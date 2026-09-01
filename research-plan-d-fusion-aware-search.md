# 研究方案 D (v3.0)：融合感知 + LLM 增强的约束引导分层自动调优

> **状态**：研究提案 (v3.0) | **目标会议**：MLSys 2027（首选）/ ASPLOS 2027 | **预估周期**：7-8 个月
>
> **v3.0 来源与动机**：本方案继承方案 C (v2.0) 的核心架构（约束图 + CGAS + 双向剪枝 + 解耦代价模型），基于 2024–2026 年最新顶会论文的系统性文献调研，在以下方面做了重大扩展：

| # | 新增模块                                        | 灵感来源                                         | 解决的问题                                              |
| - | ----------------------------------------------- | ------------------------------------------------ | ------------------------------------------------------- |
| 1 | **融合感知预变换** (§4)                  | Nautilus (2026), Neptune (2025)                  | 方案 C 仅处理单算子，无法覆盖 Attention 等融合 kernel   |
| 2 | **LLM 选择性代价代理** (§6)              | GPU Forecasters (2026), Meta LLM Compiler (2024) | 在代价模型中引入 LLM 作为可选的折损率预测器             |
| 3 | **RDU 主动采样** (§7.2)                  | TCL (2026)                                       | 用 10% 的 micro-benchmark 数据达到原精度                |
| 4 | **因果回溯机制** (§3.3 改进)             | ECCO (2025)                                      | 用因果推断替代启发式回溯触发条件                        |
| 5 | **融合算子实验** (§10)                   | Neptune (2025)                                   | 新增 FlashAttention / LayerNorm+Residual 等融合算子评估 |
| 6 | **与 Nautilus 的系统性对比** (§2.4, §9) | Nautilus (2026)                                  | 明确差异化定位：搜索效率优化 vs. 端到端 math-to-kernel  |

---

## 目录

1. [问题定位与相关工作](#1-问题定位)
2. [核心洞察与系统定位](#2-核心洞察)
3. [算法设计：CGAS + 因果回溯](#3-算法设计)
4. [融合感知预变换](#4-融合感知预变换)
5. [搜索空间生成：双向剪枝](#5-搜索空间生成)
6. [LLM 增强的解耦代价模型](#6-llm-增强)
7. [RDU 主动采样与训练数据](#7-rdu-主动采样)
8. [Tensor Core 路径的约束图](#8-tensor-core-路径)
9. [理论分析](#9-理论分析)
10. [实验设计](#10-实验设计)
11. [详细实施计划](#11-实施计划)
12. [预期成果与创新点](#12-预期成果)
13. [风险与对策](#13-风险)
14. [时间线](#14-时间线)
15. [附录：v3.0 修订记录](#15-附录)

---

## 1. 问题定位

### 1.1 拒绝采样 vs. 可行域搜索（继承方案 C §1.1）

Meta Schedule 的搜索低效源于一个结构性缺陷：**在高维约束空间中做拒绝采样**。`MultiLevelTiling` 将跨内存层级的决策全部塞进同一条规则，四条子规则 (`TileLoopNest → AddWriteReuse → AddReadReuse → AddAsyncPipeline`) 顺序执行，输出笛卡尔积，全部交给 Postproc 的事后过滤。

我们在 A100 上实测了 MatMul (256×256×256) 的 Postproc 行为：

| 演化迭代 | VerifyGPUCode 拒绝数 | 说明                                      |
| -------- | -------------------- | ----------------------------------------- |
| 第 1 轮  | 468                  | 生成了 ~484 个候选，仅 16 个通过 Postproc |
| 第 2 轮  | 925                  | 生成了 ~941 个候选，仅 16 个通过          |
| 后续每轮 | 58–90               | 每轮仍有大量候选被丢弃                    |

最终仅有约 46.5% 的测量预算花在有效候选上。本方案将搜索从**拒绝采样**转变为**可行域搜索**——约束在生成阶段就限制搜索范围。

### 1.2 新挑战：融合算子的搜索空间

方案 C 假设搜索对象是单个算子（MatMul, Conv2D, Reduction）。但现代深度学习编译器的实际瓶颈已转移到**融合算子**（fused operators）：

- **FlashAttention**：`QK^T → Softmax → PV`，三次矩阵运算融合为单个 kernel
- **LayerNorm + Residual + GeLU**：Transformer 中的经典融合模式
- **Conv + BN + ReLU**：推理图中的标准融合

Neptune (2025) 展示了代数融合可以打破循环依赖，生成等价于 FlashAttention 的 kernel。Nautilus (2026) 更进一步——从数学描述出发，自动发现 FlashAttention-3 级别的融合 kernel。

**融合算子对方案 C 提出的新问题**：

1. 融合改变了访存模式和共享内存需求——原来按单算子计算的 Roofline 上界不再准确
2. 融合后的约束图拓扑更复杂——多个算子的资源消耗叠加在同一个 kernel 中
3. 融合算子的搜索空间更大——不同融合策略（fused vs. separated）本身就是一个搜索维度

### 1.3 与 Nautilus 的根本差异（v3.0 新增）

Nautilus (2026) 是方案 D 最直接的相关工作。两者的根本差异：

| 维度               | Nautilus                           | 方案 D (本方案)/                       |
| ------------------ | ---------------------------------- | -------------------------------------- |
| **目标**     | 端到端 math-to-kernel 编译         | 在现有生产编译器（TVM）中提升搜索效率  |
| **输入**     | 数学表达式                         | TIR schedule + 融合后的子图            |
| **优化策略** | 连续 lowering + expression rewrite | 约束引导的分层搜索 + LLM 增强代价模型  |
| **Fusion**   | 内建 reduction fusion              | 融合感知预变换 + 约束图扩展            |
| **搜索算法** | 自研 auto-scheduler                | CGAS（在 TVM Meta Schedule 框架内）    |
| **代价模型** | 未详细公开                         | 白盒 Roofline + 黑盒折损率（可选 LLM） |
| **部署方式** | 独立编译器                         | TVM 插件（零迁移成本）                 |

**核心定位差异**：Nautilus 重新发明了整个编译栈；方案 D 在不改变 TVM 现有架构的前提下，通过约束建模和智能搜索大幅提升调优效率。

### 1.4 相关工作系统综述

| 类别                   | 代表工作                                            | 与本方案的关系                  |
| ---------------------- | --------------------------------------------------- | ------------------------------- |
| **端到端编译器** | Nautilus (2026)                                     | 竞品，定位不同（见 §1.3）      |
| **搜索空间优化** | Ansor (2020), Transfer-Tuning (2022), CATWILD       | 方案 D 的祖先/平行线            |
| **算子融合**     | Neptune (2025), FlashAttention (2022)               | 方案 D 的融合预变换灵感来源     |
| **代价模型**     | TLP, TCL (Mamba, 2026), GPU Forecasters (LLM, 2026) | 方案 D 的 LLM 代理灵感来源      |
| **LLM + 编译器** | Meta LLM Compiler (2024), DynamoLLM (HPCA 2025)     | 验证 LLM 在编译器任务中的可行性 |
| **因果推断**     | ECCO (2025)                                         | 方案 D 的回溯机制灵感来源       |
| **TVM 平台**     | MATCH (2024, 异构边缘), TVM + BayesianOpt (2023)    | 验证 TVM 平台的可扩展性         |

---

## 2. 核心洞察

### 2.1 修正后的主张（继承方案 C §2.1）

> GPU 内核的搜索空间具有由硬件资源约束诱导的**部分有序结构（Partial Order）**。不同内存层级的决策通过**资源容量约束**而非任意的性能交互耦合。通过显式建模这种约束结构，可以将搜索从"扁平笛卡尔积 + 事后过滤"变为"约束引导的分层探索 + 有限回溯"。

### 2.2 v3.0 新增主张

> **(1) 融合算子可以通过代数等价变换分解为约束图可处理的子结构。** 方案 D 引入融合感知预变换（Fusion-Aware Pre-pass），在进入 CGAS 之前对融合后的 TIR 做等价变换，拆解为约束图可建模的 tile 搜索问题。
>
> **(2) LLM 可以作为选择性代价代理，减少对真实 GPU 测量的依赖。** 在代价模型中引入 LLM 作为黑盒折损率预测的可选方案——当 LLM 对其预测置信度高时，直接用预测值；置信度低时，回退到真实 GPU 测量。
>
> **(3) 回溯决策可以通过因果推断形式化。** 用 ECCO 风格的因果分析替代方案 C 的启发式回溯触发条件，回答"放松哪个约束最可能改善目标函数？"

### 2.3 系统架构总览（v3.0）

```
                        ┌──────────────────────┐
                        │   TVM Relay/Relax     │
                        │   计算图 (含融合子图)  │
                        └──────────┬───────────┘
                                   │
                                   ▼
              ┌────────────────────────────────────────┐
              │  §4: 融合感知预变换 (Fusion-Aware Pre-pass) │  ← v3.0 新增
              │  · 代数等价变换 (Neptune 风格)            │
              │  · 融合决策编码为搜索维度                  │
              │  · 约束图拓扑生成                           │
              └────────────────┬───────────────────────┘
                               │
                               ▼
              ┌────────────────────────────────────────┐
              │  §3: 约束引导的交替搜索 (CGAS)            │  ← 方案 C 继承
              │  · 有向约束图 DAG                         │
              │  · Top-2 Beam Search                     │
              │  · 因果回溯 (ECCO 启发)  ← v3.0 改进     │
              └────────────────┬───────────────────────┘
                               │
              ┌────────────────┼───────────────────────┐
              │                ▼                        │
              │  §5: 双向剪枝空间生成器                    │  ← 方案 C 继承
              │  · 向上剪枝 (硬件天花板)                   │
              │  · 向下剪枝 (性能地板)                     │
              │  · 约束双向流动                            │
              └────────────────┬───────────────────────┘
                               │
                               ▼
              ┌────────────────────────────────────────┐
              │  §6: LLM 增强的解耦代价模型  ← v3.0 新增 │
              │  ┌──────────────────────────────────┐  │
              │  │ 白盒 Roofline (确定性)              │  │
              │  │  ↓                                 │  │
              │  │ 黑盒折损率                          │  │
              │  │  ├── MLP (方案 C 默认)              │  │
              │  │  ├── Mamba (TCL 风格，可选)         │  │
              │  │  └── LLM 代理 (选择性，可选) ← v3.0  │  │
              │  │       · 高置信度 → 直接用           │  │
              │  │       · 低置信度 → defer GPU 测量   │  │
              │  └──────────────────────────────────┘  │
              └────────────────┬───────────────────────┘
                               │
                               ▼
                      ┌────────────────┐
                      │  GPU 实测验证   │
                      │  (最终仲裁者)   │
                      └────────────────┘
```

### 2.4 与 Meta Schedule 的根本差异

|              | Meta Schedule      | 方案 C (v2.0)        | 方案 D (v3.0)                                  |
| ------------ | ------------------ | -------------------- | ---------------------------------------------- |
| 搜索空间组织 | 扁平笛卡尔积       | 有向约束图           | 有向约束图 + 融合算子子图                      |
| 约束检查时机 | 事后（Postproc）   | 事前（约束传播）     | 事前 +**融合变换后**                     |
| 代价模型     | XGBoost (扁平特征) | 白盒 + 黑盒 MLP      | 白盒 + 黑盒（MLP / Mamba /**LLM 代理**） |
| 回溯机制     | 无                 | 启发式触发           | **因果推断触发**                         |
| 训练数据采集 | 随机采样           | 随机 micro-benchmark | **RDU 主动采样 (10%)**                   |
| 融合算子     | 不支持             | 未覆盖               | **融合感知预变换**                       |

---

## 3. 算法设计：CGAS + 因果回溯

### 3.1 CGAS 算法框架（继承方案 C §3.1，微调）

```
算法: Constraint-Guided Alternating Search (CGAS) v3.0

输入:
  M: TIR 模块（经过融合感知预变换）
  H: GPU 硬件规格
  B: 总测量预算

初始化:
  d₁⁰ = HeuristicInit_L1(M, H)   // 保证 occupancy ≥ 50% 的最小 block tile
  d₂⁰ = HeuristicInit_L2(M, H)
  d₃⁰ = HeuristicInit_L3(M, H)
  best = (d₁⁰, d₂⁰, d₃⁰)
  history = []
  backtrack_count = {L1: 0, L2: 0, L3: 0}

for round = 1 to K_max:            // 外层: 交替优化 (K_max ≤ 5)
    for tier in [L1, L2, L3]:
        // 1. 从其他层传播约束
        C = PropagateConstraints(all_tiers \ {tier}, H)

        // 2. 在约束区域内搜索 (Top-2 Beam Search)
        S_tier = GenerateSearchSpace(M, tier, C)   // 双向剪枝 (见 §5)
        beams = TopK(S_tier, k=2, strategy=["conservative", "aggressive"])
        d_new, results = SearchTierWithBeam(tier, beams, budget = B_tier)

        // 3. 更新当前层决策 + 记录
        d[tier] = d_new
        history.extend(results)

        // 4. 更新全局最优
        if Runtime(d₁, d₂, d₃) < Runtime(best):
            best = (d₁, d₂, d₃)

    // 5. 因果回溯检查 (v3.0 改进，见 §3.3)
    if CausalBacktrackCondition(history, tier_boundary, backtrack_count):
        tier_to_relax = CausalIdentifyConstraint(history)
        RelaxConstraint(tier_to_relax, factor=1.5)
        backtrack_count[tier_to_relax] += 1
        continue

    // 6. 收敛检查
    if Improvement(best, last_best) < ε for consecutive(patience=3):
        break

返回: best, history

PropagateConstraints(fixed_tiers, H):
    return FeasibleRegion(
        shared_mem_avail = H.shmem_per_sm - sum(t.shmem_used for t in fixed_tiers),
        registers_avail = H.regs_per_thread - sum(t.regs_used for t in fixed_tiers),
        min_occupancy = H.min_occupancy_for_latency_hiding,
        block_tile_range = ComputeValidBlockTileRange(fixed_tiers, H),
    )
```

### 3.2 Beam Search 策略（继承方案 C，明确化）

Top-2 Beam Search 保留两条探索路径：

| Beam          | 策略         | 描述                               | 防止的问题       |
| ------------- | ------------ | ---------------------------------- | ---------------- |
| Beam 1 (保守) | 高 Occupancy | 选择 Occupancy ≥ 50% 的 tile 配置 | 延迟隐藏不足     |
| Beam 2 (激进) | 大 Tile      | 选择数据复用度最高的 tile 配置     | 过早收敛于保守解 |

两条 beam 并行探索各自的最优路径，最终取 runtime 更优者。这一设计防止了贪心算法在 L1 选择小 block tile → L2 被迫用小 warp tile → 整体性能受限的陷阱。

### 3.3 因果回溯机制（v3.0 改进，替换方案 C §3.3）

方案 C 的回溯触发条件是启发式的（"top-5 集中在边界 ±10% + 过去 2 轮无改进"）。方案 D 引入 ECCO (2025) 风格的因果推断，形式化地回答：**放松哪个约束最可能改善目标函数？**

```
CausalBacktrackCondition(history, tier_boundary, backtrack_count):

    // 条件 1: 当前层搜索停滞
    1. if Improvement(current_best, past_2_rounds_best) > ε:
          return False   // 仍在改进，不需要回溯

    // 条件 2: 回溯次数限制
    2. if sum(backtrack_count.values()) ≥ 2:
          return False   // 防止无限回溯

    // 条件 3: 因果效应估计
    3. for each constraint c in active_constraints:
          // 构建因果图: constraint_c → runtime
          // 使用 history 中的 (constraint_value, runtime) 对
          ATE[c] = EstimateCausalEffect(history, treatment=c, outcome=runtime)
          // 控制变量: 其他层的决策

    4. most_causal = argmax ATE  // 放松后改进最大的约束

    5. if ATE[most_causal] > CAUSAL_THRESHOLD:
          CausalIdentifyConstraint(history) → returns most_causal
          return True
       else:
          return False   // 约束收紧不是瓶颈，回溯无益
```

**因果效应估计的实现**：

使用 Doubly Robust Estimation（结合倾向性评分 + 回归模型），对每个约束维度的"如果放松 ±50%，runtime 预期改善量"做反事实推断。输入数据是 history 中已有的 (tile_config, runtime) 对——不需要额外采集数据。

**与方案 C 启发式回溯的对比**：

|          | 方案 C 启发式        | 方案 D 因果推断                        |
| -------- | -------------------- | -------------------------------------- |
| 判断依据 | 边界分布 + 轮次计数  | 反事实因果效应                         |
| 回溯目标 | 固定：产生约束的上层 | 动态：因果效应最大的约束               |
| 放松幅度 | 固定 ±50%           | 因果效应估计确定放松量                 |
| 误触发率 | ~30%（估计）         | 预期 < 10%                             |
| 计算开销 | 无                   | 轻量（~10ms，在 history 上做统计推断） |

**回退策略**：如果因果推断的计算开销在实际运行中成为瓶颈，回退到方案 C 的启发式触发条件作为 fallback。

---

## 4. 融合感知预变换（v3.0 核心新增）

### 4.1 动机

方案 C 的约束图假设搜索对象是单个算子（如 MatMul、Conv2D）。但实际 TVM workload 中，经过 Relax 图级融合后，进入 Meta Schedule 调优的通常已经是融合后的子图：

```
原始图:    QK^T → Softmax → PV    (三个独立算子)
Relax 融合后: FusedAttention     (一个融合 kernel，调用一次 Meta Schedule)
```

融合 kernel 的访存模式和资源需求与单算子有很大差异：

- 中间结果（如 QK^T 的结果矩阵）不再写回全局内存，而是留在共享内存/寄存器中
- 多个算子的 tile 配置需要联合优化（不是各自独立最优的简单加和）
- 共享内存同时被多个运算阶段使用，容量约束更紧张

### 4.2 代数等价变换（Neptune 启发）

方案 D 在进入 CGAS 之前，对融合后的 TIR 执行**融合感知预变换**。其核心思想来自 Neptune (2025)：通过代数等价变换，将融合算子的依赖结构转换为约束图可处理的形式。

```
Fusion-Aware Pre-pass 流程:

输入: 融合后的 TIR (如 FusedAttention，包含 QK^T + Softmax + PV)

Step 1: 融合算子分类
  根据融合模式将算子分类为:
  - Reduction Fusion: QK^T, PV (矩阵乘 → reduction)
  - Element-wise Fusion: Softmax, GeLU, LayerNorm
  - Mixed Fusion: LayerNorm + MatMul + Residual

Step 2: 代数依赖分析 (Neptune 风格)
  for each 融合边 e = (op_a, op_b):
      // 检查依赖类型
      if e 是 reduction→element-wise (如 QK^T → Softmax):
          标记为 "可流水线化的循环依赖"
          计算等效的 arithmetic intensity（考虑中间数据留存在 on-chip）
      elif e 是 element-wise→element-wise (如 GeLU → Residual):
          标记为 "无循环依赖"，可直接链式 tile
      elif e 是 reduction→reduction (如 MLP 的两个 MatMul):
          标记为 "串行 reduction"，各自独立 tile，共享内存需要分段分配

Step 3: 约束图生成
  根据融合类型生成扩展的约束图:
  - 新增 "FusionEdge" 节点：表示融合边界上的数据依赖
  - 融合后的共享内存上限 = 各阶段共享内存需求的最大值（流水线化后可以复用）
  - 融合后的 Roofline 上界 = Σ(各阶段计算量) / max(各阶段访存量)（见 §6.1）

Step 4: 融合决策编码为搜索维度（可选）
  // 不是所有融合都是有益的——有时分离执行更快
  if 融合边 e 的 arithmetic intensity < 阈值:
      将 "是否融合" 编码为一个布尔搜索维度
      S_fusion = {fused, separated}
      对两种选择分别执行 CGAS，取更优者

输出: 标注了融合结构的 TIR + 扩展的约束图
```

### 4.3 融合算子的约束图扩展

以 FlashAttention（QK^T + Softmax + PV）为例，扩展后的约束图：

```
                    ┌─────────────────────┐
                    │  融合顶层 (Fusion Top)│
                    │  · 总计算量: 4×B×N²×d │
                    │  · 中间数据留存在 SMEM│
                    └──────────┬──────────┘
                               │
            ┌──────────────────┼──────────────────┐
            ▼                  ▼                  ▼
    ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
    │ QK^T 阶段     │  │ Softmax 阶段  │  │ PV 阶段       │
    │ (G→S tile:   │  │ (全在 SMEM,   │  │ (S→R tile:   │
    │  bi, bj, bk) │  │  无全局访存)  │  │  bi, bj, bk) │
    └──────┬───────┘  └──────────────┘  └──────┬───────┘
           │                                   │
           │ 共享内存消耗                       │ 共享内存消耗
           │ Q_tile + K_tile + 输出 tile       │ QK^T 输出(已在SMEM) + V_tile + 输出
           │                                   │
           └───────────────┬───────────────────┘
                           │ 聚合共享内存
                           │ max(QK^T_SMEM, PV_SMEM)
                           │ (流水线化后可复用)
                           ▼
                  ┌─────────────────┐
                  │ Block Tile      │
                  │ (受融合 SMEM 上限│
                  │  约束，需同时满足│
                  │  两个阶段的整除) │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │ Pipeline +      │
                  │ Warp Tile       │
                  └─────────────────┘
```

**关键性质**：

- 融合算子的约束图比单算子**多了一层**（Fusion Top 节点），但底层的 tile 搜索逻辑不变
- 共享内存容量约束考虑了流水线化后的复用——两个阶段的 SMEM 需求取 max 而非 sum
- Block Tile 的整除性约束需要同时满足 QK^T 和 PV 两个阶段的 warp tile 分解

### 4.4 融合算子覆盖率

| 融合模式                  | 代表算子                       | 本方案覆盖 | 备注                                    |
| ------------------------- | ------------------------------ | ---------- | --------------------------------------- |
| Reduction → Element-wise | FlashAttention                 | ✓         | 核心覆盖                                |
| Element-wise 链           | LayerNorm + GeLU + Dropout     | ✓         | 直接链式 tile                           |
| Reduction → Reduction    | MLP (MatMul → GeLU → MatMul) | ✓         | 串行 reduction，分段 SMEM               |
| Conv → BN → ReLU        | 推理融合                       | ✓         | im2col 后视为 implicit gemm 融合        |
| Cross-Attention           | Encoder-Decoder Attention      | 部分       | 需要在 TIR 层面显式表示 cross-attention |

---

## 5. 搜索空间生成：双向剪枝（继承方案 C §3.4）

本节内容完整继承方案 C (v2.0) 的双向剪枝设计，不再重复展开。核心要点：

- **向上剪枝**：共享内存容量、线程数上限、Warp 对齐、寄存器容量——保证候选不崩溃
- **向下剪枝**：Occupancy ≥ 25%、Block/Warp 整除性、数据复用度 ≥ 阈值——保证候选有基本效率
- **约束双向流动**：`constraints_from_downstream` 允许下层约束（如 TC Swizzle 对齐）反推上层选择

唯一修改：在 §4 融合感知预变换后，向下剪枝的 Occupancy 和数据复用度阈值需要针对融合算子重新标定（因为融合后的计算量/访存量比率变化）。

---

## 6. LLM 增强的解耦代价模型（v3.0 核心新增）

### 6.1 架构：白盒 Roofline + 黑盒折损率（继承方案 C §5）

代价评估的两层架构不变：

```
Final_Score = 白盒理论上界 × 黑盒折损率
```

**白盒 Roofline**（纯分析型，无需训练）：

```
compute_roof = (total_FLOPs × SM_count × GPU_clock) / (FLOPs_per_cycle × ...)
memory_roof = (total_bytes × HBM_bandwidth) / (bytes_per_cycle × ...)
whitebox_upperbound = min(compute_roof, memory_roof) × occupancy_factor
```

**融合算子的白盒调整**：

- `total_FLOPs` = Σ(各融合阶段的 FLOPs)
- `total_bytes` = 输入数据 + 输出数据（中间结果留在 on-chip，不计入 HBM 流量）
- 例如 FlashAttention：输入 Q, K, V 各 B×N×d，输出 O (B×N×d)，总 HBM 流量 ≈ 4×B×N×d×sizeof(dtype)，远小于未融合的三个独立 kernel 的流量

### 6.2 黑盒折损率的三种实现（v3.0 新增选择）

方案 C 只用了 MLP。方案 D 提供三种可选实现，实验中做消融对比：

```
                         ┌──────────────────────┐
                         │   黑盒折损率预测器      │
                         │   输入: 指令排布特征    │
                         │         (刻意剥离容量)  │
                         └──────────┬───────────┘
                                    │
            ┌───────────────────────┼───────────────────────┐
            ▼                       ▼                       ▼
  ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────────┐
  │ MLP (方案 C)     │   │ Mamba (TCL 启发) │   │ LLM 代理 (v3.0 新增) │
  │ <100K params     │   │ <500K params     │   │ 预训练 Code LLM      │
  │ 推理 <10μs       │   │ 推理 <50μs       │   │ 推理 ~10–50ms        │
  │ 强 baseline      │   │ 捕获序列依赖     │   │ 选择性调用           │
  └─────────────────┘   └─────────────────┘   └──────────┬──────────┘
                                                          │
                                                          ▼
                                              ┌─────────────────────┐
                                              │ 置信度校准            │
                                              │ · 高置信度 → 直接用  │
                                              │ · 低置信度 → defer   │
                                              │   GPU 实测           │
                                              └─────────────────────┘
```

### 6.3 LLM 代理的设计（GPU Forecasters 启发）

GPU Forecasters (2026) 的核心发现：**LLM 可以预测 GPU kernel 的相对性能**，且在自评置信度高的预测上精度可观。

方案 D 将 LLM 作为**选择性代理**（Selective Surrogate）：

```
LLM_SelectivePredict(features, threshold=0.8):
    // 1. 构建 prompt（指令排布特征 + 算子类型 + GPU 型号）
    prompt = FormatPrompt(features, op_type, gpu_model)

    // 2. LLM 推理
    raw_output = LLM(prompt)  // 输出: 折损率 ∈ (0,1] + 置信度 ∈ (0,1]

    // 3. 置信度门控
    efficiency, confidence = ParseOutput(raw_output)
    if confidence ≥ threshold:
        return efficiency, source="LLM"
    else:
        return None, source="defer"  // 回退到 GPU 实测

    // 4. 端到端分数
    if source == "LLM":
        predicted_latency = whitebox_upperbound / max(efficiency, 0.01)
    else:
        predicted_latency = MeasureOnGPU(candidate)  // 真实测量
    return 1.0 / predicted_latency
```

**LLM prompt 设计**：

```
System: You are a GPU kernel performance predictor. Given a schedule's
instruction-level features (NOT tile sizes or memory capacities — those
are handled analytically), predict the micro-architecture efficiency
ratio: actual_runtime / roofline_upperbound. Output a float in (0, 1]
and a confidence score in (0, 1].

User: Operator: FusedAttention_fp16
GPU: NVIDIA A100 (Ampere, 312 TFLOPS fp16, 2039 GB/s HBM2e)
Schedule Features:
  - Unroll Factor: 4
  - Vectorize Width: 8
  - Pipeline Stages: 3
  - Loop Order: ijk (outer) → kij (inner)
  - Cooperative Fetch: 4 warps
  - Swizzle Pattern: 128B
  - Async Copy Depth: 2 slabs

Predict efficiency ratio and confidence:
```

**预期效果**：

- 搜索预算节省：LLM 代理替代 30–50% 的 GPU 测量（对高置信度的候选直接用预测值）
- 精度损失可控：defer 机制保证了低置信度候选仍然走真实测量
- 这是方案 C → 方案 A（LLM 代价模型）的自然桥接——通过 LLM 代理的实验，可以回答"纯 LLM 代价模型是否可行"

### 6.4 三种黑盒模型的消融对比

| 模型                 | 参数量      | 推理延迟  | 精度 (预期 ρ)       | 训练数据需求    |
| -------------------- | ----------- | --------- | -------------------- | --------------- |
| MLP (baseline)       | <100K       | <10μs    | >0.80                | 10–20K samples |
| Mamba (TCL)          | <500K       | <50μs    | >0.85                | 10–20K samples |
| LLM 代理 (selective) | 7B (预训练) | ~10–50ms | >0.75 (高置信度子集) | 0 (zero-shot)   |

---

## 7. RDU 主动采样与训练数据（v3.0 新增）

### 7.1 动机

方案 C 的 micro-benchmark 采集使用随机采样。TCL (2026) 证明：RDU (Representativeness-Diversity-Uncertainty) 主动采样策略只需 10% 的训练数据即可达到原精度。

### 7.2 RDU 采样器

```
RDU_Sampler(program_pool, budget_ratio=0.1):

    // Phase 1: 计算三项目标
    for each program p in program_pool:
        Representativeness[p] = 1 / min_{q in selected} distance(p, q)
            // 距离度量: tile 参数空间中的 L2 距离
        Diversity[p] = entropy(feature_distribution(selected ∪ {p}))
            // 特征分布的信息熵增益
        Uncertainty[p] = 1 - max_{q in selected} similarity(p, q)
            // 与已选样本的最大相似度的补数
        RDU[p] = α × Representativeness + β × Diversity + γ × Uncertainty
            // α=0.4, β=0.3, γ=0.3 (TCL 推荐的权重)

    // Phase 2: 贪心选择
    selected = []
    while len(selected) < budget_ratio × len(program_pool):
        p* = argmax RDU[p]
        selected.append(p*)
        UpdateRDU(all programs, selected)  // 增量更新

    return selected
```

### 7.3 Micro-benchmark 采集流程

```
采集流程 (RDU 增强版):

1. 程序生成: 对每个算子类型，生成 500–1000 个变体
    变体维度: tile 大小、unroll factor、vectorize width、pipeline stage
2. RDU 采样: 从变体池中选择 10% (≈ 50–100 个)
3. GPU 测量: 每个采样变体在 GPU 上实测 runtime
4. 标签计算: 折损率 = 实测 runtime / 白盒 Roofline 上界
5. 模型训练: 用 RDU 采样数据训练黑盒折损率预测器
6. 验证: 在剩余的 90% 变体上验证预测精度
```

---

## 8. Tensor Core 路径的约束图（继承方案 C §4）

本节的 v2.0 修正（联合宏节点）完整继承，不再重复。唯一新增：融合算子中的 Tensor Core 使用。

### 8.1 融合算子的 Tensor Core 约束

在 FlashAttention 中，QK^T 和 PV 都使用 Tensor Core (mma 指令)。融合后的约束图需要处理两个阶段的 mma_tile 需求：

```
融合算子的 TC 联合宏节点:

tc_macro: {
    mma_qk: m16n8k16,     // QK^T 阶段的 mma tile
    mma_pv: m16n8k16,     // PV 阶段的 mma tile
    // 约束: mma_qk.K == mma_pv.K (K 维度在两个阶段必须一致)
    //      因为中间结果 (QK^T 输出) 的 layout 由第一个 mma 决定
    swizzle: 128B          // 两个阶段共享同一 Swizzle 模式
}
```

两个 mma_tile 的 K 维度共享同一 tile 参数 `bk`，这自然满足了 M/N/K 维度上的整除约束。

---

## 9. 理论分析（继承方案 C §6，微调）

方案 C 的四个命题（局部收敛、收敛速率、约束保守性、回溯充分性）全部保留。

**v3.0 新增命题**：

**命题 5（LLM 代理的安全下界）**：在使用 LLM 选择性代理时，最终搜索到的 kernel 性能不低于不使用 LLM 代理的 baseline 性能。因为 defer 机制保证了低置信度候选走真实测量，LLM 代理只替换了高置信度候选的测量——在最坏情况下（LLM 全部 defer），退化为 baseline。

**命题 6（融合预变换的保优性）**：融合感知预变换不改变 kernel 的语义正确性，因为所有变换都是代数等价的（Neptune 已证明其代数修正的正确性）。

---

## 10. 实验设计

### 10.1 主实验：搜索效率对比

| 方法                               | 描述                                         | 搜索空间类型       |
| ---------------------------------- | -------------------------------------------- | ------------------ |
| Meta Schedule (Evolutionary)       | 扁平笛卡尔积 + 事后 Postproc 过滤            | 拒绝采样           |
| Ansor                              | TVM 上一代自调度器                           | 层次化             |
| **Nautilus** (v3.0 新增)     | 端到端 math-to-kernel + expression rewrite   | 连续 lowering      |
| CUTLASS Profiler (Oracle)          | 手工模板的参数暴力搜索                       | 手工分层           |
| Triton Auto-Tuner                  | Triton 自动调优                              | 程序员分层         |
| **CGAS w/ XGBoost** (消融)   | 分层空间 + 单一 XGBoost                      | 可行域搜索（部分） |
| **CGAS w/o 向下剪枝** (消融) | 仅向上剪枝                                   | 可行域搜索（部分） |
| **CGAS v2.0** (方案 C)       | 双向剪枝 + 混合代价 + Beam                   | 可行域搜索         |
| **CGAS v3.0** (方案 D)       | CGAS v2.0 + 融合预变换 + LLM 代理 + RDU 采样 | 可行域搜索（完整） |

**核心图**：X 轴 = wall-clock 时间，Y 轴 = 当前最优加速比（vs. CUTLASS oracle）。

### 10.2 消融实验（方案 C §8.2 全部保留 + v3.0 新增）

| 实验                               | 变量                        | 回答的问题                            | 来源 |
| ---------------------------------- | --------------------------- | ------------------------------------- | ---- |
| 分层空间 vs 扁平空间               | 只用 XGBoost，变空间结构    | 分层空间本身的收益                    | C    |
| 向上剪枝 vs 双向剪枝               | CGAS 含/不含向下剪枝        | 性能地板的价值                        | C    |
| 解耦混合 vs 单一代价               | 变代价模型结构              | 白盒 + 黑盒解耦的增益                 | C    |
| Beam Top-2 vs Top-1                | CGAS 含/不含 Beam Search    | 防贪心陷阱是否必要                    | C    |
| 交替 vs 贪心                       | CGAS 含/不含外层循环        | 交替优化是否优于贪心                  | C    |
| 启发式回溯 vs 因果回溯             | 变回溯触发机制              | **因果推断的价值 (v3.0)**       | D    |
| **RDU vs Random 采样**       | 变 micro-benchmark 采样策略 | **主动采样的数据效率 (v3.0)**   | D    |
| **LLM 代理 on/off**          | CGAS v3.0 含/不含 LLM 代理  | **LLM 选择性代理的价值 (v3.0)** | D    |
| **Mamba vs MLP vs LLM 代理** | 变黑盒模型实现              | **最优黑盒模型选择 (v3.0)**     | D    |
| **融合预变换 on/off**        | 含/不含融合感知预变换       | **融合支持的价值 (v3.0)**       | D    |

### 10.3 算子覆盖（v3.0 扩展）

| 算子                                       | 路径                         | 类型                              | 来源 |
| ------------------------------------------ | ---------------------------- | --------------------------------- | ---- |
| MatMul (fp32)                              | CUDA Core                    | 基础线性代数                      | C    |
| MatMul (fp16)                              | Tensor Core                  | AI 训练核心                       | C    |
| Conv2D (fp32)                              | CUDA Core (implicit gemm)    | 视觉模型                          | C    |
| Conv2D (fp16)                              | Tensor Core                  | 视觉模型加速                      | C    |
| **FlashAttention (fp16)**            | **Tensor Core (融合)** | **LLM 核心 (v3.0)**         | D    |
| **LayerNorm + Residual + GeLU**      | **CUDA Core (融合)**   | **Transformer 瓶颈 (v3.0)** | D    |
| **MLP Block (MatMul→GeLU→MatMul)** | **Tensor Core (融合)** | **LLM FFN 层 (v3.0)**       | D    |
| Reduction (fp32)                           | CUDA Core                    | 归约算子代表                      | C    |
| **Conv + BN + ReLU (fp16)**          | **Tensor Core (融合)** | **推理融合 (v3.0)**         | D    |

### 10.4 GPU 覆盖（继承方案 C §8.3）

| GPU         | 架构   | 代际     | Tensor Core      |
| ----------- | ------ | -------- | ---------------- |
| NVIDIA H100 | Hopper | 最新     | ✓ (第四代)      |
| NVIDIA A100 | Ampere | 前一代   | ✓ (第三代)      |
| NVIDIA T4   | Turing | 较老     | ✓ (第一代)      |
| AMD MI250   | CDNA2  | 竞争架构 | ✗ (Matrix Core) |

### 10.5 新增实验：LLM 代理的置信度校准

```
实验目标: 验证 LLM 代理的置信度与实际预测精度的相关性

方法:
  1. 准备 500 个不同 tile 配置的 evaluation set
  2. 对每个配置:
     a. LLM 预测折损率 + 置信度
     b. GPU 实测真实 runtime
     c. 计算预测误差 = |predicted - actual| / actual
  3. 绘制可靠性图 (Reliability Diagram):
     X 轴: LLM 输出置信度分桶 (0-0.2, 0.2-0.4, ..., 0.8-1.0)
     Y 轴: 每个桶内的平均预测精度
  4. 理想情况: 高置信度桶 → 高精度，对角线对齐
  5. 确定最优 defer 阈值: 在 F1(defer_rate, accuracy) 上做 trade-off 分析
```

### 10.6 新增实验：融合预变换的收益分解

```
实验目标: 量化融合感知预变换对最终 kernel 性能的贡献

设置:
  - 三组 FlashAttention 变体 (B=1, 64, 128; N=512, 1024, 2048; d=64, 128)
  - Baseline: 三个独立 MatMul + Softmax + MatMul 的 Meta Schedule 调优
  - Neptune: Neptune 编译器生成的融合 kernel
  - CGAS v3.0 (ours): 融合预变换 + CGAS
  - CGAS v3.0 w/o fusion: CGAS 但不做融合预变换 (即方案 C 在融合算子上直接调优)

指标:
  - Kernel runtime (μs)
  - 编译时间 (s)
  - HBM 带宽利用率 (%)
```

---

## 11. 详细实施计划

### 阶段 0：文献调研与定位确认（已完成）

产出：[research-plan-d-fusion-aware-search.md](research-plan-d-fusion-aware-search.md)

### 阶段 1：DoE 独立性量化验证（第 1–3 周，继承方案 C）

验证"层级间耦合主要通过约束传播"的假设。Kendall τ ≥ 0.8 判据不变。

**v3.0 新增子实验**：对融合算子（FlashAttention）重复 DoE，验证融合后层级间的 Kendall τ 是否仍保持 > 0.8。

### 阶段 2：约束图 + CGAS + 融合预变换（第 4–8 周）

| 任务                                        | 工作量         | 关键文件                                 |
| ------------------------------------------- | -------------- | ---------------------------------------- |
| DecisionGraph + FeasibleRegion 数据结构     | 3 天           | `constraint_graph.h/cc`                |
| PropagateConstraints 约束传播引擎           | 3 天           | `constraint_graph.h/cc`                |
| 分层空间生成器（含双向剪枝）                | 4 天           | `hierarchical_space.cc`                |
| **融合感知预变换（v3.0 新增）**       | **5 天** | **`fusion_prepass.cc`**          |
| **融合算子约束图生成器（v3.0 新增）** | **3 天** | **`fusion_constraint_graph.cc`** |
| CUDA Core 路径的 CGAS                       | 4 天           | `cgas_search.cc`                       |
| Tensor Core 路径的 CGAS                     | 3 天           | `cgas_search.cc`                       |
| **因果回溯引擎（v3.0 新增）**         | **3 天** | **`causal_backtrack.cc`**        |
| 层级感知变异器                              | 2 天           | `tiered_mutator.cc`                    |
| Python 接口 + 单元测试                      | 3 天           | `hierarchical_search.py`               |

### 阶段 3：LLM 增强代价模型 + RDU 采样（第 9–12 周）

| 任务                                             | 工作量         |
| ------------------------------------------------ | -------------- |
| **RDU 采样器实现（v3.0 新增）**            | 3 天           |
| Micro-benchmark 自动生成脚本                     | 3 天           |
| 白盒 Roofline 模块（含融合算子支持）             | 3 天           |
| 黑盒 MLP 训练（baseline）                        | 2 天           |
| **黑盒 Mamba 训练（v3.0 新增）**           | **3 天** |
| **LLM 代理集成 + 置信度校准（v3.0 新增）** | **4 天** |
| 对比实验：MLP vs Mamba vs LLM 代理               | 3 天           |

### 阶段 4：端到端评估（第 13–18 周）

| 实验                                          | 硬件                     | 模型                                                    |
| --------------------------------------------- | ------------------------ | ------------------------------------------------------- |
| 主实验：搜索效率对比（含 Nautilus baseline）  | H100, A100, MI250        | ResNet-50, BERT, Llama                                  |
| Tensor Core 专门评估                          | H100, A100               | fp16 matmul, bf16 attention                             |
| **融合算子评估（v3.0 新增）**           | **H100, A100**     | **FlashAttention, LayerNorm+Residual, MLP Block** |
| 消融实验（完整 10 项）                        | A100                     | 逐组件消融                                              |
| 跨架构泛化                                    | A100 → T4, H100 → A100 | 同一算子                                                |
| **LLM 代理置信度校准实验（v3.0 新增）** | **A100**           | **500 variants**                                  |
| Wall-clock 时间分析                           | A100                     | 编译/测量/推理 breakdown                                |

### 阶段 5：论文撰写（第 19–24 周）

---

## 12. 预期成果与创新点

### 12.1 论文贡献

1. **可行域搜索框架**：首次将 GPU 调度的搜索空间显式建模为带资源约束的有向图，将搜索从拒绝采样转变为可行域搜索
2. **融合感知预变换**（v3.0 新增）：将算子融合的代数等价变换与分层调优统一在同一框架中，首次使分层约束搜索能处理 Attention 等复杂融合 kernel
3. **LLM 增强的选择性代价模型**（v3.0 新增）：首次在 GPU 自动调优中引入 LLM 作为选择性性能代理，通过置信度门控平衡精度与效率
4. **因果回溯机制**（v3.0 新增）：用因果推断替代启发式回溯触发条件，形式化地回答"放松哪个约束最有价值"
5. **RDU 主动采样**（v3.0 新增）：将 TCL 的 RDU 采样策略引入 micro-benchmark 数据采集，减少 90% 的训练数据需求
6. **CGAS 算法**：约束引导的交替搜索 + Top-2 Beam Search
7. **解耦混合代价模型**：白盒 Roofline + 三种可选黑盒实现（MLP/Mamba/LLM）
8. **理论分析**：交替优化的收敛性、约束传播的保守性、LLM 代理的安全下界、融合预变换的保优性
9. **全面实验**：3 代 GPU + 2 家厂商，CUDA Core + Tensor Core，单算子 + 融合算子，9 种算子

### 12.2 预期定量结果

| 指标                                             | 期望值                       |
| ------------------------------------------------ | ---------------------------- |
| Wall-clock 搜索时间（vs Meta Schedule）          | -60% ~ -75%                  |
| **融合算子搜索时间（vs Meta Schedule）**   | **-65% ~ -80% (v3.0)** |
| 最终加速比（同等时间预算 vs Meta Schedule）      | 持平或略优 (+5%)             |
| **LLM 代理节省的 GPU 测量次数**            | **30–50% (v3.0)**     |
| 无效候选率（vs Meta Schedule ~60%）              | < 15%                        |
| 黑盒代价模型 Spearman ρ (MLP)                   | > 0.80                       |
| **黑盒代价模型 Spearman ρ (Mamba, v3.0)** | **> 0.85**             |
| 跨 GPU 代际迁移的 Kendall τ                     | > 0.70                       |
| **LLM 代理置信度-精度对齐 (ECE)**          | **< 0.10 (v3.0)**      |
| **RDU 采样 vs Random 采样的数据效率**      | **10× (v3.0)**        |

---

## 13. 风险与对策

| 风险                                       | 概率 | 影响 | 缓解措施                                                                            | 来源 |
| ------------------------------------------ | ---- | ---- | ----------------------------------------------------------------------------------- | ---- |
| DoE 验证 τ < 0.5（含融合算子）            | 30%  | 致命 | 降级为"软分层"；若仅融合算子 τ 低，将融合限制为可配置选项                          | C+D  |
| Nautilus 被接收后抢占注意力                | 15%  | 中等 | 强调差异化定位（搜索效率 vs 全新编译栈）；在 MLSys rebuttal 中准备回应              | D    |
| LLM 代理精度不足（ECE > 0.2）              | 35%  | 中等 | 回退到纯 MLP/Mamba 代价模型；LLM 代理作为附录/Future Work                           | D    |
| 融合预变换的代数等价性在某些算子类上不成立 | 25%  | 严重 | 限制适用范围到已验证的融合模式（Reduction→Element-wise）；其他模式回退到未融合版本 | D    |
| GPU 资源不足（多代 GPU + 融合实验）        | 40%  | 严重 | 利用云 GPU（Lambda Labs/VAST）；H100 按需租用；融合实验优先在 A100 上完成           | C    |
| CGAS 外循环次数 K 实际 > 期望              | 25%  | 中等 | 早期停止 + 递减预算策略                                                             | C    |
| 因果回溯的计算开销超预期                   | 20%  | 低   | 回退到方案 C 的启发式回溯作为 fallback（§3.3 已设计）                              | D    |

---

## 14. 时间线

```
Week  1-3:  ████████ 阶段 1: DoE 验证（含融合算子 τ 验证）
                          ↖ Week 3: Go/No-Go 决策点

Week  4-8:  ████████████████████ 阶段 2: 约束图 + CGAS + 融合预变换
                          ↖ Week 8: 融合预变换可用

Week  9-12: ████████████████████ 阶段 3: LLM 增强代价模型 + RDU
                          ↖ Week 12: 代价模型决策点

Week 13-18: ██████████████████████████ 阶段 4: 端到端评估
                          ↖ Week 18: 结果冻结

Week 19-24: ██████████████████████████ 阶段 5: 论文撰写
                          ↖ Week 24: 投稿 MLSys 2027
```

---

## 15. 附录：v3.0 修订记录

| 版本 | 日期    | 修订内容                                            |
| ---- | ------- | --------------------------------------------------- |
| v1.0 | 2025-10 | 初版：CGAS + 分层代价模型 + 约束图                  |
| v2.0 | 2026-06 | 双向剪枝 + 解耦代价模型 + TC 拓扑修正 + Beam Search |
| v3.0 | 2026-07 | 六项重大修订，详见下方                              |

### v3.0 修订摘要

| # | 节               | 修订类型           | 内容                                              | 灵感来源                                         |
| - | ---------------- | ------------------ | ------------------------------------------------- | ------------------------------------------------ |
| 1 | §4              | **完整新增** | 融合感知预变换：代数等价变换 + 融合算子约束图扩展 | Nautilus (2026), Neptune (2025)                  |
| 2 | §6.2–6.3       | **重大扩展** | LLM 选择性代价代理：置信度门控 + defer 机制       | GPU Forecasters (2026), Meta LLM Compiler (2024) |
| 3 | §6.2            | **新增选项** | Mamba 作为黑盒折损率预测器候选                    | TCL (2026)                                       |
| 4 | §7              | **完整新增** | RDU 主动采样策略，10% 数据替代随机采样            | TCL (2026)                                       |
| 5 | §3.3            | **重写**     | 因果回溯替代启发式回溯                            | ECCO (2025)                                      |
| 6 | §1.3–1.4, §10 | **新增**     | Nautilus 系统性对比 + 融合算子实验                | 文献调研                                         |
| 7 | §10             | **扩展**     | 新增融合算子、LLM 置信度校准、RDU 效率等实验      | —                                               |
| 8 | §9              | **新增命题** | LLM 代理安全下界 + 融合预变换保优性               | —                                               |

### v3.0 与 v2.0 的关键数值变化

| 参数         | v2.0            | v3.0                    | 变化原因                         |
| ------------ | --------------- | ----------------------- | -------------------------------- |
| 算子覆盖     | 7 种（单算子）  | 9 种（含 4 种融合算子） | 融合感知预变换                   |
| 代价模型选项 | 1 种（MLP）     | 3 种（MLP/Mamba/LLM）   | LLM + Mamba 集成                 |
| 消融实验     | 7 项            | 12 项                   | 融合、LLM、RDU、因果、Mamba      |
| 预估周期     | 6–7 个月       | 7–8 个月               | 融合预变换 + LLM 代理引入约 4 周 |
| 训练数据需求 | 10–20K samples | 1–2K samples (RDU 10%) | RDU 主动采样                     |

---

## 附录：与方案 A/B/C 的关系

```
方案 D (融合感知 + LLM 分层搜索) ← 最完整，覆盖面最广
    │
    ├── 核心架构来自 方案 C (CGAS + 双向剪枝 + 解耦代价模型)
    │
    ├── 融合能力来自 Neptune + Nautilus 的启发
    │
    ├── LLM 代理来自 GPU Forecasters，桥接 方案 A (LLM 代价模型)
    │      方案 D 的 LLM 代理实验直接验证了方案 A 的可行性
    │
    └── 因果回溯来自 ECCO，为 方案 B (LLM Agent 搜索) 中的决策
           提供了可解释的约束分析框架
```

**推荐发表策略**：

- **主发方案 D** 到 MLSys 2027（覆盖面最广，创新点最密）
- 如果时间不足，**方案 C v2.0 作为 fallback**（去融合 + 去 LLM，降为 6 个月周期）
- 方案 A（LLM 代价模型）和方案 B（LLM Agent 搜索）作为 Future Work 在 Discussion 中讨论
- 融合预变换模块如果独立做出显著贡献，可考虑单独投 CGO 2027 的 short paper
