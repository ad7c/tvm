# 研究方案 C (v2.0)：约束引导的 GPU 内核分层自动调优

> **状态**：研究提案 (v2.0 修订) | **目标会议**：MLSys（首选）/ ASPLOS | **预估周期**：6-7 个月
>
> **v2.0 修订说明**：本版基于 v1.0 的实验反馈和理论审查，在以下方面做了重大修订：
>
> - **3.4 节**：搜索空间生成从单向容量剪枝升级为双向剪枝（向上硬件天花板 + 向下性能地板）
> - **5 节**：代价模型从 4 个独立层级模型重构为解耦型混合模型（白盒 Roofline + 黑盒折损率）
> - **4.2 节**：修正 Tensor Core 约束图中 Swizzle 的因果倒置
> - **3.1 节**：CGAS 算法引入 Top-2 Beam Search 防贪心陷阱
> - **1.1 节 + 8.1 节**：强化"可行域搜索 vs. 拒绝采样"的叙事框架

---

## 目录

1. [问题定位：拒绝采样 vs. 可行域搜索](#1-问题定位)
2. [核心洞察与形式化](#2-核心洞察)
3. [算法设计：约束引导的交替搜索 + Beam Search](#3-算法设计)
4. [Tensor Core 路径的约束图](#4-tensor-core-路径)
5. [解耦型混合代价模型（方向讨论）](#5-解耦型混合代价模型方向讨论)
6. [理论分析](#6-理论分析)
7. [详细实施计划](#7-详细实施计划)
8. [实验设计](#8-实验设计)
9. [预期成果与创新点](#9-预期成果与创新点)
10. [风险与对策](#10-风险与对策)
11. [时间线](#11-时间线)
12. [附录：v2.0 修订记录](#12-附录v20-修订记录)

---

## 1. 问题定位

### 1.1 扁平笛卡尔积 → 拒绝采样的维度灾难

Meta Schedule 的搜索低效源于一个本质性的结构缺陷：**它在高维约束空间中做拒绝采样（Rejection Sampling）**。

根源在 `MultiLevelTiling` 把跨内存层级的决策全部塞进同一条规则。从源码 `multi_level_tiling.cc:124-131` 看 `ApplySubRules`：

```cpp
states = SubRule(states, [&](State s) { return TileLoopNest(s); });      // 循环拆分
states = SubRule(states, [&](State s) { return AddWriteReuse(s); });      // 写缓存
states = SubRule(states, [&](State s) { return AddReadReuse(s); });       // 读缓存（共享内存）
states = SubRule(states, [&](State s) { return AddAsyncPipeline(s); });   // 异步流水线
```

四条子规则顺序执行，输出是它们的笛卡尔积。搜索空间大小：

$$
|\mathcal{S}| = |\text{TileLoopNest}| \times |\text{WriteReuse}| \times |\text{ReadReuse}| \times |\text{AsyncPipeline}|
$$

**拒绝采样的代价**：候选在生成时不受约束检查，全部交给 Postproc 阶段的事后过滤（`VerifyGPUCode` 检查共享内存/线程数/向量化宽度等硬件限制）。我们在 A100 上实测了 MatMul (256×256×256) 的 Postproc 行为：

| 演化迭代 | VerifyGPUCode 拒绝数 | 说明                                      |
| -------- | -------------------- | ----------------------------------------- |
| 第 1 轮  | 468                  | 生成了 ~484 个候选，仅 16 个通过 Postproc |
| 第 2 轮  | 925                  | 生成了 ~941 个候选，仅 16 个通过          |
| 后续每轮 | 58-90                | 每轮仍有大量候选在 Postproc 被丢弃        |

这构成了一个**拒绝采样的典型病理**：在高维约束空间中盲目生成候选 → 事后硬约束过滤 → 绝大多数算力浪费在"造垃圾 → 验垃圾 → 扔垃圾"的循环上。最终仅有约 46.5% 的测量预算花在了有效的候选上。

**本方案的立场**：将搜索从拒绝采样转变为**可行域搜索（Feasible-Region Search）**——约束在生成阶段就限制搜索范围，确保每个进入测量的候选都满足硬件约束和性能底线。CPU 不浪费一毫秒在注定无效的候选上。

### 1.2 变异器的无方向性

`mutate_tile_size.cc:189-231` 中 `MutateSampleTileSize` 的核心逻辑：

```cpp
int x, y;                                  // 随机选两个 tile factor
tiles[x] /= divide_factor;                 // 一个缩小
tiles[y] *= divide_factor;                 // 另一个放大
```

变异器**不知道**它操作的是哪层内存层级的 tile——全局↔共享的那层还是共享↔寄存器的那层。这种盲变异导致大量无方向的随机游走。

### 1.3 代价模型的层级混淆

XGBoost 代价模型对所有调度决策使用同一套扁平特征（~200 维 PerStoreFeature），无法区分 "这个 tile 参数影响哪层硬件"。学到的关联是统计性的而非结构性的——无法给出 "shared memory 带宽饱和，再增大 tile 不会带来收益" 这类结构化的预测。

---

## 2. 核心洞察

### 2.1 修正后的主张

> GPU 内核的搜索空间具有**由硬件资源约束诱导的部分有序结构（Partial Order）**。不同内存层级的决策通过**资源容量约束**而非任意的性能交互耦合。通过显式建模这种约束结构，可以将搜索从 "扁平笛卡尔积 + 事后过滤" 变为 "约束引导的分层探索 + 有限回溯"。

与原来 "跨层级独立" 的假设相比，这个主张更弱，但更正确：

- **承认耦合的存在**：层级之间确实有依赖（如 block tile 通过 occupancy 影响下游优化收益）
- **刻画耦合的性质**：依赖是通过资源约束传播的（剩余共享内存量、寄存器数、occupancy），方向性明确
- **不假设可分离性**：不声称可以将搜索分解为独立子问题，而是利用约束结构缩小搜索范围

### 2.2 约束图模型

GPU kernel 的搜索空间可以建模为一个**有向无环约束图** $\mathcal{G} = (\mathcal{V}, \mathcal{E})$：

- **节点** $\mathcal{V}$：调度决策变量
- **边** $\mathcal{E}$：资源约束依赖关系

```
                    语义约束
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
    Block Tile     Tensor Core     Global Layout
   (i, j, k, w)    (mma tile,     (row/col, pad,
                    pipeline)       alignment)
        │              │              │
        │ 共享内存      │ Fragment     │ 128B 对齐
        │ 消耗量        │ 布局锁定      │ 约束
        │              │              │
        └──────┬───────┴──────┬───────┘
               │ 聚合约束      │
               │ occupancy     │
               │ 剩余 shmem    │
               │ warp 分配     │
               ▼              ▼
          Warp Tile      双缓冲/Swizzle
         (warp_i, j, k)  (bank conflict)
               │
               │ 剩余寄存器数
               │ swizzle pattern
               ▼
          指令级优化
        (unroll, vec, 指令排布)
```

**关键性质**：

- 约束沿边单向传递（L2 受 L1 约束，但不反向）
- 跨层级的耦合**只通过资源容量产生**，不存在任意的非线性交互
- 不同计算路径（CUDA Core / Tensor Core）有不同的图拓扑，但共享相同的约束传播框架

### 2.3 与 Meta Schedule 的根本差异

|              | Meta Schedule         | 本方案                       |
| ------------ | --------------------- | ---------------------------- |
| 搜索空间组织 | 扁平笛卡尔积          | 有向约束图                   |
| 约束检查时机 | 事后（Postproc 过滤） | 事前（约束传播限制搜索范围） |
| 层级间通信   | 无（全部混在一起）    | 约束区域（FeasibleRegion）   |
| 变异方向     | 盲（随机两个 tile）   | 有向（只在当前层级内变异）   |
| 搜索算法     | 单层演化              | 交替优化 + 有限回溯          |

### 2.4 核心优化问题

$$
\min_{d_1, d_2, d_3} \text{Runtime}(d_1, d_2, d_3)
$$

$$
\text{s.t.} \quad g_1(d_1) \leq C_{\text{shmem}}, \quad g_2(d_2, d_1) \leq C_{\text{remaining\_shmem}}, \quad g_3(d_3, d_1, d_2) \leq C_{\text{registers}}
$$

其中 $g_i$ 是层级 $i$ 的资源消耗函数，$C_*$ 是硬件容量常数。约束的嵌套结构暗示了分层优化的可行性——上层决策通过约束**范围**（而非具体值）影响下层。

---

## 3. 算法设计

### 3.1 约束引导的交替搜索

```
算法: Constraint-Guided Alternating Search (CGAS)

输入:
  M: TIR 模块
  H: GPU 硬件规格
  B: 总测量预算

初始化:
  d₁⁰ = HeuristicInit_L1(M, H)   // 保证 occupancy ≥ 50% 的最小 block tile
  d₂⁰ = HeuristicInit_L2(M, H)
  d₃⁰ = HeuristicInit_L3(M, H)
  best = (d₁⁰, d₂⁰, d₃⁰)
  history = []

for round = 1 to K_max:            // 外层: 交替优化轮次 (典型值 K_max ≤ 5)
    for tier in [L1, L2, L3]:
        // 1. 从其他层传播约束
        C = PropagateConstraints(all_tiers \ {tier}, H)

        // 2. 在约束区域内搜索
        S_tier = GenerateSearchSpace(M, tier, C)      // 只包含满足约束的候选
        d_new, results = SearchTier(tier, S_tier, budget = B_tier)

        // 3. 更新当前层决策 + 记录
        d[tier] = d_new
        history.extend(results)

        // 4. 更新全局最优
        if Runtime(d₁, d₂, d₃) < Runtime(best):
            best = (d₁, d₂, d₃)

    // 5. 回溯触发检查
    if BacktrackCondition(history, tier_boundary):
        TriggerBacktrack(tier)    // 放松某层的约束，允许重新搜索
        continue

    // 6. 收敛检查
    if Improvement(best, last_best) < ε for consecutive(patience=3):
        break

返回: best, history

PropagateConstraints(fixed_tiers, H):
    // 返回 FeasibleRegion，而非固定参数
    return FeasibleRegion(
        shared_mem_avail = H.shmem_per_sm - sum(t.shmem_used for t in fixed_tiers),
        registers_avail = H.regs_per_thread - sum(t.regs_used for t in fixed_tiers),
        min_occupancy = H.min_occupancy_for_latency_hiding,  // 如 50%
        block_tile_range = ComputeValidBlockTileRange(fixed_tiers, H),
    )
```

### 3.2 与贪心顺序搜索的对比

|            | 贪心（原方案）          | 交替优化（本方案）                    |
| ---------- | ----------------------- | ------------------------------------- |
| 搜索顺序   | 固定 L1→L2→L3，不回头 | 外层循环，每轮都更新所有层            |
| 层级间通信 | 固定参数值              | FeasibleRegion（约束范围）            |
| 回溯能力   | 无                      | 有：约束边界检测 → 触发回溯          |
| 全局最优性 | 无保证                  | $O(\frac{1}{K})$ 收敛到局部最优     |
| 额外开销   | 无                      | $K$ 倍外层循环（实测 $K \leq 5$） |

### 3.3 回溯触发条件

回溯是有限制的——只在以下条件同时满足时才触发：

```
BacktrackCondition(history, tier):
    1. 当前层搜索的 top-5 最优解都集中在约束边界 ±10% 范围内
    2. 当前层的历史最小 runtime 在过去 2 轮中无改进
    3. 触发次数 ≤ 2 (防止无限循环)

触发后:
    放松产生该约束的上层决策，将其搜索空间扩大到当前选择的 ±50% 范围
    重新进行该层的搜索（只用原预算的 30%）
```

回溯不是重新开始——只在受影响的局部重搜索。这种设计源于观察：**约束过紧通常只发生在 1-2 个维度的边界附近**。

### 3.4 搜索空间生成：双向剪枝 (Bidirectional Pruning)

原方案只做了单向的"合法性"检查（硬件容量天花板），这只能保证候选不崩溃，无法保证候选有意义。本节将空间生成器升级为**双向漏斗**：在生成阶段同时执行**向上剪枝**（硬件天花板）和**向下剪枝**（性能地板），确保每个进入搜索的候选既合法又有基本的物理合理性。

---

#### 3.4.1 双向剪枝概览

```
                    笛卡尔积全空间
                   (ti × tj × tk × ...)
                          │
                          ▼
              ┌───────────────────────┐
              │   向上剪枝 (天花板)     │
              │   共享内存 ≤ 硬件容量   │
              │   线程数 ≤ 1024        │
              │   Warp 对齐整除        │
              │   寄存器 ≤ 255         │
              └───────────────────────┘
                          │
                          ▼  剩余: ~200 候选 (已合法)
              ┌───────────────────────┐
              │   向下剪枝 (地板)       │  ← v2.0 新增
              │   Occupancy ≥ 25%      │
              │   Block/Warp 整除性    │
              │   数据复用度 ≥ 阈值     │
              │   下游结构对齐          │
              └───────────────────────┘
                          │
                          ▼  剩余: ~20 候选 (合法且高效)
                    进入搜索
```

向上剪枝保证候选"不崩溃"；向下剪枝保证候选"不浪费算力"。

---

#### 3.4.2 向上剪枝：硬件天花板

在 `SamplePerfectTile` 抽样后、候选进入 Builder 前，按层级逐级检查硬性物理上限。这些条件在 v1.0 中已基本存在，此处做系统化表述：

**L1 (Block Tile → Global↔Shared)**：

| 约束           | 表达式                                         | 说明                                          |
| -------------- | ---------------------------------------------- | --------------------------------------------- |
| 共享内存容量   | `estimate_shmem(bi, bj, bk) ≤ shmem_per_sm` | 按 GPU 型号查表（如 A100: 164KB/SM）          |
| 线程数上限     | `bi × bj ≤ max_threads_per_block`          | 通常 1024，注意线程布局按 2D (bi, bj) 而非 3D |
| Warp 对齐      | `bi % warp_size == 0`                        | 每个 block tile 维度须是 32 的整数倍          |
| Block 乘积整除 | `(bi×bj×bk) % (wi×wj×wk) == 0`           | 保证 L1 → L2 的因子化分解无碎片（见下）      |

**L2 (Warp Tile → Shared↔Register)**：

| 约束          | 表达式                                                        | 说明                                     |
| ------------- | ------------------------------------------------------------- | ---------------------------------------- |
| 寄存器容量    | `estimate_regs(wi, wj, wk, unroll, vec) ≤ regs_per_thread` | 通常 255/thread，含 spill 开销的保守估计 |
| 剩余共享内存  | `shmem_used_L1 + shmem_used_L2 ≤ shmem_per_sm`             | 累加 L1 和 L2 的共享内存消耗             |
| Warp ≤ Block | `wi ≤ bi, wj ≤ bj, wk ≤ bk`                              | 每个 warp tile 维度不超过对应 block tile |

**L3 (Instruction Tile → 指令排布)**：

| 约束          | 表达式                                                                  | 说明                         |
| ------------- | ----------------------------------------------------------------------- | ---------------------------- |
| 剩余寄存器    | `regs_used_L1+L2 + regs_used_L3 ≤ regs_per_thread`                   | 三级累加不超限               |
| 向量化宽度    | `vec_width ≤ min(inner_loop_extent, max_vector_bytes / dtype_bytes)` | fp32: max 4, fp16: max 8     |
| Unroll 合理性 | `unroll × vec_width ≤ innermost_loop_extent`                        | 展开不能超过内层循环总迭代数 |

---

#### 3.4.3 向下剪枝：性能地板（v2.0 核心贡献）

合法不等于高效。如果 block tile 只有 32 个线程——它合法（满足 `≤1024`），但 SM 的 2048 个线程槽位严重空转。向下剪枝在生成阶段就拒绝这些"合法但注定慢"的候选。

---

**地板一：Occupancy 保底**

GPU 通过 warp 级并行隐藏内存延迟。如果 block 太小，SM 无法驻留足够 warp，延迟隐藏失效。

```
occupancy_block = min(
    block_threads / max_threads_per_sm,       # 线程瓶颈
    shmem_per_block / max_shmem_per_sm,        # 共享内存瓶颈
    regs_per_block / max_regs_per_sm           # 寄存器瓶颈
)
条件: occupancy_block ≥ MIN_OCCUPANCY_FLOOR (默认 25%)
```

推导：给定 block tile (bi, bj, bk)，假设 2D 线程布局（bi × bj 个线程映射到输出的 i/j 维度），每个 SM 能并发驻留的 block 数由线程数、共享内存、寄存器三者中的瓶颈决定。如果 occupancy 低于 25%，SM 的大半算力被浪费。

阈值按 GPU 代际查表：

| GPU 代际       | SM 最大线程数 | 25% Occupancy 最小线程/block |
| -------------- | ------------- | ---------------------------- |
| Turing (T4)    | 1024          | 256                          |
| Ampere (A100)  | 2048          | 512                          |
| Hopper (H100)  | 2048          | 512                          |
| Ada (RTX 4090) | 2048          | 512                          |

**实现要点**：25% 是最低门槛，不是最优目标。在 Beam Search（见 3.1 节 v2.0 修订）中，我们保留一个最大化 occupancy 的分支和一个最大化 tile size 的分支，让下游探索两端的 trade-off。

---

**地板二：Block/Warp 整除性约束**

```
条件: bi % wi == 0 AND bj % wj == 0 AND bk % wk == 0
```

如果 block tile 不能被 warp tile 整除，会产生**尾 warp（tail warp）**——最后一个 warp 只有部分线程活跃，其余线程空转。此外，尾 warp 导致 shared memory 访问模式出现不规则的边界条件，容易触发 bank conflict。

这个约束从数学上保证 L1 → L2 的分解是**完美因式分解**，不存在任何碎片。

**示例**：block tile = (128, 128, 32)，warp tile = (32, 32, 16)

- `128 % 32 == 0` ✓
- `128 % 32 == 0` ✓
- `32 % 16 == 0` ✓
- 完美分割为 4×4×2 = 32 个 warp，每个 warp 完全填充

反例：block tile = (100, 100, 32)，warp tile = (32, 32, 16)

- `100 % 32 != 0` ✗ → 产生尾 warp，拒绝

---

**地板三：最小数据复用度（Data Reuse Floor）**

如果 tile 太小，每次从全局内存搬到共享内存的数据只被用一两次就丢弃，导致访存瓶颈压倒计算。

```
data_reuse = O(compute_bytes) / O(memory_traffic_bytes)
           = (2 × bi × bj × bk) / ((bi × bk + bk × bj) × sizeof(dtype))

条件: data_reuse ≥ MIN_REUSE_FLOOR (fp32 默认 8 FLOPs/byte, 即 reuse ≥ 2.0)
```

直观解释：`bi × bk` 是 A 矩阵 tile 的字节数，`bk × bj` 是 B 矩阵 tile 的字节数，`2 × bi × bj × bk` 的分子是 2 FLOPs/MAC × tile 体积。reuse = 2.0 意味着每 byte 数据至少做 8 次浮点运算——低于此阈值，kernel 主频受限于内存带宽而非算力，tile 太小毫无意义。

**算子感知**：此约束**仅对 GEMM 类算子启用**（MatMul、Conv2D 的 implicit gemm 路径）。对于 LayerNorm、Reduction 等天然低算术强度的算子，跳过复用度检查，只应用 Occupancy 和整除性地板。

---

#### 3.4.4 统一剪枝架构

以下伪代码将上述所有约束整合为单一的空间生成器。约束可以从下游反向传递（如 TC 的 Swizzle 模式决定了 Block Tile 的对齐需求）：

```python
def GenerateL1Space(ir_module, gpu_spec, constraints_from_downstream=None):
    """
    生成 L1 候选空间，同时执行双向剪枝。

    参数:
      ir_module: TIR 模块
      gpu_spec: GPU 硬件规格（SM 数、共享内存、寄存器、warp 大小等）
      constraints_from_downstream: 从 L2/L3/TC 反向传递的约束
        (如确定的 warp tile 需 block tile 整除，Swizzle 对齐需求等)
    """
    space = []
    for ti in valid_tile_range(i_dim, gpu_spec):
        for tj in valid_tile_range(j_dim, gpu_spec):
            for tk in valid_tile_range(k_dim, gpu_spec):
              
                # ============ 向上剪枝 (硬件天花板) ============
              
                # ① 共享内存容量
                shmem = estimate_shared_memory(ti, tj, tk, dtype)
                if shmem > gpu_spec.shmem_per_sm:
                    continue
              
                # ② 线程数上限 (2D 布局: ti × tj 个线程)
                threads = ti * tj
                if threads > gpu_spec.max_threads_per_block:
                    continue
              
                # ③ Warp 大小对齐
                if ti % gpu_spec.warp_size != 0:
                    continue
              
                # ④ L2 (warp tile) 大于 L1 的非法情况
                if constraints_from_downstream:
                    wt = constraints_from_downstream.get("warp_tile")
                    if wt and (ti < wt[0] or tj < wt[1] or tk < wt[2]):
                        continue
              
                # ============ 向下剪枝 (性能地板) ============
              
                # ⑤ Occupancy 保底
                if threads < gpu_spec.min_threads_for_occupancy(0.25):
                    continue
              
                # ⑥ Block/Warp 整除性 (保证因式分解无碎片)
                if constraints_from_downstream and wt:
                    if ti % wt[0] != 0 or tj % wt[1] != 0 or tk % wt[2] != 0:
                        continue
              
                # ⑦ 最小数据复用度 (仅 GEMM 类)
                if ir_module.is_gemm_like():
                    reuse = (2.0 * ti * tj * tk) / ((ti * tk + tk * tj) * dtype_bytes)
                    if reuse < MIN_REUSE_FLOOR:
                        continue
              
                # ⑧ 下游结构约束反向传递 (如 TC Swizzle 对齐)
                if constraints_from_downstream:
                    align = constraints_from_downstream.get("swizzle_alignment", 0)
                    if align > 0 and ti % align != 0:
                        continue
              
                space.append((ti, tj, tk))
    return space
```

**关键设计**：`constraints_from_downstream` 参数实现了约束的双向流动——不仅上层约束向下传播（`PropagateConstraints`），下层发现的对齐需求也向上反推。这在 Tensor Core 路径中尤其关键（Swizzle 模式反推 Block Tile 形状，见 §4.2 v2.0 修正）。

---

#### 3.4.5 剪枝效果的预期量化

在 A100 fp32 MatMul (1024×1024×1024) 上，双向剪枝的空间缩减预期：

| 阶段                     | 约束                               | 剩余候选      | 累积剪枝率         |
| ------------------------ | ---------------------------------- | ------------- | ------------------ |
| 笛卡尔积全空间           | ti,tj,tk ∈ {1,2,4,8,...,256}      | ~16,777,216   | —                 |
| + 向上剪枝               | 共享内存 + 线程数 + Warp 对齐      | ~200          | 99.9988%           |
| + 向下剪枝               | Occupancy ≥25% + 数据复用度 ≥2.0 | ~60           | 99.9996%           |
| + Block/Warp 整除        | bi%wi=0, bj%wj=0, bk%wk=0          | ~20           | 99.9999%           |
| **进入搜索的候选** |                                    | **~20** | **99.9999%** |

> 注意：以上量化基于枚举搜索空间的静态分析。实际运行中，CGAS 的交替优化会进一步缩小每层的搜索范围——L1 搜索仅在 ~20 个候选上进行，而不是原方案约定的 200 个，大幅降低了测量预算需求。

---

#### 3.4.6 v1.0 vs v2.0 对比

| 维度           | v1.0 原方案                        | v2.0 双向剪枝                                |
| -------------- | ---------------------------------- | -------------------------------------------- |
| 剪枝方向       | 仅向上（容量天花板）               | 向上 + 向下（性能地板）                      |
| 有效候选率     | ~45%（和 Meta Schedule 类似）      | 100%（所有进入搜索的候选都合法且有基本效率） |
| Occupancy 保证 | 无（可能选中 1 warp/block 的配置） | 硬性 ≥25%                                   |
| 因子化保证     | 未约束                             | 数学强制性 bi%wi=0                           |
| 数据复用       | 不检查                             | GEMM 类算子硬性下限                          |
| 空间大小（L1） | ~200 候选                          | ~20 候选（搜索效率 ×10）                    |
| 下游约束反传   | 单向 L1→L2→L3                    | 双向（Swizzle 可反推 Block Tile）            |

---

## 4. Tensor Core 路径

### 4.1 为什么必须处理 Tensor Core

现代 GPU 上 fp16/bf16/int8 矩阵乘的性能 90%+ 来自 Tensor Core。仅处理 fp32 CUDA Core 路径将使方案的实际适用范围严重受限。审稿人完全可能提出 "你的方法在 CUDA Core 上省了 80% 搜索时间，但 CUDA Core 只贡献了 10% 的性能——这有用吗？"

### 4.2 Tensor Core 的约束图拓扑（v2.0 修正）

Tensor Core 引入了**硬性的离散约束**，与 CUDA Core 的连续优化问题性质不同。**v2.0 重要修正**：原方案将 Swizzle 置于叶子节点，这是因果倒置。在硬件实际中，Swizzle 模式与 mma_tile 绑定决定了**共享内存的 128B 对齐布局**，该布局会**向上反推** Block Tile 的形状需求——Block Tile 的列维度必须满足 Swizzle 的对齐要求，否则产生 bank conflict。

```
修正后的 Tensor Core 约束图:

                联合宏节点 ─────────┐
            ┌── mma_tile ──────────┤
            │  (m16n8k8 / m16n8k16 │   ← 硬件固定，枚举穷举 (≤6 种)
            │   / m16n8k32)        │
            │  swizzle_pattern     │
            │  (none / 32B / 64B   │   ← 与 mma_tile 联合决定内存布局
            │   / 128B)            │
            └──────────────────────┘
                     │
                     │ 发射约束:
                     │ ① block_tile % mma_tile == 0  (M/N/K 每个维度)
                     │ ② block_tile 列维度 ≥ swizzle 对齐粒度
                     │ ③ pipeline_stage 有效范围由 swizzle + mma_tile 确定
                     ▼
            ┌──────────────────────┐
            │    Block Tile         │
            │  (离散，须满足上述双向  │   ← 搜索空间已被联合宏节点大幅缩小
            │   约束后才能进入搜索)  │
            └──────────────────────┘
                     │
                     │ 剩余共享内存、寄存器
                     ▼
            ┌──────────────────────┐
            │   Pipeline Stage      │
            │   ∈ {2, 3, 4, 5}      │   ← Swizzle 决定了 bank conflict
            │                       │     模式，影响 pipeline 双缓冲效率
            └──────────────────────┘
                     │
                     │ 剩余寄存器
                     ▼
            ┌──────────────────────┐
            │   指令级优化 (Epilogue)│
            │   Unroll / Vectorize   │
            └──────────────────────┘
```

**v1.0 错误**：`Block Tile → Swizzle`（单向决定）。Swizzle 被放在 Pipeline 之下，作为被动接收方。

**v2.0 修正**：`mma_tile + Swizzle` 绑定为**联合宏节点**，位于图的最顶层。Swizzle 不是叶子——它会向上反推 Block Tile 的对齐条件，向下决定 Pipeline 的 bank conflict 模式。约束流是**双向的**。

**核心差异**（v1.0 vs v2.0 不变的部分）：

- CUDA Core 路径的搜索瓶颈在 **L1**（200-500 种参数组合）
- Tensor Core 路径的搜索瓶颈在 **联合宏节点**（mma_tile × swizzle，共 3×4=12 种组合），一旦选定宏节点，下游 Block Tile 的选择空间被严格限制在 1-4 个合法值
- Tensor Core 的 shared memory swizzle pattern 是**离散的、不连续的**（没有 "swizzle=0.7" 这种中间状态），它和 mma_tile 的绑定是硬件的物理约束，不是搜索算法的设计选择

### 4.3 统一约束图框架

两条路径可以在同一个框架中统一处理——通过**在图中引入 TensorCore 决策节点**：

```python
class DecisionGraph:
    """统一的约束图，v2.0: TC 路径的 mma_tile + swizzle 绑定为联合宏节点"""

    def __init__(self, ir_module, gpu_spec):
        self.nodes = []
        self.edges = []   # (src, dst, constraint_type)

        # 公共节点
        self.add_node("block_tile", type="continuous", range=[1, 256])
        self.add_node("global_layout", type="discrete",
                      values=["row_major", "col_major", "padded"])

        if gpu_spec.has_tensor_core and dtype in [fp16, bf16, int8]:
            # Tensor Core: mma_tile + swizzle 绑定为联合宏节点 (v2.0 修正)
            self.add_node("tc_macro", type="discrete",
                          values=[(mma, sw) for mma in gpu_spec.mma_tile_options
                                         for sw in gpu_spec.swizzle_options])
            self.add_node("pipeline_stage", type="discrete",
                          values=[2, 3, 4, 5])

            # 联合宏节点 → Block Tile (divisibility + alignment)
            self.add_edge("tc_macro", "block_tile",
                          constraint="block_tile % mma_tile == 0")
            self.add_edge("tc_macro", "block_tile",
                          constraint="block_tile_col >= swizzle_alignment")

            # Block Tile 的反向约束传递 (FeasibleRegion 机制)
            self.add_edge("block_tile", "tc_macro",
                          constraint="remaining_shmem_after_L1")

            # 联合宏节点 → Pipeline (bank conflict 模式)
            self.add_edge("tc_macro", "pipeline_stage",
                          constraint="pipeline_bounds_from_swizzle")
        else:
            # CUDA Core 自由路径
            self.add_node("warp_tile", type="continuous", range=[1, 128])

        # 公共下游
        self.add_node("unroll", type="discrete", values=[0,1,2,4,8,16])
        self.add_node("vectorize", type="discrete", values=[1,2,4,8,16])

        # 资源约束边
        self.add_edge("block_tile", "warp_tile",
                      constraint="warp_tile <= block_tile")

    def propagate(self, fixed_nodes):
        """从已固定节点传播约束，返回剩余节点的可行区域"""
        region = FeasibleRegion()
        for edge in self.edges:
            if edge.src in fixed_nodes:
                region.apply_constraint(edge.dst, edge.constraint,
                                       fixed_nodes[edge.src])
        return region
```

这种设计的好处：

- 同一套搜索算法（交替优化）在两种约束图拓扑上都能跑
- 新硬件/新指令路径只需添加新节点和约束边
- 不需要为每种路径单独实现搜索逻辑

### 4.4 Tensor Core 路径的搜索策略

由于 mma_tile 只有 3-6 种离散选项，可以**暴力穷举 mma_tile + 固定下游搜索**：

```python
# Tensor Core 路径: 枚举 mma_tile + 交替优化下游
results = []
for mma_tile in gpu.mma_tile_options:          # 最多 6 种
    fixed = {"mma_tile": mma_tile}
    constraint = graph.propagate(fixed)
    # 在 mma_tile 固定的情况下，交替优化 block_tile 和 pipeline
    best_downstream = CGAS(ir_module, gpu, constraint,
                           init_tiers=["block_tile", "pipeline", "swizzle"])
    results.append((mma_tile, best_downstream))

return min(results, key=lambda r: r[1].runtime)
```

因为 mma_tile 只有几种，即使暴力枚举也不会爆炸。这是 Tensor Core 路径的特殊性质——**根节点的离散选择空间极小**，其余部分被强约束锁定。

---

## 5. 解耦型混合代价模型（方向讨论）

### 5.0 v1.0 方案检讨

v1.0 方案设计了 4 个独立层级代价模型（L1-Cost、L2-Cost、L3-Cost、TC-Cost），各自预测层级的局部效率。这个设计存在一个根本性问题：

> 4 个独立轻量模型违背了 GPU 硬件层级间的非线性耦合常识。例如：L3 的寄存器压力会直接拉低 L1 的 Occupancy——两个层级无法独立测算。

当 L1 的 block tile 较大时，每个 block 消耗更多寄存器。在寄存器总量固定的情况下（A100: 65536 registers/SM），剩余寄存器减少意味着 SM 能并发驻留的 warp 数下降 → Occupancy 下降 → 延迟隐藏能力退化。这是一个 **L1 决策 → 寄存器消耗 → Occupancy → L2/L3 可见性能** 的跨层级因果链，4 个独立模型无法捕获。

### 5.1 解耦策略：白盒 + 黑盒

v2.0 将代价评估体系重构为两层：

```
Final_Score = 白盒理论上界 × 黑盒折损率
```

设计原则：

- **容量和带宽是确定性物理量，硬编码比机器学习更准** → 白盒
- **指令排布和微架构延迟是黑盒效应，适合神经网络拟合** → 黑盒
- **黑盒的输入中不包含容量信息** → 强迫模型只关注指令排布，减少对训练数据的依赖

---

#### 5.1.1 白盒部分：Roofline 性能上界（纯分析型，无需训练）

给定一个调度决策（tile 配置 + 指令排布），白盒模型计算该配置的**理论上界**——即如果 GPU 完美执行该配置、没有任何微架构停顿（bank conflict、pipeline bubble、TLB miss 等），能达到的最佳性能。

```
compute_roof = (total_FLOPs × SM_count × GPU_clock) / (FLOPs_per_cycle × ...)
memory_roof = (total_bytes × HBM_bandwidth) / (bytes_per_cycle × ...)
whitebox_upperbound = min(compute_roof, memory_roof) × occupancy_factor
```

输入：

- Tile 配置 (bi, bj, bk, wi, wj, wk)（从中可精确计算数据传输量和计算量）
- GPU 硬件常数（峰值带宽、峰值 TFLOPS、SM 数、时钟频率）
- 从双向剪枝传递过来的 occupancy 估算值

输出：该配置的理论最优延迟（秒或微秒），精度约 ±30%。

白盒模型**不做 ML**：

- 容量检查（共享内存不超限）已经是 100% 精确的
- Roofline 上界在有 profiled 带宽和算力常数时误差可控
- 它给出的 "该配置理想情况下能跑多快" 本身就带排序信息——远高于 Roofline 上限的 tile 必然是低效的

---

#### 5.1.2 黑盒部分：微架构折损率预测（可训练，待后续优化）

白盒给出的只是**理论上界**。实际性能与上界之间的差距来自微架构效应——bank conflict、pipeline bubble、L1/L2 cache hit/miss pattern、warp scheduler 的发射效率等。这些效应的建模需要机器学习。

**输入**（注意：刻意剥离了容量信息）：

- 指令排布特征：Unroll Factor, Vectorize Width, Pipeline Stages
- 访存模式特征：Cooperative Fetch Width, Swizzle Pattern（仅 TC）
- 循环结构特征：Loop Order (ijk/ikj/...), Async Copy Depth
- 算子类型：MatMul / Conv2D / Reduction / Element-wise

**输出**：折损率 ∈ (0, 1]。1.0 = 实际达到白盒上界。

**模型候选**：

- **小 MLP**（<100K 参数）：轻量，推理 <10μs，适合在搜索内循环中使用
- **Pattern-aware Transformer**：将 tile 参数和指令序列视为 token，用 attention 捕获跨层级的微架构交互。训练开销大，推理开销可控（~100μs），作为未来的升级方向

**训练数据**：从 micro-benchmark 采集（沿用 v1.0 §5.2 的设计）。因为黑盒输入中不含量信息（共享内存大小、寄存器数等），同样的"Unroll=4, Vec=4, Pipeline=3"在不同 tile 大小下的折损率应该相似——这极大减少了训练数据需求量。

---

#### 5.1.3 融合打分

融合时的关键：白盒负责"上限校准"。

```
# 即使黑盒预测不准（某区域训练数据稀疏），最终分数也不超过 Roofline 上界
predicted_latency = whitebox_upperbound / max(blackbox_efficiency, 0.01)
final_score = 1.0 / predicted_latency  # 越高越好
```

此外，融合时引入一个交叉项，捕获"高带宽消耗 + 大 Unroll"的交互效应：

- 当 `memory_roof < compute_roof * 0.8`（访存瓶颈）且 `unroll > 4` → 额外扣分，因为大 Unroll 在访存瓶颈下不产生收益，反而增加寄存器溢出风险

---

### 5.2 训练数据采集（沿用 v1.0，微调）

采集策略与 v1.0 保持一致：合成 micro-benchmark 独立采集，每个层级 10K-20K 样本。与 v1.0 不同的是：黑盒模型的训练标签是**折损率**（实际延迟 / 白盒上界），而非绝对延迟。这使得训练标签的值域稳定在 (0, 1]，训练更易收敛。

---

### 5.3 待优化项（后续迭代讨论）

以下问题当前方案不做具体设计，留待后续研究：

1. **白盒 Roofline 的精度提升**：当前 ±30% 的精度受 DRAM bank conflict、L2 hit rate 等因素影响。能否通过 profiling 标定每个 GPU 型号的修正因子（如有效带宽 = 峰值带宽 × 0.72 for A100）？
2. **黑盒折损率的跨代际迁移**：在 A100 上训练的折损率预测模型，能否通过少量 H100 样本做迁移学习？折损率的物理含义（微架构效率）比绝对延迟更容易跨代际泛化。
3. **白盒 + 黑盒的融合是否需要非线性项**：当前是纯乘法。是否存在"大带宽 + 大 Unroll"的交互效应需要额外的交叉项？
4. **黑盒模型在搜索循环中的推理开销**：小 MLP (<10μs) 可用于内循环排序。搜索每轮约评估 20-60 个候选（双向剪枝后），ML 推理总开销 < 1ms——远小于 GPU 测量开销（~1-3s per candidate）。

---

### 5.4 v1.0 vs v2.0 代价模型对比

| 维度        | v1.0 分层代价模型                | v2.0 解耦混合模型                      |
| ----------- | -------------------------------- | -------------------------------------- |
| 模型结构    | 4 个独立 XGBoost/MLP             | 1 个白盒公式 + 1 个黑盒 MLP            |
| 层级耦合    | 假设各层独立（违反物理）         | 白盒天然处理跨层级容量耦合             |
| 训练数据    | 每层 10-20K 样本                 | 黑盒只需 10-20K 样本（标签值域更稳定） |
| 跨 GPU 泛化 | 需每 GPU 重训练                  | 白盒零训练；黑盒可迁移学习             |
| 上限校准    | 无（模型可能输出物理不可能的值） | 白盒做硬性上界截断                     |
| 推理开销    | 4 次推理                         | 白盒 <1ns + 黑盒 <10μs                |

---

## 6. 理论分析

### 6.1 交替优化的收敛性

**命题 1（局部收敛）**：在每层搜索能找到自身局部最优的假设下，交替优化在有限步内收敛到一个局部最优解 $(d_1^*, d_2^*, d_3^*)$，其中任意单层参数的改变都不再改善目标函数。

**证明思路**：交替优化可视为在约束图上做坐标下降（coordinate descent）。因为目标函数 $f(d_1, d_2, d_3)$ 在每个坐标方向上的搜索都单调不增，且搜索空间有限（所有参数都是离散或离散化的），算法在有限步内收敛。

**命题 2（收敛速率）**：如果目标函数满足 Polyak-Łojasiewicz 条件（$f(x) - f(x^*) \leq \frac{1}{2\mu} \|\nabla f(x)\|^2$），交替优化线性收敛：$f(x^k) - f(x^*) \leq \rho^k (f(x^0) - f(x^*))$，其中 $\rho = 1 - c \cdot \mu$。

GPU kernel 运行时间通常不满足 PL 条件（离散参数、平台效应）。但实证上，交替优化在 $K \leq 5$ 轮内稳定收敛——这需要在实验中验证。

### 6.2 约束传播的保优性

**命题 3（约束保守性）**：`PropagateConstraints` 返回的 `FeasibleRegion` 是实际可行区域的超集——即所有被排除的组合都在真实硬件上确实不可行。

**证明**：约束传播只使用硬件容量上限（共享内存/寄存器/occupancy），这些是硬性物理限制。传播时不引入任何假设或近似，因此不会错误排除可行解。

**推论**：分层搜索不会因约束传播而错过全局最优解（即不会引入 false negative 剪枝）。

### 6.3 回溯的修正率

回溯机制在局部最优陷入约束边界时触发。可以证明：

**命题 4（回溯的充分性）**：如果全局最优解 $(d_1^*, d_2^*, d_3^*)$ 被约束边界排除，则至多 2 次回溯即可发现。

**证明思路**：每次回溯将约束边界扩大 50%。因为约束过紧最多影响 1-2 个维度（GPU 的瓶颈维度有限），2 次回溯后的约束区域足以覆盖全局最优。

---

## 7. 详细实施计划

### 阶段 1：独立性量化验证（第 1-3 周）

**目标**：用 DoE 方法验证 "层级间耦合主要通过约束传播" 的假设。

**实验设计**：对每个算子（matmul, conv2d, reduction），采样 30+ 组 L2/L3 配置（正交采样），对每组扫描 L1 空间，计算 L1 排名的 Kendall τ 一致性：

```
对 matmul/conv2d/reduction:
    对 30 组 (L2, L3) 配置:
        扫描 L1 的 200 个 tile 组合
        测量每个 (L1, L2, L3) 的运行时间
    计算: 对每个 L2+L3 组合, L1 性能排名的 pairwise Kendall τ
    报告: τ 的中位数和方差
```

**判据**：如果 Kendall τ ≥ 0.8 → L1 排名在 L2/L3 变化下高度一致 → 分层搜索可行。如果 τ ≤ 0.5 → 假设不成立 → **需要重新设计方案**。

**产出**：DoE 验证报告 + Go/No-Go 决策。

### 阶段 2：约束图与搜索框架（第 4-7 周）

| 任务                                    | 工作量 | 关键文件                   |
| --------------------------------------- | ------ | -------------------------- |
| DecisionGraph + FeasibleRegion 数据结构 | 3 天   | `constraint_graph.h/cc`  |
| PropagateConstraints 约束传播引擎       | 3 天   | `constraint_graph.h/cc`  |
| 分层空间生成器（含事前约束剪枝）        | 4 天   | `hierarchical_space.cc`  |
| CUDA Core 路径的 CGAS 实现              | 4 天   | `cgas_search.cc`         |
| Tensor Core 路径的 CGAS 实现            | 3 天   | `cgas_search.cc`         |
| 层级感知变异器                          | 2 天   | `tiered_mutator.cc`      |
| Python 接口 + 单元测试                  | 3 天   | `hierarchical_search.py` |

### 阶段 3：层级代价模型（第 8-10 周）

| 任务                                 | 工作量 |
| ------------------------------------ | ------ |
| Micro-benchmark 自动生成脚本         | 3 天   |
| L1-Cost 训练（带宽利用率）           | 2 天   |
| L2-Cost 训练（bank conflict）        | 2 天   |
| L3-Cost 训练（IPC）                  | 2 天   |
| TC-Cost 训练（pipeline 延迟隐藏）    | 2 天   |
| 对比实验：分层模型 vs 端到端 XGBoost | 3 天   |

### 阶段 4：端到端评估（第 11-15 周）

| 实验                 | 硬件                     | 模型                        |
| -------------------- | ------------------------ | --------------------------- |
| 主实验：搜索效率对比 | H100, A100, MI250        | ResNet-50, BERT, Llama      |
| Tensor Core 专门评估 | H100, A100               | fp16 matmul, bf16 attention |
| 消融实验             | A100                     | 逐组件消融                  |
| 跨架构泛化           | A100 → T4, H100 → A100 | 同一算子                    |
| Wall-clock 时间分析  | A100                     | 编译/测量/推理 breakdown    |

### 阶段 5：论文撰写（第 16-20 周）

---

## 8. 实验设计

### 8.1 主实验：搜索效率 —— 可行域搜索 vs. 拒绝采样

核心叙事：本方案将搜索从 Meta Schedule 的"拒绝采样"（事后过滤）转变为"可行域搜索"（事前约束），在相同 wall-clock 时间内将更多测量预算投入有效候选空间的探索。

| 方法                          | 描述                                              | 搜索空间类型         |
| ----------------------------- | ------------------------------------------------- | -------------------- |
| Meta Schedule (Evolutionary)  | 扁平笛卡尔积 + 事后 Postproc 过滤                 | 拒绝采样             |
| Ansor                         | TVM 上一代自调度器                                | 层次化（维度假分层） |
| CUTLASS Profiler (Oracle)     | 手工模板的参数暴力搜索                            | 手工分层（性能上限） |
| Triton Auto-Tuner             | Triton 的自动调优                                 | 程序员分层           |
| **CGAS w/ XGBoost**     | 分层空间 + 单一 XGBoost（消融用，无层级代价模型） | 可行域搜索           |
| **CGAS w/o 向下剪枝**   | 仅向上剪枝（v1.0 级），消融向下剪枝的增益         | 可行域搜索（部分）   |
| **CGAS（本方案 v2.0）** | 双向剪枝 + 解耦混合代价模型 + Top-2 Beam          | 可行域搜索（完整）   |

**核心图**：7 条搜索效率曲线（X 轴 = wall-clock 时间，Y 轴 = 当前最优加速比）。

**核心指标**：

- 达到 CUTLASS oracle 95% 性能所需的 wall-clock 时间
- 500 次测量 vs 4000 次测量的最终加速比
- Wall-clock 时间 breakdown（编译/测量/代价模型推理）
- 候选有效率：数据库记录中 `run_secs` 为非哨兵值的比例（Meta Schedule ~46.5%，CGAS 预期 ~100%）

### 8.2 消融实验

| 实验                           | 变量                     | 回答的问题                         |
| ------------------------------ | ------------------------ | ---------------------------------- |
| 分层空间 vs 扁平空间           | 只用 XGBoost，变空间结构 | 分层空间本身的收益                 |
| **向上剪枝 vs 双向剪枝** | CGAS 含/不含向下剪枝     | **v2.0 新增** 性能地板的价值 |
| 解耦混合 vs 单一代价           | 变代价模型结构           | 白盒 + 黑盒解耦的增益              |
| Beam Top-2 vs Top-1            | CGAS 含/不含 Beam Search | 防贪心陷阱是否必要                 |
| 交替 vs 贪心                   | CGAS 含/不含外层循环     | 交替优化是否优于贪心               |
| 回溯 vs 不回溯                 | CGAS 含/不含回溯         | 回溯机制的价值                     |
| 事前约束 vs 事后过滤           | 约束检查时机             | 事前剪枝 vs Postproc 过滤          |

### 8.3 GPU 覆盖

| GPU         | 架构   | 代际     | Tensor Core      |
| ----------- | ------ | -------- | ---------------- |
| NVIDIA H100 | Hopper | 最新     | ✓ (第四代)      |
| NVIDIA A100 | Ampere | 前一代   | ✓ (第三代)      |
| NVIDIA T4   | Turing | 较老     | ✓ (第一代)      |
| AMD MI250   | CDNA2  | 竞争架构 | ✗ (Matrix Core) |

覆盖3代 NVIDIA + 1个 AMD，确保跨架构泛化评估有说服力。

### 8.4 算子覆盖

| 算子                 | 路径                      | 类型             |
| -------------------- | ------------------------- | ---------------- |
| MatMul (fp32)        | CUDA Core                 | 基础线性代数     |
| MatMul (fp16)        | Tensor Core (wmma/mma)    | AI 训练核心      |
| Conv2D (fp32)        | CUDA Core (implicit gemm) | 视觉模型         |
| Conv2D (fp16)        | Tensor Core               | 视觉模型加速     |
| FlashAttention-style | Tensor Core               | LLM 核心         |
| LayerNorm + Residual | CUDA Core (fused)         | Transformer 瓶颈 |
| Reduction (fp32)     | CUDA Core                 | 归约算子代表     |

---

## 9. 预期成果与创新点

### 9.1 论文贡献

1. **约束图建模**：首次将 GPU 调度的搜索空间显式建模为带资源约束的有向图，统一处理 CUDA Core 和 Tensor Core 两种路径
2. **CGAS 算法**：约束引导的交替搜索，用 FeasibleRegion 而非固定参数在层级间传递信息
3. **层级代价模型**：四个层级各自预测局部效率，避免端到端代价模型的结构混淆
4. **理论分析**：交替优化的收敛性、约束传播的保守性、有限回溯的充分性
5. **全面实验**：3 代 NVIDIA GPU + AMD，CUDA Core + Tensor Core，7 种算子

### 9.2 预期定量结果

| 指标                                    | 期望值                          |
| --------------------------------------- | ------------------------------- |
| Wall-clock 搜索时间（vs Meta Schedule） | -60% ~ -75%                     |
| 最终加速比（同等时间预算）              | 持平或略优 (+5%)                |
| 无效候选率（vs Meta Schedule ~60%）     | < 15%                           |
| 层级代价模型 Spearman ρ                | L1: >0.85, L2: >0.80, L3: >0.75 |
| 跨 GPU 代际迁移的 Kendall τ            | >0.70                           |

---

## 10. 风险与对策

| 风险                          | 概率     | 影响 | 缓解措施                                                |
| ----------------------------- | -------- | ---- | ------------------------------------------------------- |
| DoE 验证 τ < 0.5             | 中 (35%) | 致命 | 方案降级为 "软分层"：保留分层空间，放弃代价模型分离     |
| Tensor Core 路径开销超预期    | 中 (30%) | 严重 | 优先验证 CUDA Core 路径，TC 作为扩展实验                |
| GPU 资源不足（多代 GPU）      | 中 (40%) | 严重 | 利用云 GPU（Lambda Labs/VAST）；H100 按需租用           |
| CGAS 外循环次数 K 实际 > 期望 | 低 (25%) | 中等 | 用早期停止 + 递减预算策略限制外循环开销                 |
| 层级代价模型精度不如预期      | 中 (25%) | 中等 | 混合使用层级代价 + 真实测量验证（每层仍保留验证性测量） |
| AMD GPU 的 CUDA→ROCm 移植    | 中 (35%) | 中等 | 优先用 HIP 的兼容层；若不可行则降为只用 NVIDIA          |

---

## 11. 时间线

```
Week  1-3:  ████████ 阶段 1: DoE 验证 + 分层假设检验
                          ↖ Week 3: Go/No-Go 决策点

Week  4-7:  ████████████████ 阶段 2: 约束图 + CGAS 框架

Week  8-10: ████████████ 阶段 3: 层级代价模型训练
                          ↖ Week 10: 代价模型决策点

Week 11-15: ████████████████████ 阶段 4: 端到端评估
                          ↖ Week 15: 结果冻结

Week 16-20: ████████████████████ 阶段 5: 论文撰写
                          ↖ Week 20: 投稿
```

---

## 附录：与方案 A/B 的关系

三个方案形成递进的研究线：

```
方案 C (分层搜索) ← 最务实，工程可控
    │            ← 解决 "搜得慢" 的问题
    │
    ├─→ 方案 A (LLM 代价模型)
    │   用 Transformer 替代 XGBoost 做层级代价预测
    │   方案 C 的分层结构使方案 A 的训练数据更容易采集
    │
    └─→ 方案 B (LLM Agent 搜索)
        用 Agent 做约束图中的策略决策
        ("L1 在边界附近，应该回溯还是接受？")
        方案 C 的约束图为 Agent 提供了可推理的结构
```

**推荐策略**：先发方案 C（夯实基础），后续工作扩展 A 或 B。

---

## 12. 附录：v2.0 修订记录

| 版本 | 日期    | 修订内容                                |
| ---- | ------- | --------------------------------------- |
| v1.0 | 2025-10 | 初版：CGAS 算法 + 分层代价模型 + 约束图 |
| v2.0 | 2026-06 | 五项重大修订，详见下方                  |

### v2.0 修订摘要

| # | 节    | 修订类型 | 内容                                                                                                                                                              |
| - | ----- | -------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1 | §1.1 | 叙事重写 | 引入"拒绝采样 vs. 可行域搜索"的对立框架，替换模糊的"约60%无效"为实测 Postproc 数据                                                                                |
| 2 | §3.1 | 算法增强 | CGAS 内循环增加 Top-2 Beam Search（保守高 Occupancy + 激进大 Tile），MCTS 写入未来工作                                                                            |
| 3 | §3.4 | 完整替换 | 搜索空间生成从单向容量剪枝升级为**双向剪枝**（向上硬件天花板 + 向下性能地板），新增 Occupancy 保底、Block/Warp 整除性、数据复用度三个地板条件及统一剪枝架构 |
| 4 | §4.2 | 拓扑修正 | 修正 Tensor Core 约束图中 Swizzle 的因果倒置：`mma_tile + Swizzle` 绑定为联合宏节点，Swizzle 不再是被动叶子，可向上反推 Block Tile 对齐要求                     |
| 5 | §5   | 完整重写 | 从 4 个独立层级模型重构为**解耦型混合代价模型**（白盒 Roofline 上界 + 黑盒折损率 MLP），定性讨论设计方向和待优化项                                          |
| 6 | §8   | 实验更新 | Baseline 表增加"CGAS w/o 向下剪枝"消融行；消融实验增加双向剪枝、Beam Search、混合代价模型三项                                                                     |

### v2.0 与 v1.0 的关键数值变化

| 参数               | v1.0         | v2.0            | 变化原因                               |
| ------------------ | ------------ | --------------- | -------------------------------------- |
| L1 搜索空间        | ~200 候选    | ~20 候选        | 向下剪枝追加三个地板条件               |
| 候选有效率（预期） | ~45%         | ~100%           | 双向剪枝在生成阶段排除无效候选         |
| 外层循环 K_max     | ≤5          | ≤5（不变）     | Beam Search 增加了内循环开销但外层不变 |
| 代价模型数量       | 4 个独立模型 | 1 白盒 + 1 黑盒 | 解耦后不再按层级分模型                 |
| 预估周期w          | 5-6 个月     | 6-7 个月        | 双向剪枝实现 + Beam Search 增加约 3 周 |
