# 约束引导 MetaSchedule：研究方案、实现与实验说明

> 更新日期：2026-09-05
> 项目目录：`F:\tvm`
> 当前分支：`pruning-experiment`
> 基础提交：`4a1297ad59c9ff80de535a8ba898bf6d9012e5c7`
> TVM 版本：`0.26.dev90`
> 当前阶段：CS/H、修正后的 Draft-GA、G1/G2/G4 已完成 MatMul128 单 seed、16-trial CUDA 实验；G3 已完成 64-trial 真实 Runner 反馈实验，但直到最后一次反馈后才越过冷启动，冷启动后 enforce 尚未实际验证；尚未形成跨算子、多 seed、大预算的正式性能结论。

本文统一说明当前研究为什么做、准备解决什么问题、各模块如何设计和实现、已经进行过哪些
实验、目前取得了哪些可靠成果、哪些结果仍不能下结论、下一步还必须完成哪些实验，以及
所有代码和数据在仓库中的位置。

本文只把仓库中已经存在并验证过的内容写成“已实现/已完成”。第 2～12 节保留了
2026-09-03 以前的研究演进和负向实验；**第 13 节是 2026-09-05 的最新状态，若旧章节与其
冲突，以第 13 节及原始 manifest 为准**。G3 的反馈闭环已经实测，但自适应 enforce、
shared-memory 投影和多 seed 结论仍明确标为未完成，避免把研究计划和当前成果混为一谈。

---

## 1. 摘要

### 1.1 研究问题

TVM MetaSchedule 在搜索 GPU 调度时会生成大量候选。候选经过 trace replay 和一串
postprocessor 后，由 `VerifyGPUCode` 检查线程数、线程契约、shared memory 等硬件约束。
当前流程的主要问题是：很多明显非法的候选直到较晚阶段才被拒绝；GA 变异器也不了解
可行域，会持续产生相似的非法子代；在实测样本很少时，XGBoost 代价模型的排序能力又
不足。

本研究的核心思想是把同一类硬件/结构信息前移并复用：

```text
CS：初始化候选在可证明的线程可行域内采样
    ↓
H：候选 replay 前/后只拒绝可证明非法者
    ↓
Draft-LSE：Draft 是 GA 内唯一 fitness
    ├─ G2 越界投影
    ├─ G1 资源余量方向 + Draft 多 proposal 择优
    └─ G4 相对差退火
    ↓
跨代 Draft top-K
    ↓
XGBoost 仅在每批 GA 结束后精排一次
    ↓
最高分候选 build + GPU run
    ↓
G3 用真实父子延迟更新 mutator 账本（反馈已实测，自适应选臂待验证）
```

目标不是简单增加一个过滤器，而是形成一套“可行性知识贯穿搜索过程”的 MetaSchedule
轻量改造：确定非法的尽早排除，性能不确定的只软排序，变异尽量不再制造同类垃圾，代价
模型只在可行区域内学习。

### 1.2 当前模块状态

| 模块              | 当前状态       | 已验证内容                                                              | 尚缺内容                                         |
| ----------------- | -------------- | ----------------------------------------------------------------------- | ------------------------------------------------ |
| I0 可观测性       | 已实现         | 分阶段计数、Verify 失败原因、时间、源码/DLL 身份、计数守恒              | 更完整的候选流 JSONL                             |
| CS/H 可行域与硬筛 | 已实现         | 初始化约束采样、trace/post-replay/late 检查、unknown 放行、单 seed 配对 | 多算子/多 seed；安全 shared-memory 判据          |
| S 旧软过滤        | 已实现原型     | thread-only penalty、shadow/rerank、候选数保持                          | 它不是完整 Pruner，也没有进入模型训练特征        |
| Draft-LSE         | 已实现并实测   | Draft 独立指导 GA；XGBoost 退出 GA、末端每批一次 top-K 精排             | 分数校准、跨算子排序质量                         |
| G1/G2             | 已实现并实测   | 单轴线程越界投影、资源余量方向、多个合法 proposal 的 Draft 择优         | 多轴、shared 资源、降低四 proposal replay 开销   |
| G4                | 已实现并实测   | 相对差退火、shadow/enforce、温度衰减及接收计数                          | 多 seed 稳定性；与完整组合的正收益               |
| G3                | 反馈闭环已实测 | lineage、真实父子延迟、滑窗 UCB1、shadow/enforce 日志                   | 冷启动后真实 enforce；不可用 arm 的无限 UCB 修正 |
| L 温度相对标签    | 未实现         | 已确认 TVM 基线本来就用相对标签                                         | `gamma` 参数、离线学习曲线和端到端消融         |

### 1.3 当前最重要的实验结论

1. 早期 `vk != 1` 硬规则已经被证伪：40 个合法候选中会误杀 15 个，误杀率 37.5%。
   这条规则已经退出研究主线。
2. 改进后的线程硬筛在 606 条历史记录、409 条有限实测记录上没有观察到有限实测误拒；
   在 H2 shadow 中对整个 Verify 混合失败集合的覆盖从旧版 0.59% 提高到 80.20%，样本内
   precision 为 100%。
3. 线程硬筛的分析成本很低，但当前插入位置与 `VerifyGPUCode` 自带的早期
   `ThreadExtentChecker` 高度重叠。shadow 搜索 149 s、enforce 搜索 150 s，因此目前没有
   可靠的硬筛时间加速证据。
4. lowering 前的 shared buffer 声明大小不能作为硬过滤：409 条合法有限实测记录中会误判
   397 条。该 proxy 已经被证伪，只保留观察统计。
5. Pruner 草稿模型已经真正改变 top-k，单候选分析约 59 us；但 36 个唯一实测 trace 上的
   Spearman 仅为 0.314，只有 11 个分数档位，草稿头部存在明显误排。
6. 最新单 seed 四臂实验中，P 单臂找到 33.108 us 的 best，优于 baseline 的 49.038 us；
   但 H+P 为 34.784 us，且搜索时间比 P 单臂慢 8.70%。这只能视为 P 的单 seed 积极信号，
   不能声称稳定增益或组合增益。
7. GA 原型能够修复一部分线程越界 proposal，但触发频率低，尚未达到完整“方向性、
   可行性保持 GA”的目标。
8. 2026-09-05 修正后的 Draft-GA 已验证目标架构：两批搜索的 GA 内 XGBoost 调用/输入
   为 `0/0`，末端调用/输入为 `2/64`；原 baseline 为 `10/1280`，输入减少 95%。
9. 同一轮 D_G1G2 中，已知越界 proposal `6/6` 投影成功，1992 次资源方向选择共产生
   8015 个 hard-feasible divisor；但搜索从 D 的 130 s 增至 227 s，方向枚举开销过大。
10. G4 enforce 的劣解接收路径已真实触发，本 seed 的独立 D_G1G2 臂 best 从
    31.682 降至 29.784 μs，但完整 CS/H 组合加 G4 后 best 从 29.119 变为 35.662 μs。
    结果混合，不能宣称稳定提升。
11. 上述六臂结果仍只有 MatMul128、seed 20260905、16 trials。后续 G3 另完成了同 workload、
    同 seed 的 64-trial shadow/enforce 运行，验证了 Runner 反馈闭环，但未产生冷启动后的
    enforce generation；这些结果都不构成多算子或统计意义上的性能结论。

---

## 2. 为什么要做这项研究

### 2.1 MetaSchedule 当前的成本链

GPU MetaSchedule 候选的实际处理过程是：

```text
生成 trace
  → replay 成 Schedule/IR
  → 多个 postprocessor
  → VerifyGPUCode
  → 通过后才 build
  → GPU run
  → 更新数据库和代价模型
```

必须特别说明：被 `VerifyGPUCode` 拒绝的候选不会进入 build/run。因此研究的直接收益不是
“让一个非法候选少编译一次”，而是减少非法候选已经消耗或将要消耗的 replay/postproc
时间。只有当搜索更有效、达到目标性能所需轮数下降时，才可能间接减少总测量和总
build/run 次数。

这一区分决定了实验必须同时报告：

- search-only 时间；
- replay/postproc/Verify 分段时间；
- build/run 时间；
- 达到目标性能所需的测量数和墙钟时间；
- 最终 kernel 质量。

不能只看一次端到端时间，也不能把所有时间变化都归因于过滤。

### 2.2 实测暴露的浪费

在当前 MatMul 128³、RTX 3050 Ti、population 512、GA 4 代的 H2 shadow 运行中：

```text
生成候选：5089
VerifyGPUCode 失败：1419
  thread_extent：1138
  shared_memory：231
  lowering_exception：50
最终进入测量：16
```

按 Verify 失败计：

- `thread_extent` 约占 80.20%；
- `shared_memory` 约占 16.28%；
- `lowering_exception` 约占 3.52%。

这说明当前 workload 的大量搜索工作用于反复生成并拒绝线程契约不满足的候选。但“失败
数量多”不自动等于“提前检查一定加速”，因为正式 `VerifyGPUCode` 对 thread extent 的
检查本身已经处在 lowering 之前。必须用 shadow/enforce 的实际时间对照判断净收益。

### 2.3 三个相互关联的问题

#### 问题一：过滤太晚

如果候选在 replay 或多个 postproc 之后才被确认非法，已经发生的工作无法收回。真正有
价值的过滤应该尽可能靠近决策生成点，同时保持“只在确定非法时拒绝”。

#### 问题二：GA 不知道可行域

原始 `MutateTileSize` 随机选择 tile 因子进行一除一乘。即使父代合法，变异后仍可能让
线程数或 shared memory 越界。后处理再拒绝，只是在重复“造垃圾—验垃圾—扔垃圾”。

#### 问题三：小样本代价模型缺少物理先验

XGBoost 在实测样本不足时预测近似随机，且必须从数据中间接学习 warp 利用率、并行度、
内存瓶颈等规律。Pruner 的解析模型提供了一个低成本、可解释的冷启动先验，但它只能
判断“可能快慢”，不能判断“是否合法”。

### 2.4 为什么必须拆分硬筛和软打分

硬约束和性能启发式具有不同语义：

| 类型       | 示例                                       | 处理方式               | 正确性要求                 |
| ---------- | ------------------------------------------ | ---------------------- | -------------------------- |
| 硬合法性   | threads 超过上限；确定违反 thread contract | 允许直接拒绝           | 历史合法记录误拒必须为 0   |
| 软性能信号 | warp 未填满；occupancy 低；算存比不理想    | 只能降分或作为模型特征 | 允许不准确，但不得称为非法 |

低 occupancy 的 kernel 仍可能因为 ILP、cache 或 Tensor Core 使用而很快。因此 occupancy、
warp 利用率、访存强度绝不能升级为硬拒绝条件。

---

## 3. 研究命题与相关工作定位

### 3.1 四个可证伪假设

当前实施方案把研究拆成四个假设：

- **H1——硬过滤**：静态确定违反硬件/线程契约的候选可以被安全提前拒绝，历史合法
  数据库误拒为 0，并带来正的净时间收益。
- **H2——Pruner 解析先验**：低成本解析评分能够改善冷启动排序、top-k 或
  time-to-target；它只做软选择，不承担合法性判断。
- **H3——方向性 GA**：根据资源违反类型和资源余量选择变异/投影方向，可以提高合法、
  唯一后代比例，并减少达到相同性能所需的代数和测量数。
- **H4——标签校准**：温度化相对标签能改善小样本排序；若跨 workload 不稳定，则不进入
  最终系统。

其中 H 和 G 是拟作为研究主体的两个支柱；P 是明确引用 Pruner 的外部解析先验；L 是支撑
性消融，不应包装为主要原创点。

### 3.2 与 Pruner 的关系

Pruner 的核心思想是 draft-then-verify：先用廉价解析模型粗排，再让更昂贵的模型或实测
处理少量候选。本项目复用了其公开 PSA 风格特征和 latency 公式的思路，并在 TVM S-TIR
上重新实现特征提取。

本项目当前与 Pruner 的区别是：

- P 模块是 Pruner 风格软打分，明确引用，不宣称公式原创；
- H 模块使用可证明的合法性条件，承诺零误拒，不由 P 分数阈值化得到；
- G 模块尝试让 TVM 现有变异器回到可行域，而不是只在变异后打分；
- 当前只完成 P1——在已有合法 population 的 XGBoost 分数后融合，尚未完成真正位于
  昂贵 postproc 前的 P2。

### 3.3 与 Heron 的关系

Heron 已经研究过约束保持的进化搜索，因此不能声称“首次让 GA 不产生非法子代”。本项目
准备强调的差异是实现路线：

| Heron                               | 本项目                                                  |
| ----------------------------------- | ------------------------------------------------------- |
| 将搜索空间重建为 CSP，在 CSP 上进化 | 对 MetaSchedule 现有搜索器和 mutator 做轻量 retrofit    |
| 约束属于其自有空间定义              | 尝试从 trace/S-TIR 结构机械提取约束和资源影响关系       |
| 需要切换搜索框架                    | 保留 MetaSchedule 现有 schedule rules、搜索空间和数据库 |

一句话概括：Heron 是构造一个天生守约束的空间，本项目是让现有 MetaSchedule 搜索逐步
学会守约束。

---

## 4. 当前系统怎样工作

### 4.1 I0：失败原因和候选守恒

为了判断优化到底省了什么，首先扩展了可观测性：

- 按 `PickBestFromDatabase`、`SampleInit`、`Evolve` 分开统计；
- 记录生成数、trace replay 失败、每个 postproc 失败、硬筛拒绝和通过数；
- 将 Verify 失败细分为 `thread_extent`、`threads_per_block`、`shared_memory`、
  `local_memory`、`vthread`、`vector_bytes`、`kernel_count`、`lowering_exception` 和
  `unknown_verify_failure`；
- 记录硬筛分析时间和 `VerifyGPUCode` 时间；
- 检查守恒式：

```text
generated
= passed
+ trace_replay_failed
+ Σ postproc_failed
+ hard_rejected
```

- 每个实验保存 git HEAD、tracked/untracked source identity、DLL 路径与 SHA256、Target、
  seed、超参数、trace hash、逐条 latency。

没有这层可观测性，Verify 失败原因、硬筛覆盖率和搜索时间变化都无法可靠归因。

### 4.2 H：线程硬过滤 v2

#### 当前插入位置

当前 H 位于 `ThreadedTraceApply` 内、正式 `VerifyGPUCode` postprocessor 执行前。此时 trace
已经完整 replay，位于 `VerifyGPUCode` 之前的其他 postprocessor 也已经执行。因此它属于
H-late，而不是真正的 trace 生成期 H-early。

#### 当前判据

分析器沿每个 PrimFunc 的活跃 AST 路径计算 `threadIdx.x * threadIdx.y * threadIdx.z`，并
检查：

```text
actual_threads < meta_schedule_thread_extent_low_inclusive
actual_threads > meta_schedule_thread_extent_high_inclusive
actual_threads > target.max_threads_per_block
```

分别记录：

```text
thread_extent_low
thread_extent_high
threads_per_block
```

它支持：

- `For(kThreadBinding)`；
- `AttrStmt(thread_extent)`；
- `SBlock(warp_execution)` 隐含的一个 warp；
- x/y/z 多物理轴；
- 多 PrimFunc；
- 边界和整数乘法溢出保护。

若遇到动态 extent、缺失/冲突注解、不支持的 binding 或无法证明的结构，则标记 unknown 并
放行。这里的原则是“宁可漏掉，也不猜测拒绝”。

#### 三种模式

```text
TVM_PRUNE_MODE=0  off：不检查
TVM_PRUNE_MODE=1  shadow：计算预测并与 Verify 结果对照，但不拒绝
TVM_PRUNE_MODE=2  enforce：确定违反线程条件时提前返回 nullopt
```

#### shared-memory proxy

当前分析器还记录 lowering 前声明的 shared buffer 字节数，用来观察它是否能预测后续
shared-memory 失败。但这个值不参与 enforce。实验已证明 postproc 中的 compact/merge
会显著改变实际 allocation，声明大小不是安全下界或精确值。

### 4.3 S：早期 thread-only 软过滤

在完整 Pruner 草稿接入前，实现过一个很小的 thread-only 性能 proxy：

```text
penalty
= geometric_mean(
     lane_utilization,
     warp_scheduler_utilization,
     thread_occupancy_proxy)

adjusted_score = xgb_score * penalty^weight
```

控制开关为：

```text
TVM_SOFT_FILTER_MODE=0  off
TVM_SOFT_FILTER_MODE=1  shadow
TVM_SOFT_FILTER_MODE=2  rerank
TVM_SOFT_FILTER_WEIGHT  默认 0.2
```

它不删除候选，只保持候选向量长度并改变分数顺序；unknown 或非 sm_86 target 使用中性
penalty 1.0。它只包含线程相关 proxy，不是完整 Pruner，也没有修改 XGBoost 的训练特征或
标签。现在它主要保留为历史消融和简单对照，研究主线中的软先验是 P 模块。

### 4.4 P：Pruner 风格解析草稿 P1

#### 输入特征

草稿遍历 postprocessed S-TIR，在 `BufferStore` 周围收集 11 项 PSA 风格特征：

1. 每线程浮点工作量；
2. 总计算工作量；
3. register usage proxy；
4. memory workload proxy；
5. shared-memory footprint proxy；
6. DRAM workload proxy；
7. data-reuse penalty；
8. 是否为 GPU statement；
9. grid size；
10. block size；
11. virtual-thread size。

动态循环或不支持的 GPU binding 会使该候选返回 unknown，而不是生成看似精确的分数。

#### 显式硬件 profile

模型使用：峰值 GFLOPS、显存带宽、SM 数、每 SM active block 数、SM partition 数、warp
size 和 register capacity。当前 RTX 3050 Ti 开发 profile 为：

```text
peak_gflops = 10752
global_memory_gbps = 192
sm_count = 20
active_blocks_per_sm = 1
sm_partitions = 4
warp_size = 32
register_capacity = 128
```

关键硬件参数必须显式提供；缺失时整个草稿返回 unknown 和中性分数，公式内部不静默
写死 `sm_86`。

#### 解析 latency 公式

对每个 GPU statement，草稿估计：

```text
warp_penalty      = raw_warps / ceil(raw_warps)
peak_penalty      = 实际调度单元 / 完整 wave 容量
register_penalty  = max(register_usage / register_capacity, 1)
compute_throughput
  = peak_gflops * peak_penalty * warp_penalty / register_penalty

compute_time = compute_workload * reuse_penalty / compute_throughput
memory_time  = dram_workload / global_memory_bandwidth
statement_latency = max(compute_time, memory_time)
```

所有 GPU statement 的 latency 相加，最终：

```text
draft_score = 1 / total_latency
```

分数越高表示解析模型认为候选越快。

#### 与 XGBoost 的融合

当前 population 内先按最大已知草稿分数归一化：

```text
normalized_draft = draft_score / max_known_draft_score
```

然后：

```text
fused_score = xgb_score * normalized_draft^beta
```

本轮 `beta=0.2`。unknown 的 multiplier 为 1.0，因此不会因为无法分析而被误罚。由于
`normalized_draft <= 1`，当前草稿本质上是一个降权器：它保留草稿最看好的 XGBoost
分数，压低草稿认为较差的候选。

模式为：

```text
TVM_DRAFT_MODE=0  off
TVM_DRAFT_MODE=1  shadow：记录假想融合和 top-k overlap，实际仍返回 XGBoost 分数
TVM_DRAFT_MODE=2  rank：实际返回 fused_score
TVM_DRAFT_BETA    融合强度
TVM_DRAFT_EPS     rank 模式的随机探索比例
```

P1 发生在已有 population 的代价模型预测处。候选在到达这里之前已经经过 replay 和
postproc，所以 P1 可以改善候选排序或 time-to-target，但不能宣称减少 population 之前的
Verify 工作。

### 4.5 G：当前 GA 可行性投影原型

当前只改造 `MutateTileSize` 的一条路径。它从 flat trace 追踪：

```text
SamplePerfectTile
  → Split
  → Fuse（如有）
  → Bind(threadIdx.x/y/z)
```

若能够证明恰好一个物理线程轴的 extent，则计算 proposal 的线程数。proposal 合法时保持
原结果；越界时沿本次已选择的 divisor lattice，从更小 divisor 开始寻找最近的非平凡合法
变异：

```text
原 proposal 越界
  → 尝试下一个更小 divisor
  → 第一个 threads <= limit 的 proposal 作为投影结果
  → 没有非平凡合法点则返回 nullopt，让 EvolutionarySearch 重试
```

模式为：

```text
TVM_GA_T2_MODE=0  off：保持原 proposal
TVM_GA_T2_MODE=1  shadow：只观察越界，不额外采样，不改变返回 trace
TVM_GA_T2_MODE=2  project：执行线程可行性投影
```

统计包括 `checked`、`unknown`、`initial_reject`、`projected`、`projection_failed` 和
`divisor_steps`。

当前 flat trace 无法安全证明多物理轴处在同一词法作用域，零轴、多轴、warp_execution 和
不支持 provenance 都返回 unknown。因此它只是“单轴线程投影原型”，不是完整 G。

### 4.6 L：计划中的温度相对标签

TVM 当前 XGBoost 已经使用相对标签：

```text
y_i = best_cost / cost_i
```

因此不能把“引入相对标签”当成当前创新。计划测试的是温度/幂指数：

```text
y_i(gamma) = (best_cost / cost_i)^gamma
gamma ∈ {0.5, 0.75, 1.0, 1.5, 2.0}
```

`gamma=1` 必须数值等价于现有 TVM；只有在 128/256 样本档和多数 workload 的排序指标上
稳定改善，L 才进入最终系统。当前 `xgb_model.py` 尚未增加 `label_gamma`，这部分没有实现，
也没有实验数据。

---

## 5. 实验环境与测量口径

### 5.1 当前开发环境

| 项目                | 当前值                                                               |
| ------------------- | -------------------------------------------------------------------- |
| GPU                 | NVIDIA GeForce RTX 3050 Ti Laptop GPU，4 GB                          |
| Compute capability  | 8.6，target`sm_86`                                                 |
| Driver              | 591.44                                                               |
| Python              | 3.11.15，当前实验环境`.venv-tune`                                  |
| 构建目录            | `F:\tvm\build-prune`                                               |
| 当前真实 DLL        | `F:\tvm\build-prune\lib\tvm_compiler.dll`                          |
| 当前 DLL SHA256     | `1dd5493eb5c0bb95699e8633de3b770cdf77707592bdc3cb96443486e757ac62` |
| 默认开发 workload   | MatMul 128×128×128                                                 |
| Target threads 上限 | 1024 threads/block                                                   |
| Target shared 上限  | 49152 bytes/block                                                    |
| warp size           | 32                                                                   |

当前增量构建结果为 `ninja: no work to do.`，说明实验使用的 C++ 源码已经存在对应构建
产物。VS Code IntelliSense 也已经改为读取 `build-prune/compile_commands.json`。

### 5.2 实验层级

目前数据主要分为三类：

- **功能 smoke**：单 seed、8 trials，确认代码路径、计数、编译和上卡；
- **小规模 screening**：2 批共 16 trials，初步观察 XGBoost 训练前后和模块组合；
- **正式实验**：多 workload、5 seeds、256 trials、checkpoint 和置信区间——尚未进行。

因此本文所有 8/16-trial 结果都不能直接写成论文性能结论。

### 5.3 主要指标

硬筛主要看：

- 有限实测历史记录的误拒数；
- shadow TP/FP/FN/TN；
- 按失败原因的覆盖；
- hard-analysis 成本；
- 避免的 Verify/postproc 成本和净节省。

软先验主要看：

- known coverage；
- 单候选打分成本；
- Spearman、NDCG@k、top-k overlap/recall、regret；
- Verify 通过率是否因重排恶化；
- time-to-target 和最终 best。

GA 主要看：

- proposal 初始非法率；
- 投影成功/失败率；
- 投影后 Verify 合法率；
- 每代唯一合法后代比例；
- fallback 和 duplicate；
- 达到相同性能所需代数/测量数。

### 5.4 结果身份

实验脚本会保存：

- git HEAD；
- tracked diff hash；
- untracked source hash；
- 组合 source identity；
- 实际加载 DLL 的路径和 SHA256；
- 全部模式、参数和 Target。

最新 H2 四臂加载的是同一个 DLL。baseline 完成后只修复了实验日志解析正则，因此其
source identity 与另外三臂不同，但 C++ 搜索实现和加载 DLL 没有变化。机器可读汇总明确
保留了这个限制。

---

## 6. 已经完成的实验

### 6.1 E0：废弃旧 `vk != 1` 硬规则

#### 为什么做

早期剪枝用 MatMul 某个 reduction tile 因子 `vk != 1` 作为硬约束，曾在一次运行中观察到
约 8% 端到端时间下降。但它是绑定特定 MatMul/SSSRRSRS schedule 的算子指纹，无法证明
被拒候选真的非法。

#### 怎么做

从 baseline 的 `database_tuning_record.json` 提取已经通过全部 postproc、build 和 run 的
合法候选，检查其中有多少满足会被旧规则拒绝的 `vk == 1`。

#### 结果

| 数据集                     | 合法候选 | 会被旧规则误杀 | 误杀率 |
| -------------------------- | -------: | -------------: | -----: |
| verify_mode0，MatMul 128³ |        8 |              5 |  62.5% |
| prune_cmp_0，MatMul 256³  |        8 |              3 |  37.5% |
| prune_cmp_1，MatMul 256³  |        8 |              1 |  12.5% |
| prune_cmp_2，MatMul 256³  |        8 |              2 |  25.0% |
| prune_run_0，MatMul 128³  |        8 |              4 |  50.0% |
| 合计                       |       40 |             15 |  37.5% |

`vk==1` 候选同时包含很快和很慢的 kernel，既不代表非法，也不能稳定表示低性能。

#### 结论

旧规则不健全；此前 8% 时间变化受到误杀合法候选、缩小搜索多样性的污染，不能作为有效
加速证据。该实验建立了后续原则：任何硬规则必须先过合法数据库零误拒审计。

### 6.2 I0：环境、真实 DLL 与 runner 诊断

#### 完成内容

- 修复实验父进程和 LocalBuilder 子进程的 `PYTHONPATH`；
- 固定可用 Python 3.11 环境；
- 修复读取实际加载 `tvm_compiler.dll` 的方法；
- 将 TEMP/TMP 放在 ASCII、可写路径，避开中文用户名和链接器临时文件问题；
- 补齐 builder/runner error、原始 `run_secs` 和逐批 measure callback；
- 确认不是 Python 模拟，而是 C++ DLL + CUDA 真机测量。

首轮 P1 smoke 出现 `run_secs=[1e10]`。诊断显示 builder 已成功，runner 在加载 `.tar`
模块的二次链接阶段因为临时目录权限触发 `LNK1104`。在可写临时目录重新运行同一 1-trial
配置后得到有限实测 80.9279 us，证明 1e10 首因是环境权限，而不是候选变异或草稿模型。

### 6.3 旧版线程硬筛三臂 smoke

#### 配置

MatMul 128³、seed 20260902、8 trials、population 512、GA 4 代。

#### 结果

| Arm     | Search | Verify fail |     Best | 预测/拦截 |
| ------- | -----: | ----------: | -------: | --------: |
| off     |   73 s |        1184 | 36.73 us |       0/0 |
| shadow  |   71 s |        1184 | 51.79 us |       7/0 |
| enforce |   72 s |        1177 | 49.15 us |       7/7 |

shadow 检查 3047 个候选，TP/FP/FN/TN 为 7/0/1177/1863，同口径 mixed-oracle recall
只有 0.59%。它证明了旧线程上限规则在样本内没有观察到误拒，但覆盖太窄，时间差也不足以
证明加速。

### 6.4 旧软过滤与 GA 投影分臂 smoke

#### 结果

| Arm         | H / S / G              | Search |    Tune | Verify fail |     Best |
| ----------- | ---------------------- | -----: | ------: | ----------: | -------: |
| baseline    | off/off/off            |   71 s | 225.8 s |        1184 | 48.74 us |
| soft shadow | off/shadow/off         |   71 s | 223.4 s |        1184 | 48.98 us |
| soft rerank | off/rerank/off         |   88 s | 237.9 s |        1210 | 34.44 us |
| GA shadow   | off/off/shadow         |   72 s | 224.4 s |        1184 | 46.04 us |
| GA project  | off/off/project        |   77 s | 230.7 s |        1178 | 37.07 us |
| combined    | enforce/rerank/project |   88 s | 235.0 s |        1181 | 34.45 us |

旧 soft shadow 处理 2145 个候选，候选数严格保持 2145→2145，分析成本约 22.7 ms，
top-16 overlap 为 42/80。rerank 的 best 更好，但 Verify 失败增加、搜索慢 17 s，说明软
重排会改变整个搜索谱系，不能只看其自身几十毫秒的计算成本。

GA shadow 观察到 6 个初始线程越界 proposal；project 运行中观察到 13 个，成功投影 6
个、无非平凡合法点 7 个。两臂轨迹不同，所以不能把 13 与 6 当成严格配对。该实验只证明
线程投影路径在真实搜索中会触发并工作，没有证明搜索加速。

### 6.5 旧 thread-only soft weight 小规模扫描

为检查 `weight=0.2` 是否过强，使用相同源码身份，在 seed 20260902/03/04 上分别运行
8 trials。下表为三个 seed 的中位数；括号内是各 seed best 的范围。

| Weight | Search 中位数 | Tune 中位数 | Verify fail 中位数 |         Best 中位数（范围） |
| -----: | ------------: | ----------: | -----------------: | --------------------------: |
|   0.00 |          71 s |    218.80 s |                730 | 43.186 us（37.397～53.690） |
|   0.05 |          72 s |    213.60 s |                726 | 39.291 us（30.878～50.770） |
|   0.10 |          68 s |    222.41 s |                716 | 43.203 us（32.503～50.070） |
|   0.20 |          69 s |    208.59 s |                723 | 31.545 us（29.742～33.858） |

`weight=0.2` 在这三个极小预算 seed 上的 best 分布较积极，但每臂只有 8 次测量，Verify
失败数量也存在明显 seed 差异。该结果只能用于决定后续参数候选，不能证明 0.2 已经稳定
优于其他权重。

### 6.6 P1：Pruner 草稿 + 已训练 XGBoost 两批实验

#### 为什么分两批

即使设置 `XGBModel(num_warmup_samples=0)`，第一批开始时 booster 仍不存在，第一批预测
实际仍是随机分数。第一批 8 个有限实测加入训练后，第二批才是“已训练 XGBoost + P”。

#### 配置

MatMul 128³、seed 20260903、16 trials、每批 8、population 512、GA 4 代、H/S/G 关闭、
`beta=0.2`、`draft_epsilon=0.10`。

#### 结果

| Arm    | Search |     Tune | Verify fail |      Best |   P 开销/coverage |
| ------ | -----: | -------: | ----------: | --------: | ----------------: |
| off    |  142 s | 418.84 s |        1379 | 40.479 us |                — |
| shadow |  145 s | 424.24 s |        1386 | 40.156 us | 248.948 ms / 100% |
| rank   |  167 s | 449.13 s |        1401 | 44.491 us | 248.079 ms / 100% |

rank 相对 off：搜索慢 17.6%，tune 慢 7.2%，best 差 9.91%。草稿解析本身只有约 0.248 s，
主要退化来自重排后走入了需要更多 Verify/retry 的搜索路径，而不是草稿计算太慢。

#### top-k 和真实质量分析

shadow 下，原 XGBoost 与假想融合 top-16 的总 overlap 为 34/160。第一批随机 warmup 为
21/80，第二批已训练后为 13/80；这显示训练后的 XGBoost 与当前草稿在头部冲突明显。

对三臂 48 条记录按 trace hash 去重后得到 36 条唯一 trace：

```text
P score 与 inverse latency 的 Spearman：0.314
不同 P score 档位：11
P-only top-1 latency：347.345 us
oracle best：40.318 us
P-only top-8 与真实 top-8 overlap：2/8
P-only top-8 中最好：44.491 us
相对 oracle regret：10.35%
```

这说明当前 P 有一定总体方向，但头部精排能力不足，不能继续简单增大 beta。

另一个实现细节是：每批只选 8 个候选时，当前代码使用
`int(8 * draft_epsilon)`，`epsilon=0.10` 会得到 0 个随机候选。名义上的 10% 探索在该
配置下实际没有生效，后续应改为明确的取整策略或增大 batch。

### 6.7 H v2：历史数据库零误拒审计

#### 数据范围

```text
总记录：606
有限实测记录：409
唯一 schedule：432
唯一有限实测 schedule：262
trace replay error：0
```

#### 结果

改进后的线程硬筛：

```text
有限实测记录被拒：0
唯一有限实测 schedule 被拒：0
```

这为当前线程规则提供了较强的经验安全证据。它仍不是数学上覆盖所有 TVM schedule 的
证明，所以正式实验仍需要跨算子和人工边界测试。

shared pre-compaction proxy：

```text
409 条有限实测中预测拒绝 397 条
```

因此该 shared proxy 明确 No-Go，不得进入 enforce。

### 6.8 H v2：shadow 闸门

在 16/16 次真实 CUDA 测量中重新解析完整原始日志：

```text
checked = 5089
TP/FP/FN/TN = 1138/0/281/3670
precision = 100%
mixed-oracle recall = 80.197%
unknown = 0
分析成本 = 0.038173 s
search = 149 s
VerifyGPUCode = 106.264205 s
```

预测原因：

```text
thread_extent_low  = 1121
thread_extent_high = 17
hardware limit     = 0
```

正式 Verify 失败：

```text
thread_extent      = 1138
shared_memory      = 231
lowering_exception = 50
```

这里的 80.197% 是针对整个 Verify 混合失败集合计算的，不能解释为独立 thread-only
数据集的分类召回。但它说明新增 low/high contract 检查覆盖了旧版漏掉的大量
`thread_extent` 失败，而且样本内没有观察到“预测拒绝但 Verify 通过”的反例。

### 6.9 H v2 × P：四臂组合实验

#### 配置

```text
workload：MatMul 128³
seed：20260903
trials：16，每批 8
population：512
GA generations：4
XGB warmup：0
旧 soft：off
GA T2：off
P beta：0.2
P epsilon：0.10
```

关闭旧 soft 和 GA 是为了只测 H、P 及二者组合的独立效果。

#### 结果

| Arm      | Search | Tune call | VerifyGPUCode |                Best |          Top-3 mean |          Top-5 mean |
| -------- | -----: | --------: | ------------: | ------------------: | ------------------: | ------------------: |
| baseline |  193 s |  479.64 s |      151.95 s |           49.038 us |           63.096 us |           86.256 us |
| H        |  150 s |  429.90 s |      110.75 s |           37.057 us |           91.530 us |          109.593 us |
| P        |  161 s |  431.69 s |      121.24 s | **33.108 us** | **36.303 us** | **64.429 us** |
| H+P      |  175 s |  446.06 s |      133.40 s |           34.784 us |           61.513 us |           84.619 us |

H enforce：

```text
generated = 5037
hard rejected = 1105
low/high/hardware = 1091/14/0
remaining Verify failures = 272
hard analysis = 0.035383 s
```

H+P：

```text
generated = 5045
hard rejected = 1107
low/high/hardware = 1098/9/0
remaining Verify failures = 307
hard analysis = 0.036127 s
P analysis = 0.249296 s
XGB/fused top-k overlap = 72/160
```

P 单臂 coverage 100%，P analysis 0.250132 s，top-k overlap 67/160。

#### 如何解释

- P 单臂在这次 seed 上的 best、top-3、top-5 都优于 baseline，是积极信号；
- 但上一节 P1 rank 是负结果，所以不能说 P 已经稳定改善；
- H+P 相对 P 单臂 search 慢 8.70%、tune 慢 3.33%、best 差 5.06%，没有组合增益；
- baseline 193 s 明显高于同一实现的 H shadow 149 s。判断 H 时间收益时应优先比较 shadow
  149 s 与 enforce 150 s，而不是把 193→150 s 全部归因给 H；
- 正式 `VerifyGPUCode` 已经在 lowering 前执行 `ThreadExtentChecker`，当前 H 只是把同一类
  线程检查略微前移，所以没有跳过昂贵 lowering，这解释了为何拒绝 1105 个候选却没有
  稳定时间收益。

### 6.10 编译与测试

当前 C++ 不是 Python 模拟，已链接进：

```text
F:\tvm\build-prune\lib\tvm_compiler.dll
SHA256 = 1dd5493eb5c0bb95699e8633de3b770cdf77707592bdc3cb96443486e757ac62
```

相关回归最终为 42 passed：

| 测试组                            | 数量 |
| --------------------------------- | ---: |
| thread feasibility                |   22 |
| off/shadow/enforce integration    |    1 |
| Pruner draft                      |    3 |
| VerifyGPUCode postprocessor       |    8 |
| tile-size mutator / GA projection |    6 |
| VerifyGPUCode analysis API        |    2 |
| 合计                              |   42 |

Python 实验/分析脚本已通过 `py_compile`，相关 diff 通过空白检查。

---

## 7. 当前取得的成果

### 7.1 已经可靠成立的工程成果

1. 建立了可复用的 `build-prune` 编译和真实 CUDA 实验路径。
2. 建立了 off/shadow/enforce 或 off/shadow/rank 的安全上线方式。
3. 建立了候选守恒、逐 postproc、逐 Verify 原因和时间分解。
4. 用真实合法数据库否定了旧 `vk!=1` 和简单 shared 声明 proxy，避免把错误规则写成成果。
5. 实现了通用程度明显更高的线程路径/线程契约分析，不再依赖 MatMul 某个固定 tile 下标。
6. 将 Pruner 风格解析模型做成独立、可消融的 C++ 模块，并接入 XGBoost 排序。
7. 实现了单轴线程越界 GA 投影以及独立统计开关。
8. 实验结果包含源码和 DLL 身份，可追查“到底跑的是哪份代码”。

### 7.2 已经可靠成立的研究发现

1. 算子指纹式硬剪枝会严重误杀，零误拒审计必须先于性能实验。
2. 当前 MatMul 失败主体是 thread contract，而不是物理 threads/block 超上限。
3. 仅增加 thread contract 覆盖可以大幅减少后续 Verify 失败计数，但在当前位置不等于时间
   加速。
4. lowering 前 shared 声明大小与 lowering 后真实 allocation 差异很大，shared 硬筛必须
   建立在更可靠的 compact/placement 语义上。
5. 解析先验计算很便宜，真正风险是排序误差改变搜索谱系，而非模型推理成本。
6. 当前 P 更像冷启动 prior，而不是能长期压过已训练 XGBoost 的精排器。
7. “减少非法候选数”“减少搜索时间”“提高最终 kernel 质量”是三个不同命题，必须分别
   实验，不能互相替代。

### 7.3 当前可以和不可以对外表述的结论

可以表述：

- 当前线程硬筛在已审计的有限实测数据库中零误拒；
- contract-aware H 将当前 MatMul 上的 mixed-oracle coverage 提高到 80.20%；
- P 的分析开销低、能够显著改变 top-k；
- GA 原型能识别并投影部分单轴线程越界变异；
- P 在一个最新 seed 上产生了积极候选质量结果。

不可以表述：

- H 已稳定加速 MetaSchedule；
- shared-memory 硬过滤已经完成；
- P 已稳定优于 XGBoost；
- H+P 有组合增益；
- GA 已完整实现可行性保持；
- 代价模型相对标签已经修改；
- 当前方法已跨算子通用；
- 单 seed、16 trials 的 best 可以作为论文主结果。

---

## 8. 当前不足

### 8.1 H 的插入位置收益不足

当前 H-late 在完整 trace replay 和前序 postprocessor 之后，且正式
`ThreadExtentChecker` 本来就在 lowering 前。它能改善原因归因和失败计数，但没有明显
避免主要 lowering 成本。要形成性能贡献，需要：

- 将能从 trace 决策证明的 contract/thread 检查前移到 replay/postproc 之前；或
- 找到真正能跳过昂贵 lowering 的安全 shared-memory 判据；或
- 用 candidate-stream 实测证明某个新位置的净节省为正。

### 8.2 shared-memory 判据尚未解决

简单累计 lowering 前声明 buffer 会因为后续 compact/merge 产生大量误判。下一版必须
区分：

- 声明大小；
- compact 后大小；
- software pipeline stage 倍数；
- buffer hoisting；
- 静态 shared 与动态 shared；
- 能证明的下界、能证明的精确值和 unknown。

只有“真实安全下界已经超过上限”或“精确值超过上限”才能硬拒绝。

### 8.3 P 的特征和校准过粗

当前 P 有以下近似：

- 浮点操作只按 TIR 表达式节点粗略计数；
- register usage 不是后端编译后的真实寄存器数；
- DRAM proxy 未完整描述 cache、transaction、coalescing 和 bank conflict；
- Tensor Core、流水、指令延迟等没有完整建模；
- 36 个唯一 trace 只有 11 个分数档位，tie 较多；
- population max-normalization 容易让单个极端值控制其他候选比例；
- 固定 beta 会在 XGBoost 已训练后继续强行干预；
- 小 batch 下 epsilon 向下取整成 0，探索保护没有实际生效。

### 8.4 G 只覆盖单轴线程越界

当前 G 不覆盖：

- 多物理线程轴；
- warp_execution；
- shared-memory 越界；
- 一个因子同时影响多个资源；
- 余量内的方向性采样；
- 投影后的 Verify 原因闭环；
- 后代 duplicate 和搜索多样性约束；
- UCB mutator 选择和退火接受劣解。

### 8.5 L 完全未开始

当前没有 `label_gamma` API、没有相对标签单元测试、没有 trace 去重后的 train/test 划分，
也没有 32/64/128/256 样本学习曲线。因此当前项目还不能称为“代价模型已修改”。

### 8.6 实验外部有效性不足

当前主要实验局限为：

- 主要只有 MatMul 128³；
- 主要只有 RTX 3050 Ti；
- H2/P 实验只有一个 seed；
- 8/16 trials 太少，best 容易被偶然候选影响；
- search 时间来自秒级日志，存在明显运行波动；
- P1 只观察被现有搜索选中的候选，存在 selection bias；
- baseline 与 H2 其他三臂存在解析正则修复导致的 source identity 差异，虽然 DLL 一致；
- 当前代码仍位于 dirty worktree，尚未整理成独立、可复查提交。

---

## 9. 下一步实施与实验计划

### 9.1 第一优先级：冻结当前基线

目标是让后续每个数字都可复现：

1. 将 I0/H/P/G 当前改动按模块拆成独立提交；
2. 冻结 DLL、compile commands、Python 环境和 `frozen_config.json`；
3. 每个 arm 使用独立目录，不覆盖旧结果；
4. 统一保存 candidate count、source identity、DLL SHA256、GPU、driver、CUDA 和 Target；
5. 将运行状态明确标为 smoke、screening 或 formal。

### 9.2 E1：H 的位置和真实净收益

需要做三个时间位置对照：

```text
B：不启用 H
H-late-shadow/enforce：当前 Verify 前位置
H-early-shadow/enforce：trace 决策可知后、昂贵 replay/postproc 前
```

每种位置记录：

```text
H analysis time
avoided replay time
avoided postproc time
avoided Verify/lowering time
net saved time
final unique legal candidates
final kernel quality
```

H-late 只有在 `net_saved_time > 0` 且置信区间不跨 0 时才能作为加速组件；否则保留为失败
归因和兜底组件。

### 9.3 E1-shared：安全 shared-memory 研究

先做离线标定，不直接改 enforce：

1. 对历史合法记录和随机候选保存 lowering 前 IR；
2. 运行与正式路径一致的 compact/lowering；
3. 对比声明大小、候选下界、compact 后真实 shared 大小；
4. 按 software pipeline、hoisting、cache read placement 分组；
5. 找到始终满足 `estimate <= actual` 的下界，或将不可证明样本标 unknown；
6. 在全部有限实测历史记录上要求误拒为 0；
7. 再以 shadow 对每个 shared Verify failure 做逐样本确认。

如果安全下界召回仍很低，应诚实放弃 shared H-early，将 shared 只用于 P/G 软信号。

### 9.4 E2：P 的无偏质量评估

当前只看数据库会受到“原模型已经选择过这些候选”的偏差。下一步应在 B 模式旁路保存
未经过 H/P 选择的 candidate stream：

```text
candidate_id / trace_hash / stage
draft_score / feature rows / known
postproc_pass / verify_pass / failure_reason
measured / latency
```

然后按 P score 分为 5 个分位区间，对每个区间等量随机抽取已通过 Verify 的唯一候选进行
build/run。每个 workload 建议至少：

- 200 个未过滤候选；
- 每个分位区间 20 个合法唯一 trace；
- 分开报告 SampleInit 和 Evolve。

指标包括 Spearman、NDCG@16、top-k regret、各分位 Verify 通过率、每得到 N 个合法候选
需要处理的候选数，以及模拟 P2 的净节省。

### 9.5 P1 融合改进

建议按以下顺序做，不同时修改多个自由度：

#### 改进一：修复探索比例

对小 batch 使用明确策略，例如：

```text
num_random = 0                         if epsilon == 0
num_random = max(1, round(n*epsilon))  otherwise
```

并测试 `epsilon={0.05,0.10,0.20}`。

#### 改进二：用 rank/percentile 代替 population max

将粗糙、尺度不稳定的绝对分数转换为 population 百分位，降低极端值和硬 tie 对排序的控制。

#### 改进三：冷启动衰减

```text
beta(n) = beta0 * exp(-n / tau)
fused_score = xgb_score * draft_rank^beta(n)
```

XGBoost 数据少时 P 提供方向，数据增加后逐渐退出。测试 `beta0={0.05,0.10,0.20,0.40}`
和固定的少量 `tau` 候选。

#### 改进四：学习校准而不是永久手工相乘

把 P 的原始特征/分数作为 XGBoost 附加特征，或训练 residual/pairwise calibration，让数据
决定何时信任解析模型。这部分必须与“手工乘法融合”独立消融。

### 9.6 P2：真正的 draft-then-verify 前置选择

只有 E2 candidate stream 显示正收益后才实现：

```text
oversample k× traces
  → 轻量 replay
  → H-early
  → P 打分
  → 90% top + 10% random
  → 仅入选者执行完整 postproc/Verify
```

P2 必须满足：

- shadow 不改变候选、trace 和影响原搜索的 RNG；
- select 回退率低于 5%；
- 避免的 postproc/Verify 时间大于额外 replay/P/oversample 时间；
- 3 seeds × 64 trials 下最终质量相对 P1 退化不超过 2%。

任何门槛失败就回退 P1，不为了“更像 Pruner”保留负收益路径。

### 9.7 E3：方向性 GA 扩展

建议分四步：

1. **G1 多轴线程投影**：从 trace/IR 建立可靠的词法作用域关系；
2. **G1 shared 投影**：只使用经过 E1-shared 验证的安全资源估计；
3. **G2 余量方向选择**：在所有 hard-feasible divisor 中，以 `1-epsilon` 选择 P 增量最好
   的方向，以 epsilon 随机探索；
4. **G3/G4 可选**：UCB mutator 账本、相对分数尺度下的退火接受。

必须记录：

```text
proposal_total / known / unknown
initial_legal / initial_illegal[reason]
projected_success / failure
post_projection_verify_pass / fail[reason]
directed / random
duplicate_offspring
fallback_to_parent
unique_trace_ratio
```

G 只直接影响 Evolve，因此不能把 SampleInit 的变化归功于 G。

### 9.8 E4：温度相对标签

实施 `label_gamma` 后进行纯离线实验：

- trace hash 去重后再切 train/test；
- 同一 trace 重复测量不能跨集合；
- 训练前缀为 32/64/128/256；
- `best_cost` 只由训练集计算；
- `gamma={0.5,0.75,1.0,1.5,2.0}`；
- 5 个随机划分；
- 主指标 Spearman、NDCG@16、top-k regret；RMSE 只作辅助。

若多数 workload 在 128/256 样本档没有稳定提升，最终系统保持 `gamma=1`。

### 9.9 E5/E6：正式模块消融和端到端实验

正式六臂：

| Arm   | H | P | G |          L | 目的             |
| ----- | -: | -: | -: | ---------: | ---------------- |
| B     | 0 | 0 | 0 |    gamma=1 | 原始基线         |
| P     | 0 | 1 | 0 |    gamma=1 | P 独立贡献       |
| H     | 1 | 0 | 0 |    gamma=1 | H 独立贡献       |
| H+G   | 1 | 0 | 1 |    gamma=1 | G 在 H 上的增量  |
| H+P+G | 1 | 1 | 1 |    gamma=1 | 三个搜索模块组合 |
| FULL  | 1 | 1 | 1 | 选定 gamma | L 的最后增量     |

正式 workload 至少覆盖：

- GEMM：两个 shape；
- Conv2D：两个 shape；
- Reduction：两个 shape；
- Softmax：两个 shape；
- LayerNorm：两个 shape；
- Add+ReLU：两个 shape。

先在 RTX 3050 Ti 开发和排错，正式主结果使用 A100；有预算时再加入第二种架构。

统一 seeds：

```text
20260902, 20260903, 20260904, 20260905, 20260906
```

每臂 256 trials，每 32 trials 保存 checkpoint。主指标：

- 达到参考性能 90%/95%/99% 的 time-to-target 和测量数；
- best-so-far AUC；
- 256 trials 最终质量；
- search/replay/postproc/build/run 分解；
- 合法率、Verify 失败率、唯一 trace 比例。

统计使用 workload×seed 配对，中位数/IQR、配对 bootstrap 95% CI；需要显著性检验时使用
配对 Wilcoxon 并报告 effect size。

---

## 10. 代码位置说明

### 10.1 核心 C++

| 文件                                                                                                                                | 关键部分                                                                             | 当前作用                              |
| ----------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ | ------------------------------------- |
| [`src/s_tir/meta_schedule/constraints.h`](src/s_tir/meta_schedule/constraints.h)                                                   | `ThreadFeasibilityResult`、`AnalyzeThreadFeasibility` 声明                       | H 的分析结果结构和接口                |
| [`src/s_tir/meta_schedule/constraints.cc`](src/s_tir/meta_schedule/constraints.cc)                                                 | `PhysicalThreadCollector`、thread contract 检查、shared proxy、测试 FFI            | H v2 的具体分析逻辑                   |
| [`src/s_tir/meta_schedule/utils.h`](src/s_tir/meta_schedule/utils.h)                                                               | `VerifyGPUCodeFailureReason`、`ThreadedTraceApply::Apply`、`SummarizeFailures` | H 插入点、计数守恒、Verify 原因和计时 |
| [`src/s_tir/meta_schedule/draft_model.h`](src/s_tir/meta_schedule/draft_model.h)                                                   | hardware profile、11-field feature row、score API                                    | P 的数据结构和接口                    |
| [`src/s_tir/meta_schedule/draft_model.cc`](src/s_tir/meta_schedule/draft_model.cc)                                                 | `PrunerDraftFeatureExtractor`、`ScorePrunerDraftFeatures`、归一化和融合          | P 特征、公式、分数及测试 FFI          |
| [`src/s_tir/meta_schedule/search_strategy/evolutionary_search.cc`](src/s_tir/meta_schedule/search_strategy/evolutionary_search.cc) | `PredictNormalizedScore`、P/S 环境变量、top-k overlap                              | S/P 与 XGBoost 的接入和日志           |
| [`src/s_tir/meta_schedule/mutator/mutate_tile_size.cc`](src/s_tir/meta_schedule/mutator/mutate_tile_size.cc)                       | `EstimateThreadsFromTrace`、`MutateSampleTileSize`、GA T2 stats                  | G 的单轴线程投影原型                  |
| [`src/s_tir/meta_schedule/search_strategy/replay_trace.cc`](src/s_tir/meta_schedule/search_strategy/replay_trace.cc)               | Target 传递、`ThreadedTraceApply` 日志                                             | ReplayTrace 路径也使用相同 H/计数设施 |
| [`src/s_tir/meta_schedule/postproc/verify_gpu_code.cc`](src/s_tir/meta_schedule/postproc/verify_gpu_code.cc)                       | failure reason 分类、thread-local last reason                                        | I0 的 Verify 失败归因                 |
| [`src/s_tir/analysis/verify_gpu_code.cc`](src/s_tir/analysis/verify_gpu_code.cc)                                                   | `VerifyGPUCodeErrors`                                                              | 返回详细错误而不只返回 bool           |
| [`include/tvm/s_tir/analysis.h`](include/tvm/s_tir/analysis.h)                                                                     | `VerifyGPUCodeErrors` 公共声明                                                     | C++ analysis API                      |
| [`python/tvm/s_tir/analysis/__init__.py`](python/tvm/s_tir/analysis/__init__.py)                                                   | `verify_gpu_code_errors`                                                           | Python 详细错误 API                   |

### 10.2 测试

| 文件                                                                                                                                                                  | 覆盖内容                                                                      |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| [`tests/python/s_tir/meta_schedule/test_meta_schedule_thread_feasibility.py`](tests/python/s_tir/meta_schedule/test_meta_schedule_thread_feasibility.py)             | x/y/z、warp、contract low/high、31/32、1024/1025、unknown、shared observation |
| [`tests/python/s_tir/meta_schedule/test_meta_schedule_thread_filter_modes.py`](tests/python/s_tir/meta_schedule/test_meta_schedule_thread_filter_modes.py)           | H off/shadow/enforce 集成                                                     |
| [`tests/python/s_tir/meta_schedule/test_meta_schedule_pruner_draft_model.py`](tests/python/s_tir/meta_schedule/test_meta_schedule_pruner_draft_model.py)             | P 公式、归一化、unknown、中性、shadow/rank                                    |
| [`tests/python/s_tir/meta_schedule/test_meta_schedule_mutator_mutate_tile_size.py`](tests/python/s_tir/meta_schedule/test_meta_schedule_mutator_mutate_tile_size.py) | G 投影、失败回退、unknown、多轴、off 等价                                     |
| [`tests/python/s_tir/meta_schedule/test_meta_schedule_postproc_verify_gpu_code.py`](tests/python/s_tir/meta_schedule/test_meta_schedule_postproc_verify_gpu_code.py) | Verify postprocessor 和失败分类                                               |
| [`tests/python/s_tir/analysis/test_s_tir_analysis_verify_gpu_code.py`](tests/python/s_tir/analysis/test_s_tir_analysis_verify_gpu_code.py)                           | `verify_gpu_code_errors` API                                                |

### 10.3 实验和分析脚本

| 文件                                                                                          | 作用                                                                                          |
| --------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| [`experiments/e1_thread_filter.py`](experiments/e1_thread_filter.py)                         | 当前统一 MatMul-128 驱动；控制 H/S/P/G，记录 schema 5 结果、身份、measure callback 和日志解析 |
| [`experiments/audit_verify_database.py`](experiments/audit_verify_database.py)               | 重放历史数据库，检查 H 有限实测误拒和 shared proxy                                            |
| [`experiments/analyze_pruner_draft_results.py`](experiments/analyze_pruner_draft_results.py) | 重放实测 trace，计算 P 的 Spearman、top-k、regret 和 tie                                      |
| [`experiments/analyze_hard_draft_combo.py`](experiments/analyze_hard_draft_combo.py)         | 统一生成 H×P 四臂汇总、shadow gate、数据库审计和 trace overlap                               |
| [`experiments/pruner_prefilter.py`](experiments/pruner_prefilter.py)                         | 早期 MatMul 固定下标公式观察；不通用，不作为最终实现                                          |

### 10.4 方案和实施记录

| 文件                                                                                                    | 作用                                                |
| ------------------------------------------------------------------------------------------------------- | --------------------------------------------------- |
| [`研究方案-通用可行性过滤与可行性保持搜索.md`](研究方案-通用可行性过滤与可行性保持搜索.md)             | 总体研究动机、T1/T2/T3、相关工作和论文定位          |
| [`experiments/IMPLEMENTATION_EXPERIMENT_PLAN.md`](experiments/IMPLEMENTATION_EXPERIMENT_PLAN.md)       | H/P/G/L 的 v1.1 具体实施与 E0-E6 实验方案           |
| [`experiments/IMPLEMENTATION_LOG_V1_1.md`](experiments/IMPLEMENTATION_LOG_V1_1.md)                     | 2026-09-02/03 的实现、环境、P1、H2 与组合实验全过程 |
| [`experiments/FILTER_SOFT_GA_RESULTS_2026-09-02.md`](experiments/FILTER_SOFT_GA_RESULTS_2026-09-02.md) | 旧 H/S/G smoke 的完整结果和复现命令                 |
| [`GA优化工作单 (1).md`](<GA优化工作单 (1).md>)                                                         | G1～G4 的扩展设计和验收指标                         |
| [`实验进度记录.md`](实验进度记录.md)                                                                   | 早期 E0、环境修复和阶段进度                         |

---

## 11. 数据和结果位置

| 路径                                                                                                                        | 内容                                      |
| --------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------- |
| [`experiments/data/e2_soft_ga/`](experiments/data/e2_soft_ga/)                                                             | 旧 soft/GA 分臂 smoke                     |
| [`experiments/data/e3_final/`](experiments/data/e3_final/)                                                                 | 旧 H 三臂和最终联合复跑                   |
| [`experiments/data/e4_soft_weight/`](experiments/data/e4_soft_weight/)                                                     | seed 20260902 的 soft weight 扫描         |
| [`experiments/data/e5_soft_multiseed/`](experiments/data/e5_soft_multiseed/)                                               | seed 20260903/04 的 soft weight 扫描      |
| [`experiments/data/runner_diag/`](experiments/data/runner_diag/)                                                           | runner 1e10/LNK1104 失败与成功对照        |
| [`experiments/data/v1_1_p1_trained/`](experiments/data/v1_1_p1_trained/)                                                   | P1 off/shadow/rank 两批 16-trial 原始结果 |
| [`experiments/data/v1_1_p1_trained/draft_quality_pair0.json`](experiments/data/v1_1_p1_trained/draft_quality_pair0.json)   | P1 运行与离线排序质量汇总                 |
| [`experiments/data/hard_filter_v2_database_audit.json`](experiments/data/hard_filter_v2_database_audit.json)               | 606 条历史记录 H v2 审计                  |
| [`experiments/data/v1_1_h2_combo/`](experiments/data/v1_1_h2_combo/)                                                       | H/P 四臂及 shadow 原始日志/数据库         |
| [`experiments/data/v1_1_h2_combo/hard_draft_combo_pair0.json`](experiments/data/v1_1_h2_combo/hard_draft_combo_pair0.json) | 当前 H2×P 主汇总                         |

shadow H2 目录没有 `result.json`，原因是旧日志正则在最终落盘前误解析 shared 统计；其
16 条数据库记录和原始日志完整，已经由修复后的解析器重新生成到统一汇总 JSON。这个事实
必须在复查时保留。

---

## 12. 当前总体判断

当前工作已经从一个会误杀的 MatMul 专用规则，推进到一个具备以下属性的研究原型：

- 有明确硬/软语义边界；
- 有 contract-aware 线程硬筛；
- 有合法数据库审计；
- 有 Verify 原因和成本分解；
- 有独立 Pruner 风格草稿模型；
- 有真实 XGBoost 融合和 top-k 变化；
- 有初步的 GA 越界投影；
- 有真实编译、真实 CUDA 测量和可追踪结果身份。

现阶段最有价值的成果不是一个已经确定的加速百分比，而是已经排除了两条错误路线，并找
到了真正需要攻克的瓶颈：

1. 当前 H 的线程覆盖高，但位置与已有早期检查重叠，必须继续前移或找到安全 shared
   判据才能产生时间收益；
2. P 很便宜且能改变搜索，但当前分数过粗，最适合作为可衰减冷启动 prior，而不是永久
   压制已训练 XGBoost；
3. G 的正确方向是让变异留在可行域，但目前只覆盖单轴线程场景；
4. L 尚未实现，必须先用离线、多样本学习曲线证明价值；
5. 最终是否成立必须由跨算子、多 seed、time-to-target 和最终质量共同回答。

因此，当前研究可以进入下一轮 H-early/shared 标定、P 校准和 G 扩展，但还不应进入论文
主表式的性能宣传。下一轮实验最重要的纪律仍然是：先保证零误拒和对照可归因，再扩大
预算；任何模块只在自己的预注册门槛上成立，不能用另一个模块的偶然 best 替它证明。

---

## 13. 2026-09-05 最新实现与正式协议实验（覆盖旧状态）

本节记录对第二点方案纠偏后的实现。完整过程、逐项计数和解释同时写入
[`实验进度记录.md`](实验进度记录.md) 的 “SP-v1” 章节。

### 13.1 为什么纠偏

旧原型在每代先用 Draft 缩候选、随后仍让 XGBoost 参与 GA，这不符合研究目标。当前实现
把职责严格拆开：

```text
CS/H 负责可行性
  → Draft 独立作为 GA fitness，运行完整 4 轮
  → G2/G1/G4 只使用硬件资源和 Draft 信号
  → 汇总跨代 Draft top-32
  → XGBoost 每个搜索批次只调用一次
  → 选 8 条编译、上卡实测
```

因此，新实验不沿用 2026-09-04 那组“每代 XGBoost”的第二点数据。旧数据只保留作错误
接入位置的负向探索。

### 13.2 实现结构

| 模块     | 代码                                                                   | 实际职责                                                                                         |
| -------- | ---------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| CS       | `constraints.{h,cc}`、`utils.h`                                    | 建立`TraceConstraintPlan`，初始化时只从已知线程可行 decision 中采样                            |
| H        | `constraints.cc`、`utils.h`                                        | trace、post-replay、late 三个位置拒绝可证明线程非法者；unknown 放行                              |
| Draft    | `draft_model.{h,cc}`                                                 | 从 Schedule/IR 提取解析特征，按显式 RTX 3050 Ti profile 给性能分数                               |
| Draft-GA | `draft_guided_ga.{h,cc}`、`search_strategy/evolutionary_search.cc` | Draft fitness、跨代 archive、末端 top-K XGBoost 精排                                             |
| G2       | `mutator/mutate_tile_size.cc`                                        | 已知越界后沿 divisor lattice 或反向转移缩回合法域                                                |
| G1       | `mutate_tile_size.cc`、`evolutionary_search.cc`                    | 线程余量方向；每次生成 4 个合法 proposal，再以 Draft 增量/ε 探索选择                            |
| G4       | `draft_guided_ga.cc`、`evolutionary_search.cc`                     | 以相对 Draft 分数差计算退火概率；支持 off/shadow/enforce                                         |
| G3       | `draft_mutator_credit.{h,cc}`、`evolutionary_search.cc`            | 基于真实父子延迟的滑窗 UCB1、lineage、跨批 Runner feedback；早期六臂关闭，后续另做 64-trial 实验 |

相关测试源码：

- `test_meta_schedule_pruner_draft_model.py`：Draft fitness、verify top-K、方向选择、G4 相对差；
- `test_meta_schedule_mutator_mutate_tile_size.py`：off/shadow/project/directed、乘积守恒、合法投影；
- `test_meta_schedule_draft_mutator_credit.py`：G3 账本、失败臂临时排除、off/shadow、lineage
  和跨批反馈测试源码；最新执行结果需在下一次重新编译后单独持久化。

### 13.3 实验身份和口径

```text
MatMul128 / CUDA sm_86 / seed 20260905
population 128 / 4 mutation rounds / 16 trials / 8 per batch
exact legal init quota 248 / init micro-batch 64 / XGB warmup 0
Draft verify ratio 0.25 / direction trials 4 / epsilon 0.10
G4 T0 0.10 / decay 0.95
```

所有新六臂和 G4 运行使用：

```text
F:\tvm\build-prune\lib\tvm_compiler.dll
SHA256 fb3bf9abc23abdf4b88c281883706b3f3dee10d53bb42653b91b54f30a6d0365
source identity 6ea37e8f00365f6b88945feb758a60583be7467bcecb064dec6323ff107d1927
```

每臂均为 16/16 有限实测，build error=0，runner error=0；汇总器也验证了候选计数守恒、
精确 quota 和 Draft/XGBoost 架构计数。

### 13.4 六臂结果

| Arm        | 初始化尝试 | Verify 失败 | 搜索 / s | tune / s | XGB 调用/输入 | best / μs | top-3 / μs | top-5 / μs |
| ---------- | ---------: | ----------: | -------: | -------: | ------------: | ---------: | ----------: | ----------: |
| B          |       2240 |        2126 |      123 |  398.086 |     10 / 1280 |     34.656 |      60.670 |      66.505 |
| CSH        |        512 |         262 |      133 |  401.523 |     10 / 1280 |     40.133 |      60.824 |      71.701 |
| D          |       2240 |        2119 |      130 |  395.526 |        2 / 64 |     35.907 |      44.722 |      53.120 |
| D_G2       |       2240 |        2100 |      126 |  393.874 |        2 / 64 |     35.710 |      44.416 |      51.821 |
| D_G1G2     |       2176 |        2269 |      227 |  458.548 |        2 / 64 |     31.682 |      40.451 |      50.266 |
| CSH_D_G1G2 |        512 |         349 |      203 |  469.697 |        2 / 64 |     29.119 |      42.721 |      51.998 |

可以直接支持的结论：

1. CS/H 再次把初始化尝试 `2240→512`、Verify 失败 `2126→262`，但本 seed 搜索
   `123→133 s`，只复现剪枝量，没有复现上一 seed 的墙钟加速。
2. D 两批解析 1280 条候选且 unknown=0；GA 内 XGBoost 为零，末端总输入 64，严格符合
   “Draft 先搜索、XGBoost 后验证”的架构。
3. G2-only 只触发 1 次并成功投影；样本太少，无法估计性能收益。
4. D_G1G2 的 6 个已知越界 proposal 全部投影成功，8015 个候选方向产生 1992 次资源
   方向选择；质量分位有积极信号，但四 proposal replay 使搜索 `130→227 s`。
5. 完整无 G4 臂本 seed 找到最低 best 29.119 μs，但搜索/tune 分别比 B 慢 65.0%/18.0%；
   单个 best 不能替代 time-to-target 和多 seed 统计。

### 13.5 G4 结果

| D_G1G2 的 G4 模式 | 搜索 / s | tune / s | best / μs | top-3 / μs | top-5 / μs | fallback |
| ----------------- | -------: | -------: | ---------: | ----------: | ----------: | -------: |
| off               |      227 |  458.548 |     31.682 |      40.451 |      50.266 |      536 |
| shadow            |      208 |  474.700 |     33.790 |      42.853 |      53.093 |      547 |
| enforce           |      194 |  456.307 |     29.784 |      42.460 |      50.597 |       17 |

shadow 两批观察到 75 次劣解比较，其中反事实接受 55 次；enforce 两批观察到 72 次劣解，
真实概率判定接受 55 次、拒绝 17 次。G4 路径不是空开关，并显著减少了父代回退，但结果
仍然混合：独立臂 best 改善而 top-3/top-5 略差；在 CSH_D_G1G2 上开启 G4 后，best
`29.119→35.662 μs`、top-5 `51.998→63.473 μs`，本次明显退化。不能把 G4 写成已证明
的质量提升。

### 13.6 原始数据与完整性

| 目录                                                               | 内容                       |
| ------------------------------------------------------------------ | -------------------------- |
| `F:\tvm_tmp\second_point_draft_lse_20260905_r1\`                 | 六臂主实验                 |
| `F:\tvm_tmp\g4_shadow_formal_20260905_r1\`                       | D_G1G2 + G4 shadow         |
| `F:\tvm_tmp\g4_enforce_formal_20260905_r1\`                      | D_G1G2 + G4 enforce        |
| `F:\tvm_tmp\full_g4_formal_20260905_r1\`                         | CSH_D_G1G2 + G4 enforce    |
| `experiments/data/first_point_v1_1/equal_valid_e2e_20260904_r2/` | 2026-09-04 CS/H 第一点评估 |

每个新目录均保存命令计划、完成状态、paired summary、逐臂 result/数据库/日志和
`raw_manifest.sha256.json`。G4 的 `compared/accepted_worse/annealed_choices` 暂未进入 paired
summary，复核时读取各臂 `logs/tvm.s_tir.meta_schedule.logging.task_0_main.log`。本次盘点
重新计算了四份 manifest 覆盖的 45 个原始文件，`45/45` SHA256 与记录一致。

### 13.7 尚未解决

- 单 workload、单 seed、16 trials 不能证明稳定性能；下一批至少做 3～5 seeds 和多算子；
- G1 的四 proposal replay 是当前最大新增搜索开销，应消融 trials={1,2,4} 并做缓存；
- G2 触发太少，需要专门构造接近线程上限的 workload/种群统计合法率；
- G4 要看 best-so-far AUC、多样性和多 seed，而不能只看偶然 best；
- G3 已完成真实延迟反馈闭环和 64-trial 计数，但冷启动后 enforce 尚未实际发生；无限 UCB
  的不可用 arm 风险需修正并重测；
- 多轴线程和 shared-memory 仍保守 Unknown/放行，不能声称多资源可行性保持已完成；
- `F:\tvm_tmp` 是临时盘，论文冻结前应把四个原始目录完整归档进项目并保留 manifest。

### 13.8 G3：真实 Runner 奖励实验与未完成的 enforce（2026-09-05）

G3 的完整反馈路径已经接入：GA 保存候选的 mutator lineage，Runner 返回真实延迟后，只有
同时存在已实测父代与成功子代时才给对应 mutator 记一个 outcome。奖励不读取 Draft 或
XGBoost 分数。实验和审计细节见
[`experiments/G3_RUNNER_CREDIT_RESULTS_2026-09-05.md`](experiments/G3_RUNNER_CREDIT_RESULTS_2026-09-05.md)。

冻结协议为 MatMul128、CUDA sm_86、seed 20260905、64 trials、每批 8 条、population 128、
4 轮变异、direction trials=2、G4 off、冷启动阈值 32。shadow/enforce 加载相同 DLL：

```text
F:\tvm\build-prune\lib\tvm_compiler.dll
SHA256 6e0156f6b2b3133100970b50c52e7804ed8130a226d1baa49600fbee24a27474
```

两臂均为 64/64 有限实测、0 build error、0 runner error；GA 内 XGBoost 为 0/0，GA 后
XGBoost 为 8 次、共 256 条，即每批一次 top-32。Enforce 共得到 34 个真实父子 outcome，
14 个子代优于父代，反馈源均为 Runner。

但是这轮**没有真正验证冷启动后的 enforce**。八个批次开始前的累计 credit 为
`0,0,0,5,13,16,23,26`；第 8 批反馈后才从 26 增至 34，首次超过阈值 32，而实验已经
结束。第 8 批所有 GA generation 的原始日志仍为 `cold_start=1`。

最终账本为：

| arm           | accepted | credit | improved |    UCB |
| ------------- | -------: | -----: | -------: | -----: |
| TileSize      |      717 |     29 |       11 | 0.8725 |
| Unroll        |      289 |      5 |        3 | 1.7877 |
| ThreadBinding |        0 |      0 |        0 |    inf |

shadow 臂在第 7 批反馈后已经有 33 个 credit，因此第 8 批记录到了冷启动后的反事实选择：
450 次 UCB 首选全部落到从未成功的 ThreadBinding。它证明无限 UCB 的首选塌缩风险，但
不能冒充 enforce 的最终候选已经塌缩；失败臂的当前-child 临时屏蔽、无虚假奖励重试和
最终成功 arm 仍需重新编译后用至少一个冷启动后批次验证。

Shadow 与 enforce 的 best 分别为 32.3368 和 34.2492 μs，search 为 594 和 601 s，tune
为 1291.510 和 1641.919 s。由于 enforce 从未退出冷启动，且两个运行的 source identity
不同，这些差异不能归因于 G3。当前可以写进成果的是“真实反馈闭环跑通并发现选择风险”，
不能写“G3 已提升性能”。原始目录为：

```text
F:\tvm_tmp\g3_shadow_density_20260905_r1\
F:\tvm_tmp\g3_enforce_collapse_20260905_r1\
```
