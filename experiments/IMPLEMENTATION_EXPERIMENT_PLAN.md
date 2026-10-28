# Pruner 解析先验 + 硬过滤 + 方向性 GA + 温度相对标签：实施与实验方案

> 版本：v1.1（2026-09-02）  
> 目标：形成一套可实现、可归因、可复现的 TVM MetaSchedule 约束引导搜索方案。  
> 当前约束：本文只制定方案，不启动长时间 GPU 实验。

### v1.1 修订摘要

- 构建命令在同一个 `cmd.exe` 进程内显式设置 ASCII `TEMP/TMP`，避免中文用户目录导致构建工具失败。
- 将 P2（拆分 replay/postproc 的前置草稿选择）降为高风险探索项，补充 shadow、回退机制和独立 Go/No-Go 门槛；P1 是最低可交付实现。
- 明确 H-late 与 `VerifyGPUCode` 的功能重叠，只能按净节省时间评价，不能把已有 postproc 成本算成收益。
- E2 增加未经 P/H 选择的候选流和分层实测，解决只用“已测合法数据库”造成的选择偏差。

## 1. 最终要验证的命题

本研究不把 Pruner 的公式或 TVM 已有的相对标签声明为原创。核心命题是：

> 将同一套资源约束信息用于保守硬过滤和方向性变异，并把 Pruner 的解析评分作为已引用的性能先验，可以在不降低最终 kernel 质量的前提下，减少 MetaSchedule 的无效候选和达到目标性能所需的时间。

拆成四个可证伪假设：

- **H1（硬过滤）**：只拒绝静态确定违反硬件上限的候选，已测合法数据库上的误拒数为 0，并减少相应的 Verify/postproc 工作。
- **H2（Pruner 先验）**：Pruner 式解析评分在低成本下能提高候选排序质量或 time-to-target，但它只作软排序，不承担合法性判定。
- **H3（方向性 GA）**：根据违反的资源选择收缩方向，比原始随机 `MutateTileSize` 产生更高比例的合法、非重复后代。
- **H4（标签校准）**：温度化相对标签在小样本下提高性能排序质量；若无稳定收益，则不进入最终系统。

其中 H1 和 H3 是论文主体，H2 是明确引用的外部解析先验，H4 是支撑性消融。

## 2. 当前事实与起点

### 2.1 已经完成

- C++ 硬过滤：`TVM_PRUNE_MODE=0/1/2`，当前只处理静态可知的 `threads_per_block`。
- 模型外软重排：`TVM_SOFT_FILTER_MODE=0/1/2`，当前是 thread-only proxy，不是完整 Pruner 模型。
- GA 投影：`TVM_GA_T2_MODE=0/1/2`，当前只处理能从 flat trace 证明的单物理线程轴。
- 已编译并真实运行过 CUDA；现有 8-trial 数据只属于功能 smoke，不作为性能结论。
- 历史数据库审计中，当前线程硬过滤对 233 个唯一有限实测 schedule 的误拒数为 0。

### 2.2 已知不足

- 基线 seed `20260902` 中，SampleInit 的 Verify 失败为 `927/1024`，明显大于 Evolve 的 `257/2023`。只改最终 top-k 排名不能解决主要浪费。
- 当前 Verify 计数混合了 threads、shared memory、vthread、vector 和 lowering exception，尚不能给出真实失败原因分布。
- `experiments/pruner_prefilter.py` 使用固定 MatMul/SSSRRSRS 下标，只是公式观察原型，不具备通用性。
- 当前 GA 的投影触发频率低，尚不能声称“完整 GA 可行性保持”。
- TVM 当前 XGBoost 已使用 `min_cost / cost` 相对标签，不能把“引入相对标签”当成新修改。

## 3. 系统设计与对照命名

完整链路：

```text
生成/变异 trace
    -> H：保守硬过滤（确定非法才删除）
    -> P：Pruner 解析先验（软排序，保留随机探索）
    -> postproc + VerifyGPUCode
    -> G：下一次变异使用资源违反类型和余量选择方向
    -> C：XGBoost 性能模型
    -> L：温度化相对标签训练 C
    -> build + run
```

正式实验统一使用下列缩写：

| 缩写 | 含义 |
|---|---|
| B | 未启用本研究改动的 MetaSchedule 基线 |
| H | 保守硬过滤 |
| P | 复现并引用的 Pruner 解析先验 |
| G | 方向性、可行性投影的 GA 变异 |
| L | 温度化相对标签 |
| FULL | H + P + G + L |

原则：P 只作软排序；H 才能使用“拒绝”和“零误杀”措辞。P 若舍弃低分候选，应称为搜索选择，不称为非法过滤。

## 4. 实施阶段

### 阶段 I0：冻结环境与补齐可观测性

#### 修改内容

1. 在 `ThreadedTraceApply` 中记录失败发生在哪个 postproc。
2. 将 `VerifyGPUCode` 细分为至少以下原因：
   - `thread_extent`
   - `shared_memory`
   - `vthread`
   - `vector_bytes`
   - `lowering_exception`
   - `unknown_verify_failure`
3. 分开记录 `SampleInit`、`Evolve`、`PickBestFromDatabase` 三个阶段。
4. 每次运行记录源码身份、DLL SHA256、Target、随机种子、全部开关和超参数。

#### 预计修改文件

- `src/s_tir/meta_schedule/utils.h`
- `src/s_tir/meta_schedule/postproc/verify_gpu_code.cc`
- `src/s_tir/meta_schedule/search_strategy/evolutionary_search.cc`
- `experiments/e1_thread_filter.py`

#### 输出计数

```text
stage.generated
stage.trace_replay_failed
stage.postproc_failed[index]
stage.verify_failed[reason]
stage.hard_rejected[reason]
stage.passed
```

#### 通过门槛

- 各阶段满足守恒：`generated = passed + 所有失败 + hard_rejected`。
- mode=shadow 与 mode=off 在同 seed 下的最终 trace 多重集一致。
- 无法归因的 Verify 失败可以存在，但必须单独计为 `unknown`，不能混入 threads。

若守恒不成立，不进入任何性能实验。

### 阶段 I1：把 Pruner 草稿模型做成可引用的解析先验 P

#### 先做来源检查

- 核对 Pruner 论文公式、符号含义和开源代码许可证。
- 公式可依据论文重新实现，并在代码与论文中引用。
- 若复制源代码，必须确认许可证与 Apache TVM 兼容并保留要求的版权信息；否则只按论文 clean-room 重写公式。

#### 实现形式

新增独立模块，避免与硬过滤混成一个无法消融的函数：

```text
src/s_tir/meta_schedule/draft_model.h
src/s_tir/meta_schedule/draft_model.cc
```

建议接口：

```cpp
struct DraftFeatures {
  double p_l0_memory;
  double p_l0_compute;
  double p_l1_memory;
  double p_l1_compute;
  double p_l2_compute;
  double p_l2_memory;
  uint32_t known_mask;
};

struct DraftScore {
  bool known;
  double score;  // (0, 1]
  DraftFeatures features;
};

DraftScore AnalyzeDraftScore(const Schedule& sch, const Target& target);
```

硬件参数通过 `Target` 或显式 profile 表读取。禁止在公式内部静默写死 `sm_86`；缺参数时返回 unknown 和中性分数 1.0。

#### 接入模式

```text
TVM_DRAFT_MODE=0  off
TVM_DRAFT_MODE=1  shadow：计算并记录，不改变顺序
TVM_DRAFT_MODE=2  rank：参与排序
TVM_DRAFT_BETA      解析先验权重
TVM_DRAFT_EPS       随机探索比例
```

第一版组合公式：

```text
log(final_score) = log(max(xgb_score, eps)) + beta * log(max(draft_score, eps))
```

等价于：

```text
final_score = xgb_score * draft_score^beta
```

保留 `TVM_DRAFT_EPS=0.10` 的随机选择，防止经验公式把搜索锁死。

#### 两级落地

- **P1（最低可交付、先完成）**：在已有合法 population 的 XGBoost 分数后重排，验证公式、统计和排序作用。P1 只能声称改变候选排序或 time-to-target，不能声称减少进入 population 之前的 Verify/postproc 工作。
- **P2（高风险探索项）**：将 trace replay 与 postproc 拆开，在 SampleInit oversample 后、昂贵 postproc 前进行草稿选择。它会改动 `SampleInitPopulation` 的核心控制流、随机数消费、并行调度和候选分布，只有通过独立门槛后才进入 FULL。P2 不是完成论文的强制前提。

P2 的流程：

```text
生成 k * population_size 个 trace（k 初始取 2）
    -> replay 到 schedule，但不执行完整 postproc
    -> H-early：只拒绝能够证明的非法候选
    -> P 打分
    -> 90% 取高分 + 10% 随机
    -> 对入选者执行完整 postproc/Verify
```

P2 必须提供独立开关：

```text
TVM_DRAFT_PREFILTER_MODE=0  off：保持现有 SampleInitPopulation
TVM_DRAFT_PREFILTER_MODE=1  shadow：给全部 replay 后候选打分，但仍全部走原流程
TVM_DRAFT_PREFILTER_MODE=2  select：oversample 后执行 90/10 选择
```

实现时保留原始 `ThreadedTraceApply::Apply` 路径。`select` 在以下任一条件发生时按批次回退到原路径：

- P 特征 known coverage 低于预注册阈值。
- 选择后无法凑够 `init_min_unmeasured`。
- replay/postproc 拆分后的 schedule 或 trace 不满足等价性检查。
- 新路径发生异常或达到独立的失败次数上限。

P2 shadow 的目的不是要求和 off 有相同运行时间，而是证明：不执行选择时，它不改变返回候选、trace 决策和后续 RNG 状态。P2 select 必然改变候选分布，不能再用逐 trace 相等作为正确性标准。

#### P2 独立风险门槛

- **语义门槛**：拆分前后的 postproc 输入 schedule、最终 IR 和 trace 在 shadow 模式下逐候选一致。
- **随机性门槛**：shadow 不额外消费会影响原搜索的 RNG；用于打分的过程必须是确定性的。
- **可用性门槛**：select 在所有开发 workload 上都能达到 `init_min_unmeasured`，回退率低于 5%。
- **成本门槛**：`避免的 postproc/Verify 时间 - 额外 replay、打分和 oversample 时间 > 0`。
- **质量门槛**：3 seeds × 64 trials 下，最终最佳延迟中位数相对 P1 退化不超过 2%。

任何一项不满足，P2 不进入正式 FULL；保留 P1，并在 limitation 中说明没有声称前置减少 Verify。

Pruner 六项中任一项无法通用提取时，该项必须标 unknown，不能使用固定 `I[2]`、`J[2]` 等算子下标伪装通用。

#### 单元测试

- 同一结构重复计算分数完全一致。
- unknown target 返回中性分数，不删除候选。
- 构造增大 threads/shared/访存不对齐的样例，验证对应项按公式变化。
- shadow 不改变 RNG 消耗和候选顺序。
- MatMul 原型上的新旧公式逐项对齐；差异必须能由 IR/trace 语义解释。

### 阶段 I2：扩展硬过滤 H

硬过滤分两层：

1. **H-early**：trace replay 后的保守预检，只在下界已经超过上限时拒绝。
2. **H-late**：当前 `VerifyGPUCode` 前的精确静态检查，覆盖能从此时 IR 机械读出的 threads/shared 子类。

H-late 与 `VerifyGPUCode` 内部的线程/shared 检查存在功能重叠。它的成立依据不是“实现了同一项检查”，而是它是否能以更低成本提前得到同样的拒绝结论。H-late 位于前序 postproc 之后，因此：

- 不能节省已经执行的 trace replay 和前序 postproc。
- 只能节省被它拦截候选的 `VerifyGPUCode` 自身及其后续步骤。
- 如果 H-late 内部复用了与 `VerifyGPUCode` 同样昂贵的 lowering/checker，就没有加速价值。
- 即使没有净加速，它仍可作为统一结构特征和失败归因的实现基础，但必须与性能贡献分开表述。

H-late 的主评价量定义为：

```text
H_late_net_saved_time
  = avoided_verify_and_later_time
  - H_late_analysis_time_for_all_checked_candidates
```

同时报告 `H_late_recall_by_reason`。只有净节省大于 0 且置信区间不跨 0，才把 H-late 写成加速组件；否则论文加速主张只归于 H-early/P2，H-late降级为兜底与可观测性组件。

建议扩展 `ThreadFeasibilityResult` 为：

```cpp
struct StructuralFeatures {
  bool thread_known;
  int64_t thread_x;
  int64_t thread_y;
  int64_t thread_z;
  int64_t threads_per_block;

  bool shared_known;
  int64_t shared_bytes;

  int64_t warp_size;
  int64_t max_threads_per_block;
  int64_t max_shared_memory_per_block;
};
```

拒绝规则只有：

```text
thread_known && threads_per_block > max_threads_per_block
shared_known && shared_bytes > max_shared_memory_per_block
```

unknown 一律放行。occupancy、warp 利用率和算存比永远不能进入硬拒绝条件。

#### 通过门槛

- 对所有可回放、有限实测历史记录做审计，误拒数必须为 0。
- 人工构造边界：1023/1024 通过，1025 拒绝；shared limit 相同。
- 多轴、重复轴、动态 extent、warp_execution、多个 PrimFunc 都有测试。
- 每个被 H 拒绝的样本都能由对应的 Verify 子规则确认，而不是用混合 Verify 结果间接推断。
- H-early 和 H-late 的 `checked/rejected/time_saved` 分开计数，禁止把两层收益合并后反推。
- H-late 必须报告上述净节省；不以“检查器本身更简单”代替实际计时证据。

### 阶段 I3：扩展方向性 GA G

将当前“单轴溢出后沿 divisor lattice 回退”扩成两个子模式：

- **G1：可行性投影**。proposal 合法则保持原 proposal；越界才沿影响该资源的因子方向投影。
- **G2：余量内方向选择**。在多个合法 divisor 中，根据资源余量和 P 的解析增量选择方向，同时保留 epsilon 随机探索。

统一资源—因子影响关系：

```text
SamplePerfectTile factor
    -> Split 输出 loop
    -> Fuse（如存在）
    -> Bind(threadIdx.x/y/z) 或 shared tile footprint
    -> 影响 threads/shared
```

禁止按“第 2 个 split 因子一定是线程因子”硬编码。

建议模式：

```text
TVM_GA_T2_MODE=0  off
TVM_GA_T2_MODE=1  shadow
TVM_GA_T2_MODE=2  project（G1）
TVM_GA_T2_MODE=3  directed（G1 + G2）
```

G2 候选选择：

```text
以 1-epsilon 概率：选 hard-feasible 且 draft_delta 最大的 divisor
以 epsilon 概率：在 hard-feasible divisors 中随机选
```

默认 `epsilon=0.10`，只在开发集选择一次后冻结。

#### 必记指标

```text
proposal_total
proposal_known / proposal_unknown
proposal_initial_legal
proposal_initial_illegal[reason]
projected_success / projected_failure
directed_choices / random_choices
post_projection_verify_pass / fail[reason]
duplicate_offspring
fallback_to_parent
divisor_steps
```

#### 通过门槛

- tile 因子乘积、`max_innermost_factor` 和 trace 可回放性保持不变。
- shadow 与 off 同 seed 逐 proposal 一致，不额外消耗 RNG。
- G1 对其声称覆盖的线程溢出，投影后线程合法率为 100%。
- G 不得显著降低后代多样性：唯一 trace 比例相对 B 下降不超过 10%；否则提高 epsilon 或缩小定向范围。

### 阶段 I4：实现温度相对标签 L

当前 TVM 基线为：

```text
y_i = best_cost / cost_i
```

修改为：

```text
y_i(gamma) = (best_cost / cost_i)^gamma
```

在 `XGBModel` 构造函数增加显式参数：

```python
label_gamma: float = 1.0
```

不要只用隐式环境变量。实验 driver 必须把 gamma 写入 `result.json`。训练和 validation 使用同一变换。

候选集合：

```text
gamma in {0.5, 0.75, 1.0, 1.5, 2.0}
```

`gamma=1.0` 必须与当前 TVM 数值一致。初始重点观察 `0.5`：当前 PackSum 同时用标签作样本权重，开平方会提高慢样本的训练权重。

注意默认 `num_warmup_samples=100`：32/64 样本时在线 `predict()` 仍可能返回随机分数。因此：

- 离线标签实验显式训练后直接评估，或统一设 `num_warmup_samples=0`。
- 端到端实验所有对照使用相同 warmup；不能只给 L 降低 warmup。
- 若保留默认 warmup，端到端的 L 结论主要看 128/256 trials。

#### 预计修改文件

- `python/tvm/s_tir/meta_schedule/cost_model/xgb_model.py`
- 新增对应 Python 单元测试
- `experiments/eval_cost_model.py`

## 5. 实验驱动与结果格式

新建统一入口，现有 `e1_thread_filter.py` 保留为历史 smoke：

```text
experiments/workloads.py
experiments/run_constraint_guided.py
experiments/eval_cost_model.py
experiments/summarize_constraint_guided.py
```

建议命令接口：

```powershell
python experiments\run_constraint_guided.py run `
  --arm B `
  --workload matmul_m128_n128_k128 `
  --seed 20260902 `
  --trials 64 `
  --work-root F:\tvm\experiments\data\constraint_guided
```

每个运行目录只写一次，不覆盖旧结果：

```text
experiments/data/constraint_guided/
  <hardware>/<workload>/<arm>/seed_<seed>/
    result.json
    database_workload.json
    database_tuning_record.json
    candidate_stream.jsonl        # E2-B/P2 shadow 候选流；未启用时可不存在
    logs/
    environment.json
```

`result.json` 至少包含：

- git HEAD、tracked diff hash、untracked source hash、DLL 路径与 SHA256。
- GPU 名称、driver、CUDA、Target 全量属性。
- arm、所有 mode、beta、epsilon、gamma、warmup。
- P1/P2 所处阶段、oversample factor、P2 回退次数和回退原因。
- workload 和 shape。
- 生成、H-early/H-late、各 postproc 失败、GA、模型和测量计数。
- H-early/H-late 分层计时，以及 H-late 的避免时间、分析时间和净节省。
- P 的 known coverage、分数分位数、打分时间和各分位区间 Verify 结果。
- search/build/run/update 的分段时间。
- 每次测量的 trace hash、延迟和累计 best-so-far。
- 运行状态：`smoke`、`screening` 或 `formal`。

## 6. Workload 与硬件矩阵

### 6.1 开发集：只用于调参与排错

不得出现在最终主表中：

| 类型 | Workload |
|---|---|
| GEMM | M=N=K=128 |
| GEMM | M=N=K=512 |
| Reduction | `(1024, 1024)`，归约最后一维 |

### 6.2 正式测试集

每类至少两个 shape；最终按实际 TVM schedule rule 可运行性冻结：

| 类型 | 建议 shape |
|---|---|
| GEMM | `(256,256,256)`；`(1024,1024,4096)` |
| Conv2D | NCHW `(1,64,56,56)`、`K=128, R=S=3`；`(1,128,28,28)`、`K=256, R=S=3` |
| Reduction | `(64,128,128)`；`(1024,4096)`，归约最后一维 |
| Softmax | `(128,1024)`；`(1024,4096)` |
| LayerNorm | `(128,768)`；`(2048,4096)` |
| Add+ReLU | `(1024,1024)`；`(4096,4096)` |

如果某一 workload 的基线本身无法得到有效记录，先归因修复；不能把“基线 0 个合法候选”当作我们的提升。

### 6.3 硬件

- 开发与正确性：RTX 3050 Ti Laptop，`sm_86`。
- 正式主结果：A100，`sm_80`。
- 强版本：再加入一张消费级或更新架构 GPU，验证硬件 profile 不是只对 A100 调出来的。

P 的硬件参数按设备分别配置，但算法和代码不按算子修改。

## 7. 实验矩阵

### E0：失败原因与基线成本画像

配置：B，开发集和正式 workload，3 seeds，8-trial smoke 后再用 64 trials。

回答：

- SampleInit/Evolve 各自失败多少？
- 每种 Verify 原因占多少？
- 每类失败消耗多少 replay/postproc 时间？
- H 和 G 应该优先覆盖哪类失败？

输出：失败原因堆叠图、阶段成本瀑布图。

### E1：硬过滤正确性与覆盖率

对照：`H-off`、`H-shadow`、`H-enforce`。

数据：

1. 全部历史可回放、有限实测记录。
2. baseline 新生成的所有候选。
3. 人工边界与随机生成的非法候选。

主指标：

- 已测合法记录误拒数，必须为 0。
- 每种原因的拦截数。
- 被拦截候选原本会在哪个 Verify 子规则失败。
- 每个成功拦截节省的后续处理时间。
- H-early/H-late 各自的分析开销、避免开销和净节省；H-late 不计入已经发生的 replay/前序 postproc。

### E2：Pruner 解析先验质量

E2 使用两套用途不同的数据，不能只在已经被 MetaSchedule 选中并测量的合法数据库上评价。

#### E2-A：已测合法集——只回答合法候选中的性能排序

对已有合法实测候选计算 P 分数，与真实吞吐比较：

- Spearman rho。
- NDCG@16。
- top-16 recall。
- 解析耗时/候选。
- known coverage。

这一结果只能说明“P 在已经合法且已经被选中测量的候选中能否排序性能”。由于这些记录已经经过 postproc、Verify 和原代价模型选择，分布偏向合法与高分区域，不能用它证明 P 对原始候选流的筛选收益。

#### E2-B：未过滤候选流——回答 P 对实际搜索漏斗的影响

在 B 模式下，从 SampleInit 和 Evolve 旁路记录未经 H/P 选择的候选。对每个候选先计算 P，之后仍走原始流程，记录：

```text
candidate_id / trace_hash / stage
draft_score / six_penalties / known_mask
postproc_pass / verify_pass / failure_reason
measured / latency（若实测）
```

采集过程必须满足：

- P 处于 shadow，不改变候选顺序、RNG 和是否进入 Verify。
- pass/fail 样本都保留，不能只保存数据库记录。
- SampleInit 和 Evolve 分开分析。
- 按 `draft_score` 分位区间统计 Verify 通过率，观察 P 是否把候选引向更容易通过或更难通过的区域。

失败候选没有真实运行时间，不能人为赋成极慢性能。对通过 Verify 的候选，按 P 分数分层抽样实测，而不是只测 P 的 top-k：

```text
将 draft_score 分成 5 个分位区间；
每个区间等量随机抽取合法候选；
去重后 build/run；
比较各区间的真实延迟分布。
```

这样避免“只测高分候选，再用这些数据证明高分候选更好”的循环选择偏差。若预算有限，开发阶段每个 workload 至少收集 200 个未过滤候选，并在每个分位区间实测至少 20 个合法、唯一 trace；达不到数量时如实报告 coverage，不跨 workload 拼成伪样本量。

E2-B 主指标：

- P 分数各分位区间的 Verify 通过率及失败原因分布。
- 每得到 N 个 Verify 合法候选预计需要处理的候选数。
- 分层实测候选中 P 与真实吞吐的 Spearman/NDCG。
- P top-q 选择相对随机选择的合法产出率和性能 regret。
- 若模拟 P2：额外 oversample/replay/打分成本与预计避免的 postproc/Verify 成本。

只有 E2-A 排序好，最多支持 P1；E2-B 同时显示正的成本收益且通过 P2 风险门槛，才允许推进 P2。

在线开发集参数：

```text
beta in {0.05, 0.10, 0.20, 0.40}
draft_epsilon in {0.05, 0.10, 0.20}
oversample_factor in {1, 2, 4}
```

选择规则：先保证最终质量不退化，再最小化 time-to-target。参数选定后冻结，不能用正式集重新调参。

### E3：方向性 GA

对照：

```text
B-G0：原始 MutateTileSize
G-shadow：只观察可投影 proposal
G1：越界投影
G2：越界投影 + 余量内方向选择
```

主指标：

- mutation 后 Verify 通过率。
- 每代唯一合法后代数。
- fallback-to-parent 比例。
- 每 1000 次 mutation 的非法子代数。
- 达到相同性能所需代数/测量数。
- 唯一 trace 比例，检查搜索坍缩。

必须按 SampleInit 和 Evolve 分开报告；G 只能直接改善 Evolve，不能拿 SampleInit 的变化给 G 记功。

### E4：温度相对标签

数据划分：

- 先按 trace hash 去重。
- 同一 trace 的重复测量不能跨 train/test。
- 每个 workload 内以训练前缀 32/64/128/256 构建学习曲线。
- `best_cost` 只由训练集计算，不能使用 test 最优值。

对照：`gamma={0.5,0.75,1.0,1.5,2.0}`。

指标：

- Spearman rho（主）。
- NDCG@16。
- top-k regret：模型 top-k 中真实最佳与测试集真实最佳的差距。
- RMSE（辅，不用作唯一结论）。
- 5 个随机数据划分的均值和置信区间。

进入 FULL 的门槛：至少在 128/256 两个样本档中，多数 workload 的排序指标优于 `gamma=1`，且没有明显恶化端到端 time-to-target。否则 FULL 使用 `gamma=1`，诚实报告 L 无收益。

### E5：模块消融

正式六臂：

| Arm | H | P | G | L | 回答的问题 |
|---|---:|---:|---:|---:|---|
| B | 0 | 0 | 0 | gamma=1 | 原始基线 |
| P | 0 | 1 | 0 | gamma=1 | Pruner 先验单独贡献 |
| H | 1 | 0 | 0 | gamma=1 | 硬过滤单独贡献 |
| H+G | 1 | 0 | 1 | gamma=1 | 方向性 GA 增量 |
| H+P+G | 1 | 1 | 1 | gamma=1 | 三个搜索模块组合 |
| FULL | 1 | 1 | 1 | 选定 gamma | 标签的最后增量 |

必要时增加 `P+G`，检查 G 是否依赖 H；不要先跑完整 2^4=16 臂。

### E6：端到端正式实验

预算：每个正式 workload、每个 arm 使用相同的 5 个 seeds：

```text
20260902, 20260903, 20260904, 20260905, 20260906
```

每次 256 trials，每 32 trials 保存 checkpoint：

```text
32 / 64 / 96 / 128 / 160 / 192 / 224 / 256
```

主指标：

1. **time-to-target**：达到参考最优性能 90%、95%、99% 所需墙钟时间和测量数。
2. **best-so-far AUC**：归一化性能—测量次数曲线下面积。
3. **最终质量**：256 trials 后的最佳延迟，FULL 相对 B 不应稳定退化。
4. **搜索成本**：search-only 时间，以及 replay/postproc/build/run 分解。
5. **有效性**：合法候选率、Verify 失败率、唯一候选率。

参考最优值使用同一 workload 所有 arm、所有 seeds 的有效记录并集中的最优值；同时单独报告相对 B 最终中位数，避免 reference 选择掩盖退化。

## 8. 统计规则

- 所有 arm 使用完全相同的 seed 列表、trial budget、population 和测量参数。
- 以 workload × seed 配对，不只比较总平均。
- 报告中位数、IQR 和按 workload/seed 配对 bootstrap 的 95% CI。
- 需要显著性检验时用配对 Wilcoxon，并同时报告 effect size；不能只报 p-value。
- 单 seed 或 8 trials 只能标注 smoke。
- 调参开发集与正式报告集分离；正式运行后不再修改 beta、epsilon、gamma。
- 搜索耗时和端到端耗时分开，不能把 GPU runner 噪声解释成过滤器开销。

## 9. Go/No-Go 门槛

| 阶段 | Go 条件 | No-Go 后处理 |
|---|---|---|
| I0 可观测性 | 计数守恒；shadow 不改变轨迹 | 先修日志与 RNG，不跑性能 |
| H | 历史合法记录误拒 0 | 修分析器或缩小硬规则，不放宽误拒容忍度 |
| P | known coverage 足够，且开发集排序或 time-to-target 有正收益 | 保留为外部对照，不进 FULL |
| P2 前置选择 | 语义/RNG/可用性门槛通过；净节省大于 0；质量退化不超过 2% | 回退 P1；不声称减少前置 Verify/postproc |
| G | 覆盖范围内投影后合法率 100%，唯一 trace 下降不超过 10% | 缩小投影范围或提高 epsilon |
| L | 128/256 样本多数 workload 排序改善 | FULL 保持 gamma=1 |
| FULL | time-to-target 改善且最终性能基本持平 | 逐模块回退，以 E5 定位冲突 |

“最终性能基本持平”的预注册判断：FULL 相对 B 的最终最佳延迟中位数退化不超过 2%，且不存在跨多数 workload 的一致退化。

## 10. 执行顺序与计算预算

严格按下面顺序，不并行开展尚未过门槛的模块：

1. I0 失败原因与守恒计数。
2. E0 baseline 成本画像。
3. H 的历史数据库零误拒审计与 shadow/enforce。
4. P 的公式复现与 E2-A/E2-B 数据采集，先完成 P1 shadow/rank。
5. 仅当 E2-B 支持前置漏斗收益时实现 P2；P2 未过独立门槛立即回退 P1。
6. G1 多轴/多资源扩展和 shadow/project。
7. L 的纯离线数据库实验。
8. 开发集 `3 workloads × 3 seeds × 64 trials` 筛选 beta/epsilon/gamma。
9. 冻结全部参数并生成 `frozen_config.json`，同时记录 FULL 使用 P1 还是 P2。
10. E5/E6 正式六臂，5 seeds × 256 trials。
11. 汇总脚本一次性生成表格、曲线和失败原因图。

为了控制 GPU 预算，正式实验分两批：

- **第一批门槛实验**：GEMM、Reduction、Softmax 各一个 shape，六臂全部跑完。
- **第二批完整实验**：第一批 FULL 通过门槛后，再扩展剩余 workload。

## 11. 构建与最小验证

每次 C++ 修改后：

```powershell
cmd.exe /d /s /c 'call "E:\vs\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && set "TEMP=F:\tvm_tmp" && set "TMP=F:\tvm_tmp" && set "CCACHE_DISABLE=1" && cmake --build build-prune --parallel 12'
```

`TEMP/TMP` 必须在同一个 `cmd.exe` 命令链中设置，不能只在外层 PowerShell 假定已经生效。`F:\tvm_tmp` 需要预先存在并可写。继续通过 `cmake --build` 调用已配置的生成器，不在方案中假定 build-prune 一定由某个特定 `ninja.exe` 生成。

先运行最小相关测试，再跑 smoke：

```powershell
D:\conda_envs\tvm-build-venv\python.exe tests\python\s_tir\meta_schedule\test_meta_schedule_thread_feasibility.py
D:\conda_envs\tvm-build-venv\python.exe tests\python\s_tir\meta_schedule\test_meta_schedule_thread_filter_modes.py
D:\conda_envs\tvm-build-venv\python.exe tests\python\s_tir\meta_schedule\test_meta_schedule_mutator_mutate_tile_size.py
```

每个新模式都必须依次通过：

```text
单元测试 -> off/shadow 等价性 -> 8-trial smoke -> 3×64 screening -> 正式实验
```

## 12. 最终论文图表清单

- **表 1**：workload、shape、硬件和 baseline 失败原因分布。
- **表 2**：H 的零误拒、覆盖率与节省的 postproc 时间。
- **表 3**：P 的 Spearman/NDCG/known coverage 和单候选开销。
- **表 4**：G 的非法后代率、投影成功率、唯一后代率。
- **表 5**：B/P/H/H+G/H+P+G/FULL 六臂端到端结果。
- **图 1**：完整系统数据流，标明哪些是引用组件、哪些是本文贡献。
- **图 2**：SampleInit/Evolve 失败原因堆叠图。
- **图 3**：best-so-far vs measured trials。
- **图 4**：best-so-far vs wall-clock time。
- **图 5**：gamma 在 32/64/128/256 样本下的排序学习曲线。
- **图 6**：各模块 time-to-target 的配对 speedup 分布。

## 13. 结果解释边界

- 若 H 只拦下很少 threads 失败，应结论为“正确但覆盖窄”，不能声称显著加速。
- H-late 若与 `VerifyGPUCode` 做同等昂贵的工作，或净节省置信区间跨 0，只报告其正确性/归因价值，不报告加速。
- 若 P 改善最终质量但增加 Verify 失败，应同时报告，不能只展示最快 kernel。
- P 在已测合法数据库上的相关性不能外推为前置筛选有效；P2 主张必须由未过滤候选流和分层实测支持。
- P2 未通过独立门槛时，FULL 使用 P1；不能因为 P2“更像 Pruner”而保留负收益路径。
- 若 G 只减少 Evolve 失败，不能把 SampleInit 的高失败率归给 G 解决。
- 若 gamma 没有稳定收益，保留 TVM 默认 `gamma=1`，L 作为负结果消融。
- 若只有 MatMul 有效，论文定位必须缩成特定 schedule family；只有跨算子无需改规则，才能使用“通用 retrofit”。
- Pruner 的六项公式、TVM 的 XGBoost/相对标签都明确引用；本文贡献集中在约束分析的共享、硬过滤边界和现有 GA 的方向性改造。
