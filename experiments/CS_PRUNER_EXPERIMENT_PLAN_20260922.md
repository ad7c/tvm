# CS 扩展与 Draft/GA 论文实验说明

> 历史方案保留。2026-10-08之后的执行顺序、预算、数据状态与安全边界以
> [实验与论文联动计划v1.4](EXPERIMENT_PLAN_20261008.md)为准。

**版本**：v1.0  
**编写日期**：2026-09-22  
**研究对象**：TVM `s_tir` GPU schedule 搜索中的 CS（Constraint Screening）、H（硬筛）、Draft/SA、GA 和代价模型组合  
**当前状态**：已有真实 C++/CUDA 实验，但还没有完成论文级别的安全性、召回率和相同搜索资源对照实验。

## 1. 实验目的

本阶段不是继续堆叠更多启发式规则，而是用可复现实验证明下面四个问题：

1. CS 是否能在 lowering/Verify 之前减少必然非法候选？
2. CS 是否会误删合法候选，特别是高性能候选？
3. 在相同真实搜索资源下，CS 是否得到更好的最终结果，或者至少用更少时间得到相同结果？
4. Draft/GA 是否只是减少了代价模型调用，还是确实改善了候选质量？

Pruner 论文的实验启发是必须同时测量：候选质量（Best-k/Top-k）、搜索时间、代价模型输入规模、最终实测性能和多算子稳定性。Pruner 的 Draft 是性能软排序，不是合法性证明；因此本项目仍保持如下边界：

```text
CS/H：只能拒绝“可证明非法”的候选；未知情况放行
Draft/SA：只负责性能软评分，不删除候选
XGBoost：只在规定阶段排序，不参与 Draft-GA 变异方向
GA：单独做消融，不能把 G1/G2 的结果混写成 CS 的收益
```

## 2. 已完成实验（按日期）

### 2026-09-02：硬筛、软评分和 GA T2/G2 功能验证

实现和验证了三个独立开关：

- `TVM_PRUNE_MODE=off/shadow/enforce`：CS/H 硬筛；
- `TVM_SOFT_FILTER_MODE=off/shadow/rerank`：只重排、不删除候选的 thread proxy；
- `TVM_GA_T2_MODE=off/shadow/project`：对可证明的 thread overflow 变异尝试投影。

主要结果：

- 硬筛 shadow 在当前 MatMul smoke 中未观察到误拒，但混合 `VerifyGPUCode` oracle 的召回率很低，不能把它解释为纯线程约束分类精度；
- 硬筛能从 trace 中提前拦截一部分 `threads_per_block` 超限；
- 软评分保持候选数量不变，累计分析开销为毫秒级，但单 seed 下 rerank 增加了 Verify 失败，尚不能声称最终性能提升；
- G2 能发现并处理少量 thread-overflow mutation，但触发频率低，未形成稳定的端到端加速；
- G1/G3/G4 没有形成可用于论文主结论的稳定收益，G3 已暂时放弃。

证据文档和数据：

- [FILTER_SOFT_GA_RESULTS_2026-09-02.md](FILTER_SOFT_GA_RESULTS_2026-09-02.md)
- `experiments/data/e1_thread_filter/`
- `experiments/data/e2_soft_ga/`
- `experiments/data/e3_final/`
- `experiments/data/e4_soft_weight/`
- `experiments/data/e5_soft_multiseed/`
- `experiments/data/second_point_g3_pair96_20260905_r1/`

### 2026-09-03：Pruner 风格 Draft 与 XGBoost 组合

搭建了 Pruner 风格 Symbol-based Analyzer 草稿模型：

- 根据 statement 和硬件参数估计计算/访存成本；
- Draft 阶段不调用 XGBoost；
- GA 阶段不调用 XGBoost；
- 每个搜索 batch 结束后，才把 Draft archive 的 top-k 交给 XGBoost；
- unknown 使用中性分数，不删除候选。

当前实现位置：

- `src/s_tir/meta_schedule/draft_model.h`
- `src/s_tir/meta_schedule/draft_model.cc`
- `src/s_tir/meta_schedule/draft_guided_ga.h`
- `src/s_tir/meta_schedule/draft_guided_ga.cc`
- `src/s_tir/meta_schedule/search_strategy/evolutionary_search.cc`
- `tests/python/s_tir/meta_schedule/test_meta_schedule_pruner_draft_model.py`

已验证：

- Draft-GA 内部 XGBoost 调用数为 0；
- XGBoost 只在 batch 末尾处理 Draft archive；
- 组合路径能真实编译并运行；
- 但 Draft 与真实延迟的 Spearman 相关性约为 0.314，说明目前只能作为弱指导，不能单独作为可靠性能排序器。

证据文档和数据：

- `experiments/IMPLEMENTATION_LOG_V1_1.md`
- `experiments/data/second_point_v1_0/`
- `experiments/analyze_pruner_draft_results.py`
- `experiments/analyze_hard_draft_combo.py`
- `experiments/run_second_point_draft_lse.py`

### 2026-09-04：CS+H 相同合法候选配额实验

这是目前第一点优化最完整的一次单 seed 配对实验：

- MatMul128；
- seed `20260904`；
- 两个方案都精确产生相同数量的未测合法候选；
- 后续使用相同 GA、编译、runner 和 GPU 测量流程。

结果：

| 指标 | Baseline | CS+H |
|---|---:|---:|
| 初始化尝试数 | 2048 | 512 |
| 初始化通过数 | 253 | 287 |
| 凑齐 quota 时间 | 154.60 s | 112.34 s |
| 完整搜索时间 | 223 s | 138 s |
| Verify 失败 | 约 2079（多 seed 均值） | 约 278（多 seed 均值） |

这证明 CS/H 对“减少废候选”和“减少 Verify 失败”有效，但单 seed 的 best latency 不能作为最终性能结论。

证据文档和数据：

- [FIRST_POINT_V1_1_RESULTS.md](FIRST_POINT_V1_1_RESULTS.md)
- [FIRST_POINT_V1_1_IMPLEMENTATION_LOG.md](FIRST_POINT_V1_1_IMPLEMENTATION_LOG.md)
- `experiments/data/first_point_v1_1/equal_valid_e2e_20260904_r2/`
- 汇总文件：`paired_summary.json`、`paired_summary.csv`、`raw_manifest.sha256.json`

### 2026-09-05：GA/G1/G2/G3 探索

已做过 Draft-GA、G1/G2、G3 runner credit 和退火方向的探索。当前结论：

- G2 projection 功能成立，但成功投影数量很少，跨 seed 不稳定；
- G1 方向引导在当前预算下可能过早收窄搜索，不能作为已证实收益；
- G3 依赖真实 runner 奖励，实验复杂度和环境噪声较大，暂时停止；
- GA 相关结果不能替代 CS 的安全性和召回率实验。

证据文档和数据：

- `experiments/results/g1_single_proposal_v2/`
- `experiments/data/second_point_g3_smoke_20260905_r1/`
- `experiments/data/second_point_g3_pair96_20260905_r1/`
- [G3_RUNNER_CREDIT_RESULTS_2026-09-05.md](G3_RUNNER_CREDIT_RESULTS_2026-09-05.md)

### 2026-09-14：正式主线和多算子 smoke

完成了 MatMul128、MatMul256、Reduction128、Conv2D、Elementwise、Fusion 的真实编译和 CUDA runner smoke。

主要观察：

- MatMul128 的 CSH 平均生成候选约 3208 → 1516；
- VerifyGPUCode 失败约 2079 → 278；
- Conv2D 可以运行，CS/H+Draft 组合在该次 smoke 中得到较低的 best latency；
- Elementwise/Fusion 只有 6 条合法 trace，适合作为小搜索空间边界案例，不适合作为主要平均结果；
- Reduction/Elementwise 某些 workload 没有可触发的硬拒绝，属于 applicability boundary，不是模块失败。

证据文档和数据：

- [FORMAL_20260914_RESULTS.md](FORMAL_20260914_RESULTS.md)
- `experiments/results/formal_mainline_20260914/`
- `experiments/results/multiseed_multiop_20260914/`
- `experiments/results/conv_elementwise_20260914/`
- `experiments/results/conv_elementwise_20260914_r2/`
- `experiments/results/conv_elementwise_20260914_r3/`

### 2026-09-15：MatMul128 三 seed 重复实验

完整的 B、CSH、D+G2 三 seed 结果为：

| Arm | 平均 best latency | 相对 B |
|---|---:|---:|
| B | 39.0023 us | 1.000 |
| CSH | 38.9543 us | -0.12% |
| D+G2 | 37.0123 us | -5.10% |

但 D+G2 的逐 seed 变化为 `-12.52%/-5.40%/+3.21%`，说明目前不稳定。CSH 的最终质量基本持平，但候选生成效率和 Verify 失败数量显著改善。FULL 的第三个 seed 曾出现 builder timeout，不能直接并入正式平均。

证据文档和数据：

- [FORMAL_MULTISEED_20260915_RESULTS.md](FORMAL_MULTISEED_20260915_RESULTS.md)
- `experiments/results/formal_multiseed_matmul128_20260915/`
- `experiments/results/formal_multiseed_matmul128_20260915_rerun/`
- `experiments/results/msr/`

### 2026-09-21：ResNet18 端到端尝试（暂不纳入论文结果）

运行了独立的 `experiments/e2e_resnet.py`：

- baseline 8 trial 和 1000 trial；
- Full 1000 trial；
- 真实 TVM 编译和 GPU runner 成功产生记录。

但当前 baseline 结果的算子名称解析为 `?`，只得到一个聚合 best，而 Full 记录了 32 个算子的分项延迟，二者统计口径不一致。因此这批数据只能作为工程诊断，不能作为论文对照结果。

原始数据在 `F:/tvm_tmp/e2e_resnet/`，该目录是历史脚本默认路径；正式实验必须迁移到 `F:/tvm/experiments/results/`，避免把外部临时目录当作论文证据。

## 3. 当前已经可以下的结论

目前可以严谨地写：

1. 当前 CS/H 能显著减少初始化阶段线程非法候选和后续 VerifyGPUCode 失败。
2. 在 MatMul128 的相同合法候选配额实验中，CS+H 减少了凑齐 quota 的时间。
3. CS/H 的分析路径已经接入真实 C++ TVM 和 CUDA runner，off/shadow/enforce 开关可复现。
4. Draft-GA 阶段可以做到不调用 XGBoost，XGBoost 只在 Draft archive 末端调用。
5. G1/G2/完整组合目前没有稳定的跨 seed 最终性能收益。

目前不能写：

- CS 在所有算子和硬件上都零误删；
- CS 一定提高最终 kernel latency；
- Draft 分数可以替代 XGBoost；
- G1/G2 已经普遍改善 GA；
- 当前结果已经证明跨平台泛化。

## 4. 仍需完成的实验

### E1：CS 零误删与边界安全审计（P0）

**目的**：把现有混合 Verify oracle 升级成真正的安全性证据。

**数据来源**：

- `experiments/data/hard_filter_v2_database_audit.json`
- `experiments/results/formal_multiseed_matmul128_20260915/`
- 所有包含 `database_tuning_record.json` 的正式 run 目录。

**方法**：

1. 收集所有已成功 lowering、build、runner 和 GPU 实测的合法候选；
2. 对每条候选执行 CS shadow；
3. 统计每条规则的 `checked/predicted/unknown/FP`；
4. 另构造边界 trace：等于上限、刚超过上限、多轴、warp_execution、动态 extent、冲突绑定；
5. 只有合法数据库 FP=0 且边界行为正确的规则才允许进入 enforce。

**输出**：

`experiments/results/paper_v1/e1_cs_safety/`

必须包含 `result.json`、候选 manifest、规则级 confusion matrix 和完整日志。

### E2：Best-k/Top-k 候选保留率（P0）

**目的**：证明 CS/Draft 减少候选时没有丢掉真正高质量解。

**数据来源**：

- `experiments/results/formal_mainline_20260914/`
- `experiments/results/formal_multiseed_matmul128_20260915/`
- `experiments/results/conv_elementwise_20260914*`
- 后续生成的完整候选数据库。

**方法**：

1. 在每个算子上生成一个尽可能完整的候选池；
2. 对所有候选记录真实 Verify、build 和 GPU latency；
3. 比较 B、CS、CS+H、CS+Draft、CS+H+Draft；
4. 计算 `Best-1/5/10`、`Top-1/5/10 recall`；
5. Draft archive 至少测试 16、32、64、128、256 五个规模。

**输出**：

`experiments/results/paper_v1/e2_bestk_recall/`

**完成标准**：不能只报告筛掉了多少，必须同时报告保留下来的高质量候选比例。

### E3：相同资源的多 seed、多算子端到端对照（P0）

**目的**：验证 CS 的收益不是单 seed 或单算子偶然现象。

**正式 workload**：

- `matmul128`
- `matmul256`
- `reduction128`
- `conv2d`
- `relu128` 或 `elementwise128`
- `fusion128`

**实验臂**：

```text
B             全部新模块关闭
CS            仅约束感知采样
CS+H          CS 加 trace-position hard filter
CS+Draft      CS 加 Draft，但不加 GA 方向引导
CS+H+Draft    第一、第二点组合
```

G1/G2 暂时单独做消融，不放入第一轮主结论。

**预算**：

- 先做 3 seeds × 16 trials 功能复核；
- 再做 5 seeds × 64 或 128 trials 正式实验；
- 同时执行相同实测预算、相同有效候选 quota 和相同墙钟时间三种口径。

**输出**：

`experiments/results/paper_v1/e3_equal_budget/`

### E4：完整阶段耗时拆分（P0）

**目的**：证明节省时间来自减少无效候选，而不是把成本转移到 CS/Draft/replay。

**必须记录的阶段**：

```text
生成、CS 分析、replay、postproc、lowering、VerifyGPUCode、Draft、
GA、XGBoost 特征提取、XGBoost 推理、Builder、Runner/GPU 测量
```

**输出**：

`experiments/results/paper_v1/e4_stage_timing/`

每个结果同时保存绝对时间、每候选平均时间和占总时间比例。

### E5：CS 扩展可行性实验（P1）

**目的**：从“线程总数约束”扩展到更多可证明约束，但不引入误删。

按以下顺序逐项进行 shadow：

1. 目标显式提供时的 `max_thread_x/y/z` 轴向约束；
2. trace 中显式 `Vectorize` 且 extent、dtype、stride 都静态已知时的 vector bytes 约束；
3. 可从访问区域证明的 shared-memory 下界；
4. 其它资源约束的静态下界。

禁止直接把以下内容升级为硬拒绝：

- 原始 shared allocation 总大小；
- 可能被 lowering 消除的 vthread extent；
- 可能被 padding、predicate 或 scalarization 修正的 vector 宽度；
- 动态 extent、冲突绑定和未知 block scope。

**输出**：

`experiments/results/paper_v1/e5_cs_extension/`

每条新规则需要单独记录 coverage、FP、unknown、节省的 Verify 时间和最终质量变化。

### E6：Draft 与 GA 消融（P1）

**目的**：厘清 Draft、G1、G2 各自的收益，避免组合结果无法归因。

**Draft 参数**：

- archive size：16/32/64/128/256；
- Draft 权重：0.05/0.10/0.20/0.40；
- 保持 XGBoost 只在 archive 末端调用。

**GA 对照**：

- GA off；
- G2 only；
- G1 only；
- G1+G2。

G3/G4 暂不进入主线，除非后续有独立、稳定的 runner 证据。

**输出**：

`experiments/results/paper_v1/e6_draft_ga_ablation/`

### E7：目标参数和算子泛化（P2）

如果暂时没有第二块 GPU，先用不同 target 约束配置验证：

- max threads/block；
- warp size；
- shared-memory 上限；
- vector width 上限。

正式论文最好增加另一种真实 GPU 架构。还要覆盖：

- 动态 shape；
- 不完整 trace；
- 多 block；
- unknown instruction；
- 不支持的 target。

**输出**：

`experiments/results/paper_v1/e7_target_generalization/`

## 5. 统一实验口径

所有正式实验必须满足：

1. 使用同一个已编译 DLL，并记录 DLL SHA256；
2. 每个 arm 使用独立 Python 进程和独立 work directory；
3. `off/shadow/enforce` 使用相同 workload、seed、population、GA 代数和 trial budget；
4. 不使用 `F:/tvm_tmp` 作为正式证据目录，统一使用：

```text
结果：F:/tvm/experiments/results/paper_v1/<experiment>/
临时：F:/tvm/tmp/paper_v1/<experiment>/
```

5. 每个 run 保存：

```text
result.json
stdout.log
stderr.log
database_workload.json
database_tuning_record.json
manifest.json
source_identity_sha256
dll_sha256
```

6. 统计时同时报告：

- mean/std/median；
- 95% bootstrap confidence interval；
- 每个 seed 的配对结果；
- best、top-3、top-5 和均值；
- build/runner timeout，不能把失败候选当作高延迟。

## 6. 执行计划

### Phase 0：冻结环境与修复实验入口

日期：2026-09-22

- 固定 `build-prune` DLL 和 source identity；
- 确认 `experiments/e1_thread_filter.py` 的所有正式运行显式传入 `--work-root` 和 `--temp-dir`；
- 修复并重新验证 ResNet18 端到端统计口径，当前 `B` 结果的算子名解析为 `?`，暂不引用；
- 先跑现有 CS/H 单元测试和 off/shadow 等价性测试。

### Phase 1：CS 安全性闸门

日期：2026-09-23～2026-09-24

- 完成 E1 合法数据库 shadow 审计；
- 完成边界 trace；
- 输出每条约束的 FP/unknown/coverage；
- 任何出现合法候选 FP 的新规则停止进入 enforce。

### Phase 2：候选召回率

日期：2026-09-24～2026-09-26

- 完成 E2 Best-k/Top-k；
- 先在 MatMul128、Conv2D、Reduction128 上运行；
- 再补 Elementwise/Fusion 边界案例；
- 输出 Draft archive size 曲线。

### Phase 3：正式相同资源对照

日期：2026-09-26～2026-10-02

- 先做 3 seed × 16 trial pilot；
- 无环境和账本问题后，做 5 seed × 64/128 trial；
- 执行 B、CS、CS+H、CS+Draft、CS+H+Draft；
- G1/G2 保持独立消融，不混入主结论。

### Phase 4：阶段耗时和模块归因

日期：2026-10-02～2026-10-04

- 完成 E4 的完整阶段时间采集；
- 把每候选平均耗时和总比例画成图；
- 核对 CS 节省的 Verify 时间是否被 Draft/replay/Builder 成本抵消。

### Phase 5：CS 新约束可行性

日期：2026-10-04～2026-10-10

- 按 E5 顺序逐项实现 shadow；
- 每个规则单独编译和测试；
- 只有通过零误删闸门且有实际 coverage 的规则才进入 enforce；
- 共享内存原始声明大小继续保持 observation-only，直到找到 compaction-safe 下界。

### Phase 6：论文汇总

日期：2026-10-10 以后

- 汇总 E1～E7；
- 生成论文主表和消融表；
- 把负结果（G1 不稳定、G3 放弃、shared proxy 不能硬筛、ResNet 首轮统计无效）写入限制和讨论；
- 只对多 seed、多个算子和相同预算下稳定的结果下主结论。

## 7. 论文主结论的准入标准

只有满足以下条件，才能把某个模块写成论文中的有效优化：

### CS 硬筛

- 已知合法候选 FP=0；
- 边界 trace 行为正确；
- 至少两个以上算子有实际 coverage；
- 能节省可测的 Verify/lowering 时间；
- 没有显著降低 Best-k/Top-k 召回率。

### Draft

- archive size 减小时，Best-k 仍保持稳定；
- XGBoost/复杂代价模型调用量明显减少；
- 端到端时间节省大于 Draft 和额外 replay 的开销；
- 多 seed 下最终 top-k 不出现系统性回退。

### GA

- G1/G2 需要单独消融；
- 至少在多数 seed 上改善或不损害最终质量；
- 不能只凭一次 best latency 更低就宣称有效。

## 8. 当前最短执行路径

如果只保留最必要的论文实验，执行顺序是：

```text
E1 CS 零误删审计
  ↓
E2 Best-k/Top-k 召回
  ↓
E3 多 seed、多算子、相同资源端到端
  ↓
E4 阶段耗时拆分
  ↓
E5 CS 新约束 shadow 可行性
  ↓
E6 Draft/GA 消融
```

在 E1～E4 完成之前，不再增加新的 G1/G3/G4 机制，也不把 shared-memory proxy 升级成硬筛。这样可以先把“CS 安全、保留好候选、节省搜索成本”这条主线证据补完整。
## 2026-09-22 实际执行补充

本次按 Phase 0～Phase 2 推进。`build-prune` 已完成真实 C++ 编译和链接，44 个 CS/Verify/Draft 定向测试全部通过。E1 审计扫描 1393 条 tuning record，其中 1177 条有有限 GPU 实测时间，线程 CS 规则误删为 0；shared-memory 原始 proxy 对 1333 条标记超限，但其中 1117 条已实测成功，因此继续保持 observation-only。E2 已用重建 DLL 的 C++ Draft FFI 重算并输出 top-k 召回诊断；历史数据源身份不完全一致，不能作为最终 endpoint 表。

完整过程、命令和原始路径见 `experiments/results/paper_v1/EXECUTION_LOG_20260922.md`。本轮额外启动的 64-trial baseline 因 LocalBuilder 30 秒超时全部 build failure，已停止并保留为诊断，不纳入任何正式平均。
### 2026-09-22～23 实际执行补充

- `popen_pool.py` 对 Windows `psutil.AccessDenied` worker 清理增加容忍，最小 PopenPool shutdown 测试通过，第二个 seed 运行未新增遗留 worker。
- 当前版本第二个 MatMul128 seed（20260923）B/CS+H 8-trial 配对已完成：生成候选 189/131，Verify 失败 118/32，best 46.836/42.029 us；候选减少 30.7%，Verify 失败减少 72.9%。由于预算很小，best latency 仅作为 pilot 观察。

## 9. E8：Pruner 口径的收敛速度实验（2026-09-22 新增）

Pruner 的主收敛实验不是只比较最终 best，而是画“端到端模型推理延迟 - 搜索墙钟时间”曲线：最多 200 轮、每轮实测 10 个 tensor program、共 2000 trials；加速比按“达到 baseline 完整搜索最终性能所需的时间”计算。其单算子实验另使用 800 trials。因此本项目增加两级实验：

1. 可行性 pilot：MatMul128，4 轮 × 8 trials，共 32 trials；逐轮记录 best-so-far、累计 finite trials、从第一次 task pick 开始的墙钟时间、搜索/Verify/Builder/Runner 分解。
2. 正式算子曲线：MatMul、Conv2D、Reduction 至少 3 seeds；预算扩到 100～800 trials，并报告中位数及 bootstrap 95% CI。
3. 正式模型曲线：修复并统一 ResNet18/ResNet50 子图聚合口径后，以模型加权推理延迟为纵轴、搜索墙钟为横轴；不能把单个算子的 kernel latency 冒充模型收敛。

主对照先使用 `B` 与 `CS+H`，再单独加入 `CS+H+Draft` 消融。必须报告：

- 每个测量 batch 的 best-so-far；
- 达到 baseline 最终性能以及其 1%/2%/5% 邻域所需时间；
- 相同 trials 下的最终质量；
- 相同墙钟时间下的最终质量；
- 失败/超时批次，不能用 1e10 哨兵伪装成延迟。

分析入口为 `experiments/analyze_convergence.py`，输出 JSON 与 CSV。第一组 16-trial 连续曲线已验证脚本：seed 20260926 中，Baseline 在 451s/16 trials 达到 40.8845 us，CS+H 在 197s/8 trials 已达到 40.6480 us；该单 seed 结果只作为 E8 pilot，不作跨 seed 结论。

当天结束时按用户要求暂停：两次 32-trial Baseline 尝试均为中途停止的诊断数据，不纳入正式曲线。`B32` 完成 2 个 batch 后外部中断；`B32r` 完成第 1 个 batch（8/8 finite，48.4383 us，201s）并在第 2 个 batch runner 阶段被人工停止。下次必须用新 run-id 从头完成 B 与 CS+H 的 4×8 配对，不能把两个半截运行拼成连续收敛曲线。
