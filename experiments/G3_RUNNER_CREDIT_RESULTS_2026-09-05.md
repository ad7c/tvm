# G3 真实延迟记账实验记录（2026-09-05）

## 1. 本轮要回答的问题

G3 不使用 Draft 分数或 XGBoost 分数给变异器发奖励，而是等待候选真实编译、上卡运行后，
用已实测父子延迟更新滑窗 UCB1 账本。本轮专门检查：

1. lineage 能否从 GA 候选一直传到 Runner 回调；
2. 奖励是否只来自真实 Runner 延迟；
3. Draft 是否仍是 GA 内唯一 fitness，XGBoost 是否只在每批末端精排一次；
4. 冷启动结束后 UCB 选臂是否存在塌缩风险。

这是一轮 MatMul128、单 seed 的机制实验，不是跨算子或统计性能结论。

## 2. 冻结配置与二进制

```text
workload=MatMul128
target=CUDA sm_86
seed=20260905
trials=64, trials_per_iter=8
population_size=128
exact_init_population=true, init_sample_batch_size=64
genetic_iters=4
direction_trials=2, direction_eps=0.10
Draft verify ratio=0.25
G4=off
XGBoost warmup=0
G3 window=50, cold_start_measurements=32, exploration_scale=1.0
```

两次运行的配置除 G3 `shadow/enforce` 外相同，加载的 DLL 也相同：

```text
F:\tvm\build-prune\lib\tvm_compiler.dll
SHA256 6e0156f6b2b3133100970b50c52e7804ed8130a226d1baa49600fbee24a27474
```

实验后已冻结同一二进制副本：

```text
F:\tvm_tmp\g3_enforce_collapse_20260905_r1\
  tvm_compiler_pre_retry_fix_6e0156f6.dll
```

本轮运行使用真实 C++ DLL 和真实 CUDA build/run；启动本轮实验时没有现场重新编译。
shadow/enforce 的 `source_identity_sha256` 分别为 `a19d029e...8d` 和 `0e1b820b...2a`，
并不相同。因此以下性能数字只是两个独立运行的手工比较，不能写成严格同源的性能配对，
也不是分析器在同一目录产生的 paired comparison。

## 3. 数据完整性和架构契约

两臂均满足：

```text
sent_to_builder=64
sent_to_runner=64
database_records=64
finite_records=64
build_errors=0
runner_errors=0
search_batches=8
architecture_validation.passed=true
```

数据库逐行复核也得到各 64 条正有限延迟，独立最小值与 `result.json` 中的 best 一致。

Draft/XGBoost 调用契约：

```text
Draft generations=40（8 批 × 每批 iter 0..4）
Draft known/total=5120/5120
GA 内 XGBoost calls/input=0/0
GA 后 XGBoost calls/input=8/256
```

即每批先由 Draft 独立完成 GA，再只对跨代 Draft top-32 调一次 XGBoost，最终返回 8 条
候选去 build/run。G3 奖励源记录为 `source=runner`，未使用 Draft 或 XGBoost 分数代替实测。

## 4. Enforce 的逐批反馈时序

| 批次 | GA 开始前累计 credit | 本批 credit | improved | feedback 后累计 | feedback 后 cold |
|---:|---:|---:|---:|---:|---|
| 1 | 0 | 0 | 0 | 0 | 是 |
| 2 | 0 | 0 | 0 | 0 | 是 |
| 3 | 0 | 5 | 2 | 5 | 是 |
| 4 | 5 | 8 | 3 | 13 | 是 |
| 5 | 13 | 3 | 1 | 16 | 是 |
| 6 | 16 | 7 | 3 | 23 | 是 |
| 7 | 23 | 3 | 3 | 26 | 是 |
| 8 | 26 | 8 | 2 | 34 | 否 |

关键时序是：32-credit 阈值在第 8 批 Runner feedback 写入后才越过。第 8 批的四个 GA
generation 在原始日志中仍全部记录 `cold_start=1`，而且没有第 9 批。因此本轮虽然开关是
`enforce`，却**没有任何 GA generation 真正执行冷启动后的 UCB 强制选臂**。
`feedback_events[7].cold_start="0"` 只能解释为反馈后的状态，不能解释为第 8 批已经应用
UCB。

反馈漏斗合计：

```text
sent=64
child_measured=64
with_lineage=60
with_measured_parent=34
credited=34
improved=14
skipped_no_lineage=4
skipped_parent_unmeasured=26
skipped_child_error=0
```

前两批没有已实测父代，所以没有 credit；这也说明 64 trials 在当前每批 8 条的设置下，
刚好不足以观察 enforce 生效后的下一代。

## 5. 选臂风险

Enforce 最终账本：

| arm | selected | accepted | credited | improved | mean reward | 最终 UCB |
|---|---:|---:|---:|---:|---:|---:|
| TileSize | 3279 | 717 | 29 | 11 | 0.3793 | 0.8725 |
| Unroll | 289 | 289 | 5 | 3 | 0.6000 | 1.7877 |
| ThreadBinding | 76 | 0 | 0 | 0 | 0.0000 | inf |

`ThreadBinding` 76 次尝试全部没有产生 accepted 子代，因此没有真实父子样本，并一直保留
`UCB=inf`。shadow 运行在第 7 批反馈后已经累计 33 个 credit；其第 8 批实际基线选择仍为
`404/43/3`，但 UCB 的反事实首选新增 `450` 次全部落到 ThreadBinding。这个结果明确证明
“未见但不可用的 arm 永久为 inf”会使 UCB 首选塌缩。它不能证明 enforce 的最终候选一定
塌缩，因为本轮 enforce 没有进入该阶段，而且失败臂的本地重试/排除行为还需单独验证；
但它已经构成必须修正后才能扩大预算的机制风险。

建议的修正门槛是：失败 arm 只在当前 child 的重试中临时屏蔽，不写入虚假 Runner 奖励；
同时记录首选失败数、重试选臂数和最终成功 arm。修正后必须重新编译、先跑对应单测，再用
足以产生至少一个冷启动后批次的预算验证；不能沿用本次 DLL 的结果替代验证。

## 6. Shadow 与 Enforce 的观测结果

| 指标 | Shadow | Enforce | Enforce 相对 Shadow |
|---|---:|---:|---:|
| finite | 64 | 64 | 相同 |
| best / μs | 32.3368 | 34.2492 | 慢 5.91% |
| top-3 mean / μs | 33.5131 | 42.0751 | 慢 25.55% |
| top-5 mean / μs | 34.3818 | 49.8195 | 慢 44.90% |
| search / s | 594 | 601 | +1.18% |
| builder / s | 492 | 504 | +2.44% |
| runner/update / s | 182 | 513 | +181.87% |
| tune / s | 1291.510 | 1641.919 | +27.13% |

这些差异**不能归因于 G3 enforce**：enforce 臂从未在 GA generation 中退出冷启动；两次
运行的 source identity 也不同。特别是 runner/update 的大幅波动更应视为独立运行中的
系统/轨迹噪声线索，而不是算法成本。当前可靠结论只有反馈闭环、调用契约和塌缩风险。

## 7. 代码和证据位置

G3 代码路径：

- `src/s_tir/meta_schedule/draft_mutator_credit.{h,cc}`：滑窗、冷启动、UCB1、实测 outcome；
- `src/s_tir/meta_schedule/search_strategy/evolutionary_search.cc`：mutator arm、lineage、
  每批 pending 候选、Runner feedback、shadow/enforce 日志；
- `experiments/e1_thread_filter.py`：G3 开关、日志解析、feedback funnel 和架构断言；
- `experiments/analyze_second_point_v1_0.py`：跨结果汇总和 G3 完整性检查；
- `tests/python/s_tir/meta_schedule/test_meta_schedule_draft_mutator_credit.py`：G3 账本、
  shadow/off、lineage 和反馈相关测试源码。测试源码存在，但本轮记录不把“未持久化的最新
  测试执行”写成已通过。

原始目录：

```text
F:\tvm_tmp\g3_shadow_density_20260905_r1\
F:\tvm_tmp\g3_enforce_collapse_20260905_r1\
```

主要证据为各目录的 `command_plan.json`、`paired_summary.json/csv`、逐臂 `result.json`、
`database_tuning_record.json` 和 `logs/tvm.s_tir.meta_schedule.logging.task_0_main.log`。
Enforce 目录还保存本次 DLL 冻结副本。本文件的实验目录副本及 DLL 由额外 artifact manifest
记录；原有 `raw_manifest.sha256.json` 保留为运行结束时的原始结果 manifest。

## 8. 预修正版结论

- 已验证：真实父子延迟能够跨批反馈；34 个 outcome 被记账，其中 14 个为改善；奖励源只
  来自 Runner；64/64 候选真实 build/run 成功；Draft/XGBoost 分工符合设计。
- 尚未验证：冷启动后 enforce 的真实选择、候选质量或墙钟收益；当前性能差异不可归因。
- 已发现：没有成功样本的 ThreadBinding 保持无限 UCB，shadow 已观察到下一批反事实首选
  全部压向该 arm，扩大实验前必须先验证失败臂重试/排除修正。

## 9. 修正版失败臂重试与 96-trial 严格配对

### 9.1 修正内容

针对预修正版中 `ThreadBinding accepted=0` 且永久保持无限 UCB 的问题，在 G3 enforce 的
每个子代生成过程中增加失败 arm 掩码：当前 arm 的 `Apply`、trace replay/postproc 或 G4
接受阶段未产出可用子代时，本轮 retry 临时排除它并改试其他 arm；所有 arm 都尝试过后会
清空掩码重新开始。该失败不写成 0 奖励，G3 账本仍然只接收跨批、真实 Runner 父子延迟。
G3 off/shadow 不启用这个掩码，也不为它额外消耗随机数。

这是一项局部止血修正，不是完整 pending-feedback 算法：它只能处理本地无法生成子代的
arm，不能处理“能生成、但长期进不了末端 top-K、因而没有 Runner 反馈”的 arm；retry
过程中也会重新采样父代，所以不能表述为“固定同一父代换 arm”。

对应实现：

- `draft_mutator_credit.{h,cc}`：支持在临时不可用掩码内选择；
- `evolutionary_search.cc`：每个 child 的失败 arm 掩码、无放回尝试及逐代日志；
- `e1_thread_filter.py`：解析 `generation_events`，记录 batch/iter/cold/selected/accepted。

### 9.2 构建、测试与实验身份

修正版使用真实 MSVC/Ninja 构建：

```text
CCACHE_DISABLE=1
call E:\vs\VC\Auxiliary\Build\vcvars64.bat
cmake --build F:\tvm\build-prune --parallel 4
```

本轮冻结二进制：

```text
F:\tvm_tmp\g3_retry_fix_pair96_20260905_r1\tvm_compiler_pair96_ddc796a9.dll
SHA256 ddc796a942b2e3488b7dacf98076626c82980e56112dbd3f5cbd9282d75165bb
```

运行前已完成 G3 专项 `10/10 passed`，以及 CS/H、G1/G2、Draft、G3、EvolutionarySearch、
GPU verifier 的扩大回归 `74/74 passed`。另用同一 DLL 跑过 G3-off 8-trial CUDA smoke：
`8/8` finite、0 build/runner error、GA 内 XGBoost 为 0、末端一次输入 32。上述测试执行最初
只保存在终端记录中；本节如实记录结果，不伪造当时不存在的测试日志文件。

严格配对配置：MatMul128、CUDA sm_86、seed 20260905、96 trials（12 批，每批 8）、
population 128、GA 4 轮、direction trials 2、G4 off、CS/H off。两个臂顺序固定为：

```text
D_G1G2       # Draft-GA + G1/G2，G3 off
D_G1G2_G3    # 同配置，G3 enforce
```

两臂的 `git_head`、dirty-source identity、target、seed 和 DLL 全部一致：
`source_identity_sha256=7ec83fd9...65c2a`，加载 DLL 的 SHA256 均为 `ddc796a...65bb`。
完成时再次核验 live/frozen DLL，哈希未变化。因此这是严格同源、同预算配对；但运行顺序
没有交换，且只有单 workload、单 seed。

### 9.3 数据完整性与 Draft/XGBoost 架构契约

| 指标 | G3-off | G3-enforce |
|---|---:|---:|
| database / finite | 96 / 96 | 96 / 96 |
| build / runner error | 0 / 0 | 0 / 0 |
| search batches | 12 | 12 |
| Draft generation | 60 | 60 |
| Draft known / total | 7680 / 7680 | 7680 / 7680 |
| GA 内 XGBoost calls / input | 0 / 0 | 0 / 0 |
| 末端 XGBoost calls / input | 12 / 384 | 12 / 384 |
| G3 feedback / generation event | 0 / 0 | 12 / 48 |
| architecture validation | passed | passed |

G3 的 48 个逐代事件完整覆盖 `12 batch × 4 mutation iter`；12 次反馈全部标记
`source=runner`。反馈漏斗合计为：

```text
sent=96, child_measured=96, with_lineage=92
with_measured_parent=46, credited=46, improved=16
```

### 9.4 冷启动和真正的 UCB 阶段

| batch | GA 是否 cold | 本批 credited / improved | 反馈后累计 credit | 反馈后 cold |
|---:|---|---:|---:|---|
| 1 | 是 | 0 / 0 | 0 | 是 |
| 2 | 是 | 0 / 0 | 0 | 是 |
| 3 | 是 | 0 / 0 | 0 | 是 |
| 4 | 是 | 6 / 0 | 6 | 是 |
| 5 | 是 | 5 / 2 | 11 | 是 |
| 6 | 是 | 8 / 2 | 19 | 是 |
| 7 | 是 | 3 / 1 | 22 | 是 |
| 8 | 是 | 3 / 0 | 25 | 是 |
| 9 | 是 | 7 / 5 | 32 | 否 |
| 10 | **否，首个 UCB 批次** | 5 / 4 | 37 | 否 |
| 11 | 否 | 4 / 0 | 41 | 否 |
| 12 | 否 | 5 / 2 | 46 | 否 |

第 9 批的 GA 仍在 cold start；只有该批 Runner 反馈将累计 credit 推到 32 以后，第 10 批
才首次真正执行 UCB enforce。不能把 feedback 的 `cold_start=0` 倒推到同一批 GA。

冷启动前后逐臂聚合：

| 阶段 | arm | selected | accepted | selected-accepted |
|---|---|---:|---:|---:|
| batch 1–9 | TileSize | 3662 | 782 | 2880 |
| batch 1–9 | Unroll | 364 | 364 | 0 |
| batch 1–9 | ThreadBinding | 83 | 0 | 83 |
| batch 10–12 | TileSize | 0 | 0 | 0 |
| batch 10–12 | Unroll | 1184 | 1184 | 0 |
| batch 10–12 | ThreadBinding | 1344 | 0 | 1344 |

这里的 `selected-accepted` 只能称为“未接受尝试”，不能全部等同于 mutator `Apply` 失败；
例如 TileSize 还可能因 Draft 未选中 proposal 而复用父代。实验明确观察到：ThreadBinding
仍因没有实测样本而保持 `UCB=inf`，每个新 child 先尝试它并失败，局部掩码随后让 Unroll
成功，于是 adaptive 三批完全没有再选择 TileSize。修正确实避免了“整个子代生成失败”，
但选臂从“卡死 ThreadBinding”变成了“ThreadBinding 固定空耗一次、随后 Unroll 垄断”。

最终账本：TileSize `22/7`（mean 0.3182，UCB 0.9081），Unroll `24/9`
（mean 0.3750，UCB 0.9398），ThreadBinding `0/0`（UCB inf）。

### 9.5 等预算结果与因果边界

| 指标 | G3-off | G3-enforce | enforce 相对 off |
|---|---:|---:|---:|
| search / s | 936 | 865 | 快 7.59% |
| builder / s | 781 | 782 | 慢 0.13% |
| runner/update / s | 821 | 827 | 慢 0.73% |
| tune / s | 2563.891 | 2499.383 | 快 2.52% |
| best / μs | 38.0030 | 35.7334 | 低 5.97% |
| top-3 mean / μs | 47.9236 | 44.2242 | 低 7.72% |
| top-5 mean / μs | 53.2265 | 52.9278 | 低 0.56% |
| median / μs | 71.6838 | 76.1915 | **高 6.29%** |
| mean / μs | 102.1823 | 84.7071 | 低 17.10% |
| unique trace | 96 | 96 | 相同数量 |

两臂 trace 只重合 `10/96`。更重要的是，两臂全局 best 都出现在 batch 1；此时累计 credit
为 0、UCB 尚未生效。G3 enforce 在冷启动阶段也会对本地失败 arm 执行新 retry，因此
35.7334 μs 只能归入“冷启动 retry、搜索轨迹差异或测量噪声”的混合信号，**不能归因于
Runner-credit/UCB**。

仅看真正 adaptive 的 batch 10–12（末 24 条）：

| 指标 | G3-off | G3-enforce | enforce 相对 off |
|---|---:|---:|---:|
| best / μs | 62.1729 | 60.8776 | 低 2.08% |
| top-3 mean / μs | 63.2194 | 64.9836 | 高 2.79% |
| top-5 mean / μs | 64.8046 | 66.9628 | 高 3.33% |
| median / μs | 70.5416 | 73.4693 | 高 4.15% |
| mean / μs | 70.0839 | 73.1264 | 高 4.34% |

这三批的 search 缩短主要伴随 TileSize 方向提案减少：全程 direction proposals 从 9800
降到 7324，VerifyGPUCode 时间从 809.679 s 降到 745.290 s；但候选质量指标不一致。
因此当前验收结论是：**Runner-only 反馈和失败换臂机制跑通；当前 UCB 算法有效性未通过，
并暴露反馈饥饿/Unroll 垄断。**

### 9.6 原始证据和归档

运行现场保留于：

```text
F:\tvm_tmp\g3_retry_fix_pair96_20260905_r1\
```

项目内正式归档：

```text
F:\tvm\experiments\data\second_point_g3_pair96_20260905_r1\
```

归档包含两个 `result.json`、数据库、原始 task/scheduler 日志、driver 日志、命令计划、
环境快照、DLL 哈希监控、paired summary 以及 manifest；不复制 220 MB DLL，外部冻结路径和
哈希写在 `ARTIFACT_NOTE.md`。原始 raw manifest 为 `10/10`，现场 artifact manifest 为
`20/20`，复制后 archive manifest 也已逐文件核验。实验结束时没有采集
`environment_after.json`，因此不事后伪造；完成时 DLL 哈希记录仍足以证明二进制身份。

### 9.7 下一步

1. 增加 retry-only 对照臂，把冷启动失败换臂与 UCB 效果拆开；
2. 将 Runner reward 与本地可用性/pending pull 分账，未测 arm 不应永久保持无限优先级；
3. 每批为来自已实测父代的一跳 mutator probe 保留少量 Runner 名额，减轻 top-K 反馈饥饿；
4. 在算法修正后交换运行顺序并做多 seed；再扩展 Conv2D、Reduction 等 workload；
5. G1 当前仅是线程余量方向，G2 当前仅投影能证明的线程越界；shared-memory lowering 前
   proxy 误报严重，不能直接拿来做 shared 硬投影。
