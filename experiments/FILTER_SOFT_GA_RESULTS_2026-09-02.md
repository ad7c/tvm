# 硬过滤、软过滤与 GA 可行性保持实验记录（2026-09-02）

## 1. 本轮范围

本轮实现并验证三个互相独立的开关：

- 硬过滤：只拒绝可静态证明 `threads_per_block` 超过 Target 上限的候选。
- 软过滤：不删除候选，只在代价模型预测后用 thread proxy 降权重排。
- GA T2：在 `MutateTileSize` 产生 thread-overflow proposal 时，沿原 divisor 格点投影到最近的非平凡合法变异；无解则让 EvolutionarySearch 重试。

没有实现精确 shared-memory 预判，也没有改 XGBoost 的训练特征/标签。原因是现有实验已经确认真实 shared-memory 用量是 lowering 结果；在 lowering 前从 schedule IR 直接读取并不成立。

## 2. 环境与统一口径

- 分支：`pruning-experiment`
- 基础提交：`4a1297ad59c9ff80de535a8ba898bf6d9012e5c7`
- TVM：`0.26.dev90`
- GPU：NVIDIA GeForce RTX 3050 Ti Laptop GPU，driver 591.44，target `sm_86`
- Python：`D:\conda_envs\tvm-build-venv\python.exe`（CPython 3.11）
- 最终动态库：`F:\tvm\build-prune\lib\tvm_compiler.dll`，SHA256 `47083fe920dbe94560382c4ba9db343be4769f59ab11d050beca7c9b4595e195`
- workload：MatMul 128³，`max_trials=8`，单批，population 512，GA 4 代，seed `20260902`
- Target 上限：threads/block 1024、shared memory/block 49152 bytes、warp size 32

所有性能数据都是单 seed、8 trials 的功能烟测，不能当论文性能结论。搜索时间来自秒级日志时间戳；最佳延迟只用于确认生成候选可正常编译和上卡。

### 2.1 证据口径与结论边界

- 本报告中的 TP/FP/FN/TN 以整个 `VerifyGPUCode` 的结果作为对照。该检查还会因 shared memory、lowering、vector 等非线程约束失败，因此它是混合 oracle，不是纯线程合法性真值。`FP=0` 只能表示当前样本中没有观察到“线程规则预测拒绝、但 VerifyGPUCode 通过”的反例；TP、FN 以及由此得到的 precision/recall 不能直接解释成线程过滤器的分类精度或召回率。
- 所有端到端结果均为单 seed、8 trials、单批次的功能 smoke。搜索时间使用秒级日志时间戳，最佳延迟来自很小的候选集合；这些数值只能证明流程能够运行并产生可测候选，不能支持加速、搜索质量提升或统计显著性的性能结论。
- 当前硬过滤仅检查可静态证明的 `threads_per_block` 上限，不包含 shared memory、寄存器或其它硬件约束。软过滤是在代价模型输出之外执行 `raw_score * penalty^weight` 的乘法重排，没有修改模型训练特征、标签或模型结构。GA T2 仅处理 trace 中能够确定为单一物理线程轴的 `MutateTileSize` proposal；多轴、动态或不支持的 provenance 作为 Unknown 放行。
- `pruner_prefilter.py` 只是面向当前 MatMul-128/SSSRRSRS trace 的早期启发式观察脚本。它读取已通过 VerifyGPUCode 的记录，但不执行 lowering，也不读取真实 CompactBufferAllocation，因而不能作为“公式精确”“通用零误杀”或可直接启用硬过滤的证据。

## 3. 实现

### 3.1 硬过滤

环境变量：

```text
TVM_PRUNE_MODE=0  off
TVM_PRUNE_MODE=1  shadow，只预测并与 VerifyGPUCode 对照
TVM_PRUNE_MODE=2  enforce，静态确定超限时提前拒绝
```

分析器按 PrimFunc、按活跃 AST 路径计算 `threadIdx.x * y * z`；相同物理轴不重复相乘。动态 extent、冲突的重复轴和不支持结构都返回 Unknown 并放行。支持 `For(kThreadBinding)`、`AttrStmt(thread_extent)`，以及 `SBlock(warp_execution)` 隐含的 `threadIdx.x=thread_warp_size`。warp size 无效或与显式 x 冲突时保守返回 Unknown。检查点位于 `VerifyGPUCode` 之前，因此能够省掉 VerifyGPUCode 及后续 postproc，但不能省掉它之前已经运行的 postproc。

### 3.2 软过滤

环境变量：

```text
TVM_SOFT_FILTER_MODE=0  off
TVM_SOFT_FILTER_MODE=1  shadow，计算 hypothetical score 但返回 raw score
TVM_SOFT_FILTER_MODE=2  rerank，返回调整后的 score
TVM_SOFT_FILTER_WEIGHT=0.2
```

公式：

```text
adjusted_score = raw_score * penalty^weight
```

`penalty` 是仅在 sm_86 启用的 thread-only proxy：lane utilization、warp-scheduler utilization、thread occupancy proxy 三项的几何均值。Unknown 和非 sm_86 target 严格取 1.0。它是 Pruner-inspired proxy，不是 Pruner LSE/SA 的复现；没有使用共享内存、寄存器、transaction bytes 或访存强度。候选 vector 长度始终不变。

### 3.3 GA T2

环境变量：

```text
TVM_GA_T2_MODE=0  off，保持原版 RNG 和 decision
TVM_GA_T2_MODE=1  shadow，只统计 proposal
TVM_GA_T2_MODE=2  project，投影或返回 nullopt 触发原有重试
```

trace 推导链是 `SamplePerfectTile -> Split -> Fuse -> Bind(threadIdx.*)`。当前只在 trace 中恰有一个可解物理轴时判定；零轴或多轴因 flat trace 无法证明共同嵌套而返回 Unknown。Unknown 原样放行。投影只沿本次“从 x 除、向 y 乘”的同一 divisor 方向搜索更小 divisor，保持 tile 总乘积和 `max_innermost_factor` 约束。该实现只承诺处理可静态推导的 thread overflow，不处理 shared-memory 等其它非法原因。

## 4. 构建与测试

增量构建命令：

```powershell
cmd.exe /d /s /c 'call "E:\vs\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && set CCACHE_DISABLE=1 && cmake --build build-prune --parallel'
```

结果：最终一次重编译 67/67 步骤成功，包含 `constraints.cc`、`mutate_tile_size.cc`、`evolutionary_search.cc`、`replay_trace.cc`，并链接 `tvm_compiler.dll`。只有既有第三方头文件在 CP936 下的 C4819 警告。

测试结果：

| 测试 | 结果 |
|---|---:|
| thread analyzer | 13/13 PASS |
| soft-filter FFI | 4/4 PASS |
| hard mode 0/1/2 接入烟测 | 3/3 PASS |
| GA `MutateTileSize`（既有 + 投影/失败/Unknown/多轴回退） | 6/6 PASS |

新增 hard 用例覆盖 `warp_execution × threadIdx.y` 的 1056 超限、1024 边界、显式 x 冲突、无效 warp size 和离开 SBlock 后的作用域恢复。新增 soft 用例用非单位 raw score 锁定 `raw_score * penalty^weight`，防止实现漏乘原始代价模型分数。

实验脚本结果格式升级到 schema v2：落盘 predicted/enforced、hard-analysis/VerifyGPUCode 时间、DLL SHA256、tracked diff/status 哈希、untracked source 哈希和组合 source identity；三臂比较会拒绝身份不一致的结果。最终三臂实验时的 source identity 为 `64a4fbfed5226e8e83351762be82c61b2f5bebdb1688204d9bd35cdeb443cc56`。

硬 mode 接入的精确结果：

| mode | checked / predicted / enforced | TP/FP/FN/TN | VerifyGPUCode fail |
|---|---|---|---:|
| off | 0 / 0 / 0 | 0/0/0/0 | 1 |
| shadow | 1 / 1 / 0 | 1/0/0/0 | 1 |
| enforce | 1 / 1 / 1 | 0/0/0/0 | 0 |

## 5. 硬过滤三臂实验

最终 DLL 数据目录：`experiments/data/e3_final/hard_final_v2_seed20260902_*`

| arm | search (s) | Verify fail | 最佳延迟 (us) | thread 预测/拦截 |
|---|---:|---:|---:|---:|
| off | 73 | 1184 | 36.73 | 0 / 0 |
| shadow | 71 | 1184 | 51.79 | 7 / 0（TP=7，FP=0） |
| enforce | 72 | 1177 | 49.15 | 7 / 7 |

shadow 检查了 3047 个候选：TP/FP/FN/TN = 7/0/1177/1863，precision=1.0，混合-oracle recall=0.59%。off 与 shadow 的 8 条实测 trace 多重集相同，Verify 失败数相同，源码身份与 DLL SHA256 相同，计数恒等式成立，因此 enforce 闸门通过。shadow 的 hard analysis 累计 14.344 ms，VerifyGPUCode 累计 54.534 s；enforce 记录 predicted/enforced=7/7。

注意：这里用整个 `VerifyGPUCode` 作为 oracle，FN 会混入 shared memory、lowering 等非线程失败；只有 FP=0 能直接支持“未观测到误杀”。静态判据本身的安全性来自只在确定 `threads > target limit` 时拒绝。

结论：正确性成立，但覆盖很窄；enforce 只少跑了 7 次 Verify，单次搜索 73→72 s，秒级日志下没有可归因的加速证据。

## 6. 软过滤与 GA 分臂实验

数据目录：`experiments/data/e2_soft_ga/`

| arm | hard / soft / GA | search (s) | tune (s) | Verify fail | 最佳延迟 (us) |
|---|---|---:|---:|---:|---:|
| baseline | off / off / off | 71 | 225.8 | 1184 | 48.74 |
| soft shadow | off / shadow / off | 71 | 223.4 | 1184 | 48.98 |
| soft rerank | off / rerank / off | 88 | 237.9 | 1210 | 34.44 |
| GA shadow | off / off / shadow | 72 | 224.4 | 1184 | 46.04 |
| GA project | off / off / project | 77 | 230.7 | 1178 | 37.07 |
| combined | enforce / rerank / project | 88 | 235.0 | 1181 | 34.45 |

### 6.1 软过滤统计

- shadow 与 baseline 的 8 条 measured trace 多重集相同；database 写入顺序有一处不同，属于异步 runner 顺序，不是候选变化。
- shadow 共处理 2145 个候选，2145→2145，known=2145、unknown=0，累计分析 22.735 ms。
- hypothetical top-16 合计 overlap 42/80 = 52.5%，说明 proxy 有明显重排能力。
- rerank 同样保持 2145→2145，累计分析 23.442 ms；top-16 overlap 45/80 = 56.25%。
- rerank 单次最佳延迟更好，但 Verify 失败增加 26、搜索多 17 s。额外 17 s 不是 proxy 计算成本，而是重排后搜索谱系变化带来的 retry/postproc 差异。

结论：软层“只重排不删候选”和低计算开销已经验证；weight=0.2 是否有净收益尚未验证。当前数据提示它可能提高最终候选质量，也可能把搜索带到更难通过 postproc 的区域。

### 6.2 GA 统计

GA shadow：

```text
checked=1153, unknown=0, initial_reject=6
```

它与 baseline 的 measured trace 多重集完全相同，Verify fail 同为 1184，证明 shadow 不改变 RNG/返回 trace。

GA project：

```text
checked=1152, unknown=0, initial_reject=13
projected=6, projection_failed=7, divisor_steps=9
```

project 会改变后续搜索轨迹，因此 initial_reject 不能与 shadow 的 6 直接逐项配对。该组成功投影 6 次，其余 7 次无非平凡合法点并触发重试；Verify fail 从 1184 降到 1178，数值上少 6，但不能解释成 6 次投影逐事件导致 6 次 Verify 减少。搜索时间为 71→77 s。

结论：T2 已经能在真实 sm_86 搜索中发现并处理 thread-overflow 变异，功能目标成立；当前触发频率低，且投影/重试没有形成单次搜索加速。

### 6.3 联合组

联合组统计：

```text
hard: checked=3027, predicted/enforced=1
soft: 2145->2145, time=22.981 ms, top16 overlap=46/80
GA: checked=1103, initial_reject=2, projected=1, projection_failed=1
```

三者可同时工作。联合最佳延迟与 soft-rerank 基本相同（34.45 vs 34.44 us），搜索时间也基本相同（88 s），本次仍由软重排主导轨迹。由于缺少 `enforce + rerank + GA-off/shadow` 配对臂，联合组中的 Verify 变化不能拆分归因给 hard 或 GA。

### 6.4 最终 DLL 联合复跑

最终 DLL 用 schema v2 入口又跑了一次 `hard-enforce + soft-rerank + GA-project`，数据在 `experiments/data/e3_final/combined_final_v2_seed20260902_enforce_soft-rerank_ga-project/`：

- 8/8 候选可编译并上卡，search 88 s，VerifyGPUCode fail 1181，最佳延迟 37.10 us。
- hard checked=3027，predicted/enforced=1/1，analysis 14.611 ms。
- soft 共 5 次调用，候选总数严格保持 2145→2145，累计 24.782 ms，top-16 overlap 46/80。
- GA checked=1103，initial_reject=2，projected=1，projection_failed=1，divisor_steps=1。

这次复跑证明最终二进制中的三条路径能共同工作，不提供各组件的独立性能归因，也不改变“单 seed、8 trials 只是 smoke”的结论。

## 7. 失败和环境排查记录

1. 专用 CPython 3.11 环境没有 pytest，`import tvm.testing` 会失败；新测试提供不依赖 pytest 的 standalone 入口。
2. 实验父进程最初只修改 `sys.path`，LocalBuilder 子进程找不到 TVM；修复为同时设置 `PYTHONPATH=F:\tvm\python`。
3. 第一次成功调优后，结果记录使用旧字段 `tvm.base._LIB` 崩溃；当前分支应读取 `_LOADED_LIBS["tvm_compiler"]`。
4. 第一次统一构建的 PowerShell→cmd 引号多转义一层，未启动编译；改回已验证的单引号包裹命令后成功。
5. GA 测试最初从 seed 0 枚举。TVM 的乘法 LCG 在 0 上是固定点，`uniform_int_distribution` 可能无限等待；测试改为从 seed 1 开始。
6. 个别调优进程退出时打印 `psutil.AccessDenied`，发生在清理已经结束/不可终止的 worker；主进程 exit 0、8 条记录与 result.json 均已写入。
7. `src/s_tir/meta_schedule/utils.h` 在索引中是 CRLF，其余改动文件是 LF；最终将工作树版本统一为 LF，避免新增行被 `git diff --check` 识别为尾随 CR。

## 8. 下一步建议

1. 先把正式统计预算提高到至少 5 seeds × 64/128 trials，主指标用“达到目标延迟所需实测次数”，不要用 8-trial 单次最佳值。
2. 软过滤做 `weight={0.05, 0.1, 0.2}` 消融，同时记录 Verify fail、搜索 retry 和最终质量；0.2 当前可能偏强。
3. 将 soft proxy 明确保持为性能信号，绝不升级为硬拒绝规则。
4. T2 下一步先补 flat trace 的作用域推导以安全覆盖多物理轴，再分析 7 次 `projection_failed` 的 divisor 结构；暂不先做 G3/G4。
5. 硬过滤的下一增量应寻找安全的 shared-memory 下界或更早的 thread trace 预检；当前 post-early-postproc thread 过滤召回仅 0.59%。
6. XGBoost 特征/相对标签属于独立 T3，需在更大样本 database 上做，不能把本轮 soft rerank 直接写成“代价模型已修改”。

## 9. 复现实验命令

统一入口：`experiments/e1_thread_filter.py`。下面每个 arm 都应在独立 Python 进程运行：

```powershell
# baseline
D:\conda_envs\tvm-build-venv\python.exe experiments\e1_thread_filter.py run --mode off --soft-mode off --ga-t2-mode off --seed 20260902 --run-id final_base0 --work-root F:\tvm\experiments\data\e2_soft_ga --temp-dir F:\tvm\tmp\e2

# soft shadow / rerank
D:\conda_envs\tvm-build-venv\python.exe experiments\e1_thread_filter.py run --mode off --soft-mode shadow --ga-t2-mode off --seed 20260902 --run-id soft_shadow0 --work-root F:\tvm\experiments\data\e2_soft_ga --temp-dir F:\tvm\tmp\e2
D:\conda_envs\tvm-build-venv\python.exe experiments\e1_thread_filter.py run --mode off --soft-mode rerank --soft-weight 0.2 --ga-t2-mode off --seed 20260902 --run-id soft_rerank0 --work-root F:\tvm\experiments\data\e2_soft_ga --temp-dir F:\tvm\tmp\e2

# GA shadow / project
D:\conda_envs\tvm-build-venv\python.exe experiments\e1_thread_filter.py run --mode off --soft-mode off --ga-t2-mode shadow --seed 20260902 --run-id ga_shadow0 --work-root F:\tvm\experiments\data\e2_soft_ga --temp-dir F:\tvm\tmp\e2
D:\conda_envs\tvm-build-venv\python.exe experiments\e1_thread_filter.py run --mode off --soft-mode off --ga-t2-mode project --seed 20260902 --run-id ga_project0 --work-root F:\tvm\experiments\data\e2_soft_ga --temp-dir F:\tvm\tmp\e2

# combined
D:\conda_envs\tvm-build-venv\python.exe experiments\e1_thread_filter.py run --mode enforce --soft-mode rerank --soft-weight 0.2 --ga-t2-mode project --seed 20260902 --run-id combined0 --work-root F:\tvm\experiments\data\e2_soft_ga --temp-dir F:\tvm\tmp\e2
```
