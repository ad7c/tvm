<!--
Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements. See the NOTICE file
distributed with this work for additional information
regarding copyright ownership. The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License. You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied. See the License for the
specific language governing permissions and limitations
under the License.
-->

# 单算子实验重启恢复（2026-10-11）

13:06 检查用户重启后的状态。旧驱动已经停止，没有仍在运行的调优 / nvcc 进程。

## 旧尝试与失败证据

- 旧目录：`experiments/results/paper_v1/operator800_20261011`。原始日志、数据库和协议保持原样。
- S01 冒烟正常结束，4/4 候选有效。
- 正式 M128G1B01 在 12:58:20 的第 8 批 XGBoost 训练中报 `bad_malloc: Failed to allocate 1105616 bytes`，进程退出码 1。
- 7 批反馈完整，共 70 个候选；模型累计训练样本依次 10、20、30、40、50、60、70。
- 数据库已有 80 条测量记录，80/80 有正确性通过记录；第 8 批模型更新失败，所以只有 7 批完整回调。
- 此次没有完整的模型 / 搜索随机状态检查点，不能把中断前后的片段拼成一个 800 次正式结果，也不能把 80 条旧记录作为新组的热启动数据。
- 12 个正式配置尚无完整结果。

## 训练内存重放

入口：[replay_xgb_memory.py](replay_xgb_memory.py)。使用同 DLL、同 workload/target 和既有数据库反馈，在独立进程中构建一个 XGBModel，按每批 10 条持续积累训练数据。这是训练内存诊断，不是正式搜索，不重新进行硬件测量。

输出：`tmp/operator800_xgb_replay_20261011/model/result.json`；监控记录：`tmp/operator800_xgb_replay_20261011/guard/`。

- 80 条训练重放全部完成，进程正常退出（0）。
- 首次导入并训练后，私有内存为 339,714,048 bytes（约 324 MiB），之后各批保持相同。
- 80 条训练特征合计 209,920 bytes（约 205 KiB），共 320 行。
- 此重放没有复现单独训练的内存不足，完整搜索 / 编译 / 测量进程的增长和全系统内存压力仍需由新尝试的监控定位。
- 不把重放的拟合指标当成正式搜索质量结果；没有因此改变 XGBoost 参数或减少历史样本。

## 新尝试 r02

13:16 已启动受监控的串行实验流程，初始 guard PID=3324。目录：`experiments/results/paper_v1/operator800_20261011_r02`。源码、DLL、B/C/D 开关、种群、GA、预算和在线训练设置沿用原配置，并新建冻结协议；所有实验重新从空历史开始。

正式设置仍为四算子 × B/C/D × 一个 seed（20261011），每组最多 800 次提交、每批 10 次；种群 32、GA 1、Draft 入围比例 0.25、warmup=0、adaptive_training=False。每组一个 XGBModel，整组期间持续追加数据。

资源监控：[guard_experiment_resources.py](guard_experiment_resources.py)：

- 每 5 秒记录本次启动进程树中各进程 RSS / private bytes 和系统物理 / 提交内存余量。
- 物理内存保留 1 GiB，系统提交余量保留 4 GiB，本次实验进程树私有内存上限 4 GiB。
- 同一压力条件连续 3 次采样出现时，终止本次启动的实验进程树，并保存停止原因；不关闭其他应用，不自动反复重试。
- 未改变页面文件或系统设置。
- 已通过真实子进程测试，确认终止本次启动的内存超限子进程，保留另一个独立运行的测试进程。
- 外部监控与实验进程分开，B/C/D 都在相同保护设置下执行。若资源保护触发，该尝试不作为完整性能比较。

状态以以下文件为准，不能用旧目录判断新进度：

- [新 progress.json](results/paper_v1/operator800_20261011_r02/progress.json)：当前调优 / 重测 / PyTorch / 报告阶段。
- [resource_guard/status.json](results/paper_v1/operator800_20261011_r02/resource_guard/status.json)：监控进程和停止原因。
- `resource_guard/memory.jsonl`：全程内存采样。
- `resource_guard/child.console.log`：驱动 stdout / stderr；每个调优组仍有自己的 `*.console.log`。
- `execution.jsonl`：每个子任务退出码与日志哈希。

每个算子的 B/C/D 和独立重测完成后，自动输出到 `papers/figures/operator800_20261011_r02/`。全部完成才生成 `COMPLETION.json`。论文和上一批完整短预算结果未修改。

后续若监控停止，先对照内存日志里的 PID 与 progress / execution，确认增长来自调优主进程、持久测量 worker 或短时编译 worker，再修复和冻结新的尝试；不要直接重复启动同一个不完整目录。

## r02 中止与监控修正（13:32–13:36）

r02 的 S01 正常结束。MatMul128 baseline 保留了 4 批完整反馈和 40 条测量记录，未完成 800 次预算。

13:27 时曾把停止归因于监控计入零线程子进程的历史 private bytes，但后续 r03 证明这一判断不完整：零线程条目伴随真实的系统提交内存增长，最新定位见文末。清理在一个零线程子进程的 `AccessDenied` 上中断。13:32 已按 PID 与创建时间校验身份后，终止仍存活的实验主进程、runner 和 builder；清理记录为 `r02/resource_guard/cleanup_053200.json`。原始日志和数据库保持原样，`manual_stop.json` 保留了停机前的进度快照。

资源监控现先排除零线程的已退出进程，保留被排除 PID 与存活进程的命令行；清理遇到单个进程错误时继续处理其他目标，并使用此前记录的进程身份清理失去父进程的残留 worker。5 项验证通过，覆盖已退出进程的内存排除、单个清理失败、已退出父进程，以及实际内存保护不影响无关进程。原始 r01 的内存分配失败原因仍需在完整运行中定位，此修正不代表该问题已解决。

13:36:10 启动 r03，guard PID=3440。r03 冻结协议与 r02 完全一致，所有组重新从空历史开始；监控脚本单独保存新快照。当前先执行 S01 两批检查，再执行 MatMul128 的 B/C/D，后续三个算子自动串行运行。

- 当前进度：[r03 progress.json](results/paper_v1/operator800_20261011_r03/progress.json)。
- 资源状态：[r03 resource_guard/status.json](results/paper_v1/operator800_20261011_r03/resource_guard/status.json)。
- 冻结协议：[r03 protocol.json](results/paper_v1/operator800_20261011_r03/protocol.json)。
- 新图表输出：`papers/figures/operator800_20261011_r03/`。

截至启动时没有任何完整的 800 次正式结果。后续确认仍以新进度、执行回执和资源状态为准，不能以进程已启动推断运行已完成或环境故障已排除。

13:41:16 更新：S01 正常退出，4/4 候选有效，模型累计样本为 2→4，第二批更新前的模型验证日志完整。已进入正式 `M128G1B01`（MatMul128 baseline，800 次提交），初始 PID=1184。额外补充了实际存活孙进程的清理检查，资源保护共 6 项验证通过。正式结果仍未完成，原始内存分配失败是否复现继续由新日志观察。

## 每 15 分钟后台巡检（13:50 起）

根据用户要求，已启动 [watch_operator800_study.py](watch_operator800_study.py)，初始 PID=9160，每 900 秒读取当前进度、完整批次提交数、最优实测延迟、模型累计数据、错误数量、进程存活和内存采样，保存一条巡检记录。13:50 首次检查时正式 MatMul128 baseline 已完成 30/800 次提交，累计训练样本 30，未发现异常；下次常规检查约 14:05。资源守护仍每 5 秒采样并按既定阈值保护，不受巡检周期影响。

- 最新可读报告：[monitor15min/STATUS.md](results/paper_v1/operator800_20261011_r03/monitor15min/STATUS.md)。
- 最新完整快照：`monitor15min/latest.json`；历次记录：`monitor15min/checks.jsonl`。
- 巡检器身份、间隔与状态：`monitor15min/watcher.json`，启动后核验身份存活。
- 巡检器只读实验输入，不修改算法、预算、模型或候选。实验正常完成或资源守护中止时立即写最后一次检查并结束；异常状态标为 `needs_attention`。记录写入本地文件，不代表定时向当前对话发送消息。

## DLL 加载修复及最新运行 r04（14:03）

13:50:43，r03 在第 4 批测量之后因系统提交余量连续低于阈值而停止。完整模型反馈为 30 条，数据库为 40 条；停止时活跃实验进程的 private bytes 约 909 MiB，系统提交余量约 1.26 GiB。guard 正常清理，无存活残留。停止后系统提交余量恢复到约 14 GiB。r03 的原始数据保留，不能拼入新搜索。

独立诊断定位到 `python/tvm/runtime/module.py` 的 Windows 分支：每次加载 `.tar` 候选都尝试预加载库目录下所有 DLL，包含 `tvm_ffi_testing.dll`。该测试库在已有类型注册的进程内反复触发 `Overriding testing.TestIntPair`，加载阶段出现零线程 Python 子进程条目及真实提交内存增长。零线程并不充分证明只剩历史计数；不能据此排除系统内存压力。

现仅预加载链接所需的 `tvm_ffi.dll` 和 `tvm_runtime.dll`，不加载无关库，核心依赖加载失败会正常上报。对照使用同 4 个既有候选、同 compiler DLL：

| 诊断 | 加载次数 | 进程私有内存范围（MiB） | 最大残留子进程数 | 观测系统提交内存增长（MiB） |
|---|---:|---:|---:|---:|
| 修复前 | 4 | 320.79–341.20 | 4 | 1300.71 |
| 修复后 | 4 | 289.97–290.02 | 0 | 52.79 |
| 修复后连续验证 | 80 | 290.39–291.77 | 0 | 431.55 |

三次诊断均完成矩阵单位输入检查，修复后的 80 次验证进程正常退出；系统提交数受其他应用影响，不视为进程自身内存或论文性能结果。诊断原始事件位于 `tmp/module_commit_diag_20261011_r02/`、`tmp/module_commit_diag_20261011_r03/`、`tmp/module_commit_stress_20261011/`，汇总为 [MODULE_COMMIT_FIX_20261011.json](../tmp/MODULE_COMMIT_FIX_20261011.json)。原先未设置正确库路径的第一诊断失败，保留在 `tmp/module_commit_diag_20261011/`，不用于对照结论。

14:02:51 从 r04 新目录重新执行全部单算子流程，guard 初始 PID=4952。Python runtime 身份因修复而重新冻结，compiler DLL 未变；B/C/D、预算、seed、GA、代价模型及正确性设置不变，各组重新从空历史开始。14:03:26 启动新巡检器，初始 PID=3368，间隔 900 秒，下次常规巡检约 14:18:27；旧 r03 巡检器已校验身份并停止。

- 最新入口指针：[OPERATOR800_CURRENT.json](OPERATOR800_CURRENT.json)。
- 当前运行：[r04 progress.json](results/paper_v1/operator800_20261011_r04/progress.json)。
- 资源守护：[r04 resource_guard/status.json](results/paper_v1/operator800_20261011_r04/resource_guard/status.json)。
- 最新巡检报告：[r04 monitor15min/STATUS.md](results/paper_v1/operator800_20261011_r04/monitor15min/STATUS.md)。
- 巡检记录：`r04/monitor15min/checks.jsonl`；巡检器状态：`r04/monitor15min/watcher.json`。

新版巡检器在资源守护停止时立即记录一次异常，随后仍按每 15 分钟记录停止状态；完整实验正常结束后退出。它只写本地检查文件，不能自动触发本对话中的新回复。每 5 秒的内存保护继续独立执行。截至启动仍没有完整的 800 次正式结果，完整搜索的稳定性继续观察。

14:07:56 更新：r04 的 S01 正常退出，4/4 候选有效、两批模型反馈完成。驱动已自动进入正式 `M128G1B01`（800 次提交），初始 PID=23956；guard 和每 15 分钟巡检器均仍存活。

## 独立完成与恢复控制器（15:18）

15:18:28 启动控制器，初始 PID=23536，接续健康运行的 r04，没有启动第二个 GPU 调优任务。启动检查为 MatMul128 baseline 414/800、累计模型样本 414；15:20:57 为 434/800。正式完整组尚未产生。

恢复机制、验收条件及重启命令见 [OPERATOR800_COMPLETION.md](OPERATOR800_COMPLETION.md)。当前统一状态见 [恢复巡检报告](results/paper_v1/operator800_20261011_cohort/monitor15min/STATUS.md)，原始 r04 数据仍保留原目录。冻结源码不变；完整组必须同时具有成功退出记录、800 条真实提交及文件来源审核。补跑使用独立尝试目录，不拼接中断片段。14 项恢复测试、3 项模型历史审核测试和 6 项资源保护测试通过。

## 物理内存阈值和恢复目录修正（15:34）

r04 在 15:21:18 的下一批编译期间因物理余量连续低于 1 GiB 停止，完整反馈 434 次。当时提交余量约 11.33 GiB、实验树 private 约 1.64 GiB、无零线程条目，和此前的提交内存耗尽不同。第一次 baseline 补跑也触发同一物理保护；C/D 则在开始调优前因日志路径过长失败。

15:26 身份校验后停止第一版控制器及其子进程，未清理原始数据。修正后的原始目录采用短路径，最长日志路径经所有配置检查为 220 字符；物理保留调整为 256 MiB，提交保留 4 GiB、private 上限 4 GiB、5 秒采样和连续三次保护仍启用。旧故障以实际日志和资源停止采样标记修复来源，保留证据，不计入新策略的重试上限。所有正式组仍未完成，最终比较组将采用统一的新保护设置。

15:34:27 启动控制器第二版，初始 PID=11832，当前 MatMul128 baseline 原始目录为 [o8_6ff1d8_001](results/o8_6ff1d8_001/progress.json)。从空历史重新运行，不拼接旧 434 次。源码、DLL 和调优/训练参数仍冻结。18 项恢复测试通过，新增覆盖路径长度及物理低谷与提交内存不足的区别；其他 9 项相关测试此前通过。

## 首个完整 baseline 和末尾审核误报（18:14）

`o8_6ff1d8_001` 于 17:50 完成 MatMul128 baseline 的 800 次提交，800 个有限测量与数值检查齐全，81 批模型反馈累计至 800 个样本。总调优 8040.721 秒，搜索最佳 27.246462 μs，首次观察于 446 次提交、74.08 分钟。独立重测待完成。

末尾回调校验将“每批最多 10 个”错写为固定每批 10 个，六个 9 候选的中间批和最后 6 个候选因此误报。完整结果先保存后异常退出；原退出码 1 和原日志保留。自动恢复曾启动重复尝试 `o8_6ff1d8_002`，17:59 通过 PID、创建时间与命令身份校验停止控制器及重复子树，留下独立停止证据。

新增外部独立审核和每组自动绘图，不修改冻结调优源码或 DLL。独立核对真实批次长度、原数据库延迟和候选 trace、时间线、四类数值检查、GA 评分架构、累计训练与原退出日志后，可以接受这一明确的末尾审核例外；其他失败继续排除。派生审核视图与原数据分开，任务清单如实记录原退出码和审核依据。具体规则见 [OPERATOR800_COMPLETION.md](OPERATOR800_COMPLETION.md)。

18:14 控制器第三版启动（初始 PID 23248），保留首组完整数据，继续剩余正式组和算子。首组 [收敛图](../papers/figures/operator800_20261011_cohort/arms/M128G1B01/convergence.png)、[训练图](../papers/figures/operator800_20261011_cohort/arms/M128G1B01/model_training.png) 与 [说明](../papers/figures/operator800_20261011_cohort/arms/M128G1B01/RESULTS.md) 已生成。每 15 分钟本地巡检、每 5 秒资源保护继续启用。

## 保持原状态页路径，增加 30 秒刷新（20:00）

用户打开的是 cohort 的 `monitor15min/STATUS.md`。此前该页按 900 秒巡检频率写入，19:50 查询时仍显示 19:44 的 515 次反馈，而实际已经 545 次，实验并未卡住。

现新增轻量实时发布器，每 30 秒原子更新同一个 `STATUS.md`，同时显示当前编译/测量/模型反馈阶段。15 分钟巡检记录仍保留，原摘要改为 `AUDIT_STATUS.md`；两类输出由各自唯一的写入进程管理。20:00:15 只替换控制器，存活的调优进程 PID 3740 与 guard PID 11356 保持原创建时间、没有重启或重置模型。新控制器初始 PID 11624，发布器初始 PID 19096；交接记录和旧状态页副本保留在 cohort。

20:00:17，实时页正确显示 baseline 800/800、CS+H 605/800（累计模型样本 605），其余任务继续排队。34 项相关测试通过。状态入口仍为 [STATUS.md](results/paper_v1/operator800_20261011_cohort/monitor15min/STATUS.md)，正式审计摘要为 [AUDIT_STATUS.md](results/paper_v1/operator800_20261011_cohort/monitor15min/AUDIT_STATUS.md)。

## CS+H 完成，对比图与卷积/规约优先级（20:35）

20:32:10，MatMul128 CS+H 完成 800 次实际提交和 800 个累计模型样本，全部有效测量有正确性记录。独立审核接受已定位的末尾批大小误报，原退出码和原始文件保留。Baseline / CS+H 的收敛、目标时间与阶段耗时图已生成并检查，见 [两组对比](../papers/figures/operator800_20261011_cohort/comparisons/matmul128_BC/RESULTS.md)。严格 baseline 目标未被 CS+H 达到；宽松目标的时间单独呈现。独立赢家重测仍待执行。

20:32:36 只替换调度器，初始 PID 12360；CS+H 已在交接之前自然结束，没有中断 GPU 工作。按用户指示将调度优先级设为 MatMul128 → Conv2D → Reduction4096 → MatMul256。冻结协议、源码、DLL、seed、预算和模型设置均保持原身份。优先级证据保留；原证据中空 PID 导致误记录交接脚本自身的条目，由独立补充说明明确排除，原文件不覆盖。

20:34:49，MatMul128 D（CS+H+Draft-GA）在 `o8_6ff1d8_004` 开始，初始调优 PID 23916、guard PID 7708。同一算子两个完整组即自动生成对比图，三个组齐全后另生成三组图。绘图失败不影响后续调优。37 项相关测试与冻结源码检查通过；正式完成 2/12，后续自动串行执行卷积和规约。
