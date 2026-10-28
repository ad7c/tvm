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

# 800 次提交单算子实验：完成与恢复入口

2026-10-11 15:18 启动独立恢复控制器，初始 PID 23536，接续当时健康运行的 r04。15:20:57 达到 MatMul128 baseline 434/800；15:21:18 因可用物理内存连续低于 1 GiB 被保护停止，当时提交余量仍约 11.33 GiB、实验树 private 约 1.64 GiB，未重现早先的提交内存耗尽。

首次恢复还发现目录层级过深，C/D 的日志路径超过 Windows 传统路径限制，启动失败。15:26 按身份终止恢复控制器及其子进程，保留所有原始文件。修正后最长日志路径为 220 字符，物理保留阈值改为 256 MiB，提交/实验树上限继续保留。15:34:27 启动第二次控制器，PID 11832；当前 MatMul128 baseline 从空历史重跑，原 434 次不拼接。完整正式组仍为 0/12。

## 固定实验设置

- 四个 FP32 算子：MatMul128、MatMul256、标准 NCHW Conv2D 3×3、大 Reduction [128,4096]。
- B = baseline，C = CS+H，D = CS+H+Draft-GA；一个 seed 20261011，GA 1 代，每批最多 10 次提交。
- 每组 800 次真实提交，失败占预算。每组独立使用一个持续保留历史的 XGBModel；每批按全部累计、具有特征的反馈重新训练 booster。
- 不修改冻结源码、compiler DLL、训练设置或正确性检查，不混入 r01–r03 的中断数据，不拼接同一组的两次搜索。

## 实际运行机制

入口是 [complete_operator800_study.py](complete_operator800_study.py)，审核与报告入口是 [operator800_cohort.py](operator800_cohort.py)。

健康的已存在驱动会被接续，控制器只读取其状态和退出记录。发生失败时，控制器先清理具有匹配 PID/创建时间的实验残留，再保留通过审核的完整组，只运行尚未完成的任务。当前已进入逐组恢复执行阶段，每个任务独立受到 5 秒资源保护。

每次补跑使用独立目录，从空历史开始。任务清单 `cohort.json` 关联每组的结果路径、执行记录、提交数和原始文件哈希。默认每个任务最多两次未解决故障的尝试，包含已接续的失败尝试；第二次失败后保留证据、继续其他可执行组，最终标为 `needs_attention`。已诊断且实施修复的旧尝试保留 `resolved_by` 和证据，不消耗修复后的重试额度。共享的日志路径启动故障会立即中止控制器，防止在其他组中重复同一问题。搜索空间提前耗尽单独记录，不能补重复候选或把不足 800 的结果当成满预算组。

补跑前连续三次检查可用物理内存至少 1.5 GiB、提交余量至少 6 GiB；运行中的资源保护每 5 秒检查，阈值为物理保留 256 MiB、提交保留 4 GiB、实验树 private 上限 4 GiB、连续三次超限停止。原 r04 的物理保留阈值为 1 GiB；没有在两个不同阈值下保留完整正式组进行比较。控制器不会关闭其他应用或更改系统设置。

补跑的原始目录采用 `experiments/results/o8_6ff1d8_001` 等短路径，任务身份由 `recovery_task.json` 和总清单保存。创建任务前检查所有注册配置的最长日志路径不超过 240 字符；当前最大值 220。未改变冻结驱动的命名逻辑，也未修改 Windows 长路径注册表设置。

恢复阶段优先完成尚未完成的调优组，再完成可重测算子的独立重测和 PyTorch 参考。后处理故障不阻塞其他算子的调优。所有 GPU 任务串行运行。

## 状态与产物

- 最新入口：[OPERATOR800_CURRENT.json](OPERATOR800_CURRENT.json)。
- 最初源实验（已中断）：[r04 resource guard](results/paper_v1/operator800_20261011_r04/resource_guard/status.json)。
- 当前补跑：[progress.json](results/o8_6ff1d8_001/progress.json)。后续任务位置以最新入口指针为准。
- 恢复任务清单：[cohort.json](results/paper_v1/operator800_20261011_cohort/cohort.json)。
- 每 15 分钟报告：[STATUS.md](results/paper_v1/operator800_20261011_cohort/monitor15min/STATUS.md)；历史记录为同目录 `checks.jsonl`。
- 当前控制器日志：`results/paper_v1/operator800_20261011_cohort/controller.r02.stdout.log` 和 `controller.r02.stderr.log`。第一版日志、源码快照及失败尝试均保留。
- 原驱动图表：`papers/figures/operator800_20261011_r04/`；最终跨尝试汇总：`papers/figures/operator800_20261011_cohort/`。

本地巡检不会自动触发对话回复。首次越过 100 次提交的累计反馈与当时检查到的资源快照保存在任务清单的 `milestones100`，不能将该快照解释为第 100 次提交时的精确内存。

每组必须具有可核验的退出记录、800 条数据库记录、完整批次反馈、持续累计模型历史和正确性覆盖。通常要求原调优进程退出码 0；2026-10-11 发现的“固定满批次”末尾审核误报按下文的独立审核例外处理，原退出码 1 如实保留。每个算子的三个原搜索赢家固定后独立重测 30 个样本，并审核四类数值检查、原数据库及 trace 的哈希；PyTorch 参考也须完成 30 个样本和四类检查。

最终输出收敛图（提交数和时间）、达到目标的时间、阶段耗时、组件数量与预测开销、模型训练数据和训练/新批预测指标，以及 `RESULTS.md`、`summary.json`、`model_training.csv`。完整任务清单会先保存为不可变的 `report_input_*.json`，报告清单引用其哈希。

只有 12 组 × 800 次提交、四算子的独立重测、四个 PyTorch 参考及报告全部通过审核，才写 `COMPLETION.json`。完整性要求是 9600 次提交，不是 9600 次有效测量。

## 重启后的恢复命令

控制器已在后台运行时不要重复启动。电脑重启或控制器结束后，使用：

```powershell
$env:TVM_EXPERIMENT_NO_TORCH = '1'
$env:TVM_FFI_DISABLE_TORCH_C_DLPACK = '1'
$env:MKL_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_THREADING_LAYER = 'SEQUENTIAL'
$env:PYTHONPATH = 'F:\tvm\experiments\runtime_bootstrap;F:\tvm\python;F:\tvm\experiments'
$env:PATH = 'F:\tvm\.venv-tune\Library\bin;' + $env:PATH
& .\.venv-tune\python.exe -u -X utf8 experiments\complete_operator800_study.py `
  --input experiments\results\paper_v1\operator800_20261011_r04 `
  --output experiments\results\paper_v1\operator800_20261011_cohort `
  --figure-output papers\figures\operator800_20261011_cohort
```

`--dry-run` 只检查冻结源码和当前进程，不写文件、不启动调优。独立重生成报告使用：

```powershell
& .\.venv-tune\python.exe -X utf8 experiments\operator800_cohort.py `
  --manifest experiments\results\paper_v1\operator800_20261011_cohort\cohort.json `
  --output papers\figures\operator800_20261011_cohort
```

验证：18 项恢复/来源/预算/路径/资源策略测试、3 项训练历史审核测试、6 项资源保护测试通过；真实 S01 的成功退出记录、注册命令和八个原始文件哈希审核通过。冻结源码检查通过。论文正文和此前短预算实验未替换。

15:47 检查：新轮 baseline 完整反馈 49/800，模型累计样本依次 10、20、30、40、49；构建和运行错误均为 0，资源保护仍正常运行，提交余量约 12 GiB。报告入口在 `tmp/operator800_cohort_report_qa` 的只写报告检查通过，正确标为正式组 0/12、`completion_ready=False`；这些空预算检查产物不作为论文实验结果。

## 首个 800 次结果与末尾审核修正（18:14）

17:50，`o8_6ff1d8_001` 的 MatMul128 baseline 已完成 800/800 次提交，800 个有限结果均有四类数值正确性检查，XGBoost 的 81 次反馈持续累计到 800 个样本。调优时间 8040.721 秒；搜索最佳 27.246462 μs，首次在累计 446 次提交、4444.873 秒时观察到。独立 30 样本重测尚未完成，不能将搜索最小值作为独立重测结论。

冻结脚本错误地假设所有中间批次都必须满 10 个候选，因此把实际的 81 批（74 批 × 10、6 批 × 9、末批 × 6）误判为失败。搜索已经返回并保存完整结果，只有末尾架构审核触发 `RuntimeError`，原进程退出码 1。控制器随后错误地启动了 `o8_6ff1d8_002` 重跑；17:59 已按 PID、创建时间和命令核验身份后停止重复进程树，停止证据单独留档，没有拼接或删除记录。

新增 [operator800_postcheck.py](operator800_postcheck.py) 外部审核：每批不超过 10，实际批大小之和与数据库、时间线和预算一致；校验归档回调、原数据库实测延迟和排序后的 trace 身份、四类数值检查、原 GA 评分架构、训练历史、注册命令、冻结协议、控制台及资源守护退出。仅允许“全部错误恰好等于旧回调审核误报、普通退出 1、干净清理、控制台最后一行为指定末尾审核异常”的结果走此路径；内存停止、缺少正确性、其他架构错误、预算不足均拒绝。

独立派生视图写在原尝试的 `audit_v2/`，审核器版本进入文件名和哈希；原 `result.json`、数据库、日志和退出码不改写。任务清单明确标记 `completion_basis=independently_revalidated_postcheck`、`original_returncode=1`，报告继续引用原搜索结果及原赢家。模型/编译器/冻结脚本保持原哈希。

18:14 启动第三版控制器（初始 PID 23248）。每个正式组通过审核后自动生成 `papers/figures/operator800_20261011_cohort/arms/<ID>/` 下的提交数/时间双横轴实测收敛图、训练监测图、PDF/SVG/PNG、CSV、结果说明和来源清单；绘图采用同一受控运行环境，失败仅重试绘图，不重跑调优。首组图表见 [baseline RESULTS.md](../papers/figures/operator800_20261011_cohort/arms/M128G1B01/RESULTS.md)。

31 项预算/恢复/小批次审核/训练历史测试与 6 项资源保护测试通过；真实 800 次 baseline 的外部审核、绘图与冻结源码检查通过。完整套实验仍未完成。

## 状态页实时刷新（20:00）

`monitor15min/STATUS.md` 原来只在每 15 分钟完整巡检时写入，因此期间的提交数会落后。现保持用户打开的同一路径，改由 [watch_operator800_cohort.py](watch_operator800_cohort.py) 每 30 秒原子刷新：显示当前任务、编译/测量/反馈阶段、完整批次提交数、模型累计样本、已审核组和资源状态。更新时间改变而提交数不变，表示当前批次尚未完成。

每 15 分钟的 `checks.jsonl` 和 `latest.json` 审计频率保留，摘要改写 `monitor15min/AUDIT_STATUS.md`，由控制器单独写入；实时发布器单独写 `STATUS.md`，没有双写竞争。发布器读取当前 cohort 和活动目录，不启动 GPU 工作、不改变模型或实验参数，不把旧中断组拼进进度表。最新实时 JSON 为 `live_status/latest.json`。

20:00:15 只交接控制器（新初始 PID 11624），原调优进程 PID 3740 和资源守护 PID 11356 的创建时间前后相同，原 CS+H 运行继续；证据保存在 `status_handover_*.json`。控制器恢复时自动接续存活的 guard，并确保只启动一个实时发布器（初始 PID 19096）。20:00:17 的 CS+H 为 605/800、模型累计 605，baseline 800/800 保留。旧状态页已另存 `STATUS_before_live_*.md`，未改写原搜索数据。

新增 3 项状态刷新测试通过，验证同一文件的新提交数和时间戳、成功组保留、原巡检记录不被覆盖、已有发布器不重复启动及当前阶段/资源显示；总计 34 项相关测试通过。

## 两组完整结果、自动对比图与后续顺序（20:35）

MatMul128 CS+H 在 20:32:10 完成 800 次实际提交，800 个有限测量均有数值正确性记录，81 次模型反馈累计到 800 个样本。它与 baseline 一样仅触发已定位的末尾批大小审核误报；原退出码 1 保留，外部完整审核通过，任务清单记录相同的独立重审依据。正式组已审核 2/12，整套实验尚未完成。

[两组对比说明](../papers/figures/operator800_20261011_cohort/comparisons/matmul128_BC/RESULTS.md) 包含提交数和时间两个横轴的收敛曲线、达到严格 baseline 目标及 +1%/+5% 目标的时间、互不重叠的阶段耗时。Baseline / CS+H 的搜索最佳分别为 27.246462 / 27.253157 微秒，总调优时间为 134.01 / 135.21 分钟；CS+H 未达到严格 baseline 最终延迟，图表如实标注未达到。这些是搜索实测值，尚未使用独立赢家重测。

[plot_operator800_comparison.py](plot_operator800_comparison.py) 在同一算子有两个完整方法组且包含 baseline 时自动绘图，第三组完成后另生成 B/C/D 对比。来源清单保存输入及输出哈希；失败只重试绘图，不重跑调优。新增目标未达到和调度优先级测试后，37 项相关测试通过，冻结源码检查通过。

按用户要求，独立调度优先级改为 MatMul128 → Conv2D → Reduction4096 → MatMul256，原协议和全部实验参数不改。20:32:36 只交接调度器，新初始 PID 12360；CS+H 的 guard 已在交接前自然退出。优先级原记录与补充说明保留在 cohort 的 `priority_change_*.json`，补充说明排除原记录中由空 PID 误采集的交接脚本自身身份。20:34:49，MatMul128 D 在 `o8_6ff1d8_004` 启动（初始调优 PID 23916、guard PID 7708）。D 完成后自动补三组图，随后串行执行卷积和规约。
