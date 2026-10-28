# 约束引导 MetaSchedule：v1.1 实施日志

本日志对应 `experiments/IMPLEMENTATION_EXPERIMENT_PLAN.md` v1.1。所有 8-trial 结果只标记为 smoke；只有真实 C++ 编译、TVM 调优和 CUDA 运行才记为在线实验。

## 2026-09-02：环境恢复与 I0 启动

### 冻结身份

- Git HEAD：`4a1297ad59c9ff80de535a8ba898bf6d9012e5c7`
- 方案 SHA256：`AB4D85A3856F7D3CFC0E831D16E1362620F042E29F4F0192728A48204ABE89B7`
- 当前编译库：`F:\tvm\build-prune\lib\tvm_compiler.dll`
- DLL SHA256：`47083FE920DBE94560382C4BA9DB343BE4769F59AB11D050BECA7C9B4595E195`
- DLL 时间：`2026-09-02T05:44:07.6126339+08:00`
- GPU：NVIDIA GeForce RTX 3050 Ti Laptop GPU，compute capability 8.6，4096 MiB
- 驱动：591.44
- CMake 配置：Ninja，RelWithDebInfo，MSVC 14.44，CUDA Toolkit 12.6

### Python 环境故障与恢复

旧环境 `D:\conda_envs\tvm-build-venv` 的 `python.exe` 启动立即返回 `0xC0000022 / ACCESS_DENIED`。TVM C++ 构建本身正常：

```powershell
$env:TEMP='F:\tvm_tmp'
$env:TMP='F:\tvm_tmp'
$env:CCACHE_DISABLE='1'
cmake --build build-prune --parallel 12
```

结果为 `ninja: no work to do.`，所以阻塞不在 C++/CUDA 构建链。

尝试离线新建包含最新 NumPy、XGBoost、pytest、Cython 和 scikit-build-core 的环境时，本地缓存不完整，conda 返回 `OfflineError`。随后尝试最小环境求解，沙箱内因 `D:\conda_pkgs` 无法写锁而返回 `NoWritablePkgsDirError`；经典求解器扫描完整 repodata 过慢，已主动终止，未留下环境。

最终采用离线克隆，保留旧环境不变：

```powershell
$env:CONDA_NO_PLUGINS='true'
$env:TEMP='F:\tvm_tmp'
$env:TMP='F:\tvm_tmp'
D:\miniconda\Scripts\conda.exe create `
  -p F:\tvm\.venv-tune `
  --clone D:\conda_envs\tvm-build-venv `
  --offline -y
```

克隆成功，共 87 个 conda 包。新环境版本：Python 3.11.15、NumPy 2.4.6、XGBoost 3.2.0、psutil 7.2.2。未对 TVM 或 tvm-ffi 使用 editable install。

验证命令使用：

```powershell
$env:TEMP='F:\tvm_tmp'
$env:TMP='F:\tvm_tmp'
$env:PYTHONNOUSERSITE='1'
$env:PYTHONPATH='F:\tvm\python'
$env:TVM_LIBRARY_PATH='F:\tvm\build-prune\lib'
$env:PATH='F:\tvm\.venv-tune;F:\tvm\.venv-tune\Library\bin;F:\tvm\.venv-tune\Scripts;F:\tvm\build-prune\lib;' + $env:PATH
F:\tvm\.venv-tune\python.exe -c "import ctypes, numpy, tvm_ffi, tvm; print(tvm.cuda(0).exist)"
```

验证结果：

- Python 从 `F:\tvm\.venv-tune\python.exe` 启动。
- NumPy、tvm_ffi、仓库 `F:\tvm\python\tvm` 均成功导入。
- `tvm.cuda(0).exist == True`。

### I0 当前状态

- 已确认 `VerifyGPUCode` 总失败当前混合三层：`ThreadExtentChecker`、lowering exception、lowered TIR 的 GPU 资源校验。
- lowered TIR 校验器实际可能报告线程轴/threads per block、shared memory、local memory、vthread、vector bytes 和 kernel count。
- 公开分析 API 当前只返回 bool，内部已经生成详细错误字符串；I0 将暴露只读详细错误接口，并在 `ThreadedTraceApply` 中记录候选级主失败原因与守恒计数。
- 尚未产生新的性能数据；下一结果先完成编译、单元测试、off/shadow 等价性，再运行 8-trial E0 smoke。

### I0 C++ 构建与最小验证

首次全量构建在沙箱外调用 MSVC，完成 `222/222` 并链接真实的
`F:\tvm\build-prune\lib\tvm_compiler.dll`。随后发现新增详细 API 的非法线程测试
没有被校验器拒绝。定位结果不是 Python 模拟或 DLL 未更新，而是
`src/s_tir/analysis/verify_gpu_code.cc` 使用 `IterVar.var.name_hint` 识别
`threadIdx.x/y/z`；TVMScript 中变量名为 Python 标识符，真正的 GPU 轴语义位于
`IterVar.thread_tag`。这会导致直接分析尚未 canonicalize 的 TIR 时漏检线程约束。

已改为读取 `IterVar.thread_tag`，并完成 2 个目标的 MSVC 增量构建。新 DLL 身份：

- 路径：`F:\tvm\build-prune\lib\tvm_compiler.dll`
- SHA256：`17DBEB431B38FABCC9D6169C0BC53B724742B9EEBA7C5F6ECD2C65D79CC85A70`
- 大小：220055040 bytes
- 修改时间：2026-09-02 22:47:27 +08:00

测试环境最初缺少 pytest 和 TVM 测试插件依赖。仅向隔离环境
`F:\tvm\.venv-tune` 安装 `pytest==9.1.1` 与 `ml_dtypes==0.6.0`；未安装或替换
TVM/tvm-ffi。

真实 DLL 测试结果：

- `test_s_tir_analysis_verify_gpu_code.py`：2 passed。合法输入返回空错误列表；
  2048-thread 输入返回详细的 `threads per block`、2048 和 1024 信息。
- `test_meta_schedule_thread_feasibility.py`、
  `test_meta_schedule_thread_filter_modes.py`、
  `test_meta_schedule_mutator_mutate_tile_size.py`：合计 24 passed。
- 三模式集成用例的候选守恒均为 `generated=accounted=1`、
  `conservation_delta=0`。off/shadow 的 Verify 主原因为
  `threads_per_block=1`；enforce 由硬过滤拦截，Verify 失败为 0。
- 既有 `test_meta_schedule_postproc_verify_gpu_code.py`：8 passed，确认
  thread-tag 修复没有破坏已有合法/非法 GPU postproc 用例。
- `git diff --check` 通过；仅 Git 无法读取用户级 ignore 的环境警告。

实验解析脚本 `experiments/e1_thread_filter.py` 已升级为 schema 3：结果 JSON 将记录
每个 SampleInit/Evolve/PickBest 阶段的 generated、trace replay、逐 postproc、逐 Verify
原因、hard reject、passed、accounted 和 conservation delta；off/shadow 比较也会把候选
计数完全一致及两侧守恒作为 I0 阶段门。

这些结果仍是正确性测试，没有产生或声称任何性能提升。

### I0 阶段门与 E0/H 单 seed 8-trial smoke

固定配置：MatMul 128³、RTX 3050 Ti (`sm_86`)、seed 20260902、8 trials、单批、
population 512、SampleInit 最少 50、GA 4 代，P 与 GA T2 均关闭。三臂使用相同
schema-3 源码身份与 DLL SHA256。机器结果位于：

- off：`experiments/data/v1_1_i0/v1_1_i0_pair0_seed20260902_off_soft-off_ga-off/result.json`
- shadow：`experiments/data/v1_1_i0/v1_1_i0_pair0_seed20260902_shadow_soft-off_ga-off/result.json`
- enforce：`experiments/data/v1_1_i0/v1_1_i0_pair0_seed20260902_enforce_soft-off_ga-off/result.json`

off baseline 的候选归因：

| 阶段 | generated | passed | Verify 失败 | thread_extent | shared_memory | lowering_exception | delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| SampleInit | 1024 | 97 | 927 | 837 | 67 | 23 | 0 |
| Evolve（4 代合计） | 2023 | 1766 | 257 | 155 | 93 | 9 | 0 |
| PickBestFromDatabase | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 合计 | 3047 | 1863 | 1184 | 992 | 160 | 32 | 0 |

baseline 总 Verify 拒绝率为 1184/3047 = 38.86%。在 Verify 失败内部，
`thread_extent` 占 83.78%，`shared_memory` 占 13.51%，`lowering_exception` 占
2.70%；其它已分类原因和 unknown 均为 0。SampleInit 本身的失败率为
927/1024 = 90.53%，Evolve 为 257/2023 = 12.70%。992 个 thread-extent 失败中
837 个来自 SampleInit、155 个来自 Evolve，因此当前主要问题发生在初始 population
采样，不是仅由 GA 变异造成。本轮第一批没有历史数据库记录，PickBest 阶段生成数为 0，
所以这组数据不能回答有历史数据库时的 PickBest 行为。

`thread_extent` 的语义已经核对：高层 `ThreadExtentChecker` 检查 schedule block 上
`meta_schedule_thread_extent_low/high_inclusive` 与实际 x/y/z 线程乘积是否一致；它不是
简单的硬件 `max_threads_per_block` 超限。这解释了为什么目前只检查硬件线程上限的 H
无法覆盖大多数该类失败。

I0 off/shadow 阶段门结果全部通过：

- 8 条最终实测 trace 的 SHA256 多重集一致。
- 两边 generated/passed/逐 postproc/逐 Verify 原因/hard reject/accounted 账本完全一致。
- 两边 Verify 失败均为 1184，且每个阶段 `conservation_delta=0`。
- shadow 检查 3047 个候选，unknown=0，预测 7 个；TP/FP/FN/TN 为
  7/0/1177/1863，precision=100%，recall=0.59%，观察到的误拒为 0。
- shadow 线程分析总开销 15808 us，即约 5.19 us/候选。

H-enforce 结果：

- 预测 7、实际硬拒绝 7；SampleInit 拦截 1，Evolve 拦截 6。
- 总账本为 `3047 = 1863 passed + 1177 Verify failed + 7 hard rejected`，delta=0。
- Verify 原因变为 thread_extent=985、shared_memory=160、lowering_exception=32；
  与 off 相比恰好少 7 个 thread_extent，符合 shadow 真阳性结果。
- off/shadow 各有 1 个 builder error、7 个有限实测记录；enforce 为 8/8 构建和实测
  成功。尚不能证明原 builder error 就是被 H 拦截的候选，因为 enforce 会改变后续选择。
- 搜索时间为 off 75s、shadow 78s、enforce 89s，enforce 本轮没有搜索阶段加速。
  tune call 为 288.5s、299.7s、244.8s；enforce 的 builder 时间为 62s，而 off/shadow
  为 133s/136s。单 seed 且受 builder error 影响，不能据此声称端到端加速。
- 最佳延迟 off/shadow/enforce 分别为 49.49/81.90/43.14 us。即使 off/shadow 的 trace
  多重集一致，测量值仍明显波动，再次说明 8-trial 单次结果只可作功能 smoke。

当前结论：I0 已通过，E0 首个 baseline smoke 已完成；H 的当前实现观察到零误拒但召回
仅 0.59%，不应扩大性能结论。下一步应先对历史数据库执行零误拒审计，并扩展 H 覆盖
schedule thread-extent contract；shared-memory 失败是第二优先级。

## 2026-09-03：P1 Pruner 草稿模型 + 已训练 XGBoost 两批实验

### 目标与实现边界

本阶段只搭建 Pruner 解析打分草稿并与现有 XGBoost 代价模型组合，GA T2 始终为 off，
没有在本阶段修改 GA。Pruner 参考仓库为 `F:\tvm_tmp\Pruner`，HEAD 为
`0760c3f39d84b31b4b10c4d7971ff402f2c350ae`，许可证为 Apache-2.0。核对的上游文件为：

- `src/auto_scheduler/feature_psa.cc`
- `python/tvm/auto_scheduler/cost_model/psa_model.py`

实现文件：

- `src/s_tir/meta_schedule/draft_model.h`
- `src/s_tir/meta_schedule/draft_model.cc`
- `src/s_tir/meta_schedule/search_strategy/evolutionary_search.cc`
- `tests/python/s_tir/meta_schedule/test_meta_schedule_pruner_draft_model.py`

草稿复制的是 Pruner 官方实现骨架：逐 statement 计算 compute time 和 memory time，
然后使用 `total_latency += max(compute_time, memory_time)`，最终
`draft_score = 1 / total_latency`。需要特别说明，Pruner 论文正文写的是计算与访存延迟
相加，官方仓库实现却取二者最大值；本草稿以可复现官方代码为准，没有把两者混写。

组合公式为：

```text
normalized_draft = draft_score / 当前 population 最大已知 draft_score
final_score = xgb_score * normalized_draft^beta
```

unknown 使用中性乘数 1.0，不删除候选。新增环境开关为：

- `TVM_DRAFT_MODE=0/1/2`：off / shadow / rank。
- `TVM_DRAFT_BETA`：本轮固定 0.2。
- `TVM_DRAFT_EPS`：rank 模式的 epsilon，本轮固定 0.10。
- `TVM_DRAFT_*` 硬件参数：必须显式提供，公式内部不静默写死 `sm_86`。

本轮 profile 为 `rtx3050ti_sm86_explicit_v1`：peak 10752 GFLOPS、显存带宽
192 GB/s、20 SM、每 SM 1 active block、4 SM partitions、warp size 32、
register capacity 128。

旧 `TVM_SOFT_FILTER_MODE` 仍保留为 thread-only proxy，未把它改名冒充 Pruner。

### 编译与测试

真实 C++ DLL 为 `F:\tvm\build-prune\lib\tvm_compiler.dll`：

```text
SHA256 = 8F5ED2B181708C4DAD8C1388C8A3C9FF516ECB016797E73C8CD3CC248EF0CE59
size   = 220260864 bytes
```

新增源文件后执行过 `cmake -S . -B build-prune`，随后完成真实 MSVC/CUDA 编译。
本轮结束前再次运行：

```powershell
cmd.exe /d /s /c 'call "E:\vs\Common7\Tools\VsDevCmd.bat" -arch=x64 `
  -host_arch=x64 && set "TEMP=F:\tvm_tmp" && set "TMP=F:\tvm_tmp" && `
  set "CCACHE_DISABLE=1" && cmake --build build-prune --parallel 12'
```

结果为 `ninja: no work to do.`，确认实验 DLL 已是当前 C++ 源码对应产物，不是 Python
模拟。相关回归测试分两组重跑，结果为 29 passed + 8 passed = 37 passed，覆盖草稿公式、
归一化、unknown 中性、shadow 恒等、rank 重排、候选数量守恒、静态/动态特征、原硬过滤、
VerifyGPUCode 和 tile-size mutator。

### runner 的 1e10 根因

首轮单批 smoke 位于 `experiments/data/v1_1_p1/`。三臂均有 8 条数据库记录，但
`run_secs=[10000000000]`、`finite_records=0`。这批结果只能证明接入、覆盖、开销和重排，
不能评价真实 latency。

实验驱动 `experiments/e1_thread_filter.py` 已增加 measure callback，逐批保留
`BuilderResult.error_msg`、`RunnerResult.error_msg` 和原始 `run_secs`。缩小到 1 trial 后
抓到的错误发生在 runner 加载 `.tar` 模块的二次链接阶段：

```text
clang ... -shared ...
LINK : fatal error LNK1104: 无法打开文件
"F:\tvm_tmp\lnk{...}.tmp"
```

builder 成功，失败点不是候选变异、CUDA kernel 运行或草稿模型。当前受控环境只允许向
`F:\tvm` 写入，而链接器临时目录是 `F:\tvm_tmp`；同一 1-trial 配置改为在沙箱外运行后
得到有限实测 `80.9279 us`。因此本批 `1e10` 的首因是实验进程写权限，不是候选非法。

进程退出时仍可能打印 `PopenPoolExecutor.__del__ / psutil.AccessDenied`。它在后续
16/16 有限实测的三臂中也会出现，故目前证据表明它是 worker 清理告警，不是测量失败
首因。原始失败与成功对照位于：

- `experiments/data/runner_diag/runner_diag0_seed20260903_off_soft-off_draft-off_ga-off/`
- `experiments/data/runner_diag/runner_diag_escalated_seed20260903_off_soft-off_draft-off_ga-off/`

### 两批实验设计

为了保证第二批真正使用已训练代价模型，三臂均显式创建
`XGBModel(num_warmup_samples=0, adaptive_training=False, num_tuning_cores=1)`。
注意：第 1 批开始时 booster 仍不存在，因此第 1 批预测仍为随机分数；第 1 批 8 条有限
实测完成后训练 XGBoost，第 2 批才是“已训练 XGBoost + draft”。每臂最终均记录
`xgb_data_size=16`、`xgb_trained=true`。

固定配置：MatMul 128³、seed 20260903、16 trials、每批 8、population 512、
SampleInit 最少 50、GA 4 代、hard/旧 soft/GA T2 全部关闭、beta 0.2、epsilon 0.10。
由于 `int(8 * 0.10) = 0`，本轮每批最终选择没有随机候选，8 条全部来自排序头部。

三臂使用相同 source identity
`12b849261f8cf01397bc6adb18a894971db32aa0459d7e0c48b5fdea7ae5f760` 和相同 DLL：

- off：`experiments/data/v1_1_p1_trained/v1_1_p1_trained_pair0_seed20260903_off_soft-off_draft-off_ga-off/result.json`
- shadow：`experiments/data/v1_1_p1_trained/v1_1_p1_trained_pair0_seed20260903_off_soft-off_draft-shadow_ga-off/result.json`
- rank：`experiments/data/v1_1_p1_trained/v1_1_p1_trained_pair0_seed20260903_off_soft-off_draft-rank_ga-off/result.json`
- 汇总与逐 trace 分析：`experiments/data/v1_1_p1_trained/draft_quality_pair0.json`

### 运行结果

| 指标 | off | shadow | rank |
|---|---:|---:|---:|
| 有限实测 | 16/16 | 16/16 | 16/16 |
| builder / runner error | 0 / 0 | 0 / 0 | 0 / 0 |
| 搜索时间 | 142 s | 145 s | 167 s |
| tune call | 418.8 s | 424.2 s | 449.1 s |
| generated | 5019 | 5018 | 5042 |
| Verify 失败 | 1379 | 1386 | 1401 |
| VerifyGPUCode 时间 | 101.864 s | 104.303 s | 124.784 s |
| 最佳延迟 | 40.479 us | 40.156 us | 44.491 us |

rank 相对 off：搜索多 25 s，即慢 17.6%，搜索 speedup 为 0.850；tune call 多
30.29 s，即慢 7.2%，端到端 speedup 为 0.933；最佳延迟差 9.91%。因此本轮不能声称
节约时间或提升 best，当前直接乘法融合是负结果。rank 多出的约 22.9 s
`VerifyGPUCode` 时间基本解释了 25 s 搜索退化；draft 解析本身只有约 0.248 s，
并不是主要耗时。

分批实测分布：

| 臂 | batch 0 best / median / mean | batch 1 best / median / mean |
|---|---:|---:|
| off | 40.479 / 177.611 / 601.033 us | 75.599 / 162.738 / 156.610 us |
| shadow | 40.156 / 197.556 / 532.575 us | 51.479 / 104.895 / 141.850 us |
| rank | 44.491 / 84.866 / 128.465 us | 47.106 / 124.930 / 202.902 us |

rank 在冷启动 batch 0 的 top-3/top-5 实测均值为 44.827/60.843 us，off 为
98.646/130.232 us；这说明草稿在 XGBoost 尚未训练时能把一批较稳定的中上候选集中到
测量集合中。但它没有选到约 40 us 的全局 best，且在已训练 batch 1 中 top-3/top-5
均值退化为 74.569/94.714 us。该现象只能作为单 seed 的方向提示，不能扩大为性能结论。

### top-k 变化和 draft 真实性能质量

shadow 对 4228 个候选全部给出已知分数，coverage 100%，statement rows 16912，
解析总时间 248948 us，即约 58.9 us/候选。shadow 的“原 XGBoost top-16 与假想融合
top-16”重合为：

- batch 0（随机 warmup）：21/80 = 26.25%。
- batch 1（已训练 XGBoost）：13/80 = 16.25%。其中 SampleInit 为 13/16，后续四代
  GA population 均为 0/16。

这表示已训练 XGBoost 与当前 draft 在搜索头部强烈冲突。rank 的 batch 0 重合为
27/80，batch 1 为 63/80；后一个高重合不能解释为两个独立模型更一致，因为 population
已经连续受融合分数引导，存在选择反馈。

数据库原始插入顺序显示：off/shadow 的 batch 0 trace 完全相同 8/8，证明 shadow 本身
不改第一批选择；batch 1 因同一 trace 的实测噪声形成不同训练标签，仅重合 4/8。
rank 与 off/shadow 在两个 batch 都是 0/8 重合，说明融合确实显著改变了最终 measured
top-k，不是只改日志。

新增 `experiments/analyze_pruner_draft_results.py`，对三臂所有有限记录重放 trace，使用
同一 C++ 草稿函数重新打分，再计算 Spearman、draft-only top-k 和 oracle regret。
浮点并行归约只观察到 `1e-14～4.5e-13` 的末位变化；离线排名按 9 位小数合并等价分数，
并明确标记 top-k 在分数并列边界处依赖 tie-break。

36 条唯一 trace 的结果：

- draft score 与 inverse latency 的 Spearman 为 0.314，只是弱到中等正相关。
- 36 条候选只有 11 个不同的 draft 分数档位，区分度较粗。
- draft-only top-1 是 347.345 us，而唯一 trace oracle best 是 40.318 us，头部存在严重误排。
- draft-only top-8 与真实 top-8 仅重合 2/8；top-8 内最好为 44.491 us，相对 oracle
  regret 10.35%。

各臂相关性受“该臂先选择了哪些候选”的反馈偏差影响，不能直接互相比优劣；合并唯一
trace 指标更适合说明当前草稿的总体分辨率，但 36 条样本仍然只够做 smoke 诊断。

### 当前结论与进入 GA 前的决定

P1 草稿接入、真实编译、有效 runner、两批已训练 XGBoost 实验和 top-k 审计已经完成。
结果不是正向性能结论：当前 `xgb * normalized_draft^0.2` 会改变搜索方向，但增加 Verify
时间、未节约搜索/端到端时间、未改善 best。Pruner 草稿在冷启动时对候选分布有一定价值，
但分数过粗且头部误排，不能继续简单提高 beta。

在改 GA 前，下一版融合应优先做：

1. 将 draft 作为冷启动 prior，第一批后随 XGBoost 数据量增加而衰减 beta，而不是永久
   等权影响每一代。
2. 使用 population percentile / rank 或对数比值并截断极端值，避免 max-normalization 和
   粗分数档位支配头部。
3. 把 draft 特征交给代价模型学习校准或学习 residual/pairwise relative label，而不是在
   XGBoost 输出后手工相乘。
4. 在 rank 前联动已验证的硬过滤，减少融合搜索带来的额外 VerifyGPUCode 路径；再进入
   GA 方向性变异实验。

GA 尚未在本阶段改动。下一阶段应以本节数据作为 baseline，先实现可衰减/可校准的融合，
然后再比较 GA 优化的独立增益和组合增益。

## 2026-09-03：H2 硬筛改进与 Pruner 草稿组合实验

### 本轮实现

Pruner 草稿模块已经接入 evolutionary search 的代价模型排序路径。实现位于
`src/s_tir/meta_schedule/draft_model.h/.cc`，接入点位于
`src/s_tir/meta_schedule/search_strategy/evolutionary_search.cc`。草稿模型从候选 TIR
提取计算量、全局访存、并行度、warp/寄存器相关 proxy，结合显式提供的硬件 profile
形成解析式分数。rank 模式使用：

```text
normalized_draft = draft_score / population 内最大已知 draft_score
fused_score = xgb_score * normalized_draft^beta
```

本轮 `beta=0.2`、`epsilon=0.10`；无法计算草稿分数时使用中性乘数 1.0，既不删除候选，
也不伪造硬件参数。shadow 只记录排序变化，rank 才实际改变 top-k。

第一模块硬筛从仅检查物理线程上限，扩展为检查 schedule 产生的线程契约：

- 实际线程乘积小于 `meta_schedule_thread_extent_low_inclusive`；
- 实际线程乘积大于 `meta_schedule_thread_extent_high_inclusive`；
- 实际线程乘积大于 target 的 `max_threads_per_block`。

实现位于 `src/s_tir/meta_schedule/constraints.h/.cc`，调用和逐原因计数位于
`src/s_tir/meta_schedule/utils.h`。动态 extent、注解缺失或无法证明的结构仍返回 unknown
并放行。硬筛在正式 `VerifyGPUCode` 前执行，low/high/hardware 三类原因分开记录。

同时试验了 lowering 前声明 shared buffer 大小作为 shared-memory proxy，但它只进入
shadow 统计，未进入 enforce。后续审计证明这个边界非常必要：schedule 后续的 compact/
merge 会显著缩小实际 shared allocation，声明大小不能直接作为合法性结论。

### 编译与回归验证

本轮不是 Python 模拟。当前源码已真实编译并链接到：

```text
F:\tvm\build-prune\lib\tvm_compiler.dll
SHA256 = 1dd5493eb5c0bb95699e8633de3b770cdf77707592bdc3cb96443486e757ac62
```

首次增量编译发现 `GetAnn` 缺少声明，补充 schedule utils 头文件后完成链接。已通过：

| 测试组 | 结果 |
|---|---:|
| thread feasibility 单元测试 | 22 passed |
| off/shadow/enforce 集成测试 | 1 passed |
| Pruner draft 单元测试 | 3 passed |
| VerifyGPUCode postproc 测试 | 8 passed |
| tile-size mutator / GA 投影测试 | 6 passed |
| VerifyGPUCode analysis 测试 | 2 passed |
| 合计 | 42 passed |

边界测试覆盖 31/32、1024/1025、动态或缺失注解放行，以及 shared proxy 仅观察不拒绝。
扩大回归范围时，两个 GA 投影测试仍按旧的 3 参数/3 返回值调用内部硬筛测试接口；已将
它们适配为显式传入 `max_shared_memory_per_block=-1` 并只读取返回数组的前三项。生产代码
未因此修改。最终上述 42 项一次全部通过。

### 历史数据库零误拒审计

审计脚本为 `experiments/audit_verify_database.py`，机器可读结果为
`experiments/data/hard_filter_v2_database_audit.json`。共重放 606 条记录，其中 409 条有
有限实测延迟；去重后是 432 个 schedule，其中 262 个有有限实测。trace replay error 为
0，线程硬筛对有限实测记录的拒绝数为 0，去重后的有限实测误拒也为 0。

shared 声明大小 proxy 则被数据库直接证伪：409 条合法有限实测中会误拒 397 条。因此它
必须继续保持 observation-only，不能因为它命中了部分 Verify shared-memory 失败就开启
enforce。

### H2 shadow 闸门

改进后的硬筛先以 shadow 跑完 16/16 次真实 CUDA 测量。原始目录为：

```text
experiments/data/v1_1_h2_combo/
  v1_1_h2_pair0_seed20260903_shadow_soft-off_draft-off_ga-off/
```

该次运行在落盘前遇到实验脚本旧正则误解析 shared 统计，因此没有 `result.json`；数据库、
任务日志和 scheduler 日志均完整。修复正则后由原始日志重新解析，得到：

```text
generated/checked = 5089
hard TP/FP/FN/TN = 1138/0/281/3670
precision = 100%
mixed-oracle recall = 80.197%
unknown = 0
hard analysis = 0.038173 s
search = 149 s
VerifyGPUCode = 106.264205 s
```

1138 个预测由 low/high/hardware = 1121/17/0 构成。正式 Verify 失败原因为
thread_extent/shared_memory/lowering_exception = 1138/231/50。这里的 recall 是相对整个
VerifyGPUCode 混合失败集合计算的，不应解释为一个独立 thread-only 数据集的召回率；
但对比旧版同口径 0.59%，说明新增线程契约检查确实覆盖了此前漏掉的大量
`thread_extent` 拒绝，同时样本内仍保持 0 个误拒。

shared proxy 的 TP/FP/FN/TN = 231/4858/0/0，再次证明它不能 enforce。

### 四臂真实 CUDA 组合实验

固定配置为 MatMul 128³、seed 20260903、16 trials、每批 8、population 512、GA 4 代、
XGBoost warmup 0、旧 soft=off、GA T2=off。也就是说，本节只测“改进硬筛”和“已接入
Pruner 草稿”的独立及组合效果，没有把后续 GA 改进混入结果。

四臂均加载同一 SHA256 的 DLL，且每臂都有 16/16 条有限实测：

| 臂 | search | tune call | VerifyGPUCode | best | top-3 mean | top-5 mean |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 193 s | 479.64 s | 151.95 s | 49.038 us | 63.096 us | 86.256 us |
| hard | 150 s | 429.90 s | 110.75 s | 37.057 us | 91.530 us | 109.593 us |
| draft | 161 s | 431.69 s | 121.24 s | 33.108 us | 36.303 us | 64.429 us |
| hard + draft | 175 s | 446.06 s | 133.40 s | 34.784 us | 61.513 us | 84.619 us |

hard enforce 在 5037 个候选中提前拒绝 1105 个，low/high/hardware = 1091/14/0，剩余
Verify 失败 272，硬筛分析总成本仅 0.035383 s。组合臂在 5045 个候选中提前拒绝 1107
个，low/high/hardware = 1098/9/0，剩余 Verify 失败 307，硬筛分析 0.036127 s。

draft 单臂 coverage 为 100%，草稿解析总成本 0.250132 s，原 XGBoost 与融合分数的
top-k 重合为 67/160；组合臂分别为 0.249296 s 和 72/160，说明模块真正改变了搜索排序，
其自身计算开销很小。

原始数据和统一汇总位于：

- `experiments/data/v1_1_h2_combo/`：四臂数据库、日志和各臂 `result.json`；
- `experiments/data/v1_1_h2_combo/hard_draft_combo_pair0.json`：统一指标、身份、shadow
  闸门、数据库审计和 trace overlap；
- `experiments/analyze_hard_draft_combo.py`：可重复生成汇总的分析脚本。

baseline 在随后只修改实验日志正则、未改 C++/DLL 后才继续运行其余三臂，因此 baseline
的 source identity 为 `ad71f1c...`，其余三臂为 `6c69a15...`；四臂加载 DLL 的 SHA256
完全相同。汇总 JSON 已保留这一限制，不能把它隐藏成严格相同的源码文本身份。

### 结果判断

本轮可以确认的提升是：硬筛对 Verify 混合失败的覆盖从旧版 0.59% 提高到 80.20%，样本
内 precision 仍为 100%，且历史有限实测数据库中没有观察到误拒；Pruner 草稿单臂在本轮
找到 33.108 us 的 best，top-3/top-5 均值也优于 baseline，是一个积极的单 seed 信号。

目前不能确认硬筛带来端到端加速。关键对照是同一实现的 shadow 搜索 149 s，而 enforce
为 150 s、Verify 时间分别为 106.26 s 和 110.75 s。正式 `VerifyGPUCode` 在昂贵 lowering
前本来就运行 `ThreadExtentChecker`，所以把同一线程契约再向前检查，主要减少失败计数，
没有跳过主要 lowering 成本。baseline 的 193 s 更像运行波动，不能将 193→150 s 全部
归因于硬筛。

组合也尚未显示对 draft 的额外收益：相对 draft 单臂，hard+draft 的 search 慢 8.70%，
tune call 慢 3.33%，best 差 5.06%，top-3/top-5 均值也更差。本轮 draft 的正向结果还与
上一轮直接融合的负结果不一致，因此当前结论只能是“模块已缝合、能改变 top-k、单 seed
出现积极候选质量”，不能宣称稳定提升。

下一步若目标是实质节省 Verify/lowering 时间，应寻找经过 compact/merge 后仍安全的
shared-memory 下界或把检查移到更有收益的位置；当前简单 shared 声明大小 proxy 已被
否定。草稿模型则应先做随 XGBoost 数据量衰减的冷启动 prior、rank/percentile 校准或
relative-label 学习，再与方向性 GA 做独立和组合消融。
