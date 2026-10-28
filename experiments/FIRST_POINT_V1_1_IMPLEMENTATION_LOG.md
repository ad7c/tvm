# 第一点优化 v1.1：实现与实验过程记录

更新日期：2026-09-04

## 1. 目标与本轮边界

第一点优化解决两个问题：初始化阶段不要反复生成线程约束必然拒绝的 decision；GA、数据库等产生完整 decision trace 后，在能够证明不可行时尽早拒绝，避免继续支付 replay、Postproc 和 VerifyGPUCode 成本。

本轮启用：

- Constraint-Aware Sampling（CS）；
- H-trace；
- H-postreplay；
- shared-memory proxy 只观察；
- TVM 原生 EvolutionarySearch 和原生 XGBoost。

本轮关闭：

- soft filter；
- Pruner draft model；
- 定向 GA/T2；
- 相对标签代价模型。

## 2. 代码位置

### 2.1 约束分析与可行域采样

- `src/s_tir/meta_schedule/constraints.h`
- `src/s_tir/meta_schedule/constraints.cc`

实现内容：

- `TraceConstraintPlan`；
- feasible / infeasible / unknown 三态；
- `GetSBlock`、`GetLoops`、`SamplePerfectTile`、`SampleCategorical`、`Split`、`Fuse`、`Bind`、`Annotate` 和 `Unannotate` 的结构解析；
- threadIdx.x/y/z 乘积；
- schedule contract low/high、hardware limit 和 warp size；
- perfect-tile 可行约数组合枚举；
- categorical 概率重归一化；
- unknown 放行；
- shared-memory pre-compaction proxy 观察。

### 2.2 分层硬筛与统计

- `src/s_tir/meta_schedule/utils.h`

实现内容：

- SampleInit 空 decision replay 时约束采样；
- 完整 trace 的 H-trace replay 前检查；
- H-postreplay IR 兜底；
- H-late 对照；
- VerifyGPUCode 主失败原因分类；
- 分阶段候选守恒、计数和微秒级计时。

### 2.3 EvolutionarySearch 与精确 quota

- `src/s_tir/meta_schedule/search_strategy/evolutionary_search.cc`

2026-09-04 新增只用于公平配对实验的开关：

```text
TVM_EVO_EXACT_INIT_POPULATION=0|1
TVM_EVO_INIT_SAMPLE_BATCH_SIZE=<positive integer>
```

开启后，`SampleInitPopulation(num)` 不再只满足默认的 `init_min_unmeasured=50`，而是：

1. 以固定微批次反复采样；
2. 直到收集并只保留精确 `num` 条完整通过 Postproc 的候选；
3. 记录 target、returned、generated、postproc_passed、batch-tail discarded、batches 和 time_us；
4. 两边以完全相同大小的未测种群进入 GA。

这不是性能优化本身，而是实验控制变量。

### 2.4 实验驱动与数据解析

- `experiments/e1_thread_filter.py`

新增参数：

```text
--exact-init-population
--init-sample-batch-size 64
```

结果 JSON 增加 `search.exact_init_population`。源码身份哈希限定到代码、测试和实验驱动文件，避免后一臂因为前一臂新生成的数据目录而得到不同的 source identity。

## 3. 开关

```text
TVM_THREAD_CONSTRAINT_SAMPLE_MODE=off|shadow|enforce
TVM_THREAD_HARD_FILTER_MODE=off|shadow|enforce
TVM_THREAD_HARD_FILTER_POSITION=late|postreplay|trace
TVM_EVO_EXACT_INIT_POPULATION=0|1
TVM_EVO_INIT_SAMPLE_BATCH_SIZE=<N>
```

旧 `TVM_PRUNE_MODE=0|1|2` 仍兼容。

## 4. 编译与验证

构建时必须先加载 Visual Studio x64 环境，否则绝对路径调用 `cl.exe` 仍会因为没有 `INCLUDE` 而找不到 `stdint.h`：

```bat
call E:\vs\VC\Auxiliary\Build\vcvars64.bat
cmake --build F:\tvm\build-prune --parallel
```

结果：

```text
[1/2] Building ... evolutionary_search.cc.obj
[2/2] Linking CXX shared library lib\tvm_compiler.dll
exit code 0
```

编译产物：

```text
F:\tvm\build-prune\lib\tvm_compiler.dll
SHA256 ad3ea50bc94dafac397a21105ab7725568c16b97fb05c88d246b44d518a51ca9
```

回归：

```text
tests/python/s_tir/meta_schedule/test_meta_schedule_trace_constraints.py
5 passed in 65.85s

thread feasibility/filter + draft + mutation related tests
27 passed in 73.22s
```

精确 quota 小验证：

```text
target=16
returned=16
generated=88
postproc_passed=19
discarded=3
batches=11
time=5.904245s
```

## 5. 配对实验过程

有效实验目录：

```text
experiments/data/first_point_v1_1/equal_valid_e2e_20260904_r2/
```

公共参数：

```text
workload=MatMul128
seed=20260904
population=128
exact init population=true
init sample batch size=64
GA generations=4
mutation probability=0.85
trials=16
trials per iteration=8
xgb warmup samples=0
build timeout=120s
soft/draft/GA-T2=off
```

第一批没有历史数据；第一批8条测量完成后训练 XGBoost，第二批搜索实际使用已训练模型。第二次搜索时数据库有8条记录，所以其128条初始种群由8条已测记录和精确120条未测合法候选组成。

Baseline 核心命令：

```powershell
F:\tvm\.venv-tune\python.exe experiments\e1_thread_filter.py run `
  --mode off --constraint-sample-mode off --hard-filter-position trace `
  --soft-mode off --draft-mode off --ga-t2-mode off `
  --seed 20260904 --run-id B_equal128 `
  --trials 16 --trials-per-iter 8 --population-size 128 `
  --genetic-iters 4 --xgb-warmup-samples 0 `
  --exact-init-population --init-sample-batch-size 64 `
  --build-timeout 120 --run-timeout 30
```

CS+H 只改变两项：

```text
--mode enforce
--constraint-sample-mode enforce
```

## 6. 环境失败记录

`equal_valid_e2e_20260904_r1` 在沙箱内运行。搜索和 GA 完成，但 Windows linker 无权在 `F:\tvm_tmp` 创建 `lnk{...}.tmp`，16条 runner 全部失败，`finite=0`。该轮完整保留用于排查，不用于性能或质量结论。

`r2` 在允许 MSVC 临时链接文件和本机 GPU 的环境中重跑：两臂 build errors 均为0、runner errors 均为0、finite measurements 均为16。

进程退出时出现 `psutil.AccessDenied` 清理警告，但发生在结果写回以后，两个进程退出码均为0，数据库、callback 和16条有限测量完整，因此不影响本轮数据。

## 7. 原始数据完整性

两臂完全一致：

```text
git HEAD: 4a1297ad59c9ff80de535a8ba898bf6d9012e5c7
source identity: 9cc25677d833445fcae05898a7156f66a33919050096eb8ab02dc85b91961ebc
DLL SHA256: ad3ea50bc94dafac397a21105ab7725568c16b97fb05c88d246b44d518a51ca9
```

原始文件哈希在：

```text
experiments/data/first_point_v1_1/equal_valid_e2e_20260904_r2/raw_manifest.sha256.json
```

## 8. 当前状态

本轮已证明在 MatMul128、seed 20260904 上，CS+H 能以更少尝试、更短时间凑齐相同合法种群，并在相同原生 GA/XGBoost/测量预算下完成端到端搜索。尚未完成多 seed、跨算子和定向 GA/相对标签代价模型实验。
