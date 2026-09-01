# C++ Meta Schedule 完整工作流

> 源码轨迹 + 实测数据，MatMul 128³ fp32，RTX 3050 Laptop

---

## 1. 入口

```python
tune_tir(mod, target, max_trials_global=50, num_trials_per_iter=8)
```

对应的 C++ 调用链：

`python/tvm/s_tir/meta_schedule/tir_integration.py:148`
  → `python/tvm/s_tir/meta_schedule/tune.py:106`
    → `TaskScheduler::Tune()` at `src/s_tir/meta_schedule/task_scheduler.cc:193`

---

## 2. 生成设计空间（一次性）

`PostOrderApply::GenerateDesignSpace()` at `src/s_tir/meta_schedule/space_generator/post_order_apply.cc`

对 MatMul 的 3 个循环 (i, j, k) 应用 `MultiLevelTiling("SSSRRSRS")`：

```
i 循环 (128): 拆成 5 个因子 → 330 种分配
j 循环 (128): 拆成 5 个因子 → 330 种分配
k 循环 (128): 拆成 3 个因子 → 36 种分配

3 个 design space → 3 个 trace 模板（35/39/39 条指令）
```

**产出**：3 个 trace 模板，包含 `SamplePerfectTile`、`CacheRead`、`ComputeAt`、`Bind` 等指令。

---

## 3. 第一步：收集初始种群

`EvolutionarySearch::SampleInitPopulation(50)` at `evolutionary_search.cc:522`

```
while 还没凑够 50 个合法:
    随机选一个 design space
    创建 Trace(insts, {})          ← 空决策，SamplePerfectTile 随机填充
    ThreadedTraceApply::Apply()    ← trace replay + postproc 链
      → 任意 postproc 失败 → 丢弃
      → 全部通过 → 加入种群
```

**日志实测数据**：

```
20:11:11  Initializing Task
20:11:15  TaskScheduler picks task
20:12:12  Sending 8 sample(s) to builder  ← init 完成，凑够 50 个
         ─────────── 57s ───────────
         采样 ~492 次，VerifyGPUCode 拒 442 次 (89.8%)
         最终 50 个合法
```

---

## 4. 第二步：进化搜索

`EvolutionarySearch::EvolveWithCostModel()` at `evolutionary_search.cc:562`

```
每轮:
  1. XGBoost 代价模型预测候选性能 → 选 top-K
  2. MutateTileSize 变异 top-K → 产生子代
  3. Postproc 验证子代 → 保留合法的
  4. 送 8 个去 builder → build 成 CUDA binary
  5. 送 8 个去 runner → GPU 实测
  6. 用测量结果更新 XGBoost
```

**日志实测数据**：

```
轮 1: 20:12:12  8 → builder → 8 valid → runner → XGBoost(9 iters)
轮 2: 20:13:21  8 → builder → 8 valid → runner → XGBoost(9 iters)
轮 3: 20:15:25  8 → builder → 8 valid → runner → XGBoost(9 iters)
轮 4: 20:17:33  8 → builder → 8 valid → runner → XGBoost(9 iters)
轮 5: 20:19:41  8 → builder → 8 valid → runner → XGBoost(9 iters)
轮 6: 20:21:51  8 → builder → 8 valid → runner → XGBoost(9 iters)
轮 7: 20:24:14  2 → builder → 2 valid → runner → finish
```

每轮 ~2 分钟（build 60s + run/update 60s）。7 轮全 0 build error。

---

## 5. Builder：编译 CUDA

`LocalBuilder::Build()` at `builder/local_builder.cc`

```
Trace replay → postproc → TIR lowering → CodeGenCUDA → NVCC → .so 文件
```

popen_worker 子进程执行编译，主进程继续采样。

---

## 6. Runner：GPU 实测

`LocalRunner::Run()` at `runner/local_runner.cc`

```
加载 .so → 分配 GPU 内存 → time_evaluator(10次×5轮) → 平均延迟
```

---

## 7. 最优候选 → 硬件实测

取 database 中代价模型得分最高的记录：

```
Record #0: 0.031ms, 134 GFLOPS, err=0
```

---

## 时间分解（984s 总耗时）

| 阶段 | 时间 | 占比 |
|------|------|------|
| SampleInit (凑 50 个初始) | ~57s | 6% |
| Evolution round 1-6 (build+run) | ~800s | 81% |
| Evolution round 7 (build+run) | ~60s | 6% |
| Overhead (scheduler, XGBoost) | ~67s | 7% |

**瓶颈在 build（每次 60s）和 run（每次 ~60s），不在搜索。** 剪枝省 SampleInit 的 ~57s。
