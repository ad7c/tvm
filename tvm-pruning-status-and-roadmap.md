# TVM 搜索空间剪枝：现状、问题与路线图

> 更新日期：2026-08-21 ｜ 分支 `pruning-experiment` ｜ TVM `0.26.dev90`（commit `ddfec9c3d`）
> 本文是工作状态文档，非学术论文。标注 ✅ 表示已实测验证，⚠ 表示推断待核实。

---

## 0. 摘要（TL;DR）

- 目标：给 TVM Meta Schedule 的搜索空间加「约束引导剪枝」，减少无效候选的验证开销（方案 C §3.4）。
- 现状：**剪枝代码第一次真正生效**——已编译进 `tvm_compiler.dll`，实测 `Pruning rejections: 183`，端到端加速 **~8%**。
- 但当前实现是最小可用版：**单算子专用约束（`vk≠1`）、事后拒绝、启发式定位、单次观测**，离论文设想的「通用约束 + 约束采样」还有很大距离。
- 最大未决风险：**剪枝是否误杀合法候选尚未验证**（存活候选 60→50）。

关键数字（本次实测，单次运行，MatMul 128³，sm_86）：

| 指标 | baseline | 剪枝 | 变化 |
|---|---|---|---|
| Pruning rejections | 0 | 183 | 生效 |
| VerifyGPUCode 失败 | 721 | 623 | −13.6% |
| 存活候选 | 60 | 50 | −16.7% ⚠ |
| 端到端墙钟 | 227s | 209s | −7.9% |

---

## 1. 环境现状（✅ 已实测）

| 项 | 值 |
|---|---|
| GPU | RTX 3050 **Ti** Laptop，4 GB，sm_86（注：PROGRESS.md 误写为 3050） |
| CUDA | 12.6.85 |
| LLVM | 17.0.6（`D:/miniconda/Library`） |
| MSVC | 14.44（VS 2026，需 `-allow-unsupported-compiler`） |
| 可用 Python | `D:/conda_envs/tvm-build-venv`（3.11.15，tvm_ffi 0.1.13.dev1）✅ |
| 默认 Python | `D:/miniconda`（3.13，tvm_ffi 0.1.10）❌ import 失败 |
| 有效构建树 | `f:/tvm/build-prune`（本次新建，USE_LLVM=ON）✅ |
| 无效构建树 | `build/`（对象文件被 VS reconfigure 清空）、`build/py3-none-win_amd64/`（USE_LLVM=OFF） |

---

## 2. 已解决的问题（本次会话，✅）

| # | 问题 | 根因 | 修复 | 状态 |
|---|---|---|---|---|
| 1 | 剪枝从未生效 | 代码在 `tvm_compiler.dll`，但只重编过 `tvm_runtime.dll` | 重编 `tvm_compiler` | ✅ |
| 2 | 剪枝代码编译不过 | `ArrayNode` 类型不存在，应为 `ffi::ArrayObj` | 改类型 + 走 `support::AsVector` | ✅ |
| 3 | 无法区分剪枝与 replay 失败 | 共用一个计数器 | 新增 `prune_fail_counter_` + 日志单独打印 | ✅ |
| 4 | baseline 与剪枝要分别编译 | 硬编码开关 | 加 `TVM_PRUNE_MODE` 环境变量开关 | ✅ |
| 5 | CUDA configure 失败 | 用户名含中文 → `TEMP` 路径非 ASCII → nvcc 报 `Access is denied` | `TEMP=TMP=F:\tvm_tmp` | ✅ |
| 6 | CUDA 12.6 不支持 VS2026 | 超出支持范围 | `-allow-unsupported-compiler` | ✅ |
| 7 | LLVM 依赖 zlib 找不到 | 缺 CMAKE_PREFIX_PATH | `-DCMAKE_PREFIX_PATH=D:/miniconda/Library` | ✅ |

**可复现构建配方**（写入本文供后续使用）：

```bat
call E:\vs\VC\Auxiliary\Build\vcvars64.bat
set TEMP=F:\tvm_tmp
set TMP=F:\tvm_tmp
cmake -G Ninja -S F:/tvm -B F:/tvm/build-prune ^
      -DCMAKE_BUILD_TYPE=RelWithDebInfo ^
      -DCMAKE_EXPORT_COMPILE_COMMANDS=TRUE ^
      -DCMAKE_CUDA_FLAGS=-allow-unsupported-compiler ^
      -DCMAKE_PREFIX_PATH=D:/miniconda/Library
```

---

## 3. 当前问题清单

### 3.1 正确性（必须最先证伪，否则所有优化都在放大偏差）

- **P1 误杀未验证**：存活候选 60→50，未知是随机性还是误杀。若 baseline 中存在 `vk==1` 且通过全部 postproc 的候选，则 `vk≠1` 硬约束不成立，需降级为软惩罚。
  - 验证成本低：从 baseline database 提取合法候选的 trace 决策，统计 `vk==1` 比例。

### 3.2 通用性（当前约束是单算子专用的）

- **P2 约束是算子指纹**：`vk≠1` 绑死 MatMul + `SSSRRSRS` 模板 + `RewriteCooperativeFetch` 行为，对 Conv2D（多个 3 因子归约轴）、`add_rfactor`（n=2）、Tensor Core（`SamplePartitionedTile`）全部失效。
- **P3 通用约束缺失**：可移植的 `shmem 估算` / `线程数` / `buffer 层级` 三条主约束都没进 C++。这是与论文「跨算子通用」叙事之间的最大落差。

### 3.3 效率（结构性，调参救不了）

- **P4 召回率低**：初始种群阶段 `vk≠1` 只拦下 102/462 = **22%**，而 Python 解析模型预测 93%（高估 4 倍）。
- **P5 检查点过晚**：拒绝时 35 条 trace 指令已全部执行，只省 postproc。三个 `SamplePerfectTile` 都在前 6 条指令内。
- **P6 阿姆达尔上限 30%**：`max_trials=8` 下搜索仅占端到端 30%，其余是 build+run。当前 8% 加速离天花板有距离。
- **P7 仍是拒绝采样**：未做到约束采样（从可行域直接生成）。

### 3.4 测量学（数据还不能写论文）

- **P8 单次运行无方差**：C++ `SamplePerfectTile` 不受 Python seed 控制，227s vs 209s 不能下结论。
- **P9 指标被 build/run 稀释**：端到端墙钟 70% 是编译+实测，主指标应为「搜索阶段耗时」或「找到 N 个合法候选耗时」。
- **P10 计数器无分解**：只有总数，无法做 Exp-3a 消融（每条约束各拦多少 / 按 init vs evolve 分解 / 误杀多少）。

### 3.5 代价模型 + GA

- **P11 代价模型近乎无信息**：XGBoost RMSE 0.75，样本仅 32 个。
- **P12 GA 大量产垃圾**：183 次剪枝中 81 次发生在进化阶段，`MutateTileSize` 持续产生非法子代。

### 3.6 工程与文献

- **P13 零提交、无版本控制**：所有改动未提交，`pruning-experiment` 与 `main` 同 commit，误操作无法恢复。
- **P14 文档与事实脱节**：PROGRESS.md 的 2.7×/4.4×/31× 来自「valid=0」的运行，需清理。
- **P15 文献未核实**：Roller（OSDI'22）、Heron（ASPLOS'23）与本文方向强相关，但检索工具故障（见 §7），引用未核实。
- **P16 papers/ 为空**：`references.bib`、`download_failed.log` 均空模板。

---

## 4. 下一步：方案与取舍

按依赖顺序分阶段。每阶段列出可选方案与推荐。

### 阶段 0：误杀检验（地基，先做）

| 方案 | 说明 | 成本 | 推荐 |
|---|---|---|---|
| 0-A 提取 baseline 合法候选决策，统计 vk==1 比例 | 证伪/证实 P1 | 几分钟，不重跑调优 | ✅ |

**产出**：若 vk==1 合法候选数 >0，则硬约束降级；若 =0，保守性命题获得实证证据。

### 阶段 1：通用约束框架（对应 P2/P3）

| 方案 | 说明 | 成本 | 推荐 |
|---|---|---|---|
| 1-A 只做 shmem+threads 通用约束 | 快，但覆盖不完整 | 低 | 可选起步 |
| 1-B 封装 `ExtractFeasibilityFeatures(sch)` → {shmem_bytes, threads, buffer_scope_ok} | 一鱼三吃：约束/特征/可行域判定，是架构地基 | 中 | ✅ |
| 1-C 直接测 buffer 层级替代 vk≠1 | 把算子指纹换成结构量，一举解决通用性 | 中 | ✅ 与 1-B 合并 |

**核心设计**：
- 约束作用于 schedule 树结构，不作用于某个因子的取值。
- pre-check 用上界估计（保守，不误杀），verify 读真实值（校准用）。
- 接口：`Constraint::Check(sch, target) -> {reject, reason, estimate}`，每条约束独立计数器。

### 阶段 2：位掩码 + 计数器分解（支撑消融实验）

| 方案 | 说明 | 成本 | 推荐 |
|---|---|---|---|
| 2-A 把 `TVM_PRUNE_MODE` 改成位掩码 | bit0=vk≠1, bit1=shmem, bit2=threads, bit3=buffer层级 | 低 | ✅ |

**产出**：Exp-3a「硬约束贡献分解」一个二进制跑完，无需重编译。

### 阶段 3：重测取方差（补方法论窟窿）

| 方案 | 说明 | 成本 | 推荐 |
|---|---|---|---|
| 3-A 每组各跑 5 次，报均值±标准差 | 判断 8% 真伪 | ~40 分钟 | ✅ |
| 3-B 换指标为「搜索阶段耗时」/「找到 N 合法候选耗时」 | 避开 build/run 稀释 | 低 | ✅ 与 3-A 合并 |

### 阶段 4：检查点前移（对应 P5）

| 方案 | 说明 | 成本 | 推荐 |
|---|---|---|---|
| 4-A 给 trace 加「决策产生后」回调，采样完 6 条指令即判 | 省 83% replay | 中 | ✅ |
| 4-B 保持现状 | 无收益 | — | 否 |

### 阶段 5：约束保持的 GA + 代价模型先验（对应 P11/P12）

| 方案 | 说明 | 成本 | 推荐 |
|---|---|---|---|
| 5-A 变异算子只在上界内移动，越界投影回可行集 | 消掉 81 次/run 垃圾 | 中 | ✅ |
| 5-B 硬件对齐的变异步长 | 缩小搜索空间 | 低 | 可选 |
| 5-C 代价模型 = roofline 上界 + 学习残差 | 解决样本稀少导致的模型发散 | 中 | ✅ |
| 5-D 特征从因子值改为结构量（shmem/threads/occupancy） | 与约束共享计算层 | 中 | ✅ 与 1-B 复用 |
| 5-E 主动采样（不确定性/UCB） | 验证贵时收益低 | 高 | 推迟 |

**耦合点**：约束做得好 → GA 不产垃圾 → 代价模型不用学非法区 → 同样样本更准。优化顺序必须是「约束 → GA → 代价模型」，不能反过来。

### 阶段 6：约束采样（论文核心贡献，最重）

| 方案 | 说明 | 成本 | 推荐 |
|---|---|---|---|
| 6-A 从可行域直接采样 tile 三元组 | 拒绝采样 → 约束采样 | 高 | 论文纵深所在 |
| 6-B 修复 postproc 而非拒绝（让 vk==1 也可用） | 扩大可行域，另一叙事线 | 高 | 可作第二技术点 |

---

## 5. 需要的代码改动（文件级）

### 已改（本次会话，未提交）

| 文件 | 改动 |
|---|---|
| `src/s_tir/meta_schedule/utils.h` | ① 加 `<cstdlib>`；② 剪枝块改为 `PruneMode()>=1 && RejectByReductionTile()`；③ 日志加 `Pruning rejections`；④ 新增 `PruneMode()`/`RejectByReductionTile()`/`PruneFailCount()`；⑤ 新增 `prune_fail_counter_` |

### 待改（按阶段）

| 文件 | 改动 | 对应阶段 |
|---|---|---|
| `src/s_tir/meta_schedule/utils.h` | 位掩码开关、通用 shmem/threads 约束、独立计数器分解 | 阶段 1/2 |
| 新增 `src/s_tir/meta_schedule/constraints.{h,cc}` | `ExtractFeasibilityFeatures` + `Constraint` 接口 + 可行域枚举 | 阶段 1/6 |
| `src/s_tir/meta_schedule/search_strategy/evolutionary_search.cc` | 消费约束计数、接入代价模型先验 | 阶段 2/5 |
| `src/s_tir/meta_schedule/mutator/mutate_tile_size.cc` | 约束保持的变异算子、硬件对齐步长 | 阶段 5 |
| `src/s_tir/meta_schedule/schedule_rule/multi_level_tiling.cc` | 联合采样 / 采样器钩子（约束采样） | 阶段 6 |
| `src/s_tir/meta_schedule/postproc/rewrite_cooperative_fetch.cc` | （可选）修复 vk==1 的 buffer 放置 | 阶段 6-B |
| `experiments/*.py` | ① `TVM_LIBRARY_PATH` 改指 `build-prune/lib`；② 补 `PYTHONPATH`；③ 强制 ASCII TEMP；④ 锁解释器 `tvm-build-venv`；⑤ 指标改为搜索阶段/找到 N 合法耗时 | 阶段 3 |

---

## 6. 本次实测数据（✅ 单次，MatMul 128³）

分阶段：

| 阶段 | baseline prune/verify | 剪枝 prune/verify |
|---|---|---|
| SampleInit | 0 / 452 | 102 / 360 |
| Evolve#0 | 0 / 64 | 17 / 84 |
| Evolve#1 | 0 / 69 | 22 / 62 |
| Evolve#2 | 0 / 69 | 18 / 55 |
| Evolve#3 | 0 / 67 | 24 / 62 |
| **合计** | 0 / 721 | 183 / 623 |

解读：
- 初始种群阶段总拒绝数几乎不变（452 vs 102+360=462），但 102 个被廉价预检拦下，`VerifyGPUCode` 调用降 20.4%。
- 进化阶段 81 次剪枝暴露 `MutateTileSize` 产生非法子代的问题（P12）。
- `Trace replay failures` 两组均为 0 → 独立计数器干净。

---

## 7. 风险与待核实事项

- ⚠ **P1 误杀**：存活候选 60→50 未查清，是后续所有优化的前置条件。
- ⚠ **文献 Roller / Heron**：检索工具故障（WebSearch 模型名多空格、arxiv MCP 未注册进子会话），引用未核实。定位撞车风险高，需尽快读原文。
- ⚠ **TVM API 细节**：本文「约束作用于 schedule 树」的取法（`sch.Get`、block `scope`、`StmtSRef` 找循环）落地前需在 `src/s_tir` 内验证精确签名。
- ⚠ **多算子无基线数据**：`exp4_multi_op.json` 显示 MatMul/Reduction/Elemwise 有效率均 0%，通用约束落地后需先补 baseline 失败归因。

---

## 8. 环境修复备忘

- WebSearch 400：`C:\Users\徐奎\.claude\settings.json` 中 4 个模型名 `"deepseek-v4-pro [1M]"` 多空格，应为 `"deepseek-v4-pro[1M]"`；改完需重启 Claude Code。
- arxiv MCP：已安装、能启动，但本会话为子会话（`CLAUDE_CODE_CHILD_SESSION=1`）未继承。
- 非 ASCII 环境反复出问题（nvcc TEMP、权限路径乱码、GBK 日志），长期建议固定工具链临时目录到 ASCII 路径。
