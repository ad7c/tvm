# GA 优化工作单 v2：Meta Schedule 进化搜索改造
## （发给实现方的任务文档 · 自包含）

> **背景一句话**：Meta Schedule 的进化搜索每轮产生约 81 个非法子代（被 `VerifyGPUCode` 事后扔掉），mutator 选择是均匀随机，劣解直接丢。本工作单改造变异器与选择逻辑，共 4 个独立开关的优化点。
>
> **前置**：仓库分支 `pruning-experiment`（TVM 0.26.dev90），已有硬杀代码在 `src/s_tir/meta_schedule/utils.h`（`ThreadedTraceApply::Apply` 内的拒绝逻辑 + `TVM_PRUNE_MODE` 开关 + 计数器），可直接参考其写法。下文文件路径以仓库实际为准。
>
> **v2 修订**（依据第二轮评审，五处）：①`shmem_uncertain` 改为标定制（原判据会让 shmem 收缩对 MatMul 永久失效）；②影响矩阵补 reduction 行；③G3 奖励改用实测 `run_secs`（原定义是循环论证）；④G4 温度匹配分数尺度；⑤命名精确/下界分离。
>
> **总纪律**：
> 1. 四个优化点**各自一个开关位**，全关必须严格等价于原版行为
> 2. 逻辑先在 Python 侧原型验证再下沉 C++（此前踩过 ArrayNode 编译失败、构建树损坏、中文路径三个坑）
> 3. 每完成一个开关立即 git 提交
> 4. **所有资源上限从 `Target` 读**（`max_shared_memory_per_block`、`max_num_threads`），禁止硬编码 49152 / 1024

---

## 0. 共用基础设施：资源估算函数（先做，G1/G2 都依赖它）

写一个函数，输入 trace（采样完成、replay 之前），输出：

```cpp
struct ResourceEstimate {
    int64_t threads;         // 线程数（精确值，不是下界）
    int64_t shmem_lb;        // 显存下界（乘了 num_stages 的下界，见 Step 0.5）
    bool    shmem_uncertain; // 标定发现无法解释的偏差时置 true（见 Step 0.5），默认 false
};
```

**命名纪律**：`_lb` 后缀只给真下界用。`threads` 是精确值，禁止叫 `threads_lb`——整个正确性论证依赖"哪个精确、哪个是下界"，命名含糊会埋 bug。

**取数规则（禁止写死下标，这是硬性要求）**：

- **threads**：遍历 trace 中的 `Bind` 指令，凡绑到 `threadIdx.*` 的循环，其长度相乘
- **shmem_lb**：遍历 `CacheRead` 指令，确定 cache 挂在哪一级循环；该级以内的 tile 尺寸 × 归约维尺寸 × dtype 字节数 = 单个 buffer；所有 shared buffer 求和；**再乘以 num_stages**（从 trace 的软件流水决策读出，无流水则为 1）
- **shmem_uncertain**：不由 trace 信号决定，由 Step 0.5 的标定结果决定（默认 false）

**正确性论证（写进注释）**：`shmem_lb` 是真实值的下界（"下界 > 上限 ⟹ 真实值必超"），据此收缩/拒绝**不可能误杀**；不确定时放行（交给 postproc）也安全。本函数只可能少拦、不可能错拦。

**放置位置**：与现有硬杀同文件（`utils.h` 或新建 `constraints.{h,cc}`），硬杀逻辑逐步替换为调用本函数（行为保持等价）。

### Step 0.5：标定实验（纯 Python，不重编译，先于一切 C++ 改动）

1. 对 baseline database **全部**合法候选 + 一批随机候选，对比"公式值 × num_stages" vs `CompactBufferAllocation` 实际值
2. 输出三分类：
   - **精确相等** → `shmem_uncertain=false`，公式可直接当精确值用（MatMul 已有 3 个候选实测分毫不差，预期走这支）
   - **差 num_stages 倍** → 倍数乘进 `shmem_lb`，仍按精确处理
   - **其他系统性偏差** → 记录偏差模式，能解释则修公式，不能解释则该算子上 `shmem_uncertain=true`（G2 只缩 threads）
3. 标定结论**按算子记录**，写入配置文件供 C++ 侧读取

---

## 影响矩阵（G1 加权、G2 收缩共用，三行）

| 因子级别 | 影响 shmem | 影响 threads | 识别规则 |
|---|---|---|---|
| block 级 | ✅ | ❌ | 循环绑 `blockIdx.*` |
| thread 级 | ❌ | ✅ | 循环绑 `threadIdx.*` |
| reduction 级 | ✅ | ❌ | 未绑 block/thread、且位于 CacheRead buffer 区域内的归约循环（如 k 维因子） |

统一规则一句话：**extent 落在 CacheRead buffer 区域内的循环因子都影响 shmem**——block 级和 reduction 级都在区域内；thread 级在区域外（各线程拿寄存器切片），只影响 threads。

---

## G1：余量导向的变异方向（开关 bit0）

**现状**：`MutateTileSize` 均匀随机挑两个因子，一个除、一个乘。

**改为**：挑因子时按**资源余量**加权。

```text
余量率_r = (上限_r − 已用_r) / 上限_r          # r ∈ {shmem, threads}，已用取自资源估算函数
因子 f 的权重 w_f = (它影响的资源的余量率)^α     # α 默认 1.0；α=0 时退化为均匀随机（消融用）
按 w_f 轮盘赌抽"被乘大的因子"和"被除小的因子"
```

- 因子→资源的归属按上面的三行影响矩阵（block 级和 reduction 级都归 shmem）
- `shmem_uncertain=true` 时 shmem 余量率按 0 处理（不往放大 shmem 的方向偏；此时只有 threads 方向可被偏置）

**文件**：`src/s_tir/meta_schedule/mutator/mutate_tile_size.cc`

---

## G2：越界缩回（开关 bit1）——优先级最高，先做

**现状**：变异产物不合规则在 replay/postproc 阶段被扔，本次变异浪费。

**改为**：变异完成后立即用资源估算函数检查，越界则**定向二分收缩**，不扔。

```text
mutated = 原 MutateTileSize 逻辑(trace)
est = ResourceEstimate(mutated)
if est.threads > 上限_threads:
    对本次被"乘大"的 thread 级因子二分收缩（步长每次减半），直到 threads ≤ 上限
if est.shmem_lb > 上限_shmem 且 !est.shmem_uncertain:
    对本次被"乘大"的、影响 shmem 的因子（block 级或 reduction 级）二分收缩，直到 shmem_lb ≤ 上限
# shmem_uncertain=true 时不做 shmem 收缩（估算不可靠，交给 postproc 兜底）
```

**铁律**：**缩错方向是本任务唯一的坑**——按影响矩阵对号入座：threads 超限只缩 thread 级；shmem 超限只缩 block 级 / reduction 级。两个都超：先缩 shmem 再缩 threads。

**单调性依据（收敛保证）**：放大某因子 ⇒ 其影响的资源占用不减。二分收缩在离散格点上必然收敛；**设 20 次迭代上限**，极端情况缩到 1 仍超则放弃本次变异、保留父代副本（保证无死循环、无崩溃路径）。

**正确性论证（写进注释）**：收缩仅在"下界 > 上限"时触发 ⟹ 被收缩的候选本来就非法，G2 只可能减少非法子代、不可能误杀合法候选。

**文件**：`src/s_tir/meta_schedule/mutator/mutate_tile_size.cc`

**统计要求**：计数器记录 ①触发收缩次数 ②按资源分类的收缩次数（shmem / threads）③平均二分迭代次数 ④放弃次数（消融和论文用）。

---

## G3：记账本选 mutator（开关 bit2）

**现状**：变异时均匀随机选 mutator。

**改为**：UCB1 + 滑窗。

```text
每个 mutator i 维护最近 W=50 次使用记录（成功=1 / 失败=0）
成功定义：子代被实测，且 run_secs 优于父代的实测 run_secs      ← v2 修订，见下
未被实测的子代：不更新账本
选择：argmax_i [ 成功率_i + C·sqrt(ln(总使用次数) / 使用次数_i) ]，C 默认 1.0
```

**为什么不用代价模型打分定义成功（v2 修订理由，写进注释）**：代价模型分数 ≠ 真实性能（实测 32 样本下 XGBoost RMSE 0.75）。用"让模型高兴"当战绩，等于奖励"会讨好模型"的 mutator 而非"能变快"的 mutator，会放大噪声——循环论证。实测信号稀疏但真实：每轮实测约 8 个候选，几百轮累计足够 5~7 个 mutator 的账本更新。

**两个硬性细节**：

1. **滑窗**：只用最近 W 次，禁用全程平均（早晚期好用的 mutator 不同）
2. **冷启动**：database 中实测样本数 < 32 时强制均匀随机

**可选消融臂**：保留一个编译期/环境变量切换的"模型打分版"奖励（子代模型分 > 父代模型分），E4 对比两种奖励定义——"循环论证是否真的有损"由数据回答。

**文件**：`src/s_tir/meta_schedule/search_strategy/evolutionary_search.cc`（变异循环中选 mutator 的位置 + 实测结果回写战绩的钩子）

**统计要求**：运行结束导出"各 mutator 被选次数随代数变化"（论文用图）。

---

## G4：退火接受劣解（开关 bit3）

**现状**：子代打分不如父代则不进入种群。

**改为**：小概率收留，温度逐代衰减。

```text
d = (子代分 − 父代分) / |父代分|     # v2 修订：必须用相对差，无量纲，量级在 1 附近
p_accept = exp(−d / T)
T = T0 × 0.95^代数                   # T0 默认 0.1；T0=0 退化为原版（消融用）
以 p_accept 概率允许劣解进入种群
```

**尺度铁律（v2 新增，写进注释）**：若 d 用原始延迟（µs 量级），`d/T ≈ 430`、`exp(−430) ≈ 0`，退火形同虚设。**分数定义一变，T0 必须重新标定。**

**标定步骤**：首次运行时打印前一代所有 d 的分布，确认 T0（0.05~0.1）与 d 的中位数同量级；不同量级则调整 T0 后重跑。

**文件**：`src/s_tir/meta_schedule/search_strategy/evolutionary_search.cc`（子代入库/入种群判断处）

---

## 开关总控

仿照现有 `TVM_PRUNE_MODE` 的写法：

```bash
# bit0=G1余量导向  bit1=G2越界缩回  bit2=G3记账本  bit3=G4退火
export TVM_GA_OPT=0      # 全关，必须严格等价原版
export TVM_GA_OPT=2      # 只开 G2
export TVM_GA_OPT=3      # G1+G2
export TVM_GA_OPT=15     # 全开
```

加统一日志行，每次调优结束打印：`GA_OPT: G1=x G2=x G3=x G4=x | shrink_shmem=… shrink_threads=… mutator_picks=…`

---

## 实施顺序（严格按此）

| 步 | 内容 | 完成标志 |
|---|---|---|
| 0 | 资源估算函数（Python 原型先验证） | 对 baseline database 全部合法候选：threads 精确相等；shmem 公式值 ≤ 实际值（下界性抽查 20 个） |
| **0.5** | **标定实验**（纯 Python） | 输出三分类结论；确定每个算子的 `shmem_uncertain` 取值 |
| 1 | G2 越界缩回 | 非法子代计数 **= 0**（标定为精确的算子上 shmem/threads 两类收缩都要有触发记录）；最终性能与基线持平（差异 <2%） |
| 2 | G1 余量导向 | 编译过、行为正确即可（效果由实验评估，不设门槛） |
| 3 | G3 记账本 | 同上 + 战绩导出文件生成 |
| 4 | G4 退火 | 同上 + d 分布标定记录 |

每步完成即 git commit。

---

## 验收实验（对应总方案 E4，由实现方跑完出数据）

| 组 | TVM_GA_OPT | 回答什么 |
|---|---|---|
| A0 | 0 | 基线 |
| A1 | 2 | 光"不产垃圾"有多大用 |
| A2 | 3 | 可行性两件套 |
| A3 | 7 | +记账本 |
| A4 | 15 | 全开 |
| A3' | 7（模型打分版奖励） | 循环论证是否真的有损（可选） |

**指标**（每组 5 次独立运行，报均值±标准差）：

1. 每轮非法子代数（开 G2 后必须为 0）
2. 每代有效样本率
3. 达到基线最终性能所需实测轮数（目标：A4 比 A0 降低 ≥20%）
4. 最终最优性能（各组必须持平，差异 <2%，**变差即失败**）
5. 搜索阶段墙钟（非端到端）

**算子**：先用 MatMul 128³ 跑通全表；之后加 Conv2D、Reduction 各一组 shape（每个新算子先做 Step 0.5 标定）。

---

## 不做清单（防止范围蔓延）

- ❌ 不改交叉（crossover）逻辑
- ❌ 不动种群大小、代数等超参
- ❌ 不引入 LLM / 外部依赖
- ❌ Conv2D 的余量闭式反解不做（Conv2D 上 G1 退化为"只偏置方向"，G2 照常工作）
- ❌ 新颖性去重：先检查仓库现有 dedup 机制再报回来定，不默认做
