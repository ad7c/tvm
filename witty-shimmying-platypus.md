# 2 硬杀 + 6 软罚：约束引导剪枝研究计划（修订版 v2）

## Context

复用 Pruner（ASPLOS'25）的「硬件感知符号 + 经验公式」，把候选验证从"事后拒绝"前移为"生成后立即硬杀非法 + 软罚低性能"。这是对原「双向剪枝」的修正落地：

- **向上剪枝（硬杀）**：判据从被证伪的 `vk≠1` 换成 **从 schedule 结构机械提取的 shmem/threads**（零误杀）。
- **向下剪枝（软罚）**：从"硬拒"软化成 **Pruner 的 6 个 penalty 项降分**（性能差不是非法）。

**v2 关键修订**（按 review）：硬杀判据**不再写死下标**（`i[2]*j[2]` 那种算子指纹），改用通用的 `ExtractStructuralFeatures` 算法——从 trace 的 Bind / CacheRead / ComputeAt 结构推导，算子无关。这正是主方案 T1 的核心函数，一份劳动同时喂 T1（硬杀）、T2（变异预算）、T3（软罚特征）。

## 方案骨架

```
① 采样 split 因子（生成候选）
   ↓
② 【硬杀 → 软罚】← 生成之后、replay/postproc 之前
   硬杀：ExtractStructuralFeatures 算 shmem/threads，超限 → return nullopt（零误杀）
   软罚：6 个 penalty 算性能分（降分不拒，进代价模型特征）
   ↓
③ trace replay → postproc（VerifyGPUCode）→ build → run
```

## 核心：ExtractStructuralFeatures（通用，替代硬编码公式）

```cpp
struct StructuralFeatures {
    int64_t shmem_bytes;   // 所有 scope="shared" 的 CacheRead buffer 的 tile 尺寸之和
    int64_t threads;       // blockDim（threadIdx.* 各维乘积）
    // 软罚所需的其余符号（S1~S8）可同函数一并提取
};
StructuralFeatures ExtractStructuralFeatures(const Schedule& sch, const Target& target);
```

**通用推导（不写死下标）**：

- **threads**：遍历 trace 的 `Bind` 指令，把所有绑到 `threadIdx.x/y/z` 的循环 extent 相乘。
- **shmem**：遍历 trace 的 `CacheRead(scope="shared")` + `ComputeAt`，对每个 shared buffer：看它 ComputeAt 在哪一级，该级之上的空间 loop extent（**排除绑到 blockIdx 的 grid 因子**）× 该级之内的 reduction loop extent，就是 buffer tile 尺寸，求和 × dtype 字节数。

> 原理与 `CompactBufferAllocation` 一致（都是"buffer 铺在哪个循环层 → tile 尺寸"），只是从 trace 结构机械读，而非跑 lowering。shmem 这半边的推导需要 loop 类型（spatial/reduction）信息，可从 `GetSBlockVarTypes` / `GetLoopIntExtent` 取，实现难度高于 threads，但这是通用性的命门，不能省。

**零误杀原理**：`shmem_bytes` 是下界（未乘 num_stages），只当"下界 > 上限"才拒 → 被拒候选真实值必超 → VerifyGPUCode 本来也拒 → 不误杀。

## 硬杀 2 条（上限从 Target 读，不硬编码）

| 约束 | 上限来源 | 判据 |
|---|---|---|
| shared memory | `target.max_shared_memory_per_block` | `shmem_bytes > 上限` → 拒 |
| threads | `target.max_num_threads`（或 `max_threads_per_block`） | `threads > 上限` → 拒 |

## 软罚 6 项（Pruner 原版，降分不拒）

| 惩罚项 | 公式 | 信号 |
|---|---|---|
| P_l0,m | `min(m_l0/S1, 1)` | 寄存器容量 |
| P_l0,c | `1 + S2/S1` | 算存比 |
| P_l1,m | `min(m_l1/S3, 1)` | 共享内存容量（软） |
| P_l1,c | `sch/(⌈sch/pu⌉·pu)` | warp 占用率 |
| P_l2,c | `S6/(⌈S6/pu⌉·pu)` | SM 占用率 |
| P_l2,m | `S7/(⌈S7/n⌉·n)` | 访存事务对齐 |

**特征流向（v2 修订）**：6 项 penalty **不在生成时算好随车携带**，而是**挪进代价模型的特征提取器**，从 replay 后的 IR（或 trace 决策）算，作为 XGBoost 的额外特征。这与 T1b 统一，避免动数据结构。硬件参数（m_l0/m_l1/pu_l1/pu_l2/n_l1/n_l2/T_p/T_m）从 `Target` 读或按 sm_86 先填表。

## 指标（v2 修正）

- **零误杀**：baseline database 里**全部**合法候选（几百个，不是 8 个）过 `ExtractStructuralFeatures`，shmem/threads 一个都不能超限。
- **召回**（前移拦截接住多少垃圾）＝ **硬杀拦下的非法候选 ÷ VerifyGPUCode 总共拒掉的候选**。
  - 注意区分：precision（拦下的里多少真是非法）在零误杀下恒 100%，不测它。
  - 预期：乘 num_stages 前召回偏低，乘完上升；**剩余缺口是 hoisting**（cooperative fetch 把 buffer 提出循环，真实 shmem 更大），下界公式结构上抓不到，只能 replay 后读真实 IR（= T1b 的职责）。
- **加速**：搜索阶段耗时（非端到端），5 次取均值±方差。

## 代码改动

### Step 1（核心）：实现 ExtractStructuralFeatures + 硬杀

- 文件：新增 `src/s_tir/meta_schedule/constraints.{h,cc}`（或在 utils.h 内）
- `ExtractStructuralFeatures(sch, target)`：从 trace 的 Bind/CacheRead/ComputeAt 推导 threads + shmem。
- `ThreadedTraceApply::Apply` 里把 `RejectByReductionTile`（vk≠1）替换为：调 `ExtractStructuralFeatures`，shmem/threads 超限 → 计 prune 数 → return nullopt。

### Step 2：Python 验证零误杀（样本量加大）

- 用 baseline database 全部合法候选过公式，确认零误杀。
- 生成一批随机 tile 因子，统计召回（拦下多少会爆 shmem/threads 的）。

### Step 3：软罚进代价模型

- 6 项 penalty 在特征提取器算，喂 XGBoost。

### Step 4（补召回）：num_stages

- 读 software_pipeline 决策，shmem × num_stages。

### Step 5（补召回边界）：T1b

- replay 后读真实 IR，抓 hoisting 造成的额外 shmem（下界公式的缺口）。

## 实施顺序

1. **Step 0** Python 原型（已跑，8 候选零误杀；需加大到全部合法候选）。
2. **Step 1** 实现 `ExtractStructuralFeatures`（先 threads 后 shmem），重建 tvm_compiler。
3. **Step 2** 验证零误杀（全部合法候选）+ 召回（随机非法候选）。
4. **Step 3** 软罚 6 项进特征提取器。
5. **Step 4/5** num_stages + T1b 补召回。

## 验证

- 零误杀：全部合法候选无一被拒。
- 召回：拦下非法 ÷ VerifyGPUCode 总拒数（越高越好，乘 num_stages 后应显著上升）。
- 加速：搜索阶段耗时下降，5 次均值±方差。
- 软罚：代价模型 Spearman ρ 在 32/64/128 样本下的提升。

## 关键 caveat

- `ExtractStructuralFeatures` 的 shmem 推导需要 loop 类型信息，是通用性命门，不能省（不写死 SSSRRSRS 下标）。
- num_stages 未乘时零误杀但召回不全；乘完剩余缺口是 hoisting，靠 T1b（replay 后读真实 IR）。
- 硬杀上限按当前静态共享路径（`max_shared_memory_per_block`）；"扩大到 99KB 的 shared.dyn"是另一条未走通的路。
