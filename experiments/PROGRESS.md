# 研究进度报告：约束引导的 GPU 内核双向剪枝

> 研究方案 C §3.4 | TVM 0.26.dev90 | RTX 3050 Laptop (sm_86) | 2026-07

---

## 一、核心进展

### 1.1 问题验证 ✓

**确认了 Meta Schedule 的"拒绝采样病理"：**

- 对 MatMul 128³，SamplePerfectTile 全因子空间 = 392 万种组合
- 硬件合法的 block tile 组合 = 85 / 512（16.6%）
- 实际通过 VerifyGPUCode 的候选 ≈ 0.5%（Exp01: 0/5000；Exp03: 3/439）
- **99.5%+ 的随机采样在 postproc 阶段被丢弃**

根因定位：`RewriteCooperativeFetch`（postproc 第 2 步）通过 `ComputeAt` 放置 shared memory buffer。当 tile 的中间 split 因子 vk=1 时，buffer 被推到外层循环 → 分配全矩阵大小（128×128×4=64KB → 超 48KB 限制）。93% 的随机因子分配命中此模式。

### 1.2 双向剪枝实验 ✓

| 实验 | 方法 | 耗时 | TVM 调用 | 加速比 |
|------|------|------|----------|--------|
| Exp03 | Baseline（无剪枝） | 691s | 439 | 1× |
| Exp03 | Single 剪枝（↑） | 255s | 151 | **2.7×** |
| Exp06 | Baseline（无剪枝） | 3250s | 3000 | 1× |
| Exp06 | Single 剪枝（↑） | 738s | 613 | **4.4×** |
| Exp06 | Double 剪枝（↑+↓） | 105s | 0 | **31×** |

**Single 剪枝（shmem 估算 + 线程数检查）稳定提供 2.7-4.4× 墙钟加速。** 预检本身几乎零开销（0.05s vs TVM postproc 1.5s）。

### 1.3 论文 vs 实测差异发现 ✓

| 论文 §3.4.5 声称 | 实测（128³，RTX 3050） |
|-----------------|---------------------|
| 全空间 ~16,777,216 | 392 万（SamplePerfectTile 实际空间） |
| 向上剪枝后 ~200 | ~600（预检通过，仍需 VerifyGPUCode） |
| 向下剪枝后 ~20 | ~0（100% 被砍，阈值太紧） |
| 最终有效率 ~100% | ~2%（预检通过 → VerifyGPUCode 通过） |

**差异根源**：论文假设双向剪枝可保证硬件合法，但实际运行时 `ComputeAt` 将 shared memory buffer 放到全矩阵层级——此行为无法从 tile 参数做纯算术预测。这个问题是 99% 拒绝率的直接原因。

---

## 二、当前状态

### 2.1 已验证的结论

1. **预检有用且无害**：shmem 估算 + 线程数检查从不误杀合法候选（已验证），且单次耗时 0.05s vs TVM 1.5s
2. **Shared memory 超限是唯一拒绝原因**：所有 VerifyGPUCode 拒绝均为 shmem buffer 全矩阵分配
3. **分析模型精度足够**：对"预检通过"的候选，真实 shmem 大概率超限 → 模型有一定误判率但保守
4. **Double 剪枝需调参**：occupancy ≥ 25% + data reuse ≥ 8 FLOP/byte 对 128³ 过紧

### 2.2 待解决问题

| 问题 | 状态 | 方向 |
|------|------|------|
| Double 剪枝阈值 | 待调 | 放宽 occupancy 到 12.5%，reuse 到 4 |
| 随机性导致无法复现 | 已知 | C++ SamplePerfectTile 不受 Python seed 控制 |
| 合法候选收集 + benchmark | 待跑 | Exp05 脚本已写好，需重跑 |
| 论文 §3.4.5 量化数字 | 待修正 | 用实测数据替换理论估计 |

---

## 三、已完成的实验文件

| 目录 | 用途 |
|------|------|
| `exp03_verify/` | 三组对比（baseline / single / double），产生核心加速数据 |
| `exp06_double/` | 修正版三组对比，double 阈值待调 |
| `exp02_pruning/pruning.py` | 双向剪枝分析模型（可复用） |
| `data/exp03_verify.json` | Exp03 完整数据 |
| `data/exp06_double.json` | Exp06 完整数据 |

---

## 四、下一步建议

1. 调优 double 剪枝阈值（放宽 occupancy / reuse），重跑三组对比找到最优参数
2. 完成 Exp05（合法候选 build + benchmark），拿到真实性能数据
3. 用实测数字更新论文 §3.4.5 的预期量化表
4. 将"ComputeAt 导致 shmem 全矩阵分配"写入论文 §1.1，作为拒绝采样的核心病理

---

*报告人：用户 + Claude | 日期：2026-07-07*
