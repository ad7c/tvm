# TVM 学习指南

> 基于 TVM 0.26.dev90（S-TIR 版本），GPU 后端（CUDA sm_86, RTX 3050）

---

## 目录

1. [GPU 硬件基础](#1-gpu-硬件基础)
2. [TVM 架构总览](#2-tvm-架构总览)
3. [IR 层：S-TIR 写算子](#3-ir-层s-tir-写算子)
4. [Schedule：给循环做变换](#4-schedule给循环做变换)
5. [Meta Schedule：自动搜索最优调度](#5-meta-schedule自动搜索最优调度)
6. [Postproc：验证与修复](#6-postproc验证与修复)
7. [Build & Run：编译到 GPU 执行](#7-build--run编译到-gpu-执行)
8. [动手实验指南](#8-动手实验指南)

---

## 1. GPU 硬件基础

### 1.1 为什么要懂硬件？

不是写驱动，是知道**限制条件**。搜索空间的所有约束都来自硬件。

### 1.2 关键概念

```
一块 GPU（如 RTX 3050 Laptop）
│
├── 16 个 SM（Streaming Multiprocessor）
│   每个 SM 里：
│   ├── 共享内存（Shared Memory）：48KB / block ──── 极快（~1TB/s），但极小
│   ├── 寄存器：65536 个 / SM，255 个 / thread
│   └── 可同时跑 1536 个线程
│
├── Warp = 32 个线程一组（硬件调度基本单位）
│   一个 warp 内所有线程同时执行同一条指令（SIMT）
│
└── 全局内存（VRAM）：4GB ──── 慢（~200GB/s），但大
```

**核心矛盾**：共享内存很快但极小，全局内存很大但很慢。

### 1.3 分块（Tiling）——一切优化的基础

```
朴素 MatMul：每个线程直接从全局内存读数据
  → 每个元素被读一次，大量重复访问全局内存

分块 MatMul：
  1. 把矩阵切成小 tile（如 32×32）
  2. 每个 block 先把自己要用的 tile 从全局内存拷到共享内存
  3. 线程们从共享内存读数据做计算
  → 每个全局内存 byte 被复用 (tile_size / sizeof(float)) 次
```

### 1.4 Occupancy（占用率）

一个 SM 能同时跑多少个 warp，受限于：
- 线程数：`blocks × threads_per_block ≤ 1536`
- 共享内存：`blocks × shmem_per_block ≤ 100KB`
- 寄存器：`blocks × threads_per_block × regs_per_thread ≤ 65536`

三者取最小值。Occupancy 太低 = SM 算力浪费。

### 1.5 GPU 线程层级

```
Kernel（整个计算）
├── Grid = Block(0,0) + Block(0,1) + ...
│
├── Block(0,0) ── 在一个 SM 上运行，共享 shared memory
│   ├── Warp 0: Thread 0..31  (SIMT，同时执行)
│   ├── Warp 1: Thread 32..63
│   └── ...
│
└── Block(1,0) ── 在另一个 SM 上运行
```

**代码中的体现**：
```python
blockIdx.x   ← 第几个 block
threadIdx.x  ← block 里的第几个线程
vthread.x    ← 虚拟线程（TIR 概念，用于流水线）
```

### 1.6 关键约束速查表（RTX 3050 sm_86）

| 约束 | 值 | 代码中怎么查 |
|------|-----|-------------|
| Shared memory / block | 48KB (49152B) | `target.attrs['max_shared_memory_per_block']` |
| Threads / block | 1024 | `target.attrs['max_threads_per_block']` |
| Threads / SM | 1536 | `cudaDeviceGetAttribute` |
| Warp size | 32 | `target.attrs['thread_warp_size']` |
| Registers / thread | 255 | — |
| Registers / SM | 65536 | — |
| Shared memory / SM | 100KB | — |

---

## 2. TVM 架构总览

### 2.1 TVM 的"全栈"结构

```
用户代码（Python）
    │
    ▼
┌──────────────────────┐
│  S-TIR（结构化 TIR）  │  ← 算子定义 + 调度变换
│  - 写循环嵌套         │
│  - Schedule 做变换     │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  TIR（低级中间表示）   │  ← lowering 后的 IR
│  - 显式的线程绑定      │
│  - 显式的内存分配      │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  CodeGen（代码生成）   │  ← TIR → CUDA C / LLVM
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  GPU Binary           │  ← 可在 GPU 上执行的 .so/.cubin
└──────────────────────┘
```

### 2.2 关键目录

```
tvm/
├── python/tvm/
│   ├── s_tir/                     ← ** S-TIR（新 TIR）**
│   │   ├── meta_schedule/         ← 自动调优
│   │   │   ├── schedule_rule/     ← 调度规则（MultiLevelTiling 等）
│   │   │   ├── space_generator/   ← 搜索空间生成
│   │   │   ├── postproc/          ← 后处理验证
│   │   │   ├── search_strategy/   ← 搜索策略（进化算法）
│   │   │   └── mutator/           ← 变异器
│   │   └── schedule/              ← Schedule API
│   ├── tirx/                      ← TIR 扩展类型
│   ├── relax/                     ← Relax（模型级 IR）
│   └── script/                    ← TVMScript parser
│
├── src/s_tir/                     ← ** C++ 实现**
│   ├── meta_schedule/
│   │   ├── schedule_rule/
│   │   │   └── multi_level_tiling.cc  ← MultiLevelTiling 核心逻辑
│   │   ├── postproc/
│   │   │   └── verify_gpu_code.cc     ← VerifyGPUCode 检查逻辑
│   │   └── mutator/
│   │       └── mutate_tile_size.cc    ← 变异器
│   └── schedule/
│       └── primitive/
│           └── sampling.cc            ← SamplePerfectTile 采样
│
└── src/backend/cuda/              ← CUDA 代码生成
```

### 2.3 多种使用层次

| 层次 | API | 搜索？ | 性能 | 使用场景 |
|------|-----|--------|------|----------|
| L0: 直接 build | `tvm.build(mod, target)` | 无 | 最差 | — |
| L1: TOPI | `topi.nn.conv2d(...)` | 无 | 基线 | 日常推理 |
| L2: schedule 手写 | `sch.split(...); sch.bind(...)` | 无 | 好 | 专家调优 |
| L3: Meta Schedule | `tune_tir(...)` | 进化搜索 | 最优 | 研究/极致性能 |

---

## 3. IR 层：S-TIR 写算子

### 3.1 最简单的 MatMul

```python
# 文件: experiments/common.py  (make_matmul_module)
from tvm.script import tirx as T

@tvm.script.ir_module
class Matmul:
    @T.prim_func(s_tir=True)
    def main(a: T.handle, b: T.handle, c: T.handle) -> None:
        T.func_attr({'global_symbol': 'main'})
        A = T.match_buffer(a, (256, 256), 'float32')
        B = T.match_buffer(b, (256, 256), 'float32')
        C = T.match_buffer(c, (256, 256), 'float32')

        for i, j, k in T.grid(256, 256, 256):
            with T.sblock('matmul'):
                vi, vj, vk = T.axis.remap('SSR', [i, j, k])
                with T.init():
                    C[vi, vj] = T.float32(0)
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]
```

**关键语法**：
- `T.grid(M, N, K)`：嵌套循环，等价于 `for i in range(M): for j in range(N): for k in range(K):`
- `T.sblock('name')`：**S-Block（Structured Block）**，调度变换的基本单元
- `T.axis.remap('SSR', ...)`：标注循环类型
  - `S` = Spatial（数据并行，i, j 是输出维度）
  - `R` = Reduction（归约，k 是累加维度）
- `T.init()`：初始化代码块
- `T.match_buffer`：绑定函数参数到指定 shape/dtype 的 buffer

### 3.2 为什么是 SSR？

```
C[i][j] += A[i][k] * B[k][j]
  ↑  ↑       ↑  ↑      ↑  ↑
  S  S       S  R      R  S
```

- i, j 只出现在 C 的索引中 → Spatial
- k 出现在 A 和 B 的索引中，会被累加 → Reduction

### 3.3 S-TIR vs 旧 TIR

```
旧 TIR（tir）:        S-TIR（s_tir）:
for i in range(256):  for i, j, k in T.grid(256,256,256):
  for j in range(256):  with T.sblock('matmul'):
    for k in range(256):  vi,vj,vk = T.axis.remap('SSR', [i,j,k])
      C[i,j] += ...       C[vi,vj] += A[vi,vk] * B[vk,vj]
```

S-TIR 的优势：调度变换可以自动理解循环的语义（哪个是 spatial，哪个是 reduction），不需要手动标注。

---

## 4. Schedule：给循环做变换

### 4.1 核心操作

```python
from tvm.s_tir.schedule import Schedule

sch = Schedule(Matmul)            # 拿到 schedule 对象
block = sch.get_sblock('matmul')  # 获取要变换的 block
loops = sch.get_loops(block)      # 获取 block 的循环

# 操作 1: Split —— 把一个循环拆成多个
factors = sch.sample_perfect_tile(loops[0], n=5)  # 随机拆成 5 份
splits = sch.split(loops[0], factors)

# 操作 2: Fuse —— 把多个循环合并
fused = sch.fuse([splits[0], other_splits[0]])

# 操作 3: Bind —— 绑定到 GPU 线程
sch.bind(fused, 'blockIdx.x')

# 操作 4: CacheRead/Write —— 插入共享内存/本地内存
cache = sch.cache_read(block, 0, 'shared')  # 把 A 缓存到 shared memory
sch.compute_at(cache, some_loop)            # 决定缓存放在哪个循环层
```

### 4.2 MultiLevelTiling 做了什么？

**文件**：`src/s_tir/meta_schedule/schedule_rule/multi_level_tiling.cc`

```
structure = "SSSRRSRS"
tile_binds = [blockIdx.x, vthread.x, threadIdx.x]

意思是：
  每个 spatial 循环被拆成 5 份  (因为 structure 有 5 个 S)
  每个 reduction 循环被拆成 3 份 (因为 structure 有 3 个 R)
  前 3 个 level 绑定到 GPU 线程：blockIdx.x, vthread.x, threadIdx.x
  剩下的 level 是普通串行循环
```

**实际效果**（假设 256 的 i 循环）：

```
i: split(5 份) → (bi, vi, ti, si1, si2)
                    │    │    │    │     │
                    │    │    │    └─────┴── 普通串行循环
                    │    │    └── 绑定到 threadIdx.x
                    │    └── 绑定到 vthread.x
                    └── 绑定到 blockIdx.x

bi × vi × ti × si1 × si2 = 256
```

**文件**：`src/s_tir/schedule/primitive/sampling.cc` — `SamplePerfectTile`

```cpp
// 把 extent 随机分解为 n 个因子
// 例如: SamplePerfectTile(256, 5) → (4, 2, 8, 4, 1)
// 算法: 质因数分解 → 随机分配质因数幂次
```

### 4.3 Trace 机制

每次 Schedule 变换操作（split, fuse, bind...）都会被记录为一条 **Trace 指令**：

```python
trace = sch.trace
# trace.insts[4]: SamplePerfectTile(loop=l2, n=5, max_innermost=64)
# trace.decisions[4]: (2, 2, 64, 1, 1)  ← 实际采样到的因子
```

**关键**：Trace 可以**重放**（replay）。重放时，`SamplePerfectTile` 会重新随机采样（如果 decisions 没被固定）。

```python
# 重放到一个新的 schedule 上
new_sch = Schedule(Matmul)
trace.apply_to_schedule(new_sch)  # SamplePerfectTile 被重新随机采样
```

这就是搜索空间的工作原理：同一组指令，不同的随机参数。

---

## 5. Meta Schedule：自动搜索最优调度

### 5.1 组件架构

```
tune_tir(mod, target, strategy="evolutionary")
    │
    ├── SpaceGenerator（搜索空间）
    │   └── PostOrderApply: 遍历 AST，对每个 block 应用 Schedule Rules
    │       └── MultiLevelTiling: 生成 tiling trace
    │
    ├── SearchStrategy（搜索策略）
    │   └── EvolutionarySearch:
    │       1. SampleInitPopulation: 随机采样 N 个候选
    │       2. EvolveWithCostModel: XGBoost 代价模型指导进化
    │       3. MutateTileSize: 变异 tile 大小
    │
    ├── Postproc（后处理）
    │   ├── DisallowDynamicLoop
    │   ├── RewriteCooperativeFetch  ← 插入 shared memory 预取
    │   ├── RewriteUnboundBlock
    │   ├── RewriteParallelVectorizeUnroll
    │   ├── RewriteReductionBlock
    │   └── VerifyGPUCode ← 硬件合法性检查
    │
    ├── Builder → 编译为 GPU binary
    │
    └── Runner → GPU 上实际测量
```

### 5.2 进化搜索流程

**文件**：`src/s_tir/meta_schedule/search_strategy/evolutionary_search.cc`

```
1. SampleInitPopulation (50 个合法初始候选)
   对每个 trial:
     a. 随机采样 tile 参数
     b. 重放 trace → 生成 schedule
     c. 跑 postproc → 合法？
     d. 不合法 → 再采样（最多 max_fail_count 次）
     ≈ 每生成 1 个合法候选需要 ~100 次采样（1% 有效率）

2. EvolveWithCostModel (4 次遗传迭代)
   每轮:
     a. 选 8 个配置送 builder + runner 测量
     b. 用测量结果更新 XGBoost 代价模型
     c. MutateTileSize 变异 → 跑 postproc → XGBoost 预测
     d. 选最好的 512 个保留

3. PickWithEpsGreedy (epsilon-greedy 最终选择)
```

### 5.3 SamplePerfectTile 的采样空间

对于 MatMul 256³：

```
256 = 2⁸  → 约数: {1, 2, 4, 8, 16, 32, 64, 128, 256}
5 个因子，乘积 = 256

总组合数 ≈ 9⁵ = 59049（枚举所有可能的因子分配）
实际上很多因子的组合很少——比如 256 = 256×1×1×1×1 只有一种排列方式。

有效的 block tile 组合（约数枚举）:
  bi, bj, bk ∈ {1, 2, 4, ..., 256}
  = 9³ = 729 种
```

### 5.4 变异器

**文件**：`src/s_tir/meta_schedule/mutator/mutate_tile_size.cc`

```cpp
// 核心操作: 把因子从一个 split "搬" 到另一个 split
int x, y;                        // 随机选两个 tile factor 位置
tiles[x] /= divide_factor;       // 一个缩小
tiles[y] *= divide_factor;       // 另一个放大
// 乘积不变，但分布改变
```

**例子**：
```
变异前: i = (4, 2, 8, 4, 1)  → bi=4, ti=8
变异后: i = (8, 2, 4, 4, 1)  → bi=8, ti=4   (把因子 2 从 split 3 搬到 split 0)
```

### 5.5 代价模型

**文件**：`src/s_tir/meta_schedule/measure_callback/update_cost_model.cc`

XGBoost 模型，输入 ~200 维特征向量（PerStoreFeature），输出预测的运行时间。

```
特征: tile sizes, vectorization width, unroll factor, pipeline stage...
输出: log(runtime)
```

---

## 6. Postproc：验证与修复

### 6.1 Postproc 链

**文件**：`src/s_tir/meta_schedule/postproc/postproc.cc` (line 79)

```cpp
// Default CUDA postproc chain:
1. DisallowDynamicLoop     // 拒绝动态循环
2. RewriteCooperativeFetch // 插入 shared memory 预取 + 向量化加载
3. RewriteUnboundBlock     // 给未绑定的 block 添加 thread_blocks
4. RewriteParallelVectorizeUnroll // 并行化 + 向量化 + 展开
5. RewriteReductionBlock   // 重写 reduction block 结构
6. VerifyGPUCode           // 硬件约束检查（最后一道关）
7. RewriteTensorize        // Tensor Core 替换（费时，放在最后）
```

### 6.2 VerifyGPUCode 检查什么

**文件**：`src/s_tir/analysis/verify_gpu_code.cc`

```
1. max_shared_memory_per_block  ← 所有 shared scope 的 alloc 总和
2. max_local_memory_per_block   ← 所有 local scope 的 alloc 总和（reg spill）
3. max_threads_per_block        ← threadIdx.x × threadIdx.y × threadIdx.z
4. max_thread_x/y/z             ← 每个维度的线程上限
5. max_vthread                  ← 虚拟线程上限 (8)
6. max_vector_bytes             ← 向量化最大字节 (16)
7. max_kernels                  ← kernel 启动数量
```

**为什么 99% 的候选在这一步被拒？** 因为随机采样的 tile 参数几乎总是导致 shared memory 或 local memory 超限。

### 6.3 Shared memory 超限的根因

当 block tile 的某个维度太小（比如 `bj=1`），`CacheRead` 创建的 shared memory buffer 会被 `ComputeAt` 放到循环外层，导致分配大小为全矩阵而非 tile：

```
tile factors: bi=32, bj=1, bk=1
  → A_shared buffer 被 ComputeAt 到 bj 外层
  → A_shared = alloc(256×256, shared) = 256KB → 超 48KB 限制
```

**这是搜索效率低的根本原因**：SamplePerfectTile 随机产生不合理的小 tile，导致后处理必然失败。

---

## 7. Build & Run：编译到 GPU 执行

### 7.1 编译流程

```python
# 1. 从 Schedule 构建
rt_mod = tvm.build(sch.mod, target=target)

# 2. 分配 GPU 内存
dev = tvm.cuda(0)
a_gpu = tvm.runtime.ndarray.array(np_data, dev)

# 3. 执行并计时
evaluator = rt_mod.time_evaluator(
    rt_mod.entry_name, dev,
    number=10,      # 每次测量跑 10 遍取平均
    repeat=3        # 重复 3 次
)
result = evaluator(a_gpu, b_gpu, c_gpu)
print(f'Latency: {result.mean * 1000:.2f} ms')
```

### 7.2 Lowering 过程

```
S-TIR (high-level, with sblocks)
    │
    ▼ LowerSBlock: sblock → opaque block
    ▼ LowerOpaqueBlock: opaque → explicit loops
    ▼ FlattenBuffer: 简化 buffer 访问
    ▼ LowerIntrin: 替换 intrinsic 调用
    ▼ StorageRewrite: 合并内存分配
    │
    ▼
TIR (low-level, thread-bound, alloc'd)
    │
    ▼ CodeGenCUDA: TIR → CUDA C 代码
    │
    ▼ NVCC / ptxas: CUDA C → GPU binary
```

---

## 8. 动手实验指南

### 8.1 实验目录结构

```
experiments/
├── README.md
├── common.py              ← 共享工具（GPU spec, MatMul 定义, postproc 封装）
├── data/                  ← 实验结果 JSON
├── figures/               ← 图表 PNG
│
├── exp01_baseline/        ← 实验 1：全流程 demo
│   └── run.py
│
├── exp02_pruning/         ← 实验 2：双向剪枝
│   ├── model.py           ← 分析模型
│   ├── verify.py          ← TVM 验证
│   └── compare.py         ← 对比图表
│
└── check_tvm.py           ← TVM 功能检查
```

### 8.2 必读源码（按学习顺序）

| # | 文件 | 为什么读 |
|---|------|----------|
| 1 | `python/tvm/s_tir/meta_schedule/schedule_rule/multi_level_tiling.py` | 理解 search space 是怎么定义的 |
| 2 | `src/s_tir/meta_schedule/schedule_rule/multi_level_tiling.cc:200-292` | `TileLoopNest` — 核心 tiling 逻辑 |
| 3 | `src/s_tir/schedule/primitive/sampling.cc:212-310` | `SamplePerfectTile` — tile 采样算法 |
| 4 | `src/s_tir/meta_schedule/postproc/verify_gpu_code.cc` | VerifyGPUCode — 理解所有硬件约束 |
| 5 | `src/s_tir/meta_schedule/search_strategy/evolutionary_search.cc` | 进化搜索策略 |
| 6 | `src/s_tir/meta_schedule/mutator/mutate_tile_size.cc` | 变异器实现 |
| 7 | `src/s_tir/analysis/verify_gpu_code.cc:44-330` | 底层 GPU 约束检查 |
| 8 | `src/backend/cuda/codegen/codegen_cuda.cc` | CUDA 代码生成 |

### 8.3 跑代码

```bash
# 激活环境
source /d/miniconda/etc/profile.d/conda.sh
conda activate tvm-build-venv
cd F:/tvm

# 1. TVM 功能检查
python experiments/check_tvm.py

# 2. 分析模型（瞬时）
python experiments/exp02_pruning/model.py

# 3. TVM 验证（~几分钟）
python experiments/exp02_pruning/verify.py

# 4. 对比图表
python experiments/exp02_pruning/compare.py
```

### 8.4 调试技巧

```python
# 打印 schedule 的 trace 决策
dm = ds[0].trace.decisions
for key, val in dm.items():
    print(f'{key}: {val}')

# 打印 lowered TIR（看实际的 shared memory 分配）
print(sch.mod.script())

# 打印 trace JSON（可修改决策后重放）
json_obj = trace.as_json()
# json_obj[0] = instructions
# json_obj[1] = decisions (可修改)

# 重放修改后的 trace
from tvm.s_tir.schedule._ffi_api import TraceApplyJSONToSchedule
TraceApplyJSONToSchedule(modified_json, sch)
```

### 8.5 环境问题速查

| 问题 | 解决方法 |
|------|----------|
| `Path.home()` 报错 | `os.environ['USERPROFILE'] = 'C:\\tvm_home'` (before imports) |
| `module 'tvm' has no attribute 'nd'` | 用 `tvm.runtime.ndarray.array()` |
| `@tvm.script.ir_module` in `python -c` 失败 | 必须写在 .py 文件中 |
| `popen_worker` 找不到 tvm | `export PYTHONPATH=F:/tvm/python` |
| RewriteUnboundBlock InternalError | 跳过此 postproc（已知 bug） |

---

## 附录：关键术语对照

| 术语 | 英文 | 含义 |
|------|------|------|
| 分块 | Tiling | 把大循环拆成小块 |
| 调度 | Schedule | 循环变换的组合（split/fuse/bind/cache） |
| 设计空间 | Design Space | 所有可能调度参数的集合 |
| 跟踪 | Trace | 调度变换的指令序列 + 决策 |
| 后处理 | Postproc | 验证 + 修复 schedule 的步骤 |
| 变异 | Mutation | 改变 tile 大小但保持乘积不变 |
| 代价模型 | Cost Model | 预测 kernel 运行时间的 ML 模型 |
| S块 | S-Block | S-TIR 的调度基本单元 |
| 占用率 | Occupancy | SM 上活跃 warp 的比例 |
| 寄存器溢出 | Register Spill | 寄存器不够用，溢出到 local memory |

---

*生成时间: 2026-07-06 | TVM 0.26.dev90*
