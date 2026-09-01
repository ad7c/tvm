# 验证计划：为什么全是 0 合法

## 已知事实

| 实验 | remove_postproc | 合法 |
|------|----------------|------|
| Exp03 baseline | False | 3/439 ✓ |
| Exp03 pruning | False | 3/674 ✓ |
| auto_experiment baseline | True | 0/5000 ✗ |
| auto_experiment single | True | 0/5000 ✗ |
| run_experiment.py (当前) | False | 正在跑 |

## 根因假设

`remove_postproc=True` 时，Trace replay 去掉 design space 的 postproc 决策 → `enter_postproc()` 重新跑 postproc → `RewriteCooperativeFetch` 创建全新的 shared memory buffer → buffer 挂在错误层级 → 100% 被 VerifyGPUCode 拒。

`remove_postproc=False` 时，复用 design space 里的 postproc 决策 → buffer 可能放对 → 少量通过。

## 验证步骤

### 第1步：确认当前脚本正在产生合法候选
等终端里 `run_experiment.py` 跑出第一个 PASS——如果能出，说明方法没问题，继续等。

### 第2步：手动验证——拿 design space 的原始 trace 重放
SpaceGenerator 生成 design space 时 trace 自带正确决策。直接重放应该 100% 通过：

```python
# 不创建新 Trace，直接用 design space 的原始 trace
sch = Schedule(Matmul128, debug_mask='all')
ds[0].trace.apply_to_schedule(sch, remove_postproc=False)
sch.enter_postproc()
# 应该全部 PASS——因为 design space 生成时就验证过了
```

如果这都不通过 → TVM 环境有问题。

### 第3步：如果 1+2 都过不了
检查是否 TVM 源码被修改（git diff），或者 build/ 目录的二进制过期。
