# ResNet-18 / Scout 实验实时进度

更新时间：2026-10-25T21:47:54+08:00

状态：`paused_by_user`；当前阶段：`formal`；当前组：`B`。

| 正式组 | 实际提交 / 上限 | 最新整网检查点 ms | 阶段 |
|---|---:|---:|---|
| B | 173/2000 | - | paused_by_user |
| C | 0/2000 | - | 尚未开始 |
| D | 0/2000 | - | 尚未开始 |

当前输出：`F:\tvm\experiments\results\e2e_20261025_A03\formal_B`

最近日志：

```text
2026-10-25 21:44:55 [INFO] [task_scheduler.cc:206] Sending 10 sample(s) to builder
```

B = MetaSchedule；C = CS+H；D = 完整 Scout。
检查点延迟是完整模型实测；最终三组还将按轮换顺序独立重测。
此文件每 15 秒更新；完成或异常后停止。
