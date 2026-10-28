<!--
Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements. See the NOTICE file
distributed with this work for additional information
regarding copyright ownership. The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License. You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied. See the License for the
specific language governing permissions and limitations
under the License.
-->

# 单算子 800 次提交实验交接（2026-10-11）

**13:16 状态更新：本文件对应的第一尝试已因内存分配失败终止。恢复诊断、新目录和当前状态见 [OPERATOR800_RECOVERY_20261011.md](OPERATOR800_RECOVERY_20261011.md)。旧目录的 70 次完整反馈和第 8 批测量不能作为 800 次正式结果。**

用户本轮要求先跑单算子，补充同提交次数和同时间的实测收敛、总用时、首次找到最终最优的时间，并查看持续更新的 XGBoost 训练情况。

## 固定设置

- MatMul128、MatMul256、Conv2D NCHW 3×3、大 Reduction 128×4096，FP32；RTX 3050 Ti Laptop 4GB。
- Baseline（B）、CS+H（C）、CS+H+Draft-GA（D）；一个 seed：20261011。
- 每组最多 800 个**硬件测量提交**，每批 10 个。失败占预算；搜索空间提前耗尽时报告实际数量，不能复制候选补齐。
- 种群 32，初始化最小未测数 16、精确初始化、采样批 32；GA 1 代；eps-greedy=0.1；Draft 入围比例=0.25。
- 每个算子、每种配置维护一个 XGBModel 对象，贯穿整次调优。B/C/D 不共享测量历史。
- 当前 TVM 将新反馈追加到历史特征/代价数据，然后用全部累计数据从头拟合 XGBoost booster；adaptive_training=False，每批有新特征就训练。warmup=0，第一批测量后启用模型评分。
- 不能将持续积累样本等同于在旧 booster 后续追加树。不能保证训练拟合改善就必然带来实测搜索质量改善。

协议：[protocol.json](results/paper_v1/operator800_20261011/protocol.json)。源码、实验脚本、DLL、包版本已经冻结，脚本快照保存在同目录。运行期间不要修改冻结文件。

## 入口与运行状态

- 驱动：[run_operator800_study.py](run_operator800_study.py)。串行执行 smoke / B/C/D / 冻结赢家独立重测 / 单独进程 PyTorch / 报告。
- 状态：[progress.json](results/paper_v1/operator800_20261011/progress.json)；每个子进程的命令、退出码和日志哈希：[execution.jsonl](results/paper_v1/operator800_20261011/execution.jsonl)。
- 调优日志：对应 run ID 的 `*.console.log`；`callback_batches.jsonl` 逐批写入；`timeline.jsonl` 保留阶段边界和原始 XGB 日志。
- S01 是 4 次提交、两批各 2 次，用于实际验证模型训练样本持续累积、第二批先用旧模型验证。其他算子 smoke 各 2 次，不进入正式结果。
- 正式结果只有全部检查通过才进入分析。驱动遇到失败会停下并保存证据；不会静默重试、覆写或拼接中断的搜索。
- 运行状态以 progress / execution 为准，本文件不将已启动写成已完成。全部结束后才生成 `COMPLETION.json`。

12:37 已启动正式驱动，初始 orchestrator PID=10700、首组 M128G1B01 子进程 PID=27144。S01 两批已通过真实 CUDA 检查，4/4 有效；累计训练样本 2→4，第二批更新前 p-RMSE=0.115430（仅验证记录功能，不作为模型效果结论）。S01 进程退出码为 0。正式 12 组尚未完成，当前 PID 和阶段以 progress 为准。

## 指标与输出

报告入口：[report_operator800_study.py](report_operator800_study.py)。每完成一个算子的 B/C/D，自动更新 `papers/figures/operator800_20261011/`：

1. `convergence_submissions`：横轴实际累计提交 0–800，纵轴截至该批的最佳有效实测延迟。
2. `convergence_time`：横轴 Tune 从开始到当前批完成的完整墙钟时间；纵轴同上。准备时间单列。
3. `time_to_target`：到达 baseline 完整搜索最终实测延迟的时间，严格阈值为主，+1%、+5% 保留在 JSON；未达标为 NR。该目标是回顾性基准。
4. `final_performance`：原始数据库赢家固定后独立重建与复测，三轮轮换 B/C/D、每轮 10 个样本，报告 median / IQR。不能根据新计时更换赢家。
5. `stage_time`：Search、Builder、Runner+notify、Callbacks、Residual 互斥耗时；正确性检查计入 Runner，模型更新计入 Callbacks，嵌套预测时间不能重复相加。
6. `candidate_counts`：B/C 生成和 VerifyGPU 拒绝数量；`prediction_cost`：C/D 的 XGB 预测时间与 Draft 评分时间。
7. `training_samples`、`model_update_time`、`training_fit`、`next_batch_prediction`、`next_batch_ranking`：累计样本数、逐批更新耗时、训练拟合 p-RMSE、旧模型对新反馈的 p-RMSE / a-peak@32。

`RESULTS.md` 包括总 Tune 用时、首次找到本组最终最优实测候选的批末时间、到 baseline 目标的时间、准备时间、独立复测延迟及相对 B 的加速比。

`model_training.csv` 逐批保留训练数量、boosting 轮数、best_iteration、拟合与更新前的预测指标。更新计时包含特征提取、验证和训练；它不是单独的 `xgb.train` 时间。新批候选由搜索选出，不能当作独立测试集；训练误差和新批误差要分开解释。

PyTorch 参考：[benchmark_torch_operators.py](benchmark_torch_operators.py)。输入常驻 GPU，FP32、TF32 关闭；每算子四组 FP64 数值参考检查；三个计时轮次共 30 样本。CUDA events 计时包含 Python 发射间隙，Conv2D 使用 PyTorch 输出分配，MatMul/Reduction 复用输出，因此报告其计时边界，与 TVM 独立复测分开解释。

## 环境和继续工作

- 调优使用 `.venv-tune/python.exe` 和 `build-prune/lib/tvm_compiler.dll`。
- TVM/NumPy 子进程沿用排除可选 Torch 导入的 bootstrap；PyTorch 单独子进程恢复 Torch 导入。MKL SEQUENTIAL，CPU / builder / runner 单并发。
- Windows 独立赢家复测使用已有 `--terminal-worker-exit`：只有全部检查和输出完成、CUDA 同步后才退出，避免 native DLL 析构故障；失败不被掩盖。
- 本轮 16 个相关单元检查通过；报告空输入绘图已验证。性能有效性以真实 smoke、正式测量及独立复测为准。
- 本轮不修改论文正文、不覆盖 `single_seed_20261010` 原始结果。正式数据完成后再填论文。
- 如果中断或失败，先查看 stdout / stderr / 子任务日志与状态，确认进程是否仍活着，禁止直接再开一份同预算搜索。
