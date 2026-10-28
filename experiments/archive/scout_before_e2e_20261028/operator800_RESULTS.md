# 单算子 800 次提交实验

Seed：20261011。已完成 12/12 组。

预算按实际提交计数，构建/运行失败占预算。模型在每组实验内部持续积累历史样本；B/C/D 不共享测量信息。

| 算子 | 方法 | 提交 / 有效 | 最优实测 µs | 独立重测 µs | 重测相对baseline加速 | Tune总用时 s | 首次找到最终最优 s | 到baseline目标 s | 准备 s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| matmul128 | Baseline | 800 / 800 | 27.246 | 29.225 | 1.000 | 8040.721 | 4444.873 | 4444.873 | 113.290 |
| matmul128 | CS+H | 800 / 800 | 27.253 | 28.873 | 1.012 | 8112.554 | 2739.096 | NR | 110.351 |
| matmul128 | CS+H+Draft-GA | 800 / 800 | 27.717 | 31.418 | 0.930 | 7818.674 | 1970.352 | NR | 106.547 |
| matmul256 | CS+H | 800 / 800 | 29.810 | 30.715 | 0.921 | 9035.891 | 6441.839 | NR | 120.630 |
| matmul256 | CS+H+Draft-GA | 800 / 800 | 29.458 | 31.315 | 0.903 | 7954.219 | 2762.219 | NR | 102.536 |
| matmul256 | Baseline | 800 / 800 | 26.424 | 28.288 | 1.000 | 8189.073 | 2233.450 | 2233.450 | 101.427 |
| conv2d_nchw3x3 | CS+H+Draft-GA | 800 / 800 | 29.705 | 29.018 | 0.984 | 8280.249 | 5885.767 | NR | 112.876 |
| conv2d_nchw3x3 | Baseline | 800 / 800 | 29.200 | 28.563 | 1.000 | 8414.000 | 6334.292 | 6334.292 | 112.946 |
| conv2d_nchw3x3 | CS+H | 800 / 800 | 28.245 | 29.383 | 0.972 | 8353.883 | 7201.077 | 3183.891 | 73.117 |
| reduction4096 | Baseline | 85 / 85 | 31.126 | 28.598 | 1.000 | 1031.664 | 207.055 | 207.055 | 115.445 |
| reduction4096 | CS+H | 88 / 88 | 31.788 | 28.562 | 1.001 | 973.140 | 522.791 | NR | 111.140 |
| reduction4096 | CS+H+Draft-GA | 88 / 88 | 35.501 | 32.862 | 0.870 | 906.497 | 193.654 | NR | 110.147 |

首次找到最终最优按实际调优批末时间计算；独立重测不改变候选选择。NR 表示在本组预算内未达到 baseline 最终实测值。

| 算子 | 方法 | 生成数 | VerifyGPU拒绝数 | XGB评分输入 | XGB预测时间 s | Draft评分时间 s | 最终训练样本 | 模型更新合计 s |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| matmul128 | Baseline | 22018 | 17128 | 5184 | 27.915343 | 0.000000 | 800 | 22.434 |
| matmul128 | CS+H | 7436 | 1950 | 5184 | 27.550802 | 0.000000 | 800 | 25.076 |
| matmul128 | CS+H+Draft-GA | 7337 | 1878 | 800 | 4.314189 | 0.926667 | 800 | 22.541 |
| matmul256 | CS+H | 9983 | 4631 | 5184 | 30.941186 | 0.000000 | 800 | 30.582 |
| matmul256 | CS+H+Draft-GA | 9766 | 4513 | 800 | 4.255819 | 0.788715 | 800 | 28.490 |
| matmul256 | Baseline | 25529 | 20602 | 5184 | 28.239925 | 0.000000 | 800 | 28.428 |
| conv2d_nchw3x3 | CS+H+Draft-GA | 7354 | 1896 | 800 | 16.845031 | 1.718431 | 800 | 37.844 |
| conv2d_nchw3x3 | Baseline | 19841 | 14890 | 5184 | 107.515957 | 0.000000 | 800 | 40.204 |
| conv2d_nchw3x3 | CS+H | 7463 | 1944 | 5248 | 99.786674 | 0.000000 | 800 | 42.743 |
| reduction4096 | Baseline | 760 | 0 | 1088 | 1.413595 | 0.000000 | 85 | 3.814 |
| reduction4096 | CS+H | 759 | 0 | 1088 | 1.319204 | 0.000000 | 88 | 3.454 |
| reduction4096 | CS+H+Draft-GA | 732 | 0 | 88 | 0.113852 | 0.041023 | 88 | 3.296 |

| 算子 | PyTorch参考 µs | P25 / P75 µs |
|---|---:|---:|
| matmul128 | 13.831 | 12.879 / 14.982 |
| matmul256 | 16.305 | 15.576 / 17.232 |
| conv2d_nchw3x3 | 25.585 | 24.236 / 28.562 |
| reduction4096 | 10.605 | 10.484 / 11.043 |

训练样本数不一定等于有效测量数：构建失败不进入模型，具有特征的运行失败可作为高代价反馈。

`model_training.csv` 保留每批数据量、更新耗时、训练拟合与更新前的新批预测指标。训练拟合不能单独证明泛化能力；新批候选也受搜索选择影响。

待完成：无

口径说明：

- One search seed; resumed arms carry explicit history and continuation provenance.
- Failures consume submission budget; interrupted attempts are retained but excluded.
- Convergence uses actual measurements at feedback-batch completion, without interpolation.
- Targets use the baseline's observed final latency, evaluated retrospectively.
- Independent retiming freezes the original winner and uses 30 technical samples.
- Training fit is separate from pre-update predictions on selected new-batch feedback.
- UpdateCostModel includes feature extraction, validation, and training; it is inside Callbacks.
- PyTorch CUDA-event timing includes Python launch gaps and Conv2D output allocation.
- CONVG1C01: verified history continuation; active Tune joins historical and new segments. State reconstruction and pause are separate; native internal prediction timings include reconstruction.
- REDG1B01: valid early termination below the 800-submission upper limit; actual count retained.
- REDG1C01: valid early termination below the 800-submission upper limit; actual count retained.
- REDG1D01: valid early termination below the 800-submission upper limit; actual count retained.
