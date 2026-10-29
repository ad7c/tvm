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

# ResNet-18 新版 Scout：启动前准备

2026-10-29。用户最新指令：**先准备，等用户通知内存足够后再启动**。
当前未启动端到端调优、GPU 预检或后台等待队列；不得因内存恢复而自行启动。

## 实现

- `scout_feedback.py`：显式控制 GA，复用原有 C++ 约束采样、变异、回放及合法性后处理。
- `model_scout_feedback_study.py`：ResNet-18 两组试验驱动，复用已经验证过的完整 Relax VM 评估和资源保护。
- 新增 C++ 实验接口 `ScoutReplay`、`ScoutMutate`、`ScoutDraftFeatures`；原有搜索入口和草稿公式不变。
- 中途与最终预测复用同一 XGBoost；每批预测缓存只在该批有效。实测反馈后按原 adaptive 策略更新。
- 线性修正按任务名称识别的算子族共享，五项特征；64 个有效不同样本后拟合，至少 16 个后续样本进行顺序验证。
- 验证必须在纳入新标签之前进行；至少 20 对相差超过 5% 的同任务同批候选，排序一致率提高至少 5 个百分点才启用。
- 记录原始/修正评分、模型版本、候选来源、父代关系、中途注入、最终角色、原始重复计时、构建/运行错误。

## 冻结的首轮配置

| 项目 | 设置 |
|---|---|
| 模型 | ResNet-18，固定已有权重和输入，batch 1，1×3×224×224 |
| 路径 | Relax / MetaSchedule，普通 CUDA，FP32，sm_86 |
| 两组 | R1：剪枝＋XGBoost GA 控制实现；R6：新版完整反馈草稿 GA |
| 种群/变异代数 | P=64，G=2；允许合法候选不足时缩小 |
| R6 下一代 | 最多 45 草稿引导变异、6 随机父代变异、7 新采样、6 精英延续 |
| R1 下一代 | 最多 10 精英延续、54 XGBoost 排名引导变异 |
| R6 中途 | 第一代之后，16 高分＋8 多样＋8 随机；注入最多 8 个父代精英 |
| R6 最终池 | 最多 96 草稿高分＋16 多样＋16 随机，K=128 |
| R1 最终池 | 已产生候选由 XGBoost 评分，取最高分的最多 128 个 |
| 每批 GPU | R6：8 优选＋1 分歧挑战＋1 随机；R1：8 优选＋2 随机 |
| 冷启动 | XGBoost warmup=100；R6 中途/挑战关闭，4 草稿高分＋4 多样＋2 随机实测 |
| 未测档案 | 每任务最多 256 个紧凑记录；另保留最多 13 个实测优良父代记录 |
| 搜索界限 | 每批最多 1024 次生成尝试、软时间 300 秒；到限用已有候选进行最终选择 |
| 每组预算 | 最多 400 次提交，包含失败；进程上限 90 分钟，预留 5 分钟收尾 |
| 全轮预算 | 5 小时，含准备、预检、两组调优和评估；单搜索种子，探索性试验 |
| 运行环境 | builder=1，搜索线程=1，XGBoost tuning cores=1；沿用原测量环境 |
| 计时 | 算子 number=3、repeat=5、min_repeat_ms=100；完整模型 warmup=100、repeat=30 |
| 内存 | 启动可用物理内存 ≥4 GiB、可用提交 ≥6 GiB；保留运行时内存保护 |

R1 是参数匹配的**控制实现**，不是原生默认 MetaSchedule。两组共享本次 Python 调度层、原生回放和变异操作。
它用于初筛新版流程是否值得继续；不能将其结果直接写成“超过原生 MetaSchedule”。原生系统基线及各模块消融应另补。

多样性当前使用线程块大小区间和 trace 前三个 tiling/categorical 决策分桶，属于首版近似。
草稿仍保留原有按 buffer 覆盖聚合和访存量近似；新增字节单位的共享内存代理供修正使用，不作为硬拒绝规则。
原草稿 `1/score` 按名义毫秒代理转换为秒来拟合残差，不宣称它是精确物理延迟。
构建文件清理仍使用原有默认回调，两组一致，耗时计入结果；此次没有顺便改动清理策略。

## 启动后的顺序

1. 检查内存和源码/编译库身份，创建全新结果目录。
2. 准备完整模型，验证未调优 GPU 前向正确性与计时稳定性。
3. R6 独立预检最多 10 次实测，再检查应用调优记录后的完整模型正确性。预检历史不进入正式组。
4. 顺序运行 R1、R6，各自全新数据库及代价模型。每 100 次提交在时间允许时做完整模型检查点。
5. 各组最终完整模型校验和计时，再交错顺序重测两组。
6. 比较完整模型延迟、首次实测达到 R1 最终延迟 +5% 的时间、各阶段耗时和模块触发次数。

400 次/90 分钟不保证线性修正激活或所有任务收敛。未触发时明确标记覆盖不足，不据此宣称模块有效或无效。
不得把 30 次计时重复称为 30 个搜索种子；不得把检查点间插值当作实测到达时间。
软限时允许结束当前不可中断的回放；硬超时或内存保护终止的记录保留作诊断，不伪装为完整结果。

## 启动入口（等待用户通知后执行）

```powershell
.\.venv-tune\python.exe -u -X utf8 experiments\model_scout_feedback_study.py run `
  --study experiments\results\resnet18_feedback_20261029_A01 `
  --trials 400 --arm-seconds 5400 --total-seconds 18000 --memory-wait-seconds 0
```

目录已存在时必须另选新名称，不覆盖或续写旧协议的实验。
后台启动需使用隐藏窗口并保存标准输出/错误；当前没有执行此命令。

## 检查边界

构建和单元/原生接口检查用于验证流程实现；原生调用测试使用构造的测量反馈，不运行 GPU kernel。
只有用户允许启动后产生的真实端到端结果才能回答效果问题。
测试日志：`experiments/validation/scout_feedback_20261029/tests.log`。
