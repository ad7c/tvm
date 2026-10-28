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

# 2026-10-08 收工记录与下次执行计划

归档时间：2026-10-08 21:30左右（Asia/Shanghai）。用户要求“今天到这里”，不再启动实验。
本说明补充[实验计划v1.6](EXPERIMENT_PLAN_20261008.md)第11节，不覆盖历史结果或冻结协议。

## 1. 研究主线不变

下一步回答：**在什么条件下，CS/H和Draft-GA能用相同搜索资源更早找到好程序？**

- B：原始baseline；C：CS+H；D：CS+H+现有Draft-GA。
- CS/H看“少做了多少无效工作、实际节省多少时间”，不能只看拒绝条数。
- Draft-GA看“GA内少用XGB后，评分成本和搜索质量如何权衡”。当前子代仍经过Verify/lowering，
  不能宣称Draft已经省去全部lowering。
- 借鉴Pruner的收敛、成本拆分、消融及适用条件实验组织；明确Draft来源，
  不预设正结果，也不只选有利算子或seed。
- 流水线故障诊断已有阶段结论，转为补充材料，不再挤占主线实验。

## 2. 今天已完成与未完成

### 已完成

1. 阶段时间戳、逐批账本、启动身份记录和数值检查已经接入实验驱动；新开关默认关闭。
   主线两臂同时打开`--record-timeline --check-correctness`，未修改生产CS/H/Draft/GA算法。
2. 7项unit/mock测试通过；真实CUDA冒烟S02完成4/4实测，4/4候选通过数值检查（共16个输入检查）。
   完整tune为274.521810秒，最优35.673162微秒；这只证明流程可用，不证明方法加速。
   S01启动错误及修复记录保留。
3. 已冻结MatMul128开发批协议：32 submitted、4/批、population32、init16、GA1、3seed。
   协议及代码/库/workload身份见`results/paper_v1/p3_budget32/protocol.json`。
4. 此前完成的流水线反事实与一个恢复候选的真实CUDA正确性检查保留；没有把它计为新的调优对照。

### B11：中断的baseline，不是完成的32提交实验

原始目录：
`results/paper_v1/p3_budget32/B11_matmul128_seed20261011_off_sample-off_hpos-trace_soft-off_draft-off_dga-off_ga-off_g3-off_g4-off/`。
控制台日志：`results/paper_v1/p3_budget32/B11.console.log`。

| 项目 | 核实结果 |
|---|---:|
| 计划提交 | 32 |
| builder_begin累计提交 | 24（6批） |
| 完成回调/数据库记录 | 20（5批） |
| 完成候选数值检查 | 20/20通过，共80个输入检查 |
| 前5批builder/runner错误 | 0/0 |
| 未完成结果的已提交候选 | 4；不能擅自记作4条已确认builder失败 |
| 最优已测延迟 | 30.882331微秒；尚未独立重测 |
| result.json / tune_end | 均无 |

第1批结果在tune第144.501300秒可用，最优37.049285微秒；第2批在255.208503秒可用，
累计最优降到30.882331微秒。第3/4/5批可用时刻分别为360.265226/466.085189/610.238343秒，
累计最优未再降低。第6批在658.726103秒发往builder后没有完成记录。
这些是中断前的真实前缀，不是完整运行时间，也不能作为B/C/D收益对照。

20:58后日志记录`WinError 1455`（页面文件不足）：子进程导入`tvm_ffi`时加载`torch_python.dll`失败。
系统内存提交压力及子进程生命周期需排查；现有证据不能归因为CS/H算法错误或GPU显存不足。
安装的`tvm_ffi/__init__.py`有主动`import torch`路径，不能假定仅设置
`TVM_FFI_DISABLE_TORCH_C_DLPACK=1`就会跳过该导入。

21:29核查：主Python PID428及runner父进程PID31924已不在；原执行会话也已失效。
`run_state.json`仍写着`tuning`，这是未正常收尾的旧状态，不表示仍在持续调优。
保留原状态、数据库和日志；另附中断说明，不补造`result.json`。

**残留进程尚未清理成功：** 只读进程清单确认17个Python子进程的父PID为31924，
创建时间20:49–20:57，与本轮一致。尝试结束这些明确归属的进程，即使请求沙箱外执行仍全部
`Access is denied`；复查仍存在。未结束其他Python进程，不能声称机器已经完全空闲。
下次实验前需由有权限的会话处理；若今天需要释放占用，可先保存其他工作，再由用户通过管理员工具
核对进程身份后结束。不要仅凭本页旧PID直接操作（PID可能复用）。本次记录的PID为：
`26076,22652,33372,3272,23064,33888,28824,220,11940,5872,10220,1212,31736,11888,29672,21964,35544`。

C11及后续C/B/D均未启动；没有新的完整多seed主线、收益图或32提交结论。

## 3. 下次按这个顺序做

| 顺序 | 工作 | 要得到的证据/完成条件 |
|---|---|---|
| 0 | 处理残留进程与内存问题 | 核对内存提交/进程生命周期、torch导入路径；最小修正后小冒烟。未经许可不改页面文件或依赖、不结束无关程序 |
| 1 | 重开完整MatMul128 B/C配对 | seed20261011/12/13，各32提交；顺序B11→C11→C12→B12→B13→C13，使用新run目录，不覆盖本次失败 |
| 2 | 加现有Draft-GA的D组三seed | 同配置，verify_ratio=0.25，G1/G2/G3/G4关闭；无需等待新Delta创新完成 |
| 3 | 搜索强度条件实验 | 同32提交、population32/init16，比较GA1与GA4下B/C/D；先开发seed，再同矩阵补另两seed，不能只扩有利臂 |
| 4 | 多算子/shape | MatMul256、标准Conv2D、大Reduction；先补各算子数值检查并冻结定义，再做同预算B/C/D。Elementwise/Fusion保留为小空间边界组 |
| 5 | 独立重测与论文图 | 每组top-1独立重测；完成submitted轴/完整tune轴收敛、预定达标时间、阶段耗时、GA强度及多算子图 |

环境或驱动发生变化时，保留旧协议，新建协议身份；B/C/D使用同一新环境匹配重跑。
不把本次20条前缀拼进重跑结果，不把旧版本时间与新版本直接混合平均。
GPU实验串行，期间不并发CPU诊断、另一个调优或论文编译。

### 统一比较口径

- 同submitted：失败也占预算，不补抽到相同成功数；32条、4/批等于8批，不是32个GA generation。
- 完整tune时间：包含search、builder、runner、callbacks和残余间隙；进程启动/准备另列。
  数值检查在runner段，XGB更新属于callbacks；嵌套子项不重复相加。
- 预定MatMul128描述性目标40/50微秒；结果以整批可用时刻计，未达标保留。
  300/600/900秒是固定提交轨迹上的预定观察点，不是已实现墙钟停止器。
  中断run不能当成正常完成后无限延用best的run；中断之后标为无观测。
- 报告各seed及波动、不完整运行、失败成本；不用一条最优值宣称稳定提升。
- 优势条件重点看基线无效候选成本占比、XGB评分成本占比、GA强度；builder主导时收益小也是边界结果。

新Delta/父子改进打分暂列离线支线：先查与同父组absolute排序是否等价，再做同配额random/absolute对照；
没有额外信息就不扩大GPU实验。G3继续不做。CS扩展只允许可证明的约束进入硬拒绝；
shared上界超限和短循环本身均不能作为通用非法证明。正式5seed/64预算、模型级及官方Pruner复现
留在开发批完成之后，不在本次恢复时直接铺开。

## 4. 代码与记录入口

- 主驱动：`e1_thread_filter.py`；观测与正确性辅助：`p3_observability.py`。
- 测试：`test_p3_observability.py`；冒烟结果：
  `results/paper_v1/p3_readiness/READINESS_RESULTS_20261008.md`。
- 主线协议、控制台及原始账本：`results/paper_v1/p3_budget32/`。
- 离线分析：`analyze_p3_mainline.py`已写，尚未执行验证或生成主线图；
  在绘图前需验证其统计与协议检查，不能把缺少result的B11直接当完整run输入。
- 流水线补充诊断：`results/paper_v1/p1_verify_diagnostics/COUNTERFACTUAL_NOTE_20261008.md`。
- 总进度：`../实验进度记录.md`；研究执行计划：`EXPERIMENT_PLAN_20261008.md`第11节。
- 论文：`../papers/main.tex`；现有PDF14页，尚未加入本次未完成P3的性能结论。

今晚只归档并暂停，不启动重跑、不新增算法、不编译论文。待用户要求继续后，从第0项恢复。
