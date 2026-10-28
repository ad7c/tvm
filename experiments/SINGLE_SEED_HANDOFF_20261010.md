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

# 单 seed 实验交接（2026-10-10）

用户最新要求：先用一个搜索 seed 验证效果，不做多 seed 对比。seed 固定为
20261011，未根据结果选择。正式矩阵为 15 次运行、每次 32 submitted；失败占预算。

**最终状态：已完成。** 15/15正式运行、480/480提交finite且正确，三个冒烟各2/2finite。
五组原赢家R01独立重测完成，15个原赢家各30样本，共450个主计时样本；60项重测输入检查通过。
报告和PNG/SVG/PDF三种格式的五类图：`papers/figures/single_seed_20261010/`。
查看`RESULTS.md`、`summary.json`、`paper_table.tex`与`manifest.json`。
**论文已合入本批素材：** `papers/main.tex` 引入
`papers/sections/single_seed_experiments.tex`；`papers/main.pdf` 已重编为19页并完成逐页视觉检查。
正文使用专门按双栏宽度生成的五图及三张数值表；详情见 `papers/EXPERIMENT_UPDATE_20261010.md`。

本批目录：`experiments/results/paper_v1/single_seed_20261010/`。
冻结依据是其中 `protocol.json`；`progress.json` 给出调优状态，
`execution.jsonl` 给出已退出的运行。不要仅用控制台最后一行认定实验完成。

## 本次实现与已完成的工作

1. opt-in 的 `experiments/runtime_bootstrap/sitecustomize.py` 排除本批不需要的
   Torch 导入，主进程和 worker 均实测不加载 Torch；不改已安装包。
   5 项标准算子 reference/mock 测试、7 项原观测测试通过。
2. S01 真实 MatMul128 CUDA 冒烟：2/2 submitted finite，8 项输入检查通过。
3. `run_single_seed_study.py` 串行运行冻结矩阵。开始后不要修改源码身份所覆盖的
   CMake、cmake、include、python、src、tests、e1_thread_filter、p3_observability、
   clean_workloads、runtime_bootstrap，否则后续运行会拒绝继续。
4. `finalize_single_seed_study.py --wait` 已在等待上述调优完整结束；它随后串行独立
   重建每臂原 top-1，执行五组重测，最后调用报告脚本。它没有并发 GPU 测量。
5. `remeasure_top1.py` 已接入标准 Conv2D 和大 Reduction 的独立 NumPy FP64 参考。
   重测固定原赢家，100 次预热/轮、3 轮 × 10 计时样本，轮换 B/C/D 顺序。
6. `report_single_seed_study.py` 验证预算、数值覆盖、来源和互斥阶段耗时后出图。
   图片和报告目标目录为 `papers/figures/single_seed_20261010/`。

## 矩阵

- MatMul128：GA1 按 B/C/D，GA4 按 D/C/B，六次。
- MatMul256：GA1 B/C/D，三次。
- 标准 Conv2D：FP32 NCHW/OIHW，输入 1×16×32×32，核 32×16×3×3，
  stride1/padding1/dilation1/groups1，GA1 C/D/B，三次。
- 大 Reduction：FP32 128×4096 按 axis1 求和，GA1 D/B/C，三次。
- Conv2D 和 Reduction 各先执行一次 2 submitted 的真实冒烟；均不计入正式曲线。

B=baseline，C=CS+H，D=CS+H+Draft-GA，Draft 比例 0.25。
population32/init16，exact init/sample batch32，eps0.1，XGB warmup0。
GA 参数1意味着初代加一代突变，参数4意味着初代加四代突变。
全部臂使用相同环境、检查和记录配置。

## 完成判据与后续

- 正式结果须有完整 `result.json`、架构检查通过，实际提交数明确。
- 失败尝试完整保留；不补录失败预算，不覆盖日志，不拼接旧 B11。
- 需监控两个进程：串行调优和独立重测/出图。重测最终状态在
  `finalization.json`；图生成后还需要代理视觉检查 PNG 并验证 manifest。
- 三类核心图：同提交次数的搜索质量、达到目标性能所需时间、互斥各阶段耗时。
  另有 GA1/GA4 评分开销图及独立重测箱线图；导出 PNG/SVG/PDF。
- MatMul128 目标预定40/50μs；扩展目标按 baseline 终点×1/1.01/1.05，
  明确属于回顾性描述。目标只在 batch callback 完成时判定，不插值。
- 本批一个搜索 seed，30 个重测样本属于技术重复，不声称跨 seed 稳定性。
- 根据实际结果更新实验进度与交接；若 Draft 只省预测而不省总时间或损失质量，
  报告应照实说明，不能为了“有效果”选有利 seed 或替换原赢家。

以上执行步骤已经完成，所有正式结果经预算、来源、数值覆盖和阶段守恒审计。
112项输入/输出文件哈希核验通过，五张PNG与PDF渲染页均已目视检查，无重叠裁切。

## 结论与运行环境追加处理

- GA1 C→D：XGB输入512→64，预测2.560564→0.333087秒，节省87.5%输入/2.227477秒。
  GA4：1280→64，6.427177→0.330386秒，节省95%输入/6.096791秒，绝对评分节省约2.74倍。
  但GA4 Search并未更快，总tune仅减少0.67%，并慢于baseline；不能写成普遍端到端加速。
- Builder占全批完整tune约69.22%，含实际编译及Windows逐批worker启动。
- 原赢家独立重测B/C/D中位数μs：Mat128GA1=31.7013/31.8705/34.0597；
  Mat128GA4=37.5262/36.1484/35.1035；Mat256=31.4930/44.1429/34.4057；
  标准Conv=37.7946/37.6410/37.6904；大Reduction=35.9612/36.8367/32.4199。
  后者D较B快9.85%，总tune较B少5.56%。这些均为单seed描述，技术重复不是搜索seed。
- Mat256按baseline终点+5%目标，D4提交/124.729秒达标，B12提交/312.693秒，C未达。
  严格终点和+1%目标下D也未达；全部阈值列在报告中，不选择有利阈值替代完整结论。
- Conv R01完成数值检查及90个主样本后，父编排器收到非零退出；原子进程码未保留。
  R02诊断和R03普通os._exit处理均复现0xc0000409，所有诊断数据完整保留，不替换主R01。
  R04验证Windows独立worker在检查、计时、文件关闭、device同步后仅终止自身，退出0。
  新Reduce R01使用该选项，退出0。具体故障DLL根因未定位；这是生命周期规避，失败路径仍异常退出。
- 绘图必须使用注册环境的Library/bin和MKL SEQUENTIAL；未配置的手动重绘遇到0xc06d007f。
  报告绘图入口已加入相同DLL/线程设置，注册环境下最终重绘退出0，异常空日志也保留。
- 若重复分析，使用注册environment()启动报告；无需重跑480提交或改变原赢家。

借鉴本地 Pruner 第6节图6、图7、表9及消融组织；不是官方复现。
详细环境、哈希与限制见本批 `ENVIRONMENT.md` 与 `protocol.json`。

## 论文更新完成（2026-10-10）

用户授权按结果报告更新实验部分，现已完成。主实验改为一个预固定seed、32提交、
15正式运行；新增五类论文图和主结果/GA成本/全部目标三表，同步校准摘要、引言、讨论与结论，
删除过时“待补”和普遍加速或保证质量的表述。历史固定池机制证据保留，明确与主实验不同版本。
保留Mat128 GA1质量退化、Mat256严格及+1%目标未达、Conv无明确质量优势等负结果；
区分调优观测与固定原赢家重测、技术重复与搜索seed，说明Windows退出异常的边界。
累计完整Tune取15个原tune_s之和11587.414秒；五阶段之和11587.407秒，相差约7.3毫秒，
二者不可混用；Builder占比四舍五入仍为69.22%。原报告及原始数据保持不变。
最终XeLaTeX连续两次退出0，无未定义引用、超宽盒或缺字；19页PNG逐页检查通过。
渲染及本次修订哈希记录见 `papers/experiment_update_20261010.json`。
