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

# 论文实验更新（2026-10-10）

已根据 `figures/single_seed_20261010/RESULTS.md` 和审计后的 `summary.json`
更新当前主稿 `main.tex`，重新编译 `main.pdf`（19页）。实验正文位于
`sections/single_seed_experiments.tex`，由主稿直接引入。

本批主比较为 seed20261011、每臂32次提交、B/C/D、五组15次运行。
没有增加搜索seed、重跑GPU或用诊断重测替换原赢家。

## 已合入内容

- 实验协议：共同环境、两个MatMul规模、标准Conv、大Reduction、GA1/4、失败占预算、数值检查。
- 图3–7：同提交次数质量、完整Tune达标时间、互斥阶段、GA评分成本、固定原top-1重测中位数及IQR。
- 表3：15臂生成/Verify、Search、Tune、调优最优和独立重测；表4：GA1/4 C到D评分成本；
  表7：13个阈值、39臂记录，包含全部未达标与截止时间。
- 摘要、引言、讨论与结论同步校准，删除旧待补和普遍加速/保证质量表述。
- 历史固定池与首因诊断保留，明确历史DLL、CPU机制计时及技术重复边界；旧16提交pilot不混入主实验。

## 结果表达的边界

GA4的绝对预测节省约为GA1的2.74倍，但完整Tune减少仅0.67%，Search反而增加。
Reduction的D原赢家重测中位数较B低9.85%、Tune少5.56%；Mat128GA1 D退化，
Mat256 D仍慢于B，Conv没有明确质量优势。Mat256的+5%目标达标优势不能覆盖严格及+1%未达结果；
Reduction宽松目标较早达到、严格目标较晚达到。一个搜索seed和每赢家30个技术样本不支持统计显著性。
Windows完成测量后的退出异常及其worker规避在正文如实交代，未宣称根因修复。

正文累计Tune为11587.414秒，取15个`tune_s`之和；阶段之和11587.407秒因计时边界
相差约7.3毫秒。Builder占比均四舍五入为69.22%。未修改原报告或其已冻结的manifest。

## 生成与核验

数值表由 `../experiments/export_single_seed_paper_tables.py` 从唯一审计summary生成，
验证15运行、15原赢家和39达标记录。图由 `../experiments/plot_single_seed_paper.py`
按6.25英寸双栏宽度生成，PDF/PNG/SVG及输入哈希位于
`figures/single_seed_paper_20261010/`；最低字号8pt。

使用既有MiKTeX XeLaTeX连续编译两次，均退出0；最终日志无未定义引用、超宽盒或缺字。
既有CTeX/LaTeX版本提示和少量underfull行不影响本次渲染。19页均渲染并逐页检查，
未发现缺字、重叠或裁切。渲染预览位于
`../tmp/pdfs/paper_updated_single_seed_20261010_final/`。

修订来源、源码/图表/PDF哈希与视觉核验记录见 `experiment_update_20261010.json`。
