# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements. See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership. The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied. See the License for the
# specific language governing permissions and limitations
# under the License.
"""Seed31-only tables and three core figures from archived, audited measurements.

No TVM import or new measurements. Does not overwrite older pilot figures.
"""

import json
from pathlib import Path

import plot_paper_pilot_20261008 as pilot

ROOT = pilot.ROOT
OUT = ROOT / "papers/figures/seed31_20261010"
REPORT = ROOT / "experiments/SEED31_RESULTS_20261010.md"
TABLE = ROOT / "papers/tables/seed31_ablation_20261010.tex"
SEED = 20260931
MAIN = ("Baseline", "CS+H", "CS+H+Draft 0.25")
NAMES = {"Baseline": "Baseline", "CS+H": "CS+H", "CS+H+Draft 0.25": "CS+H+Draft-GA"}
OUTPUTS = {}


def save(fig, name, note, bottom=0.09):
    fig.text(0.5, 0.012, note, ha="center", va="bottom", fontsize=8, color="#444444")
    fig.tight_layout(rect=(0, bottom, 1, 0.94))
    for suffix in ("png", "svg"):
        path = OUT / f"{name}.{suffix}"
        fig.savefig(path, dpi=220, bbox_inches="tight")
        OUTPUTS[path.relative_to(ROOT).as_posix()] = pilot.digest_bytes(path.read_bytes())
    pilot.plt.close(fig)


def main():
    all_runs, identity = pilot.load_runs()
    runs = {name: all_runs[SEED, name] for name in MAIN}
    sensitivity = all_runs[SEED, "CS+H+Draft 0.50"]
    records = []
    baseline = runs["Baseline"]["result"]
    for name, run in [*runs.items(), ("CS+H+Draft 0.50", sensitivity)]:
        result, curve = run["result"], run["curve"]
        account = result["search"]["totals"]["accounting"]
        draft = result["search"]["draft_ga"]
        records.append({
            "arm": name, "result_path": run["path"].relative_to(ROOT).as_posix(),
            "generated": account["generated"], "verify_failed": account["verify_failure_total"],
            "postproc_passed": account["passed"], "submitted": result["measurement"]["sent_to_builder"],
            "finite": result["measurement"]["finite_records"],
            "best_us": result["measurement"]["best_latency_us"], "timing": result["timing"],
            "xgb_ga_calls": draft["ga_xgb_calls"], "xgb_ga_inputs": draft["ga_xgb_input"],
            "xgb_final_calls": draft["final_xgb_calls"], "xgb_final_inputs": draft["final_xgb_input"],
            "xgb_predict_s": (draft["ga_xgb_time_us"] + draft["final_xgb_time_us"]) / 1e6,
            "draft_score_s": draft["draft_time_us"] / 1e6,
            "tune_reduction_pct": 100 * (1 - result["timing"]["tune_call_s"] / baseline["timing"]["tune_call_s"]),
            "curve": curve["points"],
            "targets": pilot.target_results(curve, baseline["measurement"]["best_latency_us"]),
        })
    OUT.mkdir(parents=True, exist_ok=True)
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    pilot.plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

    fig, axes = pilot.plt.subplots(1, 2, figsize=(10, 4.3), sharey=True)
    for name, run in runs.items():
        points = run["curve"]["points"]
        values = [point["best_so_far_latency_us"] for point in points]
        for axis, key in zip(axes, ("measured_trials", "elapsed_tuning_loop_s")):
            axis.step([point[key] for point in points], values, where="post", marker="o",
                      color=pilot.COLORS[name], label=NAMES[name])
            axis.grid(alpha=0.2)
    axes[0].set(xlabel="Submitted candidates (all 16 succeeded)", ylabel="Best observed kernel latency (us)", xticks=[4, 8, 12, 16])
    axes[1].set_xlabel("Scheduler-loop elapsed time (s)")
    axes[0].legend(fontsize=8)
    fig.suptitle("MatMul128, seed31: search quality at matched submitted budget")
    save(fig, "f1_search_quality", "Batch endpoints only; no values before first result. Historical timing excludes setup and tune residual.\n"
         "Draft-GA final shortlist ratio = 0.25; measurements are tuning observations, not independent retiming.")

    fig, axes = pilot.plt.subplots(1, 4, figsize=(12, 4), sharey=True)
    for index, axis in enumerate(axes):
        for row, record in enumerate(records[:3]):
            target = record["targets"][index]
            end = target["scheduler_elapsed_s_to_target"] if target["reached"] else target["observation_end_scheduler_elapsed_s"]
            color = pilot.COLORS[record["arm"]]
            if target["reached"]:
                axis.barh(row, end, color=color, height=0.5)
                axis.text(end + 6, row, f"{end:.0f}s", va="center", fontsize=9)
            else:
                axis.plot(end, row, marker="x", color=color, markersize=9, mew=2)
                axis.text(end - 8, row + 0.25, f"NR by {end:.0f}s", ha="right", fontsize=8)
        target = records[0]["targets"][index]
        axis.set_title(f"Baseline final +{100 * target['relative_tolerance']:.0f}%\nTarget {target['target_latency_us']:.2f} us", fontsize=10)
        axis.set(xlim=(0, 590), xlabel="Scheduler time (s)", yticks=range(3))
        axis.grid(axis="x", alpha=0.2)
    axes[0].set_yticklabels([NAMES[name] for name in MAIN])
    axes[0].set_ylim(2.65, -0.55)
    fig.suptitle("Seed31 time-to-target: sensitivity to the required performance")
    save(fig, "f2_time_to_target", "Retrospective thresholds from baseline's tuning minimum, not an independent optimal reference.\n"
         "NR = not reached within the observed run; crosses are censoring times, not attainment times.")

    fig, axis = pilot.plt.subplots(figsize=(9, 4))
    left = pilot.np.zeros(3)
    fields = (("search_s", "Search (Replay/Verify included)", "#009E73"),
              ("builder_s", "Builder", "#0072B2"),
              ("runner_and_update_s", "Runner + model update", "#E69F00"),
              ("unaccounted_inside_tune_s", "Tune residual", "#999999"))
    for field, label, color in fields:
        values = pilot.np.array([record["timing"][field] for record in records[:3]])
        axis.barh(range(3), values, left=left, color=color, label=label, height=0.55)
        for row, (offset, value) in enumerate(zip(left, values)):
            axis.text(offset + value / 2, row, f"{value:.1f}", ha="center", va="center", fontsize=8, color="white")
        left += values
    for row, total in enumerate(left):
        axis.text(total + 5, row, f"{total:.1f}s", va="center", fontsize=9)
    axis.set(yticks=range(3), yticklabels=[NAMES[name] for name in MAIN], xlabel="Full tune-call time (s)", xlim=(0, 590))
    axis.invert_yaxis()
    fig.legend(*axis.get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, 0.10), ncol=2, fontsize=8)
    fig.suptitle("Seed31: stage costs and total tuning time")
    save(fig, "f3_stage_time", "Historical log stages have one-second resolution. Runner and model update cannot be separated retrospectively.\n"
         "Setup is excluded from tune; nested Draft/XGB times are not added again.", bottom=0.30)

    lines = ["# seed31实验表与核心图（2026-10-10）", "",
             "按用户要求先聚焦seed31效果，不等待多seed稳定性实验。以下是历史真实编译/GPU测量的重新整理，非新增调优。",
             "统一设置：MatMul128 FP32，seed=20260931，16次提交、每批4次，population32/init16、GA1，RTX3050Ti。",
             "三组源代码/库/预算身份相同；G1/G2/G3/G4关闭，D采用既定verify_ratio=0.25。", "",
             "## 1. 三组主消融", "",
             "| 方法 | 生成尝试 | Verify失败 | GA内XGB调用/输入数 | 末端XGB调用/输入数 | 搜索秒 | 总调优秒 | 最优延迟μs |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for record in records[:3]:
        lines.append(f"| {NAMES[record['arm']]} | {record['generated']} | {record['verify_failed']} | "
                     f"{record['xgb_ga_calls']}/{record['xgb_ga_inputs']} | {record['xgb_final_calls']}/{record['xgb_final_inputs']} | "
                     f"{record['timing']['search_s']:.1f} | {record['timing']['tune_call_s']:.3f} | {record['best_us']:.4f} |")
    lines += ["", "三组均16提交/16有限测量。生成尝试不等于去重trace数量；内核延迟与调优耗时是不同指标。",
              "CS+H相对baseline：生成尝试减少71.1%，Verify失败减少90.1%，搜索时间减少19.4%，总调优时间减少5.3%；",
              "最终内核延迟增加约0.46%，即这组主要收益是少做无效工作和更早得到接近基线最终质量的程序。",
              "Draft-GA移除了GA内XGB评分，但其最终内核更慢；这组不能写成Draft改善搜索质量。", "",
              "## 2. 达标时间与同提交质量", "",
              "达标阈值基于baseline最终实测32.9672μs，属于回顾性描述。历史时间轴从首次scheduler pick算起，",
              "不是完整tune起点；缺失的逐批完整tune时间不由总残差均摊或倒推。", "",
              "| 目标 | Baseline：秒/提交数 | CS+H：秒/提交数 | CS+H+Draft-GA：秒/提交数 |",
              "|---|---:|---:|---:|"]
    for index in range(4):
        target = records[0]["targets"][index]
        cells = []
        for record in records[:3]:
            hit = record["targets"][index]
            cells.append(f"{hit['scheduler_elapsed_s_to_target']:.0f}/{hit['finite_trials_to_target']}" if hit["reached"] else "未达到")
        lines.append(f"| baseline最终+{100 * target['relative_tolerance']:.0f}%（{target['target_latency_us']:.2f}μs） | " + " | ".join(cells) + " |")
    lines += ["", "5%目标下：CS+H用4条/129秒，baseline用16条/495秒，提交数减少75%，scheduler达标时间减少73.9%（3.84倍）。",
              "1%目标下：两组均需16条，时间470对495秒；因此达标优势依赖目标精度，不能仅展示最有利阈值。", "",
              "## 3. 阶段拆分", "",
              "| 方法 | Search秒 | Builder秒 | Runner+更新秒 | Tune残差秒 | 另列Setup秒 |",
              "|---|---:|---:|---:|---:|---:|"]
    for record in records[:3]:
        lines.append("| " + NAMES[record["arm"]] + " | " + " | ".join(f"{record['timing'][key]:.3f}" for key in
                     ("search_s", "builder_s", "runner_and_update_s", "unaccounted_inside_tune_s", "setup_s")) + " |")
    lines += ["", "Builder约占总调优的三分之二，解释了为何搜索阶段节省19.4%并未变成同幅度总时间下降。",
              "历史日志无法独立拆开Runner和模型更新；新实验使用完整timeline。", "",
              "## 4. Draft比例敏感性（不替换主配置）", "",
              f"保留50%：生成{records[3]['generated']}，Verify失败{records[3]['verify_failed']}，"
              f"搜索{records[3]['timing']['search_s']:.0f}秒，总调优{records[3]['timing']['tune_call_s']:.3f}秒，最优{records[3]['best_us']:.4f}μs。",
              "它仍差于baseline的最终内核延迟；不要按结果在25%/50%间挑选主表。", "",
              "## 5. 图和来源", "",
              "- 同提交质量及scheduler收敛：[图1](../papers/figures/seed31_20261010/f1_search_quality.png)。",
              "- 达标时间：[图2](../papers/figures/seed31_20261010/f2_time_to_target.png)。",
              "- 阶段耗时：[图3](../papers/figures/seed31_20261010/f3_stage_time.png)。",
              "- 三线表LaTeX：[表格](../papers/tables/seed31_ablation_20261010.tex)，依赖booktabs。",
              "- 每张图另有SVG；同目录manifest.json记录源文件hash、原始数值、配置及图hash。",
              "- 重现：`.venv-tune/python.exe -X utf8 experiments/report_seed31_20261010.py`。", "",
              "独立重测不是将历史最小值换个名称：需固定各组原始top-1 trace，在相同环境下重新编译、检查数值、",
              "预热及交错重复计时，另存所有样本/中位数。不能用重测后的终点改写过去的收敛轨迹。",
              "MatMul256、标准Conv2D及大Reduction扩展另存新批次，不将历史自定义截取卷积冒充标准卷积。", ""]
    retime_path = ROOT / "experiments/results/paper_v1/seed31_retime_20261010/R01/result.json"
    if retime_path.exists():
        retime = pilot.read_json(retime_path)
        if retime["status"] == "completed":
            pilot.require(retime["loaded_library_sha256"] == identity["loaded_library_sha256"], "Retiming DLL mismatch")
            fig, axis = pilot.plt.subplots(figsize=(8.4, 4.7))
            samples = [row["samples_us"] for row in retime["arms"]]
            boxes = axis.boxplot(samples, tick_labels=[NAMES[name] for name in MAIN], patch_artist=True, showfliers=False)
            for patch, name in zip(boxes["boxes"], MAIN):
                patch.set_facecolor(pilot.COLORS[name])
                patch.set_alpha(0.45)
            for index, row in enumerate(retime["arms"]):
                jitter = pilot.np.linspace(-0.13, 0.13, len(row["samples_us"]))
                axis.scatter(index + 1 + jitter, row["samples_us"], s=12, color=pilot.COLORS[MAIN[index]], alpha=0.7)
                axis.text(index + 1, max(row["samples_us"]) + 2, f"Median {row['median_us']:.2f} us", ha="center", fontsize=9)
            axis.set_ylabel("Independently remeasured kernel latency (us)")
            axis.grid(axis="y", alpha=0.2)
            fig.suptitle("Fixed seed31 winners: independent retiming, 30 samples per arm")
            save(fig, "f4_independent_top1", "Original trace and DLL fixed; 100 warmup calls/round; 3 rotating rounds x 10 repeats.\n"
                 "Technical repeats, not independent search seeds. Shared desktop GPU; activity snapshots retained.")
            lines += ["## 6. 原始top-1独立重测（本日新增真实CUDA测量）", "",
                      "固定原数据库每组最优trace及原DLL，重新编译；四类输入检查全部通过。",
                      "每轮预热100次，三组顺序轮换，3轮×10次计时；每次number=3、min_repeat_ms=100。",
                      "这些是技术重复，不是30个搜索seed；不用于回填历史收敛曲线。", "",
                      "| 方法 | 原调优最小值μs | 独立重测中位数μs | P25–P75 μs | 相对重测baseline延迟下降 |",
                      "|---|---:|---:|---:|---:|"]
            for index, row in enumerate(retime["arms"]):
                improvement = 100 * (1 - row["median_us"] / retime["arms"][0]["median_us"])
                lines.append(f"| {NAMES[MAIN[index]]} | {row['original_latency_us']:.4f} | {row['median_us']:.4f} | "
                             f"{row['p25_us']:.2f}–{row['p75_us']:.2f} | {improvement:.2f}% |")
            lines += ["", "本次独立重测排序与调优最小值排序不同，二者分开呈现；共同环境仍有桌面GPU活动，",
                      "样本分布和各轮GPU利用率/温度已保留，不能把本次中位数差异推广为稳定跨环境收益。",
                      "[独立重测图](../papers/figures/seed31_20261010/f4_independent_top1.png)，",
                      "[原始记录](results/paper_v1/seed31_retime_20261010/R01/result.json)。", ""]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    tex = [r"% Generated from audited historical seed31 measurements; see manifest.json.",
           r"\begin{table}[t]", r"\centering", r"\small",
           r"\caption{MatMul128 ablation, seed 20260931, 16 submitted measurements per method. Lower is better. Draft-GA uses a 0.25 final shortlist ratio.}",
           r"\label{tab:seed31-ablation-20261010}", r"\begin{tabular}{lrrrrr}", r"\toprule",
           r"Method & Generated & Verify fail & Search (s) & Tune (s) & Best ($\mu$s) \\", r"\midrule"]
    for record in records[:3]:
        tex.append(f"{NAMES[record['arm']]} & {record['generated']} & {record['verify_failed']} & "
                   f"{record['timing']['search_s']:.1f} & {record['timing']['tune_call_s']:.2f} & {record['best_us']:.2f} " + r"\\")
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    TABLE.write_text("\n".join(tex), encoding="utf-8")
    for path in (REPORT, TABLE):
        OUTPUTS[path.relative_to(ROOT).as_posix()] = pilot.digest_bytes(path.read_bytes())
    manifest = {"date": "2026-10-10", "scope": "Historical single-seed descriptive analysis; no new GPU measurements",
                "script_sha256": pilot.digest_bytes(Path(__file__).read_bytes()), "inputs": pilot.INPUTS,
                "identity": identity, "runs": records, "outputs": OUTPUTS,
                "time_axis": "Scheduler loop, one-second log resolution; not full tune",
                "targets": "Retrospective baseline final x (1 + {0, .01, .02, .05})"}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps([{key: row[key] for key in ("arm", "generated", "verify_failed", "best_us", "tune_reduction_pct",
                                                "xgb_ga_inputs", "xgb_final_inputs", "xgb_predict_s")} for row in records], indent=2))
    print(f"Report: {REPORT}\nFigures: {OUT}")


if __name__ == "__main__":
    main()
