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
"""Analyze completed Scout whole-model runs; never launch or extend tuning."""

import argparse
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def report(study, make_plots=False):
    if read(study / "progress.json")["status"] != "completed":
        raise ValueError("Formal study must complete before final analysis")
    comparison = read(study / "comparison.json")
    initial = read(study / "untuned/result.json")
    curves, records = {}, {}
    for arm in "BCD":
        root = study / ("formal_" + arm)
        record = read(root / "result.json")
        if not record["architecture_passed"]:
            raise ValueError("Architecture validation failed: " + arm)
        points = [{"submitted": 0, "tune_s": 0, "median_ms": initial["median_ms"],
                   "p25_ms": initial["p25_ms"], "p75_ms": initial["p75_ms"]}]
        for checkpoint in record["checkpoints"]:
            measured = read(Path(checkpoint["result"]))
            if measured["status"] != "completed" or not all(c["passed"] for c in measured["checks"]):
                raise ValueError("Invalid whole-model checkpoint: " + str(checkpoint))
            points.append({"submitted": checkpoint["submitted"], "tune_s": checkpoint["tune_elapsed_s"],
                           "median_ms": measured["median_ms"], "p25_ms": measured["p25_ms"],
                           "p75_ms": measured["p75_ms"],
                           "tuned_functions": len(measured["applied_tuned_functions"])})
        curves[arm], records[arm] = points, record
    target = curves["B"][-1]["median_ms"]
    attainments = []
    for ratio in (1.0, 1.01, 1.05):
        for arm in "BCD":
            matches = [p for p in curves[arm] if p["median_ms"] <= target * ratio]
            first = matches[0] if matches else None
            attainments.append({"arm": arm, "ratio": ratio, "threshold_ms": target * ratio,
                                "tune_s": first["tune_s"] if first else None,
                                "submitted": first["submitted"] if first else None})
    result = {"curves": curves, "target_baseline_checkpoint_ms": target,
              "attainments": attainments, "final_independent_comparison": comparison}
    (study / "analysis.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    baseline = comparison[0]
    lines = ["# ResNet-18 / Scout 端到端结果", "",
             "FP32，batch 1，RTX 3050 Ti；单搜索 seed，固定随机权重。", "",
             "## 最终固定程序的独立轮换重测", "",
             "| 配置 | 实际提交 | Tune 秒 | 整网中位延迟 ms | P25–P75 ms | B/本组推理延迟 | B/本组调优时间 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for row in comparison:
        lines.append(f"| {row['arm']} | {row['submitted']} | {row['tune_s']:.2f} | "
                     f"{row['full_model_median_ms']:.5f} | {row['full_model_p25_ms']:.5f}–"
                     f"{row['full_model_p75_ms']:.5f} | "
                     f"{baseline['full_model_median_ms']/row['full_model_median_ms']:.4f} | "
                     f"{baseline['tune_s']/row['tune_s']:.4f} |")
    lines += ["", f"共同未调优参考：{initial['median_ms']:.5f} ms。", "",
              "Tune 秒扣除同步检查点评估等待；原始墙钟、编译和评估成本保留在各组结果中。",
              "延迟为完整 Relax VM 前向，输入和参数常驻 GPU，不含结果取回。", "",
              "## 达到基线最终检查点质量的时间", "",
              f"目标取 B 最后一个检查点的整网中位延迟 {target:.5f} ms；这是回顾性阈值。",
              "从共同未调优起点开始，仅使用实际检查点，不插值。NR 表示未达到；0 表示起点已满足。", "",
              "| 阈值 | B 秒 / 提交 | C 秒 / 提交 | D 秒 / 提交 |", "|---|---:|---:|---:|"]
    for ratio in (1.0, 1.01, 1.05):
        cells = []
        for arm in "BCD":
            row = next(r for r in attainments if r["ratio"] == ratio and r["arm"] == arm)
            cells.append("NR" if row["tune_s"] is None else f"{row['tune_s']:.2f} / {row['submitted']}")
        lines.append(f"| {ratio:.2f} × B | " + " | ".join(cells) + " |")
    lines += ["", "## 覆盖与成本", "",
              "| 配置 | 有限调优记录 | 末尾已回填函数 | 检查点评估总秒 | GA 内学习模型调用 | 批末学习模型调用 |",
              "|---|---:|---:|---:|---:|---:|"]
    for arm in "BCD":
        record = records[arm]
        stats = read(study / ("formal_" + arm) / "architecture.json")["draft_ga"]
        lines.append(f"| {arm} | {record['finite_records']} | {curves[arm][-1]['tuned_functions']} | "
                     f"{record['checkpoint_evaluation_total_s']:.2f} | {stats['ga_xgb_calls']} | {stats['final_xgb_calls']} |")
    lines += ["", "B = MetaSchedule；C = CS+H；D = 完整 Scout。未覆盖函数采用共同 Fallback。",
              "这是一组单 seed 性能结果；30 个计时样本属于技术重复，不能据此宣称跨 seed 稳定性或准确率收益。",
              "候选数量、评分成本、Tune 时间与整网推理延迟分别报告。", "",
              "原始曲线和达标点见 [analysis.json](analysis.json)。"]
    (study / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not make_plots:
        return
    # Plot only measured checkpoint medians; do not replace them by a cumulative minimum.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), constrained_layout=True)
    for arm, color in zip("BCD", ("#3366AA", "#339977", "#DD7722")):
        points = curves[arm]
        y = [p["median_ms"] for p in points]
        for ax, key, scale in ((axes[0], "submitted", 1), (axes[1], "tune_s", 60)):
            x = [p[key] / scale for p in points]
            ax.plot(x, y, "o-", color=color, ms=3, lw=1.3, label=arm)
            ax.fill_between(x, [p["p25_ms"] for p in points], [p["p75_ms"] for p in points],
                            color=color, alpha=0.12, linewidth=0)
    for ax in axes:
        ax.set_ylabel("Full-model latency (ms)")
        ax.grid(alpha=0.2)
        ax.legend(frameon=False, ncol=3, fontsize=8)
    axes[0].set_xlabel("Submitted measurements")
    axes[1].set_xlabel("Tuning time (min; evaluation excluded)")
    fig.savefig(study / "end_to_end_curves.pdf")
    fig.savefig(study / "end_to_end_curves.svg")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--plots", action="store_true", help="Also render PDF/SVG figures after the experiment")
    args = parser.parse_args()
    report(args.study.resolve(), make_plots=args.plots)
