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
"""Audit completed 800-trial arms and regenerate figures without changing raw data."""

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import statistics

from report_single_seed_study import audit, audit_retiming, digest, target_point, write_json, COLORS, LABELS


WORKLOADS = ["matmul128", "matmul256", "conv2d_nchw3x3", "reduction4096"]
TITLES = ["MatMul 128", "MatMul 256", "Conv2D 3x3", "Reduction 128x4096"]


def metrics(message):
    return {key: float(value) for key, value in re.findall(
        r"((?:tr-)?(?:p-rmse|a-peak@\d+)):\s*([+\-\d.eE]+)", message or "")}


def training_progress(result, row, events):
    points = []
    previous = 0
    for batch, curve, phase in zip(result["measurement"]["callback_batches"], row["curve"],
                                    result["timeline"]["batches"]):
        state = batch["cost_model"]
        size = state["data_size"]
        if not previous <= size <= curve["submitted"] or state["last_train_size"] != size:
            raise ValueError("Accumulating XGB training history is inconsistent")
        if size and not state["trained"]:
            raise ValueError("Feature-bearing feedback did not train XGB")
        messages = [event["message"] for event in events if event["event"] == "xgb_log"
                    and event["batch_index"] == batch["timeline_batch_index"]]
        validation = next((metrics(msg) for msg in messages if msg.startswith("XGB validation:")), {})
        if previous and size > previous and not validation:
            raise ValueError("Missing pre-update model validation on new feedback")
        fit = metrics(state["best_metrics"])
        points.append({"submitted": curve["submitted"], "tune_s": curve["tune_s"], **state,
                       "update_s": phase["callback_s"]["UpdateCostModel"],
                       "train_rmse": float(state["best_score"]) if state["best_score"] else None,
                       "train_peak32": fit.get("tr-a-peak@32"),
                       "next_batch_rmse": validation.get("p-rmse"),
                       "next_batch_peak32": validation.get("a-peak@32"),
                       "training_log_entries": len(messages)})
        previous = size
    if len(points) != len(row["curve"]) or previous != row["xgb_data_size"]:
        raise ValueError("Training progress coverage mismatch")
    return points


def collect(directory):
    protocol = json.loads((directory / "protocol.json").read_text(encoding="utf-8"))
    inputs = {str(directory / "protocol.json"): digest(directory / "protocol.json")}
    report = {"seed": protocol["seed"], "protocol_sha256": inputs[str(directory / "protocol.json")],
              "generated_utc": datetime.now(timezone.utc).isoformat(), "runs": [], "pending": [],
              "retiming": [], "pytorch": [], "targets": [], "input_sha256": inputs,
              "notes": ["One search seed; no cross-seed confidence intervals.",
                        "Submitted candidates include failures. Search-space exhaustion is retained.",
                        "Convergence uses actual tuning measurements at batch completion; no interpolation.",
                        "Targets use the baseline's observed final latency, evaluated retrospectively.",
                        "Independent retiming freezes the original winner; it does not select another winner.",
                        "Training fit uses accumulated samples; next-batch validation uses the previous model "
                        "on feedback selected by the search, before training. It is not an independent test set.",
                        "UpdateCostModel time includes feature extraction and pre-update validation as well as training.",
                        "PyTorch is a separate-process CUDA-event reference; its Python launch/allocation costs "
                        "differ from TVM's runtime timing."]}
    for spec in protocol["order"]:
        if spec["smoke"]:
            continue
        matches = list(directory.glob(spec["id"] + "_*/result.json"))
        if not matches:
            report["pending"].append(spec["id"])
            continue
        if len(matches) != 1:
            raise ValueError(f"Duplicate results for {spec['id']}")
        path = matches[0]
        result = json.loads(path.read_text(encoding="utf-8"))
        if not result["config"].get("record_model_training"):
            raise ValueError("Model monitoring is required for this cohort")
        row = audit(path, spec, protocol)
        events_path = path.with_name("timeline.jsonl")
        events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
        row["model_progress"] = training_progress(result, row, events)
        row["own_best"] = target_point(row, row["best_us"]) if row["best_us"] else None
        report["runs"].append(row)
        for raw in [path, events_path, path.with_name("callback_batches.jsonl")]:
            inputs[str(raw)] = digest(raw)
    for workload in WORKLOADS:
        rows = [r for r in report["runs"] if r["workload"] == workload]
        baseline = next((r for r in rows if r["arm"] == "B"), None)
        if baseline and baseline["best_us"]:
            for row in rows:
                for ratio in protocol["targets"]["relative_to_baseline_final"]:
                    target = baseline["best_us"] * ratio
                    point = target_point(row, target)
                    report["targets"].append({"workload": workload, "arm": row["arm"], "ratio": ratio,
                                              "target_us": target, **point,
                                              "speedup_vs_baseline_full_tune":
                                              baseline["tune_s"] / point["tune_s"] if point["attained"] else None})
        timing_dir = directory / "retiming" / f"{workload}_ga1_R01"
        if (timing_dir / "result.json").exists():
            timing = json.loads((timing_dir / "result.json").read_text(encoding="utf-8"))
            if timing["status"] == "completed":
                inputs.update(audit_retiming(timing, rows, protocol, timing_dir))
                inputs[str(timing_dir / "result.json")] = digest(timing_dir / "result.json")
                report["retiming"].append({"workload": workload, "arms": timing["arms"]})
        torch_path = directory / "pytorch" / f"{workload}.json"
        if torch_path.exists():
            torch = json.loads(torch_path.read_text(encoding="utf-8"))
            if torch["status"] == "completed":
                samples = torch["samples_us"]
                if (len(samples) != 30 or len(torch["checks"]) != 4
                        or not all(c["passed"] for c in torch["checks"])
                        or any(not math.isfinite(v) or v <= 0 for v in samples)
                        or torch["median_us"] != statistics.median(samples)
                        or torch["workload"] != workload or torch["tf32"]
                        or torch["script_sha256"] != protocol["script_sha256"]["benchmark_torch_operators.py"]
                        or torch["reference_sha256"] != protocol["script_sha256"]["clean_workloads.py"]):
                    raise ValueError("PyTorch reference audit failed")
                inputs[str(torch_path)] = digest(torch_path)
                report["pytorch"].append(torch)
    return report


def plots(report, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def panels(title):
        fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
        fig.suptitle(title)
        return fig, list(axes.flat)

    def save(fig, name, note):
        fig.text(0.5, 0.012, note, ha="center", fontsize=8)
        fig.tight_layout(rect=(0, 0.055, 1, 0.95))
        fig.savefig(output / (name + ".png"), dpi=180)
        fig.savefig(output / (name + ".svg"))
        plt.close(fig)

    def selected(workload):
        return sorted([r for r in report["runs"] if r["workload"] == workload], key=lambda r: r["arm"])

    for key, name, title, xlabel in [
            ("submitted", "convergence_submissions", "Search quality at equal submitted budget", "Submitted candidates"),
            ("tune_s", "convergence_time", "Search quality versus full tuning wall time", "Full tuning time (s)")]:
        fig, axes = panels(title)
        for axis, workload, label in zip(axes, WORKLOADS, TITLES):
            for row in selected(workload):
                points = [p for p in row["curve"] if p["best_us"] is not None]
                axis.step([p[key] for p in points], [p["best_us"] for p in points], where="post",
                          color=COLORS[row["arm"]], label=LABELS[row["arm"]])
            axis.set(title=label, xlabel=xlabel, ylabel="Best observed latency (us)")
            if key == "submitted":
                axis.set_xlim(0, 800)
            axis.grid(alpha=0.2)
            if axis.lines:
                axis.legend(fontsize=8)
        save(fig, name, "One seed; actual measurements at batch endpoints; independent winner retiming reported separately.")

    fig, axes = panels("Time to baseline final observed performance")
    for axis, workload, label in zip(axes, WORKLOADS, TITLES):
        rows = selected(workload)
        points = [p for p in report["targets"] if p["workload"] == workload and p["ratio"] == 1.0]
        for i, arm in enumerate("BCD"):
            hit = next((p for p in points if p["arm"] == arm), None)
            if not hit:
                continue
            seconds = hit["tune_s"] if hit["attained"] else hit["observed_until_s"]
            if hit["attained"]:
                axis.barh(i, seconds, color=COLORS[arm])
                axis.text(seconds, i, f" {seconds:.0f}s", va="center", fontsize=8)
            else:
                axis.scatter(seconds, i, marker="x", color=COLORS[arm])
                axis.annotate("NR", (seconds, i), xytext=(-15, 8), textcoords="offset points")
        axis.set(title=label, xlabel="Full tuning time (s)")
        axis.set_yticks(range(3), [LABELS[a] for a in "BCD"], fontsize=8)
        axis.invert_yaxis()
        if rows:
            axis.set_xlim(0, max(r["tune_s"] for r in rows) * 1.25)
    save(fig, "time_to_target", "Retrospective strict baseline endpoint; NR = not reached within the submitted budget.")

    fig, axes = panels("Independent latency of the fixed winning candidates")
    for axis, workload, label in zip(axes, WORKLOADS, TITLES):
        timing = next((t for t in report["retiming"] if t["workload"] == workload), None)
        torch = next((t for t in report["pytorch"] if t["workload"] == workload), None)
        data = timing["arms"] if timing else []
        if torch:
            data = data + [{"name": "PyTorch", **torch}]
        for i, arm in enumerate(data):
            axis.bar(i, arm["median_us"], color=COLORS.get(arm["name"], "#777777"))
            axis.errorbar(i, arm["median_us"], yerr=[[arm["median_us"] - arm["p25_us"]],
                                                    [arm["p75_us"] - arm["median_us"]]],
                          color="black", capsize=3)
        axis.set_xticks(range(len(data)), [LABELS.get(a["name"], a["name"]) for a in data], fontsize=8)
        axis.set(title=label, ylabel="Median latency (us)")
    save(fig, "final_performance", "30 samples per fixed candidate; bars show medians, whiskers IQR. PyTorch uses separate CUDA-event timing.")

    fig, axes = panels("Candidate generation and VerifyGPU rejection counts")
    for axis, workload, label in zip(axes, WORKLOADS, TITLES):
        rows = [r for r in selected(workload) if r["arm"] in "BC"]
        for offset, key, legend in [(-0.2, "generated", "Generated"), (0.2, "verify_failed", "VerifyGPU rejected")]:
            axis.bar([i + offset for i in range(len(rows))], [r[key] for r in rows], width=0.4, label=legend)
        axis.set_xticks(range(len(rows)), [LABELS[r["arm"]] for r in rows])
        axis.set(title=label, ylabel="Candidates")
        axis.legend(fontsize=8)
    save(fig, "candidate_counts", "Counts only; same submission cap. Actual completed budgets are listed in RESULTS.md.")

    fig, axes = panels("Cost-model prediction time and Draft scoring time")
    for axis, workload, label in zip(axes, WORKLOADS, TITLES):
        rows = [r for r in selected(workload) if r["arm"] in "CD"]
        axis.bar(range(len(rows)), [r["xgb_prediction_s"] for r in rows], label="XGB prediction")
        axis.bar(range(len(rows)), [r["draft_scoring_s"] for r in rows],
                 bottom=[r["xgb_prediction_s"] for r in rows], label="Draft scoring")
        axis.set_xticks(range(len(rows)), [LABELS[r["arm"]] for r in rows], fontsize=8)
        axis.set(title=label, ylabel="Nested scoring time (s)")
        axis.legend(fontsize=8)
    save(fig, "prediction_cost", "Internal GA/final prediction counters; these costs are nested in Search and are not added to wall time.")

    fig, axes = panels("Accumulated feature-bearing training measurements")
    for axis, workload, label in zip(axes, WORKLOADS, TITLES):
        for row in selected(workload):
            points = row["model_progress"]
            axis.plot([p["submitted"] for p in points], [p["data_size"] for p in points],
                      color=COLORS[row["arm"]], label=LABELS[row["arm"]])
        axis.set(title=label, xlabel="Submitted candidates", ylabel="Cumulative training samples", xlim=(0, 800))
        if axis.lines:
            axis.legend(fontsize=8)
    save(fig, "training_samples", "Each arm retains one XGBModel; all feature-bearing history accumulates throughout tuning.")

    for key, name, title, ylabel in [
            ("update_s", "model_update_time", "Cost-model update cost per feedback batch", "UpdateCostModel time (s)"),
            ("train_rmse", "training_fit", "Training fit on accumulated samples", "Best training p-RMSE"),
            ("next_batch_rmse", "next_batch_prediction", "Previous-model predictions on new feedback", "Pre-update next-batch p-RMSE"),
            ("next_batch_peak32", "next_batch_ranking", "Previous-model ranking of new feedback", "Pre-update a-peak@32")]:
        fig, axes = panels(title)
        for axis, workload, label in zip(axes, WORKLOADS, TITLES):
            for row in selected(workload):
                points = [p for p in row["model_progress"] if p[key] is not None and math.isfinite(p[key])]
                axis.plot([p["submitted"] for p in points], [p[key] for p in points],
                          color=COLORS[row["arm"]], label=LABELS[row["arm"]])
            axis.set(title=label, xlabel="Submitted candidates", ylabel=ylabel, xlim=(0, 800))
            if axis.lines:
                axis.legend(fontsize=8)
        save(fig, name, "One search seed; next-batch checks use selected feedback before it is added to training history.")

    rows = report["runs"]
    fig, axis = plt.subplots(figsize=(11, max(4, len(rows) * 0.4 + 2)))
    left = [0.0] * len(rows)
    for key, label in [("search_s", "Search"), ("builder_s", "Builder"),
                       ("runner_and_notify_s", "Runner + notify"), ("callbacks_s", "Callbacks"),
                       ("residual_s", "Residual")]:
        values = [r["phases_s"][key] for r in rows]
        axis.barh(range(len(rows)), values, left=left, label=label)
        left = [a + b for a, b in zip(left, values)]
    axis.set_yticks(range(len(rows)), [f"{r['workload']} | {LABELS[r['arm']]}" for r in rows], fontsize=8)
    axis.invert_yaxis()
    axis.set(xlabel="Full tuning time (s)", title="Non-overlapping tuning stage costs")
    axis.legend(fontsize=8)
    save(fig, "stage_time", "Correctness is inside Runner; model updates inside Callbacks; preparation is reported separately.")


def markdown(report):
    def number(value):
        return f"{value:.3f}" if value is not None else "—"

    lines = ["# 单算子 800 次提交实验", "", f"Seed：{report['seed']}。已完成 {len(report['runs'])}/12 组。", "",
             "预算按实际提交计数，构建/运行失败占预算。模型在每组实验内部持续积累历史样本；B/C/D 不共享测量信息。", "",
             "| 算子 | 方法 | 提交 / 有效 | 最优实测 µs | 独立重测 µs | 重测相对baseline加速 | Tune总用时 s | 首次找到最终最优 s | 到baseline目标 s | 准备 s |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in report["runs"]:
        timing = next((t for t in report["retiming"] if t["workload"] == row["workload"]), None)
        winner = next((a for a in timing["arms"] if a["name"] == row["arm"]), None) if timing else None
        baseline = next((a for a in timing["arms"] if a["name"] == "B"), None) if timing else None
        speedup = baseline["median_us"] / winner["median_us"] if baseline and winner else None
        target = next((p for p in report["targets"] if p["workload"] == row["workload"]
                       and p["arm"] == row["arm"] and p["ratio"] == 1), None)
        hit = number(target["tune_s"]) if target and target["attained"] else ("NR" if target else "—")
        lines.append(f"| {row['workload']} | {LABELS[row['arm']]} | {row['budget']['actual_submitted']} / "
                     f"{row['outcomes'].get('finite', 0)} | {number(row['best_us'])} | "
                     f"{number(winner['median_us'] if winner else None)} | {number(speedup)} | {row['tune_s']:.3f} | "
                     f"{number(row['own_best']['tune_s'] if row['own_best'] else None)} | {hit} | {row['preparation_s']:.3f} |")
    lines += ["", "首次找到最终最优按实际调优批末时间计算；独立重测不改变候选选择。NR 表示在本组预算内未达到 baseline 最终实测值。", "",
              "| 算子 | 方法 | 生成数 | VerifyGPU拒绝数 | XGB评分输入 | XGB预测时间 s | Draft评分时间 s | 最终训练样本 | 模型更新合计 s |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in report["runs"]:
        lines.append(f"| {row['workload']} | {LABELS[row['arm']]} | {row['generated']} | {row['verify_failed']} | "
                     f"{row['xgb_inputs']} | {row['xgb_prediction_s']:.6f} | {row['draft_scoring_s']:.6f} | "
                     f"{row['xgb_data_size']} | {sum(p['update_s'] for p in row['model_progress']):.3f} |")
    lines += ["", "| 算子 | PyTorch参考 µs | P25 / P75 µs |", "|---|---:|---:|"]
    for row in report["pytorch"]:
        lines.append(f"| {row['workload']} | {row['median_us']:.3f} | {row['p25_us']:.3f} / {row['p75_us']:.3f} |")
    lines += ["", "训练样本数不一定等于有效测量数：构建失败不进入模型，具有特征的运行失败可作为高代价反馈。", "",
              "`model_training.csv` 保留每批数据量、更新耗时、训练拟合与更新前的新批预测指标。训练拟合不能单独证明泛化能力；新批候选也受搜索选择影响。", "",
              "待完成：" + (", ".join(report["pending"]) or "无"), "", "口径说明：", ""]
    lines.extend("- " + note for note in report["notes"])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    report = collect(args.input.resolve())
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "summary.json", report)
    (args.output / "RESULTS.md").write_text(markdown(report), encoding="utf-8")
    keys = ["workload", "arm", "submitted", "data_size", "last_train_size", "update_s", "boosted_rounds",
            "best_iteration", "train_rmse", "train_peak32", "next_batch_rmse", "next_batch_peak32", "tune_s"]
    with (args.output / "model_training.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in report["runs"]:
            for point in row["model_progress"]:
                writer.writerow({"workload": row["workload"], "arm": row["arm"], **point})
    if not args.no_plots:
        plots(report, args.output)
    write_json(args.output / "manifest.json", {"report_script_sha256": digest(__file__),
               "protocol_sha256": report["protocol_sha256"], "input_sha256": report["input_sha256"],
               "output_sha256": {p.name: digest(p) for p in args.output.iterdir()
                                 if p.is_file() and p.name != "manifest.json"}})
    print(f"Audited {len(report['runs'])}/12 formal runs; figures: {args.output}")


if __name__ == "__main__":
    main()
