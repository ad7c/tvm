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
"""Audit and plot the registered study; never tune or replace measurements."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys

from analyze_p3_mainline import summarize_run


ROOT = Path(__file__).resolve().parents[1]
COLORS = {"B": "#0072B2", "C": "#009E73", "D": "#E69F00"}
LABELS = {"B": "Baseline", "C": "CS+H", "D": "CS+H+Draft-GA"}
GROUPS = [("matmul128", 1), ("matmul128", 4), ("matmul256", 1),
          ("conv2d_nchw3x3", 1), ("reduction4096", 1)]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def target_point(row, target):
    hit = next((p for p in row["curve"] if p["best_us"] is not None and p["best_us"] <= target), None)
    return {"attained": hit is not None, "submitted": hit["submitted"] if hit else None,
            "tune_s": hit["tune_s"] if hit else None, "observed_until_s": row["tune_s"]}


def reduction_percent(before, after):
    return 100 * (before - after) / before if before else None


def compare_groups(report):
    comparisons = []
    for workload, ga in GROUPS:
        rows = {r["arm"]: r for r in report["runs"] if (r["workload"], r["ga"]) == (workload, ga)}
        retiming = next((r for r in report["retiming"] if (r["workload"], r["ga"]) == (workload, ga)), None)
        medians = {r["name"]: r["median_us"] for r in retiming["arms"]} if retiming else {}
        for before_arm, after_arm in [("B", "C"), ("C", "D")]:
            if before_arm not in rows or after_arm not in rows:
                continue
            before, after = rows[before_arm], rows[after_arm]
            values = {key: reduction_percent(before[key], after[key])
                      for key in ("generated", "verify_failed", "tune_s", "xgb_inputs", "xgb_prediction_s")}
            values["search_s"] = reduction_percent(before["phases_s"]["search_s"], after["phases_s"]["search_s"])
            values["observed_best_us"] = reduction_percent(before["best_us"], after["best_us"]) if before["best_us"] and after["best_us"] else None
            values["retimed_median_us"] = reduction_percent(medians[before_arm], medians[after_arm]) if medians else None
            comparisons.append({"workload": workload, "ga": ga, "before": before_arm, "after": after_arm,
                                "reduction_percent": values,
                                "xgb_prediction_saved_s": before["xgb_prediction_s"] - after["xgb_prediction_s"],
                                "search_saved_s": before["phases_s"]["search_s"] - after["phases_s"]["search_s"],
                                "tune_saved_s": before["tune_s"] - after["tune_s"]})
    return comparisons


def audit(path, spec, protocol):
    result = json.loads(path.read_text(encoding="utf-8"))
    config = dict(protocol["common_config"])
    config.update({"trials": spec["trials"], "trials_per_iter": spec["trials_per_iter"],
                   "genetic_iters": spec["genetic_iters"], "hard_filter_position": "trace"})
    legacy_protocol = {"source_identity_sha256": protocol["identity"]["source_identity_sha256"],
                       "loaded_library_sha256": protocol["runtime"]["compiler_sha256"],
                       "config": config, "workload": spec["workload"],
                       "next_existing_draft_arm": {"draft_ga_verify_ratio": 0.25},
                       "absolute_target_us": [40, 50] if spec["workload"] == "matmul128" else [],
                       "tune_snapshot_s": [300, 600, 900]}
    row = summarize_run(result, legacy_protocol)
    if row["arm"] != spec["arm"] or row["seed"] != protocol["seed"]:
        raise ValueError("Arm or seed differs from registration")
    if row["budget"]["actual_submitted"] > spec["trials"]:
        raise ValueError("Submission budget exceeded")
    model = result["search"]["draft_ga"]
    row.update({"id": spec["id"], "ga": spec["genetic_iters"], "result_path": str(path),
                "result_sha256": digest(path), "config": result["config"],
                "generated": result["search"]["totals"]["accounting"]["generated"],
                "verify_failed": result["search"]["totals"]["verify_gpu_fail"],
                "xgb_prediction_s": (model["ga_xgb_time_us"] + model["final_xgb_time_us"]) / 1e6,
                "xgb_inputs": model["ga_xgb_input"] + model["final_xgb_input"],
                "draft_scoring_s": model["draft_time_us"] / 1e6,
                "setup_s": result["timing"]["setup_s"],
                "preparation_s": result["timing"]["preparation_from_driver_entry_s"]})
    phase_sum = sum(row["phases_s"].values())
    if abs(phase_sum - row["tune_s"]) > 0.1:
        raise ValueError(f"Phase conservation failed: {phase_sum} != {row['tune_s']}")
    return row


def audit_retiming(timing, rows, protocol, directory):
    expected = {r["arm"]: r["result_sha256"] for r in rows}
    actual = {a["name"]: a["result_sha256"] for a in timing["arms"]}
    if actual != expected or timing["loaded_library_sha256"] != protocol["runtime"]["compiler_sha256"]:
        raise ValueError("Independent retiming provenance mismatch")
    for key, value in {"warmup_each_round": 100, "rounds": 3, "repeat_per_round": 10,
                       "number": 3, "min_repeat_ms": 100, "input_seed": 20261010}.items():
        if timing[key] != value:
            raise ValueError(f"Retiming configuration changed: {key}")
    if len({a["workload_sha256"] for a in timing["arms"]}) != 1:
        raise ValueError("Retiming workload differs across arms")
    if [(m["round"], m["arm"]) for m in timing["measurements"]] != [
            (i, "BCD"[(i + j) % 3]) for i in range(3) for j in range(3)]:
        raise ValueError("Retiming rotation changed")
    inputs = {}
    for arm in timing["arms"]:
        samples = arm["samples_us"]
        flattened = [v for m in timing["measurements"] if m["arm"] == arm["name"] for v in m["samples_us"]]
        if (len(samples) != 30 or samples != flattened
                or any(not math.isfinite(v) or v <= 0 for v in samples)
                or not math.isclose(statistics.median(samples), arm["median_us"], rel_tol=1e-12)
                or len(arm["checks"]) != 4 or not all(c["passed"] for c in arm["checks"])):
            raise ValueError("Incomplete or invalid independent retiming samples/checks")
        for path, expected_hash in [(Path(arm["result_path"]).parent / "database_tuning_record.json", arm["database_sha256"]),
                                    (directory / arm["name"] / "trace.txt", arm["trace_sha256"]),
                                    (directory / arm["name"] / "scheduled_mod.json", arm["scheduled_mod_sha256"])]:
            if digest(path) != expected_hash:
                raise ValueError(f"Retiming lineage file changed: {path}")
            inputs[str(path)] = expected_hash
    for name in ("driver_snapshot.py", "clean_workloads_snapshot.py", "command.json", "timings.jsonl"):
        path = directory / name
        inputs[str(path)] = digest(path)
    return inputs


def panels(plt, title):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    axes.flat[-1].axis("off")
    fig.suptitle(title, fontsize=14)
    return fig, list(axes.flat)


def save(fig, output, name, plt, note):
    fig.text(0.5, 0.012, note, ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.055, 1, 0.95))
    fig.savefig(output / (name + ".png"), dpi=180)
    fig.savefig(output / (name + ".svg"))
    fig.savefig(output / (name + ".pdf"))
    plt.close(fig)


def plot(report, output):
    if os.name == "nt":
        # Match the registered subprocess profile before NumPy/Matplotlib load
        # their native dependencies, including the venv's MKL DLL directory.
        library_bin = str(Path(sys.executable).parent / "Library/bin")
        os.environ["PATH"] = library_bin + os.pathsep + os.environ["PATH"]
        os.environ["MKL_THREADING_LAYER"] = "SEQUENTIAL"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["OMP_NUM_THREADS"] = "1"
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = panels(plt, "Search quality at equal submitted budget")
    for axis, group in zip(axes, GROUPS):
        for row in sorted(report["runs"], key=lambda r: r["arm"]):
            if (row["workload"], row["ga"]) != group:
                continue
            points = [p for p in row["curve"] if p["best_us"] is not None]
            axis.step([p["submitted"] for p in points], [p["best_us"] for p in points],
                      where="post", marker="o", markersize=3, color=COLORS[row["arm"]], label=LABELS[row["arm"]])
        axis.set_title(f"{group[0]} | GA{group[1]}")
        axis.set_xlabel("Submitted candidates (failures included)")
        axis.set_ylabel("Best observed kernel latency (us)")
        axis.set_xlim(0, 33)
        axis.set_xticks([0, 8, 16, 24, 32])
        axis.grid(alpha=0.2)
        if axis.lines:
            axis.legend(fontsize=8)
    save(fig, output, "f1_search_quality", plt,
         "One search seed; batch endpoints. Tuning observations, separate from independent winner retiming.")

    fig, axes = panels(plt, "Time to target performance: full tune wall time")
    for axis, group in zip(axes, GROUPS):
        rows = [r for r in report["runs"] if (r["workload"], r["ga"]) == group]
        baseline = next((r for r in rows if r["arm"] == "B"), None)
        target = 40.0 if group[0] == "matmul128" else (baseline["best_us"] * 1.05 if baseline and baseline["best_us"] else None)
        axis.set_title(f"{group[0]} | GA{group[1]}" + (f" | target {target:.2f} us" if target else " | baseline pending"))
        for index, arm in enumerate("BCD"):
            row = next((r for r in rows if r["arm"] == arm), None)
            if row is None or target is None:
                continue
            hit = target_point(row, target)
            seconds = hit["tune_s"] if hit["attained"] else row["tune_s"]
            if hit["attained"]:
                axis.barh(index, seconds, color=COLORS[arm], height=0.55)
                axis.text(seconds, index, f" {seconds:.1f}s / {hit['submitted']} submitted", va="center", fontsize=8)
            else:
                axis.scatter(seconds, index, marker="x", s=70, color=COLORS[arm])
                axis.annotate(f"NR by {seconds:.1f}s", (seconds, index), xytext=(-5, -15),
                              textcoords="offset points", ha="right", fontsize=8)
        axis.set_yticks(range(3), [LABELS[x] for x in "BCD"], fontsize=8)
        axis.invert_yaxis()
        axis.set_xlabel("Full tune seconds (preparation separate)")
        axis.set_xlim(left=0)
        if rows:
            axis.set_xlim(right=max(r["tune_s"] for r in rows) * 1.38)
        axis.grid(axis="x", alpha=0.2)
    save(fig, output, "f2_time_to_target", plt,
         "MatMul128: preregistered 40 us. Extensions: retrospective baseline final +5%. NR = not reached; no interpolation.")

    rows = sorted(report["runs"], key=lambda r: (GROUPS.index((r["workload"], r["ga"])), r["arm"]))
    fig, axis = plt.subplots(figsize=(13, max(4, len(rows) * 0.43 + 2)))
    bottoms = [0.0] * len(rows)
    phases = [("search_s", "Search"), ("builder_s", "Builder"),
              ("runner_and_notify_s", "Runner + notify"), ("callbacks_s", "Callbacks"),
              ("residual_s", "Tune residual")]
    for key, label in phases:
        values = [r["phases_s"][key] for r in rows]
        axis.barh(range(len(rows)), values, left=bottoms, label=label)
        bottoms = [a + b for a, b in zip(bottoms, values)]
    axis.set_yticks(range(len(rows)), [f"{r['workload']} GA{r['ga']} | {LABELS[r['arm']]}" for r in rows], fontsize=9)
    for i, row in enumerate(rows):
        axis.text(row["tune_s"], i, f" {row['tune_s']:.1f}s", va="center", fontsize=8)
    axis.invert_yaxis()
    axis.set_xlabel("Full tune wall time (s)")
    axis.set_title("Non-overlapping stage costs")
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=5, fontsize=9)
    save(fig, output, "f3_stage_time", plt,
         "Setup/preparation separate. Correctness is inside Runner; XGB update inside Callbacks; nested prediction times are not added.")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))
    for arm in "BCD":
        selected = [next((r for r in rows if r["workload"] == "matmul128" and r["ga"] == ga and r["arm"] == arm), None)
                    for ga in [1, 4]]
        for axis, key in zip(axes, ["xgb_prediction_s", "xgb_inputs"]):
            points = [(ga, r[key]) for ga, r in zip([1, 4], selected) if r]
            if points:
                axis.plot([p[0] for p in points], [p[1] for p in points], marker="o", color=COLORS[arm], label=LABELS[arm])
            axis.set_xticks([1, 4])
            axis.set_xlabel("GA iteration parameter")
            axis.grid(alpha=0.2)
    axes[0].set_ylabel("Cumulative XGB prediction time (s)")
    axes[1].set_ylabel("Total XGB prediction inputs")
    axes[0].legend(fontsize=8)
    fig.suptitle("Does more intensive search increase the value of Draft?")
    save(fig, output, "f4_ga_model_cost", plt, "Prediction is nested inside Search. Compare quality and total tune alongside these savings.")

    if report["retiming"]:
        fig, axes = panels(plt, "Fixed archived winners: independent retiming")
        for axis, group in zip(axes, GROUPS):
            timing = next((r for r in report["retiming"] if (r["workload"], r["ga"]) == group), None)
            if timing is None:
                axis.set_title(f"{group[0]} GA{group[1]} | pending")
                continue
            ordered = [next(a for a in timing["arms"] if a["name"] == arm) for arm in "BCD"]
            boxes = axis.boxplot([a["samples_us"] for a in ordered], patch_artist=True)
            for box, arm in zip(boxes["boxes"], "BCD"):
                box.set_facecolor(COLORS[arm]); box.set_alpha(0.55)
            axis.set_xticks([1, 2, 3], ["B", "C", "D"])
            axis.set_title(f"{group[0]} | GA{group[1]}")
            axis.set_ylabel("Retimed kernel latency (us)")
            axis.grid(axis="y", alpha=0.2)
        save(fig, output, "f5_independent_top1", plt,
             "Original winners fixed; 100 warmups/round; 3 rotating rounds x 10 samples. Technical repeats, not search seeds.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    protocol_path = args.input / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    report = {"seed": protocol["seed"], "scope": protocol["scope"], "runs": [], "pending": [],
              "retiming": [], "targets": {}, "inputs": {str(protocol_path): digest(protocol_path)}}
    for spec in protocol["order"]:
        if spec["smoke"]:
            continue
        paths = list(args.input.glob(spec["id"] + "_*/result.json"))
        if not paths:
            report["pending"].append(spec["id"])
            continue
        row = audit(paths[0], spec, protocol)
        report["runs"].append(row)
        report["inputs"][str(paths[0])] = digest(paths[0])
    for group in GROUPS:
        rows = [r for r in report["runs"] if (r["workload"], r["ga"]) == group]
        baseline = next((r for r in rows if r["arm"] == "B"), None)
        if baseline and baseline["best_us"] is not None:
            targets = [40, 50] if group[0] == "matmul128" else [baseline["best_us"] * ratio for ratio in [1, 1.01, 1.05]]
            report["targets"][f"{group[0]}_ga{group[1]}"] = {
                str(target): {r["arm"]: target_point(r, target) for r in rows} for target in targets}
        retime_path = args.input / "retiming" / f"{group[0]}_ga{group[1]}_R01/result.json"
        if retime_path.exists():
            timing = json.loads(retime_path.read_text(encoding="utf-8"))
            if timing["status"] != "completed":
                raise ValueError(f"Incomplete independent retiming: {retime_path}")
            report["inputs"][str(retime_path)] = digest(retime_path)
            report["inputs"].update(audit_retiming(timing, rows, protocol, retime_path.parent))
            timing.update({"workload": group[0], "ga": group[1]})
            report["retiming"].append(timing)
    report["lifecycle_diagnostics"] = {}
    for name in ("retiming_shutdown_diagnostic.json", "retiming_exit_check.json", "retiming_terminal_check.json"):
        path = args.input / name
        if path.exists():
            report["lifecycle_diagnostics"][name] = json.loads(path.read_text(encoding="utf-8"))
            report["inputs"][str(path)] = digest(path)
            for field in ("log", "diagnostic_result"):
                if field in report["lifecycle_diagnostics"][name]:
                    referenced = Path(report["lifecycle_diagnostics"][name][field])
                    report["inputs"][str(referenced)] = digest(referenced)
    report["comparisons"] = compare_groups(report)
    write_json(args.output / "summary.json", report)
    lines = ["# 单 seed 同预算搜索实验（2026-10-10）", "", report["scope"], "",
             "32 submitted；失败占预算；population32/init16；Draft比例0.25；GA参数1/4；逐候选四类输入检查。", "",
             "| 算子 | GA | 方法 | 提交/finite | 生成 | Verify失败 | Search秒 | Tune秒 | 调优最优μs | XGB输入 | XGB预测秒 |",
             "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in report["runs"]:
        best = f"{row['best_us']:.4f}" if row["best_us"] is not None else "无有限结果"
        lines.append(f"| {row['workload']} | {row['ga']} | {LABELS[row['arm']]} | "
                     f"{row['budget']['actual_submitted']}/{row['outcomes'].get('finite', 0)} | {row['generated']} | "
                     f"{row['verify_failed']} | {row['phases_s']['search_s']:.3f} | {row['tune_s']:.3f} | "
                     f"{best} | {row['xgb_inputs']} | {row['xgb_prediction_s']:.6f} |")
    lines += ["", f"待完成：{', '.join(report['pending']) or '无'}。", "",
              "MatMul128达标目标预定40/50μs；扩展算子按baseline终点×1/1.01/1.05作回顾性描述。",
              "完整tune时间以整批callback完成为结果可用时刻，不插值；setup另列；嵌套预测时间不重复累加。", "",
              "| 算子 | GA | 方法 | 固定原top-1重测中位数μs | P25–P75 μs | 样本数 |",
              "|---|---:|---|---:|---|---:|"]
    for timing in report["retiming"]:
        for arm in timing["arms"]:
            lines.append(f"| {timing['workload']} | {timing['ga']} | {LABELS[arm['name']]} | {arm['median_us']:.4f} | "
                         f"{arm['p25_us']:.4f}–{arm['p75_us']:.4f} | {len(arm['samples_us'])} |")
    lines += ["", "独立重测固定原数据库最佳候选，不回填调优曲线。30计时样本属于技术重复；本报告只有一个搜索seed。", ""]
    lines += ["## 达到目标性能的完整 tune 时间", "",
              "MatMul128为预定绝对目标；扩展为各自baseline终点的1/1.01/1.05倍，均列出以便检查阈值敏感性。NR表示观测结束仍未达到。", "",
              "| 算子/GA | 目标μs | 方法 | 达标提交数 | 达标tune秒 / NR观测截止秒 |",
              "|---|---:|---|---:|---:|"]
    for group, targets in report["targets"].items():
        for target, arms in targets.items():
            for arm in "BCD":
                if arm not in arms:
                    continue
                hit = arms[arm]
                count = str(hit["submitted"]) if hit["attained"] else "NR"
                seconds = f"{hit['tune_s']:.3f}" if hit["attained"] else f"NR by {hit['observed_until_s']:.3f}"
                lines.append(f"| {group} | {float(target):.4f} | {LABELS[arm]} | {count} | {seconds} |")
    lines += ["", "## 准备与各阶段耗时", "", "准备时间包括setup，两者不相加；准备不计入上面的tune达标时间。", "",
              "| 运行 | 准备秒 | 其中setup秒 | Search秒 | Builder秒 | Runner+notify秒 | Callbacks秒 | Tune residual秒 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in report["runs"]:
        phases = row["phases_s"]
        lines.append(f"| {row['id']} | {row['preparation_s']:.3f} | {row['setup_s']:.3f} | "
                     + " | ".join(f"{phases[k]:.3f}" for k in ("search_s", "builder_s", "runner_and_notify_s", "callbacks_s", "residual_s")) + " |")
    lines += ["## 方法增量比较", "", "正值表示后者减少该指标；延迟正值表示更快，负值表示退化。单 seed 描述性结果。", "",
              "| 算子 | GA | 比较 | Search减少% | Tune减少% | XGB输入减少% | 预测节省秒 | 调优最优延迟减少% | 独立重测延迟减少% |",
              "|---|---:|---|---:|---:|---:|---:|---:|---:|"]
    for row in report["comparisons"]:
        values = row["reduction_percent"]
        def percent(key):
            return f"{values[key]:.2f}" if values[key] is not None else "—"
        lines.append(f"| {row['workload']} | {row['ga']} | {row['before']}→{row['after']} | "
                     f"{percent('search_s')} | {percent('tune_s')} | {percent('xgb_inputs')} | "
                     f"{row['xgb_prediction_saved_s']:.6f} | {percent('observed_best_us')} | {percent('retimed_median_us')} |")
    lines += ["", "GA1/GA4：比较 C→D 的预测节省秒数、Search减少比例、总Tune减少比例，并同时检查原赢家独立重测质量。",
              "预测开销节省只属于搜索阶段的一部分；不能单独当作总调优加速。", ""]
    if len(report["runs"]) == 15 and len(report["retiming"]) == 5:
        dense = {c["ga"]: c for c in report["comparisons"]
                 if c["workload"] == "matmul128" and c["before"] == "C" and c["after"] == "D"}
        stages = {key: sum(r["phases_s"][key] for r in report["runs"])
                  for key in report["runs"][0]["phases_s"]}
        fraction = 100 * stages["builder_s"] / sum(stages.values())
        lines += ["## 本批证据支持的结论", "",
                  f"1. 增大GA密度提高了Draft节省XGB评分的绝对量：GA1节省{dense[1]['xgb_prediction_saved_s']:.6f}秒，GA4节省{dense[4]['xgb_prediction_saved_s']:.6f}秒（约{dense[4]['xgb_prediction_saved_s']/dense[1]['xgb_prediction_saved_s']:.2f}倍）；输入节省87.5%→95%。",
                  "2. 上述节省没有同比转化为总搜索加速：MatMul128 GA4的C→D Search反而增加1.67%，总tune只减少0.67%，并慢于baseline。GA1的C→D总tune减少0.72%。",
                  f"3. 总耗时主要在Builder：本批合计占完整tune的{fraction:.2f}%。其包含实际编译及Windows逐批worker启动，不能把该成本算作XGB预测。",
                  "4. 独立重测必须与调优最小值分开：MatMul128 GA1的D中位数34.0597μs，劣于B31.7013μs；GA4的D35.1035μs优于B37.5262μs和C36.1484μs。不同组的绝对重测值不能当作同时运行的GA1/GA4配对测试。",
                  "5. MatMul256中D相对C的原赢家重测中位数降低22.06%，但仍慢于B；按baseline终点+5%回顾性目标，D以4提交/124.729秒达到，B需12提交/312.693秒，C未达到。严格终点和+1%目标下D也未达到，不能只据+5%阈值宣称普遍胜出。",
                  "6. 标准Conv三臂主R01重测中位数接近，D没有明确质量优势且tune更慢；大Reduction的D主R01中位数32.4199μs，优于B35.9612μs和C36.8367μs，tune较B减少5.56%。",
                  "7. CS+H在MatMul/Conv明显减少生成和Verify失败，但本批Search未普遍减少。少生成不自动等于少搜索耗时；条件采样与回放成本需要另行分解，不能由候选计数推定总加速。",
                  "这是一个预先固定seed的开发实验，支持工作负载相关的现象与后续研究取舍，不构成跨seed稳定性或统计显著性结论。", ""]
    if report["lifecycle_diagnostics"]:
        lines += ["## Windows退出阶段诊断", "",
                  "Conv2D R01完成12项数值检查和90个计时样本并写出completed结果后，父编排器收到非零退出状态；原子进程码未被旧编排器保留。",
                  "相同原赢家R02诊断在完整计时后复现0xc0000409；R03同步后使用os._exit仍复现同一代码。R04单轮诊断验证Windows专用worker终止处理，退出码记录在retiming_terminal_check.json。",
                  "主分析始终使用原R01，不以诊断复测替换原赢家或挑选更有利样本。新Reduce R01显式记录terminal worker exit策略；失败路径不绕过异常。",
                  "该处理在全部数值检查、计时、文件关闭和device同步完成后仅终止当前独立worker，不是底层原生库根因修复。具体故障DLL尚未定位。",
                  "Windows的ExitProcess会调用DLL退出通知；TerminateProcess跳过该通知，见[Microsoft官方文档](https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-exitprocess)。",
                  "详细退出码、原日志和诊断结果完整保存，并列入本报告manifest。", ""]
    (args.output / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    table = ["% Auto-generated from audited raw results; do not edit numeric cells.",
             "% B=Baseline, C=CS+H, D=CS+H+Draft-GA. See RESULTS.md for provenance and limitations.",
             r"\begin{table}[t]", r"\centering\small", r"\begin{tabular}{llrrr}", r"\toprule",
             r"Workload / GA & Arm & Tune (s) & Observed best ($\mu$s) & Retimed median ($\mu$s) \\", r"\midrule"]
    for row in sorted(report["runs"], key=lambda r: (GROUPS.index((r["workload"], r["ga"])), r["arm"])):
        timing = next((g for g in report["retiming"] if (g["workload"], g["ga"]) == (row["workload"], row["ga"])), None)
        arm = next((a for a in timing["arms"] if a["name"] == row["arm"]), None) if timing else None
        label = row["workload"].replace("_", r"\_")
        median = f"{arm['median_us']:.4f}" if arm else "--"
        best = f"{row['best_us']:.4f}" if row["best_us"] else "--"
        table.append(f"{label} / {row['ga']} & {row['arm']} & {row['tune_s']:.3f} & {best} & {median} " + r"\\")
    table += [r"\bottomrule", r"\end{tabular}",
              r"\caption{One fixed search seed (20261011), 32 submitted candidates per arm. Independent retiming fixes the original winner and uses 30 technical samples; it does not replace the search observations. B: baseline; C: CS+H; D: CS+H+Draft-GA.}",
              r"\label{tab:single-seed-20261010}", r"\end{table}", ""]
    (args.output / "paper_table.tex").write_text("\n".join(table), encoding="utf-8")
    if args.plot and report["runs"]:
        plot(report, args.output)
    outputs = {str(p): digest(p) for p in args.output.iterdir() if p.is_file() and p.name != "manifest.json"}
    write_json(args.output / "manifest.json", {"script_sha256": digest(__file__),
               "analysis_dependencies": {name: digest(Path(__file__).with_name(name))
                                         for name in ("analyze_p3_mainline.py", "p3_observability.py")},
               "inputs": report["inputs"], "outputs": outputs})
    print(json.dumps({"completed_runs": len(report["runs"]), "pending": report["pending"],
                      "retiming_groups": len(report["retiming"]), "output": str(args.output)}))


if __name__ == "__main__":
    main()
