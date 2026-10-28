# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Reproduce audited pilot figures from existing C++/CUDA experiment results.

No TVM import, compilation, or GPU execution occurs.  Run from any directory:
    F:\\tvm\\.venv-tune\\python.exe experiments/plot_paper_pilot_20261008.py

All plots are descriptive pilot observations, not significance estimates.
Convergence uses batch endpoints; the elapsed axis excludes setup and the
unattributed tune-call residual.  The stage chart includes that residual.
"""

import hashlib
import json
import math
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath

# This plotting process uses only tiny BLAS operations.  The local Conda-style
# environment needs its own DLL directory and sequential MKL initialization.
library_bin = Path(sys.executable).parent / "Library/bin"
if os.name == "nt" and library_bin.is_dir():
    os.environ["PATH"] = str(library_bin) + os.pathsep + os.environ.get("PATH", "")
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_THREADING_LAYER"] = "SEQUENTIAL"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # pylint: disable=wrong-import-position
import numpy as np  # pylint: disable=wrong-import-position


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "papers/figures/pilot_20261008"
SOURCE_BASE = "experiments/results/paper_v1"
CURVES = (
    f"{SOURCE_BASE}/e8_convergence_small/seed20260931/convergence_4x4.json",
    f"{SOURCE_BASE}/e10_seedpair/seed20260932/convergence_4x4.json",
    f"{SOURCE_BASE}/e8_convergence_small/seed20260931/convergence_4x4_draft_ratio.json",
    f"{SOURCE_BASE}/e8_convergence_small/seed20260931/convergence_4x4_h_ablation.json",
)
NAMES = {
    "B": "Baseline", "Baseline": "Baseline", "CSH": "CS+H", "CS+H": "CS+H",
    "CSH_Draft25": "CS+H+Draft 0.25", "CSH_Draft50": "CS+H+Draft 0.50",
    "H_only": "H-only",
}
COLORS = {
    "Baseline": "#0072B2", "CS+H": "#009E73", "CS+H+Draft 0.25": "#E69F00",
    "CS+H+Draft 0.50": "#CC79A7", "H-only": "#777777",
}
VARIED_CONFIG_KEYS = {
    "constraint_sample_mode", "hard_filter_mode", "draft_ga_mode", "draft_ga_verify_ratio",
}
INPUTS = {}


def require(condition, message):
    """Fail rather than silently plot an incompatible or stale input."""
    if not condition:
        raise ValueError(message)


def close(left, right, label):
    require(math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-8), label)


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def digest_json(value):
    return digest_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def read_json(path):
    raw = path.read_bytes()
    INPUTS[path.relative_to(ROOT).as_posix()] = {"sha256": digest_bytes(raw), "bytes": len(raw)}
    return json.loads(raw.decode("utf-8-sig"))


def local_result_path(value):
    """Resolve recorded F:\\tvm paths under the current checkout for portability."""
    parts = PureWindowsPath(value).parts
    require("experiments" in parts, f"Unexpected result path: {value}")
    return ROOT.joinpath(*parts[parts.index("experiments"):])


def audit_arm(arm, result):
    name = f"{result['seed']} {arm['arm']}"
    for key in ("seed", "workload", "source_identity_sha256", "loaded_library_sha256"):
        require(arm[key] == result[key], f"{name}: curve/result {key} mismatch")
    measurement = result["measurement"]
    close(arm["final_best_latency_us"], measurement["best_latency_us"], f"{name}: final best")
    close(arm["tune_call_s"], result["timing"]["tune_call_s"], f"{name}: tune time")
    require(result["architecture_validation"]["passed"], f"{name}: architecture validation")
    points, batches = arm["points"], measurement["batch_summaries"]
    callbacks = measurement["callback_batches"]
    require(len(points) == len(batches) == len(callbacks) == 4, f"{name}: batch count")
    finite_total, best, previous_time = 0, math.inf, -1
    for point, batch, callback in zip(points, batches, callbacks):
        require(all(error is None for error in callback["runner_errors"]),
                f"{name}: runner error in complete pilot")
        samples = [
            sum(samples) / len(samples) * 1e6
            for samples in callback["runner_run_secs"]
            if samples and all(math.isfinite(sample) and sample > 0 for sample in samples)
        ]
        require(len(samples) == batch["finite"] == point["batch_finite"], f"{name}: batch finite")
        close(min(samples), batch["best_latency_us"], f"{name}: callback/batch best")
        close(point["batch_best_latency_us"], batch["best_latency_us"], f"{name}: curve batch best")
        finite_total += len(samples)
        best = min(best, min(samples))
        require(point["measured_trials"] == finite_total, f"{name}: cumulative finite")
        close(point["best_so_far_latency_us"], best, f"{name}: best-so-far")
        require(point["elapsed_tuning_loop_s"] > previous_time, f"{name}: nonmonotone time")
        previous_time = point["elapsed_tuning_loop_s"]
    require(finite_total == measurement["finite_records"] == 16, f"{name}: finite total")
    require(measurement["sent_to_builder"] == measurement["sent_to_runner"] == 16,
            f"{name}: submitted measurement count")
    close(best, min(measurement["record_latencies_us"]), f"{name}: record minimum")
    close(best, measurement["best_latency_us"], f"{name}: callback/final best")
    timing = result["timing"]
    stages = sum(timing[key] for key in ("search_s", "builder_s", "runner_and_update_s"))
    close(stages, previous_time, f"{name}: endpoint/stage total")
    close(stages + timing["unaccounted_inside_tune_s"], timing["tune_call_s"],
          f"{name}: tune decomposition")
    require(timing["unaccounted_inside_tune_s"] >= 0, f"{name}: negative residual")
    totals = result["search"]["totals"]
    accounting = totals["accounting"]
    hard = sum(accounting["hard_rejected"].values())
    close(hard, sum(position["enforced"] for position in totals["hard_positions"].values()),
          f"{name}: hard position total")
    require(sum(accounting["verify_failed"].values()) == accounting["verify_failure_total"],
            f"{name}: Verify failure reasons")
    require(accounting["conservation_ok"] and accounting["conservation_delta"] == 0,
            f"{name}: recorded conservation")
    accounted = (accounting["passed"] + hard + sum(accounting["postproc_failed"].values())
                 + accounting["trace_replay_failed"] + accounting["constraint_sampling_abandoned"])
    require(accounted == accounting["generated"], f"{name}: recomputed conservation")


def load_runs():
    runs, reference = {}, None
    for curve_path in CURVES:
        document = read_json(ROOT / curve_path)
        require(all(document["identity"].values()), f"Input identity check failed: {curve_path}")
        for arm in document["arms"]:
            result_path = local_result_path(arm["result"])
            result = read_json(result_path)
            audit_arm(arm, result)
            config = result["config"]
            common = {key: value for key, value in config.items() if key not in VARIED_CONFIG_KEYS}
            identity = {key: result[key] for key in (
                "workload", "source_identity_sha256", "loaded_library_sha256", "target", "tvm_version")}
            identity["shared_config"] = common
            if reference is None:
                reference = identity
            require(identity == reference, f"Different shared configuration/identity: {result_path}")
            key = (result["seed"], NAMES[arm["arm"]])
            if key in runs:
                require(runs[key]["curve"]["points"] == arm["points"], f"Duplicate curve mismatch: {key}")
                require(runs[key]["path"] == result_path, f"Duplicate source mismatch: {key}")
                runs[key]["curve_inputs"].append(curve_path)
            else:
                runs[key] = {"curve": arm, "result": result, "path": result_path,
                             "curve_inputs": [curve_path]}
    require(len(runs) == 7, "Expected five seed31 arms and two seed32 arms")
    return runs, reference


def outcome_counts(result):
    totals = result["search"]["totals"]
    accounting, positions = totals["accounting"], totals["hard_positions"]
    verify = accounting["verify_failed"]
    counts = {
        "Passed postprocessing": accounting["passed"],
        "Verify: thread constraint": verify["thread_extent"] + verify["threads_per_block"],
        "Verify: shared memory": verify["shared_memory"],
        "Verify: lowering exception": verify["lowering_exception"],
        "H: before replay": positions["trace"]["enforced"],
        "H: after replay": positions["postreplay"]["enforced"] + positions["late"]["enforced"],
    }
    counts["Other failure"] = accounting["generated"] - sum(counts.values())
    require(counts["Other failure"] >= 0, "Candidate outcome partition must be nonnegative")
    return counts


def target_results(curve, baseline_best):
    results = []
    for tolerance in (0, 0.01, 0.02, 0.05):
        threshold = baseline_best * (1 + tolerance)
        hit = next((point for point in curve["points"]
                    if point["best_so_far_latency_us"] <= threshold), None)
        results.append({
            "relative_tolerance": tolerance, "target_latency_us": threshold,
            "reached": hit is not None,
            "first_observed_batch": None if hit is None else hit["batch"],
            "finite_trials_to_target": None if hit is None else hit["measured_trials"],
            "scheduler_elapsed_s_to_target": None if hit is None else hit["elapsed_tuning_loop_s"],
            "observation_end_finite_trials": curve["points"][-1]["measured_trials"],
            "observation_end_scheduler_elapsed_s": curve["points"][-1]["elapsed_tuning_loop_s"],
        })
    return results


def save_figure(figure, name, note):
    figure.text(0.5, 0.015, note, ha="center", va="bottom", fontsize=8, color="#444444")
    for extension in ("png", "svg"):
        figure.savefig(OUTPUT / f"{name}.{extension}", dpi=220, bbox_inches="tight",
                       metadata={"Creator": "plot_paper_pilot_20261008.py"})
    plt.close(figure)


def convergence_figure(runs):
    figure, axes = plt.subplots(2, 2, figsize=(9.6, 6.6), sharey=True)
    for row, seed in enumerate((20260931, 20260932)):
        for name in ("Baseline", "CS+H"):
            points = runs[seed, name]["curve"]["points"]
            latency = [point["best_so_far_latency_us"] for point in points]
            for column, field in enumerate(("measured_trials", "elapsed_tuning_loop_s")):
                axes[row, column].step([point[field] for point in points], latency, where="post",
                                       marker="o", markersize=4, color=COLORS[name], label=name)
        for column in range(2):
            axis = axes[row, column]
            axis.set_title(f"Seed {seed}", fontsize=10)
            axis.set_ylabel("Best measured latency ($\\mu$s)")
            axis.set_ylim(30, 63)
            axis.grid(axis="both", alpha=0.2)
        axes[row, 0].set_xticks((4, 8, 12, 16))
        axes[row, 0].set_xlabel("Cumulative finite measurements")
        axes[row, 1].set_xlabel("Scheduler elapsed time (s; excludes setup)")
        axes[row, 1].set_xlim(100, 515)
    axes[0, 0].legend(frameon=False)
    figure.suptitle("Pilot: MatMul128 best-so-far at four measured batch endpoints", fontsize=12)
    figure.tight_layout(rect=(0, 0.085, 1, 0.95))
    save_figure(figure, "pilot_convergence",
                "16 measurements, batch=4, population=32; no seed pooling or interpolation.\n"
                "Time starts at the first scheduler pick; batch-level time excludes setup and tune residual.")


def candidate_figure(runs):
    figure, axes = plt.subplots(1, 2, figsize=(9.6, 5.8), sharey=True)
    palette = ["#009E73", "#D55E00", "#E69F00", "#CC79A7", "#56B4E9", "#0072B2", "#999999"]
    for axis, seed in zip(axes, (20260931, 20260932)):
        values = [runs[seed, name]["result"] for name in ("Baseline", "CS+H")]
        counts = [outcome_counts(value) for value in values]
        bottom = np.zeros(2)
        for (category, _), color in zip(counts[0].items(), palette):
            heights = np.array([value[category] for value in counts])
            if not heights.any():
                continue
            axis.bar([0, 1], heights, bottom=bottom, width=0.58, label=category,
                     color=color, edgecolor="white", linewidth=0.5)
            for index, height in enumerate(heights):
                if height >= 28:
                    axis.text(index, bottom[index] + height / 2, str(height), ha="center", va="center",
                              fontsize=9, color="white" if color in ("#009E73", "#D55E00") else "black")
            bottom += heights
        for index, value in enumerate(values):
            accounting = value["search"]["totals"]["accounting"]
            ratio = accounting["passed"] / accounting["generated"] * 100
            axis.text(index, accounting["generated"] + 25,
                      f"{accounting['generated']} total\n{ratio:.1f}% passed", ha="center", fontsize=9)
        axis.set_title(f"Seed {seed}", fontsize=10)
        axis.set_xticks([0, 1], ["Baseline", "CS+H"])
        axis.set_ylim(0, 1530)
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)
    axes[0].set_ylabel("Candidate attempts (not unique traces)")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.08), ncol=3,
                  frameon=False, fontsize=8)
    figure.suptitle("Pilot: candidate outcomes under the same 16-measurement budget", fontsize=12)
    figure.tight_layout(rect=(0, 0.19, 1, 0.94))
    save_figure(figure, "pilot_candidate_outcomes",
                "Passed = postprocessing passed, not a GPU measurement or a unique candidate.\n"
                "H positions are distinct; unknown at trace level may become known after replay.")


def timing_figure(runs):
    figure, axes = plt.subplots(1, 2, figsize=(9.6, 4.4), sharex=True)
    stages = [("search_s", "Search (includes Replay/Verify)", "#009E73"),
              ("builder_s", "Builder", "#0072B2"),
              ("runner_and_update_s", "Runner + model update", "#E69F00"),
              ("unaccounted_inside_tune_s", "Tune-call residual", "#BBBBBB")]
    for axis, seed in zip(axes, (20260931, 20260932)):
        left = np.zeros(2)
        for field, label, color in stages:
            widths = np.array([runs[seed, name]["result"]["timing"][field]
                               for name in ("Baseline", "CS+H")])
            axis.barh([0, 1], widths, left=left, height=0.5, color=color, label=label,
                      edgecolor="white", linewidth=0.5)
            for index, width in enumerate(widths):
                axis.text(left[index] + width / 2, index, f"{width:.0f}", ha="center", va="center",
                          color="white" if field in ("search_s", "builder_s") else "black", fontsize=8)
            left += widths
        for index, total in enumerate(left):
            axis.text(total + 5, index, f"{total:.1f}", ha="left", va="center", fontsize=9)
        axis.set_yticks([0, 1], ["Baseline", "CS+H"])
        axis.invert_yaxis()
        axis.set_xlim(0, 605)
        axis.set_xlabel("Tune-call time (s; excludes setup)")
        axis.set_title(f"Seed {seed}", fontsize=10)
        axis.grid(axis="x", alpha=0.2)
        axis.set_axisbelow(True)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.105), ncol=2,
                  frameon=False, fontsize=8)
    figure.suptitle("Pilot: non-overlapping top-level tuning time", fontsize=12)
    figure.tight_layout(rect=(0, 0.25, 1, 0.91))
    save_figure(figure, "pilot_stage_timing",
                "Replay, H, Verify and GA are nested inside Search, never added again.\n"
                "Residual = tune call minus logged Search/Builder/Runner+update; stage logs have second resolution.")


def ablation_figure(runs):
    names = ["Baseline", "CS+H", "CS+H+Draft 0.25", "CS+H+Draft 0.50", "H-only"]
    values = [runs[20260931, name]["result"] for name in names]
    figure, axes = plt.subplots(1, 2, figsize=(9.6, 4.7), sharey=True)
    for axis, data, xlabel in (
            (axes[0], [value["measurement"]["best_latency_us"] for value in values],
             "Best measured latency ($\\mu$s; lower is better)"),
            (axes[1], [value["timing"]["tune_call_s"] for value in values],
             "Tune-call time (s; lower is better)")):
        bars = axis.barh(np.arange(len(names)), data, color=[COLORS[name] for name in names], height=0.6)
        axis.bar_label(bars, labels=[f"{value:.2f}" for value in data], padding=4, fontsize=9)
        axis.set_xlim(0, max(data) * 1.2)
        axis.set_yticks(np.arange(len(names)), names)
        axis.set_xlabel(xlabel, fontsize=9)
        axis.grid(axis="x", alpha=0.2)
        axis.set_axisbelow(True)
    axes[0].invert_yaxis()
    figure.suptitle("Pilot ablation: MatMul128, seed 20260931, 16/16 finite measurements", fontsize=12)
    figure.tight_layout(rect=(0, 0.12, 1, 0.93))
    save_figure(figure, "pilot_draft_ablation",
                "Draft 0.25/0.50 = Draft-GA verification ratio; GA uses no XGBoost in these two arms.\n"
                "Single-seed observations; same source/DLL and shared budget, but different candidate trajectories.")


def make_manifest(runs, reference):
    records = []
    for (seed, name), run in sorted(runs.items()):
        result, curve = run["result"], run["curve"]
        baseline = runs[seed, "Baseline"]["result"]
        accounting = result["search"]["totals"]["accounting"]
        records.append({
            "seed": seed, "arm": name, "result_path": run["path"].relative_to(ROOT).as_posix(),
            "curve_input_paths": run["curve_inputs"],
            "source_identity_sha256": result["source_identity_sha256"],
            "loaded_library_sha256": result["loaded_library_sha256"],
            "config": result["config"], "config_sha256": digest_json(result["config"]),
            "varied_config": {key: result["config"][key] for key in sorted(VARIED_CONFIG_KEYS)},
            "candidate_accounting": accounting,
            "hard_positions": result["search"]["totals"]["hard_positions"],
            "aggregate_filter": result["search"]["totals"]["filter"],
            "plotted_candidate_outcomes": outcome_counts(result),
            "passed_attempt_fraction": accounting["passed"] / accounting["generated"],
            "generated_reduction_vs_baseline": 1 - accounting["generated"] /
            baseline["search"]["totals"]["accounting"]["generated"],
            "verify_failure_reduction_vs_baseline": 1 - accounting["verify_failure_total"] /
            baseline["search"]["totals"]["accounting"]["verify_failure_total"],
            "timing": result["timing"], "finite_records": result["measurement"]["finite_records"],
            "best_latency_us": result["measurement"]["best_latency_us"],
            "latency_change_vs_baseline": result["measurement"]["best_latency_us"] /
            baseline["measurement"]["best_latency_us"] - 1,
            "tune_time_change_vs_baseline": result["timing"]["tune_call_s"] /
            baseline["timing"]["tune_call_s"] - 1,
            "draft_ga": {key: result["search"]["draft_ga"][key] for key in (
                "ga_xgb_calls", "ga_xgb_input", "final_xgb_calls", "final_xgb_input",
                "draft_time_us", "ga_xgb_time_us", "final_xgb_time_us")},
            "convergence_points": curve["points"],
            "time_to_baseline_final": target_results(curve, baseline["measurement"]["best_latency_us"]),
        })
    return {
        "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Existing MatMul128 4x4 pilot observations; no new GPU measurements",
        "script_path": Path(__file__).relative_to(ROOT).as_posix(),
        "script_sha256": digest_bytes(Path(__file__).read_bytes()),
        "python": platform.python_version(), "matplotlib": matplotlib.__version__,
        "inputs": INPUTS, "shared_identity_and_config": reference,
        "shared_config_sha256": digest_json(reference["shared_config"]),
        "validation": {"passed": True, "unique_result_files": len(runs), "curve_files": len(CURVES),
                       "checks": ["source/DLL/workload/target/shared config equal",
                                  "curve vs raw callbacks vs batch summaries: finite counts and all best values",
                                  "candidate accounting and H-position conservation",
                                  "top-level timing sum and curve endpoint equal",
                                  "duplicate curve copies agree"]},
        "measurement_notes": [
            "Source/DLL identities are archived run fields, not a claim about the currently installed binary.",
            "Seed identifiers are RNG integers, not calendar dates.",
            "Same seed and budgets do not imply identical candidate trajectories.",
            "Passed candidate attempts are not deduplicated legal traces or GPU measurements.",
            "Trace-level unknown is preserved independently from aggregate filter unknown.",
            "Convergence is observed at batch endpoints only, with integer-second log resolution.",
            "Scheduler elapsed excludes setup and tune-call residual; total tune time includes residual.",
            "Time-to-target threshold = each seed's baseline final latency times (1+tolerance).",
            "Targets are retrospective pilot targets, not preregistered confirmatory endpoints.",
            "Unreached targets are right-censored at each arm's observation end; null is not zero.",
            "No confidence intervals or aggregate cross-seed means are inferred from these pilot runs.",
        ],
        "runs": records,
    }


def main():
    runs, reference = load_runs()
    print(f"Validated {len(runs)} input runs; drawing convergence", flush=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "svg.hashsalt": "tvm-pilot-20261008"})
    convergence_figure(runs)
    print("Drawing candidate outcomes", flush=True)
    candidate_figure(runs)
    print("Drawing stage timing", flush=True)
    timing_figure(runs)
    print("Drawing Draft ablation", flush=True)
    ablation_figure(runs)
    manifest = make_manifest(runs, reference)
    manifest["outputs"] = {path.name: {"sha256": digest_bytes(path.read_bytes()),
                                      "bytes": path.stat().st_size}
                           for path in sorted(OUTPUT.iterdir()) if path.suffix in (".png", ".svg")}
    require(len(manifest["outputs"]) == 8, "Expected four PNG/SVG output pairs")
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Validated {len(runs)} runs; wrote four PNG/SVG figures and manifest to {OUTPUT}")
    for record in manifest["runs"]:
        targets = [(target["relative_tolerance"], target["finite_trials_to_target"],
                    target["scheduler_elapsed_s_to_target"]) for target in record["time_to_baseline_final"]]
        print(f"{record['seed']} {record['arm']}: best={record['best_latency_us']:.4f} us; "
              f"tune={record['timing']['tune_call_s']:.1f} s; targets={targets}")


if __name__ == "__main__":
    main()
