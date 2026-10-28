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
"""Analyze one preregistered P3 seed from completed raw result files, never resample."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics

from p3_observability import classify_measurements, write_json


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def arm_name(result):
    switches = (result["mode"], result["constraint_sample_mode"], result["draft_ga_mode"])
    return {("off", "off", "off"): "B", ("enforce", "enforce", "off"): "C",
            ("enforce", "enforce", "enforce"): "D"}[switches]


def summarize_run(result, protocol):
    if not result["architecture_validation"]["passed"]:
        raise ValueError("Invalid run: retain its artifacts, but do not present a valid comparison")
    if not result["config"]["check_correctness"] or not result["config"]["record_timeline"]:
        raise ValueError("P3 requires numerical checking and monotonic timestamps")
    if result["loaded_library_sha256"] != protocol["loaded_library_sha256"]:
        raise ValueError("DLL differs from preregistered P3 identity")
    if result["source_identity_sha256"] != protocol["source_identity_sha256"]:
        raise ValueError("Source differs from preregistered P3 identity")
    for key in ("trials", "trials_per_iter", "population_size", "init_min_unmeasured",
                "init_sample_batch_size", "exact_init_population", "genetic_iters",
                "eps_greedy", "xgb_warmup_samples", "hard_filter_position"):
        if result["config"][key] != protocol["config"][key]:
            raise ValueError(f"Protocol mismatch: {key}")
    if result["workload"] != protocol["workload"]:
        raise ValueError("Workload name differs")
    for name in ("soft_mode", "draft_mode", "ga_t2_mode", "draft_ga_g3_mode", "draft_ga_anneal_mode"):
        if result[name] != "off":
            raise ValueError(f"Unplanned optimization: {name}")
    if arm_name(result) == "D" and result["draft_ga_verify_ratio"] != protocol["next_existing_draft_arm"]["draft_ga_verify_ratio"]:
        raise ValueError("Draft ratio differs from protocol")
    submitted, finite, best = 0, 0, None
    curve, outcomes = [], Counter()
    batches = result["measurement"]["callback_batches"]
    for batch in batches:
        kinds = classify_measurements(batch)
        outcomes.update(kinds)
        submitted += len(kinds)
        for kind, values in zip(kinds, batch["runner_run_secs"]):
            if kind == "finite":
                finite += 1
                latency = statistics.median(values) * 1e6
                best = latency if best is None else min(best, latency)
        curve.append({"submitted": submitted, "finite": finite, "best_us": best,
                      "tune_s": batch["available_at_tune_s"]})
    if submitted != result["budget"]["actual_submitted"]:
        raise ValueError("Submission count differs")
    if finite != result["measurement"]["finite_records"] or finite != result["correctness"]["finite_checked"]:
        raise ValueError("Finite/correctness coverage mismatch")
    best_reported = result["measurement"]["best_latency_us"]
    if best != best_reported:
        raise ValueError("Callback and database best latency differ")
    phases = {key: sum(row[key] for row in result["timeline"]["batches"])
              for key in ("search_s", "builder_s", "runner_and_notify_s", "callbacks_s")}
    phases["residual_s"] = result["timeline"]["residual_s"]
    ttt = {}
    for target in protocol["absolute_target_us"]:
        hit = next((row for row in curve if row["best_us"] is not None and row["best_us"] <= target), None)
        ttt[str(target)] = {"attained": hit is not None, "submitted": hit["submitted"] if hit else None,
                            "tune_s": hit["tune_s"] if hit else None,
                            "observed_until_s": result["timing"]["tune_call_s"]}
    snapshots = {}
    for cutoff in protocol["tune_snapshot_s"]:
        eligible = [row for row in curve if row["tune_s"] <= cutoff and row["best_us"] is not None]
        snapshots[str(cutoff)] = eligible[-1]["best_us"] if eligible else None
    return {"arm": arm_name(result), "seed": result["seed"], "workload": result["workload"],
            "budget": result["budget"], "outcomes": dict(outcomes), "curve": curve,
            "best_us": best, "tune_s": result["timing"]["tune_call_s"], "phases_s": phases,
            "targets": ttt, "snapshots_us": snapshots,
            "correctness_s": result["correctness"]["correctness_wall_s"],
            "search_totals": result["search"]["totals"],
            "draft_ga": result["search"]["draft_ga"],
            "xgb_data_size": result["measurement"]["xgb_data_size"]}


def plot_report(report, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"B": "#0072B2", "C": "#009E73", "D": "#E69F00"}
    labels = {"B": "Baseline", "C": "CS+H", "D": "CS+H+Draft-GA"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for row in report["runs"]:
        for axis, key in zip(axes, ("submitted", "tune_s")):
            points = [point for point in row["curve"] if point["best_us"] is not None]
            axis.step([p[key] for p in points], [p["best_us"] for p in points], where="post",
                      marker="o", markersize=3, color=colors[row["arm"]], label=labels[row["arm"]])
    axes[0].set_xlabel("Submitted candidates (failures included)")
    axes[1].set_xlabel("Full tuning wall time (s)")
    for axis in axes:
        axis.set_ylabel("Best measured latency (us; lower is better)")
        axis.grid(alpha=0.2)
        axis.legend()
    axes[1].axhline(40, color="0.5", linestyle=":", linewidth=1)
    title = f"MatMul128 | seed {report['runs'][0]['seed']} | fixed 32-submission budget"
    fig.suptitle(title)
    fig.text(0.5, 0.01, "Development run(s); batch-completion timestamps. Independent top-1 remeasurement not included.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig.savefig(output / "convergence.png", dpi=180)
    fig.savefig(output / "convergence.svg")
    plt.close(fig)
    keys = ["search_s", "builder_s", "runner_and_notify_s", "callbacks_s", "residual_s"]
    names = ["Search", "Builder", "Runner + notify", "Callbacks", "Setup / residual"]
    fig, axis = plt.subplots(figsize=(7.4, 3.6))
    bottoms = [0.0] * len(report["runs"])
    for key, label in zip(keys, names):
        values = [row["phases_s"][key] for row in report["runs"]]
        axis.bar([labels[row["arm"]] for row in report["runs"]], values, bottom=bottoms, label=label)
        bottoms = [a + b for a, b in zip(bottoms, values)]
    axis.set_ylabel("Full tuning wall time (s)")
    axis.legend(loc="upper left", bbox_to_anchor=(1, 1))
    axis.set_title(f"Observed, non-overlapping phases | seed {report['runs'][0]['seed']}")
    fig.tight_layout()
    fig.savefig(output / "phases.png", dpi=180)
    fig.savefig(output / "phases.svg")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    originals = [json.loads(path.read_text(encoding="utf-8")) for path in args.runs]
    rows = [summarize_run(result, protocol) for result in originals]
    if len({row["seed"] for row in rows}) != 1 or len({row["arm"] for row in rows}) != len(rows):
        raise ValueError("Expected unique arms of one seed")
    for path in args.runs:
        identity = json.loads(path.with_name("identity.json").read_text(encoding="utf-8"))
        if identity["workload_sha256"] != protocol["workload_sha256"]:
            raise ValueError("Workload identity changed")
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"schema_version": 1, "runs": rows,
              "has_B_C_pair": {"B", "C"}.issubset({row["arm"] for row in rows}),
              "protocol_sha256": sha256(args.protocol),
              "input_sha256": {str(path.resolve()): sha256(path) for path in args.runs},
              "analyzer_sha256": sha256(__file__),
              "limitations": ["Single development seed, not a significance test.",
                              "Search-time measured best, not independently remeasured final performance.",
                              "Fixed submitted budget, not a wall-clock stopping experiment."]}
    write_json(args.output / "summary.json", report)
    if args.plot:
        plot_report(report, args.output)
    print(json.dumps([{key: row[key] for key in ("arm", "budget", "best_us", "tune_s", "targets")}
                      for row in rows], indent=2))


if __name__ == "__main__":
    main()
