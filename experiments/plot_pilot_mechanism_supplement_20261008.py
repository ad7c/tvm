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
"""Offline F5/F11/F12 figures from audited, archived pilot measurements.

No TVM import, compilation, sampling or GPU measurement. Run:
    .venv-tune/python.exe experiments/plot_pilot_mechanism_supplement_20261008.py

The online two-seed pilot and the separate frozen-pool CPU experiment are
never pooled. Unknown is an intermediate check result, not a final outcome.
"""

import json
import platform
import re
import statistics
from datetime import datetime, timezone
from pathlib import Path

# This module initializes the known plotting environment before matplotlib.
import plot_paper_pilot_20261008 as pilot
import plot_fixed_trace_pool as frozen
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

plt = pilot.plt
np = pilot.np
ROOT = pilot.ROOT
OUTPUT = ROOT / "papers/figures/pilot_mechanism_20261008"
SEEDS = (20260931, 20260932)
ARMS = ("Baseline", "CS+H")
STAGES = ("sample_init", "evolve", "pick_best")
OUTCOMES = {
    "H before replay": "#0072B2",
    "H after replay": "#56B4E9",
    "Verify rejected": "#E69F00",
    "Other failure": "#999999",
    "Passed postprocessing": "#009E73",
}
COHORTS = {
    "Rejected by trace H": "#0072B2",
    "Verify rejected after H": "#E69F00",
    "Passed postprocessing": "#009E73",
}
OUTPUTS = {}


def read_text_input(path):
    raw = path.read_bytes()
    pilot.INPUTS[path.relative_to(ROOT).as_posix()] = {
        "sha256": pilot.digest_bytes(raw), "bytes": len(raw)
    }
    return raw.decode("utf-8-sig")


def stage_partition(stage):
    accounting = stage["accounting"]
    positions = stage["hard_positions"]
    hard_total = sum(accounting["hard_rejected"].values())
    pilot.require(hard_total == sum(value["enforced"] for value in positions.values()),
                  "Stage hard-rejection conservation")
    verify_total = sum(accounting["verify_failed"].values())
    pilot.require(verify_total == accounting["verify_failure_total"] == stage["verify_gpu_fail"],
                  "Stage Verify-rejection conservation")
    # VerifyGPUCode occupies index 5 in this archived postprocessing chain.
    pilot.require(accounting["postproc_failed"]["5"] == verify_total,
                  "Unexpected Verify postprocessor index")
    other = (accounting["trace_replay_failed"] + accounting["constraint_sampling_abandoned"]
             + sum(value for key, value in accounting["postproc_failed"].items() if key != "5"))
    counts = dict(zip(OUTCOMES, (positions["trace"]["enforced"],
                               positions["postreplay"]["enforced"] + positions["late"]["enforced"],
                               verify_total, other, accounting["passed"])))
    pilot.require(all(value >= 0 for value in counts.values()), "Negative stage outcome")
    pilot.require(sum(counts.values()) == accounting["generated"] == accounting["accounted"]
                  == accounting["calculated_accounted"] and accounting["conservation_ok"]
                  and accounting["conservation_delta"] == 0, "Stage outcome conservation")
    pilot.require(positions["late"]["checked"] == 0,
                  "This figure assumes all after-replay checks are at the postreplay position")
    for value in positions.values():
        pilot.require(0 <= value["unknown"] <= value["checked"], "Invalid unknown count")
    return counts


def audit_online(runs):
    records = []
    fields = ("target", "returned", "generated", "postproc_passed", "discarded", "batches", "time_us")
    pattern = re.compile(r"Exact init population: " + " ".join(
        field + r"=(?P<" + field + r">\d+)" for field in fields))
    for seed in SEEDS:
        quotas = []
        for arm in ARMS:
            run = runs[seed, arm]
            result = run["result"]
            init = result["search"]["exact_init_population"]
            pilot.require(result["config"]["exact_init_population"], "Exact init mode not enabled")
            pilot.require(init["all_targets_met"] and init["calls"] == len(init["events"]) == 4,
                          "Incomplete initialization quota")
            logs = run["path"].parent / "logs/tvm.s_tir.meta_schedule.logging.task_0_main.log"
            logged = [{key: int(value) for key, value in match.groupdict().items()}
                      for match in pattern.finditer(read_text_input(logs))]
            pilot.require(logged == init["events"], "Raw C++ initialization logs disagree with JSON")
            for field in fields:
                pilot.require(sum(event[field] for event in logged) == init[field + "_total"],
                              f"Initialization {field} event sum")
            for event in logged:
                pilot.require(event["target"] == event["returned"]
                              and event["postproc_passed"] == event["returned"] + event["discarded"],
                              "Returned/overflow accounting differs")
                batch_size = min(event["target"], result["config"]["init_sample_batch_size"])
                pilot.require(event["generated"] == event["batches"] * batch_size,
                              "Generation/batch accounting differs")
            pilot.require(init["returned_total"] == init["target_total"] == 112,
                          "Plot requires equal cumulative returned quota of 112")
            quotas.append([event["target"] for event in logged])
            stages = result["search"]["by_stage"]
            pilot.require(set(stages) == set(STAGES), "Unexpected online search stage")
            outcomes = {name: stage_partition(stages[name]) for name in STAGES}
            for field in ("generated", "passed", "trace_replay_failed", "constraint_sampling_abandoned",
                          "verify_failure_total"):
                pilot.require(sum(stage["accounting"][field] for stage in stages.values())
                              == result["search"]["totals"]["accounting"][field],
                              f"Stage/total {field} mismatch")
            pilot.require(stages["sample_init"]["accounting"]["generated"] == init["generated_total"]
                          and stages["sample_init"]["accounting"]["passed"] == init["postproc_passed_total"],
                          "Init/stage counts differ")
            records.append({"seed": seed, "arm": arm,
                            "result_path": run["path"].relative_to(ROOT).as_posix(),
                            "exact_init_population": init,
                            "init_inclusive_seconds": init["time_us_total"] / 1e6,
                            "stages": {name: {"generated": stages[name]["accounting"]["generated"],
                                               "outcomes": outcomes[name],
                                               "hard_positions": stages[name]["hard_positions"],
                                               "verify_failure_reasons": stages[name]["accounting"]["verify_failed"]}
                                       for name in STAGES}})
        pilot.require(quotas[0] == quotas[1] == [32, 28, 26, 26], "Paired init quota mismatch")
    return records


def save(figure, name, note):
    figure.text(0.5, 0.015, note, ha="center", va="bottom", fontsize=8, color="#444444")
    for suffix in ("png", "svg"):
        path = OUTPUT / f"{name}.{suffix}"
        figure.savefig(path, dpi=220, facecolor="white", bbox_inches="tight",
                       metadata={"Creator": Path(__file__).name})
        raw = path.read_bytes()
        OUTPUTS[path.name] = {"sha256": pilot.digest_bytes(raw), "bytes": len(raw)}
    plt.close(figure)


def plot_init(records):
    lookup = {(row["seed"], row["arm"]): row for row in records}
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    positions = np.arange(2)
    width = 0.33
    for index, arm in enumerate(ARMS):
        rows = [lookup[seed, arm] for seed in SEEDS]
        for axis, values, precision in (
                (axes[0], [row["exact_init_population"]["generated_total"] for row in rows], 0),
                (axes[1], [row["init_inclusive_seconds"] for row in rows], 2)):
            bars = axis.bar(positions + (index - 0.5) * width, values, width=width,
                            color=pilot.COLORS[arm], label=arm)
            axis.bar_label(bars, labels=[f"{value:.{precision}f}" for value in values], padding=4)
    for axis in axes:
        axis.set_xticks(positions, [f"Seed {seed}" for seed in SEEDS])
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)
    axes[0].set_ylim(0, 1400)
    axes[1].set_ylim(0, 80)
    axes[0].set_title("(a) Generated initialization attempts")
    axes[0].set_ylabel("Candidate attempts (not unique traces)")
    axes[1].set_title("(b) Inclusive initialization time")
    axes[1].set_ylabel("Sum of four SampleInitPopulation calls (s)")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.115), ncol=2, frameon=False)
    figure.suptitle("F5 | Equal returned initialization quotas: 112 per arm", fontsize=12)
    figure.tight_layout(rect=(0, 0.21, 1, 0.93))
    save(figure, "f5_equal_returned_init",
         "Quotas = 32 + 28 + 26 + 26; overflow passes are discarded (B: 5/6, CS+H: 18/18).\n"
         "Returned schedules are not deduplicated. Initialization time includes replay/postprocessing; two-seed pilot only.")


def plot_stages(records):
    lookup = {(row["seed"], row["arm"]): row for row in records}
    figure, axes = plt.subplots(2, 3, figsize=(12, 7))
    limits = (1400, 170, 21)
    for row_index, seed in enumerate(SEEDS):
        for column, stage in enumerate(STAGES):
            axis = axes[row_index, column]
            stages = [lookup[seed, arm]["stages"][stage] for arm in ARMS]
            left = np.zeros(2)
            for category, color in OUTCOMES.items():
                widths = np.array([item["outcomes"][category] for item in stages])
                axis.barh([0, 1], widths, left=left, height=0.46, color=color,
                          edgecolor="white", linewidth=0.5)
                for index, value in enumerate(widths):
                    if value >= limits[column] * 0.020:
                        axis.text(left[index] + value / 2, index, str(value),
                                  ha="center", va="center", fontsize=8,
                                  color="white" if category in ("H before replay", "Passed postprocessing") else "black")
                left += widths
            for index, value in enumerate(left):
                axis.text(value + limits[column] * 0.014, index, f"N={int(value)}",
                          fontsize=8, ha="left", va="center")
            hard = stages[1]["hard_positions"]
            axis.text(0.01, 0.04,
                      f"H unknown / checked (CS+H):\n"
                      f"trace {hard['trace']['unknown']}/{hard['trace']['checked']}; "
                      f"IR {hard['postreplay']['unknown']}/{hard['postreplay']['checked']}",
                      transform=axis.transAxes, fontsize=8, va="bottom", color="#444444")
            axis.set_yticks([0, 1], ARMS)
            axis.set_ylim(2.2, -0.55)
            axis.set_xlim(0, limits[column])
            axis.set_title(f"{stage} | seed {seed}", fontsize=10)
            axis.set_xlabel("Candidate attempts", fontsize=9)
            axis.grid(axis="x", alpha=0.2)
            axis.set_axisbelow(True)
    handles = [Patch(facecolor=color, label=name) for name, color in OUTCOMES.items()]
    figure.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.085), ncol=5,
                  frameon=False, fontsize=8)
    figure.suptitle("F11 | Candidate outcomes by search stage (four tuning batches)", fontsize=12)
    figure.tight_layout(rect=(0, 0.18, 1, 0.94), h_pad=2.0)
    save(figure, "f11_stage_candidate_outcomes",
         "H is off in Baseline; zero IR rejection is shown in the legend, not omitted from accounting.\n"
         "Unknown is intermediate and overlaps final outcomes. Stage attempts may revisit the same trace; column scales differ.")


def audit_cost_cohorts():
    path = ROOT / "experiments/results/paper_v1/p1_fixed_pool/pilot02/result.json"
    result, audited = frozen.audit(path)
    events = frozen.read_input(path.parent / "evaluations.jsonl", audited["inputs"], jsonl=True)
    pilot.INPUTS.update(audited["inputs"])
    ids = {name: set() for name in COHORTS}
    statuses = {"hard_trace_reject": "Rejected by trace H",
                "postproc_5_VerifyGPUCode_reject": "Verify rejected after H",
                "passed": "Passed postprocessing"}
    for event in events:
        if event["arm"] == "trace" and event["repeat"] == 0:
            pilot.require(event["status"] in statuses, "Unclassified fixed-pool outcome")
            ids[statuses[event["status"]]].add(event["candidate_id"])
    pilot.require([len(value) for value in ids.values()] == [111, 17, 127],
                  "Unexpected fixed-pool cohort sizes")
    repeats = []
    baseline = {}
    for repeat in range(result["repeats"]):
        rows = [event for event in events if event["arm"] == "baseline" and event["repeat"] == repeat]
        total = sum(event["total_s"] for event in rows)
        costs = {name: sum(event["total_s"] for event in rows if event["candidate_id"] in members)
                 for name, members in ids.items()}
        pilot.close(sum(costs.values()), total, "Fixed-pool cost partition")
        baseline[repeat] = {event["candidate_id"]: event for event in rows}
        repeats.append({"repeat": repeat, "baseline_total_s": total, "cohort_path_s": costs,
                        "cohort_path_fraction": {name: cost / total for name, cost in costs.items()}})
    long_tail = []
    for candidate_id in sorted(ids["Verify rejected after H"]):
        verify = [baseline[index][candidate_id]["timing_s"]["postproc_5_VerifyGPUCode"]
                  for index in range(result["repeats"])]
        if statistics.median(verify) > 1.0:
            long_tail.append({"candidate_id": candidate_id, "baseline_verify_s": verify,
                              "median_verify_s": statistics.median(verify),
                              "baseline_path_s": [baseline[index][candidate_id]["total_s"]
                                                  for index in range(result["repeats"])]})
    for index, repeat in enumerate(repeats):
        repeat["long_tail_path_s"] = sum(row["baseline_path_s"][index] for row in long_tail)
        repeat["long_tail_path_fraction"] = repeat["long_tail_path_s"] / repeat["baseline_total_s"]
    return {"result_path": path.relative_to(ROOT).as_posix(), "seed": result["seed"],
            "repeats": result["repeats"], "pool_counts": audited["pool_counts"],
            "oracle_gate_passed": audited["gate_passed"],
            "cohort_candidate_ids": {name: sorted(members) for name, members in ids.items()},
            "cohort_counts": {name: len(members) for name, members in ids.items()},
            "repeat_costs": repeats, "long_tail_definition": "Median baseline Verify time > 1 second",
            "long_tail_candidates": long_tail}


def plot_cost(cohorts):
    figure, axes = plt.subplots(1, 2, figsize=(11, 5.3), sharey=True)
    names = list(COHORTS)
    total = sum(cohorts["cohort_counts"].values())
    for index, (name, color) in enumerate(COHORTS.items()):
        count = cohorts["cohort_counts"][name]
        fraction = 100 * count / total
        axes[0].barh(index, fraction, height=0.48, color=color)
        axes[0].text(fraction + 1.5, index, f"{count}/{total} ({fraction:.1f}%)", va="center", fontsize=9)
        for repeat_index, marker in enumerate(("o", "s", "^")):
            cost = cohorts["repeat_costs"][repeat_index]["cohort_path_fraction"][name] * 100
            y_position = index + (repeat_index - 1) * 0.19
            axes[1].scatter(cost, y_position, color=color, marker=marker, s=35, zorder=3)
            axes[1].text(cost + 1.6, y_position, f"{cost:.2f}%", va="center", fontsize=8)
    for axis in axes:
        axis.set_xlim(0, 105)
        axis.set_xticks([0, 25, 50, 75, 100])
        axis.set_yticks(range(3), ["Rejected by\ntrace H", "Verify rejected\nafter H", "Passed\npostprocessing"])
        axis.grid(axis="x", alpha=0.2)
        axis.set_axisbelow(True)
    axes[0].invert_yaxis()
    axes[0].set_title("(a) Candidate share")
    axes[0].set_xlabel("Share of 255 complete attempts (%)")
    axes[1].set_title("(b) Baseline path-time share, same candidates")
    axes[1].set_xlabel("Share of baseline candidate path time (%)")
    handles = [Line2D([], [], linestyle="none", marker=marker, color="#333333",
                      label=f"Replay round {index + 1}: {cohorts['repeat_costs'][index]['baseline_total_s']:.3f} s")
               for index, marker in enumerate(("o", "s", "^"))]
    figure.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.125), ncol=3,
                  frameon=False, fontsize=8)
    figure.suptitle("F12 | Rejection count does not identify the expensive candidate cohort", fontsize=12)
    figure.tight_layout(rect=(0, 0.23, 1, 0.92))
    save(figure, "f12_fixed_pool_cost_cohorts",
         "One frozen pool (seed 20261012), three technical replay rounds; CS off, no Builder/Runner/GPU timing.\n"
         "Cohorts use trace-H outcomes, then link candidate IDs to Baseline. Verify failure causes are not resolved here.")


def main():
    runs, identity = pilot.load_runs()
    records = audit_online(runs)
    cohorts = audit_cost_cohorts()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "svg.hashsalt": "tvm-pilot-mechanism-20261008"})
    plot_init(records)
    plot_stages(records)
    plot_cost(cohorts)
    scripts = {}
    for path in (Path(__file__), Path(pilot.__file__), Path(frozen.__file__)):
        raw = path.read_bytes()
        scripts[path.relative_to(ROOT).as_posix()] = {
            "sha256": pilot.digest_bytes(raw), "bytes": len(raw)}
    manifest = {
        "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Offline F5/F11 online-pilot plots and separate F12 CPU fixed-pool diagnostic",
        "scripts": scripts, "inputs": pilot.INPUTS, "outputs": OUTPUTS,
        "python": platform.python_version(), "matplotlib": pilot.matplotlib.__version__,
        "online_shared_identity_and_config": identity, "online_plotted_runs": records,
        "fixed_pool_cost_cohorts": cohorts,
        "validation": {"passed": True, "shared_loader_audited_runs": len(runs),
                       "plotted_online_runs": len(records), "checks": [
                           "Existing pilot.load_runs audits all seven source runs and four convergence files.",
                           "Exact initialization events match raw C++ logs and all summary totals.",
                           "Both arms satisfy identical per-call quotas; passed equals returned plus overflow.",
                           "Stage outcome counts are conserved and sum to run accounting totals.",
                           "Frozen-pool audit checks hashes, lineage, all candidate/arm/repeat events and oracle agreement.",
                           "F12 recomputes baseline costs using candidate IDs, preserving all three technical repeats."]},
        "boundaries": [
            "Two online seeds are descriptive pilot observations, not statistical significance evidence.",
            "Equal returned quotas are not equal unique schedules, equal passed counts or identical trajectories.",
            "Initialization time is inclusive; replay/H/Verify times are not added to it.",
            "SampleInitPopulation includes parallel replay/postprocessing, allocation, book-keeping and summary logging.",
            "Stage attempts can revisit prior candidates; there is no conserved unique population flow across stages.",
            "Trace unknown and IR unknown are separate intermediate annotations, not disjoint final outcomes.",
            "F12 is a different CPU mechanism experiment; its seed is not an additional online tuning seed.",
            "F12 times sum candidate paths, excluding pool collection, loading, warmup and structural-equality checks.",
            "F12 cohorts do not classify shared-memory or lowering-exception causes.",
            "F12 long-tail threshold is retrospective exploratory analysis, not a preregistered endpoint.",
            "No GPU speedup or safe whole-space pruning theorem is inferred from these plots.",
        ],
    }
    pilot.require(len(OUTPUTS) == 6, "Expected three PNG/SVG pairs")
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    for row in records:
        init = row["exact_init_population"]
        print(f"{row['seed']} {row['arm']}: returned={init['returned_total']}, "
              f"generated={init['generated_total']}, init_s={row['init_inclusive_seconds']:.6f}")
    print(json.dumps({"fixed_pool_cohort_counts": cohorts["cohort_counts"],
                      "repeat_costs": cohorts["repeat_costs"],
                      "long_tail_ids": [row["candidate_id"] for row in cohorts["long_tail_candidates"]],
                      "output": str(OUTPUT)}, indent=2))


if __name__ == "__main__":
    main()
