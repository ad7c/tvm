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
"""Audit and plot a completed P1 frozen-trace experiment without importing TVM.

Example:
    python experiments/plot_fixed_trace_pool.py --result RUN/result.json --output FIGURES

All plotted quantities are recomputed from the persisted pool and evaluation
journal. These are CPU replay/postprocessing observations, not GPU timings.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PureWindowsPath
import platform
import statistics
import sys

# Match the local plotting environment before importing matplotlib/numpy.
library_bin = Path(sys.executable).parent / "Library/bin"
if os.name == "nt" and library_bin.is_dir():
    os.environ["PATH"] = str(library_bin) + os.pathsep + os.environ.get("PATH", "")
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_THREADING_LAYER"] = "SEQUENTIAL"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # pylint: disable=wrong-import-position


ROOT = Path(__file__).resolve().parents[1]
ARMS = ("baseline", "postreplay", "trace")
STAGES = ("initialization", "mutation")
ARM_LABELS = ("No H", "H after replay", "Before replay H\n+ IR H on survivors")
ARM_COLORS = ("#0072B2", "#E69F00", "#009E73")
OUTCOME_COLORS = {
    "Passed postprocessing": "#009E73",
    "H before replay": "#D55E00",
    "H after replay": "#E69F00",
    "Verify rejection": "#CC79A7",
    "Replay exception": "#56B4E9",
    "Verify exception": "#882255",
    "Other postproc rejection": "#999999",
    "Other postproc exception": "#332288",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, label):
    require(math.isfinite(actual) and math.isfinite(expected)
            and math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9),
            f"{label}: {actual!r} != {expected!r}")


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def json_hash(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def display_path(path):
    path = Path(path).resolve()
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else path.as_posix()


def archived_path(value):
    """Resolve the archived Windows reference under a relocated checkout."""
    path = Path(value)
    if path.is_file():
        return path.resolve()
    parts = PureWindowsPath(value).parts
    require("experiments" in parts, f"Cannot resolve archived reference: {value}")
    return ROOT.joinpath(*parts[parts.index("experiments"):])


def read_input(path, inputs, *, jsonl=False, expected_hash=None):
    raw = path.read_bytes()
    digest = sha256(raw)
    if expected_hash is not None:
        require(digest == expected_hash, f"Hash mismatch: {path}")
    inputs[display_path(path)] = {"sha256": digest, "bytes": len(raw)}
    text = raw.decode("utf-8-sig")
    return [json.loads(line) for line in text.splitlines()] if jsonl else json.loads(text)


def outcome(status):
    fixed = {"passed": "Passed postprocessing", "hard_trace_reject": "H before replay",
             "hard_postreplay_reject": "H after replay", "replay_exception": "Replay exception"}
    if status in fixed:
        return fixed[status]
    require(status.startswith("postproc_"), f"Unrecognized outcome: {status}")
    if status.endswith("_reject"):
        return "Verify rejection" if "_VerifyGPUCode_" in status else "Other postproc rejection"
    require(status.endswith("_exception"), f"Unrecognized postprocessor outcome: {status}")
    return "Verify exception" if "_VerifyGPUCode_" in status else "Other postproc exception"


def audit_event(row, postprocs):
    """Validate observed paths and their time ledger, not merely the final boolean."""
    label = f"{row['arm']} {row['status']}"
    require(row["arm"] in ARMS, f"Unknown arm: {row['arm']}")
    require(type(row["accepted"]) is bool, f"Invalid accepted flag: {label}")
    require(row["accepted"] == (row["status"] == "passed"), f"Outcome flag mismatch: {label}")
    outcome(row["status"])
    times = row["timing_s"]
    require("orchestration_residual" in times, f"Missing residual: {label}")
    require(all(math.isfinite(value) and value >= 0 for value in times.values()),
            f"Invalid component time: {label}")
    require(math.isfinite(row["total_s"]) and row["total_s"] > 0, f"Invalid total: {label}")
    close(sum(times.values()), row["total_s"], f"Exclusive time conservation: {label}")
    trace = row["trace_check"]
    require((trace is not None) == (row["arm"] == "trace"), f"Trace check path: {label}")
    if trace is not None:
        require(trace["truth"] in (0, 1, 2), f"Unknown trace truth: {label}")
        require("plan_and_trace_check" in times, f"Missing trace timing: {label}")
        require((trace["truth"] == 1) == (row["status"] == "hard_trace_reject"),
                f"Trace rejection inconsistent: {label}")
    if row["status"] == "hard_trace_reject":
        require("replay" not in times and row["ir_check"] is None and not row["postprocs"],
                f"Early-rejected row executed later stages: {label}")
    else:
        require("schedule_setup" in times and "replay" in times, f"Missing replay: {label}")
        expected_ir = row["arm"] != "baseline" and row["status"] != "replay_exception"
        require((row["ir_check"] is not None) == expected_ir, f"IR check path: {label}")
        if expected_ir:
            require("postreplay_check" in times, f"Missing IR check timing: {label}")
            require(bool(row["ir_check"][1]) == (row["status"] == "hard_postreplay_reject"),
                    f"IR rejection inconsistent: {label}")
    for index, postproc in enumerate(row["postprocs"]):
        require(postproc["index"] == index and postproc["name"] == postprocs[index],
                f"Postprocessor order differs: {label}")
        key = f"postproc_{index}_{postproc['name']}"
        require(key in times, f"Missing postprocessor timing: {label}")
        if not postproc["passed"]:
            require(index == len(row["postprocs"]) - 1 and row["status"] == key + "_reject",
                    f"Postprocessor rejection differs: {label}")
    if row["accepted"]:
        require(len(row["postprocs"]) == len(postprocs)
                and all(item["passed"] for item in row["postprocs"]),
                f"Accepted candidate missed postprocessing: {label}")


def summarize(events, stage, arm, repeats):
    rows = [row for row in events if row["arm"] == arm
            and (stage == "all" or row["stage"] == stage)]
    rounds = [[row for row in rows if row["repeat"] == repeat] for repeat in range(repeats)]
    totals = [sum(row["total_s"] for row in group) for group in rounds]
    statuses = [dict(Counter(row["status"] for row in group)) for group in rounds]
    require(all(counts == statuses[0] for counts in statuses),
            f"Repeat status counts drift: {stage}/{arm}")
    timings = Counter()
    for row in rows:
        timings.update(row["timing_s"])
    return {
        "stage": stage, "arm": arm, "events": len(rows), "repeat_total_s": totals,
        "median_total_s": statistics.median(totals), "mean_total_s": statistics.mean(totals),
        "status_counts_all_repeats": dict(Counter(row["status"] for row in rows)),
        "stage_mean_s": {key: value / repeats for key, value in timings.items()},
        "trace_unknown_all_repeats": sum(row["trace_check"] is not None
                                        and row["trace_check"]["truth"] == 2 for row in rows),
        "status_counts_per_repeat": statuses,
        "plotted_outcomes_per_repeat": dict(Counter(outcome(row["status"]) for row in rounds[0])),
        "calls_per_repeat": [dict(Counter(key for row in group for key in row["timing_s"]
                                          if key != "orchestration_residual")) for group in rounds],
    }


def compare_summary(actual, recorded):
    label = f"{actual['stage']}/{actual['arm']}"
    for key in ("stage", "arm", "events", "status_counts_all_repeats", "trace_unknown_all_repeats"):
        require(actual[key] == recorded[key], f"Summary mismatch {label}: {key}")
    for key in ("median_total_s", "mean_total_s"):
        close(actual[key], recorded[key], f"Summary {label}: {key}")
    require(len(actual["repeat_total_s"]) == len(recorded["repeat_total_s"]), f"Repeat count: {label}")
    for index, value in enumerate(actual["repeat_total_s"]):
        close(value, recorded["repeat_total_s"][index], f"Summary {label}: repeat {index}")
    require(actual["stage_mean_s"].keys() == recorded["stage_mean_s"].keys(),
            f"Summary timing keys: {label}")
    for key, value in actual["stage_mean_s"].items():
        close(value, recorded["stage_mean_s"][key], f"Summary {label}: {key}")


def rejection_cost_partition(events, stage, repeats):
    """Attribute baseline costs to the exact candidates rejected by trace H."""
    selected = [row for row in events if stage == "all" or row["stage"] == stage]
    early_ids = sorted({row["candidate_id"] for row in selected
                        if row["arm"] == "trace" and row["status"] == "hard_trace_reject"})
    early_set = set(early_ids)
    survivor_ids = sorted({row["candidate_id"] for row in selected} - early_set)
    rows = []
    for repeat in range(repeats):
        partition = {"repeat": repeat}
        for arm in ("baseline", "trace"):
            arm_rows = [row for row in selected if row["arm"] == arm and row["repeat"] == repeat]
            early = sum(row["total_s"] for row in arm_rows if row["candidate_id"] in early_set)
            survivors = sum(row["total_s"] for row in arm_rows if row["candidate_id"] not in early_set)
            early_verify = [value for row in arm_rows if row["candidate_id"] in early_set
                            for key, value in row["timing_s"].items() if "_VerifyGPUCode" in key]
            survivor_verify = [value for row in arm_rows if row["candidate_id"] not in early_set
                               for key, value in row["timing_s"].items() if "_VerifyGPUCode" in key]
            partition[arm] = {"early_rejected_cohort_s": early, "survivor_cohort_s": survivors,
                              "early_cohort_verify_s": sum(early_verify),
                              "survivor_cohort_verify_s": sum(survivor_verify),
                              "early_cohort_verify_calls": len(early_verify),
                              "survivor_cohort_verify_calls": len(survivor_verify),
                              "verify_total_s": sum(early_verify) + sum(survivor_verify),
                              "total_s": early + survivors,
                              "early_cohort_time_fraction": early / (early + survivors)
                              if arm_rows else None}
        rows.append(partition)
    return {"stage": stage, "early_rejected_ids": early_ids, "survivor_ids": survivor_ids,
            "early_rejected_count": len(early_ids), "survivor_count": len(survivor_ids),
            "repeat_costs": rows,
            "interpretation": "Candidate cohorts are defined by trace H, not by latency. "
            "Their baseline cost shows the available filtering opportunity. Replays at different "
            "times are not a noise-free counterfactual; no lowering reason is inferred."}


def audit(result_path):
    inputs = {}
    result = read_input(result_path, inputs)
    directory = result_path.parent
    identity = read_input(directory / "identity.json", inputs)
    pool = read_input(directory / "pool.json", inputs, expected_hash=result["pool_sha256"])
    collection = read_input(directory / "collection.jsonl", inputs, jsonl=True)
    events = read_input(directory / "evaluations.jsonl", inputs, jsonl=True)
    reference = read_input(archived_path(identity["reference_result"]), inputs,
                           expected_hash=identity["reference_sha256"])
    read_input(directory / "workload_tvm.json", inputs, expected_hash=identity["workload_sha256"])
    require(result["schema_version"] == pool["schema_version"] == 1, "Unsupported P1 schema")
    require(result["identity"] == pool["identity"] == identity, "Identity files disagree")
    require(result["seed"] == pool["seed"], "Pool seed differs")
    require(identity["loaded_library_sha256"] == reference["loaded_library_sha256"], "Reference DLL differs")
    require(identity["reference_source_identity_sha256"] == reference["source_identity_sha256"],
            "Reference source identity differs")
    require(identity["workload"] == reference["workload"]
            and json.loads(identity["target"]) == json.loads(reference["target"]),
            "Reference workload/target differs")
    candidates = pool["candidates"]
    require(candidates == collection, "Pool differs from collection journal")
    ids = {row["id"]: row for row in candidates}
    require(len(ids) == len(candidates), "Duplicate candidate ID")
    require(all(row["stage"] in STAGES for row in candidates), "Unknown collection stage")
    requested = {"initialization": result["initial_attempts"], "mutation": result["mutation_attempts"]}
    require(Counter(row["stage"] for row in candidates) == Counter(requested), "Attempt quota differs")
    generation_counts = dict(Counter(row["generation_status"] for row in candidates))
    require(generation_counts == result["generation_status"], "Generation summary differs")
    complete = [row for row in candidates if row["generation_status"] == "complete"]
    for row in candidates:
        require(math.isfinite(row["generation_s"]) and row["generation_s"] >= 0,
                f"Invalid generation time: {row['id']}")
        if row["stage"] == "mutation":
            parent = ids[row["parent_id"]]
            require(parent["stage"] == "initialization" and parent["generation_status"] == "complete"
                    and parent["collection_oracle"]["accepted"], f"Invalid mutation parent: {row['id']}")
            require(row["mutator"] in identity["mutators"] and row["template_id"] == parent["template_id"],
                    f"Mutation metadata differs: {row['id']}")
        else:
            require(row["parent_id"] is None and row["mutator"] is None, "Initialization has a parent")
        if row["generation_status"] == "complete":
            require(json_hash(row["trace_json"]) == row["trace_sha256"], f"Trace hash differs: {row['id']}")
            require(isinstance(row["trace_tvm_json"], str), f"Missing serialized trace: {row['id']}")
            audit_event(row["collection_oracle"], identity["postprocs"])
            require(row["collection_oracle"]["arm"] == "baseline", "Collection oracle is not baseline")
        else:
            require(row["generation_status"] in ("initial_replay_exception", "no_proposal"),
                    f"Unknown generation status: {row['generation_status']}")
            require("trace_sha256" not in row and "collection_oracle" not in row,
                    "Incomplete proposal was presented as a complete trace")
    unique = len({row["trace_sha256"] for row in complete})
    require(unique == result["complete_unique_traces"], "Unique trace count differs")
    close(sum(row["generation_s"] for row in candidates), result["collection_generation_s"], "Generation time")
    close(sum(row["collection_oracle"]["total_s"] for row in complete), result["collection_oracle_s"],
          "Collection oracle time")
    repeats = result["repeats"]
    require(isinstance(repeats, int) and repeats > 0, "Invalid repeat count")
    expected_keys = {(row["id"], arm, repeat) for row in complete for arm in ARMS for repeat in range(repeats)}
    event_keys = [(row["candidate_id"], row["arm"], row["repeat"]) for row in events]
    require(len(event_keys) == len(expected_keys) and set(event_keys) == expected_keys,
            "Candidate/arm/repeat coverage missing or duplicated")
    disagreements, false_rejects, signatures = [], [], {}
    for row in events:
        audit_event(row, identity["postprocs"])
        candidate = ids[row["candidate_id"]]
        oracle = candidate["collection_oracle"]
        require(row["stage"] == candidate["stage"], "Evaluation stage differs from pool")
        repeat = row["repeat"]
        order = ARMS[repeat % 3:] + ARMS[:repeat % 3]
        require(row["arm_order"] == list(order), "Recorded arm rotation differs")
        require(row["oracle_accepted"] == oracle["accepted"], "Recorded oracle label differs")
        if row["accepted"] != oracle["accepted"]:
            disagreements.append({"id": candidate["id"], "repeat": repeat, "arm": row["arm"],
                                  "status": row["status"]})
        if row["status"].startswith("hard_") and oracle["accepted"]:
            false_rejects.append({"id": candidate["id"], "repeat": repeat, "arm": row["arm"]})
        if row["arm"] == "baseline" and row["status"] != oracle["status"]:
            disagreements.append({"id": candidate["id"], "repeat": repeat, "arm": row["arm"],
                                  "status_drift": row["status"]})
        signature = {key: row[key] for key in ("accepted", "status", "trace_check", "ir_check", "postprocs")}
        key = (candidate["id"], row["arm"])
        if key in signatures:
            require(signature == signatures[key], f"Per-candidate repeat outcome drift: {key}")
        signatures[key] = signature
    require(disagreements == result["oracle_disagreements"] and false_rejects == result["false_rejects"],
            "Oracle agreement summary differs")
    gate = not disagreements and not false_rejects
    require(result["gate_passed"] == gate and gate, "Oracle safety gate did not pass; do not plot as validated")
    summaries = [summarize(events, stage, arm, repeats) for stage in ("all", *STAGES) for arm in ARMS]
    recorded = {(item["stage"], item["arm"]): item for item in result["summaries"]}
    require(len(recorded) == len(result["summaries"]) == len(summaries), "Summary arm/stage coverage differs")
    for summary in summaries:
        compare_summary(summary, recorded[(summary["stage"], summary["arm"])])
    pool_stats = {}
    for stage in ("all", *STAGES):
        rows = [row for row in candidates if stage == "all" or row["stage"] == stage]
        ready = [row for row in rows if row["generation_status"] == "complete"]
        pool_stats[stage] = {"attempts": len(rows), "complete": len(ready),
                             "generation_status": dict(Counter(row["generation_status"] for row in rows)),
                             "unique_complete_traces": len({row["trace_sha256"] for row in ready}),
                             "oracle_passed": sum(row["collection_oracle"]["accepted"] for row in ready),
                             "unique_oracle_passed_traces": len({row["trace_sha256"] for row in ready
                                                                 if row["collection_oracle"]["accepted"]})}
    return result, {"inputs": inputs, "identity": identity, "pool_counts": pool_stats,
                    "summaries": summaries, "oracle_disagreements": disagreements,
                    "observed_false_rejects": false_rejects, "gate_passed": gate,
                    "baseline_cost_partition": [rejection_cost_partition(events, stage, repeats)
                                                for stage in ("all", *STAGES)]}


def setup_axis(axis):
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", alpha=0.2)
    axis.set_axisbelow(True)
    axis.set_xticks(range(len(ARMS)), ARM_LABELS)


def save_figure(figure, output, name, outputs):
    for suffix in ("png", "svg"):
        path = output / f"{name}.{suffix}"
        figure.savefig(path, dpi=180, facecolor="white")
        raw = path.read_bytes()
        outputs[path.name] = {"sha256": sha256(raw), "bytes": len(raw)}
    plt.close(figure)


def plot(audited, repeats, output):
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "svg.fonttype": "none"})
    summaries = {(row["stage"], row["arm"]): row for row in audited["summaries"]}
    outputs = {}
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.5))
    for axis, stage in zip(axes, STAGES):
        setup_axis(axis)
        counts = audited["pool_counts"][stage]
        axis.set_title(f"{stage.capitalize()}: {counts['complete']} complete attempts\n"
                       f"{counts['unique_complete_traces']} unique traces / {counts['attempts']} proposals")
        axis.set_ylabel("Sum of candidate evaluation times per round (ms)")
        if counts["complete"] == 0:
            axis.text(0.5, 0.5, "No complete traces to replay", ha="center", transform=axis.transAxes)
            continue
        for index, (arm, color) in enumerate(zip(ARMS, ARM_COLORS)):
            totals = [value * 1000 for value in summaries[(stage, arm)]["repeat_total_s"]]
            offsets = [0.0] if repeats == 1 else [0.24 * i / (repeats - 1) - 0.12 for i in range(repeats)]
            axis.scatter([index + offset for offset in offsets], totals, color=color, s=35, zorder=3)
            median = statistics.median(totals)
            axis.plot([index - 0.22, index + 0.22], [median, median], color="black", lw=2, zorder=4)
            axis.annotate(f"{median:.1f}", (index, max(totals)), xytext=(0, 8),
                          textcoords="offset points", ha="center", fontsize=8)
        axis.set_ylim(bottom=0, top=axis.get_ylim()[1] * 1.13)
    figure.suptitle("Frozen-trace H-position pilot: CPU replay and postprocessing", y=0.99)
    figure.text(0.5, 0.03, f"Dots: {repeats} replay round(s) of one pool, not independent seeds; black bar: median.\n"
                "Collection, setup, deserialization and warmup excluded; Python/FFI overhead included. No GPU measurement.",
                ha="center", va="bottom", fontsize=8)
    figure.tight_layout(rect=(0, 0.15, 1, 0.94))
    save_figure(figure, output, "fixed_pool_round_timing", outputs)

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.9))
    categories = [name for name in OUTCOME_COLORS if any(
        name in summaries[(stage, arm)]["plotted_outcomes_per_repeat"] for stage in STAGES for arm in ARMS)]
    for axis, stage in zip(axes, STAGES):
        setup_axis(axis)
        axis.set_title(stage.capitalize())
        axis.set_ylabel("Complete candidate attempts per replay round")
        bottom = [0] * len(ARMS)
        for category in categories:
            values = [summaries[(stage, arm)]["plotted_outcomes_per_repeat"].get(category, 0) for arm in ARMS]
            axis.bar(range(len(ARMS)), values, bottom=bottom, width=0.62,
                     label=category, color=OUTCOME_COLORS[category])
            for index, value in enumerate(values):
                if value:
                    axis.text(index, bottom[index] + value / 2, str(value), ha="center", va="center", fontsize=8)
            bottom = [left + right for left, right in zip(bottom, values)]
        require(all(value == audited["pool_counts"][stage]["complete"] for value in bottom),
                f"Plot outcome conservation: {stage}")
        if not audited["pool_counts"][stage]["complete"]:
            axis.text(0.5, 0.5, "No complete traces", ha="center", transform=axis.transAxes)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.045), ncol=3,
                  frameon=False, fontsize=8)
    figure.suptitle("Rejection positions on the identical frozen pool", y=0.99)
    figure.text(0.5, 0.015, "Counts identical across all audited repeats. Passed postprocessing does not certify GPU correctness.",
                ha="center", fontsize=8)
    figure.tight_layout(rect=(0, 0.19, 1, 0.94))
    save_figure(figure, output, "fixed_pool_rejection_positions", outputs)
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result, audited = audit(args.result.resolve())
    args.output.mkdir(parents=True, exist_ok=True)
    outputs = plot(audited, result["repeats"], args.output)
    manifest = {
        "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Completed-trace CPU replay/postprocessing pilot; no new GPU measurements",
        "script_path": display_path(__file__), "script_sha256": sha256(Path(__file__).read_bytes()),
        "python": platform.python_version(), "matplotlib": matplotlib.__version__,
        "result_path": display_path(args.result), "seed": result["seed"], "repeats": result["repeats"],
        "excluded_costs_s": {key: result[key] for key in ("setup_s", "collection_s", "collection_generation_s",
                                                         "collection_oracle_s", "warmup_s")},
        "validation_checks": [
            "Pool, workload and reference file hashes match archived identity/result.",
            "Pool equals collection journal; candidate IDs, trace hashes, lineage and quotas are checked.",
            "Exactly one evaluation exists per complete candidate, arm and repeat.",
            "Per-candidate outcomes and rejection counts are stable across repeats.",
            "Oracle agreement and observed hard-filter false rejections are recomputed; gate passed.",
            "All recorded summary fields and exclusive per-candidate time ledgers are recomputed.",
        ],
        "measurement_boundaries": [
            "One seeded proposal pool, not the historical online GA trajectory or independent tuning seeds.",
            "Only complete frozen traces enter replay comparisons; incomplete attempts remain in pool counts.",
            "Initialization required collection replay to obtain complete decisions; this cost was not saved.",
            "CS is disabled in the frozen-pool comparison; these figures isolate H position, not CS effects.",
            "Each round is a sum of candidate total_s, not the entire driver process wall clock.",
            "Time includes schedule setup, plan construction, Python/FFI orchestration and applicable postprocessors.",
            "Pool collection, warmup, trace deserialization and structural-equality checks are outside plotted timing.",
            "Structural output equivalence was checked by the experiment, not re-executed by this plotting script.",
            "Verify rejection is not separated into shared-memory, lowering or other resource causes by this FFI.",
            "Zero observed false rejections applies to this pool and baseline oracle, not the whole search space.",
            "Repeated rounds are not independent seeds; no significance or GPU speedup is inferred.",
            "Archived source/DLL identity is checked against the reference, not asserted to equal currently loaded code.",
            "Evaluation journal receives a content hash here; the original schema pins the pool but not this journal.",
        ],
        **audited, "outputs": outputs,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                                               encoding="utf-8")
    print(json.dumps({"gate_passed": audited["gate_passed"], "pool_counts": audited["pool_counts"],
                      "output": str(args.output.resolve())}, indent=2))


if __name__ == "__main__":
    main()
