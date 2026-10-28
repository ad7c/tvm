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
"""Audit a cohort of complete operator runs without joining interrupted searches."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics

from report_operator800_study import WORKLOADS, markdown, plots, training_progress
from report_single_seed_study import audit, audit_retiming, digest, target_point
from operator800_postcheck import revalidate
from run_operator800_study import command
from watch_operator800_study import complete_lines, read_json, write_json_atomic


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def matching_receipt(root, phase, identifier):
    ledger = root / "execution.jsonl"
    matches = [(i, entry) for i, entry in enumerate(complete_lines(ledger))
               if entry.get("phase") == phase and entry.get("id") == identifier]
    if len(matches) != 1 or matches[0][1].get("returncode") != 0:
        raise ValueError(f"No unique successful exit receipt: {phase} {identifier} in {root}")
    index, entry = matches[0]
    return {"ledger_path": str(ledger.resolve()), "line_index": index,
            "entry_sha256": object_hash(entry)}


def check_receipt(reference):
    entries = complete_lines(Path(reference["ledger_path"]))
    try:
        entry = entries[reference["line_index"]]
    except IndexError as error:
        raise ValueError("Missing execution receipt") from error
    if (object_hash(entry) != reference["entry_sha256"] or entry.get("returncode") != 0
            or entry.get("guard_status", "completed") != "completed"):
        raise ValueError("Execution receipt changed or child did not exit successfully")
    return entry


def check_console(root, entry, phase, identifier):
    name = identifier + ".console.log" if phase in ("tuning", "smoke") else identifier + "." + phase + ".log"
    path = root / name
    if digest(path) != entry["console_sha256"]:
        raise ValueError(f"Console log changed: {path}")
    return path


def check_protocol(root, protocol):
    if read_json(root / "protocol.json") != protocol:
        raise ValueError(f"Attempt protocol differs from the frozen cohort: {root}")


def accept_tuning(root, spec, protocol):
    check_protocol(root, protocol)
    if (root / "resume_state.json").exists():
        from operator800_history_resume import accept_resumed_tuning
        return accept_resumed_tuning(root, spec, protocol)
    paths = list(root.glob(spec["id"] + "_*/result.json"))
    if len(paths) != 1:
        raise ValueError(f"Expected one result for {spec['id']}, found {len(paths)}")
    path = paths[0]
    if (spec["workload"] == "reduction4096"
            and read_json(path)["budget"]["terminal_status"] == "early_exhaustion"):
        from operator800_exhaustion_audit import accept_exhausted
        return accept_exhausted(root, spec, protocol)
    phase = "smoke" if spec["smoke"] else "tuning"
    entries = [(index, value) for index, value in enumerate(complete_lines(root / "execution.jsonl"))
               if value.get("phase") == phase and value.get("id") == spec["id"]]
    if len(entries) != 1:
        raise ValueError("No unique tuning exit receipt")
    index, entry = entries[0]
    if entry.get("returncode") == 0:
        receipt = matching_receipt(root, phase, spec["id"])
        entry = check_receipt(receipt)
    elif entry.get("returncode") == 1:
        receipt = {"ledger_path": str((root / "execution.jsonl").resolve()), "line_index": index,
                   "entry_sha256": object_hash(entry)}
    else:
        raise ValueError("Tuning process did not exit successfully or with the known final postcheck")
    if entry["argv"] != command(spec, root, protocol["seed"]):
        raise ValueError("Tuning command differs from the registered command")
    console = check_console(root, entry, entry["phase"], spec["id"])
    result = read_json(path)
    derived = None
    if entry["returncode"] == 0:
        row = audit(path, spec, protocol)
    else:
        derived, row = revalidate(path, root, spec, protocol, entry)
    if row["budget"]["actual_submitted"] != spec["trials"]:
        raise ValueError(f"Short budget: {row['budget']['actual_submitted']}/{spec['trials']}; "
                         "inspect search-space exhaustion, do not pad or mark complete")
    if row["best_us"] is None or not result["config"].get("record_model_training"):
        raise ValueError("Missing finite winner or model-training monitoring")
    if result["config"].get("xgb_adaptive_training") != protocol["common_config"]["adaptive_training"]:
        raise ValueError("Model update policy differs")
    events_path = path.with_name("timeline.jsonl")
    row["model_progress"] = training_progress(result, row, complete_lines(events_path))
    artifacts = [path, events_path, path.with_name("callback_batches.jsonl"),
                 path.with_name("database_workload.json"), path.with_name("database_tuning_record.json"),
                 path.with_name("correctness.jsonl"), path.with_name("identity.json"), console]
    if derived:
        artifacts.extend([derived, root / "resource_guard/status.json", path.with_name("runner_inputs.jsonl")])
    records = complete_lines(path.with_name("database_tuning_record.json"))
    if len(records) != row["budget"]["actual_submitted"]:
        raise ValueError("Database record count differs from the complete callback budget")
    reference = {"root": str(root.resolve()), "result_path": str(path.resolve()),
            "receipt": receipt, "submitted": row["budget"]["actual_submitted"],
            "artifacts_sha256": {str(p.resolve()): digest(p) for p in artifacts}}
    if derived:
        reference.update(completion_basis="independently_revalidated_postcheck",
                         original_returncode=1, postcheck_audit_path=str(derived.resolve()))
    return reference


def load_tuning(reference, spec, protocol):
    root = Path(reference["root"])
    checked = accept_tuning(root, spec, protocol)
    if checked != reference:
        raise ValueError(f"Accepted result or provenance changed: {spec['id']}")
    path = Path(reference["result_path"])
    result = read_json(path)
    row = audit(Path(reference.get("postcheck_audit_path", path)), spec, protocol)
    row.update(result_path=str(path), result_sha256=digest(path))
    if reference.get("completion_basis") == "verified_history_replay_continuation":
        row["resumption"] = result["resumption"]
    if reference.get("completion_basis") == "verified_early_termination":
        row["early_termination_audit"] = result["early_termination_audit"]
    if reference.get("postcheck_audit_path"):
        row["postcheck_revalidation"] = read_json(Path(reference["postcheck_audit_path"]))["postcheck_revalidation"]
    row["model_progress"] = training_progress(result, row, complete_lines(path.with_name("timeline.jsonl")))
    row["own_best"] = target_point(row, row["best_us"])
    return row


def accept_phase(root, phase, workload, path, rows, protocol):
    receipt = matching_receipt(root, phase, workload)
    entry = check_receipt(receipt)
    console = check_console(root, entry, phase, workload)
    result = read_json(path)
    if result.get("status") != "completed":
        raise ValueError(f"Incomplete {phase}: {path}")
    if phase == "retiming":
        if len(rows) != 3 or {row["arm"] for row in rows} != set("BCD"):
            raise ValueError("Retiming requires all three accepted arms")
        inputs = audit_retiming(result, rows, protocol, path.parent)
        # The original selected database index and archived workload must still match.
        expected = {row["arm"]: row for row in rows}
        for arm in result["arms"]:
            original = expected[arm["name"]]
            if Path(arm["result_path"]).resolve() != Path(original["result_path"]).resolve():
                raise ValueError("Retiming winner came from a different attempt")
    elif phase == "pytorch":
        samples = result["samples_us"]
        if (len(samples) != 30 or len(result["checks"]) != 4
                or not all(c["passed"] for c in result["checks"])
                or any(not math.isfinite(v) or v <= 0 for v in samples)
                or result["median_us"] != statistics.median(samples)
                or result["workload"] != workload or result["tf32"]
                or result["script_sha256"] != protocol["script_sha256"]["benchmark_torch_operators.py"]
                or result["reference_sha256"] != protocol["script_sha256"]["clean_workloads.py"]):
            raise ValueError("PyTorch reference audit failed")
        inputs = {}
    else:
        raise ValueError(f"Unknown measurement phase {phase}")
    inputs.update({str(path.resolve()): digest(path), str(console.resolve()): digest(console)})
    return {"root": str(root.resolve()), "result_path": str(path.resolve()),
            "receipt": receipt, "artifacts_sha256": inputs}


def check_phase(reference, phase, workload, rows, protocol):
    checked = accept_phase(Path(reference["root"]), phase, workload,
                           Path(reference["result_path"]), rows, protocol)
    if checked != reference:
        raise ValueError(f"Accepted {phase} provenance changed: {workload}")
    return read_json(Path(reference["result_path"]))


def collect_cohort(manifest_path):
    manifest = read_json(manifest_path)
    protocol_path = Path(manifest["protocol_path"])
    if digest(protocol_path) != manifest["protocol_sha256"]:
        raise ValueError("Frozen protocol hash changed")
    protocol = read_json(protocol_path)
    report = {"seed": protocol["seed"], "protocol_sha256": manifest["protocol_sha256"],
              "generated_utc": utc_now(), "runs": [], "pending": [], "retiming": [],
              "pytorch": [], "targets": [], "input_sha256": {
                  str(manifest_path.resolve()): digest(manifest_path),
                  str(protocol_path.resolve()): digest(protocol_path)},
              "notes": ["One search seed; resumed arms carry explicit history and continuation provenance.",
                        "Failures consume submission budget; interrupted attempts are retained but excluded.",
                        "Convergence uses actual measurements at feedback-batch completion, without interpolation.",
                        "Targets use the baseline's observed final latency, evaluated retrospectively.",
                        "Independent retiming freezes the original winner and uses 30 technical samples.",
                        "Training fit is separate from pre-update predictions on selected new-batch feedback.",
                        "UpdateCostModel includes feature extraction, validation, and training; it is inside Callbacks.",
                        "PyTorch CUDA-event timing includes Python launch gaps and Conv2D output allocation."]}
    for spec in protocol["order"]:
        if spec["smoke"]:
            continue
        reference = manifest["accepted"].get(spec["id"])
        if reference is None:
            report["pending"].append(spec["id"])
            continue
        row = load_tuning(reference, spec, protocol)
        report["runs"].append(row)
        if "resumption" in row:
            report["notes"].append(
                spec["id"] + ": verified history continuation; active Tune joins historical and new segments. "
                "State reconstruction and pause are separate; native internal prediction timings include reconstruction.")
        if "early_termination_audit" in row:
            report["notes"].append(
                spec["id"] + ": valid early termination below the 800-submission upper limit; actual count retained.")
        report["input_sha256"].update(reference["artifacts_sha256"])
        report["input_sha256"][reference["receipt"]["ledger_path"]] = digest(Path(reference["receipt"]["ledger_path"]))
    for workload in WORKLOADS:
        rows = [row for row in report["runs"] if row["workload"] == workload]
        baseline = next((row for row in rows if row["arm"] == "B"), None)
        if baseline:
            for row in rows:
                for ratio in protocol["targets"]["relative_to_baseline_final"]:
                    target = baseline["best_us"] * ratio
                    point = target_point(row, target)
                    report["targets"].append({"workload": workload, "arm": row["arm"],
                        "ratio": ratio, "target_us": target, **point,
                        "speedup_vs_baseline_full_tune": baseline["tune_s"] / point["tune_s"]
                            if point["attained"] else None})
        for phase in ("retiming", "pytorch"):
            reference = manifest[phase].get(workload)
            if reference:
                result = check_phase(reference, phase, workload, rows, protocol)
                report["input_sha256"].update(reference["artifacts_sha256"])
                report["input_sha256"][reference["receipt"]["ledger_path"]] = digest(Path(reference["receipt"]["ledger_path"]))
                report[phase].append({"workload": workload, "arms": result["arms"]}
                                     if phase == "retiming" else result)
    report["completion_ready"] = (not report["pending"] and len(report["runs"]) == 12
                                  and len(report["retiming"]) == 4 and len(report["pytorch"]) == 4)
    return report


def write_report(manifest_path, output, no_plots=False):
    report = collect_cohort(manifest_path)
    output.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output / "summary.json", report)
    (output / "RESULTS.md").write_text(markdown(report), encoding="utf-8")
    keys = ["workload", "arm", "submitted", "data_size", "last_train_size", "update_s", "boosted_rounds",
            "best_iteration", "train_rmse", "train_peak32", "next_batch_rmse", "next_batch_peak32", "tune_s"]
    with (output / "model_training.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in report["runs"]:
            for point in row["model_progress"]:
                writer.writerow({"workload": row["workload"], "arm": row["arm"], **point})
    if not no_plots:
        plots(report, output)
    write_json_atomic(output / "manifest.json", {"report_script_sha256": digest(Path(__file__)),
        "plot_script_sha256": digest(Path(__file__).with_name("report_operator800_study.py")),
        "protocol_sha256": report["protocol_sha256"], "input_sha256": report["input_sha256"],
        "completion_ready": report["completion_ready"],
        "output_sha256": {p.name: digest(p) for p in output.iterdir()
                          if p.is_file() and p.name != "manifest.json"}})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    report = write_report(args.manifest.resolve(), args.output.resolve(), args.no_plots)
    print(f"Audited {len(report['runs'])}/12; completion_ready={report['completion_ready']}")


if __name__ == "__main__":
    main()
