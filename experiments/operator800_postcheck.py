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
"""Revalidate a known post-search batch-count error without changing frozen code.

The original process exit and result remain untouched. Only the fixed-size
callback assumption can be superseded, after independent evidence checks.
"""

import copy
import hashlib
import math
from pathlib import Path
import statistics

from p3_observability import classify_measurements
from report_single_seed_study import audit, digest
from watch_operator800_study import complete_lines, read_json, write_json_atomic


FIELDS = ("builder_errors", "runner_errors", "runner_run_secs")


def validate_batches(batches, search_batches, trials, maximum, effective):
    """Check actual batch sizes, allowing short intermediate batches."""
    errors = []
    if not 0 <= effective <= trials or maximum <= 0:
        errors.append("Invalid effective submission budget or batch maximum")
    if len(batches) != search_batches:
        errors.append("Search/callback batch coverage differs")
    total = 0
    empty_seen = False
    for index, batch in enumerate(batches):
        lengths = [len(batch[name]) for name in FIELDS]
        if len(set(lengths)) != 1:
            errors.append(f"Batch {index}: callback fields differ in length")
        count = lengths[1]
        if count > maximum:
            errors.append(f"Batch {index}: exceeds batch maximum")
        if count and empty_seen:
            errors.append(f"Batch {index}: nonempty callback follows exhaustion")
        empty_seen |= count == 0
        total += count
    if total != effective:
        errors.append(f"Callback submissions {total} differ from records {effective}")
    return errors


def callback_only_failure(result):
    # Reproduce the frozen validator exactly; a different error cannot use this recovery.
    from e1_thread_filter import _validate_callback_batch_contract

    architecture = result["architecture_validation"]
    errors = _validate_callback_batch_contract(
        result["measurement"]["callback_batches"], result["search"]["search_batches"],
        result["config"]["trials"], result["config"]["trials_per_iter"],
        result["measurement"]["database_records"])
    if (architecture["passed"] or not errors or architecture["errors"] != errors
            or architecture["callback_batch_errors"] != errors):
        raise ValueError("Not solely the known fixed-size callback postcheck failure")
    return errors


def validate_evidence(path, result):
    measurement = result["measurement"]
    batches = measurement["callback_batches"]
    actual = measurement["database_records"]
    errors = validate_batches(batches, result["search"]["search_batches"],
                              result["config"]["trials"],
                              result["config"]["trials_per_iter"], actual)
    if errors:
        raise ValueError("; ".join(errors))
    if complete_lines(path.with_name("callback_batches.jsonl")) != batches:
        raise ValueError("Archived callback stream differs from embedded callbacks")
    records = complete_lines(path.with_name("database_tuning_record.json"))
    if len(records) != actual or result["budget"]["actual_submitted"] != actual:
        raise ValueError("Database/callback/budget submission counts differ")
    database_latencies = []
    for record in records:
        values = record[1][1]
        usable = values and all(math.isfinite(value) and 0 < value < 1e9 for value in values)
        database_latencies.append(statistics.median(values) * 1e6 if usable else None)
    # get_all_tuning_records returns records ranked by latency, whereas the
    # JSONL database preserves insertion order. Compare the actual multiset.
    if (sum(value is None for value in database_latencies)
            != sum(value is None for value in measurement["record_latencies_us"])
            or sorted(value for value in database_latencies if value is not None)
            != sorted(value for value in measurement["record_latencies_us"] if value is not None)):
        raise ValueError("Raw database timings differ from archived measurement timings")
    if sorted(trace for batch in batches for trace in batch["trace_sha256"]) != sorted(measurement["trace_hashes"]):
        raise ValueError("Callback/database trace identities differ")
    phases = result["timeline"]["batches"]
    if len(phases) != len(batches):
        raise ValueError("Timeline does not cover every feedback batch")
    previous = 0
    observed = []
    inputs = complete_lines(path.with_name("runner_inputs.jsonl"))
    artifact_ids = {item["artifact_path"]: item["candidate_id"] for item in inputs}
    if len(artifact_ids) != len(inputs) or len({item["candidate_id"] for item in inputs}) != len(inputs):
        raise ValueError("Duplicate runner identity")
    checks = [item for item in complete_lines(path.with_name("correctness.jsonl"))
              if item["event"] == "correctness_result"]
    passed = {item["candidate_id"] for item in checks
              if item["passed"] and len(item["cases"]) == 4
              and all(case["passed"] for case in item["cases"])}
    if len(checks) != len({item["candidate_id"] for item in checks}):
        raise ValueError("Duplicate correctness result")
    for index, (batch, phase) in enumerate(zip(batches, phases)):
        count = len(batch["runner_errors"])
        if (phase["submitted"] != count or phase["batch_index"] != index
                or batch["timeline_batch_index"] != index
                or len(batch["artifact_paths"]) != count
                or len(batch["trace_sha256"]) != count):
            raise ValueError("Timeline, artifact or trace coverage differs from callback")
        when = batch["available_at_tune_s"]
        if (not math.isfinite(when) or not previous <= when <= result["timing"]["tune_call_s"]
                or when != phase["available_at_tune_s"]):
            raise ValueError("Invalid feedback completion timestamp")
        previous = when
        for outcome, values, artifact in zip(classify_measurements(batch),
                                             batch["runner_run_secs"], batch["artifact_paths"]):
            if outcome == "finite":
                if artifact_ids.get(artifact) not in passed:
                    raise ValueError("Finite timing lacks four passed correctness cases")
                observed.append(statistics.median(values) * 1e6)
    archived_latencies = measurement["record_latencies_us"]
    finite_archived = [value for value in archived_latencies if value is not None]
    if (len(archived_latencies) != actual or sorted(observed) != sorted(finite_archived)
            or len(observed) != measurement["finite_records"]
            or len(observed) != result["correctness"]["finite_checked"]
            or result["correctness"]["passed_candidates"] != len(passed)
            or result["correctness"]["runner_inputs"] != len(inputs)
            or (min(observed) if observed else None) != measurement["best_latency_us"]):
        raise ValueError("Finite measurement/correctness/database summary differs")


def revalidate(path, root, spec, protocol, entry):
    """Write a separately identified audit view, never editing original artifacts."""
    result = read_json(path)
    original_errors = callback_only_failure(result)
    guard_path = root / "resource_guard/status.json"
    guard = read_json(guard_path)
    if (entry["returncode"] != 1 or guard["status"] != "child_failed"
            or guard.get("returncode") != 1
            or guard.get("cleanup", {}).get("remaining_live")
            or guard.get("cleanup", {}).get("errors") or not guard.get("ended_utc")):
        raise ValueError("Postcheck recovery requires ordinary exit 1 and clean resource-guard exit")
    console = root / (spec["id"] + ".console.log")
    if digest(console) != entry["console_sha256"]:
        raise ValueError("Original console hash differs")
    text = console.read_text(encoding="utf-8")
    terminal = "RuntimeError: Draft-GA architecture contract failed; raw result was retained at "
    last = text.rstrip().splitlines()[-1]
    expected = terminal + str(path) + ": " + repr(original_errors)
    if last != expected or f"result={path}\n" not in text:
        raise ValueError("Console does not end exclusively with the known final postcheck error")
    # Check the native GA accounting again independently of the callback assumption.
    from e1_thread_filter import _validate_draft_ga_contract, make_parser
    from run_operator800_study import command

    args = make_parser().parse_args(command(spec, root, protocol["seed"])[5:])
    native = _validate_draft_ga_contract(result["search"], args)
    if not native["passed"]:
        raise ValueError("Independent GA architecture validation failed")
    validate_evidence(path, result)
    from tvm.s_tir.meta_schedule.database import JSONDatabase

    database = JSONDatabase(path_workload=str(path.with_name("database_workload.json")),
                            path_tuning_record=str(path.with_name("database_tuning_record.json")))
    ranked = database.get_all_tuning_records()
    if [hashlib.sha256(str(record.trace).encode("utf-8")).hexdigest() for record in ranked] != result["measurement"]["trace_hashes"]:
        raise ValueError("Raw database trace identities differ from archived ranked traces")
    evidence = {"validator_sha256": digest(Path(__file__)), "original_result_path": str(path.resolve()),
                "original_result_sha256": digest(path), "original_returncode": entry["returncode"],
                "guard_path": str(guard_path.resolve()), "guard_sha256": digest(guard_path),
                "superseded_callback_errors": original_errors,
                "rule": "Each batch contains at most trials_per_iter candidates; actual sizes sum to the budget",
                "batch_sizes": [len(batch["runner_errors"]) for batch in result["measurement"]["callback_batches"]]}
    corrected = copy.deepcopy(result)
    corrected["architecture_validation"]["passed"] = True
    corrected["architecture_validation"]["errors"] = []
    corrected["architecture_validation"]["callback_batch_errors"] = []
    corrected["status"] = "independently_revalidated_postcheck"
    corrected["postcheck_revalidation"] = evidence
    target = root / "audit_v2" / (spec["id"] + "_derived_" + evidence["validator_sha256"][:12] + ".json")
    if target.exists():
        if read_json(target) != corrected:
            raise ValueError("Existing independent audit view changed")
    else:
        target.parent.mkdir(exist_ok=True)
        write_json_atomic(target, corrected)
    # All old protocol, phase conservation, measured convergence and model checks still apply.
    row = audit(target, spec, protocol)
    row.update(result_path=str(path), result_sha256=digest(path),
               postcheck_revalidation=evidence)
    return target, row
