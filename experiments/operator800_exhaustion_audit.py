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
"""Externally audit frozen Reduction searches that reached the empty-candidate stop rule."""

import copy
import hashlib
from pathlib import Path
import types

from e1_thread_filter import _validate_callback_batch_contract, _validate_draft_ga_contract, make_parser
import operator800_postcheck as original_audit
from report_single_seed_study import audit, digest
from run_operator800_study import command
from watch_operator800_study import complete_lines, read_json, write_json_atomic


def validate_variable_batches(batches, search_batches, trials, maximum, effective):
    """Empty iterations may be followed by candidates; the last search iteration has no callback."""
    errors = []
    if len(batches) + 1 != search_batches or not 0 < effective < trials or maximum <= 0:
        errors.append("Unexpected terminal-search/budget accounting")
    total = 0
    for batch in batches:
        lengths = [len(batch[key]) for key in original_audit.FIELDS]
        if len(set(lengths)) != 1 or not 0 <= lengths[0] <= maximum:
            errors.append("Variable callback field lengths differ or exceed the batch limit")
        total += lengths[0]
    if total != effective:
        errors.append("Variable callback counts differ from the database")
    return errors


def accept_exhausted(root, spec, protocol):
    from operator800_cohort import check_console, object_hash
    from tvm.s_tir.meta_schedule.database import JSONDatabase

    if spec["workload"] != "reduction4096" or "do not pad" not in protocol["short_space_policy"]:
        raise ValueError("This external audit is restricted to the registered Reduction early-stop policy")
    path = next(root.glob(spec["id"] + "_*/result.json"))
    result = read_json(path)
    actual = result["budget"]["actual_submitted"]
    if (result["budget"]["terminal_status"] != "early_exhaustion"
            or result["budget"]["budget_completed"] or not 0 < actual < spec["trials"]):
        raise ValueError("Not a below-budget search termination")
    entries = [(i, e) for i, e in enumerate(complete_lines(root / "execution.jsonl"))
               if e.get("phase") == "tuning" and e.get("id") == spec["id"]]
    if len(entries) != 1:
        raise ValueError("No unique original tuning receipt")
    index, entry = entries[0]
    if entry["returncode"] != 1 or entry["argv"] != command(spec, root, protocol["seed"]):
        raise ValueError("Original command or terminal exit differs")
    guard = read_json(root / "resource_guard/status.json")
    if (guard["status"] != "child_failed" or guard.get("returncode") != 1
            or guard.get("cleanup", {}).get("remaining_live") or guard.get("cleanup", {}).get("errors")):
        raise ValueError("Original run ended with a resource or process-cleanup problem")
    console = check_console(root, entry, "tuning", spec["id"])
    expected_tail = ("RuntimeError: Draft-GA architecture contract failed; raw result was retained at "
                     + str(path) + ": " + repr(result["architecture_validation"]["errors"]))
    if console.read_text(encoding="utf-8").rstrip().splitlines()[-1] != expected_tail:
        raise ValueError("Console contains a different terminal exception")
    args = make_parser().parse_args(command(spec, root, protocol["seed"])[5:])
    native = _validate_draft_ga_contract(result["search"], args)
    callback_errors = _validate_callback_batch_contract(
        result["measurement"]["callback_batches"], result["search"]["search_batches"],
        spec["trials"], spec["trials_per_iter"], effective_trials=actual)
    if result["architecture_validation"]["errors"] != native["errors"] + callback_errors:
        raise ValueError("Original architecture errors cannot be reproduced")
    events = result["search"]["draft_ga"]["summary_events"]
    if (len(events) != result["search"]["search_batches"]
            or len(events) != len(result["measurement"]["callback_batches"]) + 1
            or sum(e["returned"] == 0 for e in events) != 5 or events[-1]["returned"] != 0):
        raise ValueError("Search did not terminate at the frozen five-empty-iteration rule")
    allowed_native = []
    if spec["arm"] == "D":
        for i, event in enumerate(events):
            if (event["final_xgb_calls"] != int(event["draft_archive"] > 0)
                    or event["final_xgb_input"] != min(event["verify_top_k"], event["draft_archive"])):
                raise ValueError("Nonempty Draft archive lacks the required final scoring")
            if event["draft_archive"] == 0:
                allowed_native.append(f"batch {i}: final_xgb_calls=0 (expected exactly 1)")
    if native["errors"] != allowed_native:
        raise ValueError("An architecture error remains beyond empty-archive scoring")
    # Reuse every original data/trace/correctness/timeline check. Only its explicit
    # fixed-batch-shape policy is replaced in a private function namespace.
    namespace = dict(original_audit.validate_evidence.__globals__, validate_batches=validate_variable_batches)
    evidence_check = types.FunctionType(original_audit.validate_evidence.__code__, namespace)
    evidence_check(path, result)
    records = JSONDatabase(path_workload=str(path.with_name("database_workload.json")),
                           path_tuning_record=str(path.with_name("database_tuning_record.json"))).get_all_tuning_records()
    hashes = [hashlib.sha256(str(record.trace).encode("utf-8")).hexdigest() for record in records]
    if hashes != result["measurement"]["trace_hashes"]:
        raise ValueError("Raw database trace identities differ")
    total = 0
    for batch in result["measurement"]["callback_batches"]:
        total += len(batch["runner_errors"])
        if batch["cost_model"]["data_size"] != total:
            raise ValueError("Cumulative model history differs from actual feedback")
    corrected = copy.deepcopy(result)
    corrected["architecture_validation"].update(passed=True, errors=[], callback_batch_errors=[])
    corrected["status"] = "verified_early_termination"
    corrected["early_termination_audit"] = {
        "auditor_sha256": digest(Path(__file__)), "shared_evidence_checker_sha256": digest(Path(original_audit.__file__)),
        "original_result_sha256": digest(path), "original_returncode": 1,
        "superseded_errors": result["architecture_validation"]["errors"],
        "termination": "Five cumulative empty search iterations; final iteration returns no callback",
        "scope": "Search stopping condition; no assertion that every legal schedule has been enumerated"}
    derived = path.with_name("result.exhaustion.json")
    if derived.exists() and read_json(derived) != corrected:
        raise ValueError("Existing external early-termination audit changed")
    if not derived.exists():
        write_json_atomic(derived, corrected)
    audit(derived, spec, protocol)
    artifacts = [path, derived, console, root / "resource_guard/status.json"]
    artifacts += [path.with_name(name) for name in ("timeline.jsonl", "callback_batches.jsonl",
                  "database_workload.json", "database_tuning_record.json", "correctness.jsonl",
                  "identity.json", "runner_inputs.jsonl")]
    return {"root": str(root.resolve()), "result_path": str(derived.resolve()), "submitted": actual,
            "receipt": {"ledger_path": str((root / "execution.jsonl").resolve()),
                        "line_index": index, "entry_sha256": object_hash(entry)},
            "artifacts_sha256": {str(p.resolve()): digest(p) for p in artifacts},
            "completion_basis": "verified_early_termination", "original_returncode": 1}
