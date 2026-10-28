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
"""Bridge a diagnostic DLL to an archived pool, then instrument failed Verify calls.

Uses the existing replay driver and original postprocessor objects. No sampling,
Builder, Runner, GPU measurements, or copied lowering pipeline. Diagnostic times
include observer overhead and never replace the archived performance measurements.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import time

from fixed_trace_pool import (
    ARMS, ROOT, SOURCE_FILES, ReplayExperiment, configure_environment,
    json_hash, plain_json, sha256, write_json,
)


class VerifyObserver:
    """Proxy preserves the actual postprocessor, reading thread-local state immediately."""

    def __init__(self, postproc, getter, tvm):
        self.postproc, self.getter, self.tvm = postproc, getter, tvm
        self.instrument = None
        self.reset()

    def reset(self):
        self.reason = None
        self.called = False
        self.input_mod = None

    def read_reason(self):
        try:
            return str(self.getter())
        except Exception as error:
            # Do not let a ScheduleError from the observer become a candidate failure.
            raise RuntimeError("Verify diagnostic getter failed") from error

    def apply(self, schedule):
        self.called = True
        self.input_mod = schedule.mod
        if self.instrument is None:
            passed = self.postproc.apply(schedule)
            self.reason = self.read_reason()
        else:
            current = self.tvm.transform.PassContext.current()
            with self.tvm.transform.PassContext(
                opt_level=current.opt_level, required_pass=current.required_pass,
                disabled_pass=current.disabled_pass, config=current.config,
                instruments=list(current.instruments) + [self.instrument],
            ):
                passed = self.postproc.apply(schedule)
                self.reason = self.read_reason()
        if bool(passed) != (self.reason == "none"):
            raise AssertionError("Verify return value and immediate failure reason disagree")
        return passed


def make_instrument(tvm):
    """Observe the unmodified pass pipeline; unfinished nested frames locate a throw."""
    @tvm.instrument.pass_instrument
    class PassJournal:
        def __init__(self):
            self.events = []
            self.stack = []
            self.anomalies = []
            self.last_completed_mod = None
            self.started = time.perf_counter()

        def run_before_pass(self, mod, info):
            try:
                event = {"id": len(self.events), "name": str(info.name),
                         "parent": self.stack[-1][0] if self.stack else None,
                         "depth": len(self.stack), "start_s": time.perf_counter() - self.started,
                         "completed": False}
                self.events.append(event)
                self.stack.append((event["id"], mod))
            except Exception as error:
                # Verify catches C++/FFI exceptions, so keep a sticky observer error.
                self.anomalies.append(f"before-pass observer error: {type(error).__name__}")

        def run_after_pass(self, mod, info):
            try:
                name = str(info.name)
                if not self.stack or self.events[self.stack[-1][0]]["name"] != name:
                    self.anomalies.append(f"Unmatched after-pass: {name}")
                    return
                index, _ = self.stack.pop()
                event = self.events[index]
                event["elapsed_inclusive_s"] = time.perf_counter() - self.started - event["start_s"]
                event["completed"] = True
                self.last_completed_mod = mod
            except Exception as error:
                self.anomalies.append(f"after-pass observer error: {type(error).__name__}")

        def report(self):
            elapsed = time.perf_counter() - self.started
            for index, _ in self.stack:
                self.events[index]["elapsed_to_verify_return_s"] = elapsed - self.events[index]["start_s"]
            return {"events": self.events, "anomalies": self.anomalies,
                    "unfinished_stack": [self.events[index]["name"] for index, _ in self.stack],
                    "deepest_unfinished_pass": self.events[self.stack[-1][0]]["name"] if self.stack else None,
                    "timing_note": "Instrumented inclusive times; nested frames must not be summed."}

    return PassJournal()


def append_row(stream, row):
    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    stream.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--library-dir", type=Path, required=True)
    parser.add_argument("--library-sha256", required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "command.json", {
        "argv": sys.argv, "utc": datetime.now(timezone.utc).isoformat(), "python": sys.executable,
    })
    archived = json.loads(args.pool.read_text(encoding="utf-8"))
    original_path = args.pool.parent / "result.json"
    original = json.loads(original_path.read_text(encoding="utf-8"))
    old_identity = archived["identity"]
    if not original["gate_passed"] or sha256(args.pool) != original["pool_sha256"]:
        raise ValueError("Input must be the unchanged pool from a validated run")
    current_source = {name: sha256(ROOT / name) for name in SOURCE_FILES}
    changed = [name for name in SOURCE_FILES if current_source[name] != old_identity["current_source_files"][name]]
    if changed != ["src/s_tir/meta_schedule/postproc/verify_gpu_code.cc"]:
        raise ValueError(f"Expected only the Verify diagnostic source change, found {changed}")
    reference = Path(old_identity["reference_result"])
    if sha256(reference) != old_identity["reference_sha256"]:
        raise ValueError("Archived reference changed")
    journal_path = args.pool.parent / "evaluations.jsonl"
    old_events = [json.loads(line) for line in journal_path.read_text(encoding="utf-8").splitlines()]
    old_status = {(event["candidate_id"], event["arm"]): event["status"]
                  for event in old_events if event["repeat"] == 0}
    candidates = [row for row in archived["candidates"] if row["generation_status"] == "complete"]
    if len(old_status) != len(candidates) * len(ARMS):
        raise ValueError("Archived event coverage is incomplete")
    configure_environment(args.library_dir.resolve())
    print("Loading diagnostic DLL; retaining archived pool/target/postprocessors", flush=True)
    started = time.perf_counter()
    experiment = ReplayExperiment(reference, archived["seed"],
                                  expected_library_sha256=args.library_sha256.lower())
    tvm = experiment.tvm
    if experiment.postproc_names != old_identity["postprocs"]:
        raise ValueError("Postprocessor chain changed")
    if json.loads(str(experiment.target)) != json.loads(old_identity["target"]):
        raise ValueError("Target changed")
    workload_path = args.pool.parent / "workload_tvm.json"
    if sha256(workload_path) != old_identity["workload_sha256"]:
        raise ValueError("Archived workload changed")
    tvm.ir.assert_structural_equal(experiment.mod, tvm.ir.load_json(workload_path.read_text(encoding="utf-8")))
    getter = tvm.get_global_func("s_tir.meta_schedule.testing.GetLastVerifyGPUCodeFailureReason")
    verify_index = experiment.postproc_names.index("VerifyGPUCode")
    observer = VerifyObserver(experiment.postprocs[verify_index], getter, tvm)
    experiment.postprocs[verify_index] = observer
    traces, oracle_modules = {}, {}
    for candidate in candidates:
        trace = experiment.decode_trace(candidate["trace_tvm_json"])
        if json_hash(plain_json(trace.as_json(True))) != candidate["trace_sha256"]:
            raise ValueError(f"Trace restore mismatch: {candidate['id']}")
        traces[candidate["id"]] = trace
        if candidate["collection_oracle"]["accepted"]:
            oracle_modules[candidate["id"]] = tvm.ir.load_json(candidate["oracle_mod_tvm_json"])
    identity = {
        "schema_version": 1, "pool_sha256": sha256(args.pool),
        "original_result_sha256": sha256(original_path), "old_events_sha256": sha256(journal_path),
        "old_identity": old_identity, "loaded_library": str(experiment.library),
        "loaded_library_sha256": sha256(experiment.library), "current_source_files": current_source,
        "driver_sha256": sha256(__file__), "replay_driver_sha256": sha256(Path(__file__).with_name("fixed_trace_pool.py")),
        "setup_s": time.perf_counter() - started,
        "scope": "New diagnostic DLL; no new proposals, Builder/Runner, GPU measurements or speedup claims",
    }
    write_json(args.output / "identity.json", identity)
    shutil.copy2(__file__, args.output / "driver_snapshot.py")
    shutil.copy2(Path(__file__).with_name("fixed_trace_pool.py"), args.output / "replay_driver_snapshot.py")
    bridge_rows = []
    with (args.output / "bridge.jsonl").open("x", encoding="utf-8") as stream:
        for arm in ARMS:
            for index, candidate in enumerate(candidates):
                observer.reset()
                row, schedule = experiment.evaluate(traces[candidate["id"]], candidate["seed"], arm)
                row.update(candidate_id=candidate["id"], stage=candidate["stage"],
                           verify_called=observer.called, verify_reason=observer.reason)
                expected = old_status[candidate["id"], arm]
                row["old_status"] = expected
                row["status_agrees"] = row["status"] == expected
                row["oracle_ir_equal"] = None
                if row["accepted"]:
                    tvm.ir.assert_structural_equal(schedule.mod, oracle_modules[candidate["id"]])
                    row["oracle_ir_equal"] = True
                append_row(stream, row)
                bridge_rows.append(row)
                if not row["status_agrees"]:
                    raise AssertionError(f"Bridge disagreement: {candidate['id']} {arm}")
                if (index + 1) % 32 == 0 or index + 1 == len(candidates):
                    print(f"Bridge {arm}: {index + 1}/{len(candidates)}", flush=True)
    baseline = {row["candidate_id"]: row for row in bridge_rows if row["arm"] == "baseline"}
    residual_ids = {row["candidate_id"] for row in bridge_rows
                    if row["arm"] == "trace" and row["status"] == f"postproc_{verify_index}_VerifyGPUCode_reject"}
    write_json(args.output / "bridge_summary.json", {
        "gate_passed": True, "complete_candidates": len(candidates), "evaluations": len(bridge_rows),
        "accepted_ir_checks": sum(row["oracle_ir_equal"] is True for row in bridge_rows),
        "baseline_reason_counts": dict(Counter(row["verify_reason"] for row in baseline.values())),
        "residual_reason_counts": dict(Counter(baseline[key]["verify_reason"] for key in residual_ids)),
        "bridge_sha256": sha256(args.output / "bridge.jsonl"),
    })
    print(f"Full bridge passed. Instrumenting {len(residual_ids)} residual failures separately", flush=True)
    profiles = []
    with (args.output / "pass_profiles.jsonl").open("x", encoding="utf-8") as stream:
        for candidate in candidates:
            key = candidate["id"]
            if key not in residual_ids:
                continue
            observer.reset()
            instrument = make_instrument(tvm)
            observer.instrument = instrument
            row, _ = experiment.evaluate(traces[key], candidate["seed"], "baseline")
            profile = {"candidate_id": key, "stage": candidate["stage"], "trace_sha256": candidate["trace_sha256"],
                       "status": row["status"], "verify_reason": observer.reason, **instrument.report()}
            profile["status_and_reason_agree"] = (row["status"] == baseline[key]["status"]
                                                  and observer.reason == baseline[key]["verify_reason"])
            snapshots = {"verify_input": observer.input_mod}
            if instrument.stack:
                snapshots["deepest_unfinished_input"] = instrument.stack[-1][1]
            if instrument.last_completed_mod is not None:
                snapshots["last_completed_output"] = instrument.last_completed_mod
            profile["ir_snapshots"] = {}
            for label, mod in snapshots.items():
                path = args.output / f"{key}.{label}.json"
                path.write_text(tvm.ir.save_json(mod), encoding="utf-8")
                profile["ir_snapshots"][label] = {"path": path.name, "sha256": sha256(path)}
            append_row(stream, profile)
            profiles.append(profile)
            if not profile["status_and_reason_agree"] or profile["anomalies"]:
                raise AssertionError(f"Instrumentation altered outcome or became inconsistent: {key}")
            print(f"Profile {key}: {observer.reason}; unfinished={profile['deepest_unfinished_pass']}", flush=True)
    observer.instrument = None
    write_json(args.output / "diagnosis.json", {
        "schema_version": 1, "gate_passed": True, "identity": identity,
        "bridge_sha256": sha256(args.output / "bridge.jsonl"),
        "profiles_sha256": sha256(args.output / "pass_profiles.jsonl"),
        "residual_candidates": len(profiles),
        "residual_reason_counts": dict(Counter(profile["verify_reason"] for profile in profiles)),
        "unfinished_pass_counts": dict(Counter(profile["deepest_unfinished_pass"] or "none" for profile in profiles)),
        "new_proposals": 0, "new_gpu_measurements": 0,
        "limitations": ["Only first failure reason, not all simultaneous violations.",
                        "A swallowed exception's text is not exposed by this getter.",
                        "Pass observation adds overhead; these times are diagnostic only.",
                        "Unfinished pass stack is not by itself a safe pruning rule."],
    })
    print("Diagnostic bridge and separate pass profiles completed", flush=True)


if __name__ == "__main__":
    main()
