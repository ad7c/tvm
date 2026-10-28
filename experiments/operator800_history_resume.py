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
"""Restore a frozen search by replaying verified feedback without GPU remeasurement."""

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys

from report_single_seed_study import audit, digest
from run_operator800_study import ROOT, command, verify_sources
from watch_operator800_study import complete_lines, read_json, write_json_atomic


def utc_now():
    return datetime.now(timezone.utc).isoformat()


class History:
    """Fail closed if regeneration or feedback accounting differs from the saved prefix."""

    def __init__(self, source, output, budget=800):
        self.source, self.output, self.budget = Path(source), Path(output), budget
        self.batches = complete_lines(self.source / "callback_batches.jsonl")
        self.records = complete_lines(self.source / "database_tuning_record.json")
        self.inputs = complete_lines(self.source / "runner_inputs.jsonl")
        self.checks = complete_lines(self.source / "correctness.jsonl")
        self.total = sum(len(b["runner_errors"]) for b in self.batches)
        if not self.batches or self.total != len(self.records) or not 0 < self.total < budget:
            raise ValueError("Incomplete or inconsistent saved feedback prefix")
        for batch in self.batches:
            n = len(batch["runner_errors"])
            if not n or any(len(batch[k]) != n for k in
                            ("builder_errors", "runner_run_secs", "trace_sha256", "artifact_paths")):
                raise ValueError("Saved batch field lengths differ")
            if any(batch["builder_errors"]) or any(batch["runner_errors"]):
                raise ValueError("This recovery requires a fully successful saved prefix")
            if any(not values or any(not math.isfinite(v) or v <= 0 for v in values)
                   for values in batch["runner_run_secs"]):
                raise ValueError("Saved feedback contains invalid measurement values")
        if sorted(statistics.median(v) for b in self.batches for v in b["runner_run_secs"]) != sorted(
                statistics.median(record[1][1]) for record in self.records):
            raise ValueError("Saved database and feedback measurements differ")
        self.index = self.offset = self.restored = self.new = 0
        self.model = None
        self.restore_boundary_s = None
        self.input_ids = {r["artifact_path"]: r["candidate_id"] for r in self.inputs}
        self.check_by_id = {}
        for item in self.checks:
            self.check_by_id.setdefault(item["candidate_id"], []).append(item)
        for batch in self.batches:
            for artifact in batch["artifact_paths"]:
                ident = self.input_ids[artifact]
                passed = [r for r in self.check_by_id.get(ident, [])
                          if r["event"] == "correctness_result" and r["passed"]]
                if len(passed) != 1 or len(passed[0]["cases"]) != 4 or not all(
                        case["passed"] for case in passed[0]["cases"]):
                    raise ValueError("Saved candidate lacks four successful correctness cases")
        self.source_hashes = {str(p.resolve()): digest(p) for p in
            (self.source / name for name in ("callback_batches.jsonl", "database_tuning_record.json",
             "database_workload.json", "runner_inputs.jsonl", "correctness.jsonl", "identity.json",
             "timeline.jsonl", "command.json"))}
        self.best_us = min(statistics.median(v) * 1e6 for b in self.batches
                           for v in b["runner_run_secs"])

    @property
    def replaying(self):
        return self.index < len(self.batches)

    def save(self, status):
        self.output.mkdir(parents=True, exist_ok=True)
        write_json_atomic(self.output / "resume_state.json", {
            "utc": utc_now(), "status": status, "source": str(self.source.resolve()),
            "historical_submitted": self.total, "restored_submitted": self.restored,
            "new_submitted": self.new, "actual_submitted": self.total + self.new,
            "budget": self.budget, "restored_batches": self.index,
            "historical_batches": len(self.batches), "source_hashes": self.source_hashes,
            "historical_best_us": self.best_us, "resume_driver_sha256": digest(Path(__file__)),
            "restore_boundary_s": self.restore_boundary_s,
            "resume_kind": "verified_feedback_replay_then_continuation"})

    def finish_batch(self, traces, count, model_size, tune_elapsed_s):
        if self.replaying:
            batch = self.batches[self.index]
            if traces != batch["trace_sha256"] or count != len(batch["runner_errors"]):
                raise ValueError(f"History replay differs at batch {self.index}; GPU continuation refused")
            self.restored += count
            if model_size != self.restored:
                raise ValueError("Restored model sample history differs")
            self.index += 1
            self.offset = 0
            if not self.replaying:
                self.restore_boundary_s = tune_elapsed_s
            self.save("restoring_history" if self.replaying else "history_restored")
        else:
            self.new += count
            if self.total + self.new > self.budget or model_size != self.total + self.new:
                raise ValueError("Continuation budget or cumulative model history differs")
            self.save("continuing")


def install_replay(history):
    """Use the original builder/runner after every historical batch has matched."""
    import tvm
    from tvm.ir.utils import derived_object
    import tvm.s_tir.meta_schedule.builder as builder_module
    from tvm.s_tir.meta_schedule.builder.builder import BuilderResult, PyBuilder
    import tvm.s_tir.meta_schedule.runner as runner_module
    from tvm.s_tir.meta_schedule.runner.local_runner import LocalRunnerFuture
    from tvm.s_tir.meta_schedule.measure_callback import PyMeasureCallback
    from tvm.s_tir.meta_schedule import tir_integration
    from tvm.s_tir.meta_schedule.utils import remove_build_dir
    from p3_observability import append_json
    from tvm_ffi import register_global_func

    real_builder, real_runner = builder_module.LocalBuilder, runner_module.LocalRunner
    real_tune = tir_integration.tune_tir

    @derived_object
    class ReplayBuilder(PyBuilder):
        def __init__(self, **kwargs):
            self.inner = real_builder(**kwargs)

        def build(self, inputs):
            if not history.replaying:
                return self.inner.build(inputs)
            batch = history.batches[history.index]
            if len(inputs) != len(batch["runner_errors"]):
                raise ValueError(f"Historical builder batch size differs at {history.index}")
            return [BuilderResult(path, None) for path in batch["artifact_paths"]]

    class ReplayRunner:
        def __init__(self, **kwargs):
            self.inner = real_runner(**kwargs)
            self.f_run_evaluator = None

        def run(self, inputs):
            if not history.replaying:
                self.inner.f_run_evaluator = self.f_run_evaluator
                return self.inner.run(inputs)
            batch = history.batches[history.index]
            result = []
            for inp in inputs:
                index = history.offset
                if index >= len(batch["runner_errors"]) or str(inp.artifact_path) != batch["artifact_paths"][index]:
                    raise ValueError("Historical runner identity differs")
                options = self.f_run_evaluator.keywords
                new_id = options["candidate_id"]
                old_id = history.input_ids[batch["artifact_paths"][index]]
                if new_id != old_id:
                    raise ValueError("Historical runner sequence differs")
                for original in history.check_by_id[old_id]:
                    append_json(options["audit_path"], dict(original, recovered_from=str(history.source)))
                result.append(LocalRunnerFuture(batch["runner_run_secs"][index], None))
                history.offset += 1
            return result

    @derived_object
    class ReplayCompleted(PyMeasureCallback):
        def apply(self, scheduler, task_id, candidates, builds, results):
            traces = [hashlib.sha256(str(c.sch.trace).encode("utf-8")).hexdigest() for c in candidates]
            current = next(history.output.glob("CONVG1C01_*/callback_batches.jsonl"))
            batch = complete_lines(current)[-1]
            history.finish_batch(traces, len(results), history.model.data_size,
                                 batch["available_at_tune_s"])
            print(f"RESUME {history.restored}/{history.total} restored; "
                  f"{history.new} new; total {history.total + history.new}/{history.budget}", flush=True)

    def tune(*args, **kwargs):
        history.model = kwargs["cost_model"]
        kwargs["measure_callbacks"] = [*kwargs["measure_callbacks"], ReplayCompleted()]
        return real_tune(*args, **kwargs)

    def cleanup(path):
        if not history.replaying:
            remove_build_dir(path)

    builder_module.LocalBuilder = ReplayBuilder
    runner_module.LocalRunner = ReplayRunner
    tir_integration.tune_tir = tune
    register_global_func("s_tir.meta_schedule.remove_build_dir", override=True)(cleanup)


def finish_result(history, protocol, spec):
    """Preserve raw timing and provide a separately identified resumed audit view."""
    from operator800_postcheck import callback_only_failure, validate_evidence
    from p3_observability import summarize_timeline

    path = next(history.output.glob(spec["id"] + "_*/result.json"))
    result = read_json(path)
    if history.replaying or history.total + history.new != spec["trials"]:
        raise ValueError("History restoration or continuation did not finish")
    if not result["architecture_validation"]["passed"]:
        callback_only_failure(result)
    validate_evidence(path, result)
    for source, expected in history.source_hashes.items():
        if digest(Path(source)) != expected:
            raise ValueError("Original history changed during continuation")
    old_batches = history.batches
    events = [e for e in complete_lines(history.source / "timeline.jsonl")
              if e["batch_index"] < len(old_batches)]
    old_end = old_batches[-1]["available_at_tune_s"]
    events.append({"event": "tune_end", "batch_index": len(old_batches)-1, "tune_elapsed_s": old_end})
    old_timeline = summarize_timeline(events)
    corrected = copy.deepcopy(result)
    offset = old_end - history.restore_boundary_s
    corrected["measurement"]["callback_batches"][:len(old_batches)] = copy.deepcopy(old_batches)
    for batch in corrected["measurement"]["callback_batches"][len(old_batches):]:
        batch["available_at_tune_s"] += offset
    corrected["timeline"]["batches"][:len(old_batches)] = old_timeline["batches"]
    for batch in corrected["timeline"]["batches"][len(old_batches):]:
        batch["available_at_tune_s"] += offset
    corrected["timing"]["tune_call_s"] += offset
    corrected["timeline"]["tune_s"] = corrected["timing"]["tune_call_s"]
    phase_keys = ("search_s", "builder_s", "runner_and_notify_s", "callbacks_s")
    accounted = sum(sum(batch[k] for k in phase_keys) for batch in corrected["timeline"]["batches"])
    corrected["timeline"]["residual_s"] = corrected["timing"]["tune_call_s"] - accounted
    if corrected["timeline"]["residual_s"] < -1e-6:
        raise ValueError("Resumed stage accounting overlaps")
    corrected["architecture_validation"].update(passed=True, errors=[], callback_batch_errors=[])
    corrected["status"] = "verified_resumed_search"
    corrected["resumption"] = {
        "kind": "verified_feedback_replay_then_continuation", "source_hashes": history.source_hashes,
        "historical_submitted": history.total, "new_submitted": history.new,
        "raw_result_sha256": digest(path), "historical_active_tune_s": old_end,
        "state_reconstruction_s": history.restore_boundary_s,
        "continuation_active_tune_s": result["timing"]["tune_call_s"] - history.restore_boundary_s,
        "timing_policy": "Historical active Tune plus continuation active Tune; reconstruction and pause excluded",
        "internal_counter_timing_policy": "Native prediction/search counters observed during reconstruction and continuation",
        "superseded_callback_errors": result["architecture_validation"]["errors"]}
    derived = path.with_name("result.resumed.json")
    write_json_atomic(derived, corrected)
    audit(derived, spec, protocol)
    history.save("completed")
    print(f"VERIFIED RESUMED RESULT: {derived}", flush=True)
    return derived


def accept_resumed_tuning(root, spec, protocol):
    """Accept a resumed arm with explicit segment provenance and an actual wrapper receipt."""
    from operator800_cohort import matching_receipt, check_receipt, check_console
    from operator800_postcheck import validate_evidence

    state = read_json(root / "resume_state.json")
    if state["status"] != "completed" or state["actual_submitted"] != spec["trials"]:
        raise ValueError("Resumed search has not completed the total budget")
    receipt = matching_receipt(root, "history_resume", spec["id"])
    entry = check_receipt(receipt)
    expected = [sys.executable, "-u", "-X", "utf8", str(ROOT / "experiments/operator800_history_resume.py"),
                "--source", state["source"], "--output", str(root.resolve())]
    if entry["argv"] != expected or state["resume_driver_sha256"] != digest(Path(__file__)):
        raise ValueError("Resumption command or adapter identity changed")
    console = check_console(root, entry, "history_resume", spec["id"])
    path = next(root.glob(spec["id"] + "_*/result.resumed.json"))
    result = read_json(path)
    raw = path.with_name("result.json")
    if result["resumption"]["raw_result_sha256"] != digest(raw):
        raise ValueError("Raw resumed result changed")
    sources = result["resumption"]["source_hashes"]
    if sources != state["source_hashes"] or any(digest(Path(p)) != h for p, h in sources.items()):
        raise ValueError("Historical resumption inputs changed")
    if (state["historical_submitted"] + state["new_submitted"] != spec["trials"]
            or result["measurement"]["xgb_data_size"] != spec["trials"]):
        raise ValueError("Resumed model history or submission count differs")
    validate_evidence(raw, read_json(raw))
    audit(path, spec, protocol)
    artifacts = [path, raw, root / "resume_state.json", root / "resource_guard/status.json", console]
    artifacts += [path.with_name(name) for name in ("timeline.jsonl", "callback_batches.jsonl",
                  "database_workload.json", "database_tuning_record.json", "correctness.jsonl",
                  "identity.json", "runner_inputs.jsonl")]
    hashes = {str(p.resolve()): digest(p) for p in artifacts}
    hashes.update(sources)
    return {"root": str(root.resolve()), "result_path": str(path.resolve()), "receipt": receipt,
            "submitted": spec["trials"], "artifacts_sha256": hashes,
            "completion_basis": "verified_history_replay_continuation",
            "historical_submitted": state["historical_submitted"], "new_submitted": state["new_submitted"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = read_json(args.output / "protocol.json")
    verify_sources(protocol)
    history = History(args.source.resolve(), args.output.resolve())
    identity = read_json(history.source / "identity.json")
    if (identity["source_identity_sha256"] != protocol["identity"]["source_identity_sha256"]
            or identity["loaded_library_sha256"] != protocol["runtime"]["compiler_sha256"]):
        raise ValueError("Saved history version differs from frozen environment")
    history.save("preparing_history")
    os.environ["TVM_LIBRARY_PATH"] = str(ROOT / "build-prune/lib")
    install_replay(history)
    import e1_thread_filter as driver
    spec = next(s for s in protocol["order"] if s["id"] == "CONVG1C01")
    parsed = driver.make_parser().parse_args(command(spec, args.output.resolve(), protocol["seed"])[5:])
    try:
        driver.run_experiment(parsed)
    except RuntimeError:
        paths = list(history.output.glob("CONVG1C01_*/result.json"))
        if not paths:
            history.save("failed")
            raise
        from operator800_postcheck import callback_only_failure
        callback_only_failure(read_json(paths[0]))
    finish_result(history, protocol, spec)


if __name__ == "__main__":
    main()
