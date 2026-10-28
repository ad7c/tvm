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
"""Bounded ResNet-18 study: baseline, constraints, and Draft pools 64/128.

All model preparation/evaluation uses the previously verified Relax VM path.
This experiment deliberately retains the legacy Draft formula and stopping
rules; larger pools and a matched final exploration quota are the changes.
"""

import argparse
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys
import time
import traceback

import model_scout_study as base
from guard_experiment_resources import kill_launched_tree, system_memory
from p3_observability import append_json

ROOT = base.ROOT
DRIVER = Path(__file__).resolve()
ARMS = ("B", "C", "D64", "D128")
CONFIG = {
    "population_size": 128, "genetic_num_iters": 2,
    "init_min_unmeasured": 16, "init_measured_ratio": 0.2,
    "max_fail_count": 5, "genetic_mutate_prob": 0.85,
    "genetic_max_fail_count": 10, "eps_greedy": 0.1,
}
read, sha, utc = base.read, base.sha, base.utc


def save(path, value):
    """Publish status atomically; never overwrite a historical run directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".writing")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def environment(arm="B"):
    env = base.environment("D" if arm.startswith("D") else arm)
    env.update({
        "TVM_EVO_INIT_SAMPLE_BATCH_SIZE": "128",
        "TVM_DRAFT_GA_VERIFY_RATIO": "1.0" if arm == "D128" else "0.5",
        "TVM_DRAFT_GA_FINAL_EPS": "0.1",
        "OPENBLAS_NUM_THREADS": "1",
    })
    return env


def identity():
    files = [DRIVER, Path(base.__file__), ROOT / "src/s_tir/meta_schedule/draft_model.cc",
             ROOT / "src/s_tir/meta_schedule/draft_guided_ga.cc",
             ROOT / "experiments/guard_experiment_resources.py",
             ROOT / "experiments/runtime_bootstrap/sitecustomize.py"]
    return {**base.identity(), **{p.relative_to(ROOT).as_posix(): sha(p) for p in files}}


def publish(study, state, **extra):
    progress = {"utc": utc(), "status": state, "controller_pid": os.getpid(), **extra}
    save(study / "progress.json", progress)
    lines = ["# ResNet-18 candidate-pool study", "", f"Updated: {progress['utc']}",
             f"State: {state}", f"Phase: {extra.get('phase', '-')} / {extra.get('arm', '-')}",
             f"Working directory: {extra.get('output', '-')}",
             "", "| Arm | Feedback | Latest full-model ms | State |",
             "|---|---:|---:|---|"]
    for arm in ARMS:
        folder = study / ("formal_" + arm)
        item = read(folder / "progress.json") if (folder / "progress.json").exists() else {}
        checkpoints = read(folder / "checkpoints.json") if (folder / "checkpoints.json").exists() else []
        latency = "-"
        if checkpoints:
            evaluation = Path(checkpoints[-1]["result"])
            if evaluation.exists():
                latency = f"{read(evaluation)['median_ms']:.5f}"
        lines.append(f"| {arm} | {item.get('submitted', 0)} | {latency} | {item.get('status', 'pending')} |")
    lines += ["", "B: matched MetaSchedule; C: constraints; D64/D128: constraints + legacy Draft.",
              "All groups: population 128, GA 2, at most 10 GPU submissions per batch.",
              "One search seed; independent final timing is reported separately."]
    (study / "STATUS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def memory_ready():
    memory = system_memory()
    return memory["available_physical"] >= 1.5 * 2**30 and memory["available_commit"] >= 6 * 2**30


def run_child(stage, args, output, *, arm="B", trials=0, database=None,
              timeout_s=900, soft_limit_s=None, monitor=False):
    import psutil
    output.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-u", "-X", "utf8", str(DRIVER), stage,
               "--study", str(args.study), "--output", str(output), "--arm", arm,
               "--trials", str(trials)]
    if database is not None:
        command += ["--database", str(database)]
    guard = [sys.executable, "-u", "-X", "utf8", str(ROOT / "experiments/guard_experiment_resources.py"),
             "--output", str(output / "guard"), "--interval", "5",
             "--physical-reserve-mib", "1024", "--commit-reserve-gib", "4",
             "--private-limit-gib", "4", "--consecutive-samples", "3", "--"]
    env = environment(arm)
    started = time.time()
    deadline = min(started + timeout_s, getattr(args, "global_deadline", float("inf")))
    env["SCOUT_POOL_DEADLINE"] = str(min(deadline - 30, started + (soft_limit_s or timeout_s - 60)))
    env["SCOUT_POOL_GLOBAL_DEADLINE"] = str(getattr(args, "global_deadline", deadline))
    save(output / "command.json", {"argv": guard + command, "arm": arm,
                                    "started_utc": utc(), "deadline_epoch": deadline})
    limited = False
    known = {}
    with (output / "launcher.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(guard + command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        owned = psutil.Process(process.pid)
        while process.poll() is None:
            try:
                for child in [owned, *owned.children(recursive=True)]:
                    known[child.pid] = child
            except psutil.NoSuchProcess:
                pass
            if monitor:
                publish(args.study, "running", phase=stage, arm=arm, output=str(output))
            if time.time() >= deadline:
                cleanup = kill_launched_tree(owned, known.values())
                save(output / "deadline_cleanup.json", cleanup)
                if cleanup["errors"] or cleanup["remaining_live"]:
                    raise RuntimeError("Deadline cleanup incomplete")
                limited = True
                break
            time.sleep(5)
        code = process.wait(timeout=30)
    receipt = {"returncode": code, "wall_s": time.time() - started,
               "status": "time_limit" if limited else "completed" if code == 0 else "failed", "utc": utc()}
    save(output / "receipt.json", receipt)
    if not limited:
        state = read(output / "guard/status.json")
        if code or state.get("status") != "completed" or state.get("cleanup", {}).get("errors"):
            raise RuntimeError(f"{stage}/{arm} failed; see {output / 'guard/status.json'}")
    return receipt


def copy_database(source, destination):
    """Copy complete database lines; retain any incomplete suffix in the original."""
    destination.mkdir(parents=True, exist_ok=False)
    counts = {}
    for name in ("database_workload.json", "database_tuning_record.json"):
        path = source / name
        valid = []
        if path.exists():
            lines = path.read_text(encoding="utf-8").splitlines()
            for index, line in enumerate(lines):
                try:
                    json.loads(line)
                except json.JSONDecodeError:
                    if index != len(lines) - 1:
                        raise
                    break
                valid.append(line)
        (destination / name).write_text("\n".join(valid) + ("\n" if valid else ""), encoding="utf-8")
        counts[name] = len(valid)
    save(destination / "snapshot.json", {"source": str(source), "complete_lines": counts, "utc": utc()})
    return counts["database_tuning_record.json"]


def trace_record(candidate, task):
    # Trace.as_json() retains typed IntImm/FloatImm values in this TVM revision.
    # Encode these scalars separately; generic IR serialization of a whole Trace
    # does not restore the instruction-kind callbacks needed for replay.
    import tvm
    def scalar(value):
        if isinstance(value, (tvm.tirx.IntImm, tvm.tirx.FloatImm)):
            return {"__tvm_scalar__": tvm.ir.save_json(value)}
        raise TypeError(f"Unsupported trace value: {type(value).__name__}")
    serialized = json.dumps(candidate.sch.trace.as_json(), default=scalar, separators=(",", ":"))
    return hashlib.sha256((task + "\n" + serialized).encode()).hexdigest(), serialized


def replay_trace_record(serialized, schedule):
    """Restore a saved trace including explicitly typed scalar attributes."""
    import tvm
    from tvm.s_tir.schedule import Trace
    def scalar(value):
        return tvm.ir.load_json(value["__tvm_scalar__"]) if set(value) == {"__tvm_scalar__"} else value
    Trace.apply_json_to_schedule(json.loads(serialized, object_hook=scalar), schedule)


def tune(args):
    import numpy as np
    import tvm
    from tvm.ir.utils import derived_object
    from tvm.s_tir.meta_schedule import relax_integration
    from tvm.s_tir.meta_schedule.builder import LocalBuilder
    from tvm.s_tir.meta_schedule.cost_model import PyCostModel, XGBModel
    from tvm.s_tir.meta_schedule.database import JSONDatabase
    from tvm.s_tir.meta_schedule.logging import get_loggers_from_work_dir
    from tvm.s_tir.meta_schedule.measure_callback import MeasureCallback, PyMeasureCallback
    from tvm.s_tir.meta_schedule.runner import EvaluatorConfig, LocalRunner
    from tvm.s_tir.meta_schedule.search_strategy import EvolutionarySearch, PySearchStrategy
    from tvm.s_tir.meta_schedule.tune import tune_tasks
    from tvm.s_tir.meta_schedule.tune_context import TuneContext
    from tvm.s_tir.meta_schedule.utils import fork_seed
    from p3_observability import TuneJournal

    random.seed(base.SEED)
    np.random.seed(base.SEED)
    started = time.perf_counter()
    deadline = float(os.environ["SCOUT_POOL_DEADLINE"])
    args.global_deadline = float(os.environ["SCOUT_POOL_GLOBAL_DEADLINE"])
    core = XGBModel(num_warmup_samples=100, adaptive_training=True, num_tuning_cores=1)
    seen = set()
    audit_seconds = 0.0
    predict_calls = 0
    submitted = 0
    measured_ids = []
    checkpoints = []
    evaluation_s = 0.0

    def record_candidates(candidates, task):
        ids = []
        with gzip.open(args.output / "candidate_traces.jsonl.gz", "at", encoding="utf-8") as stream:
            for candidate in candidates:
                key, trace = trace_record(candidate, task)
                ids.append(key)
                if key not in seen:
                    stream.write(json.dumps({"task": task, "trace_sha256": key,
                                             "trace_format": "tvm.Trace.as_json+typed_scalars_v1",
                                             "trace": json.loads(trace)}) + "\n")
                    seen.add(key)
        return ids

    @derived_object
    class ObservedModel(PyCostModel):
        def load(self, path):
            core.load(path)

        def save(self, path):
            core.save(path)

        def update(self, context, candidates, results):
            begin = time.perf_counter()
            before = int(core.data_size)
            core.update(context, candidates, results)
            append_json(args.output / "model_updates.jsonl", {
                "task": context.task_name, "input_count": len(candidates),
                "samples_before": before, "samples_after": int(core.data_size),
                "seconds": time.perf_counter() - begin, "utc": utc()})

        def predict(self, context, candidates):
            nonlocal predict_calls, audit_seconds
            begin = time.perf_counter()
            scores = core.predict(context, candidates)
            predict_s = time.perf_counter() - begin
            audit_start = time.perf_counter()
            ids = record_candidates(candidates, str(context.task_name))
            draft = None
            if args.arm.startswith("D"):
                analyzer = tvm.get_global_func("s_tir.meta_schedule.testing.AnalyzePrunerDraftScore")
                draft = [[float(x) for x in analyzer(c.sch.mod, 10752., 192., 20, 1, 4, 32, 128.)]
                         for c in candidates]
            predict_calls += 1
            row = {"call": predict_calls, "task": context.task_name, "utc": utc(),
                   "input_count": len(candidates), "data_size": int(core.data_size),
                   "warmup": int(core.data_size) < 100, "prediction_s": predict_s,
                   "trace_sha256": ids,
                   "scores": [float(x) if math.isfinite(float(x)) else None for x in scores],
                   "draft_known_score_statements": draft}
            append_json(args.output / "model_predictions.jsonl", row)
            audit_seconds += time.perf_counter() - audit_start
            return scores

    @derived_object
    class DeadlineStrategy(PySearchStrategy):
        def __init__(self):
            self.inner = EvolutionarySearch(**CONFIG)
            self.context = None

        def _initialize_with_tune_context(self, context):
            self.context = context
            self.inner._initialize_with_tune_context(context)

        def pre_tuning(self, max_trials, num_trials_per_iter, design_spaces, database=None, cost_model=None):
            self.inner.pre_tuning(max_trials, num_trials_per_iter, design_spaces, database, cost_model)

        def post_tuning(self):
            self.inner.post_tuning()

        def generate_measure_candidates(self):
            if time.time() >= deadline:
                append_json(args.output / "task_stops.jsonl", {"task": self.context.task_name,
                            "reason": "arm_time_limit", "utc": utc()})
                return None
            result = self.inner.generate_measure_candidates()
            append_json(args.output / "candidate_batches.jsonl", {"task": self.context.task_name,
                        "returned": None if result is None else len(result), "utc": utc()})
            return result

        def notify_runner_results(self, candidates, results):
            self.inner.notify_runner_results(candidates, results)

        def clone(self):
            return DeadlineStrategy()

    lowered = tvm.ir.load_json((args.study / "prepare/model.lowered.json").read_text(encoding="utf-8"))
    extracted = relax_integration.extract_tasks(lowered, base.target(tvm))
    expected = read(args.study / "prepare/tasks.json")
    if [t.task_name for t in extracted] != [t["name"] for t in expected]:
        raise RuntimeError("Task inventory differs from frozen preparation")
    names = [f"t{i:02d}" for i in range(len(extracted))]
    loggers = get_loggers_from_work_dir(str(args.output), names)
    contexts = [TuneContext(mod=t.dispatched[0], target=t.target, space_generator="post-order-apply",
                search_strategy=DeadlineStrategy(), task_name=name, logger=logger,
                rand_state=seed, num_threads=1).clone()
                for t, name, logger, seed in zip(extracted, names, loggers, fork_seed(base.SEED, n=len(extracted)))]
    database = JSONDatabase(work_dir=str(args.output))
    builder = LocalBuilder(max_workers=1, timeout_sec=120)
    runner = LocalRunner(timeout_sec=30,
                         evaluator_config=EvaluatorConfig(number=3, repeat=5, min_repeat_ms=100))

    def checkpoint():
        nonlocal evaluation_s
        folder = args.output / "checkpoints" / f"n{submitted:04d}"
        copy_database(args.output, folder)
        wall_before = time.perf_counter() - started
        receipt = run_child("evaluate", args, folder / "eval", arm=args.arm,
                            database=folder, timeout_s=900)
        if receipt["status"] != "completed":
            raise RuntimeError("Checkpoint evaluation exceeded its deadline")
        evaluation_s += receipt["wall_s"]
        checkpoints.append({"submitted": submitted, "wall_before_evaluation_s": wall_before,
                            "available_wall_s": time.perf_counter() - started,
                            "active_tune_s": time.perf_counter() - started - evaluation_s,
                            "result": str(folder / "eval/result.json")})
        save(args.output / "checkpoints.json", checkpoints)

    @derived_object
    class Capture(PyMeasureCallback):
        def apply(self, scheduler, task_id, candidates, builds, runs):
            nonlocal submitted
            submitted += len(candidates)
            ids = record_candidates(candidates, names[int(task_id)])
            measured_ids.extend(ids)
            row = {"utc": utc(), "task_id": int(task_id), "submitted": submitted,
                   "candidates": len(candidates), "trace_sha256": ids,
                   "builder_errors": [r.error_msg for r in builds],
                   "runner_errors": [r.error_msg for r in runs],
                   "run_secs": [None if r.run_secs is None else [float(x) for x in r.run_secs] for r in runs],
                   "xgb_data_size": int(core.data_size), "wall_s": time.perf_counter() - started}
            append_json(args.output / "batches.jsonl", row)
            save(args.output / "progress.json", {"status": "tuning", "arm": args.arm, **row})
            if args.trials > 10 and submitted >= (len(checkpoints) + 1) * 100 and time.time() < deadline - 600:
                checkpoint()

    with TuneJournal(args.output / "timeline.jsonl", record_model_training=True):
        tune_tasks(tasks=contexts, task_weights=[float(t.weight) for t in extracted],
                   work_dir=str(args.output), max_trials_global=args.trials, num_trials_per_iter=10,
                   builder=builder, runner=runner, database=database, cost_model=ObservedModel(),
                   measure_callbacks=[*MeasureCallback.create("default"), Capture()], task_scheduler="gradient")
    wall_s = time.perf_counter() - started
    core.save(str(args.output / "xgb_model.tar"))
    summary = {"status": "time_limit" if time.time() >= deadline else "completed", "arm": args.arm,
               "submitted": submitted, "requested": args.trials, "tune_wall_s": wall_s,
               "active_tune_s": wall_s - evaluation_s, "checkpoint_evaluation_s": evaluation_s,
               "prediction_audit_s": audit_seconds, "predict_calls": predict_calls,
               "distinct_measured_traces": len(set(measured_ids)), "checkpoints": checkpoints,
               "database_records": len(database.get_all_tuning_records())}
    save(args.output / "result.json", summary)
    save(args.output / "progress.json", summary)
    base.terminal_success()


def audit_arm(folder, arm):
    calls = [json.loads(x) for x in (folder / "model_predictions.jsonl").read_text(encoding="utf-8").splitlines()]
    source = "\n".join(p.read_text(encoding="utf-8") for p in (folder / "logs").glob("*.log"))
    from e1_thread_filter import _parse_draft_ga_log
    summaries = _parse_draft_ga_log(source)["summary_events"]
    nonempty = [e for e in summaries if e["returned"] > 0]
    expected_pool = int(arm[1:]) if arm.startswith("D") else None
    errors = []
    if not calls or not nonempty:
        errors.append("No nonempty scoring events were recorded")
    for event in nonempty:
        if expected_pool:
            if event["ga_xgb_calls"] != 0 or event["final_xgb_calls"] != 1 or event["verify_top_k"] != expected_pool:
                errors.append("Draft scoring architecture differs from registered configuration")
            if event["final_xgb_input"] > expected_pool:
                errors.append("Verify pool exceeds configured cap")
        elif event["ga_xgb_calls"] != 3 or event["final_xgb_calls"] != 0:
            errors.append("Baseline GA does not perform initial plus two generation scores")
    quotas = re.findall(r"Final selection quota: epsilon=([\d.]+) model=(\d+) random=(\d+)", source)
    if not quotas or any(float(e) != 0.1 for e, _, _ in quotas):
        errors.append("Final exploration epsilon was not applied")
    result = {"passed": not errors, "errors": errors, "scoring_events": summaries,
              "actual_xgb_input_counts": [c["input_count"] for c in calls],
              "expected_pool": expected_pool,
              "full_verify_pool_observed": expected_pool is None or any(c["input_count"] == expected_pool for c in calls),
              "source": "Actual model calls and native search telemetry"}
    save(folder / "architecture.json", result)
    if errors:
        raise RuntimeError("Candidate-pool architecture audit failed: " + "; ".join(errors))
    return result


def retime(args):
    import numpy as np
    import tvm
    from tvm import relax
    device = tvm.cuda(0)
    with np.load(base.MODEL / "params.npz") as data:
        params = [tvm.runtime.tensor(data[f"p{i}"], device) for i in range(len(data.files))]
    with np.load(base.MODEL / "inputs.npz") as data:
        x = tvm.runtime.tensor(data["normal"], device)
    with np.load(base.MODEL / "reference.npz") as data:
        reference = data["normal"].copy()
    vms, samples = {}, {}
    for arm in ARMS:
        source = args.study / ("formal_" + arm) / "final_eval/model.scheduled.json"
        mod = tvm.ir.load_json(source.read_text(encoding="utf-8"))
        exe = tvm.compile(mod, target=base.target(tvm), relax_pipeline="default_build", tir_pipeline="s_tir")
        vm = relax.VirtualMachine(exe, device)
        if not np.allclose(vm["main"](x, *params).numpy(), reference, rtol=2e-4, atol=2e-4):
            raise RuntimeError("Remeasurement correctness failure: " + arm)
        vm.save_function("main", "benchmark", x, *params, include_return=False)
        vms[arm], samples[arm] = vm, []
    orders = [list(ARMS[i:] + ARMS[:i]) for i in range(len(ARMS))]
    round_medians = {a: [] for a in ARMS}
    for order in orders:
        for arm in order:
            vm = vms[arm]
            for _ in range(100):
                vm["benchmark"]()
            device.sync()
            values = list(vm.time_evaluator("benchmark", device, number=3, repeat=8, min_repeat_ms=100)().results)
            samples[arm].extend(values)
            round_medians[arm].append(float(np.median(values) * 1000))
    save(args.output / "result.json", {"status": "completed", "orders": orders, "arms": {
        a: {"median_ms": float(np.median(samples[a]) * 1000), "latency_s": samples[a],
            "round_medians_ms": round_medians[a]} for a in ARMS}})
    base.terminal_success()


def report(study):
    retiming = read(study / "retime/result.json")["arms"] if (study / "retime/result.json").exists() else {}
    rows = []
    for arm in ARMS:
        folder = study / ("formal_" + arm)
        if not (folder / "final_eval/result.json").exists():
            continue
        result, evaluation = read(folder / "result.json"), read(folder / "final_eval/result.json")
        predictions = [json.loads(line) for line in
                       (folder / "model_predictions.jsonl").read_text(encoding="utf-8").splitlines()]
        rows.append({"arm": arm, "submitted": result["submitted"],
                     "tune_wall_s": result["tune_wall_s"],
                     "final_median_ms": retiming.get(arm, {}).get("median_ms", evaluation["median_ms"]),
                     "independent_retiming": arm in retiming,
                     "trained_prediction_calls": sum(not p["warmup"] for p in predictions),
                     "warmup_prediction_calls": sum(p["warmup"] for p in predictions),
                     "tuned_functions": len(evaluation["applied_tuned_functions"]),
                     "fallback_functions": evaluation["fallback_functions"]})
    baseline = next((r for r in rows if r["arm"] == "B"), None)
    if baseline:
        for row in rows:
            row["within_5_percent"] = row["final_median_ms"] <= baseline["final_median_ms"] * 1.05
            checkpoints = study / ("formal_" + row["arm"]) / "checkpoints.json"
            checkpoints = read(checkpoints) if checkpoints.exists() else []
            attained = [c["available_wall_s"] for c in checkpoints
                        if read(c["result"])["median_ms"] <= baseline["final_median_ms"] * 1.05]
            if row["within_5_percent"]:
                attained.append(row["tune_wall_s"] + read(study / ("formal_" + row["arm"]) / "final_eval/receipt.json")["wall_s"])
            row["first_observed_target_s"] = min(attained) if attained else None
    save(study / "comparison.json", rows)
    lines = ["# ResNet-18 pool-size comparison", "", "Matched parameters, legacy Draft, single search seed.",
             "Complete GPU VM forward; checkpoint observations only, no interpolation.", "",
             "| Arm | Feedback | Tune wall min | Final ms | Tuned functions | Trained scoring calls | Within B +5% |",
             "|---|---:|---:|---:|---:|---:|---|"]
    for row in rows:
        lines.append(f"| {row['arm']} | {row['submitted']} | {row['tune_wall_s']/60:.2f} | "
                     f"{row['final_median_ms']:.5f} | {row['tuned_functions']} | "
                     f"{row['trained_prediction_calls']} | {row.get('within_5_percent')} |")
    lines += ["", "Timing repeats are not independent search seeds. Actual budgets may differ under time limits.",
              "Prediction audit overhead is recorded separately and remains included in wall time.",
              "Zero trained scoring calls means that arm did not test learned XGBoost ranking beyond warmup.",
              "Legacy initialization/empty-candidate stopping rules are retained and must be considered with task coverage."]
    (study / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args):
    if args.study.exists():
        raise FileExistsError("Use a fresh study directory; do not resume historical arms in place")
    args.study.mkdir(parents=True)
    frozen = identity()
    protocol = {"created_utc": utc(), "model": "resnet18", "shape": [1, 3, 224, 224], "dtype": "float32",
                "weights": "existing frozen random weights; performance, not accuracy study",
                "arms": ARMS, "strategy": CONFIG, "seed": base.SEED, "trials_per_arm": args.trials,
                "arm_seconds": args.arm_seconds, "total_seconds": args.total_seconds,
                "gpu_batch": 10, "verify_pools": {"D64": 64, "D128": 128},
                "draft_formula": "legacy unchanged", "xgb_warmup": 100, "adaptive_training": True,
                "runner": {"number": 3, "repeat": 5, "min_repeat_ms": 100, "feedback": "mean of raw repeats"},
                "checkpoint_every": 100, "identity": frozen,
                "preflight": "D64 and D128: 10 submissions each, excluded from formal results; included in total time limit",
                "limits": "Matched-parameter B, not default MetaSchedule; fixed random weights; single seed",
                "environment": {a: {k:v for k,v in environment(a).items() if k.startswith('TVM_')} for a in ARMS}}
    save(args.study / "protocol.json", protocol)
    shutil.copy2(DRIVER, args.study / "driver_snapshot.py")
    save(args.study / "controller.json", {"pid": os.getpid(), "created_utc": utc()})
    try:
        wait_started = time.time()
        while not memory_ready():
            publish(args.study, "waiting_for_memory", required_free_gib=1.5,
                    available_free_gib=system_memory()["available_physical"] / 2**30)
            if time.time() - wait_started >= args.memory_wait_seconds:
                publish(args.study, "blocked_for_memory")
                return
            time.sleep(15)
        args.global_deadline = time.time() + args.total_seconds
        save(args.study / "budget.json", {"started_epoch": time.time(), "deadline_epoch": args.global_deadline})
        for stage in ("prepare", "untuned"):
            receipt = run_child("prepare" if stage == "prepare" else "evaluate", args,
                                args.study / stage, timeout_s=900, monitor=True)
            if receipt["status"] != "completed":
                raise RuntimeError("Preparation timed out")
        untuned = read(args.study / "untuned/result.json")
        if (untuned["p75_ms"] - untuned["p25_ms"]) / untuned["median_ms"] > 0.05:
            raise RuntimeError("Untuned full-model timing IQR exceeds 5%; tuning not started")
        for arm in ("D64", "D128"):
            folder = args.study / ("preflight_" + arm)
            receipt = run_child("tune", args, folder, arm=arm, trials=10,
                                timeout_s=900, soft_limit_s=600, monitor=True)
            if receipt["status"] != "completed":
                raise RuntimeError("Preflight exceeded its deadline")
            architecture = audit_arm(folder, arm)
            if not architecture["full_verify_pool_observed"]:
                raise RuntimeError("Preflight did not exercise the full requested candidate pool")
            if copy_database(folder, folder / "final_database") == 0:
                raise RuntimeError("Preflight produced no valid measured schedules")
            receipt = run_child("evaluate", args, folder / "final_eval", arm=arm,
                                database=folder / "final_database", timeout_s=900, monitor=True)
            if receipt["status"] != "completed":
                raise RuntimeError("Preflight model evaluation exceeded its deadline")
        for arm in ARMS:
            if identity() != frozen:
                raise RuntimeError("Frozen implementation or model changed")
            if args.global_deadline - time.time() < 1200:
                publish(args.study, "total_time_limit")
                report(args.study)
                return
            folder = args.study / ("formal_" + arm)
            receipt = run_child("tune", args, folder, arm=arm, trials=args.trials,
                                timeout_s=args.arm_seconds, soft_limit_s=args.arm_seconds - 300, monitor=True)
            if not (folder / "database_tuning_record.json").exists():
                raise RuntimeError("No tuning database was produced")
            records = copy_database(folder, folder / "final_database")
            if records == 0:
                raise RuntimeError("No measured schedule available for full-model evaluation")
            if not (folder / "result.json").exists():
                batches = [json.loads(x) for x in (folder / "batches.jsonl").read_text(encoding="utf-8").splitlines()]
                save(folder / "result.json", {"status": "time_limit", "submitted": sum(x["candidates"] for x in batches),
                     "database_records": records, "tune_wall_s": receipt["wall_s"]})
                save(folder / "progress.json", read(folder / "result.json"))
            audit_arm(folder, arm)
            final = run_child("evaluate", args, folder / "final_eval", arm=arm,
                              database=folder / "final_database", timeout_s=900, monitor=True)
            if final["status"] != "completed":
                raise RuntimeError("Final full-model evaluation timed out")
            report(args.study)
        final = run_child("retime", args, args.study / "retime", timeout_s=1800, monitor=True)
        if final["status"] != "completed":
            raise RuntimeError("Independent remeasurement timed out")
        report(args.study)
        publish(args.study, "completed", results=str(args.study / "RESULTS.md"))
    except BaseException as error:
        publish(args.study, "failed", error=repr(error), traceback=traceback.format_exc())
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("run", "prepare", "evaluate", "tune", "retime", "report"))
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--arm", choices=ARMS, default="B")
    parser.add_argument("--trials", type=int, default=400)
    parser.add_argument("--arm-seconds", type=int, default=5400)
    parser.add_argument("--total-seconds", type=int, default=28800)
    parser.add_argument("--memory-wait-seconds", type=int, default=18000)
    parser.add_argument("--database", type=Path)
    args = parser.parse_args()
    args.study = args.study.resolve()
    if args.output:
        args.output = args.output.resolve()
        args.output.mkdir(parents=True, exist_ok=True)
    if args.stage != "run":
        os.environ.update(environment(args.arm))
    try:
        if args.stage == "report":
            report(args.study)
        else:
            {"run": run, "prepare": base.prepare, "evaluate": base.evaluate,
             "tune": tune, "retime": retime}[args.stage](args)
    except BaseException as error:
        if args.output:
            save(args.output / "failure.json", {"utc": utc(), "error": repr(error), "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
