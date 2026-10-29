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
"""Paired ResNet-18 Scout study with real full-model checkpoint measurements."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
DRIVER = Path(__file__).resolve()
MODEL = ROOT / "experiments/fixtures/resnet18_fp32_sm86"
FALLBACK = MODEL / "model.scheduled.json"
SEED = 20261011


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def environment(arm="B"):
    (ROOT / "tmp").mkdir(exist_ok=True)
    env = dict(os.environ)
    env.update({
        "TVM_EXPERIMENT_NO_TORCH": "1", "TVM_FFI_DISABLE_TORCH_C_DLPACK": "1",
        "TVM_LIBRARY_PATH": str(ROOT / "build-prune/lib"),
        "MKL_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_THREADING_LAYER": "SEQUENTIAL",
        "PYTHONUNBUFFERED": "1", "PYTHONUTF8": "1", "TEMP": str(ROOT / "tmp"),
        "TMP": str(ROOT / "tmp"), "TVM_PRUNE_MODE": "0" if arm == "B" else "2",
        "TVM_THREAD_HARD_FILTER_MODE": "off" if arm == "B" else "enforce",
        "TVM_THREAD_HARD_FILTER_POSITION": "trace",
        "TVM_THREAD_CONSTRAINT_SAMPLE_MODE": "off" if arm == "B" else "enforce",
        "TVM_SOFT_FILTER_MODE": "0", "TVM_DRAFT_MODE": "0",
        "TVM_DRAFT_BETA": "0.2", "TVM_DRAFT_EPS": "0.1",
        "TVM_DRAFT_GA_MODE": "2" if arm == "D" else "0",
        "TVM_DRAFT_GA_VERIFY_RATIO": "0.25", "TVM_DRAFT_GA_G3_MODE": "0",
        "TVM_DRAFT_GA_ANNEAL_MODE": "0", "TVM_GA_T2_MODE": "0",
        "TVM_EVO_EXACT_INIT_POPULATION": "1", "TVM_EVO_INIT_SAMPLE_BATCH_SIZE": "32",
        "TVM_DRAFT_PROFILE": "rtx3050ti_sm86_explicit_v1", "TVM_DRAFT_PEAK_GFLOPS": "10752.0",
        "TVM_DRAFT_MEMORY_GBPS": "192.0", "TVM_DRAFT_SM_COUNT": "20",
        "TVM_DRAFT_ACTIVE_BLOCKS_PER_SM": "1", "TVM_DRAFT_SM_PARTITIONS": "4",
        "TVM_DRAFT_REG_CAP": "128.0",
    })
    env["PYTHONPATH"] = os.pathsep.join(str(ROOT / part) for part in
                                        ("experiments/runtime_bootstrap", "python", "experiments"))
    env["PATH"] = str(Path(sys.executable).parent / "Library/bin") + os.pathsep + env["PATH"]
    return env


def terminal_success():
    # The model-readiness worker verified this policy for Windows TVM DLL teardown.
    import ctypes
    sys.stdout.flush()
    sys.stderr.flush()
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    kernel.TerminateProcess.argtypes = [ctypes.c_void_p, ctypes.c_uint]
    kernel.TerminateProcess.restype = ctypes.c_int
    if not kernel.TerminateProcess(kernel.GetCurrentProcess(), 0):
        raise ctypes.WinError(ctypes.get_last_error())


def identity():
    files = [DRIVER, ROOT / "build-prune/lib/tvm_compiler.dll",
             ROOT / "src/s_tir/meta_schedule/search_strategy/evolutionary_search.cc",
             ROOT / "src/s_tir/meta_schedule/mutator/mutate_tile_size.cc",
             ROOT / "src/s_tir/meta_schedule/utils.h",
             ROOT / "src/s_tir/analysis/verify_gpu_code.cc",
             ROOT / "src/s_tir/meta_schedule/postproc/verify_gpu_code.cc",
             ROOT / "python/tvm/s_tir/meta_schedule/relax_integration.py",
             ROOT / "experiments/guard_experiment_resources.py",
             ROOT / "experiments/runtime_bootstrap/sitecustomize.py",
             ROOT / "experiments/p3_observability.py", ROOT / "experiments/e1_thread_filter.py",
             MODEL / "model.relax.json", MODEL / "params.npz", MODEL / "inputs.npz",
             MODEL / "reference.npz", FALLBACK]
    return {path.relative_to(ROOT).as_posix(): sha(path) for path in files}


def target(tvm):
    return tvm.target.Target({"kind": "cuda", "arch": "sm_86", "max_threads_per_block": 1024,
                              "max_num_threads": 1024, "max_shared_memory_per_block": 49152,
                              "thread_warp_size": 32})


def child(stage, study, output, *, arm="B", trials=0, database=None):
    output.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-u", "-X", "utf8", str(DRIVER), stage,
           "--study", str(study), "--output", str(output), "--arm", arm,
           "--trials", str(trials)]
    if database is not None:
        cmd += ["--database", str(database)]
    guard = [sys.executable, "-u", "-X", "utf8", str(ROOT / "experiments/guard_experiment_resources.py"),
             "--output", str(output / "guard"), "--interval", "5", "--physical-reserve-mib", "256",
             "--commit-reserve-gib", "4", "--private-limit-gib", "4", "--consecutive-samples", "3", "--"]
    save(output / "command.json", {"utc": utc(), "argv": guard + cmd, "arm": arm})
    started = time.perf_counter()
    with (output / "launcher.log").open("w", encoding="utf-8") as stream:
        code = subprocess.call(guard + cmd, cwd=ROOT, env=environment(arm),
                               stdout=stream, stderr=subprocess.STDOUT)
    receipt = {"returncode": code, "wall_s": time.perf_counter() - started, "ended_utc": utc()}
    save(output / "receipt.json", receipt)
    status = read(output / "guard/status.json")
    if (code or status.get("status") != "completed" or status.get("returncode") != 0
            or status.get("cleanup", {}).get("errors")
            or status.get("cleanup", {}).get("remaining_live")):
        raise RuntimeError(f"{stage} {arm} failed; preserved {output}")
    return receipt


def prepare(args):
    import tvm
    from tvm import relax, tirx
    from tvm.s_tir.meta_schedule import relax_integration
    mod = tvm.ir.load_json((MODEL / "model.relax.json").read_text(encoding="utf-8"))
    with target(tvm):
        lowered = tvm.transform.Sequential([relax.transform.DecomposeOpsForInference(),
                    relax.transform.CanonicalizeBindings(), relax.get_pipeline("zero")])(mod)
    cached = tvm.ir.load_json(FALLBACK.read_text(encoding="utf-8"))
    current_names = {gv.name_hint for gv in lowered.get_global_vars()}
    if current_names != {gv.name_hint for gv in cached.get_global_vars()}:
        raise ValueError("Cached fallback and prepared model function names differ")
    for gv, func in lowered.functions_items():
        if isinstance(func, relax.Function):
            tvm.ir.assert_structural_equal(func, cached[gv.name_hint], map_free_vars=True)
    tasks = relax_integration.extract_tasks(lowered, target(tvm))
    rows = [{"id": i, "name": t.task_name, "weight": int(t.weight),
             "module_sha256": hashlib.sha256(tvm.ir.save_json(t.dispatched[0]).encode()).hexdigest()}
            for i, t in enumerate(tasks)]
    (args.output / "model.lowered.json").write_text(tvm.ir.save_json(lowered), encoding="utf-8")
    save(args.output / "tasks.json", rows)
    device = tvm.cuda(0)
    if device.compute_version != "8.6":
        raise ValueError("This protocol is frozen for sm_86")
    save(args.output / "result.json", {"status": "completed", "tasks": len(rows),
          "tir_functions": sum(isinstance(f, tirx.PrimFunc) for _, f in lowered.functions_items()),
          "device": device.device_name, "compute_version": device.compute_version})
    terminal_success()


def evaluate(args):
    import numpy as np
    import tvm
    from tvm import relax, tirx
    from tvm.s_tir.meta_schedule.database import JSONDatabase
    started = time.perf_counter()
    tgt = target(tvm)
    cached = tvm.ir.load_json(FALLBACK.read_text(encoding="utf-8"))
    lowered = tvm.ir.load_json((args.study / "prepare/model.lowered.json").read_text(encoding="utf-8"))
    tuned_names = []
    if args.database:
        db = JSONDatabase(work_dir=str(args.database), allow_missing=False)
        with tgt, db:
            lowered = relax.transform.MetaScheduleApplyDatabase(enable_warning=True)(lowered)
    for gv, func in list(lowered.functions_items()):
        if isinstance(func, tirx.PrimFunc):
            if func.attrs and func.attrs.get("tirx.is_scheduled", False):
                tuned_names.append(gv.name_hint)
            else:
                lowered[gv] = cached[gv.name_hint]
    if args.database and not tuned_names:
        raise RuntimeError("No measured schedule was applied to the full model")
    (args.output / "model.scheduled.json").write_text(tvm.ir.save_json(lowered), encoding="utf-8")
    executable = tvm.compile(lowered, target=tgt, relax_pipeline="default_build", tir_pipeline="s_tir")
    compile_s = time.perf_counter() - started
    device = tvm.cuda(0)
    vm = relax.VirtualMachine(executable, device)
    with np.load(MODEL / "params.npz") as archive:
        params = [tvm.runtime.tensor(archive[f"p{i}"], device) for i in range(len(archive.files))]
    checks = []
    with np.load(MODEL / "inputs.npz") as inputs, np.load(MODEL / "reference.npz") as reference:
        for case in inputs.files:
            x = tvm.runtime.tensor(inputs[case], device)
            actual = vm["main"](x, *params).numpy()
            device.sync()
            passed = bool(np.allclose(actual, reference[case], rtol=2e-4, atol=2e-4))
            checks.append({"case": case, "passed": passed, "max_abs_error":
                           float(np.max(np.abs(actual - reference[case]))), "shape": list(actual.shape)})
            if not passed:
                save(args.output / "result.json", {"status": "correctness_failed", "checks": checks})
                raise RuntimeError("Full-model correctness failed: " + case)
        x = tvm.runtime.tensor(inputs["normal"], device)
    vm.save_function("main", "benchmark", x, *params, include_return=False)
    for _ in range(100):
        vm["benchmark"]()
    device.sync()
    samples = list(vm.time_evaluator("benchmark", device, number=3, repeat=30, min_repeat_ms=100)().results)
    save(args.output / "result.json", {"status": "completed", "arm": args.arm,
         "device": str(device), "device_name": device.device_name, "target": str(tgt),
         "checks": checks, "applied_tuned_functions": tuned_names,
         "fallback_functions": [gv.name_hint for gv, func in lowered.functions_items()
                                 if isinstance(func, tirx.PrimFunc) and gv.name_hint not in tuned_names],
         "compile_and_apply_s": compile_s, "latency_s": samples,
         "median_ms": float(np.median(samples) * 1000),
         "p25_ms": float(np.percentile(samples, 25) * 1000),
         "p75_ms": float(np.percentile(samples, 75) * 1000),
         "timing": {"warmup": 100, "repeat": 30, "number": 3, "min_repeat_ms": 100,
                    "scope": "Complete Relax VM forward; resident GPU input and parameters; no output download"}})
    device.sync()
    terminal_success()


def tune(args):
    import numpy as np
    import tvm
    from tvm.ir.utils import derived_object
    from tvm.s_tir.meta_schedule import relax_integration
    from tvm.s_tir.meta_schedule.builder import LocalBuilder
    from tvm.s_tir.meta_schedule.cost_model import XGBModel
    from tvm.s_tir.meta_schedule.database import JSONDatabase
    from tvm.s_tir.meta_schedule.logging import get_loggers_from_work_dir
    from tvm.s_tir.meta_schedule.measure_callback import MeasureCallback, PyMeasureCallback
    from tvm.s_tir.meta_schedule.runner import LocalRunner, EvaluatorConfig
    from tvm.s_tir.meta_schedule.search_strategy import EvolutionarySearch
    from tvm.s_tir.meta_schedule.tune import tune_tasks
    from tvm.s_tir.meta_schedule.tune_context import TuneContext
    from tvm.s_tir.meta_schedule.utils import fork_seed
    from p3_observability import TuneJournal, append_json
    from e1_thread_filter import _parse_draft_ga_log

    random.seed(SEED)
    np.random.seed(SEED)

    lowered = tvm.ir.load_json((args.study / "prepare/model.lowered.json").read_text(encoding="utf-8"))
    extracted = relax_integration.extract_tasks(lowered, target(tvm))
    expected = read(args.study / "prepare/tasks.json")
    if [t.task_name for t in extracted] != [row["name"] for row in expected]:
        raise ValueError("Tuning tasks differ from prepared inventory")
    strategy = EvolutionarySearch(population_size=32, init_measured_ratio=0.2,
                 init_min_unmeasured=16, max_fail_count=5, genetic_num_iters=1,
                 genetic_mutate_prob=0.85, genetic_max_fail_count=10, eps_greedy=0.1)
    names = [f"t{i:02d}" for i in range(len(extracted))]
    loggers = get_loggers_from_work_dir(str(args.output), names)
    contexts = [TuneContext(mod=t.dispatched[0], target=t.target, space_generator="post-order-apply",
                 search_strategy=strategy, task_name=name, logger=logger, rand_state=seed, num_threads=1).clone()
                for t, name, logger, seed in zip(extracted, names, loggers, fork_seed(SEED, n=len(extracted)))]
    builder = LocalBuilder(max_workers=1, timeout_sec=120)
    runner = LocalRunner(timeout_sec=30, evaluator_config=EvaluatorConfig(number=3, repeat=1, min_repeat_ms=100))
    model = XGBModel(num_warmup_samples=0, num_tuning_cores=1, adaptive_training=False)
    database = JSONDatabase(work_dir=str(args.output))
    checkpoints = []
    submitted = 0
    excluded_s = 0.0
    journal = TuneJournal(args.output / "timeline.jsonl", record_model_training=True)

    def checkpoint(count):
        nonlocal excluded_s
        folder = args.output / "checkpoints" / f"n{count:04d}"
        folder.mkdir(parents=True, exist_ok=False)
        for name in ("database_workload.json", "database_tuning_record.json"):
            shutil.copy2(args.output / name, folder / name)
        available_s = time.perf_counter() - started - excluded_s
        receipt = child("evaluate", args.study, folder / "eval", arm=args.arm, database=folder)
        excluded_s += receipt["wall_s"]
        row = {"submitted": count, "tune_elapsed_s": available_s,
               "evaluation_wall_s": receipt["wall_s"], "result": str(folder / "eval/result.json")}
        checkpoints.append(row)
        save(args.output / "checkpoints.json", checkpoints)

    @derived_object
    class Capture(PyMeasureCallback):
        def apply(self, scheduler, task_id, candidates, builds, runs):
            nonlocal submitted
            submitted += len(candidates)
            row = {"utc": utc(), "task_id": int(task_id), "submitted_total": submitted,
                   "candidates": len(candidates), "builder_errors": [r.error_msg for r in builds],
                   "runner_errors": [r.error_msg for r in runs],
                   "run_secs": [None if r.run_secs is None else [float(x) for x in r.run_secs] for r in runs],
                   "trace_sha256": [hashlib.sha256(str(c.sch.trace).encode()).hexdigest() for c in candidates],
                   "xgb_data_size": int(model.data_size),
                   "tune_elapsed_s": time.perf_counter() - started - excluded_s}
            append_json(args.output / "batches.jsonl", row)
            save(args.output / "progress.json", {"status": "tuning", "arm": args.arm, **row})
            if args.trials > 2 and submitted >= (len(checkpoints) + 1) * 200:
                checkpoint(submitted)

    started = time.perf_counter()
    with journal:
        tune_tasks(tasks=contexts, task_weights=[float(t.weight) for t in extracted],
                   work_dir=str(args.output), max_trials_global=args.trials, num_trials_per_iter=min(10, args.trials),
                   builder=builder, runner=runner, database=database, cost_model=model,
                   measure_callbacks=[*MeasureCallback.create("default"), Capture()], task_scheduler="gradient")
    tune_wall_s = time.perf_counter() - started
    in_tune_evaluation_s = excluded_s
    logs = "\n".join(path.read_text(encoding="utf-8", errors="replace")
                       for path in (args.output / "logs").glob("*.log") if "task_scheduler" not in path.name)
    stats = _parse_draft_ga_log(logs)
    errors = []
    nonempty = [event for event in stats["summary_events"] if event["returned"] > 0]
    if sum(event["returned"] for event in nonempty) != submitted:
        errors.append("Scoring telemetry and submitted candidate counts disagree")
    for event in nonempty:
        if args.arm == "D":
            if event["mode"] != "draft-enforce" or event["ga_xgb_calls"] != 0 or event["final_xgb_calls"] != 1:
                errors.append("Scout scoring architecture did not match the registered policy")
        elif event["mode"] != "baseline" or event["ga_xgb_calls"] != 2 or event["final_xgb_calls"] != 0:
            errors.append("Baseline scoring architecture did not match the registered policy")
    save(args.output / "architecture.json", {"passed": not errors, "errors": errors, "draft_ga": stats})
    if errors:
        raise RuntimeError("; ".join(errors))
    if not checkpoints or checkpoints[-1]["submitted"] != submitted:
        checkpoint(submitted)
    records = list(database.get_all_tuning_records())
    finite = sum(bool(r.run_secs) and all(math.isfinite(float(s)) and 0 < float(s) < 1e9
                 for s in r.run_secs) for r in records)
    save(args.output / "result.json", {"status": "completed", "arm": args.arm, "seed": SEED,
         "budget": args.trials, "submitted": submitted, "finite_records": finite,
         "tune_s": tune_wall_s - in_tune_evaluation_s,
         "tune_entry_to_return_wall_s": tune_wall_s,
         "checkpoint_evaluation_total_s": sum(r["evaluation_wall_s"] for r in checkpoints),
         "checkpoints": checkpoints, "architecture_passed": True})
    save(args.output / "progress.json", {"status": "completed", "submitted": submitted, "arm": args.arm})
    terminal_success()


def retime(args):
    """Remeasure frozen final programs in rotated order in one CUDA process."""
    import numpy as np
    import tvm
    from tvm import relax
    device = tvm.cuda(0)
    with np.load(MODEL / "params.npz") as archive:
        params = [tvm.runtime.tensor(archive[f"p{i}"], device) for i in range(len(archive.files))]
    with np.load(MODEL / "inputs.npz") as archive:
        x = tvm.runtime.tensor(archive["normal"], device)
    with np.load(MODEL / "reference.npz") as archive:
        expected = archive["normal"].copy()
    vms, samples, provenance = {}, {}, {}
    for arm in "BCD":
        tuned = read(args.study / ("formal_" + arm) / "result.json")
        source = Path(tuned["checkpoints"][-1]["result"]).parent / "model.scheduled.json"
        mod = tvm.ir.load_json(source.read_text(encoding="utf-8"))
        executable = tvm.compile(mod, target=target(tvm), relax_pipeline="default_build", tir_pipeline="s_tir")
        vm = relax.VirtualMachine(executable, device)
        actual = vm["main"](x, *params).numpy()
        if not np.allclose(actual, expected, rtol=2e-4, atol=2e-4):
            raise RuntimeError("Final remeasurement correctness failed: " + arm)
        vm.save_function("main", "benchmark", x, *params, include_return=False)
        vms[arm], samples[arm] = vm, []
        provenance[arm] = {"source": str(source), "source_sha256": sha(source)}
    orders = ["BCD", "CDB", "DBC"]
    for order in orders:
        for arm in order:
            vm = vms[arm]
            for _ in range(100):
                vm["benchmark"]()
            device.sync()
            samples[arm].extend(vm.time_evaluator("benchmark", device, number=3, repeat=10, min_repeat_ms=100)().results)
    result = {"status": "completed", "device_name": device.device_name, "orders": orders, "arms": {}}
    for arm in "BCD":
        result["arms"][arm] = {**provenance[arm], "latency_s": samples[arm],
            "median_ms": float(np.median(samples[arm]) * 1000),
            "p25_ms": float(np.percentile(samples[arm], 25) * 1000),
            "p75_ms": float(np.percentile(samples[arm], 75) * 1000)}
    save(args.output / "result.json", result)
    device.sync()
    terminal_success()


def status(study, state, **extra):
    save(study / "progress.json", {"utc": utc(), "status": state, "controller_pid": os.getpid(), **extra})


def run(args):
    from guard_experiment_resources import system_memory
    study = args.study
    if study.exists():
        raise FileExistsError("Use a new study directory; prior attempts are retained")
    study.mkdir(parents=True)
    args.created_study = True
    frozen = identity()
    protocol = {"registered_utc": utc(), "model": "resnet18", "input_shape": [1, 3, 224, 224],
        "dtype": "float32", "weights": "Fixed random weights from existing verified export; no accuracy study",
        "arms": {"B": "MetaSchedule", "C": "CS+H", "D": "Scout: CS+H+Draft-GA"},
        "seed": SEED, "numpy_seed": SEED, "python_random_seed": SEED,
        "global_trials_per_arm": args.trials, "trials_per_iter": 10,
        "checkpoint_every_submissions": 200, "population_size": 32, "genetic_iters": 1,
        "init_min_unmeasured": 16, "exact_init_population": True, "eps_greedy": 0.1,
        "xgb_warmup": 0, "adaptive_training": False, "draft_ga_verify_ratio": 0.25,
        "order": ["B", "C", "D"], "pilot_trials_per_arm": 2, "identity": frozen,
        "budget_policy": "Failures count; no refill; no reruns or extension in place",
        "timing_policy": "Tune entry-to-return minus synchronous checkpoint evaluation waits; preparation separate",
        "fallback_policy": "Same cached DLight Fallback per untuned function for every arm",
        "final_remeasurement": {"orders": ["BCD", "CDB", "DBC"], "warmup_per_round": 100,
                                "samples_per_round": 10, "number": 3, "min_repeat_ms": 100},
        "limits": "Single search seed; fixed random weights; technical timing samples are not independent searches",
        "environment": {arm: {k: v for k, v in environment(arm).items() if k.startswith("TVM_")} for arm in "BCD"}}
    save(study / "protocol.json", protocol)
    shutil.copy2(DRIVER, study / "driver_snapshot.py")
    if args.reuse_prepared:
        source = args.reuse_prepared.resolve()
        old_identity = {key.replace("\\", "/"): value for key, value in
                        read(source / "protocol.json")["identity"].items()}
        for name in ("experiments/results/model_smoke_20261025_A01/resnet18/model.relax.json",
                     "experiments/results/model_smoke_20261025_A01/resnet18/params.npz",
                     "experiments/results/model_smoke_20261025_A01/resnet18/inputs.npz",
                     "experiments/results/model_smoke_20261025_A01/resnet18/reference.npz",
                     "experiments/results/model_smoke_20261025_A01/resnet18/gpu_A05_terminal/model.scheduled.json",
                     "build-prune/lib/tvm_compiler.dll"):
            if old_identity[name] != frozen[name]:
                raise ValueError("Prepared artifact provenance differs: " + name)
        retained = []
        for phase in ("prepare", "untuned"):
            if read(source / phase / "result.json")["status"] != "completed":
                raise ValueError("Only completed prerequisites can be reused")
            shutil.copytree(source / phase, study / phase)
            retained.append({"phase": phase, "source": str(source / phase),
                             "files": {str(p.relative_to(source / phase)): sha(p)
                                       for p in (source / phase).rglob("*") if p.is_file()}})
        save(study / "retained_prerequisites.json", retained)
    for phase, arm, budget in [("prepare", "B", 0), ("untuned", "B", 0)] + [
            ("pilot", arm, 2) for arm in "BCD"] + [("formal", arm, args.trials) for arm in "BCD"]:
        if identity() != frozen:
            raise RuntimeError("Registered source, model or compiler changed")
        for attempt in range(3):
            memory = system_memory()
            if memory["available_physical"] < 1.5 * 2**30 or memory["available_commit"] < 6 * 2**30:
                raise RuntimeError("Insufficient free physical memory or commit for the registered study")
            time.sleep(1)
        folder = study / (phase if phase in ("prepare", "untuned") else phase + "_" + arm)
        if phase in ("prepare", "untuned") and args.reuse_prepared:
            continue
        status(study, "running", phase=phase, arm=arm, output=str(folder))
        child("prepare" if phase == "prepare" else "evaluate" if phase == "untuned" else "tune",
              study, folder, arm=arm, trials=budget)
        save(study / "completed_stages" / (folder.name + ".json"), read(folder / "result.json"))
    if identity() != frozen:
        raise RuntimeError("Registered source, model or compiler changed before final retiming")
    status(study, "running", phase="retime", output=str(study / "retime"))
    child("retime", study, study / "retime")
    retiming = read(study / "retime/result.json")["arms"]
    rows = []
    for arm in "BCD":
        result = read(study / ("formal_" + arm) / "result.json")
        measured = read(result["checkpoints"][-1]["result"])
        rows.append({"arm": arm, "submitted": result["submitted"], "tune_s": result["tune_s"],
                     "full_model_median_ms": retiming[arm]["median_ms"],
                     "full_model_p25_ms": retiming[arm]["p25_ms"],
                     "full_model_p75_ms": retiming[arm]["p75_ms"],
                     "checkpoint_median_ms": measured["median_ms"],
                     "tuned_functions": len(measured["applied_tuned_functions"])})
    save(study / "comparison.json", rows)
    baseline = rows[0]
    lines = ["# ResNet-18 Scout end-to-end study", "", "Single seed; FP32 batch 1; fixed random weights.", "",
             "| Arm | Submitted | Tune s | Full-model median ms | Tuned functions | Latency B/arm |",
             "|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['arm']} | {row['submitted']} | {row['tune_s']:.2f} | "
                     f"{row['full_model_median_ms']:.5f} | {row['tuned_functions']} | "
                     f"{baseline['full_model_median_ms'] / row['full_model_median_ms']:.4f} |")
    lines += ["", "Latency measures the complete VM forward with resident GPU inputs and parameters.",
              "Checkpoint evaluation costs are saved separately. Raw failures and histories are retained."]
    (study / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    status(study, "completed", comparison=str(study / "comparison.json"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("run", "prepare", "tune", "evaluate", "retime"))
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--arm", choices=list("BCD"), default="B")
    parser.add_argument("--trials", type=int, default=2000)
    parser.add_argument("--database", type=Path)
    parser.add_argument("--reuse-prepared", type=Path, help="Reuse verified preparation and untuned timing only")
    args = parser.parse_args()
    args.created_study = False
    args.study = args.study.resolve()
    if args.output:
        args.output = args.output.resolve()
        args.output.mkdir(parents=True, exist_ok=True)
    if args.stage != "run":
        os.environ.update(environment(args.arm))
    try:
        {"run": run, "prepare": prepare, "tune": tune, "evaluate": evaluate, "retime": retime}[args.stage](args)
    except BaseException as exc:
        if args.stage == "run" and args.created_study:
            status(args.study, "failed", error=repr(exc))
        elif args.output:
            save(args.output / "failure.json", {"utc": utc(), "error": repr(exc), "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
