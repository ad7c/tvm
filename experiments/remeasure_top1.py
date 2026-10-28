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
"""Independently rebuild and interleave timing of each arm's archived top-1.

No tuning, resampling, or selection using the new measurements.
"""

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import sys
import time
import traceback

from fixed_trace_pool import configure_environment, sha256, write_json
from p3_observability import append_json, gpu_status
from clean_workloads import SPECS, numerical_cases


def record_mean(record):
    values = [float(value) for value in record.run_secs] if record.run_secs is not None else []
    return statistics.mean(values) if values and all(0 < value < 1e9 and math.isfinite(value) for value in values) else math.inf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", action="append", required=True, help="ARM=path/to/result.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--library-dir", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=10, help="Per round; default total is 30 samples/arm")
    parser.add_argument("--terminal-worker-exit", action="store_true",
                        help="Windows-only: flush completed artifacts and exit before native DLL finalizers")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "command.json", {"argv": sys.argv, "utc": datetime.now(timezone.utc).isoformat()})
    shutil.copy2(__file__, args.output / "driver_snapshot.py")
    shutil.copy2(Path(__file__).with_name("clean_workloads.py"), args.output / "clean_workloads_snapshot.py")
    summary = {"status": "preparing", "scope": "Independent retiming of fixed archived winners, not another search seed",
               "warmup_each_round": args.warmup, "rounds": args.rounds, "repeat_per_round": args.repeat,
               "number": 3, "min_repeat_ms": 100, "input_seed": 20261010,
               "rtol": 2e-4, "atol": 2e-4, "gpu_before": gpu_status(), "arms": [], "measurements": []}
    summary["exit_policy"] = "Windows TerminateProcess after completed artifacts" if args.terminal_worker_exit else "normal Python shutdown"
    if args.terminal_worker_exit and os.name != "nt":
        raise ValueError("Terminal worker exit is scoped to the Windows native-finalization workaround")
    write_json(args.output / "result.json", summary)
    try:
        configure_environment(args.library_dir.resolve())
        os.environ["CUDA_HOME"] = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"
        os.environ["CUDA_PATH"] = os.environ["CUDA_HOME"]
        print("Importing TVM for archived top-1 retiming", flush=True)
        import numpy as np
        import tvm
        from tvm.s_tir.meta_schedule import database
        from tvm.s_tir.meta_schedule.builder.local_builder import default_build

        lib = Path(tvm.base._LOADED_LIBS["tvm_compiler"]._name).resolve()
        summary["loaded_library"] = str(lib)
        summary["loaded_library_sha256"] = sha256(lib)
        device = tvm.cuda(0)
        executable_arms = []
        for specification in args.result:
            name, result_name = specification.split("=", 1)
            result_path = Path(result_name).resolve()
            original = json.loads(result_path.read_text(encoding="utf-8"))
            if original["loaded_library_sha256"] != summary["loaded_library_sha256"]:
                raise ValueError("Retiming DLL must match the archived tuning DLL")
            workload = original["workload"]
            if workload not in ("matmul128", "matmul256", *SPECS):
                raise ValueError(f"Unsupported retiming workload: {workload}")
            db = database.JSONDatabase(work_dir=str(result_path.parent), allow_missing=False)
            records = list(db.get_all_tuning_records())
            selected_index, record = min(enumerate(records), key=lambda pair: record_mean(pair[1]))
            latency = record_mean(record) * 1e6
            if not math.isclose(latency, original["measurement"]["best_latency_us"], rel_tol=1e-9):
                raise ValueError("Archived top-1 disagrees with database")
            candidate = record.as_measure_candidate()
            target = tvm.target.Target(json.loads(original["target"]))
            arm_dir = args.output / name
            arm_dir.mkdir()
            (arm_dir / "scheduled_mod.json").write_text(tvm.ir.save_json(candidate.sch.mod), encoding="utf-8")
            (arm_dir / "trace.txt").write_text(str(record.trace), encoding="utf-8")
            print(f"{name}: rebuild archived record {selected_index}, original {latency:.4f} us", flush=True)
            started = time.perf_counter()
            module = default_build(candidate.sch.mod, target, None)
            build_s = time.perf_counter() - started
            if workload in SPECS:
                cases = list(numerical_cases(workload))
                timing_arrays = cases[0][1]
            else:
                n = 128 if workload == "matmul128" else 256
                rng = np.random.default_rng(20261010)
                a = rng.normal(size=(n, n)).astype("float32")
                b = rng.normal(size=(n, n)).astype("float32")
                pairs = [("normal", a, b), ("uniform", rng.uniform(-1, 1, (n, n)).astype("float32"), rng.uniform(-1, 1, (n, n)).astype("float32")),
                         ("identity", np.eye(n, dtype="float32"), b), ("zero", np.zeros((n, n), dtype="float32"), b)]
                cases = [(label, [lhs, rhs], lhs.astype("float64") @ rhs.astype("float64"))
                         for label, lhs, rhs in pairs]
                timing_arrays = [a, b]
            checks = []
            for label, arrays, reference in cases:
                buffers = [tvm.runtime.tensor(value, device) for value in arrays]
                buffers.append(tvm.runtime.tensor(np.full(reference.shape, np.nan, dtype="float32"), device))
                module[module.entry_name](*buffers)
                device.sync()
                actual = buffers[-1].numpy()
                error = np.abs(actual.astype("float64") - reference)
                tolerance = summary["atol"] + summary["rtol"] * np.abs(reference)
                row = {"case": label, "passed": bool(np.isfinite(actual).all() and (error <= tolerance).all()),
                       "max_abs_error": float(error.max()), "max_tolerance_ratio": float((error / tolerance).max())}
                checks.append(row)
                append_json(arm_dir / "checks.jsonl", row)
                if not row["passed"]:
                    raise AssertionError(f"{name} numerical failure: {row}")
            timing_args = [tvm.runtime.tensor(value, device) for value in timing_arrays]
            timing_args.append(tvm.runtime.tensor(np.zeros(cases[0][2].shape, dtype="float32"), device))
            summary["arms"].append({"name": name, "workload": workload, "result_path": str(result_path), "result_sha256": sha256(result_path),
                                    "database_sha256": sha256(result_path.parent / "database_tuning_record.json"),
                                    "record_index": selected_index, "original_latency_us": latency,
                                    "trace_sha256": sha256(arm_dir / "trace.txt"), "scheduled_mod_sha256": sha256(arm_dir / "scheduled_mod.json"),
                                    "workload_sha256": __import__("hashlib").sha256(tvm.ir.save_json(record.workload.mod).encode()).hexdigest(),
                                    "build_wall_s": build_s, "checks": checks, "samples_us": []})
            executable_arms.append((module, timing_args))
            write_json(args.output / "result.json", summary)
        if len({row["workload_sha256"] for row in summary["arms"]}) != 1:
            raise ValueError("All retiming arms must have the same workload")
        summary["status"] = "measuring"
        for round_index in range(args.rounds):
            order = [(round_index + index) % len(executable_arms) for index in range(len(executable_arms))]
            for arm_index in order:
                module, timing_args = executable_arms[arm_index]
                arm = summary["arms"][arm_index]
                before = gpu_status()
                for _ in range(args.warmup):
                    module[module.entry_name](*timing_args)
                device.sync()
                timer = module.time_evaluator(module.entry_name, device, number=3, repeat=args.repeat, min_repeat_ms=100)
                samples = [float(value) * 1e6 for value in timer(*timing_args).results]
                if any(not math.isfinite(value) or value <= 0 for value in samples):
                    raise ValueError("Invalid timing sample")
                arm["samples_us"].extend(samples)
                row = {"round": round_index, "arm": arm["name"], "samples_us": samples,
                       "gpu_before": before, "gpu_after": gpu_status()}
                summary["measurements"].append(row)
                append_json(args.output / "timings.jsonl", row)
                write_json(args.output / "result.json", summary)
                print(f"round {round_index + 1}, {arm['name']}: median {statistics.median(samples):.4f} us", flush=True)
        for arm in summary["arms"]:
            arm["median_us"] = statistics.median(arm["samples_us"])
            arm["p25_us"], arm["p75_us"] = [float(x) for x in np.percentile(arm["samples_us"], [25, 75])]
        summary["status"] = "completed"
    except BaseException:
        summary["status"] = "failed"
        summary["exception"] = traceback.format_exc()
        raise
    finally:
        summary["gpu_after"] = gpu_status()
        write_json(args.output / "result.json", summary)
    print(json.dumps([{key: row[key] for key in ("name", "original_latency_us", "median_us", "p25_us", "p75_us")} for row in summary["arms"]], indent=2), flush=True)
    if args.terminal_worker_exit:
        # These are isolated benchmark workers. All measurements, files and
        # correctness checks are complete; keep native objects alive until
        # process termination to avoid Windows TVM/FFI teardown fail-fast.
        # Failure paths still raise normally and never reach this branch.
        print("TERMINAL: final device sync", flush=True)
        device.sync()
        print("TERMINAL: artifacts flushed; terminate this isolated worker before DLL detach", flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        # Windows CRT _exit still runs DLL detach via ExitProcess. The Win32
        # terminal-worker pattern bypasses detach, which reproduces fail-fast
        # here even after all measurements and writes have completed.
        import ctypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        kernel.TerminateProcess.argtypes = [ctypes.c_void_p, ctypes.c_uint]
        kernel.TerminateProcess.restype = ctypes.c_int
        if not kernel.TerminateProcess(kernel.GetCurrentProcess(), 0):
            raise ctypes.WinError(ctypes.get_last_error())


if __name__ == "__main__":
    main()
