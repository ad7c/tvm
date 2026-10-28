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
"""Run the frozen, single-seed study serially, preserving every attempt."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from e1_thread_filter import _git_source_identity


ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def environment():
    result = dict(os.environ)
    result.update({
        "TVM_EXPERIMENT_NO_TORCH": "1", "TVM_FFI_DISABLE_TORCH_C_DLPACK": "1",
        "MKL_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_THREADING_LAYER": "SEQUENTIAL",
        "PYTHONUNBUFFERED": "1", "PYTHONUTF8": "1",
    })
    result["PYTHONPATH"] = os.pathsep.join(str(ROOT / p) for p in (
        "experiments/runtime_bootstrap", "python", "experiments"
    ))
    result["PATH"] = str(Path(sys.executable).parent / "Library/bin") + os.pathsep + result["PATH"]
    return result


def run_spec(identifier, workload, ga, arm, smoke=False):
    return {"id": identifier, "workload": workload, "genetic_iters": ga, "arm": arm,
            "trials": 2 if smoke else 32, "trials_per_iter": 2 if smoke else 4,
            "smoke": smoke}


def matrix():
    runs = [run_spec("S01", "matmul128", 1, "B", True)]
    for ga, arms in [(1, "BCD"), (4, "DCB")]:
        runs += [run_spec(f"M128G{ga}{arm}01", "matmul128", ga, arm) for arm in arms]
    runs += [run_spec(f"M256G1{arm}01", "matmul256", 1, arm) for arm in "BCD"]
    runs += [run_spec("S02", "conv2d_nchw3x3", 1, "B", True)]
    runs += [run_spec(f"CONVG1{arm}01", "conv2d_nchw3x3", 1, arm) for arm in "CDB"]
    runs += [run_spec("S03", "reduction4096", 1, "B", True)]
    runs += [run_spec(f"REDG1{arm}01", "reduction4096", 1, arm) for arm in "DBC"]
    return runs


def command(spec, output, seed):
    switches = {"B": ("off", "off", "off"), "C": ("enforce", "enforce", "off"),
                "D": ("enforce", "enforce", "enforce")}[spec["arm"]]
    return [sys.executable, "-u", "-X", "utf8", str(ROOT / "experiments/e1_thread_filter.py"),
            "run", "--workload", spec["workload"], "--seed", str(seed), "--run-id", spec["id"],
            "--trials", str(spec["trials"]), "--trials-per-iter", str(spec["trials_per_iter"]),
            "--population-size", "32", "--init-min-unmeasured", "16", "--exact-init-population",
            "--init-sample-batch-size", "32", "--genetic-iters", str(spec["genetic_iters"]),
            "--eps-greedy", "0.1", "--xgb-warmup-samples", "0", "--build-timeout", "120",
            "--run-timeout", "30", "--hard-filter-position", "trace", "--mode", switches[0],
            "--constraint-sample-mode", switches[1], "--draft-ga-mode", switches[2],
            "--draft-ga-verify-ratio", "0.25", "--record-timeline", "--check-correctness",
            "--temp-dir", str(ROOT / "tmp/single_seed_20261010"), "--work-root", str(output),
            "--repetitions-note", "single-seed-development-study"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/results/paper_v1/single_seed_20261010")
    parser.add_argument("--seed", type=int, default=20261011)
    parser.add_argument("--only", nargs="+", help="Run selected registered IDs; completed runs are retained")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    identity = _git_source_identity()
    runtime = {
        "python": sys.version, "compiler_sha256": file_hash(ROOT / "build-prune/lib/tvm_compiler.dll"),
        "bootstrap_sha256": file_hash(ROOT / "experiments/runtime_bootstrap/sitecustomize.py"),
        "ffi_init_sha256": file_hash(Path(sys.executable).parent / "Lib/site-packages/tvm_ffi/__init__.py"),
        "packages": {name: importlib.metadata.version(name) for name in ["numpy", "xgboost", "apache-tvm-ffi"]},
        "torch_policy": "Opt-in import exclusion for NumPy/CUDA experiments; installed packages unchanged",
    }
    protocol = {
        "registered_local_date": "2026-10-10", "seed": args.seed, "identity": identity,
        "runtime": runtime, "order": matrix(),
        "common_config": {"population_size": 32, "init_min_unmeasured": 16, "exact_init_population": True,
                          "init_sample_batch_size": 32, "eps_greedy": 0.1, "xgb_warmup_samples": 0,
                          "adaptive_training": False, "draft_ga_verify_ratio": 0.25,
                          "record_timeline": True, "check_correctness": True,
                          "soft_legacy_draft_g1_g2_g3_g4": "off"},
        "targets": {"matmul128": {"absolute_us": [40, 50]},
                    "extensions": {"relative_to_baseline_final": [1.0, 1.01, 1.05],
                                   "interpretation": "Retrospective descriptive thresholds, following Pruner's baseline endpoint comparison"}},
        "budget_policy": "32 submitted, failures count, no refill. Space exhaustion is reported, not duplicated.",
        "timing_policy": "Full tune start through completion; targets at batch completion, no interpolation; preparation separate",
        "scope": "One search seed; GA 1/4 on MatMul128, GA1 across four workloads; no cross-seed stability claim",
    }
    protocol_path = output / "protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text(encoding="utf-8")) != protocol:
            raise ValueError("Frozen protocol changed; use a new output directory")
    else:
        write_json(protocol_path, protocol)
    env = environment()
    ledger_path = output / "execution.jsonl"
    for spec in protocol["order"]:
        if args.only and spec["id"] not in args.only:
            continue
        matches = list(output.glob(spec["id"] + "_*/result.json"))
        if matches:
            result = json.loads(matches[0].read_text(encoding="utf-8"))
            if not result["architecture_validation"]["passed"]:
                raise ValueError(f"Existing invalid run {spec['id']}; retain and review before continuing")
            print(f"RETAIN {spec['id']}", flush=True)
            continue
        if list(output.glob(spec["id"] + "_*/")):
            raise ValueError(f"Incomplete attempt {spec['id']} exists; choose a new ID explicitly")
        if _git_source_identity() != identity:
            raise ValueError("Production/driver source changed after registration")
        cmd = command(spec, output, args.seed)
        log_path = output / (spec["id"] + ".console.log")
        if log_path.exists():
            raise FileExistsError(log_path)
        started = time.monotonic()
        print(f"START {spec['id']} {spec['workload']} GA{spec['genetic_iters']} {spec['arm']}", flush=True)
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            write_json(output / "progress.json", {"status": "running", "current": spec,
                       "pid": process.pid, "log": str(log_path), "started_utc": datetime.now(timezone.utc).isoformat()})
            returncode = process.wait()
        entry = {**spec, "returncode": returncode, "wall_s": time.monotonic() - started,
                 "console_sha256": file_hash(log_path)}
        with ledger_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(entry) + "\n")
        matches = list(output.glob(spec["id"] + "_*/result.json"))
        if returncode != 0 or len(matches) != 1:
            write_json(output / "progress.json", {"status": "failed", "current": spec, "entry": entry})
            raise RuntimeError(f"Run failed: {spec['id']}; full log: {log_path}")
        result = json.loads(matches[0].read_text(encoding="utf-8"))
        if not result["architecture_validation"]["passed"]:
            raise ValueError(f"Run contract failed: {spec['id']}")
        print(f"DONE {spec['id']} submitted={result['budget']['actual_submitted']} "
              f"finite={result['measurement']['finite_records']} "
              f"best_us={result['measurement']['best_latency_us']} tune_s={result['timing']['tune_call_s']:.2f}", flush=True)
    write_json(output / "progress.json", {"status": "selected_runs_completed" if args.only else "completed"})


if __name__ == "__main__":
    main()
