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
"""Serial operator study: frozen 800-submission arms, retiming and Torch reference."""

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

from e1_thread_filter import _git_source_identity, make_parser
from report_single_seed_study import audit
from run_single_seed_study import ROOT, command as previous_command, environment, file_hash, write_json


WORKLOADS = [("M128", "matmul128", "BCD"), ("M256", "matmul256", "CDB"),
             ("CONV", "conv2d_nchw3x3", "DBC"), ("RED", "reduction4096", "BCD")]
SCRIPTS = ["run_operator800_study.py", "report_operator800_study.py", "benchmark_torch_operators.py",
           "e1_thread_filter.py", "p3_observability.py", "clean_workloads.py", "remeasure_top1.py",
           "run_single_seed_study.py", "report_single_seed_study.py", "analyze_p3_mainline.py"]


def matrix():
    rows = []
    for index, (short, workload, arms) in enumerate(WORKLOADS, 1):
        rows.append({"id": f"S{index:02d}", "workload": workload, "genetic_iters": 1,
                     "arm": "B", "trials": 4 if index == 1 else 2,
                     "trials_per_iter": 2, "smoke": True})
        rows.extend({"id": f"{short}G1{arm}01", "workload": workload, "genetic_iters": 1,
                     "arm": arm, "trials": 800, "trials_per_iter": 10, "smoke": False}
                    for arm in arms)
    return rows


def command(spec, output, seed):
    cmd = previous_command(spec, output, seed)
    cmd[cmd.index("--temp-dir") + 1] = str(ROOT / "tmp" / output.name)
    cmd[cmd.index("--repetitions-note") + 1] = "operator800-single-seed"
    cmd.append("--record-model-training")
    return cmd


def runtime():
    return {"python": sys.version,
            "compiler_sha256": file_hash(ROOT / "build-prune/lib/tvm_compiler.dll"),
            "bootstrap_sha256": file_hash(ROOT / "experiments/runtime_bootstrap/sitecustomize.py"),
            "ffi_init_sha256": file_hash(Path(sys.executable).parent / "Lib/site-packages/tvm_ffi/__init__.py"),
            "packages": {name: importlib.metadata.version(name)
                         for name in ("numpy", "xgboost", "apache-tvm-ffi")}}


def verify_sources(protocol):
    if _git_source_identity() != protocol["identity"] or runtime() != protocol["runtime"]:
        raise ValueError("Frozen source/runtime changed; do not mix experiment versions")
    for name, expected in protocol["script_sha256"].items():
        if file_hash(ROOT / "experiments" / name) != expected:
            raise ValueError(f"Frozen experiment script changed: {name}")


def register(output, seed):
    path = output / "protocol.json"
    if path.exists():
        protocol = json.loads(path.read_text(encoding="utf-8"))
        if protocol["seed"] != seed or protocol["order"] != matrix():
            raise ValueError("Registered matrix differs")
        verify_sources(protocol)
        return protocol
    # Validate actual CLI arguments before paying any CUDA/worker startup cost.
    for spec in matrix():
        parsed = make_parser().parse_args(command(spec, output, seed)[5:])
        assert parsed.trials == spec["trials"] and parsed.trials_per_iter == spec["trials_per_iter"]
        assert parsed.record_model_training and parsed.check_correctness and parsed.record_timeline
    protocol = {"registered_local_date": "2026-10-11", "seed": seed,
                "identity": _git_source_identity(), "runtime": runtime(), "order": matrix(),
                "script_sha256": {name: file_hash(ROOT / "experiments" / name) for name in SCRIPTS},
                "common_config": {"population_size": 32, "init_min_unmeasured": 16,
                                  "exact_init_population": True, "init_sample_batch_size": 32,
                                  "eps_greedy": 0.1, "xgb_warmup_samples": 0,
                                  "adaptive_training": False, "draft_ga_verify_ratio": 0.25},
                "targets": {"relative_to_baseline_final": [1.0, 1.01, 1.05], "primary_ratio": 1.0},
                "budget_policy": "800 submitted per arm; failures count; no refill or duplicate padding",
                "scope": "Four FP32 operators, B/C/D, one seed; GA1; 800 submitted maximum per arm",
                "timing_policy": "Tune entry through return; preparation separate; batch endpoints only",
                "model_training": "One XGBModel per arm; accumulate all feature-bearing measurements; "
                                  "retrain booster from scratch after feedback; adaptive training disabled",
                "short_space_policy": "Retain actual submissions and report exhaustion; do not pad to 800",
                "expected_formal_runs": 12, "expected_max_submitted": 9600}
    output.mkdir(parents=True, exist_ok=True)
    write_json(path, protocol)
    for name in SCRIPTS:
        shutil.copy2(ROOT / "experiments" / name, output / ("snapshot_" + name))
    return protocol


def result_path(output, spec):
    paths = list(output.glob(spec["id"] + "_*/result.json"))
    if len(paths) != 1:
        raise ValueError(f"Expected exactly one result for {spec['id']}: {paths}")
    return paths[0]


def execute(cmd, log, output, phase, label, env):
    if log.exists():
        raise FileExistsError(f"Prior attempt retained; choose a new explicit attempt: {log}")
    print(f"START {phase} {label}", flush=True)
    started = time.monotonic()
    with log.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        write_json(output / "progress.json", {"status": "running", "phase": phase, "current": label,
                   "pid": process.pid, "orchestrator_pid": os.getpid(), "log": str(log),
                   "started_utc": datetime.now(timezone.utc).isoformat()})
        code = process.wait()
    receipt = {"phase": phase, "id": label, "pid": process.pid, "argv": cmd, "returncode": code,
               "windows_hex": hex(code & 0xffffffff), "wall_s": time.monotonic() - started,
               "console_sha256": file_hash(log)}
    with (output / "execution.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(receipt, ensure_ascii=False) + "\n")
    if code:
        raise RuntimeError(f"{label} exited {code}; preserved log: {log}")
    print(f"DONE {phase} {label} wall_s={receipt['wall_s']:.2f}", flush=True)


def finalize_group(output, protocol, workload, env, figure_output):
    specs = [s for s in protocol["order"] if s["workload"] == workload and not s["smoke"]]
    originals = {s["arm"]: result_path(output, s) for s in specs}
    target = output / "retiming" / f"{workload}_ga1_R01"
    if not target.exists():
        cmd = [sys.executable, "-u", "-X", "utf8", str(ROOT / "experiments/remeasure_top1.py"),
               "--output", str(target), "--library-dir", str(ROOT / "build-prune/lib"),
               "--terminal-worker-exit"]
        for arm in "BCD":
            cmd += ["--result", f"{arm}={originals[arm]}"]
        execute(cmd, output / f"{workload}.retiming.log", output, "retiming", workload, env)
    timing = json.loads((target / "result.json").read_text(encoding="utf-8"))
    if timing["status"] != "completed" or {a["name"]: a["result_sha256"] for a in timing["arms"]} != {
            arm: file_hash(path) for arm, path in originals.items()}:
        raise ValueError("Retiming status or original-winner provenance differs")
    torch_output = output / "pytorch" / f"{workload}.json"
    if not torch_output.exists():
        torch_env = dict(env)
        torch_env["TVM_EXPERIMENT_NO_TORCH"] = "0"
        cmd = [sys.executable, "-u", "-X", "utf8", str(ROOT / "experiments/benchmark_torch_operators.py"),
               "--workload", workload, "--output", str(torch_output)]
        execute(cmd, output / f"{workload}.pytorch.log", output, "pytorch", workload, torch_env)
    if json.loads(torch_output.read_text(encoding="utf-8"))["status"] != "completed":
        raise ValueError("Incomplete PyTorch reference")
    cmd = [sys.executable, "-u", "-X", "utf8", str(ROOT / "experiments/report_operator800_study.py"),
           "--input", str(output), "--output", str(figure_output)]
    execute(cmd, output / f"{workload}.report.log", output, "reporting", workload, env)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/results/paper_v1/operator800_20261011")
    parser.add_argument("--figure-output", type=Path, default=ROOT / "papers/figures/operator800_20261011")
    parser.add_argument("--seed", type=int, default=20261011)
    parser.add_argument("--register-only", action="store_true")
    parser.add_argument("--only", nargs="+", help="Registered tuning IDs; no group finalization")
    args = parser.parse_args()
    output = args.output.resolve()
    protocol = register(output, args.seed)
    if args.register_only:
        print(json.dumps({"registered": str(output), "formal_runs": 12, "max_submitted": 9600}))
        return
    env = environment()
    try:
        for spec in protocol["order"]:
            if args.only and spec["id"] not in args.only:
                continue
            verify_sources(protocol)
            if list(output.glob(spec["id"] + "_*/result.json")):
                receipts = [json.loads(line) for line in (output / "execution.jsonl").read_text(
                    encoding="utf-8").splitlines() if json.loads(line).get("id") == spec["id"]]
                if len(receipts) != 1 or receipts[0]["returncode"] != 0:
                    raise ValueError(f"No successful process receipt for retained run {spec['id']}")
                row = audit(result_path(output, spec), spec, protocol)
                print(f"RETAIN {spec['id']} submitted={row['budget']['actual_submitted']}", flush=True)
            else:
                if list(output.glob(spec["id"] + "_*/")):
                    raise ValueError(f"Incomplete attempt retained for {spec['id']}; do not overwrite")
                execute(command(spec, output, args.seed), output / (spec["id"] + ".console.log"),
                        output, "smoke" if spec["smoke"] else "tuning", spec["id"], env)
                row = audit(result_path(output, spec), spec, protocol)
            print(f"VALID {spec['id']} submitted={row['budget']['actual_submitted']} "
                  f"finite={row['outcomes'].get('finite', 0)} best_us={row['best_us']}", flush=True)
            group = [s for s in protocol["order"] if s["workload"] == spec["workload"] and not s["smoke"]]
            if not args.only and spec == group[-1]:
                finalize_group(output, protocol, spec["workload"], env, args.figure_output.resolve())
        status = "selected_runs_completed" if args.only else "completed"
        write_json(output / "progress.json", {"status": status, "orchestrator_pid": os.getpid()})
        if not args.only:
            rows = [audit(result_path(output, s), s, protocol) for s in protocol["order"] if not s["smoke"]]
            write_json(output / "COMPLETION.json", {"status": "completed", "formal_runs": len(rows),
                       "submitted": sum(r['budget']['actual_submitted'] for r in rows),
                       "finite_correct": sum(r['outcomes'].get('finite', 0) for r in rows),
                       "report": str(args.figure_output.resolve() / "RESULTS.md")})
    except BaseException:
        write_json(output / "progress.json", {"status": "failed", "exception": traceback.format_exc(),
                   "orchestrator_pid": os.getpid()})
        raise


if __name__ == "__main__":
    main()
