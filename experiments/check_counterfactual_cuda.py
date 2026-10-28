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
"""Build and numerically check the one recovered MatMul candidate on CUDA.

No latency benchmark or tuner run. Test inputs/reference are independent of TVM.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

from diagnose_fixed_trace_pool import append_row
from fixed_trace_pool import configure_environment, sha256, write_json


def gpu_status():
    result = subprocess.run([
        "nvidia-smi", "--query-gpu=index,name,driver_version,memory.used,utilization.gpu",
        "--format=csv",
    ], capture_output=True, text=True, check=False)
    return {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counterfactual", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--library-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "command.json", {
        "argv": sys.argv, "utc": datetime.now(timezone.utc).isoformat(), "python": sys.executable,
    })
    shutil.copy2(__file__, args.output / "driver_snapshot.py")
    summary = json.loads((args.counterfactual / "summary.json").read_text(encoding="utf-8"))
    identity = json.loads((args.counterfactual / "identity.json").read_text(encoding="utf-8"))
    events_path = args.counterfactual / "results.jsonl"
    if not summary["completed"] or not summary["original_gate_passed"] or sha256(events_path) != summary["results_sha256"]:
        raise ValueError("Counterfactual gate/identity invalid")
    accepted = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()
                if json.loads(line)["variant"] != "original" and json.loads(line)["accepted"]]
    if [(row["candidate_id"], row["variant"]) for row in accepted] != [("mutation-0037", "no_pipeline")]:
        raise ValueError("This narrow check expects the single recovered candidate")
    mod_path = args.counterfactual / "mutation-0037.no_pipeline.accepted_mod.json"
    identity.update(accepted_mod_sha256=sha256(mod_path), parent_results_sha256=sha256(events_path),
                    checker_sha256=sha256(__file__), gpu_before=gpu_status(),
                    scope="Numerical correctness only; no speedup or latency measurement")
    write_json(args.output / "identity.json", identity)
    print("Loading TVM for one archived accepted candidate; correctness only", flush=True)
    configure_environment(args.library_dir.resolve())
    import numpy as np
    import tvm

    library = Path(tvm.base._LOADED_LIBS["tvm_compiler"]._name).resolve()
    if sha256(library) != identity["loaded_library_sha256"]:
        raise ValueError("DLL changed")
    mod = tvm.ir.load_json(mod_path.read_text(encoding="utf-8"))
    buffers = [mod["main"].buffer_map[param] for param in mod["main"].params]
    if len(buffers) != 3 or any([int(x) for x in buf.shape] != [128, 128] or buf.dtype != "float32" for buf in buffers):
        raise ValueError("Only archived FP32 MatMul128 is supported")
    target = tvm.target.Target(json.loads(identity["target"]))
    device = tvm.cuda(0)
    if not device.exist:
        raise RuntimeError("CUDA device unavailable")
    result = {"candidate_id": "mutation-0037", "variant": "no_pipeline", "status": "running",
              "rtol": 2e-4, "atol": 2e-4, "tests": [], "performance_measured": False}
    write_json(args.output / "result.json", result)
    try:
        start = time.perf_counter()
        executable = tvm.compile(mod, target=target)
        result["build_wall_s"] = time.perf_counter() - start
        print(f"CUDA build completed ({result['build_wall_s']:.3f}s); checking values", flush=True)
        rng = np.random.default_rng(20261008)
        cases = []
        for index in range(3):
            cases.append((f"normal_{index}", rng.normal(size=(128, 128)).astype("float32"),
                          rng.normal(size=(128, 128)).astype("float32")))
        cases.extend([
            ("uniform", rng.uniform(-1, 1, (128, 128)).astype("float32"), rng.uniform(-1, 1, (128, 128)).astype("float32")),
            ("identity", np.eye(128, dtype="float32"), rng.normal(size=(128, 128)).astype("float32")),
            ("zero", np.zeros((128, 128), dtype="float32"), rng.normal(size=(128, 128)).astype("float32")),
        ])
        with (args.output / "checks.jsonl").open("x", encoding="utf-8") as stream:
            for label, lhs, rhs in cases:
                expected = lhs.astype("float64") @ rhs.astype("float64")
                a_dev, b_dev = tvm.runtime.tensor(lhs, device), tvm.runtime.tensor(rhs, device)
                # Two different output sentinels check complete output initialization.
                for sentinel in (float("nan"), 123.0):
                    out = tvm.runtime.tensor(np.full((128, 128), sentinel, dtype="float32"), device)
                    executable(a_dev, b_dev, out)
                    device.sync()
                    actual = out.numpy()
                    error = np.abs(actual.astype("float64") - expected)
                    tolerance = result["atol"] + result["rtol"] * np.abs(expected)
                    row = {"case": label, "output_sentinel": str(sentinel),
                           "passed": bool(np.isfinite(actual).all() and (error <= tolerance).all()),
                           "max_abs_error": float(error.max()),
                           "max_tolerance_ratio": float((error / tolerance).max())}
                    np.savez(args.output / f"{label}.{str(sentinel)}.npz", a=lhs, b=rhs, expected=expected, actual=actual)
                    append_row(stream, row)
                    result["tests"].append(row)
                    if not row["passed"]:
                        raise AssertionError(f"Numerical mismatch: {row}")
        result["status"] = "passed"
    except Exception:
        result["status"] = "failed"
        result["exception"] = traceback.format_exc()
        raise
    finally:
        result["gpu_after"] = gpu_status()
        write_json(args.output / "result.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
