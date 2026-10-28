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
"""Separate-process FP32 PyTorch reference for the four operator workloads."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import time
import traceback

from clean_workloads import numerical_cases


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def cases_for(workload):
    import numpy as np

    if workload not in ("matmul128", "matmul256"):
        return list(numerical_cases(workload))
    n = 128 if workload == "matmul128" else 256
    rng = np.random.default_rng(20261010)
    a, b = [rng.normal(size=(n, n)).astype("float32") for _ in range(2)]
    pairs = [("normal", a, b),
             ("uniform", rng.uniform(-1, 1, (n, n)).astype("float32"),
              rng.uniform(-1, 1, (n, n)).astype("float32")),
             ("identity", np.eye(n, dtype="float32"), b), ("zero", np.zeros_like(a), b)]
    return [(label, [lhs, rhs], lhs.astype("float64") @ rhs.astype("float64"))
            for label, lhs, rhs in pairs]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", required=True,
                        choices=["matmul128", "matmul256", "conv2d_nchw3x3", "reduction4096"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    result = {"status": "preparing", "workload": args.workload, "input_seed": 20261010,
              "utc": datetime.now(timezone.utc).isoformat(), "dtype": "float32",
              "rtol": 2e-4, "atol": 2e-4, "warmup_each_round": 100,
              "rounds": 3, "repeat_per_round": 10, "min_sample_ms": 100,
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "reference_sha256": hashlib.sha256(Path(__file__).with_name("clean_workloads.py").read_bytes()).hexdigest(),
              "scope": "Resident CUDA inputs; CUDA-event elapsed time including gaps between Python launches. "
                       "Conv2D uses native output allocation; matmul/reduction reuse output buffers. "
                       "Separate-process reference, not interleaved with TVM arms.",
              "checks": [], "measurements": [], "samples_us": []}
    write_json(args.output, result)
    started = time.perf_counter()
    try:
        print("Importing standalone PyTorch reference", flush=True)
        import numpy as np
        import torch
        import torch.nn.functional as functional

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA PyTorch is unavailable")
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = True
        device = torch.device("cuda:0")
        properties = torch.cuda.get_device_properties(device)
        result.update({"torch_version": torch.__version__, "cuda_version": torch.version.cuda,
                       "gpu": properties.name, "gpu_memory_bytes": properties.total_memory,
                       "tf32": False, "cudnn_benchmark": True})

        def operation(inputs, output):
            if args.workload.startswith("matmul"):
                return torch.mm(inputs[0], inputs[1], out=output)
            if args.workload == "conv2d_nchw3x3":
                return functional.conv2d(inputs[0], inputs[1], stride=1, padding=1)
            return torch.sum(inputs[0], dim=1, out=output)

        cases = cases_for(args.workload)
        with torch.inference_mode():
            for label, arrays, expected in cases:
                inputs = [torch.from_numpy(array).to(device) for array in arrays]
                output = torch.full(expected.shape, float("nan"), device=device, dtype=torch.float32)
                actual = operation(inputs, output).cpu().numpy()
                error = np.abs(actual.astype("float64") - expected)
                tolerance = result["atol"] + result["rtol"] * np.abs(expected)
                check = {"case": label, "passed": bool(np.isfinite(actual).all() and (error <= tolerance).all()),
                         "max_abs_error": float(error.max()),
                         "max_tolerance_ratio": float((error / tolerance).max())}
                result["checks"].append(check)
                if not check["passed"]:
                    raise AssertionError(f"Incorrect PyTorch reference: {check}")
            inputs = [torch.from_numpy(array).to(device) for array in cases[0][1]]
            output = torch.empty(cases[0][2].shape, device=device, dtype=torch.float32)
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

            def timed(number):
                begin.record()
                for _ in range(number):
                    operation(inputs, output)
                end.record()
                end.synchronize()
                return begin.elapsed_time(end)

            for _ in range(100):
                operation(inputs, output)
            torch.cuda.synchronize()
            number = 100
            while True:
                elapsed = timed(number)
                if elapsed >= 100 or number >= 100000:
                    break
                number = min(100000, max(number + 1, int(number * 110 / max(elapsed, 0.001))))
            if elapsed < 100:
                raise RuntimeError("Cannot reach minimum sample duration")
            result["number"] = number
            result["status"] = "measuring"
            for round_index in range(3):
                for _ in range(100):
                    operation(inputs, output)
                torch.cuda.synchronize()
                samples = []
                for sample_index in range(10):
                    elapsed = timed(number)
                    latency = elapsed * 1000 / number
                    if not math.isfinite(latency) or latency <= 0:
                        raise ValueError("Invalid CUDA-event timing")
                    samples.append(latency)
                    result["measurements"].append({"round": round_index, "sample": sample_index,
                                                   "elapsed_ms": elapsed, "number": number,
                                                   "latency_us": latency})
                result["samples_us"].extend(samples)
                write_json(args.output, result)
                print(f"round {round_index + 1}: median {statistics.median(samples):.4f} us", flush=True)
            result["median_us"] = statistics.median(result["samples_us"])
            result["p25_us"], result["p75_us"] = map(float, np.percentile(result["samples_us"], [25, 75]))
            torch.cuda.synchronize()
        result["status"] = "completed"
    except BaseException:
        result["status"] = "failed"
        result["exception"] = traceback.format_exc()
        raise
    finally:
        result["process_wall_s"] = time.perf_counter() - started
        write_json(args.output, result)


if __name__ == "__main__":
    main()
