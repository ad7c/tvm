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
"""Bounded CUDA execution proof using the cached ResNet-18 smoke artifacts."""

import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--scheduled", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=30)
    args = parser.parse_args()
    if not 1 <= args.seconds <= 60:
        raise ValueError("Proof duration must be between 1 and 60 seconds")
    args.output.mkdir(parents=True, exist_ok=True)
    result = {
        "status": "importing_tvm", "pid": os.getpid(), "hostname": socket.gethostname(),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Cached untuned full-model CUDA execution proof; fixed random weights",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "scheduled_sha256": hashlib.sha256(args.scheduled.read_bytes()).hexdigest(),
    }

    def record(stage):
        result["status"] = stage
        (args.output / "proof.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(datetime.now(timezone.utc).isoformat(), stage, flush=True)

    record("importing_tvm")
    import numpy as np
    import tvm
    from tvm import relax

    device = tvm.cuda(0)
    if not device.exist or device.dlpack_device_type() != 2:
        raise RuntimeError("A real CUDA device is required")
    result.update(device=str(device), device_name=device.device_name,
                  compute_version=device.compute_version, device_type=device.dlpack_device_type())
    target = tvm.target.Target({
        "kind": "cuda", "arch": "sm_" + device.compute_version.replace(".", ""),
        "max_threads_per_block": 1024, "max_num_threads": 1024,
        "max_shared_memory_per_block": 49152, "thread_warp_size": 32,
    })
    result["target"] = str(target)
    record("compiling_cached_schedule")
    scheduled = tvm.ir.load_json(args.scheduled.read_text(encoding="utf-8"))
    executable = tvm.compile(scheduled, target=target,
                             relax_pipeline="default_build", tir_pipeline="s_tir")
    vm = relax.VirtualMachine(executable, device)
    with np.load(args.source / "params.npz") as archive:
        params = [tvm.runtime.tensor(archive[f"p{i}"], device) for i in range(len(archive.files))]
    with np.load(args.source / "inputs.npz") as archive:
        x = tvm.runtime.tensor(archive["normal"], device)
    with np.load(args.source / "reference.npz") as archive:
        expected = archive["normal"].copy()
    result.update(input_device=str(x.device), parameter_devices=sorted({str(p.device) for p in params}),
                  parameter_tensor_count=len(params))
    record("warming_up")
    for _ in range(3):
        output = vm["main"](x, *params)
        device.sync()
    if not np.allclose(output.numpy(), expected, rtol=2e-4, atol=2e-4):
        raise RuntimeError("Warmup correctness check failed")
    query = ["nvidia-smi", "--query-gpu=timestamp,index,name,uuid,pci.bus_id,utilization.gpu,memory.used,power.draw,temperature.gpu",
             "--format=csv,noheader,nounits", "-l", "1"]
    with (args.output / "gpu_samples.csv").open("w", encoding="utf-8") as samples:
        sampler = subprocess.Popen(query, stdout=samples, stderr=subprocess.STDOUT)
        try:
            result["loop_started_utc"] = datetime.now(timezone.utc).isoformat()
            record("running_continuous_cuda_inference")
            start = time.perf_counter()
            count = 0
            while time.perf_counter() - start < args.seconds:
                output = vm["main"](x, *params)
                device.sync()
                count += 1
            elapsed = time.perf_counter() - start
            result.update(iterations=count, elapsed_s=elapsed,
                          loop_ended_utc=datetime.now(timezone.utc).isoformat())
        finally:
            sampler.terminate()
            sampler.wait(timeout=10)
    actual = output.numpy()
    passed = bool(np.allclose(actual, expected, rtol=2e-4, atol=2e-4))
    result.update(output_device=str(output.device), output_shape=list(actual.shape),
                  correctness_passed=passed, max_abs_error=float(np.max(np.abs(actual - expected))),
                  exit_policy="Windows isolated terminal worker; results flushed before DLL detach")
    if not passed:
        record("correctness_failed")
        raise RuntimeError("Final correctness check failed")
    record("completed")
    device.sync()
    sys.stdout.flush()
    sys.stderr.flush()
    # Match the already validated smoke-worker policy for Windows TVM DLL teardown.
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    kernel.TerminateProcess.argtypes = [ctypes.c_void_p, ctypes.c_uint]
    kernel.TerminateProcess.restype = ctypes.c_int
    if not kernel.TerminateProcess(kernel.GetCurrentProcess(), 0):
        raise ctypes.WinError(ctypes.get_last_error())


if __name__ == "__main__":
    main()
