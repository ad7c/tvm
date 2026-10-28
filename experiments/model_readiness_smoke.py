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
"""Offline full-model import, CUDA execution and numerical readiness checks."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parents[1]


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def progress(folder, stage, **extra):
    save(folder / "progress.json", {"utc": datetime.now(timezone.utc).isoformat(),
                                   "stage": stage, **extra})
    print(stage, flush=True)


def export_model(folder, name):
    import numpy as np
    import torch
    import torchvision
    from torchvision import models

    torch.set_num_threads(1)
    torch.manual_seed(20261025)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    progress(folder, "create_model", model=name)
    model = getattr(models, name)(weights=None).eval()
    rng = np.random.default_rng(20261025)
    arrays = {"normal": rng.normal(size=(1, 3, 224, 224)).astype("float32"),
              "uniform": rng.uniform(-1, 1, (1, 3, 224, 224)).astype("float32"),
              "zero": np.zeros((1, 3, 224, 224), dtype="float32")}
    progress(folder, "torch_export")
    exported = torch.export.export(model, (torch.from_numpy(arrays["normal"]),))
    torch.export.save(exported, folder / "model.pt2")
    progress(folder, "import_relax")
    import tvm
    from tvm import relax
    from tvm.relax.frontend import detach_params
    from tvm.relax.frontend.torch import from_exported_program

    mod = from_exported_program(exported, keep_params_as_input=True, unwrap_unit_return_tuple=True)
    mod, params = detach_params(mod)
    (folder / "model.relax.json").write_text(tvm.ir.save_json(mod), encoding="utf-8")
    np.savez(folder / "params.npz", **{f"p{i}": p.numpy() for i, p in enumerate(params["main"])})
    (folder / "model.relax.py").write_text(mod.script(), encoding="utf-8")
    progress(folder, "torch_full_model_cuda_reference")
    torch.cuda.reset_peak_memory_stats()
    model = model.cuda()
    references = {}
    timings = {}
    with torch.inference_mode():
        for case, array in arrays.items():
            x = torch.from_numpy(array).cuda()
            torch.cuda.synchronize()
            started = time.perf_counter()
            output = model(x)
            torch.cuda.synchronize()
            timings[case] = time.perf_counter() - started
            references[case] = output.cpu().numpy()
    np.savez(folder / "inputs.npz", **arrays)
    np.savez(folder / "reference.npz", **references)
    result = {"status": "completed", "model": name, "seed": 20261025,
              "weights": "fixed random initialization; architecture readiness only",
              "input_shape": [1, 3, 224, 224], "dtype": "float32",
              "torch": torch.__version__, "torchvision": torchvision.__version__,
              "tvm": tvm.__version__, "parameter_count": sum(p.numel() for p in model.parameters()),
              "exported_parameter_tensors": len(params["main"]),
              "torch_cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
              "torch_cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
              "torch_single_forward_wall_s": timings,
              "output_shapes": {key: list(v.shape) for key, v in references.items()},
              "compiler_sha256": hashlib.sha256((ROOT / "build-prune/lib/tvm_compiler.dll").read_bytes()).hexdigest()}
    save(folder / "export_result.json", result)
    progress(folder, "export_completed")


def compile_and_run(folder, name, source=None, scheduler="dlight", terminal_worker_exit=False):
    import numpy as np
    import tvm
    from tvm import relax, tirx
    from tvm.s_tir import dlight as dl
    from tvm.s_tir.meta_schedule import relax_integration

    source = source or folder
    result = {"status": "running", "model": name, "formal_tuning_submissions": 0,
              "scope": "Untuned full-model execution and numerical check; fixed random weights",
              "scheduler": scheduler, "execution_device": "cuda:0", "stages_s": {}, "checks": [],
              "exit_policy": "Windows isolated terminal worker" if terminal_worker_exit else "normal Python shutdown"}
    save(folder / "gpu_result.json", result)
    mod = tvm.ir.load_json((source / "model.relax.json").read_text(encoding="utf-8"))
    target = tvm.target.Target({"kind": "cuda", "arch": "sm_86", "max_threads_per_block": 1024,
                               "max_num_threads": 1024, "max_shared_memory_per_block": 49152,
                               "thread_warp_size": 32})
    result["target"] = str(target)
    progress(folder, "legalize_and_fuse")
    started = time.perf_counter()
    with target:
        lowered = tvm.transform.Sequential([
            relax.transform.DecomposeOpsForInference(), relax.transform.CanonicalizeBindings(),
            relax.get_pipeline("zero"),
        ])(mod)
    result["stages_s"]["legalize_and_fuse"] = time.perf_counter() - started
    (folder / "model.lowered.py").write_text(lowered.script(), encoding="utf-8")
    progress(folder, "extract_tuning_tasks")
    started = time.perf_counter()
    tasks = relax_integration.extract_tasks(lowered, target)
    task_rows = [{"name": task.task_name, "weight": int(task.weight),
                  "dispatched_variants": len(task.dispatched)} for task in tasks]
    save(folder / "tasks.json", task_rows)
    result["tuning_tasks"] = len(tasks)
    result["tir_functions"] = sum(isinstance(lowered[g], tirx.PrimFunc) for g in lowered.get_global_vars())
    result["stages_s"]["extract_tasks"] = time.perf_counter() - started
    save(folder / "gpu_result.json", result)
    progress(folder, "gpu_schedule_" + scheduler, tasks=len(tasks))
    started = time.perf_counter()
    with target:
        if scheduler == "basic":
            scheduled = tvm.s_tir.transform.DefaultGPUSchedule()(lowered)
        else:
            scheduled = lowered
            rules = ((dl.gpu.Fallback(),) if scheduler == "fallback" else
                     (dl.gpu.Matmul(), dl.gpu.GEMV(), dl.gpu.Reduction(),
                      dl.gpu.GeneralReduction(), dl.gpu.Fallback()))
            for gv, func in list(lowered.functions_items()):
                if isinstance(func, tirx.PrimFunc):
                    progress(folder, "schedule_function_" + gv.name_hint)
                    one = dl.ApplyDefaultSchedule(*rules)(tvm.IRModule({gv: func}))
                    scheduled[gv] = one[gv]
    result["stages_s"]["default_gpu_schedule"] = time.perf_counter() - started
    (folder / "model.scheduled.json").write_text(tvm.ir.save_json(scheduled), encoding="utf-8")
    progress(folder, "compile_complete_model")
    started = time.perf_counter()
    executable = tvm.compile(scheduled, target=target, relax_pipeline="default_build", tir_pipeline="s_tir")
    result["stages_s"]["compile"] = time.perf_counter() - started
    progress(folder, "load_cuda_vm")
    device = tvm.cuda(0)
    vm = relax.VirtualMachine(executable, device)
    parameters = np.load(source / "params.npz")
    params = [tvm.runtime.tensor(parameters[f"p{i}"], device)
              for i in range(len(parameters.files))]
    inputs, expected = np.load(source / "inputs.npz"), np.load(source / "reference.npz")
    timings = {}
    for case in inputs.files:
        progress(folder, "full_model_cuda_forward_" + case)
        x = tvm.runtime.tensor(inputs[case], device)
        started = time.perf_counter()
        actual = vm["main"](x, *params).numpy()
        device.sync()
        timings[case] = time.perf_counter() - started
        difference = np.abs(actual - expected[case])
        passed = bool(np.allclose(actual, expected[case], rtol=2e-4, atol=2e-4))
        result["checks"].append({"case": case, "passed": passed, "output_shape": list(actual.shape),
                                  "max_abs_error": float(difference.max()),
                                  "rtol": 2e-4, "atol": 2e-4})
        save(folder / "gpu_result.json", result)
        if not passed:
            np.save(folder / ("actual_" + case + ".npy"), actual)
            raise ValueError("Full model numerical check failed: " + case)
    result.update(status="completed", full_model_single_forward_wall_s=timings,
                  compiler_sha256=hashlib.sha256((ROOT / "build-prune/lib/tvm_compiler.dll").read_bytes()).hexdigest())
    save(folder / "gpu_result.json", result)
    progress(folder, "gpu_smoke_completed")
    if terminal_worker_exit:
        if os.name != "nt":
            raise ValueError("Terminal worker exit is restricted to Windows")
        device.sync()
        print("TERMINAL: verified CUDA results flushed; exit isolated worker before DLL detach", flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        import ctypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        kernel.TerminateProcess.argtypes = [ctypes.c_void_p, ctypes.c_uint]
        kernel.TerminateProcess.restype = ctypes.c_int
        if not kernel.TerminateProcess(kernel.GetCurrentProcess(), 0):
            raise ctypes.WinError(ctypes.get_last_error())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("resnet18", "mobilenet_v2"), required=True)
    parser.add_argument("--stage", choices=("export", "gpu"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input", type=Path, help="Reuse exported artifacts from an earlier stage")
    parser.add_argument("--scheduler", choices=("dlight", "basic", "fallback"), default="dlight")
    parser.add_argument("--terminal-worker-exit", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    stage_result = args.output / ("export_result.json" if args.stage == "export" else "gpu_result.json")
    if stage_result.exists():
        raise FileExistsError("Stage output already exists; preserve it and use a new attempt")
    shutil.copy2(__file__, args.output / "driver_snapshot.py")
    save(args.output / (args.stage + "_command.json"), {"argv": sys.argv,
         "utc": datetime.now(timezone.utc).isoformat(), "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    os.environ["TVM_LIBRARY_PATH"] = str(ROOT / "build-prune/lib")
    try:
        if args.stage == "export":
            export_model(args.output, args.model)
        else:
            compile_and_run(args.output, args.model, args.input, args.scheduler, args.terminal_worker_exit)
    except BaseException as error:
        save(args.output / (args.stage + "_error.json"), {"error": repr(error), "traceback": traceback.format_exc()})
        progress(args.output, "failed_" + args.stage, error=repr(error))
        raise


if __name__ == "__main__":
    main()
