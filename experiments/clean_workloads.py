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
"""Explicit new workloads and independent NumPy references; old workloads unchanged."""

SPECS = {
    "conv2d_nchw3x3": {"op": "conv2d", "layout": "NCHW/OIHW", "dtype": "float32",
                       "inputs": [[1, 16, 32, 32], [32, 16, 3, 3]], "output": [1, 32, 32, 32],
                       "stride": [1, 1], "padding": [1, 1], "dilation": [1, 1], "groups": 1},
    "reduction4096": {"op": "sum", "dtype": "float32", "inputs": [[128, 4096]],
                      "output": [128], "axis": 1},
}


def make_workload(name):
    import tvm
    from tvm.script import tirx as T

    if name == "conv2d_nchw3x3":
        @tvm.script.ir_module
        class Conv3x3:
            @T.prim_func(s_tir=True)
            def main(x: T.handle, w: T.handle, y: T.handle) -> None:
                T.func_attr({"global_symbol": "main"})
                X = T.match_buffer(x, (1, 16, 32, 32), "float32")
                W = T.match_buffer(w, (32, 16, 3, 3), "float32")
                Y = T.match_buffer(y, (1, 32, 32, 32), "float32")
                P = T.alloc_buffer((1, 16, 34, 34), "float32")
                for n, c, h, k in T.grid(1, 16, 34, 34):
                    with T.sblock("pad"):
                        vn, vc, vh, vk = T.axis.remap("SSSS", [n, c, h, k])
                        P[vn, vc, vh, vk] = T.if_then_else(
                            vh >= 1 and vh < 33 and vk >= 1 and vk < 33,
                            X[vn, vc, vh - 1, vk - 1], T.float32(0))
                for n, oc, oh, ow, ic, kh, kw in T.grid(1, 32, 32, 32, 16, 3, 3):
                    with T.sblock("conv"):
                        vn, voc, voh, vow, vic, vkh, vkw = T.axis.remap(
                            "SSSSRRR", [n, oc, oh, ow, ic, kh, kw])
                        with T.init():
                            Y[vn, voc, voh, vow] = T.float32(0)
                        Y[vn, voc, voh, vow] = Y[vn, voc, voh, vow] + P[
                            vn, vic, voh + vkh, vow + vkw] * W[voc, vic, vkh, vkw]
        return Conv3x3
    if name == "reduction4096":
        @tvm.script.ir_module
        class Reduction4096:
            @T.prim_func(s_tir=True)
            def main(a: T.handle, b: T.handle) -> None:
                T.func_attr({"global_symbol": "main"})
                A = T.match_buffer(a, (128, 4096), "float32")
                B = T.match_buffer(b, (128,), "float32")
                for i, j in T.grid(128, 4096):
                    with T.sblock("r"):
                        vi, vj = T.axis.remap("SR", [i, j])
                        with T.init():
                            B[vi] = T.float32(0)
                        B[vi] = B[vi] + A[vi, vj]
        return Reduction4096
    raise ValueError(f"Unknown clean workload: {name}")


def reference(name, arrays):
    import numpy as np

    if name == "reduction4096":
        return arrays[0].astype("float64").sum(axis=1)
    if name == "conv2d_nchw3x3":
        image, weight = [value.astype("float64") for value in arrays]
        padded = np.pad(image, ((0, 0), (0, 0), (1, 1), (1, 1)))
        patches = np.lib.stride_tricks.sliding_window_view(padded, (3, 3), axis=(2, 3))
        return np.einsum("ncyxij,ocij->noyx", patches, weight, optimize=True)
    raise ValueError(name)


def numerical_cases(name):
    import numpy as np

    spec = SPECS[name]
    rng = np.random.default_rng(20261010)
    for label in ("normal", "uniform", "ones", "zero"):
        arrays = []
        for shape in spec["inputs"]:
            if label == "normal":
                value = rng.normal(size=shape)
            elif label == "uniform":
                value = rng.uniform(-1, 1, size=shape)
            else:
                value = np.ones(shape) if label == "ones" else np.zeros(shape)
            arrays.append(value.astype("float32"))
        yield label, arrays, reference(name, arrays)


def checked_clean_evaluator(rt_mod, device, config, repeated_args, *, audit_path, candidate_id, workload):
    """Check independent buffers before delegating unchanged measurement inputs."""
    import time
    import traceback
    import numpy as np
    import tvm
    from tvm.s_tir.meta_schedule.runner.local_runner import default_run_evaluator
    from p3_observability import append_json

    start = time.perf_counter()
    append_json(audit_path, {"event": "correctness_begin", "candidate_id": candidate_id})
    row = {"event": "correctness_result", "candidate_id": candidate_id, "workload": workload,
           "passed": False, "cases": [], "rtol": 2e-4, "atol": 2e-4, "seed": 20261010}
    try:
        shapes = [list(value.shape) for value in repeated_args[0]]
        if shapes != [*SPECS[workload]["inputs"], SPECS[workload]["output"]]:
            raise ValueError(f"Unexpected buffers for {workload}: {shapes}")
        for label, arrays, expected in numerical_cases(workload):
            buffers = [tvm.runtime.tensor(value, device) for value in arrays]
            output = tvm.runtime.tensor(np.full(expected.shape, np.nan, dtype="float32"), device)
            rt_mod[rt_mod.entry_name](*buffers, output)
            device.sync()
            actual = output.numpy()
            error = np.abs(actual.astype("float64") - expected)
            tolerance = row["atol"] + row["rtol"] * np.abs(expected)
            passed = bool(np.isfinite(actual).all() and (error <= tolerance).all())
            row["cases"].append({"name": label, "passed": passed, "max_abs_error": float(error.max()),
                                 "max_tolerance_ratio": float((error / tolerance).max())})
            if not passed:
                raise AssertionError(f"Numerical correctness failed for {workload}/{label}")
        row["passed"] = True
    except Exception:
        row["exception"] = traceback.format_exc()
        raise
    finally:
        row["correctness_wall_s"] = time.perf_counter() - start
        append_json(audit_path, row)
    values = default_run_evaluator(rt_mod, device, config, repeated_args)
    append_json(audit_path, {"event": "evaluator_complete", "candidate_id": candidate_id,
                             "costs_s": [float(value) for value in values]})
    return values
