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
"""Isolate Windows commit growth during loading, launching and releasing CUDA modules."""

import argparse
import ctypes
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import shutil
import subprocess
import traceback

import psutil

from guard_experiment_resources import system_memory


def exit_code(pid):
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
    kernel.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = kernel.OpenProcess(0x1000, False, pid)
    if not handle:
        return {"open_error": ctypes.get_last_error()}
    try:
        code = ctypes.c_uint32()
        if kernel.GetExitCodeProcess(handle, ctypes.byref(code)):
            return {"exit_code": code.value}
        return {"query_error": ctypes.get_last_error()}
    finally:
        kernel.CloseHandle(handle)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, action="append", required=True)
    parser.add_argument("--library-dir", type=Path, default=Path("build-prune/lib"))
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, args.output / "snapshot.py")
    root = psutil.Process()

    def emit(event, **fields):
        row = {"utc": datetime.now(timezone.utc).isoformat(), "event": event, **fields}
        with (args.output / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, default=str) + "\n")
        print(json.dumps(row, default=str), flush=True)

    def sample(stage, **fields):
        processes = []
        for process in [root, *root.children(recursive=True)]:
            try:
                memory = process.memory_info()
                row = {"pid": process.pid, "parent": process.ppid(), "name": process.name(),
                       "threads": process.num_threads(), "private": memory.private,
                       "rss": memory.rss, "handles": process.num_handles(), **exit_code(process.pid)}
                try:
                    row["argv"] = process.cmdline()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    row["argv"] = None
                processes.append(row)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        emit(stage, system=system_memory(), processes=processes, **fields)

    original_popen = subprocess.Popen

    class ObservedPopen(original_popen):
        def __init__(self, *values, **kwargs):
            super().__init__(*values, **kwargs)
            emit("python_spawn", pid=self.pid, argv=values[0] if values else kwargs.get("args"),
                 stack=[f"{frame.filename}:{frame.lineno}:{frame.name}"
                        for frame in traceback.extract_stack(limit=10)[:-1]])

    subprocess.Popen = ObservedPopen
    sample("before_import")
    os.environ["TVM_LIBRARY_PATH"] = str(args.library_dir.resolve())
    import numpy as np
    import tvm

    sample("after_import", library=str(tvm.base._LOADED_LIBS["tvm_compiler"]._name),
           environment={key: os.environ.get(key) for key in
                        ("TVM_EXPERIMENT_NO_TORCH", "TVM_FFI_DISABLE_TORCH_C_DLPACK")})
    device = tvm.cuda()
    lhs = tvm.runtime.tensor(np.eye(128, dtype="float32"), device)
    rhs = tvm.runtime.tensor(np.ones((128, 128), dtype="float32"), device)
    out = tvm.runtime.tensor(np.zeros((128, 128), dtype="float32"), device)
    sample("after_buffers")
    for index, artifact in enumerate(args.artifact * args.repeat):
        directory = args.output / f"artifact_{index:02d}"
        directory.mkdir()
        copied = directory / "module.tar"
        shutil.copy2(artifact, copied)
        sample("before_load", index=index)
        module = tvm.runtime.load_module(str(copied.resolve()))
        sample("after_load", index=index)
        module[module.entry_name](lhs, rhs, out)
        device.sync()
        np.testing.assert_allclose(out.numpy(), np.ones((128, 128)), rtol=2e-4, atol=2e-4)
        sample("after_launch", index=index)
        del module
        gc.collect()
        device.sync()
        sample("after_release", index=index)
    emit("completed", loaded_modules=len(args.artifact) * args.repeat)


if __name__ == "__main__":
    main()
