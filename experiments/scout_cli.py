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
"""Relocatable Scout entrypoint. Only the explicit run action launches tuning."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "experiments/fixtures/resnet18_fp32_sm86"


def check_assets():
    manifest = json.loads((ASSETS / "manifest.json").read_text(encoding="utf-8"))
    total = 0
    for name, expected in manifest["files"].items():
        path = ASSETS / name
        if path.parent != ASSETS or not path.is_file():
            raise ValueError(f"Missing or invalid fixture: {name}")
        data = path.read_bytes()
        if len(data) != expected["bytes"] or hashlib.sha256(data).hexdigest() != expected["sha256"]:
            raise ValueError(f"Fixture checksum mismatch: {name}")
        total += len(data)
    return {"assets":"verified", "files":len(manifest["files"]),"bytes":total}


def doctor(require_memory=False):
    import tvm
    from tvm import base as tvm_base
    from guard_experiment_resources import system_memory

    result = check_assets()
    if not Path(tvm.__file__).resolve().is_relative_to(ROOT / "python"):
        raise RuntimeError(f"TVM imported from another checkout: {tvm.__file__}")
    library = tvm_base._LOADED_LIBS.get("tvm_compiler")
    expected = (ROOT / "build-prune/lib/tvm_compiler.dll").resolve()
    if library is None or Path(library._name).resolve() != expected:
        raise RuntimeError("TVM compiler must load from this checkout's build-prune/lib")
    for name in ("ScoutReplay","ScoutMutate","ScoutDraftFeatures"):
        tvm.get_global_func("s_tir.meta_schedule.testing." + name)
    if shutil.which("nvcc") is None:
        raise RuntimeError("nvcc is missing from PATH; install CUDA Toolkit and open the build shell")
    device = tvm.cuda(0)
    if not device.exist:
        raise RuntimeError("No visible CUDA GPU")
    if device.compute_version != "8.6" or device.multi_processor_count != 20:
        raise RuntimeError("Frozen Scout profile requires sm_86 / 20 SMs (RTX 3050 Ti). Adapt the profile before other GPUs.")
    memory = system_memory()
    result.update(python=sys.version.split()[0], tvm_source=str(tvm.__file__),
                  compiler=str(expected), gpu=device.device_name,
                  compute_capability=device.compute_version, sms=device.multi_processor_count,
                  free_physical_gib=memory["available_physical"]/2**30,
                  free_commit_gib=memory["available_commit"]/2**30,
                  dependencies={n:importlib.metadata.version(n) for n in
                      ("numpy","xgboost","psutil","apache-tvm-ffi")},
                  scope="Configuration/device query only; no model execution or tuning")
    print(json.dumps(result,indent=2),flush=True)
    if require_memory and (memory["available_physical"] < 4*2**30 or memory["available_commit"] < 6*2**30):
        raise RuntimeError("Need >=4 GiB free physical and >=6 GiB free commit memory; no automatic wait/start")


def worker(action):
    code = 0
    try:
        if action in ("doctor","ready"):
            doctor(require_memory=action == "ready")
        else:
            import pytest
            code = pytest.main(["experiments/test_scout_feedback.py",
                "experiments/test_model_scout_pool_study.py",
                "tests/python/s_tir/meta_schedule/test_meta_schedule_pruner_draft_model.py",
                "-q","-p","no:cacheprovider"])
    except BaseException:
        traceback.print_exc()
        code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    # Isolated Windows worker: avoid TVM DLL teardown after reporting its result.
    os._exit(int(code))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("assets","doctor","check","run"))
    parser.add_argument("--study",type=Path,help="New result directory; required for run")
    parser.add_argument("--trials",type=int,default=400)
    parser.add_argument("--arm-seconds",type=int,default=5400)
    parser.add_argument("--total-seconds",type=int,default=18000)
    args = parser.parse_args()
    if args.action == "assets":
        print(json.dumps(check_assets(),indent=2))
        return 0
    if os.name != "nt":
        parser.error("This experiment harness is currently Windows-only; see README_SCOUT_1029.md")
    if args.action == "run" and args.study is None:
        parser.error("run requires --study with a fresh directory")
    if not (ROOT / "build-prune/lib/tvm_compiler.dll").is_file():
        parser.error("Build this checkout first; build-prune/lib/tvm_compiler.dll is missing")
    from model_scout_feedback_study import environment
    env = environment()
    action = "ready" if args.action == "run" else args.action
    code = subprocess.call([sys.executable,"-u","-X","utf8",str(Path(__file__).resolve()),
                            "_worker",action],cwd=ROOT,env=env)
    if code or args.action != "run":
        return code
    return subprocess.call([sys.executable,"-u","-X","utf8",
        str(ROOT / "experiments/model_scout_feedback_study.py"),"run",
        "--study",str(args.study.resolve()),"--trials",str(args.trials),
        "--arm-seconds",str(args.arm_seconds),"--total-seconds",str(args.total_seconds),
        "--memory-wait-seconds","0"],cwd=ROOT,env=env)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "_worker":
        worker(sys.argv[2])
    else:
        raise SystemExit(main())
