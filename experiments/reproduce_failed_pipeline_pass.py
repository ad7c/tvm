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
"""Isolate one failing pass from exact, validated pass-input snapshots.

This is exception diagnosis, not a replacement for full-pipeline verification,
not an additional invalid-candidate count, and not a speedup measurement.
No trace generation, TuneContext, Builder, Runner, or GPU execution is used.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import time
import traceback

from fixed_trace_pool import ROOT, configure_environment, sha256, write_json


EXPECTED_LIBRARY = "c2853006ea7217da30f5029c4038c20118dfb5d1a4864b99de066178602ef3c3"
PASS_NAME = "s_tir.InjectSoftwarePipeline"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridge", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--library-dir", type=Path, default=ROOT / "build-prune/lib")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "command.json", {
        "argv": sys.argv, "utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.executable,
    })
    diagnosis_path = args.bridge / "diagnosis.json"
    diagnosis = json.loads(diagnosis_path.read_text(encoding="utf-8"))
    profiles_path = args.bridge / "pass_profiles.jsonl"
    if not diagnosis["gate_passed"] or sha256(profiles_path) != diagnosis["profiles_sha256"]:
        raise ValueError("Input profiles are not from the unchanged validated bridge")
    if diagnosis["identity"]["loaded_library_sha256"] != EXPECTED_LIBRARY:
        raise ValueError("Bridge used a different compiler DLL")
    profiles = [json.loads(line) for line in profiles_path.read_text(encoding="utf-8").splitlines()]
    selected = [row for row in profiles if row["verify_reason"] == "lowering_exception"]
    if len(selected) != 8 or any(row["deepest_unfinished_pass"] != PASS_NAME for row in selected):
        raise ValueError("This diagnostic expects exactly the eight archived pipeline failures")
    configure_environment(args.library_dir.resolve())
    print("Loading exact diagnostic DLL; no TuneContext or GPU work", flush=True)
    setup_started = time.perf_counter()
    import tvm  # pylint: disable=import-outside-toplevel

    library = Path(tvm.base._LOADED_LIBS["tvm_compiler"]._name).resolve()
    if sha256(library) != EXPECTED_LIBRARY:
        raise ValueError("Loaded DLL differs from the validated bridge")
    current = tvm.transform.PassContext.current()
    if current.opt_level != 2 or current.config or current.required_pass or current.disabled_pass:
        raise ValueError("Unexpected process-default PassContext")
    identity = {
        "schema_version": 1, "diagnosis_sha256": sha256(diagnosis_path),
        "profiles_sha256": sha256(profiles_path), "source_bridge": str(args.bridge.resolve()),
        "loaded_library": str(library), "loaded_library_sha256": sha256(library),
        "driver_sha256": sha256(__file__),
        "configure_environment_source_sha256": sha256(Path(__file__).with_name("fixed_trace_pool.py")),
        "pass_source_sha256": sha256(ROOT / "src/s_tir/transform/inject_software_pipeline.cc"),
        "pass_context": {"opt_level": 2, "config": {}, "required_pass": [], "disabled_pass": []},
        "setup_s": time.perf_counter() - setup_started,
        "scope": "Single original pass on exact archived input; diagnosis only, no new candidates",
    }
    write_json(args.output / "identity.json", identity)
    shutil.copy2(__file__, args.output / "driver_snapshot.py")
    rows = []
    with (args.output / "results.jsonl").open("x", encoding="utf-8") as stream:
        for profile in selected:
            snapshot = profile["ir_snapshots"]["deepest_unfinished_input"]
            snapshot_path = args.bridge / snapshot["path"]
            if sha256(snapshot_path) != snapshot["sha256"]:
                raise ValueError(f"Pass input snapshot changed: {snapshot_path}")
            mod = tvm.ir.load_json(snapshot_path.read_text(encoding="utf-8"))
            key = profile["candidate_id"]
            ir_path = args.output / f"{key}.input.tir.txt"
            ir_path.write_text(mod.script(), encoding="utf-8")
            row = {
                "candidate_id": key, "stage": profile["stage"], "trace_sha256": profile["trace_sha256"],
                "input_path": str(snapshot_path.resolve()), "input_sha256": snapshot["sha256"],
                "readable_input_sha256": sha256(ir_path), "pass": PASS_NAME,
                "archived_verify_reason": profile["verify_reason"],
                "targets": {str(var.name_hint): str(func.attrs["target"])
                            for var, func in mod.functions.items()},
            }
            started = time.perf_counter()
            try:
                with tvm.transform.PassContext(opt_level=2):
                    output = tvm.s_tir.transform.InjectSoftwarePipeline()(mod)
            except Exception as error:  # Intentionally capture exact isolated diagnostics, not legality.
                row.update(status="exception", exception_type=f"{type(error).__module__}.{type(error).__name__}",
                           exception_message=str(error), traceback=traceback.format_exc())
            else:
                output_path = args.output / f"{key}.unexpected_output.json"
                output_path.write_text(tvm.ir.save_json(output), encoding="utf-8")
                row.update(status="unexpected_pass_success", output_sha256=sha256(output_path))
            row["isolated_diagnostic_s"] = time.perf_counter() - started
            rows.append(row)
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            print(f"{key}: {row['status']} {row.get('exception_type', '')}", flush=True)
            if row["status"] == "exception":
                print(row["exception_message"], flush=True)
    write_json(args.output / "summary.json", {
        "schema_version": 1, "identity": identity, "candidate_count": len(rows),
        "status_counts": dict(Counter(row["status"] for row in rows)),
        "exception_type_counts": dict(Counter(row.get("exception_type", "none") for row in rows)),
        "results_sha256": sha256(args.output / "results.jsonl"),
        "all_eight_reproduced_an_exception": all(row["status"] == "exception" for row in rows),
        "new_candidates": 0, "new_gpu_measurements": 0,
        "limitations": ["An isolated exception does not independently establish a safe early-pruning rule.",
                        "Snapshot inputs have already traversed preceding lowering passes.",
                        "Diagnostic elapsed times include Python exception formatting; not performance data.",
                        "Preserve the original full-pipeline bridge oracle and timing."],
    })
    print("All archived failure inputs processed; see summary.json and full results.jsonl", flush=True)


if __name__ == "__main__":
    main()
