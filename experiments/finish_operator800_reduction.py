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
"""After the Conv/MatMul queue, audit and remeasure the original Reduction winners."""

import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import complete_operator800_study as recovery
from operator800_cohort import accept_tuning, collect_cohort, write_report
from resume_operator800_20261024 import COHORT, FIGURES, BoundedController, make_command
from run_operator800_study import ROOT, verify_sources
from watch_operator800_cohort import ensure_running
from watch_operator800_study import read_json, write_json_atomic


def main():
    print("Waiting for the authorized Conv/MatMul controller to finish", flush=True)
    while True:
        manifest = read_json(COHORT / "cohort.json")
        owner = manifest.get("controller", {})
        if recovery.verified_process(owner.get("pid"), owner.get("created")) is None:
            break
        time.sleep(15)
    if manifest["status"] != "authorized_queue_completed_reduction_audit_pending":
        raise RuntimeError("The earlier queue needs attention; no extra GPU phases launched")
    protocol = read_json(Path(manifest["protocol_path"]))
    verify_sources(protocol)
    lock = recovery.acquire_lock(COHORT)
    ctl = None
    try:
        config = SimpleNamespace(input=Path(manifest["protocol_path"]).parent, output=COHORT,
                                 figure_output=FIGURES, interval=900, poll=5, max_attempts=1)
        ctl = BoundedController(config, manifest, protocol)
        manifest["controller"] = {**lock, "mode": "finish_reduction_evidence"}
        ctl.save(status="auditing_reduction", active_root=None, current="reduction4096")
        for arm, suffix in zip("BCD", ("011", "012", "013")):
            identifier = f"REDG1{arm}01"
            if identifier in manifest["accepted"]:
                continue
            spec = next(s for s in protocol["order"] if s["id"] == identifier)
            root = ROOT / ("experiments/results/o8_6ff1d8_" + suffix)
            reference = accept_tuning(root, spec, protocol)
            manifest["accepted"][identifier] = reference
            ctl.save()
            print(f"AUDITED {identifier}: {reference['submitted']}/800 upper limit", flush=True)
        publisher_path = COHORT / "live_status/watcher.json"
        if publisher_path.exists():
            publisher = read_json(publisher_path)
            process = recovery.verified_process(publisher.get("pid"), publisher.get("created"))
            if process is not None:
                if not any(Path(arg).name == "watch_operator800_cohort.py" for arg in process.cmdline()):
                    raise RuntimeError("Status publisher identity changed")
                process.terminate()
                process.wait(timeout=15)
        ensure_running(COHORT)
        for phase in ("retiming", "pytorch"):
            if "reduction4096" in manifest[phase]:
                continue
            root = ctl.new_root("reduction4096", phase)
            cmd, env = make_command(ctl, root, phase, "reduction4096")
            if not ctl.execute(root, cmd, phase, "reduction4096", env) or "reduction4096" not in manifest[phase]:
                raise RuntimeError("Reduction measurement phase failed; evidence retained")
        ctl.save(status="reporting", active_root=None)
        report_input = COHORT / ("report_input_final_" + str(time.time_ns()) + ".json")
        write_json_atomic(report_input, manifest)
        report = write_report(report_input, FIGURES)
        if not report["completion_ready"]:
            raise RuntimeError("Final operator evidence set remains incomplete")
        write_json_atomic(COHORT / "COMPLETION.json", {
            "status": "completed", "utc": recovery.utc_now(), "formal_groups": len(report["runs"]),
            "submitted": sum(r["budget"]["actual_submitted"] for r in report["runs"]),
            "early_terminated_groups": [key for key, ref in manifest["accepted"].items()
                                        if ref.get("completion_basis") == "verified_early_termination"],
            "resumed_groups": [key for key, ref in manifest["accepted"].items()
                               if ref.get("completion_basis") == "verified_history_replay_continuation"],
            "report": str(FIGURES / "RESULTS.md")})
        ctl.save(status="completed", active_root=None, pending=[], active_guard_pid=None,
                 active_guard_created=None, closeout_error=None)
        ctl.monitor(force=True)
        print("DONE: all operator groups, fixed-winner measurements and Torch references audited", flush=True)
    except BaseException as error:
        if ctl is not None:
            ctl.save(status="needs_attention", closeout_error=repr(error))
        raise
    finally:
        path = COHORT / "controller.lock.json"
        if path.exists() and read_json(path) == lock:
            path.unlink()


if __name__ == "__main__":
    main()
