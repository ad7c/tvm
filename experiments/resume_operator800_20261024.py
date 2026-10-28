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
"""Run the authorized remaining Conv/MatMul arms with the unchanged frozen driver."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import psutil

import complete_operator800_study as recovery
from operator800_cohort import collect_cohort, write_report
from report_single_seed_study import digest
from run_operator800_study import ROOT, environment, verify_sources
from watch_operator800_cohort import ensure_running
from watch_operator800_study import read_json, write_json_atomic


COHORT = ROOT / "experiments/results/paper_v1/operator800_20261011_cohort"
FIGURES = ROOT / "papers/figures/operator800_20261011_cohort"
STEPS = (
    ("tuning", "CONVG1C01"),
    ("retiming", "conv2d_nchw3x3"),
    ("pytorch", "conv2d_nchw3x3"),
    ("tuning", "M256G1D01"),
    ("tuning", "M256G1B01"),
    ("retiming", "matmul256"),
    ("pytorch", "matmul256"),
)


def completed(manifest, phase, identifier):
    return identifier in manifest["accepted" if phase == "tuning" else phase]


def require_prerequisites(manifest, protocol, phase, identifier):
    if phase == "tuning":
        spec = next(s for s in protocol["order"] if s["id"] == identifier)
        if spec["smoke"] or spec["trials"] != 800:
            raise ValueError("The authorized task is not a frozen 800-submission arm")
        required = [s["id"] for s in protocol["order"]
                    if s["smoke"] and s["workload"] == spec["workload"]]
    else:
        required = [s["id"] for s in protocol["order"]
                    if not s["smoke"] and s["workload"] == identifier]
        if len(required) != 3:
            raise ValueError("Expected the frozen B/C/D workload group")
    if not required or any(task not in manifest["accepted"] for task in required):
        raise RuntimeError(f"Required accepted results missing for {phase} {identifier}: {required}")


def make_command(ctl, root, phase, identifier):
    env = environment()
    if phase == "tuning" and identifier == "CONVG1C01":
        source = next((ROOT / "experiments/results/o8_6ff1d8_008").glob("CONVG1C01_*/"))
        cmd = [sys.executable, "-u", "-X", "utf8", str(ROOT / "experiments/operator800_history_resume.py"),
               "--source", str(source.resolve()), "--output", str(root.resolve())]
    elif phase == "tuning":
        cmd = [sys.executable, "-u", "-X", "utf8", str(ROOT / "experiments/run_operator800_study.py"),
               "--output", str(root), "--seed", str(ctl.protocol["seed"]), "--only", identifier]
    elif phase == "retiming":
        rows = [row for row in collect_cohort(ctl.manifest_path)["runs"]
                if row["workload"] == identifier]
        cmd = [sys.executable, "-u", "-X", "utf8", str(ROOT / "experiments/remeasure_top1.py"),
               "--output", str(root / "retiming" / f"{identifier}_ga1_R01"),
               "--library-dir", str(ROOT / "build-prune/lib"), "--terminal-worker-exit"]
        for arm in "BCD":
            row = next(row for row in rows if row["arm"] == arm)
            cmd += ["--result", f"{arm}={row['result_path']}"]
    else:
        cmd = [sys.executable, "-u", "-X", "utf8", str(ROOT / "experiments/benchmark_torch_operators.py"),
               "--workload", identifier, "--output", str(root / "pytorch" / f"{identifier}.json")]
        env["TVM_EXPERIMENT_NO_TORCH"] = "0"
    return cmd, env


def run_sequence(ctl):
    """Each missing phase gets one fresh attempt; an unaccepted result stops the queue."""
    for phase, identifier in STEPS:
        if completed(ctl.manifest, phase, identifier):
            print(f"SKIP ACCEPTED {phase} {identifier}", flush=True)
            continue
        require_prerequisites(ctl.manifest, ctl.protocol, phase, identifier)
        root = ctl.new_root(identifier, phase)
        cmd, env = make_command(ctl, root, phase, identifier)
        print(f"START AUTHORIZED {phase} {identifier}: {root}", flush=True)
        execution_phase = "history_resume" if phase == "tuning" and identifier == "CONVG1C01" else phase
        ok = ctl.execute(root, cmd, execution_phase, identifier, env)
        reference = ctl.manifest["accepted" if phase == "tuning" else phase].get(identifier)
        if not ok or not reference or Path(reference["root"]) != root:
            raise RuntimeError(f"Stopped after {phase} {identifier}; attempt retained at {root}")


class BoundedController(recovery.Controller):
    """Keep report failures separate from successful tuning acceptance."""

    def _plot(self, method):
        try:
            method()
        except Exception as error:
            print(f"REPORT NEEDS ATTENTION: {error!r}", flush=True)
            self.manifest.setdefault("report_errors", []).append(
                {"utc": recovery.utc_now(), "error": repr(error)})
            self.save()

    def plot_completed_arms(self):
        self._plot(super().plot_completed_arms)

    def plot_workload_comparisons(self):
        self._plot(super().plot_workload_comparisons)


def preflight():
    manifest = read_json(COHORT / "cohort.json")
    protocol = read_json(Path(manifest["protocol_path"]))
    if digest(Path(manifest["protocol_path"])) != manifest["protocol_sha256"]:
        raise ValueError("Frozen protocol hash changed")
    verify_sources(protocol)
    report = collect_cohort(COHORT / "cohort.json")
    active = [str(root) for root in recovery.task_roots(manifest) if recovery.guard_is_alive(root)]
    scripts = {Path(__file__).name, "complete_operator800_study.py", "finish_conv_d_only.py",
               "restart_operator800_conv_c_only.py", "guard_experiment_resources.py",
               "run_operator800_study.py", "e1_thread_filter.py", "remeasure_top1.py",
               "benchmark_torch_operators.py"}
    scripts.add("operator800_history_resume.py")
    for process in psutil.process_iter(["pid", "cmdline"]):
        if process.pid != os.getpid() and any(Path(arg).name in scripts
                for arg in (process.info["cmdline"] or [])):
            active.append(str(process.pid))
    if active:
        raise RuntimeError(f"Another experiment/controller is active: {active}")
    first = next(((phase, task) for phase, task in STEPS
                  if not completed(manifest, phase, task)), None)
    if first:
        require_prerequisites(manifest, protocol, *first)
    return manifest, protocol, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Read-only source and queue check")
    args = parser.parse_args()
    manifest, protocol, report = preflight()
    queue = [{"phase": phase, "id": task} for phase, task in STEPS
             if not completed(manifest, phase, task)]
    if args.check:
        print(json.dumps({"sources_verified": True, "retained_formal_groups": len(report["runs"]),
                          "queue": queue, "writes": False, "launches": False}, ensure_ascii=False))
        return
    lock = recovery.acquire_lock(COHORT)
    ctl = None
    try:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        control = COHORT / "resumes" / ("remaining_operators_" + stamp)
        control.mkdir(parents=True, exist_ok=False)
        shutil.copy2(COHORT / "cohort.json", control / "cohort.before.json")
        request = {"utc": recovery.utc_now(), "queue": queue,
                   "instruction": "Keep the earlier experiment environment unchanged and finish these operators.",
                   "protocol_sha256": manifest["protocol_sha256"], "controller": lock,
                   "wrapper_sha256": digest(Path(__file__)),
                   "attempt_policy": "Conv C restores the verified 298-measurement history; unstarted MatMul arms start empty",
                   "automatic_retries": 0, "reduction": "existing early termination evidence requires audit",
                   "paper_sha256": {str(p): digest(p) for p in
                       (ROOT / "papers/main.tex", ROOT / "papers/sections/single_seed_experiments.tex")}}
        write_json_atomic(control / "request.json", request)
        config = SimpleNamespace(input=Path(manifest["protocol_path"]).parent, output=COHORT,
                                 figure_output=FIGURES, interval=900, poll=5, max_attempts=1)
        ctl = BoundedController(config, manifest, protocol)
        manifest["controller"] = {**lock, "mode": "authorized_remaining_operators",
                                  "source_sha256": request["wrapper_sha256"], "queue": queue}
        ctl.save(status="resuming_authorized_operators", active_root=None,
                 requires_explicit_continue=False, resume_request=str(control / "request.json"),
                 stop_after="authorized_queue", closeout_error=None, paused_task=None,
                 pending=[s["id"] for s in protocol["order"]
                          if not s["smoke"] and s["id"] not in manifest["accepted"]])
        ensure_running(COHORT)
        run_sequence(ctl)
        ctl.save(status="reporting", active_root=None)
        report_input = control / "report_input.json"
        shutil.copy2(ctl.manifest_path, report_input)
        report = write_report(report_input, FIGURES)
        write_json_atomic(control / "DONE.json", {
            "utc": recovery.utc_now(), "completed_queue": queue, "pending": report["pending"],
            "paper_files_unchanged": {p: digest(Path(p)) == expected
                                      for p, expected in request["paper_sha256"].items()}})
        ctl.save(status="authorized_queue_completed_reduction_audit_pending", active_root=None,
                 pending=report["pending"], active_guard_pid=None, active_guard_created=None)
        ctl.monitor(force=True)
        print("DONE: Conv/MatMul tuning and retiming complete; Reduction audit still pending", flush=True)
    except BaseException as error:
        if ctl is not None:
            # Preserve a live guard identity if an observer failed while its child was running.
            live = ctl.active_root is not None and recovery.guard_is_alive(ctl.active_root)
            ctl.save(status="needs_attention", closeout_error=repr(error),
                     active_root=str(ctl.active_root) if live else None)
            write_json_atomic(control / "ERROR.json", {"utc": recovery.utc_now(), "error": repr(error)})
        raise
    finally:
        lock_path = COHORT / "controller.lock.json"
        if lock_path.exists() and read_json(lock_path) == lock:
            lock_path.unlink()


if __name__ == "__main__":
    main()
