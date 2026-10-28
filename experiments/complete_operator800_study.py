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
"""Adopt an operator study, retain complete arms and recover individual failed tasks."""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import psutil

from guard_experiment_resources import kill_launched_tree, system_memory
from e1_thread_filter import make_parser
from operator800_cohort import (accept_phase, accept_tuning, collect_cohort, load_tuning,
                               object_hash, utc_now, write_report)
from report_single_seed_study import digest
from run_operator800_study import ROOT, command, environment, verify_sources
from watch_operator800_study import (complete_lines, latest_memory, process_alive, read_json,
                                    snapshot, tail_text, write_json_atomic)
from watch_operator800_cohort import ensure_running


def verified_process(pid, created):
    if not pid or pid < 0 or created is None:
        return None
    try:
        process = psutil.Process(pid)
        if process.create_time() == created and process.num_threads() > 0:
            return process
    except psutil.NoSuchProcess:
        pass
    return None


def guard_wrapper_alive(root):
    path = root / "resource_guard/status.json"
    if not path.exists():
        return False
    state = read_json(path)
    if state.get("status") not in ("starting", "running"):
        return False
    if process_alive(state.get("guard_pid")):
        try:
            process = psutil.Process(state["guard_pid"])
            if state.get("child_created") and process.create_time() > state["child_created"]:
                return False  # A later process reused the terminated guard's PID.
            argv = process.cmdline()
            if any(Path(arg).name == "guard_experiment_resources.py" for arg in argv):
                index = argv.index("--output") + 1
                if (ROOT / argv[index]).resolve() == path.parent.resolve():
                    return True
        except (psutil.NoSuchProcess, ValueError):
            pass
    return False


def guard_is_alive(root):
    if guard_wrapper_alive(root):
        return True
    path = root / "resource_guard/status.json"
    if not path.exists():
        return False
    state = read_json(path)
    return verified_process(state.get("child_pid"), state.get("child_created")) is not None


def clean_stale_children(root):
    """Only clean identities recorded by the guard after its verified exit."""
    guard = read_json(root / "resource_guard/status.json")
    memory = latest_memory(root / "resource_guard/memory.jsonl") or {"processes": []}
    known = []
    for row in memory["processes"]:
        process = verified_process(row["pid"], row["created"])
        if process is not None:
            known.append(process)
    parent = verified_process(guard.get("child_pid", -1), guard.get("child_created"))
    if parent is None and not known:
        return
    # The original root object is optional once its children have become orphans.
    root_process = parent or known[0]
    cleanup = kill_launched_tree(root_process, known)
    write_json_atomic(root / ("recovery_cleanup_" + datetime.now().strftime("%H%M%S") + ".json"), cleanup)
    if cleanup["remaining_live"]:
        raise RuntimeError(f"Experiment descendants still alive: {cleanup['remaining_live']}")


def acquire_lock(output):
    path = output / "controller.lock.json"
    if path.exists():
        previous = read_json(path)
        if process_alive(previous.get("pid"), previous.get("created")):
            raise RuntimeError(f"Another recovery controller owns this cohort: {previous['pid']}")
        path.rename(output / ("stale_controller_lock_" + str(time.time_ns()) + ".json"))
    identity = {"pid": os.getpid(), "created": psutil.Process().create_time(), "utc": utc_now()}
    with path.open("x", encoding="utf-8") as stream:
        json.dump(identity, stream)
    return identity


def task_roots(manifest):
    return [Path(path) for path in manifest["attempt_roots"]]


def failed_attempts(manifest, spec):
    return [failure for failure in manifest["failures"]
            if failure["phase"] in ("smoke", "tuning") and failure["id"] == spec["id"]
            and not failure.get("resolved_by")]


def tuning_log_path(root, spec, seed):
    """Validate the existing driver's longest log path before starting any worker."""
    args = make_parser().parse_args(command(spec, root, seed)[5:])
    name = (f"{args.run_id}_{args.workload}_seed{args.seed}_{args.mode}"
            f"_sample-{args.constraint_sample_mode}_hpos-{args.hard_filter_position}"
            f"_soft-{args.soft_mode}_draft-{args.draft_mode}"
            f"_dga-{args.draft_ga_mode}_ga-{args.ga_t2_mode}"
            f"_g3-{args.draft_ga_g3_mode}_g4-{args.draft_ga_anneal_mode}")
    return root / name / "logs/tvm.s_tir.meta_schedule.logging.task_scheduler.log"


def validate_tuning_paths(root, protocol):
    paths = [tuning_log_path(root, spec, protocol["seed"]) for spec in protocol["order"]]
    longest = max(paths, key=lambda p: len(str(p)))
    if os.name == "nt" and len(str(longest)) > 240:
        raise ValueError(f"Windows experiment path too long ({len(str(longest))}): {longest}")
    return longest


def tuning_order(manifest, protocol):
    """Change execution priority separately from the immutable tuning protocol."""
    registered = list(dict.fromkeys(spec["workload"] for spec in protocol["order"]))
    priority = manifest.get("scheduling", {}).get("workload_priority", registered)
    if len(priority) != len(registered) or set(priority) != set(registered):
        raise ValueError("Scheduling priority must include each registered workload exactly once")
    positions = {workload: index for index, workload in enumerate(priority)}
    return sorted(protocol["order"], key=lambda spec: positions[spec["workload"]])


def choose_tuning(manifest, protocol, max_attempts):
    for spec in tuning_order(manifest, protocol):
        if spec["id"] in manifest["accepted"]:
            continue
        smoke = next(s for s in protocol["order"] if s["smoke"] and s["workload"] == spec["workload"])
        if not spec["smoke"] and smoke["id"] not in manifest["accepted"]:
            continue
        failures = failed_attempts(manifest, spec)
        if len(failures) >= max_attempts or any(f.get("short_budget") for f in failures):
            continue
        return spec
    return None


class Controller:
    def __init__(self, args, manifest, protocol):
        self.args, self.manifest, self.protocol = args, manifest, protocol
        self.output = args.output.resolve()
        self.manifest_path = self.output / "cohort.json"
        self.next_check = 0
        self.active_root = None

    def save(self, **state):
        self.manifest.update(state, updated_utc=utc_now())
        write_json_atomic(self.manifest_path, self.manifest)
        current = Path(self.manifest.get("active_root") or self.args.input)
        write_json_atomic(ROOT / "experiments/OPERATOR800_CURRENT.json", {
            "directory": str(current), "source_directory": str(self.args.input),
            "cohort_directory": str(self.output), "cohort": str(self.manifest_path),
            "progress": str(current / "progress.json"),
            "monitor": str(self.output / "monitor15min/AUDIT_STATUS.md"),
            "interval_s": self.args.interval, "updated_utc": utc_now()})

    def failure(self, root, phase, identifier, error):
        key = (str(root.resolve()), phase, identifier)
        if any((f["root"], f["phase"], f["id"]) == key for f in self.manifest["failures"]):
            return
        guard_path = root / "resource_guard/status.json"
        guard = read_json(guard_path) if guard_path.exists() else {}
        failure = {"root": key[0], "phase": phase, "id": identifier, "utc": utc_now(),
                   "reason": str(error), "guard_status": guard.get("status"),
                   "short_budget": "Short budget:" in str(error)}
        log = root / (identifier + ".console.log")
        failure["console_tail"] = tail_text(log if log.exists() else root / "resource_guard/child.console.log", 6000)
        self.manifest["failures"].append(failure)
        print(f"PRESERVE FAILED {identifier} {phase}: {error}", flush=True)
        self.save()
        if ("FileNotFoundError" in failure["console_tail"]
                and "meta_schedule.logging.task_scheduler.log" in failure["console_tail"]):
            raise RuntimeError("Diagnose shared Windows path failure before launching another task")

    def refresh(self, terminal=False):
        changed = False
        for root in task_roots(self.manifest):
            for spec in self.protocol["order"]:
                if spec["id"] in self.manifest["accepted"]:
                    continue
                present = list(root.glob(spec["id"] + "_*/")) or (root / (spec["id"] + ".console.log")).exists()
                if not present:
                    continue
                phase = "smoke" if spec["smoke"] else "tuning"
                receipts = [e for e in complete_lines(root / "execution.jsonl")
                            if e.get("phase") == phase and e.get("id") == spec["id"]]
                if not receipts and not terminal:
                    continue
                try:
                    reference = accept_tuning(root, spec, self.protocol)
                except (OSError, ValueError, KeyError) as error:
                    if terminal or receipts:
                        self.failure(root, phase, spec["id"], error)
                    continue
                self.manifest["accepted"][spec["id"]] = reference
                if reference.get("postcheck_audit_path"):
                    for failure in self.manifest["failures"]:
                        if (failure["root"] == str(root.resolve()) and failure["id"] == spec["id"]
                                and failure["phase"] == phase):
                            failure.update(resolved_by="independent-variable-batch-postcheck-v2",
                                           resolution_evidence=reference["postcheck_audit_path"])
                changed = True
                print(f"ACCEPT {spec['id']} {reference['submitted']}/{spec['trials']}", flush=True)
        if changed:
            self.save()
        self.plot_completed_arms()
        self.plot_workload_comparisons()
        # Retain independently measured phases of the adopted original driver.
        report = collect_cohort(self.manifest_path)
        for workload in dict.fromkeys(s["workload"] for s in self.protocol["order"]):
            rows = [row for row in report["runs"] if row["workload"] == workload]
            if len(rows) != 3:
                continue
            for phase in ("retiming", "pytorch"):
                if workload in self.manifest[phase]:
                    continue
                for root in task_roots(self.manifest):
                    path = (root / "retiming" / f"{workload}_ga1_R01/result.json" if phase == "retiming"
                            else root / "pytorch" / f"{workload}.json")
                    receipts = [e for e in complete_lines(root / "execution.jsonl")
                                if e.get("phase") == phase and e.get("id") == workload]
                    if not path.exists() and not receipts:
                        continue
                    if not receipts and not terminal:
                        continue
                    try:
                        reference = accept_phase(root, phase, workload, path, rows, self.protocol)
                    except (OSError, ValueError, KeyError) as error:
                        if terminal or receipts:
                            self.failure(root, phase, workload, error)
                        continue
                    self.manifest[phase][workload] = reference
                    self.save()
                    print(f"ACCEPT {phase} {workload}", flush=True)
                    break

    def plot_completed_arms(self):
        """Publish each completed arm before starting another tuning process."""
        script = ROOT / "experiments/plot_operator800_arm.py"
        for spec in self.protocol["order"]:
            reference = self.manifest["accepted"].get(spec["id"])
            if spec["smoke"] or reference is None:
                continue
            output = self.args.figure_output.resolve() / "arms" / spec["id"]
            manifest_path = output / "manifest.json"
            if manifest_path.exists():
                saved = read_json(manifest_path)
                if (saved.get("accepted_tuning") == reference
                        and saved.get("plot_script_sha256") == digest(script)
                        and all((output / name).is_file() and digest(output / name) == expected
                                for name, expected in saved.get("output_sha256", {}).items())
                        and (output / "convergence.png").is_file()
                        and (output / "model_training.png").is_file()):
                    continue
            failures = [failure for failure in self.manifest["failures"]
                        if failure["phase"] == "plotting" and failure["id"] == spec["id"]
                        and not failure.get("resolved_by")]
            if len(failures) >= self.args.max_attempts:
                continue
            logs = self.output / "plot_logs"
            logs.mkdir(exist_ok=True)
            log = logs / (spec["id"] + "_" + str(time.time_ns()) + ".log")
            cmd = [sys.executable, "-u", "-X", "utf8", str(script), "--root", reference["root"],
                   "--id", spec["id"], "--output", str(output)]
            with log.open("w", encoding="utf-8") as stream:
                code = subprocess.run(cmd, cwd=ROOT, env=environment(), stdout=stream,
                                      stderr=subprocess.STDOUT).returncode
            if code:
                # Plotting never discards successful tuning or prevents remaining GPU tasks.
                self.manifest["failures"].append({"root": reference["root"], "phase": "plotting",
                    "id": spec["id"], "utc": utc_now(), "reason": f"Plot process exited {code}",
                    "log": str(log), "log_sha256": digest(log), "console_tail": tail_text(log, 6000)})
                self.save()
                print(f"PRESERVE FAILED plotting {spec['id']}; see {log}", flush=True)
            else:
                for failure in failures:
                    failure.update(resolved_by="per-arm-plot-retry", resolution_evidence=str(manifest_path))
                self.manifest.setdefault("arm_plots", {})[spec["id"]] = {
                    "manifest_path": str(manifest_path), "manifest_sha256": digest(manifest_path)}
                self.save()
                print(f"PLOT {spec['id']} {output}", flush=True)

    def plot_workload_comparisons(self):
        """Draw a comparison as soon as two complete arms exist, then all three."""
        script = ROOT / "experiments/plot_operator800_comparison.py"
        for workload in dict.fromkeys(spec["workload"] for spec in tuning_order(self.manifest, self.protocol)):
            specs = [spec for spec in self.protocol["order"] if not spec["smoke"]
                     and spec["workload"] == workload and spec["id"] in self.manifest["accepted"]]
            specs.sort(key=lambda spec: spec["arm"])
            if len(specs) < 2 or not any(spec["arm"] == "B" for spec in specs):
                continue
            label = workload + "_" + "".join(spec["arm"] for spec in specs)
            signature = object_hash({"script_sha256": digest(script),
                "inputs": {spec["id"]: self.manifest["accepted"][spec["id"]] for spec in specs}})
            previous = self.manifest.get("comparison_plots", {}).get(label)
            if previous and previous["input_signature"] == signature:
                path = Path(previous["manifest_path"])
                if path.exists() and digest(path) == previous["manifest_sha256"]:
                    saved = read_json(path)
                    if all((path.parent / name).is_file() and digest(path.parent / name) == expected
                           for name, expected in saved["output_sha256"].items()):
                        continue
            failures = [failure for failure in self.manifest["failures"]
                        if failure["phase"] == "comparison_plot" and failure["id"] == label
                        and not failure.get("resolved_by")]
            if len(failures) >= self.args.max_attempts:
                continue
            output = self.args.figure_output.resolve() / "comparisons" / label
            inputs = self.output / "comparison_inputs"
            inputs.mkdir(exist_ok=True)
            report_input = inputs / (label + "_" + str(time.time_ns()) + ".json")
            shutil.copy2(self.manifest_path, report_input)
            log = inputs / (report_input.stem + ".log")
            cmd = [sys.executable, "-u", "-X", "utf8", str(script), "--manifest", str(report_input),
                   "--workload", workload, "--output", str(output)]
            with log.open("w", encoding="utf-8") as stream:
                code = subprocess.run(cmd, cwd=ROOT, env=environment(), stdout=stream,
                                      stderr=subprocess.STDOUT).returncode
            path = output / "manifest.json"
            if code or not path.exists():
                self.manifest["failures"].append({"root": str(self.output), "phase": "comparison_plot",
                    "id": label, "utc": utc_now(), "reason": f"Comparison plot process exited {code}",
                    "log": str(log), "log_sha256": digest(log), "console_tail": tail_text(log, 6000)})
                print(f"PRESERVE FAILED comparison {label}; see {log}", flush=True)
            else:
                for failure in failures:
                    failure.update(resolved_by="comparison-plot-retry", resolution_evidence=str(path))
                self.manifest.setdefault("comparison_plots", {})[label] = {
                    "input_signature": signature, "manifest_path": str(path), "manifest_sha256": digest(path)}
                print(f"COMPARE {label} {output}", flush=True)
            self.save()

    def monitor(self, force=False):
        if not force and time.monotonic() < self.next_check:
            return
        current = None
        issues = []
        if self.active_root:
            try:
                current = snapshot(self.active_root)
                issues.extend(current["issues"])
            except (OSError, ValueError, KeyError) as error:
                issues.append("Current task snapshot unavailable: " + str(error))
        check = {"utc": utc_now(), "local_time": datetime.now().astimezone().isoformat(),
                 "controller_status": self.manifest["status"], "controller_pid": os.getpid(),
                 "active_root": str(self.active_root) if self.active_root else None,
                 "accepted": {identifier: ref["submitted"] for identifier, ref in self.manifest["accepted"].items()},
                 "current": current, "failures": self.manifest["failures"], "issues": issues}
        if current:
            for run in current["runs"]:
                if run["smoke"] or run["submitted_with_feedback"] < 100:
                    continue
                key = str(self.active_root) + ":" + run["id"]
                if key not in self.manifest["milestones100"]:
                    batches = complete_lines(next(self.active_root.glob(run["id"] + "_*/callback_batches.jsonl")))
                    sizes = [(b.get("cost_model") or {}).get("data_size", 0) for b in batches]
                    if any(b < a for a, b in zip(sizes, sizes[1:])):
                        issues.append("Model history reset: " + run["id"])
                    self.manifest["milestones100"][key] = {"utc": utc_now(), "run": run,
                        "latest_resource_sample": current["memory"], "history_monotonic": sizes == sorted(sizes)}
                    self.save()
        monitor_dir = self.output / "monitor15min"
        monitor_dir.mkdir(exist_ok=True)
        with (monitor_dir / "checks.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(check, ensure_ascii=False) + "\n")
        write_json_atomic(monitor_dir / "latest.json", check)
        rows = ["# 800 次提交实验进度", "", "时间：" + check["local_time"],
                "状态：" + self.manifest["status"],
                "已审核完整正式组：" + str(sum(not s["smoke"] and s["id"] in self.manifest["accepted"]
                                           for s in self.protocol["order"])) + "/12", "",
                "| 任务 | 完整反馈 / 预算 | 模型累计样本 | 最佳实测 µs |", "|---|---:|---:|---:|"]
        active_runs = {run["id"]: run for run in current["runs"]} if current else {}
        for spec in self.protocol["order"]:
            if spec["smoke"]:
                continue
            run = active_runs.get(spec["id"])
            ref = self.manifest["accepted"].get(spec["id"])
            if ref:
                result = read_json(Path(ref["result_path"]))
                count, size, best = ref["submitted"], result["measurement"]["xgb_data_size"], result["measurement"]["best_latency_us"]
            elif run:
                count, size, best = run["submitted_with_feedback"], (run["model"] or {}).get("data_size", 0), run["best_observed_us"]
            else:
                count, size, best = 0, 0, None
            rows.append(f"| {spec['id']} | {count}/800 | {size} | {f'{best:.3f}' if best is not None else '—'} |")
        rows += ["", "本次异常：" + ("；".join(issues) or "未发现异常"),
                 "保留失败尝试：" + str(len(self.manifest["failures"])),
                 "", "每 15 分钟更新本地报告；不会自动向对话发送消息。"]
        (monitor_dir / "AUDIT_CHECKPOINT.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
        print(f"CHECK {check['local_time']} accepted={len(self.manifest['accepted'])} issues={issues}", flush=True)
        self.next_check = time.monotonic() + self.args.interval

    def wait_existing(self, root):
        self.active_root = root
        self.save(status="adopting_existing", active_root=str(root))
        last_ledger_count = -1
        self.monitor(force=True)
        while guard_wrapper_alive(root):
            # Observe all new successful receipts; never start a competing GPU process.
            count = len(complete_lines(root / "execution.jsonl"))
            if count != last_ledger_count:
                self.refresh()
                last_ledger_count = count
            self.monitor()
            time.sleep(self.args.poll)
        clean_stale_children(root)
        self.refresh(terminal=True)
        self.monitor(force=True)
        self.active_root = None

    def wait_memory(self):
        good = 0
        while good < 3:
            memory = system_memory()
            good = good + 1 if (memory["available_physical"] >= 1.5 * 2**30
                                and memory["available_commit"] >= 6 * 2**30) else 0
            self.save(status="waiting_for_memory", available_memory=memory)
            self.monitor()
            time.sleep(self.args.poll)

    def new_root(self, label, phase):
        # The frozen driver uses long descriptive per-run names. Keep the raw
        # parent short without changing that driver or Windows global settings.
        parent = ROOT / "experiments/results"
        key = object_hash({"cohort": str(self.output), "protocol": self.manifest["protocol_sha256"]})[:6]
        index = 1
        while (parent / f"o8_{key}_{index:03d}").exists():
            index += 1
        root = parent / f"o8_{key}_{index:03d}"
        validate_tuning_paths(root, self.protocol)
        root.mkdir()
        shutil.copy2(self.manifest["protocol_path"], root / "protocol.json")
        write_json_atomic(root / "recovery_task.json", {"id": label, "phase": phase,
                          "cohort": str(self.manifest_path), "utc": utc_now()})
        self.manifest["attempt_roots"].append(str(root))
        self.save()
        return root

    def execute(self, root, cmd, phase, label, env):
        verify_sources(self.protocol)
        for previous in task_roots(self.manifest):
            if previous != root and guard_is_alive(previous):
                raise RuntimeError(f"Another registered experiment is still active: {previous}")
        self.wait_memory()
        verify_sources(self.protocol)
        guard = [sys.executable, "-u", "-X", "utf8", str(ROOT / "experiments/guard_experiment_resources.py"),
                 "--output", str(root / "resource_guard"), "--interval", "5",
                 "--physical-reserve-mib", "256", "--commit-reserve-gib", "4",
                 "--private-limit-gib", "4", "--consecutive-samples", "3", "--", *cmd]
        self.active_root = root
        self.save(status="running", active_root=str(root), phase=phase, current=label)
        write_json_atomic(root / "progress.json", {"status": "starting", "phase": phase, "current": label})
        started = time.monotonic()
        with (root / "guard.console.log").open("w", encoding="utf-8") as stream:
            process = subprocess.Popen(guard, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
            self.save(active_guard_pid=process.pid, active_guard_created=psutil.Process(process.pid).create_time())
            while process.poll() is None:
                self.monitor()
                time.sleep(self.args.poll)
        clean_stale_children(root)
        state = read_json(root / "resource_guard/status.json")
        if phase not in ("tuning", "smoke"):
            console = root / f"{label}.{phase}.log"
            shutil.copy2(root / "resource_guard/child.console.log", console)
            receipt = {"phase": phase, "id": label, "argv": cmd,
                       "returncode": state.get("returncode", process.returncode),
                       "guard_status": state["status"], "wall_s": time.monotonic() - started,
                       "console_sha256": digest(console)}
            with (root / "execution.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(receipt, ensure_ascii=False) + "\n")
        self.refresh(terminal=True)
        self.monitor(force=True)
        self.active_root = None
        if (phase in ("tuning", "smoke") and label in self.manifest["accepted"]
                and self.manifest["accepted"][label].get("completion_basis") == "independently_revalidated_postcheck"):
            print(f"REVALIDATED {label}; original postcheck exit remains 1", flush=True)
            return True
        if state["status"] != "completed":
            self.failure(root, phase, label, "Resource guard ended: " + state["status"])
            return False
        return True

    def run(self):
        # Resume safely after the controller itself was interrupted.
        for root in task_roots(self.manifest):
            if (root / "resource_guard/status.json").exists():
                if guard_is_alive(root):
                    self.wait_existing(root)
                else:
                    clean_stale_children(root)
        self.refresh(terminal=True)
        while True:
            spec = choose_tuning(self.manifest, self.protocol, self.args.max_attempts)
            if spec is None:
                break
            phase = "smoke" if spec["smoke"] else "tuning"
            root = self.new_root(spec["id"], phase)
            cmd = [sys.executable, "-u", "-X", "utf8", str(ROOT / "experiments/run_operator800_study.py"),
                   "--output", str(root), "--seed", str(self.protocol["seed"]), "--only", spec["id"]]
            self.execute(root, cmd, phase, spec["id"], environment())
        # A postprocessing failure never prevents the remaining tuning arms from running.
        report = collect_cohort(self.manifest_path)
        for workload in dict.fromkeys(s["workload"] for s in self.protocol["order"]):
            rows = [row for row in report["runs"] if row["workload"] == workload]
            if len(rows) != 3:
                continue
            for phase in ("retiming", "pytorch"):
                while workload not in self.manifest[phase]:
                    failures = [f for f in self.manifest["failures"] if f["phase"] == phase and f["id"] == workload
                                and not f.get("resolved_by")]
                    if len(failures) >= self.args.max_attempts:
                        break
                    root = self.new_root(workload, phase)
                    if phase == "retiming":
                        target = root / "retiming" / f"{workload}_ga1_R01"
                        cmd = [sys.executable, "-u", "-X", "utf8", str(ROOT / "experiments/remeasure_top1.py"),
                               "--output", str(target), "--library-dir", str(ROOT / "build-prune/lib"), "--terminal-worker-exit"]
                        for arm in "BCD":
                            row = next(row for row in rows if row["arm"] == arm)
                            cmd += ["--result", f"{arm}={row['result_path']}"]
                        env = environment()
                    else:
                        target = root / "pytorch" / f"{workload}.json"
                        cmd = [sys.executable, "-u", "-X", "utf8", str(ROOT / "experiments/benchmark_torch_operators.py"),
                               "--workload", workload, "--output", str(target)]
                        env = environment()
                        env["TVM_EXPERIMENT_NO_TORCH"] = "0"
                    self.execute(root, cmd, phase, workload, env)
                    # Catch successful child exits whose artifact audit failed.
                    if workload not in self.manifest[phase]:
                        self.failure(root, phase, workload, "Phase did not produce accepted artifacts")
        self.save(status="reporting", active_root=None)
        # The live controller manifest keeps changing; freeze the exact report input.
        report_input = self.output / ("report_input_" + str(time.time_ns()) + ".json")
        shutil.copy2(self.manifest_path, report_input)
        report = write_report(report_input, self.args.figure_output.resolve())
        if report["completion_ready"]:
            completion = {"status": "completed", "utc": utc_now(), "formal_runs": 12,
                          "submitted": sum(row["budget"]["actual_submitted"] for row in report["runs"]),
                          "finite_correct": sum(row["outcomes"].get("finite", 0) for row in report["runs"]),
                          "cohort_manifest": str(self.manifest_path), "report_input": str(report_input),
                          "cohort_sha256_at_report": digest(report_input),
                          "report": str(self.args.figure_output.resolve() / "RESULTS.md"),
                          "report_manifest_sha256": digest(self.args.figure_output.resolve() / "manifest.json")}
            write_json_atomic(self.output / "COMPLETION.json", completion)
            self.save(status="completed")
        else:
            self.save(status="needs_attention", pending=report["pending"])
        self.monitor(force=True)
        return 0 if report["completion_ready"] else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Existing frozen operator800 study")
    parser.add_argument("--output", type=Path, required=True, help="Separate recovery cohort directory")
    parser.add_argument("--figure-output", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=900)
    parser.add_argument("--poll", type=float, default=5)
    parser.add_argument("--max-attempts", type=int, default=2, help="Per task, including failed adopted attempts")
    parser.add_argument("--dry-run", action="store_true", help="Inspect protocol and guard without writing or launching")
    args = parser.parse_args()
    if args.interval <= 0 or args.poll <= 0 or args.max_attempts < 1:
        parser.error("Intervals and attempt limit must be positive")
    args.input = args.input.resolve()
    protocol = read_json(args.input / "protocol.json")
    verify_sources(protocol)
    if args.dry_run:
        print(json.dumps({"source_verified": True, "adopt_active_guard": guard_is_alive(args.input),
                          "formal_runs": sum(not s["smoke"] for s in protocol["order"]),
                          "expected_submitted": 9600, "writes": False, "launches": False}, ensure_ascii=False))
        return
    args.output = args.output.resolve()
    if args.output == args.input:
        parser.error("Recovery output must be separate from the adopted raw study")
    args.output.mkdir(parents=True, exist_ok=True)
    lock = acquire_lock(args.output)
    controller = None
    try:
        manifest_path = args.output / "cohort.json"
        if manifest_path.exists():
            manifest = read_json(manifest_path)
            if (manifest["protocol_path"] != str(args.input / "protocol.json")
                    or manifest["protocol_sha256"] != digest(args.input / "protocol.json")
                    or manifest["figure_output"] != str(args.figure_output.resolve())):
                raise ValueError("Recovery cohort input, protocol or figure output changed")
        else:
            manifest = {"schema_version": 1, "status": "initializing", "created_utc": utc_now(),
                        "protocol_path": str(args.input / "protocol.json"),
                        "protocol_sha256": digest(args.input / "protocol.json"),
                        "figure_output": str(args.figure_output.resolve()),
                        "attempt_roots": [str(args.input)], "accepted": {}, "retiming": {},
                        "pytorch": {}, "failures": [], "milestones100": {}}
        manifest["controller"] = {**lock, "source_sha256": digest(Path(__file__)),
                                  "cohort_code_sha256": digest(Path(__file__).with_name("operator800_cohort.py")),
                                  "postcheck_code_sha256": digest(Path(__file__).with_name("operator800_postcheck.py")),
                                  "arm_plot_code_sha256": digest(Path(__file__).with_name("plot_operator800_arm.py")),
                                  "interval_s": args.interval, "max_attempts": args.max_attempts,
                                  "physical_reserve_mib": 256, "commit_reserve_gib": 4,
                                  "private_limit_gib": 4, "preflight_physical_gib": 1.5,
                                  "preflight_commit_gib": 6, "raw_path_limit_chars": 240}
        stamp = str(time.time_ns())
        for name in ("complete_operator800_study.py", "operator800_cohort.py", "operator800_postcheck.py", "plot_operator800_arm.py", "watch_operator800_cohort.py", "plot_operator800_comparison.py"):
            shutil.copy2(Path(__file__).with_name(name), args.output / ("snapshot_" + stamp + "_" + name))
        controller = Controller(args, manifest, protocol)
        controller.save()
        ensure_running(args.output)
        code = controller.run()
    except BaseException as error:
        if controller is not None:
            controller.save(status="controller_error", error=repr(error), traceback=traceback.format_exc())
            controller.monitor(force=True)
        raise
    finally:
        path = args.output / "controller.lock.json"
        if path.exists() and read_json(path) == lock:
            path.unlink()
    raise SystemExit(code)


if __name__ == "__main__":
    main()
