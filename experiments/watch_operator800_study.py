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
"""Read-only periodic checks of an operator study; preserve each check on disk."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import time

import psutil


def read_json(path):
    # The experiment's existing status writer can briefly leave a partial file.
    for attempt in range(3):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            if attempt == 2:
                raise
            time.sleep(0.1)


def tail_text(path, limit=32768):
    if not path.exists():
        return ""
    with path.open("rb") as stream:
        stream.seek(max(0, path.stat().st_size - limit))
        return stream.read().decode("utf-8", errors="replace")


def complete_lines(path):
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            records.append(json.loads(line))
        except ValueError:
            # A final append may still be in flight. Previously complete lines remain usable.
            continue
    return records


def latest_memory(path):
    for line in reversed(tail_text(path, 131072).splitlines()):
        try:
            return json.loads(line)
        except ValueError:
            continue
    return None


def process_alive(pid, created=None):
    if not pid:
        return False
    try:
        process = psutil.Process(pid)
        return (created is None or process.create_time() == created) and process.num_threads() > 0
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False


def snapshot(root):
    progress = read_json(root / "progress.json")
    guard = read_json(root / "resource_guard/status.json")
    protocol = read_json(root / "protocol.json")
    memory = latest_memory(root / "resource_guard/memory.jsonl")
    issues, runs = [], []
    for spec in protocol["order"]:
        matches = list(root.glob(spec["id"] + "_*/"))
        if not matches:
            continue
        directory = matches[0]
        batches = complete_lines(directory / "callback_batches.jsonl")
        submitted = sum(len(batch["builder_errors"]) for batch in batches)
        costs = [statistics.mean(values) for batch in batches
                 for build_error, run_error, values in zip(batch["builder_errors"],
                                                           batch["runner_errors"], batch["runner_run_secs"])
                 if build_error is None and run_error is None and values
                 and all(math.isfinite(value) and 0 < value < 1e9 for value in values)]
        result = read_json(directory / "result.json") if (directory / "result.json").exists() else None
        latest = batches[-1] if batches else {}
        runs.append({"id": spec["id"], "workload": spec["workload"], "arm": spec["arm"],
                     "smoke": spec["smoke"], "budget": spec["trials"],
                     "complete_callback_batches": len(batches), "submitted_with_feedback": submitted,
                     "best_observed_us": min(costs) * 1e6 if costs else None,
                     "build_errors": sum(error is not None for b in batches for error in b["builder_errors"]),
                     "runner_errors": sum(error is not None for b in batches for error in b["runner_errors"]),
                     "model": latest.get("cost_model"), "result_present": result is not None})
    guard_alive = process_alive(guard["guard_pid"])
    driver_alive = process_alive(guard.get("child_pid"), guard.get("child_created"))
    terminal = guard["status"] not in ("starting", "running")
    if guard["status"] == "running" and (not guard_alive or not driver_alive):
        issues.append("Guard or driver exited while its status still says running")
        terminal = True
    if terminal and guard["status"] != "completed":
        issues.append("Experiment stopped: " + guard["status"])
    if progress["status"] == "failed":
        issues.append("Experiment driver reported failure")
    if memory:
        age = datetime.now(timezone.utc).timestamp() - datetime.fromisoformat(memory["utc"]).timestamp()
        if not terminal and age > 60:
            issues.append(f"Resource samples stale for {age:.0f}s")
    touched = [p.stat().st_mtime for p in root.glob(progress.get("current", "") + "_*/timeline.jsonl")]
    if not terminal and touched and time.time() - max(touched) > 900:
        issues.append("No timeline progress for more than 15 minutes; inspect for a stall")
    log = Path(progress["log"]) if progress.get("log") else root / "resource_guard/child.console.log"
    return {"utc": datetime.now(timezone.utc).isoformat(),
            "local_time": datetime.now().astimezone().isoformat(), "progress": progress,
            "guard_status": guard["status"], "guard_alive": guard_alive, "driver_alive": driver_alive,
            "current_process_alive": process_alive(progress.get("pid")),
            "memory": memory, "runs": runs, "issues": issues, "terminal": terminal,
            "console_tail": tail_text(log, 8000),
            "guard_stderr_tail": tail_text(root / "guard.stderr.log", 4000)}


def write_json_atomic(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def record_check(output, check):
    with (output / "checks.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(check, ensure_ascii=False) + "\n")
    write_json_atomic(output / "latest.json", check)
    lines = ["# 单算子实验巡检", "", "时间：" + check["local_time"],
             "", "当前任务：" + check["progress"].get("current", check["progress"]["status"]),
             "资源守护状态：" + check["guard_status"], "",
             "| 实验 | 完整反馈提交数 / 预算 | 最佳实测延迟（µs） | 模型累计样本 | 结果文件 |",
             "|---|---:|---:|---:|---|"]
    for run in check["runs"]:
        best = f"{run['best_observed_us']:.3f}" if run["best_observed_us"] is not None else "—"
        size = (run["model"] or {}).get("data_size", 0)
        lines.append(f"| {run['id']} | {run['submitted_with_feedback']} / {run['budget']} | "
                     f"{best} | {size} | {'已生成' if run['result_present'] else '待完成'} |")
    if check["memory"]:
        memory = check["memory"]
        lines += ["", f"存活实验进程私有内存：{memory['tree_private_bytes'] / 2**20:.0f} MiB；"
                  f"可用物理内存：{memory['system']['available_physical'] / 2**20:.0f} MiB；"
                  f"提交余量：{memory['system']['available_commit'] / 2**30:.2f} GiB。"]
    lines += ["", "异常：" + ("；".join(check["issues"]) if check["issues"] else "本次未发现异常。"),
              "", "提交数按已完成模型更新的批次统计；正在进行的批次暂不计入。启动检查单独标为 S01–S04。",
              "", "后台巡检只读取实验数据，不修改搜索参数、模型或候选。"]
    (output / "STATUS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"CHECK {check['local_time']} current={check['progress'].get('current')} "
          f"guard={check['guard_status']} issues={check['issues']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=900)
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("The interval must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, args.output / "watcher_snapshot.py")
    identity = {"status": "running", "pid": os.getpid(),
                "created": psutil.Process().create_time(), "interval_s": args.interval,
                "input": str(args.input.resolve()),
                "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    write_json_atomic(args.output / "watcher.json", identity)
    deadline = time.monotonic()
    last_status = None
    try:
        while True:
            guard = read_json(args.input / "resource_guard/status.json")
            terminal = guard["status"] not in ("starting", "running") or not process_alive(guard["guard_pid"])
            if (terminal and guard["status"] != last_status) or time.monotonic() >= deadline:
                check = snapshot(args.input)
                record_check(args.output, check)
                last_status = check["guard_status"]
                if check["guard_status"] == "completed":
                    identity["status"] = "completed"
                    break
                identity["status"] = "monitoring_stopped_experiment" if check["terminal"] else "running"
                write_json_atomic(args.output / "watcher.json", identity)
                deadline = time.monotonic() + args.interval
            time.sleep(min(5, max(0.1, deadline - time.monotonic())))
    except BaseException as error:
        identity.update(status="watcher_error", error=repr(error))
        raise
    finally:
        identity["ended_utc"] = datetime.now(timezone.utc).isoformat()
        write_json_atomic(args.output / "watcher.json", identity)


if __name__ == "__main__":
    main()
