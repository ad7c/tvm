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
"""Read-only observation of worker and compiler lifetimes in a running study."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import psutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-pid", type=int, required=True)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=150)
    args = parser.parse_args()
    if not 1 <= args.seconds <= 180:
        raise ValueError("Observation must be bounded by 180 seconds")
    root = psutil.Process(args.root_pid)
    command = root.cmdline()
    if not any("model_scout_study.py" in item for item in command) or "tune" not in command:
        raise ValueError("Root must be the registered tuning worker")
    if str(args.study.resolve()).lower() not in " ".join(command).lower():
        raise ValueError("Root belongs to another study")
    started = time.time()
    rows = {}
    args.output.mkdir(parents=True, exist_ok=False)
    print("Observing existing tuning tree; no compiler or CUDA workload launched", flush=True)
    while time.time() - started < args.seconds:
        try:
            processes = root.children(recursive=True)
        except psutil.NoSuchProcess:
            break
        live = set()
        for proc in processes:
            try:
                with proc.oneshot():
                    key = f"{proc.pid}:{proc.create_time()}"
                    live.add(key)
                    if key not in rows:
                        rows[key] = {"pid": proc.pid, "ppid": proc.ppid(), "name": proc.name(),
                                     "created": proc.create_time(), "argv": proc.cmdline(),
                                     "first_seen": time.time(), "preexisting": proc.create_time() < started,
                                     "max_rss": 0, "cpu_s": 0}
                    row = rows[key]
                    row["last_seen"] = time.time()
                    row["max_rss"] = max(row["max_rss"], proc.memory_info().rss)
                    row["cpu_s"] = sum(proc.cpu_times()[:2])
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        for key, row in rows.items():
            if key not in live and "exit_observed" not in row:
                row["exit_observed"] = time.time()
                print("Process exited:", row["pid"], row["name"], flush=True)
        time.sleep(0.2)
    payload = {"started_utc": datetime.fromtimestamp(started, timezone.utc).isoformat(),
               "ended_utc": datetime.now(timezone.utc).isoformat(), "root_pid": root.pid,
               "root_argv": command, "poll_interval_s": 0.2, "processes": list(rows.values()),
               "scope": "Read-only process metadata; pre-compiler time also includes import and TIR lowering"}
    (args.output / "processes.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("Saved observation:", args.output, flush=True)


if __name__ == "__main__":
    main()
