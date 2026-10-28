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
"""Monitor a launched experiment tree; stop only that tree under sustained memory pressure."""

import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import psutil


class MemoryStatus(ctypes.Structure):
    _fields_ = [("length", ctypes.c_uint32), ("load", ctypes.c_uint32),
                ("total_physical", ctypes.c_uint64), ("available_physical", ctypes.c_uint64),
                ("total_commit", ctypes.c_uint64), ("available_commit", ctypes.c_uint64),
                ("total_virtual", ctypes.c_uint64), ("available_virtual", ctypes.c_uint64),
                ("available_extended_virtual", ctypes.c_uint64)]


def system_memory():
    if os.name != "nt":
        raise RuntimeError("This guard's commit accounting is Windows-specific")
    state = MemoryStatus()
    state.length = ctypes.sizeof(state)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(state)):
        raise ctypes.WinError()
    return {key: int(getattr(state, key)) for key in
            ("total_physical", "available_physical", "total_commit", "available_commit")}


def pressure_reasons(sample, physical_reserve, commit_reserve, private_limit):
    reasons = []
    if sample["system"]["available_physical"] < physical_reserve:
        reasons.append("physical_memory_reserve")
    if sample["system"]["available_commit"] < commit_reserve:
        reasons.append("system_commit_reserve")
    if sample["tree_private_bytes"] > private_limit:
        reasons.append("experiment_private_memory_limit")
    return reasons


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def collect_processes(owned):
    """Count executing processes; system commit also covers threadless snapshots."""
    live, exited, errors = [], [], []
    for process in owned:
        try:
            threads = process.num_threads()
            if threads == 0:
                exited.append(process.pid)
                continue
            memory = process.memory_info()
            if process.num_threads() == 0:
                exited.append(process.pid)
                continue
            record = {"pid": process.pid, "name": process.name(),
                      "created": process.create_time(), "threads": threads,
                      "rss": memory.rss, "private": memory.private}
            try:
                record["argv"] = process.cmdline()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                record["argv"] = None
            live.append(record)
        except psutil.NoSuchProcess:
            exited.append(process.pid)
        except psutil.AccessDenied:
            errors.append({"pid": process.pid, "error": "access_denied"})
    return live, exited, errors


def kill_launched_tree(root, known=()):
    # Retain already verified identities in case the parent has exited first.
    targets = {process.pid: process for process in [root, *known]}
    errors, terminated = [], []
    try:
        targets.update({process.pid: process for process in root.children(recursive=True)})
    except psutil.NoSuchProcess:
        pass
    except psutil.AccessDenied:
        errors.append({"pid": root.pid, "operation": "list_children", "error": "access_denied"})
    for process in targets.values():
        try:
            if process.num_threads() == 0:
                continue
            process.kill()
            terminated.append(process.pid)
        except psutil.NoSuchProcess:
            pass
        except psutil.AccessDenied:
            # Threadless Windows entries can reject kill; continue cleaning live workers.
            try:
                if process.num_threads() > 0:
                    errors.append({"pid": process.pid, "operation": "kill", "error": "access_denied"})
            except psutil.NoSuchProcess:
                pass
            except psutil.AccessDenied:
                errors.append({"pid": process.pid, "operation": "verify_exit", "error": "access_denied"})
    deadline = time.monotonic() + 5
    remaining = []
    while True:
        remaining = []
        for process in targets.values():
            try:
                if process.is_running() and process.num_threads() > 0:
                    remaining.append(process.pid)
            except psutil.NoSuchProcess:
                pass
            except psutil.AccessDenied:
                remaining.append(process.pid)
        if not remaining or time.monotonic() >= deadline:
            break
        time.sleep(0.1)
    return {"terminated": terminated, "errors": errors, "remaining_live": remaining}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=5)
    parser.add_argument("--physical-reserve-mib", type=float, default=1024)
    parser.add_argument("--commit-reserve-gib", type=float, default=4)
    parser.add_argument("--private-limit-gib", type=float, default=4)
    parser.add_argument("--consecutive-samples", type=int, default=3)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    cmd = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not cmd or args.interval <= 0 or args.consecutive_samples < 1:
        parser.error("A command and positive sampling settings are required")
    args.output.mkdir(parents=True, exist_ok=False)
    meta = {"status": "starting", "argv": cmd, "guard_pid": os.getpid(),
            "utc": datetime.now(timezone.utc).isoformat(), "interval_s": args.interval,
            "physical_reserve_bytes": int(args.physical_reserve_mib * 2**20),
            "commit_reserve_bytes": int(args.commit_reserve_gib * 2**30),
            "private_limit_bytes": int(args.private_limit_gib * 2**30),
            "consecutive_samples": args.consecutive_samples,
            "guard_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "psutil_version": psutil.__version__, "scope": "Only the child launched here and its descendants"}
    shutil.copy2(__file__, args.output / "guard_snapshot.py")
    write_json(args.output / "status.json", meta)
    with (args.output / "child.console.log").open("w", encoding="utf-8") as stream:
        child = subprocess.Popen(cmd, stdout=stream, stderr=subprocess.STDOUT)
        root = psutil.Process(child.pid)
        meta.update(status="running", child_pid=child.pid, child_created=root.create_time())
        write_json(args.output / "status.json", meta)
        count = 0
        known = {root.pid: root}
        try:
            while child.poll() is None:
                try:
                    owned = [root, *root.children(recursive=True)]
                except psutil.NoSuchProcess:
                    owned = []
                processes, exited, errors = collect_processes(owned)
                live_pids = {record["pid"] for record in processes}
                known.update({process.pid: process for process in owned if process.pid in live_pids})
                sample = {"utc": datetime.now(timezone.utc).isoformat(), "system": system_memory(),
                          "processes": processes, "excluded_threadless_pids": exited,
                          "inspection_errors": errors,
                          "tree_private_bytes": sum(p["private"] for p in processes)}
                reasons = pressure_reasons(sample, meta["physical_reserve_bytes"],
                                           meta["commit_reserve_bytes"], meta["private_limit_bytes"])
                count = count + 1 if reasons else 0
                sample.update(pressure_reasons=reasons, pressure_count=count)
                with (args.output / "memory.jsonl").open("a", encoding="utf-8") as journal:
                    journal.write(json.dumps(sample, ensure_ascii=False) + "\n")
                if count >= args.consecutive_samples:
                    meta.update(status="stopped_for_memory_pressure", stop_sample=sample)
                    write_json(args.output / "status.json", meta)
                    meta["cleanup"] = kill_launched_tree(root, known.values())
                    child.wait(timeout=30)
                    print("Resource guard stopped its experiment tree: " + ", ".join(reasons), flush=True)
                    break
                time.sleep(args.interval)
            code = child.wait()
            if meta["status"] == "running":
                meta["status"] = "completed" if code == 0 else "child_failed"
                meta["cleanup"] = kill_launched_tree(root, known.values())
            if meta.get("cleanup", {}).get("remaining_live"):
                meta["status_before_cleanup_error"] = meta["status"]
                meta["status"] = "cleanup_incomplete"
            meta.update(returncode=code, windows_hex=hex(code & 0xffffffff),
                        ended_utc=datetime.now(timezone.utc).isoformat())
            write_json(args.output / "status.json", meta)
        except BaseException as error:
            meta["status_before_guard_error"] = meta["status"]
            meta.update(status="guard_error", error=repr(error),
                        cleanup=kill_launched_tree(root, known.values()),
                        ended_utc=datetime.now(timezone.utc).isoformat())
            write_json(args.output / "status.json", meta)
            raise
    if meta["status"] != "completed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
