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
"""Publish a lightweight live status page without importing TVM or changing tuning."""

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil

from watch_operator800_study import complete_lines, process_alive, read_json, snapshot, write_json_atomic


SHANGHAI = timezone(timedelta(hours=8))
STAGES = {"search_begin": "候选搜索", "builder_begin": "候选编译", "builder_end": "编译结束",
          "runner_begin": "硬件测量与正确性检查", "runner_end": "测量结束",
          "callback_begin": "测量反馈", "xgb_log": "代价模型更新",
          "callback_end": "测量反馈", "batch_complete": "批次反馈完成",
          "scheduler_updated": "准备下一批"}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def watcher_alive(output):
    path = output / "live_status/watcher.json"
    if not path.exists():
        return False
    state = read_json(path)
    if not process_alive(state.get("pid"), state.get("created")):
        return False
    try:
        return any(Path(arg).name == "watch_operator800_cohort.py"
                   for arg in psutil.Process(state["pid"]).cmdline())
    except psutil.NoSuchProcess:
        return False


def ensure_running(output):
    if watcher_alive(output):
        return
    folder = output / "live_status"
    folder.mkdir(exist_ok=True)
    stamp = str(time.time_ns())
    with (folder / ("stdout_" + stamp + ".log")).open("w", encoding="utf-8") as stdout, \
         (folder / ("stderr_" + stamp + ".log")).open("w", encoding="utf-8") as stderr:
        subprocess.Popen([sys.executable, "-u", "-X", "utf8", str(Path(__file__).resolve()),
                          "--cohort", str(output / "cohort.json"), "--interval", "30"],
                         stdout=stdout, stderr=stderr, cwd=Path(__file__).resolve().parents[1],
                         creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    # Wait for the child's ownership record before another caller could launch it.
    deadline = time.monotonic() + 10
    while not watcher_alive(output):
        if time.monotonic() >= deadline:
            raise RuntimeError("Live status publisher did not start; inspect live_status/stderr logs")
        time.sleep(0.1)


def live_check(manifest_path, cache):
    manifest = read_json(manifest_path)
    protocol = read_json(Path(manifest["protocol_path"]))
    root = Path(manifest["active_root"]) if manifest.get("active_root") else None
    active = snapshot(root) if root and (root / "resource_guard/status.json").exists() else None
    active_runs = {row["id"]: row for row in active["runs"]} if active else {}
    resume_path = root / "resume_state.json" if root else None
    resumed = read_json(resume_path) if resume_path and resume_path.exists() else None
    rows = []
    for spec in protocol["order"]:
        if spec["smoke"]:
            continue
        reference = manifest["accepted"].get(spec["id"])
        if reference:
            path = Path(reference["result_path"])
            key = (str(path), path.stat().st_mtime_ns, reference["artifacts_sha256"][str(path)])
            if key not in cache:
                if sha256(path) != key[-1]:
                    raise ValueError("Accepted result changed: " + spec["id"])
                result = read_json(path)
                cache[key] = {"submitted": reference["submitted"],
                              "data_size": result["measurement"]["xgb_data_size"],
                              "best_us": result["measurement"]["best_latency_us"]}
            values = cache[key]
            state = "已审核完整组" if values["submitted"] == spec["trials"] else "已审核提前终止"
        elif resumed and spec["id"] == "CONVG1C01":
            run = active_runs.get(spec["id"], {})
            latest_best = run.get("best_observed_us")
            best = min(resumed["historical_best_us"], latest_best) if latest_best is not None else resumed["historical_best_us"]
            values = {"submitted": resumed["actual_submitted"],
                      "data_size": resumed["actual_submitted"], "best_us": best}
            state = (f"恢复旧反馈 {resumed['restored_submitted']}/{resumed['historical_submitted']}"
                     if resumed["status"] in ("preparing_history", "restoring_history")
                     else f"续跑：历史 {resumed['historical_submitted']} + 新增 {resumed['new_submitted']}")
        elif spec["id"] in active_runs:
            run = active_runs[spec["id"]]
            values = {"submitted": run["submitted_with_feedback"],
                      "data_size": (run["model"] or {}).get("data_size", 0),
                      "best_us": run["best_observed_us"]}
            state = "调优中" if active["guard_status"] == "running" else active["guard_status"]
        else:
            values = {"submitted": 0, "data_size": 0, "best_us": None}
            state = "待运行"
            failures = [failure for failure in manifest.get("failures", [])
                        if failure["id"] == spec["id"] and failure["phase"] == "tuning"
                        and not failure.get("resolved_by")]
            if failures:
                latest = failures[-1]
                key = ("failed_attempt", latest["root"], spec["id"])
                if key not in cache:
                    previous = snapshot(Path(latest["root"]))
                    run = next((run for run in previous["runs"] if run["id"] == spec["id"]), None)
                    cache[key] = run
                run = cache[key]
                if run:
                    values = {"submitted": run["submitted_with_feedback"],
                              "data_size": (run["model"] or {}).get("data_size", 0),
                              "best_us": run["best_observed_us"]}
                state = ("资源保护中断，待处理" if latest.get("guard_status") == "stopped_for_memory_pressure"
                         else "提前结束或审核失败，待处理")
            failures = [failure for failure in manifest.get("failures", [])
                        if failure["id"] == spec["id"] and failure["phase"] == "tuning"
                        and not failure.get("resolved_by")]
            if failures:
                latest = failures[-1]
                key = ("failed_attempt", latest["root"], spec["id"])
                if key not in cache:
                    previous = snapshot(Path(latest["root"]))
                    run = next((run for run in previous["runs"] if run["id"] == spec["id"]), None)
                    cache[key] = run
                run = cache[key]
                if run:
                    values = {"submitted": run["submitted_with_feedback"],
                              "data_size": (run["model"] or {}).get("data_size", 0),
                              "best_us": run["best_observed_us"]}
                state = ("资源保护中断，待处理" if latest.get("guard_status") == "stopped_for_memory_pressure"
                         else "提前结束或审核失败，待处理")
        rows.append({"id": spec["id"], "budget": spec["trials"], "state": state, **values})
    stage = None
    if active:
        identifier = active["progress"].get("current", "")
        paths = list(root.glob(identifier + "_*/timeline.jsonl"))
        if len(paths) == 1:
            events = complete_lines(paths[0])
            if events:
                last = events[-1]
                stage = {"event": last["event"], "name": STAGES.get(last["event"], last["event"]),
                         "batch_index": last.get("batch_index"), "tune_s": last.get("tune_elapsed_s")}
    if resumed and resumed["status"] in ("preparing_history", "restoring_history"):
        stage = {"event": "restoring_history",
                 "name": f"恢复搜索状态：核对旧反馈 {resumed['restored_submitted']}/{resumed['historical_submitted']}，无新增测量"}
    audit_path = manifest_path.parent / "monitor15min/latest.json"
    audit = read_json(audit_path) if audit_path.exists() else {}
    controller = manifest.get("controller", {})
    return {"utc": datetime.now(timezone.utc).isoformat(),
            "local_time": datetime.now(SHANGHAI).isoformat(), "manifest_status": manifest["status"],
            "controller_alive": process_alive(controller.get("pid"), controller.get("created")),
            "active_root": str(root) if root else None, "active": active, "stage": stage,
            "last_audit_time": audit.get("local_time"), "rows": rows,
            "formal_complete": sum(row["state"].startswith("已审核") for row in rows)}


def render(check, interval):
    active = check["active"]
    current = active["progress"].get("current") if active else "—"
    stage = check["stage"]
    lines = ["# 800 次提交实验实时状态", "", "更新时间：" + check["local_time"],
             f"刷新间隔：{interval:g} 秒；完整巡检每 15 分钟记录。",
             "最近完整巡检：" + (check["last_audit_time"] or "—"), "",
             "当前任务：" + (current or "—"),
             "当前阶段：" + (stage["name"] if stage else "准备 / 等待"),
             "控制器：" + ("运行中" if check["controller_alive"] else "未运行，请检查控制器"),
             f"已审核正式组：{check['formal_complete']}/12（提前终止组保留实际提交数）", "",
             "| 任务 | 完整反馈 / 预算 | 模型累计样本 | 最佳实测 µs | 状态 |",
             "|---|---:|---:|---:|---|"]
    for row in check["rows"]:
        best = f"{row['best_us']:.3f}" if row["best_us"] is not None else "—"
        lines.append(f"| {row['id']} | {row['submitted']}/{row['budget']} | {row['data_size']} | {best} | {row['state']} |")
    if active:
        issues = active["issues"]
        lines += ["", "资源守护：" + active["guard_status"],
                  "本次异常：" + ("；".join(issues) or "未发现异常")]
        memory = active["memory"]
        if memory:
            lines.append(f"实验树 private：{memory['tree_private_bytes'] / 2**20:.0f} MiB；"
                         f"可用物理内存：{memory['system']['available_physical'] / 2**20:.0f} MiB；"
                         f"提交余量：{memory['system']['available_commit'] / 2**30:.2f} GiB。")
    lines += ["", "提交数按已完成反馈的批次统计；正在编译、测量或更新模型的批次暂不计入。",
              "刷新时间变化而提交数暂时不变，表示同一批次仍在运行。",
              "", "[15 分钟巡检摘要](AUDIT_CHECKPOINT.md) · [原始巡检记录](checks.jsonl)"]
    return "\n".join(lines) + "\n"


def publish(output, check, interval):
    folder = output / "live_status"
    folder.mkdir(exist_ok=True)
    write_json_atomic(folder / "latest.json", check)
    content = render(check, interval)
    for name in ("AUDIT_STATUS.md", "STATUS.md", "CURRENT_RUN.md"):
        status = output / "monitor15min" / name
        temporary = status.with_name(status.stem + "." + str(os.getpid()) + ".tmp")
        temporary.write_text(content, encoding="utf-8")
        for attempt in range(3):
            try:
                temporary.replace(status)
                break
            except PermissionError:
                if attempt == 2:
                    raise
                time.sleep(0.1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=30)
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("Interval must be positive")
    manifest_path = args.cohort.resolve()
    output = manifest_path.parent
    folder = output / "live_status"
    folder.mkdir(exist_ok=True)
    if watcher_alive(output):
        return
    identity = {"status": "running", "pid": os.getpid(), "created": psutil.Process().create_time(),
                "interval_s": args.interval, "source_sha256": sha256(Path(__file__)),
                "cohort": str(manifest_path), "status_path": str(output / "monitor15min/AUDIT_STATUS.md"),
                "mirror_path": str(output / "monitor15min/STATUS.md")}
    write_json_atomic(folder / "watcher.json", identity)
    cache = {}
    try:
        while True:
            check = live_check(manifest_path, cache)
            publish(output, check, args.interval)
            print(f"LIVE {check['local_time']} completed={check['formal_complete']}/12", flush=True)
            if check["manifest_status"] == "completed":
                identity["status"] = "completed"
                break
            time.sleep(args.interval)
    except BaseException as error:
        identity.update(status="watcher_error", error=repr(error))
        raise
    finally:
        identity["ended_utc"] = datetime.now(timezone.utc).isoformat()
        write_json_atomic(folder / "watcher.json", identity)


if __name__ == "__main__":
    main()
