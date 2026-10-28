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
"""Publish a local status page for one registered full-model experiment."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import time

import psutil


def read(path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default or {}


def publish(study):
    progress = read(study / "progress.json")
    protocol = read(study / "protocol.json")
    state = progress.get("status", "waiting")
    pid = progress.get("controller_pid")
    if state == "running" and pid and not psutil.pid_exists(pid):
        state = "controller_missing"
    phase, arm = progress.get("phase", "-"), progress.get("arm", "-")
    lines = ["# ResNet-18 / Scout 实验实时进度", "",
             f"更新时间：{datetime.now().astimezone().isoformat(timespec='seconds')}", "",
             f"状态：`{state}`；当前阶段：`{phase}`；当前组：`{arm}`。", "",
             "| 正式组 | 实际提交 / 上限 | 最新整网检查点 ms | 阶段 |",
             "|---|---:|---:|---|"]
    for name in "BCD":
        folder = study / ("formal_" + name)
        result = read(folder / "result.json")
        batch = read(folder / "progress.json")
        submitted = result.get("submitted", batch.get("submitted_total", batch.get("submitted", 0)))
        checkpoints = read(folder / "checkpoints.json", [])
        latency = "-"
        if checkpoints:
            measured = read(Path(checkpoints[-1]["result"]))
            if "median_ms" in measured:
                latency = f"{measured['median_ms']:.5f}"
        label = result.get("status", batch.get("status", "尚未开始"))
        lines.append(f"| {name} | {submitted}/{protocol.get('global_trials_per_arm', 2000)} | {latency} | {label} |")
    output = Path(progress["output"]) if progress.get("output") else None
    if output:
        batch = read(output / "progress.json")
        lines += ["", f"当前输出：`{output}`"]
        if phase == "pilot":
            lines += ["", f"链路验证提交：{batch.get('submitted_total', batch.get('submitted', 0))}/2；单独记录，不计入正式预算。"]
        log = output / "guard/child.console.log"
        if log.exists():
            with log.open("rb") as stream:
                stream.seek(max(0, log.stat().st_size - 4096))
                tail = stream.read().decode("utf-8", errors="replace").splitlines()
            events = [line for line in tail if any(s in line for s in
                      ("Initializing Task", "TaskScheduler picks", "Sending ", "Traceback", "Error:"))]
            if events:
                lines += ["", "最近日志：", "", "```text", events[-1], "```"]
    if progress.get("error"):
        lines += ["", "错误：`" + str(progress["error"]) + "`"]
    lines += ["", "B = MetaSchedule；C = CS+H；D = 完整 Scout。",
              "检查点延迟是完整模型实测；最终三组还将按轮换顺序独立重测。",
              "此文件每 15 秒更新；完成或异常后停止。"]
    (study / "STATUS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    args = parser.parse_args()
    while True:
        state = publish(args.study)
        if state == "completed":
            from report_model_scout_study import report
            report(args.study)
            return
        if state in ("failed", "controller_missing"):
            return
        time.sleep(15)


if __name__ == "__main__":
    main()
