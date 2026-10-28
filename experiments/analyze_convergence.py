# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Build Pruner-style best-so-far convergence curves from tuning results.

The x axes are measured trials and wall time from the first scheduler pick.
The y axis is the best finite GPU latency observed up to each measurement
batch.  The script does not treat build/runner failures as large latencies.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import re
from typing import Any


TIMESTAMP_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")


def _parse_timestamp(line: str) -> datetime | None:
    match = TIMESTAMP_RE.match(line)
    if match is None:
        return None
    return datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S")


def _resolve_result(path: Path) -> Path:
    if path.is_file():
        return path
    matches = sorted(path.rglob("result.json"))
    if len(matches) != 1:
        raise ValueError(f"Expected one result.json below {path}, found {len(matches)}")
    return matches[0]


def _scheduler_batch_times(result_path: Path) -> list[float]:
    log_path = result_path.parent / "logs" / "tvm.s_tir.meta_schedule.logging.task_scheduler.log"
    if not log_path.is_file():
        return []
    start: datetime | None = None
    updates: list[datetime] = []
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        timestamp = _parse_timestamp(line)
        if timestamp is None:
            continue
        if start is None and "TaskScheduler picks Task" in line:
            start = timestamp
        if "[Updated] Task" in line:
            updates.append(timestamp)
    if start is None:
        return []
    return [(timestamp - start).total_seconds() for timestamp in updates]


def _finite_batch_latencies(batch: dict[str, Any]) -> list[float]:
    latencies: list[float] = []
    runner_errors = batch.get("runner_errors", [])
    for index, values in enumerate(batch.get("runner_run_secs", [])):
        error = runner_errors[index] if index < len(runner_errors) else None
        if error is not None or values is None:
            continue
        finite = [float(value) for value in values if math.isfinite(float(value)) and value > 0]
        if finite:
            latencies.append(sum(finite) / len(finite) * 1e6)
    return latencies


def _arm_curve(name: str, result_path: Path) -> dict[str, Any]:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    measurement = result["measurement"]
    elapsed = _scheduler_batch_times(result_path)
    points: list[dict[str, Any]] = []
    best_so_far = math.inf
    measured = 0
    for batch_index, batch in enumerate(measurement.get("callback_batches", [])):
        latencies = _finite_batch_latencies(batch)
        measured += len(latencies)
        if latencies:
            best_so_far = min(best_so_far, min(latencies))
        points.append(
            {
                "batch": batch_index + 1,
                "measured_trials": measured,
                "batch_finite": len(latencies),
                "batch_best_latency_us": min(latencies) if latencies else None,
                "best_so_far_latency_us": best_so_far if math.isfinite(best_so_far) else None,
                "elapsed_tuning_loop_s": elapsed[batch_index] if batch_index < len(elapsed) else None,
            }
        )
    return {
        "arm": name,
        "result": str(result_path.resolve()),
        "seed": result["seed"],
        "workload": result["workload"],
        "source_identity_sha256": result.get("source_identity_sha256"),
        "loaded_library_sha256": result.get("loaded_library_sha256"),
        "final_best_latency_us": measurement.get("best_latency_us"),
        "tune_call_s": result["timing"].get("tune_call_s"),
        "points": points,
    }


def _parse_arm(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--arm must be NAME=RESULT_OR_DIRECTORY")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("--arm must be NAME=RESULT_OR_DIRECTORY")
    return name, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", action="append", type=_parse_arm, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    arms = [_arm_curve(name, _resolve_result(path)) for name, path in args.arm]
    source_ids = {arm["source_identity_sha256"] for arm in arms}
    dll_ids = {arm["loaded_library_sha256"] for arm in arms}
    workloads = {arm["workload"] for arm in arms}
    seeds = {arm["seed"] for arm in arms}
    summary = {
        "schema_version": 1,
        "definition": {
            "x_trials": "cumulative finite GPU measurements",
            "x_time": "seconds from first scheduler pick to post-measurement model update",
            "y": "minimum finite measured GPU latency observed so far (us)",
        },
        "identity": {
            "source_identity_equal": len(source_ids) == 1,
            "loaded_library_equal": len(dll_ids) == 1,
            "workload_equal": len(workloads) == 1,
            "seed_equal": len(seeds) == 1,
        },
        "arms": arms,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=(
                "arm",
                "seed",
                "workload",
                "batch",
                "measured_trials",
                "batch_finite",
                "batch_best_latency_us",
                "best_so_far_latency_us",
                "elapsed_tuning_loop_s",
            ),
        )
        writer.writeheader()
        for arm in arms:
            for point in arm["points"]:
                writer.writerow(
                    {
                        "arm": arm["arm"],
                        "seed": arm["seed"],
                        "workload": arm["workload"],
                        **point,
                    }
                )
    print(args.output)
    print(csv_path)


if __name__ == "__main__":
    main()
