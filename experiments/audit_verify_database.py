#!/usr/bin/env python3
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
"""Audit measured tuning records against the cheap thread feasibility check.

This script is deliberately read-only.  A record with finite run times has
already passed all postprocessors, compilation, and device execution.  Such a
record is therefore a useful false-rejection oracle for a proposed hard filter.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("TVM_LIBRARY_PATH", str(REPO_ROOT / "build-prune" / "lib"))
sys.path.insert(0, str(REPO_ROOT / "python"))

import tvm  # pylint: disable=wrong-import-position
from tvm.s_tir.meta_schedule.database import JSONDatabase  # pylint: disable=wrong-import-position


DEFAULT_ROOTS = (REPO_ROOT / "experiments" / "data",)


def _finite_run_times(record: Any) -> list[float]:
    values = [] if record.run_secs is None else [float(value) for value in record.run_secs]
    return [value for value in values if math.isfinite(value) and 0 < value < 1e9]


def audit(
    roots: list[Path], max_threads: int, warp_size: int, max_shared_memory: int
) -> dict[str, Any]:
    analyze = tvm.get_global_func("s_tir.meta_schedule.testing.AnalyzeThreadFeasibility")
    work_dirs = sorted(
        path.parent
        for root in roots
        if root.exists()
        for path in root.rglob("database_tuning_record.json")
    )
    arms = []
    unique_schedules: dict[str, dict[str, Any]] = {}
    trace_hashes: set[str] = set()
    classification_conflicts = []
    skipped_databases = []

    for work_dir in work_dirs:
        workload_path = work_dir / "database_workload.json"
        if not workload_path.exists():
            skipped_databases.append(
                {"work_dir": str(work_dir), "reason": "missing database_workload.json"}
            )
            continue
        result_path = work_dir / "result.json"
        metadata = (
            json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else {}
        )
        database = JSONDatabase(work_dir=str(work_dir), allow_missing=False)
        records = list(database.get_all_tuning_records())
        counts = {
            "records": len(records),
            "finite_records": 0,
            "known": 0,
            "unknown": 0,
            "rejected": 0,
            "finite_rejected": 0,
            "shared_proxy_rejected": 0,
            "finite_shared_proxy_rejected": 0,
            "replay_errors": 0,
        }
        reject_reasons: Counter[str] = Counter()
        max_threads_histogram: Counter[int] = Counter()
        record_rows = []

        for record_id, record in enumerate(records):
            trace_hash = hashlib.sha256(str(record.trace).encode("utf-8")).hexdigest()
            schedule_hash = hashlib.sha256(
                (str(record.workload.mod) + "\0" + str(record.trace)).encode("utf-8")
            ).hexdigest()
            trace_hashes.add(trace_hash)
            run_times = _finite_run_times(record)
            counts["finite_records"] += bool(run_times)
            try:
                candidate = record.as_measure_candidate()
                raw = [
                    int(value)
                    for value in analyze(
                        candidate.sch.mod, max_threads, warp_size, max_shared_memory
                    )
                ]
                extents_known = bool(raw[0])
                rejected = bool(raw[1])
                measured_threads = raw[2]
                contracts_known = bool(raw[3])
                known = extents_known and contracts_known
                reason_flags = {
                    "thread_extent_low": bool(raw[4]),
                    "thread_extent_high": bool(raw[5]),
                    "threads_per_block": bool(raw[6]),
                }
                shared_known = bool(raw[7])
                shared_proxy_rejected = bool(raw[8])
                max_declared_shared_bytes = raw[9]
            except Exception as err:  # pylint: disable=broad-exception-caught
                counts["replay_errors"] += 1
                record_rows.append(
                    {
                        "record": record_id,
                        "trace_sha256": trace_hash,
                        "finite": bool(run_times),
                        "error": f"{type(err).__name__}: {err}",
                    }
                )
                continue

            counts["known" if known else "unknown"] += 1
            counts["rejected"] += rejected
            counts["finite_rejected"] += bool(run_times) and rejected
            counts["shared_proxy_rejected"] += shared_proxy_rejected
            counts["finite_shared_proxy_rejected"] += (
                bool(run_times) and shared_proxy_rejected
            )
            for reason, active in reason_flags.items():
                reject_reasons[reason] += active
            if extents_known:
                max_threads_histogram[measured_threads] += 1
            classification = {
                "known": known,
                "extents_known": extents_known,
                "contracts_known": contracts_known,
                "rejected": rejected,
                "max_threads": measured_threads,
                "reject_reasons": reason_flags,
                "shared_known": shared_known,
                "shared_proxy_rejected": shared_proxy_rejected,
                "max_declared_shared_bytes": max_declared_shared_bytes,
                "finite": bool(run_times),
            }
            previous = unique_schedules.get(schedule_hash)
            if previous is None:
                unique_schedules[schedule_hash] = classification
            elif any(
                previous[key] != classification[key]
                for key in ("known", "rejected", "max_threads")
            ):
                classification_conflicts.append(
                    {
                        "schedule_sha256": schedule_hash,
                        "trace_sha256": trace_hash,
                        "first": previous,
                        "later": classification,
                    }
                )
            else:
                previous["finite"] = previous["finite"] or classification["finite"]
            record_rows.append(
                {
                    "record": record_id,
                    "schedule_sha256": schedule_hash,
                    "trace_sha256": trace_hash,
                    "finite": bool(run_times),
                    "best_run_us": min(run_times) * 1e6 if run_times else None,
                    **classification,
                }
            )

        arms.append(
            {
                "work_dir": str(work_dir),
                "seed": metadata.get("seed"),
                "mode": metadata.get("mode"),
                "soft_mode": metadata.get("soft_mode"),
                "soft_weight": metadata.get("soft_weight"),
                "ga_t2_mode": metadata.get("ga_t2_mode"),
                **counts,
                "reject_reasons": dict(sorted(reject_reasons.items())),
                "max_threads_histogram": {
                    str(key): value for key, value in sorted(max_threads_histogram.items())
                },
                "record_details": record_rows,
            }
        )

    totals = {
        key: sum(int(arm[key]) for arm in arms)
        for key in (
            "records",
            "finite_records",
            "known",
            "unknown",
            "rejected",
            "finite_rejected",
            "shared_proxy_rejected",
            "finite_shared_proxy_rejected",
            "replay_errors",
        )
    }
    unique_counts = {
        "unique_trace_hashes": len(trace_hashes),
        "unique_schedules": len(unique_schedules),
        "unique_known": sum(row["known"] for row in unique_schedules.values()),
        "unique_unknown": sum(not row["known"] for row in unique_schedules.values()),
        "unique_rejected": sum(row["rejected"] for row in unique_schedules.values()),
        "unique_finite_schedules": sum(row["finite"] for row in unique_schedules.values()),
        "unique_finite_rejected": sum(
            row["finite"] and row["rejected"] for row in unique_schedules.values()
        ),
        "max_threads_histogram": {
            str(key): value
            for key, value in sorted(
                Counter(
                    row["max_threads"] for row in unique_schedules.values() if row["known"]
                ).items()
            )
        },
    }
    return {
        "max_threads_per_block": max_threads,
        "thread_warp_size": warp_size,
        "max_shared_memory_per_block": max_shared_memory,
        "arms": arms,
        "totals": totals,
        "unique": unique_counts,
        "classification_conflicts": classification_conflicts,
        "skipped_databases": skipped_databases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="*", type=Path, default=list(DEFAULT_ROOTS))
    parser.add_argument("--max-threads", type=int, default=1024)
    parser.add_argument("--warp-size", type=int, default=32)
    parser.add_argument("--max-shared-memory", type=int, default=49152)
    parser.add_argument("--details", action="store_true", help="Keep per-record rows in output")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "experiments" / "data" / "hard_filter_v2_database_audit.json",
        help="Write the JSON audit here in addition to stdout",
    )
    args = parser.parse_args()
    result = audit(
        args.roots, args.max_threads, args.warp_size, args.max_shared_memory
    )
    if not args.details:
        for arm in result["arms"]:
            arm.pop("record_details")
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
