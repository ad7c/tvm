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
"""Summarize the four first-point v1.1 pruning experiment arms."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


ARM_ARGUMENTS = (
    ("B", "baseline"),
    ("H", "hard"),
    ("CS", "constraint_sampling"),
    ("CS+H", "combined"),
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sum_values(values: dict[str, int]) -> int:
    return sum(int(value) for value in values.values())


def _row(arm: str, path: Path, result: dict[str, Any]) -> dict[str, Any]:
    totals = result["search"]["totals"]
    accounting = totals["accounting"]
    sample_init = result["search"]["by_stage"]["sample_init"]["accounting"]
    sample_metrics = totals["constraint_sampling"]
    hard_positions = totals["hard_positions"]
    return {
        "arm": arm,
        "result": str(path),
        "hard_mode": result["mode"],
        "sample_mode": result["constraint_sample_mode"],
        "hard_position": result["hard_filter_position"],
        "seed": result["seed"],
        "dll_sha256": result["loaded_library_sha256"],
        "search_s": result["timing"]["search_s"],
        "tune_call_s": result["timing"]["tune_call_s"],
        "builder_s": result["timing"]["builder_s"],
        "runner_update_s": result["timing"]["runner_and_update_s"],
        "generated": accounting["generated"],
        "passed": accounting["passed"],
        "verify_failures": accounting["verify_failure_total"],
        "hard_rejected": _sum_values(accounting["hard_rejected"]),
        "sampling_abandoned": accounting["constraint_sampling_abandoned"],
        "conservation_ok": accounting["conservation_ok"],
        "sample_init_generated": sample_init["generated"],
        "sample_init_passed": sample_init["passed"],
        "sample_init_thread_failures": (
            sample_init["verify_failed"]["thread_extent"]
            + sample_init["verify_failed"]["threads_per_block"]
        ),
        "sample_init_shared_failures": sample_init["verify_failed"]["shared_memory"],
        "sample_init_lowering_failures": sample_init["verify_failed"][
            "lowering_exception"
        ],
        "sample_decisions_replaced": sample_metrics["decisions_replaced"],
        "sample_retries": sample_metrics["retries"],
        "h_trace_rejected": hard_positions["trace"]["enforced"],
        "h_postreplay_rejected": hard_positions["postreplay"]["enforced"],
        "verify_gpu_time_s": totals["filter"]["verify_gpu_time_us"] / 1e6,
        "hard_filter_time_s": totals["filter"]["analysis_time_us"] / 1e6,
        "constraint_plan_time_s": sample_metrics["plan_time_us"] / 1e6,
        "constraint_decision_time_s": sample_metrics[
            "decision_sampling_time_us"
        ]
        / 1e6,
        "database_records": result["measurement"]["database_records"],
        "finite_records": result["measurement"]["finite_records"],
        "best_latency_us": result["measurement"]["best_latency_us"],
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for _, argument in ARM_ARGUMENTS:
        parser.add_argument(f"--{argument.replace('_', '-')}", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    paths = {
        arm: getattr(args, argument) for arm, argument in ARM_ARGUMENTS
    }
    results = {arm: _load(path) for arm, path in paths.items()}
    rows = [_row(arm, paths[arm], results[arm]) for arm, _ in ARM_ARGUMENTS]

    dll_hashes = {row["dll_sha256"] for row in rows}
    if len(dll_hashes) != 1:
        raise RuntimeError(f"Experiment arms loaded different TVM DLLs: {dll_hashes}")
    if not all(row["finite_records"] == row["database_records"] for row in rows):
        raise RuntimeError("At least one arm contains a non-finite measurement")
    if not all(row["conservation_ok"] for row in rows):
        raise RuntimeError("At least one arm violates candidate accounting conservation")

    baseline = rows[0]
    for row in rows:
        row["search_speedup_vs_B"] = baseline["search_s"] / row["search_s"]
        row["tune_speedup_vs_B"] = baseline["tune_call_s"] / row["tune_call_s"]
        row["sample_init_pass_rate"] = (
            row["sample_init_passed"] / row["sample_init_generated"]
        )
        row["sample_init_thread_failure_rate"] = (
            row["sample_init_thread_failures"] / row["sample_init_generated"]
        )

    summary = {
        "schema_version": 1,
        "status": "single-seed-smoke",
        "strict_performance_conclusion": False,
        "common_loaded_library_sha256": next(iter(dll_hashes)),
        "rows": rows,
        "observations": {
            "constraint_sampling_removed_sample_init_thread_failures": (
                rows[0]["sample_init_thread_failures"]
                - rows[2]["sample_init_thread_failures"]
            ),
            "constraint_sampling_passed_gain": (
                rows[2]["sample_init_passed"] - rows[0]["sample_init_passed"]
            ),
            "hard_verify_failures_removed": (
                rows[0]["verify_failures"] - rows[1]["verify_failures"]
            ),
            "hard_search_seconds_saved": rows[0]["search_s"] - rows[1]["search_s"],
            "combined_vs_constraint_sampling_search_seconds_saved": (
                rows[2]["search_s"] - rows[3]["search_s"]
            ),
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "smoke_summary.json"
    csv_path = args.output_dir / "smoke_summary.csv"
    manifest_path = args.output_dir / "manifest.sha256.json"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        path.name: _sha256(path)
        for path in sorted(args.output_dir.rglob("*"))
        if path.is_file() and path != manifest_path
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
