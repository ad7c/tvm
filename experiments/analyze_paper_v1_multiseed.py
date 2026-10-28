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
"""Summarize paired paper-v1 baseline and CS+H result JSON files."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


METRICS = (
    "generated",
    "passed",
    "verify_failures",
    "best_latency_us",
    "search_s",
    "builder_s",
    "runner_s",
    "tune_s",
    "verify_s",
    "verify_candidates",
    "valid_yield",
)


def _load(path: Path, expected_arm: str) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    mode = result["mode"]
    sample_mode = result["constraint_sample_mode"]
    if expected_arm == "baseline" and (mode != "off" or sample_mode != "off"):
        raise ValueError(f"{path}: expected baseline off/off, got {mode}/{sample_mode}")
    if expected_arm == "csh" and (mode != "enforce" or sample_mode != "enforce"):
        raise ValueError(f"{path}: expected CSH enforce/enforce, got {mode}/{sample_mode}")

    accounting = result["search"]["totals"]["accounting"]
    filt = result["search"]["totals"]["filter"]
    timing = result["timing"]
    generated = int(accounting["generated"])
    passed = int(accounting["passed"])
    verify_failures = int(result["search"]["totals"]["verify_gpu_fail"])
    verify_candidates = passed + verify_failures
    reasons = {key: int(value) for key, value in accounting["verify_failed"].items()}
    return {
        "path": str(path),
        "seed": int(result["seed"]),
        "workload": result["workload"],
        "loaded_library_sha256": result["loaded_library_sha256"],
        "source_identity_sha256": result["source_identity_sha256"],
        "config_identity": {
            key: result["config"][key]
            for key in (
                "trials",
                "trials_per_iter",
                "population_size",
                "init_min_unmeasured",
                "genetic_iters",
                "build_timeout_s",
                "run_timeout_s",
            )
        },
        "generated": generated,
        "passed": passed,
        "verify_failures": verify_failures,
        "best_latency_us": float(result["measurement"]["best_latency_us"]),
        "finite_records": int(result["measurement"]["finite_records"]),
        "search_s": float(timing["search_s"]),
        "builder_s": float(timing["builder_s"]),
        "runner_s": float(timing["runner_and_update_s"]),
        "tune_s": float(timing["tune_call_s"]),
        "verify_s": float(filt["verify_gpu_time_us"]) / 1.0e6,
        "verify_candidates": verify_candidates,
        "verify_ms_per_candidate": (
            float(filt["verify_gpu_time_us"]) / 1000.0 / verify_candidates
            if verify_candidates
            else None
        ),
        "valid_yield": passed / generated if generated else None,
        "hard_checked": int(filt["checked"]),
        "hard_predicted": int(filt["predicted"]),
        "hard_unknown": int(filt["unknown"]),
        "verify_failure_reasons": reasons,
    }


def _mean_std(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    values = [float(row[key]) for row in rows]
    return {
        "mean": statistics.fmean(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, action="append", required=True)
    parser.add_argument("--csh", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    baseline_rows = [_load(path, "baseline") for path in args.baseline]
    csh_rows = [_load(path, "csh") for path in args.csh]
    baseline = {row["seed"]: row for row in baseline_rows}
    csh = {row["seed"]: row for row in csh_rows}
    if baseline.keys() != csh.keys():
        raise ValueError(f"seed mismatch: baseline={sorted(baseline)}, csh={sorted(csh)}")

    pairs = []
    for seed in sorted(baseline):
        reference = baseline[seed]
        variant = csh[seed]
        for key in ("workload", "loaded_library_sha256", "source_identity_sha256", "config_identity"):
            if reference[key] != variant[key]:
                raise ValueError(f"seed {seed}: paired {key} differs")
        deltas = {
            key: (float(variant[key]) / float(reference[key]) - 1.0) * 100.0
            for key in METRICS
            if float(reference[key]) != 0.0
        }
        pairs.append({"seed": seed, "baseline": reference, "csh": variant, "csh_change_pct": deltas})

    baseline_rows = [pair["baseline"] for pair in pairs]
    csh_rows = [pair["csh"] for pair in pairs]
    summary = {
        "baseline": {key: _mean_std(baseline_rows, key) for key in METRICS},
        "csh": {key: _mean_std(csh_rows, key) for key in METRICS},
    }
    summary["csh_change_of_means_pct"] = {
        key: (summary["csh"][key]["mean"] / summary["baseline"][key]["mean"] - 1.0) * 100.0
        for key in METRICS
        if summary["baseline"][key]["mean"] != 0.0
    }
    summary["verify_failure_reason_totals"] = {
        arm: {
            reason: sum(row["verify_failure_reasons"].get(reason, 0) for row in rows)
            for reason in sorted(
                set().union(*(row["verify_failure_reasons"] for row in rows))
            )
        }
        for arm, rows in (("baseline", baseline_rows), ("csh", csh_rows))
    }

    output = {
        "schema_version": 1,
        "experiment": "paper-v1 current-version MatMul128 paired multi-seed pilot",
        "seeds": sorted(baseline),
        "pair_count": len(pairs),
        "all_pairs_same_library": len(
            {row["loaded_library_sha256"] for row in baseline_rows + csh_rows}
        )
        == 1,
        "all_pairs_same_source_identity": len(
            {row["source_identity_sha256"] for row in baseline_rows + csh_rows}
        )
        == 1,
        "pairs": pairs,
        "summary": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output["summary"], indent=2))


if __name__ == "__main__":
    main()
