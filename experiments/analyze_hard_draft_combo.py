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
"""Summarize the paired hard-filter and Pruner-draft experiment."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import statistics
from pathlib import Path
from typing import Any

from e1_thread_filter import _parse_search_log


ARM_DIRS = {
    "baseline": "v1_1_h2_pair0_seed20260903_off_soft-off_draft-off_ga-off",
    "hard": "v1_1_h2_pair0_seed20260903_enforce_soft-off_draft-off_ga-off",
    "draft": "v1_1_h2_pair0_seed20260903_off_soft-off_draft-rank_ga-off",
    "combined": "v1_1_h2_pair0_seed20260903_enforce_soft-off_draft-rank_ga-off",
}
SHADOW_DIR = "v1_1_h2_pair0_seed20260903_shadow_soft-off_draft-off_ga-off"


def _top_k_mean(values: list[float], k: int) -> float:
    ordered = sorted(values)
    return statistics.fmean(ordered[: min(k, len(ordered))])


def _arm_summary(result: dict[str, Any]) -> dict[str, Any]:
    latencies = [
        float(value)
        for value in result["measurement"]["record_latencies_us"]
        if value is not None
    ]
    accounting = result["search"]["totals"]["accounting"]
    filt = result["search"]["totals"]["filter"]
    draft = {key: value for key, value in result["search"]["draft_model"].items() if key != "events"}
    return {
        "finite_records": len(latencies),
        "best_us": min(latencies),
        "median_us": statistics.median(latencies),
        "mean_us": statistics.fmean(latencies),
        "top3_mean_us": _top_k_mean(latencies, 3),
        "top5_mean_us": _top_k_mean(latencies, 5),
        "batch_summaries": result["measurement"]["batch_summaries"],
        "search_s": result["timing"]["search_s"],
        "tune_call_s": result["timing"]["tune_call_s"],
        "verify_gpu_s": filt["verify_gpu_time_us"] / 1e6,
        "hard_analysis_s": filt["analysis_time_us"] / 1e6,
        "generated": accounting["generated"],
        "verify_failures": result["search"]["totals"]["verify_gpu_fail"],
        "hard_rejected": filt["enforced"],
        "hard_rejected_by_reason": accounting["hard_rejected"],
        "draft_model": draft,
    }


def _comparison(reference: dict[str, Any], variant: dict[str, Any]) -> dict[str, float]:
    return {
        "search_speedup": reference["search_s"] / variant["search_s"],
        "search_change_pct": (variant["search_s"] / reference["search_s"] - 1) * 100,
        "tune_speedup": reference["tune_call_s"] / variant["tune_call_s"],
        "tune_change_pct": (variant["tune_call_s"] / reference["tune_call_s"] - 1) * 100,
        "verify_speedup": reference["verify_gpu_s"] / variant["verify_gpu_s"],
        "verify_change_pct": (variant["verify_gpu_s"] / reference["verify_gpu_s"] - 1) * 100,
        "best_change_pct": (variant["best_us"] / reference["best_us"] - 1) * 100,
        "top3_mean_change_pct":
            (variant["top3_mean_us"] / reference["top3_mean_us"] - 1) * 100,
        "top5_mean_change_pct":
            (variant["top5_mean_us"] / reference["top5_mean_us"] - 1) * 100,
    }


def _overlap(first: list[str], second: list[str]) -> int:
    return sum((Counter(first) & Counter(second)).values())


def analyze(root: Path, audit_path: Path) -> dict[str, Any]:
    results = {
        name: json.loads((root / directory / "result.json").read_text(encoding="utf-8"))
        for name, directory in ARM_DIRS.items()
    }
    arms = {name: _arm_summary(result) for name, result in results.items()}

    shadow_root = root / SHADOW_DIR
    shadow_log = shadow_root / "logs" / "tvm.s_tir.meta_schedule.logging.task_0_main.log"
    shadow_search = _parse_search_log(shadow_log, "shadow")
    shadow_records = sum(
        bool(line.strip())
        for line in (shadow_root / "database_tuning_record.json")
        .read_text(encoding="utf-8")
        .splitlines()
    )

    trace_hashes = {
        name: result["measurement"]["trace_hashes"] for name, result in results.items()
    }
    trace_overlap = {}
    for first, second in (
        ("baseline", "hard"),
        ("draft", "combined"),
        ("baseline", "draft"),
        ("hard", "combined"),
    ):
        trace_overlap[f"{first}_vs_{second}"] = {
            "batch0": _overlap(trace_hashes[first][:8], trace_hashes[second][:8]),
            "batch1": _overlap(trace_hashes[first][8:16], trace_hashes[second][8:16]),
            "all": _overlap(trace_hashes[first], trace_hashes[second]),
            "denominator_per_batch": 8,
            "denominator_all": 16,
        }

    normalized_configs = []
    for result in results.values():
        config = dict(result["config"])
        config.pop("draft_mode")
        normalized_configs.append(config)

    source_identities = {
        name: result["source_identity_sha256"] for name, result in results.items()
    }
    dll_identities = {
        name: result["loaded_library_sha256"] for name, result in results.items()
    }
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    return {
        "experiment": "V1.1 H2 hard-filter-v2 x Pruner-draft pair0",
        "root": str(root),
        "configuration": {
            "seed": results["baseline"]["seed"],
            "trials": results["baseline"]["config"]["trials"],
            "trials_per_iter": results["baseline"]["config"]["trials_per_iter"],
            "population_size": results["baseline"]["config"]["population_size"],
            "genetic_iters": results["baseline"]["config"]["genetic_iters"],
            "xgb_warmup_samples": results["baseline"]["config"]["xgb_warmup_samples"],
            "soft_mode": results["baseline"]["soft_mode"],
            "ga_t2_mode": results["baseline"]["ga_t2_mode"],
            "draft_beta": results["draft"]["draft_beta"],
            "draft_eps": results["draft"]["draft_eps"],
        },
        "identity": {
            "dll_sha256_by_arm": dll_identities,
            "single_dll_identity": len(set(dll_identities.values())) == 1,
            "source_identity_by_arm": source_identities,
            "single_source_identity": len(set(source_identities.values())) == 1,
            "source_identity_note": (
                "Baseline was run immediately before a parser-only regex fix; the loaded DLL and "
                "all search settings are identical. The other three performance arms share one "
                "source identity."
            ),
            "config_equal_except_draft_mode": all(
                config == normalized_configs[0] for config in normalized_configs[1:]
            ),
        },
        "arms": arms,
        "comparisons": {
            "hard_vs_baseline": _comparison(arms["baseline"], arms["hard"]),
            "draft_vs_baseline": _comparison(arms["baseline"], arms["draft"]),
            "combined_vs_baseline": _comparison(arms["baseline"], arms["combined"]),
            "combined_vs_draft": _comparison(arms["draft"], arms["combined"]),
        },
        "shadow_gate": {
            "database_records": shadow_records,
            "search_s": shadow_search["search_s"],
            "filter": shadow_search["totals"]["filter"],
            "verify_failure_reasons": shadow_search["totals"]["accounting"]["verify_failed"],
            "candidate_accounting": shadow_search["totals"]["accounting"],
        },
        "historical_database_audit": {
            "totals": audit["totals"],
            "unique": audit["unique"],
            "classification_conflicts": audit["classification_conflicts"],
            "skipped_databases": audit["skipped_databases"],
        },
        "trace_overlap": trace_overlap,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path("experiments/data/v1_1_h2_combo")
    )
    parser.add_argument(
        "--audit",
        type=Path,
        default=Path("experiments/data/hard_filter_v2_database_audit.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/data/v1_1_h2_combo/hard_draft_combo_pair0.json"),
    )
    args = parser.parse_args()
    result = analyze(args.root, args.audit)
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
