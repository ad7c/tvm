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
"""Compare the Pruner draft score with measured tuning-record latency."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def _median(values: list[float]) -> float:
    values = sorted(values)
    middle = len(values) // 2
    if len(values) % 2:
        return values[middle]
    return 0.5 * (values[middle - 1] + values[middle])


def _ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    result = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average_rank = 0.5 * (start + 1 + end)
        for offset in range(start, end):
            result[order[offset]] = average_rank
        start = end
    return result


def _pearson(lhs: list[float], rhs: list[float]) -> float | None:
    if len(lhs) < 2:
        return None
    lhs_mean = sum(lhs) / len(lhs)
    rhs_mean = sum(rhs) / len(rhs)
    numerator = sum((x - lhs_mean) * (y - rhs_mean) for x, y in zip(lhs, rhs))
    lhs_norm = sum((x - lhs_mean) ** 2 for x in lhs)
    rhs_norm = sum((y - rhs_mean) ** 2 for y in rhs)
    denominator = math.sqrt(lhs_norm * rhs_norm)
    return numerator / denominator if denominator else None


def _quality(records: list[dict[str, Any]]) -> dict[str, Any]:
    known = [record for record in records if record["draft_known"]]
    # Parallel reductions can differ at ~1e-13 across processes.  Treat those
    # values as analytical ties instead of giving them an arbitrary rank.
    scores = [round(record["draft_score"], 9) for record in known]
    performance = [1.0 / record["latency_us"] for record in known]
    score_order = sorted(range(len(known)), key=lambda index: scores[index], reverse=True)
    truth_order = sorted(range(len(known)), key=lambda index: known[index]["latency_us"])
    oracle_best = known[truth_order[0]]["latency_us"] if known else None
    max_score = max(scores) if scores else None
    max_score_latencies = [
        record["latency_us"]
        for record, rounded_score in zip(known, scores)
        if rounded_score == max_score
    ]
    top_k = {}
    for requested_k in (1, 3, 5, 8, 16):
        k = min(requested_k, len(known))
        if k == 0:
            continue
        predicted = set(score_order[:k])
        truth = set(truth_order[:k])
        predicted_latencies = [known[index]["latency_us"] for index in predicted]
        best_predicted = min(predicted_latencies)
        top_k[str(requested_k)] = {
            "effective_k": k,
            "overlap": len(predicted & truth),
            "overlap_ratio": len(predicted & truth) / k,
            "best_latency_us": best_predicted,
            "best_regret_vs_oracle": best_predicted / oracle_best - 1.0,
            "mean_latency_us": sum(predicted_latencies) / k,
        }
    return {
        "records": len(records),
        "known": len(known),
        "unknown": len(records) - len(known),
        "spearman_score_vs_inverse_latency": _pearson(_ranks(scores), _ranks(performance)),
        "draft_score_round_decimal_places": 9,
        "unique_draft_scores": len(set(scores)),
        "max_score_tie_count": len(max_score_latencies),
        "max_score_latency_us": {
            "min": min(max_score_latencies) if max_score_latencies else None,
            "median": _median(max_score_latencies) if max_score_latencies else None,
            "max": max(max_score_latencies) if max_score_latencies else None,
        },
        "oracle_best_latency_us": oracle_best,
        "top_k_ties_are_input_order_dependent": True,
        "top_k": top_k,
    }


def _average_duplicate_traces(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(record["trace_sha256"], []).append(record)
    result = []
    for trace_hash, duplicates in grouped.items():
        item = dict(duplicates[0])
        item["trace_sha256"] = trace_hash
        item["latency_us"] = sum(record["latency_us"] for record in duplicates) / len(duplicates)
        item["duplicate_measurements"] = len(duplicates)
        result.append(item)
    return result


def _top_k_overlap(events: list[dict[str, Any]]) -> dict[str, Any]:
    overlap = sum(event["top_k_overlap"] for event in events)
    denominator = sum(event["top_k_denominator"] for event in events)
    return {
        "overlap": overlap,
        "denominator": denominator,
        "ratio": overlap / denominator if denominator else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import tvm  # pylint: disable=import-outside-toplevel
    from tvm.s_tir.meta_schedule.database import (  # pylint: disable=import-outside-toplevel
        JSONDatabase,
    )

    analyze = tvm.get_global_func("s_tir.meta_schedule.testing.AnalyzePrunerDraftScore")
    per_arm = {}
    pooled = []
    identities = set()
    libraries = set()
    for result_path in args.result:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        arm = result["draft_mode"]
        identities.add(result["source_identity_sha256"])
        libraries.add(result["loaded_library_sha256"])
        profile = result["config"]["draft_profile"]
        database = JSONDatabase(work_dir=str(result_path.parent), allow_missing=False)
        analyzed_records = []
        for record in database.get_all_tuning_records():
            values = [float(value) for value in record.run_secs]
            values = [value for value in values if math.isfinite(value) and 0 < value < 1e9]
            if not values:
                continue
            candidate = record.as_measure_candidate()
            known, score, statements = analyze(
                candidate.sch.mod,
                profile["peak_gflops"],
                profile["global_memory_gbps"],
                profile["sm_count"],
                profile["active_blocks_per_sm"],
                profile["sm_partitions"],
                profile.get("warp_size", 32),
                profile["register_capacity"],
            )
            item = {
                "arm": arm,
                "trace_sha256": hashlib.sha256(str(record.trace).encode("utf-8")).hexdigest(),
                "latency_us": _median(values) * 1e6,
                "draft_known": bool(known),
                "draft_score": float(score),
                "draft_statements": int(statements),
            }
            analyzed_records.append(item)
            pooled.append(item)
        analyzed_records.sort(key=lambda item: item["trace_sha256"])
        draft = result["search"]["draft_model"]
        events = draft.get("events", [])
        calls_per_batch = result["config"]["genetic_iters"] + 1
        per_arm[arm] = {
            "runtime": {
                "search_s": result["timing"]["search_s"],
                "tune_call_s": result["timing"]["tune_call_s"],
                "generated": result["search"]["totals"]["accounting"]["generated"],
                "verify_gpu_fail": result["search"]["totals"]["verify_gpu_fail"],
                "verify_failed": result["search"]["totals"]["accounting"]["verify_failed"],
                "finite_records": result["measurement"]["finite_records"],
                "best_latency_us": result["measurement"]["best_latency_us"],
                "batch_summaries": result["measurement"]["batch_summaries"],
                "xgb_data_size": result["measurement"]["xgb_data_size"],
                "xgb_trained": result["measurement"]["xgb_trained"],
                "draft_time_us": draft.get("time_us_total", 0),
                "draft_candidates": draft.get("candidates_total", 0),
                "draft_known_coverage": draft.get("known_coverage"),
                "draft_top_k_all": _top_k_overlap(events),
                "draft_top_k_batch_0": _top_k_overlap(events[:calls_per_batch]),
                "draft_top_k_trained_batch_1": _top_k_overlap(events[calls_per_batch:]),
            },
            "quality": _quality(analyzed_records),
            "records": analyzed_records,
        }

    pooled.sort(key=lambda item: (item["trace_sha256"], item["arm"]))
    comparisons = {}
    if all(arm in per_arm for arm in ("off", "shadow", "rank")):
        off = per_arm["off"]["runtime"]
        shadow = per_arm["shadow"]["runtime"]
        rank = per_arm["rank"]["runtime"]
        trace_sets = {
            arm: {record["trace_sha256"] for record in per_arm[arm]["records"]}
            for arm in ("off", "shadow", "rank")
        }
        comparisons = {
            "off_shadow_trace_overlap": len(trace_sets["off"] & trace_sets["shadow"]),
            "off_rank_trace_overlap": len(trace_sets["off"] & trace_sets["rank"]),
            "shadow_rank_trace_overlap": len(trace_sets["shadow"] & trace_sets["rank"]),
            "rank_search_delta_s_vs_off": rank["search_s"] - off["search_s"],
            "rank_search_speedup_vs_off": off["search_s"] / rank["search_s"],
            "rank_tune_delta_s_vs_off": rank["tune_call_s"] - off["tune_call_s"],
            "rank_tune_speedup_vs_off": off["tune_call_s"] / rank["tune_call_s"],
            "rank_best_latency_ratio_vs_off": rank["best_latency_us"] / off["best_latency_us"],
            "shadow_search_delta_s_vs_off": shadow["search_s"] - off["search_s"],
        }
    output = {
        "schema_version": 1,
        "source_identity_equal": len(identities) == 1,
        "loaded_library_equal": len(libraries) == 1,
        "source_identity_sha256": next(iter(identities)) if len(identities) == 1 else None,
        "loaded_library_sha256": next(iter(libraries)) if len(libraries) == 1 else None,
        "per_arm": per_arm,
        "comparisons": comparisons,
        "pooled_quality": _quality(pooled),
        "pooled_unique_trace_quality": _quality(_average_duplicate_traces(pooled)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({**output, "per_arm": {k: v["quality"] for k, v in per_arm.items()}}, indent=2))


if __name__ == "__main__":
    main()
