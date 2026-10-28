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
"""Reproducible MatMul-128 experiment for the first hard thread filter.

Each pruning arm runs in a fresh Python process so that its environment and
output directory are isolated.  The hard-filter modes are:

* off: classify nothing
* shadow: classify candidates but never reject
* enforce: reject candidates proven thread-illegal

``--constraint-sample-mode`` independently controls constraint-aware
initialization sampling, while ``--hard-filter-position`` selects H-late,
H-postreplay, or H-trace.  Soft filtering, the legacy post-XGBoost draft
ranker, Draft-guided GA, trace-only G1 direction, G2 projection, and
Runner-measured G3 mutator credit remain independently controllable and default to off.  In
``--draft-ga-mode enforce`` the analytic Draft score is the only GA fitness;
the learned cost model is called exactly once per
search batch, after GA, to rank the cross-generation Draft archive.  The
driver validates that contract from the C++ log before writing a successful
result.

The default smoke run uses one tuning batch.  Multi-batch runs are supported so
that later batches can exercise a cost model trained by earlier measurements.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import time
import shutil
import traceback
from datetime import datetime
from functools import partial
from typing import Any

from p3_observability import (
    CORRECTNESS_CONFIG, TuneJournal, append_json, checked_matmul_evaluator,
    classify_measurements, gpu_status, summarize_timeline, write_json,
)
from clean_workloads import SPECS as CLEAN_WORKLOADS, make_workload, checked_clean_evaluator


MODE_TO_VALUE = {"off": 0, "shadow": 1, "enforce": 2}
HARD_FILTER_POSITIONS = ("late", "postreplay", "trace")
SOFT_MODE_TO_VALUE = {"off": 0, "shadow": 1, "rerank": 2}
GA_T2_MODE_TO_VALUE = {"off": 0, "shadow": 1, "project": 2, "directed": 3}
DRAFT_MODE_TO_VALUE = {"off": 0, "shadow": 1, "rank": 2}
DRAFT_GA_MODE_TO_VALUE = {"off": 0, "shadow": 1, "enforce": 2}
DRAFT_GA_G3_MODE_TO_VALUE = {"off": 0, "shadow": 1, "enforce": 2}
DRAFT_GA_ANNEAL_MODE_TO_VALUE = {"off": 0, "shadow": 1, "enforce": 2}
G3_FIXED_CONFIG = {"window_size": 50, "cold_start_measurements": 32, "exploration_scale": 1.0}
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORK_ROOT = Path("F:/tvm_tmp/e1_thread_filter")
TIMESTAMP_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
VERIFY_REASONS = (
    "thread_extent",
    "threads_per_block",
    "shared_memory",
    "local_memory",
    "vthread",
    "vector_bytes",
    "kernel_count",
    "lowering_exception",
    "unknown_verify_failure",
)


def _timestamp(line: str) -> datetime | None:
    match = TIMESTAMP_RE.match(line)
    if match is None:
        return None
    return datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S")


def _first_int(text: str, patterns: list[str]) -> int | None:
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            value = matches[-1]
            if isinstance(value, tuple):
                value = value[-1]
            return int(value)
    return None


def _filter_metrics(text: str) -> dict[str, int | None]:
    """Parse both individual and compact TP/FP/FN/TN log formats."""
    metrics: dict[str, int | None] = {
        "checked": _first_int(
            text,
            [r"Thread filter checked\s*[:=]\s*(\d+)", r"Thread filter.*?checked\s+(\d+)"],
        ),
        "unknown": _first_int(
            text,
            [r"Thread filter unknown\s*[:=]\s*(\d+)", r"Thread filter.*?unknown\s+(\d+)"],
        ),
        "predicted": _first_int(
            text,
            [
                r"Thread filter predicted(?: rejections?)?\s*[:=]\s*(\d+)",
                r"Thread filter.*?predicted(?: rejections?)?\s+(\d+)",
            ],
        ),
        "enforced": _first_int(
            text,
            [
                r"Thread filter enforced(?: rejections?)?\s*[:=]\s*(\d+)",
                r"Thread filter.*?enforced(?: rejections?)?\s+(\d+)",
            ],
        ),
        "analysis_time_us": _first_int(
            text, [r"Thread filter analysis time \(us\)\s*[:=]\s*(\d+)"]
        ),
        "verify_gpu_time_us": _first_int(
            text, [r"VerifyGPUCode time \(us\)\s*[:=]\s*(\d+)"]
        ),
        "tp": _first_int(text, [r"(?:Thread filter\s+)?(?:shadow\s+)?TP\s*[:=]\s*(\d+)"]),
        "fp": _first_int(text, [r"(?:Thread filter\s+)?(?:shadow\s+)?FP\s*[:=]\s*(\d+)"]),
        "fn": _first_int(text, [r"(?:Thread filter\s+)?(?:shadow\s+)?FN\s*[:=]\s*(\d+)"]),
        "tn": _first_int(text, [r"(?:Thread filter\s+)?(?:shadow\s+)?TN\s*[:=]\s*(\d+)"]),
        "predicted_low": None,
        "predicted_high": None,
        "predicted_hardware": None,
        "shared_checked": _first_int(
            text, [r"Shared-memory pre-compaction proxy checked\s*[:=]\s*(\d+)"]
        ),
        "shared_unknown": _first_int(
            text, [r"Shared-memory pre-compaction proxy unknown\s*[:=]\s*(\d+)"]
        ),
        "shared_predicted": _first_int(
            text,
            [r"Shared-memory pre-compaction proxy predicted rejections\s*[:=]\s*(\d+)"],
        ),
        "shared_tp": None,
        "shared_fp": None,
        "shared_fn": None,
        "shared_tn": None,
    }
    compact = re.findall(
        r"^\s*Thread filter shadow TP/FP/FN/TN\s*[:=]\s*"
        r"(\d+)\s*[/,]\s*(\d+)\s*[/,]\s*(\d+)\s*[/,]\s*(\d+)",
        text,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    if compact:
        metrics["tp"], metrics["fp"], metrics["fn"], metrics["tn"] = map(int, compact[-1])
    reasons = re.findall(
        r"Hard filter predicted by reason low/high/hardware\s*[:=]\s*"
        r"(\d+)\s*[/,]\s*(\d+)\s*[/,]\s*(\d+)",
        text,
        flags=re.IGNORECASE,
    )
    if reasons:
        (
            metrics["predicted_low"],
            metrics["predicted_high"],
            metrics["predicted_hardware"],
        ) = map(int, reasons[-1])
    shared_compact = re.findall(
        r"^\s*Shared-memory pre-compaction proxy TP/FP/FN/TN\s*[:=]\s*"
        r"(\d+)\s*[/,]\s*(\d+)\s*[/,]\s*(\d+)\s*[/,]\s*(\d+)",
        text,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    if shared_compact:
        (
            metrics["shared_tp"],
            metrics["shared_fp"],
            metrics["shared_fn"],
            metrics["shared_tn"],
        ) = map(int, shared_compact[-1])
    return metrics


def _stage_accounting(text: str) -> dict[str, Any]:
    """Parse the machine-readable candidate accounting for one search stage."""

    def scalar(name: str) -> int:
        matches = re.findall(rf"{re.escape(name)}\s*:\s*(-?\d+)", text)
        if not matches:
            raise RuntimeError(f"Missing {name} in evolutionary-search summary")
        return int(matches[-1])

    postproc_failed = {
        int(index): int(value)
        for index, value in re.findall(r"stage\.postproc_failed\[(\d+)\]\s*:\s*(\d+)", text)
    }
    verify_failed = {
        reason: int(value)
        for reason, value in re.findall(r"stage\.verify_failed\[([^\]]+)\]\s*:\s*(\d+)", text)
    }
    hard_rejected = {
        reason: int(value)
        for reason, value in re.findall(r"stage\.hard_rejected\[([^\]]+)\]\s*:\s*(\d+)", text)
    }
    missing_reasons = [reason for reason in VERIFY_REASONS if reason not in verify_failed]
    if missing_reasons:
        raise RuntimeError(f"Missing VerifyGPUCode reason counters: {missing_reasons}")

    generated = scalar("stage.generated")
    trace_replay_failed = scalar("stage.trace_replay_failed")
    constraint_sampling_abandoned = scalar("stage.constraint_sampling_abandoned")
    passed = scalar("stage.passed")
    accounted = scalar("stage.accounted")
    conservation_delta = scalar("stage.conservation_delta")
    calculated_accounted = (
        passed
        + trace_replay_failed
        + constraint_sampling_abandoned
        + sum(postproc_failed.values())
        + sum(hard_rejected.values())
    )
    return {
        "generated": generated,
        "trace_replay_failed": trace_replay_failed,
        "constraint_sampling_abandoned": constraint_sampling_abandoned,
        "postproc_failed": {str(key): value for key, value in sorted(postproc_failed.items())},
        "verify_failed": verify_failed,
        "verify_failure_total": sum(verify_failed.values()),
        "hard_rejected": hard_rejected,
        "passed": passed,
        "accounted": accounted,
        "calculated_accounted": calculated_accounted,
        "conservation_delta": conservation_delta,
        "conservation_ok": (
            generated == accounted == calculated_accounted and conservation_delta == 0
        ),
    }


def _aggregate_accounting(blocks: list[dict[str, Any]]) -> dict[str, Any]:
    """Sum candidate accounting without losing indexed/reasoned failure buckets."""

    scalar_names = (
        "generated",
        "trace_replay_failed",
        "constraint_sampling_abandoned",
        "verify_failure_total",
        "passed",
        "accounted",
        "calculated_accounted",
        "conservation_delta",
    )
    result: dict[str, Any] = {
        name: sum(block["accounting"][name] for block in blocks) for name in scalar_names
    }
    for field in ("postproc_failed", "verify_failed", "hard_rejected"):
        keys = sorted({key for block in blocks for key in block["accounting"][field]})
        result[field] = {
            key: sum(block["accounting"][field].get(key, 0) for block in blocks) for key in keys
        }
    result["conservation_ok"] = all(
        block["accounting"]["conservation_ok"] for block in blocks
    )
    return result


def _constraint_sampling_metrics(text: str) -> dict[str, Any]:
    """Parse initialization constraint-plan and constrained-sampling counters."""

    names = {
        "plans_built": "Constraint plans built",
        "contract_blocks": "Constraint plan contract blocks",
        "enforceable_constraints": "Constraint plan enforceable constraints",
        "decisions_handled": "Constraint sampling decisions handled",
        "decisions_replaced": "Constraint sampling decisions replaced",
        "decision_unknown": "Constraint sampling decision unknown",
        "feasible_domain_total": "Constraint sampling feasible domain total",
        "empty_domains": "Constraint sampling empty domains",
        "retries": "Constraint sampling retries",
        "abandoned": "Constraint sampling abandoned",
        "candidates_checked": "Constraint sampling candidates checked",
        "candidates_unknown": "Constraint sampling candidates unknown",
        "predicted_rejections": "Constraint sampling predicted rejections",
        "plan_time_us": "Constraint plan time (us)",
        "decision_sampling_time_us": "Constraint decision sampling time (us)",
        "check_time_us": "Constraint sampling check time (us)",
        "trace_replay_time_us": "Trace replay time (us)",
    }
    result: dict[str, Any] = {
        name: _first_int(text, [rf"{re.escape(label)}\s*[:=]\s*(\d+)"])
        for name, label in names.items()
    }
    mode = re.findall(r"Constraint sampling mode\s*[:=]\s*(\w+)", text)
    result["mode"] = mode[-1].lower() if mode else None
    compact = re.findall(
        r"Constraint sampling shadow TP/FP/FN/TN \(thread-only oracle\)\s*[:=]\s*"
        r"(\d+)\s*[/,]\s*(\d+)\s*[/,]\s*(\d+)\s*[/,]\s*(\d+)",
        text,
        flags=re.IGNORECASE,
    )
    if compact:
        result["tp"], result["fp"], result["fn"], result["tn"] = map(
            int, compact[-1]
        )
    else:
        result.update({"tp": None, "fp": None, "fn": None, "tn": None})
    return result


def _hard_position_metrics(text: str) -> dict[str, dict[str, int | None]]:
    """Parse checked/unknown/predicted/enforced and time for each H position."""

    result: dict[str, dict[str, int | None]] = {}
    labels = {"trace": "H-trace", "postreplay": "H-postreplay", "late": "H-late"}
    for position, label in labels.items():
        compact = re.findall(
            rf"{re.escape(label)} checked/unknown/predicted/enforced\s*[:=]\s*"
            r"(\d+)\s*[/,]\s*(\d+)\s*[/,]\s*(\d+)\s*[/,]\s*(\d+)",
            text,
            flags=re.IGNORECASE,
        )
        values = tuple(map(int, compact[-1])) if compact else (None, None, None, None)
        result[position] = {
            "checked": values[0],
            "unknown": values[1],
            "predicted": values[2],
            "enforced": values[3],
            "time_us": _first_int(
                text, [rf"{re.escape(label)} time \(us\)\s*[:=]\s*(\d+)"]
            ),
        }
    return result


def _postproc_times(text: str) -> dict[str, int]:
    return {
        index: int(value)
        for index, value in re.findall(
            r"Postproc #(\d+) \[[^\]]+\]\s*:\s*\d+ failure\(s\), time \(us\):\s*(\d+)",
            text,
        )
    }


def _parse_summary_blocks(task_text: str) -> list[dict[str, Any]]:
    """Extract failure/filter counters from each evolutionary-search summary."""
    blocks: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    generation = -1

    def finish() -> None:
        nonlocal current
        if current is None:
            return
        text = "\n".join(current.pop("lines"))
        current["verify_gpu_fail"] = _first_int(
            text,
            [r"Postproc #\d+ \[[^\]]*VerifyGPUCode[^\]]*\]\s*:\s*(\d+) failure"],
        )
        current["trace_fail"] = _first_int(text, [r"Trace replay failures:\s*(\d+)"])
        current["filter"] = _filter_metrics(text)
        current["constraint_sampling"] = _constraint_sampling_metrics(text)
        current["hard_positions"] = _hard_position_metrics(text)
        current["postproc_time_us"] = _postproc_times(text)
        current["accounting"] = _stage_accounting(text)
        if current["verify_gpu_fail"] != current["accounting"]["verify_failure_total"]:
            raise RuntimeError(
                "VerifyGPUCode total does not match reason buckets: "
                f"total={current['verify_gpu_fail']} "
                f"reasons={current['accounting']['verify_failed']}"
            )
        blocks.append(current)
        current = None

    for line in task_text.splitlines():
        if "Generating candidates" in line:
            finish()
            generation += 1
            continue
        header: tuple[str, int | None] | None = None
        if "Pick-Best-From-Database summary:" in line:
            header = ("pick_best", None)
        elif "Sample-Init-Population summary:" in line:
            header = ("sample_init", None)
        else:
            match = re.search(r"Evolve iter #(\d+) done\. Summary:", line)
            if match:
                header = ("evolve", int(match.group(1)))
        if header is not None:
            finish()
            current = {
                "generation": generation,
                "stage": header[0],
                "iteration": header[1],
                "lines": [],
            }
            continue
        if current is not None:
            # Counter lines do not have timestamps.  The next timestamp starts
            # another log event and therefore closes the current summary.
            if _timestamp(line) is not None:
                finish()
            else:
                current["lines"].append(line)
    finish()
    return blocks


def _effective_blocks(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Avoid double counting cumulative SampleInitPopulation retry summaries."""
    effective: list[dict[str, Any]] = []
    generations = sorted({block["generation"] for block in blocks})
    for generation in generations:
        generation_blocks = [block for block in blocks if block["generation"] == generation]
        for stage in ("pick_best", "evolve"):
            effective.extend(block for block in generation_blocks if block["stage"] == stage)
        sample_blocks = [block for block in generation_blocks if block["stage"] == "sample_init"]
        if sample_blocks:
            # One ThreadedTraceApply instance is reused across retries, so the
            # last SampleInit summary already contains all previous retries.
            effective.append(sample_blocks[-1])
    return effective


def _sum_optional(blocks: list[dict[str, Any]], path: tuple[str, ...]) -> int | None:
    values: list[int] = []
    for block in blocks:
        value: Any = block
        for key in path:
            value = value[key]
        if value is not None:
            values.append(int(value))
    return sum(values) if values else None


def _parse_soft_filter_log(text: str) -> dict[str, Any]:
    pattern = re.compile(
        r"Thread soft filter: mode=(\w+) profile=(\S+) "
        r"candidates=(\d+)->(\d+) known=(\d+) unknown=(\d+) weight=([0-9.]+) "
        r"penalty\[min/p50/mean/max\]=([0-9.eE+-]+)/([0-9.eE+-]+)/"
        r"([0-9.eE+-]+)/([0-9.eE+-]+) top(\d+)_overlap=(\d+)/(\d+) "
        r"time_us=(\d+)"
    )
    events = []
    for match in pattern.finditer(text):
        events.append(
            {
                "mode": match.group(1),
                "profile": match.group(2),
                "candidates_in": int(match.group(3)),
                "candidates_out": int(match.group(4)),
                "known": int(match.group(5)),
                "unknown": int(match.group(6)),
                "weight": float(match.group(7)),
                "penalty_min": float(match.group(8)),
                "penalty_p50": float(match.group(9)),
                "penalty_mean": float(match.group(10)),
                "penalty_max": float(match.group(11)),
                "top_k": int(match.group(12)),
                "top_k_overlap": int(match.group(13)),
                "top_k_denominator": int(match.group(14)),
                "time_us": int(match.group(15)),
            }
        )
    if not events:
        return {"calls": 0, "events": []}
    overlap = sum(event["top_k_overlap"] for event in events)
    denominator = sum(event["top_k_denominator"] for event in events)
    return {
        "calls": len(events),
        "candidate_count_preserved": all(
            event["candidates_in"] == event["candidates_out"] for event in events
        ),
        "candidates_total": sum(event["candidates_in"] for event in events),
        "known_total": sum(event["known"] for event in events),
        "unknown_total": sum(event["unknown"] for event in events),
        "top_k_overlap": overlap,
        "top_k_denominator": denominator,
        "top_k_overlap_ratio": overlap / denominator if denominator else None,
        "time_us_total": sum(event["time_us"] for event in events),
        "events": events,
    }


def _parse_draft_model_log(text: str) -> dict[str, Any]:
    number = r"[0-9.eE+-]+"
    pattern = re.compile(
        rf"Pruner draft model: mode=(\w+) profile=(\S+) "
        rf"candidates=(\d+)->(\d+) known=(\d+) unknown=(\d+) statements=(\d+) "
        rf"beta=({number}) epsilon=({number}) "
        rf"score\[min/p50/mean/max\]=({number})/({number})/({number})/({number}) "
        rf"normalized\[min/p50/mean/max\]=({number})/({number})/({number})/({number}) "
        rf"top(\d+)_overlap=(\d+)/(\d+) time_us=(\d+)"
    )
    events = []
    for match in pattern.finditer(text):
        events.append(
            {
                "mode": match.group(1),
                "profile": match.group(2),
                "candidates_in": int(match.group(3)),
                "candidates_out": int(match.group(4)),
                "known": int(match.group(5)),
                "unknown": int(match.group(6)),
                "statements": int(match.group(7)),
                "beta": float(match.group(8)),
                "epsilon": float(match.group(9)),
                "score_min": float(match.group(10)),
                "score_p50": float(match.group(11)),
                "score_mean": float(match.group(12)),
                "score_max": float(match.group(13)),
                "normalized_min": float(match.group(14)),
                "normalized_p50": float(match.group(15)),
                "normalized_mean": float(match.group(16)),
                "normalized_max": float(match.group(17)),
                "top_k": int(match.group(18)),
                "top_k_overlap": int(match.group(19)),
                "top_k_denominator": int(match.group(20)),
                "time_us": int(match.group(21)),
            }
        )
    if not events:
        return {"calls": 0, "events": []}
    top_k_overlap = sum(event["top_k_overlap"] for event in events)
    top_k_denominator = sum(event["top_k_denominator"] for event in events)
    candidates = sum(event["candidates_in"] for event in events)
    known = sum(event["known"] for event in events)
    return {
        "calls": len(events),
        "candidate_count_preserved": all(
            event["candidates_in"] == event["candidates_out"] for event in events
        ),
        "candidates_total": candidates,
        "known_total": known,
        "unknown_total": sum(event["unknown"] for event in events),
        "known_coverage": known / candidates if candidates else None,
        "statements_total": sum(event["statements"] for event in events),
        "top_k_overlap": top_k_overlap,
        "top_k_denominator": top_k_denominator,
        "top_k_overlap_ratio": (
            top_k_overlap / top_k_denominator if top_k_denominator else None
        ),
        "time_us_total": sum(event["time_us"] for event in events),
        "events": events,
    }


def _parse_draft_ga_log(text: str) -> dict[str, Any]:
    generation_pattern = re.compile(
        r"Draft GA generation: iter=(\d+) mode=(\w+) profile=(\S+) "
        r"candidates=(\d+) known=(\d+) unknown=(\d+) statements=(\d+) "
        r"archive=(\d+)/(\d+) ga_xgb_calls=(\d+) draft_time_us=(\d+)"
    )
    summary_pattern = re.compile(
        r"Evolutionary scoring summary: mode=([\w-]+) ga_generations=(\d+) "
        r"ga_xgb_calls=(\d+) ga_xgb_input=(\d+) final_xgb_calls=(\d+) "
        r"final_xgb_input=(\d+) draft_archive=(\d+) verify_top_k=(\d+) "
        r"returned=(\d+) draft_time_us=(\d+) ga_xgb_time_us=(\d+) "
        r"final_xgb_time_us=(\d+)"
    )
    generations = [
        {
            "iter": int(match.group(1)),
            "mode": match.group(2),
            "profile": match.group(3),
            "candidates": int(match.group(4)),
            "known": int(match.group(5)),
            "unknown": int(match.group(6)),
            "statements": int(match.group(7)),
            "archive": int(match.group(8)),
            "verify_top_k": int(match.group(9)),
            "ga_xgb_calls_so_far": int(match.group(10)),
            "draft_time_us": int(match.group(11)),
        }
        for match in generation_pattern.finditer(text)
    ]
    summaries = [
        {
            "mode": match.group(1),
            "ga_generations": int(match.group(2)),
            "ga_xgb_calls": int(match.group(3)),
            "ga_xgb_input": int(match.group(4)),
            "final_xgb_calls": int(match.group(5)),
            "final_xgb_input": int(match.group(6)),
            "draft_archive": int(match.group(7)),
            "verify_top_k": int(match.group(8)),
            "returned": int(match.group(9)),
            "draft_time_us": int(match.group(10)),
            "ga_xgb_time_us": int(match.group(11)),
            "final_xgb_time_us": int(match.group(12)),
        }
        for match in summary_pattern.finditer(text)
    ]
    return {
        "generation_calls": len(generations),
        "summaries": len(summaries),
        "candidates_total": sum(event["candidates"] for event in generations),
        "known_total": sum(event["known"] for event in generations),
        "unknown_total": sum(event["unknown"] for event in generations),
        "ga_xgb_calls": sum(event["ga_xgb_calls"] for event in summaries),
        "ga_xgb_input": sum(event["ga_xgb_input"] for event in summaries),
        "final_xgb_calls": sum(event["final_xgb_calls"] for event in summaries),
        "final_xgb_input": sum(event["final_xgb_input"] for event in summaries),
        "draft_time_us": sum(event["draft_time_us"] for event in summaries),
        "ga_xgb_time_us": sum(event["ga_xgb_time_us"] for event in summaries),
        "final_xgb_time_us": sum(event["final_xgb_time_us"] for event in summaries),
        "generations": generations,
        "summary_events": summaries,
    }


def _parse_draft_direction_log(text: str) -> dict[str, Any]:
    """Parse Draft-directed mutation summaries without coupling to field order."""
    prefix = "Draft GA direction summary:"
    integer_fields = (
        "trials",
        "proposal_total",
        "replay_pass",
        "known",
        "unknown",
        "directed_choices",
        "random_choices",
        "fallback_to_parent",
        "draft_time_us",
    )
    events: list[dict[str, Any]] = []
    for line in text.splitlines():
        if prefix not in line:
            continue
        payload = line.split(prefix, maxsplit=1)[1]
        fields = dict(re.findall(r"([A-Za-z_][A-Za-z0-9_]*)=([^\s]+)", payload))
        event: dict[str, Any] = {
            "mode": fields.pop("mode", None),
            "epsilon": float(fields.pop("epsilon")) if "epsilon" in fields else None,
        }
        for name in integer_fields:
            event[name] = int(fields.pop(name)) if name in fields else None
        # Preserve future counters so an older driver does not silently hide
        # evidence emitted by a newer implementation.
        event["extra_fields"] = fields
        events.append(event)

    totals = {
        name: sum(int(event[name]) for event in events if event[name] is not None)
        for name in integer_fields
        if name != "trials"
    }
    return {"calls": len(events), "totals": totals, "events": events}


def _parse_draft_g3_log(text: str) -> dict[str, Any]:
    """Parse Runner feedback plus embedded or standalone per-mutator telemetry."""

    def scalar_fields(payload: str) -> dict[str, str]:
        return dict(re.findall(r"([A-Za-z_][A-Za-z0-9_]*)=([^\s,\]}]+)", payload))

    def optional_int(fields: dict[str, str], name: str) -> int | None:
        value = fields.get(name)
        return int(value) if value is not None and re.fullmatch(r"-?\d+", value) else None

    def optional_float(fields: dict[str, str], name: str) -> float | str | None:
        value = fields.get(name)
        if value is None:
            return None
        parsed = float(value)
        return parsed if math.isfinite(parsed) else value.lower()

    def arm_counts(payload: str, name: str) -> dict[int, int]:
        match = re.search(rf"(?:^|\s){re.escape(name)}=\[([^\]]*)\]", payload)
        if match is None:
            return {}
        counts: dict[int, int] = {}
        for index, count in re.findall(r"(-?\d+)\s*:\s*(-?\d+)", match.group(1)):
            counts[int(index)] = int(count)
        return counts

    base_prob_by_arm: dict[int, tuple[str, float]] = {}
    feedback_events: list[dict[str, Any]] = []
    arm_events: list[dict[str, Any]] = []
    generation_events: list[dict[str, Any]] = []
    for line in text.splitlines():
        if "Draft GA G3 arms:" in line:
            for index, type_key, probability in re.findall(
                r"(\d+):([^,@\]]+)@([0-9.eE+-]+)", line
            ):
                base_prob_by_arm[int(index)] = (type_key, float(probability))
            continue
        if "Draft GA G3 generation:" in line:
            payload = line.split("Draft GA G3 generation:", maxsplit=1)[1]
            header = payload.split(" selected=[", maxsplit=1)[0]
            fields = scalar_fields(header)
            generation_events.append(
                {
                    "batch": optional_int(fields, "batch"),
                    "iter": optional_int(fields, "iter"),
                    "mode": fields.get("mode"),
                    "cold_start": optional_int(fields, "cold_start"),
                    "selected": arm_counts(payload, "selected"),
                    "shadow_selected": arm_counts(payload, "shadow_selected"),
                    "accepted": arm_counts(payload, "accepted"),
                }
            )
            continue
        if "Draft GA G3 feedback:" in line:
            payload = line.split("Draft GA G3 feedback:", maxsplit=1)[1]
            header = payload.split(" arms=[", maxsplit=1)[0]
            fields = scalar_fields(header)
            event: dict[str, Any] = {
                "mode": fields.get("mode"),
                "source": fields.get("source"),
                "cold_start": optional_int(fields, "cold_start"),
                "total_outcomes": optional_int(fields, "total_outcomes"),
            }
            for name in (
                "batch",
                "sent",
                "child_measured",
                "with_lineage",
                "with_measured_parent",
                "credited",
                "improved",
            ):
                event[name] = optional_int(fields, name)
            event["skip"] = {
                name: int(value)
                for name, value in fields.items()
                if (name.startswith("skip_") or name.startswith("skipped_"))
                and re.fullmatch(r"-?\d+", value)
            }
            feedback_events.append(event)

            for index_text, arm_payload in re.findall(r"(\d+):\{([^}]*)\}", payload):
                index = int(index_text)
                arm_fields = scalar_fields(arm_payload)
                base_type, base_prob = base_prob_by_arm.get(index, (None, None))
                arm_events.append(
                    {
                        "batch": event["batch"],
                        "mode": event["mode"],
                        "index": index,
                        "type": arm_fields.get("type", base_type),
                        "base_prob": optional_float(arm_fields, "base_prob") or base_prob,
                        "selected": optional_int(arm_fields, "selected"),
                        "selected_total": optional_int(arm_fields, "selected_total"),
                        "selected_batch": optional_int(arm_fields, "selected_batch"),
                        "shadow_selected": optional_int(arm_fields, "shadow_selected"),
                        "accepted": optional_int(arm_fields, "accepted"),
                        "accepted_total": optional_int(arm_fields, "accepted_total"),
                        "credited": optional_int(arm_fields, "credited"),
                        "improved_batch": optional_int(arm_fields, "improved_batch"),
                        "window_count": (
                            optional_int(arm_fields, "window_count")
                            if "window_count" in arm_fields
                            else optional_int(
                                arm_fields,
                                (
                                    "window_outcomes"
                                    if "window_outcomes" in arm_fields
                                    else "outcomes"
                                ),
                            )
                        ),
                        "window_improved": optional_int(
                            arm_fields,
                            "window_improved" if "window_improved" in arm_fields else "improved",
                        ),
                        "mean_reward": (
                            optional_float(arm_fields, "mean_reward")
                            if "mean_reward" in arm_fields
                            else optional_float(arm_fields, "mean")
                        ),
                        "ucb": optional_float(arm_fields, "ucb"),
                    }
                )
            continue
        if "Draft GA G3 arm:" in line:
            fields = scalar_fields(line.split("Draft GA G3 arm:", maxsplit=1)[1])
            index = optional_int(fields, "index")
            base_type, base_prob = base_prob_by_arm.get(index, (None, None))
            arm_events.append(
                {
                    "batch": optional_int(fields, "batch"),
                    "mode": fields.get("mode"),
                    "index": index,
                    "type": fields.get("type", base_type),
                    "base_prob": optional_float(fields, "base_prob") or base_prob,
                    "selected": optional_int(fields, "selected"),
                    "selected_total": optional_int(fields, "selected_total"),
                    "selected_batch": optional_int(fields, "selected_batch"),
                    "shadow_selected": optional_int(fields, "shadow_selected"),
                    "accepted": optional_int(fields, "accepted"),
                    "accepted_total": optional_int(fields, "accepted_total"),
                    "credited": optional_int(fields, "credited"),
                    "improved_batch": optional_int(fields, "improved_batch"),
                    "window_count": optional_int(fields, "window_count"),
                    "window_improved": optional_int(fields, "window_improved"),
                    "mean_reward": optional_float(fields, "mean_reward"),
                    "ucb": optional_float(fields, "ucb"),
                }
            )

    feedback_fields = (
        "sent",
        "child_measured",
        "with_lineage",
        "with_measured_parent",
        "credited",
        "improved",
    )
    feedback_totals: dict[str, Any] = {
        name: sum(event[name] or 0 for event in feedback_events) for name in feedback_fields
    }
    skip_names = sorted({name for event in feedback_events for name in event["skip"]})
    feedback_totals["skip"] = {
        name: sum(event["skip"].get(name, 0) for event in feedback_events)
        for name in skip_names
    }

    by_arm: dict[str, dict[str, Any]] = {}
    for event in arm_events:
        key = f"{event['index']}:{event['type']}"
        aggregate = by_arm.setdefault(
            key,
            {
                "index": event["index"],
                "type": event["type"],
                "base_prob": event["base_prob"],
                "selected": 0,
                "shadow_selected": 0,
                "accepted": 0,
                "credited": 0,
                "improved": 0,
            },
        )
        selected = (
            event["selected_batch"]
            if event["selected_batch"] is not None
            else event["selected"]
        )
        aggregate["selected"] += selected or 0
        aggregate["shadow_selected"] += event["shadow_selected"] or 0
        aggregate["accepted"] += event["accepted"] or 0
        aggregate["credited"] += event["credited"] or 0
        aggregate["improved"] += event["improved_batch"] or 0
        aggregate["selected_total"] = event["selected_total"]
        aggregate["accepted_total"] = event["accepted_total"]
        for source, destination in (
            ("window_count", "final_window_count"),
            ("window_improved", "final_window_improved"),
            ("mean_reward", "final_mean_reward"),
            ("ucb", "final_ucb"),
        ):
            aggregate[destination] = event[source]

    return {
        "feedback_calls": len(feedback_events),
        "arm_events": len(arm_events),
        "generation_calls": len(generation_events),
        "runner_only": all(event["source"] == "runner" for event in feedback_events),
        "feedback_totals": feedback_totals,
        "by_arm": by_arm,
        "feedback_events": feedback_events,
        "arm_event_records": arm_events,
        "generation_events": generation_events,
    }


def _parse_draft_annealing_log(text: str) -> dict[str, Any]:
    """Parse G4 summaries while retaining counters needed for ablation checks."""
    prefix = "Draft GA annealing summary:"
    integer_fields = (
        "compared",
        "improved_or_equal",
        "worse",
        "accepted_worse",
        "rejected_worse",
        "unknown",
        "draft_time_us",
    )
    events: list[dict[str, Any]] = []
    for line in text.splitlines():
        if prefix not in line:
            continue
        payload = line.split(prefix, maxsplit=1)[1]
        fields = dict(re.findall(r"([A-Za-z_][A-Za-z0-9_]*)=([^\s]+)", payload))
        event: dict[str, Any] = {
            "mode": fields.pop("mode", None),
            "t0": float(fields.pop("t0")) if "t0" in fields else None,
            "decay": float(fields.pop("decay")) if "decay" in fields else None,
        }
        for name in integer_fields:
            event[name] = int(fields.pop(name)) if name in fields else None
        event["extra_fields"] = fields
        events.append(event)
    totals = {
        name: sum(int(event[name]) for event in events if event[name] is not None)
        for name in integer_fields
    }
    return {"calls": len(events), "totals": totals, "events": events}


def _validate_draft_ga_contract(
    search: dict[str, Any], args: argparse.Namespace
) -> dict[str, Any]:
    """Fail closed when a run does not execute the requested scoring architecture."""
    stats = search["draft_ga"]
    events = stats["summary_events"]
    batches = int(search["search_batches"])
    expected_ga_generations = int(args.genetic_iters) + 1
    expected_generation_events = (
        batches * expected_ga_generations if args.draft_ga_mode != "off" else 0
    )
    errors: list[str] = []

    if len(events) != batches:
        errors.append(
            f"expected {batches} evolutionary scoring summaries, found {len(events)}"
        )
    if stats["generation_calls"] != expected_generation_events:
        errors.append(
            "Draft generation event count mismatch: "
            f"expected {expected_generation_events}, found {stats['generation_calls']}"
        )

    for batch_index, event in enumerate(events):
        if event["ga_generations"] != expected_ga_generations:
            errors.append(
                f"batch {batch_index}: ga_generations={event['ga_generations']} "
                f"(expected {expected_ga_generations})"
            )
        if args.draft_ga_mode == "enforce":
            if event["mode"] != "draft-enforce":
                errors.append(f"batch {batch_index}: unexpected mode={event['mode']}")
            if event["ga_xgb_calls"] != 0 or event["ga_xgb_input"] != 0:
                errors.append(
                    f"batch {batch_index}: XGBoost participated inside GA "
                    f"(calls={event['ga_xgb_calls']}, input={event['ga_xgb_input']})"
                )
            if event["final_xgb_calls"] != 1:
                errors.append(
                    f"batch {batch_index}: final_xgb_calls={event['final_xgb_calls']} "
                    "(expected exactly 1)"
                )
            expected_top_k = min(event["verify_top_k"], event["draft_archive"])
            if event["final_xgb_input"] != expected_top_k:
                errors.append(
                    f"batch {batch_index}: final_xgb_input={event['final_xgb_input']} "
                    f"does not equal effective_top_k={expected_top_k}"
                )
            if event["draft_archive"] < expected_top_k:
                errors.append(
                    f"batch {batch_index}: draft_archive={event['draft_archive']} "
                    f"is smaller than effective_top_k={expected_top_k}"
                )
        else:
            expected_mode = "baseline" if args.draft_ga_mode == "off" else "draft-shadow"
            if event["mode"] != expected_mode:
                errors.append(f"batch {batch_index}: unexpected mode={event['mode']}")
            if event["ga_xgb_calls"] != expected_ga_generations:
                errors.append(
                    f"batch {batch_index}: ga_xgb_calls={event['ga_xgb_calls']} "
                    f"(expected {expected_ga_generations})"
                )
            if event["final_xgb_calls"] != 0 or event["final_xgb_input"] != 0:
                errors.append(
                    f"batch {batch_index}: baseline/shadow unexpectedly used final XGBoost"
                )

    if args.ga_t2_mode == "directed" and search["draft_direction"]["calls"] != 0:
        errors.append(
            "trace-only G1 unexpectedly entered the retired Draft best-of-N replay path"
        )

    g3 = search["draft_g3"]
    if args.draft_ga_g3_mode == "off":
        if g3["generation_calls"] != 0:
            errors.append(
                "G3 off mode emitted generation telemetry "
                f"(calls={g3['generation_calls']})"
            )
        for event in g3["feedback_events"]:
            if event["mode"] != "off" or (event["credited"] or 0) != 0:
                errors.append("G3 off mode emitted active feedback telemetry")
    else:
        if args.draft_ga_mode != "enforce":
            errors.append("G3 requires --draft-ga-mode enforce")
        if g3["feedback_calls"] != batches:
            errors.append(
                f"expected {batches} G3 feedback summaries, found {g3['feedback_calls']}"
            )
        if not g3["runner_only"]:
            errors.append("G3 reward source was not exclusively Runner")
        feedback_events_by_batch: dict[int, list[dict[str, Any]]] = {}
        for event in g3["feedback_events"]:
            batch = event["batch"]
            if batch is not None:
                feedback_events_by_batch.setdefault(batch, []).append(event)
            if event["mode"] != args.draft_ga_g3_mode:
                errors.append(
                    f"G3 batch {batch}: mode={event['mode']} "
                    f"(expected {args.draft_ga_g3_mode})"
                )
            counters = [
                event[name]
                for name in (
                    "sent",
                    "child_measured",
                    "with_lineage",
                    "with_measured_parent",
                    "credited",
                )
            ]
            if any(value is None for value in counters):
                errors.append(f"G3 batch {batch}: incomplete feedback counters")
            elif any(left < right for left, right in zip(counters, counters[1:])):
                errors.append(f"G3 batch {batch}: feedback funnel is not monotonic")
            if event["improved"] is None or (
                event["credited"] is not None and event["improved"] > event["credited"]
            ):
                errors.append(f"G3 batch {batch}: improved exceeds credited")
            if event["cold_start"] not in (0, 1):
                errors.append(f"G3 batch {batch}: invalid cold_start={event['cold_start']}")
            batch_arms = [
                arm_event
                for arm_event in g3["arm_event_records"]
                if arm_event["batch"] == batch
            ]
            if not batch_arms:
                errors.append(f"G3 batch {batch}: missing per-mutator arm telemetry")
            else:
                required_arm_fields = (
                    "index",
                    "type",
                    "base_prob",
                    "selected",
                    "credited",
                    "improved_batch",
                    "window_count",
                    "mean_reward",
                    "ucb",
                )
                for arm_event in batch_arms:
                    missing = [
                        name for name in required_arm_fields if arm_event[name] is None
                    ]
                    if missing:
                        errors.append(
                            f"G3 batch {batch}: incomplete arm telemetry fields={missing}"
                        )
                if event["credited"] is not None and sum(
                    arm_event["credited"] or 0 for arm_event in batch_arms
                ) != event["credited"]:
                    errors.append(f"G3 batch {batch}: arm credited total does not reconcile")
                if event["improved"] is not None and sum(
                    arm_event["improved_batch"] or 0 for arm_event in batch_arms
                ) != event["improved"]:
                    errors.append(f"G3 batch {batch}: arm improved total does not reconcile")

        expected_feedback_batches = set(range(1, batches + 1))
        actual_feedback_batches = set(feedback_events_by_batch)
        if actual_feedback_batches != expected_feedback_batches:
            errors.append(
                "G3 feedback batch coverage mismatch: "
                f"missing={sorted(expected_feedback_batches - actual_feedback_batches)}, "
                f"unexpected={sorted(actual_feedback_batches - expected_feedback_batches)}"
            )
        duplicate_feedback_batches = sorted(
            batch
            for batch, batch_events in feedback_events_by_batch.items()
            if len(batch_events) != 1
        )
        if duplicate_feedback_batches:
            errors.append(f"G3 feedback duplicate batches: {duplicate_feedback_batches}")

        expected_g3_generations = batches * int(args.genetic_iters)
        generation_events = g3["generation_events"]
        if g3["generation_calls"] != expected_g3_generations:
            errors.append(
                "G3 generation event count mismatch: "
                f"expected {expected_g3_generations}, found {g3['generation_calls']}"
            )
        events_by_key: dict[tuple[int, int], list[dict[str, Any]]] = {}
        for event in generation_events:
            batch = event["batch"]
            iteration = event["iter"]
            if batch is None or iteration is None:
                errors.append("G3 generation telemetry is missing batch or iter")
                continue
            events_by_key.setdefault((batch, iteration), []).append(event)
            if event["mode"] != args.draft_ga_g3_mode:
                errors.append(
                    f"G3 generation {batch}/{iteration}: mode={event['mode']} "
                    f"(expected {args.draft_ga_g3_mode})"
                )
            if event["cold_start"] not in (0, 1):
                errors.append(
                    f"G3 generation {batch}/{iteration}: "
                    f"invalid cold_start={event['cold_start']}"
                )
            for name in ("selected", "shadow_selected", "accepted"):
                counts = event[name]
                if any(index < 0 or count < 0 for index, count in counts.items()):
                    errors.append(
                        f"G3 generation {batch}/{iteration}: invalid {name} counters"
                    )
            selected = event["selected"]
            accepted = event["accepted"]
            if set(selected) != set(accepted):
                errors.append(
                    f"G3 generation {batch}/{iteration}: selected/accepted arms differ"
                )
            elif any(accepted[index] > selected[index] for index in selected):
                errors.append(
                    f"G3 generation {batch}/{iteration}: accepted exceeds selected"
                )

        expected_keys = {
            (batch, iteration)
            for batch in range(1, batches + 1)
            for iteration in range(int(args.genetic_iters))
        }
        actual_keys = set(events_by_key)
        if actual_keys != expected_keys:
            errors.append(
                "G3 generation batch/iter coverage mismatch: "
                f"missing={sorted(expected_keys - actual_keys)}, "
                f"unexpected={sorted(actual_keys - expected_keys)}"
            )
        duplicate_keys = sorted(key for key, values in events_by_key.items() if len(values) != 1)
        if duplicate_keys:
            errors.append(f"G3 generation telemetry has duplicate keys: {duplicate_keys}")

        feedback_cold = {
            int(event["batch"]): int(event["cold_start"])
            for event in g3["feedback_events"]
            if event["batch"] is not None and event["cold_start"] is not None
        }
        for batch in range(1, batches + 1):
            batch_events = [
                event for event in generation_events if event["batch"] == batch
            ]
            cold_values = {event["cold_start"] for event in batch_events}
            if len(cold_values) > 1:
                errors.append(f"G3 batch {batch}: generation cold_start changed within batch")
            expected_cold = 1 if batch == 1 else feedback_cold.get(batch - 1)
            if expected_cold is None:
                errors.append(f"G3 batch {batch}: prior feedback cold_start is unavailable")
            elif cold_values != {expected_cold}:
                errors.append(
                    f"G3 batch {batch}: generation cold_start={sorted(cold_values)} "
                    f"does not match prior feedback={expected_cold}"
                )

    annealing = search["draft_annealing"]
    if args.draft_ga_anneal_mode == "off":
        if annealing["calls"] != 0:
            errors.append("G4 summaries were emitted although annealing mode is off")
    else:
        if args.draft_ga_mode != "enforce":
            errors.append("G4 requires --draft-ga-mode enforce")
        if annealing["calls"] != batches:
            errors.append(
                f"expected {batches} G4 summaries, found {annealing['calls']}"
            )
        for batch_index, event in enumerate(annealing["events"]):
            if event["mode"] != args.draft_ga_anneal_mode:
                errors.append(
                    f"batch {batch_index}: G4 mode={event['mode']} "
                    f"(expected {args.draft_ga_anneal_mode})"
                )
            if event["t0"] is None or not math.isclose(
                event["t0"], args.draft_ga_anneal_t0, abs_tol=5e-5
            ):
                errors.append(
                    f"batch {batch_index}: G4 t0={event['t0']} "
                    f"(expected {args.draft_ga_anneal_t0})"
                )
            if event["decay"] is None or not math.isclose(
                event["decay"], args.draft_ga_anneal_decay, abs_tol=5e-5
            ):
                errors.append(
                    f"batch {batch_index}: G4 decay={event['decay']} "
                    f"(expected {args.draft_ga_anneal_decay})"
                )

    adaptive_batches = sorted(
        {
            int(event["batch"])
            for event in g3["generation_events"]
            if event["batch"] is not None and event["cold_start"] == 0
        }
    )
    result = {
        "passed": not errors,
        "draft_ga_mode": args.draft_ga_mode,
        "search_batches": batches,
        "expected_ga_generations_per_batch": expected_ga_generations,
        "ga_xgb_calls": stats["ga_xgb_calls"],
        "ga_xgb_input": stats["ga_xgb_input"],
        "final_xgb_calls": stats["final_xgb_calls"],
        "final_xgb_input": stats["final_xgb_input"],
        "g3_mode": args.draft_ga_g3_mode,
        "g3_runner_only": g3["runner_only"],
        "g3_credited": g3["feedback_totals"].get("credited", 0),
        "g3_improved": g3["feedback_totals"].get("improved", 0),
        "g3_generation_calls": g3["generation_calls"],
        "g3_expected_generation_calls": (
            batches * int(args.genetic_iters) if args.draft_ga_g3_mode != "off" else 0
        ),
        "g3_first_adaptive_batch": adaptive_batches[0] if adaptive_batches else None,
        "g3_adaptive_generation_seen": bool(adaptive_batches),
        "errors": errors,
    }
    return result


def _parse_exact_init_log(text: str) -> dict[str, Any]:
    pattern = re.compile(
        r"Exact init population: target=(\d+) returned=(\d+) generated=(\d+) "
        r"postproc_passed=(\d+) discarded=(\d+) batches=(\d+) time_us=(\d+)"
    )
    events = [
        {
            "target": int(match.group(1)),
            "returned": int(match.group(2)),
            "generated": int(match.group(3)),
            "postproc_passed": int(match.group(4)),
            "discarded": int(match.group(5)),
            "batches": int(match.group(6)),
            "time_us": int(match.group(7)),
        }
        for match in pattern.finditer(text)
    ]
    if not events:
        return {"calls": 0, "events": []}
    return {
        "calls": len(events),
        "all_targets_met": all(event["returned"] == event["target"] for event in events),
        "target_total": sum(event["target"] for event in events),
        "returned_total": sum(event["returned"] for event in events),
        "generated_total": sum(event["generated"] for event in events),
        "postproc_passed_total": sum(event["postproc_passed"] for event in events),
        "discarded_total": sum(event["discarded"] for event in events),
        "batches_total": sum(event["batches"] for event in events),
        "time_us_total": sum(event["time_us"] for event in events),
        "events": events,
    }


def _parse_search_log(task_log: Path, mode: str) -> dict[str, Any]:
    text = task_log.read_text(encoding="utf-8", errors="replace")
    blocks = _parse_summary_blocks(text)
    effective = _effective_blocks(blocks)

    starts: list[datetime] = []
    ends: list[datetime] = []
    for line in text.splitlines():
        stamp = _timestamp(line)
        if stamp is None:
            continue
        if "Generating candidates" in line:
            starts.append(stamp)
        elif re.search(r"Sending \d+ candidates\(s\) for measurement", line):
            ends.append(stamp)
    if len(starts) != len(ends):
        raise RuntimeError(
            f"Cannot pair search intervals in {task_log}: starts={len(starts)}, ends={len(ends)}"
        )

    totals = {
        "trace_fail": _sum_optional(effective, ("trace_fail",)),
        "verify_gpu_fail": _sum_optional(effective, ("verify_gpu_fail",)),
        "accounting": _aggregate_accounting(effective),
        "filter": {
            name: _sum_optional(effective, ("filter", name))
            for name in (
                "checked",
                "unknown",
                "predicted",
                "enforced",
                "analysis_time_us",
                "verify_gpu_time_us",
                "tp",
                "fp",
                "fn",
                "tn",
                "predicted_low",
                "predicted_high",
                "predicted_hardware",
                "shared_checked",
                "shared_unknown",
                "shared_predicted",
                "shared_tp",
                "shared_fp",
                "shared_fn",
                "shared_tn",
            )
        },
        "constraint_sampling": {
            name: _sum_optional(effective, ("constraint_sampling", name))
            for name in (
                "plans_built",
                "contract_blocks",
                "enforceable_constraints",
                "decisions_handled",
                "decisions_replaced",
                "decision_unknown",
                "feasible_domain_total",
                "empty_domains",
                "retries",
                "abandoned",
                "candidates_checked",
                "candidates_unknown",
                "predicted_rejections",
                "plan_time_us",
                "decision_sampling_time_us",
                "check_time_us",
                "trace_replay_time_us",
                "tp",
                "fp",
                "fn",
                "tn",
            )
        },
        "hard_positions": {
            position: {
                name: _sum_optional(effective, ("hard_positions", position, name))
                for name in ("checked", "unknown", "predicted", "enforced", "time_us")
            }
            for position in HARD_FILTER_POSITIONS
        },
        "postproc_time_us": {
            key: sum(block["postproc_time_us"].get(key, 0) for block in effective)
            for key in sorted({key for block in effective for key in block["postproc_time_us"]})
        },
    }
    required = (
        "checked",
        "unknown",
        "predicted",
        "enforced",
        "analysis_time_us",
        "verify_gpu_time_us",
        "predicted_low",
        "predicted_high",
        "predicted_hardware",
        "shared_checked",
        "shared_unknown",
        "shared_predicted",
        "shared_tp",
        "shared_fp",
        "shared_fn",
        "shared_tn",
    )
    if mode == "shadow":
        required += ("tp", "fp", "fn", "tn")
    missing = [name for name in required if totals["filter"][name] is None]
    if missing:
        raise RuntimeError(f"Missing thread-filter statistics in {task_log}: {missing}")

    if mode == "shadow":
        filt = totals["filter"]
        classified = sum(filt[name] for name in ("tp", "fp", "fn", "tn"))
        predicted_classified = filt["tp"] + filt["fp"]
        if classified > filt["checked"] or predicted_classified > filt["predicted"]:
            raise RuntimeError(
                "Inconsistent early-shadow counters: classified candidates and predictions "
                "cannot exceed their evaluated totals; "
                f"got {filt}"
            )
        filt["classified"] = classified
        filt["unclassified"] = filt["checked"] - classified
        filt["predicted_classified"] = predicted_classified
        filt["predicted_unclassified"] = filt["predicted"] - predicted_classified
        illegal = filt["tp"] + filt["fn"]
        filt["precision"] = filt["tp"] / predicted_classified if predicted_classified else None
        filt["recall"] = filt["tp"] / illegal if illegal else None
        filt["observed_false_positive_free"] = filt["fp"] == 0
        if filt["predicted"] != (
            filt["predicted_low"]
            + filt["predicted_high"]
            + filt["predicted_hardware"]
        ):
            raise RuntimeError(
                "Hard-filter reason counters do not sum to predicted total: "
                f"{filt}"
            )
        if filt["shared_predicted"] != filt["shared_tp"] + filt["shared_fp"]:
            raise RuntimeError(
                "Shared proxy predicted must equal shared TP + FP; " f"got {filt}"
            )
        if filt["shared_checked"] != sum(
            filt[name] for name in ("shared_tp", "shared_fp", "shared_fn", "shared_tn")
        ):
            raise RuntimeError(f"Shared proxy confusion counters are incomplete: {filt}")

    if mode == "enforce":
        filt = totals["filter"]
        # Zero is a valid outcome for operators with no statically provable
        # thread rejection (for example pure elementwise kernels).  The
        # enforce contract is equality, not a requirement that a rejection
        # must exist.
        if filt["enforced"] != filt["predicted"]:
            raise RuntimeError(
                "Enforce mode must exercise and apply every predicted rejection; "
                f"got predicted={filt['predicted']}, enforced={filt['enforced']}"
            )

    stage_totals: dict[str, dict[str, Any]] = {}
    for stage in ("pick_best", "sample_init", "evolve"):
        selected = [block for block in effective if block["stage"] == stage]
        stage_totals[stage] = {
            "summaries": len(selected),
            "verify_gpu_fail": _sum_optional(selected, ("verify_gpu_fail",)),
            "accounting": _aggregate_accounting(selected),
            "filter": {
                name: _sum_optional(selected, ("filter", name))
                for name in (
                    "checked",
                    "unknown",
                    "predicted",
                    "enforced",
                    "analysis_time_us",
                    "verify_gpu_time_us",
                    "tp",
                    "fp",
                    "fn",
                    "tn",
                    "predicted_low",
                    "predicted_high",
                    "predicted_hardware",
                    "shared_checked",
                    "shared_unknown",
                    "shared_predicted",
                    "shared_tp",
                    "shared_fp",
                    "shared_fn",
                    "shared_tn",
                )
            },
            "constraint_sampling": {
                name: _sum_optional(selected, ("constraint_sampling", name))
                for name in (
                    "plans_built",
                    "contract_blocks",
                    "enforceable_constraints",
                    "decisions_handled",
                    "decisions_replaced",
                    "decision_unknown",
                    "feasible_domain_total",
                    "empty_domains",
                    "retries",
                    "abandoned",
                    "candidates_checked",
                    "candidates_unknown",
                    "predicted_rejections",
                    "plan_time_us",
                    "decision_sampling_time_us",
                    "check_time_us",
                    "trace_replay_time_us",
                    "tp",
                    "fp",
                    "fn",
                    "tn",
                )
            },
            "hard_positions": {
                position: {
                    name: _sum_optional(selected, ("hard_positions", position, name))
                    for name in ("checked", "unknown", "predicted", "enforced", "time_us")
                }
                for position in HARD_FILTER_POSITIONS
            },
            "postproc_time_us": {
                key: sum(block["postproc_time_us"].get(key, 0) for block in selected)
                for key in sorted({key for block in selected for key in block["postproc_time_us"]})
            },
        }

    return {
        "search_s": sum((end - start).total_seconds() for start, end in zip(starts, ends)),
        "search_batches": len(starts),
        "soft_filter": _parse_soft_filter_log(text),
        "draft_model": _parse_draft_model_log(text),
        "draft_ga": _parse_draft_ga_log(text),
        "draft_direction": _parse_draft_direction_log(text),
        "draft_g3": _parse_draft_g3_log(text),
        "draft_annealing": _parse_draft_annealing_log(text),
        "exact_init_population": _parse_exact_init_log(text),
        "totals": totals,
        "by_stage": stage_totals,
        "summary_blocks": blocks,
    }


def _parse_scheduler_log(scheduler_log: Path) -> dict[str, Any]:
    text = scheduler_log.read_text(encoding="utf-8", errors="replace")
    builder_start: datetime | None = None
    runner_start: datetime | None = None
    builder_s = 0.0
    runner_update_s = 0.0
    sent_to_builder = 0
    sent_to_runner = 0
    build_errors = 0

    for line in text.splitlines():
        stamp = _timestamp(line)
        if stamp is None:
            continue
        match = re.search(r"Sending (\d+) sample\(s\) to builder", line)
        if match:
            sent_to_builder += int(match.group(1))
            builder_start = stamp
            continue
        match = re.search(r"Build errors: (\d+) sample\(s\)", line)
        if match:
            build_errors += int(match.group(1))
            continue
        match = re.search(r"Sending (\d+) valid sample\(s\) to runner", line)
        if match:
            sent_to_runner += int(match.group(1))
            if builder_start is not None:
                builder_s += (stamp - builder_start).total_seconds()
                builder_start = None
            runner_start = stamp
            continue
        if "[Updated]" in line and runner_start is not None:
            runner_update_s += (stamp - runner_start).total_seconds()
            runner_start = None

    return {
        "builder_s": builder_s,
        "runner_and_update_s": runner_update_s,
        "sent_to_builder": sent_to_builder,
        "build_errors": build_errors,
        "sent_to_runner": sent_to_runner,
    }


def _run_git_bytes(*args: str) -> bytes:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as err:
        raise RuntimeError(f"Cannot capture source identity with git {' '.join(args)}") from err


def _git_source_identity() -> dict[str, str]:
    """Hash HEAD, repository state, tracked diff, and untracked source contents."""
    source_paths = (
        "CMakeLists.txt",
        "cmake",
        "include",
        "python",
        "src",
        "tests",
        "experiments/e1_thread_filter.py",
        "experiments/p3_observability.py",
        "experiments/clean_workloads.py",
        "experiments/runtime_bootstrap",
    )
    head = _run_git_bytes("rev-parse", "HEAD").strip().decode("ascii")
    status = _run_git_bytes(
        "status", "--porcelain=v1", "-z", "--untracked-files=normal", "--", *source_paths
    )
    diff = _run_git_bytes("diff", "--binary", "--no-ext-diff", "HEAD", "--", *source_paths)
    untracked_paths = _run_git_bytes(
        "ls-files",
        "--others",
        "--exclude-standard",
        "-z",
        "--",
        *source_paths,
    ).split(b"\0")

    untracked_source = hashlib.sha256()
    for raw_path in sorted(path for path in untracked_paths if path):
        path = REPO_ROOT / os.fsdecode(raw_path)
        content_hash = _sha256_file(path)
        untracked_source.update(len(raw_path).to_bytes(8, "big"))
        untracked_source.update(raw_path)
        untracked_source.update(bytes.fromhex(content_hash))

    status_hash = hashlib.sha256(status).hexdigest()
    diff_hash = hashlib.sha256(diff).hexdigest()
    untracked_source_hash = untracked_source.hexdigest()
    combined = hashlib.sha256()
    for component in (head, status_hash, diff_hash, untracked_source_hash):
        combined.update(component.encode("ascii"))
        combined.update(b"\0")
    return {
        "git_head": head,
        "git_status_sha256": status_hash,
        "git_diff_sha256": diff_hash,
        "git_untracked_source_sha256": untracked_source_hash,
        "source_identity_sha256": combined.hexdigest(),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        while chunk := file_obj.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_soft_weight(value: float) -> float:
    weight = float(value)
    if not math.isfinite(weight) or not 0.0 <= weight <= 1.0:
        raise ValueError("--soft-weight must be a finite number in [0, 1]")
    return 0.0 if weight == 0.0 else weight


def _validate_nonnegative(value: float) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError("value must be a finite non-negative number")
    return result


def _validate_nonnegative_int(value: str) -> int:
    result = int(value)
    if result < 0:
        raise ValueError("value must be a non-negative integer")
    return result


def _validate_positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise ValueError("value must be a positive integer")
    return result


def _validate_probability(value: float) -> float:
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError("value must be a finite number in [0, 1]")
    return result


def _validate_keep_ratio(value: float) -> float:
    result = _validate_probability(value)
    if result == 0.0:
        raise ValueError("keep ratio must be within (0, 1]")
    return result


def _validate_positive(value: float) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError("value must be a finite positive number")
    return result


def _record_latency_s(record: Any) -> float | None:
    values = [float(value) for value in record.run_secs]
    values = [value for value in values if math.isfinite(value) and 0 < value < 1e9]
    if not values:
        return None
    values.sort()
    middle = len(values) // 2
    if len(values) % 2:
        return values[middle]
    return 0.5 * (values[middle - 1] + values[middle])


def _summarize_callback_batches(batches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries = []
    for index, batch in enumerate(batches):
        latencies = []
        for run_secs in batch["runner_run_secs"]:
            if run_secs is None:
                continue
            values = sorted(
                value for value in run_secs if math.isfinite(value) and 0 < value < 1e9
            )
            if values:
                middle = len(values) // 2
                latency = values[middle]
                if len(values) % 2 == 0:
                    latency = 0.5 * (values[middle - 1] + values[middle])
                latencies.append(latency)
        sorted_latencies = sorted(latencies)
        median_s = None
        if sorted_latencies:
            middle = len(sorted_latencies) // 2
            median_s = sorted_latencies[middle]
            if len(sorted_latencies) % 2 == 0:
                median_s = 0.5 * (
                    sorted_latencies[middle - 1] + sorted_latencies[middle]
                )
        summaries.append(
            {
                "batch_index": index,
                "candidates": len(batch["runner_errors"]),
                "builder_errors": sum(error is not None for error in batch["builder_errors"]),
                "runner_errors": sum(error is not None for error in batch["runner_errors"]),
                "finite": len(latencies),
                "exclusive_outcomes": {
                    name: classify_measurements(batch).count(name)
                    for name in ("build_error", "runner_error", "invalid_latency", "finite")
                },
                "best_latency_us": min(latencies) * 1e6 if latencies else None,
                "median_latency_us": median_s * 1e6 if median_s is not None else None,
                "mean_latency_us": (
                    sum(latencies) / len(latencies) * 1e6 if latencies else None
                ),
            }
        )
    return summaries


def _validate_callback_batch_contract(
    batches: list[dict[str, Any]],
    search_batches: int,
    trials: int,
    trials_per_iter: int,
    effective_trials: int | None = None,
) -> list[str]:
    """Validate the schema-11 mapping from search batches to Runner callbacks."""
    errors: list[str] = []
    # Some tiny operator spaces exhaust before max_trials_global.  TVM still
    # emits trailing empty search batches; validate the non-empty prefix
    # against the number of records actually measured instead of treating
    # those empty callbacks as an architecture error.
    target_trials = trials if effective_trials is None else min(trials, effective_trials)
    nonempty = [
        batch
        for batch in batches
        if any(batch[name] for name in ("builder_errors", "runner_errors", "runner_run_secs"))
    ]
    expected_batches = (
        (target_trials + trials_per_iter - 1) // trials_per_iter if target_trials else 0
    )
    if len(nonempty) != expected_batches:
        errors.append(
            f"callback nonempty batches={len(nonempty)}, expected={expected_batches}"
        )
    total_candidates = 0
    for index, batch in enumerate(nonempty):
        expected = min(trials_per_iter, max(0, target_trials - index * trials_per_iter))
        field_lengths = {
            name: len(batch[name])
            for name in ("builder_errors", "runner_errors", "runner_run_secs")
        }
        total_candidates += field_lengths["runner_errors"]
        if any(length != expected for length in field_lengths.values()):
            errors.append(
                f"callback batch {index + 1}: field lengths={field_lengths}, "
                f"expected candidates={expected}"
            )
    if total_candidates != target_trials:
        errors.append(
            f"callback candidates={total_candidates}, expected trials={target_trials}"
        )
    return errors


def run_experiment(args: argparse.Namespace) -> Path:
    process_start = time.perf_counter()
    if args.check_correctness and args.workload not in ("matmul128", "matmul256", *CLEAN_WORKLOADS):
        raise ValueError("--check-correctness supports square MatMul and explicit clean workloads")
    mode_value = MODE_TO_VALUE[args.mode]
    constraint_sample_mode_value = MODE_TO_VALUE[args.constraint_sample_mode]
    soft_mode_value = SOFT_MODE_TO_VALUE[args.soft_mode]
    ga_t2_mode_value = GA_T2_MODE_TO_VALUE[args.ga_t2_mode]
    draft_mode_value = DRAFT_MODE_TO_VALUE[args.draft_mode]
    draft_ga_mode_value = DRAFT_GA_MODE_TO_VALUE[args.draft_ga_mode]
    draft_ga_g3_mode_value = DRAFT_GA_G3_MODE_TO_VALUE[args.draft_ga_g3_mode]
    soft_weight = _validate_soft_weight(args.soft_weight)
    if args.trials < args.trials_per_iter:
        raise ValueError("--trials must be greater than or equal to --trials-per-iter")
    if args.draft_ga_mode != "off" and args.draft_mode != "off":
        raise ValueError(
            "Draft-guided GA and the legacy post-XGBoost --draft-mode must not be enabled "
            "together"
        )
    if args.draft_ga_g3_mode != "off" and args.draft_ga_mode != "enforce":
        raise ValueError("G3 measured mutator credit requires --draft-ga-mode enforce")
    if args.draft_ga_anneal_mode != "off" and args.draft_ga_mode != "enforce":
        raise ValueError("G4 annealing requires --draft-ga-mode enforce")
    if args.draft_ga_direction_trials > 32:
        raise ValueError("--draft-ga-direction-trials must be at most 32")

    source_identity = _git_source_identity()

    work_dir = args.work_root / (
        f"{args.run_id}_{args.workload}_seed{args.seed}_{args.mode}"
        f"_sample-{args.constraint_sample_mode}_hpos-{args.hard_filter_position}"
        f"_soft-{args.soft_mode}_draft-{args.draft_mode}"
        f"_dga-{args.draft_ga_mode}_ga-{args.ga_t2_mode}"
        f"_g3-{args.draft_ga_g3_mode}"
        f"_g4-{args.draft_ga_anneal_mode}"
    )
    if work_dir.exists():
        raise FileExistsError(
            f"Refusing to reuse existing work directory: {work_dir}. Choose another --run-id."
        )
    work_dir.mkdir(parents=True)
    args._active_work_dir = work_dir
    write_json(work_dir / "command.json", {
        "argv": sys.argv, "python": sys.executable, "started_local": datetime.now().isoformat(),
        "config": {key: str(value) if isinstance(value, Path) else value
                   for key, value in vars(args).items() if key != "func" and not key.startswith("_")},
    })
    write_json(work_dir / "run_state.json", {"status": "preparing"})
    for name in ("e1_thread_filter.py", "p3_observability.py", "clean_workloads.py"):
        shutil.copy2(REPO_ROOT / "experiments" / name, work_dir / ("snapshot_" + name))
    # The logging backend writes directly into this directory when tuning
    # starts; create it here so standalone invocations do not race/fail before
    # the first task logger is configured.
    (work_dir / "logs").mkdir(parents=True, exist_ok=True)
    for log_name in (
        "tvm.s_tir.meta_schedule.logging.task_scheduler.log",
        "tvm.s_tir.meta_schedule.logging.task_0_main.log",
    ):
        (work_dir / "logs" / log_name).touch(exist_ok=True)

    build_lib = REPO_ROOT / "build-prune" / "lib"
    if not build_lib.is_dir():
        raise FileNotFoundError(f"Missing build-prune library directory: {build_lib}")
    args.temp_dir.mkdir(parents=True, exist_ok=True)

    # These must be set before constructing the search strategy, whose C++
    # ThreadedTraceApply instances cache all three first-point switches.
    os.environ["TVM_PRUNE_MODE"] = str(mode_value)
    os.environ["TVM_THREAD_CONSTRAINT_SAMPLE_MODE"] = args.constraint_sample_mode
    os.environ["TVM_THREAD_HARD_FILTER_MODE"] = args.mode
    os.environ["TVM_THREAD_HARD_FILTER_POSITION"] = args.hard_filter_position
    os.environ["TVM_SOFT_FILTER_MODE"] = str(soft_mode_value)
    os.environ["TVM_SOFT_FILTER_WEIGHT"] = str(soft_weight)
    os.environ["TVM_DRAFT_MODE"] = str(draft_mode_value)
    os.environ["TVM_DRAFT_BETA"] = str(args.draft_beta)
    os.environ["TVM_DRAFT_EPS"] = str(args.draft_eps)
    os.environ["TVM_DRAFT_GA_MODE"] = str(draft_ga_mode_value)
    os.environ["TVM_DRAFT_GA_VERIFY_RATIO"] = str(args.draft_ga_verify_ratio)
    os.environ["TVM_DRAFT_GA_G3_MODE"] = str(draft_ga_g3_mode_value)
    os.environ["TVM_DRAFT_GA_ANNEAL_MODE"] = str(
        DRAFT_GA_ANNEAL_MODE_TO_VALUE[args.draft_ga_anneal_mode]
    )
    os.environ["TVM_DRAFT_GA_ANNEAL_T0"] = str(args.draft_ga_anneal_t0)
    os.environ["TVM_DRAFT_GA_ANNEAL_DECAY"] = str(args.draft_ga_anneal_decay)
    os.environ["TVM_DRAFT_PROFILE"] = args.draft_profile
    os.environ["TVM_DRAFT_PEAK_GFLOPS"] = str(args.draft_peak_gflops)
    os.environ["TVM_DRAFT_MEMORY_GBPS"] = str(args.draft_memory_gbps)
    os.environ["TVM_DRAFT_SM_COUNT"] = str(args.draft_sm_count)
    os.environ["TVM_DRAFT_ACTIVE_BLOCKS_PER_SM"] = str(args.draft_active_blocks_per_sm)
    os.environ["TVM_DRAFT_SM_PARTITIONS"] = str(args.draft_sm_partitions)
    os.environ["TVM_DRAFT_REG_CAP"] = str(args.draft_reg_cap)
    os.environ["TVM_GA_T2_MODE"] = str(ga_t2_mode_value)
    os.environ["TVM_EVO_EXACT_INIT_POPULATION"] = "1" if args.exact_init_population else "0"
    os.environ["TVM_EVO_INIT_SAMPLE_BATCH_SIZE"] = str(args.init_sample_batch_size)
    os.environ["TVM_LIBRARY_PATH"] = str(build_lib)
    os.environ["TEMP"] = str(args.temp_dir)
    os.environ["TMP"] = str(args.temp_dir)
    if args.cuda_home is not None:
        os.environ["CUDA_HOME"] = str(args.cuda_home)
        os.environ["CUDA_PATH"] = str(args.cuda_home)
    python_dir = str(REPO_ROOT / "python")
    os.environ["PYTHONPATH"] = os.pathsep.join(
        part for part in (python_dir, str(REPO_ROOT / "experiments"), os.environ.get("PYTHONPATH")) if part
    )
    sys.path.insert(0, python_dir)

    import numpy as np  # pylint: disable=import-outside-toplevel
    import tvm  # pylint: disable=import-outside-toplevel
    from tvm.ir.utils import derived_object  # pylint: disable=import-outside-toplevel
    from tvm.script import tirx as T  # pylint: disable=import-outside-toplevel
    from tvm.s_tir.meta_schedule import tir_integration  # pylint: disable=import-outside-toplevel
    from tvm.s_tir.meta_schedule.builder import LocalBuilder  # pylint: disable=import-outside-toplevel
    from tvm.s_tir.meta_schedule.builder.local_builder import (  # pylint: disable=import-outside-toplevel
        default_build,
        default_export,
    )
    from tvm.s_tir.meta_schedule.cost_model import XGBModel  # pylint: disable=import-outside-toplevel
    from tvm.s_tir.meta_schedule.measure_callback import (  # pylint: disable=import-outside-toplevel
        MeasureCallback,
        PyMeasureCallback,
    )
    from tvm.s_tir.meta_schedule.runner import (  # pylint: disable=import-outside-toplevel
        EvaluatorConfig,
        LocalRunner,
        PyRunner,
    )
    from tvm.s_tir.meta_schedule.search_strategy import (  # pylint: disable=import-outside-toplevel
        EvolutionarySearch,
    )
    from tvm.target import Target  # pylint: disable=import-outside-toplevel
    from tvm_ffi import register_global_func  # pylint: disable=import-outside-toplevel

    loaded_library = Path(tvm.base._LOADED_LIBS["tvm_compiler"]._name).resolve()
    loaded_library_sha256 = _sha256_file(loaded_library)

    register_global_func("default", override=True)(default_build)
    register_global_func("default_export", override=True)(default_export)
    reset_ga_t2_stats = tvm.get_global_func(
        "s_tir.meta_schedule.testing.ResetMutateTileSizeThreadStats"
    )
    get_ga_t2_stats = tvm.get_global_func(
        "s_tir.meta_schedule.testing.GetMutateTileSizeThreadStats"
    )
    reset_ga_t2_stats()
    journal = (TuneJournal(work_dir / "timeline.jsonl", args.record_model_training)
               if args.record_timeline else None)
    if args.record_model_training:
        if not args.record_timeline:
            raise ValueError("Model training observation requires --record-timeline")
        import logging
        from tvm.s_tir.meta_schedule.cost_model.xgb_model import logger as xgb_logger
        xgb_logger.setLevel(logging.DEBUG)

    @derived_object
    class CheckedRunner(PyRunner):
        """Retain LocalRunner's sequential behavior, attach a worker check ID to each input."""

        def __init__(self, local_runner):
            self.local_runner = local_runner
            self.next_id = 0

        def run(self, runner_inputs):
            results = []
            for runner_input in runner_inputs:
                candidate_id = f"runner-{self.next_id:04d}"
                self.next_id += 1
                append_json(work_dir / "runner_inputs.jsonl", {
                    "candidate_id": candidate_id, "artifact_path": str(runner_input.artifact_path),
                })
                evaluator = (partial(checked_clean_evaluator, workload=args.workload)
                             if args.workload in CLEAN_WORKLOADS else checked_matmul_evaluator)
                self.local_runner.f_run_evaluator = partial(
                    evaluator,
                    audit_path=str((work_dir / "correctness.jsonl").resolve()),
                    candidate_id=candidate_id,
                )
                results.extend(self.local_runner.run([runner_input]))
            return results

    @derived_object
    class TimedCallback(PyMeasureCallback):
        def __init__(self, callback, index):
            self.callback, self.index = callback, index
            self.name = type(callback).__name__

        def apply(self, task_scheduler, task_id, candidates, build_results, run_results):
            journal.emit("callback_begin", callback_index=self.index, name=self.name)
            self.callback.apply(task_scheduler, task_id, candidates, build_results, run_results)
            journal.emit("callback_end", callback_index=self.index, name=self.name)

    @derived_object
    class CaptureMeasureResults(PyMeasureCallback):
        """Keep the errors that JSONDatabase intentionally reduces to 1e10."""

        def __init__(self) -> None:
            self.batches: list[dict[str, Any]] = []

        def apply(
            self,
            _task_scheduler: Any,
            task_id: int,
            _measure_candidates: list[Any],
            builder_results: list[Any],
            runner_results: list[Any],
        ) -> None:
            self.batches.append(
                {
                    "task_id": int(task_id),
                    "builder_errors": [
                        result.error_msg if result.error_msg is not None else None
                        for result in builder_results
                    ],
                    "runner_errors": [
                        result.error_msg if result.error_msg is not None else None
                        for result in runner_results
                    ],
                    "runner_run_secs": [
                        [float(value) for value in result.run_secs]
                        if result.run_secs is not None
                        else None
                        for result in runner_results
                    ],
                    "artifact_paths": [str(result.artifact_path) if result.artifact_path is not None
                                       else None for result in builder_results],
                    "trace_sha256": [hashlib.sha256(str(candidate.sch.trace).encode("utf-8")).hexdigest()
                                     for candidate in _measure_candidates],
                }
            )
            if journal:
                event = journal.emit("batch_complete", candidates=len(runner_results))
                self.batches[-1]["available_at_tune_s"] = event["tune_elapsed_s"]
                self.batches[-1]["timeline_batch_index"] = event["batch_index"]
            if args.record_model_training:
                booster = cost_model.booster
                self.batches[-1]["cost_model"] = {
                    "data_size": int(cost_model.data_size),
                    "last_train_size": int(cost_model.last_train_size),
                    "trained": booster is not None,
                    "boosted_rounds": int(booster.num_boosted_rounds()) if booster is not None else 0,
                    "best_iteration": booster.attr("best_iteration") if booster is not None else None,
                    "best_score": booster.attr("best_score") if booster is not None else None,
                    "best_metrics": booster.attr("best_msg") if booster is not None else None,
                    "training_policy": "retrain-from-scratch-on-all-accumulated-features",
                }
            append_json(work_dir / "callback_batches.jsonl", self.batches[-1])

    @tvm.script.ir_module
    class M128:
        @T.prim_func(s_tir=True)
        def main(a: T.handle, b: T.handle, c: T.handle) -> None:
            T.func_attr({"global_symbol": "main"})
            A = T.match_buffer(a, (128, 128), "float32")
            B = T.match_buffer(b, (128, 128), "float32")
            C = T.match_buffer(c, (128, 128), "float32")
            for i, j, k in T.grid(128, 128, 128):
                with T.sblock("m"):
                    vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                    with T.init():
                        C[vi, vj] = T.float32(0)
                    C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]

    @tvm.script.ir_module
    class M256:
        @T.prim_func(s_tir=True)
        def main(a: T.handle, b: T.handle, c: T.handle) -> None:
            T.func_attr({"global_symbol": "main"})
            A = T.match_buffer(a, (256, 256), "float32")
            B = T.match_buffer(b, (256, 256), "float32")
            C = T.match_buffer(c, (256, 256), "float32")
            for i, j, k in T.grid(256, 256, 256):
                with T.sblock("m"):
                    vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                    with T.init():
                        C[vi, vj] = T.float32(0)
                    C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]

    @tvm.script.ir_module
    class Reduction128:
        @T.prim_func(s_tir=True)
        def main(a: T.handle, b: T.handle) -> None:
            T.func_attr({"global_symbol": "main"})
            A = T.match_buffer(a, (128, 128), "float32")
            B = T.match_buffer(b, (128,), "float32")
            for i, j in T.grid(128, 128):
                with T.sblock("r"):
                    vi, vj = T.axis.remap("SR", [i, j])
                    with T.init():
                        B[vi] = T.float32(0)
                    B[vi] = B[vi] + A[vi, vj]

    @tvm.script.ir_module
    class ReLU128:
        @T.prim_func(s_tir=True)
        def main(a: T.handle, b: T.handle) -> None:
            T.func_attr({"global_symbol": "main"})
            A = T.match_buffer(a, (128, 128), "float32")
            B = T.match_buffer(b, (128, 128), "float32")
            for i, j in T.grid(128, 128):
                with T.sblock("relu"):
                    vi, vj = T.axis.remap("SS", [i, j])
                    B[vi, vj] = T.max(A[vi, vj], T.float32(0))

    @tvm.script.ir_module
    class Conv2D:
        @T.prim_func(s_tir=True)
        def main(x: T.handle, w: T.handle, y: T.handle) -> None:
            T.func_attr({"global_symbol": "main"})
            X = T.match_buffer(x, (1, 3, 224, 224), "float32")
            W = T.match_buffer(w, (64, 3, 7, 7), "float32")
            Y = T.match_buffer(y, (1, 64, 112, 112), "float32")
            for n, oc, oh, ow, ic, kh, kw in T.grid(1, 64, 112, 112, 3, 7, 7):
                with T.sblock("conv"):
                    vn, voc, voh, vow, vic, vkh, vkw = T.axis.remap(
                        "SSSSRRR", [n, oc, oh, ow, ic, kh, kw]
                    )
                    with T.init():
                        Y[vn, voc, voh, vow] = T.float32(0)
                    Y[vn, voc, voh, vow] = Y[vn, voc, voh, vow] + X[
                        vn, vic, voh + vkh, vow + vkw
                    ] * W[voc, vic, vkh, vkw]

    @tvm.script.ir_module
    class Elementwise128:
        @T.prim_func(s_tir=True)
        def main(a: T.handle, b: T.handle, c: T.handle) -> None:
            T.func_attr({"global_symbol": "main"})
            A = T.match_buffer(a, (128, 128), "float32")
            B = T.match_buffer(b, (128, 128), "float32")
            C = T.match_buffer(c, (128, 128), "float32")
            for i, j in T.grid(128, 128):
                with T.sblock("add"):
                    vi, vj = T.axis.remap("SS", [i, j])
                    C[vi, vj] = A[vi, vj] + B[vi, vj]

    @tvm.script.ir_module
    class Fusion128:
        @T.prim_func(s_tir=True)
        def main(a: T.handle, b: T.handle, c: T.handle) -> None:
            T.func_attr({"global_symbol": "main"})
            A = T.match_buffer(a, (128, 128), "float32")
            B = T.match_buffer(b, (128, 128), "float32")
            C = T.match_buffer(c, (128, 128), "float32")
            for i, j in T.grid(128, 128):
                with T.sblock("fused_add_relu"):
                    vi, vj = T.axis.remap("SS", [i, j])
                    C[vi, vj] = T.max(A[vi, vj] + B[vi, vj], T.float32(0))

    workloads = {
        "matmul128": M128,
        "matmul256": M256,
        "reduction128": Reduction128,
        "relu128": ReLU128,
        "conv2d": Conv2D,
        "elementwise128": Elementwise128,
        "fusion128": Fusion128,
    }
    workload_mod = (make_workload(args.workload) if args.workload in CLEAN_WORKLOADS
                    else workloads[args.workload])

    target = Target(
        {
            "kind": "cuda",
            "arch": "sm_86",
            "max_threads_per_block": 1024,
            "max_num_threads": 1024,
            "max_shared_memory_per_block": 49152,
            "thread_warp_size": 32,
        }
    )
    workload_json = tvm.ir.save_json(workload_mod)
    (work_dir / "workload.json").write_text(workload_json, encoding="utf-8")
    write_json(work_dir / "identity.json", {
        **source_identity, "loaded_library": str(loaded_library),
        "loaded_library_sha256": loaded_library_sha256, "target": str(target),
        "workload_sha256": _sha256_file(work_dir / "workload.json"),
        "driver_sha256": _sha256_file(Path(__file__)),
        "observability_sha256": _sha256_file(REPO_ROOT / "experiments/p3_observability.py"),
        "gpu_before": gpu_status(),
        "correctness": CORRECTNESS_CONFIG if args.check_correctness else None,
    })
    strategy = EvolutionarySearch(
        population_size=args.population_size,
        init_measured_ratio=0.2,
        init_min_unmeasured=args.init_min_unmeasured,
        max_fail_count=5,
        genetic_num_iters=args.genetic_iters,
        genetic_mutate_prob=0.85,
        genetic_max_fail_count=10,
        eps_greedy=args.eps_greedy,
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    setup_start = time.perf_counter()
    builder = LocalBuilder(max_workers=1, timeout_sec=args.build_timeout)
    runner = LocalRunner(
        timeout_sec=args.run_timeout,
        evaluator_config=EvaluatorConfig(number=3, repeat=1, min_repeat_ms=100),
    )
    if args.check_correctness:
        runner = CheckedRunner(runner)
    result_capture = CaptureMeasureResults()
    default_callbacks = list(MeasureCallback.create("default"))
    if journal:
        default_callbacks = [TimedCallback(callback, index)
                             for index, callback in enumerate(default_callbacks)]
    measure_callbacks = [*default_callbacks, result_capture]
    cost_model = XGBModel(
        num_warmup_samples=args.xgb_warmup_samples,
        num_tuning_cores=1,
        adaptive_training=False,
    )
    setup_s = time.perf_counter() - setup_start

    preparation_s = time.perf_counter() - process_start
    write_json(work_dir / "run_state.json", {"status": "tuning", "preparation_s": preparation_s})
    tune_start = time.perf_counter()
    if journal:
        journal.__enter__()
    try:
        database = tir_integration.tune_tir(
            mod=workload_mod,
            target=target,
            work_dir=str(work_dir),
            max_trials_global=args.trials,
            num_trials_per_iter=args.trials_per_iter,
            builder=builder,
            runner=runner,
            cost_model=cost_model,
            measure_callbacks=measure_callbacks,
            strategy=strategy,
            space="post-order-apply",
            num_tuning_cores=1,
            seed=args.seed,
        )
    except BaseException:
        if journal:
            journal.__exit__(*sys.exc_info())
        raise
    else:
        if journal:
            journal.__exit__(None, None, None)
    tune_s = time.perf_counter() - tune_start
    timeline = summarize_timeline(journal.events) if journal else None
    if timeline:
        write_json(work_dir / "timeline_summary.json", timeline)

    task_log = work_dir / "logs" / "tvm.s_tir.meta_schedule.logging.task_0_main.log"
    scheduler_log = work_dir / "logs" / "tvm.s_tir.meta_schedule.logging.task_scheduler.log"
    if not task_log.is_file() or not scheduler_log.is_file():
        raise FileNotFoundError(f"Expected tuning logs under {work_dir / 'logs'}")

    records = list(database.get_all_tuning_records())
    latencies_s = [_record_latency_s(record) for record in records]
    finite_latencies_s = [value for value in latencies_s if value is not None]
    trace_hashes = [
        hashlib.sha256(str(record.trace).encode("utf-8")).hexdigest() for record in records
    ]
    search = _parse_search_log(task_log, args.mode)
    architecture_validation = _validate_draft_ga_contract(search, args)
    callback_errors = _validate_callback_batch_contract(
        result_capture.batches,
        int(search["search_batches"]),
        int(args.trials),
        int(args.trials_per_iter),
        effective_trials=len(records),
    )
    architecture_validation["callback_batch_calls"] = len(result_capture.batches)
    architecture_validation["callback_batch_errors"] = callback_errors
    architecture_validation["errors"].extend(callback_errors)
    actual_submitted = sum(len(batch["runner_errors"]) for batch in result_capture.batches)
    if actual_submitted != len(records):
        architecture_validation["errors"].append("Callback slots disagree with database records")
    if timeline and sum(row["submitted"] for row in timeline["batches"]) != actual_submitted:
        architecture_validation["errors"].append("Builder-send events disagree with callback slots")
    correctness = None
    if args.check_correctness:
        input_path = work_dir / "runner_inputs.jsonl"
        check_path = work_dir / "correctness.jsonl"
        inputs = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines()] if input_path.exists() else []
        checks = [json.loads(line) for line in check_path.read_text(encoding="utf-8").splitlines()] if check_path.exists() else []
        passed_checks = {row["candidate_id"] for row in checks
                         if row["event"] == "correctness_result" and row["passed"]}
        artifact_ids = {row["artifact_path"]: row["candidate_id"] for row in inputs}
        finite_ids = []
        for batch in result_capture.batches:
            for artifact, outcome in zip(batch["artifact_paths"], classify_measurements(batch)):
                if outcome == "finite":
                    finite_ids.append(artifact_ids.get(artifact))
        if any(key not in passed_checks for key in finite_ids):
            architecture_validation["errors"].append("Finite measurement lacks matching numerical check")
        correctness_config = (dict(CORRECTNESS_CONFIG, seed=20261010, cases=["normal", "uniform", "ones", "zero"],
                                   workload=CLEAN_WORKLOADS[args.workload], reference="independent NumPy FP64")
                              if args.workload in CLEAN_WORKLOADS else CORRECTNESS_CONFIG)
        correctness = {"config": correctness_config, "runner_inputs": len(inputs),
                       "passed_candidates": len(passed_checks), "finite_checked": len(finite_ids),
                       "correctness_wall_s": sum(row["correctness_wall_s"] for row in checks
                                                 if row["event"] == "correctness_result")}
    architecture_validation["passed"] = not architecture_validation["errors"]
    measurement = _parse_scheduler_log(scheduler_log)
    ga_t2_stats = {str(key): int(value) for key, value in get_ga_t2_stats().items()}
    if args.ga_t2_mode == "directed":
        proposal_total = ga_t2_stats.get("proposal_total", 0)
        proposal_accounted = ga_t2_stats.get("proposal_known", 0) + ga_t2_stats.get(
            "proposal_unknown", 0
        )
        if proposal_total != proposal_accounted:
            architecture_validation["errors"].append(
                "G1 proposal accounting mismatch: "
                f"total={proposal_total}, known+unknown={proposal_accounted}"
            )
        architecture_validation["passed"] = not architecture_validation["errors"]
    accounted_s = search["search_s"] + measurement["builder_s"] + measurement["runner_and_update_s"]

    result = {
        "schema_version": 13,
        "status": (
            "invalid_architecture"
            if not architecture_validation["passed"]
            else ("smoke" if args.repetitions_note == "smoke" else args.repetitions_note)
        ),
        "mode": args.mode,
        "mode_value": mode_value,
        "constraint_sample_mode": args.constraint_sample_mode,
        "constraint_sample_mode_value": constraint_sample_mode_value,
        "hard_filter_position": args.hard_filter_position,
        "soft_mode": args.soft_mode,
        "soft_mode_value": soft_mode_value,
        "soft_weight": soft_weight,
        "draft_mode": args.draft_mode,
        "draft_mode_value": draft_mode_value,
        "draft_beta": args.draft_beta,
        "draft_eps": args.draft_eps,
        "draft_ga_mode": args.draft_ga_mode,
        "draft_ga_mode_value": draft_ga_mode_value,
        "draft_ga_verify_ratio": args.draft_ga_verify_ratio,
        "draft_ga_direction_trials": args.draft_ga_direction_trials,
        "draft_ga_direction_eps": args.draft_ga_direction_eps,
        "draft_ga_g3_mode": args.draft_ga_g3_mode,
        "draft_ga_g3_mode_value": draft_ga_g3_mode_value,
        "draft_ga_anneal_mode": args.draft_ga_anneal_mode,
        "draft_ga_anneal_t0": args.draft_ga_anneal_t0,
        "draft_ga_anneal_decay": args.draft_ga_anneal_decay,
        "draft_profile": args.draft_profile,
        "ga_t2_mode": args.ga_t2_mode,
        "ga_t2_mode_value": ga_t2_mode_value,
        "g1_implementation": "trace-only-single-replay-v2",
        "seed": args.seed,
        "workload": args.workload,
        "work_dir": str(work_dir),
        **source_identity,
        "tvm_version": getattr(tvm, "__version__", None),
        "loaded_library": str(loaded_library),
        "loaded_library_sha256": loaded_library_sha256,
        "target": str(target),
        "config": {
            "workload": args.workload,
            "check_correctness": args.check_correctness,
            "record_timeline": args.record_timeline,
            "record_model_training": args.record_model_training,
            "trials": args.trials,
            "trials_per_iter": args.trials_per_iter,
            "population_size": args.population_size,
            "init_min_unmeasured": args.init_min_unmeasured,
            "exact_init_population": args.exact_init_population,
            "init_sample_batch_size": args.init_sample_batch_size,
            "genetic_iters": args.genetic_iters,
            "constraint_sample_mode": args.constraint_sample_mode,
            "hard_filter_mode": args.mode,
            "hard_filter_position": args.hard_filter_position,
            "soft_mode": args.soft_mode,
            "soft_weight": soft_weight,
            "draft_mode": args.draft_mode,
            "draft_beta": args.draft_beta,
            "draft_eps": args.draft_eps,
            "draft_ga_mode": args.draft_ga_mode,
            "draft_ga_verify_ratio": args.draft_ga_verify_ratio,
            "draft_ga_direction_trials": args.draft_ga_direction_trials,
            "draft_ga_direction_eps": args.draft_ga_direction_eps,
            "draft_ga_g3_mode": args.draft_ga_g3_mode,
            "draft_ga_g3_fixed_config": G3_FIXED_CONFIG,
            "draft_ga_anneal_mode": args.draft_ga_anneal_mode,
            "draft_ga_anneal_t0": args.draft_ga_anneal_t0,
            "draft_ga_anneal_decay": args.draft_ga_anneal_decay,
            "draft_profile": {
                "name": args.draft_profile,
                "peak_gflops": args.draft_peak_gflops,
                "global_memory_gbps": args.draft_memory_gbps,
                "sm_count": args.draft_sm_count,
                "active_blocks_per_sm": args.draft_active_blocks_per_sm,
                "sm_partitions": args.draft_sm_partitions,
                "warp_size": 32,
                "register_capacity": args.draft_reg_cap,
            },
            "ga_t2_mode": args.ga_t2_mode,
            "g1_implementation": "trace-only-single-replay-v2",
            "eps_greedy": args.eps_greedy,
            "num_tuning_cores": 1,
            "builder_workers": 1,
            "build_timeout_s": args.build_timeout,
            "run_timeout_s": args.run_timeout,
            "evaluator": {"number": 3, "repeat": 1, "min_repeat_ms": 100},
            "xgb_warmup_samples": args.xgb_warmup_samples,
            "xgb_adaptive_training": False,
        },
        "timing": {
            "setup_s": setup_s,
            "preparation_from_driver_entry_s": preparation_s,
            "tune_call_s": tune_s,
            "search_s": search["search_s"],
            "builder_s": measurement["builder_s"],
            "runner_and_update_s": measurement["runner_and_update_s"],
            "unaccounted_inside_tune_s": max(0.0, tune_s - accounted_s),
        },
        "search": search,
        "architecture_validation": architecture_validation,
        "timeline": timeline,
        "correctness": correctness,
        "budget": {"requested_submitted": args.trials, "actual_submitted": actual_submitted,
                   "budget_completed": actual_submitted == args.trials,
                   "terminal_status": "budget_completed" if actual_submitted == args.trials else
                                      ("early_exhaustion" if actual_submitted < args.trials else "over_budget")},
        "ga_t2_stats": ga_t2_stats,
        "measurement": {
            **measurement,
            "database_records": len(records),
            "finite_records": len(finite_latencies_s),
            "best_latency_us": min(finite_latencies_s) * 1e6 if finite_latencies_s else None,
            "record_latencies_us": [value * 1e6 if value is not None else None for value in latencies_s],
            "trace_hashes": trace_hashes,
            "callback_batches": result_capture.batches,
            "batch_summaries": _summarize_callback_batches(result_capture.batches),
            "xgb_data_size": int(cost_model.data_size),
            "xgb_trained": cost_model.booster is not None,
        },
    }
    result_path = work_dir / "result.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_json(work_dir / "run_state.json", {
        "status": "completed" if architecture_validation["passed"] else "invalid_architecture",
        "budget": result["budget"], "gpu_after": gpu_status(),
    })

    filt = search["totals"]["filter"]
    print(
        f"hard={args.mode}@{args.hard_filter_position} "
        f"sample={args.constraint_sample_mode} seed={args.seed} work_dir={work_dir}"
    )
    print(
        f"search={search['search_s']:.1f}s builder={measurement['builder_s']:.1f}s "
        f"runner/update={measurement['runner_and_update_s']:.1f}s tune={tune_s:.1f}s"
    )
    print(
        f"thread checked={filt['checked']} unknown={filt['unknown']} "
        f"predicted={filt['predicted']} enforced={filt['enforced']} TP/FP/FN/TN="
        f"{filt['tp']}/{filt['fp']}/{filt['fn']}/{filt['tn']}"
    )
    print(
        f"hard analysis={filt['analysis_time_us']} us "
        f"VerifyGPUCode={filt['verify_gpu_time_us']} us"
    )
    print(
        "hard predicted low/high/hardware="
        f"{filt['predicted_low']}/{filt['predicted_high']}/{filt['predicted_hardware']} "
        "shared proxy predicted/TP/FP/FN="
        f"{filt['shared_predicted']}/{filt['shared_tp']}/{filt['shared_fp']}/"
        f"{filt['shared_fn']}"
    )
    print(
        f"VerifyGPUCode failures={search['totals']['verify_gpu_fail']} "
        f"records={len(records)} finite={len(finite_latencies_s)} "
        f"best_us={result['measurement']['best_latency_us']}"
    )
    print(f"GA T2 stats={ga_t2_stats}")
    print(f"constraint sampling stats={search['totals']['constraint_sampling']}")
    print(f"hard-position stats={search['totals']['hard_positions']}")
    print(f"postproc time us={search['totals']['postproc_time_us']}")
    print(f"soft filter stats={search['soft_filter']}")
    print(f"Pruner draft stats={search['draft_model']}")
    print(f"Draft-guided GA stats={search['draft_ga']}")
    print(f"Draft direction stats={search['draft_direction']}")
    print(f"Draft G3 stats={search['draft_g3']}")
    print(f"Draft annealing stats={search['draft_annealing']}")
    print(f"Architecture contract={architecture_validation}")
    print(f"result={result_path}")
    if not architecture_validation["passed"]:
        raise RuntimeError(
            "Draft-GA architecture contract failed; raw result was retained at "
            f"{result_path}: {architecture_validation['errors']}"
        )
    return result_path


def compare_results(args: argparse.Namespace) -> None:
    results = {}
    modes = ("off", "shadow") + (("enforce",) if args.enforce is not None else ())
    for mode in modes:
        path = getattr(args, mode)
        result = json.loads(path.read_text(encoding="utf-8"))
        if result["mode"] != mode:
            raise ValueError(f"{path} says mode={result['mode']}, expected {mode}")
        results[mode] = result

    schema_versions = [int(results[mode].get("schema_version", 1)) for mode in modes]
    all_legacy = all(version < 2 for version in schema_versions)
    if any(version < 2 for version in schema_versions) and not all_legacy:
        raise ValueError("Cannot pair legacy result.json files with schema-version-2 results")

    comparison_warnings = []
    if all_legacy:
        comparison_warnings.append(
            "Legacy schema: DLL/source identity and enforced-count checks are unavailable"
        )
        print(f"WARNING: {comparison_warnings[-1]}", file=sys.stderr)
        comparable_keys = ("seed", "git_head", "target", "config")
    else:
        comparable_keys = (
            "schema_version",
            "seed",
            "git_head",
            "git_status_sha256",
            "git_diff_sha256",
            "git_untracked_source_sha256",
            "source_identity_sha256",
            "tvm_version",
            "loaded_library",
            "loaded_library_sha256",
            "target",
            "config",
        )
        for mode in modes:
            missing = [
                key
                for key in comparable_keys
                if key not in results[mode] or results[mode][key] in (None, "")
            ]
            if missing:
                raise ValueError(f"The {mode} result is missing paired identity fields: {missing}")

    for key in comparable_keys:
        values = [results[mode][key] for mode in modes]
        if any(value != values[0] for value in values[1:]):
            raise ValueError(f"The experiment arms differ in {key}; they are not paired")

    off = results["off"]
    shadow = results["shadow"]
    shadow_filter = shadow["search"]["totals"]["filter"]
    # Database insertion order can differ when runner results complete
    # asynchronously; the measured trace multiset is the semantic invariant.
    same_traces = sorted(off["measurement"]["trace_hashes"]) == sorted(
        shadow["measurement"]["trace_hashes"]
    )
    same_verify_fail = (
        off["search"]["totals"]["verify_gpu_fail"]
        == shadow["search"]["totals"]["verify_gpu_fail"]
    )
    off_accounting = off["search"]["totals"]["accounting"]
    shadow_accounting = shadow["search"]["totals"]["accounting"]
    confusion_total = sum(shadow_filter[name] for name in ("tp", "fp", "fn", "tn"))
    # Accept either documented convention: Unknown may be included in the
    # pass-side FN/TN buckets, or kept as a disjoint fifth bucket.
    checked_is_consistent = shadow_filter["checked"] in (
        confusion_total,
        confusion_total + shadow_filter["unknown"],
    )
    counters_consistent = (
        shadow_filter["predicted"] == shadow_filter["tp"] + shadow_filter["fp"]
        and checked_is_consistent
        and shadow_filter["unknown"] <= shadow_filter["checked"]
    )
    gate_checks = {
        "zero_false_positives": shadow_filter["fp"] == 0,
        "off_shadow_same_measured_trace_multiset": same_traces,
        "off_shadow_same_verify_failures": same_verify_fail,
        "off_candidate_accounting_conserves": off_accounting["conservation_ok"],
        "shadow_candidate_accounting_conserves": shadow_accounting["conservation_ok"],
        "off_shadow_same_candidate_accounting": off_accounting == shadow_accounting,
        "counters_consistent": counters_consistent,
        # If this is false, the workload did not exercise the rejection path.
        # It is not evidence that the rule is wrong, but enforce mode would be
        # an uninformative experiment on this corpus.
        "at_least_one_true_positive": shadow_filter["tp"] > 0,
    }

    summary = {
        "seed": off["seed"],
        "config": off["config"],
        "identity_verified": not all_legacy,
        "comparison_warnings": comparison_warnings,
        "gate_to_enforce": all(gate_checks.values()),
        "gate_checks": gate_checks,
        "shadow_filter": shadow_filter,
        "search_s": {mode: results[mode]["timing"]["search_s"] for mode in modes},
        "verify_gpu_fail": {
            mode: results[mode]["search"]["totals"]["verify_gpu_fail"] for mode in modes
        },
        "candidate_accounting": {
            mode: results[mode]["search"]["totals"]["accounting"] for mode in modes
        },
        "best_latency_us": {
            mode: results[mode]["measurement"]["best_latency_us"] for mode in modes
        },
    }
    if "enforce" in results:
        enforce = results["enforce"]
        enforce_filter = enforce["search"]["totals"]["filter"]
        if not all_legacy:
            enforced = enforce_filter.get("enforced")
            predicted = enforce_filter.get("predicted")
            if enforced != predicted or not isinstance(enforced, int):
                raise ValueError(
                    "The enforce arm did not apply every predicted rejection: "
                    f"predicted={predicted}, enforced={enforced}"
                )
        summary["enforce_filter"] = enforce_filter
        summary["enforce_search_speedup_vs_off"] = (
            off["timing"]["search_s"] / enforce["timing"]["search_s"]
            if enforce["timing"]["search_s"]
            else None
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary["gate_to_enforce"]:
        failed = [name for name, passed in gate_checks.items() if not passed]
        raise RuntimeError(f"Shadow-to-enforce gate failed: {failed}")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Run one mode in this fresh process")
    run.add_argument("--mode", choices=MODE_TO_VALUE, required=True)
    run.add_argument(
        "--constraint-sample-mode", choices=MODE_TO_VALUE, default="off"
    )
    run.add_argument(
        "--hard-filter-position", choices=HARD_FILTER_POSITIONS, default="late"
    )
    run.add_argument("--soft-mode", choices=SOFT_MODE_TO_VALUE, default="off")
    run.add_argument("--soft-weight", type=_validate_soft_weight, default=0.2)
    run.add_argument("--draft-mode", choices=DRAFT_MODE_TO_VALUE, default="off")
    run.add_argument("--draft-beta", type=_validate_nonnegative, default=0.2)
    run.add_argument("--draft-eps", type=_validate_probability, default=0.10)
    run.add_argument("--draft-ga-mode", choices=DRAFT_GA_MODE_TO_VALUE, default="off")
    run.add_argument("--draft-ga-verify-ratio", type=_validate_keep_ratio, default=0.25)
    run.add_argument(
        "--draft-ga-direction-trials",
        type=_validate_positive_int,
        default=4,
        help="Deprecated compatibility field; trace-only G1 always performs one full replay",
    )
    run.add_argument(
        "--draft-ga-direction-eps",
        type=_validate_probability,
        default=0.10,
        help="Deprecated compatibility field; no longer used by trace-only G1",
    )
    run.add_argument(
        "--draft-ga-g3-mode",
        choices=DRAFT_GA_G3_MODE_TO_VALUE,
        default="off",
        help="Runner-measured mutator credit: off, telemetry-only shadow, or UCB1 enforce",
    )
    run.add_argument(
        "--draft-ga-anneal-mode",
        choices=DRAFT_GA_ANNEAL_MODE_TO_VALUE,
        default="off",
    )
    run.add_argument("--draft-ga-anneal-t0", type=_validate_nonnegative, default=0.10)
    run.add_argument(
        "--draft-ga-anneal-decay", type=_validate_keep_ratio, default=0.95
    )
    run.add_argument("--draft-profile", default="rtx3050ti_sm86_explicit_v1")
    run.add_argument("--draft-peak-gflops", type=_validate_positive, default=10752.0)
    run.add_argument("--draft-memory-gbps", type=_validate_positive, default=192.0)
    run.add_argument("--draft-sm-count", type=int, default=20)
    run.add_argument("--draft-active-blocks-per-sm", type=int, default=1)
    run.add_argument("--draft-sm-partitions", type=int, default=4)
    run.add_argument("--draft-reg-cap", type=_validate_positive, default=128.0)
    run.add_argument("--ga-t2-mode", choices=GA_T2_MODE_TO_VALUE, default="off")
    run.add_argument(
        "--workload",
        choices=(
            "matmul128",
            "matmul256",
            "reduction128",
            "relu128",
            "conv2d",
            "conv2d_nchw3x3",
            "reduction4096",
            "elementwise128",
            "fusion128",
        ),
        default="matmul128",
        help="TIR workload; matmul128 preserves the historical experiment default",
    )
    run.add_argument("--seed", type=int, required=True)
    run.add_argument("--run-id", required=True, help="Unique paired-run label, e.g. pair0")
    run.add_argument("--work-root", type=Path, default=DEFAULT_WORK_ROOT)
    run.add_argument("--temp-dir", type=Path, default=Path("F:/tvm_tmp"))
    run.add_argument(
        "--cuda-home",
        type=Path,
        default=Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6"),
    )
    run.add_argument("--trials", type=int, default=8)
    run.add_argument("--trials-per-iter", type=int, default=8)
    run.add_argument("--population-size", type=int, default=512)
    run.add_argument("--init-min-unmeasured", type=int, default=50)
    run.add_argument(
        "--exact-init-population",
        action="store_true",
        help="Fill and retain exactly the requested unmeasured population before GA",
    )
    run.add_argument("--init-sample-batch-size", type=_validate_positive_int, default=64)
    run.add_argument("--genetic-iters", type=int, default=4)
    run.add_argument("--eps-greedy", type=_validate_probability, default=0.10)
    run.add_argument(
        "--xgb-warmup-samples",
        type=_validate_nonnegative_int,
        default=100,
        help="Set to 0 for a two-batch experiment where batch 2 uses the model trained on batch 1",
    )
    run.add_argument("--build-timeout", type=float, default=30.0)
    run.add_argument("--run-timeout", type=float, default=30.0)
    run.add_argument("--record-timeline", action="store_true", help="Persist monotonic batch/callback boundaries")
    run.add_argument("--record-model-training", action="store_true",
                     help="Observe accumulated XGB state and existing train/next-batch metrics")
    run.add_argument("--check-correctness", action="store_true", help="Check each MatMul candidate before timing")
    run.add_argument(
        "--repetitions-note",
        default="smoke",
        help="Label only; a single three-arm run is not a performance conclusion",
    )
    run.set_defaults(func=run_experiment)

    compare = subparsers.add_parser(
        "compare", help="Validate off+shadow before enforce; optionally compare the full triplet"
    )
    compare.add_argument("--off", type=Path, required=True)
    compare.add_argument("--shadow", type=Path, required=True)
    compare.add_argument("--enforce", type=Path)
    compare.set_defaults(func=compare_results)
    return parser


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()
    try:
        args.func(args)
    except BaseException:
        if hasattr(args, "_active_work_dir"):
            write_json(args._active_work_dir / "run_state.json", {
                "status": "failed", "exception": traceback.format_exc(),
            })
        raise


if __name__ == "__main__":
    main()
