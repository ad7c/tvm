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
"""Validate and summarize the corrected Draft-LSE + CS/H + G1/G2/G3 experiment.

Formal results must implement this scoring boundary:

* Draft is the only fitness used during every GA generation.
* The learned cost model is absent from GA.
* After GA, exactly one learned-model call ranks the Draft top-K archive.

Results from the superseded per-generation Draft-prefilter experiment remain
readable, but are written to a separately labelled negative-exploration
summary and can never satisfy the formal validation gate.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable


FORMAL_ARM_ORDER = (
    "B",
    "CSH",
    "D",
    "D_G2",
    "D_G1G2",
    "CSH_D_G1G2",
    "D_G1G2_G3_SHADOW",
    "D_G1G2_G3",
    "CSH_D_G1G2_G3",
)
OPTIONAL_FORMAL_ARMS = (
    "D_SHADOW",
    "CSH_D",
    "CSH_D_G2",
    "CSH_D_SHADOW",
)
LEGACY_ARM_ORDER = ("B", "CSH", "D", "CSH_D", "CSH_D_SHADOW")
IDENTITY_FIELDS = (
    "git_head",
    "git_status_sha256",
    "git_diff_sha256",
    "git_untracked_source_sha256",
    "source_identity_sha256",
    "loaded_library",
    "loaded_library_sha256",
    "target",
)
COMMON_CONFIG_FIELDS = (
    "trials",
    "trials_per_iter",
    "population_size",
    "init_min_unmeasured",
    "exact_init_population",
    "init_sample_batch_size",
    "genetic_iters",
    "hard_filter_position",
    "soft_mode",
    "soft_weight",
    "draft_mode",
    "draft_beta",
    "draft_eps",
    "draft_ga_verify_ratio",
    "draft_ga_direction_trials",
    "draft_ga_direction_eps",
    "eps_greedy",
    "xgb_warmup_samples",
    "xgb_adaptive_training",
)


def _config(result: dict[str, Any], field: str, default: Any = None) -> Any:
    return result.get("config", {}).get(field, result.get(field, default))


def _formal_arm(result: dict[str, Any]) -> str | None:
    key = (
        result.get("mode"),
        result.get("constraint_sample_mode"),
        result.get("draft_ga_mode"),
        result.get("ga_t2_mode"),
        result.get("draft_ga_g3_mode", "off"),
        result.get("draft_ga_anneal_mode", "off"),
    )
    return {
        ("off", "off", "off", "off", "off", "off"): "B",
        ("enforce", "enforce", "off", "off", "off", "off"): "CSH",
        ("off", "off", "shadow", "off", "off", "off"): "D_SHADOW",
        ("off", "off", "enforce", "off", "off", "off"): "D",
        ("off", "off", "enforce", "project", "off", "off"): "D_G2",
        ("off", "off", "enforce", "directed", "off", "off"): "D_G1G2",
        ("enforce", "enforce", "shadow", "off", "off", "off"): "CSH_D_SHADOW",
        ("enforce", "enforce", "enforce", "off", "off", "off"): "CSH_D",
        ("enforce", "enforce", "enforce", "project", "off", "off"): "CSH_D_G2",
        ("enforce", "enforce", "enforce", "directed", "off", "off"):
            "CSH_D_G1G2",
        ("off", "off", "enforce", "directed", "shadow", "off"):
            "D_G1G2_G3_SHADOW",
        ("off", "off", "enforce", "directed", "enforce", "off"):
            "D_G1G2_G3",
        ("enforce", "enforce", "enforce", "directed", "enforce", "off"):
            "CSH_D_G1G2_G3",
    }.get(key)


def _legacy_arm(result: dict[str, Any]) -> str | None:
    if "draft_prefilter_mode" not in result:
        return None
    key = (
        result.get("mode"),
        result.get("constraint_sample_mode"),
        result.get("draft_prefilter_mode"),
    )
    return {
        ("off", "off", "off"): "B",
        ("enforce", "enforce", "off"): "CSH",
        ("off", "off", "enforce"): "D",
        ("enforce", "enforce", "enforce"): "CSH_D",
        ("enforce", "enforce", "shadow"): "CSH_D_SHADOW",
    }.get(key)


def _load_results(
    root: Path,
) -> tuple[
    dict[tuple[int, str], tuple[Path, dict[str, Any]]],
    dict[tuple[int, str], tuple[Path, dict[str, Any]]],
    list[Path],
]:
    formal: dict[tuple[int, str], tuple[Path, dict[str, Any]]] = {}
    legacy: dict[tuple[int, str], tuple[Path, dict[str, Any]]] = {}
    ignored: list[Path] = []
    for path in sorted(root.rglob("result.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        seed = int(result["seed"])
        arm = _formal_arm(result)
        destination = formal
        if arm is None:
            arm = _legacy_arm(result)
            destination = legacy
        if arm is None:
            ignored.append(path)
            continue
        key = (seed, arm)
        if key in destination:
            raise ValueError(f"Duplicate seed/arm {key}: {destination[key][0]} and {path}")
        destination[key] = (path, result)
    return formal, legacy, ignored


def _finite(values: Iterable[float | None]) -> list[float]:
    return sorted(
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value)) and float(value) > 0.0
    )


def _latency_stats(values: list[float | None]) -> dict[str, float | None]:
    finite = _finite(values)
    if not finite:
        return {
            "best_us": None,
            "top3_mean_us": None,
            "top5_mean_us": None,
            "median_us": None,
            "mean_us": None,
        }
    middle = len(finite) // 2
    median = finite[middle]
    if len(finite) % 2 == 0:
        median = 0.5 * (finite[middle - 1] + finite[middle])
    return {
        "best_us": finite[0],
        "top3_mean_us": sum(finite[:3]) / min(3, len(finite)),
        "top5_mean_us": sum(finite[:5]) / min(5, len(finite)),
        "median_us": median,
        "mean_us": sum(finite) / len(finite),
    }


def _batch_best_so_far_auc(result: dict[str, Any]) -> float | None:
    """Return the mean batch-level best-so-far latency (lower is better)."""
    best_so_far = math.inf
    curve: list[float] = []
    for batch in result.get("measurement", {}).get("batch_summaries", []):
        value = batch.get("best_latency_us")
        if value is None or not math.isfinite(float(value)) or float(value) <= 0.0:
            continue
        best_so_far = min(best_so_far, float(value))
        curve.append(best_so_far)
    return sum(curve) / len(curve) if curve else None


def _best_batch(result: dict[str, Any]) -> int | None:
    batches = result.get("measurement", {}).get("batch_summaries", [])
    finite = [
        (index + 1, float(batch["best_latency_us"]))
        for index, batch in enumerate(batches)
        if batch.get("best_latency_us") is not None
        and math.isfinite(float(batch["best_latency_us"]))
        and float(batch["best_latency_us"]) > 0.0
    ]
    return min(finite, key=lambda item: item[1])[0] if finite else None


def _g3_generation_summary(result: dict[str, Any]) -> dict[str, Any]:
    """Summarize cold-start/adaptive G3 attempts without inventing per-child transitions."""
    search = result.get("search", {})
    g3 = search.get("draft_g3", {})
    events = g3.get("generation_events")
    if events is None:
        return {"available": False, "reason": "generation_events absent from result schema"}

    arms = {
        int(stats["index"]): stats.get("type")
        for stats in g3.get("by_arm", {}).values()
        if stats.get("index") is not None
    }
    by_phase: dict[str, dict[str, dict[str, int | str | None]]] = {
        "cold_start": {},
        "adaptive": {},
    }
    adaptive_batches: set[int] = set()
    for event in events:
        cold_value = event.get("cold_start")
        # Pre-schema-11 parsers may retain the key with a null value.  Treat
        # that as unknown/cold for diagnostics; schema 11 rejects it earlier.
        cold = int(cold_value) if cold_value is not None else 1
        phase = "cold_start" if cold else "adaptive"
        batch = event.get("batch")
        if not cold and batch is not None:
            adaptive_batches.add(int(batch))
        selected = event.get("selected", {})
        accepted = event.get("accepted", {})
        indexes = {int(index) for index in selected} | {int(index) for index in accepted}
        for index in indexes:
            key = f"{index}:{arms.get(index)}"
            aggregate = by_phase[phase].setdefault(
                key,
                {
                    "index": index,
                    "type": arms.get(index),
                    "selected": 0,
                    "accepted": 0,
                    "not_accepted": 0,
                },
            )
            selected_count = int(selected.get(str(index), selected.get(index, 0)))
            accepted_count = int(accepted.get(str(index), accepted.get(index, 0)))
            aggregate["selected"] += selected_count
            aggregate["accepted"] += accepted_count
            aggregate["not_accepted"] += selected_count - accepted_count

    feedback = [
        {
            "batch": event.get("batch"),
            "credited": event.get("credited"),
            "improved": event.get("improved"),
            "total_outcomes": event.get("total_outcomes"),
            "cold_after_feedback": (
                int(event["cold_start"]) if event.get("cold_start") is not None else None
            ),
        }
        for event in g3.get("feedback_events", [])
    ]
    first_adaptive = min(adaptive_batches) if adaptive_batches else None
    return {
        "available": True,
        "generation_calls": len(events),
        "first_adaptive_batch": first_adaptive,
        "adaptive_batches": sorted(adaptive_batches),
        "by_phase": by_phase,
        "feedback_timeline": feedback,
    }


def _post_adaptive_latency_stats(
    result: dict[str, Any], first_adaptive_batch: int | None
) -> dict[str, float | None]:
    if first_adaptive_batch is None:
        return _latency_stats([])
    callback_batches = result["measurement"].get("callback_batches", [])
    if callback_batches:
        values_us: list[float | None] = []
        for batch in callback_batches[first_adaptive_batch - 1 :]:
            for run_secs in batch.get("runner_run_secs", []):
                finite = _finite(run_secs or [])
                if not finite:
                    values_us.append(None)
                    continue
                middle = len(finite) // 2
                median = finite[middle]
                if len(finite) % 2 == 0:
                    median = 0.5 * (finite[middle - 1] + finite[middle])
                values_us.append(median * 1e6)
        return _latency_stats(values_us)

    # Legacy results may lack callback batches.  Database order is not a strict
    # batch invariant, so this fallback is retained only for old-schema diagnostics.
    per_batch = int(_config(result, "trials_per_iter"))
    values = result["measurement"]["record_latencies_us"]
    return _latency_stats(values[(first_adaptive_batch - 1) * per_batch :])


def _schema11_callback_errors(result: dict[str, Any]) -> list[str]:
    """Validate the callback-batch boundary used by post-adaptive comparisons."""
    if int(result.get("schema_version", 1)) < 11:
        return []

    errors: list[str] = []
    search_batches = int(result["search"]["search_batches"])
    trials = int(_config(result, "trials"))
    trials_per_iter = int(_config(result, "trials_per_iter"))
    measurement = result["measurement"]
    callback_batches = measurement.get("callback_batches")
    batch_summaries = measurement.get("batch_summaries")
    if not isinstance(callback_batches, list):
        return ["schema >= 11 result is missing callback_batches"]
    if not isinstance(batch_summaries, list):
        return ["schema >= 11 result is missing batch_summaries"]

    if len(callback_batches) != search_batches:
        errors.append(
            f"callback batches={len(callback_batches)}, expected search_batches={search_batches}"
        )
    if len(batch_summaries) != search_batches:
        errors.append(
            f"batch summaries={len(batch_summaries)}, expected search_batches={search_batches}"
        )

    total_candidates = 0
    for index, batch in enumerate(callback_batches):
        expected = min(trials_per_iter, max(0, trials - index * trials_per_iter))
        field_lengths = {
            name: len(batch.get(name, []))
            for name in ("builder_errors", "runner_errors", "runner_run_secs")
        }
        total_candidates += field_lengths["runner_errors"]
        if any(length != expected for length in field_lengths.values()):
            errors.append(
                f"callback batch {index + 1}: field lengths={field_lengths}, "
                f"expected candidates={expected}"
            )
    if total_candidates != trials:
        errors.append(f"callback candidates={total_candidates}, expected trials={trials}")

    for index, summary in enumerate(batch_summaries):
        expected = min(trials_per_iter, max(0, trials - index * trials_per_iter))
        if summary.get("batch_index") != index or summary.get("candidates") != expected:
            errors.append(
                f"batch summary {index + 1}: batch_index={summary.get('batch_index')}, "
                f"candidates={summary.get('candidates')}, expected={index}/{expected}"
            )
    return errors


def _speedup(reference: float | None, value: float | None) -> float | None:
    return reference / value if reference is not None and value else None


def _ratio(value: float | None, reference: float | None) -> float | None:
    return value / reference if value is not None and reference else None


def _trace_overlap_summary(
    reference: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    reference_hashes = reference["measurement"].get("trace_hashes", [])
    candidate_hashes = candidate["measurement"].get("trace_hashes", [])
    reference_counter = Counter(reference_hashes)
    candidate_counter = Counter(candidate_hashes)
    multiset_overlap = sum((reference_counter & candidate_counter).values())
    denominator = max(sum(reference_counter.values()), sum(candidate_counter.values()))
    reference_unique = set(reference_hashes)
    candidate_unique = set(candidate_hashes)
    unique_overlap = len(reference_unique & candidate_unique)
    unique_denominator = max(len(reference_unique), len(candidate_unique))
    return {
        "same_trace_multiset": reference_counter == candidate_counter,
        "trace_multiset_overlap": multiset_overlap,
        "trace_multiset_denominator": denominator,
        "trace_multiset_overlap_ratio": (
            multiset_overlap / denominator if denominator else None
        ),
        "reference_unique_traces": len(reference_unique),
        "candidate_unique_traces": len(candidate_unique),
        "unique_trace_overlap": unique_overlap,
        "unique_trace_overlap_ratio": (
            unique_overlap / unique_denominator if unique_denominator else None
        ),
    }


def _same_identity(
    entries: dict[tuple[int, str], tuple[Path, dict[str, Any]]]
) -> dict[str, Any]:
    if not entries:
        return {}
    first = next(iter(entries.values()))[1]
    identity: dict[str, Any] = {}
    for field in IDENTITY_FIELDS:
        expected = first.get(field)
        values = {str(key): result.get(field) for key, (_, result) in entries.items()}
        if any(value != expected for value in values.values()):
            raise ValueError(f"Experiment arms differ in {field}: {values}")
        identity[field] = expected
    return identity


def _check_common_config(
    entries: dict[tuple[int, str], tuple[Path, dict[str, Any]]]
) -> None:
    for seed in sorted({key[0] for key in entries}):
        per_seed = {arm: item for (entry_seed, arm), item in entries.items() if entry_seed == seed}
        reference = next(iter(per_seed.values()))[1]
        for field in COMMON_CONFIG_FIELDS:
            expected = _config(reference, field)
            values = {arm: _config(result, field) for arm, (_, result) in per_seed.items()}
            if any(value != expected for value in values.values()):
                raise ValueError(f"Seed {seed} arms differ in config.{field}: {values}")


def _require_arms(
    entries: dict[tuple[int, str], tuple[Path, dict[str, Any]]], required: tuple[str, ...]
) -> None:
    for seed in sorted({key[0] for key in entries}):
        present = {arm for entry_seed, arm in entries if entry_seed == seed}
        missing = [arm for arm in required if arm not in present]
        if missing:
            raise ValueError(
                f"Seed {seed} is missing formal arms: {missing}; present={sorted(present)}"
            )


def _validate_architecture(arm: str, result: dict[str, Any]) -> dict[str, Any]:
    search = result["search"]
    draft = search.get("draft_ga", {})
    events = draft.get("summary_events", [])
    batches = int(search["search_batches"])
    generations = int(_config(result, "genetic_iters")) + 1
    draft_mode = result["draft_ga_mode"]
    schema_version = int(result.get("schema_version", 1))
    errors: list[str] = []

    if _config(result, "draft_mode") != "off":
        errors.append("legacy post-XGBoost draft ranker was enabled")
    if _config(result, "soft_mode") != "off":
        errors.append("soft filter was enabled and confounds this ablation")
    if len(events) != batches:
        errors.append(f"scoring summaries={len(events)}, expected search_batches={batches}")

    expected_draft_generations = batches * generations if draft_mode != "off" else 0
    if int(draft.get("generation_calls", -1)) != expected_draft_generations:
        errors.append(
            f"Draft generation calls={draft.get('generation_calls')}, "
            f"expected={expected_draft_generations}"
        )

    for index, event in enumerate(events):
        if event.get("ga_generations") != generations:
            errors.append(f"batch {index}: wrong GA generation count")
        if draft_mode == "enforce":
            if event.get("mode") != "draft-enforce":
                errors.append(f"batch {index}: mode={event.get('mode')}")
            if event.get("ga_xgb_calls") != 0 or event.get("ga_xgb_input") != 0:
                errors.append(f"batch {index}: XGBoost participated inside GA")
            if event.get("final_xgb_calls") != 1:
                errors.append(f"batch {index}: final XGBoost calls != 1")
            if event.get("final_xgb_input") != event.get("verify_top_k"):
                errors.append(f"batch {index}: final XGBoost did not receive the full top-K")
            if event.get("draft_archive") != event.get("verify_top_k"):
                errors.append(f"batch {index}: Draft archive did not fill top-K")
        else:
            expected_mode = "baseline" if draft_mode == "off" else "draft-shadow"
            if event.get("mode") != expected_mode:
                errors.append(f"batch {index}: mode={event.get('mode')}")
            if event.get("ga_xgb_calls") != generations:
                errors.append(f"batch {index}: baseline/shadow GA XGBoost calls are incomplete")
            if event.get("final_xgb_calls") != 0:
                errors.append(f"batch {index}: baseline/shadow unexpectedly used final XGBoost")

    if result.get("ga_t2_mode") == "directed":
        direction = search.get("draft_direction", {})
        if draft_mode != "enforce":
            errors.append("directed mutation did not run with Draft-GA enforce")
        if direction.get("calls") != batches:
            errors.append(
                f"Draft direction summaries={direction.get('calls')}, expected={batches}"
            )

    g3_mode = result.get("draft_ga_g3_mode", "off")
    g3 = search.get(
        "draft_g3",
        {
            "feedback_calls": 0,
            "runner_only": True,
            "feedback_totals": {},
            "feedback_events": [],
            "arm_event_records": [],
        },
    )
    if g3_mode != "off":
        if draft_mode != "enforce":
            errors.append("G3 ran without Draft-GA enforce")
        if g3.get("feedback_calls") != batches:
            errors.append(
                f"G3 feedback summaries={g3.get('feedback_calls')}, expected={batches}"
            )
        if not g3.get("runner_only"):
            errors.append("G3 reward source was not exclusively Runner")
        feedback_events_by_batch: dict[int, list[dict[str, Any]]] = {}
        for event in g3.get("feedback_events", []):
            batch = event.get("batch")
            if batch is not None:
                feedback_events_by_batch.setdefault(int(batch), []).append(event)
            if event.get("mode") != g3_mode:
                errors.append(f"G3 batch {batch}: wrong mode={event.get('mode')}")
            if event.get("source") != "runner":
                errors.append(f"G3 batch {batch}: non-Runner reward source")
            funnel = [
                event.get(name)
                for name in (
                    "sent",
                    "child_measured",
                    "with_lineage",
                    "with_measured_parent",
                    "credited",
                )
            ]
            if any(value is None for value in funnel):
                errors.append(f"G3 batch {batch}: incomplete feedback funnel")
            elif any(left < right for left, right in zip(funnel, funnel[1:])):
                errors.append(f"G3 batch {batch}: non-monotonic feedback funnel")
            if event.get("improved") is None or (
                event.get("credited") is not None
                and event["improved"] > event["credited"]
            ):
                errors.append(f"G3 batch {batch}: invalid improvement count")
            if schema_version >= 11 and event.get("cold_start") not in (0, 1):
                errors.append(
                    f"G3 batch {batch}: invalid cold_start={event.get('cold_start')}"
                )
            batch_arms = [
                arm_event
                for arm_event in g3.get("arm_event_records", [])
                if arm_event.get("batch") == batch
            ]
            if not batch_arms:
                errors.append(f"G3 batch {batch}: missing per-arm telemetry")
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
                        name for name in required_arm_fields if arm_event.get(name) is None
                    ]
                    if missing:
                        errors.append(
                            f"G3 batch {batch}: incomplete arm telemetry fields={missing}"
                        )
                if sum(arm_event.get("credited") or 0 for arm_event in batch_arms) != (
                    event.get("credited") or 0
                ):
                    errors.append(f"G3 batch {batch}: credited arm totals do not reconcile")
                if sum(
                    arm_event.get("improved_batch") or 0 for arm_event in batch_arms
                ) != (event.get("improved") or 0):
                    errors.append(f"G3 batch {batch}: improved arm totals do not reconcile")

        if schema_version >= 11:
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

        has_generation_telemetry = (
            "generation_calls" in g3 and "generation_events" in g3
        )
        if schema_version >= 11 and not has_generation_telemetry:
            errors.append("schema >= 11 result is missing G3 generation telemetry")
        if has_generation_telemetry:
            generation_events = g3.get("generation_events", [])
            expected_generation_calls = batches * int(_config(result, "genetic_iters"))
            if g3.get("generation_calls") != expected_generation_calls:
                errors.append(
                    f"G3 generation calls={g3.get('generation_calls')}, "
                    f"expected={expected_generation_calls}"
                )
            events_by_key: dict[tuple[int, int], list[dict[str, Any]]] = {}
            for event in generation_events:
                batch = event.get("batch")
                iteration = event.get("iter")
                if batch is None or iteration is None:
                    errors.append("G3 generation event is missing batch or iter")
                    continue
                key = (int(batch), int(iteration))
                events_by_key.setdefault(key, []).append(event)
                if event.get("mode") != g3_mode:
                    errors.append(
                        f"G3 generation {key}: mode={event.get('mode')}, expected={g3_mode}"
                    )
                if schema_version >= 11 and event.get("cold_start") not in (0, 1):
                    errors.append(
                        f"G3 generation {key}: "
                        f"invalid cold_start={event.get('cold_start')}"
                    )
                selected = event.get("selected", {})
                accepted = event.get("accepted", {})
                if set(selected) != set(accepted):
                    errors.append(f"G3 generation {key}: selected/accepted arms differ")
                else:
                    for index, selected_count in selected.items():
                        accepted_count = accepted.get(index, 0)
                        if int(selected_count) < 0 or int(accepted_count) < 0:
                            errors.append(f"G3 generation {key}: negative arm count")
                        if int(accepted_count) > int(selected_count):
                            errors.append(f"G3 generation {key}: accepted exceeds selected")

            expected_keys = {
                (batch, iteration)
                for batch in range(1, batches + 1)
                for iteration in range(int(_config(result, "genetic_iters")))
            }
            actual_keys = set(events_by_key)
            if actual_keys != expected_keys:
                errors.append(
                    "G3 generation coverage mismatch: "
                    f"missing={sorted(expected_keys - actual_keys)}, "
                    f"unexpected={sorted(actual_keys - expected_keys)}"
                )
            duplicates = sorted(
                key for key, key_events in events_by_key.items() if len(key_events) != 1
            )
            if duplicates:
                errors.append(f"G3 generation duplicate batch/iter keys={duplicates}")

            feedback_cold = {
                int(event["batch"]): int(event["cold_start"])
                for event in g3.get("feedback_events", [])
                if event.get("batch") is not None and event.get("cold_start") is not None
            }
            for batch in range(1, batches + 1):
                batch_events = [
                    event for event in generation_events if event.get("batch") == batch
                ]
                cold_values = {
                    int(event["cold_start"])
                    for event in batch_events
                    if event.get("cold_start") is not None
                }
                if len(cold_values) > 1:
                    errors.append(f"G3 batch {batch}: cold_start changed within generation")
                expected_cold = 1 if batch == 1 else feedback_cold.get(batch - 1)
                if schema_version >= 11 and expected_cold is None:
                    errors.append(f"G3 batch {batch}: prior feedback cold_start is unavailable")
                elif expected_cold is not None and cold_values and cold_values != {expected_cold}:
                    errors.append(
                        f"G3 batch {batch}: generation cold_start={sorted(cold_values)}, "
                        f"prior feedback={expected_cold}"
                    )
    elif any((event.get("credited") or 0) for event in g3.get("feedback_events", [])):
        errors.append("G3 off arm unexpectedly credited measured outcomes")
    elif g3.get("generation_calls", 0) != 0:
        errors.append("G3 off arm unexpectedly emitted generation telemetry")

    driver_contract = result.get("architecture_validation")
    if driver_contract is not None and not driver_contract.get("passed"):
        errors.append("experiment-driver architecture contract failed")

    measurement = result["measurement"]
    errors.extend(_schema11_callback_errors(result))
    trials = int(_config(result, "trials"))
    accounting = search["totals"]["accounting"]
    exact = search.get("exact_init_population", {})
    checks = {
        "candidate_accounting_conserves": bool(accounting["conservation_ok"]),
        "exact_init_targets_met": (
            bool(exact.get("all_targets_met"))
            if _config(result, "exact_init_population")
            else True
        ),
        "finite_measurements_complete": measurement["finite_records"] == trials,
        "zero_build_errors": measurement["build_errors"] == 0,
        "all_trials_sent_to_runner": measurement["sent_to_runner"] == trials,
    }
    failed_checks = [name for name, passed in checks.items() if not passed]
    if failed_checks:
        errors.append(f"run-integrity checks failed: {failed_checks}")
    if errors:
        raise ValueError(f"Formal arm {arm} failed validation: {errors}")
    return {"passed": True, **checks}


def _formal_row(seed: int, arm: str, result: dict[str, Any]) -> dict[str, Any]:
    search = result["search"]
    draft = search["draft_ga"]
    direction = search.get("draft_direction", {})
    direction_totals = direction.get("totals", {})
    g3 = search.get("draft_g3", {})
    g3_totals = g3.get("feedback_totals", {})
    exact = search["exact_init_population"]
    accounting = search["totals"]["accounting"]
    filt = search["totals"]["filter"]
    ga_stats = result.get("ga_t2_stats", {})
    stats = _latency_stats(result["measurement"]["record_latencies_us"])
    trace_hashes = result["measurement"].get("trace_hashes", [])
    generation = _g3_generation_summary(result)
    return {
        "seed": seed,
        "arm": arm,
        "hard_mode": result["mode"],
        "constraint_sample_mode": result["constraint_sample_mode"],
        "draft_ga_mode": result["draft_ga_mode"],
        "draft_ga_g3_mode": result.get("draft_ga_g3_mode", "off"),
        "ga_t2_mode": result["ga_t2_mode"],
        "init_legal_returned": exact.get("returned_total"),
        "init_generated": exact.get("generated_total"),
        "init_quota_time_s": (exact.get("time_us_total") or 0) / 1e6,
        "all_candidates_generated": accounting["generated"],
        "verify_failures": search["totals"]["verify_gpu_fail"],
        "verify_time_s": (filt.get("verify_gpu_time_us") or 0) / 1e6,
        "hard_rejected": filt.get("enforced"),
        "draft_generations": draft["generation_calls"],
        "draft_candidates": draft["candidates_total"],
        "draft_known": draft["known_total"],
        "draft_unknown": draft["unknown_total"],
        "draft_time_s": draft["draft_time_us"] / 1e6,
        "ga_xgb_calls": draft["ga_xgb_calls"],
        "ga_xgb_input": draft["ga_xgb_input"],
        "final_xgb_calls": draft["final_xgb_calls"],
        "final_xgb_input": draft["final_xgb_input"],
        "total_xgb_calls": draft["ga_xgb_calls"] + draft["final_xgb_calls"],
        "total_xgb_input": draft["ga_xgb_input"] + draft["final_xgb_input"],
        "ga_xgb_time_s": draft["ga_xgb_time_us"] / 1e6,
        "final_xgb_time_s": draft["final_xgb_time_us"] / 1e6,
        "direction_proposals": direction_totals.get("proposal_total", 0),
        "direction_replay_pass": direction_totals.get("replay_pass", 0),
        "direction_known": direction_totals.get("known", 0),
        "directed_choices": direction_totals.get("directed_choices", 0),
        "random_choices": direction_totals.get("random_choices", 0),
        "direction_fallback_to_parent": direction_totals.get("fallback_to_parent", 0),
        "g3_feedback_calls": g3.get("feedback_calls", 0),
        "g3_runner_only": g3.get("runner_only", True),
        "g3_sent": g3_totals.get("sent", 0),
        "g3_child_measured": g3_totals.get("child_measured", 0),
        "g3_with_lineage": g3_totals.get("with_lineage", 0),
        "g3_with_measured_parent": g3_totals.get("with_measured_parent", 0),
        "g3_credited": g3_totals.get("credited", 0),
        "g3_improved": g3_totals.get("improved", 0),
        "g3_selected": sum(
            arm_stats.get("selected", 0) for arm_stats in g3.get("by_arm", {}).values()
        ),
        "g3_arm_count": len(g3.get("by_arm", {})),
        "g3_by_arm": g3.get("by_arm", {}),
        "g3_generation_calls": g3.get("generation_calls", 0),
        "g3_generation_telemetry_available": generation["available"],
        "g3_first_adaptive_batch": generation.get("first_adaptive_batch"),
        "g3_adaptive_batches": generation.get("adaptive_batches", []),
        "g3_cold_start_by_arm": generation.get("by_phase", {}).get("cold_start", {}),
        "g3_adaptive_by_arm": generation.get("by_phase", {}).get("adaptive", {}),
        "g2_initial_reject": ga_stats.get("initial_reject", 0),
        "g2_projected": ga_stats.get("projected", 0),
        "g2_projection_failed": ga_stats.get("projection_failed", 0),
        "g1_direction_candidates": ga_stats.get("direction_candidates", 0),
        "g1_directed_choices": ga_stats.get("directed_choices", 0),
        "g1_direction_fallback": ga_stats.get("direction_fallback", 0),
        "search_s": result["timing"]["search_s"],
        "tune_s": result["timing"]["tune_call_s"],
        "builder_s": result["timing"]["builder_s"],
        "runner_update_s": result["timing"]["runner_and_update_s"],
        "finite": result["measurement"]["finite_records"],
        "build_errors": result["measurement"]["build_errors"],
        "runner_errors": (
            result["measurement"]["sent_to_runner"]
            - result["measurement"]["finite_records"]
        ),
        "unique_traces": len(set(trace_hashes)),
        "batch_best_so_far_auc_us": _batch_best_so_far_auc(result),
        **stats,
    }


def _summarize_formal(
    root: Path,
    entries: dict[tuple[int, str], tuple[Path, dict[str, Any]]],
    required_arms: tuple[str, ...],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not entries:
        raise ValueError("No corrected Draft-GA result.json files found")
    _require_arms(entries, required_arms)
    identity = _same_identity(entries)
    _check_common_config(entries)
    validation = {
        f"seed{seed}/{arm}": _validate_architecture(arm, result)
        for (seed, arm), (_, result) in sorted(entries.items())
    }
    arm_rank = {arm: index for index, arm in enumerate(FORMAL_ARM_ORDER + OPTIONAL_FORMAL_ARMS)}
    rows = [
        _formal_row(seed, arm, result)
        for (seed, arm), (_, result) in sorted(
            entries.items(), key=lambda item: (item[0][0], arm_rank.get(item[0][1], 999))
        )
    ]
    rows_by_seed = {
        seed: {row["arm"]: row for row in rows if row["seed"] == seed}
        for seed in sorted({row["seed"] for row in rows})
    }
    comparisons: dict[str, Any] = {}
    g3_comparisons: dict[str, Any] = {}
    for seed, by_arm in rows_by_seed.items():
        baseline = by_arm.get("B")
        if baseline is not None:
            comparisons[str(seed)] = {
                arm: {
                    "search_speedup_vs_B": _speedup(baseline["search_s"], row["search_s"]),
                    "tune_speedup_vs_B": _speedup(baseline["tune_s"], row["tune_s"]),
                    "xgb_call_reduction_vs_B": (
                        1.0 - row["total_xgb_calls"] / baseline["total_xgb_calls"]
                        if baseline["total_xgb_calls"]
                        else None
                    ),
                    "xgb_input_reduction_vs_B": (
                        1.0 - row["total_xgb_input"] / baseline["total_xgb_input"]
                        if baseline["total_xgb_input"]
                        else None
                    ),
                    "best_latency_ratio_vs_B": _ratio(row["best_us"], baseline["best_us"]),
                    "top5_latency_ratio_vs_B": _ratio(
                        row["top5_mean_us"], baseline["top5_mean_us"]
                    ),
                }
                for arm, row in by_arm.items()
                if arm != "B"
            }

        g3_off = by_arm.get("D_G1G2")
        g3_shadow = by_arm.get("D_G1G2_G3_SHADOW")
        g3_enforce = by_arm.get("D_G1G2_G3")
        if g3_off is not None and (g3_shadow is not None or g3_enforce is not None):
            off_result = entries[(seed, "D_G1G2")][1]
            comparison: dict[str, Any] = {}
            if g3_shadow is not None:
                shadow_result = entries[(seed, "D_G1G2_G3_SHADOW")][1]
                shadow_traces = _trace_overlap_summary(off_result, shadow_result)
                comparison["shadow"] = {
                    "is_observational": shadow_traces["same_trace_multiset"],
                    **shadow_traces,
                    "credited": g3_shadow["g3_credited"],
                    "improved": g3_shadow["g3_improved"],
                    "note": (
                        "Diagnostic only: dynamic task assignment may change trajectories even "
                        "when shadow consumes no RNG."
                    ),
                }
            if g3_enforce is not None:
                enforce_result = entries[(seed, "D_G1G2_G3")][1]
                generation = _g3_generation_summary(enforce_result)
                first_adaptive = generation.get("first_adaptive_batch")
                off_post = _post_adaptive_latency_stats(off_result, first_adaptive)
                enforce_post = _post_adaptive_latency_stats(enforce_result, first_adaptive)
                comparison["enforce"] = {
                    "trace_overlap_vs_off": _trace_overlap_summary(
                        off_result, enforce_result
                    ),
                    "search_speedup_vs_off": _speedup(
                        g3_off["search_s"], g3_enforce["search_s"]
                    ),
                    "tune_speedup_vs_off": _speedup(
                        g3_off["tune_s"], g3_enforce["tune_s"]
                    ),
                    "latency_ratio_vs_off": {
                        name: _ratio(g3_enforce[name], g3_off[name])
                        for name in (
                            "best_us",
                            "top3_mean_us",
                            "top5_mean_us",
                            "median_us",
                            "mean_us",
                        )
                    },
                    "best_batch": _best_batch(enforce_result),
                    "off_best_batch": _best_batch(off_result),
                    "generation": generation,
                    "post_adaptive_latency": {
                        "first_batch": first_adaptive,
                        "off": off_post,
                        "enforce": enforce_post,
                        "ratio_vs_off": {
                            name: _ratio(enforce_post[name], off_post[name])
                            for name in enforce_post
                        },
                    },
                    "causal_warning": (
                        "The enforce arm also retries locally failed mutators during cold start. "
                        "Whole-run quality differences therefore cannot be attributed solely "
                        "to measured-reward UCB; use the post-adaptive slice only as a diagnostic."
                    ),
                }
            g3_comparisons[str(seed)] = comparison

    return (
        {
            "schema_version": 3,
            "experiment": "second_point_corrected_draft_lse_g1_g2_g3",
            "status": "formal_architecture_validated",
            "root": str(root),
            "required_arms": list(required_arms),
            "identity": identity,
            "validation": validation,
            "arms": {f"seed{row['seed']}/{row['arm']}": row for row in rows},
            "comparisons": comparisons,
            "g3_comparisons": g3_comparisons,
            "architecture_claim": {
                "ga_fitness": "analytic Draft only in enforce arms",
                "xgboost_inside_ga": 0,
                "xgboost_after_ga": "exactly one call per search batch over Draft top-K",
                "measurement": "highest learned-model candidates are compiled and run",
                "g3_reward": "paired parent/child latency from Runner only",
            },
            "limitations": [
                "Single-workload or single-seed runs are smoke evidence, "
                "not a stable performance conclusion.",
                "G3 enforce combines failed-arm retry with measured-reward UCB; a retry-only "
                "arm is required to separate their effects.",
                "G4 annealing is held off in this G3 ablation.",
                "Wall-clock comparisons include trajectory-dependent replay, lowering, "
                "build, and run work.",
            ],
        },
        rows,
    )


def _legacy_row(seed: int, arm: str, result: dict[str, Any]) -> dict[str, Any]:
    prefilter = result["search"]["draft_prefilter"]
    stats = _latency_stats(result["measurement"]["record_latencies_us"])
    return {
        "seed": seed,
        "arm": arm,
        "hard_mode": result["mode"],
        "constraint_sample_mode": result["constraint_sample_mode"],
        "draft_prefilter_mode": result["draft_prefilter_mode"],
        "prefilter_calls": prefilter.get("calls", 0),
        "draft_candidates_in": prefilter.get("candidates_in_total", 0),
        "xgb_input": prefilter.get("xgb_input_total", 0),
        "draft_time_s": prefilter.get("draft_time_us_total", 0) / 1e6,
        "predict_time_s": prefilter.get("predict_time_us_total", 0) / 1e6,
        "search_s": result["timing"]["search_s"],
        "tune_s": result["timing"]["tune_call_s"],
        **stats,
    }


def _summarize_legacy(
    root: Path, entries: dict[tuple[int, str], tuple[Path, dict[str, Any]]]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rank = {arm: index for index, arm in enumerate(LEGACY_ARM_ORDER)}
    rows = [
        _legacy_row(seed, arm, result)
        for (seed, arm), (_, result) in sorted(
            entries.items(), key=lambda item: (item[0][0], rank.get(item[0][1], 999))
        )
    ]
    return (
        {
            "schema_version": 1,
            "experiment": "superseded_per_generation_draft_prefilter",
            "status": "negative_exploration_wrong_architecture_not_formal_evidence",
            "root": str(root),
            "reason": (
                "These runs sent Draft-filtered candidates to XGBoost every generation; "
                "XGBoost therefore still guided GA and the runs do not implement Draft-LSE."
            ),
            "arms": {f"seed{row['seed']}/{row['arm']}": row for row in rows},
        },
        rows,
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _manifest(root: Path, output_path: Path) -> dict[str, str]:
    files: list[Path] = []
    for pattern in (
        "**/result.json",
        "**/database_workload.json",
        "**/database_tuning_record.json",
        "**/logs/*.log",
    ):
        files.extend(root.glob(pattern))
    manifest = {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(set(files))
        if path.resolve() != output_path.resolve()
    }
    output_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _parse_required_arms(value: str) -> tuple[str, ...]:
    if value.strip().lower() in ("", "none"):
        return ()
    arms = tuple(part.strip() for part in value.split(",") if part.strip())
    unknown = [arm for arm in arms if arm not in FORMAL_ARM_ORDER + OPTIONAL_FORMAL_ARMS]
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown formal arms: {unknown}")
    return arms


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--required-arms",
        type=_parse_required_arms,
        default=FORMAL_ARM_ORDER,
        help=(
            "Comma-separated completeness gate; defaults to the full G1/G2/G3 ablation. "
            "Use 'none' only for an explicitly incomplete smoke run."
        ),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    formal, legacy, ignored = _load_results(root)
    outputs: dict[str, Any] = {"ignored_result_files": [str(path) for path in ignored]}

    if formal:
        summary, rows = _summarize_formal(root, formal, args.required_arms)
        summary_path = root / "paired_summary.json"
        csv_path = root / "paired_summary.csv"
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        _write_csv(csv_path, rows)
        outputs["formal"] = {
            "summary": str(summary_path),
            "csv": str(csv_path),
            "runs": len(rows),
        }

    if legacy:
        legacy_summary, legacy_rows = _summarize_legacy(root, legacy)
        legacy_path = root / "legacy_negative_exploration_summary.json"
        legacy_csv = root / "legacy_negative_exploration_summary.csv"
        legacy_path.write_text(
            json.dumps(legacy_summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        _write_csv(legacy_csv, legacy_rows)
        outputs["legacy_negative_exploration"] = {
            "summary": str(legacy_path),
            "csv": str(legacy_csv),
            "runs": len(legacy_rows),
            "formal_evidence": False,
        }

    if not formal and not legacy:
        raise ValueError(f"No recognized result.json files below {root}")
    manifest = _manifest(root, root / "raw_manifest.sha256.json")
    outputs["manifest"] = str(root / "raw_manifest.sha256.json")
    outputs["manifest_files"] = len(manifest)
    print(json.dumps(outputs, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
