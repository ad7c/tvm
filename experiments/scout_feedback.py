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
"""Experimental bounded feedback GA. No implicit experiment launch on import."""

import gzip
import json
import math
import time
from collections import defaultdict, deque

import numpy as np


def mixed_pool(records, scores, quotas, rng):
    """Disjoint score/diversity/random quotas; unknowns remain eligible."""
    ranked = sorted(records, key=lambda k: scores.get(k, -math.inf), reverse=True)
    selected = ranked[:quotas[0]]
    remaining = [k for k in ranked if k not in selected]
    rng.shuffle(remaining)
    buckets = {records[k]["bucket"] for k in selected}
    diverse = []
    for key in remaining:
        if records[key]["bucket"] not in buckets:
            diverse.append(key)
            buckets.add(records[key]["bucket"])
            if len(diverse) == quotas[1]:
                break
    if quotas[1] == 0:
        diverse = []
    # Fill unavailable diversity buckets without sacrificing the random quota.
    diverse += [k for k in remaining if k not in diverse][:max(0, quotas[1] - len(diverse))]
    selected += diverse
    remaining = [k for k in remaining if k not in diverse]
    random_keys = remaining[:quotas[2]]
    return selected + random_keys, dict(zip(selected + random_keys,
        ["score"] * min(quotas[0], len(ranked)) + ["diversity"] * len(diverse)
        + ["random"] * len(random_keys)))


def percentiles(keys, scores):
    """Equal scores get equal midranks, including a fully tied population."""
    values = sorted(set(scores[k] for k in keys))
    result = {}
    offset = 0
    for value in values:
        tied = [k for k in keys if scores[k] == value]
        rank = (offset + (len(tied) - 1) / 2) / max(1, len(keys) - 1)
        result.update(dict.fromkeys(tied, rank))
        offset += len(tied)
    return result


def select_measurements(pool, archive, draft, learned, warm, rng, count=10):
    """8 best + 1 genuine rank-disagreement challenge + 1 archive random."""
    count = min(count, len(archive))
    best_count = max(0, count - 2)
    score = learned if warm else draft
    if warm:
        best = sorted(pool, key=lambda k: score[k], reverse=True)[:best_count]
    else:
        subset = {k:archive[k] for k in pool}
        best, _ = mixed_pool(subset, draft, (best_count // 2, best_count - best_count // 2, 0), rng)
    roles = dict.fromkeys(best, "xgb_best" if warm else "draft_warmup")
    if warm and len(best) < count:
        xp, dp = percentiles(pool, learned), percentiles(pool, draft)
        challenges = [k for k in pool if k not in roles and xp[k] >= .75 and xp[k] - dp[k] >= .25]
        if challenges:
            key = max(challenges, key=lambda k: xp[k] - dp[k])
            best.append(key)
            roles[key] = "challenge"
    remaining = [k for k in archive if k not in roles]
    rng.shuffle(remaining)
    for key in remaining[:count - len(best)]:
        best.append(key)
        roles[key] = "random"
    return best, roles


class Residual:
    """Family ridge residual with prequential validation and bounded influence."""

    def __init__(self):
        self.samples = deque(maxlen=512)
        self.seen = set()
        self.fit = None
        self.version = 0
        self.active = False
        self.new = 0
        self.validation = []
        self.bad_windows = 0

    def predict(self, row, shadow=False):
        raw = row["raw"]
        if raw is None or self.fit is None:
            return raw
        mean, scale, weights = self.fit
        shape = float(((np.asarray(row["features"]) - mean) / scale) @ weights[1:])
        alpha = .5 if shadow or self.active else 0.0
        # Intercept is family-wide: retain it for diagnostics, omit from ranking.
        return raw * math.exp(-alpha * float(np.clip(shape, -math.log(2), math.log(2))))

    def update(self, rows):
        """Rows belong to one task/batch. Evaluate BEFORE adding labels or refitting."""
        valid = [r for r in rows if r["raw"] is not None and r["key"] not in self.seen
                 and math.isfinite(r["seconds"]) and r["seconds"] > 0]
        event = {"version_before": self.version, "active_before": self.active,
                 "validation": [], "new_samples": [r["key"] for r in valid]}
        if self.fit is not None:
            for i, a in enumerate(valid):
                for b in valid[i + 1:]:
                    if max(a["seconds"], b["seconds"]) / min(a["seconds"], b["seconds"]) <= 1.05:
                        continue
                    truth = a["seconds"] < b["seconds"]
                    def concordance(sa, sb):
                        return .5 if sa == sb else float((sa > sb) == truth)
                    pair = [concordance(a["raw"], b["raw"]),
                            concordance(self.predict(a, True), self.predict(b, True))]
                    self.validation.append(pair)
                    event["validation"].append({"a": a["key"], "b": b["key"], "concordance": pair})
        for row in valid:
            self.samples.append(row)
            self.seen.add(row["key"])
        self.new += len(valid)
        if len(self.samples) >= 64 and (self.fit is None or self.new >= 16):
            if self.fit is not None:
                if len(self.validation) >= 20:
                    raw, corrected = np.mean(self.validation, axis=0)
                    event["validation_concordance"] = [float(raw), float(corrected)]
                    if corrected >= raw + .05:
                        self.active = True
                        self.bad_windows = 0
                    elif corrected < raw:
                        self.bad_windows += 1
                        if self.bad_windows >= 2:
                            self.active = False
                self.validation = []
            x = np.asarray([r["features"] for r in self.samples], dtype=float)
            # Native legacy proxy is nominal ms: convert to seconds, not a truth claim.
            y = np.asarray([math.log(r["seconds"] * r["raw"] * 1000) for r in self.samples])
            mean, scale = x.mean(0), x.std(0)
            scale[scale < 1e-8] = 1.0
            z = np.column_stack([np.ones(len(x)), (x - mean) / scale])
            penalty = np.diag([0.0] + [.1] * x.shape[1])
            weights = np.linalg.solve(z.T @ z / len(z) + penalty, z.T @ y / len(z))
            self.fit = (mean, scale, weights)
            self.new = 0
            self.version += 1
            event["fit"] = {"mean": mean.tolist(), "scale": scale.tolist(),
                            "weights": weights.tolist(), "keys": [r["key"] for r in self.samples]}
        event.update(version_after=self.version, active_after=self.active)
        return event


def make_strategy(core, residuals, output, arm, deadline, emit, model_state=None):
    """Use native replay/constraint/postprocessing and existing CUDA mutator weights."""
    import tvm
    from tvm_ffi import structural_hash
    from tvm.ir.utils import derived_object
    from tvm.s_tir import Schedule
    from tvm.s_tir.meta_schedule.arg_info import ArgInfo
    from tvm.s_tir.meta_schedule.search_strategy import EvolutionarySearch, PySearchStrategy
    from tvm.s_tir.meta_schedule.search_strategy.search_strategy import MeasureCandidate
    from model_scout_pool_study import replay_trace_record, trace_record

    replay = tvm.get_global_func("s_tir.meta_schedule.testing.ScoutReplay")
    mutate = tvm.get_global_func("s_tir.meta_schedule.testing.ScoutMutate")
    features = tvm.get_global_func("s_tir.meta_schedule.testing.ScoutDraftFeatures")

    @derived_object
    class FeedbackStrategy(PySearchStrategy):
        def __init__(self):
            self.inner = EvolutionarySearch(population_size=64, genetic_num_iters=2)
            self.archive = {}
            self.measured = set()
            self.history = []
            self.pending = {}
            self.batch = 0
            self.sent = 0
            self.empty = 0
            self.rng = None

        def _initialize_with_tune_context(self, context):
            self.ctx = context
            self.inner._initialize_with_tune_context(context)
            self.rng = np.random.default_rng(abs(int(context.rand_state)))
            name = str(context.task_name).lower()
            self.family = "conv" if "conv" in name else "matmul" if any(s in name for s in ("matmul", "dense")) else "reduction" if any(s in name for s in ("sum", "mean", "pool", "reduce")) else "other"

        def pre_tuning(self, max_trials, num_trials_per_iter, design_spaces, database=None, cost_model=None):
            self.limit = max_trials
            self.batch_size = num_trials_per_iter
            self.inner.pre_tuning(max_trials, num_trials_per_iter, design_spaces, database, cost_model)

        def post_tuning(self):
            self.inner.post_tuning()

        def clone(self):
            return FeedbackStrategy()

        def event(self, kind, **data):
            emit(kind, task=str(self.ctx.task_name), batch=self.batch, **data)

        def restore(self, key):
            if key not in self.live:
                sch = Schedule(self.ctx.mod, seed=1, debug_mask=0)
                replay_trace_record(self.archive[key]["trace"], sch)
                self.live[key] = sch
            return self.live[key]

        def accept(self, sch, source, parent=None):
            if sch is None:
                return None
            # Module equality follows native structural identity, rather than trace text.
            key = str(self.ctx.task_name) + ":" + str(structural_hash(sch.mod))
            if key in self.measured and source != "history":
                return None
            if key not in self.archive:
                begin = time.perf_counter()
                vals = [float(x) for x in features(sch.mod)]
                known = bool(vals[0]) and all(math.isfinite(x) for x in vals) and vals[1] > 0
                raw = vals[1] if known else None
                trace_key, trace = trace_record(type("C", (), {"sch": sch})(), str(self.ctx.task_name))
                # Broad trace/launch buckets, independent of the scalar draft score.
                decisions = []
                for inst, decision in sch.trace.decisions.items():
                    if str(inst.kind.name) in ("SamplePerfectTile", "SampleCategorical"):
                        decisions.append(str(decision))
                bucket = str((int(vals[2] / math.log(2)), tuple(decisions[:3])))
                self.archive[key] = {"key": key, "trace": trace, "trace_sha": trace_key,
                                     "raw": raw, "features": vals[2:] if known else None,
                                     "bucket": bucket, "source": source, "parent": parent,
                                     "priority": float(self.rng.random())}
                with gzip.open(output / "candidate_traces.jsonl.gz", "at", encoding="utf-8") as stream:
                    stream.write(json.dumps({"task": str(self.ctx.task_name), **self.archive[key]}) + "\n")
                self.event("candidate", key=key, source=source, parent=parent, raw=raw,
                           features=vals[2:], feature_s=time.perf_counter() - begin)
            self.live[key] = sch
            return key

        def generate_one(self, source, parent=None):
            if self.attempts >= 1024 or time.time() >= self.batch_deadline:
                return None
            self.attempts += 1
            begin = time.perf_counter()
            trace = None
            if parent is not None:
                trace = mutate(self.inner, self.restore(parent).trace.simplified(True))
                if trace is None:
                    self.event("attempt", source=source, parent=parent, valid=False,
                               seconds=time.perf_counter() - begin, reason="mutator")
                    return None
            sch = replay(self.inner, trace)
            key = self.accept(sch, source, parent)
            self.event("attempt", source=source, parent=parent, valid=sch is not None,
                       key=key, seconds=time.perf_counter() - begin)
            return key

        def fill(self, chosen, count, source, parents=None, weights=None):
            attempts = 0
            while len(chosen) < count and attempts < max(16, 16 * count):
                if self.attempts >= 1024 or time.time() >= self.batch_deadline:
                    break
                attempts += 1
                parent = str(self.rng.choice(parents, p=weights)) if parents else None
                key = self.generate_one(source, parent)
                if key is not None and key not in chosen:
                    chosen.append(key)

        def drafts(self, keys):
            residual = residuals[self.family]
            return {k: residual.predict(self.archive[k]) if self.archive[k]["raw"] is not None
                    else -math.inf for k in keys}

        def score_xgb(self, keys, phase):
            missing = [k for k in keys if k not in self.xcache]
            begin = time.perf_counter()
            if missing:
                candidates = [MeasureCandidate(self.restore(k), ArgInfo.from_entry_func(self.restore(k).mod)) for k in missing]
                values = core.predict(self.ctx, candidates)
                for key, value in zip(missing, values):
                    self.xcache[key] = float(value) if math.isfinite(float(value)) else -1e30
            result = {k: self.xcache[k] for k in keys}
            self.event("xgb", phase=phase, keys=keys, actual_inputs=missing,
                       scores=result, warm=self.warm, model_samples=int(core.data_size),
                       model_state=dict(model_state or {}),
                       seconds=time.perf_counter() - begin)
            return result

        def trim(self):
            eligible = {k:r for k,r in self.archive.items() if k not in self.measured}
            if len(eligible) > 256:
                scores = self.drafts(eligible) if arm == "R6" else {k:self.xcache.get(k, -math.inf) for k in eligible}
                # Persistent random priorities provide a bounded bypass reservoir.
                bypass = sorted(eligible, key=lambda k: eligible[k]["priority"], reverse=True)[:32]
                remainder = {k:r for k,r in eligible.items() if k not in bypass}
                keep, _ = mixed_pool(remainder, scores, (160, 64, 0), self.rng)
                eligible = {k:eligible[k] for k in keep + bypass}
            history = {r[1]:r[2] for r in self.history[:13]}
            self.archive = {**eligible, **history}

        def generate_measure_candidates(self):
            if time.time() >= deadline or self.sent >= self.limit or self.empty >= 3:
                self.event("stop", reason="deadline" if time.time() >= deadline else "trial_limit" if self.sent >= self.limit else "three_bounded_empty_batches")
                return None
            self.batch += 1
            self.batch_deadline = min(deadline, time.time() + 300)
            self.attempts = 0
            self.live, self.xcache = {}, {}
            self.warm = int(core.data_size) >= 100 and core.booster is not None
            residual = residuals[self.family]
            self.event("batch_start", warm=self.warm, residual_version=residual.version,
                       residual_active=residual.active, family=self.family,
                       model_state=dict(model_state or {}))
            population = [r[1] for r in self.history[:13]]
            self.fill(population, 64, "initial")
            elites = []
            for generation in range(3):
                if not population:
                    break
                draft = self.drafts(population)
                scores = draft if arm == "R6" else self.score_xgb(population, "ga_" + str(generation))
                self.event("generation", generation=generation, population=population,
                           draft={k: v if math.isfinite(v) else None for k,v in draft.items()},
                           attempts=self.attempts)
                if generation == 2 or time.time() >= self.batch_deadline:
                    break
                if generation == 1 and arm == "R6" and self.warm:
                    eligible = {k:r for k,r in self.archive.items() if k not in self.measured}
                    middle, sources = mixed_pool(eligible, self.drafts(eligible), (16, 8, 8), self.rng)
                    middle_scores = self.score_xgb(middle, "middle")
                    elites = sorted(middle, key=lambda k: middle_scores[k], reverse=True)[:8]
                    ranked = sorted(population, key=lambda k: scores[k], reverse=True)
                    population = elites + [k for k in ranked if k not in elites][:64-len(elites)]
                    scores = self.drafts(population)
                    self.event("middle", pool=middle, sources=sources, elites=elites,
                               population_size=len(population))
                ranked = sorted(population, key=lambda k: scores[k], reverse=True)
                weights = 1 / np.sqrt(np.arange(1, len(ranked) + 1))
                weights /= weights.sum()
                next_population = []
                if arm == "R6":
                    # Carry corrected elites first; all eight may be parents.
                    carries = list(dict.fromkeys(elites + ranked))[:6]
                    next_population.extend(carries)
                    self.fill(next_population, len(carries) + 45, "guided_mutation", ranked, weights)
                    random_target = min(64, len(next_population) + 6)
                    self.fill(next_population, random_target, "random_parent", population)
                    self.fill(next_population, min(64, len(next_population) + 7), "immigrant")
                else:
                    # Controlled XGB-GA comparator: same native mutators and bounds.
                    next_population.extend(ranked[:10])
                    self.fill(next_population, 64, "xgb_mutation", ranked, weights)
                self.event("offspring", generation=generation + 1, keys=next_population,
                           carry=next_population[:6 if arm == "R6" else 10])
                population = next_population
            eligible = {k:r for k,r in self.archive.items() if k not in self.measured}
            draft = self.drafts(eligible)
            if arm == "R6":
                pool, sources = mixed_pool(eligible, draft, (96, 16, 16), self.rng)
                learned = self.score_xgb(pool, "final") if self.warm else {}
                chosen, roles = select_measurements(pool, eligible, draft, learned, self.warm,
                                                    self.rng, min(10, self.limit-self.sent))
            else:
                # Every candidate generated by the comparator receives learned-model scoring.
                learned_all = self.score_xgb(list(eligible), "archive")
                pool = sorted(eligible, key=lambda k: learned_all[k], reverse=True)[:128]
                sources = dict.fromkeys(pool, "xgb")
                chosen, roles = select_measurements(pool, eligible, learned_all, learned_all,
                                                    True, self.rng, min(10, self.limit-self.sent))
            self.pending = {k:self.archive[k] for k in chosen}
            self.event("selection", pool=pool, pool_sources=sources, chosen=chosen, roles=roles,
                       corrected_draft={k:v if math.isfinite(v) else None for k,v in draft.items()},
                       attempts=self.attempts, actual_population=len(population),
                       time_limited=time.time() >= self.batch_deadline,
                       residual_version=residual.version, residual_active=residual.active)
            result = [MeasureCandidate(self.restore(k), ArgInfo.from_entry_func(self.restore(k).mod)) for k in chosen]
            self.measured.update(chosen)
            self.sent += len(chosen)
            self.empty = self.empty + 1 if not chosen else 0
            self.trim()
            self.live = {}
            return result

        def notify_runner_results(self, candidates, results):
            rows = []
            for candidate, result in zip(candidates, results):
                key = str(self.ctx.task_name) + ":" + str(structural_hash(candidate.sch.mod))
                if key not in self.pending:
                    raise RuntimeError("Feedback is not from this task's outstanding batch")
                if result.error_msg is None and result.run_secs:
                    seconds = float(np.mean([float(x) for x in result.run_secs]))
                    if math.isfinite(seconds) and seconds > 0:
                        row = self.pending[key]
                        rows.append({**row, "seconds": seconds})
                        self.history.append((seconds, key, row))
            self.history.sort(key=lambda r:r[0])
            self.history = self.history[:13]
            # History records must survive the next initial population restore.
            self.archive.update({r[1]:r[2] for r in self.history})
            if arm == "R6":
                self.event("residual_update", family=self.family,
                           update=residuals[self.family].update(rows))
            self.event("feedback", valid=len(rows), best_seconds=self.history[0][0] if self.history else None,
                       rows=[{"key":r["key"], "seconds":r["seconds"]} for r in rows])
            self.pending = {}

    return FeedbackStrategy()
