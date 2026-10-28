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
"""Tests for the Pruner-style analytic draft model."""

import io
import logging
import math

import numpy as np
import pytest
import tvm
import tvm.testing
from tvm.ir.utils import derived_object
from tvm.s_tir import meta_schedule as ms
from tvm.s_tir.schedule import Schedule
from tvm.script import tirx as T


def test_pruner_draft_formula_matches_public_psa_implementation():
    score_statement = tvm.get_global_func(
        "s_tir.meta_schedule.testing.ScorePrunerDraftStatement"
    )
    actual = float(
        score_statement(
            600.0,  # compute workload
            256.0,  # register usage
            1024.0,  # DRAM workload
            2.0,  # data reuse penalty
            10.0,  # grid size
            48.0,  # block size
            1000.0,  # peak GFLOP/s
            100.0,  # global-memory GB/s
            20,  # SMs
            1,  # active blocks per SM
            4,  # SM partitions
            32,  # warp size
            128.0,  # register capacity
        )
    )

    raw_warps = 48.0 / 32.0
    scheduled_warps = math.ceil(raw_warps)
    warp_penalty = raw_warps / scheduled_warps
    scheduling_units = 10.0 * scheduled_warps
    scheduling_capacity = 4.0 * 20.0
    peak_penalty = scheduling_units / (
        math.ceil(scheduling_units / scheduling_capacity) * scheduling_capacity
    )
    register_penalty = 256.0 / 128.0
    compute_throughput = 1000.0 * peak_penalty * warp_penalty / register_penalty
    compute_time = 600.0 * (1.0 + 1.0 / 2.0) / compute_throughput / 1e6
    memory_time = 1024.0 / 100.0 * 1000.0 / (1024.0**3)
    expected = 1.0 / max(compute_time, memory_time)
    assert math.isclose(actual, expected, rel_tol=1e-12)


def test_pruner_draft_normalization_unknown_is_neutral_and_candidate_count_is_preserved():
    normalize = tvm.get_global_func(
        "s_tir.meta_schedule.testing.NormalizePrunerDraftScores"
    )
    apply_scores = tvm.get_global_func("s_tir.meta_schedule.testing.ApplyPrunerDraftScores")

    normalized = [float(value) for value in normalize([2.0, 4.0, 100.0], [1, 1, 0])]
    assert normalized == [0.5, 1.0, 1.0]

    learned = [0.9, 0.6, 0.4]
    shadow = [float(value) for value in apply_scores(learned, normalized, 1, 1.0)]
    ranked = [float(value) for value in apply_scores(learned, normalized, 2, 1.0)]
    assert shadow == learned
    assert ranked == [0.45, 0.6, 0.4]
    assert len(ranked) == len(learned)
    assert ranked.index(max(ranked)) == 1


def test_pruner_draft_ga_fitness_is_independent_of_learned_scores():
    normalize = tvm.get_global_func(
        "s_tir.meta_schedule.testing.NormalizePrunerDraftGAFitness"
    )
    fitness = [float(value) for value in normalize([2.0, 4.0, 100.0], [1, 1, 0])]

    assert fitness[0] == 0.5
    assert fitness[1] == 1.0
    assert 0.0 < fitness[2] < fitness[0]


def test_pruner_draft_ga_all_unknown_falls_back_to_uniform_selection():
    normalize = tvm.get_global_func(
        "s_tir.meta_schedule.testing.NormalizePrunerDraftGAFitness"
    )
    fitness = [float(value) for value in normalize([10.0, 20.0, 30.0], [0, 0, 0])]
    assert fitness == [1.0, 1.0, 1.0]


def test_pruner_draft_ga_fitness_ignores_non_finite_and_non_positive_scores():
    normalize = tvm.get_global_func(
        "s_tir.meta_schedule.testing.NormalizePrunerDraftGAFitness"
    )
    fitness = [
        float(value)
        for value in normalize([4.0, float("nan"), float("inf"), 0.0, -1.0], [1] * 5)
    ]

    assert fitness[0] == 1.0
    assert all(0.0 < value < fitness[0] for value in fitness[1:])


def test_pruner_draft_ga_computes_verify_pool_size():
    verify_count = tvm.get_global_func(
        "s_tir.meta_schedule.testing.PrunerDraftGAVerifyCount"
    )
    assert int(verify_count(128, 8, 0.25)) == 32
    assert int(verify_count(16, 8, 0.25)) == 8
    assert int(verify_count(10, 1, 0.21)) == 3
    assert int(verify_count(10, 12, 0.25)) == 12

    with pytest.raises(ValueError):
        verify_count(0, 8, 0.25)
    with pytest.raises(ValueError):
        verify_count(128, 0, 0.25)
    with pytest.raises(ValueError):
        verify_count(128, 8, 0.0)
    with pytest.raises(ValueError):
        verify_count(128, 8, 1.01)


def test_pruner_draft_ga_direction_exploitation_uses_only_draft_scores():
    select = tvm.get_global_func(
        "s_tir.meta_schedule.testing.SelectPrunerDraftMutationDirection"
    )

    assert int(select(True, 4.0, [1, 1, 1], [4.5, 8.0, 6.0], False, 0)) == 1
    assert int(select(True, 8.0, [1, 1], [8.0, 7.0], False, 0)) == -1
    assert int(select(False, 1.0, [0, 1, 1], [100.0, 3.0, 5.0], False, 0)) == 2
    assert int(select(True, 4.0, [0, 0], [100.0, 200.0], False, 0)) == -1


def test_pruner_draft_ga_direction_exploration_can_keep_diversity():
    select = tvm.get_global_func(
        "s_tir.meta_schedule.testing.SelectPrunerDraftMutationDirection"
    )

    # Exploration deliberately may choose a lower-scoring or Draft-unknown legal proposal.
    assert int(select(True, 10.0, [1, 0, 1], [1.0, 100.0, 9.0], True, 0)) == 0
    assert int(select(True, 10.0, [1, 0, 1], [1.0, 100.0, 9.0], True, 1)) == 1

    with pytest.raises(ValueError):
        select(True, 10.0, [1], [9.0], True, 1)


def test_pruner_draft_ga_annealing_uses_relative_draft_score_drop():
    probability = tvm.get_global_func(
        "s_tir.meta_schedule.testing.PrunerDraftAnnealAcceptanceProbability"
    )

    assert float(probability(10.0, 10.0, 0.1)) == 1.0
    assert float(probability(10.0, 12.0, 0.1)) == 1.0
    expected = math.exp(-2.0)
    assert math.isclose(float(probability(10.0, 8.0, 0.1)), expected, rel_tol=1e-12)
    assert math.isclose(float(probability(100.0, 80.0, 0.1)), expected, rel_tol=1e-12)


def test_pruner_draft_ga_annealing_zero_temperature_rejects_only_worse_children():
    probability = tvm.get_global_func(
        "s_tir.meta_schedule.testing.PrunerDraftAnnealAcceptanceProbability"
    )

    assert float(probability(10.0, 9.0, 0.0)) == 0.0
    assert float(probability(10.0, 9.0, -1.0)) == 0.0
    assert float(probability(10.0, 11.0, 0.0)) == 1.0


@tvm.script.ir_module
class KnownStaticGPU:
    @T.prim_func(s_tir=True)
    def main(a: T.handle, c: T.handle) -> None:
        A = T.match_buffer(a, (2, 48, 4), "float32")
        C = T.match_buffer(c, (2, 48), "float32")
        for bx in T.thread_binding(2, thread="blockIdx.x"):
            for tx in T.thread_binding(48, thread="threadIdx.x"):
                for k in range(4):
                    with T.sblock("compute"):
                        vbx, vtx = T.axis.remap("SS", [bx, tx])
                        vk = T.axis.reduce(4, k)
                        T.reads(C[vbx, vtx], A[vbx, vtx, vk])
                        T.writes(C[vbx, vtx])
                        C[vbx, vtx] = C[vbx, vtx] + A[vbx, vtx, vk]


@tvm.script.ir_module
class UnknownDynamicGPU:
    @T.prim_func(s_tir=True)
    def main(a: T.handle, c: T.handle, n: T.int64) -> None:
        A = T.match_buffer(a, (n,), "float32")
        C = T.match_buffer(c, (n,), "float32")
        for tx in T.thread_binding(n, thread="threadIdx.x"):
            with T.sblock("copy"):
                vtx = T.axis.spatial(n, tx)
                T.reads(A[vtx])
                T.writes(C[vtx])
                C[vtx] = A[vtx] + T.float32(1)


@tvm.script.ir_module
class UnknownTiled:
    @T.prim_func(s_tir=True)
    def main(a: T.handle, c: T.handle) -> None:
        A = T.match_buffer(a, (16, 16), "float32")
        C = T.match_buffer(c, (16, 16), "float32")
        for i, j in T.grid(16, 16):
            with T.sblock("copy"):
                vi, vj = T.axis.remap("SS", [i, j])
                T.reads(A[vi, vj])
                T.writes(C[vi, vj])
                C[vi, vj] = A[vi, vj]


def _identity_schedule(_sch):
    return None


def _tile_unknown(sch):
    block = sch.get_sblock("copy")
    i, j = sch.get_loops(block)
    i0, i1 = sch.split(i, sch.sample_perfect_tile(i, n=2))
    j0, j1 = sch.split(j, sch.sample_perfect_tile(j, n=2))
    sch.reorder(i0, j0, i1, j1)


@derived_object
class _RecordingCostModel(ms.cost_model.PyCostModel):
    def __init__(self):
        self.calls = []

    def load(self, _path):
        pass

    def save(self, _path):
        pass

    def update(self, _context, _candidates, _results):
        pass

    def predict(self, _context, candidates):
        self.calls.append(
            [int(candidate.sch.mod["main"].attrs["draft_test_id"]) for candidate in candidates]
        )
        return np.arange(len(candidates), 0, -1, dtype="float64")


def _draft_target():
    return tvm.target.Target(
        {
            "kind": "cuda",
            "arch": "sm_86",
            "max_threads_per_block": 1024,
            "max_num_threads": 1024,
            "max_shared_memory_per_block": 49152,
            "thread_warp_size": 32,
        }
    )


def _tagged_schedule(mod, test_id):
    func = mod["main"].with_attr("draft_test_id", test_id)
    return Schedule(tvm.IRModule({"main": func}))


def _prepare_draft_strategy(
    monkeypatch,
    *,
    mod=KnownStaticGPU,
    schedule_fn=_identity_schedule,
    population_size,
    genetic_num_iters=0,
    mutator_probs=None,
    logger=None,
    direction=False,
):
    for name, value in {
        "TVM_GA_T2_MODE": "3" if direction else "0",
        "TVM_THREAD_CONSTRAINT_SAMPLE_MODE": "off",
        "TVM_THREAD_HARD_FILTER_MODE": "off",
        "TVM_SOFT_FILTER_MODE": "0",
        "TVM_DRAFT_MODE": "0",
        "TVM_DRAFT_GA_MODE": "2",
        "TVM_DRAFT_GA_G3_MODE": "off",
        "TVM_DRAFT_GA_ANNEAL_MODE": "0",
        "TVM_DRAFT_GA_VERIFY_RATIO": "0.01",
        "TVM_DRAFT_GA_DIRECTION_TRIALS": "2",
        "TVM_DRAFT_GA_DIRECTION_EPS": "1.0",
        "TVM_DRAFT_PEAK_GFLOPS": "10000",
        "TVM_DRAFT_MEMORY_GBPS": "500",
        "TVM_DRAFT_SM_COUNT": "16",
    }.items():
        monkeypatch.setenv(name, value)

    cost_model = _RecordingCostModel()
    space_generator = ms.space_generator.ScheduleFn(
        sch_fn=schedule_fn,
        sch_rules=[],
        postprocs=[],
        mutator_probs=mutator_probs or {},
    )
    context = ms.TuneContext(
        mod=mod,
        target=_draft_target(),
        space_generator=space_generator,
        search_strategy=ms.search_strategy.EvolutionarySearch(
            population_size=population_size,
            init_measured_ratio=0.0,
            init_min_unmeasured=population_size,
            max_fail_count=1,
            genetic_num_iters=genetic_num_iters,
            genetic_mutate_prob=1.0 if mutator_probs else 0.0,
            genetic_max_fail_count=2,
            eps_greedy=0.0,
        ),
        rand_state=1,
        num_threads=1,
        logger=logger,
    )
    strategy = context.search_strategy
    assert strategy is not None
    strategy.pre_tuning(
        max_trials=max(1, population_size),
        num_trials_per_iter=max(1, population_size),
        design_spaces=context.generate_design_space(),
        database=ms.database.MemoryDatabase(),
        cost_model=cost_model,
    )
    # State keeps a non-owning C++ pointer to TuneContext.  Return the context
    # so each test holds it through post_tuning.
    return context, strategy, cost_model


def test_pruner_draft_ir_extraction_known_and_dynamic_unknown():
    analyze = tvm.get_global_func("s_tir.meta_schedule.testing.AnalyzePrunerDraftScore")
    profile = (1000.0, 100.0, 20, 1, 4, 32, 128.0)

    known, known_score, statements = analyze(KnownStaticGPU, *profile)
    assert int(known) == 1
    assert float(known_score) > 0.0
    assert int(statements) == 1

    dynamic_known, dynamic_score, _ = analyze(UnknownDynamicGPU, *profile)
    assert int(dynamic_known) == 0
    assert float(dynamic_score) == 1.0


def test_pruner_draft_ga_empty_archive_returns_without_cost_model(monkeypatch):
    _context, strategy, cost_model = _prepare_draft_strategy(
        monkeypatch, population_size=2
    )
    evolve = tvm.get_global_func(
        "s_tir.meta_schedule.SearchStrategyEvolutionarySearchEvolveWithCostModel"
    )
    try:
        assert list(evolve(strategy, [], 1)) == []
        assert cost_model.calls == []

        first = list(evolve(strategy, [_tagged_schedule(UnknownTiled, 7)], 1))
        assert len(first) == 1
        assert cost_model.calls == [[7]]
        cost_model.calls.clear()

        # The FFI wrapper records returned modules as measured.  Re-evolving
        # only that module leaves a non-empty population but an empty archive.
        assert list(evolve(strategy, first, 1)) == []
        assert cost_model.calls == []
    finally:
        strategy.post_tuning()


def test_pruner_draft_ga_unknown_archive_rotates_and_known_stays_prioritized(monkeypatch):
    evolve = tvm.get_global_func(
        "s_tir.meta_schedule.SearchStrategyEvolutionarySearchEvolveWithCostModel"
    )

    def run(population):
        _context, strategy, cost_model = _prepare_draft_strategy(
            monkeypatch, population_size=len(population)
        )
        try:
            results = list(evolve(strategy, population, 2))
            assert len(results) == 2
            assert len(cost_model.calls) == 1
            return cost_model.calls[0]
        finally:
            strategy.post_tuning()

    unknown = [_tagged_schedule(UnknownTiled, index) for index in range(4)]
    # Equal unknown scores use a deterministic increasing tie-breaker, so the
    # archive does not remain locked to candidates 0 and 1.
    assert run(unknown) == [3, 2]
    assert run(unknown) == [3, 2]

    mixed = [_tagged_schedule(KnownStaticGPU, 100), *unknown]
    selected = run(mixed)
    assert selected[0] == 100
    assert selected[1] == 3


def test_pruner_draft_ga_trace_direction_uses_standard_single_child_path(monkeypatch):
    log_stream = io.StringIO()
    logger = logging.getLogger("pruner-draft-all-unknown-direction")
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler(log_stream))

    tagged_mod = _tagged_schedule(UnknownTiled, -1).mod
    context, strategy, cost_model = _prepare_draft_strategy(
        monkeypatch,
        mod=tagged_mod,
        schedule_fn=_tile_unknown,
        population_size=1,
        genetic_num_iters=1,
        mutator_probs={ms.mutator.MutateTileSize(): 1.0},
        logger=logger,
        direction=True,
    )
    try:
        parent = context.generate_design_space()[0]
        parent_trace = parent.trace.as_json(remove_postproc=True)
        evolve = tvm.get_global_func(
            "s_tir.meta_schedule.SearchStrategyEvolutionarySearchEvolveWithCostModel"
        )
        candidates = list(evolve(strategy, [parent], 1))
        assert len(candidates) == 1
        assert candidates[0].trace.as_json(remove_postproc=True) != parent_trace
        assert cost_model.calls == [[-1]]
    finally:
        strategy.post_tuning()

    # G1 now chooses its direction inside MutateTileSize using trace arithmetic.
    # Even with the retired best-of-N environment variables set by the helper,
    # EvolutionarySearch must not replay several Draft-scored proposals.
    assert "Draft GA direction summary:" not in log_stream.getvalue()


if __name__ == "__main__":
    tvm.testing.main()
