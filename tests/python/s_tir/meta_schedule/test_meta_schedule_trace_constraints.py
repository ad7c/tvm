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
"""Tests for structural constraints inferred directly from schedule traces."""

import os

import tvm
import tvm.testing
from tvm.s_tir import Schedule
from tvm.s_tir import meta_schedule as ms
from tvm.script import tirx as T


@T.prim_func(s_tir=True)
def elementwise(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, [64, 64])
    B = T.match_buffer(b, [64, 64])
    for i, j in T.grid(64, 64):  # type: ignore
        with T.sblock("B"):
            vi, vj = T.axis.remap("SS", [i, j])  # type: ignore
            B[vi, vj] = A[vi, vj] + 1.0  # type: ignore


def _evaluate(sch: Schedule, max_threads_per_block: int = 1024):
    evaluate = tvm.get_global_func(
        "s_tir.meta_schedule.testing.EvaluateTraceConstraints"
    )
    return {
        str(key): int(value)
        for key, value in evaluate(
            sch.trace, max_threads_per_block, 32
        ).items()
    }


def _annotate_contract(sch: Schedule, block, low: int = 32, high: int = 1024):
    sch.annotate(block, "meta_schedule.thread_extent_low_inclusive", low)
    sch.annotate(block, "meta_schedule.thread_extent_high_inclusive", high)


def _two_axis_schedule(i_decision, j_decision, i_index=1, j_index=1):
    sch = Schedule(elementwise, debug_mask="all")
    block = sch.get_sblock("B")
    loop_i, loop_j = sch.get_loops(block)
    i_factors = sch.sample_perfect_tile(
        loop_i, n=len(i_decision), max_innermost_factor=64, decision=i_decision
    )
    j_factors = sch.sample_perfect_tile(
        loop_j, n=len(j_decision), max_innermost_factor=64, decision=j_decision
    )
    i_loops = sch.split(loop_i, factors=i_factors)
    j_loops = sch.split(loop_j, factors=j_factors)
    sch.bind(i_loops[i_index], "threadIdx.x")
    sch.bind(j_loops[j_index], "threadIdx.y")
    _annotate_contract(sch, block)
    return sch


def test_trace_constraint_multiaxis_product_and_boundaries():
    exact_low = _evaluate(_two_axis_schedule([8, 8], [8, 8]))
    assert exact_low["truth"] == 0
    assert exact_low["max_threads"] == 64

    below = _evaluate(_two_axis_schedule([64, 1], [64, 1]))
    assert below["truth"] == 1
    assert below["reject_low"] == 1

    hardware = _evaluate(
        _two_axis_schedule([8, 8], [8, 8]), max_threads_per_block=32
    )
    assert hardware["truth"] == 1
    assert hardware["reject_hardware"] == 1


def test_trace_constraint_uses_structural_tile_indices():
    middle_factors = _evaluate(
        _two_axis_schedule([4, 2, 8], [2, 4, 8], i_index=1, j_index=1)
    )
    assert middle_factors["max_threads"] == 8
    assert middle_factors["reject_low"] == 1

    last_factors = _evaluate(
        _two_axis_schedule([4, 2, 8], [2, 4, 8], i_index=2, j_index=2)
    )
    assert last_factors["truth"] == 0
    assert last_factors["max_threads"] == 64


def test_trace_constraint_resolves_categorical_decision():
    sch = Schedule(elementwise, debug_mask="all")
    block = sch.get_sblock("B")
    loop_i, _ = sch.get_loops(block)
    inner_extent = sch.sample_categorical(
        candidates=[1, 8, 32], probs=[0.2, 0.3, 0.5], decision=1
    )
    _, inner = sch.split(loop_i, factors=[None, inner_extent])
    sch.bind(inner, "threadIdx.x")
    _annotate_contract(sch, block)

    result = _evaluate(sch)
    assert result["truth"] == 1
    assert result["max_threads"] == 8
    assert result["reject_low"] == 1


def test_trace_constraint_unknown_is_not_rejected():
    sch = Schedule(elementwise, debug_mask="all")
    block = sch.get_sblock("B")
    _, loop_j = sch.get_loops(block)
    sch.bind(loop_j, "threadIdx.x")
    _annotate_contract(sch, block)

    result = _evaluate(sch)
    assert result["truth"] == 2
    assert not any(
        result[name]
        for name in ("reject_low", "reject_high", "reject_hardware")
    )


def _thread_tiled_space(sch: Schedule):
    block = sch.get_sblock("B")
    loop_i, loop_j = sch.get_loops(block)
    i0, i1 = sch.split(
        loop_i,
        sch.sample_perfect_tile(loop_i, n=2, max_innermost_factor=64),
    )
    j0, j1 = sch.split(
        loop_j,
        sch.sample_perfect_tile(loop_j, n=2, max_innermost_factor=64),
    )
    sch.reorder(i0, j0, i1, j1)
    sch.bind(sch.fuse(i1, j1), "threadIdx.x")
    _annotate_contract(sch, block)


def test_constraint_aware_sample_init_only_emits_thread_feasible_traces():
    env_names = (
        "TVM_THREAD_CONSTRAINT_SAMPLE_MODE",
        "TVM_THREAD_HARD_FILTER_MODE",
        "TVM_THREAD_HARD_FILTER_POSITION",
    )
    previous = {name: os.environ.get(name) for name in env_names}
    try:
        os.environ["TVM_THREAD_CONSTRAINT_SAMPLE_MODE"] = "enforce"
        os.environ["TVM_THREAD_HARD_FILTER_MODE"] = "off"
        os.environ["TVM_THREAD_HARD_FILTER_POSITION"] = "trace"
        space_generator = ms.space_generator.ScheduleFn(
            sch_fn=_thread_tiled_space,
            sch_rules=[],
            postprocs=[],
            mutator_probs={},
        )
        context = ms.TuneContext(
            mod=elementwise,
            target=tvm.target.Target(
                {
                    "kind": "cuda",
                    "arch": "sm_86",
                    "max_threads_per_block": 1024,
                    "thread_warp_size": 32,
                }
            ),
            space_generator=space_generator,
            search_strategy=ms.search_strategy.EvolutionarySearch(
                population_size=32,
                init_measured_ratio=0.0,
                init_min_unmeasured=8,
                max_fail_count=2,
                genetic_num_iters=0,
                genetic_mutate_prob=0.0,
                genetic_max_fail_count=1,
                eps_greedy=0.0,
            ),
            num_threads=1,
            rand_state=20260904,
        )
        strategy = context.search_strategy
        strategy.pre_tuning(
            max_trials=8,
            num_trials_per_iter=8,
            design_spaces=space_generator.generate_design_space(context.mod),
            database=ms.database.MemoryDatabase(),
            cost_model=ms.cost_model.RandomModel(),
        )
        candidates = strategy.generate_measure_candidates()
        strategy.post_tuning()

        assert candidates is not None
        assert len(candidates) == 8
        for candidate in candidates:
            result = _evaluate(candidate.sch)
            assert result["truth"] == 0
            assert 32 <= result["max_threads"] <= 1024
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


if __name__ == "__main__":
    tvm.testing.main()
