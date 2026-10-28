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
# pylint: disable=missing-module-docstring,missing-function-docstring,missing-class-docstring
import operator
import os
from contextlib import contextmanager
from functools import reduce

import tvm
from tvm.s_tir import Schedule
from tvm.s_tir import meta_schedule as ms
from tvm.s_tir.meta_schedule import _ffi_api as ms_ffi_api
from tvm.script import tirx as T
from tvm.target import Target

# pylint: disable=invalid-name, no-member


@T.prim_func(s_tir=True)
def matmul(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, [512, 512])
    B = T.match_buffer(b, [512, 512])
    C = T.match_buffer(c, [512, 512])
    for i, j, k in T.grid(512, 512, 512):  # type: ignore
        with T.sblock("C"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])  # type: ignore
            with T.init():
                C[vi, vj] = 0.0  # type: ignore
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]


@T.prim_func(s_tir=True)
def elementwise(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, [64, 64])
    B = T.match_buffer(b, [64, 64])
    for i, j in T.grid(64, 64):  # type: ignore
        with T.sblock("B"):
            vi, vj = T.axis.remap("SS", [i, j])  # type: ignore
            B[vi, vj] = A[vi, vj] + 1.0  # type: ignore


# pylint: enable=invalid-name, no-member


def _sch(decisions: list[list[int]]) -> Schedule:
    sch = Schedule(matmul, debug_mask="all")
    # pylint: disable=invalid-name
    (d0,) = decisions
    b0 = sch.get_sblock(name="C", func_name="main")
    sch.get_consumers(block=b0)
    b1 = sch.cache_write(block=b0, write_buffer_index=0, storage_scope="global")
    l2, l3, l4 = sch.get_loops(block=b0)
    v5, v6, v7, v8 = sch.sample_perfect_tile(
        loop=l2,
        n=4,
        max_innermost_factor=64,
        decision=d0,
    )
    l9, l10, l11, l12 = sch.split(loop=l2, factors=[v5, v6, v7, v8])
    l17, l18, l19, l20 = sch.split(loop=l3, factors=[8, 4, 8, 2])
    l23, l24 = sch.split(loop=l4, factors=[512, 1])
    sch.reorder(l9, l17, l10, l18, l23, l11, l19, l24, l12, l20)
    sch.reverse_compute_at(block=b1, loop=l18, preserve_unit_loops=True)
    # pylint: enable=invalid-name
    return sch


def _make_mutator(target: Target, mod=matmul) -> ms.Mutator:
    ctx = ms.TuneContext(
        mod=mod,
        target=target,
        space_generator=ms.space_generator.PostOrderApply(
            sch_rules=[],
            postprocs=[],
            mutator_probs={ms.mutator.MutateTileSize(): 1.0},
        ),
    )
    return next(iter(ctx.space_generator.mutator_probs.keys()))


def _gpu_target(max_threads_per_block: int) -> Target:
    return Target(
        {
            "kind": "cuda",
            "max_num_threads": max_threads_per_block,
            "max_shared_memory_per_block": 49152,
            "thread_warp_size": 32,
        }
    )


def _fused_thread_sch() -> Schedule:
    sch = Schedule(elementwise, debug_mask="all")
    block = sch.get_sblock(name="B", func_name="main")
    loop_i, loop_j = sch.get_loops(block=block)
    i0, i1 = sch.split(
        loop=loop_i,
        factors=sch.sample_perfect_tile(
            loop=loop_i, n=2, max_innermost_factor=64, decision=[8, 8]
        ),
    )
    j0, j1 = sch.split(
        loop=loop_j,
        factors=sch.sample_perfect_tile(
            loop=loop_j, n=2, max_innermost_factor=64, decision=[8, 8]
        ),
    )
    sch.reorder(i0, j0, i1, j1)
    sch.bind(sch.fuse(i1, j1), "threadIdx.x")
    return sch


def _multi_axis_thread_sch() -> Schedule:
    sch = Schedule(elementwise, debug_mask="all")
    block = sch.get_sblock(name="B", func_name="main")
    loop_i, loop_j = sch.get_loops(block=block)
    i0, i1 = sch.split(
        loop=loop_i,
        factors=sch.sample_perfect_tile(
            loop=loop_i, n=2, max_innermost_factor=64, decision=[8, 8]
        ),
    )
    j0, j1 = sch.split(
        loop=loop_j,
        factors=sch.sample_perfect_tile(
            loop=loop_j, n=2, max_innermost_factor=64, decision=[8, 8]
        ),
    )
    sch.reorder(i0, j0, i1, j1)
    sch.bind(i1, "threadIdx.x")
    sch.bind(j1, "threadIdx.y")
    return sch


def _single_thread_sch() -> Schedule:
    sch = Schedule(elementwise, debug_mask="all")
    block = sch.get_sblock(name="B", func_name="main")
    loop_i, _ = sch.get_loops(block=block)
    _, i1 = sch.split(
        loop=loop_i,
        factors=sch.sample_perfect_tile(
            loop=loop_i, n=2, max_innermost_factor=64, decision=[2, 32]
        ),
    )
    sch.bind(i1, "threadIdx.x")
    return sch


def _unknown_thread_sch() -> Schedule:
    sch = Schedule(elementwise, debug_mask="all")
    block = sch.get_sblock(name="B", func_name="main")
    loop_i, loop_j = sch.get_loops(block=block)
    sch.split(
        loop=loop_i,
        factors=sch.sample_perfect_tile(
            loop=loop_i, n=2, max_innermost_factor=64, decision=[8, 8]
        ),
    )
    # GetLoops does not encode the original loop extent in the trace, so this
    # binding is intentionally outside the statically supported provenance.
    sch.bind(loop_j, "threadIdx.x")
    return sch


def _apply_with_seed(mutator: ms.Mutator, trace, seed: int):
    return ms_ffi_api.MutatorApply(mutator, trace, seed)


def _analyze_trace(trace, max_threads_per_block: int):
    sch = Schedule(elementwise, debug_mask="all")
    trace.apply_to_schedule(sch, remove_postproc=True)
    analyze = tvm.get_global_func("s_tir.meta_schedule.testing.AnalyzeThreadFeasibility")
    result = analyze(sch.mod, max_threads_per_block, 32, -1)
    known, reject, max_threads = result[0], result[1], result[2]
    return bool(int(known)), bool(int(reject)), int(max_threads)


def _reset_t2_stats():
    tvm.get_global_func("s_tir.meta_schedule.testing.ResetMutateTileSizeThreadStats")()


def _get_t2_stats():
    stats = tvm.get_global_func("s_tir.meta_schedule.testing.GetMutateTileSizeThreadStats")()
    return {str(key): int(value) for key, value in stats.items()}


@contextmanager
def _t2_mode(mode: int):
    old_value = os.environ.get("TVM_GA_T2_MODE")
    os.environ["TVM_GA_T2_MODE"] = str(mode)
    try:
        yield
    finally:
        if old_value is None:
            os.environ.pop("TVM_GA_T2_MODE", None)
        else:
            os.environ["TVM_GA_T2_MODE"] = old_value


def _find_rejecting_seed(mutator: ms.Mutator, trace, max_threads_per_block: int) -> int:
    # Seed zero is a fixed point of TVM's multiplicative LCG.
    for seed in range(1, 257):
        candidate = _apply_with_seed(mutator, trace, seed)
        if candidate is not None and _analyze_trace(candidate, max_threads_per_block)[1]:
            return seed
    raise AssertionError("The test trace did not produce a thread-overflow proposal")


def test_mutate_tile_size_matmul():
    mutator = _make_mutator(
        target=Target({"kind": "llvm", "num-cores": 16}),
    )
    results = {}
    sch = _sch(decisions=[[4, 32, 4, 1]])
    for _ in range(1000):
        trace = mutator.apply(sch.trace)
        assert trace.insts[4].kind.name == "SamplePerfectTile"
        decision = trace.decisions[trace.insts[4]]
        decision = [int(x) for x in decision]
        results[str(decision)] = decision
        assert reduce(operator.mul, decision, 1) == 512
    assert len(results) > 15


def test_mutate_sample_categorical_single_candidate():
    mutator = _make_mutator(
        target=Target({"kind": "llvm", "num-cores": 16}),
    )
    sch = Schedule(matmul, debug_mask="all")
    sch.sample_categorical(candidates=[1], probs=[1.0], decision=0)

    # The mutator finds the SampleCategorical has only one candidate, and thus skips it.
    trace = mutator.apply(sch.trace)
    assert trace is None


def test_thread_projection_preserves_shadow_and_repairs_enforce():
    parent = _fused_thread_sch().trace
    target = _gpu_target(max_threads_per_block=128)

    with _t2_mode(1):
        shadow_mutator = _make_mutator(target=target, mod=elementwise)
    _reset_t2_stats()
    seed = _find_rejecting_seed(shadow_mutator, parent, max_threads_per_block=128)
    shadow = _apply_with_seed(shadow_mutator, parent, seed)
    assert shadow is not None
    assert _get_t2_stats()["initial_reject"] > 0

    with _t2_mode(0):
        off_mutator = _make_mutator(target=target, mod=elementwise)
    off = _apply_with_seed(off_mutator, parent, seed)
    assert off is not None
    assert shadow.as_json(remove_postproc=True) == off.as_json(remove_postproc=True)

    with _t2_mode(2):
        enforce_mutator = _make_mutator(target=target, mod=elementwise)
    _reset_t2_stats()
    projected = _apply_with_seed(enforce_mutator, parent, seed)
    assert projected is not None
    assert _analyze_trace(projected, max_threads_per_block=128) == (True, False, 128)
    stats = _get_t2_stats()
    assert stats["initial_reject"] == 1
    assert stats["projected"] == 1
    assert stats["projection_failed"] == 0

    for inst, decision in projected.decisions.items():
        if inst.kind.name == "SamplePerfectTile":
            assert reduce(operator.mul, (int(value) for value in decision), 1) == 64


def test_thread_direction_uses_slack_and_pressure():
    parent = _fused_thread_sch().trace
    seed = 7

    # With ample margin, mode 3 moves the selected tile toward a larger,
    # still-legal physical thread extent.
    with _t2_mode(3):
        slack_mutator = _make_mutator(
            target=_gpu_target(max_threads_per_block=256), mod=elementwise
        )
    _reset_t2_stats()
    expanded = _apply_with_seed(slack_mutator, parent, seed)
    assert expanded is not None
    known, reject, expanded_threads = _analyze_trace(expanded, max_threads_per_block=256)
    assert known and not reject
    assert 64 < expanded_threads <= 256
    stats = _get_t2_stats()
    assert stats["proposal_total"] == 1
    assert stats["proposal_known"] == 1
    assert stats["proposal_unknown"] == 0
    assert stats["proposal_initial_legal"] + stats["initial_reject"] == 1
    assert stats["direction_candidates"] > 0
    assert stats["directed_choices"] == 1
    assert stats["direction_fallback"] == 0
    assert stats["slack_expand"] == 1
    assert stats["pressure_nonincrease"] == 0

    # At the hardware ceiling, the same random proposal is redirected to a
    # non-increasing thread extent instead of growing past the limit.
    with _t2_mode(3):
        pressure_mutator = _make_mutator(
            target=_gpu_target(max_threads_per_block=64), mod=elementwise
        )
    _reset_t2_stats()
    contracted = _apply_with_seed(pressure_mutator, parent, seed)
    assert contracted is not None
    assert _analyze_trace(contracted, max_threads_per_block=64) == (True, False, 32)
    stats = _get_t2_stats()
    assert stats["directed_choices"] == 1
    assert stats["pressure_nonincrease"] == 1
    assert stats["slack_expand"] == 0

    for candidate in (expanded, contracted):
        for inst, decision in candidate.decisions.items():
            if inst.kind.name == "SamplePerfectTile":
                assert reduce(operator.mul, (int(value) for value in decision), 1) == 64


def test_thread_direction_repairs_rejecting_proposal():
    parent = _fused_thread_sch().trace
    target = _gpu_target(max_threads_per_block=32)

    with _t2_mode(1):
        shadow_mutator = _make_mutator(target=target, mod=elementwise)
    seed = _find_rejecting_seed(shadow_mutator, parent, max_threads_per_block=32)

    with _t2_mode(3):
        directed_mutator = _make_mutator(target=target, mod=elementwise)
    _reset_t2_stats()
    repaired = _apply_with_seed(directed_mutator, parent, seed)
    assert repaired is not None
    assert _analyze_trace(repaired, max_threads_per_block=32) == (True, False, 32)
    stats = _get_t2_stats()
    assert stats["initial_reject"] == 1
    assert stats["projected"] == 1
    assert stats["projection_failed"] == 0
    assert stats["directed_choices"] == 1


def test_thread_projection_failure_returns_none():
    parent = _single_thread_sch().trace
    target = _gpu_target(max_threads_per_block=32)

    with _t2_mode(1):
        shadow_mutator = _make_mutator(target=target, mod=elementwise)
    seed = _find_rejecting_seed(shadow_mutator, parent, max_threads_per_block=32)

    with _t2_mode(2):
        enforce_mutator = _make_mutator(target=target, mod=elementwise)
    _reset_t2_stats()
    assert _apply_with_seed(enforce_mutator, parent, seed) is None
    stats = _get_t2_stats()
    assert stats["initial_reject"] == 1
    assert stats["projected"] == 0
    assert stats["projection_failed"] == 1


def test_unknown_thread_provenance_is_passed_through():
    parent = _unknown_thread_sch().trace
    target = _gpu_target(max_threads_per_block=32)
    seed = 7

    with _t2_mode(1):
        shadow_mutator = _make_mutator(target=target, mod=elementwise)
    shadow = _apply_with_seed(shadow_mutator, parent, seed)
    assert shadow is not None

    with _t2_mode(2):
        enforce_mutator = _make_mutator(target=target, mod=elementwise)
    _reset_t2_stats()
    enforce = _apply_with_seed(enforce_mutator, parent, seed)
    assert enforce is not None
    assert enforce.as_json(remove_postproc=True) == shadow.as_json(remove_postproc=True)
    stats = _get_t2_stats()
    assert stats["checked"] == 0
    assert stats["unknown"] == 1


def test_multiple_thread_axes_are_conservatively_unknown():
    parent = _multi_axis_thread_sch().trace
    target = _gpu_target(max_threads_per_block=32)
    seed = 7

    with _t2_mode(0):
        off_mutator = _make_mutator(target=target, mod=elementwise)
    off = _apply_with_seed(off_mutator, parent, seed)
    assert off is not None

    with _t2_mode(2):
        project_mutator = _make_mutator(target=target, mod=elementwise)
    _reset_t2_stats()
    projected = _apply_with_seed(project_mutator, parent, seed)
    assert projected is not None
    assert projected.as_json(remove_postproc=True) == off.as_json(remove_postproc=True)
    stats = _get_t2_stats()
    assert stats["checked"] == 0
    assert stats["unknown"] == 1


if __name__ == "__main__":
    tests = (
        test_mutate_tile_size_matmul,
        test_mutate_sample_categorical_single_candidate,
        test_thread_projection_preserves_shadow_and_repairs_enforce,
        test_thread_direction_uses_slack_and_pressure,
        test_thread_direction_repairs_rejecting_proposal,
        test_thread_projection_failure_returns_none,
        test_unknown_thread_provenance_is_passed_through,
        test_multiple_thread_axes_are_conservatively_unknown,
    )
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"PASS all {len(tests)} mutate-tile-size tests")
