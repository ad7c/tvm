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
"""Tests for the cheap thread-feasibility analysis used by Meta Schedule."""

import math

import tvm
from tvm.script import ir as I
from tvm.script import tirx as T


def _analyze(
    mod: tvm.IRModule,
    max_threads_per_block: int = 1024,
    thread_warp_size: int = 32,
):
    result = _analyze_details(mod, max_threads_per_block, thread_warp_size)
    return result["extents_known"], result["reject"], result["max_threads"]


def _analyze_details(
    mod: tvm.IRModule,
    max_threads_per_block: int = 1024,
    thread_warp_size: int = 32,
    max_shared_memory_per_block: int = 49152,
):
    analyze = tvm.get_global_func("s_tir.meta_schedule.testing.AnalyzeThreadFeasibility")
    raw = [
        int(value)
        for value in analyze(
            mod,
            max_threads_per_block,
            thread_warp_size,
            max_shared_memory_per_block,
        )
    ]
    return {
        "extents_known": bool(raw[0]),
        "reject": bool(raw[1]),
        "max_threads": raw[2],
        "contracts_known": bool(raw[3]),
        "reject_low": bool(raw[4]),
        "reject_high": bool(raw[5]),
        "reject_hardware": bool(raw[6]),
        "shared_known": bool(raw[7]),
        "shared_proxy_reject": bool(raw[8]),
        "max_declared_shared_bytes": raw[9],
    }


def _contract_mod(extent: int, low: int = 32, high: int = 1024):
    @I.ir_module
    class Mod:
        @T.prim_func(s_tir=True)
        def main():
            for tx in T.thread_binding(extent, thread="threadIdx.x"):
                with T.sblock("compute"):
                    T.sblock_attr(
                        {
                            "meta_schedule.thread_extent_low_inclusive": low,
                            "meta_schedule.thread_extent_high_inclusive": high,
                        }
                    )
                    T.evaluate(tx)

    return Mod


def test_thread_contract_lower_boundary():
    below = _analyze_details(_contract_mod(31))
    assert below["contracts_known"]
    assert below["reject"] and below["reject_low"]
    assert not below["reject_high"] and not below["reject_hardware"]

    exact = _analyze_details(_contract_mod(32))
    assert exact["contracts_known"]
    assert not exact["reject"]


def test_thread_contract_upper_boundary():
    exact = _analyze_details(_contract_mod(1024))
    assert exact["contracts_known"]
    assert not exact["reject"]

    above = _analyze_details(_contract_mod(1025), max_threads_per_block=2048)
    assert above["contracts_known"]
    assert above["reject"] and above["reject_high"]
    assert not above["reject_low"] and not above["reject_hardware"]


def test_partial_thread_contract_is_unknown_and_allowed():
    @I.ir_module
    class Mod:
        @T.prim_func(s_tir=True)
        def main():
            for tx in T.thread_binding(16, thread="threadIdx.x"):
                with T.sblock("compute"):
                    T.sblock_attr(
                        {"meta_schedule.thread_extent_low_inclusive": 32}
                    )
                    T.evaluate(tx)

    result = _analyze_details(Mod)
    assert result["extents_known"]
    assert not result["contracts_known"]
    assert not result["reject"]


def test_dynamic_annotated_thread_contract_is_unknown_and_allowed():
    @I.ir_module
    class Mod:
        @T.prim_func(s_tir=True)
        def main(n: T.int32):
            for tx in T.thread_binding(n, thread="threadIdx.x"):
                with T.sblock("compute"):
                    T.sblock_attr(
                        {
                            "meta_schedule.thread_extent_low_inclusive": 32,
                            "meta_schedule.thread_extent_high_inclusive": 1024,
                        }
                    )
                    T.evaluate(tx)

    result = _analyze_details(Mod)
    assert not result["extents_known"]
    assert not result["contracts_known"]
    assert not result["reject"]


def test_shared_memory_is_observation_only_before_compaction():
    @I.ir_module
    class Mod:
        @T.prim_func(s_tir=True)
        def main():
            with T.sblock("root"):
                shared = T.sblock_alloc_buffer((16384,), "float32", scope="shared")
                shared[0] = T.float32(0)

    result = _analyze_details(Mod)
    assert result["shared_known"]
    assert result["max_declared_shared_bytes"] == 65536
    assert result["shared_proxy_reject"]
    # Schedule-level buffers can be compacted later, so the proxy must never
    # participate in the enforce decision.
    assert not result["reject"]


def _apply_thread_soft_filter(scores, thread_counts, mode, weight=0.2):
    apply = tvm.get_global_func("s_tir.meta_schedule.testing.ApplyThreadSoftFilter")
    result = [float(value) for value in apply(scores, thread_counts, 32, mode, weight)]
    size = len(scores)
    return result[:size], result[size:]


def test_xyz_product_per_kernel():
    @I.ir_module
    class Mod:
        @T.prim_func(s_tir=True)
        def main():
            for tx in T.thread_binding(8, thread="threadIdx.x"):
                for ty in T.thread_binding(4, thread="threadIdx.y"):
                    for tz in T.thread_binding(2, thread="threadIdx.z"):
                        T.evaluate(tx + ty + tz)

    assert _analyze(Mod) == (True, False, 64)


def test_repeated_thread_axis_is_not_multiplied():
    @I.ir_module
    class Mod:
        @T.prim_func(s_tir=True)
        def main():
            for tx_outer in T.thread_binding(32, thread="threadIdx.x"):
                for tx_inner in T.thread_binding(32, thread="threadIdx.x"):
                    T.evaluate(tx_outer + tx_inner)

    # Both loops describe the same hardware axis.  Treating them as two
    # independent dimensions would incorrectly report 32 * 32 threads.
    assert _analyze(Mod, max_threads_per_block=64) == (True, False, 32)


def test_conflicting_repeated_axis_is_unknown_and_not_rejected():
    @I.ir_module
    class Mod:
        @T.prim_func(s_tir=True)
        def main():
            for tx_outer in T.thread_binding(32, thread="threadIdx.x"):
                for tx_inner in T.thread_binding(64, thread="threadIdx.x"):
                    T.evaluate(tx_outer + tx_inner)

    known, reject, _ = _analyze(Mod)
    assert not known
    assert not reject


def test_launch_thread_attr_stmt_is_analyzed():
    @I.ir_module
    class Mod:
        @T.prim_func(s_tir=True)
        def main():
            tx = T.env_thread("threadIdx.x")
            ty = T.env_thread("threadIdx.y")
            T.launch_thread(tx, 8)
            T.launch_thread(ty, 4)
            T.evaluate(tx + ty)

    assert _analyze(Mod) == (True, False, 32)


def test_multiple_kernels_are_analyzed_independently():
    @I.ir_module
    class Mod:
        @T.prim_func(s_tir=True)
        def kernel_a():
            for tx in T.thread_binding(32, thread="threadIdx.x"):
                T.evaluate(tx)

        @T.prim_func(s_tir=True)
        def kernel_b():
            for tx in T.thread_binding(16, thread="threadIdx.x"):
                for ty in T.thread_binding(8, thread="threadIdx.y"):
                    T.evaluate(tx + ty)

    # The module maximum is max(32, 16 * 8), not 32 * 16 * 8.
    assert _analyze(Mod, max_threads_per_block=256) == (True, False, 128)


def test_dynamic_extent_is_unknown_and_not_rejected():
    @I.ir_module
    class Mod:
        @T.prim_func(s_tir=True)
        def main(n: T.int32):
            for tx in T.thread_binding(n, thread="threadIdx.x"):
                T.evaluate(tx)

    known, reject, _ = _analyze(Mod)
    assert not known
    assert not reject


def test_over_limit_is_rejected():
    @I.ir_module
    class Mod:
        @T.prim_func(s_tir=True)
        def main():
            for tx in T.thread_binding(33, thread="threadIdx.x"):
                for ty in T.thread_binding(32, thread="threadIdx.y"):
                    T.evaluate(tx + ty)

    assert _analyze(Mod, max_threads_per_block=1024) == (True, True, 1056)


def test_exact_limit_is_allowed():
    @I.ir_module
    class Mod:
        @T.prim_func(s_tir=True)
        def main():
            for tx in T.thread_binding(32, thread="threadIdx.x"):
                for ty in T.thread_binding(32, thread="threadIdx.y"):
                    T.evaluate(tx + ty)

    assert _analyze(Mod, max_threads_per_block=1024) == (True, False, 1024)


def test_warp_execution_contributes_implicit_thread_idx_x():
    @I.ir_module
    class Mod:
        @T.prim_func(s_tir=True)
        def main():
            with T.sblock("root"):
                T.sblock_attr({"warp_execution": True})
                for ty in T.thread_binding(33, thread="threadIdx.y"):
                    T.evaluate(ty)

    assert _analyze(Mod, max_threads_per_block=1024) == (True, True, 1056)


def test_warp_execution_at_exact_limit_is_allowed():
    @I.ir_module
    class Mod:
        @T.prim_func(s_tir=True)
        def main():
            with T.sblock("root"):
                T.sblock_attr({"warp_execution": True})
                for ty in T.thread_binding(32, thread="threadIdx.y"):
                    T.evaluate(ty)

    assert _analyze(Mod, max_threads_per_block=1024) == (True, False, 1024)


def test_warp_execution_conflict_is_unknown_and_not_rejected():
    @I.ir_module
    class Mod:
        @T.prim_func(s_tir=True)
        def main():
            for tx in T.thread_binding(64, thread="threadIdx.x"):
                with T.sblock("warp"):
                    T.sblock_attr({"warp_execution": True})
                    T.evaluate(tx)

    known, reject, _ = _analyze(Mod)
    assert not known
    assert not reject


def test_warp_execution_without_valid_warp_size_is_unknown():
    @I.ir_module
    class Mod:
        @T.prim_func(s_tir=True)
        def main():
            with T.sblock("root"):
                T.sblock_attr({"warp_execution": True})
                T.evaluate(0)

    assert _analyze(Mod, thread_warp_size=-1) == (False, False, -1)


def test_warp_execution_scope_does_not_leak_to_sibling():
    @I.ir_module
    class Mod:
        @T.prim_func(s_tir=True)
        def main():
            with T.sblock("root"):
                with T.sblock("warp"):
                    T.sblock_attr({"warp_execution": True})
                    T.evaluate(0)
                for ty in T.thread_binding(33, thread="threadIdx.y"):
                    T.evaluate(ty)

    assert _analyze(Mod, max_threads_per_block=1024) == (True, False, 33)


def test_thread_soft_filter_modes_do_not_remove_candidates():
    scores = [1.0, 1.0, 1.0, 1.0]
    thread_counts = [32, 64, 128, -1]

    off_penalties, off_scores = _apply_thread_soft_filter(scores, thread_counts, mode=0)
    assert off_penalties == [1.0] * len(scores)
    assert off_scores == scores

    shadow_penalties, shadow_scores = _apply_thread_soft_filter(
        scores, thread_counts, mode=1
    )
    assert len(shadow_penalties) == len(scores)
    assert len(shadow_scores) == len(scores)
    assert shadow_scores == scores

    rerank_penalties, rerank_scores = _apply_thread_soft_filter(
        scores, thread_counts, mode=2
    )
    assert rerank_penalties == shadow_penalties
    assert len(rerank_scores) == len(scores)
    assert all(adjusted <= raw for adjusted, raw in zip(rerank_scores, scores))
    for penalty, adjusted in zip(rerank_penalties, rerank_scores):
        assert math.isclose(adjusted, penalty**0.2, rel_tol=1e-12)


def test_thread_soft_filter_prefers_full_scheduling_units_and_unknown_is_neutral():
    penalties, adjusted = _apply_thread_soft_filter(
        [1.0, 1.0, 1.0, 1.0], [32, 64, 128, -1], mode=2
    )

    assert 0.0 < penalties[0] < penalties[1] < penalties[2]
    assert math.isclose(penalties[2], 1.0)
    assert math.isclose(penalties[3], 1.0)
    assert adjusted[0] < adjusted[1] < adjusted[2]
    assert math.isclose(adjusted[3], 1.0)


def test_thread_soft_filter_zero_weight_is_identity():
    penalties, adjusted = _apply_thread_soft_filter(
        [0.4, 0.8], [32, 128], mode=2, weight=0.0
    )
    assert penalties[0] < penalties[1]
    assert adjusted == [0.4, 0.8]


def test_thread_soft_filter_scales_non_unit_raw_scores():
    raw_scores = [0.25, 0.75]
    penalties, adjusted = _apply_thread_soft_filter(
        raw_scores, [32, 64], mode=2, weight=0.5
    )
    for raw, penalty, actual in zip(raw_scores, penalties, adjusted):
        assert math.isclose(actual, raw * penalty**0.5, rel_tol=1e-12)


def _run_hard_filter_tests_standalone():
    tests = (
        test_xyz_product_per_kernel,
        test_repeated_thread_axis_is_not_multiplied,
        test_conflicting_repeated_axis_is_unknown_and_not_rejected,
        test_launch_thread_attr_stmt_is_analyzed,
        test_multiple_kernels_are_analyzed_independently,
        test_dynamic_extent_is_unknown_and_not_rejected,
        test_over_limit_is_rejected,
        test_exact_limit_is_allowed,
        test_warp_execution_contributes_implicit_thread_idx_x,
        test_warp_execution_at_exact_limit_is_allowed,
        test_warp_execution_conflict_is_unknown_and_not_rejected,
        test_warp_execution_without_valid_warp_size_is_unknown,
        test_warp_execution_scope_does_not_leak_to_sibling,
        test_thread_contract_lower_boundary,
        test_thread_contract_upper_boundary,
        test_partial_thread_contract_is_unknown_and_allowed,
        test_dynamic_annotated_thread_contract_is_unknown_and_allowed,
        test_shared_memory_is_observation_only_before_compaction,
    )
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"PASS all {len(tests)} hard-filter tests")


def _run_standalone():
    _run_hard_filter_tests_standalone()
    soft_filter_tests = (
        test_thread_soft_filter_modes_do_not_remove_candidates,
        test_thread_soft_filter_prefers_full_scheduling_units_and_unknown_is_neutral,
        test_thread_soft_filter_zero_weight_is_identity,
        test_thread_soft_filter_scales_non_unit_raw_scores,
    )
    for test in soft_filter_tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"PASS all {len(soft_filter_tests)} soft-filter tests")


if __name__ == "__main__":
    _run_standalone()
