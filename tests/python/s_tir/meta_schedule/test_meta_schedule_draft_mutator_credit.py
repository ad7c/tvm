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

import math

import pytest
import tvm
import tvm.testing
from tvm.s_tir import meta_schedule as ms
from tvm.s_tir.schedule import Schedule
from tvm.script import tirx as T


@tvm.script.ir_module
class _G3Matmul:
    @T.prim_func(s_tir=True)
    def main(a: T.handle, b: T.handle, c: T.handle) -> None:
        T.func_attr({"global_symbol": "main"})
        A = T.match_buffer(a, (16, 16), "float32")
        B = T.match_buffer(b, (16, 16), "float32")
        C = T.match_buffer(c, (16, 16), "float32")
        for i, j, k in T.grid(16, 16, 16):
            with T.sblock("matmul"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = T.float32(0)
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]


def _g3_schedule(sch: Schedule):
    block = sch.get_sblock("matmul")
    i, j, k = sch.get_loops(block)
    i0, i1 = sch.split(i, sch.sample_perfect_tile(i, n=2))
    j0, j1 = sch.split(j, sch.sample_perfect_tile(j, n=2))
    k0, k1 = sch.split(k, sch.sample_perfect_tile(k, n=2))
    sch.reorder(i0, j0, k0, i1, j1, k1)


def _ffi(name):
    return tvm.get_global_func(f"s_tir.meta_schedule.testing.{name}")


def _make(num_mutators=3):
    return _ffi("MakeMeasuredMutatorCreditBook")(num_mutators, 50, 32, 1.0)


def _record(book, mutator_index, improved):
    parent_latency = 2.0
    child_latency = 1.0 if improved else 3.0
    _ffi("RecordMeasuredMutatorOutcome")(
        book, mutator_index, parent_latency, child_latency
    )


def test_measured_mutator_credit_keeps_last_50_per_mutator():
    book = _make(num_mutators=2)

    for _ in range(10):
        _record(book, 0, improved=True)
    for _ in range(50):
        _record(book, 0, improved=False)
    for _ in range(7):
        _record(book, 1, improved=True)

    stats_0 = list(_ffi("MeasuredMutatorCreditStats")(book, 0))
    stats_1 = list(_ffi("MeasuredMutatorCreditStats")(book, 1))
    assert stats_0 == [50.0, 0.0, 0.0]
    assert stats_1 == [7.0, 7.0, 1.0]
    assert _ffi("MeasuredMutatorCreditTotal")(book) == 67


def test_measured_mutator_credit_cold_start_uses_caller_choice():
    book = _make(num_mutators=3)
    in_cold_start = _ffi("MeasuredMutatorCreditInColdStart")
    select = _ffi("SelectMeasuredMutatorCredit")

    assert in_cold_start(book)
    assert select(book, 2) == 2

    for index in range(31):
        _record(book, index % 3, improved=False)
    assert in_cold_start(book)
    assert select(book, 1) == 1

    _record(book, 0, improved=False)
    assert not in_cold_start(book)


def test_measured_mutator_credit_ucb1_explores_under_sampled_arm():
    book = _make(num_mutators=2)
    for _ in range(32):
        _record(book, 0, improved=True)
    _record(book, 1, improved=False)

    scores = list(_ffi("MeasuredMutatorCreditUCB1Scores")(book))
    expected_arm_0 = 1.0 + math.sqrt(2.0 * math.log(33.0) / 32.0)
    expected_arm_1 = math.sqrt(2.0 * math.log(33.0))
    assert scores[0] == pytest.approx(expected_arm_0)
    assert scores[1] == pytest.approx(expected_arm_1)
    assert scores[1] > scores[0]
    assert _ffi("SelectMeasuredMutatorCredit")(book, -1) == 1


def test_measured_mutator_credit_unseen_arm_has_priority_after_cold_start():
    book = _make(num_mutators=3)
    for _ in range(32):
        _record(book, 0, improved=True)
    _record(book, 1, improved=False)

    scores = list(_ffi("MeasuredMutatorCreditUCB1Scores")(book))
    assert math.isinf(scores[2])
    assert _ffi("SelectMeasuredMutatorCredit")(book, -1) == 2


def test_failed_unseen_arm_is_temporarily_excluded_without_fake_reward():
    book = _make(num_mutators=3)
    select_excluding = _ffi("SelectMeasuredMutatorCreditExcluding")

    for _ in range(16):
        _record(book, 0, improved=True)
    for _ in range(16):
        _record(book, 1, improved=False)

    # Arm 2 is unseen and therefore has infinite UCB.  Once that attempt has
    # failed locally, the retry can use the best remaining measured arm.
    assert select_excluding(book, -1, [0, 0, 0]) == 2
    assert select_excluding(book, -1, [0, 0, 1]) == 0
    assert select_excluding(book, -1, [1, 0, 1]) == 1
    assert select_excluding(book, -1, [1, 1, 1]) == -1

    # Local retry masks are not Runner observations and must not affect UCB.
    assert _ffi("MeasuredMutatorCreditTotal")(book) == 32
    assert math.isinf(list(_ffi("MeasuredMutatorCreditUCB1Scores")(book))[2])


def test_cold_start_failed_baseline_arm_retries_without_replacement():
    book = _make(num_mutators=3)
    select_excluding = _ffi("SelectMeasuredMutatorCreditExcluding")

    assert select_excluding(book, 2, [0, 0, 0]) == 2
    assert select_excluding(book, 2, [0, 0, 1]) == 0
    assert select_excluding(book, 2, [1, 0, 1]) == 1
    assert _ffi("MeasuredMutatorCreditTotal")(book) == 0


def _prepare_g3_strategy(monkeypatch, g3_mode):
    for name, value in {
        # Keep every orthogonal experimental branch fixed so this helper only
        # varies G3.  All of these modes are read from process-global env vars.
        "TVM_GA_T2_MODE": "0",
        "TVM_THREAD_CONSTRAINT_SAMPLE_MODE": "off",
        "TVM_THREAD_HARD_FILTER_MODE": "off",
        "TVM_SOFT_FILTER_MODE": "0",
        "TVM_DRAFT_MODE": "0",
        "TVM_DRAFT_GA_MODE": "2",
        "TVM_DRAFT_GA_G3_MODE": g3_mode,
        "TVM_DRAFT_GA_ANNEAL_MODE": "0",
        "TVM_DRAFT_GA_VERIFY_RATIO": "1.0",
        "TVM_DRAFT_PEAK_GFLOPS": "10000",
        "TVM_DRAFT_MEMORY_GBPS": "500",
        "TVM_DRAFT_SM_COUNT": "16",
        "TVM_EVO_EXACT_INIT_POPULATION": "1",
        "TVM_EVO_INIT_SAMPLE_BATCH_SIZE": "8",
    }.items():
        monkeypatch.setenv(name, value)

    target = tvm.target.Target(
        {
            "kind": "cuda",
            "arch": "sm_86",
            "max_threads_per_block": 1024,
            "max_num_threads": 1024,
            "max_shared_memory_per_block": 49152,
            "thread_warp_size": 32,
        }
    )
    database = ms.database.MemoryDatabase()
    configured_strategy = ms.search_strategy.EvolutionarySearch(
        population_size=8,
        init_measured_ratio=1.0,
        init_min_unmeasured=0,
        genetic_num_iters=1,
        genetic_mutate_prob=1.0,
        genetic_max_fail_count=8,
        eps_greedy=0.0,
    )
    context = ms.TuneContext(
        mod=_G3Matmul,
        target=target,
        space_generator=ms.space_generator.ScheduleFn(
            sch_fn=_g3_schedule,
            sch_rules=[],
            postprocs=[],
            mutator_probs={ms.mutator.MutateTileSize(): 1.0},
        ),
        search_strategy=configured_strategy,
        rand_state=1,
        num_threads=1,
    )
    strategy = context.search_strategy
    assert strategy is not None
    strategy.pre_tuning(
        max_trials=64,
        num_trials_per_iter=8,
        design_spaces=context.generate_design_space(),
        database=database,
        cost_model=ms.cost_model.RandomModel(seed=1),
    )
    # The C++ search strategy stores the TuneContext as a non-owning pointer.
    # Return it so every caller keeps the context alive for the full lifecycle.
    return context, strategy, database, target


def test_g3_off_allocates_no_credit_or_pending_state(monkeypatch):
    _context, strategy, _, _ = _prepare_g3_strategy(monkeypatch, "off")
    stats = list(_ffi("EvolutionarySearchG3Stats")(strategy))

    assert stats == [0, 0, 0, 0, 0, 0, 1, 0]
    assert list(_ffi("EvolutionarySearchG3PendingMutators")(strategy)) == []
    strategy.post_tuning()


def _first_batch_trace_json(monkeypatch, g3_mode):
    _context, strategy, _, _ = _prepare_g3_strategy(monkeypatch, g3_mode)
    try:
        candidates = strategy.generate_measure_candidates()
        assert candidates is not None and candidates
        return [
            candidate.sch.trace.as_json(remove_postproc=True) for candidate in candidates
        ]
    finally:
        strategy.post_tuning()


def test_g3_shadow_preserves_off_first_batch_candidate_sequence(monkeypatch):
    off_traces = _first_batch_trace_json(monkeypatch, "off")
    shadow_traces = _first_batch_trace_json(monkeypatch, "shadow")

    assert len(shadow_traces) == len(off_traces)
    assert shadow_traces == off_traces


@pytest.mark.parametrize("g3_mode", ["shadow", "enforce"])
def test_g3_lineage_is_aligned_and_feedback_is_cross_batch(monkeypatch, g3_mode):
    _context, strategy, database, target = _prepare_g3_strategy(monkeypatch, g3_mode)

    stats = _ffi("EvolutionarySearchG3Stats")
    pending_mutators = _ffi("EvolutionarySearchG3PendingMutators")
    assert list(stats(strategy))[0] == {"shadow": 1, "enforce": 2}[g3_mode]
    workload = database.commit_workload(_G3Matmul)
    credited = False
    for batch in range(6):
        candidates = strategy.generate_measure_candidates()
        assert candidates is not None and candidates
        pending = list(pending_mutators(strategy))
        before = list(stats(strategy))
        assert len(pending) == len(candidates) == before[3]
        assert sum(index >= 0 for index in pending) == before[4]

        latency = 2.0 / (batch + 1)
        results = [ms.runner.RunnerResult([latency], None) for _ in candidates]
        expected_credited = before[5]
        old_total = before[1]
        strategy.notify_runner_results(candidates, results)
        after = list(stats(strategy))
        assert after[1] == old_total + expected_credited
        assert after[3] == 0

        for candidate in candidates:
            database.commit_tuning_record(
                ms.database.TuningRecord(
                    trace=candidate.sch.trace,
                    workload=workload,
                    run_secs=[latency],
                    target=target,
                    args_info=candidate.args_info,
                )
            )
        if expected_credited:
            credited = True
            break

    strategy.post_tuning()
    assert credited, "expected a one-hop child of an earlier RunnerResult within six batches"


if __name__ == "__main__":
    tvm.testing.main()
