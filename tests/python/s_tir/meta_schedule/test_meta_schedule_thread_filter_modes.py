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
"""Integration smoke test for the off, shadow, and enforce thread-filter modes."""

import io
import logging
import os
import re

import tvm
from tvm.s_tir import meta_schedule as ms
from tvm.script import ir as I
from tvm.script import tirx as T


@I.ir_module
class BelowAnnotatedThreadExtent:
    @T.prim_func(s_tir=True)
    def main():
        for tx in T.thread_binding(16, thread="threadIdx.x"):
            with T.sblock("compute"):
                T.sblock_attr(
                    {
                        "meta_schedule.thread_extent_low_inclusive": 32,
                        "meta_schedule.thread_extent_high_inclusive": 1024,
                    }
                )
                T.evaluate(tx)


def _identity_schedule(_sch):
    return None


def _target():
    return tvm.target.Target(
        {
            "kind": "cuda",
            "arch": "sm_86",
            "max_shared_memory_per_block": 49152,
            "max_threads_per_block": 1024,
            "thread_warp_size": 32,
        }
    )


def _last_int(log_text, label):
    values = re.findall(rf"{re.escape(label)}:\s*(\d+)", log_text)
    assert values, f"Missing {label!r} in log:\n{log_text}"
    return int(values[-1])


def _run_mode(mode):
    log_stream = io.StringIO()
    logger = logging.getLogger(f"thread-filter-mode-{mode}")
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler(log_stream))

    space_generator = ms.space_generator.ScheduleFn(
        sch_fn=_identity_schedule,
        sch_rules=[],
        postprocs=[ms.postproc.VerifyGPUCode()],
        mutator_probs={},
    )
    context = ms.TuneContext(
        mod=BelowAnnotatedThreadExtent,
        target=_target(),
        space_generator=space_generator,
        search_strategy=ms.search_strategy.EvolutionarySearch(
            population_size=1,
            init_measured_ratio=0.0,
            init_min_unmeasured=1,
            max_fail_count=1,
            genetic_num_iters=0,
            genetic_mutate_prob=0.0,
            genetic_max_fail_count=1,
            eps_greedy=0.0,
        ),
        num_threads=1,
        rand_state=1,
        logger=logger,
    )
    strategy = context.search_strategy
    strategy.pre_tuning(
        max_trials=1,
        num_trials_per_iter=1,
        design_spaces=space_generator.generate_design_space(context.mod),
        database=ms.database.MemoryDatabase(),
        cost_model=ms.cost_model.RandomModel(),
    )
    candidates = strategy.generate_measure_candidates()
    strategy.post_tuning()

    log_text = log_stream.getvalue()
    summary = {
        "generated": _last_int(log_text, "stage.generated"),
        "passed": _last_int(log_text, "stage.passed"),
        "accounted": _last_int(log_text, "stage.accounted"),
        "conservation_delta": _last_int(log_text, "stage.conservation_delta"),
        "checked": _last_int(log_text, "Thread filter checked"),
        "predicted": _last_int(log_text, "Thread filter predicted rejections"),
        "enforced": _last_int(log_text, "Thread filter enforced rejections"),
        "verify_fail": int(
            re.findall(r"VerifyGPUCode[^\]]*\]:\s*(\d+) failure", log_text)[-1]
        ),
    }
    summary["verify_reasons"] = {
        reason: _last_int(log_text, f"stage.verify_failed[{reason}]")
        for reason in (
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
    }
    shadow = re.findall(
        r"Thread filter shadow TP/FP/FN/TN:\s*(\d+)/(\d+)/(\d+)/(\d+)",
        log_text,
    )
    assert shadow, log_text
    summary["tp"], summary["fp"], summary["fn"], summary["tn"] = map(
        int, shadow[-1]
    )
    return candidates, summary


def test_thread_filter_off_shadow_enforce_modes():
    previous_mode = os.environ.get("TVM_PRUNE_MODE")
    try:
        results = {}
        for mode in (0, 1, 2):
            os.environ["TVM_PRUNE_MODE"] = str(mode)
            results[mode] = _run_mode(mode)
    finally:
        if previous_mode is None:
            os.environ.pop("TVM_PRUNE_MODE", None)
        else:
            os.environ["TVM_PRUNE_MODE"] = previous_mode

    for candidates, _ in results.values():
        assert candidates is None

    for _, summary in results.values():
        assert summary["generated"] == summary["accounted"]
        assert summary["conservation_delta"] == 0
        assert sum(summary["verify_reasons"].values()) == summary["verify_fail"]

    off = results[0][1]
    assert off["checked"] == off["predicted"] == off["enforced"] == 0
    assert off["verify_fail"] > 0
    assert off["verify_reasons"]["thread_extent"] == off["verify_fail"]

    shadow = results[1][1]
    assert shadow["checked"] > 0
    assert shadow["predicted"] == shadow["tp"] + shadow["fp"]
    assert shadow["checked"] == shadow["tp"] + shadow["fp"] + shadow["fn"] + shadow["tn"]
    assert shadow["predicted"] > 0
    assert shadow["enforced"] == 0
    assert shadow["fp"] == 0
    assert shadow["verify_fail"] > 0
    assert shadow["verify_reasons"]["thread_extent"] == shadow["verify_fail"]

    enforce = results[2][1]
    assert enforce["checked"] > 0
    assert enforce["predicted"] == enforce["enforced"]
    assert enforce["enforced"] > 0
    assert enforce["verify_fail"] == 0

    for mode, (_, summary) in results.items():
        print(f"PASS mode={mode}: {summary}")


if __name__ == "__main__":
    test_thread_filter_off_shadow_enforce_modes()
