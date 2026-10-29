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
"""Behavior tests; synthetic labels test plumbing, never performance claims."""

from collections import defaultdict
import math
import time
from types import SimpleNamespace

import numpy as np
import pytest

from scout_feedback import Residual, make_strategy, mixed_pool, percentiles, select_measurements


def test_pool_can_rescue_unknown_and_never_duplicates():
    records = {str(i):{"bucket":str(i % 7)} for i in range(160)}
    scores = {str(i):-float(i) if i < 100 else -math.inf for i in range(160)}
    keys, roles = mixed_pool(records,scores,(96,16,16),np.random.default_rng(31))
    assert len(keys) == len(set(keys)) == 128
    assert sum(v == "random" for v in roles.values()) == 16
    assert any(int(k) >= 100 for k in keys)
    small, _ = mixed_pool({k:records[k] for k in list(records)[:3]},scores,(96,16,16),np.random.default_rng(31))
    assert len(small) == len(set(small)) == 3


def test_challenge_outside_best_eight_is_actually_measured():
    records = {str(i):{"bucket":str(i)} for i in range(128)}
    keys = list(records)
    xgb = {k:128-int(k) for k in keys}
    draft = {k:128-int(k) for k in keys}
    draft["8"] = -100
    chosen,roles = select_measurements(keys,records,draft,xgb,True,np.random.default_rng(1))
    assert len(chosen) == len(set(chosen)) == 10
    assert roles["8"] == "challenge"
    assert sum(r == "xgb_best" for r in roles.values()) == 8
    assert sum(r == "random" for r in roles.values()) == 1
    _,tied = select_measurements(keys,records,draft,dict.fromkeys(keys,1),True,np.random.default_rng(1))
    assert "challenge" not in tied.values()
    assert len(set(percentiles(keys,dict.fromkeys(keys,1)).values())) == 1


def test_cold_start_never_treats_random_xgb_as_teacher():
    records = {str(i):{"bucket":str(i)} for i in range(128)}
    scores = {k:float(k) for k in records}
    chosen,roles = select_measurements(list(records),records,scores,{},False,np.random.default_rng(2))
    assert len(chosen) == 10
    assert sum(r == "random" for r in roles.values()) == 2
    assert "challenge" not in roles.values()


def test_residual_uses_future_holdout_and_bounded_shape_correction():
    model = Residual()
    def rows(start,count):
        result = []
        for i in range(start,start+count):
            x = (i % 16) / 4
            result.append({"key":str(i),"features":[x,0,0,0,0],"raw":1000.,
                           "seconds":1e-6 * math.exp(x)})
        return result
    train = model.update(rows(0,64))
    assert model.version == 1 and not model.active
    assert train["validation"] == []  # Training labels cannot validate themselves.
    probe = rows(100,1)[0]
    assert model.predict(probe) == probe["raw"]
    assert model.predict(probe,True) != probe["raw"]
    heldout = model.update(rows(64,16))
    assert heldout["version_before"] == 1
    assert heldout["validation_concordance"][1] > heldout["validation_concordance"][0]
    assert model.active and model.version == 2
    assert set(train["fit"]["keys"]).isdisjoint(heldout["new_samples"])
    extreme = {**probe,"features":[1e6,0,0,0,0]}
    assert 1/math.sqrt(2) <= model.predict(extreme)/extreme["raw"] <= math.sqrt(2)
    assert model.update(rows(64,16))["new_samples"] == []
    model.update([{"key":"invalid","raw":None,"seconds":1.,"features":None}])
    assert "invalid" not in model.seen


@pytest.mark.parametrize("warm",[False,True])
@pytest.mark.parametrize("arm",["R1","R6"])
def test_native_ga_single_teacher_midpoint_and_feedback_lifecycle(tmp_path, monkeypatch, warm, arm):
    import tvm
    from tvm.s_tir import meta_schedule as ms
    from tvm.s_tir import Schedule
    from model_scout_study import target
    monkeypatch.setenv("TVM_DRAFT_GA_G3_MODE","0")
    monkeypatch.setenv("TVM_DRAFT_GA_MODE","0")
    monkeypatch.setenv("TVM_THREAD_HARD_FILTER_MODE","off")
    monkeypatch.setenv("TVM_THREAD_CONSTRAINT_SAMPLE_MODE","off")
    mod = tvm.script.from_source('''
@T.prim_func(s_tir=True)
def main(A: T.Buffer((4096,), "float32"), B: T.Buffer((4096,), "float32")):
    for i in range(4096):
        with T.sblock("copy"):
            v = T.axis.spatial(4096, i)
            B[v] = A[v] + T.float32(1)
''')
    def schedule(sch):
        block = sch.get_sblock("copy")
        loop = sch.get_loops(block)[0]
        tiles = sch.sample_perfect_tile(loop,n=3,max_innermost_factor=256)
        outer,middle,inner = sch.split(loop,factors=tiles)
        sch.bind(outer,"blockIdx.x")
        sch.bind(inner,"threadIdx.x")
        return sch
    calls = []
    class Teacher:
        data_size = 100 if warm else 0
        booster = object() if warm else None
        def predict(self,context,candidates):
            calls.append(len(candidates))
            return np.arange(len(candidates),dtype=float)
    events = []
    def emit(kind,**row): events.append({"kind":kind,**row})
    strategy = make_strategy(Teacher(),defaultdict(Residual),tmp_path,arm,time.time()+45,emit)
    context = ms.TuneContext(mod=mod,target=target(tvm),task_name="t0_copy",num_threads=1,
        rand_state=31,search_strategy=strategy,
        space_generator=ms.space_generator.ScheduleFn(schedule,sch_rules=[],postprocs=[],
            mutator_probs={ms.mutator.MutateTileSize():1.0}))
    context = context.clone()
    strategy = context.search_strategy
    # No CUDA compilation, execution, or real performance labels in this test.
    strategy.pre_tuning(20,10,context.generate_design_space(),ms.database.MemoryDatabase(),ms.cost_model.RandomModel())
    try:
        candidates = strategy.generate_measure_candidates()
        assert 0 < len(candidates) <= 10
        generation = [r for r in events if r["kind"] == "generation"]
        assert [r["generation"] for r in generation] == [0,1,2]
        selections = [r for r in events if r["kind"] == "selection"]
        chosen = selections[0]["chosen"]
        assert len(chosen) == len(set(chosen))
        mid = [r for r in events if r["kind"] == "middle"]
        assert len(mid) == int(warm and arm == "R6")
        if warm and arm == "R6":
            xgb = [r for r in events if r["kind"] == "xgb"]
            assert [r["phase"] for r in xgb] == ["middle","final"]
            assert set(xgb[0]["actual_inputs"]).isdisjoint(xgb[1]["actual_inputs"])
            assert mid[0]["population_size"] <= 64
        elif arm == "R6":
            assert calls == []
        else:
            assert [r["phase"] for r in events if r["kind"] == "xgb"] == ["ga_0","ga_1","ga_2","archive"]
        strategy.notify_runner_results(candidates,[ms.runner.RunnerResult([.001],None) for _ in candidates])
        second = strategy.generate_measure_candidates()
        second_keys = [r for r in events if r["kind"] == "selection"][-1]["chosen"]
        assert set(chosen).isdisjoint(second_keys)
        assert len(second) <= 10
    finally:
        strategy.post_tuning()
