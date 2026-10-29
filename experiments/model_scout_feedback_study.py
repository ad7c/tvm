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
"""Fresh-history ResNet-18 comparison of XGB-GA and feedback Draft-GA."""

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import shutil
import time
import traceback

import model_scout_study as base
import model_scout_pool_study as pool
from guard_experiment_resources import system_memory
from p3_observability import append_json

ROOT = base.ROOT
DRIVER = Path(__file__).resolve()
ARMS = ("R1", "R6")
save, read = pool.save, base.read


def environment(arm="R1"):
    env = base.environment("C")
    env["TVM_DRAFT_GA_MODE"] = "0"  # The explicit Python strategy owns Draft selection.
    env["OPENBLAS_NUM_THREADS"] = "1"
    return env


def identity():
    return {**base.identity(), **{str(p.relative_to(ROOT)):base.sha(p) for p in
        [DRIVER, ROOT / "experiments/scout_feedback.py", Path(pool.__file__),
         ROOT / "src/s_tir/meta_schedule/draft_model.cc",
         ROOT / "src/s_tir/meta_schedule/draft_model.h"]}}


def publish(study, state, **extra):
    save(study / "progress.json", {"utc": base.utc(), "status": state,
                                  "controller_pid": os.getpid(), **extra})
    lines = ["# ResNet-18 feedback GA", "", f"Updated: {base.utc()}",
             f"State: {state}", f"Phase: {extra.get('phase', '-')} / {extra.get('arm', '-')}", "",
             "| Arm | Submitted | State |", "|---|---:|---|"]
    for arm in ARMS:
        path = study / ("formal_" + arm) / "progress.json"
        row = read(path) if path.exists() else {}
        lines.append(f"| {arm} | {row.get('submitted', 0)} | {row.get('status', 'pending')} |")
    lines += ["", "R1: constraints + controlled XGB-GA; R6: full feedback Draft-GA.",
              "P64/G2/K128, <=10 GPU submissions/batch; FP32 CUDA sm_86.",
              "R1 is a parameter-matched experimental comparator, not native MetaSchedule defaults.",
              "All checkpoint overhead stays in elapsed time. Partial runs are diagnostic evidence.",
              json.dumps(extra, ensure_ascii=False)]
    (study / "STATUS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def configure_launcher():
    """Reuse the guarded child launcher only inside this executable's lifecycle."""
    pool.DRIVER, pool.ARMS = DRIVER, ARMS
    pool.environment, pool.publish = environment, publish


def tune(args):
    import numpy as np
    import tvm
    from tvm_ffi import structural_hash
    from tvm.ir.utils import derived_object
    from tvm.s_tir.meta_schedule import relax_integration
    from tvm.s_tir.meta_schedule.builder import LocalBuilder
    from tvm.s_tir.meta_schedule.cost_model import XGBModel, PyCostModel
    from tvm.s_tir.meta_schedule.database import JSONDatabase
    from tvm.s_tir.meta_schedule.logging import get_loggers_from_work_dir
    from tvm.s_tir.meta_schedule.measure_callback import MeasureCallback, PyMeasureCallback
    from tvm.s_tir.meta_schedule.runner import LocalRunner, EvaluatorConfig
    from tvm.s_tir.meta_schedule.tune import tune_tasks
    from tvm.s_tir.meta_schedule.tune_context import TuneContext
    from tvm.s_tir.meta_schedule.utils import fork_seed
    from p3_observability import TuneJournal
    from scout_feedback import Residual, make_strategy

    started = time.perf_counter()
    deadline = float(os.environ["SCOUT_POOL_DEADLINE"])
    args.global_deadline = float(os.environ["SCOUT_POOL_GLOBAL_DEADLINE"])
    np.random.seed(base.SEED)
    core = XGBModel(num_warmup_samples=100, adaptive_training=True, num_tuning_cores=1)
    residuals = defaultdict(Residual)
    counts = defaultdict(int)
    def emit(kind, **values):
        counts[kind] += 1
        append_json(args.output / "search_events.jsonl", {"kind":kind, "utc":base.utc(),
                    "elapsed_s":time.perf_counter()-started, **values})
    model_state = {"updates":0,"trains":0}

    @derived_object
    class ObservedModel(PyCostModel):
        def load(self,path): core.load(path)
        def save(self,path): core.save(path)
        def predict(self,context,candidates): return core.predict(context,candidates)
        def update(self,context,candidates,results):
            begin = time.perf_counter()
            old_booster, before = core.booster, int(core.data_size)
            core.update(context,candidates,results)
            trained = core.booster is not old_booster
            model_state["updates"] += 1
            model_state["trains"] += int(trained)
            emit("xgb_update", task=str(context.task_name), samples_before=before,
                 samples_after=int(core.data_size),trained=trained,model_state=dict(model_state),
                 seconds=time.perf_counter()-begin)
    lowered = tvm.ir.load_json((args.study / "prepare/model.lowered.json").read_text(encoding="utf-8"))
    extracted = relax_integration.extract_tasks(lowered, base.target(tvm))
    if [str(t.task_name) for t in extracted] != [r["name"] for r in read(args.study / "prepare/tasks.json")]:
        raise RuntimeError("Frozen task inventory changed")
    names = [f"t{i:02d}_{t.task_name}" for i,t in enumerate(extracted)]
    loggers = get_loggers_from_work_dir(str(args.output), [f"t{i:02d}" for i in range(len(names))])
    contexts = [TuneContext(mod=t.dispatched[0], target=t.target, space_generator="post-order-apply",
        search_strategy=make_strategy(core, residuals, args.output, args.arm, deadline, emit, model_state),
        task_name=name, logger=logger, rand_state=seed, num_threads=1).clone()
        for t,name,logger,seed in zip(extracted,names,loggers,fork_seed(base.SEED,n=len(extracted)))]
    db = JSONDatabase(work_dir=str(args.output))
    submitted = 0
    checkpoints = []
    checkpoint_s = 0.0

    @derived_object
    class Capture(PyMeasureCallback):
        def apply(self, scheduler, task_id, candidates, builds, runs):
            nonlocal submitted, checkpoint_s
            submitted += len(candidates)
            row = {"task":names[int(task_id)], "submitted":submitted, "status":"tuning",
                   "utc":base.utc(), "wall_s":time.perf_counter()-started,
                   "keys":[names[int(task_id)] + ":" + str(structural_hash(c.sch.mod)) for c in candidates],
                   "builder_errors":[r.error_msg for r in builds],
                   "runner_errors":[r.error_msg for r in runs],
                   "run_secs":[None if r.run_secs is None else [float(x) for x in r.run_secs] for r in runs],
                   "xgb_samples":int(core.data_size)}
            append_json(args.output / "batches.jsonl", row)
            save(args.output / "progress.json", row)
            if args.trials > 10 and submitted >= (len(checkpoints)+1)*100 and time.time() < deadline-600:
                folder = args.output / "checkpoints" / f"n{submitted:04d}"
                pool.copy_database(args.output, folder)
                receipt = pool.run_child("evaluate", args, folder / "eval", arm=args.arm,
                                          database=folder, timeout_s=600)
                if receipt["status"] != "completed":
                    raise RuntimeError("Checkpoint evaluation failed")
                checkpoint_s += receipt["wall_s"]
                checkpoints.append({"submitted":submitted,"available_wall_s":time.perf_counter()-started,
                                    "result":str(folder / "eval/result.json")})
                save(args.output / "checkpoints.json", checkpoints)

    with TuneJournal(args.output / "timeline.jsonl", record_model_training=True):
        tune_tasks(tasks=contexts, task_weights=[float(t.weight) for t in extracted],
            work_dir=str(args.output), max_trials_global=args.trials, num_trials_per_iter=10,
            builder=LocalBuilder(max_workers=1,timeout_sec=120),
            runner=LocalRunner(timeout_sec=30,evaluator_config=EvaluatorConfig(number=3,repeat=5,min_repeat_ms=100)),
            database=db,cost_model=ObservedModel(),measure_callbacks=[*MeasureCallback.create("default"),Capture()],
            task_scheduler="gradient")
    if core.booster is not None:
        core.save(str(args.output / "xgb_model.tar"))
    summary = {"status":"time_limit" if time.time() >= deadline else "completed",
               "submitted":submitted,"requested":args.trials,"wall_s":time.perf_counter()-started,
               "checkpoint_s":checkpoint_s,"xgb_samples":int(core.data_size),"event_counts":dict(counts),
               "residuals":{k:{"samples":len(r.samples),"version":r.version,"active":r.active} for k,r in residuals.items()},
               "database_records":len(db.get_all_tuning_records())}
    save(args.output / "result.json",summary)
    save(args.output / "progress.json",summary)
    base.terminal_success()


def report(study):
    retimed = read(study / "retime/result.json")["arms"] if (study / "retime/result.json").exists() else {}
    rows = []
    for arm in ARMS:
        folder = study / ("formal_"+arm)
        if not (folder / "final_eval/result.json").exists():
            continue
        result = read(folder / "final_eval/result.json")
        search = read(folder / "result.json")
        mechanisms = defaultdict(int)
        with (folder / "search_events.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                event = json.loads(line)
                if event["kind"] == "middle": mechanisms["midpoint_calls"] += 1
                if event["kind"] == "selection":
                    mechanisms["challenges_measured"] += sum(v == "challenge" for v in event["roles"].values())
                    mechanisms["residual_active_batches"] += int(event["residual_active"])
        rows.append({"arm":arm,"submitted":search["submitted"],"tune_wall_s":search["wall_s"],
                     "median_ms":retimed.get(arm, result)["median_ms"],
                     "timing_source":"rotating_retime" if arm in retimed else "final_eval",
                     "tuned_functions":len(result["applied_tuned_functions"]),
                     "fallback_functions":len(result["fallback_functions"]),"mechanisms":dict(mechanisms)})
    if len(rows) == 2:
        rows[1]["latency_ratio_to_R1"] = rows[1]["median_ms"] / rows[0]["median_ms"]
        rows[1]["within_5_percent"] = rows[1]["latency_ratio_to_R1"] <= 1.05
        target = rows[0]["median_ms"] * 1.05
        for row in rows:
            folder = study / ("formal_" + row["arm"])
            checkpoints = read(folder / "checkpoints.json") if (folder / "checkpoints.json").exists() else []
            attained = [p["available_wall_s"] for p in checkpoints if read(p["result"])["median_ms"] <= target]
            if read(folder / "final_eval/result.json")["median_ms"] <= target:
                attained.append(row["tune_wall_s"] + read(folder / "final_eval/receipt.json")["wall_s"])
            row["first_observed_target_s"] = min(attained) if attained else None
    save(study / "comparison.json",rows)
    (study / "RESULTS.md").write_text("# Exploratory ResNet-18 results\n\n"
        "Single search seed; technical timing repetitions are not search seeds.\n\n"
        "```json\n"+json.dumps(rows,indent=2)+"\n```\n",encoding="utf-8")


def run(args):
    args.study.mkdir(parents=True,exist_ok=False)
    frozen = identity()
    save(args.study / "protocol.json",{"utc":base.utc(),"identity":frozen,"arms":ARMS,
        "model":"resnet18", "shape":[1,3,224,224],"dtype":"float32","seed":base.SEED,
        "P":64,"G":2,"M":32,"E":8,"K":128,"trials":args.trials,
        "arm_seconds":args.arm_seconds,"total_seconds":args.total_seconds,
        "start_free_gib":4,"batch_attempt_limit":1024,"batch_soft_seconds":300,
        "cleanup":"unchanged default callback, identical in both arms; included in wall time",
        "comparator":"controlled XGB-GA with native constraints/replay/mutators; not stock MetaSchedule",
        "features":"legacy raw scoring retained; nominal ms proxy, shared-byte companion; per-buffer overwrite retained"})
    for file in (DRIVER,ROOT / "experiments/scout_feedback.py"):
        shutil.copy2(file,args.study / file.name)
    try:
        wait_start = time.time()
        while system_memory()["available_physical"] < 4*2**30 or system_memory()["available_commit"] < 6*2**30:
            publish(args.study,"waiting_for_memory",available_gib=system_memory()["available_physical"]/2**30,required_gib=4)
            if time.time()-wait_start > args.memory_wait_seconds:
                publish(args.study,"blocked_for_memory")
                return
            time.sleep(15)
        if identity() != frozen:
            raise RuntimeError("Implementation changed while waiting; start a fresh run")
        args.global_deadline = time.time()+args.total_seconds
        save(args.study / "budget.json",{"deadline_epoch":args.global_deadline})
        for stage in ("prepare","untuned"):
            receipt = pool.run_child("prepare" if stage == "prepare" else "evaluate",args,args.study/stage,timeout_s=900,monitor=True)
            if receipt["status"] != "completed": raise RuntimeError(stage+" timed out")
        untuned = read(args.study / "untuned/result.json")
        if (untuned["p75_ms"]-untuned["p25_ms"])/untuned["median_ms"] > .05:
            raise RuntimeError("Untuned GPU timing IQR exceeds 5 percent")
        # One real GPU preflight, separate history; online pathways have native unit coverage.
        folder = args.study / "preflight_R6"
        receipt = pool.run_child("tune",args,folder,arm="R6",trials=10,timeout_s=900,soft_limit_s=600,monitor=True)
        if receipt["status"] != "completed" or read(folder/"result.json")["submitted"] == 0:
            raise RuntimeError("Preflight produced no GPU feedback")
        if pool.copy_database(folder,folder/"final_database") == 0:
            raise RuntimeError("Preflight produced no valid record")
        receipt = pool.run_child("evaluate",args,folder/"final_eval",arm="R6",database=folder/"final_database",timeout_s=900,monitor=True)
        if receipt["status"] != "completed": raise RuntimeError("Preflight model check timed out")
        for arm in ARMS:
            if identity() != frozen: raise RuntimeError("Frozen implementation changed")
            folder = args.study / ("formal_"+arm)
            receipt = pool.run_child("tune",args,folder,arm=arm,trials=args.trials,
                        timeout_s=args.arm_seconds,soft_limit_s=args.arm_seconds-300,monitor=True)
            if receipt["status"] != "completed":
                # Hard-killed arms are diagnostic only; preserve their raw database.
                raise RuntimeError("Arm hit process time limit; raw evidence preserved")
            if pool.copy_database(folder,folder/"final_database") == 0:
                raise RuntimeError("No valid tuning records")
            receipt = pool.run_child("evaluate",args,folder/"final_eval",arm=arm,database=folder/"final_database",timeout_s=900,monitor=True)
            if receipt["status"] != "completed": raise RuntimeError("Final model check timed out")
            report(args.study)
        receipt = pool.run_child("retime",args,args.study/"retime",timeout_s=1800,monitor=True)
        if receipt["status"] != "completed": raise RuntimeError("Retiming timed out")
        report(args.study)
        publish(args.study,"completed",results=str(args.study/"RESULTS.md"))
    except BaseException as error:
        publish(args.study,"failed",error=repr(error),traceback=traceback.format_exc())
        raise


def main():
    configure_launcher()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("run","prepare","evaluate","tune","retime","report"))
    parser.add_argument("--study",type=Path,required=True)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--arm",choices=ARMS,default="R1")
    parser.add_argument("--trials",type=int,default=400)
    parser.add_argument("--arm-seconds",type=int,default=5400)
    parser.add_argument("--total-seconds",type=int,default=18000)
    parser.add_argument("--memory-wait-seconds",type=int,default=0)
    parser.add_argument("--database",type=Path)
    args = parser.parse_args()
    args.study = args.study.resolve()
    if args.output:
        args.output = args.output.resolve()
        args.output.mkdir(parents=True,exist_ok=True)
    if args.stage != "run": os.environ.update(environment(args.arm))
    try:
        {"run":run,"prepare":base.prepare,"evaluate":base.evaluate,"tune":tune,
         "retime":pool.retime,"report":lambda a:report(a.study)}[args.stage](args)
    except BaseException as error:
        if args.output: save(args.output/"failure.json",{"error":repr(error),"traceback":traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
