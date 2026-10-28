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
"""Collect and evaluate a frozen candidate pool using real TVM C++ FFI.

This is an H-position mechanism experiment, not an end-to-end tuner. It uses
the reference workload/target and target-derived postprocessor chain, but a
separate, seeded proposal sampler. No builder, runner, XGBoost or GPU timing.
Initialization collection itself performs replay; that cost is recorded and
must not be hidden when interpreting pre-replay filtering of completed traces.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
ARMS = ("baseline", "postreplay", "trace")
SOURCE_FILES = (
    "src/s_tir/meta_schedule/constraints.cc",
    "src/s_tir/meta_schedule/constraints.h",
    "src/s_tir/meta_schedule/utils.h",
    "src/s_tir/meta_schedule/search_strategy/evolutionary_search.cc",
    "src/s_tir/meta_schedule/postproc/verify_gpu_code.cc",
    "src/s_tir/meta_schedule/space_generator/space_generator.cc",
    "src/s_tir/meta_schedule/postproc/postproc.cc",
    "src/s_tir/meta_schedule/mutator/mutate_tile_size.cc",
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_hash(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def plain_json(value):
    """Preserve numeric trace decision indices; never stringify TVM IntImm."""
    if isinstance(value, (list, tuple)):
        return [plain_json(item) for item in value]
    if isinstance(value, dict):
        return {str(key): plain_json(item) for key, item in value.items()}
    if type(value).__name__ == "IntImm":
        return int(value.value)
    if type(value).__name__ == "FloatImm":
        return float(value.value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported trace JSON value: {type(value).__name__}")


def configure_environment(library_dir=None):
    os.environ["TVM_LIBRARY_PATH"] = str(library_dir or ROOT / "build-prune/lib")
    sys.path.insert(0, str(ROOT / "python"))
    library_bin = Path(sys.executable).parent / "Library/bin"
    if os.name == "nt" and library_bin.is_dir():
        os.environ["PATH"] = str(library_bin) + os.pathsep + os.environ.get("PATH", "")
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_THREADING_LAYER"] = "SEQUENTIAL"
    for name in (
        "TVM_PRUNE_MODE", "TVM_SOFT_FILTER_MODE", "TVM_DRAFT_MODE", "TVM_DRAFT_GA_MODE",
        "TVM_DRAFT_GA_G3_MODE", "TVM_DRAFT_GA_ANNEAL_MODE", "TVM_GA_T2_MODE",
    ):
        os.environ[name] = "0"
    os.environ["TVM_THREAD_CONSTRAINT_SAMPLE_MODE"] = "off"
    os.environ["TVM_THREAD_HARD_FILTER_MODE"] = "off"


class ReplayExperiment:
    def __init__(self, reference_path, seed, *, expected_library_sha256=None):
        import tvm  # pylint: disable=import-outside-toplevel
        from tvm.s_tir import meta_schedule as ms  # pylint: disable=import-outside-toplevel

        self.tvm = tvm
        self.ms = ms
        self.reference_path = reference_path.resolve()
        self.reference = json.loads(reference_path.read_text(encoding="utf-8"))
        self.library = Path(tvm.base._LOADED_LIBS["tvm_compiler"]._name).resolve()
        # Formal pool runs default to the archived DLL. A diagnostic bridge may
        # name an exact new hash explicitly; it must separately verify the old oracle.
        expected_hash = expected_library_sha256 or self.reference["loaded_library_sha256"]
        if sha256(self.library) != expected_hash:
            raise ValueError("Loaded DLL differs from reference; freeze a new version explicitly")
        database = ms.database.JSONDatabase(work_dir=str(reference_path.parent), allow_missing=False)
        records = list(database.get_all_tuning_records())
        if not records:
            raise ValueError("Reference database has no workload records")
        self.mod = records[0].workload.mod
        for record in records:
            tvm.ir.assert_structural_equal(self.mod, record.workload.mod)
        target_config = self.reference["target"]
        self.target = tvm.target.Target(json.loads(target_config))
        self.context = ms.TuneContext(
            mod=self.mod, target=self.target, space_generator="post-order-apply",
            rand_state=seed, num_threads=1,
        )
        self.spaces = self.context.generate_design_space()
        self.postprocs = list(self.context.space_generator.postprocs)
        self.postproc_names = [type(postproc).__name__ for postproc in self.postprocs]
        self.mutators = sorted(self.context.space_generator.mutator_probs.items(),
                               key=lambda item: type(item[0]).__name__)
        self.trace_check = tvm.get_global_func("s_tir.meta_schedule.testing.EvaluateTraceConstraints")
        self.ir_check = tvm.get_global_func("s_tir.meta_schedule.testing.AnalyzeThreadFeasibility")
        self.mutate = tvm.get_global_func("s_tir.meta_schedule.MutatorApply")
        self.max_threads = int(self.target.attrs["max_threads_per_block"])
        self.warp_size = int(self.target.attrs["thread_warp_size"])
        self.max_shared = int(self.target.attrs["max_shared_memory_per_block"])

    def new_schedule(self, seed):
        return self.tvm.s_tir.Schedule(self.mod, seed=seed, debug_mask=0, error_render_level="none")

    def decode_trace(self, serialized):
        # Restore the object graph without replay, then bind executable kinds
        # from TVM's registry. Generic object JSON need not preserve callbacks.
        from tvm.s_tir.schedule import Instruction, InstructionKind

        loaded = self.tvm.ir.load_json(serialized)
        instructions, decisions = [], {}
        for old in loaded.insts:
            new = Instruction(InstructionKind.get(old.kind.name),
                              old.inputs, old.attrs, old.outputs)
            instructions.append(new)
            decision = loaded.get_decision(old)
            if decision is not None:
                decisions[new] = decision
        return self.tvm.s_tir.Trace(instructions, decisions)

    def encode_trace(self, trace):
        trace_json = plain_json(trace.as_json(remove_postproc=True))
        serialized = self.tvm.ir.save_json(trace)
        restored = self.decode_trace(serialized)
        if plain_json(restored.as_json(remove_postproc=True)) != trace_json:
            raise AssertionError("Pure TVM JSON trace round-trip changed decisions")
        return {"trace_json": trace_json, "trace_sha256": json_hash(trace_json),
                "trace_tvm_json": serialized}

    def evaluate(self, trace, seed, arm):
        """Every schedule starts fresh. Times include FFI/Python orchestration overhead."""
        started = time.perf_counter()
        row = {"arm": arm, "accepted": False, "status": None, "trace_check": None,
               "ir_check": None, "postprocs": [], "timing_s": {}, "error": None}
        schedule = None
        operation = "trace_check"

        def timed(name, callback):
            tick = time.perf_counter()
            try:
                return callback()
            finally:
                row["timing_s"][name] = time.perf_counter() - tick

        try:
            if arm == "trace":
                raw = timed("plan_and_trace_check", lambda: self.trace_check(
                    trace, self.max_threads, self.warp_size))
                row["trace_check"] = {str(key): int(value) for key, value in raw.items()}
                if row["trace_check"]["truth"] == 1:
                    row["status"] = "hard_trace_reject"
                    return row, None
            operation = "schedule_setup"
            schedule = timed(operation, lambda: self.new_schedule(seed))
            operation = "replay"
            timed(operation, lambda: trace.apply_to_schedule(schedule, remove_postproc=True))
            if arm in ("trace", "postreplay"):
                operation = "postreplay_check"
                raw = timed(operation, lambda: self.ir_check(
                    schedule.mod, self.max_threads, self.warp_size, self.max_shared))
                row["ir_check"] = [int(value) for value in raw]
                if row["ir_check"][1]:
                    row["status"] = "hard_postreplay_reject"
                    return row, None
            operation = "enter_postproc"
            timed(operation, schedule.enter_postproc)
            for index, postproc in enumerate(self.postprocs):
                operation = f"postproc_{index}_{self.postproc_names[index]}"
                passed = bool(timed(operation, lambda: postproc.apply(schedule)))
                row["postprocs"].append({"index": index, "name": self.postproc_names[index],
                                          "passed": passed})
                if not passed:
                    row["status"] = operation + "_reject"
                    return row, None
            row["accepted"] = True
            row["status"] = "passed"
            return row, schedule
        except self.tvm.s_tir.ScheduleError as error:
            # Only expected scheduling failures are candidate outcomes. API and
            # analysis bugs must abort, never make a failed oracle look valid.
            if operation != "replay" and not operation.startswith("postproc_"):
                raise
            row["status"] = operation + "_exception"
            row["error"] = f"{type(error).__name__}: {error}"
            return row, None
        finally:
            row["total_s"] = time.perf_counter() - started
            row["timing_s"]["orchestration_residual"] = row["total_s"] - sum(row["timing_s"].values())


def collect_pool(experiment, seed, initial_count, mutation_count, output):
    rng = random.Random(seed)
    rows, traces, parents = [], {}, []
    started = time.perf_counter()
    journal = output / "collection.jsonl"
    with journal.open("x", encoding="utf-8") as stream:
        def commit(row):
            rows.append(row)
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()

        for index in range(initial_count):
            candidate_seed = rng.randrange(1, 2**31 - 1)
            template_id = rng.randrange(len(experiment.spaces))
            template = experiment.spaces[template_id]
            template_trace = experiment.tvm.s_tir.Trace(template.trace.insts, {})
            row = {"id": f"init-{index:04d}", "stage": "initialization", "seed": candidate_seed,
                   "template_id": template_id, "parent_id": None, "mutator": None,
                   "generation_status": "complete", "generation_error": None}
            generation_start = time.perf_counter()
            schedule = experiment.new_schedule(candidate_seed)
            try:
                template_trace.apply_to_schedule(schedule, remove_postproc=True)
            except experiment.tvm.s_tir.ScheduleError as error:
                row["generation_status"] = "initial_replay_exception"
                row["generation_error"] = f"{type(error).__name__}: {error}"
                row["partial_trace_tvm_json"] = experiment.tvm.ir.save_json(schedule.trace)
            row["generation_s"] = time.perf_counter() - generation_start
            if row["generation_status"] == "complete":
                row.update(experiment.encode_trace(schedule.trace))
                traces[row["id"]] = schedule.trace
                oracle, oracle_schedule = experiment.evaluate(schedule.trace, candidate_seed, "baseline")
                row["collection_oracle"] = oracle
                if oracle["accepted"]:
                    row["oracle_mod_tvm_json"] = experiment.tvm.ir.save_json(oracle_schedule.mod)
                    parents.append(row)
            commit(row)
            if (index + 1) % 8 == 0 or index + 1 == initial_count:
                print(f"Collected init {index + 1}/{initial_count}; baseline-valid parents={len(parents)}",
                      flush=True)

        if mutation_count and not parents:
            raise RuntimeError("No baseline-valid parent in the fixed initial quota; do not silently refill")
        mutators, weights = zip(*experiment.mutators)
        for index in range(mutation_count):
            parent = rng.choice(parents)
            mutator = rng.choices(mutators, weights=[float(weight) for weight in weights], k=1)[0]
            mutation_seed = rng.randrange(1, 2**31 - 1)
            replay_seed = rng.randrange(1, 2**31 - 1)
            row = {"id": f"mutation-{index:04d}", "stage": "mutation", "seed": replay_seed,
                   "template_id": parent["template_id"], "parent_id": parent["id"],
                   "mutator": type(mutator).__name__, "mutation_seed": mutation_seed,
                   "generation_status": "complete", "generation_error": None}
            generation_start = time.perf_counter()
            child = experiment.mutate(mutator, traces[parent["id"]], mutation_seed)
            row["generation_s"] = time.perf_counter() - generation_start
            if child is None:
                if row["generation_status"] == "complete":
                    row["generation_status"] = "no_proposal"
            else:
                row.update(experiment.encode_trace(child))
                traces[row["id"]] = child
                oracle, oracle_schedule = experiment.evaluate(child, replay_seed, "baseline")
                row["collection_oracle"] = oracle
                if oracle["accepted"]:
                    row["oracle_mod_tvm_json"] = experiment.tvm.ir.save_json(oracle_schedule.mod)
            commit(row)
            if (index + 1) % 8 == 0 or index + 1 == mutation_count:
                print(f"Collected mutation {index + 1}/{mutation_count}", flush=True)
    return rows, time.perf_counter() - started


def compare_pool(experiment, pool, repeats, output):
    # Deserialization is pure object reconstruction, verified at collection.
    candidates = [row for row in pool if row["generation_status"] == "complete"]
    traces = {row["id"]: experiment.decode_trace(row["trace_tvm_json"]) for row in candidates}
    oracle_modules = {row["id"]: experiment.tvm.ir.load_json(row["oracle_mod_tvm_json"])
                      for row in candidates if row["collection_oracle"]["accepted"]}
    for row in candidates:
        if json_hash(plain_json(traces[row["id"]].as_json(True))) != row["trace_sha256"]:
            raise AssertionError("Frozen trace identity mismatch")
    warmup_start = time.perf_counter()
    warmup = [next(row for row in candidates if row["collection_oracle"]["accepted"])]
    rejected = next((row for row in candidates if not row["collection_oracle"]["accepted"]), None)
    if rejected:
        warmup.append(rejected)
    for row in warmup:
        for arm in ARMS:
            experiment.evaluate(traces[row["id"]], row["seed"], arm)
    warmup_s = time.perf_counter() - warmup_start
    events, disagreements, false_rejects = [], [], []
    with (output / "evaluations.jsonl").open("x", encoding="utf-8") as stream:
        for repeat in range(repeats):
            order = ARMS[repeat % len(ARMS):] + ARMS[:repeat % len(ARMS)]
            for arm in order:
                for candidate in candidates:
                    row, schedule = experiment.evaluate(traces[candidate["id"]], candidate["seed"], arm)
                    row.update({"candidate_id": candidate["id"], "stage": candidate["stage"],
                                "repeat": repeat, "arm_order": list(order)})
                    oracle = candidate["collection_oracle"]
                    row["oracle_accepted"] = oracle["accepted"]
                    if row["accepted"] and oracle["accepted"]:
                        experiment.tvm.ir.assert_structural_equal(
                            schedule.mod, oracle_modules[candidate["id"]])
                    if row["accepted"] != oracle["accepted"]:
                        disagreements.append({"id": candidate["id"], "repeat": repeat,
                                              "arm": arm, "status": row["status"]})
                    if row["status"].startswith("hard_") and oracle["accepted"]:
                        false_rejects.append({"id": candidate["id"], "repeat": repeat, "arm": arm})
                    if arm == "baseline" and row["status"] != oracle["status"]:
                        disagreements.append({"id": candidate["id"], "repeat": repeat,
                                              "arm": arm, "status_drift": row["status"]})
                    events.append(row)
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
                print(f"Repeat {repeat + 1}/{repeats} arm={arm}: evaluated {len(candidates)} frozen traces",
                      flush=True)
    summaries = []
    for stage in ("all", "initialization", "mutation"):
        for arm in ARMS:
            arm_rows = [row for row in events if row["arm"] == arm and
                        (stage == "all" or row["stage"] == stage)]
            totals = [sum(row["total_s"] for row in arm_rows if row["repeat"] == repeat)
                      for repeat in range(repeats)]
            stage_times = Counter()
            for row in arm_rows:
                stage_times.update(row["timing_s"])
            summaries.append({"stage": stage, "arm": arm, "events": len(arm_rows),
                              "repeat_total_s": totals, "median_total_s": statistics.median(totals),
                              "mean_total_s": statistics.mean(totals),
                              "status_counts_all_repeats": dict(Counter(row["status"] for row in arm_rows)),
                              "stage_mean_s": {key: value / repeats for key, value in stage_times.items()},
                              "trace_unknown_all_repeats": sum(row["trace_check"] is not None and
                                  row["trace_check"]["truth"] == 2 for row in arm_rows)})
    return {"warmup_s": warmup_s, "summaries": summaries, "false_rejects": false_rejects,
            "oracle_disagreements": disagreements,
            "gate_passed": not false_rejects and not disagreements}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path, help="Completed single-workload result.json")
    parser.add_argument("--output", required=True, type=Path, help="New directory; existing paths are refused")
    parser.add_argument("--seed", type=int, default=20261008)
    parser.add_argument("--initial", type=int, default=32)
    parser.add_argument("--mutations", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--library-dir", type=Path, help="Optional archived DLL directory; hash must match reference")
    args = parser.parse_args()
    if args.initial <= 0 or args.mutations < 0 or args.repeats <= 0:
        parser.error("initial/repeats must be positive and mutations nonnegative")
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "command.json", {"argv": sys.argv, "utc": datetime.now(timezone.utc).isoformat(),
                                              "python": sys.executable})
    configure_environment(args.library_dir)
    setup_start = time.perf_counter()
    print("Loading TVM and the archived workload; no GPU measurements", flush=True)
    experiment = ReplayExperiment(args.reference, args.seed)
    setup_s = time.perf_counter() - setup_start
    workload_path = args.output / "workload_tvm.json"
    workload_path.write_text(experiment.tvm.ir.save_json(experiment.mod), encoding="utf-8")
    identity = {"reference_result": str(args.reference.resolve()), "reference_sha256": sha256(args.reference),
                "workload": experiment.reference["workload"], "workload_sha256": sha256(workload_path),
                "loaded_library": str(experiment.library), "loaded_library_sha256": sha256(experiment.library),
                "reference_source_identity_sha256": experiment.reference["source_identity_sha256"],
                "current_source_files": {name: sha256(ROOT / name) for name in SOURCE_FILES},
                "driver_sha256": sha256(__file__), "target": str(experiment.target),
                "postprocs": experiment.postproc_names, "design_spaces": len(experiment.spaces),
                "mutators": {type(mutator).__name__: float(weight) for mutator, weight in experiment.mutators}}
    write_json(args.output / "identity.json", identity)
    print(f"DLL {identity['loaded_library_sha256'][:12]}; postprocs={identity['postprocs']}", flush=True)
    pool, collection_s = collect_pool(experiment, args.seed, args.initial, args.mutations, args.output)
    write_json(args.output / "pool.json", {"schema_version": 1, "seed": args.seed, "identity": identity,
                                          "candidates": pool})
    # Read the actual persisted pool: comparison cannot silently use different in-memory proposals.
    pool = json.loads((args.output / "pool.json").read_text(encoding="utf-8"))["candidates"]
    comparison = compare_pool(experiment, pool, args.repeats, args.output)
    result = {"schema_version": 1, "experiment": "P1 frozen-trace H-position C++ mechanism pilot",
              "identity": identity, "seed": args.seed, "initial_attempts": args.initial,
              "mutation_attempts": args.mutations, "repeats": args.repeats,
              "generation_status": dict(Counter(row["generation_status"] for row in pool)),
              "complete_unique_traces": len({row["trace_sha256"] for row in pool if "trace_sha256" in row}),
              "setup_s": setup_s, "collection_s": collection_s,
              "collection_generation_s": sum(row["generation_s"] for row in pool),
              "collection_oracle_s": sum(row.get("collection_oracle", {}).get("total_s", 0) for row in pool),
              "pool_sha256": sha256(args.output / "pool.json"), **comparison,
              "limitations": [
                  "No new GPU latency, builder, XGBoost, or end-to-end tuner evaluation.",
                  "Frozen proposals use target defaults but are not the historical GA trajectory.",
                  "Frozen trace restoration rebinds registered instruction kinds without replay.",
                  "All accepted outputs are structurally checked against the collection oracle outside timing.",
                  "Complete initial decisions required collection replay; its time is reported separately.",
                  "Plan+trace check includes plan construction; timings include Python/FFI overhead.",
                  "All trace survivors receive IR H, including those whose trace check was known.",
                  "Full baseline postproc chain is the oracle; it is not a GPU correctness certificate.",
                  "Verify-specific shared/lowering reason unavailable through current testing FFI.",
                  "Mutation parent/child IDs are real, but child GPU latency and DeltaDraft are not measured.",
                  "No-proposal and initial replay exceptions remain in collection; not fabricated as full traces.",
              ]}
    write_json(args.output / "result.json", result)
    print(json.dumps({"gate_passed": result["gate_passed"], "pool": result["generation_status"],
                      "summaries": result["summaries"]}, indent=2), flush=True)
    if not result["gate_passed"]:
        raise SystemExit("Oracle agreement failed; retain artifacts and investigate before scaling")


if __name__ == "__main__":
    main()
