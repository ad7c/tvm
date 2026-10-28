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
"""Replay archived feedback into one XGBModel; diagnostic only, no hardware measurement."""

import argparse
import hashlib
import json
import logging
from pathlib import Path
import shutil
import sys
import time
import traceback

import psutil

from fixed_trace_pool import configure_environment
from p3_observability import append_json, write_json


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    protocol_path = args.work_dir.parent / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    summary = {"status": "preparing", "scope": "Diagnostic model replay, not an experimental search or GPU retiming",
               "source_database_sha256": sha(args.work_dir / "database_tuning_record.json"),
               "protocol_sha256": sha(protocol_path), "argv": sys.argv, "batches": []}
    shutil.copy2(__file__, args.output / "driver_snapshot.py")
    write_json(args.output / "result.json", summary)
    try:
        configure_environment(root / "build-prune/lib")
        logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(message)s")
        print("Loading archived feedback for isolated XGB memory replay", flush=True)
        import numpy as np
        import tvm
        from tvm.s_tir import meta_schedule as ms
        from tvm.s_tir.meta_schedule.cost_model import XGBModel
        from tvm.s_tir.meta_schedule.runner import RunnerResult

        library = Path(tvm.base._LOADED_LIBS["tvm_compiler"]._name).resolve()
        if sha(library) != protocol["runtime"]["compiler_sha256"]:
            raise ValueError("Diagnostic replay DLL differs from original")
        database = ms.database.JSONDatabase(work_dir=str(args.work_dir.resolve()), allow_missing=False)
        records = list(database.get_all_tuning_records())
        if not records:
            raise ValueError("No archived measurements")
        summary["records"] = len(records)
        identity = json.loads((args.work_dir / "identity.json").read_text(encoding="utf-8"))
        target = tvm.target.Target(json.loads(identity["target"]))
        context = ms.TuneContext(mod=records[0].workload.mod, target=target, num_threads=1)
        model = XGBModel(num_warmup_samples=0, num_tuning_cores=1, adaptive_training=False)
        np.random.seed(protocol["seed"])
        process = psutil.Process()
        for index in range(0, len(records), 10):
            batch = records[index:index + 10]
            candidates = [r.as_measure_candidate() for r in batch]
            feedback = [RunnerResult(r.run_secs, None) for r in batch]
            start = time.perf_counter()
            before = process.memory_info().private
            model.update(context, candidates, feedback)
            feature_arrays = [array for group in model.data.values() for array in group.features]
            row = {"submitted": index + len(batch), "data_size": int(model.data_size),
                   "update_s": time.perf_counter() - start, "before_private": before,
                   "after_private": process.memory_info().private,
                   "feature_bytes": sum(a.nbytes for a in feature_arrays),
                   "feature_rows": sum(a.shape[0] for a in feature_arrays),
                   "boosted_rounds": model.booster.num_boosted_rounds(),
                   "best_score": model.booster.attr("best_score")}
            summary["batches"].append(row)
            append_json(args.output / "training.jsonl", row)
            print(json.dumps(row), flush=True)
            write_json(args.output / "result.json", summary)
        summary["status"] = "completed"
    except BaseException:
        summary.update(status="failed", exception=traceback.format_exc())
        raise
    finally:
        write_json(args.output / "result.json", summary)


if __name__ == "__main__":
    main()
