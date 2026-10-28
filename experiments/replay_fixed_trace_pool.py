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
"""Re-evaluate a persisted P1 pool in a fresh process, without re-sampling it."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

from fixed_trace_pool import (
    ReplayExperiment,
    compare_pool,
    configure_environment,
    sha256,
    write_json,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--library-dir", type=Path, help="Optional archived DLL directory; original hash is required")
    args = parser.parse_args()
    if args.repeats <= 0:
        parser.error("repeats must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "command.json", {
        "argv": sys.argv, "utc": datetime.now(timezone.utc).isoformat(), "python": sys.executable,
    })
    archived = json.loads(args.pool.read_text(encoding="utf-8"))
    original = json.loads((args.pool.parent / "result.json").read_text(encoding="utf-8"))
    if sha256(args.pool) != original["pool_sha256"] or not original["gate_passed"]:
        raise ValueError("Expected an unchanged pool from a successfully validated run")
    configure_environment(args.library_dir)
    print("Loading TVM to replay the existing pool; no new pool proposals or GPU measurements", flush=True)
    started = time.perf_counter()
    experiment = ReplayExperiment(Path(archived["identity"]["reference_result"]), archived["seed"])
    identity = archived["identity"]
    if sha256(experiment.library) != identity["loaded_library_sha256"]:
        raise ValueError("Reevaluation DLL differs from collection")
    if json.loads(str(experiment.target)) != json.loads(identity["target"]):
        raise ValueError("Reevaluation target differs from collection")
    if experiment.postproc_names != identity["postprocs"]:
        raise ValueError("Reevaluation postprocessor chain differs from collection")
    experiment.tvm.ir.assert_structural_equal(experiment.mod, experiment.tvm.ir.load_json(
        (args.pool.parent / "workload_tvm.json").read_text(encoding="utf-8")))
    setup_s = time.perf_counter() - started
    comparison = compare_pool(experiment, archived["candidates"], args.repeats, args.output)
    write_json(args.output / "reevaluation.json", {
        "schema_version": 1, "experiment": "P1 independent-process frozen pool reevaluation",
        "input_pool": str(args.pool.resolve()), "input_pool_sha256": sha256(args.pool),
        "collection_result_sha256": sha256(args.pool.parent / "result.json"),
        "collection_identity": identity, "repeats": args.repeats, "setup_s": setup_s,
        "evaluation_scripts": {name: sha256(Path(__file__).parent / name) for name in
                               ("fixed_trace_pool.py", "replay_fixed_trace_pool.py")},
        "evaluations_sha256": sha256(args.output / "evaluations.jsonl"),
        "new_proposals": 0, "new_gpu_measurements": 0,
        "note": "Collection costs belong to the input run; this is not an independent proposal seed.",
        **comparison,
    })
    print(json.dumps({"gate_passed": comparison["gate_passed"], "summaries": comparison["summaries"]},
                     indent=2), flush=True)
    if not comparison["gate_passed"]:
        raise SystemExit("Frozen-pool reevaluation disagreed with the collection oracle")


if __name__ == "__main__":
    main()
