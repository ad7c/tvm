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
"""After serial tuning, independently retime fixed winners and produce plots."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from report_single_seed_study import GROUPS, audit, digest
from run_single_seed_study import ROOT, environment, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "experiments/results/paper_v1/single_seed_20261010")
    parser.add_argument("--output", type=Path, default=ROOT / "papers/figures/single_seed_20261010")
    parser.add_argument("--wait", action="store_true", help="Wait for the existing registered tuning process")
    args = parser.parse_args()
    study = args.input.resolve()
    if args.wait:
        print("WAIT for registered serial tuning; no concurrent GPU measurements", flush=True)
        while True:
            progress = json.loads((study / "progress.json").read_text(encoding="utf-8"))
            if progress["status"] == "completed":
                break
            if progress["status"] == "failed":
                raise RuntimeError("Tuning failed; inspect preserved attempts before independent retiming")
            time.sleep(5)
    protocol = json.loads((study / "protocol.json").read_text(encoding="utf-8"))
    originals = {}
    for spec in protocol["order"]:
        if spec["smoke"]:
            continue
        paths = list(study.glob(spec["id"] + "_*/result.json"))
        if len(paths) != 1:
            raise ValueError(f"Expected one completed registered result for {spec['id']}")
        row = audit(paths[0], spec, protocol)
        originals[(spec["workload"], spec["genetic_iters"], spec["arm"])] = (paths[0], row)
    env = environment()
    for workload, ga in GROUPS:
        group = f"{workload}_ga{ga}_R01"
        output = study / "retiming" / group
        paths = [originals[(workload, ga, arm)][0] for arm in "BCD"]
        expected_hashes = {arm: digest(path) for arm, path in zip("BCD", paths)}
        if output.exists():
            result = json.loads((output / "result.json").read_text(encoding="utf-8"))
            actual_hashes = {arm["name"]: arm["result_sha256"] for arm in result["arms"]}
            valid_samples = all(len(arm["samples_us"]) == 30 and len(arm["checks"]) == 4
                                and all(check["passed"] for check in arm["checks"])
                                for arm in result["arms"])
            if (result["status"] != "completed" or actual_hashes != expected_hashes
                    or not valid_samples or len(result["measurements"]) != 9
                    or result["loaded_library_sha256"] != protocol["runtime"]["compiler_sha256"]):
                raise ValueError(f"Incomplete or mismatched retiming retained at {output}")
            print(f"RETAIN {group}", flush=True)
            continue
        cmd = [sys.executable, "-u", "-X", "utf8", str(ROOT / "experiments/remeasure_top1.py"),
               "--output", str(output), "--library-dir", str(ROOT / "build-prune/lib"),
               "--terminal-worker-exit"]
        for arm, path in zip("BCD", paths):
            cmd += ["--result", f"{arm}={path}"]
        log_path = study / (group + ".console.log")
        if log_path.exists():
            raise FileExistsError(log_path)
        print(f"RETIME {group}", flush=True)
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            write_json(study / "finalization.json", {"status": "retiming", "group": group,
                       "pid": process.pid, "log": str(log_path)})
            returncode = process.wait()
        with (study / "retiming_execution.jsonl").open("a", encoding="utf-8") as ledger:
            ledger.write(json.dumps({"group": group, "pid": process.pid, "returncode": returncode,
                                     "windows_hex": hex(returncode & 0xffffffff),
                                     "console_sha256": digest(log_path)}) + "\n")
        if returncode:
            write_json(study / "finalization.json", {"status": "failed", "group": group,
                       "returncode": returncode, "log": str(log_path)})
            raise RuntimeError(f"Independent retiming returned {returncode}; full log: {log_path}")
        print(f"DONE {group}", flush=True)
    cmd = [sys.executable, "-u", "-X", "utf8", str(ROOT / "experiments/report_single_seed_study.py"),
           "--input", str(study), "--output", str(args.output.resolve()), "--plot"]
    write_json(study / "finalization.json", {"status": "reporting"})
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)
    write_json(study / "finalization.json", {"status": "completed", "report": str(args.output.resolve()),
               "visual_qa": "pending human-agent inspection"})


if __name__ == "__main__":
    main()
