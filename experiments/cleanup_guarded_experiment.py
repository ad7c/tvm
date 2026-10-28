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
"""Stop only identity-verified processes owned by a recorded experiment guard."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import psutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--guard-dir", type=Path, required=True)
    args = parser.parse_args()
    owned = {}
    for line in (args.guard_dir / "memory.jsonl").read_text(encoding="utf-8").splitlines():
        for record in json.loads(line)["processes"]:
            owned[(record["pid"], record["created"])] = record
    targets = {}
    for pid, created in owned:
        try:
            process = psutil.Process(pid)
            if process.create_time() != created or process.num_threads() == 0:
                continue
            targets[pid] = process
            for child in process.children(recursive=True):
                if child.num_threads() > 0:
                    targets[child.pid] = child
        except psutil.NoSuchProcess:
            pass
    receipts, denied = [], []
    for process in targets.values():
        try:
            if process.num_threads() == 0:
                continue
            created = process.create_time()
            process.kill()
            receipts.append({"pid": process.pid, "created": created, "action": "terminate_owned_process"})
        except psutil.NoSuchProcess:
            pass
        except psutil.AccessDenied:
            try:
                if process.num_threads() > 0:
                    denied.append(process.pid)
            except psutil.NoSuchProcess:
                pass
    psutil.wait_procs(list(targets.values()), timeout=5)
    result = {"utc": datetime.now(timezone.utc).isoformat(), "terminated": receipts,
              "live_access_denied": denied, "scope": "Recorded guard-owned identities and their current descendants"}
    print(json.dumps(result, indent=2))
    suffix = datetime.now(timezone.utc).strftime("%H%M%S")
    (args.guard_dir / f"cleanup_{suffix}.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if denied:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
