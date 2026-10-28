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
"""Verify the resource guard terminates its own tree and preserves unrelated processes."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock

import psutil

from guard_experiment_resources import collect_processes, kill_launched_tree, pressure_reasons


class TestResourceGuard(unittest.TestCase):
    def test_exited_process_memory_is_excluded(self):
        dead = Mock(pid=1)
        dead.num_threads.return_value = 0
        live = Mock(pid=2)
        live.num_threads.return_value = 3
        live.memory_info.return_value = Mock(rss=10, private=20)
        live.name.return_value = "python.exe"
        live.create_time.return_value = 1.0
        live.cmdline.return_value = ["python.exe"]
        records, exited, errors = collect_processes([dead, live])
        self.assertEqual(sum(record["private"] for record in records), 20)
        self.assertEqual(exited, [1])
        self.assertEqual(errors, [])
        dead.memory_info.assert_not_called()

    def test_failed_child_cleanup_does_not_skip_other_children(self):
        root, denied, live = Mock(pid=1), Mock(pid=2), Mock(pid=3)
        root.children.return_value = [denied, live]
        for process in (root, denied, live):
            process.num_threads.return_value = 1
            process.is_running.return_value = False
        denied.kill.side_effect = psutil.AccessDenied(2)
        result = kill_launched_tree(root)
        live.kill.assert_called_once()
        self.assertEqual(result["terminated"], [1, 3])
        self.assertEqual(result["errors"][0]["pid"], 2)

    def test_parent_exit_does_not_hide_known_orphan(self):
        root, orphan = Mock(pid=1), Mock(pid=2)
        root.children.side_effect = psutil.NoSuchProcess(1)
        root.num_threads.side_effect = psutil.NoSuchProcess(1)
        orphan.num_threads.return_value = 1
        orphan.is_running.return_value = False
        result = kill_launched_tree(root, [orphan])
        orphan.kill.assert_called_once()
        self.assertEqual(result["remaining_live"], [])

    def test_memory_domains_independent(self):
        sample = {"system": {"available_physical": 100, "available_commit": 1000},
                  "tree_private_bytes": 500}
        self.assertEqual(pressure_reasons(sample, 200, 0, 1000), ["physical_memory_reserve"])
        self.assertEqual(pressure_reasons(sample, 0, 2000, 1000), ["system_commit_reserve"])
        self.assertEqual(pressure_reasons(sample, 0, 0, 400), ["experiment_private_memory_limit"])

    @unittest.skipUnless(os.name == "nt", "Windows-specific resource guard")
    def test_guard_owns_termination_scope(self):
        sentinel = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        try:
            with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / "tmp") as temp:
                directory = Path(temp)
                script = Path(__file__).with_name("guard_experiment_resources.py")
                cmd = [sys.executable, str(script), "--output", str(directory / "guard"),
                       "--interval", "0.2", "--consecutive-samples", "2",
                       "--physical-reserve-mib", "0", "--commit-reserve-gib", "0",
                       "--private-limit-gib", "0.0625", "--", sys.executable, "-c",
                       "import time; storage=bytearray(128*1024*1024); time.sleep(60)"]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                status = json.loads((directory / "guard/status.json").read_text(encoding="utf-8"))
                self.assertEqual(status["status"], "stopped_for_memory_pressure")
                self.assertIsNone(sentinel.poll(), "The guard must preserve a process it did not launch")
                self.assertEqual(status["cleanup"]["remaining_live"], [])
        finally:
            sentinel.terminate()
            sentinel.wait(timeout=5)

    @unittest.skipUnless(os.name == "nt", "Windows-specific resource guard")
    def test_guard_terminates_live_grandchild(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / "tmp") as temp:
            directory = Path(temp)
            ready = directory / "grandchild.json"
            child_code = (
                "import json,os,psutil,time; from pathlib import Path; "
                f"Path({str(ready)!r}).write_text(json.dumps(dict(pid=os.getpid(),"
                "created=psutil.Process().create_time()))); "
                "storage=bytearray(128*1024*1024); time.sleep(60)"
            )
            parent_code = (
                "import subprocess,sys,time; "
                f"child=subprocess.Popen([sys.executable,'-c',{child_code!r}]); time.sleep(60)"
            )
            script = Path(__file__).with_name("guard_experiment_resources.py")
            cmd = [sys.executable, str(script), "--output", str(directory / "guard"),
                   "--interval", "0.2", "--consecutive-samples", "2",
                   "--physical-reserve-mib", "0", "--commit-reserve-gib", "0",
                   "--private-limit-gib", "0.0625", "--", sys.executable, "-c", parent_code]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
            identity = json.loads(ready.read_text())
            try:
                status = json.loads((directory / "guard/status.json").read_text(encoding="utf-8"))
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                self.assertEqual(status["cleanup"]["remaining_live"], [])
                self.assertIn(identity["pid"], status["cleanup"]["terminated"])
                try:
                    process = psutil.Process(identity["pid"])
                    self.assertTrue(process.create_time() != identity["created"] or process.num_threads() == 0)
                except psutil.NoSuchProcess:
                    pass
            finally:
                try:
                    process = psutil.Process(identity["pid"])
                    if process.create_time() == identity["created"] and process.num_threads() > 0:
                        process.kill()
                except psutil.NoSuchProcess:
                    pass


if __name__ == "__main__":
    unittest.main()
