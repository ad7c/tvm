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
"""Keep the user's status path live while preserving the audit log and raw data."""

import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from watch_operator800_cohort import ensure_running, publish, render


class TestLiveStatus(unittest.TestCase):
    def check(self):
        return {"local_time": "2026-10-11T20:00:00+08:00", "last_audit_time": "19:44:45",
                "active": None, "stage": None, "controller_alive": True, "formal_complete": 1,
                "rows": [{"id": "M128G1B01", "budget": 800, "submitted": 800,
                          "data_size": 800, "best_us": 27.246, "state": "已审核完整组"},
                         {"id": "M128G1C01", "budget": 800, "submitted": 565,
                          "data_size": 565, "best_us": 27.253, "state": "调优中"}]}

    def test_existing_publisher_prevents_duplicate_launch(self):
        with patch("watch_operator800_cohort.watcher_alive", return_value=True), \
             patch("watch_operator800_cohort.subprocess.Popen") as launch:
            ensure_running(Path("unused"))
            launch.assert_not_called()

    def test_status_refresh_keeps_audit_and_completed_group(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            monitor = output / "monitor15min"
            monitor.mkdir()
            (monitor / "checks.jsonl").write_text("original audit\n", encoding="utf-8")
            (monitor / "AUDIT_CHECKPOINT.md").write_text("original checkpoint\n", encoding="utf-8")
            check = self.check()
            publish(output, check, 30)
            self.assertIn("565/800", (monitor / "STATUS.md").read_text(encoding="utf-8"))
            check["local_time"] = "2026-10-11T20:00:30+08:00"
            check["rows"][1].update(submitted=575, data_size=575)
            publish(output, check, 30)
            text = (monitor / "STATUS.md").read_text(encoding="utf-8")
            self.assertIn("575/800", text)
            self.assertIn("800/800", text)
            self.assertIn("20:00:30", text)
            self.assertEqual((monitor / "checks.jsonl").read_text(encoding="utf-8"), "original audit\n")
            self.assertEqual((monitor / "AUDIT_CHECKPOINT.md").read_text(encoding="utf-8"), "original checkpoint\n")
            self.assertEqual((monitor / "AUDIT_STATUS.md").read_text(encoding="utf-8"), text)
            self.assertEqual((monitor / "CURRENT_RUN.md").read_text(encoding="utf-8"), text)

    def test_current_stage_and_physical_resources_are_shown(self):
        check = self.check()
        check["active"] = {"progress": {"current": "M128G1C01"}, "guard_status": "running",
                           "issues": [], "memory": {"tree_private_bytes": 2**30,
                             "system": {"available_physical": 2**30, "available_commit": 10 * 2**30}}}
        check["stage"] = {"name": "候选编译"}
        text = render(check, 30)
        self.assertIn("当前阶段：候选编译", text)
        self.assertIn("10.00 GiB", text)
        self.assertIn("完整巡检每 15 分钟", text)


if __name__ == "__main__":
    unittest.main()
