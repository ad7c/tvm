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
"""Exercise bounded queue lifecycle without launching TVM or GPU workers."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import resume_operator800_20261024 as resume
from run_operator800_study import matrix


class FakeController:
    def __init__(self, folder, fail=None, wrong_root=False):
        self.folder = Path(folder)
        self.protocol = {"order": matrix(), "seed": 20261011}
        accepted = ("S01", "S02", "S03", "S04", "M128G1B01", "M128G1C01",
                    "M128G1D01", "M256G1C01", "CONVG1B01", "CONVG1D01")
        self.manifest = {"accepted": {key: {"root": "historical"} for key in accepted},
                         "retiming": {"matmul128": {}}, "pytorch": {"matmul128": {}}}
        self.calls = []
        self.roots = []
        self.fail = fail
        self.wrong_root = wrong_root

    def new_root(self, identifier, phase):
        root = self.folder / f"attempt_{len(self.roots)}"
        root.mkdir()
        self.roots.append(root)
        return root

    def execute(self, root, cmd, phase, identifier, env):
        phase = "tuning" if phase == "history_resume" else phase
        self.calls.append((phase, identifier))
        (root / "partial.log").write_text("evidence", encoding="utf-8")
        if (phase, identifier) == self.fail:
            return False
        self.manifest["accepted" if phase == "tuning" else phase][identifier] = {
            "root": str(self.folder / "wrong" if self.wrong_root else root)}
        return True


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.command_patch = patch.object(resume, "make_command", return_value=(["frozen"], {}))
        self.command_patch.start()
        self.addCleanup(self.command_patch.stop)

    def test_only_authorized_tasks_and_no_repeat_on_completed_queue(self):
        ctl = FakeController(self.folder.name)
        resume.run_sequence(ctl)
        self.assertEqual(ctl.calls, list(resume.STEPS))
        self.assertFalse(any("RED" in task for _, task in ctl.calls))
        self.assertEqual(len(set(ctl.roots)), 7)
        resume.run_sequence(ctl)
        self.assertEqual(len(ctl.calls), 7)

    def test_failure_stops_queue_and_preserves_partial_attempt(self):
        ctl = FakeController(self.folder.name, fail=("retiming", "conv2d_nchw3x3"))
        with self.assertRaisesRegex(RuntimeError, "Stopped after"):
            resume.run_sequence(ctl)
        self.assertEqual(ctl.calls, list(resume.STEPS[:2]))
        self.assertEqual((ctl.roots[-1] / "partial.log").read_text(), "evidence")
        self.assertIn("CONVG1C01", ctl.manifest["accepted"])
        self.assertNotIn("M256G1D01", ctl.manifest["accepted"])

    def test_missing_smoke_stops_before_creating_attempt(self):
        ctl = FakeController(self.folder.name)
        del ctl.manifest["accepted"]["S03"]
        with self.assertRaisesRegex(RuntimeError, "Required accepted results missing"):
            resume.run_sequence(ctl)
        self.assertEqual(ctl.roots, [])

    def test_acceptance_must_reference_the_new_attempt(self):
        ctl = FakeController(self.folder.name, wrong_root=True)
        with self.assertRaisesRegex(RuntimeError, "Stopped after"):
            resume.run_sequence(ctl)
        self.assertEqual(len(ctl.calls), 1)
        self.assertTrue((ctl.roots[0] / "partial.log").exists())


if __name__ == "__main__":
    unittest.main()
