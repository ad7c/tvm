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
"""Validate feedback restoration and remaining-budget guards without GPU work."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from operator800_history_resume import History
from complete_operator800_study import guard_wrapper_alive
from operator800_exhaustion_audit import validate_variable_batches


class HistoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.source = Path(self.temp.name) / "source"
        self.source.mkdir()
        self.output = Path(self.temp.name) / "output"
        self.batch = {"builder_errors": [None, None], "runner_errors": [None, None],
                      "runner_run_secs": [[0.01], [0.02]], "trace_sha256": ["first", "second"],
                      "artifact_paths": ["artifact1", "artifact2"]}
        self.write("callback_batches.jsonl", [self.batch])
        self.write("database_tuning_record.json", [[0, [[], [0.01]]], [0, [[], [0.02]]]])
        self.write("runner_inputs.jsonl", [{"artifact_path": f"artifact{i}", "candidate_id": f"runner-{i}"}
                                           for i in (1, 2)])
        self.write("correctness.jsonl", [{"candidate_id": f"runner-{i}", "event": "correctness_result",
                                         "passed": True, "cases": [{"passed": True}] * 4} for i in (1, 2)])
        for name in ("database_workload.json", "identity.json", "timeline.jsonl", "command.json"):
            (self.source / name).write_text("{}\n", encoding="utf-8")

    def write(self, name, rows):
        (self.source / name).write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    def test_restore_does_not_spend_new_measurement_budget(self):
        history = History(self.source, self.output, budget=5)
        history.finish_batch(["first", "second"], 2, 2, 10.0)
        self.assertFalse(history.replaying)
        self.assertEqual(history.restored, 2)
        self.assertEqual(history.new, 0)
        history.finish_batch(["new1", "new2", "new3"], 3, 5, 20.0)
        self.assertEqual(history.total + history.new, 5)

    def test_trace_mismatch_refuses_continuation(self):
        history = History(self.source, self.output, budget=5)
        with self.assertRaisesRegex(ValueError, "replay differs"):
            history.finish_batch(["second", "first"], 2, 2, 10.0)
        self.assertTrue(history.replaying)
        self.assertEqual(history.new, 0)

    def test_model_history_mismatch_is_rejected(self):
        history = History(self.source, self.output, budget=5)
        with self.assertRaisesRegex(ValueError, "model sample history"):
            history.finish_batch(["first", "second"], 2, 1, 10.0)

    def test_database_feedback_disagreement_is_rejected(self):
        self.write("database_tuning_record.json", [[0, [[], [0.03]]], [0, [[], [0.02]]]])
        with self.assertRaisesRegex(ValueError, "measurements differ"):
            History(self.source, self.output, budget=5)

    def test_remaining_budget_cannot_be_exceeded(self):
        history = History(self.source, self.output, budget=5)
        history.finish_batch(["first", "second"], 2, 2, 10.0)
        with self.assertRaisesRegex(ValueError, "budget"):
            history.finish_batch(["a", "b", "c", "d"], 4, 6, 20.0)

    def test_finished_guard_does_not_inspect_reused_pid(self):
        (self.source / "resource_guard").mkdir()
        (self.source / "resource_guard/status.json").write_text(
            json.dumps({"status": "completed", "guard_pid": 7708}), encoding="utf-8")
        with patch("complete_operator800_study.psutil.Process") as process:
            self.assertFalse(guard_wrapper_alive(self.source))
            process.assert_not_called()

    def test_early_stop_allows_intermittent_empty_callbacks(self):
        batches = [{key: [None] * n for key in ("builder_errors", "runner_errors", "runner_run_secs")}
                   for n in (10, 0, 2, 0)]
        self.assertEqual(validate_variable_batches(batches, 5, 800, 10, 12), [])
        self.assertTrue(validate_variable_batches(batches, 4, 800, 10, 12))
        self.assertTrue(validate_variable_batches(batches, 5, 800, 10, 13))


if __name__ == "__main__":
    unittest.main()
