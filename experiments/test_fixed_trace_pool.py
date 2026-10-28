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
"""Driver control-flow tests only; real C++ behavior is checked by pool runs."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from fixed_trace_pool import ReplayExperiment, plain_json


class ScheduleFailure(RuntimeError):
    pass


class DriverTests(unittest.TestCase):
    def setUp(self):
        self.experiment = ReplayExperiment.__new__(ReplayExperiment)
        exp = self.experiment
        exp.tvm = SimpleNamespace(s_tir=SimpleNamespace(ScheduleError=ScheduleFailure))
        exp.max_threads, exp.warp_size, exp.max_shared = 1024, 32, 49152
        exp.trace_check = Mock(return_value={"truth": 0})
        exp.ir_check = Mock(return_value=[1, 0, 32, 1, 0, 0, 0, 1, 0, 0])
        self.schedule = SimpleNamespace(mod=object(), enter_postproc=Mock())
        exp.new_schedule = Mock(return_value=self.schedule)
        exp.postprocs = [SimpleNamespace(apply=Mock(return_value=True))]
        exp.postproc_names = ["VerifyGPUCode"]
        self.trace = SimpleNamespace(apply_to_schedule=Mock())

    def test_known_and_unknown_survivors_both_receive_ir_check(self):
        for truth in (0, 2):
            with self.subTest(truth=truth):
                self.experiment.trace_check.return_value = {"truth": truth}
                self.experiment.ir_check.reset_mock()
                row, schedule = self.experiment.evaluate(self.trace, 1, "trace")
                self.assertTrue(row["accepted"])
                self.assertIs(schedule, self.schedule)
                self.experiment.ir_check.assert_called_once()
                self.assertAlmostEqual(sum(row["timing_s"].values()), row["total_s"])

    def test_trace_reject_does_not_create_or_replay_schedule(self):
        self.experiment.trace_check.return_value = {"truth": 1}
        row, _ = self.experiment.evaluate(self.trace, 1, "trace")
        self.assertEqual(row["status"], "hard_trace_reject")
        self.experiment.new_schedule.assert_not_called()
        self.trace.apply_to_schedule.assert_not_called()

    def test_postreplay_reject_skips_postprocs(self):
        self.experiment.ir_check.return_value = [1, 1, 8]
        row, _ = self.experiment.evaluate(self.trace, 1, "postreplay")
        self.assertEqual(row["status"], "hard_postreplay_reject")
        self.trace.apply_to_schedule.assert_called_once()
        self.experiment.postprocs[0].apply.assert_not_called()

    def test_api_bug_is_not_an_invalid_candidate(self):
        self.trace.apply_to_schedule.side_effect = AttributeError("wrong API")
        with self.assertRaises(AttributeError):
            self.experiment.evaluate(self.trace, 1, "baseline")

    def test_analysis_error_is_not_an_invalid_candidate(self):
        self.experiment.ir_check.side_effect = ScheduleFailure("analysis bug")
        with self.assertRaises(ScheduleFailure):
            self.experiment.evaluate(self.trace, 1, "postreplay")

    def test_schedule_failure_is_recorded(self):
        self.trace.apply_to_schedule.side_effect = ScheduleFailure("candidate cannot replay")
        row, schedule = self.experiment.evaluate(self.trace, 1, "baseline")
        self.assertEqual(row["status"], "replay_exception")
        self.assertFalse(row["accepted"])
        self.assertIsNone(schedule)

    def test_postproc_failure_is_recorded(self):
        self.experiment.postprocs[0].apply.return_value = False
        row, _ = self.experiment.evaluate(self.trace, 1, "baseline")
        self.assertEqual(row["status"], "postproc_0_VerifyGPUCode_reject")

    def test_plain_json_rejects_unknown_objects(self):
        with self.assertRaises(TypeError):
            plain_json(object())


if __name__ == "__main__":
    unittest.main()
