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
"""Driver unit tests; mock runtime tests are NOT CUDA experiment evidence."""

import json
import logging
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from p3_observability import TuneJournal, checked_matmul_evaluator, classify_measurements, summarize_timeline


class TestP3(unittest.TestCase):
    def test_mutually_exclusive_errors(self):
        batch = {"builder_errors": ["compile", None, None, None, None],
                 "runner_errors": ["compile", "run", None, None, None],
                 "runner_run_secs": [None, None, [float("nan")], [1.0, float("inf")], [0.1]]}
        self.assertEqual(classify_measurements(batch),
                         ["build_error", "runner_error", "invalid_latency", "invalid_latency", "finite"])

    def test_mismatched_slots(self):
        with self.assertRaises(ValueError):
            classify_measurements({"builder_errors": [], "runner_errors": [None], "runner_run_secs": []})

    def test_logger_factory_and_phases(self):
        original = logging.getLogRecordFactory()
        logger = logging.getLogger("tvm.s_tir.meta_schedule.task_scheduler.task_scheduler")
        old_level = logger.level
        logger.setLevel(logging.INFO)
        try:
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "events.jsonl"
                with TuneJournal(path) as journal:
                    logger.info("[task_scheduler.cc:1] TaskScheduler picks Task #0: main")
                    logger.info("Sending 2 sample(s) to builder")
                    logger.info("Sending 1 valid sample(s) to runner")
                    journal.emit("callback_begin", callback_index=0, name="UpdateCostModel")
                    journal.emit("callback_end", callback_index=0, name="UpdateCostModel")
                    journal.emit("batch_complete", candidates=2)
                    logger.info("[Updated] Task #0: main")
                result = summarize_timeline(journal.events)
                self.assertEqual(result["batches"][0]["submitted"], 2)
                self.assertEqual(result["batches"][0]["sent_to_runner"], 1)
                self.assertGreaterEqual(result["residual_s"], 0)
                self.assertEqual(len(path.read_text(encoding="utf-8").splitlines()), len(journal.events))
            self.assertIs(logging.getLogRecordFactory(), original)
        finally:
            logger.setLevel(old_level)

    def test_factory_restored_on_exception(self):
        original = logging.getLogRecordFactory()
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "injected"):
                with TuneJournal(Path(directory) / "events.jsonl"):
                    raise RuntimeError("injected")
        self.assertIs(logging.getLogRecordFactory(), original)

    def test_training_logs_opt_in_and_survive_handler_replacement(self):
        logger = logging.getLogger("tvm.s_tir.meta_schedule.cost_model.xgb_model")
        previous_level, previous_handlers = logger.level, logger.handlers[:]
        logger.setLevel(logging.DEBUG)
        try:
            with tempfile.TemporaryDirectory() as directory:
                for enabled in (False, True):
                    with TuneJournal(Path(directory) / f"events-{enabled}.jsonl", enabled) as journal:
                        logger.handlers = [logging.NullHandler()]
                        logger.debug("XGB validation: p-rmse: 0.123456\ta-peak@32: 0.750000")
                    captured = [e for e in journal.events if e["event"] == "xgb_log"]
                    self.assertEqual(len(captured), int(enabled))
        finally:
            logger.setLevel(previous_level)
            logger.handlers = previous_handlers

    def test_missing_callback_fails(self):
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            summarize_timeline([{"event": "builder_begin", "batch_index": 0}])

    def check_fake_module(self, wrong):
        class Tensor:
            def __init__(self, array, _device=None):
                self.array = array.copy()
                self.shape, self.dtype = array.shape, array.dtype

            def numpy(self):
                return self.array.copy()

        class RuntimeModule:
            entry_name = "main"

            def __getitem__(self, name):
                def run(a, b, c):
                    c.array[:] = 0 if wrong else a.array @ b.array
                return run

        evaluator = Mock(return_value=[0.01])
        mocks = {"tvm": SimpleNamespace(runtime=SimpleNamespace(tensor=Tensor)),
                 "tvm.s_tir.meta_schedule.runner.local_runner": SimpleNamespace(default_run_evaluator=evaluator)}
        inputs = [Tensor(np.full((4, 4), x, dtype="float32")) for x in (1, 2, 3)]
        with tempfile.TemporaryDirectory() as directory, patch.dict("sys.modules", mocks):
            path = Path(directory) / "check.jsonl"
            call = lambda: checked_matmul_evaluator(RuntimeModule(), SimpleNamespace(sync=lambda: None),
                                                    None, [inputs], audit_path=path, candidate_id="fake0")
            if wrong:
                with self.assertRaisesRegex(AssertionError, "correctness failed"):
                    call()
                evaluator.assert_not_called()
            else:
                self.assertEqual(call(), [0.01])
                evaluator.assert_called_once()
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            result = next(row for row in rows if row["event"] == "correctness_result")
            self.assertEqual(result["passed"], not wrong)
            for tensor, value in zip(inputs, (1, 2, 3)):
                np.testing.assert_array_equal(tensor.array, value)

    def test_correct_kernel_and_untouched_timing_buffers(self):
        self.check_fake_module(False)

    def test_wrong_kernel_rejected_before_timing(self):
        self.check_fake_module(True)


if __name__ == "__main__":
    unittest.main()
