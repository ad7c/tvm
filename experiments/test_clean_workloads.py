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
"""Reference and mock-checker tests; not GPU performance experiments."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from clean_workloads import SPECS, checked_clean_evaluator, numerical_cases, reference


class TestCleanWorkloads(unittest.TestCase):
    def test_conv_matches_scalar_reference_and_padding(self):
        image = np.arange(18, dtype="float32").reshape(1, 2, 3, 3)
        weight = (np.arange(36, dtype="float32") - 15).reshape(2, 2, 3, 3)
        actual = reference("conv2d_nchw3x3", [image, weight])
        expected = np.zeros((1, 2, 3, 3))
        for oc in range(2):
            for y in range(3):
                for x in range(3):
                    for ic in range(2):
                        for ky in range(3):
                            for kx in range(3):
                                iy, ix = y + ky - 1, x + kx - 1
                                if 0 <= iy < 3 and 0 <= ix < 3:
                                    expected[0, oc, y, x] += float(image[0, ic, iy, ix]) * float(weight[oc, ic, ky, kx])
        np.testing.assert_array_equal(actual, expected)

    def test_real_shape_conv_ones(self):
        image, weight = [np.ones(shape, dtype="float32") for shape in SPECS["conv2d_nchw3x3"]["inputs"]]
        result = reference("conv2d_nchw3x3", [image, weight])
        self.assertEqual(result.shape, (1, 32, 32, 32))
        self.assertTrue((result[:, :, 1:-1, 1:-1] == 144).all())
        self.assertEqual(result[0, 0, 0, 0], 64)
        self.assertEqual(result[0, 0, 0, 1], 96)

    def test_large_reduction(self):
        value = np.ones((128, 4096), dtype="float32")
        np.testing.assert_array_equal(reference("reduction4096", [value]), np.full(128, 4096))
        cases = list(numerical_cases("reduction4096"))
        self.assertEqual([row[0] for row in cases], ["normal", "uniform", "ones", "zero"])
        self.assertTrue(all(row[2].dtype == np.float64 for row in cases))

    def exercise_mock(self, wrong):
        class Tensor:
            def __init__(self, value, device=None):
                self.value = value.copy()
                self.shape = value.shape

            def numpy(self):
                return self.value.copy()

        class Module:
            entry_name = "main"

            def __getitem__(self, name):
                def run(value, out):
                    out.value[:] = 0 if wrong else value.value.astype("float64").sum(axis=1)
                return run

        evaluator = Mock(return_value=[0.001])
        mocks = {"tvm": SimpleNamespace(runtime=SimpleNamespace(tensor=Tensor)),
                 "tvm.s_tir.meta_schedule.runner.local_runner": SimpleNamespace(default_run_evaluator=evaluator)}
        timing_inputs = [Tensor(np.ones((128, 4096), dtype="float32")), Tensor(np.full(128, 123, dtype="float32"))]
        with tempfile.TemporaryDirectory() as directory, patch.dict("sys.modules", mocks):
            output = Path(directory) / "checks.jsonl"
            def run():
                return checked_clean_evaluator(Module(), SimpleNamespace(sync=lambda: None), None,
                                               [timing_inputs], audit_path=output, candidate_id="mock",
                                               workload="reduction4096")
            if wrong:
                with self.assertRaisesRegex(AssertionError, "correctness failed"):
                    run()
                evaluator.assert_not_called()
            else:
                self.assertEqual(run(), [0.001])
                evaluator.assert_called_once()
            rows = [json.loads(line) for line in output.read_text().splitlines()]
            result = next(row for row in rows if row["event"] == "correctness_result")
            self.assertEqual(result["passed"], not wrong)
            np.testing.assert_array_equal(timing_inputs[1].value, np.full(128, 123))

    def test_checker_accepts_correct_module(self):
        self.exercise_mock(False)

    def test_checker_rejects_wrong_module_before_timing(self):
        self.exercise_mock(True)


if __name__ == "__main__":
    unittest.main()
