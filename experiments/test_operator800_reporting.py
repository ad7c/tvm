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
"""Check that training-history resets and missing pre-update checks cannot pass audit."""

import unittest

from report_operator800_study import training_progress


class TestTrainingAudit(unittest.TestCase):
    def example(self, sizes=(10, 20)):
        result = {"measurement": {"callback_batches": []}, "timeline": {"batches": []}}
        row = {"curve": [], "xgb_data_size": sizes[-1]}
        for index, size in enumerate(sizes):
            result["measurement"]["callback_batches"].append({"timeline_batch_index": index,
                "cost_model": {"data_size": size, "last_train_size": size, "trained": True,
                               "best_score": "0.1", "best_metrics": "[1] tr-p-rmse:0.1\ttr-a-peak@32:0.8"}})
            result["timeline"]["batches"].append({"callback_s": {"UpdateCostModel": 0.2}})
            row["curve"].append({"submitted": (index + 1) * 10, "tune_s": (index + 1) * 5})
        events = [{"event": "xgb_log", "batch_index": 1,
                   "message": "XGB validation: p-rmse: 0.3\ta-peak@32: 0.7"}]
        return result, row, events

    def test_fit_separate_from_predictions_before_update(self):
        points = training_progress(*self.example())
        self.assertIsNone(points[0]["next_batch_rmse"])
        self.assertEqual(points[1]["train_rmse"], 0.1)
        self.assertEqual(points[1]["next_batch_rmse"], 0.3)
        self.assertEqual(points[1]["data_size"], 20)

    def test_history_reset_rejected(self):
        with self.assertRaisesRegex(ValueError, "history"):
            training_progress(*self.example(sizes=(10, 5)))

    def test_missing_pre_update_validation_rejected(self):
        result, row, _ = self.example()
        with self.assertRaisesRegex(ValueError, "pre-update"):
            training_progress(result, row, [])


if __name__ == "__main__":
    unittest.main()
