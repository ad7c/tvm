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
"""Check all predeclared targets, including targets that were never attained."""

import unittest

from plot_operator800_comparison import target_rows


class TestComparisonTargets(unittest.TestCase):
    def test_strict_target_remains_unattained_when_slack_target_is_attained(self):
        rows = [{"arm": "B", "best_us": 27.0, "tune_s": 100,
                 "curve": [{"best_us": 29.0, "submitted": 10, "tune_s": 10},
                           {"best_us": 27.0, "submitted": 800, "tune_s": 100}]},
                {"arm": "C", "best_us": 27.1, "tune_s": 110,
                 "curve": [{"best_us": 27.1, "submitted": 200, "tune_s": 25},
                           {"best_us": 27.1, "submitted": 800, "tune_s": 110}]}]
        targets = target_rows(rows, (1.0, 1.01, 1.05))
        self.assertEqual(len(targets), 6)
        strict = next(target for target in targets if target["arm"] == "C" and target["ratio"] == 1.0)
        self.assertFalse(strict["attained"])
        self.assertIsNone(strict["tune_s"])
        slack = next(target for target in targets if target["arm"] == "C" and target["ratio"] == 1.01)
        self.assertTrue(slack["attained"])
        self.assertEqual(slack["submitted"], 200)
        self.assertEqual(slack["tune_s"], 25)


if __name__ == "__main__":
    unittest.main()
