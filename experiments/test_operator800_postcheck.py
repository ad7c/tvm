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
"""Variable batch budgets and narrowly scoped postcheck recovery tests."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

from e1_thread_filter import _validate_callback_batch_contract
from operator800_postcheck import callback_only_failure, revalidate, validate_batches, validate_evidence


def batches(sizes):
    return [{"builder_errors": [None] * size, "runner_errors": [None] * size,
             "runner_run_secs": [[1e-5]] * size} for size in sizes]


class TestPostcheck(unittest.TestCase):
    def test_short_intermediate_batches_reach_full_budget(self):
        # The real run had six intermediate batches of nine and one last batch of six.
        data = batches([10] * 74 + [9] * 6 + [6])
        self.assertEqual(validate_batches(data, 81, 800, 10, 800), [])
        self.assertTrue(_validate_callback_batch_contract(data, 81, 800, 10, 800))

    def test_short_budget_is_consistent_but_never_faked_as_full(self):
        data = batches([10, 9, 7])
        self.assertEqual(validate_batches(data, 3, 800, 10, 26), [])
        self.assertTrue(validate_batches(data, 3, 800, 10, 800))

    def test_oversized_batch_and_overall_budget_rejected(self):
        self.assertTrue(validate_batches(batches([11]), 1, 800, 10, 11))
        self.assertTrue(validate_batches(batches([10] * 81), 81, 800, 10, 810))

    def test_missing_slots_and_search_batches_rejected(self):
        data = batches([10, 9])
        data[1]["runner_run_secs"].pop()
        self.assertTrue(validate_batches(data, 2, 19, 10, 19))
        self.assertTrue(validate_batches(batches([10, 9]), 3, 19, 10, 19))

    def test_exhaustion_allows_only_trailing_empty_batches(self):
        self.assertEqual(validate_batches(batches([9, 0, 0]), 3, 800, 10, 9), [])
        self.assertTrue(validate_batches(batches([9, 0, 1]), 3, 800, 10, 10))

    def failed_result(self):
        data = batches([9, 10, 1])
        errors = _validate_callback_batch_contract(data, 3, 20, 10, 20)
        return {"config": {"trials": 20, "trials_per_iter": 10},
                "search": {"search_batches": 3},
                "measurement": {"callback_batches": data, "database_records": 20},
                "architecture_validation": {"passed": False, "errors": errors,
                                            "callback_batch_errors": errors.copy()}}

    def test_only_exact_known_callback_error_can_be_recovered(self):
        result = self.failed_result()
        self.assertTrue(callback_only_failure(result))
        result["architecture_validation"]["errors"].append("Finite measurement lacks matching numerical check")
        with self.assertRaisesRegex(ValueError, "Not solely"):
            callback_only_failure(result)

    def test_altered_callback_error_or_already_valid_result_rejected(self):
        for changed in ("error", "passed"):
            result = self.failed_result()
            if changed == "error":
                result["architecture_validation"]["callback_batch_errors"] = ["other failure"]
            else:
                result["architecture_validation"]["passed"] = True
            with self.assertRaises(ValueError):
                callback_only_failure(result)

    def test_resource_abort_is_never_recovered_as_postcheck(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "result.json"
            path.write_text(json.dumps(self.failed_result()), encoding="utf-8")
            (root / "resource_guard").mkdir()
            (root / "resource_guard/status.json").write_text(
                json.dumps({"status": "stopped_for_memory_pressure", "returncode": 1}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "ordinary exit 1"):
                revalidate(path, root, {"id": "B"}, {}, {"returncode": 1})

    def evidence(self, root):
        data = batches([1])
        data[0].update(artifact_paths=["a.tar"], trace_sha256=["trace"],
                       timeline_batch_index=0, available_at_tune_s=2.0)
        result = {"measurement": {"callback_batches": data, "database_records": 1,
                    "trace_hashes": ["trace"], "record_latencies_us": [10.0],
                    "finite_records": 1, "best_latency_us": 10.0},
                  "config": {"trials": 1, "trials_per_iter": 10},
                  "search": {"search_batches": 1}, "budget": {"actual_submitted": 1},
                  "timeline": {"batches": [{"batch_index": 0, "submitted": 1,
                                             "available_at_tune_s": 2.0}]},
                  "timing": {"tune_call_s": 2.1},
                  "correctness": {"finite_checked": 1, "passed_candidates": 1, "runner_inputs": 1}}
        fixtures = {"callback_batches.jsonl": data,
                    "database_tuning_record.json": [[0, [[], [1e-5]]]],
                    "runner_inputs.jsonl": [{"artifact_path": "a.tar", "candidate_id": "runner-0"}],
                    "correctness.jsonl": [{"event": "correctness_result", "candidate_id": "runner-0",
                                            "passed": True, "cases": [{"passed": True}] * 4}]}
        for name, values in fixtures.items():
            (root / name).write_text("".join(json.dumps(value) + "\n" for value in values), encoding="utf-8")
        return result

    def test_raw_evidence_and_missing_correctness_are_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = self.evidence(root)
            validate_evidence(root / "result.json", result)
            (root / "correctness.jsonl").write_text("", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "four passed"):
                validate_evidence(root / "result.json", result)

    def test_changed_database_timing_and_trace_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = self.evidence(root)
            (root / "database_tuning_record.json").write_text(json.dumps([0, [[], [2e-5]]]) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Raw database timings"):
                validate_evidence(root / "result.json", result)
            result = self.evidence(root)
            result["measurement"]["trace_hashes"] = ["different"]
            with self.assertRaisesRegex(ValueError, "trace identities"):
                validate_evidence(root / "result.json", result)


if __name__ == "__main__":
    unittest.main()
