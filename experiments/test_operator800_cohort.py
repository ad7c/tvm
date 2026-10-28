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
"""Recovery provenance and scheduling tests; no CUDA tuning is launched."""

import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from complete_operator800_study import (Controller, choose_tuning, verified_process,
                                       tuning_log_path, validate_tuning_paths, tuning_order)
from operator800_cohort import (accept_tuning, check_receipt, collect_cohort,
                               matching_receipt, object_hash)
from run_operator800_study import command, matrix
from watch_operator800_study import write_json_atomic


class TestCohortRecovery(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.protocol = {"order": matrix(), "seed": 20261011,
                         "common_config": {"adaptive_training": False}}
        self.spec = self.protocol["order"][1]
        write_json_atomic(self.root / "protocol.json", self.protocol)

    def receipt(self, returncode=0):
        entry = {"id": self.spec["id"], "phase": "tuning", "returncode": returncode,
                 "argv": command(self.spec, self.root, self.protocol["seed"]), "console_sha256": "console"}
        (self.root / "execution.jsonl").write_text(json.dumps(entry) + "\n", encoding="utf-8")
        return entry

    def manifest(self):
        return {"accepted": {"S01": {"submitted": 4}}, "failures": []}

    def test_successful_receipt_is_required(self):
        self.receipt(1)
        with self.assertRaisesRegex(ValueError, "successful"):
            matching_receipt(self.root, "tuning", self.spec["id"])
        self.receipt(0)
        reference = matching_receipt(self.root, "tuning", self.spec["id"])
        self.assertEqual(check_receipt(reference)["returncode"], 0)

    def test_receipt_modification_is_rejected(self):
        entry = self.receipt()
        reference = matching_receipt(self.root, "tuning", self.spec["id"])
        entry["argv"][0] = "another-runtime.exe"
        (self.root / "execution.jsonl").write_text(json.dumps(entry) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "changed"):
            check_receipt(reference)

    def test_duplicate_receipts_are_rejected(self):
        entry = self.receipt()
        with (self.root / "execution.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(entry) + "\n")
        with self.assertRaisesRegex(ValueError, "unique"):
            matching_receipt(self.root, "tuning", self.spec["id"])

    def test_cleanup_failure_is_not_a_successful_phase(self):
        entry = self.receipt()
        entry["guard_status"] = "cleanup_incomplete"
        (self.root / "execution.jsonl").write_text(json.dumps(entry) + "\n", encoding="utf-8")
        reference = matching_receipt(self.root, "tuning", self.spec["id"])
        with self.assertRaisesRegex(ValueError, "successfully"):
            check_receipt(reference)

    def test_protocol_mismatch_is_rejected_before_reading_results(self):
        changed = dict(self.protocol, seed=1)
        with self.assertRaisesRegex(ValueError, "protocol differs"):
            accept_tuning(self.root, self.spec, changed)

    def test_partial_budget_cannot_be_accepted_as_800(self):
        self.receipt()
        run = self.root / (self.spec["id"] + "_example")
        run.mkdir()
        write_json_atomic(run / "result.json", {"config": {"record_model_training": True}})
        with patch("operator800_cohort.check_console"), patch("operator800_cohort.audit", return_value={
                "budget": {"actual_submitted": 790}}):
            with self.assertRaisesRegex(ValueError, "Short budget: 790/800"):
                accept_tuning(self.root, self.spec, self.protocol)

    def test_completed_arm_is_retained_and_next_arm_selected(self):
        manifest = self.manifest()
        manifest["accepted"][self.spec["id"]] = {"submitted": 800}
        selected = choose_tuning(manifest, self.protocol, 2)
        self.assertEqual(selected["id"], "M128G1C01")

    def test_user_priority_runs_convolution_then_reduction_before_matmul256(self):
        manifest = self.manifest()
        original_order = json.loads(json.dumps(self.protocol["order"]))
        manifest["scheduling"] = {"workload_priority": ["matmul128", "conv2d_nchw3x3",
                                                        "reduction4096", "matmul256"]}
        for identifier in ("M128G1B01", "M128G1C01", "M128G1D01"):
            manifest["accepted"][identifier] = {"submitted": 800}
        self.assertEqual(choose_tuning(manifest, self.protocol, 2)["id"], "S03")
        manifest["accepted"]["S03"] = {"submitted": 2}
        self.assertEqual(choose_tuning(manifest, self.protocol, 2)["id"], "CONVG1D01")
        for identifier in ("CONVG1D01", "CONVG1B01", "CONVG1C01"):
            manifest["accepted"][identifier] = {"submitted": 800}
        self.assertEqual(choose_tuning(manifest, self.protocol, 2)["id"], "S04")
        manifest["accepted"]["S04"] = {"submitted": 2}
        self.assertEqual(choose_tuning(manifest, self.protocol, 2)["id"], "REDG1B01")
        for identifier in ("REDG1B01", "REDG1C01", "REDG1D01"):
            manifest["accepted"][identifier] = {"submitted": 800}
        self.assertEqual(choose_tuning(manifest, self.protocol, 2)["id"], "S02")
        self.assertEqual(self.protocol["order"], original_order)

    def test_priority_cannot_drop_or_duplicate_a_registered_workload(self):
        manifest = self.manifest()
        for priority in (["matmul128"], ["matmul128", "matmul256", "reduction4096", "reduction4096"]):
            manifest["scheduling"] = {"workload_priority": priority}
            with self.assertRaisesRegex(ValueError, "each registered workload"):
                tuning_order(manifest, self.protocol)

    def test_retry_limit_skips_repeated_failure_but_other_arms_continue(self):
        manifest = self.manifest()
        manifest["failures"] = [{"phase": "tuning", "id": self.spec["id"]}] * 2
        selected = choose_tuning(manifest, self.protocol, 2)
        self.assertEqual(selected["id"], "M128G1C01")

    def test_exhaustion_is_not_blindly_retried(self):
        manifest = self.manifest()
        manifest["failures"] = [{"phase": "tuning", "id": self.spec["id"], "short_budget": True}]
        self.assertEqual(choose_tuning(manifest, self.protocol, 2)["id"], "M128G1C01")

    def test_failed_smoke_blocks_its_workload_but_not_other_workloads(self):
        manifest = {"accepted": {}, "failures": [{"phase": "smoke", "id": "S01"}] * 2}
        self.assertEqual(choose_tuning(manifest, self.protocol, 2)["id"], "S02")

    def test_postprocessing_failure_does_not_block_tuning(self):
        manifest = self.manifest()
        manifest["failures"] = [{"phase": "retiming", "id": "matmul128"}] * 2
        self.assertEqual(choose_tuning(manifest, self.protocol, 2), self.spec)

    def test_diagnosed_resolved_failure_can_be_retried_without_deleting_history(self):
        manifest = self.manifest()
        manifest["failures"] = [{"phase": "tuning", "id": self.spec["id"],
                                 "resolved_by": "physical-reserve-policy-20261011"}] * 2
        self.assertEqual(choose_tuning(manifest, self.protocol, 2), self.spec)
        self.assertEqual(len(manifest["failures"]), 2)

    def test_all_short_raw_paths_fit_windows_limit(self):
        root = Path("F:/tvm/experiments/results/o8_123abc_999")
        longest = validate_tuning_paths(root, self.protocol)
        self.assertLessEqual(len(str(longest)), 240)
        self.assertEqual(len(self.protocol["order"]), 16)

    def test_original_nested_recovery_paths_exceed_windows_limit(self):
        root = Path("F:/tvm/experiments/results/paper_v1/operator800_20261011_cohort/attempts/M128G1C01_tuning_A01")
        path = tuning_log_path(root, self.protocol["order"][2], self.protocol["seed"])
        self.assertGreater(len(str(path)), 259)

    def test_compile_physical_dip_with_commit_headroom_does_not_abort(self):
        from guard_experiment_resources import pressure_reasons
        sample = {"system": {"available_physical": 700 * 2**20, "available_commit": 11 * 2**30},
                  "tree_private_bytes": 1700 * 2**20}
        self.assertEqual(pressure_reasons(sample, 256 * 2**20, 4 * 2**30, 4 * 2**30), [])
        sample["system"]["available_commit"] = 3 * 2**30
        self.assertEqual(pressure_reasons(sample, 256 * 2**20, 4 * 2**30, 4 * 2**30), ["system_commit_reserve"])

    def test_invalid_process_identity_cannot_be_targeted(self):
        self.assertIsNone(verified_process(-1, None))
        self.assertIsNone(verified_process(None, 123))

    def test_empty_cohort_does_not_claim_completion(self):
        from report_single_seed_study import digest
        manifest = {"protocol_path": str(self.root / "protocol.json"),
                    "protocol_sha256": digest(self.root / "protocol.json"),
                    "accepted": {}, "retiming": {}, "pytorch": {}}
        path = self.root / "cohort.json"
        write_json_atomic(path, manifest)
        report = collect_cohort(path)
        self.assertEqual(len(report["pending"]), 12)
        self.assertFalse(report["completion_ready"])

    def test_adoption_waits_without_launching_new_gpu_task(self):
        output = self.root / "cohort"
        output.mkdir()
        args = argparse.Namespace(input=self.root, output=output, poll=0.001, interval=900)
        controller = Controller(args, {"status": "initializing"}, self.protocol)
        with patch.object(controller, "refresh"), patch.object(controller, "monitor"), \
                patch.object(controller, "save"), \
                patch("complete_operator800_study.guard_wrapper_alive", side_effect=[True, False]), \
                patch("complete_operator800_study.clean_stale_children"), \
                patch("complete_operator800_study.subprocess.Popen") as launch:
            controller.wait_existing(self.root)
            launch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
