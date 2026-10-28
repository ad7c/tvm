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
"""Regression checks for preserving interrupted experiment evidence."""

import json

import pytest

import model_scout_pool_study as study


def test_database_snapshot_retains_complete_records_and_original_suffix(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    record = source / "database_tuning_record.json"
    original = '[1, 2]\n[3, 4]\n[5,'
    record.write_text(original, encoding="utf-8")
    (source / "database_workload.json").write_text('["workload"]\n', encoding="utf-8")
    target = tmp_path / "snapshot"
    assert study.copy_database(source, target) == 2
    assert record.read_text(encoding="utf-8") == original
    assert (target / record.name).read_text(encoding="utf-8") == '[1, 2]\n[3, 4]\n'
    with pytest.raises(FileExistsError):
        study.copy_database(source, target)


def test_database_snapshot_rejects_interior_corruption(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "database_tuning_record.json").write_text('[1]\nbad\n[2]\n', encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        study.copy_database(source, tmp_path / "snapshot")


def test_architecture_audit_does_not_pass_empty_telemetry(tmp_path):
    (tmp_path / "model_predictions.jsonl").write_text("", encoding="utf-8")
    (tmp_path / "logs").mkdir()
    with pytest.raises(RuntimeError, match="No nonempty scoring events"):
        study.audit_arm(tmp_path, "D64")
