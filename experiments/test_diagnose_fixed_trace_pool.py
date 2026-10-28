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
"""CPU-only mock observer tests; these do not validate any TVM/CUDA behavior."""

from threading import get_ident
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from diagnose_fixed_trace_pool import VerifyObserver, make_instrument
from fixed_trace_pool import ReplayExperiment


class ScheduleFailure(RuntimeError):
    """Model the candidate exception that ReplayExperiment intentionally catches."""


def fake_tvm():
    # The identity decorator avoids importing TVM while retaining callback logic.
    return SimpleNamespace(
        instrument=SimpleNamespace(pass_instrument=lambda cls: cls),
        s_tir=SimpleNamespace(ScheduleError=ScheduleFailure),
    )


def fake_experiment(observer, tvm):
    experiment = ReplayExperiment.__new__(ReplayExperiment)
    experiment.tvm = tvm
    experiment.max_threads, experiment.warp_size = 1024, 32
    experiment.max_shared = 49152
    experiment.trace_check = Mock(return_value={"truth": 0})
    experiment.ir_check = Mock(return_value=[1, 0])
    schedule = SimpleNamespace(mod=object(), enter_postproc=Mock())
    experiment.new_schedule = Mock(return_value=schedule)
    experiment.postprocs = [observer]
    experiment.postproc_names = ["VerifyGPUCode"]
    return experiment


class ObserverTests(unittest.TestCase):
    def test_getter_immediately_follows_apply_on_same_thread(self):
        calls = []
        thread = get_ident()

        def apply(schedule):
            calls.append(("apply", get_ident()))
            return False

        def getter():
            calls.append(("getter", get_ident()))
            return "shared_memory"

        observer = VerifyObserver(SimpleNamespace(apply=apply), getter, fake_tvm())
        schedule = SimpleNamespace(mod=object())
        self.assertFalse(observer.apply(schedule))
        self.assertEqual(calls, [("apply", thread), ("getter", thread)])
        self.assertTrue(observer.called)
        self.assertEqual(observer.reason, "shared_memory")
        self.assertIs(observer.input_mod, schedule.mod)

    def test_reset_removes_previous_candidate_state(self):
        getter = Mock(return_value="shared_memory")
        observer = VerifyObserver(SimpleNamespace(apply=Mock(return_value=False)), getter, fake_tvm())
        observer.apply(SimpleNamespace(mod=object()))
        observer.reset()
        self.assertIsNone(observer.reason)
        self.assertFalse(observer.called)
        self.assertIsNone(observer.input_mod)
        getter.assert_called_once_with()

    def test_early_reject_does_not_read_stale_reason(self):
        tvm = fake_tvm()
        getter = Mock(return_value="shared_memory")
        observer = VerifyObserver(SimpleNamespace(apply=Mock(return_value=False)), getter, tvm)
        experiment = fake_experiment(observer, tvm)
        observer.apply(SimpleNamespace(mod=object()))
        observer.reset()
        getter.reset_mock()
        experiment.trace_check.return_value = {"truth": 1}
        trace = SimpleNamespace(apply_to_schedule=Mock())
        row, schedule = experiment.evaluate(trace, 1, "trace")
        self.assertEqual(row["status"], "hard_trace_reject")
        self.assertIsNone(schedule)
        self.assertFalse(observer.called)
        self.assertIsNone(observer.reason)
        getter.assert_not_called()

    def test_success_after_failure_records_none(self):
        postproc = SimpleNamespace(apply=Mock(side_effect=[False, True]))
        getter = Mock(side_effect=["thread_extent", "none"])
        observer = VerifyObserver(postproc, getter, fake_tvm())
        self.assertFalse(observer.apply(SimpleNamespace(mod=object())))
        observer.reset()
        self.assertTrue(observer.apply(SimpleNamespace(mod=object())))
        self.assertEqual(observer.reason, "none")

    def test_return_reason_mismatch_is_fatal(self):
        for passed, reason in ((True, "shared_memory"), (False, "none")):
            with self.subTest(passed=passed, reason=reason):
                observer = VerifyObserver(
                    SimpleNamespace(apply=Mock(return_value=passed)),
                    Mock(return_value=reason), fake_tvm(),
                )
                with self.assertRaises(AssertionError):
                    observer.apply(SimpleNamespace(mod=object()))

    def test_getter_error_does_not_become_candidate_failure(self):
        tvm = fake_tvm()
        observer = VerifyObserver(
            SimpleNamespace(apply=Mock(return_value=False)),
            Mock(side_effect=ScheduleFailure("observer API failure")), tvm,
        )
        experiment = fake_experiment(observer, tvm)
        trace = SimpleNamespace(apply_to_schedule=Mock())
        # A callback error must escape the driver, even when its class normally
        # denotes a candidate replay/postprocessing error.
        with self.assertRaises(Exception):
            experiment.evaluate(trace, 1, "baseline")

    def test_instrument_context_preserves_settings_and_getter_order(self):
        tvm = fake_tvm()
        calls = []
        captured = {}
        existing = object()
        instrument = object()
        context = SimpleNamespace(
            opt_level=3, required_pass=["Required"], disabled_pass=["Disabled"],
            config={"test.setting": 1}, instruments=[existing],
        )

        class Context:
            @staticmethod
            def current():
                return context

            def __init__(self, **kwargs):
                captured.update(kwargs)

            def __enter__(self):
                calls.append("enter")
                return self

            def __exit__(self, *args):
                calls.append("exit")

        def apply(schedule):
            calls.append("apply")
            return True

        def getter():
            calls.append("getter")
            return "none"

        tvm.transform = SimpleNamespace(PassContext=Context)
        observer = VerifyObserver(SimpleNamespace(apply=apply), getter, tvm)
        observer.instrument = instrument
        self.assertTrue(observer.apply(SimpleNamespace(mod=object())))
        self.assertEqual(calls, ["enter", "apply", "getter", "exit"])
        self.assertEqual(captured, {
            "opt_level": 3, "required_pass": ["Required"], "disabled_pass": ["Disabled"],
            "config": {"test.setting": 1}, "instruments": [existing, instrument],
        })


class PassJournalTests(unittest.TestCase):
    def setUp(self):
        self.journal = make_instrument(fake_tvm())

    def before(self, name):
        self.journal.run_before_pass(object(), SimpleNamespace(name=name))

    def after(self, name):
        self.journal.run_after_pass(object(), SimpleNamespace(name=name))

    def test_nested_passes_close_in_order(self):
        self.before("Sequential")
        self.before("Leaf")
        self.after("Leaf")
        self.after("Sequential")
        report = self.journal.report()
        self.assertEqual(report["unfinished_stack"], [])
        self.assertIsNone(report["deepest_unfinished_pass"])
        self.assertEqual(report["anomalies"], [])
        self.assertEqual([event["parent"] for event in report["events"]], [None, 0])
        self.assertTrue(all(event["completed"] for event in report["events"]))
        self.assertTrue(all(event["elapsed_inclusive_s"] >= 0 for event in report["events"]))

    def test_failed_parent_is_not_last_successful_child(self):
        self.before("Sequential")
        self.before("FailingParent")
        self.before("SuccessfulChild")
        self.after("SuccessfulChild")
        report = self.journal.report()
        self.assertEqual(report["unfinished_stack"], ["Sequential", "FailingParent"])
        self.assertEqual(report["deepest_unfinished_pass"], "FailingParent")
        self.assertTrue(report["events"][-1]["completed"])
        self.assertIn("elapsed_to_verify_return_s", report["events"][1])

    def test_same_named_nested_passes_keep_distinct_identity(self):
        self.before("Sequential")
        self.before("Sequential")
        self.after("Sequential")
        report = self.journal.report()
        self.assertEqual(report["unfinished_stack"], ["Sequential"])
        self.assertFalse(report["events"][0]["completed"])
        self.assertTrue(report["events"][1]["completed"])

    def test_unmatched_after_marks_observer_anomaly(self):
        self.before("Parent")
        self.after("Unexpected")
        report = self.journal.report()
        self.assertTrue(report["anomalies"])
        self.assertEqual(report["unfinished_stack"], ["Parent"])

    def test_callback_error_cannot_be_silently_swallowed_as_lowering_failure(self):
        class BadPassInfo:
            @property
            def name(self):
                raise RuntimeError("broken observer input")

        # Model Verify's C++ catch: an instrument exception becomes the same
        # lowering_exception result as a real pass failure. A sticky anomaly
        # must survive so main's validation refuses this attribution.
        try:
            self.journal.run_before_pass(object(), BadPassInfo())
        except Exception:
            pass
        self.assertTrue(self.journal.report()["anomalies"])


if __name__ == "__main__":
    unittest.main()
