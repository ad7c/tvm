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
"""Opt-in P3 experiment instrumentation. Does not implement search decisions."""

import json
import logging
import math
from pathlib import Path
import re
import subprocess
import time
import traceback


CORRECTNESS_CONFIG = {
    "reference": "numpy float64 matmul", "seed": 20261008,
    "rtol": 2e-4, "atol": 2e-4,
    "cases": ["normal", "uniform", "identity", "zero"],
    "output_initialization": "NaN", "timing_inputs": "original LocalRunner allocation, unchanged",
}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_json(path, value):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + "\n")


def gpu_status():
    result = subprocess.run([
        "nvidia-smi", "--query-gpu=index,name,driver_version,memory.used,utilization.gpu,temperature.gpu",
        "--format=csv",
    ], capture_output=True, text=True, check=False)
    return {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


def classify_measurements(batch):
    """Builder failure is mirrored in RunnerResult; count mutually exclusive outcomes."""
    outcomes = []
    fields = (batch["builder_errors"], batch["runner_errors"], batch["runner_run_secs"])
    if len({len(field) for field in fields}) != 1:
        raise ValueError("Measurement slot lengths disagree")
    for build_error, run_error, values in zip(*fields):
        finite = values and all(math.isfinite(x) and 0 < x < 1e9 for x in values)
        if build_error is not None:
            outcomes.append("build_error")
        elif run_error is not None:
            outcomes.append("runner_error")
        elif finite:
            outcomes.append("finite")
        else:
            outcomes.append("invalid_latency")
    return outcomes


class TuneJournal:
    """Observe original logger boundaries despite TVM rebuilding its handlers."""

    def __init__(self, path, record_model_training=False):
        self.path = Path(path)
        self.record_model_training = record_model_training
        self.events = []
        self.batch_index = -1
        self.origin_ns = None

    def emit(self, event, **payload):
        row = {"event": event, "batch_index": self.batch_index,
               "tune_elapsed_s": (time.perf_counter_ns() - self.origin_ns) / 1e9, **payload}
        append_json(self.path, row)
        self.events.append(row)
        return row

    def __enter__(self):
        self.origin_ns = time.perf_counter_ns()
        self.old_factory = logging.getLogRecordFactory()

        def factory(*args, **kwargs):
            record = self.old_factory(*args, **kwargs)
            if record.name.startswith("tvm.s_tir.meta_schedule"):
                message = record.getMessage()
                if self.record_model_training and message.startswith("XGB "):
                    self.emit("xgb_log", message=message)
                if "TaskScheduler picks Task #" in message:
                    self.batch_index += 1
                    self.emit("search_begin", message=message)
                elif match := re.search(r"Sending (\d+) sample\(s\) to builder", message):
                    self.emit("builder_begin", candidates=int(match.group(1)))
                elif match := re.search(r"Sending (\d+) valid sample\(s\) to runner", message):
                    self.emit("runner_begin", candidates=int(match.group(1)))
                elif "[Updated] Task #" in message:
                    self.emit("scheduler_updated")
            return record

        logging.setLogRecordFactory(factory)
        self.emit("tune_begin")
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            self.emit("tune_end", exception=None if exc is None else repr(exc))
        finally:
            logging.setLogRecordFactory(self.old_factory)


def summarize_timeline(events):
    """Single-task sequential phases; nested C++ counters must not be added here."""
    batches = []
    for index in sorted({r["batch_index"] for r in events if r["event"] == "builder_begin"}):
        rows = [r for r in events if r["batch_index"] == index]
        lookup = {r["event"]: r for r in rows if not r["event"].startswith("callback_")}
        callbacks = [r for r in rows if r["event"] == "callback_begin"]
        if not callbacks or "batch_complete" not in lookup:
            raise ValueError(f"Incomplete measured batch timeline {index}")
        start, build, run = (lookup[key]["tune_elapsed_s"]
                             for key in ("search_begin", "builder_begin", "runner_begin"))
        callback_start = callbacks[0]["tune_elapsed_s"]
        end = lookup["batch_complete"]["tune_elapsed_s"]
        boundaries = [start, build, run, callback_start, end]
        if boundaries != sorted(boundaries):
            raise ValueError("Nonmonotonic phase boundaries")
        callback_times = {}
        for begin in callbacks:
            finishes = [r for r in rows if r["event"] == "callback_end"
                        and r["callback_index"] == begin["callback_index"]]
            if len(finishes) != 1:
                raise ValueError("Unmatched callback boundary")
            callback_times[begin["name"]] = finishes[0]["tune_elapsed_s"] - begin["tune_elapsed_s"]
        batches.append({"batch_index": index, "submitted": lookup["builder_begin"]["candidates"],
                        "sent_to_runner": lookup["runner_begin"]["candidates"],
                        "search_s": build - start, "builder_s": run - build,
                        "runner_and_notify_s": callback_start - run,
                        "callbacks_s": end - callback_start, "callback_s": callback_times,
                        "available_at_tune_s": end})
    total = next(r["tune_elapsed_s"] for r in events if r["event"] == "tune_end")
    accounted = sum(sum(row[key] for key in
                        ("search_s", "builder_s", "runner_and_notify_s", "callbacks_s"))
                    for row in batches)
    if accounted > total + 1e-6:
        raise ValueError("Overlapping phases exceed tune wall clock")
    return {"batches": batches, "tune_s": total, "residual_s": total - accounted,
            "note": "Logger/callback observation boundaries; runner includes correctness and notify; "
                    "callback_s is nested inside callbacks_s, not additive."}


def checked_matmul_evaluator(rt_mod, device, config, repeated_args, *, audit_path, candidate_id):
    """Worker callable: validate separate buffers, then time untouched runner inputs."""
    import numpy as np
    import tvm
    from tvm.s_tir.meta_schedule.runner.local_runner import default_run_evaluator

    started = time.perf_counter()
    append_json(audit_path, {"event": "correctness_begin", "candidate_id": candidate_id})
    row = {"candidate_id": candidate_id, "event": "correctness_result", "passed": False, "cases": []}
    try:
        inputs = repeated_args[0]
        if len(inputs) != 3 or any(str(x.dtype) != "float32" for x in inputs):
            raise ValueError("P3 correctness evaluator only supports three FP32 MatMul buffers")
        m, k = (int(x) for x in inputs[0].shape)
        k2, n = (int(x) for x in inputs[1].shape)
        if k != k2 or tuple(inputs[2].shape) != (m, n) or m != k or k != n:
            raise ValueError("P3 checker currently requires square MatMul")
        rng = np.random.default_rng(CORRECTNESS_CONFIG["seed"])
        pairs = [
            ("normal", rng.normal(size=(m, k)), rng.normal(size=(k, n))),
            ("uniform", rng.uniform(-1, 1, (m, k)), rng.uniform(-1, 1, (k, n))),
            ("identity", np.eye(m), rng.normal(size=(k, n))),
            ("zero", np.zeros((m, k)), rng.normal(size=(k, n))),
        ]
        for name, lhs, rhs in pairs:
            a, b = lhs.astype("float32"), rhs.astype("float32")
            expected = a.astype("float64") @ b.astype("float64")
            a_dev, b_dev = tvm.runtime.tensor(a, device), tvm.runtime.tensor(b, device)
            out = tvm.runtime.tensor(np.full((m, n), np.nan, dtype="float32"), device)
            rt_mod[rt_mod.entry_name](a_dev, b_dev, out)
            device.sync()
            actual = out.numpy()
            error = np.abs(actual.astype("float64") - expected)
            tolerance = CORRECTNESS_CONFIG["atol"] + CORRECTNESS_CONFIG["rtol"] * np.abs(expected)
            passed = bool(np.isfinite(actual).all() and (error <= tolerance).all())
            row["cases"].append({"name": name, "passed": passed,
                                  "max_abs_error": float(error.max()),
                                  "max_tolerance_ratio": float((error / tolerance).max())})
            if not passed:
                raise AssertionError(f"P3 numerical correctness failed: {name}")
        row["passed"] = True
    except Exception:
        row["exception"] = traceback.format_exc()
        raise
    finally:
        row["correctness_wall_s"] = time.perf_counter() - started
        append_json(audit_path, row)
    costs = default_run_evaluator(rt_mod, device, config, repeated_args)
    append_json(audit_path, {"event": "evaluator_complete", "candidate_id": candidate_id,
                             "costs_s": [float(x) for x in costs]})
    return costs
