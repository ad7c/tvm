<!--
Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements. See the NOTICE file
distributed with this work for additional information
regarding copyright ownership. The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License. You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied. See the License for the
specific language governing permissions and limitations
under the License.
-->

# ResNet-18 candidate-pool experiment

Compare matched-parameter MetaSchedule (B), constraints alone (C), and
constraints plus the unchanged Draft formula with final verification pools
of 64 and 128 (D64/D128). Each full batch submits up to 10 schedules to the
GPU: nine model-ranked choices plus one exploration choice, identically
configured across arms. Deduplication or a finite candidate space can reduce
the actual count; preserve and report it.

Common settings: FP32, batch 1, input 1x3x224x224, frozen random weights,
population 128, two GA iterations after initial scoring, seed 20261011,
XGBoost warmup 100, adaptive training, one builder, one search thread.
Warmup predictions are random scores and must not be reported as trained
XGBoost rankings. After warmup, D64/D128 use XGBoost only for final verification;
B/C use it during evolution. Thus GA settings match while XGBoost input counts
intentionally differ. B is a matched-parameter baseline, not stock defaults.

Each formal arm requests 400 submissions with a 90-minute wall limit. The
entire experiment, including preflight and evaluation, is capped at eight
hours. A soft deadline stops requesting new batches before the hard limit.
Checkpoint evaluation consumes wall time and is also reported separately.
Do not infer equal measurement budgets from equal requested budgets.

Two 10-submission ResNet-18 preflights verify D64/D128 architecture and numerical
correctness before the formal arms. They have separate histories and are
excluded from formal comparisons. The first 100 submissions are warmup, so
these preflights validate wiring rather than learned-model quality.

Evaluate the complete GPU Relax VM forward every 100 submissions and at the
end, checking outputs against the frozen reference. Uncovered functions use
the same cached fallback schedules; record their names. Final models are
remeasured in rotating order in one process. Report full-model latency and
time to the baseline's final latency plus 5%, using observed checkpoints
only. One seed is exploratory evidence, not a multi-seed conclusion.

Retain protocol and source hashes, raw scheduling databases, task inventory,
compressed candidate traces, every model prediction and training event,
five raw GPU timing samples per measured schedule, builder/runner errors,
empty candidate events, native search logs, resource telemetry, full-model
checkpoints, final schedules, and independent final timing samples. Candidate
trace IDs include task names. The extra audit work is timed and remains part
of wall time. No previous history is overwritten or resumed.

Run from the repository using the existing tuning Python environment:

```powershell
.\.venv-tune\python.exe -u -X utf8 experiments/model_scout_pool_study.py run --study experiments/results/resnet18_pool_20261028_A01
```

The controller waits for at least 1.5 GiB available physical memory and 6 GiB
available commit, for at most five hours before starting its eight-hour
budget. A resource guard stops only this experiment's process tree on
sustained memory pressure. Inspect `STATUS.md`, `progress.json`, and
`RESULTS.md` in the study directory; a partial or failed run is not a completed
comparison. Leave the current manuscript unchanged.
