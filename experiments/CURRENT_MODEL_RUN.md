# Current ResNet-18 experiment

Updated: 2026-10-28 (Asia/Shanghai).

Active branch: scout/resnet18-pool64-128-20261028.
Snapshot before changes: 8723123f487c38fd83d49129ee6aa12682c200c3, pushed to fork/pruning-experiment.

- [Live status](results/resnet18_pool_20261028_A01/STATUS.md)
- [Machine-readable progress](results/resnet18_pool_20261028_A01/progress.json)
- [Frozen protocol](results/resnet18_pool_20261028_A01/protocol.json)
- [Experiment design](SCOUT_POOL_STUDY_20261028.md)

The controller starts with two separate D64/D128 preflights, then runs four formal arms:
matched MetaSchedule, constraints only, constraints + Draft64, constraints + Draft128.
All arms use population 128, GA 2, and up to 10 GPU measurements per batch.
Requested budget: 400 submissions per arm, 90 minutes per arm, eight hours total.
The live status is authoritative: waiting_for_memory is not GPU tuning.

Historical operator results and paused e2e_20261025_A03 are retained unchanged.
The manuscript is not part of this experiment change.
