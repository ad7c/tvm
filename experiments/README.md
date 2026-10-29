# Bidirectional Pruning Experiments

**2026-10-29 Scout branch:** [安装、迁移和端到端运行说明](README_SCOUT_1029.md)。
新版入口是 `python experiments/scout_cli.py`；下方保留早期实验说明。

Research Plan §3.4: GPU kernel search space reduction via bidirectional pruning.

## Structure

```
experiments/
├── common.py              # Shared utilities (GPU spec, MatMul module, pruning model)
├── data/                  # Experiment results (JSON)
├── figures/               # Charts (PNG)
│
├── exp01_baseline/        # Exp 1: Full-space baseline pass rate
│   └── run.py             #   Runs tune_tir, captures postproc rejection stats
│
└── exp02_pruning/         # Exp 2: Bidirectional pruning effect
    ├── model.py           #   Analytical model: divisor space enumeration
    ├── verify.py          #   TVM verification: trace JSON modification + postproc
    └── compare.py         #   Comparison chart: full vs single vs double pruning
```

## How to Run

```bash
# 1. Baseline pass rate (evolutionary search, ~3 min)
python experiments/exp01_baseline/run.py

# 2a. Analytical model (instant)
python experiments/exp02_pruning/model.py

# 2b. TVM verification (~2 min)
python experiments/exp02_pruning/verify.py

# 2c. Comparison chart
python experiments/exp02_pruning/compare.py
```

## Key Questions

| Experiment | Question | Method |
|------------|----------|--------|
| exp01 | What is the baseline effective rate? | tune_tir evolution → parse C++ postproc logs |
| exp02 | Does pruning improve effective rate? | Analytical model + trace JSON verification → compare 3 groups |
