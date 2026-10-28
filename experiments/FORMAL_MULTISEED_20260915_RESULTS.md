# Formal multi-seed repetition (MatMul128)

Date: 2026-09-15.  Workload: MatMul128, CUDA `sm_86`, 16 measured trials,
population 128, 8 trials per batch, 4 GA iterations, two search batches.  The
three seeds are `20260914`, `20260915`, and `20260916`.

Arms:

* `B`: all CS/H/Draft-GA/G1/G2 switches off.
* `CSH`: constraint-aware sampling plus trace-position hard filtering.
* `D_G2`: Draft-guided GA plus G2 projection; no CS/H.
* `FULL`: CSH plus Draft-guided GA and directed G1/G2.

## Best measured latency

| Seed | B (us) | CSH (us) | D_G2 (us) | FULL (us) |
|---:|---:|---:|---:|---:|
| 20260914 | 41.1956 | 41.3622 | **36.0375** | 42.7146 |
| 20260915 | 37.6701 | 36.5285 | **35.6347** | 37.8306 |
| 20260916 | 38.1413 | 38.9720 | 39.3647 | 44.2375* |

`*` The seed-20260916 FULL run produced only 14/16 finite measurements: two
selected candidates hit the 30-second LocalBuilder timeout.  Its latency is
therefore provisional and is not a clean endpoint comparison.

For the three complete B/CSH/D_G2 runs, mean +/- sample standard deviation is:

| Arm | Mean (us) | Std (us) | Mean ratio vs B |
|---|---:|---:|---:|
| B | 39.0023 | 1.9140 | 1.000 |
| CSH | 38.9543 | 2.4169 | 0.9988 (-0.12%) |
| D_G2 | 37.0123 | 2.0472 | 0.9490 (-5.10%) |

Per-seed ratios versus B are CSH: `+0.40%, -3.03%, +2.18%`, and D_G2:
`-12.52%, -5.40%, +3.21%`.  Thus D_G2 is promising but not yet stable across
seeds; CSH is essentially quality-neutral at this small budget.

Including the provisional FULL point gives `41.5942 +/- 3.3472 us` and `+6.7%`
versus the B mean.  The two complete FULL points average `40.2726 us`; this
does not establish a benefit, and the incomplete point must be rerun before a
formal FULL claim.

## Search and verification effects

| Arm | Generated candidates (mean) | VerifyGPUCode failures (mean) | Search/Build/Runner mean (s) |
|---|---:|---:|---:|
| B | 3208 | 2079 | 148.0 / 147.3 / 58.0 |
| CSH | 1516 | 278 | 152.0 / 147.0 / 53.0 |
| D_G2 | 3224 | 2103 | 151.3 / 138.7 / 53.7 |
| FULL | 1469 | 269 | 149.7 / 190.0 / 52.3* |

CSH reduces generated candidates by about 53% and VerifyGPUCode failures by
about 86.6% relative to B.  The reduction comes mainly from eliminating
thread-extent-invalid initial decisions before full post-processing.  The
shared-memory proxy remains conservative: it catches real shared-memory
failures but also reports many false positives, so it is not treated as an
exact hard proof.

D_G2 has no CS/H filtering, so its Verify failure count remains close to B.
Its G2 proposal statistics are 534/550/578 total proposals for the three
seeds, with 0/3/1 successful projections and no projection failures.

FULL has 20/34/34 trace hard rejections, 498/529/520 directed choices, and
1/0/4 G2 projections for seeds 20260914/15/16.  It reduces Verify failures by
about 87% but does not translate that saving into a better measured kernel;
directed G1 currently narrows the search too aggressively for this budget.

`*` FULL's seed-20260916 build mean is inflated by the two timeout batches;
that run has 14/16 finite records and should not be pooled as a clean timing
measurement.

## XGBoost architecture check

All completed result JSONs pass the architecture contract.  In the
Draft-guided arms, GA makes zero XGBoost calls (`ga_xgb_calls=0`); each search
batch sends only its Draft archive to the final model (`final_xgb_calls=2`,
`final_xgb_input=64`).  Baseline and CSH instead use 10 GA XGBoost calls with
1280 GA inputs.  This confirms the intended Draft-first ordering.

## Reproducibility and caveats

Raw results are under
`experiments/results/formal_multiseed_matmul128_20260915/`.  The older seed
20260914 results are under `experiments/results/formal_mainline_20260914/`
and `experiments/results/final_v1_20260914_r2/`.

The 20260916 FULL rerun exposed a standalone-driver logging initialization
issue.  `experiments/e1_thread_filter.py` now creates the `logs` directory and
the two logger files before calling `tune_tir`; this does not alter CS, H,
Draft, GA, or measurement logic.  A clean rerun with the same configuration
and a longer builder timeout is still required before using seed-20260916
FULL in a final average.
