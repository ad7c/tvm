# Formal experiment results — 2026-09-14

All runs use the compiled `build-prune/lib/tvm_compiler.dll`, CUDA `sm_86`, and writable `F:/tvm/tmp/formal_20260914`. They are real TVM build + CUDA runner runs.

## Mainline six-arm (MatMul128, seed 20260914, 16 trials)

| Arm | Finite | Best us | Search/Build/Runner s | Verify failures |
|---|---:|---:|---:|---:|
| B | 16/16 | 41.1956 | 153/140/53 | 2122 |
| CSH (CS+H-trace) | 16/16 | 41.3622 | 133/153/52 | 271 |
| D (Draft-GA) | 16/16 | 43.5091 | 140/133/52 | 2131 |
| D+G2 | 16/16 | **36.0375** | 138/130/53 | 2113 |
| D+G1v2/G2 | 16/16 | 38.0495 | 131/133/52 | 2005 |
| CSH+D+G1v2/G2 | 16/16 | 42.7146 | 139/142/52 | 266 |

All six arms passed architecture validation and had zero build/runner errors. Draft-GA arms made zero XGBoost calls inside GA; only the end-of-batch top-k was sent to XGBoost. D+G2 was 12.5% faster than B in this seed. G1v2 was 5.6% slower than D+G2 in this budget.

## Multi-seed and multi-operator smoke (8 trials, population 64)

| Workload/seed | Arm | Best us | Search/Build/Runner s |
|---|---|---:|---:|
| MatMul128/20260915 | B | 46.1407 | 39/69/27 |
| MatMul128/20260915 | Full | 44.0913 | 60/70/32 |
| MatMul256/20260914 | B | 73.3693 | 51/71/31 |
| MatMul256/20260914 | Full | 51.0467 | 31/68/27 |
| Reduction128/20260914 | B | 40.8150 | 0/67/27 |
| Reduction128/20260914 | Full | 42.0598 (tuning completed) | 0/~67/27 |

Reduction128 full did not emit `result.json`: the driver currently treats `predicted=0,enforced=0` as an enforce failure. This is an applicability boundary (no hard rejection exists), not a compiler or runner failure.

## Stage timing (mainline representative values)

| Arm | Trace replay s | VerifyGPUCode s | Draft s | GA XGB input | Final XGB input |
|---|---:|---:|---:|---:|---:|
| B | 14.047 | 131.655 | 0 | 1280 | 0 |
| CSH | 6.438 | 121.505 | 0 | 1280 | 0 |
| D | 14.150 | 121.212 | 0.216 | 0 | 64 |
| D+G2 | 14.141 | 118.418 | 0.215 | 0 | 64 |
| D+G1v2/G2 | 13.595 | 112.348 | 0.217 | 0 | 64 |
| Full | 6.669 | 129.705 | 0.214 | 0 | 64 |

`timing` and per-stage counters are retained in each arm's `result.json` under `experiments/results/formal_mainline_20260914/` and `experiments/results/multiseed_multiop_20260914/`. The experiment progress log contains a short pointer to this detailed record.

## Additional operators (Conv2D / Elementwise / Fusion)

The driver now accepts `--workload conv2d`, `--workload elementwise128`, and
`--workload fusion128` in addition to the original workloads.  These smoke
runs use six to eight trials because the elementwise search spaces contain
only six legal traces.

| Workload | Arm | Finite | Best us | Search/Build/Runner s | Status |
|---|---|---:|---:|---:|---|
| Conv2D (1x3x224x224, 64x7x7) | B | 8/8 | 207.1982 | 72/91/28 | passed |
| Conv2D | Full | 8/8 | 108.4267 | 107/85/27 | passed; about 47.7% lower latency |
| Elementwise Add 128x128 | B | 6/6 | 35.9383 | 0/76/25 | passed |
| Elementwise Add 128x128 | Full | 6/6 | 38.5428 | 1/63/22 | passed; no hard rejection exists |
| Fused Add+ReLU 128x128 | B | 6/6 | 33.7105 | 0/61/20 | passed |
| Fused Add+ReLU 128x128 | Full | 6/6 | 34.7059 | 0/69/25 | passed; no hard rejection exists |

For tiny spaces, `verify_top_k` is configured at 16 but the effective top-k
is `min(verify_top_k, draft_archive)`, so the final XGBoost call correctly
uses six candidates.  The callback validator also ignores trailing empty
batches after the legal search space is exhausted.  Zero predicted hard
rejections are treated as a valid (vacuous) enforce result.

## Formal multi-seed repetition (2026-09-15)

The three-seed MatMul128 repetition and its caveats are recorded in
[`FORMAL_MULTISEED_20260915_RESULTS.md`](FORMAL_MULTISEED_20260915_RESULTS.md).
In brief, CSH reduced generated candidates and VerifyGPUCode failures by about
53% and 86.6%, respectively.  D+G2 averaged 5.1% lower latency than B over
the three complete seeds but was not stable (`-12.5%`, `-5.4%`, `+3.2%`).
The FULL arm still regressed; seed 20260916 had two 30-second LocalBuilder
timeouts and is marked provisional rather than pooled as a clean endpoint.
