# P3 ablation: LO-RANSAC / MSAC vs fixed & adaptive RANSAC vs GNC-GM

> **One-line result**: LO-RANSAC (MSAC + 5-round local optimization + adaptive stopping) reaches the same 100% success for random outliers at p<=70%, but it is 1.03-1.52x *slower* than the plain adaptive RANSAC that was already in the main table, and it does *not* beat GNC-GM in accuracy. The fixed-1000 RANSAC is a strawman; the adaptive variant is the fair baseline, and against it GNC wins only at high outlier rates (random outliers: p>=50%).

N=100, sigma=0.01, c=0.05, cap=1000, confidence=0.999, LO rounds=5; random: 100 trials/cell, coherent: 50 trials/cell. Timings: single-call wall clock on the Slurm node, BLAS/OpenMP pinned to 1 thread; p50/p95 are across trials, paired ratios are computed within the same trial.

## 1. Protocol and seeds

* Instances reuse the exact `run_main_extended.py` seed stream (`SeedSequence([20261127, 10, type_id, 100p, trial])`), so the RANSAC-1000 / RANSAC-adaptive / GNC-GM rows are a per-trial reproduction of the main extended table (checked in section 6).
* All three RANSAC variants share the same 3-subset stream (`SeedSequence(..., 999)`); this is a paired comparison at the hypothesis level.
* `RANSAC-1000` and `RANSAC-adaptive` call `common.solve_ransac` directly, unmodified.
* No hyper-parameter was tuned on these cells: c=0.05, confidence=0.999, cap=1000, LO rounds=5 were fixed before the run.

## 2. Exact LO-RANSAC algorithm

Minimal 3-point Kabsch hypotheses. For each hypothesis the score is MSAC (truncated quadratic), `s = sum_i min(||r_i||^2, c^2)`. A hypothesis is locally optimized **only when it improves the best score**: up to 5 rounds of `{inliers = ||r||<c; refit unweighted Kabsch on inliers; rescore}`, adopting a round only if the MSAC score strictly decreases and stopping when the inlier set is unchanged (fixed point) or the score stops improving. `_lo_refine` never worsens its input score by construction; if a model explains all N points the loop exits immediately (same early exit as `common.solve_ransac`). The adaptive confidence bound `need = ceil(log(1-0.999)/log(1-w^3))`, `w = |inliers|/N`, is updated on every new best model and the loop is capped at 1000 iterations. The final estimate is the usual unweighted refit on the best model's inliers (same final estimator as both baselines).

## 3. Random outliers: accuracy

### Success rate (rot < 5 deg, trans < 0.1 m)

| method | 0% | 10% | 20% | 30% | 40% | 50% | 60% | 70% |
|---|---|---|---|---|---|---|---|---|
| RANSAC-1000 | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| RANSAC-adaptive | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| LO-RANSAC | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| GNC-GM | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |

### Median rotation error (deg)

| method | 0% | 10% | 20% | 30% | 40% | 50% | 60% | 70% |
|---|---|---|---|---|---|---|---|---|
| RANSAC-1000 | 0.106 | 0.130 | 0.126 | 0.135 | 0.143 | 0.161 | 0.169 | 0.211 |
| RANSAC-adaptive | 0.107 | 0.133 | 0.128 | 0.136 | 0.143 | 0.162 | 0.169 | 0.211 |
| LO-RANSAC | 0.106 | 0.130 | 0.126 | 0.135 | 0.143 | 0.161 | 0.169 | 0.211 |
| GNC-GM | 0.106 | 0.132 | 0.124 | 0.140 | 0.138 | 0.161 | 0.167 | 0.211 |

### Median translation error (m)

| method | 0% | 10% | 20% | 30% | 40% | 50% | 60% | 70% |
|---|---|---|---|---|---|---|---|---|
| RANSAC-1000 | 0.0017 | 0.0015 | 0.0017 | 0.0017 | 0.0018 | 0.0022 | 0.0027 | 0.0028 |
| RANSAC-adaptive | 0.0017 | 0.0016 | 0.0017 | 0.0017 | 0.0018 | 0.0022 | 0.0027 | 0.0028 |
| LO-RANSAC | 0.0017 | 0.0015 | 0.0017 | 0.0017 | 0.0018 | 0.0022 | 0.0027 | 0.0027 |
| GNC-GM | 0.0017 | 0.0015 | 0.0018 | 0.0017 | 0.0018 | 0.0022 | 0.0027 | 0.0028 |

## 4. Random outliers: runtime and iterations

### Runtime p50 (ms)

| method | 0% | 10% | 20% | 30% | 40% | 50% | 60% | 70% |
|---|---|---|---|---|---|---|---|---|
| RANSAC-1000 | 0.19 | 56.45 | 55.54 | 55.72 | 56.65 | 59.77 | 58.35 | 58.02 |
| RANSAC-adaptive | 0.19 | 0.45 | 0.68 | 1.08 | 1.78 | 3.10 | 6.15 | 14.76 |
| LO-RANSAC | 0.28 | 0.62 | 0.83 | 1.23 | 1.94 | 3.34 | 6.45 | 15.18 |
| GNC-GM | 0.44 | 1.73 | 1.72 | 1.73 | 1.76 | 1.81 | 1.84 | 1.83 |

### Runtime p95 (ms)

| method | 0% | 10% | 20% | 30% | 40% | 50% | 60% | 70% |
|---|---|---|---|---|---|---|---|---|
| RANSAC-1000 | 0.42 | 58.56 | 62.05 | 61.85 | 90.12 | 88.97 | 70.17 | 88.56 |
| RANSAC-adaptive | 0.25 | 0.53 | 0.82 | 1.21 | 2.88 | 4.90 | 8.60 | 22.55 |
| LO-RANSAC | 0.31 | 0.73 | 1.04 | 1.32 | 3.05 | 5.14 | 8.95 | 23.23 |
| GNC-GM | 0.51 | 1.80 | 2.17 | 1.92 | 2.92 | 2.96 | 2.62 | 2.90 |

### Mean RANSAC iterations (GNC-GM: —)

| method | 0% | 10% | 20% | 30% | 40% | 50% | 60% | 70% |
|---|---|---|---|---|---|---|---|---|
| RANSAC-1000 | 2 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 |
| RANSAC-adaptive | 2 | 6 | 10 | 17 | 29 | 52 | 105 | 253 |
| LO-RANSAC | 1 | 6 | 10 | 17 | 29 | 52 | 105 | 253 |
| GNC-GM | — | — | — | — | — | — | — | — |

### Runtime ratio to GNC-GM (p50 RANSAC / p50 GNC, >1 = RANSAC slower)

| outlier rate | RANSAC-1000 / GNC | RANSAC-adaptive / GNC | LO-RANSAC / GNC | GNC p50 (ms) |
|---|---|---|---|---|
| 0% | 0.4x | 0.4x | 0.6x | 0.44 |
| 10% | 32.6x | 0.3x | 0.4x | 1.73 |
| 20% | 32.3x | 0.4x | 0.5x | 1.72 |
| 30% | 32.2x | 0.6x | 0.7x | 1.73 |
| 40% | 32.2x | 1.0x | 1.1x | 1.76 |
| 50% | 33.0x | 1.7x | 1.8x | 1.81 |
| 60% | 31.8x | 3.3x | 3.5x | 1.84 |
| 70% | 31.7x | 8.1x | 8.3x | 1.83 |

### Paired runtime ratio vs GNC-GM (median over the same trials)

| outlier rate | RANSAC-1000 / GNC | RANSAC-adaptive / GNC | LO-RANSAC / GNC |
|---|---|---|---|
| 0% | 0.4x | 0.4x | 0.6x |
| 10% | 32.6x | 0.3x | 0.4x |
| 20% | 32.5x | 0.4x | 0.5x |
| 30% | 32.2x | 0.6x | 0.7x |
| 40% | 32.2x | 1.0x | 1.1x |
| 50% | 31.7x | 1.7x | 1.8x |
| 60% | 31.7x | 3.4x | 3.5x |
| 70% | 31.6x | 8.0x | 8.2x |

> The paired column is the load-drift-robust version: both methods are timed back-to-back on the same instance, so it is preferred over the standalone p50 ratios when the shared node is noisy.

### LO overhead (random outliers): mean LO calls and adopted refits

| outlier rate | mean LO calls | mean adopted LO rounds | LO p50 (ms) | adaptive p50 (ms) | LO / adaptive |
|---|---|---|---|---|---|
| 0% | 1.0 | 1.6 | 0.28 | 0.19 | 1.52x |
| 10% | 1.3 | 1.7 | 0.62 | 0.45 | 1.37x |
| 20% | 1.5 | 1.6 | 0.83 | 0.68 | 1.22x |
| 30% | 1.6 | 1.5 | 1.23 | 1.08 | 1.14x |
| 40% | 1.9 | 1.7 | 1.94 | 1.78 | 1.09x |
| 50% | 2.0 | 1.5 | 3.34 | 3.10 | 1.08x |
| 60% | 2.2 | 1.5 | 6.45 | 6.15 | 1.05x |
| 70% | 2.6 | 1.4 | 15.18 | 14.76 | 1.03x |

## 5. Coherent outliers (one competing rigid transform)

### Success rate

| method | 0% | 10% | 20% | 30% | 40% | 50% | 60% | 70% |
|---|---|---|---|---|---|---|---|---|
| RANSAC-1000 | 100% | 100% | 100% | 100% | 100% | 56% | 0% | 0% |
| RANSAC-adaptive | 100% | 100% | 100% | 100% | 100% | 56% | 0% | 0% |
| LO-RANSAC | 100% | 100% | 100% | 100% | 100% | 48% | 0% | 0% |
| GNC-GM | 100% | 100% | 100% | 100% | 100% | 52% | 0% | 0% |

### Median rotation error (deg)

| method | 0% | 10% | 20% | 30% | 40% | 50% | 60% | 70% |
|---|---|---|---|---|---|---|---|---|
| RANSAC-1000 | 0.09 | 0.11 | 0.12 | 0.13 | 0.15 | 0.29 | 41.55 | 43.79 |
| RANSAC-adaptive | 0.10 | 0.11 | 0.12 | 0.13 | 0.15 | 0.29 | 41.55 | 43.81 |
| LO-RANSAC | 0.09 | 0.11 | 0.12 | 0.13 | 0.15 | 30.79 | 41.55 | 43.79 |
| GNC-GM | 0.10 | 0.12 | 0.14 | 0.13 | 0.15 | 0.30 | 41.58 | 43.79 |

### Median translation error (m)

| method | 0% | 10% | 20% | 30% | 40% | 50% | 60% | 70% |
|---|---|---|---|---|---|---|---|---|
| RANSAC-1000 | 0.0016 | 0.0018 | 0.0016 | 0.0018 | 0.0023 | 0.0035 | 1.1326 | 0.9820 |
| RANSAC-adaptive | 0.0016 | 0.0018 | 0.0016 | 0.0018 | 0.0023 | 0.0035 | 1.1326 | 0.9820 |
| LO-RANSAC | 0.0016 | 0.0018 | 0.0016 | 0.0018 | 0.0023 | 0.7253 | 1.1326 | 0.9820 |
| GNC-GM | 0.0016 | 0.0018 | 0.0015 | 0.0017 | 0.0022 | 0.0039 | 1.1326 | 0.9821 |

### Fraction of trials whose estimate is the *outlier* pose (identifiability failure)

| method | 0% | 10% | 20% | 30% | 40% | 50% | 60% | 70% |
|---|---|---|---|---|---|---|---|---|
| RANSAC-1000 | — | 0% | 0% | 0% | 0% | 44% | 100% | 100% |
| RANSAC-adaptive | — | 0% | 0% | 0% | 0% | 44% | 100% | 100% |
| LO-RANSAC | — | 0% | 0% | 0% | 0% | 52% | 100% | 100% |
| GNC-GM | — | 0% | 0% | 0% | 0% | 48% | 100% | 100% |

### Runtime p50 (ms)

| method | 0% | 10% | 20% | 30% | 40% | 50% | 60% | 70% |
|---|---|---|---|---|---|---|---|---|
| RANSAC-1000 | 0.19 | 56.28 | 56.50 | 55.54 | 56.27 | 58.52 | 55.13 | 56.28 |
| RANSAC-adaptive | 0.19 | 0.46 | 0.70 | 1.11 | 1.77 | 3.12 | 1.76 | 1.08 |
| LO-RANSAC | 0.29 | 0.62 | 0.86 | 1.25 | 1.96 | 3.26 | 1.92 | 1.24 |
| GNC-GM | 0.45 | 1.57 | 1.54 | 1.47 | 1.48 | 1.44 | 1.42 | 1.45 |

### Mean RANSAC iterations (GNC-GM: —)

| method | 0% | 10% | 20% | 30% | 40% | 50% | 60% | 70% |
|---|---|---|---|---|---|---|---|---|
| RANSAC-1000 | 2 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 |
| RANSAC-adaptive | 2 | 6 | 10 | 17 | 29 | 52 | 29 | 17 |
| LO-RANSAC | 1 | 6 | 10 | 17 | 29 | 52 | 29 | 17 |
| GNC-GM | — | — | — | — | — | — | — | — |

> At p=50% the true and the competing transform have the same expected consensus size, so the winner is a coin flip. LO-RANSAC's MSAC tie-break picked the outlier basin in 52% of trials vs 44% for both plain RANSAC variants and 48% for GNC; that is a 2-4 trial difference out of 50 (binomial s.e. ~7%), not a systematic regression. The large LO median rotation error at p=50% is the same tie effect, not a 30-deg failure mode. At 60-70% every method returns the competing transform: local optimization cannot repair an identifiability failure (Prop. 3.7).

## 6. Reproduction / validation check

* Compared 3600 per-trial records against `synthetic/results/main_table_extended_raw.jsonl`: max |drot| = 0.00e+00 deg, max |dtrans| = 0.00e+00 m, success mismatches = 0, iteration mismatches = 0, missing = 0.
* **PASS** — the shared methods are bit-identical to the published main-table records.

## 7. Honest reading

* **Does LO-RANSAC beat GNC-GM in accuracy?** No. Success is 100% for both over 0-70% random outliers; the median rotation error differs by at most 0.006 deg (translation 0.0001 m) on random outliers and at most 0.019 deg on coherent p<=40%. LO cannot beat the final consensus refit because all methods already use the same final refit.
* **Does LO-RANSAC beat GNC-GM in runtime?** Only where the cheaper adaptive RANSAC already does, and by a smaller margin because LO adds overhead. Paired per-trial ratios: LO/GNC = 0.4x at p=10%, 1.1x at p=40%, 8.2x at p=70% (adaptive: 0.3x, 8.0x). GNC's cost is flat in p, so RANSAC overtakes it only at low p (high inlier ratio); even at p=0 GNC is ~2x slower than RANSAC because of the annealing overhead.
* **LO vs adaptive RANSAC**: LO-RANSAC is slower at every p tested (1.03-1.52x of the adaptive RANSAC p50) because the LO refits are pure overhead once the consensus set is found; LO does not improve success, accuracy, or the adaptive stopping target.
* **Fixed-1000 baseline**: GNC is up to ~33x faster than the fixed 1000-iteration RANSAC (p50 ratio), reproducing the main table. That comparison is a strawman; the paper should lead with the adaptive-RANSAC numbers instead.
* **Budget**: the 1000-iteration cap does not bind. LO-RANSAC's mean iteration count peaks at 253 (random p=70%) and its largest single run is 264 iterations; at p=0 the all-inlier early exit stops after 1 iteration. A larger budget would not change any success/accuracy number in the requested range (p<=70%).
* **Coherent outliers**: at 60-70% all methods return the competing transform (or a mixture), including LO-RANSAC; local optimization cannot fix an identifiability failure. At p=50% all methods are near a coin flip (see the note in section 5).
* **No test-time tuning**: LO rounds, confidence and cap were fixed before the run; no cell-specific parameter was selected.

## 8. Files

* `run_ransac_lo.py` — this experiment (imports `../synthetic/common.py`, which is not modified).
* `results/ransac_lo.json` — config + aggregates + validation check.
* `results/ransac_lo_raw.jsonl` — per-trial records.
* `results/ransac_lo.png` — figure.
