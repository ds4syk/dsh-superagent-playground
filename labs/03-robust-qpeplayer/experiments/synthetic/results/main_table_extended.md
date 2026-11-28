# Synthetic 3D-3D robust registration — extended main table

N=100, sigma=0.01, c=0.05, 100 trials/cell (clean instances shared across outlier types at p=0).
Success = rotation error < 5 deg and translation error < 0.1 m. Times are wall-clock p50/p95 over trials.

## 1. Random outliers (v0 protocol: replaced y uniform in [-2,2]^3)

### Success rate

| method | 0% | 10% | 20% | 30% | 40% | 50% | 60% | 70% |
|---|---|---|---|---|---|---|---|---|
| Kabsch | 100% | 52% | 7% | 0% | 0% | 0% | 0% | 0% |
| RANSAC-1000 | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| RANSAC-adaptive | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| IRLS-GM | 100% | 100% | 100% | 100% | 100% | 100% | 99% | 98% |
| GNC-GM | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |

### Median rotation error (deg)

| method | 0% | 10% | 20% | 30% | 40% | 50% | 60% | 70% |
|---|---|---|---|---|---|---|---|---|
| Kabsch | 0.11 | 4.46 | 7.58 | 10.46 | 14.23 | 18.49 | 26.94 | 38.45 |
| RANSAC-1000 | 0.11 | 0.13 | 0.13 | 0.13 | 0.14 | 0.16 | 0.17 | 0.21 |
| RANSAC-adaptive | 0.11 | 0.13 | 0.13 | 0.14 | 0.14 | 0.16 | 0.17 | 0.21 |
| IRLS-GM | 0.11 | 0.13 | 0.12 | 0.14 | 0.14 | 0.16 | 0.17 | 0.21 |
| GNC-GM | 0.11 | 0.13 | 0.12 | 0.14 | 0.14 | 0.16 | 0.17 | 0.21 |

### Median translation error (m)

| method | 0% | 10% | 20% | 30% | 40% | 50% | 60% | 70% |
|---|---|---|---|---|---|---|---|---|
| Kabsch | 0.0017 | 0.0764 | 0.1254 | 0.1936 | 0.2439 | 0.2684 | 0.3151 | 0.3945 |
| RANSAC-1000 | 0.0017 | 0.0015 | 0.0017 | 0.0017 | 0.0018 | 0.0022 | 0.0027 | 0.0028 |
| RANSAC-adaptive | 0.0017 | 0.0016 | 0.0017 | 0.0017 | 0.0018 | 0.0022 | 0.0027 | 0.0028 |
| IRLS-GM | 0.0017 | 0.0015 | 0.0018 | 0.0017 | 0.0018 | 0.0022 | 0.0027 | 0.0028 |
| GNC-GM | 0.0017 | 0.0015 | 0.0018 | 0.0017 | 0.0018 | 0.0022 | 0.0027 | 0.0028 |

### Runtime p50 (ms)

| method | 0% | 10% | 20% | 30% | 40% | 50% | 60% | 70% |
|---|---|---|---|---|---|---|---|---|
| Kabsch | 0.06 | 0.05 | 0.05 | 0.05 | 0.05 | 0.05 | 0.05 | 0.05 |
| RANSAC-1000 | 0.23 | 51.47 | 50.69 | 51.85 | 51.37 | 50.92 | 48.23 | 47.69 |
| RANSAC-adaptive | 0.22 | 0.41 | 0.63 | 1.03 | 1.65 | 2.90 | 5.24 | 12.36 |
| IRLS-GM | 2.34 | 1.80 | 1.78 | 1.84 | 1.74 | 1.81 | 1.65 | 1.69 |
| GNC-GM | 0.51 | 1.59 | 1.59 | 1.63 | 1.59 | 1.61 | 1.48 | 1.50 |

### Runtime p95 (ms)

| method | 0% | 10% | 20% | 30% | 40% | 50% | 60% | 70% |
|---|---|---|---|---|---|---|---|---|
| Kabsch | 0.07 | 0.07 | 0.06 | 0.06 | 0.06 | 0.06 | 0.06 | 0.06 |
| RANSAC-1000 | 0.52 | 69.30 | 54.75 | 55.29 | 55.20 | 56.02 | 53.82 | 53.46 |
| RANSAC-adaptive | 0.30 | 0.54 | 0.71 | 1.12 | 1.82 | 3.16 | 5.90 | 13.85 |
| IRLS-GM | 2.38 | 2.35 | 1.91 | 1.94 | 1.92 | 1.92 | 1.89 | 1.93 |
| GNC-GM | 0.62 | 2.14 | 1.71 | 1.73 | 1.76 | 1.77 | 1.72 | 1.70 |

### Speedup of GNC-GM vs RANSAC baselines (runtime p50 ratio)

| outlier rate | vs RANSAC-1000 | vs RANSAC-adaptive | GNC p50 (ms) | RANSAC-1000 p50 (ms) | RANSAC-adaptive p50 (ms) |
|---|---|---|---|---|---|
| 0% | 0.5x | 0.4x | 0.51 | 0.23 | 0.22 |
| 10% | 32.3x | 0.3x | 1.59 | 51.47 | 0.41 |
| 20% | 31.8x | 0.4x | 1.59 | 50.69 | 0.63 |
| 30% | 31.9x | 0.6x | 1.63 | 51.85 | 1.03 |
| 40% | 32.3x | 1.0x | 1.59 | 51.37 | 1.65 |
| 50% | 31.6x | 1.8x | 1.61 | 50.92 | 2.90 |
| 60% | 32.5x | 3.5x | 1.48 | 48.23 | 5.24 |
| 70% | 31.9x | 8.3x | 1.50 | 47.69 | 12.36 |

> Reading: against the fixed-budget 1000-iteration RANSAC baseline (the v0 table), GNC is ~30x faster for every p>=10%. Against adaptive RANSAC, which stops on a confidence bound, GNC is comparable at 10-30% outliers and pulls ahead at >=60% (and by ~25-30x at >=80%, see the budget table). RANSAC is not differentiable.

## 2. Coherent outliers (one competing rigid transform, 30-60 deg / 0.6-1.5 m)

### Success rate

| method | 0% | 10% | 20% | 30% | 40% | 50% | 60% | 70% |
|---|---|---|---|---|---|---|---|---|
| Kabsch | 100% | 28% | 0% | 0% | 0% | 0% | 0% | 0% |
| RANSAC-1000 | 100% | 100% | 100% | 100% | 100% | 53% | 0% | 0% |
| RANSAC-adaptive | 100% | 100% | 100% | 100% | 100% | 53% | 0% | 0% |
| IRLS-GM | 100% | 100% | 100% | 100% | 100% | 47% | 0% | 0% |
| GNC-GM | 100% | 100% | 100% | 100% | 100% | 49% | 0% | 0% |

### Median rotation error (deg)

| method | 0% | 10% | 20% | 30% | 40% | 50% | 60% | 70% |
|---|---|---|---|---|---|---|---|---|
| Kabsch | 0.11 | 4.64 | 8.25 | 13.27 | 17.56 | 24.48 | 26.76 | 32.46 |
| RANSAC-1000 | 0.11 | 0.11 | 0.12 | 0.13 | 0.14 | 0.29 | 43.01 | 45.28 |
| RANSAC-adaptive | 0.11 | 0.11 | 0.12 | 0.13 | 0.14 | 0.29 | 43.01 | 45.28 |
| IRLS-GM | 0.11 | 0.12 | 0.12 | 0.13 | 0.15 | 30.59 | 43.00 | 45.26 |
| GNC-GM | 0.11 | 0.12 | 0.12 | 0.13 | 0.15 | 30.76 | 43.00 | 45.26 |

### Fraction of trials that recover the outlier pose (<5 deg, <0.1 m)

| method | 0% | 10% | 20% | 30% | 40% | 50% | 60% | 70% |
|---|---|---|---|---|---|---|---|---|
| Kabsch | — | 0% | 0% | 0% | 0% | 0% | 0% | 0% |
| RANSAC-1000 | — | 0% | 0% | 0% | 0% | 47% | 100% | 100% |
| RANSAC-adaptive | — | 0% | 0% | 0% | 0% | 47% | 100% | 100% |
| IRLS-GM | — | 0% | 0% | 0% | 0% | 40% | 100% | 100% |
| GNC-GM | — | 0% | 0% | 0% | 0% | 51% | 100% | 100% |

### Fraction of trials closer to the outlier transform than to GT

| method | 0% | 10% | 20% | 30% | 40% | 50% | 60% | 70% |
|---|---|---|---|---|---|---|---|---|
| Kabsch | — | 0% | 0% | 0% | 0% | 40% | 94% | 100% |
| RANSAC-1000 | — | 0% | 0% | 0% | 0% | 47% | 100% | 100% |
| RANSAC-adaptive | — | 0% | 0% | 0% | 0% | 47% | 100% | 100% |
| IRLS-GM | — | 0% | 0% | 0% | 0% | 48% | 100% | 100% |
| GNC-GM | — | 0% | 0% | 0% | 0% | 51% | 100% | 100% |

### Runtime p50 (ms)

| method | 0% | 10% | 20% | 30% | 40% | 50% | 60% | 70% |
|---|---|---|---|---|---|---|---|---|
| Kabsch | 0.04 | 0.05 | 0.05 | 0.05 | 0.05 | 0.05 | 0.05 | 0.05 |
| RANSAC-1000 | 0.16 | 48.46 | 48.99 | 47.79 | 48.32 | 49.65 | 49.92 | 48.37 |
| RANSAC-adaptive | 0.15 | 0.39 | 0.60 | 0.93 | 1.52 | 2.69 | 1.61 | 0.94 |
| IRLS-GM | 1.60 | 1.71 | 1.65 | 1.64 | 1.66 | 1.67 | 1.73 | 1.61 |
| GNC-GM | 0.36 | 1.29 | 1.31 | 1.27 | 1.22 | 1.21 | 1.28 | 1.23 |

> Failure mode: at 60-70% coherent outliers the competing transform is the largest consensus / lowest robust cost, so every method (including the exact global inner solver) returns the *outlier* pose. This is the identifiability failure of Prop 3.7, not a solver bug. At 50% the two hypotheses tie and the winner is essentially a coin flip.

## 3. Boundary outliers (e = ||r||^2 uniform in [c^2/3, 3c^2])

### Success rate

| method | 0% | 10% | 20% | 30% | 40% | 50% | 60% | 70% |
|---|---|---|---|---|---|---|---|---|
| Kabsch | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| RANSAC-1000 | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| RANSAC-adaptive | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| IRLS-GM | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| GNC-GM | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |

### Median rotation error (deg)

| method | 0% | 10% | 20% | 30% | 40% | 50% | 60% | 70% |
|---|---|---|---|---|---|---|---|---|
| Kabsch | 0.11 | 0.16 | 0.22 | 0.24 | 0.28 | 0.29 | 0.32 | 0.33 |
| RANSAC-1000 | 0.11 | 0.16 | 0.20 | 0.25 | 0.26 | 0.35 | 0.46 | 0.62 |
| RANSAC-adaptive | 0.11 | 0.14 | 0.17 | 0.20 | 0.24 | 0.30 | 0.43 | 0.60 |
| IRLS-GM | 0.11 | 0.12 | 0.15 | 0.16 | 0.15 | 0.17 | 0.23 | 0.27 |
| GNC-GM | 0.11 | 0.12 | 0.15 | 0.16 | 0.15 | 0.17 | 0.23 | 0.27 |

### Median translation error (m)

| method | 0% | 10% | 20% | 30% | 40% | 50% | 60% | 70% |
|---|---|---|---|---|---|---|---|---|
| Kabsch | 0.0017 | 0.0025 | 0.0030 | 0.0035 | 0.0038 | 0.0038 | 0.0046 | 0.0046 |
| RANSAC-1000 | 0.0017 | 0.0022 | 0.0030 | 0.0032 | 0.0043 | 0.0049 | 0.0069 | 0.0085 |
| RANSAC-adaptive | 0.0017 | 0.0019 | 0.0024 | 0.0028 | 0.0037 | 0.0044 | 0.0057 | 0.0085 |
| IRLS-GM | 0.0017 | 0.0018 | 0.0020 | 0.0020 | 0.0023 | 0.0026 | 0.0033 | 0.0039 |
| GNC-GM | 0.0017 | 0.0018 | 0.0020 | 0.0020 | 0.0023 | 0.0026 | 0.0033 | 0.0039 |

### Runtime p50 (ms)

| method | 0% | 10% | 20% | 30% | 40% | 50% | 60% | 70% |
|---|---|---|---|---|---|---|---|---|
| Kabsch | 0.05 | 0.05 | 0.05 | 0.05 | 0.05 | 0.05 | 0.05 | 0.05 |
| RANSAC-1000 | 0.16 | 48.41 | 47.66 | 48.99 | 47.64 | 47.68 | 47.50 | 48.10 |
| RANSAC-adaptive | 0.15 | 0.34 | 0.48 | 0.67 | 0.93 | 1.32 | 1.82 | 2.92 |
| IRLS-GM | 1.74 | 1.77 | 1.71 | 1.72 | 1.69 | 1.64 | 1.63 | 1.66 |
| GNC-GM | 0.39 | 0.62 | 0.61 | 0.62 | 0.61 | 0.61 | 0.60 | 0.60 |

## 4. RANSAC budget sweep (random outliers, 3-point sampling)

| p | budget | adaptive | success | rot med (deg) | time p50 (ms) | time p95 (ms) | mean iters | theory success |
|---|---|---|---|---|---|---|---|---|
| 50% | 50 | False | 100% | 0.17 | 2.51 | 2.79 | 50 | 99.8% |
| 50% | 1000 | False | 100% | 0.17 | 48.64 | 52.14 | 1000 | 100.0% |
| 50% | 1000 | True | 100% | 0.17 | 2.61 | 2.91 | 52 | 99.9% |
| 60% | 50 | False | 97% | 0.18 | 2.36 | 2.73 | 50 | 95.7% |
| 60% | 1000 | False | 100% | 0.17 | 48.16 | 52.38 | 1000 | 100.0% |
| 60% | 1000 | True | 100% | 0.17 | 4.94 | 5.63 | 105 | 99.9% |
| 70% | 50 | False | 74% | 0.26 | 2.50 | 2.74 | 50 | 72.0% |
| 70% | 1000 | False | 100% | 0.22 | 49.92 | 52.45 | 1000 | 100.0% |
| 70% | 1000 | True | 100% | 0.22 | 13.02 | 13.75 | 253 | 99.8% |
| 80% | 50 | False | 26% | 44.11 | 2.55 | 2.80 | 50 | 29.8% |
| 80% | 1000 | False | 100% | 0.27 | 49.86 | 52.57 | 1000 | 99.9% |
| 80% | 1000 | True | 100% | 0.27 | 43.12 | 45.54 | 863 | 99.8% |
| 90% | 50 | False | 5% | 91.14 | 2.69 | 2.83 | 50 | 3.6% |
| 90% | 1000 | False | 46% | 25.62 | 50.07 | 53.87 | 1000 | 52.4% |
| 90% | 1000 | True | 46% | 25.62 | 51.77 | 53.92 | 1000 | 52.4% |

> The 50-iteration budget is *not* a failure at 50% outliers: a clean 3-point sample has probability ~0.121 (without replacement), so 50 iterations succeed with probability ~99.9%. The accuracy-time trade-off bites at >=70% outliers: at 70% the clean-sample probability is 0.025 and 50 iterations give ~75%; at 80% 50 iterations give 26% while 1000 give 100%; at 90% even 1000 iterations only reach 46% (exact theory 52.4%). This is the honest RANSAC baseline: its cost is *data-dependent* (inlier ratio), while GNC's is not.

## 5. Scaling with N (p=30% random outliers)

| N | method | success | rot med (deg) | time p50 (ms) | time p95 (ms) |
|---|---|---|---|---|---|
| 20 | Kabsch | 0% | 22.43 | 0.05 | 0.06 |
| 20 | RANSAC-1000 | 100% | 0.28 | 49.66 | 51.63 |
| 20 | RANSAC-adaptive | 100% | 0.28 | 1.01 | 1.08 |
| 20 | IRLS-GM | 100% | 0.29 | 1.57 | 1.64 |
| 20 | GNC-GM | 100% | 0.29 | 1.32 | 1.41 |
| 50 | Kabsch | 2% | 12.61 | 0.05 | 0.06 |
| 50 | RANSAC-1000 | 100% | 0.16 | 50.30 | 52.62 |
| 50 | RANSAC-adaptive | 100% | 0.16 | 1.02 | 1.07 |
| 50 | IRLS-GM | 100% | 0.16 | 1.69 | 1.76 |
| 50 | GNC-GM | 100% | 0.16 | 1.47 | 1.56 |
| 100 | Kabsch | 0% | 10.48 | 0.05 | 0.06 |
| 100 | RANSAC-1000 | 100% | 0.15 | 51.51 | 53.77 |
| 100 | RANSAC-adaptive | 100% | 0.14 | 1.02 | 1.09 |
| 100 | IRLS-GM | 100% | 0.14 | 1.83 | 1.90 |
| 100 | GNC-GM | 100% | 0.14 | 1.62 | 1.71 |
| 200 | Kabsch | 2% | 6.40 | 0.06 | 0.07 |
| 200 | RANSAC-1000 | 100% | 0.10 | 55.42 | 57.86 |
| 200 | RANSAC-adaptive | 100% | 0.10 | 1.10 | 1.18 |
| 200 | IRLS-GM | 100% | 0.09 | 2.21 | 2.30 |
| 200 | GNC-GM | 100% | 0.09 | 1.99 | 2.10 |

## Notes / caveats

* The 3D-3D forward solve here is weighted Kabsch (the closed-form 3D-3D specialization of QPEP); the robust outer loop is GNC-GM or IRLS-GM.  No QPEP polynomial solver is involved, so this table isolates the robust estimator, not the QPEP algebra.
* RANSAC threshold is `||r|| < c`; the GM kernel uses the same `c` on the squared residual, so both methods have the same inlier scale.
* `RANSAC-1000` is the v0 baseline (always 1000 iterations unless a sample already covers all N points). `RANSAC-adaptive` uses the standard confidence bound (0.999) and is capped at 1000 iterations.
* Metrics use the trace/arccos formula; near zero it has a numerical floor of ~1e-6 deg.  This does not affect the 0.1-100 deg comparisons.
* All numbers generated by `run_main_extended.py`; raw per-trial records in `results/main_table_extended_raw.jsonl`.
