# SDPRLayers baselines (our re-implementation) on the constrained-QPEP instances

Generated 2026-11-28 01:13:15 by `experiments/code/make_sdpr_results.py` from the raw JSON records; do not edit by hand.

**Evidence grade: independently reproduced in this repository** for the SDP values, poses and timings; **not** a reproduction of the official SDPRLayers software (see the caveats at the end).  This file is the A7 aggregate of the P2 final push: every table gives n, and every cell was measured (no estimated or interpolated numbers).

## Table 1. Synthetic desktop cells, same instances as the main table

`make_instance(trial, p, noise=0.01, bias_z=0.08)`, 50 trials per p, `seed=trial` (paired with `raw_main.json`).  Penetration / pose errors are median [IQR]; runtime is the forward wall clock of that method (certificate time excluded except for the certificate row).

| p | method | n | pen mm [IQR] | pen>1mm | rot deg [IQR] | trans mm [IQR] | feasible | certified (sound) | ms [IQR] | λ2/λ1 med | rank med | extract gap med |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 8 | Unconstrained (published) | 50 | 103.02 [84.69, 110.83] | 100% [93, 100] | 5.30 [3.26, 6.70] | 97.90 [88.42, 107.73] | 0% [0, 7] | 0% [0, 7] | – [–, –] | – | – | – |
| 8 | SDPR-unc (B1) | 50 | 103.02 [84.69, 110.83] | 100% [93, 100] | 5.30 [3.26, 6.70] | 97.90 [88.42, 107.73] | 0% [0, 7] | 0% [0, 7] | 31 [28, 47] | 2.57e-11 | 1.0 | -6.11e-14 |
| 8 | SDPR-unc + polish | 50 | 103.02 [84.69, 110.83] | 100% [93, 100] | 5.30 [3.26, 6.70] | 97.90 [88.42, 107.73] | 0% [0, 7] | 0% [0, 7] | 19 [15, 22] | – | – | – |
| 8 | SDPR-slack (B2a) | 50 | 5.16e-08 [0.00, 3.07e-07] | 0% [0, 7] | 4.17 [2.74, 5.10] | 10.63 [8.17, 12.69] | 98% [90, 100] | 98% [90, 100] | 63 [59, 71] | 0.21 | 6.0 | -5.54e-11 |
| 8 | SDPR-hard (M→∞) | 50 | 4.25e-07 [1.72e-08, 4.08e-06] | 0% [0, 7] | 4.17 [2.74, 5.10] | 10.63 [8.17, 12.69] | 96% [87, 99] | 96% [87, 99] | 30 [29, 33] | 7.12e-11 | 1.0 | -1.76e-11 |
| 8 | ALM-QPEP (published) | 50 | 3.68e-08 [2.40e-08, 7.15e-08] | 0% [0, 7] | 4.17 [2.74, 5.10] | 10.63 [8.17, 12.69] | 100% [93, 100] | 100% [93, 100] | 260 [183, 341] | – | – | – |
| 16 | Unconstrained (published) | 50 | 103.02 [84.69, 110.83] | 100% [93, 100] | 5.30 [3.26, 6.70] | 97.90 [88.42, 107.73] | 0% [0, 7] | 0% [0, 7] | – [–, –] | – | – | – |
| 16 | SDPR-unc (B1) | 50 | 103.02 [84.69, 110.83] | 100% [93, 100] | 5.30 [3.26, 6.70] | 97.90 [88.42, 107.73] | 0% [0, 7] | 0% [0, 7] | 30 [27, 36] | 2.57e-11 | 1.0 | -6.11e-14 |
| 16 | SDPR-unc + polish | 50 | 103.02 [84.69, 110.83] | 100% [93, 100] | 5.30 [3.26, 6.70] | 97.90 [88.42, 107.73] | 0% [0, 7] | 0% [0, 7] | 18 [14, 22] | – | – | – |
| 16 | SDPR-slack (B2a) | 50 | 1.47e-07 [0.00, 2.00e-06] | 0% [0, 7] | 4.17 [2.74, 5.10] | 10.63 [8.17, 12.69] | 98% [90, 100] | 98% [90, 100] | 178 [157, 209] | 0.15 | 7.0 | -4.28e-10 |
| 16 | SDPR-hard (M→∞) | 50 | 3.37e-07 [3.26e-09, 1.38e-06] | 0% [0, 7] | 4.17 [2.74, 5.10] | 10.63 [8.17, 12.69] | 98% [90, 100] | 98% [90, 100] | 33 [32, 35] | 9.57e-11 | 1.0 | -2.03e-12 |
| 16 | ALM-QPEP (published) | 50 | 3.68e-08 [9.31e-09, 6.89e-08] | 0% [0, 7] | 4.17 [2.74, 5.10] | 10.63 [8.17, 12.69] | 100% [93, 100] | 100% [93, 100] | 277 [219, 336] | – | – | – |
| 32 | Unconstrained (published) | 50 | 103.02 [84.69, 110.83] | 100% [93, 100] | 5.30 [3.26, 6.70] | 97.90 [88.42, 107.73] | 0% [0, 7] | 0% [0, 7] | – [–, –] | – | – | – |
| 32 | SDPR-unc (B1) | 50 | 103.02 [84.69, 110.83] | 100% [93, 100] | 5.30 [3.26, 6.70] | 97.90 [88.42, 107.73] | 0% [0, 7] | 0% [0, 7] | 31 [28, 45] | 2.57e-11 | 1.0 | -6.11e-14 |
| 32 | SDPR-unc + polish | 50 | 103.02 [84.69, 110.83] | 100% [93, 100] | 5.30 [3.26, 6.70] | 97.90 [88.42, 107.73] | 0% [0, 7] | 0% [0, 7] | 19 [15, 25] | – | – | – |
| 32 | SDPR-slack (B2a) | 50 | 1.94e-07 [0.00, 2.65e-06] | 0% [0, 7] | 4.17 [2.74, 5.10] | 10.63 [8.17, 12.69] | 92% [81, 97] | 92% [81, 97] | 532 [434, 682] | 0.12 | 9.0 | -1.48e-10 |
| 32 | SDPR-hard (M→∞) | 50 | 5.48e-07 [1.14e-08, 2.87e-06] | 0% [0, 7] | 4.17 [2.74, 5.10] | 10.63 [8.17, 12.69] | 98% [90, 100] | 98% [90, 100] | 42 [39, 59] | 4.40e-11 | 1.0 | -1.52e-10 |
| 32 | ALM-QPEP (published) | 50 | 3.77e-08 [2.14e-08, 6.21e-08] | 0% [0, 7] | 4.17 [2.74, 5.10] | 10.63 [8.17, 12.69] | 100% [93, 100] | 100% [93, 100] | 280 [226, 372] | – | – | – |
| 64 | Unconstrained (published) | 50 | 103.02 [84.69, 110.83] | 100% [93, 100] | 5.30 [3.26, 6.70] | 97.90 [88.42, 107.73] | 0% [0, 7] | 0% [0, 7] | – [–, –] | – | – | – |
| 64 | SDPR-unc (B1) | 50 | 103.02 [84.69, 110.83] | 100% [93, 100] | 5.30 [3.26, 6.70] | 97.90 [88.42, 107.73] | 0% [0, 7] | 0% [0, 7] | 35 [31, 47] | 2.57e-11 | 1.0 | -6.11e-14 |
| 64 | SDPR-unc + polish | 50 | 103.02 [84.69, 110.83] | 100% [93, 100] | 5.30 [3.26, 6.70] | 97.90 [88.42, 107.73] | 0% [0, 7] | 0% [0, 7] | 21 [16, 26] | – | – | – |
| 64 | SDPR-slack (B2a) | 50 | 1.98e-07 [0.00, 1.30e-06] | 0% [0, 7] | 4.17 [2.74, 5.10] | 10.63 [8.17, 12.69] | 92% [81, 97] | 92% [81, 97] | 2405 [1815, 2975] | 0.09 | 10.0 | -6.84e-10 |
| 64 | SDPR-hard (M→∞) | 50 | 2.48e-07 [0.00, 1.43e-05] | 0% [0, 7] | 4.17 [2.74, 5.10] | 10.63 [8.17, 12.69] | 92% [81, 97] | 92% [81, 97] | 59 [51, 79] | 1.33e-10 | 1.0 | 2.53e-11 |
| 64 | ALM-QPEP (published) | 50 | 3.75e-08 [1.88e-08, 6.03e-08] | 0% [0, 7] | 4.17 [2.74, 5.10] | 10.63 [8.17, 12.69] | 100% [93, 100] | 100% [93, 100] | 288 [240, 329] | – | – | – |

`certified (sound)` = feasible and `J - bound <= 1e-6`, where the bound is the P2-Cert Schur bound (fallback: dual repair) computed on the same instance; never the SCS value.  `unc*`/`ALM*` are the published rows re-read from `raw_main.json`.

## Table 2. SDP diagnostics (synthetic)

| p | PSD size unc/hard/slack | ineq hard/slack | λ2/λ1 med B1 | λ2/λ1 max B1 | rank med B1 | λ2/λ1 med B2a | rank med B2a | extract gap med B1 | B2a value − hard (med / q3) |
|---|---|---|---|---|---|---|---|---|---|
| 8 | 14/14/22 | 8/8 | 2.57e-11 | 4.95e-10 | 1.0 | 0.21 | 6.0 | -6.11e-14 | 3.50e-10 / 2.32e-09 |
| 16 | 14/14/30 | 16/16 | 2.57e-11 | 4.95e-10 | 1.0 | 0.15 | 7.0 | -6.11e-14 | 2.06e-10 / 2.18e-09 |
| 32 | 14/14/46 | 32/32 | 2.57e-11 | 4.95e-10 | 1.0 | 0.12 | 9.0 | -6.11e-14 | -2.41e-10 / 1.86e-09 |
| 64 | 14/14/78 | 64/64 | 2.57e-11 | 4.95e-10 | 1.0 | 0.09 | 10.0 | -6.11e-14 | 7.68e-10 / 5.37e-09 |

The λ2/λ1 and rank columns are read from the SCS solution; `extract gap` = J(extracted pose) − SDP value is the solver-independent tightness measure that matters for the forward layer.


## Table 4. Real BOP instances (n=90/72/84), same rows as `raw.json`

Replayed from the `rigorous/rigorous_arrays.npz` dumps of `run_bop.py RIGOROUS_ONLY=1`; the join with the published table is exact (`max |ΔJ| = 0`).  Penetration is computed on the full object mesh against the estimated support plane, exactly as in `run_bop.py`.  CIs are 95% Wilson.

| dataset | n | method | pen mm [IQR] | pen>1mm | rot deg [IQR] | trans mm [IQR] | feasible | certified (sound) | ms [IQR] | λ2/λ1 med | rank med | extract gap med |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LINEMOD | 90 | Unconstrained (published) | 11.26 [5.00, 19.72] | 86% [77, 91] | 2.08 [1.33, 2.64] | 35.63 [19.16, 47.01] | 13% [8, 22] | 13% [8, 22] | – [–, –] | – | – | – |
| LINEMOD | 90 | SDPR-unc (B1) | 11.47 [5.00, 20.58] | 87% [78, 92] | 2.19 [1.39, 2.69] | 36.02 [19.37, 50.65] | 12% [7, 21] | 9% [5, 17] | 198 [72, 441] | 3.85e-11 | 1.0 | -7.19e-13 |
| LINEMOD | 90 | SDPR-unc + polish | 11.26 [5.00, 19.72] | 86% [77, 91] | 2.08 [1.33, 2.64] | 35.63 [19.16, 47.01] | 13% [8, 22] | 13% [8, 22] | 16 [14, 21] | – | – | – |
| LINEMOD | 90 | SDPR-slack (B2a) | 2.40 [0.00, 12.07] | 53% [43, 63] | 2.50 [1.54, 3.28] | 38.27 [22.65, 55.56] | 42% [33, 53] | 12% [7, 21] | 3169 [2979, 3340] | 0.09 | 5.0 | -4.45e-07 |
| LINEMOD | 90 | SDPR-hard (M→∞) | 9.19 [3.42, 19.85] | 83% [74, 90] | 2.14 [1.35, 2.72] | 35.95 [19.72, 49.22] | 14% [9, 23] | 13% [8, 22] | 675 [640, 697] | 0.01 | 2.0 | -1.36e-06 |
| LINEMOD | 90 | ALM-QPEP (published) | 2.16e-08 [4.30e-10, 0.38] | 14% [9, 23] | 1.97 [1.26, 2.75] | 37.66 [24.57, 52.45] | 100% [96, 100] | 24% [17, 34] | – [–, –] | – | – | – |
| LM-O | 72 | Unconstrained (published) | 14.93 [8.26, 22.49] | 93% [85, 97] | 2.07 [1.40, 2.83] | 34.69 [19.52, 49.20] | 7% [3, 15] | 7% [3, 15] | – [–, –] | – | – | – |
| LM-O | 72 | SDPR-unc (B1) | 14.93 [8.26, 22.49] | 93% [85, 97] | 2.07 [1.40, 2.83] | 34.69 [19.52, 49.20] | 7% [3, 15] | 7% [3, 15] | 132 [92, 231] | 2.81e-11 | 1.0 | -5.31e-13 |
| LM-O | 72 | SDPR-unc + polish | 14.93 [8.26, 22.49] | 93% [85, 97] | 2.07 [1.40, 2.83] | 34.69 [19.52, 49.20] | 7% [3, 15] | 7% [3, 15] | 15 [13, 19] | – | – | – |
| LM-O | 72 | SDPR-slack (B2a) | 1.96 [0.00, 12.39] | 61% [50, 72] | 2.53 [1.60, 4.15] | 47.07 [24.08, 64.42] | 32% [22, 43] | 3% [1, 10] | 3150 [2974, 3336] | 0.10 | 6.0 | 9.91e-07 |
| LM-O | 72 | SDPR-hard (M→∞) | 12.70 [5.52, 19.68] | 88% [78, 93] | 2.08 [1.37, 2.82] | 34.89 [19.85, 50.04] | 12% [7, 22] | 7% [3, 15] | 669 [643, 691] | 0.01 | 2.0 | -4.89e-06 |
| LM-O | 72 | ALM-QPEP (published) | 4.83e-08 [4.08e-09, 0.35] | 10% [5, 19] | 1.99 [1.26, 2.73] | 38.14 [21.66, 51.44] | 100% [95, 100] | 15% [9, 25] | – [–, –] | – | – | – |
| YCB-Video | 84 | Unconstrained (published) | 14.09 [6.84, 18.91] | 93% [85, 97] | 1.08 [0.79, 1.55] | 17.33 [9.79, 23.84] | 6% [3, 13] | 6% [3, 13] | – [–, –] | – | – | – |
| YCB-Video | 84 | SDPR-unc (B1) | 11.40 [6.00, 18.69] | 86% [77, 92] | 1.12 [0.74, 1.55] | 19.06 [10.68, 25.12] | 13% [7, 22] | 6% [3, 13] | 238 [138, 592] | 4.67e-11 | 1.0 | -5.39e-12 |
| YCB-Video | 84 | SDPR-unc + polish | 14.09 [6.84, 18.91] | 93% [85, 97] | 1.08 [0.79, 1.55] | 17.33 [9.79, 23.84] | 6% [3, 13] | 6% [3, 13] | 16 [13, 20] | – | – | – |
| YCB-Video | 84 | SDPR-slack (B2a) | 0.28 [0.00, 7.91] | 48% [37, 58] | 1.43 [0.91, 2.39] | 28.31 [21.84, 140.51] | 46% [36, 57] | 2% [1, 8] | 3185 [3006, 3447] | 0.13 | 6.0 | 1.85e-05 |
| YCB-Video | 84 | SDPR-hard (M→∞) | 11.28 [5.42, 15.20] | 90% [82, 95] | 1.18 [0.80, 1.56] | 17.27 [10.67, 24.31] | 10% [5, 18] | 6% [3, 13] | 667 [634, 705] | 0.01 | 2.0 | -3.25e-06 |
| YCB-Video | 84 | ALM-QPEP (published) | 7.21e-09 [1.03e-09, 4.36e-05] | 1% [0, 6] | 1.11 [0.59, 1.71] | 25.18 [16.67, 36.89] | 100% [96, 100] | 13% [7, 22] | – [–, –] | – | – | – |


## Table 5. Real-data SDP diagnostics

| dataset | method | PSD size | λ2/λ1 med | rank med | extract gap med | value − stored cert value (med) | eig-extraction | mirror poses |
|---|---|---|---|---|---|---|---|---|
| LINEMOD | B1 | 14 | 3.85e-11 | 1.0 | -7.19e-13 | -1.67e-06 | 100% [96, 100] | 0 |
| LINEMOD | hard | 14 | 0.01 | 2.0 | -1.36e-06 | -2.68e-09 | 100% [96, 100] | 1 |
| LINEMOD | B2a | 30 | 0.09 | 5.0 | -4.45e-07 | 1.24e-05 | 70% [60, 78] | 7 |
| LM-O | B1 | 14 | 2.81e-11 | 1.0 | -5.31e-13 | -5.28e-06 | 100% [95, 100] | 0 |
| LM-O | hard | 14 | 0.01 | 2.0 | -4.89e-06 | -1.03e-08 | 99% [93, 100] | 4 |
| LM-O | B2a | 30 | 0.10 | 6.0 | 9.91e-07 | 7.55e-06 | 58% [47, 69] | 2 |
| YCB-Video | B1 | 14 | 4.67e-11 | 1.0 | -5.39e-12 | -1.66e-06 | 100% [96, 100] | 0 |
| YCB-Video | hard | 14 | 0.01 | 2.0 | -3.25e-06 | 0.00 | 100% [96, 100] | 2 |
| YCB-Video | B2a | 30 | 0.13 | 6.0 | 1.85e-05 | 1.93e-05 | 63% [52, 73] | 7 |

`mirror poses` counts extracted poses with fewer than half of the mesh vertices in front of the camera (cheirality mirror of the same algebraic cost); `eig-extraction` is the share of rows where the leading eigenvector (rather than the first-moment column) was the dehomogenisation route.

Instance checks: LINEMOD n=90, max|ΔJ| arrays vs rows 0.0e+00, published join 90/90 (max|ΔJ| 0.0e+00); LM-O n=72, max|ΔJ| arrays vs rows 0.0e+00, published join 72/72 (max|ΔJ| 0.0e+00); YCB-Video n=84, max|ΔJ| arrays vs rows 0.0e+00, published join 84/84 (max|ΔJ| 0.0e+00).


## Table 6. A5 penalty-weight sweep of the lifted slack form (SCS)

`min J + M sum_j s_j, g_j - s_j <= 0, s_j >= 0`, same instances as the main synthetic cells (seed = trial).  The penalised form is a *weakening* of the hard relaxation (at rank 1 it admits any g_j with s_j = max(0, g_j)), so its SDP value may not exceed the hard value; a positive `value − hard` is SCS inflation on an `optimal_inaccurate` solve, not a better relaxation.  `certified` is against the sound P2-Cert Schur bound.

| p | M | ok/n | status | value−hard med (max) | pen mm med [IQR] | pen>1mm | rot med | trans med | feasible | certified (sound) | ms med | s_max med |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 8 | 100 | 20/20 | optimal:20 | -1.07e-09 (+4.77e-08) | 0.000 [0.000, 0.000] | 0% [0, 16] | 3.55 | 10.30 | 90% [70, 97] | 90% [70, 97] | 614 | 7.78e-13 |
| 8 | 1000 | 20/20 | optimal:4, optimal_inaccurate:16 | +8.73e-07 (+1.10e-05) | 0.002 [0.000, 0.010] | 0% [0, 16] | 3.71 | 10.30 | 40% [22, 61] | 40% [22, 61] | 1655 | 3.31e-13 |
| 8 | 10000 | 20/20 | optimal:1, optimal_inaccurate:19 | +6.42e-05 (+1.29e-04) | 0.009 [0.002, 0.043] | 0% [0, 16] | 3.91 | 10.36 | 20% [8, 42] | 15% [5, 36] | 1949 | 4.71e-13 |
| 16 | 100 | 20/20 | optimal:19, optimal_inaccurate:1 | -2.98e-10 (+1.95e-07) | 0.000 [0.000, 0.000] | 0% [0, 16] | 3.55 | 10.30 | 85% [64, 95] | 85% [64, 95] | 1102 | 6.30e-13 |
| 16 | 1000 | 20/20 | optimal:9, optimal_inaccurate:11 | +2.46e-08 (+1.16e-05) | 0.000 [0.000, 0.004] | 0% [0, 16] | 3.55 | 10.30 | 65% [43, 82] | 65% [43, 82] | 3557 | 2.05e-13 |
| 16 | 10000 | 20/20 | optimal_inaccurate:20 | +3.53e-05 (+1.65e-04) | 0.016 [0.005, 0.091] | 0% [0, 16] | 4.16 | 10.36 | 20% [8, 42] | 10% [3, 30] | 3896 | 2.87e-12 |

Hard-relaxation reference rows on the same instances:

| p | n | hard value med | hard pen mm med | hard feasible | hard ms med |
|---|---|---|---|---|---|
| 8 | 20 | 3.952e-03 | 0.000 | 95% [76, 99] | 30 |
| 16 | 20 | 3.952e-03 | 0.000 | 100% [84, 100] | 33 |


## Table 7. A6 PSD block size and solver status versus p

Direct probe: the matrix CVXPY/SCS actually returns for the equality-slack lift, one instance per p (seed 0).  The claimed PSD block is 14+p.

| p | expected 14+p | returned slack X | n_eq | n_ineq | status | slack ms |
|---|---|---|---|---|---|---|
| 8 | 22 | **22×22** | 47 | 8 | optimal | 62 |
| 16 | 30 | **30×30** | 55 | 16 | optimal | 167 |
| 32 | 46 | **46×46** | 71 | 32 | optimal | 419 |
| 64 | 78 | **78×78** | 103 | 64 | optimal | 1372 |

Status census over all measured rows (`n_eq` = 39 Shor equalities + p slack equalities; `n_ineq` = p non-negativity rows; status counts from SCS; `retried` counts rows where the solver retry ladder was used):

| cell | p | n | variant | PSD | n_eq | n_ineq | status | retried | ms med | ms max |
|---|---|---|---|---|---|---|---|---|---|---|
| synth | 8 | 50 | unc | 14.0 | 39.0 | 0.0 | optimal:50 | 0 | 31 | 152 |
| synth | 8 | 50 | hard | 14.0 | 39.0 | 8.0 | optimal:50 | 0 | 30 | 50 |
| synth | 8 | 50 | slack | 22.0 | 47.0 | 8.0 | optimal:50 | 0 | 63 | 114 |
| synth | 16 | 50 | unc | 14.0 | 39.0 | 0.0 | optimal:50 | 0 | 30 | 154 |
| synth | 16 | 50 | hard | 14.0 | 39.0 | 16.0 | optimal:50 | 0 | 33 | 56 |
| synth | 16 | 50 | slack | 30.0 | 55.0 | 16.0 | optimal:50 | 0 | 178 | 306 |
| synth | 32 | 50 | unc | 14.0 | 39.0 | 0.0 | optimal:50 | 0 | 31 | 265 |
| synth | 32 | 50 | hard | 14.0 | 39.0 | 32.0 | optimal:50 | 0 | 42 | 392 |
| synth | 32 | 50 | slack | 46.0 | 71.0 | 32.0 | optimal:50 | 4 | 532 | 7841 |
| synth | 64 | 50 | unc | 14.0 | 39.0 | 0.0 | optimal:50 | 0 | 35 | 216 |
| synth | 64 | 50 | hard | 14.0 | 39.0 | 64.0 | optimal:50 | 0 | 59 | 150 |
| synth | 64 | 50 | slack | 78.0 | 103.0 | 64.0 | optimal:50 | 0 | 2405 | 4915 |
| LINEMOD | 16 | 90 | unc | 14 | 39 | 0 | optimal:71, optimal_inaccurate:19 | 0 | 198 | 658 |
| LINEMOD | 16 | 90 | hard | 14 | 39 | 16 | optimal:18, optimal_inaccurate:72 | 0 | 675 | 1028 |
| LINEMOD | 16 | 90 | slack | 30 | 55 | 16 | optimal:11, optimal_inaccurate:79 | 0 | 3169 | 3969 |
| LM-O | 16 | 72 | unc | 14 | 39 | 0 | optimal:71, optimal_inaccurate:1 | 0 | 132 | 624 |
| LM-O | 16 | 72 | hard | 14 | 39 | 16 | optimal:8, optimal_inaccurate:64 | 0 | 669 | 957 |
| LM-O | 16 | 72 | slack | 30 | 55 | 16 | optimal:18, optimal_inaccurate:54 | 0 | 3150 | 3778 |
| YCB-Video | 16 | 84 | unc | 14 | 39 | 0 | optimal:63, optimal_inaccurate:21 | 0 | 238 | 812 |
| YCB-Video | 16 | 84 | hard | 14 | 39 | 16 | optimal:20, optimal_inaccurate:64 | 0 | 667 | 852 |
| YCB-Video | 16 | 84 | slack | 30 | 55 | 16 | optimal:13, optimal_inaccurate:71 | 0 | 3185 | 4383 |

Relaxation-value ordering `slack value >= hard value` (the slack lift is tighter; a row below the hard value by more than 1e-6 is numerically invalid):

| cell | n | median | max | min | rows below hard |
|---|---|---|---|---|---|
| synth p=8 | 50 | +3.50e-10 | +1.99e-08 | -7.12e-09 | 0 |
| synth p=16 | 50 | +2.06e-10 | +1.68e-08 | -2.76e-08 | 0 |
| synth p=32 | 50 | -2.41e-10 | +1.01e-08 | -2.38e-08 | 0 |
| synth p=64 | 50 | +7.68e-10 | +1.47e-08 | -1.86e-08 | 0 |
| LINEMOD | 90 | +9.54e-06 | +2.45e-04 | -2.38e-05 | 13 |
| LM-O | 72 | +1.50e-06 | +3.57e-04 | -4.29e-05 | 11 |
| YCB-Video | 84 | +1.19e-05 | +3.11e-04 | -5.68e-05 | 14 |


## Exact commands, n, and provenance

All runs are CPU-only on the shared login node (`ls`, 40 cores), under
`nice -n 10` and with single-thread BLAS
(`OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1`); the GPU partition was
checked once at the start and was occupied by other users, so nothing was run
there. Interpreter: `/home/syk/miniconda3/envs/sdpr/bin/python`
(numpy 1.26.3, scipy 1.13.1, cvxpy 1.5.4, SCS 3.3.1). Absolute times drift with
node load and are used only for relative comparison.

| cell | n | trials / seeds | command | raw artifact |
|---|---|---|---|---|
| synth p=8 | 50 | seeds 0–49 | `QPE_SDPR_P=8 QPE_TRIALS=50 python code/run_sdpr_synth.py` | `results/baseline_sdpr_synth_p8.json` |
| synth p=16 | 50 | seeds 0–49 | `QPE_SDPR_P=16 QPE_TRIALS=50 python code/run_sdpr_synth.py` | `results/baseline_sdpr_synth_p16.json` |
| synth p=32 | 50 | seeds 0–49 | `QPE_SDPR_P=32 QPE_TRIALS=50 python code/run_sdpr_synth.py` | `results/baseline_sdpr_synth_p32.json` |
| synth p=64 | 50 | seeds 0–49 | `QPE_SDPR_P=64 QPE_TRIALS=50 python code/run_sdpr_synth.py` | `results/baseline_sdpr_synth_p64.json` |
| LINEMOD | 90 | all stored instances | `python code/run_sdpr_bop.py lm` | `results/baseline_sdpr_lm.json` |
| LM-O | 72 | all stored instances | `python code/run_sdpr_bop.py lmo` | `results/baseline_sdpr_lmo.json` |
| YCB-Video | 84 | all stored instances | `python code/run_sdpr_bop.py ycbv` | `results/baseline_sdpr_ycbv.json` |
| A5 M sweep | 20 per (p,M) | seeds 0–19, p=8/16 | `QPE_SWEEP_P=8,16 QPE_SWEEP_TRIALS=20 python code/run_sdpr_sweep.py` | `results/baseline_sdpr_sweep.json` |
| A6 status probe | 1 per p | seed 0, p=8/16/32/64 | `python code/analyze_sdpr_status.py` | `results/baseline_sdpr_status.json` |
| aggregate | – | – | `python code/make_sdpr_results.py` | this file |

The synthetic instances are `qpep_core.make_instance(trial, p, noise=0.01,
bias_z=0.08)` with `seed = trial`, i.e. **the same instances** as the published
Unconstrained / ALM rows in `results/raw_main.json` (50 trials per p); the raw
records re-read those rows instead of recomputing them.  The real instances are
the `rigorous/rigorous_arrays.npz` dumps of `run_bop.py RIGOROUS_ONLY=1` behind
`real/*/results/raw.json`; `run_sdpr_bop.py` joins each row on
`(frame, img, obj, bias_px)` and reports `max |dJ|` against the published table
(0.0 in all three cells).  No instance was regenerated for this round.

**What changed in this round.** `p=16` and LINEMOD were the two under-powered
cells (n=12 and n=3); they are now n=50 and n=90 like the rest.  The old n=12 /
n=3 files are kept as `results/baseline_sdpr_synth_p16_n12_archive.json` and
`results/baseline_sdpr_lm_n3_archive.json` and are **not** used by any table.


## What these numbers are, and what they are not

* **Our re-implementation, not the official SDPRLayers code.** We solve the same
  Shor relaxation they describe (homogeneous QCQP -> SDP, PSD lift of the
  quaternion Veronese vector plus translation) with CVXPY/SCS, and we read a pose
  out of the SDP solution with the rank-1 leading eigenvector.  We do **not**
  have their PyTorch layer, their dual-certificate backward pass, their MOSEK
  settings, their problem scaling or their learned front-end, so nothing here is
  a statement about their released implementation or its speed.  There is no
  backward pass in this report at all: only the forward solve is measured.
* **The 13/14-dim lift with the full rotation ideal is SDPRLayers' 10x10
  relaxation.** By the paper's Theorem P1-2 the Veronese constraints make the
  13x13 (v2,t) relaxation equal to the 10x10 one; we run the 14-dim homogeneous
  version with `e = 1`, which is the same SDP.
* **SCS is not dual feasible.** A solver `value` is not a rigorous bound, and we
  never use it as one.  Every "certified" number uses the sound P2-Cert Schur
  bound (fallback: dual repair), as in `results/rigorous_real.md`.
* **Tightness is solver-dependent.** λ2/λ1 and rank are read from SCS, which
  returns low-rank iterates; an interior-point solver (CLARABEL) returns the
  analytic centre of the same optimal face with a much larger rank.  The
  solver-independent statement is the extraction gap J(pose) - value.
* **Penalty-weight sensitivity.** The penalised slack form is numerically
  fragile: the penalty weight M does not change the solution (the measured slack
  mass at the SDP optimum is ~1e-13 for every M) but it changes the conditioning,
  and at M >= 1e3 SCS returns `optimal_inaccurate` on most rows and a value
  *above* the hard-relaxation value (an impossible value for this weakening of
  the hard problem).  The equality-slack form (B2a), which is what an
  equality-only layer would actually use, has no M and is stable on the synthetic
  cells.  (An earlier internal sweep also saw CLARABEL stall with
  `InsufficientProgress` on part of the M cells; CLARABEL was not re-run in this
  round, so that observation is not part of the table above.)
* **The real-data SDP values are only good to ~1e-4.** On the three BOP cells
  the slack value falls *below* the hard value on 11-14 rows (by up to
  5.7e-5), although `slack >= hard` must hold mathematically.  Both solves are
  SCS and frequently `optimal_inaccurate` (54-79 of 72-90 slack rows), so the
  SDP *values* on real data are not reliable below a few times 1e-5.  Every
  "certified" number therefore uses the separate sound Schur bound on the same
  instance, never the solver value.
* **Timing caveat for the A5 sweep.** The sweep ran concurrently with the three
  real-data jobs on the same node; its absolute milliseconds (0.6-3.9 s per
  penalised solve) are inflated by contention and are used only for the
  within-table comparison of M.
* **Synthetic p saturation is structural** (planar contact is redundant beyond
  three non-collinear points), exactly as in the main table; the p sweep probes
  cost and PSD size, not accuracy.

