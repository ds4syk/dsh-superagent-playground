# 6D gradient table: exact IFT vs Naive IRLS-StopGrad vs finite differences

N=30, c=0.05, sigma=0.01, FD step h=1e-05. 10 boundary (40%) + 10 random (20%) instances; 600 observations; skipped 0 near-singular instances.

Definitions: exact = -H_rob^{-1} dPhi/dy_i; naive = -H_naive^{-1} dPhi_naive/dy_i with w_i frozen and H_naive the exact frozen-weight Hessian (including the sphere term); no-lambda = exact RHS but H_doc without -lambda_rob diag(I_3,0). FD re-solves the polished fixed point to 1e-13 on each side.

## Metrics by residual bin

| bin | method | n | relerr p50 | relerr p90 | cos p50 | cos min |
|---|---|---|---|---|---|---|
| inlier (<c^2/3) | exact | 414 | 4.92e-08 | 5.63e-08 | +1.000000 | +1.000000 |
| inlier (<c^2/3) | naive | 414 | 2.12e-01 | 4.30e-01 | +0.985145 | +0.791009 |
| inlier (<c^2/3) | nolambda | 414 | 8.96e-04 | 2.16e-03 | +1.000000 | +0.999989 |
| boundary (c^2/3..3c^2) | exact | 122 | 3.08e-08 | 5.25e-08 | +1.000000 | +1.000000 |
| boundary (c^2/3..3c^2) | naive | 122 | 1.07e+00 | 1.19e+00 | +0.234203 | -0.357687 |
| boundary (c^2/3..3c^2) | nolambda | 122 | 9.80e-04 | 2.39e-03 | +1.000000 | +0.999994 |
| gross (>3c^2) | exact | 64 | 1.25e-04 | 5.98e-04 | +1.000000 | +0.998591 |
| gross (>3c^2) | naive | 64 | 1.16e+00 | 1.20e+00 | -0.117776 | -0.396923 |
| gross (>3c^2) | nolambda | 64 | 9.44e-04 | 3.26e-03 | +1.000000 | +0.998589 |

> The exact Jacobian matches FD to ~5e-8 in the inlier and boundary bins, and to ~1e-4 (relative) for gross outliers where ||dz*/dy_i|| is tiny and FD suffers cancellation. The naive Jacobian is already biased for inliers (~20% relative error), degrades to ~100% at the boundary, and is anti-correlated for gross outliers. `no-lambda` is a small but systematic bias here (~1e-3 relative); the constraint term is a much larger fraction of H itself when H is not data-dominated (see the Hessian lines below), so it cannot be dropped on principle.

## Sign flips by residual bin (exact vs naive)

| bin | n | mapping radial a | pose translation radial b |
|---|---|---|---|
| inlier (<c^2/3) | 414 | 0% | 0% |
| boundary (c^2/3..3c^2) | 122 | 100% | 100% |
| gross (>3c^2) | 64 | 100% | 100% |

> Mapping level: a_exact = -lambda_parallel(e) is provably opposite to a_naive = -2 rho'(e) for every e>c^2/3. Pose level: b_exact = rhat^T (dt*/dy_i) rhat is not forced to flip (H_rob^{-1} mixes directions, Theorem 3.3(d)), but in this well-conditioned 6D sample it flips for 100% of the boundary/gross observations.

## Representative boundary instance (per observation)

| i | e/c^2 | lambda_parallel | a_exact (map) | a_naive (map) | b_exact (pose t) | b_naive (pose t) | relerr exact | relerr naive | cos exact | cos naive |
|---|---|---|---|---|---|---|---|---|---|---|
| 24 | 0.003 | +1.9620 | -1.96203 | -1.98730 | +0.07553 | +0.06546 | 5.74e-08 | 2.32e-01 | +1.00000 | +0.99343 |
| 5 | 0.018 | +1.7960 | -1.79596 | -1.93078 | +0.06095 | +0.05233 | 5.70e-08 | 1.73e-01 | +1.00000 | +0.99576 |
| 17 | 0.025 | +1.7161 | -1.71611 | -1.90300 | +0.04994 | +0.04606 | 5.11e-08 | 2.30e-01 | +1.00000 | +0.99092 |
| 8 | 0.036 | +1.5995 | -1.59951 | -1.86166 | +0.05145 | +0.05525 | 5.00e-08 | 2.03e-01 | +1.00000 | +0.98818 |
| 27 | 0.039 | +1.5792 | -1.57915 | -1.85435 | +0.05924 | +0.05918 | 4.83e-08 | 1.78e-01 | +1.00000 | +0.99147 |
| 28 | 0.039 | +1.5718 | -1.57179 | -1.85170 | +0.03584 | +0.03848 | 4.66e-08 | 1.93e-01 | +1.00000 | +0.98957 |
| 14 | 0.052 | +1.4465 | -1.44654 | -1.80594 | +0.03972 | +0.04248 | 5.14e-08 | 2.62e-01 | +1.00000 | +0.98534 |
| 26 | 0.052 | +1.4461 | -1.44613 | -1.80579 | +0.04932 | +0.05166 | 5.41e-08 | 2.49e-01 | +1.00000 | +0.98360 |
| 10 | 0.069 | +1.2962 | -1.29616 | -1.74933 | +0.04755 | +0.05594 | 5.22e-08 | 1.95e-01 | +1.00000 | +0.98338 |
| 25 | 0.096 | +1.0812 | -1.08122 | -1.66481 | +0.04527 | +0.05010 | 4.93e-08 | 2.12e-01 | +1.00000 | +0.98381 |
| 15 | 0.110 | +0.9814 | -0.98139 | -1.62391 | +0.02310 | +0.03142 | 4.79e-08 | 2.07e-01 | +1.00000 | +0.98026 |
| 2 | 0.118 | +0.9259 | -0.92589 | -1.60066 | +0.02194 | +0.03311 | 4.95e-08 | 2.58e-01 | +1.00000 | +0.96906 |
| 29 | 0.136 | +0.8103 | -0.81032 | -1.55097 | +0.03732 | +0.04963 | 4.52e-08 | 2.26e-01 | +1.00000 | +0.97424 |
| 9 | 0.142 | +0.7687 | -0.76869 | -1.53261 | +0.03546 | +0.05236 | 3.55e-08 | 2.47e-01 | +1.00000 | +0.97217 |
| 18 | 0.153 | +0.7067 | -0.70670 | -1.50479 | +0.02401 | +0.04666 | 4.46e-08 | 3.49e-01 | +1.00000 | +0.94124 |
| 11 | 0.208 | +0.4252 | -0.42522 | -1.36985 | +0.01552 | +0.03450 | 4.57e-08 | 3.88e-01 | +1.00000 | +0.92294 |
| 0 | 0.259 | +0.2247 | -0.22471 | -1.26246 | +0.01083 | +0.04207 | 5.02e-08 | 5.07e-01 | +1.00000 | +0.87901 |
| 6 | 0.266 | +0.1981 | -0.19812 | -1.24729 | +0.00876 | +0.04271 | 3.76e-08 | 5.53e-01 | +1.00000 | +0.84363 |
| 23 | 0.395 | -0.1361 | +0.13608 | -1.02790 | -0.00327 | +0.01929 | 4.32e-08 | 6.12e-01 | +1.00000 | +0.79895 |
| 3 | 0.493 | -0.2879 | +0.28791 | -0.89720 | -0.01480 | +0.03039 | 4.58e-08 | 8.69e-01 | +1.00000 | +0.61412 |
| 19 | 0.600 | -0.3904 | +0.39038 | -0.78158 | -0.01236 | +0.01871 | 5.25e-08 | 9.46e-01 | +1.00000 | +0.54155 |
| 12 | 0.802 | -0.4807 | +0.48066 | -0.61562 | -0.01924 | +0.01823 | 3.57e-08 | 8.52e-01 | +1.00000 | +0.58313 |
| 22 | 1.061 | -0.4987 | +0.49872 | -0.47090 | -0.01490 | +0.01109 | 4.04e-08 | 1.05e+00 | +1.00000 | +0.35983 |
| 1 | 1.402 | -0.4627 | +0.46267 | -0.34664 | -0.01787 | +0.01034 | 3.09e-08 | 1.06e+00 | +1.00000 | +0.24554 |
| 16 | 1.609 | -0.4310 | +0.43099 | -0.29383 | -0.01448 | +0.00857 | 2.68e-08 | 1.10e+00 | +1.00000 | +0.17601 |
| 7 | 2.198 | -0.3421 | +0.34208 | -0.19556 | -0.01036 | +0.00506 | 2.26e-08 | 1.11e+00 | +1.00000 | +0.13684 |
| 21 | 2.270 | -0.3323 | +0.33227 | -0.18699 | -0.01202 | +0.00558 | 2.12e-08 | 1.12e+00 | +1.00000 | +0.09776 |
| 4 | 2.311 | -0.3270 | +0.32697 | -0.18248 | -0.01548 | +0.00631 | 2.04e-08 | 1.20e+00 | +1.00000 | -0.12887 |
| 13 | 2.351 | -0.3217 | +0.32173 | -0.17811 | -0.01553 | +0.00616 | 1.46e-08 | 1.17e+00 | +1.00000 | -0.06231 |
| 20 | 2.565 | -0.2955 | +0.29553 | -0.15736 | -0.01164 | +0.00497 | 1.59e-08 | 1.11e+00 | +1.00000 | +0.11068 |

> `a_*` is the mapping-level radial response rhat^T (dPhi_t/dy_i) rhat; exactly -lambda_parallel(e_i). `b_*` is the pose-level translation response rhat^T (dt*/dy_i) rhat; H_rob^{-1} mixes directions, so b is not forced to flip sign by lambda_parallel alone (Theorem 3.3(d)). The sign-flip column in the summary table counts e>c^2/3 observations.

## FD step-size check (one boundary observation)

| h | relerr exact | relerr naive | cos exact | cos naive |
|---|---|---|---|---|
| 1e-04 | 4.04e-06 | 1.05e+00 | +1.00000000 | +0.35983009 |
| 1e-05 | 4.04e-08 | 1.05e+00 | +1.00000000 | +0.35982828 |
| 1e-06 | 1.41e-09 | 1.05e+00 | +1.00000000 | +0.35982826 |
| 1e-07 | 1.12e-08 | 1.05e+00 | +1.00000000 | +0.35982826 |

## Hessian / lambda checks

* lambda_min(H_rob) on representative instance: +22.5623
* ||H_rob - H_fd||_F / ||H_fd||_F: 3.98e-07
* ||H_doc - H_fd||_F / ||H_fd||_F (missing -lambda): 1.08e-03  (lambda_rob=-0.096)
* ||H_naive - H_fd||_F / ||H_fd||_F (frozen weights): 2.80e-01
* max |lambda_i - 4 u_i^T W_i r_i|: 2.78e-17
* max |lambda_i - 4 e_i|: 1.89e-01
* max |a_exact + lambda_parallel|: 8.88e-16

## Notes

* FD differentiates the *polished fixed-point branch*, not a finite GNC unroll; this is the correct oracle for the IFT derivative. The GNC output itself has ||Phi||~1e-7 because it uses 5 final IRLS steps.
* The naive Hessian keeps the exact sphere term, so exact-vs-naive isolates the weight-sensitivity (rho'') error, not the constraint error. The `no-lambda` row separately isolates the constraint error.
* `relerr` can be large for gross outliers where ||dz*/dy_i|| is tiny; the cosine column and absolute errors are the meaningful diagnostics there.
