# Robustness + timing baseline study (measured in this worktree)

All numbers in this file were **measured by `experiments/code/run_robustness.py` in one run on 2026-11-27 20:56:19** (host `ls`, python 3.10.22, numpy 2.2.6), instance seed 20261127, tau=0.01, FD h=0.0001, MC budget n=200 x 3 seeds. No number is copied from `main_table.md` / `ablation.md`.

Runtime of the whole script: **62.4 s** (foreground, no srun/sbatch).


## (a) Enumeration robustness vs `n_starts`

theta grid: -0.100, +0.000, +0.020, +0.100 (4 points); reference = n_starts=1024. `rel rich` = Laplace gradient rel. error vs a Richardson-extrapolated central FD of the *same* smoothed map (h=1e-4 and 5e-5, O(h^4)); `rel cen` = vs plain central FD h=1e-4 (the `run_main.py` protocol); `rel same` = vs central FD built with the *same* n_starts (self-consistent).

| n_starts | median K | #theta K != ref | missed roots | extra roots | #theta FD set change | rel rich (median) | rel rich (max) | rel cen (median) | rel same (median) | enum ms (median) |
|-|-|-|-|-|-|-|-|-|-|-|
| 32 | 2.5 | 1 | 1 | 0 | 0 | 4.59e-09 | 1.76e-07 | 6.04e-05 | 6.04e-05 | 15.5 |
| 64 | 2.5 | 0 | 0 | 0 | 0 | 4.59e-09 | 1.76e-07 | 6.04e-05 | 6.04e-05 | 20.5 |

K (number of local minima) per theta -- the enumeration result itself:

| theta | 32 | 64 |
|-|-|-|-|
| -0.100 | 2 | 2 |
| +0.000 | 4 | 4 |
| +0.020 | 3 | 3 |
| +0.100 | 1 | 2 |

### (a3) Missed-root probability over *independent* start draws

The sweep above is nested (the n_starts=32 draw is a prefix of the 1024 draw: verified, so `extra` is structurally 0). To measure a real miss probability we draw 10 independent start sets per (n_starts, theta) at theta in -0.0200, -0.0013, +0.0000, +0.0013, +0.0200 and compare against n_starts=1024. `pose/dpose rel` is the output error of the Laplace layer when a minimum is missed, relative to the full reference.

| n_starts | runs | miss rate | miss rate at |theta|=1.3e-3 | pose rel (max, missed runs) | dpose rel (max, missed runs) |
|-|-|-|-|-|-|
| 16 | 50 | 0.10 | 0.15 | 2.83e-01 | 9.71e-01 |
| 32 | 50 | 0.00 | 0.00 | n/a | n/a |
| 64 | 50 | 0.00 | 0.00 | n/a | n/a |

### (a2) Secondary-mode fold near |theta| ~ 1.3e-3 (fine scan)

n_starts=512; coarse step 2e-4, fine step 2e-05 around the fold. Measured: the mode count changes at theta≈0.0014 (K in [3, 4]); the fragile mode (smallest 3x3 Hessian determinant) has detH falling to 3.35e-01 and its Laplace weight reaches 0.122 right before it disappears. |dpose| peaks at 3529 (median 379) and the pose map jumps 3.09e-01 across one 2e-05 step (effective FD slope up to 15445, vs |dpose|≈379 away from the fold).

| theta | K | w (sorted desc) | detH min | w(fragile) | |dpose| | pose jump | FD slope across |
|-|-|-|-|-|-|-|-|
| +0.00100 | 4 | 0.591, 0.299, 0.091, 0.020 | 5.46e-01 | 0.091 | 481 | 0.00e+00 | 0 |
| +0.00102 | 4 | 0.593, 0.295, 0.093, 0.019 | 5.29e-01 | 0.093 | 484 | 9.64e-03 | 482 |
| +0.00104 | 4 | 0.594, 0.292, 0.095, 0.019 | 5.11e-01 | 0.095 | 488 | 9.71e-03 | 486 |
| +0.00106 | 4 | 0.595, 0.289, 0.097, 0.019 | 4.93e-01 | 0.097 | 493 | 9.80e-03 | 490 |
| +0.00108 | 4 | 0.596, 0.285, 0.100, 0.018 | 4.74e-01 | 0.100 | 500 | 9.92e-03 | 496 |
| +0.00110 | 4 | 0.597, 0.282, 0.103, 0.018 | 4.54e-01 | 0.103 | 509 | 1.01e-02 | 504 |
| +0.00112 | 4 | 0.598, 0.278, 0.106, 0.018 | 4.33e-01 | 0.106 | 521 | 1.03e-02 | 515 |
| +0.00114 | 4 | 0.599, 0.275, 0.109, 0.017 | 4.11e-01 | 0.109 | 538 | 1.06e-02 | 529 |
| +0.00116 | 4 | 0.599, 0.271, 0.113, 0.017 | 3.87e-01 | 0.113 | 561 | 1.10e-02 | 549 |
| +0.00118 | 4 | 0.599, 0.268, 0.117, 0.017 | 3.62e-01 | 0.117 | 594 | 1.15e-02 | 576 |
| +0.00120 | 4 | 0.598, 0.264, 0.122, 0.016 | 3.35e-01 | 0.122 | 643 | 1.23e-02 | 617 |
| +0.00122 | 4 | 0.597, 0.259, 0.128, 0.016 | 3.06e-01 | 0.128 | 719 | 1.36e-02 | 678 |
| +0.00124 | 4 | 0.594, 0.255, 0.135, 0.016 | 2.74e-01 | 0.135 | 846 | 1.55e-02 | 776 |
| +0.00126 | 4 | 0.590, 0.250, 0.145, 0.015 | 2.36e-01 | 0.145 | 1083 | 1.90e-02 | 951 |
| +0.00128 | 4 | 0.583, 0.243, 0.159, 0.015 | 1.92e-01 | 0.159 | 1619 | 2.62e-02 | 1308 |
| +0.00130 | 4 | 0.566, 0.233, 0.187, 0.014 | 1.33e-01 | 0.187 | 3529 | 4.65e-02 | 2324 |
| +0.00132 | 3 | 0.699, 0.284, 0.017 | 1.46e+00 | 0.017 | 413 | 3.09e-01 | 15445 |
| +0.00134 | 3 | 0.702, 0.281, 0.017 | 1.47e+00 | 0.017 | 411 | 8.24e-03 | 412 |
| +0.00136 | 3 | 0.705, 0.279, 0.017 | 1.47e+00 | 0.017 | 408 | 8.19e-03 | 410 |
| +0.00138 | 3 | 0.708, 0.276, 0.016 | 1.47e+00 | 0.016 | 406 | 8.14e-03 | 407 |
| +0.00140 | 3 | 0.711, 0.273, 0.016 | 1.48e+00 | 0.016 | 404 | 8.10e-03 | 405 |
| +0.00142 | 3 | 0.713, 0.271, 0.016 | 1.48e+00 | 0.016 | 401 | 8.05e-03 | 402 |
| +0.00144 | 3 | 0.716, 0.268, 0.016 | 1.49e+00 | 0.016 | 399 | 8.00e-03 | 400 |
| +0.00146 | 3 | 0.719, 0.265, 0.015 | 1.49e+00 | 0.015 | 396 | 7.95e-03 | 398 |
| +0.00148 | 3 | 0.722, 0.263, 0.015 | 1.50e+00 | 0.015 | 394 | 7.90e-03 | 395 |
| +0.00150 | 3 | 0.725, 0.260, 0.015 | 1.50e+00 | 0.015 | 392 | 7.86e-03 | 393 |
| +0.00152 | 3 | 0.727, 0.258, 0.015 | 1.51e+00 | 0.015 | 389 | 7.81e-03 | 390 |
| +0.00154 | 3 | 0.730, 0.255, 0.015 | 1.51e+00 | 0.015 | 387 | 7.76e-03 | 388 |
| +0.00156 | 3 | 0.733, 0.253, 0.014 | 1.51e+00 | 0.014 | 384 | 7.71e-03 | 385 |
| +0.00158 | 3 | 0.736, 0.250, 0.014 | 1.52e+00 | 0.014 | 382 | 7.66e-03 | 383 |

## (b) Timing variance (20 repetitions, n_starts=128)

`enum` = `Instance.find_modes`; `deriv` = `run_main.pose_and_derivs` (poses, K ift solves, dpose Jacobians, dJ, numeric dlogdet); `fwd` and `layer` measured end-to-end; `ift` = K calls to `qpep_core.ift_dq` (H assembly + 3x3 solve), `hess3` = K assemblies, `solve3` = K pure 3x3 solves; `backward` = weight gradient + pose mixing only. All in ms per layer call; brackets are [p5, p95].

| theta | K | enum | deriv | fwd (measured) | ift x K | hess3 x K | solve3 x K | backward | layer total |
|-|-|-|-|-|-|-|-|-|-|
| -0.10 | 2 | 14.26 [14.24, 14.28] | 3.48 [3.45, 3.50] | 19.07 [17.91, 20.23] | 0.49 [0.49, 0.50] | 0.27 [0.27, 0.27] | 0.02 [0.02, 0.02] | 0.03 [0.03, 0.03] | 17.70 [17.58, 17.82] |
| +0.00 | 4 | 54.74 [54.13, 55.35] | 6.76 [6.57, 6.96] | 62.08 [61.28, 62.87] | 0.98 [0.97, 0.99] | 0.52 [0.52, 0.52] | 0.03 [0.03, 0.03] | 0.04 [0.04, 0.04] | 61.89 [61.65, 62.14] |
| +0.10 | 2 | 11.91 [11.76, 12.07] | 3.49 [3.44, 3.54] | 15.68 [15.64, 15.72] | 0.57 [0.56, 0.58] | 0.28 [0.28, 0.28] | 0.02 [0.02, 0.02] | 0.04 [0.04, 0.04] | 15.91 [15.81, 16.02] |

Relative spread (std/mean, measured over the 20 repetitions):

| theta | enum | fwd | backward | layer total |
|-|-|-|-|-|
| -0.10 | 0.2% | 9.6% | 0.3% | 1.1% |
| +0.00 | 1.8% | 2.0% | 4.5% | 0.6% |
| +0.10 | 2.1% | 0.4% | 1.5% | 1.0% |

## (c) Baseline stress: hard argmin vs multi-modal Laplace vs mode-level MC

Random symmetric dM with unit Frobenius norm, added as M(theta)+eps*dM; 2 perturbation seeds for eps>0. MC budget fixed at n=200, 3 independent seeds. `rel rich` for MC = per-seed rel. error vs the Richardson FD of the Laplace map (same target); `mean est` = rel. error of the 20-seed mean. `switch` = the hard argmin changes branch between theta +- h (then hard FD is undefined).

| eps | theta | K (seeds) | hard switch rate | lap rel rich | lap rel cen | hard rel cen | MC rel rich (median) | MC mean-est rel | MC pose std | MC dpose std |
|-|-|-|-|-|-|-|-|-|-|-|
| 0 | +0.00 | 4 | 1.00 | 1.76e-07 | 4.34e-04 | n/a | 9.35e-02 | 2.01e-02 | 1.78e-01 | 4.45e+01 |
| 0.001 | +0.00 | 4 | 1.00 | 2.55e-07 | 4.39e-04 | n/a | 9.88e-02 | 1.90e-02 | 1.85e-01 | 4.67e+01 |
| 0 | +0.02 | 3 | 0.00 | 9.18e-09 | 1.20e-04 | 1.20e-04 | 6.96e-06 | 6.96e-06 | 1.60e-16 | 4.44e-16 |
| 0.001 | +0.02 | 3 | 0.00 | 8.89e-09 | 1.18e-04 | 1.18e-04 | 7.04e-06 | 7.04e-06 | 6.94e-18 | 3.68e-15 |

### (c-ii) tau sweep

| theta | eps | tau | K | w_max | lap rel rich | lap rel cen | hard switch | MC rel rich (median) |
|-|-|-|-|-|-|-|-|-|
| +0.00 | 0 | 0.010 | 4 | 0.459 | 1.76e-07 | 4.34e-04 | True | 9.35e-02 |
| +0.00 | 0 | 0.050 | 4 | 0.290 | 1.07e-06 | 1.20e-03 | True | 8.95e-02 |
| +0.00 | 0.001 | 0.010 | 4 | 0.460 | 3.01e-07 | 4.43e-04 | True | 9.01e-02 |
| +0.00 | 0.001 | 0.050 | 4 | 0.295 | 1.83e-06 | 1.61e-03 | True | 9.28e-02 |
| +0.00 | 0.001 | 0.010 | 4 | 0.461 | 2.09e-07 | 4.35e-04 | True | 1.08e-01 |
| +0.00 | 0.001 | 0.050 | 4 | 0.293 | 1.28e-06 | 1.32e-03 | True | 7.93e-02 |
| +0.02 | 0 | 0.010 | 3 | 1.000 | 9.18e-09 | 1.20e-04 | False | 6.96e-06 |
| +0.02 | 0 | 0.050 | 3 | 0.956 | 1.23e-08 | 1.50e-04 | False | 3.03e-02 |
| +0.02 | 0.001 | 0.010 | 3 | 1.000 | 8.39e-09 | 1.16e-04 | False | 7.01e-06 |
| +0.02 | 0.001 | 0.050 | 3 | 0.956 | 1.16e-08 | 1.47e-04 | False | 3.00e-02 |
| +0.02 | 0.001 | 0.010 | 3 | 1.000 | 9.38e-09 | 1.21e-04 | False | 7.06e-06 |
| +0.02 | 0.001 | 0.050 | 3 | 0.956 | 1.22e-08 | 1.50e-04 | False | 3.01e-02 |

### (c-iii) Crossing-jump test (does the map step at theta=0?)

`|pose(+d)-pose(-d)|` at d=1e-3, 1e-4, 1e-5. A step map keeps the jump constant as d->0; a smooth map's jump decays ~linearly.

| eps | delta | hard jump | hard FD slope | Laplace jump | Laplace FD slope |
|-|-|-|-|-|-|
| 0 | 1e-03 | 2.86e+00 | 1432 | 9.73e-01 | 486 |
| 0 | 1e-04 | 2.86e+00 | 14319 | 1.00e-01 | 501 |
| 0 | 1e-05 | 2.86e+00 | 143191 | 1.00e-02 | 502 |
| 0.001 | 1e-03 | 2.86e+00 | 1432 | 9.91e-01 | 495 |
| 0.001 | 1e-04 | 2.86e+00 | 14319 | 1.01e-01 | 505 |
| 0.001 | 1e-05 | 5.33e-04 | 27 | 1.01e-02 | 505 |
| 0.001 | 1e-03 | 2.86e+00 | 1432 | 9.77e-01 | 488 |
| 0.001 | 1e-04 | 2.86e+00 | 14319 | 1.00e-01 | 502 |
| 0.001 | 1e-05 | 5.33e-04 | 27 | 1.00e-02 | 502 |

## (d) True batched forward throughput

Batched damped Newton, S=128 starts, batch B=1/8/64 (64 distinct M(theta), same starts for every instance). Batched algebra checked against `qpep_core` on single samples: max abs diff J=2.8e-17, grad=1.1e-16, H=6.7e-16, projgrad=1.1e-16; K equal to the loop for the 4 validation instances: True, max |dJ|=4.4e-16. Classify is a per-instance Python loop (K varies), included in `total`.

| B | batched newton (ms) | classify (ms) | batched total (ms) | batched samples/s | loop total (ms) | loop samples/s | speedup | newton iters |
|-|-|-|-|-|-|-|-|-|
| 1 | 31.9 | 0.7 | 32.6 | 30.7 | 14.9 | 67.2 | 0.46x | 14 |
| 8 | 142.1 | 4.8 | 146.9 | 54.5 | 113.9 | 70.2 | 0.78x | 16 |
| 64 | 1029.1 | 39.1 | 1068.2 | 59.9 | 1386.0 | 46.2 | 1.30x | 50 |

## Conclusions (measured here; negative results included)

**Start-count / missed roots.** On this instance enumeration is not the bottleneck: with the nested start stream, n_starts>=32 always reproduces the n_starts=1024 mode set at all 4 test thetas (K-ne-ref count 1, missed 1); over 10 *independent* start draws the miss rate is 0.10 at n_starts=16, 0.00 at 32 and 0.00 at 64 (always the fragile, smallest-detH mode), and a miss moves the pose by up to 28.3% and the gradient by up to 97% of the full reference. So 32 starts is enough here, 16 is not; the risk is a discrete, visible output error, not a silent one. **Gradient accuracy.** The 4.34e-4 crossing error reported by `run_main.py` is FD truncation of the *reference*, not an error of the analytic gradient: at theta=0 the error vs a Richardson FD is 1.8e-07 while the error vs the h=1e-4 central FD is 4.3e-04 (and scales as h^2), because the softmax transition scale is tau/slope=5e-3. **Hard argmin.** Where its central FD is defined, hard argmin is as self-consistent as Laplace (median rel. err 1.2e-04 over the 50% of perturbed cases where the argmin does not switch), i.e. hard is *better* on that one metric; but it switches branch inside +-h in 50% of the exact-crossing cases and 50% of the perturbed ones, and it is a step map: its pose jump stays 2.86 as delta->0 whereas the Laplace jump decays linearly (FD slope 502, i.e. the true derivative). So the hard baseline wins only a metric that ignores the discontinuity it is supposed to fix. **MC.** Mode-level MC is unbiased for the same smoothed target but with a fixed budget of n=200 x 3 seeds its per-seed gradient rel. error is 0.05 (20-seed mean 7.7e-03) and its pose std is 0.09, versus 9.3e-08 for analytic Laplace in the same regime; it does not buy robustness to the fold either, because it samples the same detH-weighted posterior. It is cheaper in memory/linalg (no K Hessians) but not more accurate at this budget. **The real fragility is a mode fold, not a missed root.** A secondary minimum exists only for |theta| < ~0.0014; as detH->0 its weight spikes to 0.12, |dpose| spikes to 3529 (vs 379 away from the fold) and the pose map steps by 0.31 (a step that stays constant under grid refinement, i.e. a genuine discontinuity of the Laplace map, not an FD artifact). Higher n_starts or smaller tau cannot remove it. **Timing.** At theta=0 a full layer call takes 61.9 ms median (p95 62.1), of which mode enumeration is 88%, the K 3x3 implicit solves are 1.6% (pure 3x3 solves only 0.0%), the rest of the derivatives 10.9% and the analytic backward (weight gradient + mixing) 0.1%; at theta=-0.1 the whole call is 17.7 ms. **Batching.** A true batched multi-start Newton gives 1.30x over the loop at B=64 (60 vs 46 samples/s), 0.78x at B=8, and is *slower* at B=1 (0.46x) because the Python/linalg overhead is not amortised; convergence heterogeneity (slowest start in the batch) and the per-instance classify loop cap the gain. A GPU/custom-kernel port is needed for a real throughput win, not just numpy flattening.

Caveat for the paper: 'smooth at the crossing' holds at theta=0 and inside |theta| < ~1.3e-3, but the multi-modal Laplace map is itself discontinuous where a minimum is born or dies (section a2), and its derivative is unbounded there. This is a property of the Laplace approximation with mode enumeration, not of the implicit differentiation formula.

