# Detector-noise robustness of the hard / Laplace / MC layer on real LINEMOD frames

> **What this is.** A controlled *simulated* detector-noise study on the same 15 real eggbox/glue frames as `run_pose_error.py`, using the **fixed clean RANSAC planar-face subset**. Gaussian pixel noise and gross 2-D outliers are injected on the bearings; the RANSAC subset is never recomputed. **What this is not.** It is **not** a trained detector (GPU blocked, no appearance, no descriptors) and it is **not** a LINEMOD accuracy benchmark: the correspondences are associated through the GT pose, so the GT pose is favoured by construction and the **absolute errors are optimistic**. The value is the *relative* degradation and the hard/Laplace/MC comparison under the same perturbation.

## Headline

- **Clean baseline** (sigma=0, f=0, 150 draws): hard rotation error 2.06 [1.47, 3.49] deg, Laplace 44.24 [33.90, 51.60] deg, MC 44.29 [33.64, 51.39] deg; ADD-S hard 3.7 [3.0, 4.2] mm, Laplace 14.2 [10.4, 20.2] mm, MC 14.2 [10.4, 19.2] mm. GT-like mode is the hard argmin in 93% of frames (93% of draws).
- **Gaussian pixel noise to 5 px** costs the hard estimator 10.0x its clean rotation error (median 20.49 deg) and 44.5x its clean ADD-S (median 164.2 mm); the degradation is steep and super-linear. Laplace median 41.83 deg (0.9x), MC 42.19 deg; K_phys stays 2 in 100% of draws; the hard argmin is still the mode *closest* to GT in 100% of frames, so the loss is not a flip switch but the GT-side minimum itself sliding ~20 deg / 220 mm along the flat valley; hard failures (>10 deg) 66% of draws.
- **Up to 1 px there is no measurable degradation** (and possibly a tiny improvement): hard rotation error 2.03 deg at 0.5 px and 1.98 deg at 1 px vs 2.06 deg clean; hard ADD-S 2.9 / 2.7 mm vs 3.7 mm clean. Averaging the pixel noise reduces the clean nearest-vertex/model bias slightly. The collapse happens between 2 and 5 px, not gradually from zero.
- **Gross outliers break the plain L2 bearing cost almost immediately.** At f=0.05 a random image pixel is ~0.1 m away from the corresponding model point, i.e. one outlier squared residual (~1e-2 m^2) outweighs the whole inlier set (~1e-3 m^2). No physical mode survives in 95% of draws (median K_phys=0); the fallback hard estimate is 152 deg (100% failures) and ADD-S 889 mm. The estimate is already broken at the first diagnostic f value: at f=0.005 (2-3 gross outliers out of 558) a physical mode survives in 93% of draws but the hard error is 71 deg with 87% failures, so the empirical breakdown point is O(1) outlier; K_phys=0 only *dominates* at f=0.05 (95% of draws). The outlier arm therefore measures the **breakdown** of an un-robustified layer, not graceful degradation; a trained front-end with outlier rejection (or a robust kernel) is doing essential work.
- **Laplace is biased toward the flip mode with or without noise.** Its median error tracks the ~half-flip compromise (flip modes at median 88.4 deg) and the flip weight stays near 0.5 (w_other median 0.482 clean, 0.481 at 5 px): Gaussian noise moves both estimators but does not remove the near-tie. Hard argmin is the more accurate estimator here (by construction) but is discontinuous; MC estimates the same biased mean as Laplace and adds seed noise (between-seed SD of the aggregate rotation 0.419 deg at 5 px).

**Framing.** Simulated detector noise, not a trained detector. The clean correspondences favour GT by construction, so the absolute errors are optimistic; the experiment supports statements about *relative* degradation and about the hard/Laplace/MC trade, not about absolute accuracy on LINEMOD.

## 0. Protocol

- **Frames / subset.** The 15 frames of `run_two_mode.py` Part A (8 eggbox `000010` + 7 glue `000011`), each with the clean RANSAC dominant-plane inlier subset stored once. RANSAC is **not** re-run on noisy data; every condition uses exactly the same 3-D points and the same subset indices. Frame-set check: two_mode.json/part_a: equal (15/15); pose_error.json/frames: equal (15/15).
- **Gaussian arm.** $\sigma\in\{0, 0.5, 1, 2, 5\}$ px, $u'=u+\mathcal{N}(0,\sigma^2)$, $v'=v+\mathcal{N}(0,\sigma^2)$, $U'=\mathrm{normalize}(K^{-1}[u',v',1])$. 10 independent noise seeds per condition.
- **Outlier arm.** A fraction $f\in\{0, 0.05, 0.1, 0.2\}$ of the *accepted* bearings is replaced by the bearing of a random pixel uniformly drawn in the 640x480 image (a random direction in the image plane = a wrong 2-D measurement for the fixed 3-D model point), at $\sigma=0$. Extra $f\in\{0.005, 0.01\}$ runs only localise the breakdown point.
- **Layer.** Same reduced bearing cost and mode enumeration as `run_two_mode.py` ($\tau=0.01$, `n_starts=384`). Physical modes have all transformed points at $z>0.05$ m. Hard = lowest-$J$ physical mode; Laplace = $(\mathrm{Proj}_{SO(3)}(\sum_k w_k R_k), \sum_k w_k t_k)$ with $w_k\propto\exp(-J_k/\tau-\tfrac12\log\det H_k)$; MC = 400 mode samples from $w$, chordal-mean pose, 10 MC seeds per (frame, noise seed). **Fallback:** if the physical filter finds no mode (L2 breakdown), the layer runs on all minima and the draw is flagged; those numbers are breakdown indicators, not accuracy.
- **Metrics.** Rotation error = geodesic angle to `cam_R_m2c` (deg). ADD-S = mean over 1000 fixed mesh query vertices of the distance to the nearest vertex of the **full** GT-transformed mesh (dense KD-tree, the `pose_error.py` protocol with the same vertex subsample; the 3-subsample spread there was $\le$ 1.9 mm and is part of the subsample noise, so one subsample is used here). $\Delta J=J_{\rm runner\text{-}up}-J_{\rm hard}$, `w_other` = posterior weight of the strongest non-hard mode.
- **Aggregation.** Per condition, per frame: mean over the 10 noise seeds; mean/median with 95% percentile bootstrap CIs over the 15 frames (1000 resamples). For MC: frame CI, MC seed-level CI (bootstrap over the 10 per-seed aggregates), combined frame+MC CI, and between-seed SD. Hard-failure rate = fraction of draws with hard rotation error >10 deg; GT-like fraction = fraction of frames where the GT-closest mode is the hard argmin in the majority of seeds.

## 1. Gaussian pixel noise

### 1.1 Rotation error (deg)

| sigma (px) | K_phys med [>=1] | Delta J med | w_other med | hard rot med [95% CI] | Laplace rot med [95% CI] | MC rot med [95% CI] | hard fail | GT-like frames |
|-|-|-|-|-|-|-|-|-|
| 0 | 2 [100%] | 1.49e-03 | 0.482 | 2.06 [1.47, 3.49] | 44.24 [33.90, 51.60] | 44.29 [33.64, 51.39] | 7% | 93% |
| 0.5 | 2 [100%] | 1.48e-03 | 0.482 | 2.03 [1.40, 3.54] | 43.64 [33.88, 46.67] | 43.50 [32.94, 50.40] | 7% | 93% |
| 1 | 2 [100%] | 1.45e-03 | 0.480 | 1.98 [1.29, 3.30] | 43.66 [33.89, 46.52] | 43.11 [33.82, 51.11] | 5% | 93% |
| 2 | 2 [100%] | 1.31e-03 | 0.476 | 2.74 [1.91, 10.84] | 42.51 [33.62, 46.47] | 42.34 [32.39, 46.21] | 23% | 93% |
| 5 | 2 [100%] | 1.13e-03 | 0.481 | 20.49 [7.54, 30.86] | 41.83 [31.87, 46.56] | 42.19 [32.06, 46.97] | 66% | 100% |

### 1.2 ADD-S (mm)

| sigma (px) | hard ADD-S med [95% CI] | Laplace ADD-S med [95% CI] | MC ADD-S med [95% CI] | hard ADD-S mean | Laplace ADD-S mean |
|-|-|-|-|-|-|
| 0 | 3.7 [3.0, 4.2] | 14.2 [10.4, 20.2] | 14.2 [10.4, 19.2] | 6.0 | 15.4 |
| 0.5 | 2.9 [2.5, 4.8] | 14.4 [10.5, 19.4] | 14.3 [9.4, 19.0] | 5.1 | 14.9 |
| 1 | 2.7 [1.7, 3.2] | 12.8 [10.9, 17.8] | 12.7 [10.9, 19.7] | 4.6 | 16.3 |
| 2 | 14.1 [7.9, 21.5] | 21.2 [17.1, 25.0] | 21.2 [17.1, 46.7] | 27.3 | 39.3 |
| 5 | 164.2 [98.6, 245.0] | 168.4 [102.0, 248.5] | 168.4 [102.1, 180.0] | 188.2 | 198.9 |

### 1.3 Translation error (mm), for interpreting ADD-S

The planar subset leaves a soft direction in which the optimum can slide a long way for little cost, so the eliminated translation (not only the rotation) degrades sharply with pixel noise; ADD-S is dominated by that translation shift.

| sigma (px) | hard trans med | Laplace trans med | MC trans med |
|-|-|-|-|
| 0 | 6.2 | 20.2 | 20.2 |
| 0.5 | 5.3 | 20.3 | 20.2 |
| 1 | 5.7 | 18.9 | 18.9 |
| 2 | 32.1 | 34.9 | 34.9 |
| 5 | 220.4 | 220.3 | 220.3 |

### 1.4 Relative degradation (median vs clean)

| sigma (px) | hard rot / clean | Laplace rot / clean | MC rot / clean | hard ADD-S / clean | Laplace ADD-S / clean | MC ADD-S / clean |
|-|-|-|-|-|-|-|
| 0 | 1.00x | 1.00x | 1.00x | 1.00x | 1.00x | 1.00x |
| 0.5 | 0.98x | 0.99x | 0.98x | 0.78x | 1.01x | 1.01x |
| 1 | 0.96x | 0.99x | 0.97x | 0.72x | 0.90x | 0.89x |
| 2 | 1.33x | 0.96x | 0.96x | 3.82x | 1.49x | 1.49x |
| 5 | 9.95x | 0.95x | 0.95x | 44.52x | 11.85x | 11.86x |

## 2. Gross 2-D outliers

A fraction of the accepted bearings is replaced by random image pixels. Because the cost is an unnormalised L2 bearing cost, the first few gross outliers dominate the inlier residuals; the table reports the breakdown. `fallback` = fraction of draws where the physical filter found **no** mode and the layer ran on all minima (flagged). The f=0 row is the shared clean baseline.

### 2.1 Required grid

| f | K_phys med [>=1] | fallback | Delta J med | w_other med | hard rot med [95% CI] | Laplace rot med [95% CI] | MC rot med [95% CI] | hard fail | GT-like frames |
|-|-|-|-|-|-|-|-|-|-|
| 0 | 2 [100%] | 0% | 1.49e-03 | 0.482 | 2.06 [1.47, 3.49] | 44.24 [33.90, 51.60] | 44.29 [33.64, 51.39] | 7% | 93% |
| 0.05 | 0 [5%] | 95% | 2.60e-04 | 0.316 | 151.70 [129.07, 165.70] | 132.98 [124.12, 153.52] | 131.01 [108.58, 134.86] | 100% | 20% |
| 0.1 | 0 [0%] | 100% | 2.44e-04 | 0.320 | 150.40 [141.10, 158.25] | 134.50 [122.59, 144.59] | 125.34 [116.01, 126.79] | 100% | 13% |
| 0.2 | 0 [0%] | 100% | 2.69e-04 | 0.417 | 145.47 [127.71, 156.89] | 128.43 [122.00, 130.33] | 123.51 [120.14, 129.43] | 100% | 20% |

| f | hard ADD-S med [95% CI] | Laplace ADD-S med [95% CI] | MC ADD-S med [95% CI] | hard ADD-S mean |
|-|-|-|-|-|
| 0 | 3.7 [3.0, 4.2] | 14.2 [10.4, 20.2] | 14.2 [10.4, 19.2] | 6.0 |
| 0.05 | 888.6 [801.5, 992.8] | 838.4 [749.1, 979.9] | 837.7 [749.4, 991.4] | 862.6 |
| 0.1 | 886.5 [722.1, 974.5] | 858.0 [708.3, 980.0] | 857.6 [761.5, 970.7] | 856.5 |
| 0.2 | 872.3 [716.7, 996.8] | 857.7 [723.2, 996.2] | 857.2 [763.3, 966.3] | 849.4 |

Under the all-minima fallback the `GT-like is hard argmin` column is close to chance (1/K_all ~ 20-25%), so it carries no accuracy information there; only `K_phys` and the failure rate identify the breakdown. The diagnostic f values in section 2.2 locate the onset.

### 2.2 Breakdown localisation (diagnostic f values, not part of the required grid)

| f | K_phys med [>=1] | fallback | hard rot med | hard ADD-S med | GT-like frames | hard fail |
|-|-|-|-|-|-|-|
| 0.005 | 2 [93%] | 7% | 70.7 | 405.1 | 73% | 87% |
| 0.01 | 2 [80%] | 20% | 77.4 | 598.3 | 60% | 98% |

The breakdown point is O(1) outlier. At f=0.005 (2-3 replaced bearings out of 558) a physical mode still survives in most draws, but the hard pose is already ~70 deg / 400 mm off and 87% of draws fail; K_phys=0 first dominates at f=0.05. A single random direction whose ray misses the model point by 0.05-0.2 m contributes 2.5e-3 to 4e-2 m^2 to the cost, versus ~1e-3 m^2 for all 558 inliers. This is the expected behaviour of a plain L2 cost and the reason a real front-end needs outlier rejection; it is **not** evidence that a detector would fail this way.

## 3. MC seed-level CI

Per-seed aggregate = mean over frames (and noise seeds) of that MC seed's metric; the CI is a percentile bootstrap over the MC seeds. The combined CI bootstraps frames and MC seeds together.

| condition | MC rot mean | frame CI | MC seed CI | combined CI | between-MC-seed SD |
|-|-|-|-|-|-|
| sigma_0 | 42.18 | [36.63, 47.39] | [41.90, 42.49] | [36.62, 47.29] | 0.503 |
| sigma_1 | 41.80 | [36.55, 47.25] | [41.60, 42.02] | [36.53, 46.96] | 0.359 |
| sigma_5 | 40.40 | [35.25, 45.99] | [40.17, 40.66] | [35.01, 45.37] | 0.419 |
| outlier_0.05 | 120.89 | [112.37, 128.76] | [119.40, 122.42] | [109.69, 131.04] | 2.618 |
| outlier_0.1 | 119.79 | [113.29, 124.95] | [116.63, 122.79] | [111.31, 128.40] | 5.460 |
| outlier_0.2 | 122.53 | [115.62, 129.09] | [120.92, 124.01] | [112.71, 130.72] | 2.675 |

## 4. Per-frame view at sigma = 5 px

Seed-averaged per frame. `K_phys` is the mean number of physical modes over seeds; `GT-like` is the fraction of seeds where the GT-closest mode is the hard argmin.

| frame/img | obj | K_phys | GT-like | hard rot | Laplace rot | MC rot | hard ADD-S | Laplace ADD-S | MC ADD-S |
|-|-|-|-|-|-|-|-|-|-|
| 000010/15 | 10 | 2.0 | 90% | 17.79 | 33.09 | 33.15 | 89.2 | 96.4 | 96.4 |
| 000010/18 | 10 | 2.0 | 100% | 9.12 | 41.82 | 41.61 | 117.6 | 126.2 | 126.2 |
| 000010/40 | 10 | 2.0 | 100% | 15.49 | 31.87 | 32.06 | 170.5 | 174.5 | 174.5 |
| 000010/60 | 10 | 2.0 | 100% | 5.70 | 46.56 | 46.97 | 55.1 | 70.6 | 70.6 |
| 000010/66 | 10 | 2.0 | 90% | 20.49 | 41.83 | 42.19 | 164.2 | 168.4 | 168.4 |
| 000010/72 | 10 | 2.0 | 50% | 44.25 | 23.37 | 23.55 | 177.6 | 177.5 | 177.5 |
| 000010/80 | 10 | 2.0 | 80% | 25.98 | 26.86 | 26.47 | 98.9 | 102.2 | 102.1 |
| 000010/90 | 10 | 2.0 | 80% | 29.88 | 26.67 | 26.77 | 174.2 | 178.2 | 178.2 |
| 000011/20 | 11 | 2.0 | 100% | 30.86 | 35.93 | 35.00 | 268.4 | 300.2 | 299.7 |
| 000011/24 | 11 | 2.0 | 100% | 7.54 | 52.56 | 53.38 | 99.9 | 106.7 | 106.8 |
| 000011/26 | 11 | 2.0 | 100% | 6.70 | 49.98 | 50.34 | 62.5 | 68.8 | 68.8 |
| 000011/32 | 11 | 2.0 | 100% | 6.29 | 45.68 | 45.14 | 65.3 | 73.7 | 73.5 |
| 000011/33 | 11 | 2.0 | 70% | 53.63 | 62.73 | 61.15 | 245.0 | 248.5 | 248.0 |
| 000011/40 | 11 | 2.0 | 100% | 26.84 | 44.23 | 43.30 | 553.1 | 575.7 | 575.5 |
| 000011/49 | 11 | 2.0 | 100% | 36.19 | 44.83 | 44.93 | 481.8 | 515.1 | 515.1 |

## 5. Interpretation and caveats

1. **Simulated detector noise, not a trained detector.** No image appearance, no descriptor, no training. Gaussian pixel noise is a model of a front-end's localisation error; the outlier injection is a model of a wrong match. Neither is calibrated to a real detector's error distribution.
2. **The absolute errors are optimistic by construction.** Every 3-D point is the nearest mesh vertex after mapping the depth pixel through the GT pose, so the GT-like mode is favoured and is the lowest-J mode in most clean frames. The experiment measures *relative* degradation and the hard/Laplace/MC trade under noise, not LINEMOD accuracy.
3. **The noise sensitivity is specific to the planar two-mode construction.** The layer is evaluated on the dominant planar face only (that is what creates the two-mode cost). A plane patch has a soft direction that trades depth against tilt, so the eliminated translation is much more noise-sensitive than it would be for a full-object correspondence set: at 5 px the GT-side minimum slides ~220 mm in translation while the cost changes by a few percent. This is a property of the *two-mode cost on a plane*, and it makes the absolute sigma=5 errors pessimistic in that one respect, even though the clean baseline is optimistic in the GT-association respect. The relative ordering hard < Laplace/MC on rotation is stable across all conditions.
4. **Hard vs Laplace.** Hard argmin is the most accurate estimator on this data and is expected to be, because the GT pose is the global minimum of the clean cost; Laplace is a posterior mean over a near-tie and is pulled toward the flip (w_other ~0.5). Pixel noise moves both curves but does not remove the near-tie; hard stays better on rotation at every sigma (2.1 vs 44.2 deg clean, 20.5 vs 41.8 deg at 5 px), but its advantage narrows as it degrades. Laplace's value is smoothness/differentiability, not accuracy.
5. **Hard argmin is fragile for two reasons.** (i) The selected mode can switch to the flip (the GT-like fraction drops below 100% in a few frames even at 2-5 px); (ii) more importantly, the GT-side minimum itself slides along the flat valley, so the rotation error grows to ~20 deg and the translation to ~220 mm at 5 px *without* a mode switch. The Laplace/MC mean cannot jump, but it is already ~40 deg off because of the near-tie bias, so it is not an accuracy rescue.
6. **`GT-like` is relative to the current minima.** At every condition `kgt` is the physical mode closest to GT *after* the noise; at 5 px that mode is itself ~20 deg off, so `GT-like = 100%` must not be read as 'the estimate is correct'. It only says no flip switch happened.
7. **The outlier arm is a breakdown measurement.** The plain L2 bearing cost has essentially no outlier tolerance: a single random direction at ~0.1 m residual (~1e-2 m^2) outweighs the 558 inlier residuals (~1e-3 m^2), and the z>0 physical filter then rejects all physical modes in 95% of the f=0.05 draws. We report the all-minima fallback (flagged) so the collapse is visible rather than hidden as a missing cell; these numbers must not be read as the accuracy of the layer on 'slightly corrupted' data. A robust kernel / front-end outlier rejection is a prerequisite, and its absence is exactly why this simulation cannot replace a trained front-end.
8. **Statistics.** 15 frames give wide CIs; the eggbox/glue groups are not separately powered. The frame bootstrap treats frames as exchangeable; noise/MC seeds are averaged within frame first, and MC seed noise is reported separately. The CIs describe this frame population, not LINEMOD.
9. **ADD-S details.** 1000 fixed mesh-vertex queries against the full GT mesh (dense KD-tree), one vertex subsample (the 3-subsample spread in `pose_error.py` was <=1.9 mm). Both objects have a discrete ~180 deg symmetry, so ADD-S (not ADD) is the appropriate metric.
10. **Runtime.** 461 s on the login-node CPU with 10 noise seeds x 10 MC seeds x 15 frames x 10 conditions; no condition needed a seed reduction.

## Repro

```bash
cd labs/01-laplace-qpeplayer/experiments/real/linemod
/home/syk/miniconda3/envs/WM/bin/python run_detector_noise.py
```

Deterministic: `Instance(seed=20261127)`, noise seeds `(20261128, cond, frame, seed)`, MC seeds `(20261128, cond, frame, seed)`, vertex subsamples `default_rng(20261127)`, bootstrap seeds fixed. Outputs: `results/detector_noise.{json,md,png}`. The JSON stores per-frame arrays over noise seeds and the MC seed draws, plus the reproduction check against `results/pose_error.json`: **ok** (max abs diff hard rot 6.6489036498751375e-12).
