# Classical external-style PnP baselines on the P1 real two-mode correspondences

BOP LINEMOD eggbox (obj 10) / glue (obj 11); 15 frames, same correspondences as `run_two_mode.py` Part A. Generated 2026-11-27 22:39:07 on ls with python 3.11.17, numpy 1.26.3, scipy 1.13.1, cv2 5.0.0 (sdpr env), runtime 3.0 s.

Reproduce: `/home/syk/miniconda3/envs/sdpr/bin/python labs/01-laplace-qpeplayer/experiments/real/linemod/run_classical_baselines.py` (the WM env has no cv2; sdpr has cv2 5.0.0 + scipy). No commit/push.

**Scope.** This is a forward-accuracy / ambiguity comparison on real correspondences. EPnP, ITERATIVE, SQPNP and IPPE are classical **forward solvers, not differentiable layers**; the local GN/LM baseline is a plain forward optimizer and is *not* BPnP. Nothing here differentiates through PnP or makes a gradient claim. A negative or inconclusive outcome would be reported as such.

## 0. Method list

| method | what it is |
|-|-|
| EPnP | `cv2.solvePnP(SOLVEPNP_EPNP)`, single solution |
| ITERATIVE | `cv2.solvePnP(SOLVEPNP_ITERATIVE)`, planar homography/DLT init + LM |
| SQPNP | `cv2.solvePnPGeneric(SOLVEPNP_SQPNP)`; cv2 5.0 docs: "Only 1 solution is returned" for non-P3P/IPPE flags — observed count is reported, not assumed |
| IPPE first | `cv2.solvePnPGeneric(SOLVEPNP_IPPE)` on the RANSAC inliers **projected onto the fitted plane** (IPPE requires exact coplanarity); first returned solution |
| IPPE oracle | best of the 2 planar solutions chosen against GT — an upper bound, not deployable |
| LM (identity / GT / flip init) | "local GN/LM (BPnP-style IFT baseline)": `scipy.optimize.least_squares(method='lm')`, 6-DoF rotvec+t, residual `Pm_i (R P_i + t)` stacked (3n), i.e. the same bearing cost P1 minimises |

## 1. Correspondence reconstruction sanity check

Real depth pixel -> bearing (real intrinsics) -> nearest real mesh vertex (model frame). The **original pixel** of each selected depth sample is kept, so PnP sees the true measurement, nearest-vertex quantization included. Frame inclusion is reproduced by the same loop/RANSAC/K>=2 filter and then asserted equal to `results/two_mode.json`.

- frames: **15/15**, identical `(frame, img)` set and identical `n_plane` counts to `run_two_mode.py` Part A: **True**.
- correspondences are bitwise identical to `run_two_mode.real_correspondences` (P and U compared with `array_equal`).
- dropped by the K>=2 filter: `000011/11` (n_plane=358, K_phys=1).
- GT-pose re-projection of the model point onto the original pixel: median residual **1.47 px** over all 900 selected samples (median of per-frame medians; inlier-only median 1.54 px), per-frame p90 median 2.89 px, worst sample 54 px. The ~1 px level is the mesh-vertex spacing (~1.6–1.8 mm ≈ 1.1 px at 0.86 m): the residual is the quantization of the nearest-vertex model point along the ray. The worst samples are mask/depth boundary pixels whose nearest mesh vertex is 20–170 mm away; like P1 we apply **no mesh-distance outlier filter**, so this small tail is shared by every method below (it does not move the medians).
- **independent cross-check against P1:** the local GN/LM from the GT init converges to P1's hard-argmin pose on **14/15 frames** with max |Δrot| = 0.000° and max |Δt| = 0.000 mm. The single mismatch is `000010/72` (P1 hard 44.5° = the flip mode, GT-init LM 3.8° = the true mode). This is what one expects if, and only if, the correspondence set and the bearing cost here are exactly P1's.

## 2. Per-frame results

### 2a. Single-solution forward solvers

| frame/img | obj | n_plane | reproj px | EPnP rot° | EPnP t mm | EPnP ADD-S | ITER rot° | ITER t mm | ITER ADD-S | SQPNP n | SQPNP rot° | SQPNP t mm | SQPNP ADD-S |
|-|-|-|-|-|-|-|-|-|-|-|-|-|-|
| 000010/15 | 10 | 558 | 0.99 | 1.41 | 5.8 | 4.5 | 1.60 | 7.7 | 5.4 | 1 | 1.47 | 5.7 | 4.5 |
| 000010/18 | 10 | 474 | 1.10 | 0.51 | 10.0 | 6.3 | 0.65 | 11.7 | 7.0 | 1 | 0.51 | 8.6 | 5.8 |
| 000010/40 | 10 | 482 | 1.43 | 2.34 | 16.1 | 9.0 | 2.21 | 15.3 | 8.7 | 1 | 2.20 | 12.2 | 7.4 |
| 000010/60 | 10 | 452 | 1.32 | 1.01 | 5.8 | 4.6 | 1.02 | 7.9 | 5.5 | 1 | 0.96 | 5.9 | 4.6 |
| 000010/66 | 10 | 460 | 1.81 | 94.96 | 46.1 | 23.2 | 2.50 | 9.3 | 6.1 | 1 | 2.29 | 5.0 | 4.0 |
| 000010/72 | 10 | 501 | 1.30 | 44.34 | 25.9 | 13.5 | 43.99 | 26.4 | 13.5 | 1 | 3.84 | 13.5 | 8.1 |
| 000010/80 | 10 | 508 | 1.25 | 1.23 | 10.0 | 6.5 | 1.17 | 8.9 | 6.0 | 1 | 1.08 | 7.1 | 5.2 |
| 000010/90 | 10 | 474 | 1.66 | 3.71 | 11.6 | 7.2 | 3.54 | 8.6 | 5.9 | 1 | 3.48 | 5.6 | 4.4 |
| 000011/20 | 11 | 413 | 1.86 | 10.37 | 78.4 | 30.6 | 10.00 | 72.1 | 27.1 | 1 | 8.12 | 44.0 | 14.4 |
| 000011/24 | 11 | 423 | 2.34 | 1.39 | 9.4 | 5.6 | 1.46 | 12.0 | 6.5 | 1 | 1.35 | 4.5 | 3.5 |
| 000011/26 | 11 | 439 | 2.82 | 2.05 | 10.3 | 6.1 | 2.07 | 12.1 | 6.8 | 1 | 2.04 | 6.1 | 4.4 |
| 000011/32 | 11 | 474 | 1.85 | 1.80 | 8.3 | 5.0 | 1.84 | 10.3 | 5.7 | 1 | 1.77 | 5.6 | 4.0 |
| 000011/33 | 11 | 365 | 1.58 | 1.61 | 1.5 | 1.6 | 1.76 | 2.2 | 2.2 | 1 | 1.61 | 11.0 | 4.7 |
| 000011/40 | 11 | 428 | 1.13 | 3.07 | 52.3 | 16.8 | 122.16 | 8165.1 | 8067.4 | 1 | 2.77 | 3.7 | 3.2 |
| 000011/49 | 11 | 347 | 1.47 | 6.58 | 85.0 | 35.9 | 178.97 | 1818.8 | 1729.6 | 1 | 4.94 | 60.8 | 22.8 |

### 2b. Planar multi-solution set (the ambiguity)

SQPNP returns **one** solution for every frame in cv2 5.0, so it exposes no multi-solution set here. IPPE on exactly-coplanar plane-projected inliers returns **2** solutions; `pair angle` is the rotation between them and `sol 1` is the planar flip. On these 15 frames the solution IPPE lists first is also the GT-closer one, so `first` == `oracle`; that ordering is empirical here (OpenCV does not document a sort order), and the oracle-vs-GT column is reported because it is the upper bound.

| frame/img | pair angle° | sol0 rot° | sol0 t mm | sol0 ADD-S | sol0 reproj px | sol1 rot° | sol1 t mm | sol1 ADD-S | sol1 reproj px | oracle rot° |
|-|-|-|-|-|-|-|-|-|-|-|
| 000010/15 | 68.7 | 1.40 | 4.9 | 4.0 | 0.81 | 69.07 | 35.2 | 18.4 | 1.19 | 1.40 |
| 000010/18 | 91.5 | 0.44 | 8.5 | 5.7 | 0.84 | 91.11 | 44.2 | 23.3 | 1.27 | 0.44 |
| 000010/40 | 69.2 | 2.34 | 13.1 | 7.8 | 0.77 | 69.75 | 36.1 | 17.0 | 1.11 | 2.34 |
| 000010/60 | 103.3 | 0.53 | 5.1 | 4.2 | 0.94 | 103.10 | 48.6 | 27.4 | 1.66 | 0.53 |
| 000010/66 | 86.8 | 1.87 | 5.7 | 4.4 | 0.92 | 88.45 | 42.4 | 21.8 | 1.16 | 1.87 |
| 000010/72 | 43.3 | 4.12 | 12.0 | 7.4 | 0.85 | 43.95 | 24.1 | 13.3 | 0.80 | 4.12 |
| 000010/80 | 54.7 | 1.59 | 6.9 | 5.1 | 0.75 | 54.63 | 28.5 | 15.3 | 1.00 | 1.59 |
| 000010/90 | 53.4 | 4.39 | 6.5 | 4.8 | 0.79 | 55.16 | 27.4 | 15.3 | 0.95 | 4.39 |
| 000011/20 | 70.2 | 8.29 | 61.3 | 21.5 | 1.26 | 74.88 | 58.5 | 17.0 | 2.24 | 8.29 |
| 000011/24 | 129.8 | 1.68 | 9.6 | 5.5 | 1.44 | 130.05 | 45.2 | 24.6 | 2.10 | 1.68 |
| 000011/26 | 126.2 | 2.76 | 10.1 | 5.8 | 1.60 | 125.82 | 45.5 | 24.1 | 2.42 | 2.76 |
| 000011/32 | 123.5 | 2.21 | 9.8 | 5.3 | 1.25 | 124.23 | 52.4 | 23.4 | 2.16 | 2.21 |
| 000011/33 | 147.3 | 2.70 | 3.2 | 2.6 | 1.30 | 148.97 | 33.0 | 27.4 | 1.72 | 2.70 |
| 000011/40 | 103.7 | 2.07 | 28.6 | 8.7 | 1.18 | 104.50 | 40.0 | 31.4 | 1.88 | 2.07 |
| 000011/49 | 77.7 | 7.45 | 92.5 | 40.7 | 1.04 | 84.63 | 81.5 | 26.9 | 1.85 | 7.45 |

### 2c. Local GN/LM (BPnP-style IFT baseline)

`mode` is the physical basin the optimizer ended in: `true` (<10° from GT), `flip` (<10° from the IPPE flip), `other`, or `mirror (z<=0)` when the estimate puts the object behind the camera (non-physical; P1 rejects such modes).

| frame/img | init | rot° | t mm | ADD-S | phys | mode | J_est | J_gt | nfev |
|-|-|-|-|-|-|-|-|-|-|
| 000010/15 | identity | 179.97 | 1766.8 | 1708.7 | NO | mirror (z<=0) | 8.16e-04 | 1.47e-03 | 73 |
| 000010/15 | gt | 1.47 | 5.7 | 4.5 | yes | true | 7.52e-04 | 1.47e-03 | 36 |
| 000010/15 | flip | 69.28 | 35.8 | 18.1 | yes | flip | 1.71e-03 | 1.47e-03 | 36 |
| 000010/18 | identity | 179.93 | 1961.2 | 1895.2 | NO | mirror (z<=0) | 1.25e-03 | 2.62e-03 | 81 |
| 000010/18 | gt | 0.51 | 8.6 | 5.8 | yes | true | 9.06e-04 | 2.62e-03 | 29 |
| 000010/18 | flip | 91.90 | 44.7 | 23.5 | yes | flip | 2.16e-03 | 2.62e-03 | 43 |
| 000010/40 | identity | 2.20 | 12.2 | 7.4 | yes | true | 8.71e-04 | 4.40e-03 | 65 |
| 000010/40 | gt | 2.20 | 12.2 | 7.4 | yes | true | 8.71e-04 | 4.40e-03 | 57 |
| 000010/40 | flip | 69.08 | 36.4 | 16.9 | yes | flip | 1.84e-03 | 4.40e-03 | 36 |
| 000010/60 | identity | 104.05 | 49.3 | 28.0 | yes | flip | 2.37e-03 | 2.16e-03 | 74 |
| 000010/60 | gt | 0.96 | 5.9 | 4.6 | yes | true | 6.80e-04 | 2.16e-03 | 29 |
| 000010/60 | flip | 104.05 | 49.3 | 28.0 | yes | flip | 2.37e-03 | 2.16e-03 | 36 |
| 000010/66 | identity | 2.29 | 4.9 | 4.0 | yes | true | 1.24e-03 | 5.96e-03 | 73 |
| 000010/66 | gt | 2.29 | 4.9 | 4.0 | yes | true | 1.24e-03 | 5.96e-03 | 36 |
| 000010/66 | flip | 88.39 | 42.9 | 21.6 | yes | flip | 1.94e-03 | 5.96e-03 | 50 |
| 000010/72 | identity | 178.82 | 2064.0 | 2010.8 | NO | mirror (z<=0) | 1.19e-03 | 3.28e-03 | 86 |
| 000010/72 | gt | 3.84 | 13.5 | 8.1 | yes | true | 1.14e-03 | 3.28e-03 | 50 |
| 000010/72 | flip | 44.46 | 24.8 | 13.4 | yes | flip | 1.01e-03 | 3.28e-03 | 36 |
| 000010/80 | identity | 54.71 | 29.3 | 15.4 | yes | flip | 1.15e-03 | 2.04e-03 | 97 |
| 000010/80 | gt | 1.08 | 7.1 | 5.2 | yes | true | 6.43e-04 | 2.04e-03 | 36 |
| 000010/80 | flip | 54.71 | 29.3 | 15.4 | yes | flip | 1.15e-03 | 2.04e-03 | 52 |
| 000010/90 | identity | 3.49 | 5.6 | 4.4 | yes | true | 9.06e-04 | 5.38e-03 | 73 |
| 000010/90 | gt | 3.49 | 5.6 | 4.4 | yes | true | 9.06e-04 | 5.38e-03 | 50 |
| 000010/90 | flip | 54.80 | 27.4 | 15.3 | yes | flip | 1.32e-03 | 5.38e-03 | 36 |
| 000011/20 | identity | 69.14 | 99.7 | 31.2 | yes | flip | 3.01e-03 | 2.47e-03 | 136 |
| 000011/20 | gt | 8.14 | 44.2 | 14.5 | yes | true | 9.48e-04 | 2.47e-03 | 77 |
| 000011/20 | flip | 69.14 | 99.7 | 31.2 | yes | flip | 3.01e-03 | 2.47e-03 | 139 |
| 000011/24 | identity | 179.94 | 1542.2 | 1494.3 | NO | mirror (z<=0) | 2.54e-03 | 4.77e-03 | 65 |
| 000011/24 | gt | 1.37 | 4.6 | 3.5 | yes | true | 1.60e-03 | 4.77e-03 | 36 |
| 000011/24 | flip | 131.05 | 47.0 | 24.4 | yes | flip | 3.25e-03 | 4.77e-03 | 36 |
| 000011/26 | identity | 179.81 | 1374.8 | 1328.5 | NO | mirror (z<=0) | 2.16e-03 | 5.57e-03 | 65 |
| 000011/26 | gt | 2.06 | 6.2 | 4.5 | yes | true | 1.40e-03 | 5.57e-03 | 36 |
| 000011/26 | flip | 126.87 | 47.5 | 24.2 | yes | flip | 3.30e-03 | 5.57e-03 | 43 |
| 000011/32 | identity | 179.82 | 1362.3 | 1310.9 | NO | mirror (z<=0) | 2.20e-03 | 2.80e-03 | 58 |
| 000011/32 | gt | 1.78 | 5.5 | 3.9 | yes | true | 9.69e-04 | 2.80e-03 | 43 |
| 000011/32 | flip | 123.91 | 58.9 | 22.9 | yes | flip | 3.07e-03 | 2.80e-03 | 43 |
| 000011/33 | identity | 179.75 | 2045.2 | 1998.7 | NO | mirror (z<=0) | 2.57e-03 | 2.99e-03 | 51 |
| 000011/33 | gt | 1.62 | 11.1 | 4.8 | yes | true | 1.85e-03 | 2.99e-03 | 36 |
| 000011/33 | flip | 149.58 | 30.5 | 29.3 | yes | flip | 3.35e-03 | 2.99e-03 | 36 |
| 000011/40 | identity | 106.01 | 38.5 | 40.7 | yes | flip | 4.20e-03 | 2.97e-03 | 69 |
| 000011/40 | gt | 2.77 | 3.9 | 3.3 | yes | true | 1.64e-03 | 2.97e-03 | 57 |
| 000011/40 | flip | 106.01 | 38.5 | 40.7 | yes | flip | 4.20e-03 | 2.97e-03 | 57 |
| 000011/49 | identity | 83.71 | 96.4 | 32.4 | yes | flip | 3.04e-03 | 2.34e-03 | 99 |
| 000011/49 | gt | 4.95 | 60.6 | 22.7 | yes | true | 9.41e-04 | 2.34e-03 | 57 |
| 000011/49 | flip | 83.71 | 96.4 | 32.4 | yes | flip | 3.04e-03 | 2.34e-03 | 78 |

### 2d. Is a single-solution error a wrong mode or just inaccuracy?

`reproj` is the RMS re-projection residual of the returned pose on the same inlier pixels; compare it with the GT pose's residual (last column). A large rotation error with a small reprojection residual means the solver confidently committed to the *other* planar solution, not that it failed to converge.

| frame/img | EPnP rot° | EPnP reproj px | EPnP J | ITER rot° | ITER reproj px | ITER J | SQPNP reproj px | SQPNP J | GT reproj px | J_gt |
|-|-|-|-|-|-|-|-|-|-|
| 000010/15 | 1.41 | 0.78 | 7.53e-04 | 1.60 | 0.78 | 7.54e-04 | 0.78 | 7.52e-04 | 0.99 | 1.47e-03 |
| 000010/18 | 0.51 | 0.81 | 9.07e-04 | 0.65 | 0.81 | 9.09e-04 | 0.81 | 9.06e-04 | 1.10 | 2.62e-03 |
| 000010/40 | 2.34 | 0.74 | 8.75e-04 | 2.21 | 0.74 | 8.74e-04 | 0.75 | 8.71e-04 | 1.43 | 4.40e-03 |
| 000010/60 | 1.01 | 0.88 | 6.81e-04 | 1.02 | 0.88 | 6.82e-04 | 0.88 | 6.80e-04 | 1.32 | 2.16e-03 |
| 000010/66 | 94.96 | 4.01 | 2.35e-02 | 2.50 | 0.91 | 1.24e-03 | 0.91 | 1.24e-03 | 1.81 | 5.96e-03 |
| 000010/72 | 44.34 | 0.79 | 1.02e-03 | 43.99 | 0.78 | 1.02e-03 | 0.84 | 1.14e-03 | 1.30 | 3.28e-03 |
| 000010/80 | 1.23 | 0.73 | 6.47e-04 | 1.17 | 0.73 | 6.45e-04 | 0.73 | 6.43e-04 | 1.25 | 2.04e-03 |
| 000010/90 | 3.71 | 0.78 | 9.17e-04 | 3.54 | 0.78 | 9.09e-04 | 0.78 | 9.06e-04 | 1.66 | 5.38e-03 |
| 000011/20 | 10.37 | 1.25 | 1.01e-03 | 10.00 | 1.25 | 9.88e-04 | 1.28 | 9.48e-04 | 1.86 | 2.47e-03 |
| 000011/24 | 1.39 | 1.43 | 1.61e-03 | 1.46 | 1.43 | 1.62e-03 | 1.44 | 1.60e-03 | 2.34 | 4.77e-03 |
| 000011/26 | 2.05 | 1.50 | 1.41e-03 | 2.07 | 1.50 | 1.42e-03 | 1.51 | 1.40e-03 | 2.82 | 5.57e-03 |
| 000011/32 | 1.80 | 1.18 | 9.71e-04 | 1.84 | 1.18 | 9.77e-04 | 1.19 | 9.69e-04 | 1.85 | 2.80e-03 |
| 000011/33 | 1.61 | 1.26 | 1.87e-03 | 1.76 | 1.26 | 1.88e-03 | 1.27 | 1.85e-03 | 1.58 | 2.99e-03 |
| 000011/40 | 3.07 | 1.15 | 1.71e-03 | 122.16 | 10.98 | 1.28e+01 | 1.19 | 1.64e-03 | 1.13 | 2.97e-03 |
| 000011/49 | 6.58 | 1.03 | 9.59e-04 | 178.97 | 1.34 | 1.80e-03 | 1.05 | 9.41e-04 | 1.47 | 2.34e-03 |

## 3. Aggregate (mean / median with 95% bootstrap CI over frames, 2000 resamples)

Cell format: `mean [CI] / median [CI]`, degrees or millimetres.

| method | n | rot° | trans mm | ADD-S mm | ADD mm | true-mode |
|-|-|-|-|-|-|-|
| EPnP | 15 | 11.76 [2.30, 26.69] / 2.05 [1.41, 3.71] | 25.1 [12.8, 38.9] / 10.3 [8.3, 25.9] | 11.8 [7.0, 17.1] / 6.5 [5.0, 13.5] | 27.8 [14.1, 43.4] / 10.1 [8.0, 40.0] | 12/15 |
| ITERATIVE | 15 | 25.00 [3.40, 53.88] / 2.07 [1.60, 3.54] | 679.2 [12.0, 1883.1] / 11.7 [8.6, 15.3] | 660.2 [7.0, 1848.6] / 6.5 [5.7, 8.7] | 678.3 [13.2, 1878.6] / 11.7 [8.7, 15.3] | 11/15 |
| SQPNP first | 15 | 2.56 [1.72, 3.59] / 2.04 [1.47, 2.77] | 13.3 [6.7, 21.8] / 6.1 [5.6, 11.0] | 6.7 [4.6, 9.5] / 4.6 [4.0, 5.8] | 13.2 [6.6, 21.7] / 5.9 [5.4, 11.0] | 15/15 |
| IPPE first | 15 | 2.92 [1.91, 4.09] / 2.21 [1.68, 2.76] | 18.5 [8.2, 32.2] / 9.6 [5.7, 12.0] | 8.9 [5.1, 14.1] / 5.5 [4.4, 7.4] | 18.3 [8.1, 31.8] / 9.3 [5.7, 12.1] | 15/15 |
| IPPE oracle (GT) | 15 | 2.92 [1.91, 4.09] / 2.21 [1.68, 2.76] | 18.5 [8.2, 32.2] / 9.6 [5.7, 12.0] | 8.9 [5.1, 14.1] / 5.5 [4.4, 7.4] | 18.3 [8.1, 31.8] / 9.3 [5.7, 12.1] | 15/15 |
| LM identity init | 15 | 112.24 [75.95, 147.27] / 106.01 [69.14, 179.82] | 830.2 [404.1, 1272.4] / 99.7 [38.5, 1766.8] | 794.0 [372.3, 1226.7] / 40.7 [28.0, 1708.7] | 846.8 [413.8, 1287.7] / 93.5 [83.8, 1771.1] | 3/15 |
| LM GT init | 15 | 2.57 [1.72, 3.60] / 2.06 [1.47, 2.77] | 13.3 [6.7, 21.9] / 6.2 [5.5, 11.1] | 6.7 [4.6, 9.5] / 4.6 [4.0, 5.8] | 13.2 [6.6, 21.7] / 6.0 [5.3, 11.1] | 15/15 |
| LM flip init | 15 | 91.13 [76.93, 106.85] / 88.39 [69.08, 106.01] | 47.3 [37.0, 59.1] / 42.9 [35.8, 47.5] | 23.8 [20.3, 27.5] / 23.5 [16.9, 28.0] | 70.4 [62.2, 77.8] / 73.4 [60.4, 76.4] | 0/15 |

Reference rows (P1, **cited not recomputed**; `results/two_mode.md` + its JSON):

- P1 hard argmin: median rotation error **2.06°** (fails >10° in 1/15 frames), median translation error **6.2 mm** (median of the per-frame `hard_trans_err` recorded in `two_mode.json`).
- P1 Laplace mean: median rotation error **58.28°**, median translation error **20.2 mm**.

## 4. Per-frame comparison with P1 (P1 values cited from `two_mode.json`)

| frame/img | P1 hard rot° | P1 hard t mm | P1 Laplace rot° | P1 Laplace t mm | EPnP rot° | IPPE-oracle rot° | LM(GT) rot° | LM(id) rot° (mode) |
|-|-|-|-|-|-|-|-|-|
| 000010/15 | 1.47 | 5.7 | 46.56 | 17.6 | 1.41 | 1.40 | 1.47 | 180.0 (mirror (z<=0)) |
| 000010/18 | 0.51 | 8.6 | 59.17 | 20.9 | 0.51 | 0.44 | 0.51 | 179.9 (mirror (z<=0)) |
| 000010/40 | 2.20 | 12.2 | 46.41 | 19.4 | 2.34 | 2.34 | 2.20 | 2.2 (true) |
| 000010/60 | 0.96 | 5.9 | 64.56 | 21.8 | 1.01 | 0.53 | 0.96 | 104.0 (flip) |
| 000010/66 | 2.29 | 4.9 | 58.28 | 20.2 | 94.96 | 1.87 | 2.29 | 2.3 (true) |
| 000010/72 | 44.46 | 24.8 | 31.02 | 16.0 | 44.34 | 4.12 | 3.84 | 178.8 (mirror (z<=0)) |
| 000010/80 | 1.08 | 7.1 | 37.98 | 15.4 | 1.23 | 1.59 | 1.08 | 54.7 (flip) |
| 000010/90 | 3.49 | 5.6 | 37.96 | 13.2 | 3.71 | 4.39 | 3.49 | 3.5 (true) |
| 000011/20 | 8.14 | 44.2 | 52.80 | 77.4 | 10.37 | 8.29 | 8.14 | 69.1 (flip) |
| 000011/24 | 1.37 | 4.6 | 75.98 | 22.3 | 1.39 | 1.68 | 1.37 | 179.9 (mirror (z<=0)) |
| 000011/26 | 2.06 | 6.2 | 73.75 | 23.1 | 2.05 | 2.76 | 2.06 | 179.8 (mirror (z<=0)) |
| 000011/32 | 1.78 | 5.5 | 72.70 | 28.6 | 1.80 | 2.21 | 1.78 | 179.8 (mirror (z<=0)) |
| 000011/33 | 1.62 | 11.1 | 81.79 | 12.6 | 1.61 | 2.70 | 1.62 | 179.7 (mirror (z<=0)) |
| 000011/40 | 2.77 | 3.9 | 65.39 | 16.2 | 3.07 | 2.07 | 2.77 | 106.0 (flip) |
| 000011/49 | 4.95 | 60.6 | 58.11 | 78.5 | 6.58 | 7.45 | 4.95 | 83.7 (flip) |

## 5. Findings

1. **Forward point accuracy is at least as good as P1 hard argmin.** Median rotation error: EPnP 2.05°, ITERATIVE 2.07°, SQPNP 2.04°, IPPE first 2.21° — all at P1's cited 2.06° hard-argmin level. Median translation is 6.1 mm for SQPNP (matching P1's cited 6.2 mm), but 10.3 mm for EPnP and 9.6 mm for IPPE first, i.e. translation is the metric where the solver choice matters; median ADD-S is 4.6–6.5 mm. P1's Laplace-mean median is 58.28° (cited), far outside every classical row. So a classical forward solver is the right null hypothesis for the point estimate, and the comparison supports P1's own negative reading of the posterior mean rather than an accuracy win.

2. **Single-solution solvers are confidently wrong on some frames — and reproduce P1's one hard-argmin failure.** EPnP is >10° from GT on 3/15 frames (`000010/66` (95°, reproj 4.0 px), `000010/72` (44°, reproj 0.8 px), `000011/20` (10°, reproj 1.3 px)); ITERATIVE on 4/15 (`000010/72` (44°, reproj 0.8 px), `000011/20` (10°, reproj 1.2 px), `000011/40` (122°, reproj 11.0 px), `000011/49` (179°, reproj 1.3 px)). This is not just non-convergence: on `000010/72` EPnP and ITERATIVE both return the planar flip at 44.3° / 44.0° with a reprojection residual of 0.79 px, and their bearing cost J = 1.02e-03 is *lower* than 1.16e-03 of the true-mode planar solution on the same raw inliers; the data itself prefers the flip. P1's hard argmin fails on exactly that frame (44.46°, cited). On the other failing frames the solver is simply inaccurate or returns a behind-camera (mirror) solution, with reprojection residuals of several px, so those are solver failures, not ambiguity.

3. **The planar multi-solution set is real, but the SQPNP API does not expose it in cv2 5.0.** `solvePnPGeneric(SOLVEPNP_SQPNP)` returned 1 solution on every frame (cv2 5.0's own docstring: "Only 1 solution is returned" for non-P3P/IPPE flags), so SQPNP is a single-solution solver in this toolchain. IPPE on exactly-coplanar plane-projected inliers returns 2 solutions on 15/15 frames with a pair rotation of 43–147°; the second solution is 44–149° and 24–81 mm from GT (full per-solution rows in §2b). **First vs oracle:** on these frames the GT-closer IPPE solution is always the one IPPE lists first, so the oracle gain is exactly zero (first = oracle = 2.21° / 9.6 mm); the oracle column is reported because it is the upper bound, and the equality is an empirical property of these 15 frames, not a documented ordering guarantee.

4. **The unconstrained local GN/LM baseline fails without a physical filter and multi-start.** From identity/zero the algebraic bearing cost is sign-symmetric (`Pm(-x) = -Pm(x)`), so the LM iterate can settle on a stationary point with **all points behind the camera**: 7/15 frames end in a mirror (z ≤ 0) pose, 3/15 in the true mode, 5/15 in the flip mode (median rotation error 106.01°). From the GT init it stays in the true mode on 15/15 frames (median 2.06° / 6.2 mm), and from the IPPE flip init it stays out of the true mode (0/15 true, median 88.39°). This is an honest negative for a single-init local bearing-cost baseline: P1's explicit rejection of z ≤ 0 modes and its multi-start enumeration are doing real work here.

5. **GT is not the minimiser of the bearing cost** — the cost is fitted on nearest-vertex-quantised model points. The solver poses have a lower algebraic cost than the GT pose on 15/15 frames, which is why an argmin sits ~1–2° from the annotation rather than on it, and why ADD-S/translation errors have a ~1 px floor. Statements about 'the true pose being the global minimum' (P1) are about the true *mode*, not about GT being the exact minimiser; our per-frame rows are consistent with P1's hard errors.

6. **Scope: forward only.** These are classical forward solvers. Nothing above differentiates through PnP or supports/denies a gradient claim; the numbers bound the *forward* accuracy/ambiguity that any differentiable or sampling layer on these correspondences would have to be compared against, and here that bound is aggressive (2° median for a plain EPnP/SQPNP).

## 6. Caveats

- **Same correspondences, same subset.** The 15 frames and the RANSAC inlier masks are reproduced bitwise (verified against `two_mode.json`); nothing was re-tuned.
- **Sample size.** 15 frames, one object instance each; bootstrap CIs are over frames and therefore only describe this fixed set, not a population.
- **Vertex quantization dominates the noise floor.** The model point is the nearest mesh vertex (~1.6–1.8 mm spacing ≈ 1.1 px at 0.86 m), and the plane inlier band is 2.5 mm; the GT re-projection residual of ~1 px is this quantization, so absolute accuracy below ~1 px is not meaningful for any method here.
- **IPPE uses plane-projected points** (exactly coplanar, planarity RMS 0.83–1.47 mm); EPnP/ITERATIVE/SQPNP/LM use the raw inlier points. This is a classical modelling step, stated so the numbers are not read as identical inputs.
- **No front-end.** These poses start from the same real 3-D–2-D correspondences and K; no detector, no RANSAC-PnP outlier rejection, no refinement with re-projection error, no test-time augmentation. The numbers characterise solvers, not a pipeline.
- **Oracle columns are not deployable** and are shown only to separate 'solver picked the wrong planar solution' from 'solver is inaccurate'.
- **P1 numbers are cited, not recomputed**: medians quoted from `results/two_mode.md`; per-frame P1 values read from `results/two_mode.json` (same frames). ADD/ADD-S for the P1 poses were not recorded by P1 and are therefore absent rather than invented.
- **cv2 version.** cv2 5.0.0 changes the multi-solution contract of `solvePnPGeneric`; on OpenCV 4.x SQPNP may return up to 3 planar solutions, so the solution counts here are tool-version-specific and were measured, not assumed.

