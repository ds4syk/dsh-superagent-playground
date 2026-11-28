# 3DMatch cross-scene robustness check (2 unseen scenes)

The parent table (`experiments/real/threedmatch/`) concluded on redkitchen and sun3d-home_at that (i) GNC-GM keeps RANSAC-level success at 0-70% random outliers while being ~27-30x faster than RANSAC-1000, (ii) coherent outliers at 60-70% defeat every method (model identifiability), and (iii) boundary outliers are harmless (all methods 100%, GNC lower median rotation error). This run repeats the identical protocol on two *unseen* official scenes.

Scenes: `sun3d-hotel_uc-scan3` (hotel, 55 fragments) and `sun3d-home_md-home_md_scan9_2012_sep_30` (home_md, 60 fragments). N=300, c=0.05 m, tau=0.06 m, voxel=0.03 m, 20 trials/cell, 8 high-overlap pairs/scene, p=0-0.7, 3 outlier types, 5 methods. Base correspondences are GT-NN oracle matches on real fragments (real scan/occlusion residual), not a learned descriptor.

**GT direction**: `verify_gt_direction_extra.py` re-checked both new scenes; the stored 4x4 maps id2->id1 (inverse median NN distance << forward), so the runner inverts it exactly as the parent does. See `results/verify_gt_direction_extra.json`.

## sun3d-hotel_uc

Selected pairs: 16-22 (overlap 100%, med res 12.7 mm), 12-48 (overlap 100%, med res 12.7 mm), 27-29 (overlap 100%, med res 13.2 mm), 27-28 (overlap 100%, med res 13.4 mm), 14-24 (overlap 100%, med res 13.5 mm), 29-30 (overlap 100%, med res 13.5 mm), 25-48 (overlap 100%, med res 13.6 mm), 32-48 (overlap 100%, med res 13.7 mm).

### random outliers: success rate (%)

| method | 0% | 10% | 20% | 30% | 40% | 50% | 60% | 70% |
|---|---|---|---|---|---|---|---|---|
| Kabsch | 100 | 80 | 55 | 20 | 5 | 0 | 0 | 0 |
| RANSAC-1000 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| RANSAC-adaptive | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| IRLS-GM | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| GNC-GM | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |

### random outliers: median rotation error (deg)

| method | 0% | 10% | 20% | 30% | 40% | 50% | 60% | 70% |
|---|---|---|---|---|---|---|---|---|
| Kabsch | 0.29 | 1.56 | 1.97 | 3.64 | 4.26 | 7.50 | 10.40 | 14.04 |
| RANSAC-1000 | 0.30 | 0.32 | 0.19 | 0.28 | 0.32 | 0.28 | 0.28 | 0.28 |
| RANSAC-adaptive | 0.32 | 0.31 | 0.22 | 0.31 | 0.32 | 0.29 | 0.30 | 0.30 |
| IRLS-GM | 0.29 | 0.33 | 0.20 | 0.27 | 0.32 | 0.29 | 0.26 | 0.28 |
| GNC-GM | 0.29 | 0.33 | 0.20 | 0.27 | 0.32 | 0.29 | 0.26 | 0.28 |

### random outliers: runtime p50 (ms)

| method | 0% | 10% | 20% | 30% | 40% | 50% | 60% | 70% |
|---|---|---|---|---|---|---|---|---|
| Kabsch | 0.11 | 0.11 | 0.10 | 0.11 | 0.11 | 0.11 | 0.10 | 0.11 |
| RANSAC-1000 | 63.35 | 63.19 | 62.46 | 62.96 | 62.91 | 63.14 | 63.64 | 64.15 |
| RANSAC-adaptive | 0.33 | 0.53 | 0.82 | 1.28 | 2.06 | 3.50 | 7.21 | 17.01 |
| IRLS-GM | 2.74 | 2.68 | 2.66 | 2.68 | 2.67 | 2.69 | 2.73 | 2.74 |
| GNC-GM | 0.85 | 2.31 | 2.28 | 2.29 | 2.29 | 2.30 | 2.35 | 2.34 |

### coherent outliers: success rate (%)

| method | 0% | 10% | 20% | 30% | 40% | 50% | 60% | 70% |
|---|---|---|---|---|---|---|---|---|
| Kabsch | 100 | 35 | 0 | 0 | 0 | 0 | 0 | 0 |
| RANSAC-1000 | 100 | 100 | 100 | 100 | 100 | 0 | 0 | 0 |
| RANSAC-adaptive | 100 | 100 | 100 | 100 | 100 | 0 | 0 | 0 |
| IRLS-GM | 100 | 100 | 100 | 100 | 100 | 75 | 0 | 0 |
| GNC-GM | 100 | 100 | 100 | 100 | 100 | 25 | 0 | 0 |

### boundary outliers: median rotation error (deg)

| method | 0% | 10% | 20% | 30% | 40% | 50% | 60% | 70% |
|---|---|---|---|---|---|---|---|---|
| Kabsch | 0.256 | 0.225 | 0.273 | 0.265 | 0.193 | 0.260 | 0.311 | 0.279 |
| RANSAC-1000 | 0.259 | 0.272 | 0.406 | 0.367 | 0.321 | 0.436 | 0.503 | 0.517 |
| RANSAC-adaptive | 0.240 | 0.293 | 0.387 | 0.345 | 0.325 | 0.288 | 0.503 | 0.619 |
| IRLS-GM | 0.223 | 0.236 | 0.370 | 0.231 | 0.191 | 0.262 | 0.264 | 0.307 |
| GNC-GM | 0.223 | 0.236 | 0.370 | 0.231 | 0.191 | 0.262 | 0.264 | 0.307 |

## sun3d-home_md

Selected pairs: 17-18 (overlap 100%, med res 11.4 mm), 17-19 (overlap 100%, med res 11.6 mm), 20-21 (overlap 100%, med res 11.9 mm), 21-22 (overlap 100%, med res 11.9 mm), 18-19 (overlap 100%, med res 11.9 mm), 9-37 (overlap 100%, med res 12.3 mm), 53-54 (overlap 100%, med res 12.6 mm), 19-20 (overlap 100%, med res 12.6 mm).

### random outliers: success rate (%)

| method | 0% | 10% | 20% | 30% | 40% | 50% | 60% | 70% |
|---|---|---|---|---|---|---|---|---|
| Kabsch | 100 | 85 | 80 | 55 | 50 | 45 | 20 | 0 |
| RANSAC-1000 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| RANSAC-adaptive | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| IRLS-GM | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 95 |
| GNC-GM | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |

### random outliers: median rotation error (deg)

| method | 0% | 10% | 20% | 30% | 40% | 50% | 60% | 70% |
|---|---|---|---|---|---|---|---|---|
| Kabsch | 0.34 | 1.67 | 2.65 | 3.22 | 2.93 | 4.88 | 8.06 | 10.89 |
| RANSAC-1000 | 0.31 | 0.35 | 0.33 | 0.27 | 0.40 | 0.42 | 0.33 | 0.57 |
| RANSAC-adaptive | 0.35 | 0.40 | 0.38 | 0.31 | 0.42 | 0.45 | 0.35 | 0.59 |
| IRLS-GM | 0.26 | 0.35 | 0.27 | 0.26 | 0.32 | 0.29 | 0.34 | 0.52 |
| GNC-GM | 0.26 | 0.35 | 0.27 | 0.26 | 0.32 | 0.29 | 0.34 | 0.52 |

### random outliers: runtime p50 (ms)

| method | 0% | 10% | 20% | 30% | 40% | 50% | 60% | 70% |
|---|---|---|---|---|---|---|---|---|
| Kabsch | 0.11 | 0.14 | 0.12 | 0.11 | 0.11 | 0.11 | 0.10 | 0.11 |
| RANSAC-1000 | 61.75 | 85.20 | 80.86 | 65.14 | 63.99 | 63.76 | 61.54 | 62.11 |
| RANSAC-adaptive | 0.30 | 0.75 | 1.14 | 1.32 | 2.06 | 3.67 | 6.56 | 16.13 |
| IRLS-GM | 2.63 | 4.12 | 4.08 | 2.77 | 2.72 | 2.73 | 2.65 | 2.66 |
| GNC-GM | 0.82 | 2.81 | 2.77 | 2.10 | 1.97 | 1.96 | 1.98 | 2.10 |

### coherent outliers: success rate (%)

| method | 0% | 10% | 20% | 30% | 40% | 50% | 60% | 70% |
|---|---|---|---|---|---|---|---|---|
| Kabsch | 100 | 5 | 0 | 0 | 0 | 0 | 0 | 0 |
| RANSAC-1000 | 100 | 100 | 100 | 100 | 100 | 0 | 0 | 0 |
| RANSAC-adaptive | 100 | 100 | 100 | 100 | 100 | 0 | 0 | 0 |
| IRLS-GM | 100 | 100 | 100 | 100 | 100 | 40 | 0 | 0 |
| GNC-GM | 100 | 100 | 100 | 100 | 100 | 50 | 0 | 0 |

### boundary outliers: median rotation error (deg)

| method | 0% | 10% | 20% | 30% | 40% | 50% | 60% | 70% |
|---|---|---|---|---|---|---|---|---|
| Kabsch | 0.331 | 0.321 | 0.541 | 0.431 | 0.471 | 0.465 | 0.488 | 0.641 |
| RANSAC-1000 | 0.331 | 0.357 | 0.449 | 0.572 | 0.595 | 0.817 | 0.931 | 0.919 |
| RANSAC-adaptive | 0.363 | 0.309 | 0.468 | 0.459 | 0.558 | 0.840 | 0.795 | 1.282 |
| IRLS-GM | 0.301 | 0.271 | 0.373 | 0.328 | 0.303 | 0.424 | 0.441 | 0.440 |
| GNC-GM | 0.301 | 0.271 | 0.373 | 0.328 | 0.303 | 0.424 | 0.441 | 0.440 |

## Pooled over the two extra scenes (macro average of scene cells)

### random outliers: success rate (%)

| method | 0% | 10% | 20% | 30% | 40% | 50% | 60% | 70% |
|---|---|---|---|---|---|---|---|---|
| Kabsch | 100 | 82 | 68 | 38 | 28 | 22 | 10 | 0 |
| RANSAC-1000 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| RANSAC-adaptive | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| IRLS-GM | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 98 |
| GNC-GM | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |

### random outliers: median rotation error (deg)

| method | 0% | 10% | 20% | 30% | 40% | 50% | 60% | 70% |
|---|---|---|---|---|---|---|---|---|
| Kabsch | 0.31 | 1.62 | 2.31 | 3.43 | 3.59 | 6.19 | 9.23 | 12.47 |
| RANSAC-1000 | 0.31 | 0.34 | 0.26 | 0.28 | 0.36 | 0.35 | 0.31 | 0.43 |
| RANSAC-adaptive | 0.33 | 0.35 | 0.30 | 0.31 | 0.37 | 0.37 | 0.33 | 0.45 |
| IRLS-GM | 0.27 | 0.34 | 0.24 | 0.27 | 0.32 | 0.29 | 0.30 | 0.40 |
| GNC-GM | 0.27 | 0.34 | 0.24 | 0.27 | 0.32 | 0.29 | 0.30 | 0.40 |

### random outliers: runtime p50 (ms)

| method | 0% | 10% | 20% | 30% | 40% | 50% | 60% | 70% |
|---|---|---|---|---|---|---|---|---|
| Kabsch | 0.11 | 0.12 | 0.11 | 0.11 | 0.11 | 0.11 | 0.10 | 0.11 |
| RANSAC-1000 | 62.55 | 74.19 | 71.66 | 64.05 | 63.45 | 63.45 | 62.59 | 63.13 |
| RANSAC-adaptive | 0.32 | 0.64 | 0.98 | 1.30 | 2.06 | 3.58 | 6.89 | 16.57 |
| IRLS-GM | 2.69 | 3.40 | 3.37 | 2.73 | 2.69 | 2.71 | 2.69 | 2.70 |
| GNC-GM | 0.83 | 2.56 | 2.53 | 2.19 | 2.13 | 2.13 | 2.16 | 2.22 |

### coherent outliers: success rate (%)

| method | 0% | 10% | 20% | 30% | 40% | 50% | 60% | 70% |
|---|---|---|---|---|---|---|---|---|
| Kabsch | 100 | 20 | 0 | 0 | 0 | 0 | 0 | 0 |
| RANSAC-1000 | 100 | 100 | 100 | 100 | 100 | 0 | 0 | 0 |
| RANSAC-adaptive | 100 | 100 | 100 | 100 | 100 | 0 | 0 | 0 |
| IRLS-GM | 100 | 100 | 100 | 100 | 100 | 57 | 0 | 0 |
| GNC-GM | 100 | 100 | 100 | 100 | 100 | 38 | 0 | 0 |

### boundary outliers: success rate (%)

| method | 0% | 10% | 20% | 30% | 40% | 50% | 60% | 70% |
|---|---|---|---|---|---|---|---|---|
| Kabsch | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| RANSAC-1000 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| RANSAC-adaptive | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| IRLS-GM | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| GNC-GM | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |

### boundary outliers: median rotation error (deg)

| method | 0% | 10% | 20% | 30% | 40% | 50% | 60% | 70% |
|---|---|---|---|---|---|---|---|---|
| Kabsch | 0.294 | 0.273 | 0.407 | 0.348 | 0.332 | 0.362 | 0.400 | 0.460 |
| RANSAC-1000 | 0.295 | 0.315 | 0.427 | 0.470 | 0.458 | 0.627 | 0.717 | 0.718 |
| RANSAC-adaptive | 0.302 | 0.301 | 0.427 | 0.402 | 0.441 | 0.564 | 0.649 | 0.950 |
| IRLS-GM | 0.262 | 0.253 | 0.371 | 0.280 | 0.247 | 0.343 | 0.352 | 0.373 |
| GNC-GM | 0.262 | 0.253 | 0.371 | 0.280 | 0.247 | 0.343 | 0.352 | 0.373 |

## Do the parent two-scene conclusions hold?

| check | parent claim | extra-scene measurement | holds |
|---|---|---|---|
| random_gnc_success_ge_95 | parent: GNC-GM success 100% at 0-70% random outliers (both scenes) | 100.0% | YES |
| random_gnc_parity_ge_m005 | parent: GNC-GM keeps RANSAC-1000-level random-outlier success | 0.000 | YES |
| gnc_speedup_ge_10x | parent: GNC-GM ~27-30x faster than RANSAC-1000 (p50) | 28.4x | YES |
| coherent_60_70_all_zero | parent: coherent 60-70% -> all methods 0% (identifiability limit) | 0.0% | YES |
| boundary_all_success_ge_95 | parent: boundary outliers -> all methods 100% (consistency, no breakdown) | 100.0% | YES |
| boundary_gnc_rot_le_ransac | parent: boundary 70% GNC median rot 0.25 deg vs RANSAC 0.74 deg | 0.373 deg (GNC) vs 0.718 deg (RANSAC) | YES |

**Overall: parent conclusions hold on the two extra scenes.**

### Parent vs extra scenes, same pooled cells

| cell | parent (2 scenes) | extra (2 scenes) |
|---|---|---|
| random 0% GNC success | 1.000 | 1.000 |
| random 70% GNC success | 1.000 | 1.000 |
| random 70% IRLS success | 0.975 | 0.975 |
| random 70% RANSAC-1000 success | 1.000 | 1.000 |
| coherent 50% GNC success | 0.500 | 0.375 |
| coherent 60% GNC success | 0.000 | 0.000 |
| coherent 70% GNC success | 0.000 | 0.000 |
| boundary 70% GNC med rot (deg) | 0.252 | 0.373 |

* random p=0% p50: parent RANSAC 55.0 ms / GNC 0.81 ms (68.1x); extra RANSAC 62.5 ms / GNC 0.83 ms (75.2x).
* random p=30% p50: parent RANSAC 62.7 ms / GNC 2.24 ms (28.0x); extra RANSAC 64.1 ms / GNC 2.19 ms (29.2x).
* random p=50% p50: parent RANSAC 64.3 ms / GNC 2.35 ms (27.4x); extra RANSAC 63.4 ms / GNC 2.13 ms (29.8x).
* random p=70% p50: parent RANSAC 67.5 ms / GNC 2.45 ms (27.6x); extra RANSAC 63.1 ms / GNC 2.22 ms (28.4x).

## Caveats

* GT-NN base correspondences use the ground-truth transform to establish overlap; they are oracle matches, not a learned descriptor. Real scanner noise/occlusion/partial overlap are present (natural inlier fraction below 1).
* Outliers are injected in the observation space, exactly as in the parent and synthetic protocols; this tests the robust outer layer, not the front-end.
* Pooled numbers are macro averages over two scenes, so they describe these two extra scenes only; they do not replace the full 3DMatch benchmark (this is a 4-scene cross-scene check in total).
