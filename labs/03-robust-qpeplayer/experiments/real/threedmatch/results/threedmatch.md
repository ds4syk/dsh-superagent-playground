# 3DMatch real-fragment robust 3D-3D registration

Two official 3DMatch geometric-registration scenes. Base correspondences are GT-mapped nearest neighbours on real overlapping fragments (oracle matches with real scan/occlusion residual); controlled outliers are then injected. N=300, c=0.05 m, tau=0.06 m, 20 trials/cell, trials sample one of 8 high-overlap fragment pairs per scene.

This is a real-geometry robust-outer-layer test, not a descriptor front-end. See `run_descriptor.py` for the no-GT-leakage hand-crafted descriptor variant.

## redkitchen

Selected pairs: 51-52 (overlap 100%, med res 12.0 mm), 50-51 (overlap 100%, med res 12.2 mm), 50-52 (overlap 100%, med res 12.2 mm), 13-14 (overlap 100%, med res 12.4 mm) ...

### random outliers: success rate

| method | 0% | 10% | 20% | 30% | 40% | 50% | 60% | 70% |
|---|---|---|---|---|---|---|---|---|
| Kabsch | 100 | 100 | 65 | 45 | 30 | 20 | 5 | 0 |
| RANSAC-1000 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| RANSAC-adaptive | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| IRLS-GM | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 95 |
| GNC-GM | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |

### random outliers: median rotation error (deg)

| method | 0% | 10% | 20% | 30% | 40% | 50% | 60% | 70% |
|---|---|---|---|---|---|---|---|---|
| Kabsch | 0.20 | 0.75 | 2.59 | 2.43 | 4.25 | 6.02 | 6.43 | 9.59 |
| RANSAC-1000 | 0.20 | 0.20 | 0.26 | 0.25 | 0.28 | 0.25 | 0.26 | 0.29 |
| RANSAC-adaptive | 0.20 | 0.17 | 0.27 | 0.25 | 0.26 | 0.25 | 0.25 | 0.33 |
| IRLS-GM | 0.18 | 0.17 | 0.26 | 0.23 | 0.26 | 0.24 | 0.24 | 0.24 |
| GNC-GM | 0.18 | 0.17 | 0.26 | 0.23 | 0.26 | 0.24 | 0.24 | 0.24 |

### random outliers: runtime p50 (ms)

| method | 0% | 10% | 20% | 30% | 40% | 50% | 60% | 70% |
|---|---|---|---|---|---|---|---|---|
| Kabsch | 0.11 | 0.13 | 0.10 | 0.12 | 0.11 | 0.12 | 0.11 | 0.13 |
| RANSAC-1000 | 49.03 | 70.06 | 61.43 | 62.83 | 62.29 | 63.37 | 62.23 | 71.03 |
| RANSAC-adaptive | 0.32 | 0.58 | 0.75 | 1.26 | 2.01 | 3.94 | 6.89 | 18.70 |
| IRLS-GM | 2.73 | 2.79 | 2.62 | 2.63 | 2.70 | 2.75 | 2.66 | 2.99 |
| GNC-GM | 0.81 | 2.31 | 2.20 | 2.22 | 2.30 | 2.37 | 2.28 | 2.52 |

### coherent outliers: success rate

| method | 0% | 10% | 20% | 30% | 40% | 50% | 60% | 70% |
|---|---|---|---|---|---|---|---|---|
| Kabsch | 100 | 25 | 0 | 0 | 0 | 0 | 0 | 0 |
| RANSAC-1000 | 100 | 100 | 100 | 100 | 100 | 0 | 0 | 0 |
| RANSAC-adaptive | 100 | 100 | 100 | 100 | 100 | 0 | 0 | 0 |
| IRLS-GM | 100 | 100 | 100 | 100 | 100 | 55 | 0 | 0 |
| GNC-GM | 100 | 100 | 100 | 100 | 100 | 65 | 0 | 0 |

### coherent outliers: median rotation error (deg)

| method | 0% | 10% | 20% | 30% | 40% | 50% | 60% | 70% |
|---|---|---|---|---|---|---|---|---|
| Kabsch | 0.23 | 4.82 | 9.87 | 13.24 | 19.25 | 23.93 | 26.69 | 30.94 |
| RANSAC-1000 | 0.22 | 0.25 | 0.23 | 0.27 | 0.25 | 43.86 | 43.50 | 41.36 |
| RANSAC-adaptive | 0.22 | 0.25 | 0.23 | 0.24 | 0.26 | 43.86 | 43.50 | 41.36 |
| IRLS-GM | 0.20 | 0.23 | 0.19 | 0.22 | 0.24 | 0.42 | 43.50 | 41.36 |
| GNC-GM | 0.20 | 0.23 | 0.19 | 0.22 | 0.24 | 0.35 | 43.50 | 41.36 |

### coherent outliers: runtime p50 (ms)

| method | 0% | 10% | 20% | 30% | 40% | 50% | 60% | 70% |
|---|---|---|---|---|---|---|---|---|
| Kabsch | 0.13 | 0.10 | 0.11 | 0.12 | 0.10 | 0.10 | 0.11 | 0.10 |
| RANSAC-1000 | 34.75 | 62.83 | 65.45 | 78.93 | 62.83 | 64.89 | 65.82 | 63.06 |
| RANSAC-adaptive | 0.30 | 0.53 | 0.83 | 1.57 | 2.03 | 3.43 | 2.04 | 1.19 |
| IRLS-GM | 3.04 | 2.65 | 2.69 | 3.31 | 2.68 | 2.68 | 2.79 | 2.68 |
| GNC-GM | 0.86 | 2.17 | 2.16 | 2.37 | 2.03 | 2.03 | 2.16 | 2.07 |

### boundary outliers: success rate

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
| Kabsch | 0.25 | 0.26 | 0.23 | 0.27 | 0.31 | 0.26 | 0.27 | 0.37 |
| RANSAC-1000 | 0.25 | 0.26 | 0.26 | 0.32 | 0.48 | 0.45 | 0.56 | 0.74 |
| RANSAC-adaptive | 0.26 | 0.27 | 0.23 | 0.34 | 0.41 | 0.39 | 0.51 | 0.81 |
| IRLS-GM | 0.23 | 0.22 | 0.19 | 0.22 | 0.30 | 0.21 | 0.23 | 0.25 |
| GNC-GM | 0.23 | 0.22 | 0.19 | 0.22 | 0.30 | 0.21 | 0.23 | 0.25 |

### boundary outliers: runtime p50 (ms)

| method | 0% | 10% | 20% | 30% | 40% | 50% | 60% | 70% |
|---|---|---|---|---|---|---|---|---|
| Kabsch | 0.11 | 0.12 | 0.10 | 0.11 | 0.10 | 0.10 | 0.12 | 0.12 |
| RANSAC-1000 | 62.87 | 71.56 | 61.50 | 65.54 | 61.30 | 61.43 | 61.91 | 72.91 |
| RANSAC-adaptive | 0.32 | 0.54 | 0.64 | 0.97 | 1.22 | 1.83 | 2.90 | 4.53 |
| IRLS-GM | 2.81 | 2.96 | 2.67 | 2.68 | 2.67 | 2.64 | 2.65 | 2.98 |
| GNC-GM | 0.86 | 1.07 | 0.96 | 0.98 | 0.96 | 0.95 | 0.95 | 1.14 |


## sun3d-home_at

Selected pairs: 31-32 (overlap 100%, med res 12.5 mm), 14-16 (overlap 100%, med res 12.8 mm), 24-29 (overlap 100%, med res 12.9 mm), 1-2 (overlap 100%, med res 13.3 mm) ...

### random outliers: success rate

| method | 0% | 10% | 20% | 30% | 40% | 50% | 60% | 70% |
|---|---|---|---|---|---|---|---|---|
| Kabsch | 100 | 100 | 50 | 15 | 30 | 5 | 0 | 0 |
| RANSAC-1000 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| RANSAC-adaptive | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| IRLS-GM | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| GNC-GM | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |

### random outliers: median rotation error (deg)

| method | 0% | 10% | 20% | 30% | 40% | 50% | 60% | 70% |
|---|---|---|---|---|---|---|---|---|
| Kabsch | 0.17 | 1.32 | 2.08 | 2.49 | 4.73 | 4.77 | 6.46 | 10.21 |
| RANSAC-1000 | 0.16 | 0.21 | 0.29 | 0.17 | 0.27 | 0.21 | 0.28 | 0.23 |
| RANSAC-adaptive | 0.17 | 0.21 | 0.28 | 0.18 | 0.25 | 0.21 | 0.28 | 0.21 |
| IRLS-GM | 0.15 | 0.19 | 0.26 | 0.18 | 0.23 | 0.20 | 0.27 | 0.21 |
| GNC-GM | 0.15 | 0.19 | 0.26 | 0.18 | 0.23 | 0.20 | 0.27 | 0.21 |

### random outliers: runtime p50 (ms)

| method | 0% | 10% | 20% | 30% | 40% | 50% | 60% | 70% |
|---|---|---|---|---|---|---|---|---|
| Kabsch | 0.10 | 0.12 | 0.10 | 0.13 | 0.13 | 0.13 | 0.12 | 0.13 |
| RANSAC-1000 | 61.07 | 61.60 | 62.15 | 62.59 | 62.60 | 65.25 | 62.90 | 64.06 |
| RANSAC-adaptive | 0.31 | 0.56 | 0.80 | 1.27 | 1.92 | 3.65 | 7.00 | 16.71 |
| IRLS-GM | 2.67 | 2.64 | 2.67 | 2.63 | 2.64 | 2.70 | 2.70 | 2.75 |
| GNC-GM | 0.81 | 2.25 | 2.26 | 2.26 | 2.22 | 2.33 | 2.31 | 2.37 |

### coherent outliers: success rate

| method | 0% | 10% | 20% | 30% | 40% | 50% | 60% | 70% |
|---|---|---|---|---|---|---|---|---|
| Kabsch | 100 | 15 | 0 | 0 | 0 | 0 | 0 | 0 |
| RANSAC-1000 | 100 | 100 | 100 | 100 | 100 | 0 | 0 | 0 |
| RANSAC-adaptive | 100 | 100 | 100 | 100 | 100 | 0 | 0 | 0 |
| IRLS-GM | 100 | 100 | 100 | 100 | 100 | 55 | 0 | 0 |
| GNC-GM | 100 | 100 | 100 | 100 | 100 | 35 | 0 | 0 |

### coherent outliers: median rotation error (deg)

| method | 0% | 10% | 20% | 30% | 40% | 50% | 60% | 70% |
|---|---|---|---|---|---|---|---|---|
| Kabsch | 0.28 | 5.07 | 9.08 | 14.82 | 14.95 | 23.17 | 28.25 | 33.32 |
| RANSAC-1000 | 0.27 | 0.18 | 0.19 | 0.40 | 0.27 | 43.05 | 46.73 | 47.60 |
| RANSAC-adaptive | 0.31 | 0.16 | 0.20 | 0.45 | 0.27 | 43.05 | 46.73 | 47.60 |
| IRLS-GM | 0.33 | 0.17 | 0.18 | 0.49 | 0.26 | 0.88 | 46.73 | 47.60 |
| GNC-GM | 0.33 | 0.17 | 0.18 | 0.49 | 0.26 | 37.39 | 46.73 | 47.60 |

### coherent outliers: runtime p50 (ms)

| method | 0% | 10% | 20% | 30% | 40% | 50% | 60% | 70% |
|---|---|---|---|---|---|---|---|---|
| Kabsch | 0.12 | 0.12 | 0.12 | 0.11 | 0.09 | 0.11 | 0.12 | 0.10 |
| RANSAC-1000 | 63.87 | 78.93 | 78.38 | 61.11 | 61.05 | 62.60 | 65.70 | 62.36 |
| RANSAC-adaptive | 0.33 | 0.61 | 0.97 | 1.22 | 1.96 | 3.34 | 1.94 | 1.18 |
| IRLS-GM | 2.84 | 3.03 | 3.34 | 2.59 | 2.61 | 2.66 | 2.77 | 2.66 |
| GNC-GM | 0.87 | 2.35 | 2.58 | 2.08 | 2.02 | 2.00 | 2.13 | 2.19 |

### boundary outliers: success rate

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
| Kabsch | 0.23 | 0.20 | 0.31 | 0.22 | 0.28 | 0.25 | 0.24 | 0.27 |
| RANSAC-1000 | 0.23 | 0.23 | 0.35 | 0.35 | 0.42 | 0.35 | 0.36 | 0.60 |
| RANSAC-adaptive | 0.23 | 0.20 | 0.49 | 0.45 | 0.43 | 0.38 | 0.36 | 0.69 |
| IRLS-GM | 0.22 | 0.18 | 0.45 | 0.37 | 0.28 | 0.26 | 0.21 | 0.25 |
| GNC-GM | 0.22 | 0.18 | 0.45 | 0.37 | 0.28 | 0.26 | 0.21 | 0.25 |

### boundary outliers: runtime p50 (ms)

| method | 0% | 10% | 20% | 30% | 40% | 50% | 60% | 70% |
|---|---|---|---|---|---|---|---|---|
| Kabsch | 0.11 | 0.10 | 0.11 | 0.12 | 0.10 | 0.12 | 0.12 | 0.09 |
| RANSAC-1000 | 34.72 | 62.19 | 63.73 | 72.08 | 68.83 | 78.63 | 72.76 | 61.11 |
| RANSAC-adaptive | 0.29 | 0.48 | 0.68 | 0.99 | 1.36 | 2.16 | 2.83 | 3.98 |
| IRLS-GM | 2.70 | 2.65 | 2.72 | 2.71 | 2.85 | 3.37 | 3.03 | 2.64 |
| GNC-GM | 0.82 | 0.97 | 0.99 | 1.01 | 1.02 | 1.21 | 1.08 | 0.95 |


## Caveats

* GT-NN base correspondences use the ground-truth transform to establish overlap; they are oracle matches, not a learned descriptor. Real scanner noise, occlusion, and partial overlap are present, so the clean success rate is not 100% for a naive Kabsch.
* The transform direction in `gt.log` was verified empirically: it maps fragment `id2` into `id1`; the runner inverts it. See `results/verify_gt_direction.json`.
* Coherent outliers at >=60% are expected to make the robust global optimum wrong (model identifiability), exactly as in the synthetic protocol.
