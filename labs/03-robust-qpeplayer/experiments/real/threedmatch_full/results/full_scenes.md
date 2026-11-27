# 3DMatch full official set: robust 3D-3D registration

All eight official 3DMatch geometric-registration scene-fragment sets. Base correspondences are GT-mapped nearest neighbours on real overlapping fragments (oracle matches with real scan/occlusion residual); controlled outliers are then injected. N=300, c=0.05 m, tau=0.06 m, voxel=0.03 m, 20 trials/cell, trials sample one of 8 high-overlap fragment pairs per scene.

GT direction was verified on every scene (`verify_gt_direction_full.json`): the stored 4x4 maps id2->id1, so the runner inverts it. Base matches are oracle correspondences, not a learned descriptor.

## Pooled across 8 scenes

Pooled over 8 scenes x 20 trials/cell (equal scene weight).

### random: success rate (%)

| method | 0% | 10% | 20% | 30% | 40% | 50% | 60% | 70% |
|---|---|---|---|---|---|---|---|---|
| Kabsch | 100 | 91 | 65 | 37 | 21 | 15 | 5 | 2 |
| RANSAC-1000 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| RANSAC-adaptive | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| IRLS-GM | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 99 |
| GNC-GM | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |

### random: median rotation error (deg)

| method | 0% | 10% | 20% | 30% | 40% | 50% | 60% | 70% |
|---|---|---|---|---|---|---|---|---|
| Kabsch | 0.22 | 1.33 | 2.04 | 2.84 | 3.82 | 5.13 | 6.75 | 9.77 |
| RANSAC-1000 | 0.22 | 0.24 | 0.23 | 0.24 | 0.23 | 0.22 | 0.24 | 0.27 |
| RANSAC-adaptive | 0.22 | 0.24 | 0.22 | 0.22 | 0.23 | 0.21 | 0.24 | 0.27 |
| IRLS-GM | 0.21 | 0.22 | 0.22 | 0.21 | 0.21 | 0.19 | 0.23 | 0.25 |
| GNC-GM | 0.21 | 0.22 | 0.22 | 0.21 | 0.21 | 0.19 | 0.23 | 0.25 |

### coherent: success rate (%)

| method | 0% | 10% | 20% | 30% | 40% | 50% | 60% | 70% |
|---|---|---|---|---|---|---|---|---|
| Kabsch | 100 | 18 | 0 | 0 | 0 | 0 | 0 | 0 |
| RANSAC-1000 | 100 | 100 | 100 | 100 | 100 | 3 | 0 | 0 |
| RANSAC-adaptive | 100 | 100 | 100 | 100 | 100 | 3 | 0 | 0 |
| IRLS-GM | 100 | 100 | 100 | 100 | 100 | 43 | 0 | 0 |
| GNC-GM | 100 | 100 | 100 | 100 | 100 | 46 | 0 | 0 |

### coherent: median rotation error (deg)

| method | 0% | 10% | 20% | 30% | 40% | 50% | 60% | 70% |
|---|---|---|---|---|---|---|---|---|
| Kabsch | 0.21 | 4.58 | 9.30 | 13.64 | 17.72 | 22.82 | 25.21 | 33.02 |
| RANSAC-1000 | 0.22 | 0.22 | 0.23 | 0.22 | 0.24 | 46.49 | 42.11 | 46.14 |
| RANSAC-adaptive | 0.21 | 0.22 | 0.22 | 0.22 | 0.25 | 46.49 | 42.11 | 46.14 |
| IRLS-GM | 0.20 | 0.21 | 0.21 | 0.20 | 0.22 | 33.45 | 42.11 | 46.14 |
| GNC-GM | 0.20 | 0.21 | 0.21 | 0.20 | 0.22 | 31.55 | 42.11 | 46.14 |

### boundary: success rate (%)

| method | 0% | 10% | 20% | 30% | 40% | 50% | 60% | 70% |
|---|---|---|---|---|---|---|---|---|
| Kabsch | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| RANSAC-1000 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| RANSAC-adaptive | 100 | 100 | 100 | 100 | 100 | 100 | 99 | 100 |
| IRLS-GM | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| GNC-GM | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |

### boundary: median rotation error (deg)

| method | 0% | 10% | 20% | 30% | 40% | 50% | 60% | 70% |
|---|---|---|---|---|---|---|---|---|
| Kabsch | 0.24 | 0.21 | 0.24 | 0.25 | 0.23 | 0.26 | 0.28 | 0.24 |
| RANSAC-1000 | 0.24 | 0.24 | 0.27 | 0.28 | 0.30 | 0.37 | 0.47 | 0.52 |
| RANSAC-adaptive | 0.23 | 0.23 | 0.25 | 0.26 | 0.29 | 0.36 | 0.43 | 0.60 |
| IRLS-GM | 0.22 | 0.21 | 0.22 | 0.22 | 0.21 | 0.22 | 0.24 | 0.27 |
| GNC-GM | 0.22 | 0.21 | 0.22 | 0.22 | 0.21 | 0.22 | 0.24 | 0.27 |

### runtime p50 (ms), random outliers

| method | 0% | 10% | 20% | 30% | 40% | 50% | 60% | 70% |
|---|---|---|---|---|---|---|---|---|
| Kabsch | 0.11 | 0.11 | 0.11 | 0.11 | 0.11 | 0.11 | 0.10 | 0.10 |
| RANSAC-1000 | 62.80 | 66.10 | 64.76 | 70.66 | 73.38 | 64.63 | 62.12 | 62.62 |
| RANSAC-adaptive | 0.34 | 0.58 | 0.84 | 1.31 | 2.17 | 3.67 | 7.03 | 16.42 |
| IRLS-GM | 2.71 | 2.68 | 2.67 | 2.67 | 2.71 | 2.67 | 2.64 | 2.65 |
| GNC-GM | 0.83 | 2.30 | 2.31 | 2.30 | 2.36 | 2.31 | 2.27 | 2.26 |

## Per-scene

### redkitchen (`7-scenes-redkitchen`)

Selected pairs: 50-52 (overlap 99%, med res 11.9 mm), 51-52 (overlap 90%, med res 12.1 mm), 16-17 (overlap 90%, med res 13.4 mm), 6-7 (overlap 90%, med res 15.9 mm) ...

#### random: success rate (%)

| method | 0% | 10% | 20% | 30% | 40% | 50% | 60% | 70% |
|---|---|---|---|---|---|---|---|---|
| Kabsch | 100 | 95 | 75 | 40 | 30 | 30 | 5 | 0 |
| RANSAC-1000 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| RANSAC-adaptive | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| IRLS-GM | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 95 |
| GNC-GM | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |

#### coherent: success rate (%)

| method | 0% | 10% | 20% | 30% | 40% | 50% | 60% | 70% |
|---|---|---|---|---|---|---|---|---|
| Kabsch | 100 | 20 | 0 | 0 | 0 | 0 | 0 | 0 |
| RANSAC-1000 | 100 | 100 | 100 | 100 | 100 | 0 | 0 | 0 |
| RANSAC-adaptive | 100 | 100 | 100 | 100 | 100 | 0 | 0 | 0 |
| IRLS-GM | 100 | 100 | 100 | 100 | 100 | 50 | 0 | 0 |
| GNC-GM | 100 | 100 | 100 | 100 | 100 | 25 | 0 | 0 |

#### boundary: success rate (%)

| method | 0% | 10% | 20% | 30% | 40% | 50% | 60% | 70% |
|---|---|---|---|---|---|---|---|---|
| Kabsch | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| RANSAC-1000 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| RANSAC-adaptive | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| IRLS-GM | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| GNC-GM | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |


### sun3d-home_at (`sun3d-home_at-home_at_scan1_2013_jan_1`)

Selected pairs: 44-45 (overlap 98%, med res 14.0 mm), 24-29 (overlap 91%, med res 13.2 mm), 1-2 (overlap 90%, med res 13.3 mm), 43-45 (overlap 90%, med res 16.0 mm) ...

#### random: success rate (%)

| method | 0% | 10% | 20% | 30% | 40% | 50% | 60% | 70% |
|---|---|---|---|---|---|---|---|---|
| Kabsch | 100 | 100 | 85 | 30 | 35 | 10 | 5 | 5 |
| RANSAC-1000 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| RANSAC-adaptive | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| IRLS-GM | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| GNC-GM | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |

#### coherent: success rate (%)

| method | 0% | 10% | 20% | 30% | 40% | 50% | 60% | 70% |
|---|---|---|---|---|---|---|---|---|
| Kabsch | 100 | 25 | 0 | 0 | 0 | 0 | 0 | 0 |
| RANSAC-1000 | 100 | 100 | 100 | 100 | 100 | 5 | 0 | 0 |
| RANSAC-adaptive | 100 | 100 | 100 | 100 | 100 | 5 | 0 | 0 |
| IRLS-GM | 100 | 100 | 100 | 100 | 100 | 40 | 0 | 0 |
| GNC-GM | 100 | 100 | 100 | 100 | 100 | 60 | 0 | 0 |

#### boundary: success rate (%)

| method | 0% | 10% | 20% | 30% | 40% | 50% | 60% | 70% |
|---|---|---|---|---|---|---|---|---|
| Kabsch | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| RANSAC-1000 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| RANSAC-adaptive | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| IRLS-GM | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| GNC-GM | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |


### sun3d-hotel_uc (`sun3d-hotel_uc-scan3`)

Selected pairs: 31-33 (overlap 94%, med res 16.1 mm), 31-32 (overlap 90%, med res 14.9 mm), 40-41 (overlap 90%, med res 13.2 mm), 5-6 (overlap 89%, med res 16.4 mm) ...

#### random: success rate (%)

| method | 0% | 10% | 20% | 30% | 40% | 50% | 60% | 70% |
|---|---|---|---|---|---|---|---|---|
| Kabsch | 100 | 100 | 70 | 40 | 25 | 10 | 0 | 0 |
| RANSAC-1000 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| RANSAC-adaptive | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| IRLS-GM | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| GNC-GM | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |

#### coherent: success rate (%)

| method | 0% | 10% | 20% | 30% | 40% | 50% | 60% | 70% |
|---|---|---|---|---|---|---|---|---|
| Kabsch | 100 | 30 | 0 | 0 | 0 | 0 | 0 | 0 |
| RANSAC-1000 | 100 | 100 | 100 | 100 | 100 | 10 | 0 | 0 |
| RANSAC-adaptive | 100 | 100 | 100 | 100 | 100 | 10 | 0 | 0 |
| IRLS-GM | 100 | 100 | 100 | 100 | 100 | 45 | 0 | 0 |
| GNC-GM | 100 | 100 | 100 | 100 | 100 | 50 | 0 | 0 |

#### boundary: success rate (%)

| method | 0% | 10% | 20% | 30% | 40% | 50% | 60% | 70% |
|---|---|---|---|---|---|---|---|---|
| Kabsch | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| RANSAC-1000 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| RANSAC-adaptive | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| IRLS-GM | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| GNC-GM | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |


### sun3d-home_md (`sun3d-home_md-home_md_scan9_2012_sep_30`)

Selected pairs: 41-47 (overlap 95%, med res 12.4 mm), 56-57 (overlap 93%, med res 13.3 mm), 39-46 (overlap 91%, med res 13.6 mm), 18-19 (overlap 91%, med res 11.9 mm) ...

#### random: success rate (%)

| method | 0% | 10% | 20% | 30% | 40% | 50% | 60% | 70% |
|---|---|---|---|---|---|---|---|---|
| Kabsch | 100 | 80 | 60 | 40 | 25 | 25 | 0 | 5 |
| RANSAC-1000 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| RANSAC-adaptive | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| IRLS-GM | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| GNC-GM | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |

#### coherent: success rate (%)

| method | 0% | 10% | 20% | 30% | 40% | 50% | 60% | 70% |
|---|---|---|---|---|---|---|---|---|
| Kabsch | 100 | 15 | 0 | 0 | 0 | 0 | 0 | 0 |
| RANSAC-1000 | 100 | 100 | 100 | 100 | 100 | 0 | 0 | 0 |
| RANSAC-adaptive | 100 | 100 | 100 | 100 | 100 | 0 | 0 | 0 |
| IRLS-GM | 100 | 100 | 100 | 100 | 100 | 40 | 0 | 0 |
| GNC-GM | 100 | 100 | 100 | 100 | 100 | 65 | 0 | 0 |

#### boundary: success rate (%)

| method | 0% | 10% | 20% | 30% | 40% | 50% | 60% | 70% |
|---|---|---|---|---|---|---|---|---|
| Kabsch | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| RANSAC-1000 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| RANSAC-adaptive | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| IRLS-GM | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| GNC-GM | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |


### sun3d-hotel1 (`sun3d-hotel_umd-maryland_hotel1`)

Selected pairs: 36-37 (overlap 95%, med res 12.7 mm), 39-56 (overlap 95%, med res 18.5 mm), 40-56 (overlap 92%, med res 18.3 mm), 21-31 (overlap 91%, med res 23.4 mm) ...

#### random: success rate (%)

| method | 0% | 10% | 20% | 30% | 40% | 50% | 60% | 70% |
|---|---|---|---|---|---|---|---|---|
| Kabsch | 100 | 80 | 25 | 10 | 5 | 0 | 0 | 5 |
| RANSAC-1000 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| RANSAC-adaptive | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| IRLS-GM | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| GNC-GM | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |

#### coherent: success rate (%)

| method | 0% | 10% | 20% | 30% | 40% | 50% | 60% | 70% |
|---|---|---|---|---|---|---|---|---|
| Kabsch | 100 | 15 | 0 | 0 | 0 | 0 | 0 | 0 |
| RANSAC-1000 | 100 | 100 | 100 | 100 | 100 | 0 | 0 | 0 |
| RANSAC-adaptive | 100 | 100 | 100 | 100 | 100 | 0 | 0 | 0 |
| IRLS-GM | 100 | 100 | 100 | 100 | 100 | 55 | 0 | 0 |
| GNC-GM | 100 | 100 | 100 | 100 | 100 | 25 | 0 | 0 |

#### boundary: success rate (%)

| method | 0% | 10% | 20% | 30% | 40% | 50% | 60% | 70% |
|---|---|---|---|---|---|---|---|---|
| Kabsch | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| RANSAC-1000 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| RANSAC-adaptive | 100 | 100 | 100 | 100 | 100 | 100 | 95 | 100 |
| IRLS-GM | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| GNC-GM | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |


### sun3d-hotel3 (`sun3d-hotel_umd-maryland_hotel3`)

Selected pairs: 11-13 (overlap 79%, med res 16.2 mm), 33-34 (overlap 78%, med res 11.9 mm), 22-23 (overlap 75%, med res 14.8 mm), 24-28 (overlap 73%, med res 15.4 mm) ...

#### random: success rate (%)

| method | 0% | 10% | 20% | 30% | 40% | 50% | 60% | 70% |
|---|---|---|---|---|---|---|---|---|
| Kabsch | 100 | 75 | 45 | 45 | 10 | 20 | 5 | 0 |
| RANSAC-1000 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| RANSAC-adaptive | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| IRLS-GM | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| GNC-GM | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |

#### coherent: success rate (%)

| method | 0% | 10% | 20% | 30% | 40% | 50% | 60% | 70% |
|---|---|---|---|---|---|---|---|---|
| Kabsch | 100 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| RANSAC-1000 | 100 | 100 | 100 | 100 | 100 | 5 | 0 | 0 |
| RANSAC-adaptive | 100 | 100 | 100 | 100 | 100 | 5 | 0 | 0 |
| IRLS-GM | 100 | 100 | 100 | 100 | 100 | 55 | 0 | 0 |
| GNC-GM | 100 | 100 | 100 | 100 | 100 | 30 | 0 | 0 |

#### boundary: success rate (%)

| method | 0% | 10% | 20% | 30% | 40% | 50% | 60% | 70% |
|---|---|---|---|---|---|---|---|---|
| Kabsch | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| RANSAC-1000 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| RANSAC-adaptive | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| IRLS-GM | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| GNC-GM | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |


### sun3d-studyroom (`sun3d-mit_76_studyroom-76-1studyroom2`)

Selected pairs: 41-42 (overlap 96%, med res 15.4 mm), 36-37 (overlap 96%, med res 13.9 mm), 2-48 (overlap 92%, med res 14.9 mm), 13-56 (overlap 91%, med res 13.8 mm) ...

#### random: success rate (%)

| method | 0% | 10% | 20% | 30% | 40% | 50% | 60% | 70% |
|---|---|---|---|---|---|---|---|---|
| Kabsch | 100 | 100 | 75 | 40 | 5 | 10 | 10 | 0 |
| RANSAC-1000 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| RANSAC-adaptive | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| IRLS-GM | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| GNC-GM | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |

#### coherent: success rate (%)

| method | 0% | 10% | 20% | 30% | 40% | 50% | 60% | 70% |
|---|---|---|---|---|---|---|---|---|
| Kabsch | 100 | 10 | 0 | 0 | 0 | 0 | 0 | 0 |
| RANSAC-1000 | 100 | 100 | 100 | 100 | 100 | 0 | 0 | 0 |
| RANSAC-adaptive | 100 | 100 | 100 | 100 | 100 | 0 | 0 | 0 |
| IRLS-GM | 100 | 100 | 100 | 100 | 100 | 40 | 0 | 0 |
| GNC-GM | 100 | 100 | 100 | 100 | 100 | 65 | 0 | 0 |

#### boundary: success rate (%)

| method | 0% | 10% | 20% | 30% | 40% | 50% | 60% | 70% |
|---|---|---|---|---|---|---|---|---|
| Kabsch | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| RANSAC-1000 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| RANSAC-adaptive | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| IRLS-GM | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| GNC-GM | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |


### sun3d-mit_lab_hj (`sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika`)

Selected pairs: 32-36 (overlap 97%, med res 15.2 mm), 32-33 (overlap 95%, med res 15.3 mm), 34-36 (overlap 95%, med res 16.1 mm), 33-37 (overlap 93%, med res 15.3 mm) ...

#### random: success rate (%)

| method | 0% | 10% | 20% | 30% | 40% | 50% | 60% | 70% |
|---|---|---|---|---|---|---|---|---|
| Kabsch | 100 | 100 | 85 | 50 | 30 | 15 | 15 | 0 |
| RANSAC-1000 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| RANSAC-adaptive | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| IRLS-GM | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| GNC-GM | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |

#### coherent: success rate (%)

| method | 0% | 10% | 20% | 30% | 40% | 50% | 60% | 70% |
|---|---|---|---|---|---|---|---|---|
| Kabsch | 100 | 25 | 0 | 0 | 0 | 0 | 0 | 0 |
| RANSAC-1000 | 100 | 100 | 100 | 100 | 100 | 5 | 0 | 0 |
| RANSAC-adaptive | 100 | 100 | 100 | 100 | 100 | 5 | 0 | 0 |
| IRLS-GM | 100 | 100 | 100 | 100 | 100 | 20 | 0 | 0 |
| GNC-GM | 100 | 100 | 100 | 100 | 100 | 50 | 0 | 0 |

#### boundary: success rate (%)

| method | 0% | 10% | 20% | 30% | 40% | 50% | 60% | 70% |
|---|---|---|---|---|---|---|---|---|
| Kabsch | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| RANSAC-1000 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| RANSAC-adaptive | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| IRLS-GM | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| GNC-GM | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |


## Caveats

* GT-NN base correspondences use the ground-truth transform to establish overlap: they are oracle matches, not a learned descriptor. Real scanner noise, occlusion and partial overlap are present.
* Coherent outliers at >=60% are expected to defeat every method (model identifiability), exactly as in the synthetic protocol.
* For low-overlap pairs the valid GT-NN set can be smaller than N; the runner resamples with replacement. This is only relevant for the high-overlap table in rare cases; the dedicated low/mid-overlap study reports the raw counts.
