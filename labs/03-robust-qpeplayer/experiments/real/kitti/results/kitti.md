# KITTI raw drive 0001: real LiDAR robust 3D-3D registration

Ego motion from OXTS (KITTI devkit Mercator + roll/pitch/yaw), mapped to Velodyne with `calib_imu_to_velo`. N=300, c=0.25 m, tau=0.45 m, 20 trials/cell, 10 consecutive-frame pairs. Base matches are GT-mapped nearest neighbours on real LiDAR (sparse, occluded, dynamic), with controlled outliers injected.

## random outliers: success rate

| method | 0% | 10% | 20% | 30% | 40% | 50% | 60% | 70% |
|---|---|---|---|---|---|---|---|---|
| Kabsch | 100 | 10 | 0 | 5 | 0 | 0 | 0 | 0 |
| RANSAC-1000 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| RANSAC-adaptive | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| IRLS-GM | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| GNC-GM | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |

## random outliers: median rotation error (deg)

| method | 0% | 10% | 20% | 30% | 40% | 50% | 60% | 70% |
|---|---|---|---|---|---|---|---|---|
| Kabsch | 0.04 | 0.49 | 0.91 | 1.75 | 1.44 | 1.80 | 4.29 | 5.77 |
| RANSAC-1000 | 0.04 | 0.03 | 0.04 | 0.04 | 0.05 | 0.05 | 0.05 | 0.06 |
| RANSAC-adaptive | 0.04 | 0.03 | 0.04 | 0.04 | 0.05 | 0.04 | 0.05 | 0.06 |
| IRLS-GM | 0.03 | 0.03 | 0.03 | 0.04 | 0.04 | 0.04 | 0.04 | 0.06 |
| GNC-GM | 0.03 | 0.03 | 0.03 | 0.04 | 0.04 | 0.04 | 0.04 | 0.06 |

## random outliers: runtime p50 (ms)

| method | 0% | 10% | 20% | 30% | 40% | 50% | 60% | 70% |
|---|---|---|---|---|---|---|---|---|
| Kabsch | 0.07 | 0.07 | 0.08 | 0.07 | 0.07 | 0.07 | 0.07 | 0.07 |
| RANSAC-1000 | 61.20 | 63.57 | 62.06 | 62.56 | 62.66 | 61.60 | 61.04 | 61.06 |
| RANSAC-adaptive | 0.62 | 0.86 | 1.29 | 1.91 | 3.19 | 5.64 | 10.59 | 25.75 |
| IRLS-GM | 2.62 | 2.63 | 2.64 | 2.67 | 2.67 | 2.65 | 2.58 | 2.62 |
| GNC-GM | 0.94 | 2.73 | 2.76 | 2.78 | 2.78 | 2.79 | 2.74 | 2.75 |

## coherent outliers: success rate

| method | 0% | 10% | 20% | 30% | 40% | 50% | 60% | 70% |
|---|---|---|---|---|---|---|---|---|
| Kabsch | 100 | 10 | 0 | 0 | 0 | 0 | 0 | 0 |
| RANSAC-1000 | 100 | 100 | 100 | 100 | 100 | 0 | 0 | 0 |
| RANSAC-adaptive | 100 | 100 | 100 | 100 | 100 | 0 | 0 | 0 |
| IRLS-GM | 100 | 100 | 100 | 100 | 100 | 45 | 0 | 0 |
| GNC-GM | 100 | 100 | 100 | 100 | 100 | 35 | 0 | 0 |

## coherent outliers: median rotation error (deg)

| method | 0% | 10% | 20% | 30% | 40% | 50% | 60% | 70% |
|---|---|---|---|---|---|---|---|---|
| Kabsch | 0.05 | 4.00 | 8.34 | 12.65 | 18.80 | 20.07 | 26.82 | 34.85 |
| RANSAC-1000 | 0.04 | 0.04 | 0.04 | 0.04 | 0.04 | 39.90 | 46.73 | 50.47 |
| RANSAC-adaptive | 0.04 | 0.05 | 0.04 | 0.04 | 0.04 | 39.90 | 46.73 | 50.47 |
| IRLS-GM | 0.04 | 0.04 | 0.03 | 0.04 | 0.04 | 26.14 | 46.73 | 50.47 |
| GNC-GM | 0.04 | 0.04 | 0.03 | 0.04 | 0.04 | 34.56 | 46.73 | 50.47 |

## coherent outliers: runtime p50 (ms)

| method | 0% | 10% | 20% | 30% | 40% | 50% | 60% | 70% |
|---|---|---|---|---|---|---|---|---|
| Kabsch | 0.07 | 0.07 | 0.07 | 0.07 | 0.07 | 0.08 | 0.08 | 0.08 |
| RANSAC-1000 | 61.11 | 61.25 | 61.35 | 67.05 | 62.31 | 67.27 | 68.29 | 62.49 |
| RANSAC-adaptive | 0.61 | 0.94 | 1.27 | 2.20 | 3.16 | 3.76 | 2.01 | 1.20 |
| IRLS-GM | 2.62 | 2.58 | 2.60 | 2.77 | 2.63 | 2.75 | 2.82 | 2.64 |
| GNC-GM | 0.94 | 2.35 | 2.34 | 2.41 | 2.27 | 2.31 | 2.50 | 2.34 |

## boundary outliers: success rate

| method | 0% | 10% | 20% | 30% | 40% | 50% | 60% | 70% |
|---|---|---|---|---|---|---|---|---|
| Kabsch | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| RANSAC-1000 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| RANSAC-adaptive | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 95 |
| IRLS-GM | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| GNC-GM | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |

## boundary outliers: median rotation error (deg)

| method | 0% | 10% | 20% | 30% | 40% | 50% | 60% | 70% |
|---|---|---|---|---|---|---|---|---|
| Kabsch | 0.04 | 0.04 | 0.05 | 0.06 | 0.06 | 0.06 | 0.06 | 0.07 |
| RANSAC-1000 | 0.04 | 0.04 | 0.05 | 0.06 | 0.07 | 0.08 | 0.11 | 0.14 |
| RANSAC-adaptive | 0.04 | 0.05 | 0.05 | 0.05 | 0.08 | 0.09 | 0.12 | 0.18 |
| IRLS-GM | 0.04 | 0.03 | 0.04 | 0.04 | 0.04 | 0.04 | 0.05 | 0.07 |
| GNC-GM | 0.04 | 0.03 | 0.04 | 0.04 | 0.04 | 0.04 | 0.05 | 0.07 |

## boundary outliers: runtime p50 (ms)

| method | 0% | 10% | 20% | 30% | 40% | 50% | 60% | 70% |
|---|---|---|---|---|---|---|---|---|
| Kabsch | 0.07 | 0.07 | 0.07 | 0.08 | 0.08 | 0.07 | 0.07 | 0.07 |
| RANSAC-1000 | 62.31 | 68.53 | 62.64 | 63.06 | 63.10 | 62.13 | 62.05 | 61.71 |
| RANSAC-adaptive | 0.62 | 0.83 | 1.02 | 1.35 | 1.98 | 2.61 | 3.97 | 6.25 |
| IRLS-GM | 2.65 | 2.86 | 2.62 | 2.63 | 2.66 | 2.63 | 2.63 | 2.63 |
| GNC-GM | 0.96 | 0.99 | 0.96 | 0.95 | 0.96 | 0.96 | 0.94 | 0.95 |

## Caveats

* KITTI raw provides OXTS, not per-frame ground-truth poses for this sync drive; the relative transform is derived from GPS/IMU and is accurate to the OXTS/calibration level, not to the millimetre. Clean Kabsch reaches ~0.02 deg / ~0.01-0.03 m on the sampled base pairs, which is the practical noise floor.
* LiDAR is sparse (~0.1 m point spacing), so c=0.25 m is used; a 3DMatch-style c=0.05 m would discard most correct matches.
* Base correspondences are GT-mapped NN, not a learned descriptor; this isolates the robust outer layer, like the 3DMatch oracle table.
* Consecutive frames at 10 Hz have |t|≈1.6-2.2 m; the 0.1 m success threshold is absolute pose error, not relative-to-motion error.
