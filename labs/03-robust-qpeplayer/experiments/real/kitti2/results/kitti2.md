# KITTI raw drive 0002: real LiDAR robust 3D-3D registration

Ego motion from OXTS (KITTI devkit Mercator + roll/pitch/yaw), mapped to Velodyne with `calib_imu_to_velo`. N=300, c=0.25 m, tau=0.45 m, 20 trials/cell, 10 consecutive-frame pairs. Base matches are GT-mapped nearest neighbours on real LiDAR (sparse, occluded, dynamic), with controlled outliers injected.

## random outliers: success rate

| method | 0% | 10% | 20% | 30% | 40% | 50% | 60% | 70% |
|---|---|---|---|---|---|---|---|---|
| Kabsch | 100 | 10 | 0 | 0 | 0 | 0 | 0 | 0 |
| RANSAC-1000 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| RANSAC-adaptive | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| IRLS-GM | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| GNC-GM | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |

## random outliers: median rotation error (deg)

| method | 0% | 10% | 20% | 30% | 40% | 50% | 60% | 70% |
|---|---|---|---|---|---|---|---|---|
| Kabsch | 0.04 | 0.70 | 0.77 | 0.83 | 1.24 | 2.11 | 3.22 | 5.13 |
| RANSAC-1000 | 0.05 | 0.05 | 0.05 | 0.05 | 0.05 | 0.04 | 0.07 | 0.07 |
| RANSAC-adaptive | 0.05 | 0.06 | 0.06 | 0.05 | 0.06 | 0.05 | 0.07 | 0.06 |
| IRLS-GM | 0.04 | 0.05 | 0.05 | 0.04 | 0.05 | 0.05 | 0.07 | 0.05 |
| GNC-GM | 0.04 | 0.05 | 0.05 | 0.04 | 0.05 | 0.05 | 0.07 | 0.05 |

## random outliers: runtime p50 (ms)

| method | 0% | 10% | 20% | 30% | 40% | 50% | 60% | 70% |
|---|---|---|---|---|---|---|---|---|
| Kabsch | 0.07 | 0.07 | 0.07 | 0.09 | 0.13 | 0.14 | 0.14 | 0.14 |
| RANSAC-1000 | 61.95 | 62.38 | 65.50 | 85.97 | 118.87 | 118.13 | 115.46 | 115.52 |
| RANSAC-adaptive | 0.67 | 0.94 | 1.31 | 3.07 | 5.54 | 10.77 | 19.38 | 48.22 |
| IRLS-GM | 2.61 | 2.66 | 2.69 | 3.75 | 4.83 | 4.87 | 4.81 | 4.83 |
| GNC-GM | 0.95 | 2.70 | 2.80 | 4.14 | 5.07 | 5.01 | 4.95 | 4.97 |

## coherent outliers: success rate

| method | 0% | 10% | 20% | 30% | 40% | 50% | 60% | 70% |
|---|---|---|---|---|---|---|---|---|
| Kabsch | 100 | 5 | 0 | 0 | 0 | 0 | 0 | 0 |
| RANSAC-1000 | 100 | 100 | 100 | 100 | 100 | 0 | 0 | 0 |
| RANSAC-adaptive | 100 | 100 | 100 | 100 | 100 | 0 | 0 | 0 |
| IRLS-GM | 100 | 100 | 100 | 100 | 100 | 30 | 5 | 0 |
| GNC-GM | 100 | 100 | 100 | 100 | 100 | 40 | 0 | 0 |

## coherent outliers: median rotation error (deg)

| method | 0% | 10% | 20% | 30% | 40% | 50% | 60% | 70% |
|---|---|---|---|---|---|---|---|---|
| Kabsch | 0.04 | 4.13 | 8.66 | 13.11 | 19.22 | 21.58 | 26.27 | 35.03 |
| RANSAC-1000 | 0.03 | 0.05 | 0.05 | 0.05 | 0.06 | 39.90 | 46.73 | 50.47 |
| RANSAC-adaptive | 0.04 | 0.06 | 0.06 | 0.05 | 0.07 | 39.90 | 46.73 | 50.47 |
| IRLS-GM | 0.03 | 0.04 | 0.05 | 0.04 | 0.06 | 34.87 | 42.83 | 50.47 |
| GNC-GM | 0.03 | 0.04 | 0.05 | 0.04 | 0.06 | 33.93 | 46.73 | 50.47 |

## coherent outliers: runtime p50 (ms)

| method | 0% | 10% | 20% | 30% | 40% | 50% | 60% | 70% |
|---|---|---|---|---|---|---|---|---|
| Kabsch | 0.13 | 0.08 | 0.07 | 0.07 | 0.07 | 0.07 | 0.07 | 0.08 |
| RANSAC-1000 | 108.24 | 67.84 | 63.10 | 63.66 | 63.48 | 63.32 | 64.10 | 79.06 |
| RANSAC-adaptive | 1.07 | 0.97 | 1.33 | 1.98 | 3.23 | 3.42 | 1.98 | 1.33 |
| IRLS-GM | 4.58 | 2.67 | 2.66 | 2.67 | 2.63 | 2.63 | 2.66 | 2.82 |
| GNC-GM | 1.61 | 2.45 | 2.37 | 2.35 | 2.30 | 2.16 | 2.30 | 2.56 |

## boundary outliers: success rate

| method | 0% | 10% | 20% | 30% | 40% | 50% | 60% | 70% |
|---|---|---|---|---|---|---|---|---|
| Kabsch | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| RANSAC-1000 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| RANSAC-adaptive | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| IRLS-GM | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| GNC-GM | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |

## boundary outliers: median rotation error (deg)

| method | 0% | 10% | 20% | 30% | 40% | 50% | 60% | 70% |
|---|---|---|---|---|---|---|---|---|
| Kabsch | 0.05 | 0.04 | 0.06 | 0.06 | 0.05 | 0.06 | 0.07 | 0.08 |
| RANSAC-1000 | 0.05 | 0.05 | 0.06 | 0.06 | 0.06 | 0.08 | 0.12 | 0.16 |
| RANSAC-adaptive | 0.05 | 0.04 | 0.05 | 0.07 | 0.09 | 0.09 | 0.14 | 0.17 |
| IRLS-GM | 0.04 | 0.04 | 0.05 | 0.04 | 0.04 | 0.05 | 0.08 | 0.09 |
| GNC-GM | 0.04 | 0.04 | 0.05 | 0.04 | 0.04 | 0.05 | 0.08 | 0.09 |

## boundary outliers: runtime p50 (ms)

| method | 0% | 10% | 20% | 30% | 40% | 50% | 60% | 70% |
|---|---|---|---|---|---|---|---|---|
| Kabsch | 0.09 | 0.08 | 0.07 | 0.07 | 0.08 | 0.08 | 0.07 | 0.07 |
| RANSAC-1000 | 96.71 | 64.53 | 63.05 | 63.14 | 63.10 | 62.86 | 62.79 | 62.34 |
| RANSAC-adaptive | 0.92 | 0.82 | 1.05 | 1.45 | 1.93 | 2.85 | 3.83 | 5.94 |
| IRLS-GM | 4.07 | 2.65 | 2.61 | 2.63 | 2.65 | 2.61 | 2.62 | 2.60 |
| GNC-GM | 1.43 | 0.96 | 0.96 | 0.95 | 0.96 | 0.96 | 0.96 | 0.95 |

## Caveats

* KITTI raw provides OXTS, not per-frame ground-truth poses for this sync drive; the relative transform is derived from GPS/IMU and is accurate to the OXTS/calibration level, not to the millimetre. Clean Kabsch reaches ~0.02 deg / ~0.01-0.03 m on the sampled base pairs, which is the practical noise floor.
* LiDAR is sparse (~0.1 m point spacing), so c=0.25 m is used; a 3DMatch-style c=0.05 m would discard most correct matches.
* Base correspondences are GT-mapped NN, not a learned descriptor; this isolates the robust outer layer, like the 3DMatch oracle table.
* Consecutive frames at 10 Hz have |t|≈1.6-2.2 m; the 0.1 m success threshold is absolute pose error, not relative-to-motion error.

## Cross-drive comparison (drive 0001)

See `results/kitti2_compare.md` / `kitti2_compare.json` / `kitti2_compare.png`. Random and boundary success agree with drive 0001 to within 5 points in all 80 cells; the coherent majority-outlier transition stays at p≈0.5-0.6; GNC remains ~22x faster than fixed RANSAC-1000.
