# KITTI odometry 01/02/03/04/06/08/09/10: robust 3D-3D extension

Protocol imported unchanged from `../kitti_odom/run_kitti_odom.py`: N=300, c=0.25 m, tau=0.45 m, voxel=0.15 m, random outliers at p in [0.0, 0.3, 0.5, 0.7], **10 trials/cell**, 10 consecutive-frame pairs per sequence. Success = rot < 5 deg and |t| < 0.1 m. The parent 00/05/07 rows use 20 trials/cell (read from `../kitti_odom/results/kitti_odom.json`), so compare means, not paired tests.

## Data actually used

| seq | GT frames | frames fetched by protocol | pairs used | MB |
|---|---|---|---|---|
| 01 | 1101 | 33 | 10 | 8.4 |
| 02 | 4661 | 33 | 10 | 65.6 |
| 03 | 801 | 33 | 10 | 65.9 |
| 04 | 271 | 32 | 10 | 64.4 |
| 06 | 1101 | 33 | 10 | 64.7 |
| 08 | 4071 | 33 | 10 | 64.5 |
| 09 | 1591 | 33 | 10 | 65.9 |
| 10 | 1201 | 33 | 10 | 66.3 |

The official odometry Velodyne archive is a single 84.8 GB zip; only the ~33 frames per sequence touched by the protocol were range-fetched from the HuggingFace mirror. **Coverage claim: all 11 sequences 00-10 are represented by the frames this protocol uses; the full 00-10 LiDAR streams are NOT downloaded.**

## Success per sequence (GNC / RANSAC-1000), p = 0.0/0.3/0.5/0.7

| seq | base inlier | GNC p0 | GNC p.3 | GNC p.5 | GNC p.7 | RANSAC p.7 |
|---|---|---|---|---|---|---|
| 01 | 0.889 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 02 | 0.941 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 03 | 0.926 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 04 | 0.908 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 06 | 0.916 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 08 | 0.914 | 0.80 | 0.80 | 1.00 | 1.00 | 1.00 |
| 09 | 0.934 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 10 | 0.935 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

## Pooled

| set | p | GNC | RANSAC-1000 | IRLS | RANSAC/GNC time |
|---|---|---|---|---|---|
| new 8 | 0.0 | 0.97 | 0.99 | 0.97 | 65.7x |
| new 8 | 0.3 | 0.97 | 0.97 | 0.97 | 22.7x |
| new 8 | 0.5 | 1.00 | 1.00 | 1.00 | 22.4x |
| new 8 | 0.7 | 1.00 | 1.00 | 0.99 | 22.4x |
| all 11 | 0.0 | 0.99 | 0.99 | 0.99 | 65.7x |
| all 11 | 0.3 | 0.99 | 0.99 | 0.99 | 22.8x |
| all 11 | 0.5 | 1.00 | 1.00 | 1.00 | 22.4x |
| all 11 | 0.7 | 1.00 | 1.00 | 0.99 | 22.4x |

## Baseline 00/05/07 at the same p (20 trials/cell, parent run)

| seq | GNC p0 | GNC p.3 | GNC p.5 | GNC p.7 |
|---|---|---|---|---|
| 00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 05 | 1.00 | 1.00 | 1.00 | 1.00 |
| 07 | 1.00 | 1.00 | 1.00 | 1.00 |

