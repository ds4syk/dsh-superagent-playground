# KITTI threshold c sensitivity

Random outliers, 20 trials/cell, N=300, GT-NN LiDAR base. c is the GM scale on squared residual and the RANSAC distance threshold.

| c | p | method | base inlier | success | rot med (deg) | p50 (ms) |
|---|---|---|---|---|---|---|
| 0.15 | 0.0 | Kabsch | 69% | 100% | 0.03 | 0.07 |
| 0.15 | 0.0 | RANSAC-1000 | 69% | 100% | 0.04 | 62.23 |
| 0.15 | 0.0 | GNC-GM | 69% | 100% | 0.04 | 1.17 |
| 0.15 | 0.3 | Kabsch | 69% | 0% | 1.29 | 0.09 |
| 0.15 | 0.3 | RANSAC-1000 | 69% | 100% | 0.05 | 68.33 |
| 0.15 | 0.3 | GNC-GM | 69% | 100% | 0.04 | 3.80 |
| 0.15 | 0.5 | Kabsch | 69% | 0% | 3.80 | 0.07 |
| 0.15 | 0.5 | RANSAC-1000 | 69% | 100% | 0.05 | 62.39 |
| 0.15 | 0.5 | GNC-GM | 69% | 100% | 0.05 | 2.95 |
| 0.15 | 0.7 | Kabsch | 68% | 0% | 5.00 | 0.07 |
| 0.15 | 0.7 | RANSAC-1000 | 68% | 100% | 0.05 | 62.28 |
| 0.15 | 0.7 | GNC-GM | 68% | 100% | 0.04 | 3.01 |
| 0.25 | 0.0 | Kabsch | 88% | 100% | 0.03 | 0.07 |
| 0.25 | 0.0 | RANSAC-1000 | 88% | 100% | 0.03 | 62.50 |
| 0.25 | 0.0 | GNC-GM | 88% | 100% | 0.03 | 0.97 |
| 0.25 | 0.3 | Kabsch | 87% | 0% | 1.24 | 0.08 |
| 0.25 | 0.3 | RANSAC-1000 | 87% | 100% | 0.05 | 64.82 |
| 0.25 | 0.3 | GNC-GM | 87% | 100% | 0.04 | 2.89 |
| 0.25 | 0.5 | Kabsch | 87% | 0% | 2.83 | 0.14 |
| 0.25 | 0.5 | RANSAC-1000 | 87% | 100% | 0.04 | 120.73 |
| 0.25 | 0.5 | GNC-GM | 87% | 100% | 0.03 | 5.21 |
| 0.25 | 0.7 | Kabsch | 86% | 0% | 6.33 | 0.13 |
| 0.25 | 0.7 | RANSAC-1000 | 86% | 100% | 0.05 | 120.79 |
| 0.25 | 0.7 | GNC-GM | 86% | 100% | 0.05 | 5.27 |
| 0.35 | 0.0 | Kabsch | 96% | 100% | 0.03 | 0.13 |
| 0.35 | 0.0 | RANSAC-1000 | 96% | 100% | 0.03 | 120.70 |
| 0.35 | 0.0 | GNC-GM | 96% | 100% | 0.03 | 1.53 |
| 0.35 | 0.3 | Kabsch | 95% | 0% | 1.82 | 0.07 |
| 0.35 | 0.3 | RANSAC-1000 | 95% | 100% | 0.04 | 63.50 |
| 0.35 | 0.3 | GNC-GM | 95% | 100% | 0.04 | 2.66 |
| 0.35 | 0.5 | Kabsch | 95% | 5% | 2.11 | 0.07 |
| 0.35 | 0.5 | RANSAC-1000 | 95% | 100% | 0.05 | 63.75 |
| 0.35 | 0.5 | GNC-GM | 95% | 100% | 0.05 | 2.71 |
| 0.35 | 0.7 | Kabsch | 95% | 0% | 3.17 | 0.07 |
| 0.35 | 0.7 | RANSAC-1000 | 95% | 100% | 0.06 | 64.90 |
| 0.35 | 0.7 | GNC-GM | 95% | 100% | 0.05 | 2.75 |

## Reading

* At c=0.15 m most correct base matches (median residual ~0.10 m) are still inside, but the margin is thin; c=0.25 m is the defensible LiDAR choice. c=0.35 m adds more boundary/gross points and can bias the robust estimate slightly.
* The absolute rotation error is small (<0.1 deg) for all robust methods here; the interesting difference is success under injected outliers.
