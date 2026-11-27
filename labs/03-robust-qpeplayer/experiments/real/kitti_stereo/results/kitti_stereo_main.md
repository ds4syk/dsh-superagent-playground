# KITTI raw stereo front-end -> 3D-3D robust layer (fallback, not In-The-Dark)

Real hand-crafted stereo front-end (SIFT/ORB + sub-pixel + SGBM or sparse triangulation), natural mismatch outliers, no injected outliers and no trials (one deterministic instance per frame pair). `floor` = Kabsch on GT-labeled inliers only: the front-end noise floor. Success = rot < 5 deg and trans < 0.1 m (loose: trans < 0.25 m).

## dense/orb/c0.5  (30/30 pairs)

- correspondences median 600 (min 125), natural inlier rate median 20.7% (min 0.7%), inlier median depth 29.8 m
- GT motion: rot 0.197 deg, trans 1.92 m; inverse-pose residual ratio 4.1x
- front-end time p50 0.21 s / p95 0.28 s

| method | success@0.1 m | success@0.25 m | med rot (deg) | med trans (m) | p50 ms | p95 ms | max ms |
|-|-|-|-|-|-|-|-|
| Kabsch | 0% | 7% | 0.285 | 0.694 | 0.17 | 0.24 | 0.25 |
| RANSAC-1000 | 0% | 0% | 0.117 | 0.735 | 72.01 | 115.65 | 144.68 |
| RANSAC-adaptive | 0% | 0% | 0.115 | 0.737 | 6.58 | 26.63 | 44.53 |
| IRLS-GM | 0% | 0% | 0.084 | 0.739 | 3.90 | 5.83 | 7.48 |
| GNC-GM | 0% | 0% | 0.084 | 0.739 | 3.39 | 5.18 | 6.68 |
| floor | 0% | 86% | 0.087 | 0.202 | nan | nan | nan |

## dense/sift/c0.3  (30/30 pairs)

- correspondences median 890 (min 278), natural inlier rate median 5.9% (min 0.0%), inlier median depth 29.3 m
- GT motion: rot 0.197 deg, trans 1.92 m; inverse-pose residual ratio 4.9x
- front-end time p50 0.35 s / p95 0.41 s

| method | success@0.1 m | success@0.25 m | med rot (deg) | med trans (m) | p50 ms | p95 ms | max ms |
|-|-|-|-|-|-|-|-|
| Kabsch | 0% | 0% | 0.224 | 0.663 | 0.12 | 0.14 | 0.14 |
| RANSAC-1000 | 0% | 0% | 0.060 | 0.695 | 77.95 | 85.45 | 88.27 |
| RANSAC-adaptive | 0% | 0% | 0.064 | 0.718 | 4.38 | 46.54 | 51.21 |
| IRLS-GM | 0% | 0% | 0.050 | 0.700 | 4.74 | 5.46 | 5.87 |
| GNC-GM | 0% | 0% | 0.050 | 0.700 | 4.64 | 5.32 | 5.46 |
| floor | 5% | 100% | 0.111 | 0.129 | nan | nan | nan |

## dense/sift/c0.5  (30/30 pairs)

- correspondences median 890 (min 278), natural inlier rate median 17.8% (min 0.9%), inlier median depth 27.3 m
- GT motion: rot 0.197 deg, trans 1.92 m; inverse-pose residual ratio 4.9x
- front-end time p50 0.35 s / p95 0.41 s

| method | success@0.1 m | success@0.25 m | med rot (deg) | med trans (m) | p50 ms | p95 ms | max ms |
|-|-|-|-|-|-|-|-|
| Kabsch | 0% | 0% | 0.224 | 0.663 | 0.19 | 0.25 | 0.27 |
| RANSAC-1000 | 0% | 0% | 0.068 | 0.710 | 81.28 | 117.96 | 124.86 |
| RANSAC-adaptive | 0% | 0% | 0.072 | 0.700 | 2.66 | 14.12 | 19.54 |
| IRLS-GM | 0% | 0% | 0.052 | 0.704 | 4.85 | 7.14 | 7.58 |
| GNC-GM | 0% | 0% | 0.052 | 0.704 | 4.21 | 6.46 | 6.62 |
| floor | 0% | 54% | 0.071 | 0.239 | nan | nan | nan |

## dense/sift/c1.0  (30/30 pairs)

- correspondences median 890 (min 278), natural inlier rate median 69.7% (min 7.5%), inlier median depth 18.3 m
- GT motion: rot 0.197 deg, trans 1.92 m; inverse-pose residual ratio 4.9x
- front-end time p50 0.35 s / p95 0.41 s

| method | success@0.1 m | success@0.25 m | med rot (deg) | med trans (m) | p50 ms | p95 ms | max ms |
|-|-|-|-|-|-|-|-|
| Kabsch | 0% | 0% | 0.224 | 0.663 | 0.12 | 0.14 | 0.20 |
| RANSAC-1000 | 0% | 0% | 0.110 | 0.796 | 78.15 | 82.48 | 82.90 |
| RANSAC-adaptive | 0% | 0% | 0.076 | 0.714 | 1.01 | 2.61 | 3.10 |
| IRLS-GM | 0% | 0% | 0.068 | 0.706 | 4.68 | 5.24 | 5.39 |
| GNC-GM | 0% | 0% | 0.068 | 0.706 | 3.81 | 4.19 | 4.44 |
| floor | 0% | 0% | 0.081 | 0.473 | nan | nan | nan |

## sparse/sift/c0.5  (30/30 pairs)

- correspondences median 229 (min 74), natural inlier rate median 17.7% (min 0.0%), inlier median depth 27.9 m
- GT motion: rot 0.197 deg, trans 1.92 m; inverse-pose residual ratio 4.2x
- front-end time p50 0.49 s / p95 0.56 s

| method | success@0.1 m | success@0.25 m | med rot (deg) | med trans (m) | p50 ms | p95 ms | max ms |
|-|-|-|-|-|-|-|-|
| Kabsch | 0% | 0% | 0.514 | 0.858 | 0.16 | 0.21 | 0.22 |
| RANSAC-1000 | 0% | 0% | 0.070 | 0.735 | 63.24 | 105.50 | 119.29 |
| RANSAC-adaptive | 0% | 0% | 0.068 | 0.681 | 4.61 | 32.94 | 51.25 |
| IRLS-GM | 0% | 0% | 0.047 | 0.694 | 2.43 | 4.07 | 4.78 |
| GNC-GM | 0% | 0% | 0.047 | 0.694 | 2.12 | 3.45 | 4.23 |
| floor | 0% | 70% | 0.142 | 0.233 | nan | nan | nan |

## Per-pair front-end stats (dense/SIFT/c=0.5)

| pair | n | inlier rate | med depth (m) | GT trans (m) |
|-|-|-|-|-|
| adj:0->1 | 1000 | 11.6% | 27.6 | 2.20 |
| adj:3->4 | 922 | 10.1% | 32.0 | 2.18 |
| adj:6->7 | 825 | 36.2% | 21.4 | 1.95 |
| adj:9->10 | 784 | 12.6% | 29.3 | 2.11 |
| adj:12->13 | 892 | 26.3% | 23.8 | 1.87 |
| adj:15->16 | 887 | 19.4% | 27.5 | 2.01 |
| adj:18->19 | 885 | 32.2% | 22.6 | 1.81 |
| adj:21->22 | 902 | 33.0% | 21.1 | 1.79 |
| adj:24->25 | 897 | 43.8% | 18.7 | 1.77 |
| adj:27->28 | 871 | 34.9% | 19.2 | 1.76 |
| adj:30->31 | 962 | 28.9% | 18.6 | 1.75 |
| adj:33->34 | 991 | 31.6% | 22.7 | 1.73 |
| adj:36->37 | 940 | 18.4% | 29.1 | 1.88 |
| adj:39->40 | 1020 | 17.2% | 27.2 | 1.87 |
| adj:42->43 | 1023 | 16.3% | 26.1 | 1.85 |
| adj:45->46 | 927 | 18.6% | 25.9 | 1.83 |
| adj:48->49 | 969 | 22.0% | 25.6 | 1.79 |
| adj:51->52 | 997 | 31.2% | 25.1 | 1.75 |
| adj:54->55 | 1115 | 45.9% | 17.8 | 1.55 |
| adj:57->58 | 1123 | 20.7% | 25.3 | 1.66 |
| gap3:0->3 | 358 | 2.2% | 49.7 | 6.17 |
| gap3:3->6 | 278 | 1.8% | 48.4 | 6.11 |
| gap3:6->9 | 288 | 2.4% | 41.8 | 6.00 |
| gap3:9->12 | 334 | 0.9% | 47.7 | 5.89 |
| gap3:12->15 | 330 | 1.8% | 46.9 | 5.74 |
| gap3:15->18 | 320 | 1.6% | 38.2 | 5.64 |
| gap3:18->21 | 331 | 2.1% | 48.0 | 5.59 |
| gap3:21->24 | 324 | 1.2% | 47.5 | 5.53 |
| gap3:24->27 | 317 | 2.2% | 46.6 | 5.49 |
| gap3:27->30 | 293 | 2.7% | 41.7 | 5.43 |
