# 3DMatch true low overlap (uncapped 0.20-0.50) robust protocol

Overlap = uncapped GT-NN fraction (2000 uniform source points, mapped by inv(gt.log), nearest target NN < tau=0.06 m). Band low_20_50 = [0.20, 0.50]; control high > 0.80. N=300, c=0.05 m, random outliers at p in [0.0, 0.3, 0.5, 0.7], 10 trials/cell, up to 4 low + 2 high pairs per scene. Success = rot < 5 deg and |t| < 0.1 m.

## Overlap scan of all 8 official scene sets

| scene | fragments | gt pairs | <0.20 | **0.20-0.50** | 0.50-0.80 | >0.80 | p0 / p50 / p100 |
|---|---|---|---|---|---|---|---|
| redkitchen | 60 | 506 | 0 | 202 | 275 | 29 | 0.21 / 0.54 / 0.99 |
| sun3d-home_at | 60 | 156 | 0 | 66 | 79 | 11 | 0.27 / 0.53 / 0.99 |
| sun3d-hotel_uc | 55 | 226 | 0 | 88 | 126 | 12 | 0.27 / 0.54 / 0.95 |
| sun3d-home_md | 60 | 208 | 0 | 68 | 104 | 36 | 0.21 / 0.59 / 0.94 |
| sun3d-hotel1 | 57 | 104 | 0 | 39 | 59 | 6 | 0.28 / 0.54 / 0.94 |
| sun3d-hotel3 | 37 | 54 | 0 | 21 | 33 | 0 | 0.30 / 0.53 / 0.80 |
| sun3d-studyroom | 66 | 292 | 0 | 102 | 159 | 31 | 0.26 / 0.55 / 0.96 |
| sun3d-mit_lab_hj | 38 | 77 | 0 | 22 | 39 | 16 | 0.25 / 0.60 / 0.97 |

## Selected low-overlap pairs: GT-NN base quality

| scene | pair | overlap | n_kept/2000 | n_valid (of 300) | clean base inlier | padded |
|---|---|---|---|---|---|---|
| redkitchen | 15-21 | 0.212 | 424 | 249 | 0.955 | yes |
| redkitchen | 28-31 | 0.404 | 808 | 300 | 0.946 | no |
| redkitchen | 35-37 | 0.500 | 1000 | 300 | 0.945 | no |
| redkitchen | 4-25 | 0.457 | 914 | 300 | 0.955 | no |
| sun3d-home_at | 3-6 | 0.406 | 813 | 300 | 0.962 | no |
| sun3d-home_at | 38-39 | 0.498 | 996 | 300 | 0.933 | no |
| sun3d-home_at | 41-43 | 0.266 | 531 | 300 | 0.981 | no |
| sun3d-home_at | 41-46 | 0.454 | 908 | 300 | 0.973 | no |
| sun3d-home_md | 11-33 | 0.213 | 426 | 239 | 0.911 | yes |
| sun3d-home_md | 18-20 | 0.452 | 904 | 300 | 0.968 | no |
| sun3d-home_md | 28-51 | 0.498 | 996 | 300 | 0.881 | no |
| sun3d-home_md | 29-47 | 0.385 | 770 | 300 | 0.988 | no |
| sun3d-hotel_uc | 1-54 | 0.449 | 898 | 300 | 0.976 | no |
| sun3d-hotel_uc | 33-53 | 0.392 | 783 | 300 | 0.955 | no |
| sun3d-hotel_uc | 37-39 | 0.498 | 997 | 300 | 0.934 | no |
| sun3d-hotel_uc | 48-53 | 0.273 | 545 | 300 | 0.944 | no |
| sun3d-hotel1 | 1-16 | 0.427 | 855 | 300 | 0.928 | no |
| sun3d-hotel1 | 10-29 | 0.276 | 551 | 300 | 0.973 | no |
| sun3d-hotel1 | 14-26 | 0.498 | 997 | 300 | 0.957 | no |
| sun3d-hotel1 | 2-23 | 0.380 | 760 | 300 | 0.919 | no |
| sun3d-hotel3 | 14-15 | 0.300 | 600 | 300 | 0.995 | no |
| sun3d-hotel3 | 20-21 | 0.430 | 860 | 300 | 0.968 | no |
| sun3d-hotel3 | 23-30 | 0.394 | 788 | 300 | 0.961 | no |
| sun3d-hotel3 | 6-7 | 0.487 | 974 | 300 | 0.979 | no |
| sun3d-studyroom | 22-34 | 0.500 | 999 | 300 | 0.959 | no |
| sun3d-studyroom | 4-45 | 0.256 | 511 | 300 | 0.952 | no |
| sun3d-studyroom | 6-35 | 0.446 | 892 | 300 | 0.961 | no |
| sun3d-studyroom | 8-48 | 0.378 | 756 | 300 | 0.960 | no |
| sun3d-mit_lab_hj | 0-11 | 0.248 | 496 | 297 | 0.951 | yes |
| sun3d-mit_lab_hj | 1-10 | 0.401 | 802 | 300 | 0.947 | no |
| sun3d-mit_lab_hj | 30-31 | 0.487 | 975 | 300 | 0.932 | no |
| sun3d-mit_lab_hj | 31-34 | 0.437 | 874 | 300 | 0.943 | no |

## Robust success: true low overlap vs matched high-overlap control

| p | method | low success | low rot med | low t med | high success | high rot med | high t med |
|---|---|---|---|---|---|---|---|
| 0.0 | Kabsch | 1.00 | 0.49 | 0.016 | 1.00 | 0.20 | 0.005 |
| 0.0 | RANSAC-1000 | 1.00 | 0.52 | 0.018 | 1.00 | 0.20 | 0.005 |
| 0.0 | RANSAC-adaptive | 1.00 | 0.51 | 0.017 | 1.00 | 0.19 | 0.006 |
| 0.0 | IRLS-GM | 1.00 | 0.47 | 0.016 | 1.00 | 0.19 | 0.005 |
| 0.0 | GNC-GM | 1.00 | 0.47 | 0.016 | 1.00 | 0.19 | 0.005 |
| 0.3 | Kabsch | 0.13 | 3.68 | 0.203 | 0.50 | 2.59 | 0.100 |
| 0.3 | RANSAC-1000 | 1.00 | 0.51 | 0.017 | 1.00 | 0.21 | 0.006 |
| 0.3 | RANSAC-adaptive | 1.00 | 0.53 | 0.018 | 1.00 | 0.20 | 0.006 |
| 0.3 | IRLS-GM | 1.00 | 0.47 | 0.016 | 1.00 | 0.20 | 0.006 |
| 0.3 | GNC-GM | 1.00 | 0.47 | 0.016 | 1.00 | 0.20 | 0.006 |
| 0.5 | Kabsch | 0.02 | 6.73 | 0.349 | 0.15 | 4.88 | 0.184 |
| 0.5 | RANSAC-1000 | 1.00 | 0.50 | 0.017 | 1.00 | 0.23 | 0.006 |
| 0.5 | RANSAC-adaptive | 1.00 | 0.51 | 0.018 | 1.00 | 0.22 | 0.006 |
| 0.5 | IRLS-GM | 1.00 | 0.46 | 0.015 | 1.00 | 0.21 | 0.006 |
| 0.5 | GNC-GM | 1.00 | 0.46 | 0.015 | 1.00 | 0.21 | 0.006 |
| 0.7 | Kabsch | 0.01 | 13.09 | 0.524 | 0.04 | 9.19 | 0.303 |
| 0.7 | RANSAC-1000 | 1.00 | 0.55 | 0.018 | 1.00 | 0.24 | 0.007 |
| 0.7 | RANSAC-adaptive | 1.00 | 0.54 | 0.018 | 1.00 | 0.24 | 0.007 |
| 0.7 | IRLS-GM | 0.92 | 0.53 | 0.016 | 0.99 | 0.22 | 0.007 |
| 0.7 | GNC-GM | 1.00 | 0.51 | 0.016 | 1.00 | 0.22 | 0.007 |

## GNC vs RANSAC (true low overlap)

| p | succ GNC | succ RANSAC-1000 | succ IRLS | Δ(GNC-RANSAC) | RANSAC/GNC time |
|---|---|---|---|---|---|
| 0.0 | 1.00 | 1.00 | 1.00 | +0.00 | 75.3x |
| 0.3 | 1.00 | 1.00 | 1.00 | +0.00 | 27.1x |
| 0.5 | 1.00 | 1.00 | 1.00 | +0.00 | 27.1x |
| 0.7 | 1.00 | 1.00 | 0.92 | +0.00 | 27.2x |

## Existing high-overlap bin reference (8 trials/cell)

| p | method | overlap mean | base inlier | success | rot med |
|---|---|---|---|---|---|
| 0.0 | GNC-GM | 0.880 | 0.982 | 1.00 | 0.18 |
| 0.0 | IRLS-GM | 0.880 | 0.982 | 1.00 | 0.18 |
| 0.0 | Kabsch | 0.880 | 0.982 | 1.00 | 0.18 |
| 0.0 | RANSAC-1000 | 0.880 | 0.982 | 1.00 | 0.19 |
| 0.0 | RANSAC-adaptive | 0.880 | 0.982 | 1.00 | 0.19 |
| 0.3 | GNC-GM | 0.880 | 0.982 | 1.00 | 0.20 |
| 0.3 | IRLS-GM | 0.880 | 0.982 | 1.00 | 0.20 |
| 0.3 | Kabsch | 0.880 | 0.982 | 0.45 | 2.68 |
| 0.3 | RANSAC-1000 | 0.880 | 0.982 | 1.00 | 0.22 |
| 0.3 | RANSAC-adaptive | 0.880 | 0.982 | 1.00 | 0.22 |
| 0.5 | GNC-GM | 0.880 | 0.981 | 1.00 | 0.21 |
| 0.5 | IRLS-GM | 0.880 | 0.981 | 1.00 | 0.21 |
| 0.5 | Kabsch | 0.880 | 0.981 | 0.17 | 4.95 |
| 0.5 | RANSAC-1000 | 0.880 | 0.981 | 1.00 | 0.22 |
| 0.5 | RANSAC-adaptive | 0.880 | 0.981 | 1.00 | 0.23 |
| 0.7 | GNC-GM | 0.880 | 0.983 | 1.00 | 0.22 |
| 0.7 | IRLS-GM | 0.880 | 0.983 | 0.99 | 0.22 |
| 0.7 | Kabsch | 0.880 | 0.983 | 0.02 | 10.12 |
| 0.7 | RANSAC-1000 | 0.880 | 0.983 | 1.00 | 0.24 |
| 0.7 | RANSAC-adaptive | 0.880 | 0.983 | 1.00 | 0.24 |

## Reading of the result

The clean GT-NN base inlier fraction is measured on **oracle** matches: every correspondence already passes the tau gate by construction, so geometric overlap cannot lower it below ~0.9 unless the probe budget is too small. What low overlap *does* reduce is the number of usable correspondences (n_kept/2000 and n_valid); below overlap ~0.25 the 300-point budget is padded by resampling, which is recorded in every row. Therefore in this protocol 'low overlap' is a **geometric** label (shared surface fraction), not a **correspondence-quality** label (inlier ratio). A learned front-end with real descriptor noise is out of scope, and the paper should not present these cells as evidence that low-overlap registration is easy for a real matcher.

