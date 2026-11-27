# 3DMatch low / mid / high overlap study (all 8 scenes)

Overlap = uncapped GT-NN fraction (fraction of 2000 sampled source points mapped by inv(gt.log) with a target NN within tau=0.06 m). Bins: low < 0.5, mid 0.5-0.8, high > 0.8. N=300, c=0.05 m, random outliers at p in [0.0, 0.3, 0.5, 0.7], plus coherent outliers at p=0.3; 8 trials/cell; up to 3 pairs per bin per scene.

## Overlap distribution of all gt.log pairs

| scene | fragments | pairs | low | mid | high | overlap p0 / p50 / p100 |
|---|---|---|---|---|---|---|
| redkitchen | 60 | 506 | 200 | 277 | 29 | 0.21 / 0.54 / 0.99 |
| sun3d-home_at | 60 | 156 | 66 | 79 | 11 | 0.27 / 0.53 / 0.99 |
| sun3d-hotel_uc | 55 | 226 | 88 | 126 | 12 | 0.27 / 0.54 / 0.95 |
| sun3d-home_md | 60 | 208 | 68 | 104 | 36 | 0.21 / 0.59 / 0.94 |
| sun3d-hotel1 | 57 | 104 | 39 | 59 | 6 | 0.28 / 0.54 / 0.94 |
| sun3d-hotel3 | 37 | 54 | 21 | 33 | 0 | 0.30 / 0.53 / 0.80 |
| sun3d-studyroom | 66 | 292 | 102 | 159 | 31 | 0.26 / 0.55 / 0.96 |
| sun3d-mit_lab_hj | 38 | 77 | 22 | 39 | 16 | 0.25 / 0.60 / 0.97 |

## Success rate by overlap bin (random outliers)

| bin | p | Kabsch | RANSAC-1000 | RANSAC-adaptive | IRLS-GM | GNC-GM |
|---|---|---|---|---|---|---|
| low | 0% | 100 | 100 | 100 | 100 | 100 |
| low | 30% | 11 | 100 | 100 | 100 | 100 |
| low | 50% | 3 | 100 | 100 | 100 | 100 |
| low | 70% | 0 | 100 | 100 | 97 | 100 |
| mid | 0% | 100 | 100 | 100 | 100 | 100 |
| mid | 30% | 25 | 100 | 100 | 100 | 100 |
| mid | 50% | 6 | 100 | 100 | 100 | 100 |
| mid | 70% | 2 | 100 | 100 | 89 | 100 |
| high | 0% | 100 | 100 | 100 | 100 | 100 |
| high | 30% | 45 | 100 | 100 | 100 | 100 |
| high | 50% | 17 | 100 | 100 | 100 | 100 |
| high | 70% | 2 | 100 | 100 | 99 | 100 |

## Median rotation error by overlap bin (random outliers, deg)

| bin | p | Kabsch | RANSAC-1000 | RANSAC-adaptive | IRLS-GM | GNC-GM |
|---|---|---|---|---|---|---|
| low | 0% | 0.51 | 0.53 | 0.54 | 0.52 | 0.52 |
| low | 30% | 3.86 | 0.51 | 0.54 | 0.51 | 0.51 |
| low | 50% | 6.85 | 0.52 | 0.53 | 0.51 | 0.51 |
| low | 70% | 13.34 | 0.56 | 0.55 | 0.51 | 0.50 |
| mid | 0% | 0.26 | 0.27 | 0.28 | 0.24 | 0.24 |
| mid | 30% | 3.00 | 0.32 | 0.30 | 0.26 | 0.26 |
| mid | 50% | 5.79 | 0.31 | 0.32 | 0.29 | 0.29 |
| mid | 70% | 11.03 | 0.35 | 0.33 | 0.32 | 0.30 |
| high | 0% | 0.18 | 0.19 | 0.19 | 0.18 | 0.18 |
| high | 30% | 2.68 | 0.22 | 0.22 | 0.20 | 0.20 |
| high | 50% | 4.95 | 0.22 | 0.23 | 0.21 | 0.21 |
| high | 70% | 10.12 | 0.24 | 0.24 | 0.22 | 0.22 |

## GNC advantage vs overlap (random outliers)

| bin | p | GNC success | RANSAC-1000 success | Δ success | GNC ms p50 | RANSAC ms p50 | speedup |
|---|---|---|---|---|---|---|---|
| low | 0% | 100 | 100 | +0 | 0.81 | 62.0 | 76.6x |
| low | 30% | 100 | 100 | +0 | 2.20 | 64.0 | 29.1x |
| low | 50% | 100 | 100 | +0 | 2.22 | 67.4 | 30.4x |
| low | 70% | 100 | 100 | +0 | 2.19 | 61.3 | 28.0x |
| mid | 0% | 100 | 100 | +0 | 0.80 | 61.6 | 76.8x |
| mid | 30% | 100 | 100 | +0 | 2.23 | 62.5 | 28.0x |
| mid | 50% | 100 | 100 | +0 | 2.24 | 62.2 | 27.8x |
| mid | 70% | 100 | 100 | +0 | 2.23 | 61.8 | 27.8x |
| high | 0% | 100 | 100 | +0 | 0.79 | 60.8 | 77.2x |
| high | 30% | 100 | 100 | +0 | 2.21 | 62.4 | 28.2x |
| high | 50% | 100 | 100 | +0 | 2.22 | 62.0 | 27.9x |
| high | 70% | 100 | 100 | +0 | 2.23 | 61.8 | 27.7x |

## Per-pair overlap dependence

* GNC success vs pair overlap at p=0.5: Spearman rho=+nan (p=nan)
* RANSAC-1000 success vs pair overlap at p=0.5: rho=+nan (p=nan)
* clean median rotation error vs pair overlap: rho=-0.62 (p=0.000)

| scene | pair | bin | overlap | n_valid | base inlier | clean rot | GNC succ p=0.5 | RANSAC succ p=0.5 |
|---|---|---|---|---|---|---|---|---|
| redkitchen | 15-21 | low | 0.21 | 238 | 0.96 | 0.512 | 1.00 | 1.00 |
| sun3d-home_md | 11-33 | low | 0.21 | 248 | 0.91 | 0.898 | 1.00 | 1.00 |
| sun3d-mit_lab_hj | 0-11 | low | 0.25 | 298 | 0.95 | 0.464 | 1.00 | 1.00 |
| sun3d-studyroom | 4-45 | low | 0.26 | 300 | 0.95 | 0.646 | 1.00 | 1.00 |
| sun3d-home_at | 41-43 | low | 0.27 | 300 | 0.98 | 0.405 | 1.00 | 1.00 |
| sun3d-hotel_uc | 48-53 | low | 0.27 | 300 | 0.94 | 0.616 | 1.00 | 1.00 |
| sun3d-hotel1 | 10-29 | low | 0.28 | 300 | 0.97 | 0.976 | 1.00 | 1.00 |
| sun3d-hotel3 | 14-15 | low | 0.30 | 300 | 0.99 | 0.518 | 1.00 | 1.00 |
| sun3d-hotel1 | 43-44 | low | 0.40 | 300 | 0.97 | 0.535 | 1.00 | 1.00 |
| sun3d-hotel3 | 31-34 | low | 0.41 | 300 | 0.95 | 0.856 | 1.00 | 1.00 |
| sun3d-mit_lab_hj | 3-6 | low | 0.41 | 300 | 0.90 | 0.707 | 1.00 | 1.00 |
| sun3d-home_md | 32-36 | low | 0.42 | 300 | 0.92 | 0.794 | 1.00 | 1.00 |
| sun3d-studyroom | 28-30 | low | 0.42 | 300 | 0.99 | 0.325 | 1.00 | 1.00 |
| sun3d-home_at | 3-47 | low | 0.42 | 300 | 0.94 | 0.490 | 1.00 | 1.00 |
| sun3d-hotel_uc | 39-43 | low | 0.43 | 300 | 0.97 | 0.975 | 1.00 | 1.00 |
| redkitchen | 6-24 | low | 0.43 | 300 | 0.92 | 0.281 | 1.00 | 1.00 |
| sun3d-hotel3 | 6-7 | low | 0.49 | 300 | 0.98 | 0.424 | 1.00 | 1.00 |
| sun3d-mit_lab_hj | 30-31 | low | 0.49 | 300 | 0.93 | 0.412 | 1.00 | 1.00 |
| sun3d-home_at | 38-39 | low | 0.50 | 300 | 0.94 | 0.393 | 1.00 | 1.00 |
| sun3d-home_md | 28-51 | low | 0.50 | 300 | 0.88 | 0.339 | 1.00 | 1.00 |
| sun3d-hotel_uc | 37-39 | low | 0.50 | 300 | 0.93 | 0.566 | 1.00 | 1.00 |
| sun3d-hotel1 | 14-26 | low | 0.50 | 300 | 0.96 | 0.463 | 1.00 | 1.00 |
| redkitchen | 22-54 | low | 0.50 | 300 | 0.95 | 0.249 | 1.00 | 1.00 |
| sun3d-studyroom | 22-34 | low | 0.50 | 300 | 0.96 | 0.538 | 1.00 | 1.00 |
| redkitchen | 19-38 | mid | 0.50 | 300 | 0.97 | 0.168 | 1.00 | 1.00 |
| sun3d-home_at | 31-48 | mid | 0.50 | 300 | 0.95 | 0.186 | 1.00 | 1.00 |
| sun3d-studyroom | 37-47 | mid | 0.50 | 300 | 0.95 | 0.411 | 1.00 | 1.00 |
| sun3d-hotel_uc | 19-36 | mid | 0.50 | 300 | 0.97 | 0.161 | 1.00 | 1.00 |
| sun3d-hotel1 | 49-53 | mid | 0.50 | 300 | 0.98 | 0.932 | 1.00 | 1.00 |
| sun3d-mit_lab_hj | 23-25 | mid | 0.50 | 300 | 0.98 | 0.374 | 1.00 | 1.00 |
| sun3d-home_md | 38-47 | mid | 0.50 | 300 | 0.94 | 0.959 | 1.00 | 1.00 |
| sun3d-hotel3 | 25-27 | mid | 0.51 | 300 | 0.97 | 0.245 | 1.00 | 1.00 |
| sun3d-hotel3 | 23-29 | mid | 0.59 | 300 | 0.98 | 0.165 | 1.00 | 1.00 |
| sun3d-home_at | 15-16 | mid | 0.60 | 300 | 0.98 | 0.228 | 1.00 | 1.00 |
| redkitchen | 26-27 | mid | 0.61 | 300 | 0.96 | 0.226 | 1.00 | 1.00 |
| sun3d-hotel1 | 13-32 | mid | 0.62 | 300 | 0.95 | 1.077 | 1.00 | 1.00 |
| sun3d-studyroom | 35-49 | mid | 0.62 | 300 | 0.95 | 0.293 | 1.00 | 1.00 |
| sun3d-mit_lab_hj | 5-7 | mid | 0.62 | 300 | 0.93 | 0.423 | 1.00 | 1.00 |
| sun3d-hotel_uc | 15-16 | mid | 0.63 | 300 | 0.98 | 0.215 | 1.00 | 1.00 |
| sun3d-home_md | 4-46 | mid | 0.63 | 300 | 0.92 | 0.306 | 1.00 | 1.00 |
| sun3d-home_at | 49-50 | mid | 0.79 | 300 | 0.98 | 0.097 | 1.00 | 1.00 |
| sun3d-studyroom | 25-27 | mid | 0.79 | 300 | 0.97 | 0.139 | 1.00 | 1.00 |
| sun3d-mit_lab_hj | 19-20 | mid | 0.79 | 300 | 0.96 | 0.721 | 1.00 | 1.00 |
| sun3d-hotel_uc | 43-44 | mid | 0.80 | 300 | 0.98 | 0.212 | 1.00 | 1.00 |
| redkitchen | 43-44 | mid | 0.80 | 300 | 0.99 | 0.173 | 1.00 | 1.00 |
| sun3d-hotel3 | 11-13 | mid | 0.80 | 300 | 0.98 | 0.216 | 1.00 | 1.00 |
| sun3d-home_md | 20-54 | mid | 0.80 | 300 | 0.93 | 1.261 | 1.00 | 1.00 |
| sun3d-hotel1 | 38-39 | mid | 0.80 | 300 | 0.98 | 0.216 | 1.00 | 1.00 |
| redkitchen | 47-48 | high | 0.80 | 300 | 0.99 | 0.307 | 1.00 | 1.00 |
| sun3d-home_md | 52-59 | high | 0.80 | 300 | 0.96 | 0.309 | 1.00 | 1.00 |
| sun3d-studyroom | 46-47 | high | 0.80 | 300 | 0.99 | 0.136 | 1.00 | 1.00 |
| sun3d-home_at | 4-6 | high | 0.80 | 300 | 0.98 | 0.111 | 1.00 | 1.00 |
| sun3d-hotel_uc | 6-7 | high | 0.81 | 300 | 0.98 | 0.165 | 1.00 | 1.00 |
| sun3d-mit_lab_hj | 14-16 | high | 0.81 | 300 | 0.98 | 0.171 | 1.00 | 1.00 |
| sun3d-hotel1 | 30-31 | high | 0.82 | 300 | 0.99 | 0.234 | 1.00 | 1.00 |
| redkitchen | 42-43 | high | 0.84 | 300 | 0.99 | 0.180 | 1.00 | 1.00 |
| sun3d-studyroom | 2-6 | high | 0.84 | 300 | 0.98 | 0.132 | 1.00 | 1.00 |
| sun3d-home_md | 55-58 | high | 0.85 | 300 | 0.96 | 0.210 | 1.00 | 1.00 |
| sun3d-hotel_uc | 39-40 | high | 0.86 | 300 | 0.98 | 0.354 | 1.00 | 1.00 |
| sun3d-mit_lab_hj | 35-36 | high | 0.89 | 300 | 0.99 | 0.081 | 1.00 | 1.00 |
| sun3d-home_at | 43-45 | high | 0.89 | 300 | 0.98 | 0.204 | 1.00 | 1.00 |
| sun3d-hotel1 | 40-56 | high | 0.92 | 300 | 0.98 | 0.580 | 1.00 | 1.00 |
| sun3d-home_md | 41-47 | high | 0.94 | 300 | 0.99 | 0.154 | 1.00 | 1.00 |
| sun3d-hotel1 | 36-37 | high | 0.94 | 300 | 1.00 | 0.089 | 1.00 | 1.00 |
| sun3d-hotel_uc | 31-33 | high | 0.95 | 300 | 0.96 | 0.552 | 1.00 | 1.00 |
| sun3d-studyroom | 36-37 | high | 0.96 | 300 | 0.99 | 0.171 | 1.00 | 1.00 |
| sun3d-mit_lab_hj | 32-36 | high | 0.97 | 300 | 0.99 | 0.113 | 1.00 | 1.00 |
| sun3d-home_at | 44-45 | high | 0.99 | 300 | 0.99 | 0.128 | 1.00 | 1.00 |
| redkitchen | 50-52 | high | 0.99 | 300 | 0.99 | 0.184 | 1.00 | 1.00 |

## Caveats

* Base correspondences are GT-NN oracle matches on real fragments, so low overlap does not degrade *correctness* of base matches; it changes the residual distribution and the share of boundary correspondences near c. The clean p=0 rows and `base_inlier` quantify that real-geometry effect.
* Pairs whose valid GT-NN count is below N=300 are padded by resampling with replacement; `n_valid` is reported per pair.
* The overlap estimator is a surface-proximity proxy with tau=0.06 m; it can overestimate overlap when large planar structures (floors/ceilings) are shared by fragments that see different parts of the room.
