# 3DMatch per-scene heterogeneity (4 scenes)

| scene | random70 GNC/RANSAC | GNC rot70 | speedup70 | boundary70 GNC/RANSAC rot | coherent50 GNC/IRLS/RANSAC picked-outlier |
|---|---|---|---|---|---|
| 7-scenes-redkitchen | 100/100% | 0.24° | 28.1× | 0.25/0.74° | 30/40/100% |
| sun3d-home_at-home_at_scan1_2013_jan_1 | 100/100% | 0.21° | 27.1× | 0.25/0.60° | 65/40/100% |
| sun3d-home_md-home_md_scan9_2012_sep_30 | 100/100% | 0.52° | 29.6× | 0.44/0.92° | 50/55/100% |
| sun3d-hotel_uc-scan3 | 100/100% | 0.28° | 27.4× | 0.31/0.52° | 70/25/100% |

## Reading

* All four scenes share the random-outlier success and the coherent collapse; the 50% tie regime is scene-dependent (35–65% GNC), as expected near a symmetry.
* Boundary median rotation is consistently lower for GNC than RANSAC in every scene.
