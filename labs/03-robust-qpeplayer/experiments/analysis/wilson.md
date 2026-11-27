# Wilson 95% intervals for headline success rates

| cell | k/n | rate | 95% Wilson |
|---|---|---|---|
| synthetic random p=0.5 GNC | 100/100 | 100.0% | [96.3, 100.0]% |
| synthetic random p=0.7 GNC | 100/100 | 100.0% | [96.3, 100.0]% |
| 3DMatch-4 random p=0.5 GNC | 80/80 | 100.0% | [95.4, 100.0]% |
| 3DMatch-4 random p=0.7 GNC | 80/80 | 100.0% | [95.4, 100.0]% |
| 3DMatch-4 coherent p=0.5 GNC | 35/80 | 43.8% | [33.4, 54.7]% |
| 3DMatch-4 coherent p=0.6 GNC | 0/80 | 0.0% | [0.0, 4.6]% |
| 3DMatch-8 random p=0.5 GNC | 160/160 | 100.0% | [97.7, 100.0]% |
| 3DMatch-8 random p=0.7 GNC | 160/160 | 100.0% | [97.7, 100.0]% |
| KITTI-2 random p=0.5 GNC | 40/40 | 100.0% | [91.2, 100.0]% |
| KITTI-2 random p=0.7 GNC | 40/40 | 100.0% | [91.2, 100.0]% |
| KITTI-odom pooled random p=0.5 GNC | 60/60 | 100.0% | [94.0, 100.0]% |
| KITTI-odom pooled random p=0.7 GNC | 60/60 | 100.0% | [94.0, 100.0]% |

## Reading

The lower bound for a 100% cell with n=20 is ~83.9%, n=40 ~91.2%, n=60 ~94.0%, n=100 ~96.3%, n=160 ~97.7%. Do not present 20-trial 100% cells as exact 100%. The 3DMatch-8 rows use the 20 trials/cell x 8 scenes count from `real/threedmatch_full/results/full_scenes.md`; the per-trial JSON is gitignored and not committed.
