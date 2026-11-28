# 3DMatch real FPFH front-end audit

Open3D FPFH mutual-NN matches + pairwise-distance consistency, no GT in matching. N=300, c=0.05 m, 10 trials/pair, 6 high-overlap pairs/scene. GT is used only for evaluation.

| scene | method | mean natural inlier | success | median rot (deg) | p90 rot | median trans (m) | p50 (ms) |
|---|---|---|---|---|---|---|---|
| redkitchen | Kabsch | 80.0% | 100% | 0.92 | 2.85 | 0.0130 | 0.17 |
| redkitchen | RANSAC-1000 | 80.0% | 100% | 0.90 | 1.44 | 0.0136 | 71.68 |
| redkitchen | RANSAC-adaptive | 80.0% | 100% | 0.95 | 1.60 | 0.0142 | 0.74 |
| redkitchen | IRLS-GM | 80.0% | 100% | 0.95 | 1.34 | 0.0127 | 2.73 |
| redkitchen | GNC-GM | 80.0% | 100% | 0.96 | 1.34 | 0.0128 | 1.22 |
| sun3d-home_at | Kabsch | 67.8% | 83% | 1.23 | 5.55 | 0.0386 | 0.17 |
| sun3d-home_at | RANSAC-1000 | 67.8% | 97% | 0.99 | 2.53 | 0.0307 | 64.69 |
| sun3d-home_at | RANSAC-adaptive | 67.8% | 93% | 1.01 | 2.58 | 0.0328 | 1.15 |
| sun3d-home_at | IRLS-GM | 67.8% | 100% | 0.85 | 2.54 | 0.0265 | 2.72 |
| sun3d-home_at | GNC-GM | 67.8% | 100% | 0.85 | 2.54 | 0.0264 | 1.46 |

## Caveats

* This is a hand-crafted FPFH front-end, not FCGF/Predator; it is deliberately weak and is reported as a reality check, not as a learned-descriptor result.
* The pairwise-distance consistency filter uses no GT; it does add a strong geometric prior, so the natural inlier fraction is higher than raw FPFH.
* The oracle GT-NN table (`threedmatch.md`) is the controlled outlier experiment; this table shows what happens when a real front-end supplies the matches.
