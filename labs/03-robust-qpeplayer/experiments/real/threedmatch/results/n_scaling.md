# 3DMatch real fragments: N scaling

20 trials/cell, p in [0.3, 0.7], GT-NN oracle base. Runtime p50/p95 per call.

| scene | p | N | method | success | rot med (deg) | p50 (ms) | p95 (ms) |
|---|---|---|---|---|---|---|---|
| redkitchen | 0.3 | 50 | Kabsch | 25% | 7.60 | 0.05 | 0.08 |
| redkitchen | 0.3 | 50 | RANSAC-1000 | 100% | 0.39 | 55.49 | 64.67 |
| redkitchen | 0.3 | 50 | IRLS-GM | 100% | 0.32 | 1.89 | 2.04 |
| redkitchen | 0.3 | 50 | GNC-GM | 100% | 0.32 | 1.43 | 1.55 |
| redkitchen | 0.7 | 50 | Kabsch | 0% | 27.28 | 0.06 | 0.09 |
| redkitchen | 0.7 | 50 | RANSAC-1000 | 100% | 0.63 | 56.26 | 89.30 |
| redkitchen | 0.7 | 50 | IRLS-GM | 75% | 0.58 | 1.91 | 3.14 |
| redkitchen | 0.7 | 50 | GNC-GM | 85% | 0.58 | 1.47 | 2.40 |
| redkitchen | 0.3 | 100 | Kabsch | 50% | 3.95 | 0.06 | 0.07 |
| redkitchen | 0.3 | 100 | RANSAC-1000 | 100% | 0.38 | 57.19 | 58.88 |
| redkitchen | 0.3 | 100 | IRLS-GM | 100% | 0.33 | 1.97 | 2.19 |
| redkitchen | 0.3 | 100 | GNC-GM | 100% | 0.33 | 1.58 | 1.83 |
| redkitchen | 0.7 | 100 | Kabsch | 0% | 20.22 | 0.07 | 0.10 |
| redkitchen | 0.7 | 100 | RANSAC-1000 | 100% | 0.41 | 64.58 | 65.87 |
| redkitchen | 0.7 | 100 | IRLS-GM | 80% | 0.43 | 2.09 | 2.59 |
| redkitchen | 0.7 | 100 | GNC-GM | 95% | 0.39 | 1.70 | 2.08 |
| redkitchen | 0.3 | 300 | Kabsch | 60% | 2.44 | 0.07 | 0.10 |
| redkitchen | 0.3 | 300 | RANSAC-1000 | 100% | 0.27 | 63.67 | 80.84 |
| redkitchen | 0.3 | 300 | IRLS-GM | 100% | 0.22 | 2.70 | 4.20 |
| redkitchen | 0.3 | 300 | GNC-GM | 100% | 0.22 | 2.22 | 3.38 |
| redkitchen | 0.7 | 300 | Kabsch | 0% | 9.84 | 0.07 | 0.11 |
| redkitchen | 0.7 | 300 | RANSAC-1000 | 100% | 0.32 | 62.68 | 64.51 |
| redkitchen | 0.7 | 300 | IRLS-GM | 95% | 0.32 | 2.71 | 2.78 |
| redkitchen | 0.7 | 300 | GNC-GM | 100% | 0.32 | 2.18 | 2.35 |
| redkitchen | 0.3 | 1000 | Kabsch | 70% | 1.26 | 0.11 | 0.15 |
| redkitchen | 0.3 | 1000 | RANSAC-1000 | 100% | 0.24 | 82.35 | 98.38 |
| redkitchen | 0.3 | 1000 | IRLS-GM | 100% | 0.22 | 5.12 | 5.73 |
| redkitchen | 0.3 | 1000 | GNC-GM | 100% | 0.22 | 4.22 | 4.89 |
| redkitchen | 0.7 | 1000 | Kabsch | 15% | 5.92 | 0.11 | 0.15 |
| redkitchen | 0.7 | 1000 | RANSAC-1000 | 100% | 0.24 | 81.92 | 105.57 |
| redkitchen | 0.7 | 1000 | IRLS-GM | 100% | 0.23 | 5.09 | 5.48 |
| redkitchen | 0.7 | 1000 | GNC-GM | 100% | 0.23 | 4.17 | 5.05 |
| sun3d-home_at | 0.3 | 50 | Kabsch | 5% | 5.87 | 0.05 | 0.08 |
| sun3d-home_at | 0.3 | 50 | RANSAC-1000 | 100% | 0.28 | 55.16 | 87.75 |
| sun3d-home_at | 0.3 | 50 | IRLS-GM | 100% | 0.25 | 1.88 | 2.94 |
| sun3d-home_at | 0.3 | 50 | GNC-GM | 100% | 0.25 | 1.52 | 2.39 |
| sun3d-home_at | 0.7 | 50 | Kabsch | 0% | 19.75 | 0.05 | 0.07 |
| sun3d-home_at | 0.7 | 50 | RANSAC-1000 | 100% | 0.29 | 54.53 | 55.42 |
| sun3d-home_at | 0.7 | 50 | IRLS-GM | 75% | 0.33 | 1.90 | 1.95 |
| sun3d-home_at | 0.7 | 50 | GNC-GM | 100% | 0.29 | 1.51 | 1.57 |
| sun3d-home_at | 0.3 | 100 | Kabsch | 5% | 5.14 | 0.06 | 0.07 |
| sun3d-home_at | 0.3 | 100 | RANSAC-1000 | 100% | 0.25 | 56.85 | 57.51 |
| sun3d-home_at | 0.3 | 100 | IRLS-GM | 100% | 0.20 | 1.99 | 2.13 |
| sun3d-home_at | 0.3 | 100 | GNC-GM | 100% | 0.20 | 1.65 | 1.77 |
| sun3d-home_at | 0.7 | 100 | Kabsch | 0% | 17.52 | 0.06 | 0.07 |
| sun3d-home_at | 0.7 | 100 | RANSAC-1000 | 100% | 0.26 | 56.38 | 57.96 |
| sun3d-home_at | 0.7 | 100 | IRLS-GM | 100% | 0.24 | 1.96 | 2.09 |
| sun3d-home_at | 0.7 | 100 | GNC-GM | 100% | 0.24 | 1.65 | 1.76 |
| sun3d-home_at | 0.3 | 300 | Kabsch | 40% | 3.40 | 0.08 | 0.10 |
| sun3d-home_at | 0.3 | 300 | RANSAC-1000 | 100% | 0.18 | 68.75 | 75.51 |
| sun3d-home_at | 0.3 | 300 | IRLS-GM | 100% | 0.16 | 2.83 | 3.41 |
| sun3d-home_at | 0.3 | 300 | GNC-GM | 100% | 0.16 | 2.43 | 2.87 |
| sun3d-home_at | 0.7 | 300 | Kabsch | 0% | 14.18 | 0.07 | 0.09 |
| sun3d-home_at | 0.7 | 300 | RANSAC-1000 | 100% | 0.21 | 62.81 | 63.57 |
| sun3d-home_at | 0.7 | 300 | IRLS-GM | 100% | 0.17 | 2.67 | 2.78 |
| sun3d-home_at | 0.7 | 300 | GNC-GM | 100% | 0.17 | 2.30 | 2.56 |
| sun3d-home_at | 0.3 | 1000 | Kabsch | 70% | 1.72 | 0.11 | 0.12 |
| sun3d-home_at | 0.3 | 1000 | RANSAC-1000 | 100% | 0.19 | 81.32 | 82.73 |
| sun3d-home_at | 0.3 | 1000 | IRLS-GM | 100% | 0.14 | 5.04 | 5.26 |
| sun3d-home_at | 0.3 | 1000 | GNC-GM | 100% | 0.14 | 4.38 | 4.47 |
| sun3d-home_at | 0.7 | 1000 | Kabsch | 5% | 5.88 | 0.11 | 0.14 |
| sun3d-home_at | 0.7 | 1000 | RANSAC-1000 | 100% | 0.13 | 81.85 | 88.82 |
| sun3d-home_at | 0.7 | 1000 | IRLS-GM | 100% | 0.12 | 5.08 | 5.31 |
| sun3d-home_at | 0.7 | 1000 | GNC-GM | 100% | 0.12 | 4.37 | 4.77 |

## Reading

* If GNC p50 grows approximately linearly in N while RANSAC-1000 stays flat, the fixed-budget RANSAC advantage is only at small N; at large N its per-iteration scoring cost dominates.
* Kabsch is the non-robust lower bound on runtime and a failure baseline at p>0.
