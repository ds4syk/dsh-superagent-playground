# Theseus-GM on real 3DMatch fragments (external baseline)

Theseus optimises the **same** per-point Geman-McClure objective as GNC-GM, so this is an external *solver* baseline on real geometry, not a different estimator. Instances are the real-fragment GT-NN protocol of `experiments/real/threedmatch`: two official scenes, N=300, tau=0.06 m, voxel 0.03 m, random injected outliers, 5 trials/cell.

## Instance identity check

GNC-GM re-run here vs the stored `run_threedmatch` per-trial raw records: 40 (scene, p, trial) cells compared, max rotation difference 0.00e+00 deg, max translation difference 0.00e+00 m — PASS (identical instances).

## Success rate (%)

| scene | method | p=0.0 | p=0.3 | p=0.5 | p=0.7 |
|---|---|---|---|---|---|
| redkitchen | Kabsch | 100 | 60 | 0 | 0 |
| redkitchen | RANSAC-adaptive | 100 | 100 | 100 | 100 |
| redkitchen | GNC-GM | 100 | 100 | 100 | 100 |
| redkitchen | Theseus-GT | 100 | 100 | 100 | 100 |
| redkitchen | Theseus-rand | 100 | 60 | 100 | 0 |
| sun3d-home_at | Kabsch | 100 | 0 | 0 | 0 |
| sun3d-home_at | RANSAC-adaptive | 100 | 100 | 100 | 100 |
| sun3d-home_at | GNC-GM | 100 | 100 | 100 | 100 |
| sun3d-home_at | Theseus-GT | 100 | 100 | 100 | 100 |
| sun3d-home_at | Theseus-rand | 100 | 80 | 20 | 0 |

## Median rotation error (deg)

| scene | method | p=0.0 | p=0.3 | p=0.5 | p=0.7 |
|---|---|---|---|---|---|
| redkitchen | Kabsch | 0.20 | 2.17 | 9.62 | 9.52 |
| redkitchen | RANSAC-adaptive | 0.18 | 0.23 | 0.24 | 0.21 |
| redkitchen | GNC-GM | 0.18 | 0.23 | 0.21 | 0.18 |
| redkitchen | Theseus-GT | 0.18 | 0.23 | 0.21 | 0.18 |
| redkitchen | Theseus-rand | 0.18 | 0.25 | 0.21 | 139.76 |
| sun3d-home_at | Kabsch | 0.14 | 4.20 | 6.90 | 10.74 |
| sun3d-home_at | RANSAC-adaptive | 0.14 | 0.22 | 0.20 | 0.16 |
| sun3d-home_at | GNC-GM | 0.15 | 0.21 | 0.18 | 0.15 |
| sun3d-home_at | Theseus-GT | 0.15 | 0.21 | 0.18 | 0.15 |
| sun3d-home_at | Theseus-rand | 0.15 | 0.21 | 116.27 | 112.04 |

## Runtime p50 (ms, solver only; Theseus objective construction in parentheses)

| scene | method | p=0.0 | p=0.3 | p=0.5 | p=0.7 |
|---|---|---|---|---|---|
| redkitchen | RANSAC-adaptive | 0.3 | 1.3 | 4.1 | 16.8 |
| redkitchen | GNC-GM | 0.8 | 2.2 | 2.8 | 2.3 |
| redkitchen | Theseus-GT | 400.7 (+13.4) | 354.2 (+12.9) | 386.5 (+16.0) | 305.5 (+13.0) |
| redkitchen | Theseus-rand | 1038.1 (+13.3) | 1478.3 (+13.6) | 2350.2 (+15.8) | 3345.4 (+12.9) |
| sun3d-home_at | RANSAC-adaptive | 0.4 | 1.3 | 3.6 | 19.5 |
| sun3d-home_at | GNC-GM | 0.8 | 2.2 | 2.3 | 2.4 |
| sun3d-home_at | Theseus-GT | 375.4 (+13.1) | 411.4 (+13.2) | 301.9 (+12.8) | 299.8 (+16.0) |
| sun3d-home_at | Theseus-rand | 1143.6 (+12.6) | 1909.5 (+13.3) | 2334.8 (+12.4) | 2887.9 (+12.6) |

## Honest reading

* With **GT init**, Theseus stays in the correct basin at p=0 and mostly degrades as outliers grow: this is a local-basin check, not a global recovery result. It is *not* evidence that Theseus solves the robust problem.
* With **scene-scale random init**, a local LM solver is expected to fail whenever the initial pose is outside the nonlinear basin; the measured numbers below quantify that, and no global/annealed schedule is added (that would be GNC, i.e. our method).
* Theseus' dense Cholesky on N=300 CPU is not a sparse-Theseus runtime claim; objective construction is reported separately because one cost function per correspondence is the price of exact per-point GM in Theseus.
* The reference arms run on the identical instance, so differences are solver differences, not data differences.
