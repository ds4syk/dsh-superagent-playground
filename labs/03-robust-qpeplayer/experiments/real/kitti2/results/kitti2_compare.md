# KITTI drive 0002: cross-drive check against drive 0001

Same loader/runner protocol as the parent `real/kitti/` (drive 0001): OXTS Mercator (Earth radius 6378137 m), `calib_imu_to_velo`, T_rel_velo, direction checked on 6 consecutive pairs, voxel 0.15 m, tau 0.45 m, c 0.25 m, N=300, 10 consecutive-frame pairs spread over the 77-frame sequence, 20 trials/cell, p=0..0.7.

Determinism: the two independent runs differ only in timing fields (non-timing differences: 0).

## random: success % (drive 0002 / drive 0001)

| method | 0% | 10% | 20% | 30% | 40% | 50% | 60% | 70% |
|---|---|---|---|---|---|---|---|---|
| Kabsch | 100 / 100 | 10 / 10 | 0 / 0 | 0 / 5 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| RANSAC-1000 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 |
| RANSAC-adaptive | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 |
| IRLS-GM | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 |
| GNC-GM | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 |

## random: median rotation error deg (drive 0002 / drive 0001)

| method | 0% | 10% | 20% | 30% | 40% | 50% | 60% | 70% |
|---|---|---|---|---|---|---|---|---|
| Kabsch | 0.04 / 0.04 | 0.70 / 0.49 | 0.77 / 0.91 | 0.83 / 1.75 | 1.24 / 1.44 | 2.11 / 1.80 | 3.22 / 4.29 | 5.13 / 5.77 |
| RANSAC-1000 | 0.05 / 0.04 | 0.05 / 0.03 | 0.05 / 0.04 | 0.05 / 0.04 | 0.05 / 0.05 | 0.04 / 0.05 | 0.07 / 0.05 | 0.07 / 0.06 |
| RANSAC-adaptive | 0.05 / 0.04 | 0.06 / 0.03 | 0.06 / 0.04 | 0.05 / 0.04 | 0.06 / 0.05 | 0.05 / 0.04 | 0.07 / 0.05 | 0.06 / 0.06 |
| IRLS-GM | 0.04 / 0.03 | 0.05 / 0.03 | 0.05 / 0.03 | 0.04 / 0.04 | 0.05 / 0.04 | 0.05 / 0.04 | 0.07 / 0.04 | 0.05 / 0.06 |
| GNC-GM | 0.04 / 0.03 | 0.05 / 0.03 | 0.05 / 0.03 | 0.04 / 0.04 | 0.05 / 0.04 | 0.05 / 0.04 | 0.07 / 0.04 | 0.05 / 0.06 |

## random: runtime p50 ms (drive 0002, min of 2 runs)

| method | 0% | 10% | 20% | 30% | 40% | 50% | 60% | 70% |
|---|---|---|---|---|---|---|---|---|
| Kabsch | 0.06 | 0.07 | 0.07 | 0.09 | 0.12 | 0.07 | 0.07 | 0.07 |
| RANSAC-1000 | 61.57 | 61.00 | 61.25 | 85.97 | 104.11 | 62.61 | 62.33 | 62.56 |
| RANSAC-adaptive | 0.67 | 0.90 | 1.25 | 3.07 | 4.67 | 5.70 | 10.68 | 26.11 |
| IRLS-GM | 2.61 | 2.60 | 2.60 | 3.75 | 4.48 | 2.65 | 2.69 | 2.65 |
| GNC-GM | 0.95 | 2.66 | 2.71 | 4.14 | 4.69 | 2.77 | 2.81 | 2.81 |

## coherent: success % (drive 0002 / drive 0001)

| method | 0% | 10% | 20% | 30% | 40% | 50% | 60% | 70% |
|---|---|---|---|---|---|---|---|---|
| Kabsch | 100 / 100 | 5 / 10 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| RANSAC-1000 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 | 0 / 0 | 0 / 0 | 0 / 0 |
| RANSAC-adaptive | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 | 0 / 0 | 0 / 0 | 0 / 0 |
| IRLS-GM | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 | 30 / 45 | 5 / 0 | 0 / 0 |
| GNC-GM | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 | 40 / 35 | 0 / 0 | 0 / 0 |

## coherent: median rotation error deg (drive 0002 / drive 0001)

| method | 0% | 10% | 20% | 30% | 40% | 50% | 60% | 70% |
|---|---|---|---|---|---|---|---|---|
| Kabsch | 0.04 / 0.05 | 4.13 / 4.00 | 8.66 / 8.34 | 13.11 / 12.65 | 19.22 / 18.80 | 21.58 / 20.07 | 26.27 / 26.82 | 35.03 / 34.85 |
| RANSAC-1000 | 0.03 / 0.04 | 0.05 / 0.04 | 0.05 / 0.04 | 0.05 / 0.04 | 0.06 / 0.04 | 39.90 / 39.90 | 46.73 / 46.73 | 50.47 / 50.47 |
| RANSAC-adaptive | 0.04 / 0.04 | 0.06 / 0.05 | 0.06 / 0.04 | 0.05 / 0.04 | 0.07 / 0.04 | 39.90 / 39.90 | 46.73 / 46.73 | 50.47 / 50.47 |
| IRLS-GM | 0.03 / 0.04 | 0.04 / 0.04 | 0.05 / 0.03 | 0.04 / 0.04 | 0.06 / 0.04 | 34.87 / 26.14 | 42.83 / 46.73 | 50.47 / 50.47 |
| GNC-GM | 0.03 / 0.04 | 0.04 / 0.04 | 0.05 / 0.03 | 0.04 / 0.04 | 0.06 / 0.04 | 33.93 / 34.56 | 46.73 / 46.73 | 50.47 / 50.47 |

## coherent: runtime p50 ms (drive 0002, min of 2 runs)

| method | 0% | 10% | 20% | 30% | 40% | 50% | 60% | 70% |
|---|---|---|---|---|---|---|---|---|
| Kabsch | 0.11 | 0.08 | 0.07 | 0.07 | 0.07 | 0.07 | 0.07 | 0.07 |
| RANSAC-1000 | 99.42 | 67.84 | 63.04 | 63.26 | 63.24 | 63.11 | 63.92 | 63.32 |
| RANSAC-adaptive | 0.99 | 0.97 | 1.32 | 1.92 | 3.18 | 3.39 | 1.96 | 1.19 |
| IRLS-GM | 4.32 | 2.67 | 2.66 | 2.65 | 2.63 | 2.62 | 2.66 | 2.67 |
| GNC-GM | 1.56 | 2.45 | 2.37 | 2.30 | 2.29 | 2.16 | 2.30 | 2.35 |

## boundary: success % (drive 0002 / drive 0001)

| method | 0% | 10% | 20% | 30% | 40% | 50% | 60% | 70% |
|---|---|---|---|---|---|---|---|---|
| Kabsch | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 |
| RANSAC-1000 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 |
| RANSAC-adaptive | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 95 |
| IRLS-GM | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 |
| GNC-GM | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 |

## boundary: median rotation error deg (drive 0002 / drive 0001)

| method | 0% | 10% | 20% | 30% | 40% | 50% | 60% | 70% |
|---|---|---|---|---|---|---|---|---|
| Kabsch | 0.05 / 0.04 | 0.04 / 0.04 | 0.06 / 0.05 | 0.06 / 0.06 | 0.05 / 0.06 | 0.06 / 0.06 | 0.07 / 0.06 | 0.08 / 0.07 |
| RANSAC-1000 | 0.05 / 0.04 | 0.05 / 0.04 | 0.06 / 0.05 | 0.06 / 0.06 | 0.06 / 0.07 | 0.08 / 0.08 | 0.12 / 0.11 | 0.16 / 0.14 |
| RANSAC-adaptive | 0.05 / 0.04 | 0.04 / 0.05 | 0.05 / 0.05 | 0.07 / 0.05 | 0.09 / 0.08 | 0.09 / 0.09 | 0.14 / 0.12 | 0.17 / 0.18 |
| IRLS-GM | 0.04 / 0.04 | 0.04 / 0.03 | 0.05 / 0.04 | 0.04 / 0.04 | 0.04 / 0.04 | 0.05 / 0.04 | 0.08 / 0.05 | 0.09 / 0.07 |
| GNC-GM | 0.04 / 0.04 | 0.04 / 0.03 | 0.05 / 0.04 | 0.04 / 0.04 | 0.04 / 0.04 | 0.05 / 0.04 | 0.08 / 0.05 | 0.09 / 0.07 |

## boundary: runtime p50 ms (drive 0002, min of 2 runs)

| method | 0% | 10% | 20% | 30% | 40% | 50% | 60% | 70% |
|---|---|---|---|---|---|---|---|---|
| Kabsch | 0.07 | 0.07 | 0.07 | 0.07 | 0.07 | 0.07 | 0.07 | 0.07 |
| RANSAC-1000 | 63.41 | 62.15 | 62.91 | 62.61 | 62.94 | 62.56 | 61.62 | 60.80 |
| RANSAC-adaptive | 0.65 | 0.76 | 1.04 | 1.40 | 1.91 | 2.74 | 3.71 | 5.69 |
| IRLS-GM | 2.65 | 2.62 | 2.61 | 2.62 | 2.64 | 2.61 | 2.60 | 2.58 |
| GNC-GM | 0.96 | 0.95 | 0.95 | 0.95 | 0.96 | 0.95 | 0.95 | 0.94 |

## Speedup vs RANSAC-1000 (drive 0002)

| outlier | method | 0% | 10% | 20% | 30% | 40% | 50% | 60% | 70% |
|---|---|---|---|---|---|---|---|---|---|
| random | GNC-GM | 64.9x | 22.9x | 22.6x | 20.8x | 22.2x | 22.6x | 22.2x | 22.2x |
| random | IRLS-GM | 23.6x | 23.5x | 23.6x | 22.9x | 23.3x | 23.6x | 23.2x | 23.6x |
| random | RANSAC-adaptive | 91.9x | 67.7x | 49.0x | 28.0x | 22.3x | 11.0x | 5.8x | 2.4x |
| coherent | GNC-GM | 63.6x | 27.7x | 26.6x | 27.4x | 27.6x | 29.3x | 27.8x | 27.0x |
| coherent | IRLS-GM | 23.0x | 25.4x | 23.7x | 23.9x | 24.1x | 24.1x | 24.1x | 23.7x |
| coherent | RANSAC-adaptive | 100.4x | 70.0x | 47.6x | 33.0x | 19.9x | 18.6x | 32.5x | 53.1x |
| boundary | GNC-GM | 66.1x | 65.2x | 65.9x | 65.8x | 65.5x | 65.7x | 64.8x | 64.9x |
| boundary | IRLS-GM | 23.9x | 23.8x | 24.1x | 23.9x | 23.9x | 24.0x | 23.7x | 23.6x |
| boundary | RANSAC-adaptive | 98.3x | 81.3x | 60.2x | 44.9x | 33.0x | 22.8x | 16.6x | 10.7x |

## Coherent outliers: fraction of trials whose estimate is closer to the injected competing transform than to the GT pose (drive 0002 / drive 0001)

| method | 0% | 10% | 20% | 30% | 40% | 50% | 60% | 70% |
|---|---|---|---|---|---|---|---|---|
| Kabsch | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.20 / 0.30 | 0.75 / 0.95 | 0.85 / 0.90 |
| RANSAC-1000 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
| RANSAC-adaptive | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
| IRLS-GM | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.70 / 0.55 | 0.95 / 1.00 | 1.00 / 1.00 |
| GNC-GM | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.60 / 0.65 | 1.00 / 1.00 | 1.00 / 1.00 |

## Verdict on the drive-0001 conclusions

- PASS — random: all four robust methods 100% success for p=0.1..0.7 (drive 0002 RANSAC-1000:1.00, RANSAC-adaptive:1.00, IRLS-GM:1.00, GNC-GM:1.00)
- PASS — random: Kabsch collapses once p>=0.1 (success <=15%) (drive 0002 Kabsch p=0.1:10%, p=0.2:0%, p=0.3:0%, p=0.4:0%, p=0.5:0%, p=0.6:0%, p=0.7:0%)
- PASS — random: GNC approx. 20x faster than fixed 1000-iteration RANSAC (drive 0002 GNC speedup p=0.1:22.9x, p=0.2:22.6x, p=0.3:20.8x, p=0.4:22.2x, p=0.5:22.6x, p=0.6:22.2x, p=0.7:22.2x)
- PASS — random: robust methods keep sub-0.1 deg median rotation error (max median rot over cells = 0.073 deg)
- PASS — coherent: majority-outlier regime (p>=0.6) loses the GT pose (<=10% success) (drive 0002 coherent success at p=0.6/0.7: RANSAC-1000:0/0%, RANSAC-adaptive:0/0%, IRLS-GM:5/0%, GNC-GM:0/0%)
- PASS — coherent at p=0.5: GNC/IRLS retain a minority of successes, RANSAC does not (drive 0002 p=0.5 GNC 40%, IRLS 30%, RANSAC 0% (drive 0001: GNC 35%, IRLS 45%, RANSAC 0%))
- PASS — boundary: 100% success for all methods, all p (drive 0002 all 5 methods 100% at p=0..0.7)
- PASS — boundary: GNC median rotation error below RANSAC's at p>=0.4 (drive 0002 p=0.7 GNC 0.092 deg vs RANSAC 0.160 deg)
- PASS — cross-drive: random/boundary success agrees with drive 0001 to <=10 points in all cells (max |Delta success| over 80 random+boundary cells = 5 points)
- PASS — cross-drive: coherent p=0.5 GNC success within 10 points of drive 0001 (drive 0002 40% vs drive 0001 35%)

All checked conclusions of the drive-0001 README therefore hold on drive 0002, with one expected difference: the absolute noise floor is slightly higher on this drive (GNC random translation median 0.014-0.026 m vs 0.008-0.012 m on drive 0001; rotation median 0.04-0.07 deg vs 0.03-0.06 deg), still far below the 0.1 m success threshold. The coherent-majority transition sits at p≈0.5-0.6 on both drives.

