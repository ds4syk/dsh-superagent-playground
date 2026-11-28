# Inlier-noise sweep

N=100, c=0.05, 50 trials/cell, random outliers. Fixed kernel scale; sigma is the Gaussian inlier noise.

## p=0.3: success rate

| method | σ=0.005 | σ=0.01 | σ=0.02 | σ=0.03 | σ=0.05 |
|---|---|---|---|---|---|
| Kabsch | 2 | 0 | 0 | 0 | 4 |
| RANSAC-1000 | 100 | 100 | 100 | 100 | 100 |
| IRLS-GM | 100 | 100 | 100 | 100 | 100 |
| GNC-GM | 100 | 100 | 100 | 100 | 100 |

## p=0.3: median rotation error (deg)

| method | σ=0.005 | σ=0.01 | σ=0.02 | σ=0.03 | σ=0.05 |
|---|---|---|---|---|---|
| Kabsch | 9.30 | 9.70 | 10.02 | 9.22 | 11.15 |
| RANSAC-1000 | 0.06 | 0.14 | 0.32 | 0.81 | 1.86 |
| IRLS-GM | 0.06 | 0.14 | 0.29 | 0.50 | 1.08 |
| GNC-GM | 0.06 | 0.14 | 0.29 | 0.49 | 1.12 |

## p=0.5: success rate

| method | σ=0.005 | σ=0.01 | σ=0.02 | σ=0.03 | σ=0.05 |
|---|---|---|---|---|---|
| Kabsch | 0 | 0 | 0 | 0 | 0 |
| RANSAC-1000 | 100 | 100 | 100 | 100 | 100 |
| IRLS-GM | 100 | 100 | 100 | 100 | 100 |
| GNC-GM | 100 | 100 | 100 | 100 | 100 |

## p=0.5: median rotation error (deg)

| method | σ=0.005 | σ=0.01 | σ=0.02 | σ=0.03 | σ=0.05 |
|---|---|---|---|---|---|
| Kabsch | 19.67 | 18.46 | 19.29 | 18.91 | 18.49 |
| RANSAC-1000 | 0.08 | 0.16 | 0.43 | 0.83 | 2.36 |
| IRLS-GM | 0.08 | 0.17 | 0.42 | 0.51 | 1.22 |
| GNC-GM | 0.08 | 0.17 | 0.42 | 0.51 | 1.16 |

## p=0.7: success rate

| method | σ=0.005 | σ=0.01 | σ=0.02 | σ=0.03 | σ=0.05 |
|---|---|---|---|---|---|
| Kabsch | 0 | 0 | 0 | 0 | 0 |
| RANSAC-1000 | 100 | 100 | 100 | 100 | 90 |
| IRLS-GM | 98 | 94 | 94 | 96 | 94 |
| GNC-GM | 100 | 100 | 100 | 100 | 100 |

## p=0.7: median rotation error (deg)

| method | σ=0.005 | σ=0.01 | σ=0.02 | σ=0.03 | σ=0.05 |
|---|---|---|---|---|---|
| Kabsch | 36.56 | 40.50 | 35.98 | 30.43 | 38.99 |
| RANSAC-1000 | 0.09 | 0.22 | 0.54 | 1.06 | 2.59 |
| IRLS-GM | 0.09 | 0.23 | 0.50 | 0.78 | 1.73 |
| GNC-GM | 0.09 | 0.22 | 0.49 | 0.78 | 1.67 |

## Reading

* The success threshold is 5 deg / 0.1 m, so high sigma can fail even at 0% outliers; the interesting question is whether the robust ranking changes.
* If GNC only wins when sigma << c, that is an important scope restriction.
