# Breakdown sweep

N=100, c=0.05; random 100 trials/cell, coherent 200.

## Random outliers: success rate

| method | 70% | 75% | 80% | 85% | 90% |
|---|---|---|---|---|---|
| Kabsch | 0 | 0 | 0 | 0 | 0 |
| RANSAC-1000 | 100 | 100 | 100 | 94 | 51 |
| RANSAC-adaptive | 100 | 100 | 100 | 94 | 51 |
| IRLS-GM | 95 | 81 | 76 | 64 | 32 |
| GNC-GM | 100 | 99 | 89 | 86 | 62 |

## Coherent outliers: success rate

| method | 40% | 45% | 50% | 55% | 60% |
|---|---|---|---|---|---|
| Kabsch | 0 | 0 | 0 | 0 | 0 |
| RANSAC-1000 | 100 | 100 | 50 | 0 | 0 |
| RANSAC-adaptive | 100 | 100 | 50 | 0 | 0 |
| IRLS-GM | 100 | 98 | 40 | 0 | 0 |
| GNC-GM | 100 | 98 | 52 | 2 | 0 |

## Coherent outliers: fraction that recovers the outlier transform

| method | 40% | 45% | 50% | 55% | 60% |
|---|---|---|---|---|---|
| Kabsch | 0 | 1 | 38 | 79 | 96 |
| RANSAC-1000 | 0 | 0 | 50 | 100 | 100 |
| RANSAC-adaptive | 0 | 0 | 50 | 100 | 100 |
| IRLS-GM | 0 | 0 | 48 | 100 | 100 |
| GNC-GM | 0 | 2 | 48 | 98 | 100 |

## Reading

* Random 80–90% is the fixed-RANSAC failure regime; GNC may still work because its cost does not depend on the clean-sample probability.
* Coherent 45–55% is a coin-flip / identifiability transition, not a solver bug; the recovered-outlier column makes that explicit.
