# ModelNet10 vs ModelNet40: does the conclusion hold?

Same protocol in both tables: N=200, sigma=0.01, c=0.05, 50 trials/cell, uniform outliers, seed 20261127, Kabsch / RANSAC-1000 / IRLS-GM / GNC-GM. Cells are `ModelNet40 / ModelNet10`.

## success rate (%)

| method | 0% | 10% | 20% | 30% | 40% | 50% | 60% | 70% |
|-|---|---|---|---|---|---|---|---|
| Kabsch | 100 / 100 | 88 / 100 | 48 / 60 | 22 / 24 | 4 / 2 | 0 / 0 | 2 / 0 | 0 / 0 |
| RANSAC+Kabsch | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 |
| IRLS-GM | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 | 98 / 100 |
| GNC-GM (ours) | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 |

## success delta, ModelNet10 - ModelNet40 (percentage points)

| method | 0% | 10% | 20% | 30% | 40% | 50% | 60% | 70% |
|-|---|---|---|---|---|---|---|---|
| Kabsch | +0 | +12 | +12 | +2 | -2 | +0 | -2 | +0 |
| RANSAC+Kabsch | +0 | +0 | +0 | +0 | +0 | +0 | +0 | +0 |
| IRLS-GM | +0 | +0 | +0 | +0 | +0 | +0 | +0 | +2 |
| GNC-GM (ours) | +0 | +0 | +0 | +0 | +0 | +0 | +0 | +0 |

## median rotation error, deg

| method | 0% | 10% | 20% | 30% | 40% | 50% | 60% | 70% |
|-|---|---|---|---|---|---|---|---|
| Kabsch | 0.15 / 0.13 | 2.36 / 2.76 | 4.67 / 4.11 | 5.19 / 5.72 | 8.84 / 8.23 | 12.29 / 10.81 | 14.87 / 15.68 | 24.94 / 22.96 |
| RANSAC+Kabsch | 0.15 / 0.13 | 0.15 / 0.14 | 0.17 / 0.14 | 0.20 / 0.16 | 0.19 / 0.15 | 0.20 / 0.21 | 0.20 / 0.21 | 0.25 / 0.28 |
| IRLS-GM | 0.15 / 0.13 | 0.15 / 0.14 | 0.17 / 0.14 | 0.19 / 0.16 | 0.18 / 0.15 | 0.21 / 0.21 | 0.21 / 0.22 | 0.26 / 0.29 |
| GNC-GM (ours) | 0.15 / 0.13 | 0.15 / 0.14 | 0.17 / 0.14 | 0.19 / 0.16 | 0.18 / 0.15 | 0.21 / 0.21 | 0.21 / 0.22 | 0.26 / 0.29 |

## median runtime, ms

| method | 0% | 10% | 20% | 30% | 40% | 50% | 60% | 70% |
|-|---|---|---|---|---|---|---|---|
| Kabsch | 0.1 / 0.1 | 0.1 / 0.1 | 0.1 / 0.1 | 0.1 / 0.1 | 0.1 / 0.1 | 0.1 / 0.1 | 0.1 / 0.1 | 0.1 / 0.1 |
| RANSAC+Kabsch | 0.2 / 0.2 | 56.1 / 61.2 | 55.4 / 61.5 | 55.1 / 61.2 | 54.0 / 60.6 | 54.7 / 59.6 | 53.7 / 58.9 | 53.5 / 61.2 |
| IRLS-GM | 2.9 / 2.3 | 2.3 / 2.3 | 2.2 / 2.4 | 2.1 / 2.3 | 2.2 / 2.3 | 2.2 / 2.3 | 2.1 / 2.3 | 2.1 / 2.4 |
| GNC-GM (ours) | 0.7 / 0.5 | 1.8 / 1.9 | 1.7 / 1.9 | 1.8 / 1.9 | 1.8 / 1.9 | 1.8 / 1.9 | 1.7 / 1.9 | 1.7 / 1.9 |

## checks

- GNC-GM success at all p: ModelNet40 100%..100%, ModelNet10 100%..100%.
- RANSAC+Kabsch success at all p: ModelNet40 100%..100%, ModelNet10 100%..100%.
- IRLS-GM worst cell: ModelNet40 98%, ModelNet10 100%.
- Largest success deviation on any robust-method cell: 2 pp.
- Largest median-rotation deviation on any robust-method cell: 0.038 deg.
- RANSAC-1000 / GNC runtime ratio (p>0): ModelNet40 30.1x..31.7x, ModelNet10 31.7x..32.5x.
- Kabsch collapse (first p with 0% success): ModelNet40 50%, ModelNet10 50%.

## verdict

**Holds.** 
On the second real dataset GNC-GM keeps 100% success across 0-70% outliers, matches RANSAC/IRLS rotation accuracy to <0.05 deg, keeps the ~30x speedup over fixed 1000-iteration RANSAC, and keeps the same degradation pattern for the non-robust Kabsch baseline. The two real tables are therefore consistent with each other and with the synthetic main table.
