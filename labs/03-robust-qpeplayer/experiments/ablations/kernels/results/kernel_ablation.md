# Kernel ablation: GM vs Cauchy vs Welsch

N=100, c=0.05, 50 trials/cell, gamma=1.4, identical weighted-Kabsch inner and annealing schedule. Success = rot<5 deg and trans<0.1 m.

## random outliers: success rate

| kernel | 0% | 10% | 20% | 30% | 40% | 50% | 60% | 70% |
|---|---|---|---|---|---|---|---|---|
| GM | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| Cauchy | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| Welsch | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |

## random outliers: median rotation error (deg)

| kernel | 0% | 10% | 20% | 30% | 40% | 50% | 60% | 70% |
|---|---|---|---|---|---|---|---|---|
| GM | 0.09 | 0.11 | 0.11 | 0.14 | 0.13 | 0.16 | 0.16 | 0.19 |
| Cauchy | 0.09 | 0.11 | 0.12 | 0.14 | 0.13 | 0.16 | 0.16 | 0.20 |
| Welsch | 0.09 | 0.11 | 0.12 | 0.14 | 0.13 | 0.15 | 0.16 | 0.20 |

## random outliers: runtime p50 (ms)

| kernel | 0% | 10% | 20% | 30% | 40% | 50% | 60% | 70% |
|---|---|---|---|---|---|---|---|---|
| GM | 0.64 | 2.74 | 1.64 | 1.69 | 1.68 | 1.84 | 2.67 | 2.73 |
| Cauchy | 0.68 | 2.70 | 1.66 | 1.71 | 1.65 | 2.74 | 2.62 | 1.97 |
| Welsch | 0.65 | 1.68 | 1.75 | 1.72 | 1.74 | 2.78 | 2.78 | 2.64 |

## boundary outliers: success rate

| kernel | 0% | 10% | 20% | 30% | 40% | 50% | 60% | 70% |
|---|---|---|---|---|---|---|---|---|
| GM | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| Cauchy | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| Welsch | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |

## boundary outliers: median rotation error (deg)

| kernel | 0% | 10% | 20% | 30% | 40% | 50% | 60% | 70% |
|---|---|---|---|---|---|---|---|---|
| GM | 0.11 | 0.13 | 0.14 | 0.14 | 0.15 | 0.20 | 0.23 | 0.34 |
| Cauchy | 0.11 | 0.13 | 0.14 | 0.16 | 0.19 | 0.24 | 0.26 | 0.37 |
| Welsch | 0.11 | 0.14 | 0.15 | 0.14 | 0.16 | 0.22 | 0.27 | 0.39 |

## boundary outliers: runtime p50 (ms)

| kernel | 0% | 10% | 20% | 30% | 40% | 50% | 60% | 70% |
|---|---|---|---|---|---|---|---|---|
| GM | 0.70 | 0.67 | 0.68 | 0.68 | 0.68 | 1.08 | 1.03 | 0.71 |
| Cauchy | 0.44 | 0.66 | 0.66 | 0.68 | 0.68 | 0.98 | 1.16 | 0.67 |
| Welsch | 0.43 | 0.66 | 0.68 | 0.68 | 0.72 | 1.17 | 1.04 | 0.68 |

## Reading

* All kernels are run with the same schedule, so differences are kernel shape, not tuning. GM is the default because it has a closed-form BR dual and is smooth; Cauchy is heavier-tailed near zero and Welsch is flat near zero.
* A kernel that wins on random outliers but loses on boundary outliers would be a bad default for this paper's gradient story; report both columns.
