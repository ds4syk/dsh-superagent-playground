# P2 inner solver baseline: multi-start BFGS

Inner problem: $\min_{q\in S^3} v_2(q)^\top M v_2(q)$; `solve_inner(M, n_starts=K)` with K random starts, BFGS gtol=1e-10.
Instances: p in {8,16,32,64} x 6 seeds, noise=0.01, bias_z=0.08; cases `unc` (unconstrained Mred) and `alm` (M_A at the ALM active set).
Reference = best value over the tested start counts. Gap = f_K - f_ref.

## case: all

| starts K | n | # not best (>1e-9) | # gap > 1e-6 | max gap | median gap | time median (ms) |
|---|---|---|---|---|---|---|
| 4 | 48 | 8 | 8 | 1.17e-03 | 4.1e-18 | 11.96 |
| 8 | 48 | 0 | 0 | 2.86e-17 | 2.2e-18 | 26.35 |
| 16 | 48 | 0 | 0 | 1.65e-17 | 1.3e-18 | 52.31 |
| 64 | 48 | 0 | 0 | 9.54e-18 | 4.3e-19 | 165.76 |
| 256 | 48 | 0 | 0 | 0.00e+00 | 0.0e+00 | 677.71 |

## case: unc

| starts K | n | # not best (>1e-9) | # gap > 1e-6 | max gap | median gap | time median (ms) |
|---|---|---|---|---|---|---|
| 4 | 24 | 8 | 8 | 1.17e-03 | 4.6e-18 | 10.92 |
| 8 | 24 | 0 | 0 | 1.04e-17 | 8.7e-19 | 27.04 |
| 16 | 24 | 0 | 0 | 1.04e-17 | 6.5e-19 | 54.56 |
| 64 | 24 | 0 | 0 | 3.47e-18 | 4.3e-19 | 157.99 |
| 256 | 24 | 0 | 0 | 0.00e+00 | 0.0e+00 | 617.77 |

## case: alm

| starts K | n | # not best (>1e-9) | # gap > 1e-6 | max gap | median gap | time median (ms) |
|---|---|---|---|---|---|---|
| 4 | 24 | 0 | 0 | 3.12e-17 | 4.1e-18 | 12.74 |
| 8 | 24 | 0 | 0 | 2.86e-17 | 2.8e-18 | 24.75 |
| 16 | 24 | 0 | 0 | 1.65e-17 | 1.7e-18 | 49.94 |
| 64 | 24 | 0 | 0 | 9.54e-18 | 0.0e+00 | 190.28 |
| 256 | 24 | 0 | 0 | 0.00e+00 | 0.0e+00 | 740.89 |

Notes:
- `unc` 4 starts under-solves 8/24 instances (gap up to 1.17e-3); 8 starts is already at the reference on all 48 cases, 256 starts costs ~25x more time for no objective gain.
- `alm` M_A is benign here: even 4 starts reaches the reference. So the value of an exact enumerator is *certification of the inner global minimum and removal of the 'numerical global' caveat*, not accuracy on this synthetic benchmark.
- This is a *numerical* baseline; it does not certify a global minimum. The exact enumerator (round R2) is judged against `f_ref` here.

Source: `results/inner_baseline.json` (this script).
