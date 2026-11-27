# A6: slack PSD block size and solver status versus p

Generated 2026-11-28 01:10:44 by `code/analyze_sdpr_status.py`.

## Direct probe (one instance per p, seed 0)

| p | expected PSD (14+p) | returned slack X | n_eq | n_ineq | status | solver | attempts | slack ms | hard X | unc X |
|---|---|---|---|---|---|---|---|---|---|---|
| 8 | 22 | **22×22** | 47 | 8 | optimal | SCS | 1 | 62 | 14 | 14 |
| 16 | 30 | **30×30** | 55 | 16 | optimal | SCS | 1 | 167 | 14 | 14 |
| 32 | 46 | **46×46** | 71 | 32 | optimal | SCS | 1 | 419 | 14 | 14 |
| 64 | 78 | **78×78** | 103 | 64 | optimal | SCS | 1 | 1372 | 14 | 14 |

## Status census over all measured rows

| cell | p | n | variant | PSD | n_eq | n_ineq | status | retried | valid | solver used | ms med | ms max |
|---|---|---|---|---|---|---|---|---|---|---|---|
| synth | 8 | 50 | unc | 14.0 | 39.0 | 0.0 | optimal:50 | 0 | 50 | SCS:50 | 31.5 | 151.9 |
| synth | 8 | 50 | hard | 14.0 | 39.0 | 8.0 | optimal:50 | 0 | 50 | SCS:50 | 30.0 | 50.0 |
| synth | 8 | 50 | slack | 22.0 | 47.0 | 8.0 | optimal:50 | 0 | 50 | SCS:50 | 62.9 | 114.1 |
| synth | 16 | 50 | unc | 14.0 | 39.0 | 0.0 | optimal:50 | 0 | 50 | SCS:50 | 29.9 | 154.2 |
| synth | 16 | 50 | hard | 14.0 | 39.0 | 16.0 | optimal:50 | 0 | 50 | SCS:50 | 33.0 | 55.5 |
| synth | 16 | 50 | slack | 30.0 | 55.0 | 16.0 | optimal:50 | 0 | 50 | SCS:50 | 178.3 | 306.0 |
| synth | 32 | 50 | unc | 14.0 | 39.0 | 0.0 | optimal:50 | 0 | 50 | SCS:50 | 31.5 | 265.0 |
| synth | 32 | 50 | hard | 14.0 | 39.0 | 32.0 | optimal:50 | 0 | 50 | SCS:50 | 41.7 | 391.7 |
| synth | 32 | 50 | slack | 46.0 | 71.0 | 32.0 | optimal:50 | 4 | 50 | CLARABEL:4, SCS:46 | 532.4 | 7840.6 |
| synth | 64 | 50 | unc | 14.0 | 39.0 | 0.0 | optimal:50 | 0 | 50 | SCS:50 | 34.7 | 215.8 |
| synth | 64 | 50 | hard | 14.0 | 39.0 | 64.0 | optimal:50 | 0 | 50 | SCS:50 | 59.1 | 150.3 |
| synth | 64 | 50 | slack | 78.0 | 103.0 | 64.0 | optimal:50 | 0 | 50 | SCS:50 | 2405.1 | 4915.4 |
| LINEMOD | 16 | 90 | unc | 14 | 39 | 0 | optimal:71, optimal_inaccurate:19 | 0 | 90 | None:90 | 197.7 | 658.0 |
| LINEMOD | 16 | 90 | hard | 14 | 39 | 16 | optimal:18, optimal_inaccurate:72 | 0 | 90 | None:90 | 675.1 | 1027.7 |
| LINEMOD | 16 | 90 | slack | 30 | 55 | 16 | optimal:11, optimal_inaccurate:79 | 0 | 90 | None:90 | 3168.6 | 3969.1 |
| LM-O | 16 | 72 | unc | 14 | 39 | 0 | optimal:71, optimal_inaccurate:1 | 0 | 72 | None:72 | 132.1 | 623.7 |
| LM-O | 16 | 72 | hard | 14 | 39 | 16 | optimal:8, optimal_inaccurate:64 | 0 | 72 | None:72 | 669.1 | 956.9 |
| LM-O | 16 | 72 | slack | 30 | 55 | 16 | optimal:18, optimal_inaccurate:54 | 0 | 72 | None:72 | 3149.8 | 3778.3 |
| YCB-Video | 16 | 84 | unc | 14 | 39 | 0 | optimal:63, optimal_inaccurate:21 | 0 | 84 | None:84 | 238.3 | 811.6 |
| YCB-Video | 16 | 84 | hard | 14 | 39 | 16 | optimal:20, optimal_inaccurate:64 | 0 | 84 | None:84 | 666.5 | 852.1 |
| YCB-Video | 16 | 84 | slack | 30 | 55 | 16 | optimal:13, optimal_inaccurate:71 | 0 | 84 | None:84 | 3185.1 | 4383.0 |

## Relaxation-value sanity: slack value − hard value

The slack lift is the *tighter* relaxation (its feasible set projects into the hard one, and the objective ignores the slacks), so `hard_value <= slack_value` must hold; a row below the hard value by more than 1e-6 is numerically invalid.  A positive difference is the measured tightening gap.

| cell | n | median | max | min | rows below hard by >1e-6 |
|---|---|---|---|---|---|
| synth p=8 | 50 | +3.50e-10 | +1.99e-08 | -7.12e-09 | 0 |
| synth p=16 | 50 | +2.06e-10 | +1.68e-08 | -2.76e-08 | 0 |
| synth p=32 | 50 | -2.41e-10 | +1.01e-08 | -2.38e-08 | 0 |
| synth p=64 | 50 | +7.68e-10 | +1.47e-08 | -1.86e-08 | 0 |
| LINEMOD | 90 | +9.54e-06 | +2.45e-04 | -2.38e-05 | 13 |
| LM-O | 72 | +1.50e-06 | +3.57e-04 | -4.29e-05 | 11 |
| YCB-Video | 84 | +1.19e-05 | +3.11e-04 | -5.68e-05 | 14 |

