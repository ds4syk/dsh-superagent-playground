# A5: penalty-weight M sweep of the lifted slack form (SCS)

Generated 2026-11-28 01:00:00 by `code/run_sdpr_sweep.py`; p in [8, 16], n=20 per cell, instances identical to `run_sdpr_synth.py` (seed = trial). `certified` uses the sound P2-Cert Schur bound on the same instance; the SCS value is never used as a bound.  `value−hard` is the penalised SDP value minus the hard-relaxation value (a relaxation value may not exceed it).

| p | M | ok/n | status | value−hard med (max) | pen mm med [q1,q3] | pen>1mm | rot med | trans med | feasible | certified (sound) | ms med | s_max med | n_s>0 med |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 8 | 100 | 20/20 | optimal:20 | -1.07e-09 (+4.77e-08) | 0.000 [0.000, 0.000] | 0% | 3.55 | 10.30 | 90% [70, 97] | 90% [70, 97] | 614 | 7.78e-13 | 0.0 |
| 8 | 1000 | 20/20 | optimal:4, optimal_inaccurate:16 | +8.73e-07 (+1.10e-05) | 0.002 [0.000, 0.010] | 0% | 3.71 | 10.30 | 40% [22, 61] | 40% [22, 61] | 1655 | 3.31e-13 | 0.0 |
| 8 | 10000 | 20/20 | optimal:1, optimal_inaccurate:19 | +6.42e-05 (+1.29e-04) | 0.009 [0.002, 0.043] | 0% | 3.91 | 10.36 | 20% [8, 42] | 15% [5, 36] | 1949 | 4.71e-13 | 0.0 |
| 16 | 100 | 20/20 | optimal:19, optimal_inaccurate:1 | -2.98e-10 (+1.95e-07) | 0.000 [0.000, 0.000] | 0% | 3.55 | 10.30 | 85% [64, 95] | 85% [64, 95] | 1102 | 6.30e-13 | 0.0 |
| 16 | 1000 | 20/20 | optimal:9, optimal_inaccurate:11 | +2.46e-08 (+1.16e-05) | 0.000 [0.000, 0.004] | 0% | 3.55 | 10.30 | 65% [43, 82] | 65% [43, 82] | 3557 | 2.05e-13 | 0.0 |
| 16 | 10000 | 20/20 | optimal_inaccurate:20 | +3.53e-05 (+1.65e-04) | 0.016 [0.005, 0.091] | 0% | 4.16 | 10.36 | 20% [8, 42] | 10% [3, 30] | 3896 | 2.87e-12 | 0.0 |

Hard-relaxation reference on the same instances:

| p | n | hard value med | hard pen mm med | hard feasible | hard ms med |
|---|---|---|---|---|---|
| 8 | 20 | 3.952e-03 | 0.000 | 95% | 30 |
| 16 | 20 | 3.952e-03 | 0.000 | 100% | 33 |

