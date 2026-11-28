# round-rlt raw tables

Sound cut families + rigorous dual repair on the 14x14 Veronese-Shor SDP.
`value` = solver-reported `<Q,X>` (**not** a rigorous bound); `rig` = rigorous dual
bound from `certificate.py repair_dual=True`; `rig gap = J - rig`.

## Summary (16 instances = 4 noise cells x 4 seeds)

| solver | variant | cert naive | cert rigorous | max rig gap | median rig gap | max dual shift | median n_ineq | median t(s) |
|-|-|-|-|-|-|-|-|-|
| SCS | base | 9/16 | 9/16 | 1.15e-02 | 2.37e-08 | 1.2e-06 | 16 | 0.04 |
| SCS | rlt | 10/16 | 10/16 | 9.82e-03 | 2.14e-08 | 3.2e-07 | 152 | 0.16 |
| SCS | bounds | 9/16 | 9/16 | 1.15e-02 | 2.02e-08 | 1.5e-07 | 29 | 0.05 |
| SCS | bound_rlt | 16/16 | 16/16 | 2.28e-08 | 2.01e-08 | 1.1e-08 | 352 | 0.18 |
| SCS | all | 15/16 | 13/16 | 1.32e-03 | 2.00e-08 | 9.0e-04 | 732 | 0.45 |
| CLARABEL | base | 9/16 | 9/16 | 9.44e-03 | 3.00e-08 | 1.2e-08 | 16 | 0.03 |
| CLARABEL | rlt | 10/16 | 10/16 | 9.43e-03 | 2.95e-08 | 5.8e-06 | 152 | 0.11 |
| CLARABEL | bounds | 9/16 | 9/16 | 1.15e-02 | 3.44e-08 | 1.3e-08 | 29 | 0.04 |
| CLARABEL | bound_rlt | 13/16 | 13/16 | 8.51e-05 | 5.56e-08 | 3.8e-05 | 352 | 0.16 |
| CLARABEL | all | 16/16 | 16/16 | 1.46e-07 | 4.33e-08 | 8.5e-08 | 732 | 0.36 |

## Soundness checks

| check | ok | detail |
|-|-|-|
| cuts_valid_on_sampled_feasible_points | True | `{"max_violation": {"g": 9.975437142983878e-11, "rlt": 8.207434731843932e-11, "bounds": -6.769885857986989e-26, "bound_rlt": 1.0918177473229207e-10, "box_rlt": -3.839055851029165e-47}, "tol": 1e-08, "note": "GT + 3 ALM + up to 25 random q with LP-feasible t, per instance (16 instances)"}` |
| solver_failures | True | `{"n": 4, "rows": [{"noise": 0.1, "seed": 1, "variant": "bounds", "solver": "CLARABEL", "status": "error:Solver 'CLARABEL' failed. Try another solver, or solve with verbose=True for more information."}, {"noise": 0.2, "seed": 0, "variant": "rlt", "solver": "CLARABEL", "status": "error:Solver 'CLARABEL' failed. Try another solver, or solve with verbose=True for more information."}, {"noise": 0.2, "seed": 1, "variant": "base", "solver": "CLARABEL", "status": "error:Solver 'CLARABEL' failed. Try another solver, or solve with verbose=True for more information."}, {"noise": 0.2, "seed": 2, "variant": "bound_rlt", "solver": "CLARABEL", "status": "error:Solver 'CLARABEL' failed. Try another solver, or solve with verbose=True for more information."}], "note": "CLARABEL raised on 4/160 solves; SCS solved all 160. Not a cut-validity issue."}` |
| bound_with_cuts_ge_base_naive_value | False | `{"n_violations": 3, "violations": [{"noise": 0.02, "seed": 2, "solver": "SCS", "variant": "all", "base": 0.00802023217710588, "cut": 0.007692618238360196, "delta": -0.0003276139387456845, "status": "optimal_inaccurate", "shift": 0.00021686873895937283, "time": 4.007567271124572}, {"noise": 0.05, "seed": 0, "solver": "CLARABEL", "variant": "bound_rlt", "base": 0.15252599360932204, "cut": 0.15252251769445876, "delta": -3.4759148632712566e-06, "status": "optimal_inaccurate", "shift": 1.9094779712169395e-06, "time": 0.16496706986799836}, {"noise": 0.05, "seed": 3, "solver": "CLARABEL", "variant": "bound_rlt", "base": 0.1236940208616774, "cut": 0.1236871413390433, "delta": -6.879522634103963e-06, "status": "optimal_inaccurate", "shift": 3.827258294279046e-05, "time": 0.15577312419191003}], "note": "all violations are SCS/CLARABEL status=optimal_inaccurate on larger SDPs (non-convergence), not cut unsoundness"}` |
| bound_with_cuts_ge_base_rigorous_bound | False | `{"n_violations": 5, "violations": [{"noise": 0.02, "seed": 2, "solver": "SCS", "variant": "all", "base_rig": 0.008020212150219285, "cut_rig": 0.007113145060868247, "shift": 0.00021686873895937283, "status": "optimal_inaccurate"}, {"noise": 0.02, "seed": 3, "solver": "SCS", "variant": "all", "base_rig": 0.016541655750770105, "cut_rig": 0.016528665447438667, "shift": 6.2523009781295285e-06, "status": "optimal_inaccurate"}, {"noise": 0.05, "seed": 0, "solver": "CLARABEL", "variant": "bound_rlt", "base_rig": 0.15252597365161458, "cut_rig": 0.1525186190715349, "shift": 1.9094779712169395e-06, "status": "optimal_inaccurate"}, {"noise": 0.05, "seed": 2, "solver": "SCS", "variant": "all", "base_rig": 0.06845510413569506, "cut_rig": 0.06713411049892049, "shift": 0.0008953665419385569, "status": "optimal_inaccurate"}, {"noise": 0.05, "seed": 3, "solver": "CLARABEL", "variant": "bound_rlt", "base_rig": 0.12369399975109718, "cut_rig": 0.1236089270539044, "shift": 3.827258294279046e-05, "status": "optimal_inaccurate"}], "note": "rigorous bound is only as good as the solver duals; larger cut sets hurt SCS dual accuracy"}` |
| repaired_dual_slack_psd | True | `{"n_invalid": 0, "invalid": [], "note": "lambda_min(S_rep) >= -1e-10 required for validity"}` |
| rank1_implies_bound_le_J | True | `{"n_rank1": 115, "n_violations": 0, "violations": []}` |
| naive_cert_vs_rigorous_cert_disagreement | True | `{"n": 2, "rows": [{"noise": 0.02, "seed": 3, "variant": "all", "solver": "SCS", "status": "optimal_inaccurate", "gap": -3.9552729971456535e-07, "rgap": 1.3010302043869881e-05, "shift": 6.2523009781295285e-06}, {"noise": 0.05, "seed": 2, "variant": "all", "solver": "SCS", "status": "optimal_inaccurate", "gap": -0.001136036874509222, "rgap": 0.0013210136378428994, "shift": 0.0008953665419385569}], "note": "rows where solver-tolerance cert (naive) differs from the rigorous cert"}` |

## per-instance (SCS)

| noise | seed | variant | J | value | gap | rig | rig gap | ratio | λmin(S_rep) | n_ineq | status | t(s) | cert naive | cert rig |
|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|
| 0.02 | 0 | base | 1.7220e-02 | 1.7220e-02 | +1.75e-09 | 1.7220e-02 | +2.18e-08 | 1.5e-09 | +8.2e-09 | 16.0 | optimal | 0.04 | True | True |
| 0.02 | 0 | rlt | 1.7220e-02 | 1.7220e-02 | -9.04e-11 | 1.7220e-02 | +2.00e-08 | 9.9e-11 | +7.5e-09 | 152.0 | optimal | 0.24 | True | True |
| 0.02 | 0 | bounds | 1.7220e-02 | 1.7220e-02 | +4.96e-11 | 1.7220e-02 | +2.01e-08 | 1.3e-10 | +7.5e-09 | 29.0 | optimal | 0.04 | True | True |
| 0.02 | 0 | bound_rlt | 1.7220e-02 | 1.7220e-02 | -1.25e-10 | 1.7220e-02 | +2.07e-08 | 2.1e-11 | +7.8e-09 | 352.0 | optimal | 0.19 | True | True |
| 0.02 | 0 | all | 1.7220e-02 | 1.7220e-02 | -6.02e-12 | 1.7220e-02 | +2.00e-08 | 8.0e-13 | +7.5e-09 | 732.0 | optimal | 0.45 | True | True |
| 0.02 | 1 | base | 1.3687e-02 | 1.3687e-02 | +1.87e-10 | 1.3687e-02 | +2.02e-08 | 8.1e-12 | +7.7e-09 | 16.0 | optimal | 0.03 | True | True |
| 0.02 | 1 | rlt | 1.3687e-02 | 1.3687e-02 | -1.02e-10 | 1.3687e-02 | +2.00e-08 | 1.2e-11 | +7.6e-09 | 152.0 | optimal | 0.16 | True | True |
| 0.02 | 1 | bounds | 1.3687e-02 | 1.3687e-02 | -1.34e-11 | 1.3687e-02 | +2.00e-08 | 9.2e-12 | +7.6e-09 | 29.0 | optimal | 0.04 | True | True |
| 0.02 | 1 | bound_rlt | 1.3687e-02 | 1.3687e-02 | +1.63e-09 | 1.3687e-02 | +2.18e-08 | 1.8e-11 | +8.2e-09 | 352.0 | optimal | 0.19 | True | True |
| 0.02 | 1 | all | 1.3687e-02 | 1.3687e-02 | -5.77e-11 | 1.3687e-02 | +2.00e-08 | 3.2e-12 | +7.6e-09 | 732.0 | optimal | 0.53 | True | True |
| 0.02 | 2 | base | 8.0202e-03 | 8.0202e-03 | -2.73e-11 | 8.0202e-03 | +2.00e-08 | 2.5e-12 | +7.5e-09 | 16.0 | optimal | 0.03 | True | True |
| 0.02 | 2 | rlt | 8.0202e-03 | 8.0202e-03 | -9.33e-10 | 8.0202e-03 | +2.00e-08 | 8.3e-11 | +7.5e-09 | 152.0 | optimal | 0.13 | True | True |
| 0.02 | 2 | bounds | 8.0202e-03 | 8.0202e-03 | -8.18e-12 | 8.0202e-03 | +2.00e-08 | 1.4e-12 | +7.5e-09 | 29.0 | optimal | 0.04 | True | True |
| 0.02 | 2 | bound_rlt | 8.0202e-03 | 8.0202e-03 | -2.35e-11 | 8.0202e-03 | +2.00e-08 | 1.0e-12 | +7.5e-09 | 352.0 | optimal | 0.20 | True | True |
| 0.02 | 2 | all | 8.0202e-03 | 7.6926e-03 | +3.28e-04 | 7.1131e-03 | +9.07e-04 | 6.9e-05 | +2.8e-09 | 732.0 | optimal_inaccurate | 4.01 | False | False |
| 0.02 | 3 | base | 1.6542e-02 | 1.6542e-02 | -1.08e-10 | 1.6542e-02 | +2.00e-08 | 1.3e-11 | +7.6e-09 | 16.0 | optimal | 0.03 | True | True |
| 0.02 | 3 | rlt | 1.6542e-02 | 1.6542e-02 | -7.95e-11 | 1.6542e-02 | +2.00e-08 | 6.1e-12 | +7.6e-09 | 152.0 | optimal | 0.16 | True | True |
| 0.02 | 3 | bounds | 1.6542e-02 | 1.6542e-02 | +3.40e-10 | 1.6542e-02 | +2.04e-08 | 5.5e-11 | +7.7e-09 | 29.0 | optimal | 0.05 | True | True |
| 0.02 | 3 | bound_rlt | 1.6542e-02 | 1.6542e-02 | -2.83e-11 | 1.6542e-02 | +2.01e-08 | 3.6e-11 | +7.6e-09 | 352.0 | optimal | 1.09 | True | True |
| 0.02 | 3 | all | 1.6542e-02 | 1.6542e-02 | -3.96e-07 | 1.6529e-02 | +1.30e-05 | 8.3e-07 | +2.3e-09 | 732.0 | optimal_inaccurate | 3.92 | True | False |
| 0.05 | 0 | base | 1.5253e-01 | 1.5253e-01 | -7.27e-10 | 1.5253e-01 | +2.00e-08 | 4.2e-10 | +7.5e-09 | 16.0 | optimal | 0.03 | True | True |
| 0.05 | 0 | rlt | 1.5253e-01 | 1.5253e-01 | -7.44e-12 | 1.5253e-01 | +2.00e-08 | 1.9e-12 | +7.5e-09 | 152.0 | optimal | 0.13 | True | True |
| 0.05 | 0 | bounds | 1.5253e-01 | 1.5253e-01 | -1.56e-11 | 1.5253e-01 | +2.00e-08 | 1.8e-12 | +7.5e-09 | 29.0 | optimal | 0.04 | True | True |
| 0.05 | 0 | bound_rlt | 1.5253e-01 | 1.5253e-01 | +1.85e-09 | 1.5253e-01 | +2.28e-08 | 1.4e-11 | +8.6e-09 | 352.0 | optimal | 0.22 | True | True |
| 0.05 | 0 | all | 1.5253e-01 | 1.5253e-01 | +4.51e-10 | 1.5253e-01 | +2.12e-08 | 8.1e-12 | +8.0e-09 | 732.0 | optimal | 0.51 | True | True |
| 0.05 | 1 | base | 1.1757e-01 | 1.1757e-01 | +5.59e-09 | 1.1757e-01 | +2.57e-08 | 1.1e-10 | +9.7e-09 | 16.0 | optimal | 0.03 | True | True |
| 0.05 | 1 | rlt | 1.1757e-01 | 1.1757e-01 | +5.69e-10 | 1.1757e-01 | +2.08e-08 | 6.8e-11 | +7.9e-09 | 152.0 | optimal | 0.09 | True | True |
| 0.05 | 1 | bounds | 1.1757e-01 | 1.1757e-01 | -9.27e-09 | 1.1757e-01 | +2.00e-08 | 1.3e-10 | +7.6e-09 | 29.0 | optimal | 0.03 | True | True |
| 0.05 | 1 | bound_rlt | 1.1757e-01 | 1.1757e-01 | -6.48e-10 | 1.1757e-01 | +2.00e-08 | 2.0e-10 | +7.6e-09 | 352.0 | optimal | 0.19 | True | True |
| 0.05 | 1 | all | 1.1757e-01 | 1.1757e-01 | -3.87e-09 | 1.1757e-01 | +2.00e-08 | 1.3e-10 | +7.6e-09 | 732.0 | optimal | 0.37 | True | True |
| 0.05 | 2 | base | 6.8455e-02 | 6.8455e-02 | -1.21e-11 | 6.8455e-02 | +2.00e-08 | 1.7e-12 | +7.5e-09 | 16.0 | optimal | 0.04 | True | True |
| 0.05 | 2 | rlt | 6.8455e-02 | 6.8455e-02 | +5.64e-09 | 6.8455e-02 | +2.66e-08 | 1.6e-10 | +1.0e-08 | 152.0 | optimal | 0.14 | True | True |
| 0.05 | 2 | bounds | 6.8455e-02 | 6.8455e-02 | -2.67e-11 | 6.8455e-02 | +2.00e-08 | 7.1e-12 | +7.5e-09 | 29.0 | optimal | 0.04 | True | True |
| 0.05 | 2 | bound_rlt | 6.8455e-02 | 6.8455e-02 | +2.69e-10 | 6.8455e-02 | +2.03e-08 | 1.6e-11 | +7.6e-09 | 352.0 | optimal | 0.23 | True | True |
| 0.05 | 2 | all | 6.8455e-02 | 6.9591e-02 | -1.14e-03 | 6.7134e-02 | +1.32e-03 | 2.0e-03 | +7.6e-09 | 732.0 | optimal_inaccurate | 3.90 | True | False |
| 0.05 | 3 | base | 1.2369e-01 | 1.2369e-01 | -2.83e-09 | 1.2369e-01 | +2.00e-08 | 3.9e-10 | +7.6e-09 | 16.0 | optimal | 0.03 | True | True |
| 0.05 | 3 | rlt | 1.2369e-01 | 1.2369e-01 | -1.88e-11 | 1.2369e-01 | +2.00e-08 | 1.0e-13 | +7.6e-09 | 152.0 | optimal | 0.09 | True | True |
| 0.05 | 3 | bounds | 1.2369e-01 | 1.2369e-01 | -2.00e-11 | 1.2369e-01 | +2.00e-08 | 9.0e-14 | +7.6e-09 | 29.0 | optimal | 0.04 | True | True |
| 0.05 | 3 | bound_rlt | 1.2369e-01 | 1.2369e-01 | -9.75e-12 | 1.2369e-01 | +2.00e-08 | 2.1e-14 | +7.6e-09 | 352.0 | optimal | 0.17 | True | True |
| 0.05 | 3 | all | 1.2369e-01 | 1.2369e-01 | -5.52e-11 | 1.2369e-01 | +2.00e-08 | 2.8e-11 | +7.6e-09 | 732.0 | optimal | 0.67 | True | True |
| 0.10 | 0 | base | 9.0579e-01 | 9.0259e-01 | +3.19e-03 | 9.0259e-01 | +3.19e-03 | 8.3e-03 | +8.9e-09 | 16.0 | optimal_inaccurate | 2.01 | False | False |
| 0.10 | 0 | rlt | 9.0579e-01 | 9.0270e-01 | +3.09e-03 | 9.0270e-01 | +3.09e-03 | 8.3e-03 | +8.9e-09 | 152.0 | optimal_inaccurate | 2.62 | False | False |
| 0.10 | 0 | bounds | 9.0579e-01 | 9.0260e-01 | +3.19e-03 | 9.0260e-01 | +3.19e-03 | 8.3e-03 | +9.2e-09 | 29.0 | optimal_inaccurate | 1.64 | False | False |
| 0.10 | 0 | bound_rlt | 9.0579e-01 | 9.0579e-01 | -6.83e-11 | 9.0579e-01 | +2.00e-08 | 1.8e-12 | +7.6e-09 | 352.0 | optimal | 0.20 | True | True |
| 0.10 | 0 | all | 9.0579e-01 | 9.0579e-01 | +2.08e-11 | 9.0579e-01 | +2.01e-08 | 2.3e-12 | +7.6e-09 | 732.0 | optimal | 0.56 | True | True |
| 0.10 | 1 | base | 6.8500e-01 | 6.8476e-01 | +2.38e-04 | 6.8475e-01 | +2.41e-04 | 8.6e-03 | +9.8e-09 | 16.0 | optimal_inaccurate | 1.68 | False | False |
| 0.10 | 1 | rlt | 6.8500e-01 | 6.8500e-01 | +7.61e-12 | 6.8500e-01 | +2.00e-08 | 2.3e-11 | +7.6e-09 | 152.0 | optimal | 0.15 | True | True |
| 0.10 | 1 | bounds | 6.8500e-01 | 6.8476e-01 | +2.38e-04 | 6.8476e-01 | +2.38e-04 | 8.6e-03 | +9.8e-09 | 29.0 | optimal_inaccurate | 1.78 | False | False |
| 0.10 | 1 | bound_rlt | 6.8500e-01 | 6.8500e-01 | -8.18e-10 | 6.8500e-01 | +2.00e-08 | 2.1e-11 | +7.6e-09 | 352.0 | optimal | 0.16 | True | True |
| 0.10 | 1 | all | 6.8500e-01 | 6.8500e-01 | -9.40e-12 | 6.8500e-01 | +2.00e-08 | 3.9e-12 | +7.6e-09 | 732.0 | optimal | 0.36 | True | True |
| 0.10 | 2 | base | 4.3643e-01 | 4.3582e-01 | +6.09e-04 | 4.3582e-01 | +6.09e-04 | 8.3e-03 | +9.5e-09 | 16.0 | optimal_inaccurate | 1.52 | False | False |
| 0.10 | 2 | rlt | 4.3643e-01 | 4.3585e-01 | +5.79e-04 | 4.3585e-01 | +5.79e-04 | 8.3e-03 | +9.8e-09 | 152.0 | optimal | 1.55 | False | False |
| 0.10 | 2 | bounds | 4.3643e-01 | 4.3582e-01 | +6.10e-04 | 4.3582e-01 | +6.10e-04 | 8.3e-03 | +1.0e-08 | 29.0 | optimal_inaccurate | 1.69 | False | False |
| 0.10 | 2 | bound_rlt | 4.3643e-01 | 4.3643e-01 | -8.53e-10 | 4.3643e-01 | +2.00e-08 | 2.3e-11 | +7.6e-09 | 352.0 | optimal | 0.16 | True | True |
| 0.10 | 2 | all | 4.3643e-01 | 4.3643e-01 | +1.48e-10 | 4.3643e-01 | +2.03e-08 | 9.3e-12 | +7.7e-09 | 732.0 | optimal | 0.36 | True | True |
| 0.10 | 3 | base | 5.7689e-01 | 5.7689e-01 | -2.30e-10 | 5.7689e-01 | +2.01e-08 | 3.4e-11 | +7.6e-09 | 16.0 | optimal | 0.03 | True | True |
| 0.10 | 3 | rlt | 5.7689e-01 | 5.7689e-01 | +2.03e-09 | 5.7689e-01 | +2.21e-08 | 2.6e-11 | +8.3e-09 | 152.0 | optimal | 0.10 | True | True |
| 0.10 | 3 | bounds | 5.7689e-01 | 5.7689e-01 | +5.75e-11 | 5.7689e-01 | +2.01e-08 | 3.3e-13 | +7.6e-09 | 29.0 | optimal | 0.04 | True | True |
| 0.10 | 3 | bound_rlt | 5.7689e-01 | 5.7689e-01 | +2.63e-10 | 5.7689e-01 | +2.03e-08 | 4.9e-12 | +7.7e-09 | 352.0 | optimal | 0.17 | True | True |
| 0.10 | 3 | all | 5.7689e-01 | 5.7689e-01 | +5.48e-11 | 5.7689e-01 | +2.01e-08 | 1.8e-12 | +7.6e-09 | 732.0 | optimal | 0.43 | True | True |
| 0.20 | 0 | base | 3.2392e+00 | 3.2297e+00 | +9.45e-03 | 3.2297e+00 | +9.45e-03 | 9.5e-03 | +9.6e-09 | 16.0 | optimal_inaccurate | 1.94 | False | False |
| 0.20 | 0 | rlt | 3.2392e+00 | 3.2306e+00 | +8.54e-03 | 3.2306e+00 | +8.54e-03 | 3.8e-03 | +9.4e-09 | 152.0 | optimal_inaccurate | 2.62 | False | False |
| 0.20 | 0 | bounds | 3.2392e+00 | 3.2298e+00 | +9.39e-03 | 3.2298e+00 | +9.39e-03 | 9.8e-03 | +1.0e-08 | 29.0 | optimal_inaccurate | 1.79 | False | False |
| 0.20 | 0 | bound_rlt | 3.2392e+00 | 3.2392e+00 | +2.71e-10 | 3.2392e+00 | +2.04e-08 | 8.2e-12 | +7.7e-09 | 352.0 | optimal | 0.16 | True | True |
| 0.20 | 0 | all | 3.2392e+00 | 3.2392e+00 | +2.36e-10 | 3.2392e+00 | +2.05e-08 | 4.1e-11 | +7.7e-09 | 732.0 | optimal | 0.45 | True | True |
| 0.20 | 1 | base | 3.1128e+00 | 3.1013e+00 | +1.15e-02 | 3.1013e+00 | +1.15e-02 | 1.4e-02 | +9.3e-09 | 16.0 | optimal_inaccurate | 1.66 | False | False |
| 0.20 | 1 | rlt | 3.1128e+00 | 3.1030e+00 | +9.83e-03 | 3.1030e+00 | +9.82e-03 | 1.5e-02 | +8.1e-09 | 152.0 | optimal_inaccurate | 2.62 | False | False |
| 0.20 | 1 | bounds | 3.1128e+00 | 3.1013e+00 | +1.15e-02 | 3.1013e+00 | +1.15e-02 | 1.4e-02 | +9.4e-09 | 29.0 | optimal_inaccurate | 1.71 | False | False |
| 0.20 | 1 | bound_rlt | 3.1128e+00 | 3.1128e+00 | -4.23e-11 | 3.1128e+00 | +2.00e-08 | 2.4e-11 | +7.6e-09 | 352.0 | optimal | 0.16 | True | True |
| 0.20 | 1 | all | 3.1128e+00 | 3.1128e+00 | +2.07e-11 | 3.1128e+00 | +2.00e-08 | 3.3e-13 | +7.6e-09 | 732.0 | optimal | 0.36 | True | True |
| 0.20 | 2 | base | 3.0309e+00 | 3.0231e+00 | +7.76e-03 | 3.0231e+00 | +7.76e-03 | 9.1e-03 | +9.2e-09 | 16.0 | optimal_inaccurate | 1.66 | False | False |
| 0.20 | 2 | rlt | 3.0309e+00 | 3.0237e+00 | +7.15e-03 | 3.0237e+00 | +7.15e-03 | 8.9e-03 | +9.5e-09 | 152.0 | optimal_inaccurate | 2.71 | False | False |
| 0.20 | 2 | bounds | 3.0309e+00 | 3.0231e+00 | +7.75e-03 | 3.0231e+00 | +7.75e-03 | 9.1e-03 | +1.0e-08 | 29.0 | optimal_inaccurate | 1.78 | False | False |
| 0.20 | 2 | bound_rlt | 3.0309e+00 | 3.0309e+00 | -6.21e-12 | 3.0309e+00 | +2.01e-08 | 9.4e-13 | +7.6e-09 | 352.0 | optimal | 0.17 | True | True |
| 0.20 | 2 | all | 3.0309e+00 | 3.0309e+00 | -4.83e-11 | 3.0309e+00 | +2.00e-08 | 4.1e-13 | +7.5e-09 | 732.0 | optimal | 0.35 | True | True |
| 0.20 | 3 | base | 2.2501e+00 | 2.2435e+00 | +6.59e-03 | 2.2435e+00 | +6.59e-03 | 8.9e-03 | +8.9e-09 | 16.0 | optimal_inaccurate | 1.70 | False | False |
| 0.20 | 3 | rlt | 2.2501e+00 | 2.2446e+00 | +5.51e-03 | 2.2446e+00 | +5.49e-03 | 8.5e-03 | +1.0e-08 | 152.0 | optimal_inaccurate | 2.52 | False | False |
| 0.20 | 3 | bounds | 2.2501e+00 | 2.2440e+00 | +6.10e-03 | 2.2440e+00 | +6.10e-03 | 8.8e-03 | +9.1e-09 | 29.0 | optimal_inaccurate | 1.89 | False | False |
| 0.20 | 3 | bound_rlt | 2.2501e+00 | 2.2501e+00 | -4.34e-10 | 2.2501e+00 | +2.00e-08 | 1.2e-11 | +7.6e-09 | 352.0 | optimal | 0.17 | True | True |
| 0.20 | 3 | all | 2.2501e+00 | 2.2501e+00 | -2.71e-11 | 2.2501e+00 | +2.00e-08 | 4.2e-12 | +7.6e-09 | 732.0 | optimal | 0.45 | True | True |

## per-instance (CLARABEL)

| noise | seed | variant | J | value | gap | rig | rig gap | ratio | λmin(S_rep) | n_ineq | status | t(s) | cert naive | cert rig |
|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|
| 0.02 | 0 | base | 1.7220e-02 | 1.7220e-02 | +6.25e-09 | 1.7220e-02 | +2.90e-08 | 9.5e-09 | +7.5e-09 | 16.0 | optimal | 0.04 | True | True |
| 0.02 | 0 | rlt | 1.7220e-02 | 1.7220e-02 | -2.17e-10 | 1.7220e-02 | +2.12e-08 | 2.7e-09 | +7.5e-09 | 152.0 | optimal | 0.10 | True | True |
| 0.02 | 0 | bounds | 1.7220e-02 | 1.7220e-02 | +1.06e-08 | 1.7220e-02 | +3.57e-08 | 7.5e-09 | +7.5e-09 | 29.0 | optimal | 0.03 | True | True |
| 0.02 | 0 | bound_rlt | 1.7220e-02 | 1.7220e-02 | +9.08e-09 | 1.7220e-02 | +4.20e-08 | 1.9e-09 | +7.5e-09 | 352.0 | optimal_inaccurate | 0.16 | True | True |
| 0.02 | 0 | all | 1.7220e-02 | 1.7220e-02 | -3.56e-09 | 1.7220e-02 | +5.36e-08 | 7.3e-09 | +7.5e-09 | 732.0 | optimal_inaccurate | 0.39 | True | True |
| 0.02 | 1 | base | 1.3687e-02 | 1.3687e-02 | +4.68e-10 | 1.3687e-02 | +2.09e-08 | 1.6e-09 | +7.6e-09 | 16.0 | optimal | 0.03 | True | True |
| 0.02 | 1 | rlt | 1.3687e-02 | 1.3687e-02 | -1.09e-09 | 1.3687e-02 | +2.08e-08 | 1.1e-08 | +7.6e-09 | 152.0 | optimal | 0.10 | True | True |
| 0.02 | 1 | bounds | 1.3687e-02 | 1.3687e-02 | +1.80e-09 | 1.3687e-02 | +2.23e-08 | 1.8e-09 | +7.6e-09 | 29.0 | optimal | 0.03 | True | True |
| 0.02 | 1 | bound_rlt | 1.3687e-02 | 1.3687e-02 | +3.65e-10 | 1.3687e-02 | +2.33e-08 | 2.3e-09 | +7.6e-09 | 352.0 | optimal | 0.19 | True | True |
| 0.02 | 1 | all | 1.3687e-02 | 1.3687e-02 | -3.55e-09 | 1.3687e-02 | +2.54e-08 | 3.6e-09 | +7.6e-09 | 732.0 | optimal_inaccurate | 0.35 | True | True |
| 0.02 | 2 | base | 8.0202e-03 | 8.0202e-03 | +8.21e-09 | 8.0202e-03 | +3.00e-08 | 1.5e-08 | +7.5e-09 | 16.0 | optimal | 0.03 | True | True |
| 0.02 | 2 | rlt | 8.0202e-03 | 8.0202e-03 | +8.46e-11 | 8.0202e-03 | +2.04e-08 | 7.4e-10 | +7.5e-09 | 152.0 | optimal_inaccurate | 0.10 | True | True |
| 0.02 | 2 | bounds | 8.0202e-03 | 8.0202e-03 | +3.01e-09 | 8.0202e-03 | +2.41e-08 | 4.1e-09 | +7.5e-09 | 29.0 | optimal | 0.03 | True | True |
| 0.02 | 2 | bound_rlt | 8.0202e-03 | 8.0202e-03 | +4.32e-09 | 8.0202e-03 | +2.95e-08 | 1.4e-09 | +7.5e-09 | 352.0 | optimal | 0.18 | True | True |
| 0.02 | 2 | all | 8.0202e-03 | 8.0202e-03 | +6.38e-10 | 8.0202e-03 | +3.72e-08 | 1.5e-09 | +7.5e-09 | 732.0 | optimal_inaccurate | 0.37 | True | True |
| 0.02 | 3 | base | 1.6542e-02 | 1.6542e-02 | +5.01e-09 | 1.6542e-02 | +2.68e-08 | 7.6e-09 | +7.6e-09 | 16.0 | optimal | 0.04 | True | True |
| 0.02 | 3 | rlt | 1.6542e-02 | 1.6542e-02 | -5.40e-10 | 1.6542e-02 | +2.58e-08 | 6.8e-09 | +7.6e-09 | 152.0 | optimal_inaccurate | 0.18 | True | True |
| 0.02 | 3 | bounds | 1.6542e-02 | 1.6542e-02 | +8.24e-09 | 1.6542e-02 | +3.11e-08 | 6.2e-09 | +7.6e-09 | 29.0 | optimal | 0.03 | True | True |
| 0.02 | 3 | bound_rlt | 1.6542e-02 | 1.6542e-02 | +2.48e-08 | 1.6542e-02 | +6.85e-08 | 6.9e-09 | +7.6e-09 | 352.0 | optimal_inaccurate | 0.16 | True | True |
| 0.02 | 3 | all | 1.6542e-02 | 1.6542e-02 | -1.07e-09 | 1.6542e-02 | +7.49e-08 | 5.9e-09 | +7.6e-09 | 732.0 | optimal_inaccurate | 0.36 | True | True |
| 0.05 | 0 | base | 1.5253e-01 | 1.5253e-01 | +7.18e-10 | 1.5253e-01 | +2.07e-08 | 1.3e-10 | +7.5e-09 | 16.0 | optimal | 0.03 | True | True |
| 0.05 | 0 | rlt | 1.5253e-01 | 1.5253e-01 | -1.14e-09 | 1.5253e-01 | +2.95e-08 | 4.1e-09 | +7.5e-09 | 152.0 | optimal_inaccurate | 0.10 | True | True |
| 0.05 | 0 | bounds | 1.5253e-01 | 1.5253e-01 | +2.64e-09 | 1.5253e-01 | +2.37e-08 | 7.0e-10 | +7.5e-09 | 29.0 | optimal | 0.04 | True | True |
| 0.05 | 0 | bound_rlt | 1.5253e-01 | 1.5252e-01 | +3.48e-06 | 1.5252e-01 | +7.38e-06 | 2.4e-07 | +7.5e-09 | 352.0 | optimal_inaccurate | 0.16 | False | False |
| 0.05 | 0 | all | 1.5253e-01 | 1.5253e-01 | -4.39e-09 | 1.5253e-01 | +9.01e-08 | 1.7e-09 | +7.5e-09 | 732.0 | optimal_inaccurate | 0.36 | True | True |
| 0.05 | 1 | base | 1.1757e-01 | 1.1757e-01 | +4.52e-10 | 1.1757e-01 | +2.05e-08 | 5.2e-10 | +7.6e-09 | 16.0 | optimal | 0.03 | True | True |
| 0.05 | 1 | rlt | 1.1757e-01 | 1.1757e-01 | -2.79e-09 | 1.1757e-01 | +2.18e-08 | 6.1e-09 | +7.6e-09 | 152.0 | optimal_inaccurate | 0.12 | True | True |
| 0.05 | 1 | bounds | 1.1757e-01 | 1.1757e-01 | +8.72e-09 | 1.1757e-01 | +3.44e-08 | 1.3e-08 | +7.6e-09 | 29.0 | optimal | 0.03 | True | True |
| 0.05 | 1 | bound_rlt | 1.1757e-01 | 1.1757e-01 | +5.64e-10 | 1.1757e-01 | +2.38e-08 | 2.5e-09 | +7.6e-09 | 352.0 | optimal | 0.16 | True | True |
| 0.05 | 1 | all | 1.1757e-01 | 1.1757e-01 | -1.30e-08 | 1.1757e-01 | +4.02e-08 | 1.5e-08 | +7.6e-09 | 732.0 | optimal_inaccurate | 0.35 | True | True |
| 0.05 | 2 | base | 6.8455e-02 | 6.8455e-02 | +1.57e-09 | 6.8455e-02 | +2.20e-08 | 1.3e-09 | +7.5e-09 | 16.0 | optimal | 0.03 | True | True |
| 0.05 | 2 | rlt | 6.8455e-02 | 6.8455e-02 | -8.50e-10 | 6.8455e-02 | +2.47e-08 | 3.6e-09 | +7.5e-09 | 152.0 | optimal_inaccurate | 0.12 | True | True |
| 0.05 | 2 | bounds | 6.8455e-02 | 6.8455e-02 | +1.71e-09 | 6.8455e-02 | +2.24e-08 | 8.9e-10 | +7.5e-09 | 29.0 | optimal | 0.04 | True | True |
| 0.05 | 2 | bound_rlt | 6.8455e-02 | 6.8455e-02 | +5.28e-08 | 6.8455e-02 | +1.64e-07 | 1.1e-08 | +7.5e-09 | 352.0 | optimal_inaccurate | 0.16 | True | True |
| 0.05 | 2 | all | 6.8455e-02 | 6.8455e-02 | -1.29e-08 | 6.8455e-02 | +1.46e-07 | 1.1e-08 | +7.5e-09 | 732.0 | optimal_inaccurate | 0.35 | True | True |
| 0.05 | 3 | base | 1.2369e-01 | 1.2369e-01 | +4.73e-09 | 1.2369e-01 | +2.58e-08 | 4.1e-09 | +7.6e-09 | 16.0 | optimal | 0.03 | True | True |
| 0.05 | 3 | rlt | 1.2369e-01 | 1.2369e-01 | -6.32e-10 | 1.2369e-01 | +2.04e-08 | 1.2e-09 | +7.6e-09 | 152.0 | optimal | 0.11 | True | True |
| 0.05 | 3 | bounds | 1.2369e-01 | 1.2369e-01 | +8.91e-09 | 1.2369e-01 | +3.43e-08 | 7.1e-09 | +7.6e-09 | 29.0 | optimal | 0.03 | True | True |
| 0.05 | 3 | bound_rlt | 1.2369e-01 | 1.2369e-01 | +6.88e-06 | 1.2361e-01 | +8.51e-05 | 4.0e-05 | +7.6e-09 | 352.0 | optimal_inaccurate | 0.16 | False | False |
| 0.05 | 3 | all | 1.2369e-01 | 1.2369e-01 | -1.62e-08 | 1.2369e-01 | +4.64e-08 | 8.0e-09 | +7.6e-09 | 732.0 | optimal_inaccurate | 0.36 | True | True |
| 0.10 | 0 | base | 9.0579e-01 | 9.0259e-01 | +3.19e-03 | 9.0259e-01 | +3.19e-03 | 8.3e-03 | +7.7e-09 | 16.0 | optimal | 0.03 | False | False |
| 0.10 | 0 | rlt | 9.0579e-01 | 9.0270e-01 | +3.09e-03 | 9.0268e-01 | +3.10e-03 | 8.3e-03 | +7.5e-09 | 152.0 | optimal_inaccurate | 0.12 | False | False |
| 0.10 | 0 | bounds | 9.0579e-01 | 9.0260e-01 | +3.19e-03 | 9.0260e-01 | +3.19e-03 | 8.3e-03 | +7.5e-09 | 29.0 | optimal_inaccurate | 0.04 | False | False |
| 0.10 | 0 | bound_rlt | 9.0579e-01 | 9.0579e-01 | +2.13e-08 | 9.0579e-01 | +1.17e-07 | 7.6e-08 | +7.6e-09 | 352.0 | optimal_inaccurate | 0.16 | True | True |
| 0.10 | 0 | all | 9.0579e-01 | 9.0579e-01 | -6.17e-09 | 9.0579e-01 | +3.49e-08 | 5.0e-09 | +7.6e-09 | 732.0 | optimal_inaccurate | 0.45 | True | True |
| 0.10 | 1 | base | 6.8500e-01 | 6.8476e-01 | +2.37e-04 | 6.8476e-01 | +2.37e-04 | 8.6e-03 | +7.5e-09 | 16.0 | optimal_inaccurate | 0.03 | False | False |
| 0.10 | 1 | rlt | 6.8500e-01 | 6.8500e-01 | -5.19e-08 | 6.8500e-01 | +6.40e-08 | 7.4e-06 | +7.6e-09 | 152.0 | optimal_inaccurate | 0.12 | True | True |
| 0.10 | 1 | bounds | 6.8500e-01 | — | — | — | — | — | — | 29.0 | error:Solver 'CLARABEL' failed. Try another solver, or solve with verbose=True for more information. | 0.04 | False | False |
| 0.10 | 1 | bound_rlt | 6.8500e-01 | 6.8500e-01 | +7.55e-10 | 6.8500e-01 | +2.46e-08 | 1.3e-09 | +7.6e-09 | 352.0 | optimal | 0.15 | True | True |
| 0.10 | 1 | all | 6.8500e-01 | 6.8500e-01 | -7.47e-09 | 6.8500e-01 | +3.59e-08 | 4.7e-09 | +7.6e-09 | 732.0 | optimal_inaccurate | 0.34 | True | True |
| 0.10 | 2 | base | 4.3643e-01 | 4.3582e-01 | +6.08e-04 | 4.3582e-01 | +6.08e-04 | 8.3e-03 | +7.5e-09 | 16.0 | optimal_inaccurate | 0.03 | False | False |
| 0.10 | 2 | rlt | 4.3643e-01 | 4.3585e-01 | +5.79e-04 | 4.3585e-01 | +5.79e-04 | 8.3e-03 | +7.5e-09 | 152.0 | optimal_inaccurate | 0.13 | False | False |
| 0.10 | 2 | bounds | 4.3643e-01 | 4.3582e-01 | +6.08e-04 | 4.3582e-01 | +6.08e-04 | 8.3e-03 | +7.5e-09 | 29.0 | optimal_inaccurate | 0.04 | False | False |
| 0.10 | 2 | bound_rlt | 4.3643e-01 | 4.3643e-01 | +2.62e-10 | 4.3643e-01 | +4.14e-08 | 2.0e-08 | +7.6e-09 | 352.0 | optimal_inaccurate | 0.16 | True | True |
| 0.10 | 2 | all | 4.3643e-01 | 4.3643e-01 | -6.77e-09 | 4.3643e-01 | +3.39e-08 | 8.2e-09 | +7.6e-09 | 732.0 | optimal_inaccurate | 0.47 | True | True |
| 0.10 | 3 | base | 5.7689e-01 | 5.7689e-01 | +8.65e-09 | 5.7689e-01 | +3.03e-08 | 1.9e-08 | +7.6e-09 | 16.0 | optimal | 0.03 | True | True |
| 0.10 | 3 | rlt | 5.7689e-01 | 5.7689e-01 | -1.68e-08 | 5.7689e-01 | +3.14e-08 | 6.5e-08 | +7.6e-09 | 152.0 | optimal_inaccurate | 0.10 | True | True |
| 0.10 | 3 | bounds | 5.7689e-01 | 5.7689e-01 | +3.43e-09 | 5.7689e-01 | +2.49e-08 | 3.9e-09 | +7.6e-09 | 29.0 | optimal | 0.04 | True | True |
| 0.10 | 3 | bound_rlt | 5.7689e-01 | 5.7689e-01 | +6.16e-09 | 5.7689e-01 | +5.56e-08 | 6.6e-09 | +7.6e-09 | 352.0 | optimal_inaccurate | 0.19 | True | True |
| 0.10 | 3 | all | 5.7689e-01 | 5.7689e-01 | -7.90e-09 | 5.7689e-01 | +3.40e-08 | 2.2e-09 | +7.6e-09 | 732.0 | optimal_inaccurate | 0.39 | True | True |
| 0.20 | 0 | base | 3.2392e+00 | 3.2297e+00 | +9.44e-03 | 3.2297e+00 | +9.44e-03 | 9.5e-03 | +8.8e-09 | 16.0 | optimal_inaccurate | 0.03 | False | False |
| 0.20 | 0 | rlt | 3.2392e+00 | — | — | — | — | — | — | 152.0 | error:Solver 'CLARABEL' failed. Try another solver, or solve with verbose=True for more information. | 0.09 | False | False |
| 0.20 | 0 | bounds | 3.2392e+00 | 3.2298e+00 | +9.39e-03 | 3.2298e+00 | +9.39e-03 | 9.8e-03 | +7.5e-09 | 29.0 | optimal_inaccurate | 0.04 | False | False |
| 0.20 | 0 | bound_rlt | 3.2392e+00 | 3.2392e+00 | +4.54e-08 | 3.2392e+00 | +1.27e-07 | 1.0e-08 | +7.5e-09 | 352.0 | optimal_inaccurate | 0.18 | True | True |
| 0.20 | 0 | all | 3.2392e+00 | 3.2392e+00 | -2.94e-08 | 3.2392e+00 | +1.22e-07 | 5.7e-09 | +7.5e-09 | 732.0 | optimal_inaccurate | 0.36 | True | True |
| 0.20 | 1 | base | 3.1128e+00 | — | — | — | — | — | — | 16.0 | error:Solver 'CLARABEL' failed. Try another solver, or solve with verbose=True for more information. | 0.03 | False | False |
| 0.20 | 1 | rlt | 3.1128e+00 | 3.1034e+00 | +9.43e-03 | 3.1034e+00 | +9.43e-03 | 1.5e-02 | +7.5e-09 | 152.0 | optimal_inaccurate | 0.10 | False | False |
| 0.20 | 1 | bounds | 3.1128e+00 | 3.1013e+00 | +1.15e-02 | 3.1013e+00 | +1.15e-02 | 1.4e-02 | +7.5e-09 | 29.0 | optimal_inaccurate | 0.04 | False | False |
| 0.20 | 1 | bound_rlt | 3.1128e+00 | 3.1128e+00 | +3.02e-09 | 3.1128e+00 | +2.84e-08 | 1.8e-09 | +7.6e-09 | 352.0 | optimal | 0.18 | True | True |
| 0.20 | 1 | all | 3.1128e+00 | 3.1128e+00 | -3.30e-08 | 3.1128e+00 | +1.38e-07 | 1.3e-08 | +7.6e-09 | 732.0 | optimal_inaccurate | 0.37 | True | True |
| 0.20 | 2 | base | 3.0309e+00 | 3.0231e+00 | +7.76e-03 | 3.0231e+00 | +7.76e-03 | 9.1e-03 | +7.5e-09 | 16.0 | optimal_inaccurate | 0.04 | False | False |
| 0.20 | 2 | rlt | 3.0309e+00 | 3.0239e+00 | +6.94e-03 | 3.0239e+00 | +6.94e-03 | 8.8e-03 | +7.5e-09 | 152.0 | optimal_inaccurate | 0.11 | False | False |
| 0.20 | 2 | bounds | 3.0309e+00 | 3.0231e+00 | +7.75e-03 | 3.0231e+00 | +7.75e-03 | 9.1e-03 | +7.5e-09 | 29.0 | optimal_inaccurate | 0.05 | False | False |
| 0.20 | 2 | bound_rlt | 3.0309e+00 | — | — | — | — | — | — | 352.0 | error:Solver 'CLARABEL' failed. Try another solver, or solve with verbose=True for more information. | 0.18 | False | False |
| 0.20 | 2 | all | 3.0309e+00 | 3.0309e+00 | -3.21e-08 | 3.0309e+00 | +1.28e-07 | 5.6e-09 | +7.5e-09 | 732.0 | optimal_inaccurate | 0.38 | True | True |
| 0.20 | 3 | base | 2.2501e+00 | 2.2435e+00 | +6.59e-03 | 2.2435e+00 | +6.59e-03 | 8.9e-03 | +7.4e-09 | 16.0 | optimal_inaccurate | 0.03 | False | False |
| 0.20 | 3 | rlt | 2.2501e+00 | 2.2448e+00 | +5.30e-03 | 2.2448e+00 | +5.30e-03 | 8.4e-03 | +7.6e-09 | 152.0 | optimal_inaccurate | 0.10 | False | False |
| 0.20 | 3 | bounds | 2.2501e+00 | 2.2440e+00 | +6.09e-03 | 2.2440e+00 | +6.09e-03 | 8.8e-03 | +7.8e-09 | 29.0 | optimal_inaccurate | 0.04 | False | False |
| 0.20 | 3 | bound_rlt | 2.2501e+00 | 2.2501e+00 | +4.72e-08 | 2.2501e+00 | +1.55e-07 | 2.0e-08 | +7.6e-09 | 352.0 | optimal_inaccurate | 0.26 | True | True |
| 0.20 | 3 | all | 2.2501e+00 | 2.2501e+00 | -7.63e-09 | 2.2501e+00 | +3.49e-08 | 9.1e-09 | +7.6e-09 | 732.0 | optimal_inaccurate | 0.42 | True | True |

## cut-audit on sampled feasible points (max violation; <=0 = satisfied)

| noise | seed | points | g | rlt | bounds | bound_rlt | box_rlt |
|-|-|-|-|-|-|-|-|
| 0.02 | 0 | 29 | +8.3e-11 | +6.8e-11 | -3.7e-10 | +8.6e-11 | -1.1e-15 |
| 0.02 | 1 | 29 | +8.4e-11 | +6.7e-11 | -2.7e-23 | +1.1e-10 | -6.1e-42 |
| 0.02 | 2 | 29 | +9.3e-12 | +7.6e-12 | -4.7e-09 | +9.1e-12 | -1.9e-13 |
| 0.02 | 3 | 29 | +8.1e-11 | +6.5e-11 | -1.6e-10 | +8.1e-11 | -2.1e-16 |
| 0.05 | 0 | 29 | +1.0e-10 | +8.2e-11 | -4.4e-09 | +1.1e-10 | -1.6e-13 |
| 0.05 | 1 | 29 | +5.6e-11 | +4.5e-11 | -9.3e-24 | +7.2e-11 | -7.3e-43 |
| 0.05 | 2 | 29 | +5.7e-11 | +4.7e-11 | -2.2e-10 | +5.6e-11 | -4.0e-16 |
| 0.05 | 3 | 29 | +9.1e-11 | +7.2e-11 | -2.6e-23 | +9.1e-11 | -5.5e-42 |
| 0.10 | 0 | 29 | +3.4e-11 | +2.7e-11 | -1.2e-24 | +3.7e-11 | -1.1e-44 |
| 0.10 | 1 | 29 | +8.2e-11 | +6.5e-11 | -2.8e-23 | +1.0e-10 | -6.4e-42 |
| 0.10 | 2 | 29 | +7.7e-11 | +6.1e-11 | -2.2e-24 | +7.5e-11 | -4.1e-44 |
| 0.10 | 3 | 29 | +8.8e-11 | +7.0e-11 | -3.6e-24 | +8.8e-11 | -1.1e-43 |
| 0.20 | 0 | 29 | +5.6e-11 | +4.5e-11 | -6.8e-26 | +6.7e-11 | -3.8e-47 |
| 0.20 | 1 | 29 | +3.5e-11 | +2.8e-11 | -2.4e-23 | +4.5e-11 | -4.9e-42 |
| 0.20 | 2 | 29 | +8.8e-11 | +7.3e-11 | -3.8e-24 | +8.9e-11 | -1.2e-43 |
| 0.20 | 3 | 29 | +9.2e-11 | +7.4e-11 | -7.0e-24 | +9.1e-11 | -4.1e-43 |
