# Exact vs multi-start inner solve in the P2 ALM

Generated: 2026-11-27T21:41:44 (host `ls`, numpy 1.26.3, threads pinned: False).

`inner="multistart"`: `qpep_core.solve_inner`, 6 starts, hot start `q0=q` + perturbations. `inner="exact"`: `quartic_roots.solve_inner_exact`, 162-path total-degree homotopy over the 40 projective critical points of the reduced problem (generic complex case), with multi-start fallback when a path is lost or the KKT Jacobian is singular.

## S20 — p=16, 4 seeds, noise=0.01, bias_z=0.08

| p | seed | J_ms | J_ex | ΔJ (ex−ms) | ‖q_ex−q_ms‖ | viol_ms | viol_ex | iters ms/ex | fb ms/ex | enum calls (fb) | t_ms (s) | t_ex (s) | ratio total | t_inner_ms (ms/call) | t_inner_ex (ms/call) | feasible ms/ex |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 16 | 0 | 0.004921786 | 0.004921786 | -1.69e-13 | 1.2e-10 | 5.5e-11 | 4.3e-11 | 16/16 | 0/0 | 17 (0) | 0.374 | 9.896 | 26.5x | 21.6 | 581.8 | True/True |
| 16 | 1 | 0.003942046 | 0.003942046 | -3.44e-14 | 5.9e-10 | 7.2e-11 | 5.8e-11 | 10/10 | 0/0 | 9 (4) | 0.165 | 6.069 | 36.9x | 17.9 | 673.9 | True/True |
| 16 | 2 | 0.002449209 | 0.002449209 | -1.42e-14 | 1.6e-10 | 8.7e-12 | 8.2e-12 | 9/9 | 0/0 | 12 (4) | 0.148 | 9.749 | 65.8x | 12.2 | 812.1 | True/True |
| 16 | 3 | 0.004911537 | 0.004911537 | -2.04e-13 | 6.5e-10 | 5.3e-11 | 4.9e-11 | 23/23 | 0/0 | 25 (5) | 0.383 | 15.447 | 40.3x | 15.1 | 617.4 | True/True |

Total-time ratio exact/multistart: min 26.5x, max 65.8x, median 38.6x. A single enumeration costs 653 ms on average here (162 paths).


*Sensitivity run (`--no-pin`): S20 only; see `exact_alm.md` for the correctness / completeness verdict.*

