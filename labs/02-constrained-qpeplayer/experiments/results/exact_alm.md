# Exact vs multi-start inner solve in the P2 ALM

Generated: 2026-11-27T21:42:55 (host `ls`, numpy 1.26.3, threads pinned: True).

`inner="multistart"`: `qpep_core.solve_inner`, 6 starts, hot start `q0=q` + perturbations. `inner="exact"`: `quartic_roots.solve_inner_exact`, 162-path total-degree homotopy over the 40 projective critical points of the reduced problem (generic complex case), with multi-start fallback when a path is lost or the KKT Jacobian is singular.

## S20 — p=16, 4 seeds, noise=0.01, bias_z=0.08

| p | seed | J_ms | J_ex | ΔJ (ex−ms) | ‖q_ex−q_ms‖ | viol_ms | viol_ex | iters ms/ex | fb ms/ex | enum calls (fb) | t_ms (s) | t_ex (s) | ratio total | t_inner_ms (ms/call) | t_inner_ex (ms/call) | feasible ms/ex |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 16 | 0 | 0.004921786 | 0.004921786 | -1.69e-13 | 1.2e-10 | 5.5e-11 | 4.3e-11 | 16/16 | 0/0 | 17 (0) | 0.281 | 10.492 | 37.3x | 16.3 | 616.9 | True/True |
| 16 | 1 | 0.003942046 | 0.003942046 | -3.44e-14 | 5.9e-10 | 7.2e-11 | 5.8e-11 | 10/10 | 0/0 | 9 (4) | 0.142 | 6.596 | 46.3x | 15.6 | 732.6 | True/True |
| 16 | 2 | 0.002449209 | 0.002449209 | -1.42e-14 | 1.6e-10 | 8.7e-12 | 8.2e-12 | 9/9 | 0/0 | 12 (4) | 0.153 | 8.090 | 53.0x | 12.6 | 674.0 | True/True |
| 16 | 3 | 0.004911537 | 0.004911537 | -2.04e-13 | 6.5e-10 | 5.3e-11 | 4.9e-11 | 23/23 | 0/0 | 25 (5) | 0.248 | 15.934 | 64.2x | 9.8 | 637.1 | True/True |

Total-time ratio exact/multistart: min 37.3x, max 64.2x, median 46.6x. A single enumeration costs 649 ms on average here (162 paths).

## S21 — p=8 and p=64, 2 seeds each

| p | seed | J_ms | J_ex | ΔJ (ex−ms) | ‖q_ex−q_ms‖ | viol_ms | viol_ex | iters ms/ex | fb ms/ex | enum calls (fb) | t_ms (s) | t_ex (s) | ratio total | t_inner_ms (ms/call) | t_inner_ex (ms/call) | feasible ms/ex |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 8 | 0 | 0.004921786 | 0.004921786 | +2.73e-13 | 1.0e-10 | 2.5e-11 | 4.3e-11 | 16/16 | 0/0 | 16 (0) | 0.220 | 10.325 | 46.8x | 13.6 | 645.1 | True/True |
| 8 | 1 | 0.003942046 | 0.003942046 | +1.83e-14 | 2.4e-11 | 3.8e-11 | 3.4e-11 | 14/14 | 0/0 | 13 (6) | 0.164 | 8.470 | 51.7x | 12.4 | 651.2 | True/True |
| 64 | 0 | 0.004921786 | 0.004921786 | +1.81e-12 | 7.9e-10 | 7.6e-11 | 6.6e-11 | 14/13 | 0/0 | 15 (2) | 0.212 | 9.115 | 42.9x | 13.0 | 607.3 | True/True |
| 64 | 1 | 0.003942046 | 0.003942046 | +5.02e-14 | 2.4e-10 | 1.9e-11 | 2.7e-11 | 8/8 | 0/0 | 7 (4) | 0.122 | 4.974 | 40.7x | 16.7 | 709.9 | True/True |

S20/S21 totals: 8 exact ALM trials, 114 enumerations, 25 fallbacks (21.9% of calls).

## S22 — cost reduction

| variant | trial | J | t_total (s) | enum calls | t_enum/call (ms) | fallbacks | ΔJ vs full exact | ‖q−q_full‖ |
|---|---|---|---|---|---|---|---|---|
| n_gamma=1 | 0 | 0.004921786 | 9.468 | 17 | 557 | 0 | — | — |
| n_gamma=1 | 1 | 0.003942046 | 5.718 | 9 | 635 | 4 | — | — |
| n_gamma=2 | 0 | 0.004921786 | 22.705 | 17 | 1335 | 0 | — | — |
| n_gamma=2 | 1 | 0.003942046 | 13.571 | 10 | 1357 | 0 | — | — |
| fallback_starts=8 | 0 | 0.004921786 | 9.588 | 17 | 564 | 0 | — | — |
| fallback_starts=8 | 1 | 0.003942046 | 5.576 | 10 | 557 | 5 | — | — |
| fallback_starts=64 | 0 | 0.004921786 | 8.971 | 17 | 527 | 0 | — | — |
| fallback_starts=64 | 1 | 0.003942046 | 5.998 | 9 | 666 | 4 | — | — |
| hybrid:first | 0 | 0.004921786 | 1.996 | 3 | 596 | 0 | +9.44e-13 | 3.3e-10 |
| hybrid:first | 1 | 0.003942046 | 0.839 | 1 | 709 | 1 | +1.42e-13 | 5.7e-10 |
| hybrid:stable2 | 0 | 0.004921786 | 6.673 | 12 | 552 | 0 | -5.55e-17 | 1.8e-15 |
| hybrid:stable2 | 1 | 0.003942046 | 5.107 | 8 | 632 | 3 | +1.99e-13 | 7.5e-10 |
| hybrid:every3 | 0 | 0.004921786 | 3.953 | 7 | 545 | 0 | -7.55e-14 | 4.1e-11 |
| hybrid:every3 | 1 | 0.003942046 | 2.097 | 3 | 657 | 1 | -8.32e-14 | 6.4e-10 |
| hybrid:never | 0 | 0.004921786 | 0.239 | 0 | 0 | 0 | -1.69e-13 | 1.2e-10 |
| hybrid:never | 1 | 0.003942046 | 0.134 | 0 | 0 | 0 | -3.44e-14 | 5.9e-10 |
  * `hybrid:first` (2 trials): 4.7-6.8× cheaper than full exact, still 6.2-7.7× slower than pure multistart, max ΔJ 9.4e-13.
  * `hybrid:stable2` (2 trials): 1.1-1.4× cheaper than full exact, still 29.2-38.3× slower than pure multistart, max ΔJ 2.0e-13.
  * `hybrid:every3` (2 trials): 2.4-2.7× cheaper than full exact, still 15.1-16.7× slower than pure multistart, max ΔJ 8.3e-14.

Hybrid loop sanity: with the switch forced off ("never") the copied loop reproduces `alm_qpep(inner="multistart")` J within 0.0e+00, 0.0e+00, with identical outer-iteration counts (True) and fallback counts (True).

## S23 — correctness

| metric | value |
|---|---|
| completed trials (S20+S21) | 8 |
| max ‖q_ex−q_ms‖ (up to ±) | 7.9e-10 |
| max |J_ex−J_ms| | 1.8e-12 |
| trials with ‖Δq‖ > 1e-6 | 0 |
| trials with |ΔJ| > 1e-9 | 0 |
| trials where exact J is lower by >1e-9 (ms missed a min) | 0 |
| trials where multistart J is lower by >1e-9 (enum lost a root) | 0 |
| enumeration calls | 114 |
| enumerations that fell back (exact=False) | 25 |
| fallback rate | 21.9% |
| enumerations with singular KKT Jacobian | 0 |
| degenerate rate | 0.0% |
| enumerations with well-conditioned Mred | 114 |
| enum calls with <40 finite projective roots | 25 |
| min over calls of eig_min(Mred) | 8.2e-04 |
| max over calls of eig_min(Mred) | 1.8e-03 |
| enumeration wall time (mean) | 0.649 s |
| independent 64-start BFGS probes | 114 |
| probes where 64-start BFGS missed the enumerated min | 0 |
| probes where enumeration was above 64-start BFGS | 0 |
| max (f_ms64 − f_enum) | 8.7e-19 |
| min (f_ms64 − f_enum) | -4.9e-17 |

finite projective root counts over all enum calls: `{'37': 1, '38': 2, '39': 22, '40': 89}`; rank(Mred): `{'10': 114}`.

**Completeness caveat (P2-1b).** 89/114 enumerations certified the generic count of 40 finite projective roots (`exact=True`); 25/114 (21.9%) found only 37-39 and fell back to a 64-start BFGS (`exact=False`). At noise=0.01 every reduced matrix was full rank (rank 10 in all calls, min eig 8.2e-04 .. 1.8e-03) and no KKT Jacobian was singular (`degenerate`=0), so these fallbacks are *homotopy path loss*, not rank deficiency; a second complex phase (`n_gamma=2`, S22) recovered them at ~2.4x enumeration cost. The 40-projective/80-affine count is the generic *complex* count (P2-1b), so it is a completeness criterion only when the homotopy actually reaches it; the unconstrained PSD `Mred` at noise=0 is rank-deficient (probe below) and happened to still yield 40 here, which is a measurement, not a guarantee. `inner="exact"` therefore means: enumerate and certify at the current active set when `exact=True`, else report the fallback -- it is not a proof that the outer ALM found the global constrained optimum.

### Rank-deficiency probe (unconstrained P2 `Mred`)

The ALM assembles its reduced matrix from `M_A/D_A/G_A`, so the active-constraint terms regularise it (rank counts over all 114 enumerations above: `{'10': 114}`). The *unconstrained* P2 matrix `obj_min_t(M0,D0,G0)` is a different story:

| noise | seed | rank | eig_min | finite projective | real affine | degenerate | fallback | f |
|---|---|---|---|---|---|---|---|---|
| 0.00 | 0 | 8 | 5.0e-18 | 40 | 32 | False | False | 1.37e-16 |
| 0.00 | 1 | 8 | -1.4e-17 | 40 | 32 | False | False | -1.29e-17 |
| 0.01 | 0 | 9 | -1.3e-17 | 40 | 32 | False | False | 2.33e-03 |
| 0.01 | 1 | 9 | 2.3e-17 | 40 | 32 | False | False | 2.07e-03 |

So the singular-`Mred` regime is real (rank 8-9, eig_min down to -1.4e-17) and in these instances the homotopy still reached the generic 40 finite projective roots with no fallback. Completeness is therefore a *measured* property of each call, not a consequence of the 40-root theorem: the theorem counts the generic complex case (P2-1b), and the code only claims `exact=True` when no path was lost and the best KKT Jacobian is non-singular. On the ALM instances the observed failures were 37-39 finite roots (path loss), repaired by the 64-start fallback / a second complex phase.

## S24 — verdict

`inner="exact"` is **usable end-to-end** on all 8 completed ALM trials (p=8/16/64): it returns the same KKT point as multi-start (max ‖Δq‖ 7.9e-10, max |ΔJ| 1.8e-12) with identical feasibility, but it is **37×–64× slower end-to-end** (648.8 ms per 162-path enumeration vs 13.8 ms per 6-start BFGS call; 7–25 enumerations per ALM trial), because the reduced problem is rebuilt and re-solved at every outer iteration. The per-enumeration cost is p-independent (the reduced problem is 10x10 for every p). Completeness is conditional and measured: 89/114 enumerations found the generic 40 finite projective roots (exact=True), 25/114 (21.9%) found only 37-39 and fell back to 64-start BFGS; all reduced matrices were full rank (min eig 8.2e-04) and no KKT Jacobian was singular here, so the fallbacks are homotopy path loss, not rank deficiency (a second complex phase recovered them at ~2.4x cost). The 40-root count is the generic *complex* count (P2-1b), so it is a per-call measured criterion; the unconstrained PSD `Mred` is rank-deficient at noise=0 (rank 8, eig_min ~5e-18) yet still produced 40 finite roots on the 2 probe instances, while ALM's regularised active-set matrices were full rank. **Paper claim:** exact enumeration is a correctness / global-optimality *check* of the inner block at a fixed active set (it agrees with multi-start here, and on every one of 114 probed enumerations a fresh 64-start BFGS found the same global min), not a competitive inner solver; report it as an offline certificate with the fallback rate (21.9%) and the conditional-completeness caveat stated explicitly. Cost lever (measured on 2 trials only, not a general claim): a hybrid schedule that runs exact only at the first outer iteration, or only after the active set has been stable for 2 iterations, reproduces the full-exact J (ΔJ<=9.4e-13) at 4.7-6.8x / 1.1-1.4x lower cost, i.e. still 6-38x slower than pure multi-start -- use it only where a certified inner solve is actually needed.
