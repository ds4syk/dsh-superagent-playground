# Rigorous real-data certificates — LMO (n=72)

## Rebuild fidelity

- replay vs published `raw.json`: max |ΔJ| = 0.0, compared 72, missing 0 → identical instances

## Certificate rates (95% Wilson CI)

Primary bound = `rigorous_schur` (Theorem P2-Cert, 13-dim Schur); `dual-repair` (`rigorous`, global PSD shift) is the conservative fallback.
`cert` = ALM feasible and `J - bound <= 1e-6`; the Schur column counts only verified cases A/B (A_unverified excluded).
`naive` = same rule with the solver value (NOT a bound; diagnostic only).

| cuts | n | Schur cert (95% CI) | dual-repair cert (95% CI) | naive(value) | Schur case |
|-|-|-|-|-|-|
| base 14x14 | 72 | 11 (15.3%; [8.8, 25.3]) | 11 (15.3%; [8.8, 25.3]) | 19 | {'A': 72} |
| zero-dim RLT g_i g_j | 72 | 6 (8.3%; [3.9, 17.0]) | 6 (8.3%; [3.9, 17.0]) | 19 | {'A': 72} |
| bound-RLT g_i(y_k-lo_k) | 72 | 30 (41.7%; [31.0, 53.2]) | 24 (33.3%; [23.5, 44.8]) | 45 | {'A': 72} |

## Rigorous gap `J - bound` (<=1e-6 = certified)

| cuts | bound | n | min | median | max | #negative | median solve s |
|-|-|-|-|-|-|-|-|
| base 14x14 | Schur (primary) | 72 | 1.647e-11 | 1.546e-05 | 2.559e-04 | 0 | 1.66 |
| base 14x14 | dual-repair (fallback) | 72 | 2.000e-08 | 1.631e-05 | 2.565e-04 | 0 | 1.66 |
| zero-dim RLT g_i g_j | Schur (primary) | 72 | 3.177e-11 | 6.396e-05 | 3.277e+00 | 0 | 2.60 |
| zero-dim RLT g_i g_j | dual-repair (fallback) | 72 | 2.000e-08 | 6.355e-05 | 3.247e+00 | 0 | 2.60 |
| bound-RLT g_i(y_k-lo_k) | Schur (primary) | 72 | 5.599e-12 | 2.128e-06 | 9.368e-02 | 0 | 2.51 |
| bound-RLT g_i(y_k-lo_k) | dual-repair (fallback) | 72 | 2.000e-08 | 8.027e-06 | 9.368e-02 | 0 | 2.51 |

## Dual-repair vs Schur (same solver dual point)

- base: Schur - dual median 1.5694440620303768e-07, min -2.51343531626911e-06, max 1.278387278932187e-05, n>0 68, n<0 4; Schur case {'A': 72}; dual unrepaired 0; dual_shift max 3.630906157261498e-05; min λmin(S_rep) 3.218130297869549e-09; max cond(H_tt) 792.7750561711487
- rlt: Schur - dual median 2.0165152780648478e-08, min -0.03005542390072291, max 0.04655596412917751, n>0 49, n<0 23; Schur case {'A': 72}; dual unrepaired 0; dual_shift max 0.05193775511755266; min λmin(S_rep) 2.9644785422091163e-09; max cond(H_tt) 792.7750561711487
- brlt: Schur - dual median 2.4406024679953973e-08, min -0.004409674750862395, max 0.0006351612993813205, n>0 66, n<0 6; Schur case {'A': 72}; dual unrepaired 0; dual_shift max 0.0035115166002902595; min λmin(S_rep) 3.2286694524890535e-09; max cond(H_tt) 792.7750561711487

## Solver status

- base: {'optimal_inaccurate': 61, 'optimal': 11}
- rlt: {'optimal_inaccurate': 36, 'optimal': 36}
- brlt: {'optimal': 37, 'optimal_inaccurate': 35}

## Soundness audit

- ALM feasible: 72/72 (max viol 9.10e-11)
- dual-repair bound > J + 1e-08: 0 rows
- Schur bound > J + 1e-08: 0 rows
- Schur bound > solver value + 1e-08: 0 rows
- repaired slack λmin < -1e-10: 0 rows
- NaN Schur bounds: 0; B_noRt: 0
- worst violations: {'dual_repair': None, 'schur': None, 'schur_minus_value': None}

## RLT paired (rigorous Schur cert flags)

- base → brlt: 11/72 → 30/72 (+26.4 pp), 0→1 22, 1→0 3, McNemar p=0.000157
- base → rlt: 11/72 → 6/72 (-6.9 pp), 0→1 0, 1→0 5, McNemar p=0.0625
