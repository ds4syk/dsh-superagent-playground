# Rigorous real-data certificates — LM (n=90)

## Rebuild fidelity

- replay vs published `raw.json`: max |ΔJ| = 0.0, compared 90, missing 0 → identical instances

## Certificate rates (95% Wilson CI)

Primary bound = `rigorous_schur` (Theorem P2-Cert, 13-dim Schur); `dual-repair` (`rigorous`, global PSD shift) is the conservative fallback.
`cert` = ALM feasible and `J - bound <= 1e-6`; the Schur column counts only verified cases A/B (A_unverified excluded).
`naive` = same rule with the solver value (NOT a bound; diagnostic only).

| cuts | n | Schur cert (95% CI) | dual-repair cert (95% CI) | naive(value) | Schur case |
|-|-|-|-|-|-|
| base 14x14 | 90 | 22 (24.4%; [16.7, 34.2]) | 23 (25.6%; [17.7, 35.4]) | 25 | {'A': 90} |
| zero-dim RLT g_i g_j | 90 | 19 (21.1%; [14.0, 30.6]) | 18 (20.0%; [13.0, 29.4]) | 35 | {'A': 90} |
| bound-RLT g_i(y_k-lo_k) | 90 | 46 (51.1%; [41.0, 61.2]) | 43 (47.8%; [37.8, 58.0]) | 61 | {'A': 90} |

## Rigorous gap `J - bound` (<=1e-6 = certified)

| cuts | bound | n | min | median | max | #negative | median solve s |
|-|-|-|-|-|-|-|-|
| base 14x14 | Schur (primary) | 90 | 3.983e-12 | 9.107e-06 | 3.990e-04 | 0 | 1.65 |
| base 14x14 | dual-repair (fallback) | 90 | 2.000e-08 | 9.298e-06 | 3.993e-04 | 0 | 1.65 |
| zero-dim RLT g_i g_j | Schur (primary) | 90 | 9.257e-12 | 1.635e-05 | 2.269e-01 | 0 | 0.96 |
| zero-dim RLT g_i g_j | dual-repair (fallback) | 90 | 2.000e-08 | 2.337e-05 | 3.573e-01 | 0 | 0.96 |
| bound-RLT g_i(y_k-lo_k) | Schur (primary) | 90 | 3.986e-12 | 3.369e-07 | 2.773e-01 | 0 | 1.96 |
| bound-RLT g_i(y_k-lo_k) | dual-repair (fallback) | 90 | 2.000e-08 | 2.052e-06 | 2.782e-01 | 0 | 1.96 |

## Dual-repair vs Schur (same solver dual point)

- base: Schur - dual median 8.49090739640226e-08, min -1.505778884930897e-06, max 3.3723957617989235e-06, n>0 82, n<0 8; Schur case {'A': 90}; dual unrepaired 0; dual_shift max 2.491206081827134e-05; min λmin(S_rep) 3.0603592840748965e-09; max cond(H_tt) 1668.0891130917041
- rlt: Schur - dual median 2.0305027439833042e-08, min -0.024004235587919323, max 0.24561152576557, n>0 77, n<0 13; Schur case {'A': 90}; dual unrepaired 0; dual_shift max 0.24977133010453675; min λmin(S_rep) 3.0624785408305953e-09; max cond(H_tt) 1668.0891130917041
- brlt: Schur - dual median 1.998970718902033e-08, min -0.007157250098995177, max 0.0009741713095260583, n>0 80, n<0 10; Schur case {'A': 90}; dual unrepaired 0; dual_shift max 0.005539221839014329; min λmin(S_rep) 3.1682806950570346e-09; max cond(H_tt) 1668.0891130917041

## Solver status

- base: {'optimal_inaccurate': 66, 'optimal': 24}
- rlt: {'optimal': 61, 'optimal_inaccurate': 29}
- brlt: {'optimal_inaccurate': 31, 'optimal': 59}

## Soundness audit

- ALM feasible: 90/90 (max viol 8.59e-11)
- dual-repair bound > J + 1e-08: 0 rows
- Schur bound > J + 1e-08: 0 rows
- Schur bound > solver value + 1e-08: 0 rows
- repaired slack λmin < -1e-10: 0 rows
- NaN Schur bounds: 0; B_noRt: 0
- worst violations: {'dual_repair': None, 'schur': None, 'schur_minus_value': None}

## RLT paired (rigorous Schur cert flags)

- base → brlt: 22/90 → 46/90 (+26.7 pp), 0→1 27, 1→0 3, McNemar p=8.43e-06
- base → rlt: 22/90 → 19/90 (-3.3 pp), 0→1 2, 1→0 5, McNemar p=0.453
