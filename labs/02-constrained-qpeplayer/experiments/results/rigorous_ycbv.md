# Rigorous real-data certificates — YCBV (n=84)

## Rebuild fidelity

- replay vs published `raw.json`: max |ΔJ| = 0.0, compared 84, missing 0 → identical instances

## Certificate rates (95% Wilson CI)

Primary bound = `rigorous_schur` (Theorem P2-Cert, 13-dim Schur); `dual-repair` (`rigorous`, global PSD shift) is the conservative fallback.
`cert` = ALM feasible and `J - bound <= 1e-6`; the Schur column counts only verified cases A/B (A_unverified excluded).
`naive` = same rule with the solver value (NOT a bound; diagnostic only).

| cuts | n | Schur cert (95% CI) | dual-repair cert (95% CI) | naive(value) | Schur case |
|-|-|-|-|-|-|
| base 14x14 | 84 | 11 (13.1%; [7.5, 21.9]) | 10 (11.9%; [6.6, 20.5]) | 17 | {'A': 84} |
| zero-dim RLT g_i g_j | 84 | 6 (7.1%; [3.3, 14.7]) | 6 (7.1%; [3.3, 14.7]) | 11 | {'A': 84} |
| bound-RLT g_i(y_k-lo_k) | 84 | 20 (23.8%; [16.0, 33.9]) | 19 (22.6%; [15.0, 32.6]) | 31 | {'A': 84} |

## Rigorous gap `J - bound` (<=1e-6 = certified)

| cuts | bound | n | min | median | max | #negative | median solve s |
|-|-|-|-|-|-|-|-|
| base 14x14 | Schur (primary) | 84 | 1.005e-11 | 2.104e-05 | 4.015e-04 | 0 | 1.68 |
| base 14x14 | dual-repair (fallback) | 84 | 2.000e-08 | 2.161e-05 | 3.904e-04 | 0 | 1.68 |
| zero-dim RLT g_i g_j | Schur (primary) | 84 | 1.003e-11 | 3.815e-05 | 1.074e+00 | 0 | 1.12 |
| zero-dim RLT g_i g_j | dual-repair (fallback) | 84 | 2.000e-08 | 3.959e-05 | 1.069e+00 | 0 | 1.12 |
| bound-RLT g_i(y_k-lo_k) | Schur (primary) | 84 | 4.096e-12 | 1.248e-05 | 3.010e+00 | 0 | 1.52 |
| bound-RLT g_i(y_k-lo_k) | dual-repair (fallback) | 84 | 2.000e-08 | 1.341e-05 | 2.996e+00 | 0 | 1.52 |

## Dual-repair vs Schur (same solver dual point)

- base: Schur - dual median 7.183546414074193e-08, min -1.1081893349630082e-05, max 2.162798978554526e-05, n>0 78, n<0 6; Schur case {'A': 84}; dual unrepaired 0; dual_shift max 5.674932789766132e-05; min λmin(S_rep) 2.9554392815132727e-09; max cond(H_tt) 861.0059145504672
- rlt: Schur - dual median 2.0210129613494143e-08, min -0.018213574721558407, max 0.0014016421622236042, n>0 68, n<0 16; Schur case {'A': 84}; dual unrepaired 0; dual_shift max 0.0374553781529788; min λmin(S_rep) 2.944343982531327e-09; max cond(H_tt) 861.0059145504672
- brlt: Schur - dual median 2.1550572969585686e-08, min -0.014882399497457038, max 0.00011672949267551975, n>0 73, n<0 11; Schur case {'A': 84}; dual unrepaired 0; dual_shift max 0.024409388231589522; min λmin(S_rep) 2.964547103015206e-09; max cond(H_tt) 861.0059145504672

## Solver status

- base: {'optimal': 29, 'optimal_inaccurate': 55}
- rlt: {'optimal': 62, 'optimal_inaccurate': 22}
- brlt: {'optimal': 61, 'optimal_inaccurate': 23}

## Soundness audit

- ALM feasible: 84/84 (max viol 8.71e-11)
- dual-repair bound > J + 1e-08: 0 rows
- Schur bound > J + 1e-08: 0 rows
- Schur bound > solver value + 1e-08: 0 rows
- repaired slack λmin < -1e-10: 0 rows
- NaN Schur bounds: 0; B_noRt: 0
- worst violations: {'dual_repair': None, 'schur': None, 'schur_minus_value': None}

## RLT paired (rigorous Schur cert flags)

- base → brlt: 11/84 → 20/84 (+10.7 pp), 0→1 12, 1→0 3, McNemar p=0.0352
- base → rlt: 11/84 → 6/84 (-6.0 pp), 0→1 0, 1→0 5, McNemar p=0.0625
