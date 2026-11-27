# P2 rigorous real-data certificates (rounds S1-S9)

Honest lower bounds on the *published* BOP instances (replayed with the identical instance/solver RNG stream):

- **primary**: `rigorous_schur`, the P2-Cert 13-dim Schur bound (Theorem P2-Cert, case A/B; only verified cases are counted as cert);
- **fallback**: `rigorous`, the global `repair_dual` lower bound (dual slack + PSD shift; conservative by the 2c shift).

`cert` = ALM feasible and `J - bound <= 1e-6`. The naive one-sided rule uses the solver value (`value`, NOT a bound) and is listed only to show how much of the old published rate was not honest.

## Per-dataset rigorous certificate rates

| dataset | n | cuts | Schur cert (95% CI) | dual-repair cert (95% CI) | naive(value) | Schur gap min/median/max |
|-|-|-|-|-|-|-|
| lm | 90 | base | 22 (24.4%; [16.7,34.2]) | 23 (25.6%; [17.7,35.4]) | 25 | 3.98e-12/9.11e-06/3.99e-04 |
| lm | 90 | rlt | 19 (21.1%; [14.0,30.6]) | 18 (20.0%; [13.0,29.4]) | 35 | 9.26e-12/1.64e-05/2.27e-01 |
| lm | 90 | brlt | 46 (51.1%; [41.0,61.2]) | 43 (47.8%; [37.8,58.0]) | 61 | 3.99e-12/3.37e-07/2.77e-01 |
| lmo | 72 | base | 11 (15.3%; [8.8,25.3]) | 11 (15.3%; [8.8,25.3]) | 19 | 1.65e-11/1.55e-05/2.56e-04 |
| lmo | 72 | rlt | 6 (8.3%; [3.9,17.0]) | 6 (8.3%; [3.9,17.0]) | 19 | 3.18e-11/6.40e-05/3.28e+00 |
| lmo | 72 | brlt | 30 (41.7%; [31.0,53.2]) | 24 (33.3%; [23.5,44.8]) | 45 | 5.60e-12/2.13e-06/9.37e-02 |
| ycbv | 84 | base | 11 (13.1%; [7.5,21.9]) | 10 (11.9%; [6.6,20.5]) | 17 | 1.00e-11/2.10e-05/4.02e-04 |
| ycbv | 84 | rlt | 6 (7.1%; [3.3,14.7]) | 6 (7.1%; [3.3,14.7]) | 11 | 1.00e-11/3.81e-05/1.07e+00 |
| ycbv | 84 | brlt | 20 (23.8%; [16.0,33.9]) | 19 (22.6%; [15.0,32.6]) | 31 | 4.10e-12/1.25e-05/3.01e+00 |

## Schur case distribution (A/B are proven; A_unverified is not quoted)

| dataset | cuts | A | A_unverified | B | B_noRt | other | solver status |
|-|-|-|-|-|-|-|-|
| lm | base | 90 | 0 | 0 | 0 | 0 | {'optimal_inaccurate': 66, 'optimal': 24} |
| lm | rlt | 90 | 0 | 0 | 0 | 0 | {'optimal': 61, 'optimal_inaccurate': 29} |
| lm | brlt | 90 | 0 | 0 | 0 | 0 | {'optimal_inaccurate': 31, 'optimal': 59} |
| lmo | base | 72 | 0 | 0 | 0 | 0 | {'optimal_inaccurate': 61, 'optimal': 11} |
| lmo | rlt | 72 | 0 | 0 | 0 | 0 | {'optimal_inaccurate': 36, 'optimal': 36} |
| lmo | brlt | 72 | 0 | 0 | 0 | 0 | {'optimal': 37, 'optimal_inaccurate': 35} |
| ycbv | base | 84 | 0 | 0 | 0 | 0 | {'optimal': 29, 'optimal_inaccurate': 55} |
| ycbv | rlt | 84 | 0 | 0 | 0 | 0 | {'optimal': 62, 'optimal_inaccurate': 22} |
| ycbv | brlt | 84 | 0 | 0 | 0 | 0 | {'optimal': 61, 'optimal_inaccurate': 23} |

## Dual-repair vs Schur (same solver dual point)

| dataset | cuts | Schur-dual median | min | max | #Schur tighter | #Schur looser | dual unrepaired | dual_shift max | max cond(H_tt) |
|-|-|-|-|-|-|-|-|-|-|
| lm | base | 8.49e-08 | -1.51e-06 | 3.37e-06 | 82 | 8 | 0 | 2.49e-05 | 1668.1 |
| lm | rlt | 2.03e-08 | -2.40e-02 | 2.46e-01 | 77 | 13 | 0 | 2.50e-01 | 1668.1 |
| lm | brlt | 2.00e-08 | -7.16e-03 | 9.74e-04 | 80 | 10 | 0 | 5.54e-03 | 1668.1 |
| lmo | base | 1.57e-07 | -2.51e-06 | 1.28e-05 | 68 | 4 | 0 | 3.63e-05 | 792.8 |
| lmo | rlt | 2.02e-08 | -3.01e-02 | 4.66e-02 | 49 | 23 | 0 | 5.19e-02 | 792.8 |
| lmo | brlt | 2.44e-08 | -4.41e-03 | 6.35e-04 | 66 | 6 | 0 | 3.51e-03 | 792.8 |
| ycbv | base | 7.18e-08 | -1.11e-05 | 2.16e-05 | 78 | 6 | 0 | 5.67e-05 | 861.0 |
| ycbv | rlt | 2.02e-08 | -1.82e-02 | 1.40e-03 | 68 | 16 | 0 | 3.75e-02 | 861.0 |
| ycbv | brlt | 2.16e-08 | -1.49e-02 | 1.17e-04 | 73 | 11 | 0 | 2.44e-02 | 861.0 |

## Runtime (per config, seconds per solve)

| dataset | cuts | cert solve median | cert solve max | Schur post-process median | median #ineq |
|-|-|-|-|-|-|
| lm | base | 1.65 | 2.72 | 2.02e-04 | 16 |
| lm | rlt | 0.96 | 4.45 | 2.66e-04 | 152 |
| lm | brlt | 1.96 | 3.82 | 3.51e-04 | 336 |
| lmo | base | 1.66 | 2.25 | 2.07e-04 | 16 |
| lmo | rlt | 2.60 | 3.59 | 2.73e-04 | 152 |
| lmo | brlt | 2.51 | 3.96 | 3.52e-04 | 336 |
| ycbv | base | 1.68 | 2.20 | 2.10e-04 | 16 |
| ycbv | rlt | 1.12 | 3.73 | 2.69e-04 | 152 |
| ycbv | brlt | 1.52 | 3.88 | 3.55e-04 | 336 |

## Soundness audit

Tolerances: gap `<= 1e-06` certifies; bound `> J + 1e-08` is a violation.

| dataset | n rows | ALM feasible | dual>J | Schur>J | Schur>value | invalid shift | NaN Schur | B_noRt | worst (Schur-J) |
|-|-|-|-|-|-|-|-|-|-|
| lm | 90 | 90 | 0 | 0 | 0 | 0 | 0 | 0 | None |
| lmo | 72 | 72 | 0 | 0 | 0 | 0 | 0 | 0 | None |
| ycbv | 84 | 84 | 0 | 0 | 0 | 0 | 0 | 0 | None |

## RLT paired (rigorous Schur cert flags)

- lm base→brlt: 22/90 → 46/90 (+26.7 pp), 0→1 27, 1→0 3, p=8.43e-06
- lm base→rlt: 22/90 → 19/90 (-3.3 pp), 0→1 2, 1→0 5, p=0.453
- lmo base→brlt: 11/72 → 30/72 (+26.4 pp), 0→1 22, 1→0 3, p=0.000157
- lmo base→rlt: 11/72 → 6/72 (-6.9 pp), 0→1 0, 1→0 5, p=0.0625
- ycbv base→brlt: 11/84 → 20/84 (+10.7 pp), 0→1 12, 1→0 3, p=0.0352
- ycbv base→rlt: 11/84 → 6/84 (-6.0 pp), 0→1 0, 1→0 5, p=0.0625

## LM-O depth+upright augmentation

- LM-O base-plane-only Schur cert before: 11/72; uncertified rows selected: 6 (selection=worst, 6 re-solved); certified after augmentation: base cuts 0/6, +bound-RLT 6/6, +full RLT 6/6 (alpha=12.0 deg, n_depth=4, zmin=0.02 m).

| frame | img | obj | bias | J_old | gap_old | J_new | gap_new(base) | gap_new(+brlt) | gap_new(+rlt) | cert(base) | cert(+brlt) | cert(+rlt) |
|-|-|-|-|-|-|-|-|-|-|-|-|-|
| 000002 | 1 | 8 | 30.0 | 4.5417e-04 | 2.56e-04 | 4.5417e-04 | 2.54e-04 | 9.02e-12 | 1.20e-09 | False | True | True |
| 000002 | 0 | 8 | 30.0 | 5.4827e-04 | 2.56e-04 | 5.4827e-04 | 2.37e-04 | 5.17e-11 | 5.09e-11 | False | True | True |
| 000002 | 1 | 5 | 30.0 | 3.0391e-04 | 1.78e-04 | 3.0391e-04 | 1.77e-04 | 1.03e-11 | 5.34e-09 | False | True | True |
| 000002 | 0 | 5 | 30.0 | 2.5315e-04 | 1.15e-04 | 2.5315e-04 | 1.15e-04 | 9.90e-11 | 9.78e-11 | False | True | True |
| 000002 | 0 | 8 | 20.0 | 2.9761e-04 | 1.02e-04 | 2.9761e-04 | 1.02e-04 | 1.95e-11 | 1.93e-11 | False | True | True |
| 000002 | 0 | 12 | 30.0 | 2.3687e-04 | 1.00e-04 | 2.3687e-04 | 9.93e-05 | 6.01e-11 | 2.36e-10 | False | True | True |

Caveat: depth/upright rows restrict the feasible set; this certifies the augmented physical problem only, not the original 16-plane-only problem.


## Reproduction

- lm: n=90, max|ΔJ| vs raw.json = 0.0, missing 0 → OK
- lmo: n=72, max|ΔJ| vs raw.json = 0.0, missing 0 → OK
- ycbv: n=84, max|ΔJ| vs raw.json = 0.0, missing 0 → OK

## Bottom line

Base cuts = the published plane-only instance; bound-RLT adds only *valid* cuts, so its certificates are for the same problem.  The `naive(value)` column uses this run's SCS point (eps=1e-9, 50k iters) and therefore differs slightly from the earlier one-sided table (20k iters); both are upper bounds on the sound rate.

- lm [base]: Schur 22/90 (24.4%, CI [16.7,34.2]); dual-repair 23/90 (25.6%); naive 25/90.
- lm [brlt]: Schur 46/90 (51.1%, CI [41.0,61.2]); dual-repair 43/90 (47.8%); naive 61/90.
- lmo [base]: Schur 11/72 (15.3%, CI [8.8,25.3]); dual-repair 11/72 (15.3%); naive 19/72.
- lmo [brlt]: Schur 30/72 (41.7%, CI [31.0,53.2]); dual-repair 24/72 (33.3%); naive 45/72.
- ycbv [base]: Schur 11/84 (13.1%, CI [7.5,21.9]); dual-repair 10/84 (11.9%); naive 17/84.
- ycbv [brlt]: Schur 20/84 (23.8%, CI [16.0,33.9]); dual-repair 19/84 (22.6%); naive 31/84.

## Solver-quality caveat (brlt)

The bound-RLT gap distribution is bimodal: median 3.4e-7 (tight) but max 0.28 (loose).  A loose `rigorous_schur` is still a valid lower bound, but it is far below the true SDP optimum (which is >= the base optimum); the cause is the SCS dual point, not an invalid cut.  On the worst LM rows CLARABEL reports `Solver failed`, and raising SCS to 200k iterations did not reliably repair the dual (one of two probed rows improved to 1.2e-5, one stayed at 2.7).  So the brlt rates likely understate what this cut family could certify with a better SDP solve; base is the stable reference.


The rigorous rates **replace** the one-sided upper bounds: the one-sided rule counts `J - value <= 1e-6` with a non-bound `value` and is not a certificate.  On the published plane-only instances the honest Schur rate is well below the one-sided number (base), while the valid bound-RLT cuts recover part of it (brlt).  The remaining gap is not an invalid bound (738 rows×configs checked, 0 violations): on the large-gap brlt rows it is demonstrably solver-dual quality (see caveat), while base gaps may additionally reflect genuine relaxation looseness.

