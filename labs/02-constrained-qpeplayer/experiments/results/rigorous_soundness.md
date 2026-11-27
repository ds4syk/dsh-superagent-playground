# P2 rigorous soundness audit (all collected rows)

Checks over all three datasets and all three cut configs (`base`, `rlt`, `brlt`):

- `dual>J`: `rigorous > J + 1e-8` (a lower bound above a feasible value)
- `Schur>J`: `rigorous_schur > J + 1e-8`
- `Schur>value`: `rigorous_schur > value + 1e-8` (Schur cannot exceed the solver value)
- `invalid shift`: repaired `λmin(S_rep) < -1e-10`
- `NaN Schur`: non-finite `rigorous_schur` (e.g. `B_noRt`)
- `A_unverified`: finite Schur bound whose Cholesky/Rump check failed
- `B_noRt`: case B without a proven `||t|| <= R_t`

`optimal_inaccurate` is the SCS status count per config (base/rlt/brlt); the rigorous bounds do not depend on primal solver accuracy.

| dataset | rows | checks | dual>J | Schur>J | Schur>value | invalid shift | NaN Schur | A_unverified | B_noRt | optimal_inaccurate (base/rlt/brlt) |
|-|-|-|-|-|-|-|-|-|-|-|
| lm | 90 | 270 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 66/29/31 |
| lmo | 72 | 216 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 61/36/35 |
| ycbv | 84 | 252 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 55/22/23 |
| **total** | 246 | 738 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 182/87/89 |

Verdict: **PASS** (0 violations in 738 bound checks; 0 NaN Schur bounds).

Worst `Schur - J` per dataset: {"lm": null, "lmo": null, "ycbv": null}

