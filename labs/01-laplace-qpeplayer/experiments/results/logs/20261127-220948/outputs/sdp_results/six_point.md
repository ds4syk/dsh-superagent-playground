# 6-point physical 3D-3D counterexample (new P1-Q2/Q13 proof)

eps=0.05, W eigs [0.4, 1.0, 1.6], p_i=+-e1,+-e2,+-e3, y_i=eps p_i, N=6, G=6W, D=0; value at R=I: 5.41500000.

| quantity | measured here | proof claim |
|-|-|-|
| f*_10 (21 constraints, full 20D) | 5.41500000 | 5.41500000 |
| f*_13 full (24 + 3 row-orth = full 20D) | 5.41500000 | 5.41500000 |
| f*_13 upstream (24 constraints, 17D) | 3.59431373 | 3.59430886 |
| f*_10 - f*_13_up | 1.82068627 | 1.82069114 |
| rank(X*_10) | 1 | 1 |
| rank(X*_13 upstream) | 4 | 4 |
| f_nonconvex (exact: max tr(WR)=3 at R=I) | 5.41500000 | 5.415 |
| f_nonconvex (300k quaternion samples, sanity) | 5.41522789 | 5.415 |
| gap_10 = f_nonconvex - f*_10 | -8.38e-13 | 0 |
| gap_13_full | -3.38e-13 | 0 |
| gap_13_up | 1.820686 | 1.82069114 |

Independent solver cross-check (CLARABEL, interior point): f10=5.41500000, f13_full=5.41500000, f13_up=3.59431371; f13_up eigenvalues [1.133562, 0.994552, 0.971246, 0.90063, 6e-06, 4e-06]. SCS and CLARABEL agree to 2.0e-8, so the 4.9e-6 difference from the proof's quoted 3.59430886 is not solver tolerance; the qualitative separation is confirmed but the last five digits are not reproduced. rank(X*_13up) is 4 at a 1e-5 threshold and 6 at 1e-6 (four O(1) eigenvalues then ~1e-5 ones); the theorem only needs rank > 1.

Constraints: 10x10 21 (exact rank 20), 13x13 upstream 24, 13x13 full 27.
Solver status: 10=optimal, 13full=optimal, 13up=optimal.
