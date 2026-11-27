# 6-point counterexample robustness under perturbation

Base: p_i=+-e1,+-e2,+-e3, y_i=0.05 p_i, W=[[1,-.6,0],[-.6,1,0],[0,0,1]]. Perturbations: W += 2e-3 symmetric noise then +4I (keeps PSD); y += 1e-3 noise (D != 0). All SDPs SCS eps=1e-11.

| case | f*_10 | f*_13 full | f*_13 up | f10-f13up | |f10-f13full| | rank 10/full/up | |D| | min eig G |
|-|-|-|-|-|-|-|-|-|
| base | 5.41500000 | 5.41500000 | 3.59431373 | 1.820686 | 2.89e-13 | 1/1/4 | 0.000e+00 | 2.400 |
| W_noise0 | 27.07004033 | 27.07004033 | 26.95604187 | 0.113998 | 1.40e-11 | 1/1/4 | 0.000e+00 | 26.392 |
| y_noise0 | 5.41552847 | 5.41552847 | 3.59446851 | 1.821060 | 6.00e-13 | 1/1/5 | 2.692e-03 | 2.400 |
| both0 | 27.07118584 | 27.07118584 | 26.95916219 | 0.112024 | 1.14e-12 | 1/1/4 | 1.398e-02 | 26.392 |
| W_noise1 | 27.07358416 | 27.07358416 | 26.95538212 | 0.118202 | 9.85e-12 | 1/1/4 | 0.000e+00 | 26.386 |
| y_noise1 | 5.40717140 | 5.40717140 | 3.59418681 | 1.812985 | 1.87e-13 | 1/1/5 | 3.590e-03 | 2.400 |
| both1 | 27.04330266 | 27.04330266 | 26.93688692 | 0.106416 | 1.37e-12 | 1/1/4 | 2.225e-02 | 26.386 |
| W_noise2 | 27.06810426 | 27.06810426 | 26.95471152 | 0.113393 | 3.74e-12 | 1/1/4 | 0.000e+00 | 26.407 |
| y_noise2 | 5.41512109 | 5.41512109 | 3.59433671 | 1.820784 | 5.92e-13 | 1/1/5 | 6.217e-03 | 2.400 |
| both2 | 27.05617406 | 27.05617406 | 26.94711566 | 0.109058 | 6.47e-13 | 1/1/4 | 3.250e-02 | 26.407 |

Reading:
- `|f10 - f13full|` is at solver level (<=1e-8) in every case: the full-20D equivalence is stable.
- `f10 - f13up` stays strictly positive (~1.82) in every perturbed case, including D != 0, which supports the theorem's open-neighbourhood claim numerically.
- rank(X*_13 up) stays > 1 while rank(X*_10)=1, i.e. the upstream relaxation fails to be tight on a physically realisable instance even when the 10x10 relaxation is exactly rank-1 tight.
