# 6D fold: standard vs tau^(1/3)-regularized Laplace

tau=0.01, eps_reg=tau^(1/3)=0.2154, grid step 2e-05, n_starts=512. `det(H+eps I)` replaces `det H` in the volume weight; its theta-derivative is `trace((H+eps I)^-1 dH/dtheta)`.

## Continuity across the fold

| variant | max consecutive pose jump | jump / grid step | K values |
|-|-|-|-|
| std | 3.089e-01 | 1.545e+04 | [3, 4] |
| reg | 1.070e-01 | 5.350e+03 | [3, 4] |

A genuine step map keeps the pose jump constant as the grid step shrinks, so `jump/step -> infinity`; a continuous map has `jump/step` roughly constant (= the derivative).

## Gradient checks

| theta | variant | mode set stable | grad rel vs central FD |
|-|-|-|-|
| 0.1 | std | True | 8.46e-07 |
| 0.1 | reg | True | 8.46e-07 |
| 0.02 | std | True | 1.20e-04 |
| 0.02 | reg | True | 1.20e-04 |
| 0.005 | std | True | 5.73e-04 |
| 0.005 | reg | True | 5.57e-04 |
| 0.0012 | std | True | 2.90e-01 |
| 0.0012 | reg | True | 1.19e-02 |

## Window around the mode death

| theta | K std | K reg | detH min std | detH min reg | w_fragile std | w_fragile reg | |dpose| std | |dpose| reg |
|-|-|-|-|-|-|-|-|-|
| 1.260e-3 | 4 | 4 | 2.36e-01 | 2.36e-01 | 0.145 | 0.063 | 1083 | 423 |
| 1.280e-3 | 4 | 4 | 1.92e-01 | 1.92e-01 | 0.159 | 0.064 | 1619 | 428 |
| 1.300e-3 | 4 | 4 | 1.33e-01 | 1.33e-01 | 0.187 | 0.065 | 3529 | 443 |
| 1.320e-3 | 3 | 3 | 1.46e+00 | 1.46e+00 | 0.017 | 0.013 | 413 | 407 |
| 1.340e-3 | 3 | 3 | 1.47e+00 | 1.47e+00 | 0.017 | 0.013 | 411 | 405 |
| 1.360e-3 | 3 | 3 | 1.47e+00 | 1.47e+00 | 0.017 | 0.013 | 408 | 403 |

## Reading (measured, 2026-11-27)

- The **derivative** singularity is largely cured: at theta=1.30e-3 the standard |dpose| is 3529 while the regularized one is 443, and the near-fold gradient error falls from 2.90e-1 to 1.19e-2 (24x). Away from the fold (theta=0.1, 0.02) the two variants agree to the printed precision, so the regularizer does not damage the ordinary backward pass.
- The **value** discontinuity at mode death is only reduced (pose jump 3.09e-1 -> 1.07e-1), not removed: K still changes 4 -> 3 and `jump/step` stays ~5e3, i.e. the map is still a step there. det(H+eps I) cannot fix this because the cause is the birth/death of a minimum (the Laplace sum drops a term), not the 1/sqrt(det H) divergence.
- Consequence for the paper: tau^(1/3) Hessian regularization is a defensible cheap mitigation for the fold neighbourhood, but a continuity claim across the fold needs uniform (Airy) asymptotics or a local numerical integral over the merging pair. The existing 'smooth at the crossing' claim must stay restricted to |theta| below the fold.
