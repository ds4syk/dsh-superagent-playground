# P1-Q5c: Scorer / semi-line Airy fold continuation — numerical validation

Script `theory/run_scorer_fold.py`; raw data `theory/results/scorer_fold.json`; figure `theory/results/scorer_fold.png`. tau=0.01 (6D), tau=0.02 (1D).

## 1. 1D three-way validation

Model: canonical A2 normal form $J_\alpha(x)=x^3/3-\alpha x$ ($c_3=2$, fold at $\alpha=0$; minimum $x_+=\sqrt{\alpha}$, saddle $x_-=-\sqrt{\alpha}$ for $\alpha>0$, no real stationary point for $\alpha<0$). The reported quantity is the **fold-mode posterior fraction** $w/(w+Z_{\rm rest})$; the proof does not state its background $Z_{\rm rest}$, so we report raw weights as well and reconstruct the normalisation.

| variant | $\alpha=+10^{-5}$ raw | fraction | $\alpha=-10^{-5}$ raw | fraction | proof |
|-|-|-|-|-|-|
| standard real-root Laplace | 4.4575 | 0.867032 | 0.0000 | 0.000000 | 0.867 → 0 |
| $\tau^{1/3}$-regularised | 0.6726 | 0.495950 | 0.0000 | 0.000000 | 0.497 → 0 |
| **Scorer continuation** | 0.3496 | 0.338382 | 0.3496 | 0.338337 | 0.339113 → 0.339069 |

With $Z_{\rm rest}=0.6836$ the fractions are **0.8670 / 0.4959 / 0.33838 → 0.33834**, i.e. the proof's 0.867 / 0.497 / 0.339113 / 0.339069 to $\le 1.1\times10^-3$. A free 2-parameter fit (tau, $Z_{\rm rest}$) gives tau=0.01954, $Z_{\rm rest}$=0.67587 and reproduces **all four** printed numbers to $\le 1.6e-04$ — so the proof's triplet is this fraction with a fixed background, and the qualitative claim is normalisation-independent.

- The Scorer weight is continuous by construction ($\mathcal{S}$ entire, all four invariants symmetric functions of the root pair) and numerically $w_S(+)/w_S(-)=1.000131$ vs the proof's 1.000130.
- Standard Laplace and $\tau^{1/3}$ regularisation both jump to exactly 0 for $\alpha<0$ (the real minimum no longer exists); this is theorem P1-Q5b.

## 2. 6D application on `instance.py` (tau=0.01)

Fold located at **$\theta_f=1.3185485e-03$** (positive-$\theta$ fold; the mirror one sits at $-\theta_f$ by the exact $P$-symmetry $M(-\theta)=PM(\theta)P^T$). It is a genuine saddle-node: the fragile minimum's soft Hessian eigenvalue crosses zero there and $\det H\to0$. $c_3$ measured at the fold point ($d^3J/ds^3$ along the null eigenvector) is **2.55981**.

### What is exact vs approximate

| item | status |
|-|-|
| fold point / $c_3$ / $J_0$ / $\alpha$ / $\det H_\perp$ | **measured** at each $\theta$ from Newton-polished roots; no normal-form fit is used for them |
| saddle $q_-$ | **measured** (pure Newton from $q_+ + t e_{\min}$, exactly one negative eigenvalue, $|\nabla J|<10^{-9}$) |
| complex pair $q_\pm$ for $\alpha<0$ | **measured** — complex Newton on the complexified critical-point system converges; $\mathrm{Im}\,J$ and $|\lambda_1^{\mathbb C}|$ give $|\alpha|$ and $c_3$ independently |
| $w_{\rm fold}$ | exact given the measured invariants and the proof's formula |
| normal-form corrections | the real problem is quartic, so $\varphi''''\approx8.3$; the invariants are exact, the *identification* with the normal form is leading-order |

### Continuity of the normal-form invariants across $\alpha=0$

| $\theta-\theta_f$ | side | $\alpha$ | $c_3$ | $J_0$ | $\det H_\perp$ | $\zeta$ | $\mathcal{S}(\zeta)$ |
|-|-|-|-|-|-|-|-|
| -0.0160e-3 | real | +5.398e-05 | 2.58518 | -0.455815760 | 7.41712 | +1.068e-03 | 1.288902 |
| -0.0120e-3 | real | +4.047e-05 | 2.58168 | -0.455838088 | 7.42671 | +8.007e-04 | 1.288651 |
| -0.0080e-3 | real | +2.696e-05 | 2.57757 | -0.455860418 | 7.43790 | +5.338e-04 | 1.288401 |
| -0.0060e-3 | real | +2.021e-05 | 2.57514 | -0.455871583 | 7.44448 | +4.003e-04 | 1.288275 |
| -0.0040e-3 | real | +1.347e-05 | 2.57227 | -0.455882748 | 7.45219 | +2.669e-04 | 1.288150 |
| -0.0025e-3 | real | +8.416e-06 | 2.56963 | -0.455891122 | 7.45929 | +1.668e-04 | 1.288056 |
| -0.0010e-3 | real | +3.365e-06 | 2.56599 | -0.455899496 | 7.46900 | +6.671e-05 | 1.287962 |
| +0.0015e-3 | cplx | -5.043e-06 | 2.55974 | -0.455913454 | 7.48568 | -1.001e-04 | 1.287805 |
| +0.0030e-3 | cplx | -1.009e-05 | 2.55966 | -0.455921828 | 7.48605 | -2.001e-04 | 1.287711 |
| +0.0060e-3 | cplx | -2.017e-05 | 2.55949 | -0.455938577 | 7.48679 | -4.002e-04 | 1.287524 |
| +0.0100e-3 | cplx | -3.361e-05 | 2.55927 | -0.455960911 | 7.48777 | -6.670e-04 | 1.287273 |
| +0.0150e-3 | cplx | -5.041e-05 | 2.55899 | -0.455988829 | 7.48900 | -1.000e-03 | 1.286960 |
| +0.0250e-3 | cplx | -8.401e-05 | 2.55843 | -0.456044668 | 7.49145 | -1.667e-03 | 1.286335 |
| +0.0400e-3 | cplx | -1.344e-04 | 2.55759 | -0.456128438 | 7.49514 | -2.667e-03 | 1.285399 |

Low-order polynomial residual over the 14 two-sided points: $\alpha$ rel. 9.93e-04 (slope -3.3626/unit $\theta$), $c_3$ rel. 6.96e-03, $J_0$ rel. 1.66e-12, $\det H_\perp$ rel. 1.82e-03. No jump at $\theta_f$. The complex side's $|\alpha|/(\theta-\theta_f)=3.3620$ agrees with the real side's $|\alpha'|=3.3626$ to 0.02%, and the complex $c_3$ is constant at 2.5598 (fold value 2.5598).

### Posterior / pose-map jump across the fold

Grid $\theta=\theta_f+[-1.0,+1.0]\times10^{-4}$ step $2\times10^{-5}$; mode enumeration `find_modes(n_starts=512)` for the non-fold minima, the fold mode handled by each variant. Pose is `Instance.pose` (6-vector); jump is the norm of the consecutive difference.

| variant | max consecutive pose jump | jump / step | median jump | K over the grid |
|-|-|-|-|-|
| standard real-root Laplace | 3.0414e-01 | 1.5207e+04 | 1.179e-02 | [3, 4] |
| det(H+tau^(1/3) I) | 1.0684e-01 | 5.3420e+03 | 8.272e-03 | [3, 4] |
| **Scorer continuation** | 1.2057e-02 | 6.0283e+02 | 8.183e-03 | [4] |

A genuine step keeps its jump constant as the grid is refined (`jump/step -> infinity`); a $C^\infty$ map has `jump/step -> |dpose|`.

## 3. Verdict

- **1D**: the proof's three-way comparison is reproduced (0.8670 / 0.4959 / 0.33838 → 0.33834).
- **6D**: complex root tracking is *not* numerically infeasible — it works. $w_{\rm fold}$ and all four invariants are continuous through $\theta_f$; the pose jump is 1.206e-02 vs 3.041e-01 (standard) and 1.068e-01 (regularised), i.e. the Scorer continuation removes the fold step.
- **Caveats**: (i) the 6D fold is very shallow ($\Delta J\sim10^-6\ll\tau=0.01$, $\zeta\sim10^-3$), so the Scorer weight is far from its Laplace limit *at* the fold and differs from standard Laplace by a large factor there — the theorem only promises agreement for $\Delta J\gg\tau$; (ii) the pose jump is measured with `find_modes` for the non-fold minima, so an enumeration miss would show up as a spurious step (none is seen for the Scorer variant); (iii) $w_{\rm fold}$ uses the measured invariants but the normal form is the leading-order local model ($\varphi''''=8.3$), so the *weight* is exact given $(J_0,\alpha,c_3,H_\perp)$ while the identification of the latter with the normal form is asymptotic.
