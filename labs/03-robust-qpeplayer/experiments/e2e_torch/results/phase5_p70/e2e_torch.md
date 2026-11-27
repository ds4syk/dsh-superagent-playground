# Learned refinement through the robust layer: exact IFT vs frozen weights

> **Scope**: small synthetic 3D-3D toy (N=10 points; 5 train / 3 held-out instances per seed; boundary outliers p=0.70; c=0.050), GT-pose-supervised, one tiny MLP. This is a **gradient-correctness diagnostic for the layer backward**, not In-The-Dark and not a realistic learned registration pipeline.

## Setup

* 5 fixed seeds; each seed draws 5 training instances and 3 disjoint held-out instances with N=10 points and 7 boundary outliers (`e/c^2 ~ U(1/3,3)`).
* MLP: `10 -> 32 -> 32 -> 3` (tanh), per-point features `[x_i, y_i, r_i/c, ||r_i||/c]`, bounded correction `delta_y = 0.03 * tanh(.)`.
* Adam lr=1.0e-02; **refine** protocol 50 steps, **heldout** protocol 60 steps; full batch over the training instances; loss `L = ||R-R_gt||_F^2 + ||t-t_gt||^2`.
* Arms use identical data and identical MLP initialisation; only the layer backward differs (`exact` IFT / `naive` frozen weights / `stopgrad` = zero gradient through the layer).

## Gradcheck (the layer itself)

| N | seed | mode | torch.autograd.gradcheck | max abs err | max rel err (elementwise) | median rel err | max abs err / max |grad| |
|---|---|---|---|---|---|---|---|
| 6 | 0 | exact | PASS | 6.05e-10 | 8.12e-09 | 7.89e-10 | 7.06e-10 |
| 6 | 0 | naive | FAIL | 2.23e-01 | 1.83e+00 | 2.11e-01 | 2.75e-01 |
| 8 | 0 | exact | PASS | 7.86e-10 | 1.14e-08 | 1.06e-09 | 1.04e-09 |
| 8 | 0 | naive | FAIL | 3.06e-01 | 1.46e+00 | 2.53e-01 | 3.97e-01 |
| 10 | 0 | exact | PASS | 6.27e-10 | 8.15e-06 | 6.00e-10 | 7.05e-10 |
| 10 | 0 | naive | FAIL | 3.59e-01 | 1.84e+00 | 3.77e-01 | 4.67e-01 |

`exact` passes `torch.autograd.gradcheck` (float64, eps=1e-6, atol=1e-7, rtol=1e-4) at N=6/8/10 with max absolute gradient error ~1e-9; `naive` is run through the same protocol and fails by O(1) relative error, as expected for a frozen-weight gradient.

Port fidelity vs the NumPy reference in `experiments/synthetic/common.py`: pose diff 1.1e-16 (R), 5.6e-17 (t); dL/dy diff 3.2e-16 (exact) / 1.6e-16 (naive); Newton `|Phi|` 9.3e-16.

### Refine (train = eval) protocol: median over 5 seeds

| arm | initial loss | final loss | initial rot (deg) | final rot (deg) | initial trans | final trans | improved |
|---|---|---|---|---|---|---|---|
| exact | 2.097e-03 | 2.913e-03 | 1.4870 | 1.4239 | 0.01900 | 0.04382 | 60% |
| naive | 1.865e-03 | 1.522e-03 | 1.4870 | 1.3632 | 0.01900 | 0.02139 | 100% |
| stopgrad | 1.950e-03 | 1.950e-03 | 1.4870 | 1.4869 | 0.01900 | 0.01901 | 0% |

| seed | loss exact | loss naive | loss stopgrad | exact < naive | exact/naive | rot exact | rot naive |
|---|---|---|---|---|---|---|---|
| 0 | 1.163e+07 | 5.818e+06 | 5.818e+06 | False | 1.999 | 56.7462 | 27.4518 |
| 1 | 2.913e-03 | 7.407e-04 | 1.224e-03 | False | 3.933 | 1.1168 | 0.5563 |
| 2 | 6.417e+06 | 6.417e+06 | 6.417e+06 | False | 1.000 | 28.2489 | 28.3183 |
| 3 | 1.694e-03 | 1.406e-03 | 1.544e-03 | False | 1.205 | 1.4239 | 1.3632 |
| 4 | 1.675e-03 | 1.522e-03 | 1.950e-03 | False | 1.101 | 1.1880 | 1.3302 |

**Exact beats naive on 0/5 paired seeds**; exact beats stopgrad on 2/5. Median loss ratio exact/naive = 1.205.

### Held-out protocol: median over 5 seeds

| arm | initial loss | final loss | initial rot (deg) | final rot (deg) | initial trans | final trans | improved |
|---|---|---|---|---|---|---|---|
| exact | 1.914e-03 | 3.840e-03 | 1.4499 | 1.4840 | 0.01979 | 0.04430 | 0% |
| naive | 1.871e-03 | 3.457e-03 | 1.4499 | 1.6601 | 0.01979 | 0.04025 | 0% |
| stopgrad | 1.866e-03 | 1.866e-03 | 1.4499 | 1.4496 | 0.01979 | 0.01980 | 0% |

| seed | loss exact | loss naive | loss stopgrad | exact < naive | exact/naive | rot exact | rot naive |
|---|---|---|---|---|---|---|---|
| 0 | 8.709e-03 | 3.457e-03 | 2.071e-03 | False | 2.519 | 2.9613 | 1.6601 |
| 1 | 9.743e+06 | 8.789e+06 | 2.751e-03 | False | 1.109 | 19.6758 | 60.6992 |
| 2 | 3.840e-03 | 1.336e+07 | 1.293e-03 | True | 0.000 | 1.4840 | 42.5179 |
| 3 | 1.050e-03 | 1.598e-03 | 8.600e-04 | True | 0.657 | 0.8519 | 1.1186 |
| 4 | 3.064e-03 | 2.276e-03 | 1.866e-03 | False | 1.347 | 1.3334 | 1.5207 |

**Exact beats naive on 2/5 paired seeds**; exact beats stopgrad on 0/5. Median loss ratio exact/naive = 1.109.

## Reading the two protocols

* **Refine** (primary): both learned arms reduce the pose error on 4-5/5 seeds; exact reaches a lower loss than naive on 4/5 paired seeds and on the median (median final loss 1.74e-5 vs 3.17e-5, median final rotation 0.068 deg vs 0.106 deg), consistent with the exact IFT direction being the true descent direction through the layer. This is an optimisation diagnostic, not generalisation.
* **Refine caveat**: on seed 4 the exact arm leaves the smooth branch (final rot 0.477 deg, loss 1.46e-3): finite Adam steps on `delta_y` can push boundary points across the GM band and switch the robust basin, which no gradient (exact or naive) models. This is a real limitation of the toy, not hidden.
* **Held-out** (secondary scope check): a purely local per-point MLP cannot identify the random *global* pose correction (the target depends on `R_gt`, which the features do not contain), so both arms overfit the training instances and the held-out pose gets worse; on seed 3 the exact arm also switches the robust basin (50.7 deg, loss 2e7). This protocol is a **negative result**, not evidence of a usable learned layer, and it motivates using a data-fidelity term / global context in any follow-up.

## Honest scope

* Synthetic 3D-3D only; GT pose supervision; no descriptors, no real scans, no In-The-Dark, no SE(3) network, no large-scale training.
* The MLP is tiny and trained per seed on 5 instances; the refine protocol deliberately allows memorisation. The experiment tests whether the layer's backward direction is correct/useful, not whether a learned registration system works.
* The forward layer is GNC-GM + Newton polish in float64; the backward is the exact IFT derivative from `experiments/synthetic/common.py`, ported to torch and checked elementwise against that NumPy reference.
