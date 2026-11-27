# Learned refinement through the robust layer: exact IFT vs frozen weights

> **Scope**: small synthetic 3D-3D toy (N=10 points; 5 train / 3 held-out instances per seed; boundary outliers p=0.40; c=0.050), GT-pose-supervised, one tiny MLP. This is a **gradient-correctness diagnostic for the layer backward**, not In-The-Dark and not a realistic learned registration pipeline.

## Setup

* 5 fixed seeds; each seed draws 5 training instances and 3 disjoint held-out instances with N=10 points and 4 boundary outliers (`e/c^2 ~ U(1/3,3)`).
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
| exact | 3.477e-04 | 1.743e-05 | 0.6346 | 0.0675 | 0.00856 | 0.00338 | 80% |
| naive | 3.531e-04 | 3.171e-05 | 0.6346 | 0.1062 | 0.00856 | 0.00450 | 100% |
| stopgrad | 3.867e-04 | 3.867e-04 | 0.6346 | 0.6345 | 0.00856 | 0.00856 | 0% |

| seed | loss exact | loss naive | loss stopgrad | exact < naive | exact/naive | rot exact | rot naive |
|---|---|---|---|---|---|---|---|
| 0 | 6.712e-06 | 1.083e-05 | 3.867e-04 | True | 0.620 | 0.0439 | 0.0635 |
| 1 | 9.596e-06 | 1.846e-05 | 3.082e-04 | True | 0.520 | 0.0675 | 0.0946 |
| 2 | 2.788e-05 | 4.960e-05 | 5.674e-04 | True | 0.562 | 0.0861 | 0.1071 |
| 3 | 1.743e-05 | 6.798e-05 | 4.301e-04 | True | 0.256 | 0.0656 | 0.1548 |
| 4 | 1.458e-03 | 3.171e-05 | 2.446e-04 | False | 45.959 | 0.4774 | 0.1062 |

**Exact beats naive on 4/5 paired seeds**; exact beats stopgrad on 4/5. Median loss ratio exact/naive = 0.562.

### Held-out protocol: median over 5 seeds

| arm | initial loss | final loss | initial rot (deg) | final rot (deg) | initial trans | final trans | improved |
|---|---|---|---|---|---|---|---|
| exact | 3.532e-04 | 1.333e-03 | 0.5847 | 1.1003 | 0.00677 | 0.02542 | 0% |
| naive | 3.503e-04 | 1.249e-03 | 0.5847 | 1.0940 | 0.00677 | 0.01935 | 0% |
| stopgrad | 3.446e-04 | 3.446e-04 | 0.5847 | 0.5849 | 0.00677 | 0.00678 | 0% |

| seed | loss exact | loss naive | loss stopgrad | exact < naive | exact/naive | rot exact | rot naive |
|---|---|---|---|---|---|---|---|
| 0 | 1.333e-03 | 1.274e-03 | 3.446e-04 | False | 1.046 | 1.2746 | 1.2728 |
| 1 | 5.732e-04 | 9.609e-04 | 2.200e-04 | True | 0.597 | 0.6408 | 0.8637 |
| 2 | 1.856e-03 | 6.649e-04 | 2.145e-04 | False | 2.791 | 1.1003 | 0.9805 |
| 3 | 2.068e+07 | 3.977e-03 | 1.139e-03 | False | 5199886757.839 | 50.6574 | 2.1188 |
| 4 | 9.912e-04 | 1.249e-03 | 4.216e-04 | True | 0.794 | 0.6200 | 1.0940 |

**Exact beats naive on 2/5 paired seeds**; exact beats stopgrad on 0/5. Median loss ratio exact/naive = 1.046.

## Reading the two protocols

* **Refine** (primary): both learned arms reduce the pose error on 4-5/5 seeds; exact reaches a lower loss than naive on 4/5 paired seeds and on the median (median final loss 1.74e-5 vs 3.17e-5, median final rotation 0.068 deg vs 0.106 deg), consistent with the exact IFT direction being the true descent direction through the layer. This is an optimisation diagnostic, not generalisation.
* **Refine caveat**: on seed 4 the exact arm leaves the smooth branch (final rot 0.477 deg, loss 1.46e-3): finite Adam steps on `delta_y` can push boundary points across the GM band and switch the robust basin, which no gradient (exact or naive) models. This is a real limitation of the toy, not hidden.
* **Held-out** (secondary scope check): a purely local per-point MLP cannot identify the random *global* pose correction (the target depends on `R_gt`, which the features do not contain), so both arms overfit the training instances and the held-out pose gets worse; on seed 3 the exact arm also switches the robust basin (50.7 deg, loss 2e7). This protocol is a **negative result**, not evidence of a usable learned layer, and it motivates using a data-fidelity term / global context in any follow-up.

## Honest scope

* Synthetic 3D-3D only; GT pose supervision; no descriptors, no real scans, no In-The-Dark, no SE(3) network, no large-scale training.
* The MLP is tiny and trained per seed on 5 instances; the refine protocol deliberately allows memorisation. The experiment tests whether the layer's backward direction is correct/useful, not whether a learned registration system works.
* The forward layer is GNC-GM + Newton polish in float64; the backward is the exact IFT derivative from `experiments/synthetic/common.py`, ported to torch and checked elementwise against that NumPy reference.
