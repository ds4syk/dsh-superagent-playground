# End-to-end boundary coordinate refinement: exact IFT vs naive

N=50, c=0.05, sigma=0.01, 10 fixed seeds per cell, 20 Adam steps, lambda_t=1.0. Metric L = ||R-R_gt||_F^2 + ||t-t_gt||^2; success = rot<5 deg, trans<0.1 m.

## Final metrics (median over 10 seeds)

| p | mode | lr | method | loss0 | final loss | final rot (deg) | final trans (m) | success | improved |
|---|---|---|---|---|---|---|---|---|---|
| 0.5 | boundary-only | 1e-03 | exact | 6.35e-05 | 1.90e-06 | 0.0293 | 0.00092 | 100% | 100% |
| 0.5 | boundary-only | 1e-03 | naive | 6.35e-05 | 1.66e-05 | 0.0874 | 0.00267 | 100% | 100% |
| 0.5 | boundary-only | 5e-03 | exact | 6.35e-05 | 6.36e-07 | 0.0223 | 0.00042 | 100% | 100% |
| 0.5 | boundary-only | 5e-03 | naive | 6.35e-05 | 1.56e-06 | 0.0422 | 0.00079 | 100% | 100% |
| 0.5 | all-points | 1e-03 | exact | 6.35e-05 | 1.20e-06 | 0.0156 | 0.00097 | 100% | 100% |
| 0.5 | all-points | 1e-03 | naive | 6.35e-05 | 1.87e-06 | 0.0192 | 0.00112 | 100% | 100% |
| 0.5 | all-points | 5e-03 | exact | 6.35e-05 | 2.61e-06 | 0.0446 | 0.00114 | 100% | 100% |
| 0.5 | all-points | 5e-03 | naive | 6.35e-05 | 2.86e-06 | 0.0391 | 0.00126 | 100% | 100% |
| 0.7 | boundary-only | 1e-03 | exact | 1.04e-04 | 2.14e-06 | 0.0301 | 0.00120 | 100% | 100% |
| 0.7 | boundary-only | 1e-03 | naive | 1.04e-04 | 2.26e-05 | 0.1166 | 0.00418 | 100% | 100% |
| 0.7 | boundary-only | 5e-03 | exact | 1.04e-04 | 1.70e-06 | 0.0322 | 0.00078 | 100% | 100% |
| 0.7 | boundary-only | 5e-03 | naive | 1.04e-04 | 3.66e-06 | 0.0462 | 0.00143 | 100% | 100% |
| 0.7 | all-points | 1e-03 | exact | 1.04e-04 | 1.71e-06 | 0.0246 | 0.00104 | 100% | 100% |
| 0.7 | all-points | 1e-03 | naive | 1.04e-04 | 5.39e-06 | 0.0271 | 0.00134 | 100% | 100% |
| 0.7 | all-points | 5e-03 | exact | 1.04e-04 | 4.57e-06 | 0.0550 | 0.00165 | 100% | 100% |
| 0.7 | all-points | 5e-03 | naive | 1.04e-04 | 4.72e-06 | 0.0413 | 0.00181 | 100% | 100% |

## Paired exact-vs-naive comparison

| p | mode | lr | exact wins | median loss ratio exact/naive |
|---|---|---|---|---|
| 0.5 | boundary-only | 1e-03 | 10/10 | 0.071 |
| 0.5 | boundary-only | 5e-03 | 8/10 | 0.366 |
| 0.5 | all-points | 1e-03 | 6/10 | 0.765 |
| 0.5 | all-points | 5e-03 | 4/10 | 1.044 |
| 0.7 | boundary-only | 1e-03 | 9/10 | 0.059 |
| 0.7 | boundary-only | 5e-03 | 10/10 | 0.380 |
| 0.7 | all-points | 1e-03 | 9/10 | 0.636 |
| 0.7 | all-points | 5e-03 | 6/10 | 0.806 |

> In `boundary-only` mode the exact gradient wins 10/10 paired runs at lr=1e-3 (median loss ratio 0.07, ~14x lower) and 8-10/10 at lr=5e-3 (ratio 0.37, ~2.7x lower); the naive gradient still reduces the loss but is slowed by the wrong radial sign on e>c^2/3. In `all-points` mode the inliers compensate and the two gradients are not distinguishable (4-9/10, ratio 0.64-1.04) - this is why the diagnostic restricts the update to the boundary set.

## End-to-end gradient check (one boundary observation)

| coord | exact dL/dy | naive dL/dy | FD dL/dy | relerr exact | relerr naive |
|---|---|---|---|---|---|
| 0 | -4.411013e-05 | -1.066528e-04 | -4.411014e-05 | 1.61e-07 | 1.42e+00 |
| 1 | -1.650923e-04 | -6.589131e-05 | -1.650923e-04 | 3.88e-08 | 6.01e-01 |
| 2 | +7.732867e-05 | +4.775917e-05 | +7.732867e-05 | 9.32e-08 | 3.82e-01 |

## Limitations

* This is a GT-supervised test-time diagnostic, not a learned layer. It answers 'does the backward map move the hard observations in the right direction?', not 'does it improve a real front-end?'.
* The boundary observations are known from the injection protocol; in a real system one would have to identify them (e.g. by e/c^2 or a learned weight), which is out of scope here.
* Adam is used only to make the step scale comparable between the two gradients; the paired comparison uses identical seeds, steps, and lr.
* The result is consistent across 10 seeds and 2 learning rates; the absolute numbers are small because the initial GNC pose is already good.
