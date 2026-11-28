# Theseus LM baseline on the P1 real two-mode bearing cost (BOP LINEMOD)

Cost: `C(R,t) = sum_i ||P_i^perp (R p_i + t)||^2`, `P_i^perp = I - u_i u_i^T`; same 15 frames, same real depth-pixel bearings and nearest-mesh-vertex model points, same RANSAC planar-face inliers as `run_two_mode.py` PART A.

Theseus 0.2.3, torch 2.11.0+cu128, float64 CPU, `LevenbergMarquardt` + `CholeskyDenseSolver`, max 100 iters, abs/rel tol 1e-14/1e-10. Objective: single 3N-dim residual block (cross-checked against per-point costs); 20 random inits per frame (rotation uniform, t ~ U(-1,1)^3), plus GT and identity inits. Timings are login-node CPU, objective construction excluded from solve time.

## 1. Objective equality and analytic Jacobian

Theseus objective value vs the numpy reference at GT, identity and 5 random poses per frame (21 poses x 2 objective constructions):

| construction | max abs cost diff | max rel diff |
|-|-|-|
| single 3N residual (used below) | 1.137e-13 | 2.056e-16 |
| one cost per correspondence (P3 pattern) | 1.137e-13 | 3.306e-16 |

Central-difference check of the analytic 3N x 6 Jacobian (right perturbation `T <- T @ Exp(delta)`, `delta=(v,omega)`):

| frame/img | jac rel err |
|-|-|
| 000010/15 | 6.92e-11 |
| 000010/18 | 8.03e-11 |
| 000010/40 | 1.27e-10 |

Unit-bearing deviation `max||u_i||-1| = 2.22e-16` (so `I - u u^T` is P1's `P^perp` to machine precision).

## 2. Mode labels used for the GT-vs-flip classification (from `two_mode.json`)

The mode poses (`k0` = P1's global argmin among physical modes, `k1` = the other physical mode nearest the GT rotation) are re-derived with the same deterministic construction only to *label* convergence outcomes; the hard-argmin / Laplace numbers are cited in section 6, not recomputed.

| frame/img | n_plane | K_phys | Delta J | flip deg (recomputed) | flip deg (two_mode.md) |
|-|-|-|-|-|-|
| 000010/15 | 558 | 2 | 9.58e-04 | 69.28 | 69.28 |
| 000010/18 | 474 | 2 | 1.25e-03 | 91.90 | 91.90 |
| 000010/40 | 482 | 2 | 9.71e-04 | 69.08 | 69.08 |
| 000010/60 | 452 | 2 | 1.69e-03 | 104.05 | 104.05 |
| 000010/66 | 460 | 2 | 7.00e-04 | 88.39 | 88.39 |
| 000010/72 | 501 | 2 | 1.25e-04 | 3.84 | 3.84 |
| 000010/80 | 508 | 2 | 5.11e-04 | 54.71 | 54.71 |
| 000010/90 | 474 | 2 | 4.13e-04 | 54.80 | 54.80 |
| 000011/20 | 413 | 2 | 2.07e-03 | 69.14 | 69.14 |
| 000011/24 | 423 | 2 | 1.65e-03 | 131.05 | 131.05 |
| 000011/26 | 439 | 2 | 1.90e-03 | 126.87 | 126.87 |
| 000011/32 | 474 | 2 | 2.10e-03 | 123.91 | 123.91 |
| 000011/33 | 365 | 2 | 1.49e-03 | 149.58 | 149.58 |
| 000011/40 | 428 | 2 | 2.56e-03 | 106.01 | 106.01 |
| 000011/49 | 347 | 2 | 2.10e-03 | 83.71 | 83.71 |

## 3. Initialisation at GT (local convergence check)

| frame/img | rot err (deg) | trans err (m) | final cost | iters | status | nearest P1 mode | frac z>0.05 |
|-|-|-|-|-|-|-|-|
| 000010/15 | 1.47 | 0.0057 | 7.522e-04 | 4 | 1 | k0 | 1.000 |
| 000010/18 | 0.51 | 0.0086 | 9.062e-04 | 4 | 1 | k0 | 1.000 |
| 000010/40 | 2.20 | 0.0122 | 8.710e-04 | 4 | 1 | k0 | 1.000 |
| 000010/60 | 0.96 | 0.0059 | 6.802e-04 | 4 | 1 | k0 | 1.000 |
| 000010/66 | 2.29 | 0.0049 | 1.237e-03 | 4 | 1 | k0 | 1.000 |
| 000010/72 | 3.84 | 0.0135 | 1.138e-03 | 5 | 1 | k1 | 1.000 |
| 000010/80 | 1.08 | 0.0071 | 6.433e-04 | 4 | 1 | k0 | 1.000 |
| 000010/90 | 3.49 | 0.0056 | 9.062e-04 | 5 | 1 | k0 | 1.000 |
| 000011/20 | 8.14 | 0.0442 | 9.482e-04 | 8 | 1 | k0 | 1.000 |
| 000011/24 | 1.37 | 0.0046 | 1.600e-03 | 4 | 1 | k0 | 1.000 |
| 000011/26 | 2.06 | 0.0062 | 1.401e-03 | 4 | 1 | k0 | 1.000 |
| 000011/32 | 1.78 | 0.0055 | 9.685e-04 | 4 | 1 | k0 | 1.000 |
| 000011/33 | 1.62 | 0.0111 | 1.852e-03 | 4 | 1 | k0 | 1.000 |
| 000011/40 | 2.77 | 0.0039 | 1.636e-03 | 6 | 1 | k0 | 1.000 |
| 000011/49 | 4.95 | 0.0606 | 9.405e-04 | 7 | 1 | k0 | 1.000 |

- median rot err 2.06 deg, max 8.14 deg; median trans err 0.0062 m; median iters 4; gt-rate 15/15.

## 4. Initialisation at identity (t=0, R=I)

| frame/img | rot err (deg) | trans err (m) | final cost | iters | outcome | nearest P1 mode | frac z>0.05 |
|-|-|-|-|-|-|-|-|
| 000010/15 | 179.97 | 1.7668 | 8.157e-04 | 12 | other | k1 | 0.000 |
| 000010/18 | 91.90 | 0.0447 | 2.157e-03 | 10 | flip | k1 | 1.000 |
| 000010/40 | 69.08 | 0.0364 | 1.842e-03 | 10 | flip | k1 | 1.000 |
| 000010/60 | 104.05 | 0.0493 | 2.367e-03 | 11 | flip | k1 | 1.000 |
| 000010/66 | 2.29 | 0.0049 | 1.237e-03 | 7 | gt | k0 | 1.000 |
| 000010/72 | 178.82 | 2.0640 | 1.192e-03 | 9 | other | k0 | 0.000 |
| 000010/80 | 180.00 | 1.7744 | 7.281e-04 | 11 | other | k1 | 0.000 |
| 000010/90 | 179.84 | 2.0630 | 1.070e-03 | 10 | other | k0 | 0.000 |
| 000011/20 | 69.14 | 0.0997 | 3.014e-03 | 17 | flip | k1 | 1.000 |
| 000011/24 | 179.94 | 1.5422 | 2.541e-03 | 6 | other | k1 | 0.000 |
| 000011/26 | 179.81 | 1.3748 | 2.158e-03 | 6 | other | k1 | 0.000 |
| 000011/32 | 179.82 | 1.3623 | 2.201e-03 | 7 | other | k1 | 0.000 |
| 000011/33 | 179.75 | 2.0452 | 2.573e-03 | 6 | other | k1 | 0.000 |
| 000011/40 | 106.01 | 0.0385 | 4.198e-03 | 7 | flip | k1 | 1.000 |
| 000011/49 | 83.71 | 0.0964 | 3.039e-03 | 12 | flip | k1 | 1.000 |

- gt-rate 1/15, flip 6, other 8, non-physical 8; median rot err 178.82 deg.

## 5. Random initialisation (20 trials per frame)

| frame/img | GT hit | flip | other | non-physical | median rot err (deg) | min rot err (deg) |
|-|-|-|-|-|-|-|
| 000010/15 | 6/20 | 4 | 10 | 10 | 124.39 | 1.47 |
| 000010/18 | 7/20 | 4 | 9 | 9 | 91.90 | 0.51 |
| 000010/40 | 4/20 | 1 | 15 | 15 | 178.69 | 2.20 |
| 000010/60 | 10/20 | 2 | 8 | 8 | 52.50 | 0.96 |
| 000010/66 | 5/20 | 8 | 7 | 7 | 88.39 | 2.29 |
| 000010/72 | 5/20 | 0 | 15 | 6 | 44.46 | 3.84 |
| 000010/80 | 5/20 | 6 | 9 | 9 | 54.71 | 1.08 |
| 000010/90 | 10/20 | 2 | 8 | 8 | 29.14 | 3.49 |
| 000011/20 | 7/20 | 4 | 9 | 9 | 69.14 | 8.14 |
| 000011/24 | 8/20 | 5 | 7 | 7 | 131.05 | 1.37 |
| 000011/26 | 7/20 | 3 | 10 | 10 | 152.25 | 2.06 |
| 000011/32 | 5/20 | 6 | 9 | 9 | 123.91 | 1.78 |
| 000011/33 | 8/20 | 5 | 7 | 7 | 149.58 | 1.62 |
| 000011/40 | 7/20 | 5 | 8 | 8 | 106.01 | 2.77 |
| 000011/49 | 6/20 | 2 | 12 | 12 | 178.42 | 4.95 |

- all 300 trials: gt-rate 100/300 (33.3%), flip 57, other 143, non-physical 134 (frac z>0.05 < 0.99); median rot err 123.91 deg; median time 12 ms.

## 6. Random-init multistart on representative frames (8 inits each)

| frame/img | GT hits | flip | other | non-physical | rot errs (deg) |
|-|-|-|-|-|-|
| 000010/15 | 3/8 | 0 | 5 | 5 | 1.5, 179.5, 179.5, 1.5, 1.5, 180.0, 180.0, 180.0 |
| 000010/72 | 0/8 | 0 | 8 | 3 | 44.5, 44.5, 178.8, 44.5, 44.5, 44.5, 178.8, 179.8 |
| 000010/60 | 2/8 | 2 | 4 | 4 | 104.0, 179.5, 179.9, 104.0, 179.5, 1.0, 1.0, 179.5 |
| 000011/20 | 3/8 | 1 | 4 | 4 | 175.5, 180.0, 8.1, 69.1, 180.0, 8.1, 175.5, 8.1 |
| 000011/49 | 3/8 | 2 | 3 | 3 | 4.9, 4.9, 83.7, 178.4, 4.9, 83.7, 179.1, 179.1 |

- aggregate multistart success rate (rot err < 10 deg): 11/40 = 27.5%; non-physical solutions 19/40.

## 7. Comparison with P1's hard-argmin / Laplace numbers (cited, not recomputed)

From `labs/01-laplace-qpeplayer/experiments/real/linemod/results/two_mode.md` (same 15 frames, same cost): median hard-argmin rotation error 2.06 deg, median Laplace posterior-mean rotation error 58.28 deg, hard argmin >10 deg in 1/15 frames, K=2 physical modes 15/15. (Medians recomputed from `two_mode.json` agree to 4.84e-03 deg.)

| method | what it returns | median rot err (deg) | gt-rate |
|-|-|-|-|
| hard argmin (P1, cited) | global minimum among physical modes | 2.06 | 14/15 |
| Laplace mean (P1, cited) | posterior mean over physical modes | 58.28 | 0/15 |
| Theseus LM from GT (this work) | one local solution | 2.06 | 15/15 |
| Theseus LM, identity init (this work) | one local solution | 178.82 | 1/15 |
| Theseus LM, random init (this work) | one local solution | 123.91 | 100/300 |
| Theseus LM multistart, 8 inits (this work) | best-of-8 local solutions | 4.95 | 11/40 |

**Honest framing.** Theseus is a *local* optimiser: each call returns one pose and has no notion of the enumerated mode set or of the z>0 physicality filter that P1 applies. From GT it recovers the nearest minimum (local convergence, not a baseline win). From a cold/random init it recovers whichever basin the descent enters -- on this near-tied cost that is often the P1 flip mode or a non-physical (behind-camera) pose, which the layer's mode enumeration rejects by construction. Theseus also returns no posterior, so it is not comparable to the Laplace mean as an estimator; it is the natural *no-enumeration* reference point. A negative or mixed result is reported as-is: local optimisation does not replace mode enumeration on this cost. Two concrete observations: (i) on 000010/72, the frame where P1's *global* argmin is 44 deg from GT, the GT-init solve lands on the 3.8 deg mode (P1's k1 candidate) -- a local minimum that is closer to GT than the global argmin, so the GT-init row is not simply 'hard argmin'; (ii) the identity-init solve usually ends at a rotation ~70-180 deg from GT, i.e. a non-physical (behind-camera) stationary point of the unconstrained cost.

## 8. Caveats

- The Theseus problem is the *full* (R,t) cost; P1's modes are minima of the reduced cost after eliminating t analytically. For fixed R, t*(R) is the linear least-squares optimum, so the mode sets coincide; the residual is not zero because the nearest-mesh-vertex correspondence is a discretisation (it is the same residual P1 optimises).
- The cost has no z>0 constraint, so behind-camera poses are legitimate stationary points of the Theseus objective. Rows with `frac z>0.05 < 0.99` are non-physical and correspond to poses the P1 layer discards.
- Random inits are deliberately hard (uniform rotation, t ~ U(-1,1)^3, so most starts are far outside the object's 0.6-1 m depth range). Their success rate measures basin capture from cold start, not detector accuracy.
- 'GT pose' here means the BOP ground truth; the cost's own global mode is 1.5-8 deg from it on most frames because of the nearest-vertex discretisation, so a GT-init solve cannot be more accurate than that mode.
- Solve times exclude objective construction (the 3N-residual objective is built once per frame and the pose variable is reset for each init); they are single-threaded CPU numbers on the login node.
- This is an external baseline on P1's *cost*, not a claim about Theseus' performance on published pipelines, and it does not include a posterior.

