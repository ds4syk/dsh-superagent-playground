# Pose error / ADD-S proxy on the real two-mode LINEMOD frames

**Question.** `run_two_mode.py` Part A showed that 15/15 real eggbox/glue frames have K=2 physical minima and that the Laplace posterior mean is pulled toward the flip. This report adds the missing pose-error / ADD-S proxy with per-frame and aggregate 95% CIs, comparing hard argmin, Laplace and mode-level MC.

## Headline

- The data pipeline is reproduced exactly: 15 frames, same per-frame `dJ` / weights / hard rotation error and the same 9-vector Laplace mean as `results/two_mode.json` (max abs diff 2.1e-14).
- **Hard argmin** rotation error median 2.06 deg [1.47, 3.49], ADD-S median 3.7 mm [3.0, 4.3]. It wins on accuracy, as expected: the GT-like pose is the lowest-J physical mode in 14/15 frames.
- **Laplace** posterior mean: rotation error median 44.24 deg [33.90, 46.87], ADD-S median 13.9 mm. It is biased toward the flip mode (mean paired bias vs hard: +36.9 deg, 95% CI [+27.2, +44.9]); this is the price of a deterministic, smooth (single-valued) estimate at a near-tie.
- **The earlier 58.3 deg is not a valid geodesic error.** Part A applied the geodesic formula to the *unprojected* 9-vector mean, which is not a rotation; that quantity equals arccos(sum_k w_k cos err_k) (exactly reproduced here, median 58.28 deg). On the valid chordal mean the median is 44.24 deg. The direction of the bias is unchanged; its magnitude was overstated.
- **MC** reproduces the same estimand as Laplace (mean over 20 seed means vs Laplace aggregate: 0.23 deg, within its seed-level CI) and therefore inherits the same bias; it adds seed noise (between-seed SD of the aggregate 1.011 deg).
- **1 frame(s) where hard argmin is not the GT-like mode:** `000010/72` (hard 44.5 deg, ADD-S 13 mm; GT-like mode 3.8 deg away but higher J by 1.3e-04). There Laplace/MC land between the two modes and happen to be closer than hard; that is a single-frame accident of the loss ranking, not evidence that the posterior mean is more accurate in general.

**This is a bias/smoothness trade, not an accuracy win.** No claim is made that Laplace or MC beats a detector; the GT is the global minimum in most frames, so hard argmin is expected to be the most accurate estimator here.

## 0. Metric definition

Pose of a mode $k$: $R_k = R(q_k)$, $t_k = -G^{-1}D\,v_2(q_k)$ (the translation eliminated on the RANSAC plane inlier set). Physical modes are the minima of the reduced cost with all transformed points at $z>0.05$ m; only they enter $w$, the hard argmin and MC. Weights $w_k \propto \exp(-J_k/\tau - \tfrac12 \log\det H_k)$ with $\tau=0.01$.

- **hard** = mode with minimal $J_k$.
- **Laplace** = $(\mathrm{Proj}_{SO(3)}(\sum_k w_k R_k),\ \sum_k w_k t_k)$: the weighted chordal mean of the rotations, projected back to SO(3) (SVD polar projection), and the weighted mean translation. The projection is required for a defensible pose metric: $\sum_k w_k R_k$ is not orthogonal, so a rotation error on it is not a geodesic distance and ADD-S is not defined. For reference, the unprojected 9-vector mean gives median 58.28 deg against 44.24 deg for the projected mean; the unprojected value is exactly $\arccos(\sum_k w_k\cos e_k)$ (verified to 1e-14), which overstates the geodesic error and is the source of the 58.3 deg in Part A.
- **MC** = sample $n=400$ modes i.i.d. from $w$, average the sampled poses (same chordal-mean/projection rule), repeat for 20 seeds. Per-frame MC value = mean over the 20 seeds; the 95% CI is a percentile bootstrap over those 20 seed values.

**Rotation error** $=\arccos((\mathrm{tr}(R_\mathrm{est}^\top R_\mathrm{gt})-1)/2)$ in degrees. **Translation error** $=\|t_\mathrm{est}-t_\mathrm{gt}\|$ in m (BOP `cam_t_m2c` is in mm and converted to m).

**ADD-S** (mm): draw a fixed random subsample $V$ of 1000 mesh vertices (3 independent subsamples are drawn; the reported value is their mean), transform by the estimated pose, and average the distance to the nearest vertex of the GT-transformed *full* mesh $M$ (KD-tree over all 18 473 eggbox / 7 479 glue vertices), i.e. the standard $\mathrm{mean}_x\min_y$ ADD-S with the query side Monte-Carlo'd:
$\;\mathrm{ADD\text{-}S}=\frac{1}{|V|}\sum_{x\in V}\min_{y\in M}\|R_\mathrm{est}x+t_\mathrm{est}-(R_\mathrm{gt}y+t_\mathrm{gt})\|$.
The dense GT set is the defensible choice: with a 1000-vertex GT set as well, the same-subsample variant carries a +1.1 mm discretisation floor on average (hard pose) and overstates the error; the 1000-query/dense-GT estimator agrees with the full-mesh-vs-full-mesh ADD-S to 0.51 mm (max over frames and over the hard and Laplace poses).
**ADD** (reference only): the same mean with the same-index distance $\|(R_\mathrm{est}-R_\mathrm{gt})x+(t_\mathrm{est}-t_\mathrm{gt})\|$. **ADD-S is the appropriate metric**: both eggbox (obj 10) and glue (obj 11) have a discrete symmetry of ~180 deg in `models_info.json`, so a symmetry-equivalent pose is the same physical pose and ADD would report an error of order the object diameter (0.165 m / 0.176 m). ADD-S is invariant to that relabelling; ADD is kept only to show the size of the failure mode.

## 1. Per-frame table

Rotation error (deg) and translation error (mm). `dJ` and `w_other` are exactly the Part A quantities (`w_other` = weight of the physical mode that is not the hard argmin; it is the GT-like mode in the one flagged frame). `w_gtlike` is the weight of the physical mode closest to GT.

| frame/img | obj | dJ | w_other | w_gtlike | GT-like=hard | rot hard | rot lap | rot MC | trans hard | trans lap | trans MC |
|-|-|-|-|-|-|-|-|-|-|-|-|
| 000010/15 | 10 | 9.58e-04 | 0.483 | 0.517 | yes | 1.47 | 33.90 | 33.90 | 5.7 | 17.6 | 17.6 |
| 000010/18 | 10 | 1.25e-03 | 0.472 | 0.528 | yes | 0.51 | 42.85 | 43.41 | 8.6 | 20.9 | 21.1 |
| 000010/40 | 10 | 9.71e-04 | 0.482 | 0.518 | yes | 2.20 | 33.07 | 33.03 | 12.2 | 19.4 | 19.4 |
| 000010/60 | 10 | 1.69e-03 | 0.459 | 0.541 | yes | 0.96 | 46.45 | 47.42 | 5.9 | 21.8 | 22.1 |
| 000010/66 | 10 | 7.00e-04 | 0.488 | 0.512 | yes | 2.29 | 43.67 | 43.46 | 4.9 | 20.2 | 20.1 |
| 000010/72 | 10 | 1.25e-04 | 0.504 | 0.504 | no **X** | 44.46 | 23.02 | 23.12 | 24.8 | 16.0 | 16.0 |
| 000010/80 | 10 | 5.11e-04 | 0.501 | 0.499 | yes | 1.08 | 27.71 | 28.15 | 7.1 | 15.4 | 15.6 |
| 000010/90 | 10 | 4.13e-04 | 0.497 | 0.503 | yes | 3.49 | 27.91 | 27.65 | 5.6 | 13.2 | 13.1 |
| 000011/20 | 11 | 2.07e-03 | 0.608 | 0.392 | yes | 8.14 | 44.24 | 43.80 | 44.2 | 77.4 | 77.1 |
| 000011/24 | 11 | 1.65e-03 | 0.457 | 0.543 | yes | 1.37 | 54.99 | 54.65 | 4.6 | 22.3 | 22.2 |
| 000011/26 | 11 | 1.90e-03 | 0.450 | 0.550 | yes | 2.06 | 51.65 | 51.29 | 6.2 | 23.1 | 23.0 |
| 000011/32 | 11 | 2.10e-03 | 0.451 | 0.549 | yes | 1.78 | 51.60 | 51.79 | 5.5 | 28.6 | 28.7 |
| 000011/33 | 11 | 1.49e-03 | 0.460 | 0.540 | yes | 1.62 | 59.09 | 61.50 | 11.1 | 12.6 | 12.8 |
| 000011/40 | 11 | 2.56e-03 | 0.457 | 0.543 | yes | 2.77 | 46.19 | 46.01 | 3.9 | 16.2 | 16.1 |
| 000011/49 | 11 | 2.10e-03 | 0.528 | 0.472 | yes | 4.95 | 46.75 | 47.43 | 60.6 | 78.5 | 78.8 |

ADD-S (mm) and ADD (mm). ADD-S is the primary object-space metric; ADD is shown for reference only.

| frame/img | obj | ADD-S hard | ADD-S lap | ADD-S MC | ADD hard | ADD lap | ADD MC |
|-|-|-|-|-|-|-|-|
| 000010/15 | 10 | 3.0 | 10.3 | 10.3 | 5.7 | 31.0 | 30.9 |
| 000010/18 | 10 | 4.3 | 12.2 | 12.3 | 8.7 | 37.6 | 38.0 |
| 000010/40 | 10 | 6.2 | 9.7 | 9.7 | 12.2 | 31.1 | 31.0 |
| 000010/60 | 10 | 3.0 | 13.9 | 14.1 | 5.9 | 40.7 | 41.4 |
| 000010/66 | 10 | 2.7 | 12.1 | 12.0 | 5.0 | 38.0 | 37.8 |
| 000010/72 | 10 | 12.5 | 8.4 | 8.4 | 39.7 | 22.3 | 22.3 |
| 000010/80 | 10 | 3.8 | 8.8 | 8.9 | 7.1 | 24.1 | 24.5 |
| 000010/90 | 10 | 3.1 | 8.5 | 8.4 | 5.8 | 23.3 | 23.1 |
| 000011/20 | 11 | 13.8 | 22.9 | 22.8 | 43.6 | 72.7 | 72.4 |
| 000011/24 | 11 | 2.8 | 20.2 | 20.0 | 4.2 | 40.5 | 40.3 |
| 000011/26 | 11 | 3.7 | 17.6 | 17.5 | 5.9 | 38.1 | 37.9 |
| 000011/32 | 11 | 3.1 | 19.0 | 19.0 | 5.2 | 41.5 | 41.5 |
| 000011/33 | 11 | 3.8 | 23.4 | 23.7 | 11.1 | 41.5 | 42.7 |
| 000011/40 | 11 | 2.2 | 16.8 | 16.7 | 4.3 | 44.7 | 44.5 |
| 000011/49 | 11 | 22.4 | 28.1 | 28.2 | 60.0 | 75.9 | 76.1 |

Subsample check: the three independent 1000-vertex query subsamples differ by at most 1.87 mm in any frame (Laplace pose). The 1000-vertex *GT* target set instead of the full mesh would add a discretisation floor of +1.1 mm on average (hard pose), so the full GT mesh is used as the target and the 1000-query estimator matches the full-mesh-vs-full-mesh value to 0.51 mm (hard and Laplace poses, max over frames).

## 2. Aggregate mean / median with bootstrap 95% CIs (1000 resamples over frames)

| metric | method | mean [95% CI] | median [95% CI] |
|-|-|-|-|
| rotation error (deg) | hard | 5.28 [1.82, 11.48] | 2.06 [1.47, 3.49] |
| rotation error (deg) | laplace | 42.21 [36.55, 46.94] | 44.24 [33.90, 46.87] |
| rotation error (deg) | mc | 42.44 [37.24, 47.85] | 43.80 [33.90, 47.43] |
| translation error (mm) | hard | 14.1 [6.9, 22.5] | 6.2 [5.5, 11.1] |
| translation error (mm) | laplace | 26.9 [18.4, 38.6] | 20.2 [16.2, 23.1] |
| translation error (mm) | mc | 26.9 [18.4, 38.5] | 20.1 [16.1, 22.2] |
| ADD-S (mm) | hard | 6.0 [3.6, 9.1] | 3.7 [3.0, 4.3] |
| ADD-S (mm) | laplace | 15.5 [12.5, 18.6] | 13.9 [10.3, 20.2] |
| ADD-S (mm) | mc | 15.5 [12.5, 18.4] | 14.1 [10.3, 20.0] |
| ADD (mm) | hard | 15.0 [7.1, 24.1] | 5.9 [5.2, 11.1] |
| ADD (mm) | laplace | 40.2 [33.3, 48.2] | 38.1 [31.0, 41.5] |
| ADD (mm) | mc | 40.3 [33.3, 48.0] | 38.0 [31.0, 41.5] |

The CIs are wide because there are only 15 frames; they describe this frame population, not a detector's accuracy on LINEMOD.

By object (median ADD-S, hard / Laplace): eggbox 3.5 / 10.0 mm, glue 3.7 / 20.2 mm. ADD is much larger than ADD-S (Laplace median 38.1 vs 13.9 mm; hard mean 15.0 vs 6.0 mm): the glue is long and thin (0.176 m), so same-index distances amplify even a small rotation error, and the ~180 deg discrete symmetry of both objects makes ADD invalid for a symmetry-equivalent pose.

## 3. MC seed-level CI

For each seed $s$, average that seed's per-frame metric over all frames (this is the "mean over the 20 seed means" reported by some tools); the seed-level CI is a percentile bootstrap over the 20 per-seed aggregates, so it isolates seed noise. The combined CI bootstraps frames and seeds together.

| metric | mean over 20 seed means | frame-bootstrap 95% CI | seed-level 95% CI | combined (frame+seed) 95% CI | between-seed SD | max per-frame seed CI half-width |
|-|-|-|-|-|-|-|
| rotation error (deg) | 42.44 | [37.24, 47.85] | [42.04, 42.88] | [37.20, 48.30] | 1.01 | 4.96 |
| translation error (mm) | 26.9 | [18.4, 38.5] | [26.8, 27.0] | [18.2, 37.0] | 0.2 | 0.6 |
| ADD-S (mm) | 15.5 | [12.5, 18.4] | [15.4, 15.6] | [12.6, 18.7] | 0.3 | 1.4 |
| ADD (mm) | 40.3 | [33.3, 48.0] | [40.1, 40.6] | [33.2, 47.9] | 0.6 | 2.8 |

## 4. Interpretation and caveats

1. **Hard argmin is expected to win — by geometry and by construction.** The correspondences are associated with the model *through the GT pose* (each depth point is mapped into the model frame with $R_\mathrm{gt},t_\mathrm{gt}$ and matched to the nearest mesh vertex), so the GT-like pose is favoured by the data construction; the flip mode comes from the planar two-fold ambiguity of the same points. In addition, the GT-like pose is the lowest-J physical mode in 14/15 frames. The Laplace/MC estimates are means over the two wells: they sit between the GT-like pose and the flip, with median rotation error 44.2 deg (flip modes at median 88.4 deg) and median ADD-S 13.9 mm vs 3.7 mm for hard. This is the expected behaviour of a posterior mean at a near-tie, not a defect of the estimator, and the numbers are not an accuracy benchmark of the estimator.
2. **The trade is bias versus smoothness.** Hard argmin is a discontinuous function of the data: a small perturbation of the cost can switch the selected mode. The Laplace mean is single-valued and differentiable (that is the property the P1 layer theory needs); the price is the bias quantified above. MC estimates the same smooth mean and adds seed noise, so it is not a fix for the bias.
3. **Frame `000010/72` is the honest counter-example.** The GT-like mode is 3.8 deg from GT but has *higher* J (by 1.25e-4) than a mode 44.5 deg away, so hard argmin fails; Laplace/MC land between the modes and are closer. In the other 14 frames hard argmin is the GT-like mode and is more accurate. One frame does not reverse the population conclusion, and the aggregate CIs are wide enough to show that.
4. **ADD-S, not ADD.** Both objects have a discrete ~180 deg symmetry; ADD would count a symmetry-equivalent pose as a diameter-scale error. ADD is listed only for reference. The relevant quantity for a symmetric object is ADD-S.
5. **The chordal mean degrades as the flip approaches 180 deg.** The extrinsic mean of two rotations 180 deg apart is not unique (the SVD projection is ill-conditioned). The largest flip here is 149.6 deg (`000011/33`), and its per-frame MC seed CI is the widest (5.0 deg half-width) because the mean's sensitivity to the mode weights grows with the flip angle. All reported frames are below 180 deg, but this is a limitation of any single SO(3) representative at a near-tie.
6. **Caveats.** (i) This is a pose-error proxy for a controlled two-mode bearing cost, not a detector evaluation: no image appearance, no front-end, no training. (ii) The translation is the one eliminated on the RANSAC plane inlier set, so the translation error also reflects the plane/mesh discretisation, not only the pose; for the long glue object the hard translation error reaches 44-61 mm on two frames while ADD-S stays at 14-23 mm. (iii) ADD-S queries a uniform random subsample of mesh *vertices* against the full GT mesh; the vertex distribution is not surface-area weighted, and the query-subsample spread is <= 1.87 mm per frame. (iv) 15 frames give wide CIs; the eggbox and glue groups are not separately powered, and the 8/7 frames are not reweighted. (v) The absolute error values are limited by the real depth noise and the nearest-vertex model quantisation (the GT-like mode itself has ~0.5-8 deg rotation error). (vi) Because the correspondences are built with the GT pose (item 1), the GT-like mode is the global minimum by construction as well as by measurement: the experiment quantifies bias/smoothness, it cannot establish that hard argmin would win on a detector's real correspondences.

## Repro

```bash
cd labs/01-laplace-qpeplayer/experiments/real/linemod
/home/syk/miniconda3/envs/WM/bin/python run_pose_error.py
```

Deterministic: `Instance(seed=20261127)`, `n_starts=384`, MC seeds `20261127+1000*i+s`, vertex subsamples `default_rng(20261127)`, bootstrap seeds fixed. Wall time ~10 s on the login-node CPU. Files: `results/pose_error.{json,md,png}`.

The JSON contains per-frame seeds, J/logdetH, weights and the reproduction check against `results/two_mode.json`.
