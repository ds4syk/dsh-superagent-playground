# Structured front-end noise on the 15 real LINEMOD frames: boundary-biased, correlated, heteroscedastic

> **What this is.** The same 15 real eggbox/glue frames, the same fixed clean RANSAC planar-face subset, the same reduced bearing cost and the same hard/Laplace/MC layer as `run_detector_noise.py`, driven by three **structurally realistic** front-end error models instead of iid Gaussian pixels: (a) mask-boundary-biased outliers (silhouette pixels, where depth discontinuities live), (b) a spatially correlated per-frame pixel offset plus per-point jitter, (c) heteroscedastic noise that is larger near the mask boundary. Every structured arm is compared against an **equal-variance iid twin** with paired draws, so the question is not 'more noise?' but 'the same variance, arranged differently?' **What this is not.** Not a trained detector, not calibrated to a real detector's error distribution, and the correspondences are GT-associated, so absolute errors are optimistic.

## Headline

1. **Boundary-biased outliers are less wrong per outlier but still break the layer - through a different mechanism.** At f=0.10 the boundary arm gives hard rotation 71.28 [43.64, 88.23] deg and ADD-S 447.6 [363.9, 459.9] mm versus 165.07 [133.32, 169.13] deg / 890.6 [796.1, 965.9] mm for uniform-random pixels (paired difference -82.22 [-94.87, -69.42] deg). The mechanism differs: a uniform random ray is 21.1 deg / 326 mm off the true ray and no physical mode survives (100% fallback), while a silhouette ray is *grazing* - 3.8 deg / 62 mm (inliers 2.2 mm) - so the modes stay physical (2% fallback) and the cost's **own minimum is biased**: the grazing rays form a coherent ring around the silhouette, the hard argmin is the GT-like mode on only 60% of frames, and on the eggbox frames the strongest non-hard mode is also 40-70 deg from GT - so the layer returns a *confidently wrong* pose instead of an obvious no-mode fallback (100% of draws exceed 10 deg). Either way the L2 cost has no margin: at f=0.05 both arms already fail on 93% / 100% of draws. **The iid-outlier breakdown conclusion survives for a structurally realistic silhouette outlier, with a different failure mode.**
2. **Spatially correlated noise is NOT worse than iid noise at equal total variance - the iid sigma conclusion survives.** A 1 px per-frame offset plus 0.5 px per-point jitter (sigma_eq=1.12 px) gives hard rotation 2.03 [1.32, 3.30] deg versus 2.00 [1.21, 4.21] deg for iid with exactly that variance: paired difference -0.29 [-0.69, +0.01] deg (CI includes 0). At 2 px + 1 px (sigma_eq=2.24 px) it is 2.04 [1.27, 4.37] vs 3.11 [2.03, 12.22] deg, paired -3.49 [-5.48, -1.61] deg - the correlated arm is *better*, not worse, because iid jitter of the same total variance adds independent scatter along the plane's soft direction while a common offset is largely absorbed by the pose. The one coherent-offset penalty is on the **smooth** estimators: Laplace +0.44 [+0.14, +0.79] deg and MC +0.59 [+0.02, +1.13] deg at sigma_eq=1.12 (+0.82 [+0.33, +1.38] / +0.99 [+0.36, +1.74] at 2.24), because a global shift biases every mode coherently whereas iid jitter is partly averaged out by the posterior mean. So 'sigma <= 1 px is harmless' holds as a **variance** statement, and correlation adds no hard-estimator penalty at these amplitudes (1-2 px).
3. **Heteroscedastic boundary noise IS worse than homoscedastic noise at equal RMS, but in the tail, not in the median.** Profile h1 (0.5->4 px, lambda=6 px, realised RMS 1.61 px, 21% of points above 2 px) gives hard rotation 3.45 [1.57, 13.96] deg vs 2.23 [1.42, 3.59] deg for the equal-RMS iid twin (paired mean +2.76 [+0.07, +6.93] deg, median +0.15); profile h2 (1->5 px, lambda=10 px, RMS 2.69 px) gives 13.32 [2.69, 25.52] vs 4.57 [2.82, 18.46] deg (paired mean +10.90 [+1.30, +23.70] deg, median +0.15; hard failure 38% vs 26%). The typical frame is unchanged and a few frames whose plane subset sits near the silhouette get tipped over the near-tie: boundary-weighted variance raises the **failure tail** without shifting the median. That is the opposite pattern from the outlier arms (which move the median) and consistent with 'variance matters; the arrangement matters through leverage on the near-tie'.
4. **The dense-baseline ordering hard < Laplace/MC is preserved under every structured model** (as under iid noise): Laplace stays ~the half-flip compromise (44.2 deg clean, 51.0 deg boundary f=0.10, 43.7 deg correlated), so none of these error structures creates a regime in which the smooth estimator becomes the accurate one.

**Framing.** Structured *simulated* front-end errors on GT-associated correspondences. Absolute errors are optimistic; the contribution is the equal-variance comparisons, and two of them refute the naive expectation (correlation is not worse; heteroscedasticity is worse only in the tail).

## 0. Protocol

- **Frames / subset.** The 15 frames of `run_two_mode.py` Part A, each with the **fixed clean RANSAC plane subset** (identical to `run_detector_noise.py`, asserted). RANSAC is never re-run on perturbed data; only the 2-D measurement changes.
- **Layer.** `run_detector_noise.eval_layer` called directly: same reduced cost, `n_starts=384`, `tau=0.01`, physical filter `z>0.05` m, hard = lowest-J physical mode, Laplace = weighted chordal mean, MC = 400 mode samples x 8 seeds. All-minima fallback when no physical mode survives, flagged.
- **(a) Boundary-biased outliers.** A fraction f of the accepted bearings is replaced by the bearing of a pixel drawn uniformly from the **visible-mask silhouette** (3x3 morphological boundary of `mask_visib` with a valid real depth; median pool 309 px/frame, 96.2% of the boundary pixels have valid depth). The real depth at that pixel is used, so the replacement is a real ray that *grazes* the object. Control: uniform-random-pixel outliers at the same f.
- **(b) Spatially correlated noise.** Per frame: one constant pixel offset (dx,dy) ~ N(0, sigma_c^2) shared by every point (a calibration / extrinsics drift or a systematic front-end bias), plus per-point N(0, sigma_p^2) jitter. sigma_c/sigma_p = 1/0.5 px and 2/1 px. The twin is iid Gaussian with sigma_eq = sqrt(sigma_c^2 + sigma_p^2) = 1.12 / 2.24 px, drawn with the same (frame, seed) key.
- **(c) Heteroscedastic noise.** sigma_i = sigma_in + (sigma_bd - sigma_in) * exp(-(dist_to_boundary_i - 1)/lambda) with the Euclidean distance transform of the visible mask; profiles (0.5->4 px, lambda=6 px) and (1->5 px, lambda=10 px). The twin is iid with the **same per-frame RMS** sigma_i (varies by frame: 1.30-2.43 px for h1).
- **Seeds.** 6 independent noise seeds per condition, 8 MC seeds per (frame, noise seed), frame-bootstrap CIs (`n_boot=1000`). Paired differences are formed per frame (mean over the paired seeds) and bootstrapped over frames.
- **Reproduction.** The clean condition reproduces `detector_noise.json` sigma=0 exactly: **ok** (max abs diff 0.000000000 deg).

## 1. Reference iid arm (same code, this seed set)

Consistency check against `run_detector_noise.py`; the conclusions of that report are unchanged (different seed count, so the numbers differ by seed noise only).

| condition | K_phys med | fallback | hard rot | Laplace rot | MC rot | hard ADD-S | hard fail | GT-like |
|-|-|-|-|-|-|-|-|-|
| clean (no noise) | 2 | 0% | 2.06 [1.47, 3.49] | 44.24 [33.90, 51.60] | 44.40 [33.65, 51.74] | 3.7 [3.0, 4.2] | 7% | 93% |
| iid Gaussian 1 px | 2 | 0% | 1.97 [1.50, 3.32] | 43.61 [33.84, 46.80] | 43.21 [32.96, 50.46] | 2.6 [1.9, 5.0] | 7% | 93% |
| iid Gaussian 2 px | 2 | 0% | 2.55 [1.55, 4.71] | 42.35 [33.66, 46.62] | 42.59 [33.85, 49.79] | 14.0 [8.8, 21.1] | 20% | 93% |
| iid Gaussian 5 px | 2 | 0% | 23.35 [7.07, 37.05] | 41.57 [32.53, 46.62] | 41.25 [31.25, 48.11] | 161.8 [86.1, 178.4] | 64% | 93% |

## 2. Boundary-biased outliers vs uniform-random outliers

### 2.1 Pose error

| condition | f | K_phys med | fallback | hard rot | Laplace rot | MC rot | hard ADD-S | Laplace ADD-S | hard fail | GT-like |
|-|-|-|-|-|-|-|-|-|-|-|
| uniform-random outliers f=0.05 | 0.05 | 0 | 97% | 157.20 [102.55, 168.24] | 134.73 [119.95, 161.00] | 124.09 [107.64, 139.91] | 914.6 [701.1, 978.3] | 856.1 [698.0, 971.3] | 100% | 27% |
| mask-boundary-biased outliers f=0.05 | 0.05 | 2 | 0% | 64.51 [39.41, 75.17] | 48.59 [33.01, 52.89] | 48.81 [37.27, 54.70] | 232.6 [176.9, 258.9] | 232.6 [188.3, 256.2] | 93% | 53% |
| uniform-random outliers f=0.10 | 0.1 | 0 | 100% | 165.07 [133.32, 169.13] | 140.05 [120.32, 155.13] | 128.71 [118.47, 136.56] | 890.6 [796.1, 965.9] | 857.0 [761.5, 969.5] | 100% | 27% |
| mask-boundary-biased outliers f=0.10 | 0.1 | 2 | 2% | 71.28 [43.64, 88.23] | 51.02 [37.22, 63.94] | 46.88 [37.13, 63.20] | 447.6 [363.9, 459.9] | 449.8 [362.5, 456.8] | 100% | 60% |

### 2.2 How wrong is the replaced bearing?

`out angle` = median angle between the clean bearing and the replacement (deg); `out ray res` = median distance from the *right* 3-D model point to the replacement ray at the GT pose (mm); `in ray res` = the same for the untouched inliers. A boundary replacement is grazing: a small angle and a residual close to the inlier scale, so it contributes far less squared cost than a uniform one - yet still breaks the layer.

| condition | f | out angle (deg) | out ray res (mm) | in ray res (mm) | n replaced/draw |
|-|-|-|-|-|-|
| uniform-random outliers f=0.05 | 0.05 | 21.3 | 323 | 2.2 | 23.0 |
| mask-boundary-biased outliers f=0.05 | 0.05 | 3.7 | 59 | 2.2 | 22.0 |
| uniform-random outliers f=0.10 | 0.1 | 21.1 | 326 | 2.2 | 45.5 |
| mask-boundary-biased outliers f=0.10 | 0.1 | 3.8 | 62 | 2.2 | 46.0 |

A boundary outlier costs **less** per outlier, and that changes the *mechanism* but not the *outcome*: at f=0.05 the uniform arm leaves a physical mode in 3% of draws (i.e. it destroys the physical filter), while the boundary arm leaves one in 100% but its *minimum is biased* by the coherent ring of grazing rays (93% of draws >10 deg, GT-like 53% of frames; on the eggbox frames the strongest non-hard mode is itself 38-53 deg from GT, so no mode is correct). At f=0.10 both are already past breakdown (100% / 100% hard failures). A real front-end that snaps a corner onto the silhouette would therefore produce *confidently wrong* poses (all modes physical, wrong one selected), which is arguably worse for a downstream consumer than an obvious no-mode failure.

The deeper contrast is *what* breaks: uniform outliers push the cost's minima behind the camera (the physical filter removes every mode, so the reported pose is the flagged all-minima fallback), whereas boundary outliers keep two physical minima but move them away from the truth. For a downstream consumer the second is more dangerous: the estimator looks healthy (K=2, no fallback flag) while being 65-125 deg wrong.

## 3. Correlated noise vs iid noise at equal variance

| condition | sigma_c / sigma_p (px) | sigma_eq twin (px) | K_phys med | hard rot | Laplace rot | MC rot | hard ADD-S | hard fail | GT-like |
|-|-|-|-|-|-|-|-|-|-|
| correlated: 1 px/frame offset + 0.5 px/pt jitter | 1 / 0.5 | 1.12 | 2 | 2.03 [1.32, 3.30] | 43.68 [33.09, 46.34] | 44.15 [34.00, 50.61] | 3.4 [2.5, 5.3] | 7% | 93% |
| iid, equal total variance (sigma_eq=1.12 px) | iid 1.12 | 1.12 | 2 | 2.00 [1.21, 4.21] | 43.61 [32.94, 46.50] | 43.09 [33.60, 50.25] | 2.9 [2.1, 5.3] | 7% | 93% |
| correlated: 2 px/frame offset + 1 px/pt jitter | 2 / 1 | 2.24 | 2 | 2.04 [1.27, 4.37] | 43.58 [32.89, 46.26] | 42.82 [33.59, 46.96] | 3.3 [3.0, 5.2] | 6% | 93% |
| iid, equal total variance (sigma_eq=2.24 px) | iid 2.24 | 2.24 | 2 | 3.11 [2.03, 12.22] | 42.65 [32.14, 45.58] | 43.05 [31.80, 50.23] | 19.9 [11.8, 36.7] | 27% | 93% |

**Paired difference (structured - equal-variance iid), per-frame mean over paired seeds, frame bootstrap CI:**

| pair | hard rot (deg) | hard ADD-S (mm) | Laplace rot (deg) | hard fail delta |
|-|-|-|-|-|
| corr_c1 - corr_iid_c1 | -0.29 [-0.69, +0.01] | -0.4 [-4.1, +2.6] | +0.44 [+0.14, +0.79] | +0% |
| corr_c2 - corr_iid_c2 | -3.49 [-5.48, -1.61] | -30.8 [-53.6, -12.5] | +0.82 [+0.33, +1.38] | -21% |
| het_h1 - het_iid_h1 | +2.76 [+0.07, +6.93] | +0.4 [-0.9, +1.9] | +1.21 [+0.51, +2.34] | +2% |
| het_h2 - het_iid_h2 | +10.90 [+1.30, +23.70] | +2.2 [+0.2, +4.3] | +1.88 [+0.80, +3.48] | +12% |

**Result: correlation does not amplify the hard-estimator error beyond its variance.** The paired differences for hard rotation are <= 0 at both amplitudes (CI includes 0 at 1.12 px; clearly negative at 2.24 px), so a 1-2 px systematic offset is *no worse* than iid jitter of the same total variance for the hard argmin. The mechanism is that a common offset is largely absorbed by the pose (it looks like a small camera rotation), whereas iid jitter adds independent scatter along the plane's soft direction. The correlated arm is, however, consistently slightly worse for the **smooth** estimators (Laplace/MC +0.4 to +1.0 deg, CI excludes 0), because a coherent bias survives the posterior averaging while independent jitter does not. Caveat: only 1-2 px offsets are tested, the RANSAC plane subset is held fixed (so a large offset cannot change *which* plane is selected), and 15 frames give limited power.

## 4. Heteroscedastic noise vs homoscedastic noise at equal RMS

Realised profiles: h1 = 0.5->4 px, lambda=6 px, per-frame RMS 1.30-2.43 px, 21% of points above 2 px; h2 = 1->5 px, lambda=10 px, RMS 2.36-3.65 px, 60% above 2 px. Median distance to the mask boundary of the plane-subset pixels: 13.0 px (p90 22.4 px).

| condition | K_phys med | hard rot | Laplace rot | MC rot | hard ADD-S | hard fail | GT-like |
|-|-|-|-|-|-|-|-|
| heteroscedastic 0.5->4 px, lambda=6 px | 2 | 3.45 [1.57, 13.96] | 43.99 [34.57, 47.99] | 43.06 [32.93, 53.25] | 7.3 [4.4, 14.5] | 26% | 93% |
| iid, equal per-frame RMS variance (h1) | 2 | 2.23 [1.42, 3.59] | 43.18 [33.91, 50.79] | 42.65 [32.70, 50.40] | 7.6 [4.1, 14.4] | 23% | 93% |
| heteroscedastic 1->5 px, lambda=10 px | 2 | 13.32 [2.69, 25.52] | 44.46 [33.29, 52.83] | 43.28 [33.67, 50.35] | 27.5 [15.7, 107.0] | 38% | 87% |
| iid, equal per-frame RMS variance (h2) | 2 | 4.57 [2.82, 18.46] | 42.68 [33.42, 50.69] | 41.89 [32.41, 46.24] | 30.1 [15.3, 106.6] | 26% | 100% |

The heteroscedastic arms are worse than their equal-RMS twins in the **mean** (+2.76 [+0.07, +6.93] and +10.90 [+1.30, +23.70] deg paired) but not in the median (+0.15 and +0.15 deg). The degradation is concentrated in the frames where boundary-weighted noise tips the near-tie: at 15 frames a handful of frames moves the mean and the failure rate but not the median. Because the errors remain independent, the posterior mean still averages them - which is why Laplace/MC degrade only ~1-2 deg while the hard argmin's tail grows.

## 5. Per-frame view: boundary-biased f=0.10 (worst structured arm)

| frame/img | obj | K_phys | GT-like | hard rot | Laplace rot | hard ADD-S | fallback |
|-|-|-|-|-|-|-|-|
| 000010/15 | 10 | 2.0 | 17% | 88.2 | 46.7 | 343 | 0 |
| 000010/18 | 10 | 2.0 | 17% | 101.2 | 63.9 | 402 | 0 |
| 000010/40 | 10 | 2.0 | 50% | 67.2 | 38.5 | 453 | 0 |
| 000010/60 | 10 | 2.0 | 0% | 125.1 | 88.0 | 338 | 0 |
| 000010/66 | 10 | 2.0 | 50% | 71.3 | 63.0 | 448 | 0 |
| 000010/72 | 10 | 2.0 | 17% | 77.9 | 25.9 | 453 | 0 |
| 000010/80 | 10 | 2.0 | 33% | 73.9 | 37.2 | 377 | 0 |
| 000010/90 | 10 | 2.0 | 17% | 85.3 | 37.1 | 488 | 0 |
| 000011/20 | 11 | 1.3 | 67% | 88.6 | 64.1 | 540 | 0 |
| 000011/24 | 11 | 2.0 | 83% | 44.5 | 57.0 | 364 | 0 |
| 000011/26 | 11 | 2.0 | 83% | 43.4 | 35.9 | 289 | 0 |
| 000011/32 | 11 | 2.0 | 100% | 19.8 | 34.8 | 275 | 0 |
| 000011/33 | 11 | 2.0 | 83% | 43.6 | 59.6 | 460 | 0 |
| 000011/40 | 11 | 2.0 | 100% | 29.5 | 51.0 | 597 | 0 |
| 000011/49 | 11 | 1.8 | 100% | 46.7 | 64.5 | 615 | 0 |

## 6. Interpretation and caveats

1. **Outlier breakdown survives both structures.** A handful of gross outliers - uniform *or* silhouette - already breaks the layer at f>=0.05 (93-100% hard failures). The *mode* differs (uniform destroys every physical mode and the draw is a flagged all-minima fallback; silhouette keeps two physical modes but moves their minimum), and the second is the more dangerous failure because it looks healthy. A real front-end needs outlier rejection / a robust kernel before this layer is usable.
2. **The iid sigma conclusion survives the correlation structure at 1-2 px.** The equal-variance paired test says a per-frame coherent offset is *no worse* than iid jitter for the hard argmin (paired hard-rotation difference <= 0; the 2.24 px aligned iid twin is actually worse). 'sigma <= 1 px is harmless' therefore remains a valid **variance** statement and does not need a correlation caveat at this amplitude. What *is* correlation-sensitive is the smooth estimator: Laplace/MC are +0.4 to +1.0 deg worse under the coherent offset. Caveat: offsets >2 px and offsets large enough to change the RANSAC plane selection (fixed here by design) are untested.
3. **Heteroscedasticity shows up in the tail, not the median.** Same-RMS boundary-weighted noise leaves the typical frame unchanged (paired median +0.15 deg) but raises the mean (+2.8 and +10.9 deg) and the failure rate (38% vs 26% at h2), i.e. it tips a few near-ties via leverage. So 'variance matters, arrangement matters through the near-tie' - not 'arrangement is irrelevant'.
4. **The boundary model is a real geometric structure, not a tuned failure.** The replacement uses an actual silhouette pixel and its actual depth; the 'grazing' property (small angle, small ray residual) is measured, not imposed. The outlier arm is still a breakdown measurement, not a graceful-degradation claim.
5. **Still simulated and still GT-associated.** No appearance, no descriptor, no trained detector; the correspondences favour the GT pose by construction, so absolute errors are optimistic. The equal-variance twins are the load-bearing comparison.
6. **Small sample / weak power.** 15 frames, 6 noise seeds, 8 MC seeds; CIs are wide and the paired CIs are the only adequately powered comparison. The eggbox/glue groups are not separately powered.
7. **Mode-search robustness.** On 3 of the 1440 draws the default 384-start mode search found no local minimum, because under gross outliers the reduced matrix becomes numerically singular (cond(M) ~ 1e16) and random starts converge only to saddle points. Those draws were re-run with 1024-16384 starts (which find the minimum); no draw was discarded, and `n_no_mode` in the JSON is 0. This is a numerical property of the un-robustified cost, not of the noise model.
8. **Runtime.** 284 s on the login-node CPU (16 conditions x 15 frames x 6 seeds x 8 MC seeds).

## Repro

```bash
cd labs/01-laplace-qpeplayer/experiments/real/linemod
/home/syk/miniconda3/envs/sdpr/bin/python run_structured_noise.py   # needs cv2
```

Deterministic: `Instance(seed=20261127)`, noise seeds `(20261128, cond, frame, seed)`, MC seeds `(20261128, cond, frame, seed)`, vertex subsamples `default_rng(20261127)`, bootstrap seeds fixed. Outputs `results/structured_noise.{json,md,png}`.
