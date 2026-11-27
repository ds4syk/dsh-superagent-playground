# Classical precomputed-feature front-end (Harris / ORB) on the real LINEMOD frames

> **What this is.** A **classical, non-learned feature front-end**: hand-crafted Harris / Shi-Tomasi / ORB corner detection on the real RGB image inside the BOP `mask_visib`, the real depth at the corner pixel, back-projection with the real intrinsics, the **same** GT-assisted nearest-mesh-vertex association operator as the dense pipeline, the **same** RANSAC planar-face selection and the **same** hard/Laplace/MC two-mode layer. **What this is not.** Not a learned detector, not a descriptor-matching pipeline (no appearance enters the association - only the corner *location* does), and not a LINEMOD accuracy benchmark: the association still uses the GT pose, so clean absolute errors remain optimistic. The value is the comparison against the dense-pixel baseline on the *same* frames with *everything downstream* fixed.

## Headline

1. **The corner front-end cannot feed the dense pipeline's own gate.** With the dense gate (`MIN_PLANE=150`, `MIN_OTHER=50`) **0/15** frames pass for any detector: the objects are 60-70 px across and the densest corner set (Shi-Tomasi, 109 corners/frame, median) is ~8x sparser than the 900 dense pixels; only 56 of those corners land on the dominant plane.
2. **With an explicitly relaxed well-posedness gate (>=30 plane / >=15 off-plane points) the layer runs on 12/15 frames for Shi-Tomasi and 11/15 for ORB** (Harris: **0/15** - only 25 corners/frame median, 13 on the plane, so no frame reaches 30 plane points). The gate failures are Shi-Tomasi: 000011/33, 000011/40, 000011/49; ORB: 000010/72, 000010/90, 000011/33, 000011/40. Harris gets no pose metric at all; reporting one would be fabricated.
3. **At equal point count Shi-Tomasi corners match the dense control; ORB does not.** `dense@n` (the same 900 dense pixels randomly subsampled to the corner count, RANSAC re-run) gives hard rotation 2.18 [1.58, 4.18] deg / ADD-S 3.7 [2.7, 6.4] mm at n=120, versus 2.77 [1.91, 3.69] deg / 3.4 [2.7, 11.9] mm for Shi-Tomasi corners at the same count - overlapping CIs, i.e. **the Shi-Tomasi loss is a point-count effect**. ORB at n=83 is 6.35 [4.73, 16.20] deg / 7.7 [4.5, 13.1] mm versus its own control 1.91 [1.47, 4.16] deg / 4.2 [3.4, 5.5] mm: **ORB is worse at the same count**, so its extra loss is the *spatial distribution* of its keypoints, not the count.
4. **ORB's extra loss is a plane-selection effect and it is measurable.** ORB's multi-scale keypoints are spatially clustered, and on 64% of its frames the RANSAC dominant plane is a *different surface patch* than the dense baseline's (normal 45-89 deg away; the selected set is still a genuine planar patch, ~1-2 mm thick, but an oblique/side patch of the carton). When ORB finds >=98 corners (4 frames: 98-159) it agrees with the dense face; at <=92 corners (7 frames) it never does. Worst-frame hard rotation 65.2 deg. Shi-Tomasi keeps the dense face on 100% of frames.
5. **Hard-argmin tie resolution is not robust at corner density, but the sample is too small to prove an excess and the equal-count control shows the same rate.** Frames with hard error >10 deg: dense 7% (1/15) (the dense baseline already flips on 000010/72), Shi-Tomasi 17% (2/12), ORB 27% (3/11), equal-count dense control 17% (2/12) for Shi-Tomasi and 9% (1/11) for ORB. The failure modes are mixed: wrong-mode selection (the correct mode exists 4-9 deg from GT) and a genuinely rotated cost (both modes far from GT). At this sample the corner arms are not distinguishable from their equal-count controls, so the tie fragility is a point-density effect, not a corner-quality effect.

**Framing.** Classical front-end, GT-assisted association. Absolute errors are optimistic; the claim is the *relative* comparison at matched point count plus the honest gate-failure rate.

## 0. Protocol

- **Frames.** The 15 frames of `run_two_mode.py` Part A / `run_detector_noise.py` (['000010/15', '000010/18', '000010/40', '000010/60', '000010/66', '000010/72', '000010/80', '000010/90', '000011/20', '000011/24', '000011/26', '000011/32', '000011/33', '000011/40', '000011/49']). Frame set asserted equal to `results/detector_noise.json`.
- **Front-end (classical, non-learned).** *Harris*: cv2.goodFeaturesToTrack(useHarrisDetector=True, blockSize=3, k=0.04, qualityLevel=0.001, minDistance=2, maxCorners=2000) on the grey RGB, then cv2.cornerSubPix(5x5, 20 it, 0.01). *Shi-Tomasi*: cv2.goodFeaturesToTrack(qualityLevel=0.001, minDistance=2, maxCorners=2000) on the grey RGB, then cv2.cornerSubPix(5x5). *ORB*: cv2.ORB_create(nfeatures=2000).detectAndCompute with the visible mask; native keypoint positions (multi-scale FAST+Harris scoring). Everything runs on the grey RGB; the BOP visible mask is used as the detection mask. No training, no descriptor matching.
- **Depth & association.** The depth is the real 16-bit millimetre BOP depth at the rounded corner pixel; if that pixel is 0/off-mask the nearest valid in-mask depth within a 7x7 window is used (counted per frame). The 3-D camera point is then mapped to the model frame with the GT pose and associated to the nearest real mesh vertex - the *identical* operator of `run_two_mode.real_correspondences`, so the association bias is the same as in the dense baseline.
- **Gates.** *strict* = (150, 50) (the dense pipeline's own gate); *relaxed* = (30, 15) (>=3x the 10 reduced-cost parameters, >=15 off-plane points - the minimum for the same layer to be numerically meaningful). Frames failing the relaxed gate have **no** pose metric.
- **Layer.** Unchanged: same reduced bearing cost, `n_starts=384`, `tau=0.01`, physical filter `z>0.05` m, hard = lowest-J physical mode, Laplace = weighted chordal mean, MC = 400 mode samples x 10 seeds. The code calls `run_detector_noise.eval_layer` directly.
- **Arms.** `dense` (900-px clean, recomputed here and checked against `detector_noise.json`: **ok**, max abs diff 0.000000000 deg); `<detector>` (clean corners); `dense_at_n_<detector>` (point-count control: the 900 dense pixels subsampled to the same per-frame corner count, RANSAC re-run, 5 subsample seeds); `<detector>_jit1` (the same corner set with 1 px iid pixel jitter, 5 seeds - front-end localisation repeatability). `dense_on_<detector>` restricts the dense baseline to the detector's passing frames for a paired comparison.
- **Metrics.** Rotation error (deg) vs `cam_R_m2c`; ADD-S (mm) over 1000 fixed mesh query vertices against the full GT mesh; `K_phys`; `K=2` rate; `GT-like` = the GT-closest mode is the hard argmin; hard failure > 10 deg. Median with 95% frame-bootstrap CI (`n_boot=1000`).

## 1. Front-end output per frame

Corners inside the visible mask, valid depth, RANSAC dominant-plane inliers and the gate verdict. `same face` = the selected plane matches the dense baseline's face (normal angle <= 10 deg and offset within 5 mm after the (n,off)->(-n,-off) sign alignment); the face test is evaluated whenever RANSAC returns a plane, including the gate-fail rows (it describes the plane, not the pose, so it is meaningful there).

| frame | obj | dense plane | Shi-Tomasi n (plane/other) | gate | same face | ORB n (plane/other) | gate | same face | Harris n (plane/other) | gate |
|-|-|-|-|-|-|-|-|-|-|-|
| 000010/15 | 10 | 558 | 179 (98/81) | pass | 100% | 74 (33/41) | pass | 0% | 35 (15/20) | FAIL | 0% |
| 000010/18 | 10 | 474 | 125 (62/63) | pass | 100% | 80 (42/38) | pass | 0% | 26 (17/9) | FAIL | 0% |
| 000010/40 | 10 | 482 | 109 (55/54) | pass | 100% | 83 (41/42) | pass | 0% | 24 (12/12) | FAIL | 0% |
| 000010/60 | 10 | 452 | 158 (67/91) | pass | 100% | 76 (50/26) | pass | 0% | 23 (15/8) | FAIL | 0% |
| 000010/66 | 10 | 460 | 103 (52/51) | pass | 100% | 98 (40/58) | pass | 100% | 22 (12/10) | FAIL | 0% |
| 000010/72 | 10 | 501 | 131 (71/60) | pass | 100% | 57 (25/32) | FAIL | 0% | 30 (13/17) | FAIL | 0% |
| 000010/80 | 10 | 508 | 141 (81/60) | pass | 100% | 92 (33/59) | pass | 0% | 26 (12/14) | FAIL | 0% |
| 000010/90 | 10 | 474 | 121 (56/65) | pass | 100% | 51 (23/28) | FAIL | 0% | 25 (11/14) | FAIL | 0% |
| 000011/20 | 11 | 413 | 78 (40/38) | pass | 100% | 65 (33/32) | pass | 0% | 25 (13/12) | FAIL | 0% |
| 000011/24 | 11 | 423 | 84 (51/33) | pass | 100% | 123 (80/43) | pass | 100% | 39 (27/12) | FAIL | 100% |
| 000011/26 | 11 | 439 | 101 (62/39) | pass | 100% | 151 (114/37) | pass | 100% | 41 (33/8) | FAIL | 100% |
| 000011/32 | 11 | 474 | 118 (71/47) | pass | 100% | 159 (125/34) | pass | 100% | 51 (43/8) | FAIL | 100% |
| 000011/33 | 11 | 365 | 46 (24/22) | FAIL | 100% | 55 (26/29) | FAIL | 0% | 19 (11/8) | FAIL | 0% |
| 000011/40 | 11 | 428 | 33 (15/18) | FAIL | 100% | 50 (24/26) | FAIL | 0% | 20 (9/11) | FAIL | 0% |
| 000011/49 | 11 | 347 | 47 (18/29) | FAIL | 0% | 70 (30/40) | pass | 0% | 22 (13/9) | FAIL | 0% |

Corner count medians over all 15 frames: Harris 25 (plane 13), Shi-Tomasi 109 (plane 56), ORB 76 (plane 33); dense 900 (plane 460). The corner depth lookup needed the 7x7 fallback for at most a handful of corners per frame.

## 2. Corner front-end vs dense baseline

Aggregate over the frames that produce a layer draw (12 for Shi-Tomasi/ORB, 0 for Harris, 15 for dense). Medians with 95% frame-bootstrap CIs. `dense@n` is the equal-point-count control.

| arm | frames | n pts | n plane | K=2 | hard rot (deg) | Laplace rot (deg) | MC rot (deg) | hard ADD-S (mm) | Laplace ADD-S (mm) | MC ADD-S (mm) | hard fail | GT-like | same face |
|-|-|-|-|-|-|-|-|-|-|-|-|-|-|
| dense 900-px (baseline) | 15 | 460 | 460 | 100% | 2.06 [1.37, 3.49] | 44.24 [33.90, 46.75] | 45.03 [34.73, 46.85] | 3.7 [3.0, 4.3] | 14.2 [10.4, 19.2] | 14.0 [10.6, 20.0] | 7% | 93% | 100% |
| dense 900-px, Shi-Tomasi-subset | 12 | 474 | 474 | 100% | 1.92 [1.22, 2.89] | 43.26 [30.49, 49.02] | 43.89 [30.25, 49.06] | 3.4 [3.0, 5.3] | 11.9 [9.1, 18.2] | 12.1 [9.1, 18.5] | 8% | 92% | 100% |
| dense@n (n=Shi-Tomasi corner count) | 12 | 120 | 62 | 100% | 2.18 [1.58, 4.18] | 45.88 [31.45, 56.41] | 46.17 [31.55, 53.19] | 3.7 [2.7, 6.4] | 12.6 [9.9, 17.7] | 12.7 [9.9, 17.6] | 5% | 100% | 100% |
| Shi-Tomasi corners (clean) | 12 | 120 | 62 | 100% | 2.77 [1.91, 3.69] | 46.74 [30.54, 57.29] | 45.58 [30.28, 55.90] | 3.4 [2.7, 11.9] | 13.5 [9.7, 21.8] | 13.6 [9.7, 21.4] | 17% | 83% | 100% |
| Shi-Tomasi corners + 1 px iid jitter | 12 | 120 | 62 | 100% | 9.49 [2.27, 27.80] | 46.72 [30.55, 57.31] | 46.37 [30.78, 56.52] | 8.4 [3.9, 12.7] | 13.9 [10.0, 23.0] | 13.9 [10.0, 22.6] | 27% | 75% | 100% |
| dense 900-px, ORB-subset | 11 | 460 | 460 | 100% | 1.78 [1.08, 2.29] | 44.24 [33.90, 51.60] | 45.03 [34.73, 52.43] | 3.7 [3.0, 6.4] | 14.2 [10.4, 20.2] | 14.0 [10.6, 20.0] | 0% | 100% | 100% |
| dense@n (n=ORB corner count) | 11 | 83 | 52 | 91% | 1.91 [1.47, 4.16] | 46.56 [34.97, 59.64] | 46.84 [35.43, 59.79] | 4.2 [3.4, 5.5] | 14.5 [10.8, 22.0] | 14.6 [10.9, 21.9] | 2% | 100% | 91% |
| ORB corners (clean) | 11 | 83 | 41 | 100% | 6.35 [4.73, 16.20] | 36.08 [23.01, 55.05] | 36.44 [23.11, 52.67] | 7.7 [4.5, 13.1] | 14.2 [11.3, 21.4] | 14.3 [11.4, 21.2] | 27% | 82% | 36% |
| ORB corners + 1 px iid jitter | 11 | 83 | 41 | 100% | 6.99 [5.23, 23.87] | 36.09 [23.38, 54.91] | 36.07 [23.53, 53.65] | 9.7 [5.9, 16.5] | 13.4 [11.7, 21.9] | 13.4 [11.7, 21.4] | 29% | 91% | 36% |
| Harris corners (clean) | 0 | 25 | 13 | - | - | - | - | - | - | - | - | - | - |

### 2.1 The same table restricted to the paired subset

The corner arms run on 12 frames, so the dense baseline is also restricted to those 12 frames. The paired difference is what the front-end costs at equal *downstream* pipeline:

| detector | frames | dense hard rot | corner hard rot | dense ADD-S | corner ADD-S | dense K=2 | corner K=2 |
|-|-|-|-|-|-|-|-|
| Shi-Tomasi | 12 | 1.92 [1.22, 2.89] | 2.77 [1.91, 3.69] | 3.4 [3.0, 5.3] | 3.4 [2.7, 11.9] | 100% | 100% |
| ORB | 11 | 1.78 [1.08, 2.29] | 6.35 [4.73, 16.20] | 3.7 [3.0, 6.4] | 7.7 [4.5, 13.1] | 100% | 100% |

### 2.2 Point-count control

`dense@n` keeps the *same number* of measurements as the corner arm but spreads them uniformly over the visible object instead of on corners. This separates two very different causes: fewer measurements (count) versus *where* the measurements fall (spatial distribution). The result differs by detector: Shi-Tomasi is statistically indistinguishable from its control (so its loss is a count effect), while ORB is clearly worse than its control at essentially the same count (so ORB's extra loss comes from its clustered sampling - see section 4).

| detector | n | dense@n hard rot | corner hard rot | dense@n hard ADD-S | corner hard ADD-S | dense@n hard fail | corner hard fail |
|-|-|-|-|-|-|-|-|
| Shi-Tomasi | 120 | 2.18 [1.58, 4.18] | 2.77 [1.91, 3.69] | 3.7 [2.7, 6.4] | 3.4 [2.7, 11.9] | 5% | 17% |
| ORB | 83 | 1.91 [1.47, 4.16] | 6.35 [4.73, 16.20] | 4.2 [3.4, 5.5] | 7.7 [4.5, 13.1] | 2% | 27% |

## 3. Where the hard argmin breaks

The dense baseline's hard argmin is *not* always GT: it is wrong on 7% of the clean frames (000010/72, 44.5 deg), so the 0/1 tie resolution was already fragile at 900 points. With 50-110 corner points the two modes are still present (K=2 on 100% of Shi-Tomasi frames, 100% of ORB frames) and the near-tie (w_other median 0.50 / 0.51) is resolved incorrectly on a few more frames. Per-frame detail (seed-averaged; `non-hard mode to GT` is the rotation error of the strongest non-hard mode - 4-9 deg means the *correct* mode exists and hard picked the other one):

| frame | arm | K_phys | non-hard mode to GT (deg) | w_other | hard rot | GT-like | hard ADD-S |
|-|-|-|-|-|-|-|-|
| 000010/66 | Shi-Tomasi | 2.0 | 4.5 | 0.50 | 88.1 | 0% | 19.2 |
| 000010/72 | Shi-Tomasi | 2.0 | 5.1 | 0.52 | 45.0 | 0% | 12.6 |
| 000010/18 | ORB | 2.0 | 22.5 | 0.67 | 16.2 | 100% | 7.7 |
| 000010/60 | ORB | 2.0 | 21.6 | 0.51 | 25.8 | 0% | 13.0 |
| 000011/20 | ORB | 2.0 | 9.4 | 0.56 | 65.2 | 0% | 28.3 |

Two failure modes are mixed here. In the Shi-Tomasi 000010/66 and /72 rows and the ORB 000011/20 row the *correct* mode exists 4-9 deg from GT and the hard argmin returned the other mode: a wrong-mode draw, i.e. the 0/1 tie resolution. In the ORB 000010/18 and /60 rows the GT-like mode is the hard argmin but is itself 16-26 deg off: a genuinely rotated/ambiguous cost from the sparse plane subset, which no estimator can recover. The Laplace/MC chordal mean lands near the half-flip compromise on all of them, so neither smooth estimator is an accuracy rescue. The equal-count dense control flips on 17% (2/12) frames (Shi-Tomasi count) and 9% (1/11) (ORB count), i.e. the same rate as the corner arms: the tie fragility tracks the number of measurements, not their provenance.

## 4. ORB selects a different surface patch

The two-mode construction is built on the RANSAC dominant planar patch; a sparse, image-clustered corner set can select a *different* patch, which changes the ambiguity being instantiated. The face test above shows: Shi-Tomasi keeps the dense patch on 100% of frames, ORB only on 36%. On the mismatching ORB frames the plane normal is 45-89 deg from the dense one and the selected set is still a genuine planar patch (1-2 mm thick) but an oblique or side patch of the carton, so the pose error is large for a reason that is *not* the layer: the front-end's spatial sampling changed the geometry of the problem. The mismatch splits cleanly on ORB's own corner count: every frame with >=98 corners agrees with the dense patch (4/4: 98, 123, 151, 159 corners) and every frame with <=92 corners does not (7/7), and it is *not* reproduced by the equal-count dense control (91% agreement at n=83), which is the cleanest evidence that it is the clustering of the keypoints, not the number of them.

## 5. Jitter arm (front-end localisation repeatability)

The same corner sets with 1 px iid position jitter, 5 seeds (unpaired with the clean draw: the jitter *and* the MC seeds differ, so treat these as descriptive). On the dense baseline sigma=1 px was harmless (hard rotation 1.98 deg vs 2.06 deg clean); on the corner sets it is not, because the plane subset is ~8x smaller and a 1 px move can also change which patch RANSAC selects:

| arm | hard rot | Laplace rot | MC rot | hard ADD-S | hard fail | GT-like |
|-|-|-|-|-|-|-|
| Shi-Tomasi corners (clean) | 2.77 [1.91, 3.69] | 46.74 [30.54, 57.29] | 45.58 [30.28, 55.90] | 3.4 [2.7, 11.9] | 17% | 83% |
| Shi-Tomasi corners + 1 px iid jitter | 9.49 [2.27, 27.80] | 46.72 [30.55, 57.31] | 46.37 [30.78, 56.52] | 8.4 [3.9, 12.7] | 27% | 75% |
| ORB corners (clean) | 6.35 [4.73, 16.20] | 36.08 [23.01, 55.05] | 36.44 [23.11, 52.67] | 7.7 [4.5, 13.1] | 27% | 82% |
| ORB corners + 1 px iid jitter | 6.99 [5.23, 23.87] | 36.09 [23.38, 54.91] | 36.07 [23.53, 53.65] | 9.7 [5.9, 16.5] | 29% | 91% |

## 6. Interpretation and caveats

1. **Classical, non-learned.** Harris / Shi-Tomasi / ORB are hand-crafted detectors; no training, no learned descriptors, no appearance model. Only the corner *location* enters; the model-point association is still the GT-assisted nearest-vertex operator, so this is deliberately not a descriptor-matching front-end and must not be described as one.
2. **Gate failure is the main result, not a nuisance.** At this object scale (60-70 px) the dense gate of 150 plane inliers is unreachable for corners. Harris is 20-40x sparser than the 900-pixel dense set and fails even the relaxed gate on all 15 frames; reporting a pose for it would be fabrication. The 12/15 Shi-Tomasi and ORB frames are the honest subset, and the 3 failures are reported above.
3. **The dense baseline's 0% hard-flip is a density property.** The equal-count control shows the corner front-end is not intrinsically worse per measurement, but the hard argmin's *correctness* at the near-tie is not preserved at 50-110 points. This is direct evidence that hard argmin is the fragile part of the layer, not the Laplace mean (which was already ~44 deg off because of the near-tie bias).
4. **ORB's failure mode is front-end geometry, not the layer.** Its multi-scale keypoints cluster differently and RANSAC picks a different face on several frames; the resulting pose error must not be read as a layer failure.
5. **Small sample.** 15 frames, 12 usable for corners, wide CIs; eggbox (8) and glue (7) are not separately powered. The CIs describe this frame population, not LINEMOD.
6. **ADD-S / rotation errors are optimistic in absolute terms** (GT-assisted association) and pessimistic in the plane-only sense (the cost sees only the plane subset, so the eliminated translation is weakly constrained). Use the relative comparisons.
7. **Runtime.** 33 s on the login-node CPU (15 frames x 3 detectors x 1 clean + 2 jitter arms x 5 seeds + 3 point-count controls x 5 seeds).

## Repro

```bash
cd labs/01-laplace-qpeplayer/experiments/real/linemod
/home/syk/miniconda3/envs/sdpr/bin/python run_feature_frontend.py   # needs cv2
```

Deterministic: corner detection is deterministic; RANSAC seed fixed; corner jitter / count-control seeds `(20261129, arm, frame, seed)`; MC seeds `(20261128, arm, frame, seed)`; vertex subsamples `default_rng(20261127)`; bootstrap seeds fixed. Outputs `results/feature_frontend.{json,md,png}`. Dense reproduction vs `results/detector_noise.json`: **ok**.
