# ADD-S AUC and recall-threshold curves on the 15 real LINEMOD frames

**Date:** 2026-11-28 · **Source:** [`results/pose_error.json`](pose_error.json) (per-frame ADD-S; 15 frames = 8 eggbox + 7 glue) · **Frames:** 15 · **Runtime:** 0.5 s (CPU, login node).

> **What this is.** The paper's LINEMOD AUC cells in Table 5.20 read `not measured` while the LMO rows carry an ADD-S AUC(0-0.1 m). This report computes that cell *from the per-frame ADD-S values already stored in `pose_error.json`* -- no mesh, no re-fit, no new pose, no interpolation -- per object (eggbox n=8, glue n=7) and pooled (n=15), with 95% frame-bootstrap CIs (1000 resamples) and recall@{5,10,20,50} mm. **What this is not.** Not a benchmark and not the BOP AP metric; n=15 means the recall curve has at most 15 steps, so every threshold that is not one of those steps is unconstrained by the data; the correspondences behind every pose are GT-associated, so absolute ADD-S is optimistic; there is no detector or front-end anywhere in this table. And most importantly: **with no frame above the 0.1 m cap, AUC is exactly `1 - mean(ADD-S)/0.1`** (verified to 1.1e-16), i.e. the cell is a rescaling of the mean ADD-S the paper already reports, not a new measurement.

## Headline

1. **ADD-S AUC(0-0.1 m), pooled (n=15; mean [95% frame-bootstrap CI]):** hard **0.940 [0.909, 0.964]**, Laplace 0.845 [0.814, 0.875], MC 0.845 [0.816, 0.875]. Paired with the same resampled frames: Laplace - hard -0.094 [-0.126, -0.064], MC - hard -0.094 [-0.125, -0.064]. The Laplace-vs-MC gap is 0.0001, ~243x smaller than the CI half-width -- they are the same estimator functional here, as in every other table.
2. **The cell is fully determined by the mean ADD-S.** Every one of the 15 ADD-S values is below the 0.1 m cap (max 28.2 mm), so the clipped sum collapses to `AUC = 1 - mean(ADD-S)/0.1` exactly: hard mean 6.0 mm -> 0.940 (reported 0.940), Laplace 15.5 mm -> 0.845, MC 15.5 mm -> 0.845. The CI is likewise the mean-ADD-S CI of `pose_error.json` after the monotone map, and it reproduces that stored CI to 2.2e-16 (same seeds). **So filling Table 5.20's LINEMOD AUC cell adds no measurement; it is the format-matched restatement of the mean ADD-S (hard 6.0 mm, Laplace 15.5 mm). If the paper would rather not imply a new metric, a mean-ADD-S column carries exactly this information without the 0-0.1 m framing.**
3. **recall@5/10/20/50 mm, pooled:** hard 0.733/0.800/0.933/1.000, Laplace 0.000/0.267/0.733/1.000, MC 0.000/0.267/0.733/1.000 (Laplace and MC coincide to 1/15). **recall@50 mm is 1.000 everywhere** -- the worst ADD-S in the set is 28.2 mm -- so that column carries no information and is included only because it was asked for. The Laplace/MC recall@5 mm is structurally 0: their smallest ADD-S is 8.4 mm because the posterior mean sits between the two modes.
4. **The 1/15 hard failure frame is `000010/72`** (eggbox, image 72): the hard argmin picks the mode 44.46 deg from GT (ADD-S 12.5 mm) while Laplace/MC land between the two modes and happen to be closer (8.4 mm). It is the only frame where the per-frame ordering flips, but it is not enough to make the curves cross: a 100 001-point scan of [0, 0.1] m finds no threshold where hard's recall is below Laplace's or MC's (the difference is 0 only at t=0 and above the largest ADD-S, min = +0.0000). It is also *not* the frame that sets recall@20 mm (that is `000011/49`, ADD-S 22.4 mm).
5. **Descriptive, and the CIs are wide for a structural reason.** With n=15 the curve has at most 15 steps; the pooled AUC CI half-width is ~0.027 for hard and ~0.030 for Laplace/MC. Stratifying the bootstrap by object (preserving 8/7) halves the Laplace/MC width, because object identity explains much of the between-frame spread; that variant is reported as a sensitivity in §5, not used as the primary, because every other CI in the paper is a plain frame bootstrap and this table has to be comparable to them.

## 0. Metric definition

For a threshold $t$ in metres, $\mathrm{recall}(t)=\frac1n\#\{i:\mathrm{ADD\text{-}S}_i\le t\}$. The AUC is the BOP normalisation over the fixed range $[0,T]$, $T=0.1$ m:

$$\mathrm{AUC}(0\text{-}T)=\frac1T\int_0^T\mathrm{recall}(t)\,dt=\frac1n\sum_i\Big(1-\frac{\min(\max(\mathrm{ADD\text{-}S}_i,0),T)}{T}\Big).$$

The second equality is exact (the integrand is a step function, so the integral is a sum of rectangle widths) and it is the form computed here. Four consequences bound what the number can mean:

1. **Nothing is interpolated.** The curve is the empirical CDF of 15 numbers, so recall only changes at those 15 ADD-S values; between two steps it is constant and unconstrained by the data.
2. **`mean over thresholds of recall` and `mean of clipped per-frame score` are the same number** for any uniform grid over $[0,T]$ fine enough to resolve the steps; checked with a dense 200 001-point integral (max abs diff 2.4e-07).
3. **With no frame at the cap, the clip is inactive and the sum is affine in the mean:** $\mathrm{AUC}=1-\overline{\mathrm{ADD\text{-}S}}/T$. That is the case here (max 28.2 mm < 100 mm), so this AUC is a linearly rescaled mean ADD-S and cannot rank the estimators differently from the mean ADD-S column. A 1 mm change in mean ADD-S moves the AUC by 0.01.
4. **It is not a detection rate near the boundary.** An AUC of 0.940 here means 'mean ADD-S 6.0 mm', not '94% of frames would be detected at 10 cm'; the window is 3.5x wider than the worst error in the set.

**recall@X mm** is $\mathrm{recall}(X\,\mathrm{mm})$, reported with $k/n$ so the granularity (1/15 = 0.067 per frame) is visible.

**Data availability.** The stored file has a per-frame `adds_m` for all 15 frames and all three estimators (plus the 20 per-seed MC values and the 3-vertex-subsample spread of each frame), not just medians or aggregates. Every number below is therefore computed from the stored per-frame array -- nothing is interpolated, back-solved from a median, or reconstructed from the mesh.

**Bootstrap.** Non-parametric frame bootstrap, 1000 resamples, percentile 95% CI ($[2.5, 97.5]$), plain 15-of-15 resampling and the same seeds `run_pose_error.py` used for its `adds_m` block (base `20261127`), so the AUC CI here *is* the stored mean-ADD-S CI after the monotone map -- asserted to 2.2e-16 in §5. Paired differences (Laplace - hard, MC - hard) use the same resampled frame indices for both estimators. Per-object rows resample within the object (8 of 8, 7 of 7). The MC point estimate is the AUC of the 20-seed-mean ADD-S per frame; a hierarchical bootstrap over frames *and* those 20 seeds is in §5.

## 1. Main table

Mean [95% frame-bootstrap CI]. Object rows resample within the object; the pooled row resamples 15 of 15 frames. Median and median CI are in `results/auc15.json`.

| estimator | AUC eggbox (n=8) | AUC glue (n=7) | AUC pooled (n=15) | AUC pooled, object-balanced |
|-|-|-|-|-|
| hard argmin | 0.952 [0.927, 0.968] | 0.926 [0.865, 0.969] | **0.940 [0.909, 0.964]** | 0.939 [0.908, 0.964] |
| Laplace | 0.895 [0.882, 0.909] | 0.789 [0.761, 0.814] | **0.845 [0.814, 0.875]** | 0.842 [0.827, 0.856] |
| MC | 0.895 [0.880, 0.907] | 0.789 [0.758, 0.813] | **0.845 [0.816, 0.875]** | 0.842 [0.826, 0.856] |

*`object-balanced` = mean of the two per-object AUCs (equal object weight); `pooled` = mean over the 15 frames (equal frame weight, the BOP per-instance convention). They differ by 0.001 for hard, i.e. the 8/7 imbalance is not what drives the pooled number.*

*Object split:* hard is 0.952 vs 0.926 (the two CIs overlap); Laplace is 0.895 vs 0.789 (disjoint) and MC 0.895 vs 0.789 (disjoint). The Laplace/MC gap between objects is driven by the glue frames sitting at 17-28 mm; with n=8 / n=7 and no pre-registered object contrast this is reported, not claimed as an object effect.*

### Paired differences (same resampled frames)

| comparison | eggbox | glue | pooled |
|-|-|-|-|
| Laplace - hard | -0.056 [-0.083, -0.023] | -0.137 [-0.169, -0.104] | -0.094 [-0.126, -0.064] |
| MC - hard | -0.057 [-0.083, -0.022] | -0.137 [-0.168, -0.104] | -0.094 [-0.125, -0.064] |

### Table 5.20 cell format, `median (mean [CI])`

Table 5.20 quotes LMO AUC as `median (mean)`. For LINEMOD the same cell is:

| instance | n | AUC hard | AUC Laplace | AUC MC |
|-|-|-|-|-|
| LINEMOD (E15) | 15 | 0.963 (0.940 [0.909, 0.964]) | 0.861 (0.845 [0.814, 0.875]) | 0.859 (0.845 [0.816, 0.875]) |

Feeding this row into Table 5.20 also makes the column honest about what it is: for LINEMOD `mean(ADD-S)` = 1 - AUC/0.1, so the AUC and the mean ADD-S columns of that table are the same variable up to an affine map for any row where no frame exceeds 0.1 m.

## 2. Recall at 5 / 10 / 20 / 50 mm

Value (k/n) [95% CI]. Pooled: plain 15-frame bootstrap.

| group, estimator | recall@5 mm | @10 mm | @20 mm | @50 mm |
|-|-|-|-|-|
| pooled, hard argmin | 0.733 (11/15) [0.532, 0.933] | 0.800 (12/15) [0.600, 1.000] | 0.933 (14/15) [0.800, 1.000] | 1.000 (15/15) [1.000, 1.000] |
| pooled, Laplace | 0.000 (0/15) [0.000, 0.000] | 0.267 (4/15) [0.067, 0.467] | 0.733 (11/15) [0.533, 0.933] | 1.000 (15/15) [1.000, 1.000] |
| pooled, MC | 0.000 (0/15) [0.000, 0.000] | 0.267 (4/15) [0.067, 0.468] | 0.733 (11/15) [0.467, 0.933] | 1.000 (15/15) [1.000, 1.000] |
| eggbox, hard argmin | 0.750 (6/8) | 0.875 (7/8) | 1.000 (8/8) | 1.000 (8/8) |
| eggbox, Laplace | 0.000 (0/8) | 0.500 (4/8) | 1.000 (8/8) | 1.000 (8/8) |
| eggbox, MC | 0.000 (0/8) | 0.500 (4/8) | 1.000 (8/8) | 1.000 (8/8) |
| glue, hard argmin | 0.714 (5/7) | 0.714 (5/7) | 0.857 (6/7) | 1.000 (7/7) |
| glue, Laplace | 0.000 (0/7) | 0.000 (0/7) | 0.429 (3/7) | 1.000 (7/7) |
| glue, MC | 0.000 (0/7) | 0.000 (0/7) | 0.429 (3/7) | 1.000 (7/7) |

**Frames above each threshold** (the exact failures, so the curve can be read without the figure):

| threshold | hard | Laplace | MC |
|-|-|-|-|
| 5 mm | `000010/40`, `000010/72`, `000011/20`, `000011/49` | `000010/15`, `000010/18`, `000010/40`, `000010/60`, `000010/66`, `000010/72`, `000010/80`, `000010/90`, `000011/20`, `000011/24`, `000011/26`, `000011/32`, `000011/33`, `000011/40`, `000011/49` | `000010/15`, `000010/18`, `000010/40`, `000010/60`, `000010/66`, `000010/72`, `000010/80`, `000010/90`, `000011/20`, `000011/24`, `000011/26`, `000011/32`, `000011/33`, `000011/40`, `000011/49` |
| 10 mm | `000010/72`, `000011/20`, `000011/49` | `000010/15`, `000010/18`, `000010/60`, `000010/66`, `000011/20`, `000011/24`, `000011/26`, `000011/32`, `000011/33`, `000011/40`, `000011/49` | `000010/15`, `000010/18`, `000010/60`, `000010/66`, `000011/20`, `000011/24`, `000011/26`, `000011/32`, `000011/33`, `000011/40`, `000011/49` |
| 20 mm | `000011/49` | `000011/20`, `000011/24`, `000011/33`, `000011/49` | `000011/20`, `000011/24`, `000011/33`, `000011/49` |
| 50 mm | none | none | none |

### How fragile is a single threshold?

Each row takes the frame closest to the threshold and compares its distance to the threshold with that frame's own dispersion -- for hard/Laplace the max - min ADD-S over its 3 vertex subsamples (`adds_sub_spread_m`), for MC the sd over its 20 MC seeds. If the gap is smaller than the spread, a different draw could move that frame across the threshold and the recall count would change by 1/15 = 0.067. The two spread types are not the same quantity (vertex discretisation vs posterior-seed noise) but both are the right scale for *how well that frame's ADD-S is pinned down*.

| threshold | estimator | closest frame | ADD-S (mm) | gap (mm) | spread (mm) | spread type | could flip |
|-|-|-|-|-|-|-|-|
| 5 mm | hard argmin | `000010/18` | 4.3 | 0.67 | 0.22 | vertex subsample range | no |
| 5 mm | Laplace | `000010/72` | 8.4 | 3.39 | 0.08 | vertex subsample range | no |
| 5 mm | MC | `000010/72` | 8.4 | 3.41 | 0.15 | MC seed sd | no |
| 10 mm | hard argmin | `000010/72` | 12.5 | 2.55 | 0.34 | vertex subsample range | no |
| 10 mm | Laplace | `000010/40` | 9.7 | 0.25 | 0.36 | vertex subsample range | **yes** |
| 10 mm | MC | `000010/40` | 9.7 | 0.25 | 0.38 | MC seed sd | **yes** |
| 20 mm | hard argmin | `000011/49` | 22.4 | 2.39 | 1.32 | vertex subsample range | no |
| 20 mm | Laplace | `000011/24` | 20.2 | 0.20 | 0.27 | vertex subsample range | **yes** |
| 20 mm | MC | `000011/24` | 20.0 | 0.03 | 1.65 | MC seed sd | **yes** |
| 50 mm | hard argmin | `000011/49` | 22.4 | 27.61 | 1.32 | vertex subsample range | no |
| 50 mm | Laplace | `000011/49` | 28.1 | 21.86 | 1.87 | vertex subsample range | no |
| 50 mm | MC | `000011/49` | 28.2 | 21.81 | 0.19 | MC seed sd | no |

**Reading:** the numbers that are not stable to the last frame are the ones where `could flip` is marked -- for each of them the closest frame is closer to the threshold than that frame's own spread, so a different draw would move the count by 1/15 = 0.067:
- recall@10 mm for Laplace: closest frame `000010/40` at 9.7 mm, gap 0.25 mm vs vertex subsample range 0.36 mm; the count could be 0.200, 0.267 or 0.333 instead of 0.267.
- recall@10 mm for MC: closest frame `000010/40` at 9.7 mm, gap 0.25 mm vs MC seed sd 0.38 mm; the count could be 0.200, 0.267 or 0.333 instead of 0.267.
- recall@20 mm for Laplace: closest frame `000011/24` at 20.2 mm, gap 0.20 mm vs vertex subsample range 0.27 mm; the count could be 0.667, 0.733 or 0.800 instead of 0.733.
- recall@20 mm for MC: closest frame `000011/24` at 20.0 mm, gap 0.03 mm vs MC seed sd 1.65 mm; the count could be 0.667, 0.733 or 0.800 instead of 0.733.
The hard column is not in that regime at 5 or 20 mm (gaps 0.67 mm and 2.39 mm vs spreads 0.22 mm and 1.32 mm), and the AUC is far less exposed than any recall count: moving one frame by a whole 0.4 mm shifts the pooled AUC by 0.0003, i.e. ~102x smaller than its own CI. This is the concrete reason to prefer the AUC over a recall@t cell for the paper, independent of the redundancy caveat in §0.

## 3. The 15 steps (per-frame contributions)

Each row is one step of the curve: $\mathrm{AUC}_i=1-\min(\mathrm{ADD\text{-}S}_i,100\,\mathrm{mm})/100\,\mathrm{mm}$, and the pooled AUC is the mean of that column. Sorted by object then image.

| frame/img | obj | ADD-S hard (mm) | Laplace (mm) | MC (mm) | AUC_i hard | AUC_i Lap | AUC_i MC | hard argmin = GT-like |
|-|-|-|-|-|-|-|-|-|
| `000010/15` | eggbox | 3.0 | 10.3 | 10.3 | 0.970 | 0.897 | 0.897 | yes |
| `000010/18` | eggbox | 4.3 | 12.2 | 12.3 | 0.957 | 0.878 | 0.877 | yes |
| `000010/40` | eggbox | 6.2 | 9.7 | 9.7 | 0.938 | 0.903 | 0.903 | yes |
| `000010/60` | eggbox | 3.0 | 13.9 | 14.1 | 0.970 | 0.861 | 0.859 | yes |
| `000010/66` | eggbox | 2.7 | 12.1 | 12.0 | 0.973 | 0.879 | 0.880 | yes |
| `000010/72` | eggbox | 12.5 | 8.4 | 8.4 | 0.875 | 0.916 | 0.916 | **no** |
| `000010/80` | eggbox | 3.8 | 8.8 | 8.9 | 0.962 | 0.912 | 0.911 | yes |
| `000010/90` | eggbox | 3.1 | 8.5 | 8.4 | 0.969 | 0.915 | 0.916 | yes |
| `000011/20` | glue | 13.8 | 22.9 | 22.8 | 0.862 | 0.771 | 0.772 | yes |
| `000011/24` | glue | 2.8 | 20.2 | 20.0 | 0.972 | 0.798 | 0.800 | yes |
| `000011/26` | glue | 3.7 | 17.6 | 17.5 | 0.963 | 0.824 | 0.825 | yes |
| `000011/32` | glue | 3.1 | 19.0 | 19.0 | 0.969 | 0.810 | 0.810 | yes |
| `000011/33` | glue | 3.8 | 23.4 | 23.7 | 0.962 | 0.766 | 0.763 | yes |
| `000011/40` | glue | 2.2 | 16.8 | 16.7 | 0.978 | 0.832 | 0.833 | yes |
| `000011/49` | glue | 22.4 | 28.1 | 28.2 | 0.776 | 0.719 | 0.718 | yes |
| **mean** | | **6.0** | **15.5** | **15.5** | **0.940** | **0.845** | **0.845** | 14/15 |

## 4. The 1/15 hard failure frame

`000010/72` is the only frame where the hard argmin is not the mode closest to GT (`hard_is_gt_like = false`; the mode 3.8 deg from GT has $J$ higher by 1.25e-04, so the loss ranking is wrong there). Its effect on the curves:

- hard ADD-S 12.5 mm vs Laplace/MC 8.4/8.4 mm: the one frame whose per-frame ordering is reversed.
- It is **not** enough to reverse the population ordering: over a 100 001-point scan of [0, 0.1] m, hard's recall is never below Laplace's or MC's (minimum difference +0.0000, reached only at t=0 and above the largest ADD-S), so the curves never cross and the AUC order is hard > Laplace ~ MC at every object level.
- It is **not** the frame that decides recall@20 mm: hard passes 20 mm here (12.5 mm); `000011/49` at 22.4 mm is the one that fails. Two different single frames, two different statements, both single-frame accidents.
- Leave-one-out: dropping it would move the pooled hard AUC from 0.940 to 0.944. It is **not** dropped: the frame belongs to the pre-registered 15, and removing it after seeing the result would be selection. Contrast the honest reading -- the estimator is wrong on 1/15 frames, and that is already in the rotation table (44.5 deg).

## 5. Checks, sensitivity, caveats

**Arithmetic checks.**

- Closed form vs a dense numerical integral of the recall curve (200 001 points): max abs diff 2.4e-07 (hard 0.9396078 vs 0.9396081).
- **The same formula reproduces the published LMO AUC** (`results/lmo_two_mode.json`, n=37, a completely different frame set): hard 0.8624 vs published 0.8624, Laplace 0.7656 vs 0.7656, MC 0.7662 vs 0.7662; max abs diff 0.0e+00. The LINEMOD number is therefore on the same scale as the LMO rows of Table 5.20.
- **The bootstrap reproduces the stored mean/median ADD-S CIs exactly** after the map $x\mapsto1-x/T$ (max abs diff mean 2.2e-16, median 0.0e+00; e.g. hard mean CI [0.909, 0.964] vs rescaled stored [0.909, 0.964]). This is the affine identity of §0 consequence 3, confirmed numerically rather than asserted.
- Affinity of the AUC in the mean: `|AUC - (1 - mean/0.1)|` = 0.0e+00 / 1.1e-16 / 0.0e+00 for hard / Laplace / MC; any frame at the cap: no.
- Full-mesh-vs-full-mesh ADD-S check instead of the 1000-query/dense-GT estimator (`adds_full_check_m` / `adds_full_check_laplace_m`): hard 0.940 (+0.000) and Laplace 0.845 (-0.001); MC has no such column. The ADD-S estimator choice does not move the ordering or the size.

**Sensitivity.**

- Bootstrap stratified by object (8 eggbox + 7 glue resampled within object, the design-preserving variant): hard [0.910, 0.964] (vs [0.909, 0.964]), Laplace [0.831, 0.859] (vs [0.814, 0.875]). The Laplace/MC interval is ~2x narrower when the object mix is held fixed -- object identity is a large part of the between-frame variance -- but the hard-vs-Laplace separation survives both.
- MC seed noise: hierarchical bootstrap over frames *and* the 20 MC seeds gives [0.814, 0.878] vs the frame-only [0.816, 0.875]. Seed noise is negligible next to frame noise, as expected because MC estimates the same functional as Laplace.
- Leave-one-out over the 15 frames: pooled hard AUC spans [0.937, 0.951], Laplace [0.840, 0.854]. No single frame changes the ordering.

**Caveats.**

1. **n=15, and this is descriptive.** At most 15 curve steps; pooled AUC CI half-widths are 0.027 (hard) and 0.030 (Laplace). Object rows are n=8 / n=7. No threshold-based conclusion more precise than 'hard is clearly more accurate than the posterior mean on these frames' is supported, and that conclusion was already in `pose_error.md` without any AUC.
2. **The cell is redundant with the mean ADD-S** (identity above). The only reason to put it in Table 5.20 is format consistency with the LMO rows; the only reason *not* to is that it invites reading the column as an independent metric. Both readings are stated here so the choice is explicit.
3. **Not the BOP metric.** BOP reports AP over a 3-D detection pipeline with per-object normalisation on the official test split; this is the older PoseCNN-style AUC on a 0-0.1 m window over 15 hand-picked frames. Not comparable to a leaderboard entry and not quoted as one.
4. **GT-associated correspondences.** Poses were fit to nearest-vertex correspondences selected through the GT pose with a RANSAC plane subset, so the GT is favoured by construction and the absolute ADD-S is optimistic. Only the relative ordering (hard more accurate than Laplace/MC) is measured.
5. **Frame selection.** The 15 frames are those where both objects gave K=2 physical modes with >=900 valid depth pixels; heavy occlusion and K=1 frames are outside the sample, so the AUC is conditional on K>=2.
6. **recall@50 mm is saturated and recall@5 mm for Laplace/MC is structurally 0**; only 10 and 20 mm carry signal, and the 10 mm value moves by +/-1 frame with the vertex draw (§2).

## 6. Reproduce

```bash
cd labs/01-laplace-qpeplayer/experiments/real/linemod
/home/syk/miniconda3/envs/WM/bin/python run_auc15.py
```

Runtime 0.5 s (login node, CPU). The script reads only `results/pose_error.json` and, for the cross-check, `results/lmo_two_mode.json`; it does not import or modify any other script. Constants are at the top (`T_M=0.1`, `N_BOOT=1000`, `BOOT_SEED=20261127`). Every reported number is deterministic; the only field that changes between runs is the recorded runtime.

## 7. Files

- `run_auc15.py` -- this analysis.
- `results/auc15.md` -- this report.
- `results/auc15.json` -- every number: meta, per-frame rows, stored recall curves, aggregates with CIs, checks, failure and fragility bookkeeping.
- `results/auc15.png` -- (a) pooled recall curve, (b) the 15 per-frame ADD-S values that generate its steps, (c) AUC bars with CIs per object and pooled.
- `results/pose_error.json` -- the input (per-frame ADD-S of the 15 frames).

