# 3DMatch full official set + low/mid-overlap study

This directory extends the parent 3DMatch experiment
(`../threedmatch/`, commit `3fa99be`) from four scenes to **all eight official
3DMatch geometric-registration scene-fragment sets**, and adds a
**low / mid / high overlap** study.

It imports `../threedmatch/load_3dmatch.py` read-only and the shared synthetic
solvers `../../synthetic/common.py`; no parent file is modified.

## Scenes

Data root: `/mnt/storage1/syk/data/3dmatch_fragments/`
(downloaded from `http://3dvision.princeton.edu/projects/2016/3DMatch/downloads/scene-fragments/<name>.zip`
and `<name>-evaluation.zip` through the local proxy).

| scene | status |
|---|---|
| 7-scenes-redkitchen | existing |
| sun3d-home_at-home_at_scan1_2013_jan_1 | existing |
| sun3d-hotel_uc-scan3 | existing |
| sun3d-home_md-home_md_scan9_2012_sep_30 | existing |
| sun3d-hotel_umd-maryland_hotel1 | **new** |
| sun3d-hotel_umd-maryland_hotel3 | **new** |
| sun3d-mit_76_studyroom-76-1studyroom2 | **new** |
| sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika | **new** |

Fragment counts, gt.log pair counts and archive sizes are measured in
`results/manifest.json` (script-generated), not typed into this README.

## Protocol

High-overlap table (`run_full_scenes.py`):

* Official TSDF fragments, voxel-downsampled to 0.03 m.
* `gt.log` maps id2 -> id1 on every scene (verified in
  `results/verify_gt_direction_full.json`); the runner inverts it.
* Base correspondences: GT-mapped nearest neighbours, kept when residual
  < tau = 0.06 m (oracle matches with real scan/occlusion/sampling residual).
* N = 300, c = 0.05 m, 20 trials/cell, 8 highest-overlap fragment pairs per
  scene, p = 0 … 0.7, three outlier types, five methods
  (Kabsch / RANSAC-1000 / RANSAC-adaptive / IRLS-GM / GNC-GM).
* Success: rotation < 5 deg and translation < 0.1 m. Per-scene and pooled
  tables.

Low/mid/high overlap study (`run_overlap_bins.py`):

* Overlap for **every** `gt.log` pair: uncapped fraction of 2000 source
  points mapped by inv(gt) whose nearest neighbour in the target fragment is
  within 0.06 m.
* Bins: low < 0.5, mid 0.5–0.8, high > 0.8; up to 3 pairs per bin per scene.
* N = 300, 8 trials/cell, random outliers at p ∈ {0, 0.3, 0.5, 0.7}, plus a
  coherent-outlier cell at p = 0.3; same five methods.
* Reports success / rotation error / runtime by bin, GNC-vs-RANSAC success
  delta and speedup per bin, and per-pair Spearman correlations.

### Two measurement details that matter

1. **Uncapped overlap.** The parent helper `gt_nn_correspondences` caps its
   returned set at `n` while probing `4n` points, so dividing the capped count
   by the probe count reports `min(1, 4 × true_overlap)`: almost every gt pair
   reads as 100% and the low/mid bins are invisible. `full_common.overlap_fraction`
   counts without a cap. The parent tables are unaffected (they only need
   high-overlap pairs), but the high-overlap selection here is re-ranked with
   the uncapped metric.
2. **Padding.** Pairs whose valid GT-NN count is below N = 300 are padded by
   resampling with replacement; the raw `n_valid` is recorded per trial. This
   only matters for true overlap below ≈25%, i.e. the bottom of the low bin.

## Results

| file | contents |
|---|---|
| `results/verify_gt_direction_full.json` | direction check for all 8 scenes |
| `results/full_scenes.{json,md,png}` | 8-scene per-scene + pooled high-overlap table |
| `results/full_scenes_raw.jsonl` | raw per-trial records |
| `results/pairs_<short>.json` | selected pairs per scene |
| `results/overlap_scan.json` | GT-NN overlap of every gt pair in every scene |
| `results/overlap_bins.{json,md,png}` | low/mid/high overlap tables, advantage, per-pair data |
| `results/overlap_bins_raw.jsonl` | raw per-trial records |
| `results/rounds.json` | 6 measured rounds (hypothesis / change / command / before → after / keep / artifact) |
| `results/manifest.json` | archive sizes, fragment and pair counts |
| `results/logs/` | run logs |

Figures: `results/full_scenes.png` (pooled curves, per-scene bars, runtime) and
`results/overlap_bins.png` (success vs overlap, bin tables, runtime, per-scene
overlap distributions).

## Reproduce

```bash
cd labs/03-robust-qpeplayer/experiments/real/threedmatch_full
/home/syk/miniconda3/envs/WM/bin/python verify_gt_direction_full.py
/home/syk/miniconda3/envs/WM/bin/python run_full_scenes.py
/home/syk/miniconda3/envs/WM/bin/python run_overlap_bins.py
/home/syk/miniconda3/envs/WM/bin/python make_manifest.py
/home/syk/miniconda3/envs/WM/bin/python make_rounds.py
```

The intended scheduler command is
`srun --partition=cpu --time=00:50:00 --ntasks=1 --cpus-per-task=4 ...`.
During this session the only `cpu` node was held by another user's running job
(`squeue` showed the syk job pending on `(Resources)`), so the three runs were
executed as bounded local processes under `taskset` (4 cores, one process at a
time or two at most); each run is far shorter than the 50-minute limit.

## Caveats

* Base matches are **oracle GT-NN**, not a learned descriptor (FCGF/Predator is
  out of scope). This isolates the robust outer layer; real descriptor noise is
  not measured here.
* The overlap estimator is a surface-proximity proxy with tau = 0.06 m; shared
  planar structure (floor/ceiling) can make it optimistic. It is still monotone
  enough to populate all three bins, but it is not the official 3DMatch overlap
  definition (the release does not ship per-pair overlap values).
* Coherent outliers at ≥60% are expected to defeat every method because the
  robust global optimum is no longer the ground-truth pose (identifiability),
  exactly as in the synthetic protocol.
