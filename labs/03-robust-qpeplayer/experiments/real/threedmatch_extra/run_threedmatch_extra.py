#!/usr/bin/env python3
"""Cross-scene robustness check on two *additional* official 3DMatch scenes.

The parent pipeline (`experiments/real/threedmatch/run_threedmatch.py`) built a
GT-nearest-neighbour robust-registration table on redkitchen and
sun3d-home_at.  This runner repeats the *identical* protocol on two scenes
that were not used to develop or tune it, to test whether those two-scene
conclusions are cross-scene or scene-specific:

  * ``sun3d-hotel_uc-scan3``            (55 fragments, 55 MB zip)
  * ``sun3d-home_md-home_md_scan9_2012_sep_30`` (60 fragments, 36 MB zip)

Protocol (unchanged from the parent table)
------------------------------------------
* Load official TSDF-fused fragments, voxel-downsample to 0.03 m.
* ``gt.log`` 4x4 maps id2 -> id1 (re-verified here by
  ``verify_gt_direction_extra.py``), so the runner inverts it.
* Base correspondences are GT-mapped nearest neighbours (oracle matches with
  real scan/occlusion/sampling residual), kept when residual < tau = 0.06 m.
* N = 300 pairs per trial; 20 trials/cell; trials sample one of the 8
  highest-overlap fragment pairs per scene.
* Outliers at p in {0,...,0.7}: random target points, one coherent competing
  rigid transform, or boundary residuals (distance in [c^2/3, 3c^2] around
  the GT pose).  c = 0.05 m.
* Methods: Kabsch / RANSAC-1000 / RANSAC-adaptive / IRLS-GM / GNC-GM.
* Success: rotation < 5 deg and translation < 0.1 m.

Outputs (all under ``results/``): ``extra_scenes.json``, ``extra_scenes.md``,
``extra_scenes.png``, ``extra_scenes_raw.jsonl``, ``pairs_<short>.json``,
``verify_gt_direction_extra.json`` and ``logs/``.

Run (Slurm):
  srun --partition=cpu --time=00:40:00 --ntasks=1 --cpus-per-task=4 \
    /home/syk/miniconda3/envs/WM/bin/python run_threedmatch_extra.py 2>&1 | \
    tee results/logs/extra_scenes_$(date +%Y%m%d_%H%M%S).log
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SYN = HERE.parents[1] / "synthetic"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(SYN))
import load_3dmatch as L  # noqa: E402
import common as cm  # noqa: E402

ROOT = Path("/mnt/storage1/syk/data/3dmatch_fragments")
OUT = HERE / "results"
(OUT / "logs").mkdir(parents=True, exist_ok=True)

# --- extra scenes (not used in the parent two-scene table) -----------------
SCENES = ["sun3d-hotel_uc-scan3",
          "sun3d-home_md-home_md_scan9_2012_sep_30"]
SCENE_SHORT = {"sun3d-hotel_uc-scan3": "sun3d-hotel_uc",
               "sun3d-home_md-home_md_scan9_2012_sep_30": "sun3d-home_md"}
PARENT_TWO = ["7-scenes-redkitchen", "sun3d-home_at-home_at_scan1_2013_jan_1"]
PARENT_JSON = Path("/mnt/storage1/syk/.superagent-work/three-paper-tracks-95244014fb3c/"
                   "labs/03-robust-qpeplayer/experiments/real/threedmatch/results/"
                   "threedmatch.json")

N = 300
C = 0.05
TAU = 0.06
VOXEL = 0.03
P_LIST = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
TRIALS = 20
N_PAIRS = 8
METHODS = ["Kabsch", "RANSAC-1000", "RANSAC-adaptive", "IRLS-GM", "GNC-GM"]
BASE_SEED = 20261127
OUTLIER_TYPES = ["random", "coherent", "boundary"]


def _seed(*parts):
    return np.random.SeedSequence([BASE_SEED, *[int(x) for x in parts]])


def project_rotation(R):
    U, _, Vt = np.linalg.svd(R)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    return U @ D @ Vt


def select_pairs(clouds, gt_pairs, n_probe=500, max_pairs=N_PAIRS, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for id1, id2, n_gt, T in gt_pairs:
        if id1 not in clouds or id2 not in clouds:
            continue
        Ti = np.linalg.inv(T)  # gt.log maps id2 -> id1
        x, y, d = L.gt_nn_correspondences(clouds[id1], clouds[id2], Ti,
                                          n=n_probe, rng=rng, tau=TAU)
        if len(x) < 3:
            continue
        rows.append(dict(id1=id1, id2=id2, n_gt=n_gt,
                         overlap=float(len(x) / n_probe),
                         med_res=float(np.median(d)),
                         Tinv=Ti))
    rows.sort(key=lambda r: (r["overlap"], -r["med_res"]), reverse=True)
    return rows[:max_pairs], rows


def make_methods():
    def kabsch_m(x, y, rng):
        return (*cm.kabsch(x, y, np.ones(len(x))), {})

    def ransac_fixed(x, y, rng):
        return cm.solve_ransac(x, y, rng, c=C, iters=cm.RANSAC_ITERS,
                               adaptive=False, return_info=True)

    def ransac_adapt(x, y, rng):
        return cm.solve_ransac(x, y, rng, c=C, iters=cm.RANSAC_ITERS,
                               adaptive=True, return_info=True)

    def irls(x, y, rng):
        return (*cm.solve_irls(x, y, c=C), {})

    def gnc(x, y, rng):
        return (*cm.solve_gnc(x, y, c=C), {})

    return {"Kabsch": kabsch_m, "RANSAC-1000": ransac_fixed,
            "RANSAC-adaptive": ransac_adapt, "IRLS-GM": irls, "GNC-GM": gnc}


def clean_pair_sample(clouds, pair, rng, n=N):
    id1, id2 = pair["id1"], pair["id2"]
    Ti = pair["Tinv"]
    x, y, d = L.gt_nn_correspondences(clouds[id1], clouds[id2], Ti,
                                      n=n, rng=rng, tau=TAU)
    if len(x) < 3:
        return None
    if len(x) < n:  # pad by resampling real valid pairs
        take = rng.choice(len(x), n, replace=True)
        x, y, d = x[take], y[take], d[take]
    Rgt = project_rotation(Ti[:3, :3])
    tgt = Ti[:3, 3].copy()
    return x, y, d, Rgt, tgt


def inject(x, y, Rgt, tgt, rng, p, otype, cloud_j):
    n = len(x)
    y = y.copy()
    meta = dict(idx=np.zeros(0, dtype=int), Rout=None, tout=None)
    n_out = int(round(p * n))
    if n_out <= 0:
        return y, meta
    idx = rng.choice(n, n_out, replace=False)
    meta["idx"] = idx
    if otype == "random":
        take = rng.choice(len(cloud_j), n_out, replace=True)
        y[idx] = cloud_j[take]
    elif otype == "coherent":
        axis = rng.normal(size=3)
        ang = np.deg2rad(rng.uniform(30.0, 60.0))
        R_delta = cm.rotation_about_axis(axis, ang)
        Rout = R_delta @ Rgt
        t_shift = rng.normal(size=3)
        t_shift = t_shift / np.linalg.norm(t_shift) * rng.uniform(0.5, 1.5)
        tout = R_delta @ tgt + t_shift
        y[idx] = (Rout @ x[idx].T).T + tout
        meta["Rout"] = Rout
        meta["tout"] = tout
    elif otype == "boundary":
        dirs = rng.normal(size=(n_out, 3))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        u = rng.uniform(1.0 / 3.0, 3.0, size=(n_out, 1))
        y[idx] = (Rgt @ x[idx].T).T + tgt + dirs * (C * np.sqrt(u))
    else:
        raise ValueError(otype)
    return y, meta


def run_scene(scene, methods):
    scene_dir = ROOT / scene
    gt_path = ROOT / f"{scene}-evaluation" / "gt.log"
    t0 = time.time()
    clouds = L.load_scene(scene_dir, voxel=VOXEL)
    gt_pairs = L.parse_gt_log(gt_path)
    chosen, all_pairs = select_pairs(clouds, gt_pairs)
    print(f"[{scene}] {len(clouds)} fragments, {len(gt_pairs)} gt pairs, "
          f"selected {len(chosen)}: " +
          ", ".join(f"{p['id1']}-{p['id2']}({p['overlap']:.2f})" for p in chosen), flush=True)
    rows, raw = [], []
    for otype in OUTLIER_TYPES:
        for p in P_LIST:
            per = {m: dict(re=[], te=[], time=[], base_in=[]) for m in METHODS}
            for trial in range(TRIALS):
                rng = np.random.default_rng(_seed(scene == SCENES[1], int(round(100 * p)),
                                                  {"random": 1, "coherent": 2,
                                                   "boundary": 3}[otype], trial))
                pair = chosen[int(rng.integers(len(chosen)))]
                sample = clean_pair_sample(clouds, pair, rng)
                if sample is None:
                    continue
                x, y0, d0, Rgt, tgt = sample
                y, meta = inject(x, y0, Rgt, tgt, rng, p, otype, clouds[pair["id2"]])
                base_in = float(np.mean(d0 < C))
                for name in METHODS:
                    rrng = np.random.default_rng(_seed(99, scene == SCENES[1],
                                                       int(round(100 * p)), trial))
                    t1 = time.perf_counter()
                    R, t, info = methods[name](x, y, rrng)
                    dt = time.perf_counter() - t1
                    re, te = cm.pose_errors(R, t, Rgt, tgt)
                    rec = dict(scene=scene, pair=f"{pair['id1']}-{pair['id2']}",
                               p=p, outlier_type=otype, trial=trial, method=name,
                               n=len(x), base_inlier=base_in,
                               success=cm.success(re, te), rot=re, trans=te,
                               time_ms=1000.0 * dt,
                               ransac_iters=info.get("n_iters"),
                               ransac_inliers=info.get("best_count"))
                    if otype == "coherent" and meta["Rout"] is not None:
                        re_o = cm.rot_err_deg(R, meta["Rout"])
                        te_o = float(np.linalg.norm(t - meta["tout"]))
                        rec["picked_outlier"] = bool(re_o < re and te_o < te)
                    raw.append(rec)
                    per[name]["re"].append(re)
                    per[name]["te"].append(te)
                    per[name]["time"].append(dt)
                    per[name]["base_in"].append(base_in)
            for m in METHODS:
                a = per[m]
                s = cm.summarize_errors(a["re"], a["te"], a["time"])
                row = dict(scene=scene, p=p, outlier_type=otype, method=m,
                           n_trials=len(a["re"]),
                           base_inlier=float(np.mean(a["base_in"])),
                           **s)
                if otype == "coherent":
                    vals = [r.get("picked_outlier", False) for r in raw
                            if r["p"] == p and r["outlier_type"] == otype
                            and r["method"] == m]
                    row["frac_picked_outlier"] = float(np.mean(vals)) if vals else None
                rows.append(row)
            print(f"[{scene}] {otype:8s} p={p:.1f} done "
                  f"({time.time() - t0:.1f}s)", flush=True)
    return rows, raw, chosen, all_pairs


# ---------------------------------------------------------------------------
# Pooled aggregation and parent-claim checks
# ---------------------------------------------------------------------------
def pooled_rows(rows):
    """Macro-average the two scene-level cells (each cell already averages trials)."""
    out = []
    for otype in OUTLIER_TYPES:
        for p in P_LIST:
            for m in METHODS:
                sel = [r for r in rows if r["outlier_type"] == otype
                       and abs(r["p"] - p) < 1e-9 and r["method"] == m]
                if not sel:
                    continue
                rec = dict(outlier_type=otype, p=p, method=m, n_scenes=len(sel),
                           success=float(np.mean([r["success"] for r in sel])),
                           rot_median=float(np.mean([r["rot_median"] for r in sel])),
                           rot_p90=float(np.mean([r["rot_p90"] for r in sel])),
                           trans_median=float(np.mean([r["trans_median"] for r in sel])),
                           time_ms_p50=float(np.mean([r["time_ms_p50"] for r in sel])),
                           time_ms_p95=float(np.mean([r["time_ms_p95"] for r in sel])),
                           base_inlier=float(np.mean([r["base_inlier"] for r in sel])))
                if otype == "coherent":
                    rec["frac_picked_outlier"] = float(np.mean(
                        [r["frac_picked_outlier"] for r in sel
                         if r.get("frac_picked_outlier") is not None])) if sel else None
                out.append(rec)
    return out


def load_parent_reference():
    """Parent two-scene pooled numbers, read from the parent run's JSON."""
    if not PARENT_JSON.exists():
        return None
    d = json.loads(PARENT_JSON.read_text())
    rows = d["rows"]
    pooled = []
    for otype in OUTLIER_TYPES:
        for p in P_LIST:
            for m in METHODS:
                sel = [r for r in rows if r["outlier_type"] == otype
                       and abs(r["p"] - p) < 1e-9 and r["method"] == m]
                if not sel:
                    continue
                pooled.append(dict(scene="parent_two_scene_pooled",
                                   outlier_type=otype, p=p, method=m,
                                   success=float(np.mean([r["success"] for r in sel])),
                                   rot_median=float(np.mean([r["rot_median"] for r in sel])),
                                   time_ms_p50=float(np.mean([r["time_ms_p50"] for r in sel]))))
    return dict(source=str(PARENT_JSON), scenes=d["config"]["scenes"],
                rows=pooled)


def _cell(rows, otype, p, m, key="success"):
    r = next(r for r in rows if r["outlier_type"] == otype
             and abs(r["p"] - p) < 1e-9 and r["method"] == m)
    return r[key]


def make_verdicts(pooled, parent):
    """Test the parent two-scene conclusions against the extra-scene numbers."""
    v = {}

    def speedup(p):
        return (_cell(pooled, "random", p, "RANSAC-1000", "time_ms_p50")
                / _cell(pooled, "random", p, "GNC-GM", "time_ms_p50"))

    gnc_random = {p: _cell(pooled, "random", p, "GNC-GM") for p in P_LIST}
    ransac_random = {p: _cell(pooled, "random", p, "RANSAC-1000") for p in P_LIST}
    v["random_gnc_success_min"] = dict(
        value=float(min(gnc_random.values())),
        claim="parent: GNC-GM success 100% at 0-70% random outliers (both scenes)")
    v["random_gnc_parity_with_ransac1000"] = dict(
        value=float(min(gnc_random[p] - ransac_random[p] for p in P_LIST)),
        claim="parent: GNC-GM keeps RANSAC-1000-level random-outlier success")
    v["gnc_speedup_vs_ransac1000_p07"] = dict(
        value=float(speedup(0.7)),
        claim="parent: GNC-GM ~27-30x faster than RANSAC-1000 (p50)")

    coh = {m: {p: _cell(pooled, "coherent", p, m) for p in P_LIST} for m in METHODS}
    v["coherent_60_70_all_methods_zero"] = dict(
        value=float(max(max(coh[m][p] for m in METHODS) for p in (0.6, 0.7))),
        claim="parent: coherent 60-70% -> all methods 0% (identifiability limit)")
    v["coherent_50_gnc_partial"] = dict(
        value=float(coh["GNC-GM"][0.5]),
        claim="parent: coherent 50% -> IRLS 55% / GNC 65% in redkitchen, RANSAC 0%")
    v["coherent_50_ransac1000"] = dict(
        value=float(coh["RANSAC-1000"][0.5]),
        claim="parent: RANSAC-1000 0% at coherent 50%")

    b_succ = {m: min(_cell(pooled, "boundary", p, m) for p in P_LIST) for m in METHODS}
    v["boundary_all_methods_success_min"] = dict(
        value=float(min(b_succ.values())),
        claim="parent: boundary outliers -> all methods 100% (consistency, no breakdown)")
    gnc_b = _cell(pooled, "boundary", 0.7, "GNC-GM", "rot_median")
    ran_b = _cell(pooled, "boundary", 0.7, "RANSAC-1000", "rot_median")
    v["boundary_p07_gnc_rot_minus_ransac"] = dict(
        value=float(gnc_b - ran_b),
        gnc_rot=float(gnc_b), ransac_rot=float(ran_b),
        claim="parent: boundary 70% GNC median rot 0.25 deg vs RANSAC 0.74 deg")
    v["clean_gnc_success_p0"] = dict(
        value=float(gnc_random[0.0]),
        claim="parent: clean base correspondences -> everyone succeeds at p=0")
    v["clean_gnc_rot_p0"] = dict(
        value=float(_cell(pooled, "random", 0.0, "GNC-GM", "rot_median")),
        claim="parent: clean base noise floor ~0.2 deg median rotation")

    # pass/fail with tolerances broad enough to be about the claim, not the scene
    v["holds"] = dict(
        random_gnc_success_ge_95=(v["random_gnc_success_min"]["value"] >= 0.95),
        random_gnc_parity_ge_m005=(v["random_gnc_parity_with_ransac1000"]["value"] >= -0.05),
        gnc_speedup_ge_10x=(v["gnc_speedup_vs_ransac1000_p07"]["value"] >= 10.0),
        coherent_60_70_all_zero=(v["coherent_60_70_all_methods_zero"]["value"] <= 1e-9),
        boundary_all_success_ge_95=(v["boundary_all_methods_success_min"]["value"] >= 0.95),
        boundary_gnc_rot_le_ransac=(v["boundary_p07_gnc_rot_minus_ransac"]["value"] <= 0.0),
    )
    v["all_hold"] = bool(all(v["holds"].values()))
    return v


# ---------------------------------------------------------------------------
# Markdown / figure
# ---------------------------------------------------------------------------
def _table(rows, otype, key, fmt="{:.2f}", scene=None):
    scale = 100.0 if key == "success" else 1.0
    Ls = ["| method | " + " | ".join(f"{int(100 * p)}%" for p in P_LIST) + " |",
          "|---|" + "|".join(["---"] * len(P_LIST)) + "|"]
    for m in METHODS:
        cells = []
        for p in P_LIST:
            sel = [r for r in rows if r["outlier_type"] == otype and abs(r["p"] - p) < 1e-9
                   and r["method"] == m and (scene is None or r["scene"] == scene)]
            cells.append(fmt.format(scale * sel[0][key]))
        Ls.append(f"| {m} | " + " | ".join(cells) + " |")
    return Ls


def write_markdown(rows, pooled, chosen_by_scene, verdicts, parent):
    Ls = ["# 3DMatch cross-scene robustness check (2 unseen scenes)", ""]
    Ls.append("The parent table (`experiments/real/threedmatch/`) concluded on redkitchen and "
              "sun3d-home_at that (i) GNC-GM keeps RANSAC-level success at 0-70% random "
              "outliers while being ~27-30x faster than RANSAC-1000, (ii) coherent outliers at "
              "60-70% defeat every method (model identifiability), and (iii) boundary outliers "
              "are harmless (all methods 100%, GNC lower median rotation error). "
              "This run repeats the identical protocol on two *unseen* official scenes.")
    Ls.append("")
    Ls.append(f"Scenes: `{SCENES[0]}` (hotel, 55 fragments) and `{SCENES[1]}` "
              "(home_md, 60 fragments). "
              f"N={N}, c={C} m, tau={TAU} m, voxel={VOXEL} m, {TRIALS} trials/cell, "
              f"{N_PAIRS} high-overlap pairs/scene, p=0-0.7, 3 outlier types, 5 methods. "
              "Base correspondences are GT-NN oracle matches on real fragments "
              "(real scan/occlusion residual), not a learned descriptor.")
    Ls.append("")
    Ls.append("**GT direction**: `verify_gt_direction_extra.py` re-checked both new scenes; "
              "the stored 4x4 maps id2->id1 (inverse median NN distance << forward), so the "
              "runner inverts it exactly as the parent does. See "
              "`results/verify_gt_direction_extra.json`.")
    Ls.append("")

    for scene in SCENES:
        short = SCENE_SHORT[scene]
        Ls.append(f"## {short}")
        Ls.append("")
        Ls.append("Selected pairs: " + ", ".join(
            f"{p['id1']}-{p['id2']} (overlap {100 * p['overlap']:.0f}%, "
            f"med res {1000 * p['med_res']:.1f} mm)" for p in chosen_by_scene[scene]) + ".")
        Ls.append("")
        Ls.append("### random outliers: success rate (%)")
        Ls.append("")
        Ls += _table(rows, "random", "success", "{:.0f}", scene=scene)
        Ls.append("")
        Ls.append("### random outliers: median rotation error (deg)")
        Ls.append("")
        Ls += _table(rows, "random", "rot_median", "{:.2f}", scene=scene)
        Ls.append("")
        Ls.append("### random outliers: runtime p50 (ms)")
        Ls.append("")
        Ls += _table(rows, "random", "time_ms_p50", "{:.2f}", scene=scene)
        Ls.append("")
        Ls.append("### coherent outliers: success rate (%)")
        Ls.append("")
        Ls += _table(rows, "coherent", "success", "{:.0f}", scene=scene)
        Ls.append("")
        Ls.append("### boundary outliers: median rotation error (deg)")
        Ls.append("")
        Ls += _table(rows, "boundary", "rot_median", "{:.3f}", scene=scene)
        Ls.append("")

    Ls.append("## Pooled over the two extra scenes (macro average of scene cells)")
    Ls.append("")
    Ls.append("### random outliers: success rate (%)")
    Ls.append("")
    Ls += _table(pooled, "random", "success", "{:.0f}")
    Ls.append("")
    Ls.append("### random outliers: median rotation error (deg)")
    Ls.append("")
    Ls += _table(pooled, "random", "rot_median", "{:.2f}")
    Ls.append("")
    Ls.append("### random outliers: runtime p50 (ms)")
    Ls.append("")
    Ls += _table(pooled, "random", "time_ms_p50", "{:.2f}")
    Ls.append("")
    Ls.append("### coherent outliers: success rate (%)")
    Ls.append("")
    Ls += _table(pooled, "coherent", "success", "{:.0f}")
    Ls.append("")
    Ls.append("### boundary outliers: success rate (%)")
    Ls.append("")
    Ls += _table(pooled, "boundary", "success", "{:.0f}")
    Ls.append("")
    Ls.append("### boundary outliers: median rotation error (deg)")
    Ls.append("")
    Ls += _table(pooled, "boundary", "rot_median", "{:.3f}")
    Ls.append("")

    Ls.append("## Do the parent two-scene conclusions hold?")
    Ls.append("")
    Ls.append("| check | parent claim | extra-scene measurement | holds |")
    Ls.append("|---|---|---|---|")
    for k, ok in verdicts["holds"].items():
        key = {"random_gnc_success_ge_95": "random_gnc_success_min",
               "random_gnc_parity_ge_m005": "random_gnc_parity_with_ransac1000",
               "gnc_speedup_ge_10x": "gnc_speedup_vs_ransac1000_p07",
               "coherent_60_70_all_zero": "coherent_60_70_all_methods_zero",
               "boundary_all_success_ge_95": "boundary_all_methods_success_min",
               "boundary_gnc_rot_le_ransac": "boundary_p07_gnc_rot_minus_ransac"}[k]
        meas = verdicts[key]["value"]
        if key == "gnc_speedup_vs_ransac1000_p07":
            meas_s = f"{meas:.1f}x"
        elif key == "boundary_p07_gnc_rot_minus_ransac":
            meas_s = (f"{verdicts[key]['gnc_rot']:.3f} deg (GNC) vs "
                      f"{verdicts[key]['ransac_rot']:.3f} deg (RANSAC)")
        elif "success" in key or key.startswith("coherent_60_70"):
            meas_s = f"{100 * meas:.1f}%"
        else:
            meas_s = f"{meas:.3f}"
        Ls.append(f"| {k} | {verdicts[key]['claim']} | {meas_s} | {'YES' if ok else 'NO'} |")
    Ls.append("")
    if parent is not None:
        Ls.append("### Parent vs extra scenes, same pooled cells")
        Ls.append("")
        Ls.append("| cell | parent (2 scenes) | extra (2 scenes) |")
        Ls.append("|---|---|---|")
        for otype, p, m, label in [
                ("random", 0.0, "GNC-GM", "random 0% GNC success"),
                ("random", 0.7, "GNC-GM", "random 70% GNC success"),
                ("random", 0.7, "IRLS-GM", "random 70% IRLS success"),
                ("random", 0.7, "RANSAC-1000", "random 70% RANSAC-1000 success"),
                ("coherent", 0.5, "GNC-GM", "coherent 50% GNC success"),
                ("coherent", 0.6, "GNC-GM", "coherent 60% GNC success"),
                ("coherent", 0.7, "GNC-GM", "coherent 70% GNC success"),
                ("boundary", 0.7, "GNC-GM", "boundary 70% GNC med rot (deg)")]:
            pk = "rot_median" if label.startswith("boundary") else "success"
            pv = _cell(parent["rows"], otype, p, m, pk)
            ev = _cell(pooled, otype, p, m, pk)
            Ls.append(f"| {label} | {pv:.3f} | {ev:.3f} |")
        Ls.append("")
        for p in (0.0, 0.3, 0.5, 0.7):
            pr = _cell(parent["rows"], "random", p, "RANSAC-1000", "time_ms_p50")
            pg = _cell(parent["rows"], "random", p, "GNC-GM", "time_ms_p50")
            er = _cell(pooled, "random", p, "RANSAC-1000", "time_ms_p50")
            eg = _cell(pooled, "random", p, "GNC-GM", "time_ms_p50")
            Ls.append(f"* random p={int(100 * p)}% p50: parent RANSAC {pr:.1f} ms / "
                      f"GNC {pg:.2f} ms ({pr / pg:.1f}x); extra RANSAC {er:.1f} ms / "
                      f"GNC {eg:.2f} ms ({er / eg:.1f}x).")
        Ls.append("")
    Ls.append(f"**Overall: parent conclusions {'hold' if verdicts['all_hold'] else 'do NOT all hold'} "
              "on the two extra scenes.**")
    Ls.append("")
    Ls.append("## Caveats")
    Ls.append("")
    Ls.append("* GT-NN base correspondences use the ground-truth transform to establish "
              "overlap; they are oracle matches, not a learned descriptor. Real scanner "
              "noise/occlusion/partial overlap are present (natural inlier fraction below 1).")
    Ls.append("* Outliers are injected in the observation space, exactly as in the parent "
              "and synthetic protocols; this tests the robust outer layer, not the front-end.")
    Ls.append("* Pooled numbers are macro averages over two scenes, so they describe these "
              "two extra scenes only; they do not replace the full 3DMatch benchmark "
              "(this is a 4-scene cross-scene check in total).")
    return "\n".join(Ls) + "\n"


def make_figure(pooled):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"Kabsch": "tab:gray", "RANSAC-1000": "tab:orange",
              "RANSAC-adaptive": "tab:green", "IRLS-GM": "tab:blue", "GNC-GM": "tab:red"}
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    def ys(otype, m, key, scene=None):
        out = []
        for p in P_LIST:
            sel = [r for r in pooled if r["outlier_type"] == otype
                   and abs(r["p"] - p) < 1e-9 and r["method"] == m]
            out.append(sel[0][key])
        return out

    for m in METHODS:
        axes[0, 0].plot([100 * p for p in P_LIST],
                        [100 * y for y in ys("random", m, "success")], "o-",
                        color=colors[m], label=m, lw=1.5, ms=4)
    axes[0, 0].set_title("pooled (2 extra scenes): random outliers")
    axes[0, 0].set_xlabel("outlier fraction (%)")
    axes[0, 0].set_ylabel("success (%)")
    axes[0, 0].set_ylim(-3, 103)
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].legend(fontsize=7)

    for m in METHODS:
        axes[0, 1].plot([100 * p for p in P_LIST],
                        [100 * y for y in ys("coherent", m, "success")], "o-",
                        color=colors[m], lw=1.5, ms=4)
    axes[0, 1].set_title("pooled: coherent outliers")
    axes[0, 1].set_xlabel("outlier fraction (%)")
    axes[0, 1].set_ylabel("success (%)")
    axes[0, 1].set_ylim(-3, 103)
    axes[0, 1].grid(alpha=0.3)

    for m in METHODS:
        axes[0, 2].semilogy([100 * p for p in P_LIST],
                            ys("random", m, "time_ms_p50"), "o-",
                            color=colors[m], lw=1.5, ms=4)
    axes[0, 2].set_title("pooled: runtime p50 (random outliers)")
    axes[0, 2].set_xlabel("outlier fraction (%)")
    axes[0, 2].set_ylabel("time (ms, log)")
    axes[0, 2].grid(alpha=0.3, which="both")

    for m in METHODS:
        axes[1, 0].semilogy([100 * p for p in P_LIST],
                            np.maximum(ys("boundary", m, "rot_median"), 1e-3), "o-",
                            color=colors[m], lw=1.5, ms=4)
    axes[1, 0].set_title("pooled: boundary outliers, median rot error")
    axes[1, 0].set_xlabel("outlier fraction (%)")
    axes[1, 0].set_ylabel("median rotation error (deg, log)")
    axes[1, 0].grid(alpha=0.3, which="both")

    spe = [ys("random", "RANSAC-1000", "time_ms_p50")[i]
           / ys("random", "GNC-GM", "time_ms_p50")[i] for i in range(len(P_LIST))]
    axes[1, 1].bar([100 * p for p in P_LIST], spe, width=6, color="tab:red", alpha=0.8)
    axes[1, 1].set_title("pooled: GNC speedup vs RANSAC-1000")
    axes[1, 1].set_xlabel("outlier fraction (%)")
    axes[1, 1].set_ylabel("speedup (x, p50)")
    axes[1, 1].grid(alpha=0.3, axis="y")

    for m in METHODS:
        axes[1, 2].semilogy([100 * p for p in P_LIST],
                            np.maximum(ys("random", m, "rot_median"), 1e-3), "o-",
                            color=colors[m], lw=1.5, ms=4)
    axes[1, 2].set_title("pooled: random outliers, median rot error")
    axes[1, 2].set_xlabel("outlier fraction (%)")
    axes[1, 2].set_ylabel("median rotation error (deg, log)")
    axes[1, 2].grid(alpha=0.3, which="both")

    fig.suptitle("3DMatch cross-scene check: 2 unseen scenes, oracle GT-NN + controlled outliers")
    fig.tight_layout()
    fig.savefig(OUT / "extra_scenes.png", dpi=160)
    plt.close(fig)


def main():
    methods = make_methods()
    all_rows, all_raw = [], []
    chosen_by_scene = {}
    for scene in SCENES:
        rows, raw, chosen, all_pairs = run_scene(scene, methods)
        all_rows += rows
        all_raw += raw
        chosen_by_scene[scene] = chosen
        (OUT / f"pairs_{SCENE_SHORT[scene]}.json").write_text(json.dumps(
            dict(selected=[{k: v for k, v in p.items() if k != "Tinv"} for p in chosen],
                 n_all=len(all_pairs)), indent=2))
    pooled = pooled_rows(all_rows)
    parent = load_parent_reference()
    verdicts = make_verdicts(pooled, parent)
    cm.json_dump(OUT / "extra_scenes.json", dict(
        config=dict(scenes=SCENES, scene_short=SCENE_SHORT,
                    parent_two_scene=PARENT_TWO, n=N, c=C, tau=TAU, voxel=VOXEL,
                    p=P_LIST, trials=TRIALS, n_pairs=N_PAIRS, methods=METHODS,
                    outlier_types=OUTLIER_TYPES, success_rot_deg=5.0,
                    success_trans_m=0.1, data_root=str(ROOT),
                    seed=BASE_SEED),
        scenes={s: dict(selected=[{k: v for k, v in p.items() if k != "Tinv"}
                                  for p in chosen_by_scene[s]])
                for s in SCENES},
        rows=all_rows,
        pooled=pooled,
        verdict=verdicts,
        parent_reference=parent))
    with open(OUT / "extra_scenes_raw.jsonl", "w") as f:
        for rec in all_raw:
            f.write(json.dumps(rec) + "\n")
    (OUT / "extra_scenes.md").write_text(
        write_markdown(all_rows, pooled, chosen_by_scene, verdicts, parent))
    make_figure(pooled)
    print("parent conclusions all hold:", verdicts["all_hold"], flush=True)
    print("wrote", OUT / "extra_scenes.json", "etc.", flush=True)


if __name__ == "__main__":
    main()
