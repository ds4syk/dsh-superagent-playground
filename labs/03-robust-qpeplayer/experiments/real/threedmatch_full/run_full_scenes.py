#!/usr/bin/env python3
"""Full official 3DMatch geometric-registration set: robust 3D-3D table.

Extends the parent two-scene table (``experiments/real/threedmatch``) and the
cross-scene check (``experiments/real/threedmatch_extra``) to **all eight**
official scene-fragment sets distributed with 3DMatch:

  * 7-scenes-redkitchen
  * sun3d-home_at-home_at_scan1_2013_jan_1
  * sun3d-hotel_uc-scan3
  * sun3d-home_md-home_md_scan9_2012_sep_30
  * sun3d-hotel_umd-maryland_hotel1            (new)
  * sun3d-hotel_umd-maryland_hotel3            (new)
  * sun3d-mit_76_studyroom-76-1studyroom2      (new)
  * sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika (new)

Protocol (identical to the parent, so numbers are comparable)
-------------------------------------------------------------
* Official TSDF-fused fragments, voxel-downsampled to 0.03 m.
* ``gt.log`` 4x4 maps id2 -> id1 (re-verified on every scene by
  ``verify_gt_direction_full.py``), so the runner inverts it.
* Base correspondences: GT-mapped nearest neighbours (oracle matches with
  real scan / occlusion / sampling residual), kept when residual < tau=0.06 m.
* N=300 correspondences per trial, 20 trials/cell, each trial samples one of
  the 8 highest-overlap fragment pairs of the scene.
* Outlier fraction p in {0,0.1,...,0.7}; 3 outlier types (random target
  point, one coherent competing rigid transform, boundary residual); c=0.05 m.
* Methods: Kabsch / RANSAC-1000 / RANSAC-adaptive / IRLS-GM / GNC-GM.
* Success: rotation < 5 deg and translation < 0.1 m.

Outputs (``results/``): ``full_scenes.json``, ``full_scenes.md``,
``full_scenes.png``, ``full_scenes_raw.jsonl``, ``pairs_<short>.json`` and
``logs/``.

Run (Slurm):
  srun --partition=cpu --time=00:50:00 --ntasks=1 --cpus-per-task=4 \
    /home/syk/miniconda3/envs/WM/bin/python run_full_scenes.py 2>&1 | \
    tee results/logs/full_scenes_$(date +%Y%m%d_%H%M%S).log
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import full_common as F  # noqa: E402  (re-exports parent load_3dmatch + common)
L = F.L
cm = F.cm

ROOT = Path("/mnt/storage1/syk/data/3dmatch_fragments")
OUT = HERE / "results"
(OUT / "logs").mkdir(parents=True, exist_ok=True)

SCENES = [
    "7-scenes-redkitchen",
    "sun3d-home_at-home_at_scan1_2013_jan_1",
    "sun3d-hotel_uc-scan3",
    "sun3d-home_md-home_md_scan9_2012_sep_30",
    "sun3d-hotel_umd-maryland_hotel1",
    "sun3d-hotel_umd-maryland_hotel3",
    "sun3d-mit_76_studyroom-76-1studyroom2",
    "sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika",
]
SCENE_SHORT = {
    "7-scenes-redkitchen": "redkitchen",
    "sun3d-home_at-home_at_scan1_2013_jan_1": "sun3d-home_at",
    "sun3d-hotel_uc-scan3": "sun3d-hotel_uc",
    "sun3d-home_md-home_md_scan9_2012_sep_30": "sun3d-home_md",
    "sun3d-hotel_umd-maryland_hotel1": "sun3d-hotel1",
    "sun3d-hotel_umd-maryland_hotel3": "sun3d-hotel3",
    "sun3d-mit_76_studyroom-76-1studyroom2": "sun3d-studyroom",
    "sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika": "sun3d-mit_lab_hj",
}
NEW_SCENES = SCENES[4:]

N = 300
C = 0.05
TAU = 0.06
VOXEL = 0.03
P_LIST = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
TRIALS = 20
N_PAIRS = 8
METHODS = ["Kabsch", "RANSAC-1000", "RANSAC-adaptive", "IRLS-GM", "GNC-GM"]
OUTLIER_TYPES = ["random", "coherent", "boundary"]
BASE_SEED = 20261127


def _seed(*parts):
    return np.random.SeedSequence([BASE_SEED, *[int(x) for x in parts]])


def select_pairs(clouds, gt_pairs, n_probe=2000, max_pairs=N_PAIRS, seed=0):
    """Top-``max_pairs`` pairs by the *uncapped* GT-NN overlap (see full_common)."""
    rng = np.random.default_rng(seed)
    rows = []
    for id1, id2, n_gt, T in gt_pairs:
        if id1 not in clouds or id2 not in clouds:
            continue
        Ti = np.linalg.inv(T)  # gt.log maps id2 -> id1
        s = F.overlap_fraction(clouds[id1], clouds[id2], Ti, rng,
                               n_probe=n_probe, tau=TAU)
        if s["n_kept"] < 3:
            continue
        rows.append(dict(id1=id1, id2=id2, n_gt=n_gt, overlap=s["overlap"],
                         med_res=s["med_res"], Tinv=Ti))
    rows.sort(key=lambda r: (r["overlap"], -r["med_res"]), reverse=True)
    return rows[:max_pairs], rows


def make_methods():
    return F.make_methods(c=C)


def clean_pair_sample(clouds, pair, rng, n=N):
    s = F.clean_pair_sample(clouds, pair, rng, tau=TAU, n=n)
    if s is None:
        return None
    return s["x"], s["y"], s["d"], s["Rgt"], s["tgt"]


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
    t0 = time.time()
    clouds = L.load_scene(ROOT / scene, voxel=VOXEL)
    gt_pairs = L.parse_gt_log(ROOT / f"{scene}-evaluation" / "gt.log")
    chosen, all_pairs = select_pairs(clouds, gt_pairs)
    print(f"[{scene}] {len(clouds)} fragments, {len(gt_pairs)} gt pairs, "
          f"selected {len(chosen)}: " +
          ", ".join(f"{p['id1']}-{p['id2']}({p['overlap']:.2f})" for p in chosen),
          flush=True)
    rows, raw = [], []
    for otype in OUTLIER_TYPES:
        for p in P_LIST:
            per = {m: dict(re=[], te=[], time=[], base_in=[]) for m in METHODS}
            for trial in range(TRIALS):
                rng = np.random.default_rng(_seed(SCENES.index(scene), int(round(100 * p)),
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
                    rrng = np.random.default_rng(_seed(99, SCENES.index(scene),
                                                       int(round(100 * p)), trial))
                    t1 = time.perf_counter()
                    R, t, info = methods[name](x, y, rrng)
                    dt = time.perf_counter() - t1
                    re, te = cm.pose_errors(R, t, Rgt, tgt)
                    rec = dict(scene=scene, scene_short=SCENE_SHORT[scene],
                               pair=f"{pair['id1']}-{pair['id2']}", p=p,
                               outlier_type=otype, trial=trial, method=name,
                               n=len(x), base_inlier=base_in,
                               success=cm.success(re, te), rot=re, trans=te,
                               time_ms=1000.0 * dt,
                               ransac_iters=info.get("n_iters"),
                               ransac_inliers=info.get("best_count"))
                    raw.append(rec)
                    per[name]["re"].append(re)
                    per[name]["te"].append(te)
                    per[name]["time"].append(dt)
                    per[name]["base_in"].append(base_in)
            for m in METHODS:
                a = per[m]
                s = cm.summarize_errors(a["re"], a["te"], a["time"])
                rows.append(dict(scene=scene, scene_short=SCENE_SHORT[scene], p=p,
                                 outlier_type=otype, method=m,
                                 n_trials=len(a["re"]),
                                 base_inlier=float(np.mean(a["base_in"])), **s))
            print(f"[{scene}] {otype:8s} p={p:.1f} done ({time.time()-t0:.1f}s)", flush=True)
    return rows, raw, chosen, all_pairs


def pooled_rows(raw):
    """Pool raw trials across scenes per (p, type, method); equal scene weight
    because each scene contributes the same number of trials."""
    out = []
    for otype in OUTLIER_TYPES:
        for p in P_LIST:
            for m in METHODS:
                sel = [r for r in raw if r["p"] == p and r["outlier_type"] == otype
                       and r["method"] == m]
                s = cm.summarize_errors([r["rot"] for r in sel],
                                        [r["trans"] for r in sel],
                                        [r["time_ms"] / 1000.0 for r in sel])
                out.append(dict(p=p, outlier_type=otype, method=m,
                                n_trials=len(sel),
                                base_inlier=float(np.mean([r["base_inlier"] for r in sel])),
                                **s))
    return out


def fmt_table(rows, otype, column, scenes=None):
    """Markdown success/rot table for one outlier type and metric column."""
    Ls = []
    Ls.append("| method | " + " | ".join(f"{int(100 * p)}%" for p in P_LIST) + " |")
    Ls.append("|---|" + "|".join(["---"] * len(P_LIST)) + "|")
    for m in METHODS:
        cells = []
        for p in P_LIST:
            sel = [r for r in rows if r["outlier_type"] == otype and r["method"] == m
                   and abs(r["p"] - p) < 1e-9 and (scenes is None or r["scene"] in scenes)]
            if not sel:
                cells.append("—")
            else:
                v = float(np.mean([r[column] for r in sel])) if column == "success" \
                    else float(np.median([r[column] for r in sel]))
                cells.append(f"{100 * v:.0f}" if column == "success" else f"{v:.2f}")
        Ls.append(f"| {m} | " + " | ".join(cells) + " |")
    return "\n".join(Ls)


def write_markdown(rows, pooled, chosen_by_scene):
    Ls = ["# 3DMatch full official set: robust 3D-3D registration", ""]
    Ls.append("All eight official 3DMatch geometric-registration scene-fragment sets. "
              "Base correspondences are GT-mapped nearest neighbours on real overlapping "
              "fragments (oracle matches with real scan/occlusion residual); controlled "
              f"outliers are then injected. N={N}, c={C} m, tau={TAU} m, voxel={VOXEL} m, "
              f"{TRIALS} trials/cell, trials sample one of {N_PAIRS} high-overlap fragment "
              "pairs per scene.")
    Ls.append("")
    Ls.append("GT direction was verified on every scene (`verify_gt_direction_full.json`): "
              "the stored 4x4 maps id2->id1, so the runner inverts it. "
              "Base matches are oracle correspondences, not a learned descriptor.")
    Ls.append("")
    Ls.append("## Pooled across 8 scenes")
    Ls.append("")
    Ls.append(f"Pooled over {len(SCENES)} scenes x {TRIALS} trials/cell "
              "(equal scene weight).")
    Ls.append("")
    for otype in OUTLIER_TYPES:
        Ls.append(f"### {otype}: success rate (%)")
        Ls.append("")
        Ls.append(fmt_table(pooled, otype, "success"))
        Ls.append("")
        Ls.append(f"### {otype}: median rotation error (deg)")
        Ls.append("")
        Ls.append(fmt_table(pooled, otype, "rot_median"))
        Ls.append("")
    Ls.append("### runtime p50 (ms), random outliers")
    Ls.append("")
    Ls.append(fmt_table(pooled, "random", "time_ms_p50"))
    Ls.append("")
    Ls.append("## Per-scene")
    Ls.append("")
    for scene in SCENES:
        short = SCENE_SHORT[scene]
        Ls.append(f"### {short} (`{scene}`)")
        Ls.append("")
        Ls.append("Selected pairs: " + ", ".join(
            f"{p['id1']}-{p['id2']} (overlap {100 * p['overlap']:.0f}%, "
            f"med res {1000 * p['med_res']:.1f} mm)" for p in chosen_by_scene[scene][:4])
            + " ...")
        Ls.append("")
        sub = [r for r in rows if r["scene"] == scene]
        for otype in OUTLIER_TYPES:
            Ls.append(f"#### {otype}: success rate (%)")
            Ls.append("")
            Ls.append(fmt_table(sub, otype, "success"))
            Ls.append("")
        Ls.append("")
    Ls.append("## Caveats")
    Ls.append("")
    Ls.append("* GT-NN base correspondences use the ground-truth transform to establish "
              "overlap: they are oracle matches, not a learned descriptor. Real scanner "
              "noise, occlusion and partial overlap are present.")
    Ls.append("* Coherent outliers at >=60% are expected to defeat every method "
              "(model identifiability), exactly as in the synthetic protocol.")
    Ls.append("* For low-overlap pairs the valid GT-NN set can be smaller than N; the "
              "runner resamples with replacement. This is only relevant for the "
              "high-overlap table in rare cases; the dedicated low/mid-overlap study "
              "reports the raw counts.")
    return "\n".join(Ls) + "\n"


def make_figure(pooled, rows, chosen_by_scene):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"Kabsch": "tab:gray", "RANSAC-1000": "tab:orange",
              "RANSAC-adaptive": "tab:green", "IRLS-GM": "tab:blue",
              "GNC-GM": "tab:red"}

    def get(rows_, otype, m, col):
        return [next(r for r in rows_ if r["outlier_type"] == otype and r["method"] == m
                     and abs(r["p"] - p) < 1e-9)[col] for p in P_LIST]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    ax = axes[0, 0]
    for m in METHODS:
        ax.plot([100 * p for p in P_LIST], [100 * y for y in get(pooled, "random", m, "success")],
                "o-", color=colors[m], label=m, lw=1.5, ms=4)
    ax.set_title("Pooled (8 scenes): random outliers")
    ax.set_xlabel("outlier fraction (%)"); ax.set_ylabel("success (%)")
    ax.set_ylim(-3, 103); ax.grid(alpha=0.3); ax.legend(fontsize=7)
    ax = axes[0, 1]
    for m in METHODS:
        ax.plot([100 * p for p in P_LIST], [100 * y for y in get(pooled, "coherent", m, "success")],
                "o-", color=colors[m], label=m, lw=1.5, ms=4)
    ax.set_title("Pooled: coherent outliers")
    ax.set_xlabel("outlier fraction (%)"); ax.set_ylabel("success (%)")
    ax.set_ylim(-3, 103); ax.grid(alpha=0.3)
    ax = axes[0, 2]
    for m in METHODS:
        ys = np.maximum(get(pooled, "boundary", m, "rot_median"), 1e-3)
        ax.semilogy([100 * p for p in P_LIST], ys, "o-", color=colors[m], lw=1.5, ms=4)
    ax.set_title("Pooled: boundary outliers, rot error")
    ax.set_xlabel("outlier fraction (%)"); ax.set_ylabel("median rotation error (deg)")
    ax.grid(alpha=0.3, which="both")
    # per-scene success at p=0.5 random, grouped bars
    ax = axes[1, 0]
    width = 0.16
    xpos = np.arange(len(SCENES))
    for k, m in enumerate(METHODS):
        vals = []
        for scene in SCENES:
            r = next(z for z in rows if z["scene"] == scene and z["outlier_type"] == "random"
                     and z["method"] == m and abs(z["p"] - 0.5) < 1e-9)
            vals.append(100 * r["success"])
        ax.bar(xpos + (k - 2) * width, vals, width, color=colors[m], label=m)
    ax.set_xticks(xpos)
    ax.set_xticklabels([SCENE_SHORT[s] for s in SCENES], rotation=35, ha="right", fontsize=7)
    ax.set_title("Random outliers p=0.5: per-scene success")
    ax.set_ylabel("success (%)"); ax.set_ylim(0, 105); ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=6, ncol=2)
    # per-scene coherent success p=0.5
    ax = axes[1, 1]
    for k, m in enumerate(METHODS):
        vals = []
        for scene in SCENES:
            r = next(z for z in rows if z["scene"] == scene and z["outlier_type"] == "coherent"
                     and z["method"] == m and abs(z["p"] - 0.5) < 1e-9)
            vals.append(100 * r["success"])
        ax.bar(xpos + (k - 2) * width, vals, width, color=colors[m])
    ax.set_xticks(xpos)
    ax.set_xticklabels([SCENE_SHORT[s] for s in SCENES], rotation=35, ha="right", fontsize=7)
    ax.set_title("Coherent outliers p=0.5: per-scene success")
    ax.set_ylabel("success (%)"); ax.set_ylim(0, 105); ax.grid(alpha=0.3, axis="y")
    # runtime
    ax = axes[1, 2]
    for m in METHODS:
        ax.semilogy([100 * p for p in P_LIST], get(pooled, "random", m, "time_ms_p50"),
                    "o-", color=colors[m], lw=1.5, ms=4)
    ax.set_title("Pooled: runtime p50")
    ax.set_xlabel("outlier fraction (%)"); ax.set_ylabel("time (ms, log)")
    ax.grid(alpha=0.3, which="both")
    fig.suptitle("3DMatch full official set (8 scenes): GT-NN oracle correspondences + "
                 "controlled outliers")
    fig.tight_layout()
    fig.savefig(OUT / "full_scenes.png", dpi=160)
    plt.close(fig)


def main():
    methods = make_methods()
    all_rows, all_raw = [], []
    chosen_by_scene = {}
    for scene in SCENES:
        if not (ROOT / scene).is_dir():
            raise SystemExit(f"missing scene dir {ROOT / scene}")
        rows, raw, chosen, all_pairs = run_scene(scene, methods)
        all_rows += rows
        all_raw += raw
        chosen_by_scene[scene] = chosen
        (OUT / f"pairs_{SCENE_SHORT[scene]}.json").write_text(json.dumps(
            dict(scene=scene,
                 selected=[{k: v for k, v in p.items() if k != "Tinv"} for p in chosen],
                 n_all=len(all_pairs),
                 overlap_median_all=float(np.median([p["overlap"] for p in all_pairs]))),
            indent=2))
    pooled = pooled_rows(all_raw)
    cm.json_dump(OUT / "full_scenes.json", dict(config=dict(
        scenes=SCENES, scene_short=SCENE_SHORT, new_scenes=NEW_SCENES,
        n=N, c=C, tau=TAU, voxel=VOXEL, p=P_LIST, trials=TRIALS,
        n_pairs=N_PAIRS, methods=METHODS, outlier_types=OUTLIER_TYPES,
        data_root=str(ROOT), gt_direction="id2->id1, inverted by runner"),
        per_scene=all_rows, pooled=pooled,
        selected_pairs={SCENE_SHORT[s]: [{k: v for k, v in p.items() if k != "Tinv"}
                                          for p in chosen_by_scene[s]] for s in SCENES}))
    with open(OUT / "full_scenes_raw.jsonl", "w") as f:
        for rec in all_raw:
            f.write(json.dumps(rec) + "\n")
    (OUT / "full_scenes.md").write_text(write_markdown(all_rows, pooled, chosen_by_scene))
    make_figure(pooled, all_rows, chosen_by_scene)
    print("wrote", OUT / "full_scenes.{json,md,png}")


if __name__ == "__main__":
    main()
