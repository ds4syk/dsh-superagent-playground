#!/usr/bin/env python3
"""3DMatch real-fragment robust 3D-3D registration table.

Pipeline
--------
* Two 3DMatch geometric-registration scenes (redkitchen, sun3d-home_at)
  are loaded from the official TSDF-fused fragment zips.
* Per pair, the GT transform is inverted (the file maps id2->id1) and used
  to build real-overlap nearest-neighbour correspondences: source points
  are mapped by GT and matched to the nearest target point.  The base
  pairs are therefore *oracle* correspondences with real scan/occlusion
  noise; this isolates the robust outer layer (same role as ModelNet40).
* Controlled outliers are then injected at fraction p:
  random target-scene points, one coherent competing rigid transform, or
  boundary residuals e in [c^2/3, 3c^2].
* Methods: Kabsch / RANSAC-1000 / RANSAC-adaptive / IRLS-GM / GNC-GM.
* Success: rotation < 5 deg and translation < 0.1 m.

This is NOT a descriptor front-end.  A separate, clearly labelled
hand-crafted-descriptor run is in ``run_descriptor.py``.

Run (Slurm):
  srun --partition=cpu --time=00:40:00 --ntasks=1 --cpus-per-task=4 \
    /home/syk/miniconda3/envs/WM/bin/python run_threedmatch.py 2>&1 | \
    tee results/logs/threedmatch_$(date +%Y%m%d_%H%M%S).log
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

SCENES = ["7-scenes-redkitchen", "sun3d-home_at-home_at_scan1_2013_jan_1"]
SCENE_SHORT = {"7-scenes-redkitchen": "redkitchen",
               "sun3d-home_at-home_at_scan1_2013_jan_1": "sun3d-home_at"}
N = 300
C = 0.05
TAU = 0.06
VOXEL = 0.03
P_LIST = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
TRIALS = 20
N_PAIRS = 8
METHODS = ["Kabsch", "RANSAC-1000", "RANSAC-adaptive", "IRLS-GM", "GNC-GM"]
BASE_SEED = 20261127


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
    chosen = rows[:max_pairs]
    return chosen, rows


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
    if len(x) < n:  # pad by resampling real valid pairs (high-overlap pairs rarely hit this)
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
    for otype in ["random", "coherent", "boundary"]:
        for p in P_LIST:
            per = {m: dict(re=[], te=[], time=[], base_in=[]) for m in METHODS}
            for trial in range(TRIALS):
                rng = np.random.default_rng(_seed(scene == SCENES[1], int(round(100 * p)),
                                                  {"random": 1, "coherent": 2, "boundary": 3}[otype],
                                                  trial))
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
                           n_trials=len(a["re"]), base_inlier=float(np.mean(a["base_in"])),
                           **s)
                if otype == "coherent":
                    vals = [r.get("picked_outlier", False) for r in raw
                            if r["p"] == p and r["outlier_type"] == otype and r["method"] == m]
                    row["frac_picked_outlier"] = float(np.mean(vals)) if vals else None
                rows.append(row)
            print(f"[{scene}] {otype:8s} p={p:.1f} done "
                  f"({time.time()-t0:.1f}s)", flush=True)
    return rows, raw, chosen, all_pairs


def write_markdown(rows, chosen_by_scene):
    Ls = ["# 3DMatch real-fragment robust 3D-3D registration", ""]
    Ls.append("Two official 3DMatch geometric-registration scenes. Base correspondences are "
              "GT-mapped nearest neighbours on real overlapping fragments (oracle matches with "
              "real scan/occlusion residual); controlled outliers are then injected. "
              f"N={N}, c={C} m, tau={TAU} m, {TRIALS} trials/cell, trials sample one of "
              f"{N_PAIRS} high-overlap fragment pairs per scene.")
    Ls.append("")
    Ls.append("This is a real-geometry robust-outer-layer test, not a descriptor front-end. "
              "See `run_descriptor.py` for the no-GT-leakage hand-crafted descriptor variant.")
    Ls.append("")
    for scene in SCENES:
        short = SCENE_SHORT[scene]
        Ls.append(f"## {short}")
        Ls.append("")
        Ls.append("Selected pairs: " + ", ".join(
            f"{p['id1']}-{p['id2']} (overlap {100*p['overlap']:.0f}%, med res {1000*p['med_res']:.1f} mm)"
            for p in chosen_by_scene[scene][:4]) + " ...")
        Ls.append("")
        for otype in ["random", "coherent", "boundary"]:
            Ls.append(f"### {otype} outliers: success rate")
            Ls.append("")
            Ls.append("| method | " + " | ".join(f"{int(100*p)}%" for p in P_LIST) + " |")
            Ls.append("|---|" + "|".join(["---"] * len(P_LIST)) + "|")
            for m in METHODS:
                cells = []
                for p in P_LIST:
                    r = next(z for z in rows if z["scene"] == scene and z["outlier_type"] == otype
                             and z["method"] == m and abs(z["p"] - p) < 1e-9)
                    cells.append(f"{100*r['success']:.0f}")
                Ls.append(f"| {m} | " + " | ".join(cells) + " |")
            Ls.append("")
            Ls.append(f"### {otype} outliers: median rotation error (deg)")
            Ls.append("")
            Ls.append("| method | " + " | ".join(f"{int(100*p)}%" for p in P_LIST) + " |")
            Ls.append("|---|" + "|".join(["---"] * len(P_LIST)) + "|")
            for m in METHODS:
                cells = []
                for p in P_LIST:
                    r = next(z for z in rows if z["scene"] == scene and z["outlier_type"] == otype
                             and z["method"] == m and abs(z["p"] - p) < 1e-9)
                    cells.append(f"{r['rot_median']:.2f}")
                Ls.append(f"| {m} | " + " | ".join(cells) + " |")
            Ls.append("")
            Ls.append(f"### {otype} outliers: runtime p50 (ms)")
            Ls.append("")
            Ls.append("| method | " + " | ".join(f"{int(100*p)}%" for p in P_LIST) + " |")
            Ls.append("|---|" + "|".join(["---"] * len(P_LIST)) + "|")
            for m in METHODS:
                cells = []
                for p in P_LIST:
                    r = next(z for z in rows if z["scene"] == scene and z["outlier_type"] == otype
                             and z["method"] == m and abs(z["p"] - p) < 1e-9)
                    cells.append(f"{r['time_ms_p50']:.2f}")
                Ls.append(f"| {m} | " + " | ".join(cells) + " |")
            Ls.append("")
        Ls.append("")
    Ls.append("## Caveats")
    Ls.append("")
    Ls.append("* GT-NN base correspondences use the ground-truth transform to establish overlap; "
              "they are oracle matches, not a learned descriptor. Real scanner noise, occlusion, "
              "and partial overlap are present, so the clean success rate is not 100% for a "
              "naive Kabsch.")
    Ls.append("* The transform direction in `gt.log` was verified empirically: it maps fragment "
              "`id2` into `id1`; the runner inverts it. See `results/verify_gt_direction.json`.")
    Ls.append("* Coherent outliers at >=60% are expected to make the robust global optimum wrong "
              "(model identifiability), exactly as in the synthetic protocol.")
    return "\n".join(Ls) + "\n"


def make_figure(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"Kabsch": "tab:gray", "RANSAC-1000": "tab:orange",
              "RANSAC-adaptive": "tab:green", "IRLS-GM": "tab:blue", "GNC-GM": "tab:red"}
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for col, scene in enumerate(SCENES):
        for m in METHODS:
            ys = [next(z for z in rows if z["scene"] == scene and z["outlier_type"] == "random"
                       and z["method"] == m and abs(z["p"] - p) < 1e-9)["success"]
                  for p in P_LIST]
            axes[0, col].plot([100 * p for p in P_LIST], [100 * y for y in ys], "o-",
                              color=colors[m], label=m, lw=1.5, ms=4)
        axes[0, col].set_title(f"{SCENE_SHORT[scene]}: random outliers")
        axes[0, col].set_xlabel("outlier fraction (%)")
        axes[0, col].set_ylabel("success (%)")
        axes[0, col].set_ylim(-3, 103)
        axes[0, col].grid(alpha=0.3)
    axes[0, 0].legend(fontsize=7)
    for m in METHODS:
        ys = [next(z for z in rows if z["scene"] == SCENES[0] and z["outlier_type"] == "boundary"
                   and z["method"] == m and abs(z["p"] - p) < 1e-9)["rot_median"] for p in P_LIST]
        axes[1, 0].semilogy([100 * p for p in P_LIST], np.maximum(ys, 1e-3), "o-",
                            color=colors[m], label=m, lw=1.5, ms=4)
    axes[1, 0].set_title("redkitchen: boundary rot error")
    axes[1, 0].set_xlabel("outlier fraction (%)")
    axes[1, 0].set_ylabel("median rotation error (deg)")
    axes[1, 0].grid(alpha=0.3, which="both")
    for m in METHODS:
        ys = [next(z for z in rows if z["scene"] == SCENES[1] and z["outlier_type"] == "random"
                   and z["method"] == m and abs(z["p"] - p) < 1e-9)["time_ms_p50"] for p in P_LIST]
        axes[1, 1].semilogy([100 * p for p in P_LIST], ys, "o-", color=colors[m], label=m,
                            lw=1.5, ms=4)
    axes[1, 1].set_title("sun3d-home_at: runtime p50")
    axes[1, 1].set_xlabel("outlier fraction (%)")
    axes[1, 1].set_ylabel("time (ms, log)")
    axes[1, 1].grid(alpha=0.3, which="both")
    for m in METHODS:
        ys = [next(z for z in rows if z["scene"] == SCENES[0] and z["outlier_type"] == "coherent"
                   and z["method"] == m and abs(z["p"] - p) < 1e-9)["success"] for p in P_LIST]
        axes[1, 2].plot([100 * p for p in P_LIST], [100 * y for y in ys], "o-",
                        color=colors[m], label=m, lw=1.5, ms=4)
    axes[1, 2].set_title("redkitchen: coherent outliers")
    axes[1, 2].set_xlabel("outlier fraction (%)")
    axes[1, 2].set_ylabel("success (%)")
    axes[1, 2].set_ylim(-3, 103)
    axes[1, 2].grid(alpha=0.3)
    fig.suptitle("3DMatch real fragments: oracle GT-NN correspondences + controlled outliers")
    fig.tight_layout()
    fig.savefig(OUT / "threedmatch.png", dpi=160)
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
    cm.json_dump(OUT / "threedmatch.json", dict(config=dict(
        scenes=SCENES, n=N, c=C, tau=TAU, voxel=VOXEL, p=P_LIST, trials=TRIALS,
        n_pairs=N_PAIRS, methods=METHODS, data_root=str(ROOT)),
        rows=all_rows))
    with open(OUT / "threedmatch_raw.jsonl", "w") as f:
        for rec in all_raw:
            f.write(json.dumps(rec) + "\n")
    (OUT / "threedmatch.md").write_text(write_markdown(all_rows, chosen_by_scene))
    make_figure(all_rows)
    print("wrote", OUT / "threedmatch.json", "etc.")


if __name__ == "__main__":
    main()
