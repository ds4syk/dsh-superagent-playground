#!/usr/bin/env python3
"""KITTI raw (drive 0001) real LiDAR robust 3D-3D registration table.

The ego motion comes from the OXTS GPS/IMU stream, mapped to the Velodyne
frame with the provided extrinsics (direction verified in
``verify_kitti.py``).  Base correspondences are GT-mapped nearest
neighbours on real LiDAR scans (real sparsity, occlusion, dynamic
objects); controlled outliers are then injected.  This is the outdoor
counterpart of the 3DMatch fragment table.

N=300, c=0.25 m (LiDAR point spacing is ~0.1 m, so a 0.05 m threshold
would discard most correct matches), 20 trials/cell, 10 high-overlap
consecutive-frame pairs.

Run:
  srun --partition=cpu --time=00:30:00 --ntasks=1 --cpus-per-task=2 \
    /home/syk/miniconda3/envs/WM/bin/python run_kitti.py
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
import load_kitti as K  # noqa: E402
import common as cm  # noqa: E402

OUT = HERE / "results"
(OUT / "logs").mkdir(parents=True, exist_ok=True)
N = 300
C = 0.25
TAU = 0.45
VOXEL = 0.15
P_LIST = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
TRIALS = 20
PAIR_IDS = [(0, 1), (4, 5), (8, 9), (12, 13), (16, 17),
            (20, 21), (24, 25), (28, 29), (32, 33), (36, 37)]
METHODS = ["Kabsch", "RANSAC-1000", "RANSAC-adaptive", "IRLS-GM", "GNC-GM"]
BASE_SEED = 20261127


def _seed(*parts):
    return np.random.SeedSequence([BASE_SEED, *[int(x) for x in parts]])


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


def project_rotation(R):
    U, _, Vt = np.linalg.svd(R)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    return U @ D @ Vt


def prepare_base(clouds, poses, T_velo_imu, pair, rng):
    i, j = pair
    T = K.relative_velo_pose(poses, i, j, T_velo_imu)
    Rgt, tgt = project_rotation(T[:3, :3]), T[:3, 3].copy()
    x, y, d = K.gt_nn_correspondences(clouds[i], clouds[j], T, n=1000,
                                      rng=rng, tau=TAU)
    return dict(i=i, j=j, x=x, y=y, d=d, Rgt=Rgt, tgt=tgt,
                natural_inlier=float(np.mean(d < C)) if len(d) else 0.0)


def inject(base, rng, p, otype, cloud_j):
    x, y0 = base["x"], base["y"]
    y = y0.copy()
    n = len(x)
    n_out = int(round(p * n))
    meta = dict(idx=np.zeros(0, dtype=int), Rout=None, tout=None)
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
        Rout = R_delta @ base["Rgt"]
        t_shift = rng.normal(size=3)
        t_shift = t_shift / np.linalg.norm(t_shift) * rng.uniform(0.6, 1.5)
        tout = R_delta @ base["tgt"] + t_shift
        y[idx] = (Rout @ x[idx].T).T + tout
        meta["Rout"], meta["tout"] = Rout, tout
    elif otype == "boundary":
        dirs = rng.normal(size=(n_out, 3))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        u = rng.uniform(1.0 / 3.0, 3.0, size=(n_out, 1))
        y[idx] = (base["Rgt"] @ x[idx].T).T + base["tgt"] + dirs * (C * np.sqrt(u))
    else:
        raise ValueError(otype)
    return y, meta


def main():
    t_start = time.time()
    T_velo_imu = K.read_calib_rigid(K.KITTI / "calib_imu_to_velo.txt")
    poses = K.load_oxts_poses()
    idxs = sorted({z for pair in PAIR_IDS for z in pair})
    clouds = K.load_clouds(idxs, voxel=VOXEL, max_range=30.0)
    print(f"loaded {len(clouds)} clouds; sizes " +
          ", ".join(f"{i}:{len(clouds[i])}" for i in idxs), flush=True)

    bases = []
    for k, pair in enumerate(PAIR_IDS):
        rng = np.random.default_rng(_seed(70, k))
        b = prepare_base(clouds, poses, T_velo_imu, pair, rng)
        if len(b["x"]) < N:
            print(f"skip pair {pair}: only {len(b['x'])} base pairs", flush=True)
            continue
        bases.append(b)
        print(f"pair {pair}: base={len(b['x'])}, natural inlier={100*b['natural_inlier']:.1f}%, "
              f"|t|={np.linalg.norm(b['tgt']):.2f} m", flush=True)
    methods = make_methods()

    rows, raw = [], []
    for otype in ["random", "coherent", "boundary"]:
        for p in P_LIST:
            per = {m: dict(re=[], te=[], time=[], base_in=[]) for m in METHODS}
            for trial in range(TRIALS):
                rng = np.random.default_rng(_seed(71, {"random": 1, "coherent": 2, "boundary": 3}[otype],
                                                  int(round(100 * p)), trial))
                b = bases[int(rng.integers(len(bases)))]
                take = rng.choice(len(b["x"]), N, replace=False)
                base = dict(b)
                base["x"], base["y"], base["d"] = b["x"][take], b["y"][take], b["d"][take]
                y, meta = inject(base, rng, p, otype, clouds[b["j"]])
                base_in = float(np.mean(base["d"] < C))
                for name in METHODS:
                    rrng = np.random.default_rng(_seed(72, int(round(100 * p)), trial))
                    t0 = time.perf_counter()
                    R, t, info = methods[name](base["x"], y, rrng)
                    dt = time.perf_counter() - t0
                    re, te = cm.pose_errors(R, t, base["Rgt"], base["tgt"])
                    rec = dict(p=p, outlier_type=otype, trial=trial, method=name,
                               pair=f"{b['i']}-{b['j']}", base_inlier=base_in,
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
                row = dict(p=p, outlier_type=otype, method=m, n_trials=len(a["re"]),
                           base_inlier=float(np.mean(a["base_in"])), **s)
                if otype == "coherent":
                    vals = [r.get("picked_outlier", False) for r in raw
                            if r["p"] == p and r["outlier_type"] == otype and r["method"] == m]
                    row["frac_picked_outlier"] = float(np.mean(vals)) if vals else None
                rows.append(row)
            print(f"{otype:8s} p={p:.1f} done ({time.time()-t_start:.1f}s)", flush=True)

    cm.json_dump(OUT / "kitti.json", dict(
        config=dict(drive="2011_09_26_drive_0001", n=N, c=C, tau=TAU, voxel=VOXEL,
                    p=P_LIST, trials=TRIALS, pairs=PAIR_IDS, methods=METHODS),
        rows=rows))
    with open(OUT / "kitti_raw.jsonl", "w") as f:
        for r in raw:
            f.write(json.dumps(r) + "\n")

    L = ["# KITTI raw drive 0001: real LiDAR robust 3D-3D registration", "",
         f"Ego motion from OXTS (KITTI devkit Mercator + roll/pitch/yaw), mapped to Velodyne "
         f"with `calib_imu_to_velo`. N={N}, c={C} m, tau={TAU} m, {TRIALS} trials/cell, "
         f"{len(bases)} consecutive-frame pairs. Base matches are GT-mapped nearest neighbours "
         "on real LiDAR (sparse, occluded, dynamic), with controlled outliers injected.", ""]
    for otype in ["random", "coherent", "boundary"]:
        L += [f"## {otype} outliers: success rate", "",
              "| method | " + " | ".join(f"{int(100*p)}%" for p in P_LIST) + " |",
              "|---|" + "|".join(["---"] * len(P_LIST)) + "|"]
        for m in METHODS:
            cells = []
            for p in P_LIST:
                r = next(z for z in rows if z["outlier_type"] == otype and z["method"] == m
                         and abs(z["p"] - p) < 1e-9)
                cells.append(f"{100*r['success']:.0f}")
            L.append(f"| {m} | " + " | ".join(cells) + " |")
        L += ["", f"## {otype} outliers: median rotation error (deg)", "",
              "| method | " + " | ".join(f"{int(100*p)}%" for p in P_LIST) + " |",
              "|---|" + "|".join(["---"] * len(P_LIST)) + "|"]
        for m in METHODS:
            cells = []
            for p in P_LIST:
                r = next(z for z in rows if z["outlier_type"] == otype and z["method"] == m
                         and abs(z["p"] - p) < 1e-9)
                cells.append(f"{r['rot_median']:.2f}")
            L.append(f"| {m} | " + " | ".join(cells) + " |")
        L += ["", f"## {otype} outliers: runtime p50 (ms)", "",
              "| method | " + " | ".join(f"{int(100*p)}%" for p in P_LIST) + " |",
              "|---|" + "|".join(["---"] * len(P_LIST)) + "|"]
        for m in METHODS:
            cells = []
            for p in P_LIST:
                r = next(z for z in rows if z["outlier_type"] == otype and z["method"] == m
                         and abs(z["p"] - p) < 1e-9)
                cells.append(f"{r['time_ms_p50']:.2f}")
            L.append(f"| {m} | " + " | ".join(cells) + " |")
        L.append("")
    L += ["## Caveats", "",
          "* KITTI raw provides OXTS, not per-frame ground-truth poses for this sync drive; "
          "the relative transform is derived from GPS/IMU and is accurate to the OXTS/calibration "
          "level, not to the millimetre. Clean Kabsch reaches ~0.02 deg / ~0.01-0.03 m on the "
          "sampled base pairs, which is the practical noise floor.",
          "* LiDAR is sparse (~0.1 m point spacing), so c=0.25 m is used; a 3DMatch-style c=0.05 m "
          "would discard most correct matches.",
          "* Base correspondences are GT-mapped NN, not a learned descriptor; this isolates the "
          "robust outer layer, like the 3DMatch oracle table.",
          "* Consecutive frames at 10 Hz have |t|≈1.6-2.2 m; the 0.1 m success threshold is "
          "absolute pose error, not relative-to-motion error."]
    (OUT / "kitti.md").write_text("\n".join(L) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"Kabsch": "tab:gray", "RANSAC-1000": "tab:orange",
              "RANSAC-adaptive": "tab:green", "IRLS-GM": "tab:blue", "GNC-GM": "tab:red"}
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    for col, otype in enumerate(["random", "coherent", "boundary"]):
        for m in METHODS:
            ys = [next(z for z in rows if z["outlier_type"] == otype and z["method"] == m
                       and abs(z["p"] - p) < 1e-9)["success"] for p in P_LIST]
            ax[0, col].plot([100 * p for p in P_LIST], [100 * y for y in ys], "o-",
                            color=colors[m], label=m, lw=1.5, ms=4)
        ax[0, col].set_title(f"{otype}: success")
        ax[0, col].set_ylim(-3, 103)
        ax[0, col].set_xlabel("outlier fraction (%)")
        ax[0, col].set_ylabel("success (%)")
        ax[0, col].grid(alpha=0.3)
    ax[0, 0].legend(fontsize=7)
    for m in METHODS:
        ys = [next(z for z in rows if z["outlier_type"] == "random" and z["method"] == m
                   and abs(z["p"] - p) < 1e-9)["time_ms_p50"] for p in P_LIST]
        ax[1, 0].semilogy([100 * p for p in P_LIST], ys, "o-", color=colors[m], label=m, lw=1.5)
    ax[1, 0].set_title("random: runtime p50")
    ax[1, 0].set_xlabel("outlier fraction (%)")
    ax[1, 0].set_ylabel("ms (log)")
    ax[1, 0].grid(alpha=0.3, which="both")
    for m in METHODS:
        ys = [next(z for z in rows if z["outlier_type"] == "random" and z["method"] == m
                   and abs(z["p"] - p) < 1e-9)["rot_median"] for p in P_LIST]
        ax[1, 1].semilogy([100 * p for p in P_LIST], np.maximum(ys, 1e-3), "o-",
                          color=colors[m], label=m, lw=1.5)
    ax[1, 1].set_title("random: median rotation error")
    ax[1, 1].set_xlabel("outlier fraction (%)")
    ax[1, 1].set_ylabel("deg (log)")
    ax[1, 1].grid(alpha=0.3, which="both")
    for m in METHODS:
        ys = [next(z for z in rows if z["outlier_type"] == "coherent" and z["method"] == m
                   and abs(z["p"] - p) < 1e-9)["rot_median"] for p in P_LIST]
        ax[1, 2].semilogy([100 * p for p in P_LIST], np.maximum(ys, 1e-3), "o-",
                          color=colors[m], label=m, lw=1.5)
    ax[1, 2].set_title("coherent: median rotation error")
    ax[1, 2].set_xlabel("outlier fraction (%)")
    ax[1, 2].set_ylabel("deg (log)")
    ax[1, 2].grid(alpha=0.3, which="both")
    fig.suptitle("KITTI raw 0001: real LiDAR GT-NN correspondences + controlled outliers")
    fig.tight_layout()
    fig.savefig(OUT / "kitti.png", dpi=160)
    print("wrote", OUT / "kitti.json", "etc.")


if __name__ == "__main__":
    main()
