#!/usr/bin/env python3
"""KITTI odometry sequences 00/05/07: robust 3D-3D table + motion analysis.

Same protocol as the parent raw-drive tables (``../kitti/`` for drive 0001,
``../kitti2/`` for drive 0002) so the numbers are directly comparable:

* correspondences: GT-mapped nearest neighbours on real LiDAR
  (voxel 0.15 m, range 1-30 m, tau 0.45 m), N=300 sampled per trial;
* outliers: random / coherent / boundary, p = 0 .. 0.7;
* methods: Kabsch, RANSAC-1000, RANSAC-adaptive, IRLS-GM, GNC-GM;
* 20 trials/cell, 10 consecutive-frame pairs per sequence, c = 0.25 m,
  success = rot < 5 deg and |t| < 0.1 m (absolute pose error).

Two differences from the raw drives, both documented in the README:

1. Poses are the benchmark's *exact* GT (``dataset/poses``) instead of
   OXTS-integrated motion, so the clean floor should be tighter.
2. A second phase measures per-pair and per-frame-gap behaviour, to test
   whether the robust-method conclusion depends on frame motion magnitude
   (|t| per pair, i.e. speed at 10 Hz).

Run (srun, ~10 min):
  srun --partition=cpu --time=00:40:00 --ntasks=1 --cpus-per-task=2 \
    /home/syk/miniconda3/envs/WM/bin/python run_kitti_odom.py
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
import load_kitti_odom as K  # noqa: E402
import common as cm  # noqa: E402

OUT = HERE / "results"
(OUT / "logs").mkdir(parents=True, exist_ok=True)
N = 300
C = 0.25
TAU = 0.45
VOXEL = 0.15
P_LIST = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
TRIALS = 20
N_PAIRS = 10
METHODS = ["Kabsch", "RANSAC-1000", "RANSAC-adaptive", "IRLS-GM", "GNC-GM"]
BASE_SEED = 20261127
MOTION_TRIALS = 10
MOTION_P = [0.3, 0.5, 0.7]
GAPS = [2, 4, 8]          # gap=1 is the main 10-pair set
RAW_DIRS = {"0001": HERE.parent / "kitti" / "results" / "kitti.json",
            "0002": HERE.parent / "kitti2" / "results" / "kitti2.json"}


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


def prepare_base(clouds, poses, T_cam_velo, pair, rng):
    i, j = pair
    T = K.relative_velo_pose(poses, i, j, T_cam_velo)
    Rgt, tgt = project_rotation(T[:3, :3]), T[:3, 3].copy()
    x, y, d = K.gt_nn_correspondences(clouds[i], clouds[j], T, n=1000,
                                      rng=rng, tau=TAU)
    tmag, rot = K.pose_motion(poses, i, j, T_cam_velo)
    return dict(i=i, j=j, x=x, y=y, d=d, Rgt=Rgt, tgt=tgt,
                t_m=tmag, rot_deg=rot,
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


def fit_all(methods, x, y, p, trial, tag, seed_salt=0):
    """Run the five methods on one instance, return raw records."""
    recs = []
    for name in METHODS:
        rrng = np.random.default_rng(_seed(72, int(round(100 * p)), trial, seed_salt))
        t0 = time.perf_counter()
        R, t, info = methods[name](x, y, rrng)
        dt = time.perf_counter() - t0
        recs.append((name, R, t, info, dt))
    return recs


def main():
    t_start = time.time()
    methods = make_methods()
    per_seq, raw_all, motion_pairs, gap_rows = {}, [], [], []

    for seq in K.SEQS:
        t_seq = time.time()
        poses = K.load_poses(seq)
        T_cam_velo = K.read_calib_tr(seq)
        n_frames = len(poses)
        step = max(2, (n_frames - 2) // N_PAIRS)
        pairs = [(k * step, k * step + 1) for k in range(N_PAIRS)]
        idxs = sorted({z for pr in pairs for z in pr})
        clouds = K.load_clouds(seq, idxs, voxel=VOXEL, max_range=30.0)
        print(f"seq {seq}: {n_frames} GT frames, {len(clouds)} clouds, sizes "
              + ", ".join(f"{i}:{len(clouds[i])}" for i in idxs), flush=True)

        bases = []
        for k, pair in enumerate(pairs):
            rng = np.random.default_rng(_seed(70, int(seq), k))
            b = prepare_base(clouds, poses, T_cam_velo, pair, rng)
            if len(b["x"]) < N:
                print(f"  skip pair {pair}: only {len(b['x'])} base pairs", flush=True)
                continue
            bases.append(b)
            print(f"  pair {pair}: base={len(b['x'])}, natural inlier="
                  f"{100*b['natural_inlier']:.1f}%, |t|={b['t_m']:.2f} m "
                  f"({b['t_m']*36:.1f} km/h), rot={b['rot_deg']:.2f} deg", flush=True)

        rows = []
        for otype in ["random", "coherent", "boundary"]:
            for p in P_LIST:
                per = {m: dict(re=[], te=[], time=[], base_in=[]) for m in METHODS}
                for trial in range(TRIALS):
                    rng = np.random.default_rng(_seed(
                        71, int(seq), {"random": 1, "coherent": 2, "boundary": 3}[otype],
                        int(round(100 * p)), trial))
                    b = bases[int(rng.integers(len(bases)))]
                    take = rng.choice(len(b["x"]), N, replace=False)
                    base = dict(b)
                    base["x"], base["y"], base["d"] = b["x"][take], b["y"][take], b["d"][take]
                    y, meta = inject(base, rng, p, otype, clouds[b["j"]])
                    base_in = float(np.mean(base["d"] < C))
                    for name, R, t, info, dt in fit_all(methods, base["x"], y, p, trial,
                                                        tag=(seq, otype)):
                        re, te = cm.pose_errors(R, t, base["Rgt"], base["tgt"])
                        rec = dict(dataset=f"odom{seq}", p=p, outlier_type=otype, trial=trial,
                                   method=name, pair=f"{b['i']}-{b['j']}",
                                   t_m=b["t_m"], rot_deg=b["rot_deg"],
                                   base_inlier=base_in, success=cm.success(re, te),
                                   rot=re, trans=te, time_ms=1000.0 * dt,
                                   ransac_iters=info.get("n_iters"),
                                   ransac_inliers=info.get("best_count"))
                        if otype == "coherent" and meta["Rout"] is not None:
                            re_o = cm.rot_err_deg(R, meta["Rout"])
                            te_o = float(np.linalg.norm(t - meta["tout"]))
                            rec["picked_outlier"] = bool(re_o < re and te_o < te)
                        raw_all.append(rec)
                        per[name]["re"].append(re)
                        per[name]["te"].append(te)
                        per[name]["time"].append(dt)
                        per[name]["base_in"].append(base_in)
                for m in METHODS:
                    a = per[m]
                    s = cm.summarize_errors(a["re"], a["te"], a["time"])
                    row = dict(dataset=f"odom{seq}", p=p, outlier_type=otype, method=m,
                               n_trials=len(a["re"]),
                               base_inlier=float(np.mean(a["base_in"])), **s)
                    if otype == "coherent":
                        vals = [r.get("picked_outlier", False) for r in raw_all
                                if r["dataset"] == f"odom{seq}" and r["p"] == p
                                and r["outlier_type"] == otype and r["method"] == m]
                        row["frac_picked_outlier"] = float(np.mean(vals)) if vals else None
                    rows.append(row)
                print(f"  {seq} {otype:8s} p={p:.1f} ({time.time()-t_seq:.0f}s)", flush=True)

        # --- motion phase A: fixed consecutive pair, per-pair statistics -----
        for k, b in enumerate(bases):
            for p in MOTION_P:
                agg = {m: dict(re=[], te=[], time=[]) for m in METHODS}
                for trial in range(MOTION_TRIALS):
                    rng = np.random.default_rng(_seed(73, int(seq), k, int(round(100 * p)), trial))
                    take = rng.choice(len(b["x"]), N, replace=False)
                    base = dict(b)
                    base["x"], base["y"] = b["x"][take], b["y"][take]
                    y, _ = inject(base, rng, p, "random", clouds[b["j"]])
                    for name, R, t, info, dt in fit_all(methods, base["x"], y, p, trial,
                                                        tag=(seq, "motion"), seed_salt=1):
                        re, te = cm.pose_errors(R, t, base["Rgt"], base["tgt"])
                        agg[name]["re"].append(re)
                        agg[name]["te"].append(te)
                        agg[name]["time"].append(dt)
                for m in METHODS:
                    s = cm.summarize_errors(agg[m]["re"], agg[m]["te"], agg[m]["time"])
                    motion_pairs.append(dict(dataset=f"odom{seq}", pair=f"{b['i']}-{b['j']}",
                                             t_m=b["t_m"], rot_deg=b["rot_deg"],
                                             natural_inlier=b["natural_inlier"],
                                             gap=1, p=p, method=m,
                                             n_trials=MOTION_TRIALS, **s))
        print(f"seq {seq} motion pairs done ({time.time()-t_seq:.0f}s)", flush=True)

        # --- motion phase B: larger frame gaps (bigger |t|) ------------------
        base_frames = [0, n_frames // 2]
        need = sorted({b + g for b in base_frames for g in GAPS} | set(base_frames))
        clouds2 = K.load_clouds(seq, need, voxel=VOXEL, max_range=30.0)
        for bi, i in enumerate(base_frames):
            for g in GAPS:
                j = i + g
                if j >= n_frames or j not in clouds2:
                    continue
                rng = np.random.default_rng(_seed(74, int(seq), bi, g))
                b = prepare_base(clouds2, poses, T_cam_velo, (i, j), rng)
                if len(b["x"]) < N:
                    print(f"  gap {g}: pair {i}-{j} only {len(b['x'])} base pairs, skip", flush=True)
                    continue
                print(f"  gap {g}: pair {i}-{j} base={len(b['x'])}, natural inlier="
                      f"{100*b['natural_inlier']:.1f}%, |t|={b['t_m']:.2f} m, "
                      f"rot={b['rot_deg']:.2f} deg", flush=True)
                for p in MOTION_P:
                    agg = {m: dict(re=[], te=[], time=[]) for m in METHODS}
                    for trial in range(MOTION_TRIALS):
                        rngt = np.random.default_rng(_seed(75, int(seq), bi, g,
                                                           int(round(100 * p)), trial))
                        take = rngt.choice(len(b["x"]), N, replace=False)
                        base = dict(b)
                        base["x"], base["y"] = b["x"][take], b["y"][take]
                        y, _ = inject(base, rngt, p, "random", clouds2[b["j"]])
                        for name, R, t, info, dt in fit_all(methods, base["x"], y, p, trial,
                                                            tag=(seq, "gap"), seed_salt=2):
                            re, te = cm.pose_errors(R, t, base["Rgt"], base["tgt"])
                            agg[name]["re"].append(re)
                            agg[name]["te"].append(te)
                            agg[name]["time"].append(dt)
                    for m in METHODS:
                        s = cm.summarize_errors(agg[m]["re"], agg[m]["te"], agg[m]["time"])
                        gap_rows.append(dict(dataset=f"odom{seq}", pair=f"{i}-{j}",
                                             t_m=b["t_m"], rot_deg=b["rot_deg"],
                                             natural_inlier=b["natural_inlier"],
                                             gap=g, p=p, method=m,
                                             n_trials=MOTION_TRIALS, **s))

        per_seq[seq] = dict(n_frames=n_frames, pairs=pairs, step=step,
                            base=[dict(pair=f"{b['i']}-{b['j']}", t_m=b["t_m"],
                                       rot_deg=b["rot_deg"],
                                       natural_inlier=b["natural_inlier"]) for b in bases],
                            rows=rows, n_clouds=len(clouds),
                            elapsed_s=round(time.time() - t_seq, 1))
        print(f"seq {seq} table done ({time.time()-t_seq:.0f}s, total {time.time()-t_start:.0f}s)",
              flush=True)

    # ---------------- pooled across the three sequences ---------------------
    pooled = []
    for otype in ["random", "coherent", "boundary"]:
        for p in P_LIST:
            for m in METHODS:
                sel = [r for r in raw_all if r["outlier_type"] == otype
                       and abs(r["p"] - p) < 1e-9 and r["method"] == m]
                re = [r["rot"] for r in sel]
                te = [r["trans"] for r in sel]
                tm = [r["time_ms"] / 1000.0 for r in sel]
                s = cm.summarize_errors(re, te, tm)
                pooled.append(dict(dataset="pooled_odom", p=p, outlier_type=otype,
                                   method=m, n_trials=len(sel), **s))

    # ---------------- raw-drive comparison (parent results) ----------------
    compare = {}
    for tag, path in RAW_DIRS.items():
        d = json.loads(Path(path).read_text())
        compare[tag] = d

    cfg = dict(sequences=K.SEQS, n=N, c=C, tau=TAU, voxel=VOXEL, p=P_LIST,
               trials=TRIALS, pairs_per_seq=N_PAIRS, methods=METHODS,
               motion_trials=MOTION_TRIALS, motion_p=MOTION_P, gaps=GAPS,
               data_root=str(K.DATA),
               note="exact odometry GT poses (dataset/poses), not OXTS")
    cm.json_dump(OUT / "more_drives.json", dict(
        config=cfg, per_sequence=per_seq, pooled=pooled,
        motion_pairs=motion_pairs, motion_gaps=gap_rows,
        raw_drive_rows={k: v["rows"] for k, v in compare.items()},
        raw_drive_config={k: v["config"] for k, v in compare.items()},
        elapsed_s=round(time.time() - t_start, 1)))

    with open(OUT / "more_drives_raw.jsonl", "w") as f:
        for r in raw_all:
            f.write(json.dumps(r) + "\n")
    with open(OUT / "motion_raw.jsonl", "w") as f:
        for r in motion_pairs + gap_rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {OUT/'more_drives.json'} ({len(raw_all)} raw trials, "
          f"{len(pooled)} pooled rows, {len(motion_pairs)+len(gap_rows)} motion rows)", flush=True)


if __name__ == "__main__":
    main()
