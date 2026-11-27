#!/usr/bin/env python3
"""KITTI N scaling: GNC vs fixed-budget RANSAC on real LiDAR.

N in {50,100,300,1000}, p in {0.3,0.7}, 20 trials/cell, 10 pairs,
GT-NN base precomputed at 2000 pairs.  Mirrors the 3DMatch N-scaling
experiment so the two real datasets can be compared.
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
import run_kitti as RK  # noqa: E402

OUT = HERE / "results"
(OUT / "logs").mkdir(parents=True, exist_ok=True)
N_LIST = [50, 100, 300, 1000]
P_LIST = [0.3, 0.7]
TRIALS = 20
METHODS = ["Kabsch", "RANSAC-1000", "IRLS-GM", "GNC-GM"]


def main():
    t0 = time.time()
    T_velo_imu = K.read_calib_rigid(K.KITTI / "calib_imu_to_velo.txt")
    poses = K.load_oxts_poses()
    idxs = sorted({z for pair in RK.PAIR_IDS for z in pair})
    clouds = K.load_clouds(idxs, voxel=RK.VOXEL, max_range=30.0)
    bases = []
    for k, pair in enumerate(RK.PAIR_IDS):
        rng = np.random.default_rng(RK._seed(85, k))
        T = K.relative_velo_pose(poses, pair[0], pair[1], T_velo_imu)
        Rgt = RK.project_rotation(T[:3, :3]); tgt = T[:3, 3]
        x, y, d = K.gt_nn_correspondences(clouds[pair[0]], clouds[pair[1]], T,
                                          n=2000, rng=rng, tau=0.45)
        if len(x) >= 1000:
            bases.append(dict(i=pair[0], j=pair[1], x=x, y=y, d=d, Rgt=Rgt, tgt=tgt))
    print(f"{len(bases)} bases of size {[len(b['x']) for b in bases]}", flush=True)
    rows, raw = [], []
    for n in N_LIST:
        for p in P_LIST:
            for trial in range(TRIALS):
                rng = np.random.default_rng(RK._seed(86, n, int(round(100 * p)), trial))
                b = bases[int(rng.integers(len(bases)))]
                take = rng.choice(len(b["x"]), min(n, len(b["x"])), replace=False)
                x, y, d = b["x"][take], b["y"][take], b["d"][take]
                if len(x) < n:
                    pad = rng.choice(len(x), n - len(x), replace=True)
                    x, y, d = np.concatenate([x, x[pad]]), np.concatenate([y, y[pad]]), np.concatenate([d, d[pad]])
                y = y.copy()
                nout = int(round(p * n))
                if nout:
                    idx = rng.choice(n, nout, replace=False)
                    y[idx] = clouds[b["j"]][rng.choice(len(clouds[b["j"]]), nout, replace=True)]
                for name in METHODS:
                    rrng = np.random.default_rng(RK._seed(87, n, trial))
                    t1 = time.perf_counter()
                    if name == "Kabsch":
                        R, t = cm.kabsch(x, y, np.ones(len(x)))
                    elif name == "RANSAC-1000":
                        R, t = cm.solve_ransac(x, y, rrng, c=RK.C, iters=1000, adaptive=False)
                    elif name == "IRLS-GM":
                        R, t = cm.solve_irls(x, y, c=RK.C)
                    else:
                        R, t = cm.solve_gnc(x, y, c=RK.C)
                    dt = time.perf_counter() - t1
                    re, te = cm.pose_errors(R, t, b["Rgt"], b["tgt"])
                    raw.append(dict(n=n, p=p, trial=trial, method=name,
                                    success=cm.success(re, te), rot=re, trans=te,
                                    time_ms=1000 * dt))
            for name in METHODS:
                recs = [r for r in raw if r["n"] == n and r["p"] == p and r["method"] == name]
                re = np.array([r["rot"] for r in recs]); tt = np.array([r["time_ms"] for r in recs])
                rows.append(dict(n=n, p=p, method=name,
                                 success=float(np.mean([r["success"] for r in recs])),
                                 rot_median=float(np.median(re)),
                                 time_ms_p50=float(np.median(tt)),
                                 time_ms_p95=float(np.percentile(tt, 95))))
            print(f"n={n} p={p} done ({time.time()-t0:.1f}s)", flush=True)
    cm.json_dump(OUT / "n_scaling.json", dict(
        config=dict(n_list=N_LIST, p=P_LIST, trials=TRIALS, methods=METHODS), rows=rows))
    with open(OUT / "n_scaling_raw.jsonl", "w") as f:
        for r in raw:
            f.write(json.dumps(r) + "\n")
    L = ["# KITTI N scaling", "",
         f"20 trials/cell, p in {P_LIST}, GT-NN LiDAR base. Runtime p50/p95 per call.", "",
         "| p | N | method | success | rot med (deg) | p50 (ms) | p95 (ms) |",
         "|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['p']:.1f} | {r['n']} | {r['method']} | {100*r['success']:.0f}% | "
                 f"{r['rot_median']:.2f} | {r['time_ms_p50']:.2f} | {r['time_ms_p95']:.2f} |")
    L += ["", "## Reading", "",
          "* Compare with the 3DMatch N-scaling table: both real datasets should show GNC "
          "roughly linear in N and RANSAC dominated by its iteration count.",
          "* At N=1000 GNC is expected to remain much faster; if not, report the crossover."]
    (OUT / "n_scaling.md").write_text("\n".join(L) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
