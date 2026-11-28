#!/usr/bin/env python3
"""P3 real-geometry 3D-3D registration on ModelNet40.

Each trial samples a real ModelNet40 object (2048 points), normalizes it to
the unit sphere, samples N=200 source points, applies a random rigid motion +
Gaussian noise, and replaces a fraction p by uniform outliers.  Same methods
and metrics as the synthetic table.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
SYN = HERE.parents[1] / "synthetic"
sys.path.insert(0, str(SYN))
import run_registration as rr  # noqa: E402

DATA = Path("/mnt/storage1/syk/data/modelnet40/modelnet40_ply_hdf5_2048")
OUT = HERE / "results"
OUT.mkdir(parents=True, exist_ok=True)

N = 200
SIGMA = 0.01
C = 0.05
N_TRIALS = 50
P = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
RANSAC_ITERS = 1000


def load_points():
    pts = []
    for f in sorted(DATA.glob("ply_data_test*.h5")):
        with h5py.File(f, "r") as h:
            pts.append(np.asarray(h["data"][:], dtype=np.float64))
    pts = np.concatenate(pts, axis=0)
    print("objects", pts.shape[0], "points/object", pts.shape[1])
    return pts


def make_instance(rng, clouds, p):
    obj = clouds[rng.integers(len(clouds))]
    idx = rng.choice(obj.shape[0], N, replace=False)
    x = obj[idx]
    x = x - x.mean(0)
    x = x / max(np.linalg.norm(x, axis=1).max(), 1e-9)
    R = rr.random_rotation(rng)
    t = rng.uniform(-0.3, 0.3, size=3)
    y = (R @ x.T).T + t + SIGMA * rng.normal(size=(N, 3))
    n_out = int(round(p * N))
    if n_out:
        oi = rng.choice(N, n_out, replace=False)
        y[oi] = rng.uniform(-1.0, 1.0, size=(n_out, 3))
    return x, y, R, t


def run():
    clouds = load_points()
    rng = np.random.default_rng(20261127)
    methods = {
        "Kabsch": lambda x, y: rr.kabsch(x, y, np.ones(len(x))),
        "RANSAC+Kabsch": lambda x, y: rr.solve_ransac(x, y, rng, c=C, iters=RANSAC_ITERS),
        "IRLS-GM": lambda x, y: rr.solve_irls(x, y, c=C),
        "GNC-GM (ours)": lambda x, y: rr.solve_gnc(x, y, c=C),
    }
    rows = []
    for p in P:
        acc = {m: {"rot": [], "trans": [], "succ": [], "time": []} for m in methods}
        for _ in range(N_TRIALS):
            x, y, Rgt, tgt = make_instance(rng, clouds, p)
            for name, fn in methods.items():
                t0 = time.perf_counter()
                Re, te = fn(x, y)
                dt = time.perf_counter() - t0
                re = rr.rot_err_deg(Re, Rgt)
                tr = float(np.linalg.norm(te - tgt))
                acc[name]["rot"].append(re); acc[name]["trans"].append(tr)
                acc[name]["succ"].append(re < 5.0 and tr < 0.1)
                acc[name]["time"].append(dt)
        for m in methods:
            a = acc[m]
            rows.append(dict(p=p, method=m,
                             success=float(np.mean(a["succ"])),
                             rot_median=float(np.median(a["rot"])),
                             trans_median=float(np.median(a["trans"])),
                             time_ms=float(1000 * np.median(a["time"]))))
        print("p", p, "done", flush=True)
    json.dump(rows, open(OUT / "main_table.json", "w"), indent=2)
    lines = ["# ModelNet40 3D-3D registration (real object geometry)\n",
             f"N={N}, sigma={SIGMA}, c={C}, {N_TRIALS} trials/cell, uniform outliers\n",
             "| method | " + " | ".join(f"{int(100*p)}%" for p in P) + " |",
             "|-|" + "|".join(["---"] * len(P)) + "|"]
    for m in methods:
        vals = [next(r for r in rows if r["method"] == m and r["p"] == p)["success"] for p in P]
        lines.append(f"| {m} | " + " | ".join(f"{100*v:.0f}" for v in vals) + " |")
    lines.append("\n## median rotation error (deg)\n")
    lines.append("| method | " + " | ".join(f"{int(100*p)}%" for p in P) + " |")
    lines.append("|-|" + "|".join(["---"] * len(P)) + "|")
    for m in methods:
        vals = [next(r for r in rows if r["method"] == m and r["p"] == p)["rot_median"] for p in P]
        lines.append(f"| {m} | " + " | ".join(f"{v:.2f}" for v in vals) + " |")
    lines.append("\n## median runtime (ms)\n")
    lines.append("| method | " + " | ".join(f"{int(100*p)}%" for p in P) + " |")
    lines.append("|-|" + "|".join(["---"] * len(P)) + "|")
    for m in methods:
        vals = [next(r for r in rows if r["method"] == m and r["p"] == p)["time_ms"] for p in P]
        lines.append(f"| {m} | " + " | ".join(f"{v:.2f}" for v in vals) + " |")
    (OUT / "main_table.md").write_text("\n".join(lines) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for m in methods:
        ax[0].plot([100*p for p in P], [100*next(r for r in rows if r["method"] == m and r["p"] == p)["success"] for p in P], "o-", label=m)
        ax[1].semilogy([100*p for p in P], [next(r for r in rows if r["method"] == m and r["p"] == p)["rot_median"] for p in P], "o-", label=m)
    ax[0].set_xlabel("outlier fraction (%)"); ax[0].set_ylabel("success (%)"); ax[0].set_ylim(-2, 102)
    ax[0].grid(alpha=0.3); ax[0].legend(fontsize=8)
    ax[1].set_xlabel("outlier fraction (%)"); ax[1].set_ylabel("median rotation error (deg)")
    ax[1].grid(alpha=0.3, which="both")
    fig.tight_layout(); fig.savefig(OUT / "main_table.png", dpi=160)
    print("wrote", OUT)


if __name__ == "__main__":
    run()
