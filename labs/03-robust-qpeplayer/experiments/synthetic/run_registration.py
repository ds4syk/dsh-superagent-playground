#!/usr/bin/env python3
"""Synthetic 3D-3D robust registration: main table.

Methods
-------
- Kabsch            : non-robust weighted Kabsch (all weights 1)
- RANSAC+Kabsch     : 3-point RANSAC, threshold c, refit on inliers
- IRLS-GM           : Geman-McClure IRLS without annealing (mu=1 from start)
- GNC-GM (ours)     : graduated non-convexity, weighted Kabsch inner solve

Task
----
N=100 correspondences, inlier noise sigma=0.01, outlier fraction p in
{0,10,...,70}%, 100 trials per p.  Report success (rot<5 deg, trans<0.1),
median rotation/translation error, and runtime.

The forward solve is a weighted Kabsch problem (the 3D-3D specialization of
QPEP); GNC is the robust outer loop.  This table tests the forward robust
estimation, not the exact backward (the latter is validated in ../verify).
"""
import json
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
OUT.mkdir(parents=True, exist_ok=True)

N = 100
SIGMA = 0.01
C = 0.05
N_TRIALS = 100
OUTLIER_FRACS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
RANSAC_ITERS = 1000
RANSAC_THRESH = 0.05
GNC_GAMMA = 1.4
GNC_MAX_MU_ITERS = 40


def random_rotation(rng):
    A = rng.normal(size=(3, 3))
    Q, R = np.linalg.qr(A)
    Q = Q @ np.diag(np.sign(np.diag(R)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


def kabsch(x, y, w):
    w = np.asarray(w, float)
    s = w.sum()
    if not np.isfinite(s) or s <= 1e-12:
        w = np.ones_like(w)
        s = w.sum()
    w = w / s
    cx = (w[:, None] * x).sum(0)
    cy = (w[:, None] * y).sum(0)
    H = (w[:, None] * (x - cx)).T @ (y - cy)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    t = cy - R @ cx
    return R, t


def residuals(x, y, R, t):
    return np.linalg.norm((R @ x.T).T + t - y, axis=1)


def rot_err_deg(R, Rgt):
    c = (np.trace(R.T @ Rgt) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def solve_gnc(x, y, c=C):
    R, t = kabsch(x, y, np.ones(len(x)))
    e = residuals(x, y, R, t) ** 2
    mu = max(3.0 * e.max() / (c * c), 1.0)
    for _ in range(GNC_MAX_MU_ITERS):
        w = (mu * c * c / (mu * c * c + e)) ** 2
        R, t = kabsch(x, y, w)
        e = residuals(x, y, R, t) ** 2
        if mu <= 1.0 + 1e-12:
            break
        mu = max(1.0, mu / GNC_GAMMA)
    # final exact-kernel IRLS steps at mu=1
    for _ in range(5):
        w = (c * c / (c * c + e)) ** 2
        R, t = kabsch(x, y, w)
        e = residuals(x, y, R, t) ** 2
    return R, t


def solve_irls(x, y, c=C, iters=40):
    R, t = kabsch(x, y, np.ones(len(x)))
    for _ in range(iters):
        e = residuals(x, y, R, t) ** 2
        w = (c * c / (c * c + e)) ** 2
        R, t = kabsch(x, y, w)
    return R, t


def solve_ransac(x, y, rng, c=RANSAC_THRESH, iters=RANSAC_ITERS):
    n = len(x)
    best_in = None
    best_count = -1
    for _ in range(iters):
        idx = rng.choice(n, 3, replace=False)
        R, t = kabsch(x[idx], y[idx], np.ones(3))
        e = residuals(x, y, R, t)
        inl = e < c
        cnt = int(inl.sum())
        if cnt > best_count:
            best_count = cnt
            best_in = inl
            if cnt == n:
                break
    if best_in is None or best_count < 3:
        return kabsch(x, y, np.ones(n))
    return kabsch(x[best_in], y[best_in], np.ones(best_count))


def make_instance(rng, p):
    x = rng.uniform(-1, 1, size=(N, 3))
    Rgt = random_rotation(rng)
    tgt = rng.uniform(-0.5, 0.5, size=3)
    y = (Rgt @ x.T).T + tgt + SIGMA * rng.normal(size=(N, 3))
    n_out = int(round(p * N))
    if n_out > 0:
        idx = rng.choice(N, n_out, replace=False)
        y[idx] = rng.uniform(-2.0, 2.0, size=(n_out, 3))
    return x, y, Rgt, tgt


def run():
    rng = np.random.default_rng(20261127)
    methods = {
        "Kabsch": lambda x, y: kabsch(x, y, np.ones(len(x))),
        "RANSAC+Kabsch": lambda x, y: solve_ransac(x, y, rng),
        "IRLS-GM": lambda x, y: solve_irls(x, y),
        "GNC-GM (ours)": lambda x, y: solve_gnc(x, y),
    }
    rows = []
    for p in OUTLIER_FRACS:
        acc = {m: {"rot": [], "trans": [], "succ": [], "time": []} for m in methods}
        for _ in range(N_TRIALS):
            x, y, Rgt, tgt = make_instance(rng, p)
            for name, fn in methods.items():
                t0 = time.perf_counter()
                R, t = fn(x, y)
                dt = time.perf_counter() - t0
                re = rot_err_deg(R, Rgt)
                te = float(np.linalg.norm(t - tgt))
                acc[name]["rot"].append(re)
                acc[name]["trans"].append(te)
                acc[name]["succ"].append(re < 5.0 and te < 0.1)
                acc[name]["time"].append(dt)
        for name in methods:
            a = acc[name]
            rows.append(dict(
                outlier_frac=p, method=name,
                success=float(np.mean(a["succ"])),
                rot_median=float(np.median(a["rot"])),
                rot_iqr=float(np.subtract(*np.percentile(a["rot"], [75, 25]))),
                trans_median=float(np.median(a["trans"])),
                trans_iqr=float(np.subtract(*np.percentile(a["trans"], [75, 25]))),
                time_ms_median=float(1000 * np.median(a["time"])),
            ))
        print(f"p={p:.1f} done", flush=True)
    with open(OUT / "main_table.json", "w") as f:
        json.dump(rows, f, indent=2)

    # markdown table: success rate
    lines = ["# Synthetic 3D-3D registration: success rate (rot<5°, trans<0.1)\n",
             f"N={N}, sigma={SIGMA}, c={C}, {N_TRIALS} trials/cell, uniform outliers\n",
             "| method | " + " | ".join(f"{int(100*p)}%" for p in OUTLIER_FRACS) + " |",
             "|-|" + "|".join(["---"] * len(OUTLIER_FRACS)) + "|"]
    for name in methods:
        cells = []
        for p in OUTLIER_FRACS:
            r = next(r for r in rows if r["method"] == name and r["outlier_frac"] == p)
            cells.append(f"{100*r['success']:.0f}")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines.append("\n## Median rotation error (deg)\n")
    lines.append("| method | " + " | ".join(f"{int(100*p)}%" for p in OUTLIER_FRACS) + " |")
    lines.append("|-|" + "|".join(["---"] * len(OUTLIER_FRACS)) + "|")
    for name in methods:
        cells = []
        for p in OUTLIER_FRACS:
            r = next(r for r in rows if r["method"] == name and r["outlier_frac"] == p)
            cells.append(f"{r['rot_median']:.2f}")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines.append("\n## Median translation error (m)\n")
    lines.append("| method | " + " | ".join(f"{int(100*p)}%" for p in OUTLIER_FRACS) + " |")
    lines.append("|-|" + "|".join(["---"] * len(OUTLIER_FRACS)) + "|")
    for name in methods:
        cells = []
        for p in OUTLIER_FRACS:
            r = next(r for r in rows if r["method"] == name and r["outlier_frac"] == p)
            cells.append(f"{r['trans_median']:.3f}")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    (OUT / "main_table.md").write_text("\n".join(lines) + "\n")

    # plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for name in methods:
        ys = [next(r for r in rows if r["method"] == name and r["outlier_frac"] == p)["success"] for p in OUTLIER_FRACS]
        ax[0].plot([100*p for p in OUTLIER_FRACS], [100*y for y in ys], "o-", label=name)
    ax[0].set_xlabel("outlier fraction (%)"); ax[0].set_ylabel("success rate (%)")
    ax[0].set_ylim(-2, 102); ax[0].grid(alpha=0.3); ax[0].legend(fontsize=8)
    for name in methods:
        ys = [next(r for r in rows if r["method"] == name and r["outlier_frac"] == p)["rot_median"] for p in OUTLIER_FRACS]
        ax[1].semilogy([100*p for p in OUTLIER_FRACS], ys, "o-", label=name)
    ax[1].set_xlabel("outlier fraction (%)"); ax[1].set_ylabel("median rotation error (deg)")
    ax[1].grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(OUT / "main_table.png", dpi=160)
    print("wrote", OUT / "main_table.json", OUT / "main_table.md", OUT / "main_table.png")


if __name__ == "__main__":
    run()
