"""Supplementary: SDPR vs chordal vs a local solver under measurement noise.

Pure rotation-synchronization problems (no V-JEPA 2), N=8, 13 edges, GT
trajectory; measurements corrupted by axis-angle Gaussian noise.  The local
solver minimizes the same cost with scipy least_squares from the identity.
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from model import chordal_sync  # noqa: E402
from panorama import geodesic_deg, trajectory  # noqa: E402
from rotsync import RotSyncProblem, SDPRunner, extract_rotations  # noqa: E402

EDGES = [(i, i + 1) for i in range(7)] + [(i, i + 2) for i in range(6)]
PRIOR = 10.0
SIGMAS = [0.0, 2.0, 5.0, 10.0, 20.0]
TRIALS = 10


def exp_so3(w):
    a = np.linalg.norm(w)
    if a < 1e-12:
        return np.eye(3)
    k = w / a
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * (K @ K)


def local_solve(meas, R_gt, edges=EDGES, prior=PRIOR):
    n = 8

    def residual(x):
        Rs = [exp_so3(x[3 * i : 3 * i + 3]) for i in range(n)]
        res = []
        for (i, j), Rm in zip(edges, meas):
            res.append((Rm - Rs[i] @ Rs[j].T).ravel())
        res.append(np.sqrt(prior) * (Rs[0] - np.eye(3)).ravel())
        return np.concatenate(res)

    sol = least_squares(residual, np.zeros(3 * n), method="lm", max_nfev=500)
    Rs = np.stack([exp_so3(sol.x[3 * i : 3 * i + 3]) for i in range(n)])
    return Rs


def main():
    prob = RotSyncProblem(8, redundant=False)
    runner = SDPRunner(prob, solver="SCS", eps=1e-6)
    out = {s: {"sdpr": [], "chordal": [], "local": []} for s in SIGMAS}
    t0 = time.time()
    for sigma in SIGMAS:
        for trial in range(TRIALS):
            rng = np.random.default_rng(1000 * int(sigma) + trial)
            R_gt = trajectory(8, rng)
            meas = np.stack(
                [R_gt[i] @ R_gt[j].T @ exp_so3(np.deg2rad(sigma) * rng.standard_normal(3))
                 for i, j in EDGES]
            )
            # SDPR
            Q = prob.build_Q(meas, np.ones(len(EDGES)), EDGES, PRIOR)
            o = runner.solve(Q)
            R_sdpr = extract_rotations(o["x"], 8)
            # chordal
            with torch.no_grad():
                R_ch = chordal_sync(
                    torch.tensor(meas, dtype=torch.float64)[None],
                    torch.ones(1, len(EDGES), dtype=torch.float64),
                    EDGES,
                    8,
                )[0].numpy()
            # local
            R_loc = local_solve(meas, R_gt)
            out[sigma]["sdpr"].append(float(np.mean(geodesic_deg(R_sdpr, R_gt))))
            out[sigma]["chordal"].append(float(np.mean(geodesic_deg(R_ch, R_gt))))
            out[sigma]["local"].append(float(np.mean(geodesic_deg(R_loc, R_gt))))
        print(f"sigma={sigma}: " + ", ".join(
            f"{k}={np.mean(v):.2f}" for k, v in out[sigma].items()), flush=True)

    res = {
        "sigmas": SIGMAS,
        "trials": TRIALS,
        "summary": {
            str(s): {k: {"mean": float(np.mean(v)), "median": float(np.median(v)),
                         "success<5deg": float(np.mean(np.array(v) < 5.0))}
                     for k, v in out[s].items()}
            for s in SIGMAS
        },
        "elapsed_s": round(time.time() - t0, 1),
    }
    path = os.path.join(HERE, "..", "artifacts", "results", "noise_experiment.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(res, f, indent=2)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    for k, c in [("sdpr", "#c0392b"), ("chordal", "#4c78a8"), ("local", "#9aa5b1")]:
        ax.plot(SIGMAS, [np.mean(out[s][k]) for s in SIGMAS], "o-", label=k, color=c)
    ax.set_xlabel("measurement noise σ (deg)")
    ax.set_ylabel("mean geodesic error (deg)")
    ax.set_title("Rotation synchronization: global vs local", fontsize=9)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(os.path.join(HERE, "..", "artifacts", "results", "fig_noise.png"))
    print(json.dumps(res["summary"], indent=2))


if __name__ == "__main__":
    main()
