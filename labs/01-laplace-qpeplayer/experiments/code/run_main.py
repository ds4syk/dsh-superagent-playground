#!/usr/bin/env python3
"""Laplace-QPEPLayer synthetic branch-crossing experiment.

Main table: hard argmin vs multi-modal Laplace vs mode-level Monte Carlo,
on a controlled two-global-mode QPEP instance whose global branches cross at
theta=0.  Metrics: gradient error vs finite differences, pose error, runtime,
and a small gradient-descent fit through the layer.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

import instance as inst_mod
import qpep_core as core

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "results"
OUT.mkdir(parents=True, exist_ok=True)

TAU = 0.01
FD_H = 1e-4
THETA_GRID = np.linspace(-0.2, 0.2, 81)
N_MC = 200
MC_SEEDS = 20
N_STARTS = 128
FIT_SEEDS = 0
FIT_STEPS = 0


def softmax(x):
    x = np.asarray(x, float)
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def dlogdet_numeric(q, M, dM, dq, eps=1e-6):
    """d/dtheta log det H at fixed branch, via a *tangent-space* FD.

    H is defined on the unit sphere, so perturbing q in R^4 off the sphere is
    wrong (that was the source of a spurious 9.36 discrepancy).  Use the
    orthonormal tangent basis E(q): q_t = normalize(q + E xi), and dot the
    tangent gradient with xi = E^T dq.
    """
    # explicit theta part at fixed q
    explicit = (core.logdetH(q, M + eps * dM) - core.logdetH(q, M - eps * dM)) / (2 * eps)
    Eq = core.E(q)
    xi = Eq.T @ dq
    g = np.zeros(3)
    for b in range(3):
        e = np.zeros(3); e[b] = eps
        qp = q + Eq @ e; qp = qp / np.linalg.norm(qp)
        qm = q - Eq @ e; qm = qm / np.linalg.norm(qm)
        g[b] = (core.logdetH(qp, M) - core.logdetH(qm, M)) / (2 * eps)
    return float(explicit + g @ xi)


def mode_data(I, theta):
    """Return q, J, H, logdetH for all minima at theta."""
    m = I.find_modes(I.M(theta), n_starts=N_STARTS)
    return m


def pose_and_derivs(I, theta, m=None):
    if m is None:
        m = mode_data(I, theta)
    q = m["q"]
    M, dM = I.M(theta), I.dM()
    J = np.asarray(m["J"], float)
    logdet = np.asarray(m["logdetH"], float)
    poses = np.array([I.pose(qq) for qq in q])              # (K,12)
    dq = np.array([core.ift_dq(qq, M, dM) for qq in q])    # (K,4)
    dposes = np.array([core.dpose_dq(qq, I.GinvD) @ dq[k] for k, qq in enumerate(q)])
    dJ = np.array([core.dJ_dtheta(qq, M, dM) for qq in q])
    dlogdet = np.array([dlogdet_numeric(qq, M, dM, dq[k]) for k, qq in enumerate(q)])
    return dict(q=q, J=J, logdet=logdet, poses=poses, dq=dq, dposes=dposes,
                dJ=dJ, dlogdet=dlogdet, H=m["H"])


def laplace(I, theta, tau=TAU, m=None):
    d = pose_and_derivs(I, theta, m)
    g = -d["J"] / tau - 0.5 * d["logdet"]          # log-weight value
    w = softmax(g)
    pose = w @ d["poses"]
    # d/dtheta of the log-weight (not the log-weight itself!)
    g_dot = -d["dJ"] / tau - 0.5 * d["dlogdet"]
    dw = w * (g_dot - float(w @ g_dot))
    dpose = w @ d["dposes"] + (dw[:, None] * d["poses"]).sum(0)
    return dict(pose=pose, dpose=dpose, w=w, modes=d)


def hard(I, theta):
    d = pose_and_derivs(I, theta)
    k = int(np.argmin(d["J"]))
    return dict(pose=d["poses"][k], dpose=d["dposes"][k], k=k, modes=d)


def mc(I, theta, tau=TAU, n=N_MC, rng=None, d=None):
    if d is None:
        d = pose_and_derivs(I, theta)
    g = -d["J"] / tau - 0.5 * d["logdet"]
    p = softmax(g)
    g_dot = -d["dJ"] / tau - 0.5 * d["dlogdet"]
    score = g_dot - float(p @ g_dot)
    rng = rng or np.random.default_rng(0)
    idx = rng.choice(len(p), size=n, p=p)
    poses = d["poses"][idx]
    pose = poses.mean(0)
    # score-function (likelihood-ratio) estimator of d E[pose] / d theta
    dpose = d["dposes"][idx].mean(0) + (poses * score[idx, None]).mean(0)
    return dict(pose=pose, dpose=dpose, p=p, modes=d)


def fd_laplace(I, theta, h=FD_H, **kw):
    p = laplace(I, theta + h, **kw)["pose"]
    m = laplace(I, theta - h, **kw)["pose"]
    return (p - m) / (2 * h)


def regime(theta):
    a = abs(theta)
    if a < 0.005:
        return "at"
    if a < 0.05:
        return "near"
    return "far"


def main():
    I = inst_mod.Instance(seed=20261127)
    rng = np.random.default_rng(20261127)
    rows = []
    for theta in THETA_GRID:
        t0 = time.perf_counter()
        L = laplace(I, theta)
        t_lap = time.perf_counter() - t0
        t0 = time.perf_counter()
        Hd = hard(I, theta)
        t_hard = time.perf_counter() - t0
        t0 = time.perf_counter()
        MC = mc(I, theta, rng=rng)
        t_mc = time.perf_counter() - t0
        fd = fd_laplace(I, theta)
        # hard FD: use the actual argmin branch; if the global branch switches
        # inside +-h, the finite difference is undefined -> record NaN.
        hp_full = hard(I, theta + FD_H)
        hm_full = hard(I, theta - FD_H)
        kp = hp_full["k"]; km = hm_full["k"]
        qp = hp_full["modes"]["q"][kp]; qm = hm_full["modes"]["q"][km]
        same_branch = abs(float(qp @ qm)) > 0.99
        if same_branch:
            fd_hard = (hp_full["pose"] - hm_full["pose"]) / (2 * FD_H)
            hard_defined = True
        else:
            fd_hard = np.full(12, np.nan)
            hard_defined = False
        # MC distribution over seeds: sample 20 independent MC means
        mc_means = []
        for s in range(MC_SEEDS):
            rr = np.random.default_rng(1000 + s)
            mc_means.append(mc(I, theta, rng=rr, d=L["modes"])["pose"])
        mc_means = np.array(mc_means)
        rows.append(dict(
            theta=float(theta), regime=regime(theta),
            n_modes=int(len(L["w"])), hard_k=Hd["k"],
            laplace_grad_rel=float(np.linalg.norm(L["dpose"] - fd) / max(np.linalg.norm(fd), 1e-12)),
            mc_grad_rel=float(np.linalg.norm(MC["dpose"] - fd) / max(np.linalg.norm(fd), 1e-12)),
            hard_grad_rel=(float(np.linalg.norm(Hd["dpose"] - fd_hard) / max(np.linalg.norm(fd_hard), 1e-12))
                           if hard_defined else None),
            mc_pose_std=float(np.linalg.norm(mc_means.std(0))),
            laplace_pose_norm=float(np.linalg.norm(L["pose"])),
            w_max=float(L["w"].max()),
            t_laplace_ms=1e3 * t_lap, t_hard_ms=1e3 * t_hard, t_mc_ms=1e3 * t_mc,
        ))
        if abs(theta) < 0.02:
            print("theta", f"{theta:+.4f}", "modes", len(L["w"]), "hard_k", Hd["k"],
                  "lap_grad", f"{rows[-1]['laplace_grad_rel']:.2e}",
                  "mc_grad", f"{rows[-1]['mc_grad_rel']:.2e}",
                  "hard_grad", rows[-1]["hard_grad_rel"], flush=True)
    with open(OUT / "raw_grid.json", "w") as f:
        json.dump(rows, f, indent=2)

    # aggregate by regime
    regs = {}
    for r in rows:
        regs.setdefault(r["regime"], []).append(r)

    def med(vals):
        vals = [v for v in vals if v is not None and np.isfinite(v)]
        return float(np.median(vals)) if vals else float("nan")

    table = []
    for reg in ["far", "near", "at"]:
        rs = regs.get(reg, [])
        if not rs:
            continue
        table.append(dict(
            regime=reg, n=len(rs),
            laplace_grad_rel=med([r["laplace_grad_rel"] for r in rs]),
            mc_grad_rel=med([r["mc_grad_rel"] for r in rs]),
            hard_grad_rel=med([r["hard_grad_rel"] for r in rs]),
            mc_pose_std=med([r["mc_pose_std"] for r in rs]),
            t_laplace_ms=med([r["t_laplace_ms"] for r in rs]),
            t_hard_ms=med([r["t_hard_ms"] for r in rs]),
            t_mc_ms=med([r["t_mc_ms"] for r in rs]),
        ))
    with open(OUT / "main_table.json", "w") as f:
        json.dump(table, f, indent=2)

    lines = ["# Laplace-QPEPLayer synthetic branch-crossing experiment\n",
             f"theta grid {THETA_GRID[0]:+.2f}..{THETA_GRID[-1]:+.2f} ({len(THETA_GRID)} points), "
             f"tau={TAU}, FD h={FD_H}, mode-level MC n={N_MC}, {MC_SEEDS} seeds\n",
             "| regime | n | Laplace grad rel | MC grad rel | Hard grad rel | MC pose std | t Laplace (ms) | t Hard (ms) | t MC (ms) |",
             "|-|-|-|-|-|-|-|-|-|"]
    for r in table:
        hgr = "N/A" if not np.isfinite(r["hard_grad_rel"]) else f"{r['hard_grad_rel']:.2e}"
        lines.append(f"| {r['regime']} | {r['n']} | {r['laplace_grad_rel']:.2e} | "
                     f"{r['mc_grad_rel']:.2e} | {hgr} | {r['mc_pose_std']:.2e} | "
                     f"{r['t_laplace_ms']:.2f} | {r['t_hard_ms']:.2f} | {r['t_mc_ms']:.2f} |")
    (OUT / "main_table.md").write_text("\n".join(lines) + "\n")

    # gradient-descent fit through the layer (optional; disabled when FIT_SEEDS=0)
    m0 = I.find_modes(I.M(0.0))
    order = np.argsort(m0["J"])
    if FIT_SEEDS > 0:
        target = 0.5 * (I.pose(m0["q"][order[0]]) + I.pose(m0["q"][order[1]]))
        fit_rows = []
        global N_STARTS
        N_STARTS_save = N_STARTS
        N_STARTS = 64
        for method in ["hard", "laplace", "mc"]:
            losses = []
            for seed in range(FIT_SEEDS):
                rng = np.random.default_rng(seed)
                theta = rng.uniform(-0.2, 0.2)
                lr = 0.01
                for step in range(FIT_STEPS):
                    if method == "hard":
                        out = hard(I, theta)
                    elif method == "laplace":
                        out = laplace(I, theta)
                    else:
                        out = mc(I, theta, rng=rng)
                    diff = out["pose"] - target
                    loss = float(diff @ diff)
                    g = 2.0 * diff @ out["dpose"]
                    theta = float(np.clip(theta - lr * g, -0.5, 0.5))
                losses.append(loss)
            fit_rows.append(dict(method=method, n=FIT_SEEDS, final_loss_median=float(np.median(losses)),
                                 final_loss_mean=float(np.mean(losses)),
                                 success=float(np.mean(np.array(losses) < 1e-4))))
        N_STARTS = N_STARTS_save
        with open(OUT / "fit_table.json", "w") as f:
            json.dump(fit_rows, f, indent=2)
        lines2 = ["# Gradient-descent fit through the layer\n",
                  "20 random initial theta in [-0.2,0.2], 200 steps, lr=0.01, target = average of the two global branch poses at theta=0.\n",
                  "| method | final loss median | final loss mean | success (loss<1e-4) |", "|-|-|-|-|"]
        for r in fit_rows:
            lines2.append(f"| {r['method']} | {r['final_loss_median']:.3e} | {r['final_loss_mean']:.3e} | {r['success']*100:.0f}% |")
        (OUT / "fit_table.md").write_text("\n".join(lines2) + "\n")

    # plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    th = np.array([r["theta"] for r in rows])
    ax[0].semilogy(th, [r["laplace_grad_rel"] for r in rows], label="Laplace")
    ax[0].semilogy(th, [r["mc_grad_rel"] for r in rows], label="MC")
    hard_vals = [r["hard_grad_rel"] if r["hard_grad_rel"] is not None and np.isfinite(r["hard_grad_rel"]) else np.nan for r in rows]
    ax[0].semilogy(th, hard_vals, ".", ms=2, label="Hard (N/A at crossing)")
    ax[0].set_xlabel("theta"); ax[0].set_ylabel("gradient relative error vs FD")
    ax[0].grid(alpha=0.3); ax[0].legend(fontsize=8)
    ax[1].plot(th, [r["mc_pose_std"] for r in rows], label="MC pose std")
    ax[1].set_xlabel("theta"); ax[1].set_ylabel("MC std (20 seeds)")
    ax[1].grid(alpha=0.3); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "main_table.png", dpi=160)
    print("wrote", OUT / "main_table.md", OUT / "fit_table.md", OUT / "main_table.png")


if __name__ == "__main__":
    main()
