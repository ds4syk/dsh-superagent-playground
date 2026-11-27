#!/usr/bin/env python3
"""P1 ablations after the supplement proofs:
  (A) tau sweep at crossing / near / far,
  (B) exact closed-form d logdet H vs tangent FD vs no-volume,
  (C) top-K mode truncation.
All references are branch-following finite differences of the *same* smoothed
map; absolute pose differences across variants are reported separately.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import qpep_core as core  # noqa: E402
import instance as inst_mod  # noqa: E402
import run_main as rm  # noqa: E402

OUT = HERE.parent / "results"
OUT.mkdir(parents=True, exist_ok=True)


def dlogdet_exact(q, M, dM, dq):
    """Closed form from the supplement: dE=E(dq), dJv=Jv(dq), no 3rd tensor."""
    J = core.Jval(q, M)
    H = core.hess_J(q, M)
    E = core.E(q)
    A = H - 4.0 * J * np.eye(4)
    Jv = core.Jv(q)
    dJv = core.Jv(dq)
    dE = core.E(dq)
    v = core.v2(q)
    df = float(v @ dM @ v)
    dH = 2.0 * (dJv.T @ M @ Jv + Jv.T @ dM @ Jv + Jv.T @ M @ dJv)
    for a in range(10):
        dH += 2.0 * float((dM @ v + M @ Jv @ dq)[a]) * core.HV[a]
    dHs = dE.T @ A @ E + E.T @ A @ dE + E.T @ (dH - 4.0 * df * np.eye(4)) @ E
    Hs = E.T @ A @ E
    return float(np.trace(np.linalg.solve(Hs, dHs))), Hs


def raw(I, theta, m=None):
    if m is None:
        m = I.find_modes(I.M(theta), n_starts=rm.N_STARTS)
    q = m["q"]; M, dM = I.M(theta), I.dM()
    J = np.asarray(m["J"], float)
    logdet = np.asarray(m["logdetH"], float)
    poses = np.array([I.pose(qq) for qq in q])
    dq = np.array([core.ift_dq(qq, M, dM) for qq in q])
    dposes = np.array([core.dpose_dq(qq, I.GinvD) @ dq[k] for k, qq in enumerate(q)])
    dJ = np.array([core.dJ_dtheta(qq, M, dM) for qq in q])
    dld_num = np.array([rm.dlogdet_numeric(qq, M, dM, dq[k]) for k, qq in enumerate(q)])
    dld_ex = np.array([dlogdet_exact(qq, M, dM, dq[k])[0] for k, qq in enumerate(q)])
    return dict(q=q, J=J, logdet=logdet, poses=poses, dq=dq, dposes=dposes,
                dJ=dJ, dld_num=dld_num, dld_ex=dld_ex, dld_zero=np.zeros_like(dJ))


def lap_variant(I, theta, tau=rm.TAU, mode="exact", topk=None, m=None):
    d = raw(I, theta, m)
    if mode == "exact":
        logdet = d["logdet"]; dld = d["dld_ex"]
    elif mode == "numeric":
        logdet = d["logdet"]; dld = d["dld_num"]
    elif mode == "none":
        logdet = np.zeros_like(d["logdet"]); dld = d["dld_zero"]
    else:
        raise ValueError(mode)
    g = -d["J"] / tau - 0.5 * logdet
    w = rm.softmax(g)
    if topk is not None and topk < len(w):
        keep = np.argsort(w)[-topk:]
        w2 = np.zeros_like(w); w2[keep] = w[keep]; w = w2 / w2.sum()
    pose = w @ d["poses"]
    gd = -d["dJ"] / tau - 0.5 * dld
    dw = w * (gd - float(w @ gd))
    dpose = w @ d["dposes"] + (dw[:, None] * d["poses"]).sum(0)
    return dict(pose=pose, dpose=dpose, w=w, d=d)


def fd_variant(I, theta, tau=rm.TAU, mode="exact", topk=None, h=1e-4):
    p = lap_variant(I, theta + h, tau, mode, topk)["pose"]
    m = lap_variant(I, theta - h, tau, mode, topk)["pose"]
    return (p - m) / (2 * h)


def main():
    I = inst_mod.Instance(seed=20261127)
    rows = []
    # (A) tau sweep
    for theta in [0.0, 0.01, -0.01, 0.1, -0.1]:
        for tau in [0.05, 0.02, 0.01, 0.005]:
            L = lap_variant(I, theta, tau, "exact")
            fd = fd_variant(I, theta, tau, "exact")
            rows.append(dict(exp="tau", theta=theta, tau=tau,
                             grad_rel=float(np.linalg.norm(L["dpose"] - fd) / max(np.linalg.norm(fd), 1e-12)),
                             pose_norm=float(np.linalg.norm(L["pose"])), w_max=float(L["w"].max())))
            print("tau", theta, tau, rows[-1]["grad_rel"], flush=True)
    # (B) logdet variants
    for theta in [0.0, 0.01, -0.01, 0.1, -0.1]:
        ref = lap_variant(I, theta, rm.TAU, "exact")
        fd_ref = fd_variant(I, theta, rm.TAU, "exact")
        for mode in ["exact", "numeric", "none"]:
            L = lap_variant(I, theta, rm.TAU, mode)
            fd = fd_variant(I, theta, rm.TAU, mode)
            rows.append(dict(exp="logdet", theta=theta, mode=mode,
                             grad_rel=float(np.linalg.norm(L["dpose"] - fd) / max(np.linalg.norm(fd), 1e-12)),
                             pose_diff_vs_exact=float(np.linalg.norm(L["pose"] - ref["pose"])),
                             grad_diff_vs_exact=float(np.linalg.norm(L["dpose"] - ref["dpose"])),
                             fd_diff_vs_exact=float(np.linalg.norm(fd - fd_ref))))
            print("logdet", theta, mode, rows[-1]["grad_rel"], rows[-1]["pose_diff_vs_exact"], flush=True)
    # (C) top-K modes
    for theta in [0.0, 0.01, 0.1]:
        ref = lap_variant(I, theta, rm.TAU, "exact", None)
        for k in [1, 2, 3, 4]:
            L = lap_variant(I, theta, rm.TAU, "exact", k)
            rows.append(dict(exp="topk", theta=theta, k=k,
                             pose_diff=float(np.linalg.norm(L["pose"] - ref["pose"])),
                             grad_diff=float(np.linalg.norm(L["dpose"] - ref["dpose"])),
                             w_sum=float(L["w"].sum())))
            print("topk", theta, k, rows[-1]["pose_diff"], flush=True)
    json.dump(rows, open(OUT / "ablation.json", "w"), indent=2)

    lines = ["# P1 ablation table (post-supplement)\n"]
    lines += ["## A. tau sweep (exact-logdet Laplace, gradient vs branch FD)\n",
              "| theta | tau=0.05 | 0.02 | 0.01 | 0.005 |", "|-|-|-|-|-|"]
    for theta in [0.0, 0.01, -0.01, 0.1, -0.1]:
        vals = [next(r for r in rows if r["exp"] == "tau" and r["theta"] == theta and r["tau"] == t)["grad_rel"] for t in [0.05, 0.02, 0.01, 0.005]]
        lines.append("| %+.3f | " % theta + " | ".join(f"{v:.2e}" for v in vals) + " |")
    lines += ["", "## B. logdet variants (theta=0 and near/far)\n",
              "| theta | mode | grad rel | pose diff vs exact | grad diff vs exact |",
              "|-|-|-|-|-|"]
    for r in rows:
        if r["exp"] == "logdet":
            lines.append(f"| {r['theta']:+.3f} | {r['mode']} | {r['grad_rel']:.2e} | {r['pose_diff_vs_exact']:.2e} | {r['grad_diff_vs_exact']:.2e} |")
    lines += ["", "## C. top-K mode truncation (reference: all modes)\n",
              "| theta | K | pose diff | grad diff |", "|-|-|-|-|"]
    for r in rows:
        if r["exp"] == "topk":
            lines.append(f"| {r['theta']:+.3f} | {r['k']} | {r['pose_diff']:.2e} | {r['grad_diff']:.2e} |")
    (OUT / "ablation.md").write_text("\n".join(lines) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
