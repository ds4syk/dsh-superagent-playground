#!/usr/bin/env python
"""E2: BarrierQPEP-Layer — 同伦对数障碍、O(τ) 积极集收敛、3x3 VJP。"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from core import algebra as A, modes, problems as P  # noqa: E402
from p1_riem import layer as P1  # noqa: E402
from p2_barrier import constraints as C, layer as L  # noqa: E402


def constrained_reference(prob, cs):
    """SLSQP 多起点求 min f s.t. c^T m2 <= 0, ||q||=1。"""
    cons = [{"type": "eq", "fun": lambda q: q @ q - 1, "jac": lambda q: 2 * q},
            {"type": "ineq", "fun": lambda q: -np.array([c @ A.m2(q) for c in cs]),
             "jac": lambda q: -np.array([A.Jm(q).T @ c for c in cs])}]
    rng = np.random.default_rng(0)
    best = None
    for _ in range(80):
        q0 = rng.normal(size=4); q0 /= np.linalg.norm(q0)
        r = minimize(lambda q: prob.f(q), q0, jac=lambda q: prob.grad_f(q),
                     constraints=cons, method="SLSQP",
                     options={"ftol": 1e-14, "maxiter": 800})
        if r.success and (best is None or r.fun < best.fun):
            best = r
    return None if best is None else (best.x, float(best.fun))


def slack_kkt_singularity():
    """平方松弛的 2x2 KKT 块 [[2μ,2s],[2s,0]] 在弱活跃点秩亏。"""
    rows = []
    for s, mu in [(1.0, 0.0), (0.0, 1.0), (0.0, 0.0)]:
        K = np.array([[2 * mu, 2 * s], [2 * s, 0.0]])
        rows.append(dict(s=s, mu=mu, det=float(np.linalg.det(K)),
                         sigma_min=float(np.linalg.svd(K, compute_uv=False)[-1])))
    return rows


def main():
    rng = np.random.default_rng(20261128)
    prob = P.random_instance(seed=2)
    q_unc, f_unc, _, _ = P1.forward(prob, n_starts=48)
    cs = C.tilt_constraint(30.0)[None, :]
    gq = rng.normal(size=4)

    q = np.array([1.0, 0, 0, 0])
    taus = [1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 1e-6]
    rows = []
    for tau in taus:
        q, phi = L.solve(q, prob.Qm, cs, tau)
        Gphi, H, g, mu = L.barrier_grad_hess(q, prob.Qm, cs, tau)
        eta = np.linalg.solve(H, A.E(q).T @ gq)
        Hinv, active = L.active_set_limit(q, prob.Qm, cs, tol=1e-3, mu=mu)
        eta_lim = Hinv @ A.E(q).T @ gq
        err = float(np.linalg.norm(eta - eta_lim) / np.linalg.norm(eta_lim))
        rows.append(dict(tau=tau, f=float(prob.f(q)), g=float(g[0]),
                         grad_err=err, min_eig_H=float(np.linalg.eigvalsh(H)[0]),
                         active=active.tolist()))
    slope = float(np.polyfit(np.log([r["tau"] for r in rows]),
                             np.log([r["grad_err"] for r in rows]), 1)[0])

    # 固定 τ 的 VJP 校验（Q 与 c）：必须先用同伦走到该 τ
    tau = 1e-3
    q = np.array([1.0, 0, 0, 0])
    for tt in [1e-1, 3e-2, 1e-2, 3e-3, 1e-3]:
        q, _ = L.solve(q, prob.Qm, cs, tt)
    dS = rng.normal(size=(10, 10)); dQ = 0.5 * (dS + dS.T)
    h = 1e-4
    # Q 方向：解析 vs 用 Q± 重解再求 gq·q
    qp, _ = L.solve(q, prob.Qm + h * dQ, cs, tau)
    qm, _ = L.solve(q, prob.Qm - h * dQ, cs, tau)
    fd_Q = float((gq @ qp - gq @ qm) / (2 * h))
    GQ, Gc, _ = L.vjp(q, prob.Qm, cs, tau, gq)
    an_Q = float(np.sum(GQ * dQ))
    # c 方向（单侧：沿 −m2 收紧约束，保证起点仍严格可行）
    dc = -A.m2(q)
    cp = cs + h * dc[None, :]
    qp, _ = L.solve(q, prob.Qm, cp, tau)
    fd_c = float((gq @ qp - gq @ q) / h)
    an_c = float(np.sum(Gc * dc[None, :]))

    ref = constrained_reference(prob, cs)
    out = dict(
        unconstrained=dict(f=f_unc, q=q_unc.tolist(), g=float(C.gvals(q_unc, cs)[0])),
        homotopy=rows, slope=slope,
        vjp=dict(rel_Q=abs(an_Q - fd_Q) / max(1e-14, abs(fd_Q)),
                 rel_c=abs(an_c - fd_c) / max(1e-14, abs(fd_c)),
                 an_Q=an_Q, fd_Q=fd_Q, an_c=an_c, fd_c=fd_c),
        reference=None if ref is None else dict(f=ref[1], q=ref[0].tolist()),
        slack_kkt=slack_kkt_singularity(),
    )
    with open(os.path.join(ROOT, "results", "e2_barrier.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
