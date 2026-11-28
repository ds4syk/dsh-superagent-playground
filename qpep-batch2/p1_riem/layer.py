"""Prototype 1: RiemQPEP-Layer — 3x3 切空间隐式微分。"""
from __future__ import annotations

import numpy as np

from core import algebra as A


def forward(prob_or_Q, q0=None, n_starts=64, seed=0):
    """前向：返回全局极小 q*（多起点枚举后取最低）。"""
    from core import modes
    Q = prob_or_Q.Qm if hasattr(prob_or_Q, "Qm") else prob_or_Q
    if q0 is not None:
        q, gn, ok = modes.newton(q0, Q)
        return q, A.fval(q, Q), gn, ok
    res = modes.find_minima(Q, n_starts=n_starts, seed=seed)
    if len(res["q"]) == 0:
        raise RuntimeError("no local minimum found")
    return res["q"][0], float(res["f"][0]), None, True


def riemann_hessian(q, Q):
    return A.riemann_hess(q, Q)


def vjp(Q, q, gq):
    """给定上游梯度 gq=∂ℓ/∂q*，返回 ∂ℓ/∂Q（对称 10x10）与中间量。"""
    E = A.E(q)
    J = A.Jm(q)
    v = A.m2(q)
    H = A.riemann_hess(q, Q)
    eta = np.linalg.solve(H, E.T @ gq)
    u = -2.0 * (J @ E @ eta)          # (10,)
    GQ = 0.5 * (np.outer(u, v) + np.outer(v, u))
    return GQ, dict(eta=eta, u=u, H=H, m2=v)


def translation_vjp(prob, q, gt):
    """平移损失 ℓ(t*(q*)) 对 q 的等效梯度（叠加到旋转梯度上）。"""
    # dt*/dq = -Mtt^{-1} MRt^T dvecR/dq
    dR = prob.dt_dq(q)               # (3,4)
    return dR.T @ gt


def vjp_with_translation(prob, q, gq, gt):
    gq_tot = gq + translation_vjp(prob, q, gt)
    return vjp(prob.Qm, q, gq_tot)
