"""Prototype 2: BarrierQPEP-Layer — 同伦对数障碍 + 统一 3x3 隐式微分。"""
from __future__ import annotations

import numpy as np

from core import algebra as A


def barrier_grad_hess(q, Q, cs, tau):
    v = A.m2(q)
    g = cs @ v
    if np.any(g >= 0):
        raise ValueError(f"barrier evaluated outside the feasible set: g={g}")
    mu = -tau / g
    J = A.Jm(q)
    Gf = A.grad(q, Q)
    Hf = A.hess(q, Q)
    Gphi = Gf.copy()
    Hphi = Hf.copy()
    for j, c in enumerate(cs):
        gj = J.T @ c                       # ∇g_j
        Hgj = sum(c[a] * A.B_MATS[a] for a in range(10))
        Gphi = Gphi + mu[j] * gj
        Hphi = Hphi + mu[j] * Hgj + (mu[j] ** 2 / tau) * np.outer(gj, gj)
    lam = float(q @ Gphi)
    E = A.E(q)
    H = E.T @ (Hphi - lam * np.eye(4)) @ E
    return Gphi, H, g, mu


def barrier_value(q, Q, cs, tau):
    g = cs @ A.m2(q)
    if np.any(g >= 0):
        return np.inf
    return A.fval(q, Q) - tau * np.sum(np.log(-g))


def solve(q0, Q, cs, tau, max_iter=100, tol=1e-10):
    """从可行内点 q0 出发的黎曼牛顿（回溯保持严格可行）。"""
    q = np.asarray(q0, float).copy()
    q /= np.linalg.norm(q)
    if np.any(cs @ A.m2(q) >= 0):
        raise ValueError("initial point is not strictly feasible")
    phi = barrier_value(q, Q, cs, tau)
    for _ in range(max_iter):
        Gphi, H, g, mu = barrier_grad_hess(q, Q, cs, tau)
        Eg = A.E(q).T @ Gphi
        if np.linalg.norm(Eg) < tol:
            break
        try:
            xi = np.linalg.solve(H, -Eg)
        except np.linalg.LinAlgError:
            xi = np.linalg.lstsq(H, -Eg, rcond=None)[0]
        slope = float(Eg @ xi)
        alpha = 1.0
        accepted = False
        for _ls in range(50):
            qn = q + A.E(q) @ (alpha * xi)
            qn /= np.linalg.norm(qn)
            gn = cs @ A.m2(qn)
            if np.all(gn < 0):
                phin = barrier_value(qn, Q, cs, tau)
                if phin <= phi + 1e-4 * alpha * slope:
                    q, phi = qn, phin
                    accepted = True
                    break
            alpha *= 0.5
        if not accepted:
            # 保守梯度下降兜底：只接受既可行又下降的步
            stepped = False
            for alpha in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6):
                qn = q - alpha * (A.E(q) @ Eg)
                qn /= np.linalg.norm(qn)
                if np.all(cs @ A.m2(qn) < 0):
                    phin = barrier_value(qn, Q, cs, tau)
                    if phin < phi:
                        q, phi, stepped = qn, phin, True
                        break
            if not stepped:
                break
    return q, phi


def vjp(q, Q, cs, tau, gq):
    """固定 τ 的 ∂ℓ/∂Q 与 ∂ℓ/∂c_j（对称梯度）。"""
    Gphi, H, g, mu = barrier_grad_hess(q, Q, cs, tau)
    E = A.E(q)
    eta = np.linalg.solve(H, E.T @ gq)
    v = A.m2(q)
    Jv = A.Jm(q) @ E @ eta
    u = -2.0 * Jv
    GQ = 0.5 * (np.outer(u, v) + np.outer(v, u))
    Gc = []
    for j, c in enumerate(cs):
        Gc.append(-mu[j] * Jv - (mu[j] ** 2 / tau) * float(Jv @ c) * v)
    return GQ, np.array(Gc), dict(eta=eta, u=u, Jv=Jv, H=H, mu=mu, g=g)


def active_set_limit(q, Q, cs, tol=1e-4, mu=None):
    """积极集 KKT 的约化逆（用于检验 τ->0 收敛）。

    H_L = E^T(∇²f + Σ_{j∈A} μ_j ∇²g_j − λ I)E，λ = q^T(∇f + Σ μ_j∇g_j)。
    """
    v = A.m2(q)
    g = cs @ v
    if mu is None:
        mu = np.zeros(len(cs))
    active = np.where(g > -tol)[0]
    E = A.E(q)
    Hphi = A.hess(q, Q)
    Gphi = A.grad(q, Q)
    for j in active:
        c = cs[j]
        Hgj = sum(c[a] * A.B_MATS[a] for a in range(10))
        Hphi = Hphi + mu[j] * Hgj
        Gphi = Gphi + mu[j] * (A.Jm(q).T @ c)
    lam = float(q @ Gphi)
    H_L = E.T @ (Hphi - lam * np.eye(4)) @ E
    if len(active) == 0:
        return np.linalg.inv(H_L), active
    A_A = np.stack([E.T @ (A.Jm(q).T @ cs[j]) for j in active], axis=1)
    U, s, Vt = np.linalg.svd(A_A, full_matrices=True)
    free = U[:, len(active):]
    M = free.T @ H_L @ free
    Hinv = free @ np.linalg.inv(M) @ free.T
    return Hinv, active
