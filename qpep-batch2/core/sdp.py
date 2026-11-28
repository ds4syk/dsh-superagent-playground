"""10x10 Shor/SOS 松弛下界（对应 SDPRLayers 的旋转子问题）。"""
from __future__ import annotations

import numpy as np

from . import algebra as A

try:
    import cvxpy as cp
except Exception:  # pragma: no cover
    cp = None


def sdp_bound(Qm, solver=None, eps=1e-9):
    """max γ s.t. Qm - γ u u^T + Σ_i y_i H_i ⪰ 0（21 条约束，冗余但等价）。

    返回 dict(gamma, status, M, eig)。
    """
    if cp is None:
        raise RuntimeError("cvxpy is required for the SDP lower bound")
    gamma = cp.Variable()
    ys = cp.Variable(A.H_CONSTRAINTS.shape[0])
    M = Qm - gamma * np.outer(A.U_NORM, A.U_NORM)
    for i in range(A.H_CONSTRAINTS.shape[0]):
        M = M + ys[i] * A.H_CONSTRAINTS[i]
    prob = cp.Problem(cp.Maximize(gamma), [M >> 0])
    if solver is None:
        solver = cp.CLARABEL
    prob.solve(solver=solver, **({} if solver != cp.SCS else dict(eps=eps, max_iters=50000)))
    Mv = M.value
    return dict(gamma=float(prob.value), status=prob.status,
                M=None if Mv is None else np.asarray(Mv),
                eig=None if Mv is None else np.linalg.eigvalsh(Mv))


def sdp_rank(M):
    eig = np.linalg.eigvalsh(M)
    eig = np.clip(eig, 0, None)
    if eig[-1] <= 0:
        return 0, eig
    return int(np.sum(eig > 1e-6 * eig[-1])), eig


def sdp_primal(Qm, solver=None, eps=1e-9):
    """原问题：min <Qm,X> s.t. X⪰0, <H_i,X>=0, <u u^T,X>=1。

    返回 X 及其特征值；在双极小简并处最优 X 应为秩 2。
    """
    if cp is None:
        raise RuntimeError("cvxpy is required")
    X = cp.Variable((10, 10), symmetric=True)
    cons = [X >> 0, cp.trace(np.outer(A.U_NORM, A.U_NORM) @ X) == 1]
    for H in A.H_CONSTRAINTS:
        cons.append(cp.trace(H @ X) == 0)
    prob = cp.Problem(cp.Minimize(cp.trace(Qm @ X)), cons)
    solver = solver or cp.CLARABEL
    prob.solve(solver=solver, **({} if solver != cp.SCS else dict(eps=eps, max_iters=50000)))
    Xv = np.asarray(X.value)
    return dict(value=float(prob.value), X=Xv,
                eig=np.linalg.eigvalsh(Xv), status=prob.status)
