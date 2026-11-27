#!/usr/bin/env python
"""14×14 Veronese 增强 Shor 松弛（Lemma 2.2）—— sound 后验证书。

与 verify/sanity_constraints_alm.py 的 shor14 等价，但接口化：
  shor_bound(M0,D0,G0,A,B, mu=None, add_veronese=True, enforce_ineq=True)
返回 (value, lambda2/lambda1, lambda1, X, solve_time)。

两种用法：
  * enforce_ineq=True, mu=None  : 直接对约束问题求 Shor 下界（主用法）
  * enforce_ineq=False, mu!=None: 对偶函数 d(mu) 的 Shor 下界（对照，见 proofs.md §2.2）
"""
import time

import cvxpy as cp
import numpy as np


def _veronese_constraints(X, n=14):
    """S=qq^T 的 2×2 子式线性化（v2 坐标 c_ab=1/√2）。"""
    cons = []
    idx = {}
    for i in range(4):
        idx[(i, i)] = i
    k = 4
    for i in range(4):
        for j in range(i + 1, 4):
            idx[(i, j)] = k
            k += 1
    coef = lambda a, b: 1.0 if a == b else 1.0 / np.sqrt(2.0)
    for i in range(4):
        for j in range(i + 1, 4):
            for kk in range(4):
                for ll in range(kk + 1, 4):
                    a1 = idx[(min(i, kk), max(i, kk))]
                    b1 = idx[(min(j, ll), max(j, ll))]
                    a2 = idx[(min(i, ll), max(i, ll))]
                    b2 = idx[(min(j, kk), max(j, kk))]
                    c1 = coef(min(i, kk), max(i, kk)) * coef(min(j, ll), max(j, ll))
                    c2 = coef(min(i, ll), max(i, ll)) * coef(min(j, kk), max(j, kk))
                    cons.append(c1 * X[a1, b1] == c2 * X[a2, b2])
    return cons


def shor_bound(M0, D0, G0, A, B, mu=None, add_veronese=True, enforce_ineq=True,
               add_rlt=False, solver="SCS", eps=1e-9, max_iters=20000, time_limit=None):
    """min <Qh,X> s.t. X⪰0, X[13,13]=1, e_tr^T v2=1, ||v2||^2=1, Veronese, g_j<=0.

    Qh 来自 J（mu=None）或 J+Σμ_j g_j（mu!=None，此时 enforce_ineq 应为 False）。
    """
    n = 14
    Qm = np.zeros((13, 13))
    Qm[:10, :10] = M0
    Qm[:10, 10:] = D0.T
    Qm[10:, :10] = D0
    Qm[10:, 10:] = G0
    Qh = np.zeros((n, n))
    Qh[:13, :13] = Qm
    e14 = np.zeros(n)
    e14[13] = 1.0

    lhs_lin = np.zeros((n, n))
    if mu is not None:
        for j in range(A.shape[0]):
            if mu[j] == 0.0:
                continue
            c14 = np.zeros(n)
            c14[:10] = A[j]
            c14[10:13] = B[j]
            lhs_lin += mu[j] * 0.5 * (np.outer(c14, e14) + np.outer(e14, c14))

    X = cp.Variable((n, n), symmetric=True)
    e_tr14 = np.zeros(n)
    e_tr14[:10] = np.array([1.0, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    E1 = 0.5 * (np.outer(e_tr14, e14) + np.outer(e14, e_tr14))
    cons = [X >> 0, X[13, 13] == 1.0, cp.trace(X[:10, :10]) == 1.0,
            cp.trace(E1 @ X) == 1.0]
    if add_veronese:
        cons += _veronese_constraints(X, n)
    c14s = []
    if enforce_ineq:
        for j in range(A.shape[0]):
            c14 = np.zeros(n)
            c14[:10] = A[j]
            c14[10:13] = B[j]
            c14s.append(c14)
            Cj = 0.5 * (np.outer(c14, e14) + np.outer(e14, c14))
            cons.append(cp.trace(Cj @ X) <= 0.0)
    if add_rlt and enforce_ineq:
        # Zero-dimensional RLT cuts: g_i(y) g_j(y) >= 0 for all feasible
        # y (g <= 0), linear in Y = y y^T.  Sound and adds no lifting.
        for i in range(len(c14s)):
            for j in range(i, len(c14s)):
                Cij = 0.5 * (np.outer(c14s[i], c14s[j]) + np.outer(c14s[j], c14s[i]))
                cons.append(cp.trace(Cij @ X) >= 0.0)
    prob = cp.Problem(cp.Minimize(cp.trace((Qh + 2.0 * lhs_lin) @ X)), cons)
    t0 = time.perf_counter()
    kw = dict(solver=solver)
    if solver == "SCS":
        kw["eps"] = eps
        kw["max_iters"] = max_iters
    elif solver == "CLARABEL":
        kw["tol_gap_abs"] = eps
        kw["tol_gap_rel"] = eps
        kw["tol_feas"] = eps
    if time_limit is not None:
        kw["time_limit_secs"] = time_limit
    try:
        prob.solve(**kw)
    except Exception as exc:              # 求解器失败也要返回可诊断信息
        return dict(value=float("nan"), ratio=float("nan"), lam1=float("nan"),
                    status="error:%s" % exc, time_s=time.perf_counter() - t0)
    dt = time.perf_counter() - t0
    if X.value is None:
        return dict(value=float("nan"), ratio=float("nan"), lam1=float("nan"),
                    status=str(prob.status), time_s=dt)
    ev = np.sort(np.linalg.eigvalsh(X.value))
    lam1 = float(ev[-1])
    ratio = float(ev[-2] / max(lam1, 1e-300))
    return dict(value=float(prob.value), ratio=ratio, lam1=lam1,
                status=str(prob.status), time_s=dt)
