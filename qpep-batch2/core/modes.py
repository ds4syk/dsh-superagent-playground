"""在 S^3 上枚举 QPEP 四元数四次型的临界点/局部极小点。

只用黎曼牛顿 + 多起点；这是**数值枚举**，不是代数全体根证书。
对每个返回点都检查一阶条件 ||E^T∇f|| 与二阶签名。
"""
from __future__ import annotations

import numpy as np

from . import algebra as A


def _normalize(q):
    return q / np.linalg.norm(q)


def newton(q, Q, max_iter=100, tol=1e-9, damping=0.0):
    """黎曼牛顿（带回溯线搜索）。返回 (q, grad_tangent_norm, ok)。"""
    q = _normalize(np.asarray(q, float))
    f = A.fval(q, Q)
    for _ in range(max_iter):
        g = A.grad(q, Q)
        Eg = A.E(q).T @ g
        gn = float(np.linalg.norm(Eg))
        if gn < tol:
            return q, gn, True
        H = A.riemann_hess(q, Q)
        w = np.linalg.eigvalsh(H)
        if w[0] < 1e-10:
            H = H + (1e-10 - w[0] + damping) * np.eye(3)
        elif damping > 0:
            H = H + damping * np.eye(3)
        try:
            xi = np.linalg.solve(H, -Eg)
        except np.linalg.LinAlgError:
            return q, gn, False
        nxi = float(np.linalg.norm(xi))
        if nxi > 1.0:
            xi = xi / nxi
            nxi = 1.0
        slope = float(Eg @ xi)
        alpha = 1.0
        accepted = False
        for _ls in range(25):
            qn = _normalize(q + A.E(q) @ (alpha * xi))
            fn = A.fval(qn, Q)
            if fn <= f + 1e-4 * alpha * slope + 1e-15:
                q, f = qn, fn
                accepted = True
                break
            alpha *= 0.5
        if not accepted:
            return q, gn, False
    return q, float(np.linalg.norm(A.E(q).T @ A.grad(q, Q))), False


def riemann_hess_eigs(q, Q):
    return np.linalg.eigvalsh(A.riemann_hess(q, Q))


def find_stationary(Q, n_starts=128, seed=0, tol=1e-9, extra_starts=None):
    """多起点枚举临界点。返回 dict：q, f, h_eigs, kind。"""
    rng = np.random.default_rng(seed)
    starts = [np.array([1.0, 0.0, 0.0, 0.0])]
    if extra_starts is not None:
        starts.extend(np.asarray(extra_starts, float))
    for _ in range(n_starts):
        v = rng.normal(size=4)
        starts.append(_normalize(v))
    qs, fs, eigs, ok = [], [], [], []
    for s in starts:
        q, gn, good = newton(s, Q, tol=tol)
        if gn > 1e-6:
            continue
        qs.append(q)
        fs.append(A.fval(q, Q))
        eigs.append(riemann_hess_eigs(q, Q))
        ok.append(good)
    # 去重（双覆盖 |q·q'|≈1）
    keep = []
    for i in range(len(qs)):
        dup = False
        for j in keep:
            if abs(float(qs[i] @ qs[j])) > 1.0 - 1e-7:
                dup = True
                break
        if not dup:
            keep.append(i)
    qs = np.array([qs[i] for i in keep])
    fs = np.array([fs[i] for i in keep])
    eigs = np.array([eigs[i] for i in keep])
    # 双覆盖规范化：q0 >= 0
    flip = qs[:, 0] < 0
    qs[flip] *= -1
    kind = np.array(["min" if e[0] > 1e-7 else
                     ("saddle" if e[0] < -1e-7 else "flat") for e in eigs])
    order = np.argsort(fs)
    return dict(q=qs[order], f=fs[order], h_eigs=eigs[order], kind=kind[order])


def find_minima(Q, n_starts=128, seed=0, tol=1e-9, extra_starts=None):
    res = find_stationary(Q, n_starts=n_starts, seed=seed, tol=tol,
                          extra_starts=extra_starts)
    mask = res["kind"] == "min"
    return {k: (v[mask] if isinstance(v, np.ndarray) else v) for k, v in res.items()}


def count_roots(Q, n_starts=512, seed=0):
    res = find_stationary(Q, n_starts=n_starts, seed=seed)
    return len(res["q"]), res
