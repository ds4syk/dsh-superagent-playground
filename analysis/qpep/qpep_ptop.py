"""QPEP 结构实验：point-to-plane registration（论文 Eq. (9)）。

问题
----
min_{R in SO(3), t}  L(R,t) = sum_i ( n_i^T (b_i - R u_i - t) )^2
其中 (u_i, b_i, n_i) 是第 i 组对应点/法向。

这是 QPEP 论文列出的问题族之一（Eq. (9)），其代数量（四元数 q）形式与 PnP 相同：
固定 q 时对 t 是二次的；消去 t 后 L(q) 是 q 的四次多项式，所以 KKT 是三次系统。

本模块提供：
- 四元数 -> 旋转矩阵 R(q)、dR/dq_a、d2R/dq_a dq_b 的解析式；
- L(q)、grad L(q)、Hessian L(q) 的解析式（含 t 消元）；
- full KKT 系统 (grad L - lambda q = 0, q^T q = 1) 的全部实根（向量化 LM 多起点枚举）；
- 四组 3-of-4 约化系统（lambda = 0）的全部实根；
- full KKT 的隐式微分与约化系统的隐式微分。

所有函数对 data 支持 complex（用于 complex-step 求 d/dtheta）。
"""

from __future__ import annotations

import numpy as np


# ----------------------------------------------------------------------------
# quaternion -> rotation
# ----------------------------------------------------------------------------
def quat_to_R(q):
    """q = (q0, q1, q2, q3) -> R (3, 3)."""
    q0, q1, q2, q3 = q
    return np.array(
        [
            [q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3,
             2 * (q1 * q2 - q0 * q3),
             2 * (q1 * q3 + q0 * q2)],
            [2 * (q1 * q2 + q0 * q3),
             q0 * q0 - q1 * q1 + q2 * q2 - q3 * q3,
             2 * (q2 * q3 - q0 * q1)],
            [2 * (q1 * q3 - q0 * q2),
             2 * (q2 * q3 + q0 * q1),
             q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3],
        ],
        dtype=np.result_type(q, np.float64),
    )


def quat_dR(q):
    """Return dR/dq_a for a = 0..3, each (3,3)."""
    q0, q1, q2, q3 = q
    v = np.array([q1, q2, q3], dtype=np.result_type(q, np.float64))
    I = np.eye(3, dtype=v.dtype)
    dR = np.zeros((4, 3, 3), dtype=v.dtype)

    # dR/dq0 = 2 q0 I + 2 skew(v)
    dR[0] = 2 * q0 * I + 2 * np.array(
        [[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], dtype=v.dtype
    )

    # dR/dv_j = -2 v_j I + 2 (e_j v^T + v e_j^T) + 2 q0 skew(e_j)
    for j in range(3):
        e = np.zeros(3, dtype=v.dtype)
        e[j] = 1.0
        skew_e = np.array(
            [[0, -e[2], e[1]], [e[2], 0, -e[0]], [-e[1], e[0], 0]], dtype=v.dtype
        )
        dR[j + 1] = (
            -2 * v[j] * I
            + 2 * (np.outer(e, v) + np.outer(v, e))
            + 2 * q0 * skew_e
        )
    return dR


def quat_d2R(q):
    """Return d2R/dq_a dq_b, shape (4,4,3,3)."""
    dt = np.result_type(q, np.float64)
    v = np.array([q[1], q[2], q[3]], dtype=dt)
    q0 = q[0]
    I = np.eye(3, dtype=dt)
    d2 = np.zeros((4, 4, 3, 3), dtype=dt)

    # d2R/dq0^2 = 2 I ; d2R/dq0 dv_j = 2 skew(e_j)
    d2[0, 0] = 2 * I
    for j in range(3):
        e = np.zeros(3, dtype=dt)
        e[j] = 1.0
        skew_e = np.array(
            [[0, -e[2], e[1]], [e[2], 0, -e[0]], [-e[1], e[0], 0]], dtype=dt
        )
        d2[0, j + 1] = 2 * skew_e
        d2[j + 1, 0] = 2 * skew_e

    # d2R/dv_j dv_k = -2 delta_jk I + 2 (e_j e_k^T + e_k e_j^T)
    for j in range(3):
        for k in range(3):
            ej = np.zeros(3, dtype=dt)
            ek = np.zeros(3, dtype=dt)
            ej[j] = 1.0
            ek[k] = 1.0
            d2[j + 1, k + 1] = -2 * (1.0 if j == k else 0.0) * I + 2 * (
                np.outer(ej, ek) + np.outer(ek, ej)
            )
    return d2


# ----------------------------------------------------------------------------
# objective L(q) (t eliminated) and derivatives
# ----------------------------------------------------------------------------
def _prep(data):
    u, b, n = data["u"], data["b"], data["n"]
    A = np.einsum("ij,ik->jk", n, n)
    return u, b, n, A, np.linalg.inv(A)


def loss_grad_hess(q, data):
    """L(q), grad L(q), Hessian L(q), with t = A^{-1} d(R) eliminated."""
    u, b, n, A, Ainv = _prep(data)
    N = u.shape[0]

    R = quat_to_R(q)
    dR = quat_dR(q)
    d2R = quat_d2R(q)

    Ru = u @ R.T                      # (N,3)
    e = b - Ru                        # (N,3)
    ne = np.einsum("ij,ij->i", n, e)  # n_i^T e_i
    s = np.sum(ne * ne)
    d = np.einsum("ij,i->j", n, ne)   # sum_i n_i n_i^T e_i

    L = s - d @ Ainv @ d

    # first derivatives
    dRu = np.empty((N, 4, 3), dtype=np.result_type(q, np.float64))
    for a in range(4):
        dRu[:, a, :] = u @ dR[a].T
    c = np.einsum("nj,naj->na", n, dRu)          # c[n,a] = n_n^T dR_a u_n
    ds = -2.0 * np.einsum("n,na->a", ne, c)      # ds_a = -2 sum_n ne_n c[n,a]
    dd = -np.einsum("nj,na->aj", n, c)           # dd[a,:] = -sum_n n_n c[n,a]
    gd = dd @ Ainv
    dL = ds - 2.0 * gd @ d

    # second derivatives
    H = np.empty((4, 4), dtype=np.result_type(q, np.float64))
    for a in range(4):
        for b in range(4):
            d2Ru = u @ d2R[a, b].T
            nd2Ru = np.einsum("nj,nj->n", n, d2Ru)
            d2s = (
                2.0 * np.dot(c[:, a], c[:, b])
                - 2.0 * np.einsum("n,n->", ne, nd2Ru)
            )
            d2d = -np.einsum("nj,n->j", n, nd2Ru)
            H[a, b] = d2s - 2.0 * (dd[a] @ Ainv @ dd[b]) - 2.0 * (d @ Ainv @ d2d)
    return L, dL, H


# ----------------------------------------------------------------------------
# residuals / Jacobians
# ----------------------------------------------------------------------------
def resid_full(z, data):
    """z = (q(4), lambda). F = [grad L(q) - lambda q ; q.q - 1]."""
    q = z[:4]
    lam = z[4]
    _, dL, _ = loss_grad_hess(q, data)
    F = np.empty(5, dtype=np.result_type(dL, q, np.float64))
    F[:4] = dL - lam * q
    F[4] = np.dot(q, q) - 1.0
    return F


def jac_full(z, data):
    q = z[:4]
    lam = z[4]
    _, _, H = loss_grad_hess(q, data)
    J = np.zeros((5, 5), dtype=np.result_type(z, np.float64))
    J[:4, :4] = H - lam * np.eye(4)
    J[:4, 4] = -q
    J[4, :4] = 2.0 * q
    return J


def resid_branch(q, data, k):
    """reduced system: drop row k, impose lambda = 0."""
    _, dL, _ = loss_grad_hess(q, data)
    keep = [i for i in range(4) if i != k]
    F = np.empty(4, dtype=np.result_type(dL, q, np.float64))
    F[:3] = dL[keep]
    F[3] = np.dot(q, q) - 1.0
    return F


def jac_branch(q, data, k):
    _, _, H = loss_grad_hess(q, data)
    keep = [i for i in range(4) if i != k]
    J = np.zeros((4, 4), dtype=np.result_type(q, np.float64))
    J[:3, :] = H[keep, :]
    J[3, :] = 2.0 * q
    return J


# ----------------------------------------------------------------------------
# batched Levenberg-Marquardt root enumeration
# ----------------------------------------------------------------------------
def _lm_batch(resid_one, jac_one, z0, data, maxit=50, tol=1e-13, mu0=1e-4):
    """Vectorized LM over a batch of initial guesses.

    resid_one(z, data) -> (residual vector), jac_one(z, data) -> matrix.
    z0: (B, n). Returns z (B,n) and convergence flags.
    """
    B, n = z0.shape
    z = np.array(z0, dtype=float)
    mu = np.full(B, mu0)
    active = np.ones(B, dtype=bool)
    r = np.empty((B, n))
    for b in range(B):
        r[b] = resid_one(z[b], data)
    nr = np.linalg.norm(r, axis=1)

    for it_ in range(maxit):
        done = nr < tol
        active &= ~done
        if not active.any():
            break
        idx = np.where(active)[0]
        for b in idx:
            J = jac_one(z[b], data)
            rb = r[b]
            A = J.T @ J
            g = J.T @ rb
            # LM step with adaptive mu
            for _trial in range(6):
                try:
                    dz = np.linalg.solve(A + mu[b] * np.diag(np.diag(A) + 1e-12), -g)
                except np.linalg.LinAlgError:
                    dz = np.linalg.lstsq(A + mu[b] * np.eye(n), -g, rcond=None)[0]
                znew = z[b] + dz
                rnew = resid_one(znew, data)
                if np.linalg.norm(rnew) < np.linalg.norm(rb):
                    z[b] = znew
                    r[b] = rnew
                    mu[b] = max(mu[b] * 0.3, 1e-12)
                    break
                mu[b] = min(mu[b] * 10.0, 1e6)
        nr = np.linalg.norm(r, axis=1)
        # 停滞的起点直接判失败，避免空转
        if it_ > 20:
            active &= nr > 1e-6

    return z, nr


def canonical(q, tol=1e-9):
    """Canonical representative of +-q."""
    q = np.asarray(q, dtype=float)
    for c in q:
        if abs(c) > tol:
            return q if c > 0 else -q
    return q


def dedup(points, tol=1e-6):
    """Deduplicate points up to sign, preserving order."""
    out = []
    for p in points:
        pc = canonical(p)
        if not any(np.linalg.norm(pc - canonical(o)) < tol for o in out):
            out.append(pc)
    return np.array(out) if out else np.zeros((0, points[0].shape[0]))


def enumerate_full(data, n_starts=400, seed=0, lam_scale=1.0):
    rng = np.random.default_rng(seed)
    q0 = rng.normal(size=(n_starts, 4))
    q0 /= np.linalg.norm(q0, axis=1, keepdims=True)
    z0 = np.hstack([q0, rng.normal(scale=lam_scale, size=(n_starts, 1))])
    z, nr = _lm_batch(resid_full, jac_full, z0, data)
    roots = []
    for b in range(n_starts):
        if nr[b] < 1e-9:
            roots.append(z[b, :4] / np.linalg.norm(z[b, :4]))
    if not roots:
        return np.zeros((0, 4)), np.zeros(0)
    roots = dedup(roots)
    losses = np.array([loss_grad_hess(q, data)[0] for q in roots])
    # refine each root with a couple of Newton steps on the KKT system
    return roots, losses


def enumerate_branch(data, k, n_starts=400, seed=1234):
    rng = np.random.default_rng(seed + 10 * k)
    q0 = rng.normal(size=(n_starts, 4))
    q0 /= np.linalg.norm(q0, axis=1, keepdims=True)
    z, nr = _lm_batch(lambda q, d: resid_branch(q, d, k),
                      lambda q, d: jac_branch(q, d, k), q0, data)
    roots = []
    for b in range(n_starts):
        if nr[b] < 1e-9:
            roots.append(z[b] / np.linalg.norm(z[b]))
    if not roots:
        return np.zeros((0, 4))
    return dedup(roots)


# ----------------------------------------------------------------------------
# implicit differentiation
# ----------------------------------------------------------------------------
def implicit_full(q, lam, data, theta_grad_fn):
    """dq/dtheta and dlambda/dtheta for the full KKT system.

    theta_grad_fn: function(data) -> dF/dtheta (5,) at fixed (q, lam),
    computed by complex step on the residual.
    """
    z = np.concatenate([q, [lam]])
    M = jac_full(z, data)
    N = theta_grad_fn(z, data)
    dz = -np.linalg.solve(M, N)
    return dz[:4], dz[4], M


def implicit_branch(q, data, k, theta_grad_fn):
    J = jac_branch(q, data, k)
    N = theta_grad_fn(q, data, k)
    dq = -np.linalg.solve(J, N)
    return dq, J
