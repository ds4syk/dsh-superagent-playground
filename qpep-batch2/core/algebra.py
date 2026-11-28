"""QPEP 代数内核：单位四元数、Veronese 映射、T 矩阵与 21 条 SO(3) 约束。

约定（与第二批 idea 文档一致）
--------------------------------
q = [q0, q1, q2, q3] ∈ S^3，
m2(q) = [q0^2, q1^2, q2^2, q3^2, q0q1, q0q2, q0q3, q1q2, q1q3, q2q3] ∈ R^10。
x(q) = [vec_row(R(q)); ||q||^2] = T m2(q)，T ∈ R^{10x10}，det(T) = -8192。

注意：本文件使用文档的**非正交** m2 基（系数全为 1），
因此 m2(q)^T m2(q) = ||q||^4，且 ||q||^4 = (u^T m2)^2，u = T[9] = [1,1,1,1,0,...]。
"""
from __future__ import annotations

import numpy as np

# x = [R11,R12,R13,R21,R22,R23,R31,R32,R33, h] = T @ m2(q)
# 由 sympy 精确推导，det(T) = -8192。
T = np.array([
    [1, 1, -1, -1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, -2, 2, 0, 0],
    [0, 0, 0, 0, 0, 2, 0, 0, 2, 0],
    [0, 0, 0, 0, 0, 0, 2, 2, 0, 0],
    [1, -1, 1, -1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, -2, 0, 0, 0, 0, 2],
    [0, 0, 0, 0, 0, -2, 0, 0, 2, 0],
    [0, 0, 0, 0, 2, 0, 0, 0, 0, 2],
    [1, -1, -1, 1, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
], dtype=float)
T_INV = np.linalg.inv(T)
U_NORM = T[9].copy()              # m2^T U_NORM^T U_NORM m2 = ||q||^4


def m2(q):
    """Veronese 映射。q: (...,4) -> (...,10)。"""
    q = np.asarray(q, dtype=float)
    q0, q1, q2, q3 = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack([q0 * q0, q1 * q1, q2 * q2, q3 * q3,
                     q0 * q1, q0 * q2, q0 * q3,
                     q1 * q2, q1 * q3, q2 * q3], axis=-1)


def Jm(q):
    """d m2 / d q。q: (...,4) -> (...,10,4)。"""
    q = np.asarray(q, dtype=float)
    q0, q1, q2, q3 = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    z = np.zeros_like(q0)
    rows = [
        (2 * q0, z, z, z),
        (z, 2 * q1, z, z),
        (z, z, 2 * q2, z),
        (z, z, z, 2 * q3),
        (q1, q0, z, z),
        (q2, z, q0, z),
        (q3, z, z, q0),
        (z, q2, q1, z),
        (z, q3, z, q1),
        (z, z, q3, q2),
    ]
    return np.stack([np.stack(r, axis=-1) for r in rows], axis=-2)


def _B_mats():
    """m2_a(q) = 0.5 q^T B_a q 的 10 个对称矩阵 B_a。"""
    B = np.zeros((10, 4, 4))
    for i in range(4):
        B[i, i, i] = 2.0
    for a, (i, j) in enumerate([(0, 1), (0, 2), (0, 3),
                                (1, 2), (1, 3), (2, 3)], start=4):
        B[a, i, j] = B[a, j, i] = 1.0
    return B


B_MATS = _B_mats()


def E(q):
    """S^3 切空间标准正交基，列 = q⊗i, q⊗j, q⊗k。q: (...,4)->(...,4,3)。"""
    q = np.asarray(q, dtype=float)
    q0, q1, q2, q3 = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    r0 = np.stack([-q1, -q2, -q3], axis=-1)
    r1 = np.stack([q0, -q3, q2], axis=-1)
    r2 = np.stack([q3, q0, -q1], axis=-1)
    r3 = np.stack([-q2, q1, q0], axis=-1)
    return np.stack([r0, r1, r2, r3], axis=-2)


def R_of_q(q):
    """单位四元数 -> 旋转矩阵（scalar-first）。q: (...,4) -> (...,3,3)。"""
    q = np.asarray(q, dtype=float)
    q0, q1, q2, q3 = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = np.empty(q.shape[:-1] + (3, 3))
    R[..., 0, 0] = q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3
    R[..., 0, 1] = 2 * (q1 * q2 - q0 * q3)
    R[..., 0, 2] = 2 * (q1 * q3 + q0 * q2)
    R[..., 1, 0] = 2 * (q1 * q2 + q0 * q3)
    R[..., 1, 1] = q0 * q0 - q1 * q1 + q2 * q2 - q3 * q3
    R[..., 1, 2] = 2 * (q2 * q3 - q0 * q1)
    R[..., 2, 0] = 2 * (q1 * q3 - q0 * q2)
    R[..., 2, 1] = 2 * (q2 * q3 + q0 * q1)
    R[..., 2, 2] = q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3
    return R


def x_of_q(q):
    """x(q) = [vec_row(R); ||q||^2]。q: (4,) -> (10,)。"""
    R = R_of_q(q)
    return np.concatenate([R.reshape(-1), [q @ q]])


def fval(q, Q):
    v = m2(q)
    return float(v @ Q @ v)


def grad(q, Q):
    """∇f(q) = 2 J^T Q m2。"""
    return 2.0 * Jm(q).T @ (Q @ m2(q))


def hess(q, Q):
    """∇²f(q) = 2 J^T Q J + 2 Σ_a (Q m2)_a B_a。"""
    J = Jm(q)
    v = m2(q)
    Qv = Q @ v
    return 2.0 * J.T @ Q @ J + 2.0 * np.einsum('a,aij->ij', Qv, B_MATS)


def riemann_hess(q, Q):
    """H_{S^3}(q) = E^T (∇²f - λ I) E，λ = q^T∇f = 4f。"""
    g = grad(q, Q)
    lam = float(q @ g)
    return E(q).T @ (hess(q, Q) - lam * np.eye(4)) @ E(q)


# ---------------------------------------------------------------- constraints

def _constraint_polys(x):
    """21 条 SO(3) 约束在 x ∈ R^10 上的二次型取值（标量列表）。"""
    R = x[:9].reshape(3, 3)
    h = x[9]
    out = []
    # 左正交 R^T R = h^2 I（6 条）
    for a in range(3):
        for b in range(a, 3):
            out.append(float(R[:, a] @ R[:, b]) - h * h * (1.0 if a == b else 0.0))
    # 右正交 R R^T = h^2 I（6 条）
    for a in range(3):
        for b in range(a, 3):
            out.append(float(R[a, :] @ R[b, :]) - h * h * (1.0 if a == b else 0.0))
    # 右手性 c_i × c_j = h c_k（3 个向量方程 -> 9 个标量）
    for (i, j, k) in [(0, 1, 2), (1, 2, 0), (2, 0, 1)]:
        ci, cj, ck = R[:, i], R[:, j], R[:, k]
        cross = np.cross(ci, cj)
        for l in range(3):
            out.append(float(cross[l] - h * ck[l]))
    return np.array(out)


def constraint_matrices():
    """返回 21 个对称 A_i，使 x^T A_i x 等于第 i 条约束。shape (21,10,10)。"""
    As = np.zeros((21, 10, 10))
    e = np.eye(10)
    base = np.array([_constraint_polys(e[i]) for i in range(10)])  # (10,21)
    for idx in range(21):
        A = np.zeros((10, 10))
        A[np.arange(10), np.arange(10)] = base[:, idx]
        for i in range(10):
            for j in range(i + 1, 10):
                v = _constraint_polys(e[i] + e[j])[idx]
                A[i, j] = A[j, i] = 0.5 * (v - base[i, idx] - base[j, idx])
        As[idx] = A
    return As


A_CONSTRAINTS = constraint_matrices()

# T^T A_i T：在 m2 基下的约束矩阵（21 个，张成 ker(P)，维数 20）
H_CONSTRAINTS = np.einsum('ji,kjl,lm->kim', T, A_CONSTRAINTS, T)
