"""Problem (P6) of Holmes et al. (T-RO 2025): matrix-weighted stereo pose.

This module reproduces the QCQP/SDP construction used by the clean-room
reference implementation in ``repro/cleanroom/sdp_pose.py`` so that the GPU
layer can be checked against exactly the same problem data.

    min_{C in SO(3), t}  sum_k || y_k - C m_k + t ||^2_{W_k}      (P6)

with ``x = [h, vec_col(C), t]`` and the Shor relaxation

    min_X <Q, X>  s.t.  <A_i, X> = 0,  <A_0, X> = 1,  X >= 0.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "N_VARS", "constraint_matrices", "homogenising_matrix",
    "build_objective_matrix", "objective_scale", "scale_objective",
    "build_p6_sdp",
]

N_VARS = 13
HOMOG_IDX = 0
C_IDX = slice(1, 10)
T_IDX = slice(10, 13)


def _add_block(A, p, q, val):
    A[p, q] += val
    if p != q:
        A[q, p] += val


def constraint_matrices():
    """The 24 symmetric matrices ``A_i`` with ``x^T A_i x = 0``.

    Ordering: 6 orthogonality (i <= j), 9 handedness, 9 row/column norm
    equalities.  The last 18 are the redundant constraints used to tighten
    the relaxation.
    """
    A_list = []
    for i in range(3):
        for j in range(i, 3):
            A = np.zeros((N_VARS, N_VARS))
            for k in range(3):
                p = 1 + 3 * i + k
                q = 1 + 3 * j + k
                if i == j:
                    A[p, q] += 1.0
                else:
                    _add_block(A, p, q, 0.5)
            if i == j:
                A[0, 0] -= 1.0
            A_list.append(A)

    for i, j, k in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        for l, m, n in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
            A = np.zeros((N_VARS, N_VARS))
            eps = 1.0 if (l, m, n) in ((0, 1, 2), (1, 2, 0), (2, 0, 1)) else -1.0
            p = 1 + 3 * j + m
            q = 1 + 3 * k + n
            _add_block(A, p, q, 0.5 * eps)
            p2 = 1 + 3 * j + n
            q2 = 1 + 3 * k + m
            _add_block(A, p2, q2, -0.5 * eps)
            A[1 + 3 * i + l, 0] += -0.5
            A[0, 1 + 3 * i + l] += -0.5
            A_list.append(A)

    for i in range(3):
        for j in range(3):
            A = np.zeros((N_VARS, N_VARS))
            for k in range(3):
                A[1 + 3 * j + k, 1 + 3 * j + k] += 1.0
                A[1 + i + 3 * k, 1 + i + 3 * k] -= 1.0
            A_list.append(A)
    return A_list


def homogenising_matrix():
    A0 = np.zeros((N_VARS, N_VARS))
    A0[0, 0] = 1.0
    return A0


def build_objective_matrix(m, y, W):
    """Assemble ``Q(theta)`` of (P6); see the clean-room module for details."""
    m = np.asarray(m)
    y = np.asarray(y)
    W = np.asarray(W)
    if m.shape[0] != 3:
        m = m.T
    if y.shape[0] != 3:
        y = y.T
    mT, yT = m.T, y.T
    dtype = np.result_type(m, y, W, float)
    Q = np.zeros((N_VARS, N_VARS), dtype=dtype)
    Wy = np.einsum("nab,nb->na", W, yT)
    Q[0, 0] = np.einsum("na,na->", Wy, yT)
    Q[C_IDX, C_IDX] = np.einsum("ni,nj,nab->iajb", mT, mT, W).reshape(9, 9)
    Q[T_IDX, T_IDX] = W.sum(axis=0)
    Qch = -np.einsum("ni,na->ia", mT, Wy).reshape(9)
    Q[C_IDX, 0] = Qch
    Q[0, C_IDX] = Qch
    Qth = Wy.sum(axis=0)
    Q[T_IDX, 0] = Qth
    Q[0, T_IDX] = Qth
    Qct = -np.einsum("ni,nab->iab", mT, W).reshape(9, 3)
    Q[C_IDX, T_IDX] = Qct
    Q[T_IDX, C_IDX] = Qct.T
    return 0.5 * (Q + Q.T)


def objective_scale(Q):
    Qr = np.array(Q, dtype=float)
    Qr[0, 0] = 0.0
    return 1.0 / np.linalg.norm(Qr, ord="fro")


def scale_objective(Q, scale):
    Qs = np.array(Q, dtype=float)
    Qs[0, 0] = 0.0
    Qs *= scale
    return 0.5 * (Qs + Qs.T)


def independent_constraints(A_all, tol=1e-9):
    """Greedy selection of a linearly independent subset of the matrices."""
    V = A_all.reshape(len(A_all), -1)
    basis, keep = [], []
    for i in range(len(A_all)):
        v = V[i].copy()
        for u in basis:
            v -= np.dot(v, u) * u
        if np.linalg.norm(v) > tol * np.linalg.norm(V[i]):
            basis.append(v / np.linalg.norm(v))
            keep.append(i)
    return np.array(keep)


def build_p6_sdp():
    """Return the fixed P6 SDP data and a strictly feasible dual start.

    Returns
    -------
    dict with ``A_all`` (25,13,13), ``keep`` (18 indices), ``A`` (18,13,13),
    ``b`` (18,), ``c0`` (18,) such that ``Q + s * sum_i c0_i A_i > 0`` for a
    sufficiently large ``s`` (the barrier code finds ``s``).
    """
    A_all = np.stack(constraint_matrices() + [homogenising_matrix()])
    keep = independent_constraints(A_all)
    A = A_all[keep]
    b = np.zeros(len(A))
    b[-1] = 1.0
    c0 = np.zeros(len(A))
    for pos, idx in enumerate(keep):
        if idx in (0, 3, 5):       # diagonal orthogonality constraints
            c0[pos] = 1.0
        if idx == 24:              # homogenising constraint
            c0[pos] = 4.0
    return dict(A_all=A_all, keep=keep, A=A, b=b, c0=c0)
