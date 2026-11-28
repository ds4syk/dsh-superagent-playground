#!/usr/bin/env python
"""Core algebra for the Laplace-QPEPLayer synthetic experiment.

All conventions follow ``theory/proofs.md`` / ``theory/derivations.md``:
  q = [q0,q1,q2,q3], ||q||=1,
  v2(q) = [q0^2,q1^2,q2^2,q3^2, sqrt2 q0q1, sqrt2 q0q2, sqrt2 q0q3,
           sqrt2 q1q2, sqrt2 q1q3, sqrt2 q2q3],
  J(q) = v2(q)^T M v2(q),
  E(q) = [q (x) i, q (x) j, q (x) k]  (columns, tangent basis, |dq|=|dxi|),
  H(q) = E^T (d^2 J - 4 J I) E,
  dq*/dtheta = -E H^{-1} E^T (2 Jv^T (dM/dtheta) v2).

Everything is written so that *the same* functions can be used in single mode
(shape (4,)) or batched (shape (S,4)) form; batch dimensions are leading.
"""
from __future__ import annotations

import numpy as np

SQ = np.sqrt(2.0)

# ---------------------------------------------------------------- monomials

def v2(q):
    """Veronese map. q: (...,4) -> (...,10)."""
    q = np.asarray(q, dtype=float)
    q0, q1, q2, q3 = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack(
        [q0 * q0, q1 * q1, q2 * q2, q3 * q3,
         SQ * q0 * q1, SQ * q0 * q2, SQ * q0 * q3,
         SQ * q1 * q2, SQ * q1 * q3, SQ * q2 * q3], axis=-1)


def Jv(q):
    """Jacobian d v2 / d q. q: (...,4) -> (...,10,4)."""
    q = np.asarray(q, dtype=float)
    q0, q1, q2, q3 = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    z = np.zeros_like(q0)
    rows = [
        (2 * q0, z, z, z),
        (z, 2 * q1, z, z),
        (z, z, 2 * q2, z),
        (z, z, z, 2 * q3),
        (SQ * q1, SQ * q0, z, z),
        (SQ * q2, z, SQ * q0, z),
        (SQ * q3, z, z, SQ * q0),
        (z, SQ * q2, SQ * q1, z),
        (z, SQ * q3, z, SQ * q1),
        (z, z, SQ * q3, SQ * q2),
    ]
    return np.stack([np.stack(r, axis=-1) for r in rows], axis=-2)


def _Hv():
    """Second derivatives of v2: Hv[a] = d^2 v2_a / dq dq, shape (10,4,4)."""
    H = np.zeros((10, 4, 4))
    for i in range(4):
        H[i, i, i] = 2.0
    for i, j, a in [(0, 1, 4), (0, 2, 5), (0, 3, 6),
                    (1, 2, 7), (1, 3, 8), (2, 3, 9)]:
        H[a, i, j] = H[a, j, i] = SQ
    return H


HV = _Hv()


def E(q):
    """Tangent basis, q: (...,4) -> (...,4,3).  Columns are q (x) i,j,k."""
    q = np.asarray(q, dtype=float)
    q0, q1, q2, q3 = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    r0 = np.stack([-q1, -q2, -q3], axis=-1)
    r1 = np.stack([q0, -q3, q2], axis=-1)
    r2 = np.stack([q3, q0, -q1], axis=-1)
    r3 = np.stack([-q2, q1, q0], axis=-1)
    return np.stack([r0, r1, r2, r3], axis=-2)


def dE(q):
    """d E / d q_b, b=0..3.  q: (...,4) -> (...,4,4,3) indexed [.., b, i, c]."""
    q = np.asarray(q, dtype=float)
    batch = q.shape[:-1]
    # E rows are  r0=[-q1,-q2,-q3], r1=[q0,-q3,q2], r2=[q3,q0,-q1], r3=[-q2,q1,q0].
    # The partials below were fixed on 2026-11-27 after a finite-difference audit:
    # the previous d1,d2,d3 had wrong signs / entries at rows 1-2, which silently
    # corrupted the only consumer, dHsphere_dq -> dlogdetH_dtheta (the closed-form
    # third-derivative path).  dlogdet_exact / run_main's numeric fallback used
    # E(dq) instead and were unaffected.
    d0 = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    d1 = np.array([[-1, 0, 0], [0, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float)
    d2 = np.array([[0, -1, 0], [0, 0, 1], [0, 0, 0], [-1, 0, 0]], dtype=float)
    d3 = np.array([[0, 0, -1], [0, -1, 0], [1, 0, 0], [0, 0, 0]], dtype=float)
    out = [np.broadcast_to(d, batch + (4, 3)) for d in (d0, d1, d2, d3)]
    return np.stack(out, axis=-3)


# ---------------------------------------------------------------- cost / derivatives

def Jval(q, M):
    v = v2(q)
    return np.einsum('...i,...i->...', v, v @ M.T)


def grad_J(q, M):
    """Euclidean gradient dJ/dq: (...,4)."""
    Mv = v2(q) @ M.T
    return 2.0 * np.einsum('...ai,...a->...i', Jv(q), Mv)


def hess_J(q, M):
    """Euclidean Hessian d^2 J / dq dq: (...,4,4)."""
    Jq = Jv(q)
    Mv = v2(q) @ M.T
    MJ = np.matmul(M, Jq)
    H = 2.0 * np.einsum('...ai,...aj->...ij', Jq, MJ)
    return H + 2.0 * np.einsum('...a,aij->...ij', Mv, HV)


def H_sphere(q, M):
    """Riemannian Hessian in the E basis: (...,3,3)."""
    Eq = E(q)
    Jq = Jval(q, M)
    N = hess_J(q, M) - 4.0 * Jq[..., None, None] * np.eye(4)
    return np.einsum('...ia,...ij,...jc->...ac', Eq, N, Eq)


def proj_grad(q, M):
    """E^T grad J (Riemannian gradient in E basis): (...,3)."""
    return np.einsum('...ia,...i->...a', E(q), grad_J(q, M))


def dJ_dtheta(q, M, dM):
    """partial_theta J at fixed q (envelope gives dJ_k/dtheta)."""
    v = v2(q)
    return np.einsum('...i,...i->...', v, v @ dM.T)


def dgradJ_dtheta(q, M, dM):
    """partial_theta grad J at fixed q: (...,4)."""
    dMv = v2(q) @ dM.T
    return 2.0 * np.einsum('...ai,...a->...i', Jv(q), dMv)


def dhessJ_dtheta(q, M, dM):
    """partial_theta (d^2 J) at fixed q: (...,4,4)."""
    Jq = Jv(q)
    dMv = v2(q) @ dM.T
    dMJ = np.matmul(dM, Jq)
    H = 2.0 * np.einsum('...ai,...aj->...ij', Jq, dMJ)
    return H + 2.0 * np.einsum('...a,aij->...ij', dMv, HV)


def dHsphere_dtheta_fixedq(q, M, dM):
    """partial_theta H at fixed q: (...,3,3)."""
    Eq = E(q)
    dJ = dJ_dtheta(q, M, dM)
    dN = dhessJ_dtheta(q, M, dM)
    dM4 = dN - 4.0 * dJ[..., None, None] * np.eye(4)
    return np.einsum('...ia,...ij,...jc->...ac', Eq, dM4, Eq)


def dhessJ_dq(q, M):
    """partial_q (d^2 J): (...,4,4,4) indexed [.., b, i, j] = d/dq_b Hess_J[i,j]."""
    Jq = Jv(q)                                    # (...,10,4) [a,i]
    MJ = np.matmul(M, Jq)                         # (...,10,4) [a,b]
    MHV = np.matmul(M, HV.reshape(10, 16)).reshape(10, 4, 4)   # [a,b,j]
    term1 = np.einsum('abi,...aj->...bij', HV, MJ)
    term2 = np.einsum('...ai,abj->...bij', Jq, MHV)
    term3 = np.einsum('...ab,aij->...bij', MJ, HV)
    return 2.0 * (term1 + term2) + 2.0 * term3


def dHsphere_dq(q, M):
    """partial_q H: (...,4,3,3) indexed [.., b, a, c]."""
    Eq = E(q)
    dEq = dE(q)                     # (...,4,4,3) [b,i,c]
    Jq = Jval(q, M)
    N = hess_J(q, M) - 4.0 * Jq[..., None, None] * np.eye(4)
    g = grad_J(q, M)
    dN = dhessJ_dq(q, M)            # (...,4,4,4) [b,i,j]
    dN4 = dN - 4.0 * g[..., :, None, None] * np.eye(4)
    t1 = np.einsum('...bia,...ij,...jc->...bac', dEq, N, Eq)
    t2 = np.einsum('...ia,...bij,...jc->...bac', Eq, dN4, Eq)
    t3 = np.einsum('...ia,...ij,...bjc->...bac', Eq, N, dEq)
    return t1 + t2 + t3


# ---------------------------------------------------------------- rotations

def Rmat(q):
    """Rotation matrix, row-major (...,9) output? -> (...,3,3)."""
    q = np.asarray(q, dtype=float)
    q0, q1, q2, q3 = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    c0 = np.stack([q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3,
                   2 * (q1 * q2 - q0 * q3),
                   2 * (q1 * q3 + q0 * q2)], axis=-1)
    c1 = np.stack([2 * (q1 * q2 + q0 * q3),
                   q0 * q0 - q1 * q1 + q2 * q2 - q3 * q3,
                   2 * (q2 * q3 - q0 * q1)], axis=-1)
    c2 = np.stack([2 * (q1 * q3 - q0 * q2),
                   2 * (q2 * q3 + q0 * q1),
                   q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3], axis=-1)
    return np.stack([c0, c1, c2], axis=-2)


def dRvec_dq(q):
    """Jacobian of vec(R) (row-major, length 9) w.r.t. q: (...,9,4)."""
    q = np.asarray(q, dtype=float)
    q0, q1, q2, q3 = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    batch = q.shape[:-1]

    def mat(rows):
        m = np.stack([np.stack([np.zeros(batch) + x for x in row], axis=-1)
                      for row in rows], axis=-2)
        return m.reshape(batch + (9,))

    d = {
        0: [[2 * q0, -2 * q3, 2 * q2],
            [2 * q3, 2 * q0, -2 * q1],
            [-2 * q2, 2 * q1, 2 * q0]],
        1: [[2 * q1, 2 * q2, 2 * q3],
            [2 * q2, -2 * q1, -2 * q0],
            [2 * q3, 2 * q0, -2 * q1]],
        2: [[-2 * q2, 2 * q1, 2 * q0],
            [2 * q1, 2 * q2, 2 * q3],
            [-2 * q0, 2 * q3, -2 * q2]],
        3: [[-2 * q3, -2 * q0, 2 * q1],
            [2 * q0, -2 * q3, 2 * q2],
            [2 * q1, 2 * q2, 2 * q3]],
    }
    return np.stack([mat(d[b]) for b in range(4)], axis=-1)


def rot_angle(Ra, Rb):
    """Geodesic angle (radians) between rotation matrices (broadcast)."""
    tr = np.einsum('...ij,...ij->...', Ra, Rb)
    c = np.clip((tr - 1.0) / 2.0, -1.0, 1.0)
    return np.arccos(c)


def project_SO3(A):
    """Polar projection of a 3x3 (or batched) matrix onto SO(3)."""
    A = np.asarray(A, dtype=float)
    U, s, Vt = np.linalg.svd(A)
    d = np.sign(np.linalg.det(np.einsum('...ij,...jk->...ik', U, Vt)))
    D = np.zeros_like(A)
    D[..., 0, 0] = 1.0
    D[..., 1, 1] = 1.0
    D[..., 2, 2] = d
    return np.einsum('...ij,...jk,...kl->...il', U, D, Vt)


# ---------------------------------------------------------------- IFT / Laplace pieces

def ift_dq(q, M, dM):
    """dq*/dtheta at a stationary point (Theorem 1.1), shape (...,4)."""
    Eq = E(q)
    H = H_sphere(q, M)
    dMv = v2(q) @ dM.T
    inner = np.einsum('...ai,...a->...i', Jv(q), dMv)      # Jv^T dM v2, shape (...,4)
    rhs = 2.0 * np.einsum('...ai,...a->...i', Eq, inner)   # 2 E^T Jv^T dM v2
    xi = np.linalg.solve(H, rhs)
    return -np.einsum('...ia,...a->...i', Eq, xi)


def dlogdetH_dtheta(q, M, dM, dq=None):
    """d/dtheta log det H_k, total derivative (Theorem 1.3)."""
    H = H_sphere(q, M)
    dH = dHsphere_dtheta_fixedq(q, M, dM)
    if dq is None:
        dq = ift_dq(q, M, dM)
    dH = dH + np.einsum('...b,...bac->...ac', dq, dHsphere_dq(q, M))
    return np.einsum('...ij,...ji->...', np.linalg.inv(H), dH), H, dH


def logdetH(q, M):
    return np.linalg.slogdet(H_sphere(q, M))[1]


# ---------------------------------------------------------------- pose vector

def pose_vector(q, GinvD):
    """phi(q) = [vec R(q); t*(q)] in R^12, t* = -G^{-1} D v2(q) = GinvD @ v2(q)."""
    R = Rmat(q).reshape(q.shape[:-1] + (9,))
    t = np.einsum('ij,...j->...i', GinvD, v2(q))
    return np.concatenate([R, t], axis=-1)


def dpose_dq(q, GinvD):
    """d phi / d q: (...,12,4)."""
    dR = dRvec_dq(q)
    dt = np.einsum('ij,...ja->...ia', GinvD, Jv(q))
    return np.concatenate([dR, dt], axis=-2)
