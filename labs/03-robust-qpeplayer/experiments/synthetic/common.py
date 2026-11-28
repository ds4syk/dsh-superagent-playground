#!/usr/bin/env python3
"""Shared library for the synthetic 3D-3D robust-registration experiments.

This module implements, in one place:

* instance generation (clean / random / coherent / boundary outliers),
* the four forward methods used in the main table
  (Kabsch, fixed-budget RANSAC+Kabsch, adaptive RANSAC+Kabsch,
   IRLS-GM, GNC-GM),
* pose metrics,
* the exact 3D-3D robust cost and its Riemannian derivatives
  (e_i, g_i^E, H_i^E, lambda_i, H_rob, dPhi/dy_i), used by the
  gradient and end-to-end experiments.

The legacy v0 script is ``run_registration.py``; it is kept untouched so
that the original numbers remain reproducible.  The methods here are
numerically identical for the identical RNG stream (see
``verify_repro.py`` for the 10-trial reproduction check).

Conventions
-----------
* ``x`` are model points (N,3), ``y`` are observations (N,3).
* ``e_i = ||R(q) x_i + t - y_i||^2`` (squared residual, W_i = I by
  default), Geman-McClure kernel ``rho(e) = c^2 e / (c^2 + e)``.
* RANSAC threshold ``c`` is on the *distance* ``||r_i||``; the GM scale
  is the same ``c`` on the *squared* residual, i.e. distance ``c`` is the
  kernel's half-weight point.  This is consistent and documented.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from math import ceil, log
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Problem constants (same as run_registration.py unless stated)
# ---------------------------------------------------------------------------
N_DEFAULT = 100
SIGMA = 0.01
C = 0.05
C2 = C * C
C4 = C2 * C2
RANSAC_ITERS = 1000
RANSAC_CONFIDENCE = 0.999
RANSAC_THRESH = C
GNC_GAMMA = 1.4
GNC_MAX_MU_ITERS = 40
SUCCESS_ROT_DEG = 5.0
SUCCESS_TRANS = 0.1


# ---------------------------------------------------------------------------
# Basic geometry
# ---------------------------------------------------------------------------
def random_rotation(rng):
    A = rng.normal(size=(3, 3))
    Q, R = np.linalg.qr(A)
    Q = Q @ np.diag(np.sign(np.diag(R)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


def rotation_about_axis(axis, angle):
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0.0, -axis[2], axis[1]],
                  [axis[2], 0.0, -axis[0]],
                  [-axis[1], axis[0], 0.0]])
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def Rmat(q):
    """Quaternion rotation matrix, homogeneous quadratic form (as in proofs)."""
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def quat_from_R(R):
    """Convert rotation matrix to unit quaternion (w,x,y,z) with w >= 0."""
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], float)
    q /= np.linalg.norm(q)
    if q[0] < 0:
        q = -q
    return q


def tangent_basis(q):
    """4x3 orthonormal basis of T_q S^3."""
    _, _, vh = np.linalg.svd(np.asarray(q, float).reshape(1, 4))
    return vh[1:].T


def exp_map(q, t, xi, Bq):
    """Geodesic exponential on S^3 x R^3; xi in R^6 tangent coordinates."""
    xq = Bq @ xi[:3]
    nq = np.linalg.norm(xq)
    if nq < 1e-14:
        q2 = np.array(q, float)
    else:
        q2 = np.cos(nq) * np.asarray(q, float) + np.sin(nq) * xq / nq
    return q2 / np.linalg.norm(q2), np.asarray(t, float) + xi[3:]


def Rhom(q):
    """Homogeneous (degree-2) quaternion rotation map.

    For ||q||=1 this equals Rmat(q); for general q it satisfies
    Rhom(s q) = s^2 Rhom(q), which is what the polynomial/QPEP
    parameterisation and the A_k construction require.  Do NOT use the
    standard (unit-quaternion) formula for polarisation at q=e_i+e_j.
    """
    w, x, y, z = q
    return np.array(
        [
            [w * w + x * x - y * y - z * z, 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), w * w - x * x + y * y - z * z, 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), w * w - x * x - y * y + z * z],
        ]
    )


def build_A(p):
    """Constant symmetric A_k with (Rhom(q) p)_k = q^T A_k q.

    Values at e_i and e_i+e_j determine the quadratic form; use the
    *homogeneous* map for the pair sums (a common implementation bug is
    to use the unit-quaternion formula there, which gives a different
    quadratic form and a residual map that is not R(q)p).
    """
    E = np.eye(4)
    A = np.zeros((3, 4, 4))
    for k in range(3):
        for i in range(4):
            A[k, i, i] = (Rhom(E[i]) @ p)[k]
        for i in range(4):
            for j in range(i + 1, 4):
                val = ((Rhom(E[i] + E[j]) @ p)[k] - (Rhom(E[i]) @ p)[k]
                       - (Rhom(E[j]) @ p)[k]) / 2.0
                A[k, i, j] = A[k, j, i] = val
    return A


# ---------------------------------------------------------------------------
# Weighted Kabsch and robust solvers
# ---------------------------------------------------------------------------
def kabsch(x, y, w):
    w = np.asarray(w, float)
    s = w.sum()
    if not np.isfinite(s) or s <= 1e-12:
        w = np.ones_like(w)
        s = w.sum()
    w = w / s
    cx = (w[:, None] * x).sum(0)
    cy = (w[:, None] * y).sum(0)
    H = (w[:, None] * (x - cx)).T @ (y - cy)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    t = cy - R @ cx
    return R, t


def residuals(x, y, R, t):
    return np.linalg.norm((R @ x.T).T + t - y, axis=1)


def solve_irls(x, y, c=C, iters=40):
    R, t = kabsch(x, y, np.ones(len(x)))
    for _ in range(iters):
        e = residuals(x, y, R, t) ** 2
        w = (c * c / (c * c + e)) ** 2
        R, t = kabsch(x, y, w)
    return R, t


def solve_gnc(x, y, c=C, gamma=GNC_GAMMA, max_mu_iters=GNC_MAX_MU_ITERS,
              return_info=False):
    """GNC-GM with weighted-Kabsch inner solve (3D-3D specialization).

    The schedule follows (4.8) of theory/proofs.md:
    mu0 = max(3 max_i e_i / c^2, 1), mu <- max(1, mu/gamma), plus a few
    exact-kernel IRLS polish steps at mu=1.
    """
    R, t = kabsch(x, y, np.ones(len(x)))
    e = residuals(x, y, R, t) ** 2
    mu = max(3.0 * e.max() / (c * c), 1.0)
    mu_path = [float(mu)]
    it = 0
    for it in range(max_mu_iters):
        w = (mu * c * c / (mu * c * c + e)) ** 2
        R, t = kabsch(x, y, w)
        e = residuals(x, y, R, t) ** 2
        if mu <= 1.0 + 1e-12:
            break
        mu = max(1.0, mu / gamma)
        mu_path.append(float(mu))
    for _ in range(5):
        w = (c * c / (c * c + e)) ** 2
        R, t = kabsch(x, y, w)
        e = residuals(x, y, R, t) ** 2
    if return_info:
        return R, t, dict(n_mu=len(mu_path), mu0=mu_path[0], n_outer=it + 1)
    return R, t


def solve_ransac(x, y, rng, c=RANSAC_THRESH, iters=RANSAC_ITERS,
                 adaptive=False, confidence=RANSAC_CONFIDENCE,
                 return_info=False):
    """3-point RANSAC + refit on the consensus set.

    adaptive=True uses the standard confidence-based stopping rule
    ``k >= log(1-conf)/log(1-w^3)`` with w = current inlier ratio,
    capped at ``iters``.  adaptive=False always spends the full budget
    (this is the v0 baseline).
    """
    n = len(x)
    best_in = None
    best_count = -1
    target = int(iters)
    k = 0
    while k < target:
        idx = rng.choice(n, 3, replace=False)
        R, t = kabsch(x[idx], y[idx], np.ones(3))
        e = residuals(x, y, R, t)
        inl = e < c
        cnt = int(inl.sum())
        if cnt > best_count:
            best_count = cnt
            best_in = inl
            if cnt == n:
                k += 1
                break
            if adaptive:
                w = best_count / n
                p_good = w ** 3
                if 0.0 < p_good < 1.0:
                    need = ceil(log(1.0 - confidence) / log(1.0 - p_good))
                    target = max(1, min(int(iters), need))
        k += 1
    if best_in is None or best_count < 3:
        R, t = kabsch(x, y, np.ones(n))
        info = dict(n_iters=k, best_count=best_count)
        return (R, t, info) if return_info else (R, t)
    R, t = kabsch(x[best_in], y[best_in], np.ones(best_count))
    info = dict(n_iters=k, best_count=best_count)
    return (R, t, info) if return_info else (R, t)


# ---------------------------------------------------------------------------
# Instance generation
# ---------------------------------------------------------------------------
def _clean_instance(rng, n):
    x = rng.uniform(-1, 1, size=(n, 3))
    Rgt = random_rotation(rng)
    tgt = rng.uniform(-0.5, 0.5, size=3)
    y = (Rgt @ x.T).T + tgt + SIGMA * rng.normal(size=(n, 3))
    return x, y, Rgt, tgt


def make_instance(rng, p, n=N_DEFAULT, outlier_type="random", return_meta=False):
    """Generate one instance.

    outlier_type:
      * "random"    : replaced observations uniform in [-2,2]^3 (v0 protocol)
      * "coherent"  : replaced observations generated by one rigid transform
      * "boundary"  : replaced observations = GT + r, with
                      ||r||^2 = e in [c^2/3, 3c^2] (the toxic band);
                      e is uniform on that interval.

    If return_meta, also returns a dict with the outlier indices and (for
    coherent outliers) the competing rigid transform (Rout, tout).
    """
    x, y, Rgt, tgt = _clean_instance(rng, n)
    n_out = int(round(p * n))
    meta = dict(idx=np.zeros(0, dtype=int), Rout=None, tout=None)
    if n_out <= 0:
        return (x, y, Rgt, tgt, meta) if return_meta else (x, y, Rgt, tgt)
    idx = rng.choice(n, n_out, replace=False)
    meta["idx"] = idx
    if outlier_type == "random":
        y[idx] = rng.uniform(-2.0, 2.0, size=(n_out, 3))
    elif outlier_type == "coherent":
        # A single competing rigid transform, separated from the truth by a
        # controlled rotation (30-60 deg) and translation (0.6-1.5 m).
        axis = rng.normal(size=3)
        angle = np.deg2rad(rng.uniform(30.0, 60.0))
        R_delta = rotation_about_axis(axis, angle)
        Rout = R_delta @ Rgt
        t_shift = rng.normal(size=3)
        t_shift = t_shift / np.linalg.norm(t_shift) * rng.uniform(0.6, 1.5)
        tout = R_delta @ tgt + t_shift
        meta["Rout"] = Rout
        meta["tout"] = tout
        y[idx] = (Rout @ x[idx].T).T + tout + SIGMA * rng.normal(size=(n_out, 3))
    elif outlier_type == "boundary":
        dirs = rng.normal(size=(n_out, 3))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        u = rng.uniform(1.0 / 3.0, 3.0, size=(n_out, 1))  # e / c^2
        r = dirs * (C * np.sqrt(u))
        y[idx] = (Rgt @ x[idx].T).T + tgt + r
    else:
        raise ValueError(f"unknown outlier_type {outlier_type!r}")
    return (x, y, Rgt, tgt, meta) if return_meta else (x, y, Rgt, tgt)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def rot_err_deg(R, Rgt):
    c = (np.trace(R.T @ Rgt) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def pose_errors(R, t, Rgt, tgt):
    return rot_err_deg(R, Rgt), float(np.linalg.norm(t - tgt))


def success(re, te):
    return bool(re < SUCCESS_ROT_DEG and te < SUCCESS_TRANS)


# ---------------------------------------------------------------------------
# Exact robust cost and its derivatives on S^3 x R^3 (3D-3D, W_i = I)
# ---------------------------------------------------------------------------
@dataclass
class RobustTerms:
    """Per-observation Euclidean derivatives of e_i at z=(q,t)."""

    e: np.ndarray            # (N,)
    g: np.ndarray            # (N,7) Euclidean gradient of e_i
    H: np.ndarray            # (N,7,7) Euclidean Hessian of e_i
    u: np.ndarray            # (N,3) R(q) p_i
    r: np.ndarray            # (N,3) residual
    J: np.ndarray            # (N,3,4) du/dq
    A: list = field(default_factory=list)


def robust_terms(x, y, q, t, W=None, A_list=None):
    n = len(x)
    if W is None:
        W = [np.eye(3) for _ in range(n)]
    e = np.zeros(n)
    g = np.zeros((n, 7))
    H = np.zeros((n, 7, 7))
    u = np.zeros((n, 3))
    r = np.zeros((n, 3))
    J = np.zeros((n, 3, 4))
    if A_list is None:
        A_list = [None] * n
    for i in range(n):
        A = A_list[i]
        if A is None:
            A = build_A(x[i])
            A_list[i] = A
        ui = np.array([q @ A[k] @ q for k in range(3)])
        Ji = np.array([2.0 * A[k] @ q for k in range(3)])  # 3x4
        ri = ui + t - y[i]
        Wi = W[i]
        ei = float(ri @ Wi @ ri)
        gq = 2.0 * Ji.T @ Wi @ ri
        gt = 2.0 * Wi @ ri
        Hqq = 2.0 * Ji.T @ Wi @ Ji + 4.0 * sum((Wi @ ri)[k] * A[k] for k in range(3))
        Hqt = 2.0 * Ji.T @ Wi
        Htt = 2.0 * Wi
        Hi = np.zeros((7, 7))
        Hi[:4, :4] = Hqq
        Hi[:4, 4:] = Hqt
        Hi[4:, :4] = Hqt.T
        Hi[4:, 4:] = Htt
        e[i] = ei
        g[i, :4] = gq
        g[i, 4:] = gt
        H[i] = Hi
        u[i] = ui
        r[i] = ri
        J[i] = Ji
    return RobustTerms(e=e, g=g, H=H, u=u, r=r, J=J, A=A_list)


def gm_terms(e, c=C):
    c2 = c * c
    c4 = c2 * c2
    rho = c2 * e / (c2 + e)
    drho = c4 / (c2 + e) ** 2
    d2rho = -2.0 * c4 / (c2 + e) ** 3
    return rho, drho, d2rho


def lambda_parallel(e, c=C):
    """lambda_parallel(e) = 2(rho' + 2 e rho'') = 2 c^4 (c^2 - 3e)/(c^2+e)^3."""
    e = np.asarray(e, float)
    c2 = c * c
    c4 = c2 * c2
    return 2.0 * c4 * (c2 - 3.0 * e) / (c2 + e) ** 3


def robust_hessian_exact(terms: RobustTerms, q, Bq, c=C):
    """Exact H_rob (6x6), lambda_rob, Euclidean total gradient.

    H_rob = B^T [sum_i rho' H_i^E + rho'' g_i^E g_i^E^T] B
            - lambda_rob diag(I_3, 0_3),
    lambda_rob = q^T G_q = sum_i rho'(e_i) lambda_i,
    lambda_i   = q^T g_{i,q}^E.
    """
    n = len(terms.e)
    _, drho, d2rho = gm_terms(terms.e, c)
    Bfull = np.zeros((7, 6))
    Bfull[:4, :3] = Bq
    Bfull[4:, 3:] = np.eye(3)
    Htot = np.zeros((7, 7))
    gtot = np.zeros(7)
    for i in range(n):
        Htot += drho[i] * terms.H[i] + d2rho[i] * np.outer(terms.g[i], terms.g[i])
        gtot += drho[i] * terms.g[i]
    lam_rob = float(np.asarray(q) @ gtot[:4])
    D6 = np.diag([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    H = Bfull.T @ Htot @ Bfull - lam_rob * D6
    return H, lam_rob, gtot, Bfull


def robust_hessian_naive(terms: RobustTerms, q, Bq, c=C):
    """Frozen-weight (IRLS-StopGrad) Hessian, with the exact sphere term.

    w_i = rho'(e_i) fixed, H_naive = B^T [sum w_i H_i^E] B
    - (sum w_i lambda_i) diag(I_3, 0_3).
    """
    n = len(terms.e)
    _, drho, _ = gm_terms(terms.e, c)
    Bfull = np.zeros((7, 6))
    Bfull[:4, :3] = Bq
    Bfull[4:, 3:] = np.eye(3)
    Htot = np.zeros((7, 7))
    lam_rob = 0.0
    for i in range(n):
        Htot += drho[i] * terms.H[i]
        lam_rob += drho[i] * float(np.asarray(q) @ terms.g[i, :4])
    D6 = np.diag([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    H = Bfull.T @ Htot @ Bfull - lam_rob * D6
    return H, lam_rob


def dphi_dy_exact(terms: RobustTerms, i, Bfull, c=C):
    """dPhi/dy_i (6x3) for the exact robust fixed-point map (Theorem 3.3)."""
    _, drho, d2rho = gm_terms(terms.e, c)
    Wi = np.eye(3)
    gi = terms.g[i]
    JtW = terms.J[i].T @ Wi            # 4x3
    dg = np.zeros((7, 3))
    dg[:4] = -2.0 * JtW
    dg[4:] = -2.0 * Wi
    de = -2.0 * Wi @ terms.r[i]        # 3
    # d/dy_i [rho'(e_i) g_i]
    M = drho[i] * dg + d2rho[i] * np.outer(gi, de)
    return Bfull.T @ M


def dphi_dy_naive(terms: RobustTerms, i, Bfull, c=C):
    """dPhi_naive/dy_i (6x3): freeze w_i = rho'(e_i)."""
    _, drho, _ = gm_terms(terms.e, c)
    Wi = np.eye(3)
    JtW = terms.J[i].T @ Wi
    dg = np.zeros((7, 3))
    dg[:4] = -2.0 * JtW
    dg[4:] = -2.0 * Wi
    return Bfull.T @ (drho[i] * dg)


def dL_dz(q, t, Rgt, tgt, trans_weight=1.0):
    """Gradient of L = ||R(q)-Rgt||_F^2 + w ||t-tgt||^2 in tangent coords.

    Returns dL/dz (6,), with dL/dq projected on T_q S^3.
    """
    R = Rmat(q)
    dR = 2.0 * (R - Rgt)
    # dR/dq_k for q=(w,x,y,z), see derivation in the script header.
    dR_dq = np.zeros((3, 3, 4))
    w, x, y, z = q
    dR_dq[:, :, 0] = np.array([[0, -2 * z, 2 * y],
                               [2 * z, 0, -2 * x],
                               [-2 * y, 2 * x, 0]])
    dR_dq[:, :, 1] = np.array([[0, 2 * y, 2 * z],
                               [2 * y, -4 * x, -2 * w],
                               [2 * z, 2 * w, -4 * x]])
    dR_dq[:, :, 2] = np.array([[-4 * y, 2 * x, 2 * w],
                               [2 * x, 0, 2 * z],
                               [-2 * w, 2 * z, -4 * y]])
    dR_dq[:, :, 3] = np.array([[-4 * z, -2 * w, 2 * x],
                               [2 * w, -4 * z, 2 * y],
                               [2 * x, 2 * y, 0]])
    dq = np.einsum("ab,abk->k", dR, dR_dq)
    Bq = tangent_basis(q)
    return np.concatenate([Bq.T @ dq, 2.0 * trans_weight * (t - tgt)])


def newton_polish(x, y, q, t, c=C, max_iters=30, tol=1e-13, A_list=None):
    """Newton on Phi(z)=0 in tangent coordinates, using the exact H_rob.

    Returns q,t, H (6x6), Phi_norm, n_iters.  This produces the smooth
    fixed-point branch used as the finite-difference oracle.
    """
    q = np.array(q, float) / np.linalg.norm(q)
    t = np.array(t, float)
    for it in range(max_iters):
        Bq = tangent_basis(q)
        terms = robust_terms(x, y, q, t, A_list=A_list)
        H, lam_rob, gtot, Bfull = robust_hessian_exact(terms, q, Bq, c)
        phi = Bfull.T @ gtot
        pn = float(np.linalg.norm(phi))
        if pn < tol:
            return q, t, H, pn, it
        xi = -np.linalg.solve(H, phi)
        q, t = exp_map(q, t, xi, Bq)
    Bq = tangent_basis(q)
    terms = robust_terms(x, y, q, t, A_list=A_list)
    H, lam_rob, gtot, Bfull = robust_hessian_exact(terms, q, Bq, c)
    phi = Bfull.T @ gtot
    return q, t, H, float(np.linalg.norm(phi)), max_iters


def fd_jacobian_y(x, y, i, q, t, Bq_ref, h=1e-5, c=C, A_list=None):
    """Central-difference dz*/dy_i (6x3) on the polished fixed-point branch."""
    n = len(y)
    Jfd = np.zeros((6, 3))
    for k in range(3):
        yp = y.copy()
        ym = y.copy()
        yp[i, k] += h
        ym[i, k] -= h
        qp, tp, _, pnp, _ = newton_polish(x, yp, q, t, c, A_list=A_list)
        qm, tm, _, pnm, _ = newton_polish(x, ym, q, t, c, A_list=A_list)
        if pnp > 1e-9 or pnm > 1e-9:
            raise RuntimeError(f"FD fixed-point solve did not converge: {pnp:.2e},{pnm:.2e}")
        xip = np.concatenate([Bq_ref.T @ (qp - q), tp - t])
        xim = np.concatenate([Bq_ref.T @ (qm - q), tm - t])
        Jfd[:, k] = (xip - xim) / (2.0 * h)
    return Jfd


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------
def quantile(a, q):
    return float(np.quantile(np.asarray(a, float), q))


def time_call(fn, *args, **kwargs):
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, time.perf_counter() - t0


def summarize_errors(re_list, te_list, times):
    re = np.asarray(re_list, float)
    te = np.asarray(te_list, float)
    times = np.asarray(times, float)
    return dict(
        success=float(np.mean((re < SUCCESS_ROT_DEG) & (te < SUCCESS_TRANS))),
        rot_median=float(np.median(re)),
        rot_iqr=float(np.subtract(*np.percentile(re, [75, 25]))),
        rot_p90=float(np.percentile(re, 90)),
        trans_median=float(np.median(te)),
        trans_iqr=float(np.subtract(*np.percentile(te, [75, 25]))),
        trans_p90=float(np.percentile(te, 90)),
        time_ms_p50=float(1000 * np.median(times)),
        time_ms_p95=float(1000 * np.percentile(times, 95)),
    )


def json_dump(path, obj):
    import json
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
