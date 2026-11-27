#!/usr/bin/env python3
"""Weighted 3D-3D rigid registration via Horn's 4x4 quaternion eigenproblem.

This module is the *candidate* inner solver for the Robust-QPEPLayer
experiments.  The current baseline is the weighted Kabsch/SVD solver in
``experiments/synthetic/common.py::kabsch``; both solve the same problem:

    given  x_i (model, N,3),  y_i (observation, N,3),  w_i >= 0
    minimise  F(R,t) = sum_i w_i || R x_i + t - y_i ||^2
    over      R in SO(3), t in R^3

so the solution is the weighted centroid + the best proper rotation.  The
two implementations differ only in how the rotation is extracted:

* Kabsch: 3x3 cross-covariance ``H = sum w (x-cx)(y-cy)^T``, SVD, det fix.
* Horn:   the same cross-covariance goes into a symmetric 4x4 matrix ``N``
          whose *largest-eigenvalue eigenvector* is the unit quaternion of
          the optimal rotation (Horn 1987, "Closed-form solution of
          absolute orientation using unit quaternions", JOSA A 4(4)).

Horn's quadratic form
---------------------
With the unit quaternion ``q = (w, v)``, ``v in R^3``, the rotation is the
homogeneous quadratic map ``R(q) = R_hom(q)`` (see ``Rhom`` in
``synthetic/common.py``; for ``||q||=1`` it equals ``Rmat``).  Then

    sum_i w_i y_i^T R(q) x_i = q^T N q

with the symmetric 4x4 matrix

    N[0, 0]   = trace(S)
    N[0, 1:]  = N[1:, 0] = z
    N[1:, 1:] = (S + S^T) - trace(S) I_3

where, after centring, ``S = sum_i w_i x_i y_i^T`` (3x3) and
``z = [S12-S21, S20-S02, S01-S10]`` is the (twice the) antisymmetric part
of ``S``.  Maximising a quadratic form over the unit sphere gives the
eigenvector of the largest eigenvalue, hence exact global optimality of
the *unconstrained* 6-DoF problem (no local minima).

Det correction
--------------
``R(q)`` has determinant ``(w^2+x^2+y^2+z^2)^3 = 1`` by construction, i.e.
Horn's method lives on SO(3) and needs no reflection branch.  That is in
contrast to Kabsch, whose unconstrained orthogonal Procrustes solution can
be a reflection and is corrected with ``diag(1,1,det(V U^T))``.  As a
safety net (e.g. if a caller replaces the quaternion with an externally
supplied rotation) ``solve`` checks ``det R`` and projects an improper
rotation back to SO(3); on the normal quaternion path the check is a no-op
up to round-off.  The check can be disabled with ``det_check=False`` for
timing the bare 4x4 path.

Degenerate cases
----------------
* ``w`` must be finite and non-negative; a negative or NaN/inf weight
  raises ``ValueError`` (silently clipping would hide upstream bugs).
* If ``sum(w) <= 1e-12`` (all-zero weights) the problem is undefined; we
  fall back to uniform weights, exactly like ``common.kabsch``.  The
  returned info has ``weight_fallback=True``.
* If the two largest eigenvalues of ``N`` are nearly equal, the optimal
  rotation is not unique (e.g. all points collinear / coincident).  We
  report ``eig_gap`` and ``near_degenerate`` and return the eigenvector
  LAPACK picked; the *objective value* is still the global optimum, but
  ``R`` is only one of many minimisers.

Usage
-----
    from davenport import solve
    R, t = solve(x, y, w)
    R, t, info = solve(x, y, w, return_info=True)

Conventions match ``synthetic/common.py``: ``x`` model points, ``y``
observations, solve ``y ~ R x + t``.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "solve",
    "horn_matrix",
    "quat_to_R",
    "rot_frobenius",
]

# A relative eigenvalue gap below this is flagged as (near-)degenerate.
DEGENERATE_GAP = 1e-10


def quat_to_R(q):
    """Rotation matrix from a unit quaternion ``(w, x, y, z)``.

    Form is identical to ``synthetic/common.py::Rmat`` (and to the
    homogeneous ``Rhom`` for ``||q|| = 1``), so rotation-matrix entries are
    directly comparable between the two solvers.
    """
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def horn_matrix(S):
    """Horn's symmetric 4x4 matrix from centred cross-covariance ``S``."""
    S = np.asarray(S, float)
    tr = S[0, 0] + S[1, 1] + S[2, 2]
    z0 = S[1, 2] - S[2, 1]
    z1 = S[2, 0] - S[0, 2]
    z2 = S[0, 1] - S[1, 0]
    a01 = S[0, 1] + S[1, 0]
    a02 = S[0, 2] + S[2, 0]
    a12 = S[1, 2] + S[2, 1]
    N = np.empty((4, 4), float)
    N[0, 0] = tr
    N[0, 1] = z0
    N[0, 2] = z1
    N[0, 3] = z2
    N[1, 0] = z0
    N[1, 1] = 2.0 * S[0, 0] - tr
    N[1, 2] = a01
    N[1, 3] = a02
    N[2, 0] = z1
    N[2, 1] = a01
    N[2, 2] = 2.0 * S[1, 1] - tr
    N[2, 3] = a12
    N[3, 0] = z2
    N[3, 1] = a02
    N[3, 2] = a12
    N[3, 3] = 2.0 * S[2, 2] - tr
    return N


def _det3(R):
    """Cheap 3x3 determinant (cofactor expansion)."""
    return (
        R[0, 0] * (R[1, 1] * R[2, 2] - R[1, 2] * R[2, 1])
        - R[0, 1] * (R[1, 0] * R[2, 2] - R[1, 2] * R[2, 0])
        + R[0, 2] * (R[1, 0] * R[2, 1] - R[1, 1] * R[2, 0])
    )


def _nearest_so3(R):
    """Project a 3x3 matrix to the nearest proper rotation (SVD, det-safe)."""
    U, _, Vt = np.linalg.svd(R)
    d = 1.0 if np.linalg.det(U @ Vt) >= 0.0 else -1.0
    return U @ np.diag([1.0, 1.0, d]) @ Vt


def _check_weights(w, n):
    w = np.asarray(w, float)
    if w.ndim != 1 or w.shape[0] != n:
        raise ValueError(f"weights must have shape ({n},), got {w.shape}")
    total = float(w.sum())
    if not np.isfinite(total):
        raise ValueError("weights must be finite with a finite sum")
    if np.any(w < 0.0):
        raise ValueError("weights must be non-negative")
    fallback = total <= 1e-12
    orig_total = total
    if fallback:
        w = np.ones(n, dtype=float)
        total = float(n)
    return w / total, orig_total, fallback


def solve(x, y, w, *, det_check=True, return_info=False):
    """Weighted 3D-3D rigid registration, y ~ R x + t, via Horn's 4x4.

    Parameters
    ----------
    x, y : (N, 3) float arrays
        Model points and observations (must have the same length).
    w : (N,) non-negative float array
        Per-correspondence weights.  All-zero weights fall back to
        uniform weights (same policy as ``common.kabsch``).
    det_check : bool
        Verify ``det(R) > 0`` and project to SO(3) only if round-off made
        it improper.  Default True; set False to time the bare 4x4 path.
    return_info : bool
        Also return a diagnostics dict.

    Returns
    -------
    R : (3, 3) float  proper rotation
    t : (3,) float    translation
    info : dict       only if ``return_info``; keys:
        ``q`` unit quaternion (w >= 0), ``eigvals`` (4,) ascending,
        ``eig_gap`` (largest - second largest), ``near_degenerate`` bool,
        ``weight_fallback`` bool, ``weight_sum`` original total weight,
        ``det`` determinant of the (checked) rotation.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if x.ndim != 2 or x.shape[1] != 3 or y.shape != x.shape:
        raise ValueError(f"x and y must both be (N,3), got {x.shape} and {y.shape}")
    n = x.shape[0]
    w, total_w, fallback = _check_weights(w, n)

    cx = w @ x
    cy = w @ y
    X = x - cx
    Y = y - cy
    S = (w[:, None] * X).T @ Y          # 3x3, sum w x y^T
    N = horn_matrix(S)

    eigvals, eigvecs = np.linalg.eigh(N)  # ascending
    q = eigvecs[:, -1]
    q = q / np.linalg.norm(q)
    if q[0] < 0.0:
        q = -q
    R = quat_to_R(q)
    if det_check:
        d = _det3(R)
        if d < 0.0:  # unreachable for unit q, kept as a safety net
            R = _nearest_so3(R)
            d = _det3(R)
    else:
        d = 1.0
    t = cy - R @ cx

    if not return_info:
        return R, t
    gap = float(eigvals[-1] - eigvals[-2])
    info = dict(
        q=q,
        eigvals=eigvals,
        eig_gap=gap,
        near_degenerate=bool(gap <= DEGENERATE_GAP * max(1.0, abs(float(eigvals[-1])))),
        weight_fallback=fallback,
        weight_sum=total_w,
        det=float(d),
    )
    return R, t, info


def rot_frobenius(R1, R2):
    """||R1 - R2||_F, the rotation error used for solver comparison."""
    return float(np.linalg.norm(np.asarray(R1) - np.asarray(R2)))
