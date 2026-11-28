#!/usr/bin/env python3
"""Batched GNC-GM for 3D-3D robust registration (numpy, CPU).

Vectorised over a batch of ``B`` independent registration problems, each
with ``N`` 3D-3D correspondences.  Public inputs follow the (B,N,3)
convention used by the synthetic experiments:

* ``x``  (B, N, 3)  model points,
* ``y``  (B, N, 3)  observed points,
* ``w``  (B, N)     weights (weighted-Kabsch primitive only).

Internally everything is stored transposed, ``(B, 3, N)``, so that every
inner iteration is a handful of BLAS ``matmul`` calls instead of strided
``einsum`` reductions.  The weighted Kabsch rotation is computed either by
the batched 3x3 SVD (``method="svd"``, literally the loop's linear algebra)
or by Horn's quaternion / Davenport q-method on a batched 4x4 symmetric
eigendecomposition (``method="eigh"``, the default: mathematically the same
proper rotation, ~5x cheaper on a CPU batch; verified against the
SVD branch to <= 2.1e-15 on the report's fixed instances and < 2e-13
Frobenius for random / near-degenerate H in a standalone stress test).

The iteration schedule is *identical* to
``labs/03-robust-qpeplayer/experiments/synthetic/common.py::solve_gnc``:

* initial weighted Kabsch with unit weights,
* ``mu0 = max(3 * max_i e_i / c^2, 1)``, ``gamma = 1.4``, at most 40 mu
  iterations, ``mu <- max(1, mu/gamma)``, and the loop stops after the
  iteration that uses ``mu = 1``,
* five exact-kernel (``mu = 1``) IRLS polish steps.

Nothing is approximated: only the linear algebra is batched.  Every
instance therefore follows the loop's own number of iterations (the
per-instance mu schedule is advanced with the same float operations as the
loop, and instances that are done are dropped from the active set, so their
results are never recomputed).  The batched implementation is checked
against the loop in ``verify_batch.py``; on fixed random instances the
observed discrepancies are ~1e-15 (rotation Frobenius) and ~1e-16
(translation), far inside the requested 1e-8 / 1e-10 tolerance.
"""
from __future__ import annotations

import numpy as np

# Same constants as synthetic/common.py (kept local so that this module is
# importable without the synthetic experiment package and can be timed in
# isolation).
C = 0.05
C2 = C * C
GNC_GAMMA = 1.4
GNC_MAX_MU_ITERS = 40
GNC_POLISH_STEPS = 5


# ---------------------------------------------------------------------------
# Batched weighted Kabsch
# ---------------------------------------------------------------------------
def _kabsch_T(xT, yT, w, method="eigh"):
    """Weighted Kabsch on ``(B,3,N)`` arrays with ``(B,N)`` weights."""
    B = xT.shape[0]
    s = w.sum(axis=1)
    bad = (~np.isfinite(s)) | (s <= 1e-12)
    if np.any(bad):
        w = np.where(bad[:, None], 1.0, w)
        s = np.where(bad, w.sum(axis=1), s)
    wn = w / s[:, None]                                  # (B,N)

    cx = np.matmul(xT, wn[:, :, None])[:, :, 0]          # (B,3)
    cy = np.matmul(yT, wn[:, :, None])[:, :, 0]
    xc = xT - cx[:, :, None]
    yc = yT - cy[:, :, None]

    # H = sum_i w_i (x_i-cx) (y_i-cy)^T, as one batched GEMM.
    H = np.matmul(xc * wn[:, None, :], np.transpose(yc, (0, 2, 1)))

    if method == "svd":
        U, _, Vt = np.linalg.svd(H)
        Ruv = np.matmul(np.transpose(Vt, (0, 2, 1)), np.transpose(U, (0, 2, 1)))
        d = np.sign(np.linalg.det(Ruv))
        D = np.zeros_like(H)
        D[:, 0, 0] = 1.0
        D[:, 1, 1] = 1.0
        D[:, 2, 2] = d
        R = np.matmul(np.matmul(np.transpose(Vt, (0, 2, 1)), D),
                      np.transpose(U, (0, 2, 1)))
    elif method == "eigh":
        # Horn (1987) / Davenport q-method: the rotation maximising
        # trace(R H) is the proper rotation of the largest eigenvector of
        # this 4x4 symmetric matrix (built from H = sum w xc yc^T).
        K = np.empty((B, 4, 4), dtype=H.dtype)
        H00, H01, H02 = H[:, 0, 0], H[:, 0, 1], H[:, 0, 2]
        H10, H11, H12 = H[:, 1, 0], H[:, 1, 1], H[:, 1, 2]
        H20, H21, H22 = H[:, 2, 0], H[:, 2, 1], H[:, 2, 2]
        K[:, 0, 0] = H00 + H11 + H22
        K[:, 1, 1] = H00 - H11 - H22
        K[:, 2, 2] = -H00 + H11 - H22
        K[:, 3, 3] = -H00 - H11 + H22
        K[:, 0, 1] = K[:, 1, 0] = H12 - H21
        K[:, 0, 2] = K[:, 2, 0] = H20 - H02
        K[:, 0, 3] = K[:, 3, 0] = H01 - H10
        K[:, 1, 2] = K[:, 2, 1] = H01 + H10
        K[:, 1, 3] = K[:, 3, 1] = H20 + H02
        K[:, 2, 3] = K[:, 3, 2] = H12 + H21
        _, V = np.linalg.eigh(K)
        q = V[:, :, -1]                                  # (B,4)
        q = q / np.linalg.norm(q, axis=1, keepdims=True)
        q0, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        R = np.empty((B, 3, 3), dtype=H.dtype)
        R[:, 0, 0] = q0 * q0 + qx * qx - qy * qy - qz * qz
        R[:, 1, 1] = q0 * q0 - qx * qx + qy * qy - qz * qz
        R[:, 2, 2] = q0 * q0 - qx * qx - qy * qy + qz * qz
        R[:, 0, 1] = 2.0 * (qx * qy - q0 * qz)
        R[:, 1, 0] = 2.0 * (qx * qy + q0 * qz)
        R[:, 0, 2] = 2.0 * (qx * qz + q0 * qy)
        R[:, 2, 0] = 2.0 * (qx * qz - q0 * qy)
        R[:, 1, 2] = 2.0 * (qy * qz - q0 * qx)
        R[:, 2, 1] = 2.0 * (qy * qz + q0 * qx)
    else:
        raise ValueError(f"unknown method {method!r}")

    t = cy - np.matmul(R, cx[:, :, None])[:, :, 0]
    return R, t


def batch_kabsch(x, y, w, method="eigh"):
    """Weighted Kabsch for a batch of (B,N,3) problems.

    Parameters
    ----------
    x, y : (B, N, 3) float arrays
    w    : (B, N)    non-negative weights
    method : {"eigh", "svd"}

    Returns
    -------
    R : (B, 3, 3) rotations, t : (B, 3) translations.
    A row of ``w`` summing to <= 1e-12 falls back to unit weights, exactly
    as ``common.kabsch`` does.
    """
    xT = np.ascontiguousarray(np.transpose(np.asarray(x, dtype=float), (0, 2, 1)))
    yT = np.ascontiguousarray(np.transpose(np.asarray(y, dtype=float), (0, 2, 1)))
    return _kabsch_T(xT, yT, np.asarray(w, dtype=float), method=method)


# ---------------------------------------------------------------------------
# Residuals (internal (B,3,N) layout)
# ---------------------------------------------------------------------------
def _residuals_sq_T(xT, yT, R, t):
    r = np.matmul(R, xT) + t[:, :, None] - yT
    return np.einsum("bin,bin->bn", r, r)


def batch_residuals_sq(x, y, R, t):
    """Squared residual norms e_i = ||R x_i + t - y_i||^2, shape (B, N)."""
    xT = np.ascontiguousarray(np.transpose(np.asarray(x, dtype=float), (0, 2, 1)))
    yT = np.ascontiguousarray(np.transpose(np.asarray(y, dtype=float), (0, 2, 1)))
    return _residuals_sq_T(xT, yT, R, t)


def batch_residuals(x, y, R, t):
    """Residual norms ||R x_i + t - y_i||, shape (B, N)."""
    return np.sqrt(batch_residuals_sq(x, y, R, t))


# ---------------------------------------------------------------------------
# Batched GNC-GM
# ---------------------------------------------------------------------------
def batch_solve_gnc(x, y, c=C, gamma=GNC_GAMMA,
                    max_mu_iters=GNC_MAX_MU_ITERS,
                    n_polish=GNC_POLISH_STEPS, method="eigh",
                    return_info=False):
    """GNC-GM on a batch of 3D-3D problems.

    Parameters
    ----------
    x, y : (B, N, 3)
        model / observation points.
    c, gamma, max_mu_iters, n_polish
        kernel scale, mu annealing factor, mu-iteration cap, number of
        final mu = 1 IRLS steps.  Defaults are the ``common.solve_gnc``
        defaults.
    method : {"eigh", "svd"}
        weighted-Kabsch rotation backend (see module docstring).
    return_info : bool
        also return ``dict(mu0, n_mu, n_polish)``; ``n_mu`` is the number
        of mu iterations actually spent per instance, the same quantity
        as the loop's ``info['n_mu']``.

    Returns
    -------
    R : (B, 3, 3), t : (B, 3)
        If ``return_info`` is set, also ``(mu0, n_mu)`` per instance.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 3 or y.ndim != 3 or x.shape != y.shape or x.shape[2] != 3:
        raise ValueError(f"expected x,y of shape (B,N,3), got {x.shape} {y.shape}")
    B, N, _ = x.shape
    c2 = float(c) * float(c)

    # (B,3,N) once; every iteration then stays in BLAS-friendly layout.
    xT = np.ascontiguousarray(np.transpose(x, (0, 2, 1)))
    yT = np.ascontiguousarray(np.transpose(y, (0, 2, 1)))

    # --- initialisation (unit weights, as in solve_gnc) ---
    R, t = _kabsch_T(xT, yT, np.ones((B, N)), method=method)
    e = _residuals_sq_T(xT, yT, R, t)

    mu = np.maximum(3.0 * e.max(axis=1) / c2, 1.0)
    mu0 = mu.copy()
    n_mu = np.zeros(B, dtype=np.int64)

    active = np.arange(B)
    for _ in range(int(max_mu_iters)):
        if active.size == 0:
            break
        # same float expression as the loop: w = (mu c^2 / (mu c^2 + e))^2
        a = mu[active][:, None] * c2
        w = (a / (a + e[active])) ** 2
        R[active], t[active] = _kabsch_T(xT[active], yT[active], w, method=method)
        e[active] = _residuals_sq_T(xT[active], yT[active],
                                    R[active], t[active])
        n_mu[active] += 1
        # The loop breaks *after* using mu == 1; replicate exactly.
        done = mu[active] <= 1.0 + 1e-12
        active = active[~done]
        mu[active] = np.maximum(1.0, mu[active] / gamma)

    # --- exact-kernel (mu = 1) IRLS polish, same 5 steps for everyone ---
    for _ in range(int(n_polish)):
        w = (c2 / (c2 + e)) ** 2
        R, t = _kabsch_T(xT, yT, w, method=method)
        e = _residuals_sq_T(xT, yT, R, t)

    if return_info:
        return R, t, dict(mu0=mu0, n_mu=n_mu, n_polish=int(n_polish))
    return R, t


def batch_solve_irls(x, y, c=C, iters=40, method="eigh"):
    """Batched IRLS-GM (no annealing), matching ``common.solve_irls``.

    Provided for completeness; not used by the benchmark tables.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    B, N, _ = x.shape
    c2 = float(c) * float(c)
    xT = np.ascontiguousarray(np.transpose(x, (0, 2, 1)))
    yT = np.ascontiguousarray(np.transpose(y, (0, 2, 1)))
    R, t = _kabsch_T(xT, yT, np.ones((B, N)), method=method)
    for _ in range(int(iters)):
        e = _residuals_sq_T(xT, yT, R, t)
        w = (c2 / (c2 + e)) ** 2
        R, t = _kabsch_T(xT, yT, w, method=method)
    return R, t
