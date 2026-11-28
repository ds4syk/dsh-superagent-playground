#!/usr/bin/env python3
"""Shared helpers for the P2 *real-data* layer experiments (BOP / KITTI / UTIAS).

Everything here is deliberately small and dependency-light: numpy + qpep_core.
The point is to build 3D-2D correspondences from real sensors, encode the
physical constraints that the Constrained-QPEPLayer theory accepts (affine in
``v2(q)`` and ``t``), and then hand them to the same four solvers used by the
synthetic / BOP experiments so the numbers are directly comparable.

Constraint conventions (identical to ``qpep_core``)
---------------------------------------------------
``v2(q) = [q0^2, q1^2, q2^2, q3^2, sqrt2 q0q1, sqrt2 q0q2, sqrt2 q0q3,
           sqrt2 q1q2, sqrt2 q1q3, sqrt2 q2q3]`` and ``e_tr^T v2 = 1``, so an
affine constraint is a pair ``(a, b)`` meaning ``a^T v2 + b^T t <= 0``.  Any
constant ``c`` is folded in by subtracting ``c * e_tr`` from ``a``.

New constraints added here (not in ``qpep_core``, same algebra class):

* :func:`c_tilt_axis` -- "the estimated rotation must not tilt the world-up
  axis by more than alpha": ``a . (R a) >= cos(alpha)`` with ``a`` the up
  vector expressed in the source-camera frame.  This is the *exact* roll/pitch
  cone, and it is affine in ``v2`` because ``R(q) a = m_rows(a) v2``.
* :func:`c_box` -- image-boundary / frustum constraint
  ``s * ((R p + t)_axis) - tau * ((R p + t)_z) <= 0`` for one image border.
  Four of them (left/right/top/bottom) give the full FoV frustum.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

CODE = Path(__file__).resolve().parents[2] / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))
import qpep_core as core  # noqa: E402
from certificate import shor_bound  # noqa: E402,F401


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------
def wilson(k, n, z=1.959963984540054):
    """95% Wilson score interval for k successes out of n."""
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    den = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, max(0.0, center - half), min(1.0, center + half)


def rate(xs):
    """dict with k, n, p, lo, hi (Wilson) for a list of booleans."""
    xs = [bool(x) for x in xs]
    p, lo, hi = wilson(sum(xs), len(xs))
    return dict(k=int(sum(xs)), n=len(xs), p=p, lo=lo, hi=hi)


def median_ci(xs, n_boot=2000, seed=0, alpha=0.05):
    """Percentile bootstrap CI for the median."""
    x = np.asarray([v for v in xs if np.isfinite(v)], dtype=float)
    if x.size == 0:
        return dict(n=0, median=float("nan"), q1=float("nan"), q3=float("nan"),
                    lo=float("nan"), hi=float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    med = np.median(x[idx], axis=1)
    return dict(n=int(x.size), median=float(np.median(x)),
                q1=float(np.percentile(x, 25)), q3=float(np.percentile(x, 75)),
                lo=float(np.percentile(med, 100 * alpha / 2)),
                hi=float(np.percentile(med, 100 * (1 - alpha / 2))))


def rmse(xs):
    x = np.asarray([v for v in xs if np.isfinite(v)], dtype=float)
    return float(np.sqrt(np.mean(x ** 2))) if x.size else float("nan")


def summarise(xs):
    x = np.asarray([v for v in xs if np.isfinite(v)], dtype=float)
    if x.size == 0:
        return dict(n=0, mean=float("nan"), median=float("nan"),
                    q1=float("nan"), q3=float("nan"), rmse=float("nan"))
    return dict(n=int(x.size), mean=float(x.mean()), median=float(np.median(x)),
                q1=float(np.percentile(x, 25)), q3=float(np.percentile(x, 75)),
                rmse=float(np.sqrt(np.mean(x ** 2))))


# ---------------------------------------------------------------------------
# constraints
# ---------------------------------------------------------------------------
def c_tilt_axis(a, alpha_deg):
    """``a . (R(q) a) >= cos(alpha)``  (roll/pitch cone around the up axis a).

    ``a`` is the world-up vector expressed in the source camera frame; for a
    ground vehicle it is available from the previous frame's attitude (IMU or
    odometry), see the KITTI/UTIAS scripts for the exact provenance.
    """
    a = np.asarray(a, float)
    a = a / np.linalg.norm(a)
    # cos(alpha) - a.(R a) <= 0
    A = np.cos(np.radians(alpha_deg)) * core.E_TR - a @ core.m_rows(a)
    return A, np.zeros(3)


def c_box(v, axis, tau, s):
    """One image-border (frustum) half-space on the *estimated* point position.

    ``s * z_axis - tau * z_z <= 0`` where ``z = R v + t``.  With ``axis=0``:
    a point is inside the horizontal FoV iff both ``c_box(v, 0, cx/fx, -1)``
    and ``c_box(v, 0, (W-cx)/fx, +1)`` hold (and analogously for ``axis=1``).
    """
    Mm = core.m_rows(v)
    A = float(s) * Mm[axis] - float(tau) * Mm[2]
    B = np.zeros(3)
    B[axis] = float(s)
    B[2] = -float(tau)
    return A, B


def stack(cs):
    """(A, B) from a list of (a, b) pairs."""
    A = np.array([c[0] for c in cs], dtype=float)
    B = np.array([c[1] for c in cs], dtype=float)
    return A, B


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------
def R_t_of_pose(p):
    """3x4 row-major KITTI pose -> (R, t)."""
    p = np.asarray(p, float).reshape(3, 4)
    return p[:, :3], p[:, 3]


def rel_motion(T_wi, T_wj):
    """X_j = R_rel X_i + t_rel for world poses T_w* = (R,t)."""
    R_wi, t_wi = T_wi
    R_wj, t_wj = T_wj
    R = R_wj.T @ R_wi
    t = R_wj.T @ (t_wi - t_wj)
    return R, t


def project(K, X):
    uv = (K @ X.T).T
    return uv[:, :2] / uv[:, 2:3], uv[:, 2]


def bearings(K, uv):
    U = (np.linalg.inv(K) @ np.vstack([uv.T, np.ones(uv.shape[0])])).T
    return U / np.linalg.norm(U, axis=1, keepdims=True)


def tilt_deg(a, R):
    """Angle between the up axis a and R a, in degrees."""
    a = np.asarray(a, float)
    a = a / np.linalg.norm(a)
    return float(np.degrees(np.arccos(np.clip(a @ (R @ a), -1.0, 1.0))))


def solve_all(core_mod, M0, D0, G0, A, B, q_u, t_u, rng, n_starts=(32, 8, 4, 6)):
    """The four methods of the paper on one instance (same call pattern as
    ``run_bop.py``).  Returns {name: (q, t, seconds)}."""
    out = {}
    s_unc, s_soft, s_proj, s_alm = n_starts
    t0 = core_mod.time.perf_counter()
    J, q, t = q_u, t_u[0], t_u[1]
    out["Unconstrained"] = (q, t, core_mod.time.perf_counter() - t0)
    t0 = core_mod.time.perf_counter()
    try:
        r = core_mod.soft_penalty(M0, D0, G0, A, B, 1e3, q, t, n_starts=s_soft, rng=rng)
        out["Soft penalty"] = (r["q"], r["t"], core_mod.time.perf_counter() - t0)
    except Exception:
        out["Soft penalty"] = (q, t, core_mod.time.perf_counter() - t0)
    t0 = core_mod.time.perf_counter()
    try:
        r = core_mod.project_feasible(M0, D0, G0, A, B, q, t, n_starts=s_proj, rng=rng)
        out["Projection"] = (r["q"], r["t"], core_mod.time.perf_counter() - t0)
    except Exception:
        out["Projection"] = (q, t, core_mod.time.perf_counter() - t0)
    t0 = core_mod.time.perf_counter()
    try:
        r = core_mod.alm_qpep(M0, D0, G0, A, B, q, t, rho0=10.0, n_starts=s_alm, rng=rng)
        out["ALM-QPEP"] = (r["q"], r["t"], core_mod.time.perf_counter() - t0)
    except Exception:
        out["ALM-QPEP"] = (q, t, core_mod.time.perf_counter() - t0)
    return out


def dump(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=float)
    print("wrote", path, flush=True)
