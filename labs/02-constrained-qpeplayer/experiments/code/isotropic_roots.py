#!/usr/bin/env python
"""Root counting / solving for the isotropic (Horn + plane) family, theorem P2-1b.

Provides
--------
    count_roots(M, n_gamma=2, seed=0)        -> dict (the P2-1b spectrum numbers)
    complex_roots(M, n_gamma=1, seed=0)      -> finite complex roots + Jacobian data
    solve_isotropic_plane(instance, ...)     -> global min of the ALM/canonical
                                                inner problem (verified reduced solver)
    davenport_solve(P, Y, w)                 -> the k=0 Davenport 4x4 solution

The *counting* backend is the committed 162-path total-degree homotopy
(:mod:`quartic_roots`).  For the structured (isotropic) matrices many of the
162 paths diverge (roots at infinity); this is expected and is reported as
``n_diverged``, not as failure.

Solver status (honest, see results/round_roots.md round 5/6)
------------------------------------------------------------
The theorem's 16x16 *saturated action matrix* was built offline for one
canonical rank-2 instance (symbolic Groebner basis over Q; see
``results/isotropic_gb_reference.json``).  For arbitrary numeric instances this
module uses a *verified reduced solver*:

  * one/ two Gamma-phase homotopy restricted to the known finite spectrum,
  * all finite roots are endgame-polished and re-checked on the KKT residual,
  * the isolated roots are separated by the smallest singular value of the
    KKT Jacobian (points on the positive-dimensional q^Tq=0 component, if any
    numerical shadow appears, have a singular Jacobian),
  * the global minimum is cross-checked against 64-start BFGS.

This is a *numerical* isolation, not an algebraic certification.  It is
validated against the symbolic 16-root reference on the canonical instance and
against the full 162-path enumeration on random instances.
"""
from __future__ import annotations

import time

import numpy as np

import isotropic as ISO
import quartic_roots as QR

__all__ = ["count_roots", "complex_roots", "solve_isotropic_plane",
           "davenport_solve", "kkt_jac_min_sv", "distinct_projectively"]


# ---------------------------------------------------------------------------
# Jacobian / isolation helpers
# ---------------------------------------------------------------------------
def kkt_jac_min_sv(M, q, lam):
    """Smallest singular value of the 5x5 KKT Jacobian at x=(q,lam)."""
    M = 0.5 * (np.asarray(M, float) + np.asarray(M, float).T)
    _, J = QR.kkt_eval(np.concatenate([np.asarray(q, float), [float(lam)]]), M)
    return float(np.linalg.svd(J, compute_uv=False)[-1])


def distinct_projectively(qs, tol=1e-6):
    """Cluster complex quaternions modulo q ~ -q."""
    reps = []
    for v in qs:
        nv = np.linalg.norm(v)
        hit = False
        for r in reps:
            d = min(np.linalg.norm(v - r), np.linalg.norm(v + r))
            if d <= tol * max(1.0, nv, np.linalg.norm(r)):
                hit = True
                break
        if not hit:
            reps.append(v)
    return reps


# ---------------------------------------------------------------------------
# counting
# ---------------------------------------------------------------------------
def count_roots(M, n_gamma=2, seed=0, n_gamma_min=1):
    """P2-1b root census for one reduced matrix M (10x10, v2 basis).

    Returns dict with n_finite_projective, n_finite_affine, n_real_geometric,
    n_real_affine, n_lost, n_diverged, n_paths, f_min, time.
    """
    t0 = time.perf_counter()
    roots, st = QR.critical_points(M, seed=seed, return_stats=True,
                                   n_gamma=n_gamma)
    dt = time.perf_counter() - t0
    return dict(
        n_finite_projective=int(st["n_finite_projective"]),
        n_finite_affine=int(st["n_finite_affine"]),
        n_real_geometric=int(st["n_unique"]),
        n_real_affine=int(st["n_real_affine"]),
        n_lost=int(st["n_lost"]),
        n_diverged=int(st["n_diverged"]),
        n_paths=int(st["n_paths"]),
        n_gamma=int(n_gamma),
        f_min=float(roots[0]["f"]) if roots else float("nan"),
        n_roots_listed=len(roots),
        time=float(dt),
        stats=st,
    )


def complex_roots(M, n_gamma=1, seed=0, res_tol=1e-8):
    """All finite complex KKT roots found by the homotopy (merged over phases).

    Returns list of dicts {q, lam, res, sigma_min, isolated}.  ``q`` is a
    projective representative (q ~ -q collapsed); ``lam`` is the KKT multiplier
    in the quartic_roots convention (grad f = 2 lam q).
    """
    M = np.asarray(M, dtype=float)
    M = 0.5 * (M + M.T)
    rng = np.random.default_rng(12345 + seed + 100000)
    gammas = [np.exp(2j * np.pi * rng.random())
              for _ in range(max(1, int(n_gamma)))]
    Xs, ts = [], []
    for gm in gammas:
        ph = QR._run_homotopy(M, gm)
        ts.append(ph["t_track"] + ph["t_polish"])
        idx = np.where(ph["finite"])[0]
        Xs.append(ph["X"][idx])
    Xall = np.vstack(Xs) if Xs else np.zeros((0, 5), complex)
    if len(Xall) == 0:
        return [], dict(t_homotopy=float(sum(ts)), n_finite=0)
    reps, assign = QR._proj_cluster(Xall[:, :4])
    out = []
    for k in range(len(reps)):
        members = Xall[assign == k]
        # best (smallest residual on F) member, then sign-align to the rep
        Xp, res = QR._polish_F(members.copy(), M)
        i = int(np.argmin(res))
        q = Xp[i, :4]
        lam = Xp[i, 4]
        if np.real(np.vdot(reps[k], q)) < 0:
            q, lam = -q, lam
        out.append(dict(q=q, lam=lam, res=float(res[i]),
                        sigma_min=kkt_jac_min_sv(M, np.real(q), np.real(lam))
                        if np.max(np.abs(np.imag(q))) < 1e-7 else float("nan"),
                        isolated=None))
    for d in out:
        d["isolated"] = bool(np.isfinite(d["sigma_min"]) and d["sigma_min"] > 1e-8)
    return out, dict(t_homotopy=float(sum(ts)), n_finite=int(len(Xall)))


# ---------------------------------------------------------------------------
# reduced solver for the ALM / canonical inner problem
# ---------------------------------------------------------------------------
def _multistart_min(M, n_starts=64, seed=0):
    return QR._multistart_fallback(M, n_starts=n_starts, seed=seed)


def solve_isotropic_plane(instance, n_gamma=2, seed=0, multistart=64,
                          residual_tol=1e-8, jac_tol=1e-8, return_all=False):
    """Global min of the ALM/canonical inner problem for one instance.

    ``instance`` is a dict from :func:`isotropic.make_isotropic_instance` /
    :func:`isotropic.make_canonical_instance`.  The reduced matrix is taken from
    ``instance['Mred']`` when present, else assembled from the frozen active set
    (all k constraints, mu=0, instance rho) exactly as ``alm_qpep`` does.

    Returns dict(f, q, lam, f_multistart, n_finite_projective, n_isolated,
    max_residual, time, roots, fallback, solver="reduced-homotopy").
    """
    t0 = time.perf_counter()
    if "Mred" in instance:
        Mred = np.asarray(instance["Mred"], float)
    else:
        Mred, _, _, _ = ISO.alm_reduced(instance, act=np.arange(instance["k"]),
                                        rho=instance.get("rho"))
    Mred = 0.5 * (Mred + Mred.T)
    roots, st = QR.critical_points(Mred, seed=seed, return_stats=True,
                                   n_gamma=n_gamma)
    dt_h = time.perf_counter() - t0
    # isolated real roots: nonsingular KKT Jacobian
    iso = []
    for r in roots:
        smin = kkt_jac_min_sv(Mred, r["q"], r["lam"])
        iso.append((smin, r))
    isolated = [r for smin, r in iso if smin > jac_tol]
    max_res = max([r["res"] for r in isolated], default=float("inf"))
    f_enum = float(isolated[0]["f"]) if isolated else (
        float(roots[0]["f"]) if roots else float("inf"))
    q_enum = isolated[0]["q"] if isolated else (roots[0]["q"] if roots else None)
    lam = float(isolated[0]["lam"]) if isolated else (
        float(roots[0]["lam"]) if roots else float("nan"))
    f_ms, q_ms = _multistart_min(Mred, n_starts=multistart, seed=seed)
    fallback = False
    f, q = f_enum, q_enum
    if q_ms is not None and np.isfinite(f_ms) and f_ms < f - 1e-9:
        f, q, fallback = float(f_ms), q_ms.copy(), True
    if q is not None and q[0] < 0:
        q = -q
    out = dict(f=float(f), f_enum=f_enum, q=q, lam=lam,
               f_multistart=float(f_ms) if np.isfinite(f_ms) else None,
               n_finite_projective=int(st["n_finite_projective"]),
               n_finite_affine=int(st["n_finite_affine"]),
               n_real_geometric=int(st["n_unique"]),
               n_isolated=len(isolated),
               n_real_roots=len(roots),
               max_residual=float(max_res),
               n_lost=int(st["n_lost"]),
               time=float(time.perf_counter() - t0), t_homotopy=float(dt_h),
               fallback=bool(fallback), solver="reduced-homotopy",
               roots=roots if return_all else None)
    return out


# ---------------------------------------------------------------------------
# k = 0 : Davenport 4x4 q-method (independent, exact reduction)
# ---------------------------------------------------------------------------
def _quat_quadratic(K):
    """4x4 symmetric matrix Kq with q^T Kq q == q^T K q (identity map)."""
    return np.asarray(K, float)


def davenport_solve(P, Y, w, centered=True):
    """Solve the isotropic k=0 Horn problem exactly via a 4x4 eigenproblem.

    J(R,t) = sum_i w_i ||R p_i + t - y_i||^2.  After eliminating t this is
    (const) - 2 tr(R H) with H = sum_i w_i p~_i y~_i^T (centered).  With
    tr(R(q) H) = c^T v2(q) = q^T K q, the four critical rotations are the four
    eigenvectors of the 4x4 symmetric K.  Returns (f_min, q_best, crit) where
    crit is a list of (q, f) for all four projective critical points.
    """
    P = np.atleast_2d(np.asarray(P, float))
    Y = np.atleast_2d(np.asarray(Y, float))
    w = np.asarray(w, float)
    s = w.sum()
    if centered:
        Pc = P - (w[:, None] * P).sum(0) / s
        Yc = Y - (w[:, None] * Y).sum(0) / s
    else:
        Pc, Yc = P, Y
    H = (w[:, None] * Pc).T @ Yc             # 3x3, sum w p y^T
    # constant of J after centering: sum w ||p~||^2 + sum w ||y~||^2
    const = float((w * np.sum(Pc ** 2, axis=1)).sum()
                  + (w * np.sum(Yc ** 2, axis=1)).sum())
    # c^T v2 = tr(R H):  c_i = H_ii ; c_cross = sqrt2 * H_ij
    ell = ISO.v2_linear_coeff(np.block([[H, np.zeros((3, 1))],
                                        [np.zeros((1, 4))]])[:4, :4] * 0)  # placeholder
    # build c directly: tr(R H) = sum_{a,b} H[b,a] (m(e_b) v2)_a
    c = np.zeros(10)
    for b in range(3):
        e = np.zeros(3)
        e[b] = 1.0
        c += H[:, b] @ ISO.m_rows(e)
    Kq = np.zeros((4, 4))
    Kq[np.diag_indices(4)] = c[:4]
    for k, (i, j) in enumerate([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]):
        Kq[i, j] = Kq[j, i] = c[4 + k] / np.sqrt(2.0)
    evals, evecs = np.linalg.eigh(Kq)
    crit = []
    for m in range(4):
        q = evecs[:, m]
        if q[0] < 0:
            q = -q
        f = const - 2.0 * float(evals[m])
        crit.append((q, f))
    crit.sort(key=lambda t: t[1])
    return crit[0][1], crit[0][0], crit
