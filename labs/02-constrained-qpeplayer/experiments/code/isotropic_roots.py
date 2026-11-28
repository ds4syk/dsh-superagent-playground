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

Solver status (honest; see ``results/round_roots.md``, ``results/isotropic_spectrum.md``)
------------------------------------------------------------------------------------
* The 16x16 *saturated action matrix* (Rabinowitsch saturation, 17-monomial
  Groebner staircase) is established in the **user-supplied proof** of theorem
  P2-1b (Feishu doc, 2026-11-27).  It was **not** built as a shipped numeric
  template in this round: for one canonical rank-2 instance it remains a
  symbolic reference only, and for arbitrary numeric instances no Macaulay/SVD
  quotient-basis action matrix is implemented.  The earlier claim that such a
  reference JSON existed is retracted; there is no
  ``results/isotropic_gb_reference.json``.
* What *is* implemented and validated here:
  - a corrected root **census** (``count_roots`` / ``_census``) that polishes
    every homotopy path -- including the ones classified ``diverged`` during
    tracking -- and uses a configurable ``norm_cap``.  This is what makes the
    ALM-assembled Type II matrices report their theorem-predicted 16 (the
    committed ``quartic_roots`` census drops the 4 extra roots at
    ``||q||~30`` because their paths are cut during tracking and never
    polished);
  - a **structure discriminator**: ``isotropic.canonical_fit`` measures the
    relative polynomial distance of a reduced matrix to the 15-dim canonical
    family F_n and returns the PSD-gauge rank of Sigma_A (0/1/2 -> 4/12/16).
    Isotropic Horn + plane assemblies give 1e-15, anisotropic/PnP give 1e-1;
  - the global-min **reduced solver** ``solve_isotropic_plane``: enumeration +
    endgame polish + KKT-Jacobian isolation + 64-start BFGS cross-check.
    Parity with the committed ``quartic_roots.solve_inner_exact`` on 10
    canonical rank-2 and 5 ALM instances is in
    ``results/isotropic_parity.json``.

This is a *numerical* isolation/census, not an algebraic certification.
"""
from __future__ import annotations

import time

import numpy as np

import isotropic as ISO
import quartic_roots as QR

__all__ = ["count_roots", "complex_roots", "solve_isotropic_plane",
           "davenport_solve", "kkt_jac_min_sv", "distinct_projectively",
           "classify_structure", "solve_inner_auto"]


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
# counting (corrected finite-norm classification)
# ---------------------------------------------------------------------------
# The committed quartic_roots._run_homotopy classifies a polished path as
# "finite" only when ||x|| < 1e3 with x=(q, lam) -- i.e. the *multiplier is
# included in the norm test* (hard-coded line "finite = ... & (xnorm<1e3)").
# On the P2-1b type-II family the extra complex roots (relative to the rank-1
# case) have moderate ||q|| ~ 12..34 but *large multipliers* |lam| ~ 4e3..8e3
# (for a quartic critical point lam = 4f/(q^Tq)), so they fail the 1e3 combined
# norm test and are silently discarded as "diverged".  The census below applies
# the cap to ||q|| alone (the roots are finite points of the affine KKT system;
# q^Tq=1 admits arbitrarily large components on the complex quadric) and also
# polishes the paths that tracking classified as diverged.  It reports both the
# corrected and the committed (1e3 combined-norm) counts.
def _phase_census(M, gamma, norm_cap=1e5, newton_tol=1e-11, div_tol=1e4):
    X, t, status, n_steps, n_evals, n_sing = QR._track_batch(
        M, gamma, newton_tol=newton_tol, div_tol=div_tol)
    cand = np.where(np.isin(status, [1, 2, 3, 4]))[0]
    res = np.full(X.shape[0], np.inf)
    if cand.size:
        with np.errstate(all="ignore"):
            Xp, rp = QR._polish_F(X[cand].copy(), M)
        X[cand] = Xp
        res[cand] = rp
    xnorm = np.linalg.norm(X, axis=1)            # ||(q,lam)||, for diagnostics
    qnorm = np.linalg.norm(X[:, :4], axis=1)     # the roots' own magnitude
    finite = np.isfinite(res) & (res < 1e-8) & (qnorm < norm_cap)
    st = status.copy()
    st[finite] = 1
    nonroot = ~finite
    st[nonroot & (xnorm > 1e2)] = 3
    st[nonroot & (xnorm <= 1e2)] = 2
    return dict(X=X, res=res, status=st, xnorm=xnorm, qnorm=qnorm,
                finite=finite, n_evals=n_evals, n_singular=n_sing)


def _census(M, n_gamma=2, seed=0, norm_cap=1e5, verbose=False,
            newton_tol=1e-11, div_tol=1e4):
    """All finite complex roots with a configurable norm cap (see note above).

    Returns (real_roots_sorted, stats) with the same ``real_roots`` layout as
    :func:`quartic_roots.critical_points` plus corrected census statistics.
    """
    M = np.asarray(M, dtype=float)
    M = 0.5 * (M + M.T)
    rng = np.random.default_rng(12345 + seed + 100000)
    gammas = [np.exp(2j * np.pi * rng.random())
              for _ in range(max(1, int(n_gamma)))]
    phases = []
    for gm in gammas:
        phases.append(_phase_census(M, gm, norm_cap=norm_cap,
                                    newton_tol=newton_tol, div_tol=div_tol))
    Xs = [ph["X"][ph["finite"]] for ph in phases]
    Xall = np.vstack(Xs) if Xs else np.zeros((0, 5), complex)
    if len(Xall):
        reps_p, assign_p = QR._proj_cluster(Xall[:, :4])
        n_proj = len(reps_p)
        # per-cluster norm (for the large-norm diagnostics)
        xn = np.linalg.norm(Xall[:, :4], axis=1)
        cl_norm = np.zeros(n_proj)
        for k in range(n_proj):
            cl_norm[k] = np.max(xn[assign_p == k])
    else:
        n_proj = 0
        cl_norm = np.zeros(0)
    # committed 1e3 cap (applied to ||(q,lam)||), for the before/after comparison
    Xlo = [ph["X"][ph["finite"] & (ph["xnorm"] < 1e3)] for ph in phases]
    Xlo = np.vstack(Xlo) if any(len(x) for x in Xlo) else np.zeros((0, 5), complex)
    n_proj_cap1e3 = len(QR._proj_cluster(Xlo[:, :4])[0]) if len(Xlo) else 0

    # ---- real roots (identical logic to quartic_roots.critical_points) ----
    real = []
    for ph in phases:
        for i in np.where(ph["finite"])[0]:
            x = ph["X"][i]
            im = np.max(np.abs(x.imag))
            sc = max(1.0, np.max(np.abs(x.real)))
            if im > 1e-7 * sc:
                continue
            qr = np.real(x[:4]).copy()
            lam = float(np.real(x[4]))
            nq = np.linalg.norm(qr)
            if abs(nq - 1.0) > 1e-7:
                continue
            qr = qr / nq
            Fr, _ = QR.kkt_eval(np.concatenate([qr, [lam]]), M)
            if float(np.linalg.norm(Fr)) > 1e-8:
                continue
            real.append(dict(q=qr, lam=lam, f=QR._quartic_f(qr, M),
                             res=float(np.linalg.norm(Fr))))
    uniq = []
    if real:
        reps, assign = QR._cluster([QR._canon_sign(r["q"]) for r in real],
                                   QR._dist_affine, 1e-7)
        for k in range(len(reps)):
            members = [real[i] for i in range(len(real)) if assign[i] == k]
            best = min(members, key=lambda d: d["f"])
            best["q"] = QR._canon_sign(best["q"])
            best["n_paths"] = len(members)
            uniq.append(best)
        uniq.sort(key=lambda d: d["f"])

    st = dict(
        n_paths=162 * len(phases), n_phases=len(phases),
        n_finite_paths=sum(int(ph["finite"].sum()) for ph in phases),
        n_diverged=sum(int((ph["status"] == 3).sum()) for ph in phases),
        n_newton_fail=sum(int((ph["status"] == 2).sum()) for ph in phases),
        n_lost=sum(int((ph["status"] == 2).sum()) for ph in phases),
        n_finite_projective=int(n_proj),
        n_finite_affine=int(2 * n_proj),
        n_finite_projective_cap1e3=int(n_proj_cap1e3),
        n_large_norm=int(np.sum(cl_norm >= 1e3)) if len(cl_norm) else 0,
        max_finite_norm=float(np.max(cl_norm)) if len(cl_norm) else 0.0,
        max_finite_qnorm=float(np.max([np.max(ph["qnorm"][ph["finite"]])
                                       for ph in phases if ph["finite"].any()],
                                      initial=0.0)),
        max_finite_lam=float(np.max([np.max(np.abs(ph["X"][ph["finite"], 4]))
                                     for ph in phases if ph["finite"].any()],
                                    initial=0.0)),
        n_real_geometric=len(uniq), n_unique=len(uniq),
        n_real_affine=2 * len(uniq),
        norm_cap=float(norm_cap),
        n_batch_evals=sum(ph["n_evals"] for ph in phases),
        phase_stats=[dict(n_finite=int(ph["finite"].sum()),
                          n_div=int((ph["status"] == 3).sum()),
                          n_fail=int((ph["status"] == 2).sum()))
                     for ph in phases],
    )
    if verbose:
        print("  %d paths x %d phases | finite=%d | proj=%d (cap1e3 %d, "
              "large-norm %d, max|q| %.0f max|lam| %.0f) | real=%d lost=%d div=%d"
              % (162, len(phases), st["n_finite_paths"], n_proj,
                 n_proj_cap1e3, st["n_large_norm"], st["max_finite_qnorm"],
                 st["max_finite_lam"], len(uniq), st["n_lost"], st["n_diverged"]))
    return uniq, st


def count_roots(M, n_gamma=2, seed=0, n_gamma_min=1, norm_cap=1e5,
                verbose=False):
    """P2-1b root census for one reduced matrix M (10x10, v2 basis).

    ``norm_cap`` is the ||q|| threshold for "finite" (see the note above).
    Returns dict with n_finite_projective / n_finite_affine /
    n_real_geometric / n_real_affine / n_lost / n_diverged / n_paths /
    f_min / time, plus ``n_finite_projective_cap1e3`` (the committed
    quartic_roots behaviour), ``n_large_norm`` and ``max_finite_norm``.
    """
    t0 = time.perf_counter()
    roots, st = _census(M, n_gamma=n_gamma, seed=seed, norm_cap=norm_cap,
                        verbose=verbose)
    dt = time.perf_counter() - t0
    return dict(
        n_finite_projective=int(st["n_finite_projective"]),
        n_finite_affine=int(st["n_finite_affine"]),
        n_finite_projective_cap1e3=int(st["n_finite_projective_cap1e3"]),
        n_large_norm=int(st["n_large_norm"]),
        max_finite_norm=float(st["max_finite_norm"]),
        max_finite_qnorm=float(st["max_finite_qnorm"]),
        max_finite_lam=float(st["max_finite_lam"]),
        n_real_geometric=int(st["n_real_geometric"]),
        n_real_affine=int(st["n_real_affine"]),
        n_lost=int(st["n_lost"]),
        n_diverged=int(st["n_diverged"]),
        n_paths=int(st["n_paths"]),
        n_gamma=int(n_gamma),
        norm_cap=float(norm_cap),
        f_min=float(roots[0]["f"]) if roots else float("nan"),
        n_roots_listed=len(roots),
        time=float(dt),
        stats=st,
    )


# ---------------------------------------------------------------------------
# reduced solver for the ALM / canonical inner problem
# ---------------------------------------------------------------------------
def _multistart_min(M, n_starts=64, seed=0):
    return QR._multistart_fallback(M, n_starts=n_starts, seed=seed)


def solve_isotropic_plane(instance, n_gamma=2, seed=0, multistart=64,
                          residual_tol=1e-8, jac_tol=1e-8, return_all=False,
                          norm_cap=1e5, structure=None):
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
    roots, st = _census(Mred, seed=seed, n_gamma=n_gamma, norm_cap=norm_cap)
    dt_h = time.perf_counter() - t0
    # optional P2-1b structure annotation (pure diagnostic; the real roots --
    # the only ones relevant for the global min -- are unaffected by the
    # canonical representative, which has the identical polynomial).
    fit = None
    n = None
    if structure is not None and n is None:
        n = instance.get("n") if "n" in instance else None
        if n is None and instance.get("Ns") is not None and len(instance["Ns"]):
            n = np.asarray(instance["Ns"][0], float)
    if structure is not None and n is not None:
        try:
            fit = ISO.canonical_fit(Mred, n)
        except Exception:
            fit = None
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
               n_real_geometric=int(st["n_real_geometric"]),
               n_isolated=len(isolated),
               n_real_roots=len(roots),
               max_residual=float(max_res),
               n_lost=int(st["n_lost"]),
               time=float(time.perf_counter() - t0), t_homotopy=float(dt_h),
               fallback=bool(fallback), solver="reduced-homotopy",
               n_finite_projective_cap1e3=int(st["n_finite_projective_cap1e3"]),
               n_large_norm=int(st["n_large_norm"]),
               max_finite_norm=float(st["max_finite_norm"]),
        max_finite_qnorm=float(st["max_finite_qnorm"]),
        max_finite_lam=float(st["max_finite_lam"]),
               norm_cap=float(norm_cap),
               rank_Sigma=(fit["rank_Sigma"] if fit is not None else None),
               predicted_NC=(fit["predicted_NC"] if fit is not None else None),
               family_residual=(fit["residual"] if fit is not None else None),
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


# ---------------------------------------------------------------------------
# structure-aware dispatcher (S6)
# ---------------------------------------------------------------------------
def classify_structure(M, n=None, n2=None, tol=1e-8):
    """P2-1b structure test for a reduced matrix (10x10, v2 basis).

    Returns dict(structure, rank_Sigma, rank_Sigma2, predicted_NC,
    family_residual, in_family, M_can).  ``structure`` is one of

      * ``"canonical_typeII"``            -- single normal, in F_n, N_C=4/12/16
      * ``"canonical_typeII_two_normals"``-- two normals, in F_{n1,n2}, N_C≈32
      * ``"typeI_generic"``               -- outside the canonical family, N_C=40

    ``M_can`` is the canonical representative (identical polynomial, PSD-gauge
    Sigma) and is the natural matrix to feed the 4/12/16 census.
    """
    if n is None:
        return dict(structure="typeI_generic", rank_Sigma=None, rank_Sigma2=None,
                    predicted_NC=40, family_residual=None, in_family=False,
                    M_can=np.asarray(M, float))
    fit = ISO.canonical_fit(M, n) if n2 is None else ISO.canonical_fit(M, n, n2=n2)
    in_family = bool(fit["residual"] < tol)
    if not in_family:
        return dict(structure="typeI_generic", rank_Sigma=fit["rank_Sigma"],
                    rank_Sigma2=fit.get("rank_Sigma2"), predicted_NC=40,
                    family_residual=fit["residual"], in_family=False,
                    M_can=fit["M_can"])
    if n2 is None:
        pred = {0: 4, 1: 12, 2: 16}.get(fit["rank_Sigma"], 16)
        return dict(structure="canonical_typeII", rank_Sigma=fit["rank_Sigma"],
                    rank_Sigma2=None, predicted_NC=pred,
                    family_residual=fit["residual"], in_family=True,
                    M_can=fit["M_can"])
    pred = 32 if (fit["rank_Sigma"] == 1 and fit["rank_Sigma2"] == 1) else 40
    return dict(structure="canonical_typeII_two_normals",
                rank_Sigma=fit["rank_Sigma"], rank_Sigma2=fit["rank_Sigma2"],
                predicted_NC=pred, family_residual=fit["residual"],
                in_family=True, M_can=fit["M_can"])


def solve_inner_auto(M, n=None, n2=None, data=None, n_gamma=2, seed=0,
                     norm_cap=1e5, multistart=64, jac_tol=1e-8):
    """Global min of v2(q)^T M v2(q) on S^3 with P2-1b structure dispatch.

    * canonical Type II (``n`` given and M in F_n): predicted N_C from
      rank(Sigma_A); rank 0 with ``data=(P,Y,w)`` uses the exact Davenport 4x4
      eigenproblem, otherwise the corrected census (+64-start cross-check);
    * two normals (``n`` and ``n2`` given and M in F_{n1,n2}): predicted 32/40;
    * otherwise: generic 40-root census.

    Returns dict(f, q, lam, f_multistart, n_finite_projective, n_real,
    n_isolated, structure, rank_Sigma, predicted_NC, census_consistent,
    fallback, exact, n_lost, time, solver).
    """
    t0 = time.perf_counter()
    M = np.asarray(M, float)
    M = 0.5 * (M + M.T)
    info = classify_structure(M, n=n, n2=n2)
    structure = info["structure"]
    predicted = info["predicted_NC"]
    davenport = None
    if structure == "canonical_typeII" and info["rank_Sigma"] == 0 and data is not None:
        fd, qd, crit = davenport_solve(*data)
        davenport = dict(f=float(fd), q=np.asarray(qd, float), n_crit=len(crit))
        f_enum, q_enum = float(fd), np.asarray(qd, float)
        lam = float("nan")
        n_proj = 4
        n_iso = 4
        max_res = float("nan")
        st = dict(n_finite_projective=4, n_finite_affine=8, n_lost=0)
        n_real = 4
        isolated = [dict(q=q_enum, lam=lam, f=f_enum, res=0.0)]
    else:
        roots, st = _census(M, n_gamma=n_gamma, seed=seed, norm_cap=norm_cap)
        iso = []
        for r in roots:
            smin = kkt_jac_min_sv(M, r["q"], r["lam"])
            iso.append((smin, r))
        isolated = [r for smin, r in iso if smin > jac_tol]
        max_res = max([r["res"] for r in isolated], default=float("inf"))
        f_enum = float(isolated[0]["f"]) if isolated else (
            float(roots[0]["f"]) if roots else float("inf"))
        q_enum = isolated[0]["q"] if isolated else (roots[0]["q"] if roots else None)
        lam = float(isolated[0]["lam"]) if isolated else float("nan")
        n_proj = int(st["n_finite_projective"])
        n_iso = len(isolated)
        n_real = int(st["n_real_geometric"])
    f_ms, q_ms = _multistart_min(M, n_starts=multistart, seed=seed)
    fallback = False
    f, q = f_enum, q_enum
    if q_ms is not None and np.isfinite(f_ms) and f_ms < f - 1e-9:
        f, q, fallback = float(f_ms), q_ms.copy(), True
    if q is not None and q[0] < 0:
        q = -q
    solver = ("davenport-4x4" if davenport is not None
              else "corrected-census+multistart")
    consistent = bool(n_proj == predicted) if structure != "typeI_generic" else None
    return dict(f=float(f), f_enum=float(f_enum),
                q=q, lam=(float(lam) if davenport is None else None),
                f_multistart=float(f_ms) if np.isfinite(f_ms) else None,
                n_finite_projective=int(n_proj), n_real=int(n_real),
                n_isolated=int(n_iso), n_lost=int(st.get("n_lost", 0)),
                max_residual=float(max_res), structure=structure,
                rank_Sigma=info["rank_Sigma"],
                rank_Sigma2=info["rank_Sigma2"],
                predicted_NC=int(predicted),
                family_residual=info["family_residual"],
                census_consistent=consistent,
                fallback=bool(fallback),
                exact=bool(consistent is not False and st.get("n_lost", 0) == 0),
                time=float(time.perf_counter() - t0), solver=solver)
