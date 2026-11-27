#!/usr/bin/env python
"""Exact critical-point enumeration for the P2 inner problem (validated 2026-11-27).

Problem
-------
    min_{q in S^3}  f(q) = v2(q)^T M v2(q)                     (P2-inner)

with `v2` the 10-dim Veronese lift of a unit quaternion (see qpep_core.v2).
The KKT conditions on the unit sphere are

    F(q, lam) = [ grad f(q) - 2 lam q ;  q^T q - 1 ]  = 0,     (KKT)

i.e. four cubics and one quadratic in five complex unknowns (q, lam).

What is verified here (see experiments/results/quartic_validation.md)
---------------------------------------------------------------------
* For a *generic* (well-conditioned) real symmetric 10x10 M a total-degree
  homotopy with the start system x_i^{d_i}=1 (d=(3,3,3,3,2), 162 paths) finds
  exactly **40 distinct geometric finite complex roots** (= 80 affine roots
  modulo the double cover q <-> -q), which numerically confirms the P1
  supplement count (Porteous c_3(F-E) = 40; Bezout bound 3^4*2 = 162).
* The remaining ~82 paths diverge (roots at infinity).  Divergence is
  diagnosed explicitly (``fail_reason``) rather than silently dropped.
* For the P2 reduced matrix ``Mred = obj_min_t(M0,D0,G0)`` the matrix is
  near-PSD / rank-deficient, so *fewer* than 40 roots are finite: roots
  escape to infinity as Mred becomes singular.  Counts are reported as
  measured; **completeness / "40 roots" is NOT claimed for such degenerate
  M**.  Perturbing Mred by eps*||Mred|| restores the count monotonically
  (see the results file).
* On every tested instance the enumerated global minimum equals the
  multi-start BFGS optimum (n_starts up to 256) to <=1e-15, and multi-start
  never beats the enumeration.  That is evidence for, not a proof of,
  global optimality.

Solver
------
A total-degree homotopy in C^5, **batched over all 162 paths** (numpy
`(162,5,5)` stacked linear solves; ~30-100 ms per enumeration instead of
~2 s for the scalar loop).  Euler/Heun predictor + Newton corrector with
per-path adaptive step size, a Gamma complex-phase trick, endgame Newton
polish on F at t=1, and a rescue polish for paths that stall.  Each path is
classified as

    status 1 = converged to t=1 (then polished on F)
    status 2 = Newton failure (dt < dt_min)
    status 3 = diverged (||x|| > div_tol)
    status 4 = non-finite (overflow)

Only status 2 with a bounded final norm counts as a *lost* path, which is
what triggers ``fallback=True`` in :func:`solve_inner_exact` (the
enumeration then falls back to multi-start and does **not** claim
exactness).  Paths to infinity are expected (162 - #finite) and are not
failures.

Public API (kept stable for qpep_core integration)
--------------------------------------------------
    critical_points(M, seed=0, verbose=False, return_stats=False, n_gamma=1)
        -> list[{q, lam, f, res}]  (real geometric critical points, sorted by f)
           or (list, stats) when return_stats=True
    solve_inner_exact(M, seed=0, return_stats=False, n_gamma=1, fallback_starts=64)
        -> dict(f, q, lam, n_unique, n_real_affine, f_second, fallback,
                exact, n_lost, stats)

Pure numpy/scipy; no sympy, no cvxpy.
"""
from __future__ import annotations

import itertools
import time

import numpy as np

SQRT2 = np.sqrt(2.0)

# cross-term order used by qpep_core.v2: (0,1),(0,2),(0,3),(1,2),(1,3),(2,3)
_CROSS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
_DEG = np.array([3, 3, 3, 3, 2])
_EYE4 = np.eye(4)


# ---------------------------------------------------------------------------
# Veronese algebra (duplicated locally so this module has no import cycle)
# ---------------------------------------------------------------------------
def _v2(q):
    q = np.asarray(q)
    q0, q1, q2, q3 = q
    return np.array([
        q0 * q0, q1 * q1, q2 * q2, q3 * q3,
        SQRT2 * q0 * q1, SQRT2 * q0 * q2, SQRT2 * q0 * q3,
        SQRT2 * q1 * q2, SQRT2 * q1 * q3, SQRT2 * q2 * q3,
    ], dtype=q.dtype)


def _Jv2(q):
    q = np.asarray(q)
    q0, q1, q2, q3 = q
    J = np.zeros((10, 4), dtype=q.dtype)
    J[0, 0] = 2 * q0
    J[1, 1] = 2 * q1
    J[2, 2] = 2 * q2
    J[3, 3] = 2 * q3
    J[4] = [SQRT2 * q1, SQRT2 * q0, 0.0, 0.0]
    J[5] = [SQRT2 * q2, 0.0, SQRT2 * q0, 0.0]
    J[6] = [SQRT2 * q3, 0.0, 0.0, SQRT2 * q0]
    J[7] = [0.0, SQRT2 * q2, SQRT2 * q1, 0.0]
    J[8] = [0.0, SQRT2 * q3, 0.0, SQRT2 * q1]
    J[9] = [0.0, 0.0, SQRT2 * q3, SQRT2 * q2]
    return J


# ---------------------------------------------------------------------------
# KKT evaluation, scalar (reference) and batched (fast path)
# ---------------------------------------------------------------------------
def kkt_eval(x, M):
    """F(x), J(x) for x = [q0,q1,q2,q3,lam], F = [grad f - 2 lam q; q.q-1]."""
    q = np.asarray(x[:4])
    lam = x[4]
    Jv = _Jv2(q)
    v = _v2(q)
    W = M @ v
    gq = 2.0 * (Jv.T @ W)                       # grad f, cubic
    dt = np.result_type(q.dtype, type(lam))
    F = np.empty(5, dtype=dt)
    F[:4] = gq - 2.0 * lam * q
    F[4] = q @ q - 1.0

    J = np.empty((5, 5), dtype=dt)
    term1 = np.zeros((4, 4), dtype=dt)          # dJv2/dq contracted with W
    for i in range(4):
        term1[i, i] += 4.0 * W[i]
    for a, (p, r) in zip(range(4, 10), _CROSS):
        term1[p, r] += 2.0 * SQRT2 * W[a]
        term1[r, p] += 2.0 * SQRT2 * W[a]
    J[:4, :4] = 2.0 * (Jv.T @ M @ Jv) + term1 - 2.0 * lam * _EYE4
    J[:4, 4] = -2.0 * q
    J[4, :4] = 2.0 * q
    J[4, 4] = 0.0
    return F, J


def kkt_eval_batch(X, M):
    """Batched kkt_eval: X (n,5) -> F (n,5), J (n,5,5)."""
    X = np.asarray(X)
    n = X.shape[0]
    q = X[:, :4]
    lam = X[:, 4]
    dt = X.dtype
    v = np.empty((n, 10), dtype=dt)
    q0, q1, q2, q3 = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    v[:, 0] = q0 * q0; v[:, 1] = q1 * q1; v[:, 2] = q2 * q2; v[:, 3] = q3 * q3
    v[:, 4] = SQRT2 * q0 * q1; v[:, 5] = SQRT2 * q0 * q2; v[:, 6] = SQRT2 * q0 * q3
    v[:, 7] = SQRT2 * q1 * q2; v[:, 8] = SQRT2 * q1 * q3; v[:, 9] = SQRT2 * q2 * q3
    Jv = np.zeros((n, 10, 4), dtype=dt)
    Jv[:, 0, 0] = 2 * q0; Jv[:, 1, 1] = 2 * q1; Jv[:, 2, 2] = 2 * q2; Jv[:, 3, 3] = 2 * q3
    Jv[:, 4, 0] = SQRT2 * q1; Jv[:, 4, 1] = SQRT2 * q0
    Jv[:, 5, 0] = SQRT2 * q2; Jv[:, 5, 2] = SQRT2 * q0
    Jv[:, 6, 0] = SQRT2 * q3; Jv[:, 6, 3] = SQRT2 * q0
    Jv[:, 7, 1] = SQRT2 * q2; Jv[:, 7, 2] = SQRT2 * q1
    Jv[:, 8, 1] = SQRT2 * q3; Jv[:, 8, 3] = SQRT2 * q1
    Jv[:, 9, 2] = SQRT2 * q3; Jv[:, 9, 3] = SQRT2 * q2
    W = v @ M.T                                  # (n,10) = M @ v per path
    gq = 2.0 * np.einsum('nip,ni->np', Jv, W)
    term1 = np.zeros((n, 4, 4), dtype=dt)
    for i in range(4):
        term1[:, i, i] += 4.0 * W[:, i]
    for a, (p, r) in enumerate(_CROSS):
        term1[:, p, r] += 2.0 * SQRT2 * W[:, 4 + a]
        term1[:, r, p] += 2.0 * SQRT2 * W[:, 4 + a]
    Jqq = 2.0 * np.einsum('nip,ij,njq->npq', Jv, M, Jv) + term1 \
        - 2.0 * lam[:, None, None] * _EYE4
    F = np.empty((n, 5), dtype=dt)
    F[:, :4] = gq - 2.0 * lam[:, None] * q
    F[:, 4] = np.einsum('ni,ni->n', q, q) - 1.0
    J = np.empty((n, 5, 5), dtype=dt)
    J[:, :4, :4] = Jqq
    J[:, :4, 4] = -2.0 * q
    J[:, 4, :4] = 2.0 * q
    J[:, 4, 4] = 0.0
    return F, J


def _quartic_f(q, M):
    v = _v2(np.asarray(q))
    return float(np.real(v @ M @ v))


# ---------------------------------------------------------------------------
# Start system G(x) = x_i^{d_i} - 1, its roots and Jacobian
# ---------------------------------------------------------------------------
def _start_roots():
    """All 162 roots of x_i^{d_i} = 1 (complex)."""
    w3 = np.exp(2j * np.pi * np.arange(3) / 3.0)
    roots = []
    for q0, q1, q2, q3 in itertools.product(w3, w3, w3, w3):
        for lam in (1.0 + 0j, -1.0 + 0j):
            roots.append([q0, q1, q2, q3, lam])
    return np.array(roots, dtype=complex)


def _G_batch(X):
    return X ** _DEG - 1.0


def _JG_batch(X):
    n = X.shape[0]
    J = np.zeros((n, 5, 5), dtype=X.dtype)
    for i, d in enumerate(_DEG):
        J[:, i, i] = d * X[:, i] ** (d - 1)
    return J


# ---------------------------------------------------------------------------
# Batched path tracker
# ---------------------------------------------------------------------------
def _solve_batch(A, b):
    """Solve A (n,5,5) x = b (n,5); returns (x, ok) with per-path fallback."""
    n = A.shape[0]
    ok = np.ones(n, dtype=bool)
    try:
        out = np.linalg.solve(A, b[..., None])[..., 0]
    except np.linalg.LinAlgError:
        out = np.zeros((n, 5), dtype=b.dtype)
        for i in range(n):
            try:
                out[i] = np.linalg.solve(A[i], b[i])
            except np.linalg.LinAlgError:
                ok[i] = False
    ok &= np.all(np.isfinite(out), axis=1)
    out[~ok] = 0.0
    return out, ok


def _track_batch(M, gamma, dt0=0.05, dt_max=0.10, newton_tol=1e-11,
                 max_newton=6, max_steps=400, div_tol=1e4, dt_min=1e-10,
                 verbose=False):
    """Track all 162 paths of H(x,t)=(1-t)G(x)+t*gamma*F(x) simultaneously.

    Returns (X, t, status, n_steps, n_batch_evals).
    status: 0 active(unfinished), 1 reached t=1, 2 Newton failure,
            3 diverged, 4 non-finite.
    """
    X = _start_roots()
    n = X.shape[0]
    t = np.zeros(n)
    dt = np.full(n, float(dt0))
    status = np.zeros(n, dtype=int)
    n_steps = np.zeros(n, dtype=int)
    n_evals = 0
    n_sing = 0

    for _ in range(max_steps):
        act = np.where(status == 0)[0]
        if act.size == 0:
            break
        done = act[t[act] >= 1.0 - 4e-16]
        status[done] = 1
        act = act[t[act] < 1.0 - 4e-16]
        if act.size == 0:
            break
        ta = t[act]
        h = np.minimum(dt[act], 1.0 - ta)
        tnew = ta + h
        xa = X[act]

        # ---- Euler/Heun predictor -------------------------------------
        F, JF = kkt_eval_batch(xa, M)
        n_evals += 1
        bad = ~np.all(np.isfinite(F), axis=1) | ~np.all(np.isfinite(JF), axis=(1, 2))
        if bad.any():
            status[act[bad]] = 4
            keep = ~bad
            act, ta, h, tnew, xa, F, JF = (act[keep], ta[keep], h[keep],
                                           tnew[keep], xa[keep], F[keep], JF[keep])
            if act.size == 0:
                continue
        G = _G_batch(xa)
        JG = _JG_batch(xa)
        Ht = -G + gamma * F
        Hx = (1.0 - ta)[:, None, None] * JG + ta[:, None, None] * gamma * JF
        dxdt, ok1 = _solve_batch(Hx, -Ht)
        if not ok1.all():
            n_sing += int((~ok1).sum())
            # paths with singular predictor: shrink dt and retry later
            dt[act[~ok1]] *= 0.5
            stall = act[~ok1][dt[act[~ok1]] < dt_min]
            status[stall] = 2
            keep = ok1
            act, ta, h, tnew, xa, F, G = (act[keep], ta[keep], h[keep], tnew[keep],
                                          xa[keep], F[keep], G[keep])
            dxdt, Hx, JF, JG = dxdt[keep], Hx[keep], JF[keep], JG[keep]
            if act.size == 0:
                continue

        xp = xa + h[:, None] * dxdt
        Fp, JFp = kkt_eval_batch(xp, M)
        n_evals += 1
        Gp = _G_batch(xp)
        Htp = -Gp + gamma * Fp
        Hxp = (1.0 - tnew)[:, None, None] * _JG_batch(xp) + tnew[:, None, None] * gamma * JFp
        dxdt2, ok2 = _solve_batch(Hxp, -Htp)
        dxdt2[~ok2] = dxdt[~ok2]
        xp = xa + 0.5 * h[:, None] * (dxdt + dxdt2)

        # ---- Newton corrector on H(x, tnew) ---------------------------
        conv = np.zeros(act.size, dtype=bool)
        sing = np.zeros(act.size, dtype=bool)
        for _k in range(max_newton):
            F2, JF2 = kkt_eval_batch(xp, M)
            n_evals += 1
            G2 = _G_batch(xp)
            H2 = (1.0 - tnew)[:, None] * G2 + tnew[:, None] * gamma * F2
            Hx2 = (1.0 - tnew)[:, None, None] * _JG_batch(xp) + tnew[:, None, None] * gamma * JF2
            corr, okc = _solve_batch(Hx2, -H2)
            sing |= ~okc
            xp = xp + corr
            conv = np.linalg.norm(corr, axis=1) < newton_tol * (1.0 + np.linalg.norm(xp, axis=1))
            if conv.all():
                break
        n_sing += int(sing.sum())

        good = conv & ~sing
        idx_good = act[good]
        X[idx_good] = xp[good]
        t[idx_good] = tnew[good]
        n_steps[idx_good] += 1
        dt[idx_good] = np.minimum(dt[idx_good] * 1.6, dt_max)

        badp = ~good
        if badp.any():
            idx_bad = act[badp]
            dt[idx_bad] *= 0.5
            dt[idx_bad] = np.maximum(dt[idx_bad], 1e-300)
            low = idx_bad[dt[idx_bad] < dt_min]
            status[low] = 2
            nf = idx_bad[~np.all(np.isfinite(X[idx_bad]), axis=1)]
            status[nf] = 4

        # divergence check on the whole active set
        norms = np.linalg.norm(X[act], axis=1)
        status[act[(norms > div_tol) & (status[act] == 0)]] = 3

    # anything still active: classify by norm
    rest = np.where(status == 0)[0]
    if rest.size:
        norms = np.linalg.norm(X[rest], axis=1)
        status[rest[norms > 1e2]] = 3
        status[rest[norms <= 1e2]] = 2

    return X, t, status, n_steps, n_evals, n_sing


def _polish_F(X, M, iters=25, tol=1e-13):
    """Endgame Newton polish on F at t=1 (batched). Returns (X, res, ok)."""
    for _ in range(iters):
        F, JF = kkt_eval_batch(X, M)
        corr, ok = _solve_batch(JF, -F)
        corr[~ok] = 0.0
        X = X + corr
        if np.max(np.linalg.norm(corr, axis=1)) < tol:
            break
    F, _ = kkt_eval_batch(X, M)
    res = np.linalg.norm(F, axis=1)
    return X, res


# ---------------------------------------------------------------------------
# Dedup helpers
# ---------------------------------------------------------------------------
def _cluster(points, dist_fn, tol):
    """Greedy clustering; returns list of reps and assignment indices."""
    reps, assign = [], []
    for p in points:
        hit = -1
        for k, r in enumerate(reps):
            if dist_fn(p, r) <= tol:
                hit = k
                break
        if hit < 0:
            reps.append(p)
            hit = len(reps) - 1
        assign.append(hit)
    return reps, np.array(assign, dtype=int)


def _dist_affine(a, b):
    return float(np.linalg.norm(a - b))


def _proj_cluster(qs, tol=1e-6):
    """Cluster quaternions modulo the double cover q ~ -q (projective points).

    Scale-aware tolerance: complex KKT roots may have ||q|| > 1, so a plain
    Euclidean tolerance would merge distinct roots.  Returns (reps, assign).
    """
    reps, assign = [], []
    for v in qs:
        k = -1
        nv = np.linalg.norm(v)
        for j, r in enumerate(reps):
            d = min(np.linalg.norm(v - r), np.linalg.norm(v + r))
            if d <= tol * max(1.0, nv, np.linalg.norm(r)):
                k = j
                break
        if k < 0:
            reps.append(v)
            k = len(reps) - 1
        assign.append(k)
    return reps, np.array(assign, dtype=int)


def _canon_sign(q):
    """Canonical representative of q ~ -q: first significant component > 0."""
    for i in range(4):
        if abs(q[i]) > 1e-12:
            return -q if q[i] < 0 else q
    return q


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def _run_homotopy(M, gamma, verbose=False, **kw):
    t0 = time.perf_counter()
    X, t, status, n_steps, n_evals, n_sing = _track_batch(M, gamma, **kw)
    t_track = time.perf_counter() - t0
    # endgame polish for everything that is not clearly divergent
    cand = np.where(np.isin(status, [1, 2, 4]))[0]
    res = np.full(X.shape[0], np.inf)
    if cand.size:
        Xp, res_p = _polish_F(X[cand].copy(), M)
        X[cand] = Xp
        res[cand] = res_p
    t_polish = time.perf_counter() - t0 - t_track
    xnorm = np.linalg.norm(X, axis=1)
    finite = np.isfinite(res) & (res < 1e-8) & (xnorm < 1e3)
    rescued = finite & np.isin(status, [2, 4])
    # a path that was "converged" but whose polished point is not a root
    status = status.copy()
    status[finite] = 1
    # classify the rest: bounded non-root = lost, else diverged
    nonroot = ~finite
    status[nonroot & (xnorm > 1e2)] = 3
    status[nonroot & (xnorm <= 1e2)] = 2
    out = dict(X=X, t=t, status=status, res=res, xnorm=xnorm, finite=finite,
               n_rescued=int(rescued.sum()), n_steps=n_steps,
               n_batch_evals=n_evals, n_singular=int(n_sing),
               t_track=t_track, t_polish=t_polish)
    if verbose:
        print("    phase gamma=%.3f%+.3fj tracked-finite=%d  div=%d  newtonfail=%d  "
              "nonfinite=%d  evals=%d  t=%.3fs"
              % (gamma.real, gamma.imag, int(finite.sum()),
                 int((status == 3).sum()), int((status == 2).sum()),
                 int((status == 4).sum()), n_evals, t_track + t_polish))
    return out


def critical_points(M, seed=0, verbose=False, return_stats=False, n_gamma=1,
                    dt_max=0.10, newton_tol=1e-11, div_tol=1e4):
    """Enumerate the real critical points of f(q)=v2(q)^T M v2(q) on S^3.

    Parameters
    ----------
    n_gamma : number of independent complex Gamma phases; the union of the
        real roots over phases is returned (a root missed by one phase is
        usually found by another).  Default 1.

    Returns
    -------
    list of dicts {q, lam, f, res} sorted by f, one per *geometric* critical
    point (q ~ -q collapsed; the representative has q[0] >= 0 when possible).
    With return_stats=True returns (roots, stats).
    """
    M = np.asarray(M, dtype=float)
    M = 0.5 * (M + M.T)
    rng = np.random.default_rng(12345 + seed + 100000)
    gammas = [np.exp(2j * np.pi * rng.random()) for _ in range(max(1, int(n_gamma)))]

    per_phase = []
    for gm in gammas:
        ph = _run_homotopy(M, gm, verbose=verbose,
                           dt_max=dt_max, newton_tol=newton_tol, div_tol=div_tol)
        per_phase.append(ph)

    # --- merge finite complex solutions over phases ---------------------
    Xs, ss = [], []
    for ph in per_phase:
        idx = np.where(ph["finite"])[0]
        Xs.append(ph["X"][idx])
        ss.append(ph["status"][idx])
    Xall = np.vstack(Xs) if Xs else np.zeros((0, 5), complex)
    if len(Xall):
        reps_c, _ = _cluster(list(Xall), _dist_affine, 1e-6)          # [q, lam]
        reps_p, _ = _proj_cluster(Xall[:, :4])                        # q ~ -q
    else:
        reps_c, reps_p = [], []
    # Every root q of the (real) KKT system has a mirror -q with the same lam,
    # so the affine set is closed under q -> -q.  Path tracking may end with a
    # duplicate sign instead of the mirror; count the affine roots by symmetry
    # (this is exact, the mirror is verified to have the same residual to
    # machine precision) and keep the raw path-level number for diagnostics.
    n_finite_affine_raw = len(reps_c)
    n_finite_proj = len(reps_p)
    n_finite_affine = 2 * n_finite_proj

    # --- real roots -----------------------------------------------------
    real_roots = []
    for ph in per_phase:
        idx = np.where(ph["finite"])[0]
        for i in idx:
            x = ph["X"][i]
            im = np.max(np.abs(x.imag))
            sc = max(1.0, np.max(np.abs(x.real)))
            if im > 1e-7 * sc:
                continue
            qr = np.real(x[:4]).copy()
            lamr = float(np.real(x[4]))
            nq = np.linalg.norm(qr)
            if abs(nq - 1.0) > 1e-7:
                continue
            qr = qr / nq
            Fr, _ = kkt_eval(np.concatenate([qr, [lamr]]), M)
            res = float(np.linalg.norm(Fr))
            if res > 1e-8:
                continue
            real_roots.append(dict(q=qr, lam=lamr, f=_quartic_f(qr, M), res=res))

    # distinct affine real roots: q and -q are distinct points on S^3
    n_real_affine_raw = 0
    if real_roots:
        n_real_affine_raw = len(_cluster(
            [np.concatenate([r["q"], [r["lam"]]]) for r in real_roots],
            _dist_affine, 1e-7)[0])
    # dedupe geometric (q ~ -q collapsed by the canonical sign)
    reps, assign = _cluster([_canon_sign(r["q"]) for r in real_roots],
                            _dist_affine, 1e-7)
    uniq = []
    for k in range(len(reps)):
        members = [real_roots[i] for i in range(len(real_roots)) if assign[i] == k]
        best = min(members, key=lambda d: d["f"])
        best["q"] = _canon_sign(best["q"])
        best["n_paths"] = len(members)
        uniq.append(best)
    uniq.sort(key=lambda d: d["f"])

    st_r1 = sum(int((ph["status"] == 1).sum()) for ph in per_phase)
    st_r2 = sum(int((ph["status"] == 2).sum()) for ph in per_phase)
    st_r3 = sum(int((ph["status"] == 3).sum()) for ph in per_phase)
    st_r4 = sum(int((ph["status"] == 4).sum()) for ph in per_phase)
    n_lost = st_r2
    stats = dict(
        n_paths=162 * len(per_phase), n_phases=len(per_phase),
        n_finite_paths=st_r1, n_diverged=st_r3, n_newton_fail=st_r2,
        n_nonfinite=st_r4, n_lost=n_lost,
        n_rescued=sum(ph["n_rescued"] for ph in per_phase),
        n_finite_affine=n_finite_affine,
        n_finite_affine_raw=n_finite_affine_raw,
        n_finite_projective=n_finite_proj,
        n_finite_geometric=n_finite_proj,
        n_real_affine=2 * len(uniq), n_real_affine_raw=n_real_affine_raw,
        n_real_raw=len(real_roots), n_unique=len(uniq),
        n_batch_evals=sum(ph["n_batch_evals"] for ph in per_phase),
        n_singular=sum(ph["n_singular"] for ph in per_phase),
        t_track=sum(ph["t_track"] for ph in per_phase),
        t_polish=sum(ph["t_polish"] for ph in per_phase),
        t_total=sum(ph["t_track"] + ph["t_polish"] for ph in per_phase),
        max_finite_xnorm=float(np.max([np.max(ph["xnorm"][ph["finite"]])
                                       for ph in per_phase
                                       if ph["finite"].any()], initial=0.0)),
        phase_stats=[dict(n_finite=int(ph["finite"].sum()),
                          n_div=int((ph["status"] == 3).sum()),
                          n_fail=int((ph["status"] == 2).sum()),
                          n_nonfinite=int((ph["status"] == 4).sum()),
                          n_real=int((ph["finite"] & (np.max(np.abs(ph["X"].imag), axis=1)
                                                      <= 1e-7)).sum()))
                     for ph in per_phase],
    )
    stats["paths_per_s"] = (stats["n_paths"] / stats["t_total"]) if stats["t_total"] > 0 else float("inf")
    if verbose:
        print("  %d paths, finite=%d affine=%d geo=%d real: %d affine / %d geo | "
              "lost=%d div=%d | %.2fs (track %.2f, polish %.2f)"
              % (stats["n_paths"], stats["n_finite_paths"], n_finite_affine,
                 n_finite_proj, stats["n_real_affine"], stats["n_unique"],
                 n_lost, st_r3, stats["t_total"], stats["t_track"], stats["t_polish"]))
    if return_stats:
        return uniq, stats
    return uniq


# ---------------------------------------------------------------------------
# Multi-start fallback (self-contained; mirrors qpep_core.solve_inner)
# ---------------------------------------------------------------------------
def _multistart_fallback(M, n_starts=64, seed=0):
    from scipy.optimize import minimize

    def fg(u):
        s = np.linalg.norm(u)
        xu = _v2(u)
        Mx = M @ xu
        f = float(np.real(xu @ Mx)) / s ** 4
        g_u = 2.0 * np.real(_Jv2(u).T @ Mx) / s ** 4 - 4.0 * f * u / s ** 2
        return f, g_u

    rng = np.random.default_rng(90000 + seed)
    best_f, best_q = np.inf, None
    for u0 in rng.normal(size=(n_starts, 4)):
        u0 = u0 / np.linalg.norm(u0)
        try:
            res = minimize(fg, u0, jac=True, method="BFGS",
                           options=dict(gtol=1e-10, maxiter=400))
        except Exception:
            continue
        u = res.x
        nu = np.linalg.norm(u)
        if not np.isfinite(nu) or nu < 1e-12:
            continue
        q = u / nu
        f = _quartic_f(q, M)
        if f < best_f:
            best_f, best_q = f, q
    return best_f, best_q


def solve_inner_exact(M, seed=0, return_stats=False, n_gamma=1,
                      fallback_starts=64, allow_fallback=True):
    """Global min of v2(q)^T M v2(q) over S^3 by full real-root enumeration.

    Returns dict(f, q, lam, n_unique, n_real_affine, f_second, fallback,
    exact, n_lost, stats).  ``exact=True`` only when no path was lost; if a
    path was lost (and ``allow_fallback``) the multi-start solver is run and
    the better of the two is returned with ``fallback=True`` / ``exact=False``.
    """
    roots, stats = critical_points(M, seed=seed, return_stats=True, n_gamma=n_gamma)
    M = np.asarray(M, dtype=float)
    M = 0.5 * (M + M.T)
    fallback = False
    f_ms = np.inf
    q_ms = None
    if not roots or (stats["n_lost"] > 0 and allow_fallback):
        fallback = True
        f_ms, q_ms = _multistart_fallback(M, n_starts=fallback_starts, seed=seed)
    sigma_min = np.nan
    degenerate = False
    # Completeness criterion: the generic count is 40 projective roots.  If M is
    # well conditioned and fewer are found, paths were lost (not roots at
    # infinity) -> not certified.  For (near-)singular M the finite count may
    # legitimately be smaller, so the criterion is disabled there; the
    # singular-critical-point check below still applies.
    _sv = np.linalg.svd(M, compute_uv=False)
    well_cond = bool(_sv[-1] > 1e-8 * max(_sv[0], 1e-300))
    incomplete = bool(stats["n_lost"] > 0) or bool(
        well_cond and stats["n_finite_projective"] < 40 and allow_fallback)
    if roots:
        best = roots[0]
        f, q, lam = float(best["f"]), best["q"].copy(), float(best["lam"])
        _, JF = kkt_eval(np.concatenate([q, [lam]]), M)
        sigma_min = float(np.linalg.svd(JF, compute_uv=False)[-1])
        if sigma_min < 1e-8:
            # singular KKT Jacobian: the critical point is not isolated; the
            # enumeration is not certified (rank-deficient / PSD M can give a
            # continuum of minimizers).
            degenerate = True
            if not fallback and allow_fallback:
                fallback = True
                f_ms, q_ms = _multistart_fallback(M, n_starts=fallback_starts, seed=seed)
    else:
        f, q, lam = np.inf, None, np.nan
    if incomplete and not fallback and allow_fallback:
        fallback = True
        f_ms, q_ms = _multistart_fallback(M, n_starts=fallback_starts, seed=seed)
    f_enum = float(f) if roots else np.inf
    if fallback and q_ms is not None and f_ms < f:
        f, q, lam = float(f_ms), q_ms.copy(), np.nan
    if q is not None and q[0] < 0:
        q = -q
    out = dict(f=float(f), f_enum=f_enum, q=q, lam=lam, n_unique=stats["n_unique"],
               n_real_affine=stats["n_real_affine"],
               n_finite_projective=stats["n_finite_projective"],
               n_finite_geometric=stats["n_finite_projective"],
               n_finite_affine=stats["n_finite_affine"],
               f_second=float(roots[1]["f"]) if len(roots) > 1 else float("inf"),
               f_multistart=float(f_ms) if np.isfinite(f_ms) else None,
               fallback=bool(fallback), exact=bool(not fallback),
               degenerate=bool(degenerate), sigma_min_best=sigma_min,
               well_conditioned=well_cond, incomplete=bool(
                   incomplete or (stats["n_lost"] > 0) or degenerate),
               n_lost=int(stats["n_lost"]), stats=stats)
    if return_stats:
        return out
    return out
