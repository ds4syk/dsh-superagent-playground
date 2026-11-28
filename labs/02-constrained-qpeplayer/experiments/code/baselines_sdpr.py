#!/usr/bin/env python
"""SDPRLayers-style baselines on the *same* constrained-QPEP instances.

This module re-implements, in NumPy/CVXPY, the forward optimisation layer of
SDPRLayers [Holmes, Duembgen, Barfoot, T-RO 2025] for the 6-DoF pose problem
studied in this lab, and reads a pose out of the SDP solution.  It is **our
re-implementation of their relaxation**, not their official PyTorch layer
(which we do not have and do not run): no backward pass, no torch, no learned
front-end.  Everything below is CPU CVXPY/SCS.

Two protocols (the paper's B1/B2):

  B1  ``sdpr_unconstrained``
      The homogeneous Shor relaxation of the *unconstrained* quaternion pose
      problem, lifted as y = [v2(q); t; 1] in R^14:
          min <Q, X>  s.t.  X >= 0,  X_13,13 = 1,  e_tr^T v2 = 1,
                            ||v2||^2 = 1,  Veronese 2x2 minors = 0
      (``certificate.shor_bound(..., enforce_ineq=False, add_veronese=True)``).
      The full rotation ideal (Veronese) is what makes the 13x13 relaxation
      equal to SDPRLayers' 10x10 one (Theorem P1-2 in the paper): without it
      the 10x10 relaxation is the t-marginal of the present one.
      Pose extraction: rank-1 leading eigenvector of X, dehomogenised by its
      last coordinate (sign is the only gauge freedom, and it is fixed by
      X_13,13 = 1), q from the v2 block, t from the t block; optional local
      refinement with ``qpep_core.solve_qp_global`` warm-started at the SDP
      pose.

  B2  ``sdpr_slack``
      The inequality-constrained form with p non-negative slacks,
      y = [v2; t; s_1..s_p; 1] in R^{14+p}, PSD block (14+p)x(14+p):
          min <Q, X> + M * sum_j s_j
          s.t.  X >= 0,  same equalities,  g_j - s_j <= 0,  s_j >= 0.
      At rank 1 the feasible set is exactly {g_j <= max(0,g_j)} with the L1
      exact-penalty objective J + M sum_j [g_j]_+, so M -> inf gives the hard
      constrained relaxation.  We also solve the literal ``g_j + s_j <= 0``
      sign (``sign="plus"``) that the task text names, and measure what it
      does: rank-1 feasible points of that form satisfy g_j <= -s_j <= 0, so
      it is *not* a penalty relaxation of the inequality, it is the hard
      constraint plus a bigger PSD block.  Both are reported.

The 14x14 Shor relaxation *with* the inequalities (``sdpr_hard``, i.e. the
M -> inf limit) is also solved here so that its pose can be extracted; its
value is cross-checked against ``certificate.shor_bound(enforce_ineq=True)``,
and the unc value against ``certificate.shor_bound(enforce_ineq=False)``.

Caveats that the result files repeat:
  * SCS is not dual feasible; a solver ``value`` is not a rigorous bound.
    Rigorous bounds are produced separately by ``certificate.shor_bound``
    (dual repair / P2-Cert Schur) and are the ones used for "certified".
  * This is a research re-implementation of SDPRLayers' forward SDP; the
    official layer, its dual-certificate backward pass and its solver
    settings (MOSEK, scaling) are not reproduced.
  * Absolute runtimes are machine-specific and used only for scaling.

Interpreter: /home/syk/miniconda3/envs/sdpr/bin/python (numpy 1.26, cvxpy 1.5.4).
"""
from __future__ import annotations

import time

import cvxpy as cp
import numpy as np

import certificate as CERT
import qpep_core as core

SQRT2 = np.sqrt(2.0)
_CROSS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]


# ---------------------------------------------------------------------------
# small algebra helpers
# ---------------------------------------------------------------------------


def _sym(M):
    return 0.5 * (M + M.T)


def q_full(M0, D0, G0, n2):
    """Objective matrix on the full lift: pose block in 0..12, zeros elsewhere."""
    Q = np.zeros((n2, n2))
    Q[:13, :13] = q13(M0, D0, G0)
    return Q


def q13(M0, D0, G0):
    """13x13 objective block in y13 = [v2; t]."""
    Q = np.zeros((13, 13))
    Q[:10, :10] = M0
    Q[:10, 10:] = D0.T
    Q[10:, :10] = D0
    Q[10:, 10:] = G0
    return Q


def _embed(M, n2):
    """Embed a 14-dim spec matrix (index 13 = homogeneous 1) into R^{n2}.

    Indices 0..12 stay, the homogeneous coordinate moves from 13 to n2-1.
    Spec matrices whose nonzeros only touch 0..9 (the Veronese ones) are
    simply copied into the top-left block by this same map.
    """
    assert M.shape == (14, 14), M.shape
    out = np.zeros((n2, n2))
    old = list(range(13)) + [n2 - 1]
    out[np.ix_(old, old)] = M
    return out


def _cvec(A, B, j, n2):
    """Coefficient vector of g_j = a_j^T v2 + b_j^T t in the lift.

    No homogeneous coefficient: g_j is homogeneous of degree 2 in y, and
    <sym(c h^T), X> = c^T X h = c^T y for rank-1 X (h selects the last column).
    """
    c = np.zeros(n2)
    c[:10] = A[j]
    c[10:13] = B[j]
    return c


def _slack_vec(j, n2, p):
    e = np.zeros(n2)
    e[13 + j] = 1.0
    return e


def _hvec(n2):
    e = np.zeros(n2)
    e[n2 - 1] = 1.0
    return e


def _gmat(c, h):
    """<sym(c h^T), X> = c^T X h  (= g_j when h is the homogeneous vector)."""
    return 0.5 * (np.outer(c, h) + np.outer(h, c))


def q_from_v2(v):
    """Recover a unit quaternion from a (possibly noisy) Veronese vector.

    v2 = [q0^2,q1^2,q2^2,q3^2, sqrt2 q0q1, sqrt2 q0q2, sqrt2 q0q3,
          sqrt2 q1q2, sqrt2 q1q3, sqrt2 q2q3].
    The pivot (largest |q_i|) is taken positive; the off-diagonal entries then
    fix the signs of the other components.  Returns None if v has no mass.
    """
    v = np.asarray(v, float)
    m = np.sqrt(np.clip(v[:4], 0.0, None))
    p = int(np.argmax(m))
    if m[p] < 1e-12:
        return None
    q = np.zeros(4)
    q[p] = m[p]
    for i in range(4):
        if i == p:
            continue
        idx = 4 + _CROSS.index((min(i, p), max(i, p)))
        q[i] = m[i] if v[idx] >= 0.0 else -m[i]
    n = np.linalg.norm(q)
    return q / n if n > 1e-12 else None


def rank_of(X, tol=1e-6):
    ev = np.linalg.eigvalsh(_sym(X))
    lam1 = float(ev[-1])
    if lam1 <= 0.0:
        return 0, lam1, float("nan")
    return int(np.sum(ev > tol * lam1)), lam1, float(ev[-2] / lam1)


# ---------------------------------------------------------------------------
# generic SDP solve (same assembly as certificate.shor_bound, but returns X)
# ---------------------------------------------------------------------------

_EQ_CACHE = {}


def _eq_specs(add_veronese=True):
    key = bool(add_veronese)
    if key not in _EQ_CACHE:
        specs, _, _ = CERT._base_eq_specs(14)
        if add_veronese:
            specs = specs + CERT._veronese_specs(14)
        _EQ_CACHE[key] = specs
    return _EQ_CACHE[key]


def solve_sdp(Q, n2, eq_specs, ineq_specs, eps=1e-9, max_iters=20000,
              solver="SCS", time_limit=None, verbose=False, retry=True):
    """min <Q,X> s.t. X>=0, <A_k,X>=b_k, <M_m,X><=0.  Returns X and diagnostics."""
    Xv = cp.Variable((n2, n2), symmetric=True)
    cons = [Xv >> 0]
    for Amat, b in eq_specs:
        cons.append(cp.trace(Amat @ Xv) == b)
    for M, _ in ineq_specs:
        cons.append(cp.trace(M @ Xv) <= 0.0)
    prob = cp.Problem(cp.Minimize(cp.trace(Q @ Xv)), cons)
    kw = dict(solver=solver)
    if solver == "SCS":
        kw.update(eps=eps, max_iters=max_iters)
    elif solver == "CLARABEL":
        kw.update(tol_gap_abs=eps, tol_gap_rel=eps, tol_feas=eps)
    if time_limit is not None:
        kw["time_limit_secs"] = time_limit
    t0 = time.perf_counter()
    try:
        prob.solve(**kw)
    except Exception as exc:                       # keep diagnostics on failure
        return dict(value=float("nan"), X=None, status="error:%s" % exc,
                    time_s=time.perf_counter() - t0, n_eq=len(eq_specs),
                    n_ineq=len(ineq_specs), cons=None, Q=Q)
    dt = time.perf_counter() - t0
    if Xv.value is None or prob.value is None:
        out = dict(value=float("nan"), X=None, status=str(prob.status),
                   time_s=dt, n_eq=len(eq_specs), n_ineq=len(ineq_specs),
                   cons=None, Q=Q, solver_used=solver)
        if not retry:
            out["attempts"] = 1
            return out
        return _retry(out, Q, n2, eq_specs, ineq_specs, eps, max_iters,
                      solver, time_limit, attempts=1)
    out = dict(value=float(prob.value), X=np.asarray(Xv.value),
               status=str(prob.status), time_s=dt, n_eq=len(eq_specs),
               n_ineq=len(ineq_specs), cons=cons, Q=Q, solver_used=solver)
    _resid(out, eq_specs, ineq_specs)
    if not retry:
        out["attempts"] = 1
        return out
    return _retry(out, Q, n2, eq_specs, ineq_specs, eps, max_iters, solver,
                  time_limit, attempts=1)


def _resid(out, eq_specs, ineq_specs):
    X = out["X"]
    out["eq_resid"] = float(max([abs(np.sum(A * X) - b) for A, b in eq_specs],
                                default=0.0))
    out["feas_ineq"] = float(max([np.sum(M * X) for M, _ in ineq_specs],
                                 default=-np.inf))
    out["valid"] = bool(out["status"].startswith("optimal")
                        and out["eq_resid"] <= 1e-6 and out["feas_ineq"] <= 1e-6)
    return out


_RETRY_LADDER = [("CLARABEL", 1.0, 1.0), ("SCS", 10.0, 3.0)]


def _retry(out, Q, n2, eq_specs, ineq_specs, eps, max_iters, solver,
           time_limit, attempts=1):
    """Re-solve an invalid SDP once with tighter SCS, then with CLARABEL.

    SCS occasionally returns `optimal_inaccurate` with an equality residual of
    1e-2 (a value above the hard-relaxation value, i.e. provably wrong).  The
    retry ladder keeps the solve honest without touching the common case.
    """
    if out.get("valid", False) or attempts >= 3:
        out["attempts"] = attempts
        return out
    good = out if out.get("valid", False) else None
    for solv, epsf, itf in _RETRY_LADDER:
        if solv == solver and epsf == 1.0:
            continue
        try:
            alt = solve_sdp(Q, n2, eq_specs, ineq_specs,
                            eps=max(eps / epsf, 1e-11),
                            max_iters=int(max_iters * itf), solver=solv,
                            time_limit=(20.0 if solv == "SCS" else None),
                            retry=False)
            _resid(alt, eq_specs, ineq_specs)
        except Exception:
            continue
        alt["attempts"] = attempts + 1
        alt["retried_from"] = out.get("status")
        if alt.get("valid", False):
            return alt
        if alt.get("eq_resid", np.inf) < out.get("eq_resid", np.inf):
            good = alt
        attempts += 1
    if good is not None:
        return good
    out["attempts"] = attempts
    return out


def _post(out, X, eq_specs, ineq_specs, n2, repair=False):
    """Fill rank/tightness/feasibility diagnostics (and optional dual repair)."""
    out["n_var"] = n2
    out["rank"], out["lam1"], out["ratio"] = rank_of(X)
    ev = np.linalg.eigvalsh(_sym(X))
    out["lam2"] = float(ev[-2])
    out["x_lam_min"] = float(ev[0])
    out["eq_resid"] = float(max([abs(np.sum(A * X) - b) for A, b in eq_specs],
                                default=0.0))
    out["feas_ineq"] = float(max([np.sum(M * X) for M, _ in ineq_specs],
                                 default=-np.inf))
    out["obj_from_X"] = float(np.sum(out["Q"] * X))
    if repair and out.get("cons") is not None:
        out.update(CERT._dual_repair(out["Q"], eq_specs, ineq_specs,
                                     out["cons"], 1e-8))
    else:
        out.update(rigorous=float("nan"), dual_method="off")
    return out


def solve_unc(M0, D0, G0, eps=1e-9, max_iters=20000, solver="SCS",
              add_veronese=True, repair=False, retry=True):
    """B1: the SDPRLayers homogeneous Shor relaxation without inequalities."""
    n2 = 14
    eq = _eq_specs(add_veronese)
    out = solve_sdp(q_full(M0, D0, G0, n2), n2, eq, [], eps=eps,
                    max_iters=max_iters, solver=solver, retry=retry)
    if out["X"] is None:
        return out
    return _post(out, out["X"], eq, [], n2, repair=repair)


def solve_hard(M0, D0, G0, A, B, eps=1e-9, max_iters=20000, solver="SCS",
               repair=True, schur=False, retry=True):
    """M -> inf limit of B2: the 14x14 relaxation *with* g_j <= 0."""
    n2 = 14
    eq = _eq_specs(True)
    h = _hvec(n2)
    ineq = [(_gmat(_cvec(A, B, j, n2), h), 0.0) for j in range(A.shape[0])]
    out = solve_sdp(q_full(M0, D0, G0, n2), n2, eq, ineq, eps=eps,
                    max_iters=max_iters, solver=solver, retry=retry)
    if out["X"] is None:
        return out
    out = _post(out, out["X"], eq, ineq, n2, repair=repair)
    if repair and out.get("cons") is not None and schur:
        out.update(CERT._schur_repair(out["Q"], eq, ineq, out["cons"], A, B))
    return out


def solve_slack(M0, D0, G0, A, B, M, sign="minus", eps=1e-9, max_iters=20000,
                solver="SCS", repair=False, retry=True):
    """B2: lifted [v2; t; s; 1], PSD (14+p), L1-slack penalty M."

    sign="minus": g_j - s_j <= 0, s_j >= 0   (a genuine penalty relaxation)
    sign="plus" : g_j + s_j <= 0, s_j >= 0   (literal task text; not a
                                               relaxation of g_j<=0 at rank 1)
    """
    p = int(A.shape[0])
    n2 = 14 + p
    h = _hvec(n2)
    eq = _eq_specs(True)
    ineq = []
    Svecs = []
    for j in range(p):
        c = _cvec(A, B, j, n2)
        Gj = _gmat(c, h)
        Sj = _gmat(_slack_vec(j, n2, p), h)
        Svecs.append(Sj)
        if sign == "minus":
            ineq.append((_sym(Gj - Sj), 0.0))
        elif sign == "plus":
            ineq.append((_sym(Gj + Sj), 0.0))
        else:
            raise ValueError("sign must be 'minus' or 'plus'")
        ineq.append((_sym(-Sj), 0.0))            # s_j >= 0
    Q = np.zeros((n2, n2))
    Q[:13, :13] = q13(M0, D0, G0)
    for j in range(p):
        Q = Q + M * _sym(np.outer(_slack_vec(j, n2, p), h))
    eq = [(_embed(Amat, n2), b) for Amat, b in eq]
    out = solve_sdp(Q, n2, eq, ineq, eps=eps, max_iters=max_iters,
                    solver=solver)
    if out["X"] is None:
        return out
    out = _post(out, out["X"], eq, ineq, n2, repair=repair)
    X = out["X"]
    out["s_val"] = [float(np.sum(Sj * X)) for Sj in Svecs]
    out["s_max"] = float(max(out["s_val"], default=0.0))
    out["n_s_pos"] = int(sum(1 for s in out["s_val"] if s > 1e-9))
    out["M"] = float(M)
    out["sign"] = sign
    out["g_sdp"] = [float(np.sum(_gmat(_cvec(A, B, j, n2), h) * X))
                    for j in range(p)]
    return out


def solve_slack_eq(M0, D0, G0, A, B, eps=1e-9, max_iters=20000, solver="SCS",
                   repair=False, retry=True):
    """B2a: SDPRLayers-compatible slack lift with *equalities*.

    y = [v2; t; s_1..s_p; 1], PSD (14+p), same Shor equalities, plus
        g_j + s_j = 0      (equality: the only constraint class SDPRLayers
                            supports)   and   s_j >= 0  (linear in X).
    At rank 1 this is exactly the hard constrained problem, so
        hard_value <= slack_eq_value <= J*
    and the inequality between the two SDP values measures how much the extra
    PSD block tightens the relaxation.  No penalty weight, hence no M.
    """
    p = int(A.shape[0])
    n2 = 14 + p
    h = _hvec(n2)
    eq = [(_embed(Amat, n2), b) for Amat, b in _eq_specs(True)]
    ineq = []
    for j in range(p):
        c = _cvec(A, B, j, n2)
        Gj = _gmat(c, h)
        Sj = _gmat(_slack_vec(j, n2, p), h)
        eq.append((_sym(Gj + Sj), 0.0))
        ineq.append((_sym(-Sj), 0.0))            # s_j >= 0
    Q = q_full(M0, D0, G0, n2)
    out = solve_sdp(Q, n2, eq, ineq, eps=eps, max_iters=max_iters,
                    solver=solver, retry=retry)
    if out["X"] is None:
        return out
    out = _post(out, out["X"], eq, ineq, n2, repair=repair)
    X = out["X"]
    out["s_val"] = [float(np.sum(_gmat(_slack_vec(j, n2, p), h) * X))
                    for j in range(p)]
    out["s_max"] = float(max(out["s_val"], default=0.0))
    out["n_s_pos"] = int(sum(1 for s in out["s_val"] if s > 1e-9))
    out["sign"] = "eq"
    return out


def solve_slack_pen(M0, D0, G0, A, B, M, eps=1e-9, max_iters=20000,
                    solver="SCS", repair=False, retry=True):
    """B2b: the penalised slack form of the task text (L1 exact penalty).

    g_j - s_j <= 0, s_j >= 0, min J + M sum_j s_j.  At rank 1 the feasible set
    is the original one with s_j = max(0,g_j), so M -> inf is the hard problem
    and the SDP value is a *lower* bound on J* for every M.  The opposite sign
    (g_j + s_j <= 0) is *not* a relaxation; see solve_slack_plus.
    """
    p = int(A.shape[0])
    n2 = 14 + p
    h = _hvec(n2)
    eq = [(_embed(Amat, n2), b) for Amat, b in _eq_specs(True)]
    ineq = []
    for j in range(p):
        c = _cvec(A, B, j, n2)
        Gj = _gmat(c, h)
        Sj = _gmat(_slack_vec(j, n2, p), h)
        ineq.append((_sym(Gj - Sj), 0.0))
        ineq.append((_sym(-Sj), 0.0))
    Q = q_full(M0, D0, G0, n2)
    for j in range(p):
        Q = Q + M * _sym(np.outer(_slack_vec(j, n2, p), h))
    out = solve_sdp(Q, n2, eq, ineq, eps=eps, max_iters=max_iters,
                    solver=solver, retry=retry)
    if out["X"] is None:
        return out
    out = _post(out, out["X"], eq, ineq, n2, repair=repair)
    X = out["X"]
    out["s_val"] = [float(np.sum(_gmat(_slack_vec(j, n2, p), h) * X))
                    for j in range(p)]
    out["s_max"] = float(max(out["s_val"], default=0.0))
    out["n_s_pos"] = int(sum(1 for s in out["s_val"] if s > 1e-9))
    out["M"] = float(M)
    out["sign"] = "minus"
    return out


def solve_slack_plus(M0, D0, G0, A, B, M, eps=1e-9, max_iters=20000,
                     solver="SCS", retry=True):
    """Literal task sign g_j + s_j <= 0 with s_j >= 0, min J + M sum s_j.

    At rank 1 this forces g_j <= -s_j <= 0 for every feasible point, so it is
    the *hard* constraint plus a bigger PSD block -- it does not relax the
    inequality and the penalty term is driven to zero.  Kept as a measured
    negative control for the sign convention.
    """
    p = int(A.shape[0])
    n2 = 14 + p
    h = _hvec(n2)
    eq = [(_embed(Amat, n2), b) for Amat, b in _eq_specs(True)]
    ineq = []
    for j in range(p):
        c = _cvec(A, B, j, n2)
        Gj = _gmat(c, h)
        Sj = _gmat(_slack_vec(j, n2, p), h)
        ineq.append((_sym(Gj + Sj), 0.0))
        ineq.append((_sym(-Sj), 0.0))
    Q = q_full(M0, D0, G0, n2)
    for j in range(p):
        Q = Q + M * _sym(np.outer(_slack_vec(j, n2, p), h))
    out = solve_sdp(Q, n2, eq, ineq, eps=eps, max_iters=max_iters,
                    solver=solver, retry=retry)
    if out["X"] is None:
        return out
    out = _post(out, out["X"], eq, ineq, n2)
    X = out["X"]
    out["s_val"] = [float(np.sum(_gmat(_slack_vec(j, n2, p), h) * X))
                    for j in range(p)]
    out["s_max"] = float(max(out["s_val"], default=0.0))
    out["n_s_pos"] = int(sum(1 for s in out["s_val"] if s > 1e-9))
    out["M"] = float(M)
    out["sign"] = "plus"
    return out


# ---------------------------------------------------------------------------
# pose extraction
# ---------------------------------------------------------------------------


def extract_pose(X, M0, D0, G0, P, tag=""):
    """Read (q, t) out of a lifted SDP solution.

    X is (n2,n2) with the homogeneous coordinate at index n2-1 and the pose
    block y = [v2 (0..9); t (10..12)].  The rank-1 leading eigenvector u is
    dehomogenised as u / u[-1] (the only sign freedom; X[-1,-1] = 1 fixes it),
    q is recovered from the v2 block by the Veronese sign rule and t is read
    from the t block.  For a non-tight X, |u[-1]| becomes small and normalising
    amplifies noise, so the first-moment column X[:, -1] is used instead and
    ``extract_rule`` records which was used.
    """
    X = np.asarray(X, float)
    n2 = X.shape[0]
    i_last = n2 - 1
    ev, U = np.linalg.eigh(_sym(X))
    u = U[:, -1].copy()
    u13 = float(u[i_last])
    lam1 = float(ev[-1])
    lam2 = float(ev[-2]) if n2 > 1 else 0.0
    rank = int(np.sum(ev > 1e-6 * max(lam1, 1e-300)))
    x_mom = X[:, i_last]

    use_eig = abs(u13) > 0.5
    if use_eig:
        v2_u = u[:10] / u13
        t_u = u[10:13] / u13
    else:
        v2_u = x_mom[:10]
        t_u = x_mom[10:13]

    out = dict(extract_rule=("eig" if use_eig else "moment"),
               lam1=lam1, lam2=lam2, ratio=lam2 / max(lam1, 1e-300),
               rank=rank, u_last=u13, x_last_norm=float(np.linalg.norm(x_mom)),
               t_sdp=[float(x) for x in t_u], tag=tag)

    # --- q from the v2 block ---------------------------------------------
    # NOTE: the 4x4 top-left block of X is E[v2_i v2_j] = E[q_i^2 q_j^2], NOT
    # E[q q^T]; its leading eigenvector is the normalised vector of squares.
    # The only correct read-out of q from this lift is sign recovery from the
    # Veronese vector (q and -q give the same R, so the gauge is irrelevant).
    q_v2 = q_from_v2(v2_u)
    if q_v2 is None:
        return None
    t_prof = -np.linalg.solve(G0, D0 @ core.v2(q_v2))
    Jbest = float(core.obj_value(q_v2, t_prof, M0, D0, G0))
    q_best = q_v2
    q_rule = "v2_sign"
    out.update(q_rule=q_rule, q=[float(x) for x in q_best],
               t_prof=[float(x) for x in t_prof], J_prof=Jbest)
    out["q_v2_sq_err"] = float(np.max(np.abs(v2_u[:4] - core.v2(q_best)[:4])))

    # --- pose used downstream: layer's own t, q from the block ------------
    R = core.R_of(q_best)
    z = (R @ np.asarray(P).T).T[:, 2] + np.asarray(t_u)[2]
    out["front_frac"] = float(np.mean(z > 0.0))
    out["depth_min_mm"] = float(1000.0 * z.min())
    out["J_sdp"] = float(core.obj_value(q_best, np.asarray(t_u), M0, D0, G0))
    out["extract_gap"] = out["J_sdp"] - Jbest   # cost of the SDP's own t block
    return out


def refine_from_pose(M0, D0, G0, q, n_starts=8, seed=0):
    """Unconstrained local refinement of the SDP pose (forward-only polish)."""
    t0 = time.perf_counter()
    J, qr, tr = core.solve_qp_global(M0, D0, G0, n_starts=n_starts,
                                     rng=np.random.default_rng(seed), q0=q)
    return dict(J=float(J), q=qr, t=tr, time_s=time.perf_counter() - t0)


# ---------------------------------------------------------------------------
# metrics (identical definitions to run_main.py / run_bop.py)
# ---------------------------------------------------------------------------


def metrics(ins, q, t, M0=None, D0=None, G0=None):
    """Synthetic-instance metrics; same formulas as run_main.metrics."""
    M0 = ins["M0"] if M0 is None else M0
    D0 = ins["D0"] if D0 is None else D0
    G0 = ins["G0"] if G0 is None else G0
    A, B = ins["A"], ins["B"]
    R = core.R_of(q)
    g = A @ core.v2(q) + B @ t
    viol = float(max(0.0, g.max()))
    return dict(J=float(core.obj_value(q, t, M0, D0, G0)),
                viol=viol, feasible=bool(viol <= 1e-6),
                pen_mm=core.penetration_mm(R, t, ins["V_dense"]),
                rot_deg=core.rot_deg(q, ins["q_gt"]),
                trans_mm=core.trans_mm(t, ins["t_gt"]),
                q=[float(x) for x in q], t=[float(x) for x in t],
                depth_min_mm=float(1000.0 * ((R @ ins["P"].T).T[:, 2]
                                             + np.asarray(t)[2]).min()))


def metrics_mesh(V, plane_n, plane_d, q, t, A, B, M0, D0, G0, q_gt, t_gt):
    """Real BOP-instance metrics; same formulas as run_bop.py (mesh + plane)."""
    R = core.R_of(q)
    xe = (R @ np.asarray(V).T).T + np.asarray(t)
    pen = max(0.0, -(xe @ np.asarray(plane_n) + float(plane_d)).min()) * 1000.0
    g = A @ core.v2(q) + B @ t
    viol = float(max(0.0, g.max()))
    return dict(J=float(core.obj_value(q, t, M0, D0, G0)),
                viol=viol, feasible=bool(viol <= 1e-6), pen_mm=float(pen),
                rot_deg=core.rot_deg(q, q_gt),
                trans_mm=core.trans_mm(t, t_gt),
                q=[float(x) for x in q], t=[float(x) for x in t],
                depth_min_mm=float(1000.0 * xe[:, 2].min()))


def wilson(k, n, z=1.959963984540054):
    """95% Wilson score interval for k successes out of n (as run_bop.py)."""
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    den = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, max(0.0, center - half), min(1.0, center + half)


def mad_iqr(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return dict(n=0, med=float("nan"), q1=float("nan"), q3=float("nan"),
                    mean=float("nan"), std=float("nan"))
    return dict(n=int(v.size), med=float(np.median(v)),
                q1=float(np.percentile(v, 25)), q3=float(np.percentile(v, 75)),
                mean=float(v.mean()), std=float(v.std()))
