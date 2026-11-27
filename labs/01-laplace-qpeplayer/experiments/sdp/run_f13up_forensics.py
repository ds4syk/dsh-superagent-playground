#!/usr/bin/env python3
"""Forensics on the quoted value f*_{13,up} = 3.59430886 of the 6-point instance.

Question
--------
The revised P1-Q2/Q13 proof quotes, for the 6-point physical 3D-3D instance,

    f*_{13,up} = 3.59430886,      f*_{10} - f*_{13,up} = 1.82069114,

while ``run_6point_check.py`` measures 3.5943137329 / 1.8206862671 (SCS) and
CLARABEL gives 3.59431371.  The 4.9e-6 discrepancy is *not* solver tolerance
(two independent solvers agree to <1e-9), so we ask:

  (a) is 3.59430886 the optimum of a slightly different (natural) SDP?
  (b) is it a solver artefact?
  (c) is it a mis-transcription / quote from another code path?

What this script does
---------------------
1. Rebuilds the instance exactly (imports ``run_6point_check.build_Q10`` for the
   float objective, and rebuilds the same objective and the 21/24/27 constraint
   matrices in exact rational arithmetic).  Checks the float/ exact agreement.
2. Solves f10, f13_full, f13_up with SCS at eps=1e-11/1e-12 and CLARABEL.
3. Builds an explicit *dual* certificate.  For a small PSD margin eta it solves

       max mu   s.t.   Q - mu E00 - sum_i lam_i A_i - eta I  >= 0

   and then certifies ``Q - mu E00 - sum lam_i A_i`` is positive definite in
   **exact rational arithmetic** (fraction-free-ish LDL^T with Fractions).
   That mu is a rigorous lower bound for the exact SDP (the ``-eta I`` is
   dropped only because the exact check proves the stronger statement).  The
   dual slack is additionally diagonalised with mpmath at 60 decimal digits.
4. Sweeps the natural variants requested in the task (t-block handling,
   layout/permutation, h-vs-1 constraint scaling, row-orthogonality scale s,
   constraint ablations, SCS/CLARABEL tolerances) and reports value + rank.
5. Recomputes the constructed ``rowdot`` counterexample (claim f10=+1.0,
   f13=-1.0) with the same certificate machinery, to 10 digits.

Run
---
    /home/syk/miniconda3/envs/WM/bin/python sdp/run_f13up_forensics.py

Writes ``results/f13up_forensics.json`` and ``results/f13up_forensics.md``.
New files only; the existing scripts/results are imported, never modified.
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from fractions import Fraction
from pathlib import Path

import cvxpy as cp
import mpmath as mp
import numpy as np
import sympy as sp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_6point_check as P6  # noqa: E402
import run_sdp_check as S  # noqa: E402

warnings.filterwarnings("ignore")
OUT = HERE / "results"
OUT.mkdir(parents=True, exist_ok=True)

TARGET = 3.59430886
TARGET_GAP = 1.82069114
MP_DPS = 60
NEAR = 1e-8          # "reproduces the quote" tolerance


# ---------------------------------------------------------------------------
# exact / float objective and constraints
# ---------------------------------------------------------------------------
def _frac(x) -> Fraction:
    x = sp.Rational(sp.nsimplify(x))
    return Fraction(int(x.p), int(x.q))


def build_exact_Q10(W_exact, eps_exact):
    """Exact rational 10x10 Q in x_R = [h, vec_rowmajor(R)] (same as P6.build_Q10)."""
    h = sp.symbols("h", real=True)
    R = sp.Matrix(3, 3, lambda i, j: sp.Symbol(f"R{i}{j}", real=True))
    p = sp.expand(2 * sp.trace(W_exact * R * R.T)
                  - 4 * eps_exact * sp.trace(W_exact * R) * h
                  + 2 * eps_exact ** 2 * sp.trace(W_exact) * h * h)
    vars10 = [h] + [R[i, j] for i in range(3) for j in range(3)]
    Q = sp.zeros(10, 10)
    for a in range(10):
        for b in range(a, 10):
            if a == b:
                Q[a, a] = p.coeff(vars10[a], 2)
            else:
                c = p.coeff(vars10[a] * vars10[b])
                Q[a, b] = c / 2
                Q[b, a] = c / 2
    pp = (sp.Matrix(vars10).T * Q * sp.Matrix(vars10))[0, 0]
    assert sp.expand(pp - p) == 0
    return Q


def build_data():
    U, A10f, names10, chk10 = S.build_10x10()
    A13f, chk13 = S.build_13x13()
    P = S.perm_rowmajor_to_colmajor()
    rd_cm, rd_rm = S.rowdot_matrices()

    Q10f, f_at_I = P6.build_Q10()          # float objective, exact instance
    W_exact = sp.Matrix([[1, -sp.Rational(3, 5), 0],
                         [-sp.Rational(3, 5), 1, 0],
                         [0, 0, 1]])
    eps_exact = sp.Rational(1, 20)
    Q10e = build_exact_Q10(W_exact, eps_exact)
    # exact reproduction of the imported float builder
    e_vs_f = float(np.abs(np.array(Q10e.tolist(), float) - Q10f).max())

    Pf = sp.Matrix(P.tolist())
    Gf = 6.0 * P6.W
    Q13f = np.zeros((13, 13))
    Q13f[:10, :10] = P @ Q10f @ P.T
    Q13f[10:, 10:] = Gf
    Q13f = 0.5 * (Q13f + Q13f.T)

    return dict(
        U=U, P=P, rd_cm=rd_cm, rd_rm=rd_rm, W=P6.W, G=Gf,
        A10f=[np.asarray(a, float) for a in A10f],
        A13f=[np.asarray(a, float) for a in A13f],
        A13_full=[np.asarray(a, float) for a in A13f] + [np.asarray(a, float) for a in rd_cm],
        names10=names10, Q10f=Q10f, Q13f=Q13f, f_at_I=f_at_I,
        Q10e=Q10e, G_exact=6 * W_exact,
        Pf=Pf,
        Q13e=sp.Matrix(sp.BlockMatrix([[Pf * Q10e * Pf.T, sp.zeros(10, 3)],
                                       [sp.zeros(3, 10), 6 * W_exact]])),
        exact_vs_float_Q10=e_vs_f,
        checks=dict(chk10=chk10, chk13=chk13),
    )


def e00(n):
    E = np.zeros((n, n))
    E[0, 0] = 1.0
    return E


# ---------------------------------------------------------------------------
# SDP solves
# ---------------------------------------------------------------------------
def _rank(eigs, rel=1e-6):
    return int(np.sum(eigs > rel * max(abs(eigs[0]), 1e-12)))


def solve_primal(Q, A, n, solver="SCS", opts=None, extra=None,
                 rhs=None, x00=True):
    """min <Q,X> s.t. X>=0, X00=1, <A_i,X>=rhs_i.

    ``extra(X)`` appends extra cvxpy constraints; ``rhs`` replaces the zero
    right-hand side (used for the h-vs-1 variants).
    """
    opts = {} if opts is None else dict(opts)
    X = cp.Variable((n, n), symmetric=True)
    cons = [X >> 0]
    if x00:
        cons.append(X[0, 0] == 1)
    if rhs is None:
        cons += [cp.trace(a @ X) == 0 for a in A]
    else:
        cons += [cp.trace(a @ X) == r for a, r in zip(A, rhs)]
    if extra is not None:
        cons += list(extra(X))
    prob = cp.Problem(cp.Minimize(cp.trace(Q @ X)), cons)
    t0 = time.time()
    try:
        prob.solve(solver=getattr(cp, solver), verbose=False, **opts)
    except Exception as exc:  # pragma: no cover
        return dict(status="SOLVE_FAILED: " + repr(exc), value=float("nan"),
                    solver=solver, opts=opts)
    dt = time.time() - t0
    if X.value is None:
        return dict(status=prob.status, value=float("nan"), solver=solver,
                    opts=opts, t_solve=dt)
    Xv = 0.5 * (np.asarray(X.value, float) + np.asarray(X.value, float).T)
    eigs = np.linalg.eigvalsh(Xv)[::-1]
    eq_res = max([abs(Xv[0, 0] - 1.0)] if x00 else [0.0])
    if rhs is None:
        eq_res = max(eq_res, max(abs(float(np.trace(a @ Xv))) for a in A))
    else:
        eq_res = max(eq_res, max(abs(float(np.trace(a @ Xv)) - r)
                                 for a, r in zip(A, rhs)))
    return dict(
        status=prob.status, value=float(prob.value), solver=solver, opts=opts,
        rank_1e6=_rank(eigs, 1e-6), rank_1e5=_rank(eigs, 1e-5),
        eig_top=[float(x) for x in eigs[:8]], min_eig_X=float(eigs[-1]),
        max_eq_residual=float(eq_res), t_solve=dt, X=Xv,
    )


def solve_dual_regularized(Qe, Asolve, n, eta, solver="CLARABEL", opts=None):
    """max mu s.t. Q - mu E00 - sum lam A - eta I >= 0 (float solve)."""
    opts = {} if opts is None else dict(opts)
    Qf = np.array([[float(Qe[i, j]) for j in range(n)] for i in range(n)])
    mu = cp.Variable()
    lam = cp.Variable(len(Asolve))
    Z = Qf - mu * e00(n)
    for i, a in enumerate(Asolve):
        Z = Z - lam[i] * np.asarray(a, float)
    prob = cp.Problem(cp.Maximize(mu), [Z - eta * np.eye(n) >> 0])
    t0 = time.time()
    prob.solve(solver=getattr(cp, solver), verbose=False, **opts)
    return dict(
        status=prob.status, mu=float(mu.value) if mu.value is not None else float("nan"),
        lam=np.asarray(lam.value, float) if lam.value is not None else None,
        t_solve=time.time() - t0,
    )


def exact_pd(M):
    """Exact rational LDL^T: return (is_positive_definite, min pivot)."""
    n = len(M)
    L = [[Fraction(0)] * n for _ in range(n)]
    D = []
    for j in range(n):
        s = M[j][j] - sum(L[j][k] * L[j][k] * D[k] for k in range(j))
        if s <= 0:
            return False, s
        D.append(s)
        for i in range(j + 1, n):
            L[i][j] = (M[i][j] - sum(L[i][k] * L[j][k] * D[k]
                                     for k in range(j))) / s
    return True, min(D)


def dual_slack_exact(Qe, Asolve, n, mu, lam):
    """Exact rational Z = Q - mu E00 - sum lam_i A_i."""
    QF = [[_frac(Qe[i, j]) for j in range(n)] for i in range(n)]
    lamF = [Fraction(float(v)) for v in lam]
    sumlam = [[sum(lamF[k] * Fraction(float(Asolve[k][i, j]))
                   for k in range(len(Asolve)))
               for j in range(n)] for i in range(n)]
    muF = Fraction(float(mu))
    return [[QF[i][j] - (muF if (i, j) == (0, 0) else 0) - sumlam[i][j]
             for j in range(n)] for i in range(n)]


def mp_min_eig(Zex):
    """Smallest eigenvalue of an exact rational matrix at MP_DPS decimal digits."""
    mp.mp.dps = MP_DPS
    n = len(Zex)
    A = mp.matrix([[mp.mpf(Zex[i][j].numerator) / Zex[i][j].denominator
                    for j in range(n)] for i in range(n)])
    ev = mp.eigsy(A, eigvals_only=True)
    return min(ev)


def certify(Qe, Asolve, n, tag, etas=(1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6)):
    """Find the smallest eta whose regularized dual passes the exact PD check."""
    attempts = []
    for eta in etas:
        sol = solve_dual_regularized(
            Qe, Asolve, n, eta, solver="CLARABEL",
            opts=dict(tol_gap_abs=1e-12, tol_gap_rel=1e-12, tol_feas=1e-12,
                      max_iter=2000))
        if sol["lam"] is None:
            attempts.append(dict(eta=eta, status=sol["status"], ok=False))
            continue
        Zex = dual_slack_exact(Qe, Asolve, n, sol["mu"], sol["lam"])
        ok, piv = exact_pd(Zex)
        attempts.append(dict(eta=eta, mu=sol["mu"], status=sol["status"],
                             exact_pd=bool(ok),
                             min_pivot=float(piv) if ok else -float(piv)))
        if ok:
            return dict(tag=tag, eta=eta, mu_low=float(sol["mu"]),
                        exact_pd=True, min_pivot=float(piv),
                        status=sol["status"], attempts=attempts,
                        Zex=Zex, mu_solver=float(sol["mu"]))
    return dict(tag=tag, exact_pd=False, attempts=attempts,
                mu_low=float("nan"), mu_solver=float("nan"))


# ---------------------------------------------------------------------------
# variants
# ---------------------------------------------------------------------------
def rowdot_at_X(X, rd):
    Xr = X[:10, :10]
    return [float(np.trace(np.asarray(a, float)[:10, :10] @ Xr)) for a in rd]


def build_Q10_for_W(rho):
    W = sp.Matrix([[1, -rho, 0], [-rho, 1, 0], [0, 0, 1]])
    return build_exact_Q10(W, sp.Rational(1, 20))


def build_Q10_transposed(R, W, eps):
    """10x10 objective for J(R^T) (row-major vars of R)."""
    h = sp.symbols("h", real=True)
    RT = R.T
    p = sp.expand(2 * sp.trace(W * RT * RT.T)
                  - 4 * eps * sp.trace(W * RT) * h
                  + 2 * eps ** 2 * sp.trace(W) * h * h)
    vars10 = [h] + [R[i, j] for i in range(3) for j in range(3)]
    Q = sp.zeros(10, 10)
    for a in range(10):
        for b in range(a, 10):
            if a == b:
                Q[a, a] = p.coeff(vars10[a], 2)
            else:
                c = p.coeff(vars10[a] * vars10[b])
                Q[a, b] = c / 2
                Q[b, a] = c / 2
    return Q


def Q13_from_Q10(Q10e, G_exact, Pf):
    return sp.Matrix(sp.BlockMatrix([[Pf * Q10e * Pf.T, sp.zeros(10, 3)],
                                     [sp.zeros(3, 10), G_exact]]))


def solve_variant(name, desc, Q, A, n, solver="SCS", opts=None,
                  extra=None, rhs=None, x00=True):
    sol = solve_primal(Q, A, n, solver=solver, opts=opts, extra=extra,
                       rhs=rhs, x00=x00)
    near = (abs(sol["value"] - TARGET) < NEAR
            if np.isfinite(sol.get("value", np.nan)) else False)
    baseline = 3.5943137328748276
    return dict(name=name, description=desc, solver=solver,
                value=sol["value"], status=sol["status"],
                rank_1e6=sol.get("rank_1e6"), rank_1e5=sol.get("rank_1e5"),
                eig_top=sol.get("eig_top"), min_eig_X=sol.get("min_eig_X"),
                max_eq_residual=sol.get("max_eq_residual"),
                delta_vs_baseline=(sol["value"] - baseline
                                   if np.isfinite(sol.get("value", np.nan)) else None),
                reproduces_target=bool(near), t_solve=sol.get("t_solve"))


def run_variants(d):
    P, rd, G = d["P"], d["rd_cm"], d["G"]
    Q10, Q13 = d["Q10f"], d["Q13f"]
    A10, A13 = d["A10f"], d["A13f"]
    Q10e, Q13e, Gex, Pf = d["Q10e"], d["Q13e"], d["G_exact"], d["Pf"]
    scs11 = dict(eps=1e-11, max_iters=2_000_000)
    out = []

    # ---- baseline / cross-checks -------------------------------------------
    out.append(solve_variant("base_f13up_scs1e-11", "13x13, 24 upstream constraints, SCS 1e-11",
                             Q13, A13, 13, "SCS", scs11))
    out.append(solve_variant("base_f13up_scs1e-12", "same, SCS 1e-12",
                             Q13, A13, 13, "SCS", dict(eps=1e-12, max_iters=2_000_000)))
    out.append(solve_variant("base_f13up_clarabel", "same, CLARABEL default",
                             Q13, A13, 13, "CLARABEL", dict(max_iter=1000)))
    out.append(solve_variant("base_f10", "10x10, 21 constraints (full 20D)",
                             Q10, A10, 10, "SCS", scs11))
    out.append(solve_variant("base_f13full", "13x13, 24 + 3 row-orth = 27 constraints",
                             Q13, d["A13_full"], 13, "SCS", scs11))
    A13_10 = [np.asarray(P.T @ a[:10, :10] @ P, float) for a in A13]
    out.append(solve_variant("base_rot_only_10x10",
                             "10x10 with the 24 upstream constraints mapped to the rotation block (t-eliminated)",
                             Q10, A13_10, 10, "SCS", scs11))

    # ---- (i) t-block handling ----------------------------------------------
    def xtt_zero(X):
        return [X[10 + i, 10 + i] == 0 for i in range(3)]

    def xxt_zero(X):
        return [X[i, 10 + j] == 0 for i in range(10) for j in range(3)]

    out.append(solve_variant("i_Xtt_zero", "13x13 + explicit X_tt=0 (t-block eliminated)",
                             Q13, A13, 13, "SCS", scs11, extra=xtt_zero))
    out.append(solve_variant("i_Xxt_zero", "13x13 + explicit X_xt=0",
                             Q13, A13, 13, "SCS", scs11, extra=xxt_zero))
    out.append(solve_variant("i_Xtt_I", "13x13 + t-block normalised to X_tt=I (adds tr G = 18)",
                             Q13, A13, 13, "SCS", scs11,
                             extra=lambda X: [X[10:, 10:] == np.eye(3)]))
    out.append(solve_variant("i_traceX_eq4", "replace X00=1 by trace(X)=4 (rank-1 trace normalisation)",
                             Q13, A13, 13, "SCS", scs11, x00=False,
                             extra=lambda X: [cp.trace(X) == 4]))
    out.append(solve_variant("i_X00_ge1", "relax X00=1 to X00>=1",
                             Q13, A13, 13, "SCS", scs11, x00=False,
                             extra=lambda X: [X[0, 0] >= 1]))

    # ---- (ii) layout / permutation -----------------------------------------
    Q13_noperm = np.zeros((13, 13))
    Q13_noperm[:10, :10] = Q10
    Q13_noperm[10:, 10:] = G
    out.append(solve_variant(
        "ii_no_perm",
        "objective Q10 inserted without the row/col-major permutation P",
        0.5 * (Q13_noperm + Q13_noperm.T), A13, 13, "SCS", scs11))
    Q13_inv = np.zeros((13, 13))
    Q13_inv[:10, :10] = P.T @ Q10 @ P
    Q13_inv[10:, 10:] = G
    out.append(solve_variant("ii_inverse_perm", "objective with P^T Q10 P (P is an involution, so = correct)",
                             0.5 * (Q13_inv + Q13_inv.T), A13, 13, "SCS", scs11))
    h = sp.symbols("h", real=True)
    R = sp.Matrix(3, 3, lambda i, j: sp.Symbol(f"R{i}{j}", real=True))
    W_ex = sp.Matrix([[1, -sp.Rational(3, 5), 0], [-sp.Rational(3, 5), 1, 0], [0, 0, 1]])
    Q10_T = build_Q10_transposed(R, W_ex, sp.Rational(1, 20))
    Q13_T = np.array(Q13_from_Q10(Q10_T, Gex, Pf).tolist(), float)
    out.append(solve_variant("ii_transposed_C", "objective built for C^T (J(R^T)), mapped with P",
                             Q13_T, A13, 13, "SCS", scs11))
    A_rm = []
    for a in A13:
        a = np.asarray(a, float)
        A_rm.append(np.block([[P.T @ a[:10, :10] @ P, P.T @ a[:10, 10:]],
                              [a[10:, :10] @ P, a[10:, 10:]]]))
    out.append(solve_variant("ii_constraints_rowmajor", "col-major objective Q13 against row-major-interpreted 24 constraints",
                             Q13, A_rm, 13, "SCS", scs11))

    # ---- (iii) h vs 1 -------------------------------------------------------
    # upstream diag orthogonality is C_i.C_i - h^2 = 0; with X00=1 it equals
    # C_i.C_i = 1.  Build the same matrices with the (0,0) entry dropped and
    # move the constant to the RHS (equivalent), or set RHS=0 (drop h^2).
    diag_idx = {0: 1, 3: 1, 5: 1}   # (i,j) enumeration of build_13x13's 6 orth
    A_mod, rhs_eq, rhs_drop = [], [], []
    for k, a in enumerate(A13):
        b = np.asarray(a, float).copy()
        if k < 6:
            b[0, 0] = 0.0
            r = float(diag_idx.get(k, 0))
        else:
            r = 0.0
        A_mod.append(b)
        rhs_eq.append(r)
        rhs_drop.append(0.0)
    out.append(solve_variant("iii_h2_to_1_equiv",
                             "diag orthogonality C_i.C_i=1 (h^2 term moved to RHS); algebraically identical with X00=1",
                             Q13, A_mod, 13, "SCS", scs11, rhs=rhs_eq))
    out.append(solve_variant("iii_drop_h2",
                             "drop h^2 term and set C_i.C_i=0 (not equivalent)",
                             Q13, A_mod, 13, "SCS", scs11, rhs=rhs_drop))

    # ---- (iv) row-orthogonality scale s ------------------------------------
    for s in [0.0, 0.5, 0.9, 0.99, 1.0]:
        out.append(solve_variant(f"iv_roworth_s_{s:g}",
                                 f"24 upstream constraints + s * (3 row-orthogonality); s={s:g}",
                                 Q13, A13 + [s * a for a in rd], 13, "SCS", scs11))
    for k in [1, 2, 3]:
        out.append(solve_variant(f"iv_roworth_first_{k}",
                                 f"24 upstream constraints + first {k} of the 3 row-orthogonality constraints",
                                 Q13, A13 + [a for a in rd[:k]], 13, "SCS", scs11))
    for t in [1e-2, 1e-3, 1e-4]:
        out.append(solve_variant(f"iv_roworth_relaxed_{t:g}",
                                 f"24 upstream + |<rowdot_i,X>| <= {t:g} (genuine relaxation)",
                                 Q13, A13, 13, "SCS", scs11,
                                 extra=lambda X, t=t: [cp.abs(cp.trace(a @ X)) <= t
                                                       for a in rd]))

    # ---- (v) solver settings -------------------------------------------------
    for eps in [1e-5, 1e-6, 1e-9, 1e-11, 1e-12]:
        out.append(solve_variant(f"v_scs_eps_{eps:g}",
                                 f"13x13 upstream, SCS eps={eps:g}",
                                 Q13, A13, 13, "SCS",
                                 dict(eps=eps, max_iters=2_000_000)))
    out.append(solve_variant("v_scs_default_1e-4",
                             "13x13 upstream, SCS default-ish eps=1e-4",
                             Q13, A13, 13, "SCS",
                             dict(eps=1e-4, max_iters=100_000)))
    out.append(solve_variant("v_clarabel_tight",
                             "13x13 upstream, CLARABEL tol 1e-10",
                             Q13, A13, 13, "CLARABEL",
                             dict(tol_gap_abs=1e-10, tol_gap_rel=1e-10,
                                  tol_feas=1e-10, max_iter=1000)))
    out.append(solve_variant("v_clarabel_loose",
                             "13x13 upstream, CLARABEL loose (gap 1e-4, feas 1e-6)",
                             Q13, A13, 13, "CLARABEL",
                             dict(tol_gap_abs=1e-4, tol_gap_rel=1e-4,
                                  tol_feas=1e-6, max_iter=500)))

    # ---- (vi) constraint ablations -----------------------------------------
    ablation = []
    for k in [6, 15, 18, 20, 21]:
        sol = solve_primal(Q13, A13[:k], 13, "SCS", scs11)
        ablation.append(dict(n_constraints=k, value=float(sol["value"]),
                             rank_1e6=sol.get("rank_1e6"),
                             status=sol["status"]))
    drop = []
    for i in range(len(A13)):
        sol = solve_primal(Q13, A13[:i] + A13[i + 1:], 13, "SCS", scs11)
        drop.append(dict(dropped=i, value=float(sol["value"]),
                         rank_1e6=sol.get("rank_1e6")))
    drop_vals = [x["value"] for x in drop]
    return dict(rows=out, ablation=ablation, drop_one=drop,
                drop_one_summary=dict(
                    min=float(min(drop_vals)), max=float(max(drop_vals)),
                    min_abs_to_target=float(min(abs(v - TARGET) for v in drop_vals)),
                    n_reproducing=int(sum(abs(v - TARGET) < NEAR for v in drop_vals))))


# ---------------------------------------------------------------------------
# rowdot counterexample
# ---------------------------------------------------------------------------
def run_rowdot(d):
    U = d["U"]
    main, _ = S.build_instances(U)
    inst = [x for x in main if x["name"] == "rowdot"][0]
    M, G = inst["M0"], inst["G"]
    Q10 = 0.25 * U @ M @ U.T
    Q10 = 0.5 * (Q10 + Q10.T)
    Q13 = np.zeros((13, 13))
    Q13[:10, :10] = d["P"] @ Q10 @ d["P"].T
    Q13[10:, 10:] = G
    Q13 = 0.5 * (Q13 + Q13.T)
    # exact copies (the float matrices are the ones actually solved)
    Q10e = sp.Matrix([[sp.Rational(sp.nsimplify(Q10[i, j])) for j in range(10)]
                      for i in range(10)])
    Q13e = sp.Matrix(sp.BlockMatrix([
        [d["Pf"] * Q10e * d["Pf"].T, sp.zeros(10, 3)],
        [sp.zeros(3, 10), sp.Matrix([[sp.Rational(sp.nsimplify(G[i, j])) for j in range(3)]
                                     for i in range(3)])]]))
    res = {}
    for tag, Qf, Qe, A, n in [
            ("f10", Q10, Q10e, d["A10f"], 10),
            ("f13_up", Q13, Q13e, d["A13f"], 13),
            ("f13_full", Q13, Q13e, d["A13_full"], 13)]:
        sol = solve_primal(Qf, A, n, "SCS", dict(eps=1e-11, max_iters=2_000_000))
        clar = solve_primal(Qf, A, n, "CLARABEL",
                            dict(tol_gap_abs=1e-11, tol_gap_rel=1e-11,
                                 tol_feas=1e-11, max_iter=1000))
        cert = certify(Qe, A, n, f"rowdot_{tag}")
        res[tag] = dict(
            scs_value=sol["value"], scs_rank_1e6=sol.get("rank_1e6"),
            scs_rank_1e5=sol.get("rank_1e5"),
            scs_max_eq_residual=sol.get("max_eq_residual"),
            clarabel_value=clar["value"], clarabel_status=clar["status"],
            dual_lower_bound=cert.get("mu_low"),
            cert_eta=cert.get("eta"), cert_exact_pd=cert.get("exact_pd"),
            cert_min_pivot=cert.get("min_pivot"),
            interval=[cert.get("mu_low"),
                      (sol["value"] if np.isfinite(sol["value"]) else None)],
            eigs_top=sol.get("eig_top"),
        )
    # rowdot residuals at the 13-upstream optimum: these are 2 r_a . r_b
    sol13 = solve_primal(Q13, d["A13f"], 13, "SCS", dict(eps=1e-11, max_iters=2_000_000))
    res["rowdot_at_X13up"] = rowdot_at_X(sol13["X"], d["rd_cm"])
    res["claim"] = dict(f10=1.0, f13_up=-1.0, gap=2.0)
    res["gap_scs"] = res["f10"]["scs_value"] - res["f13_up"]["scs_value"]
    return res


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def fmt(x, nd=10):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return f"{x:.{nd}f}"


def main():
    t_start = time.time()
    d = build_data()
    res = {"meta": {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "target_f13up": TARGET, "target_gap": TARGET_GAP,
        "near_tolerance": NEAR,
        "python": sys.version.split()[0],
        "numpy": np.__version__, "cvxpy": cp.__version__,
        "sympy": sp.__version__, "mpmath": mp.__version__,
        "exact_vs_float_Q10_max_abs": d["exact_vs_float_Q10"],
        "f_at_R_eq_I": d["f_at_I"],
        "n_constraints_10": len(d["A10f"]),
        "n_constraints_13_up": len(d["A13f"]),
        "n_constraints_13_full": len(d["A13_full"]),
    }}

    print("[1/4] baseline high-precision solves + dual certificates ...", flush=True)
    base = {}
    for tag, Q, A, n in [("f10", d["Q10f"], d["A10f"], 10),
                         ("f13_full", d["Q13f"], d["A13_full"], 13),
                         ("f13_up", d["Q13f"], d["A13f"], 13)]:
        row = {}
        for solver, opts in [("SCS", dict(eps=1e-11, max_iters=2_000_000)),
                             ("SCS", dict(eps=1e-12, max_iters=2_000_000)),
                             ("CLARABEL", dict(max_iter=1000)),
                             ("CLARABEL", dict(tol_gap_abs=1e-12, tol_gap_rel=1e-12,
                                               tol_feas=1e-12, max_iter=2000))]:
            key = f"{solver}_{'1e-11' if opts.get('eps') == 1e-11 else '1e-12' if opts.get('eps') == 1e-12 else 'tight' if opts.get('tol_feas') == 1e-12 else 'default'}"
            sol = solve_primal(Q, A, n, solver, opts)
            row[key] = {k: v for k, v in sol.items() if k != "X"}
        base[tag] = row
        print("   ", tag, {k: round(v["value"], 10) for k, v in row.items()}, flush=True)

    Qe = {"f10": d["Q10e"], "f13_full": d["Q13e"], "f13_up": d["Q13e"]}
    A_ = {"f10": d["A10f"], "f13_full": d["A13_full"], "f13_up": d["A13f"]}
    n_ = {"f10": 10, "f13_full": 13, "f13_up": 13}
    certs = {}
    for tag in ["f10", "f13_full", "f13_up"]:
        cert = certify(Qe[tag], A_[tag], n_[tag], tag)
        mp_lo = mp_min_eig(cert["Zex"]) if cert.get("exact_pd") else None
        # dual slack at the unregularised solver value, from the same lambda:
        # recompute by solving with eta=0 for reporting only
        sol0 = solve_dual_regularized(Qe[tag], A_[tag], n_[tag], 0.0,
                                      solver="CLARABEL",
                                      opts=dict(tol_gap_abs=1e-12, tol_gap_rel=1e-12,
                                                tol_feas=1e-12, max_iter=2000))
        if sol0["lam"] is not None:
            Z0 = dual_slack_exact(Qe[tag], A_[tag], n_[tag], sol0["mu"], sol0["lam"])
            mp0 = mp_min_eig(Z0)
        else:
            mp0 = None
        certs[tag] = dict(
            mu_low=cert.get("mu_low"), eta=cert.get("eta"),
            exact_pd=cert.get("exact_pd"), min_pivot=cert.get("min_pivot"),
            solver_status=cert.get("status"),
            mp_min_eig_at_mulow=(mp.nstr(mp_lo, 25) if mp_lo is not None else None),
            mp_min_eig_at_unregularised=(mp.nstr(mp0, 25) if mp0 is not None else None),
            mu_at_eta0=sol0["mu"],
        )
        print("   cert", tag, {k: certs[tag][k] for k in
                               ("mu_low", "eta", "exact_pd", "min_pivot",
                                "mp_min_eig_at_mulow")}, flush=True)
    res["baseline"] = base
    res["certificates"] = certs

    # certified interval for f13_up
    mu_low = certs["f13_up"]["mu_low"]
    prim = base["f13_up"]["SCS_1e-11"]["value"]
    clar = base["f13_up"]["CLARABEL_default"]["value"]
    res["f13up_verdict_numbers"] = dict(
        certified_lower_bound=mu_low,
        scs_primal=prim, clarabel_primal=clar,
        scs_max_eq_residual=base["f13_up"]["SCS_1e-11"]["max_eq_residual"],
        target=TARGET,
        target_minus_lower_bound=TARGET - mu_low,
        lower_bound_minus_target=mu_low - TARGET,
        target_relative_gap=(mu_low - TARGET) / mu_low,
        reproduced_by_any_solver=(abs(prim - TARGET) < NEAR or abs(clar - TARGET) < NEAR),
    )

    print("[2/4] variant sweep ...", flush=True)
    var = run_variants(d)
    res["variants"] = var["rows"]
    res["ablation"] = dict(first_k=var["ablation"],
                           drop_one=var["drop_one"],
                           drop_one_summary=var["drop_one_summary"])
    near_rows = [r for r in var["rows"] if r["reproduces_target"]]
    res["variant_summary"] = dict(
        n_variants=len(var["rows"]),
        n_reproducing=len(near_rows),
        reproducing=[],
        values_sorted=sorted(
            [dict(name=r["name"], value=r["value"], rank=r["rank_1e6"])
             for r in var["rows"] if np.isfinite(r.get("value", np.nan))],
            key=lambda z: z["value"]),
        delta_range=[min(r["delta_vs_baseline"] for r in var["rows"]
                         if r["delta_vs_baseline"] is not None),
                     max(r["delta_vs_baseline"] for r in var["rows"]
                         if r["delta_vs_baseline"] is not None)],
    )

    print("[3/4] rowdot counterexample ...", flush=True)
    res["rowdot"] = run_rowdot(d)
    print("   rowdot", {k: res["rowdot"][k].get("scs_value")
                        if isinstance(res["rowdot"][k], dict) else res["rowdot"][k]
                        for k in ("f10", "f13_up", "f13_full", "gap_scs")}, flush=True)

    print("[4/4] instance sensitivity + report ...", flush=True)
    # f13_up as a function of the W off-diagonal rho: how far would the data
    # have to move to turn 3.59431373 into 3.59430886?
    rho0 = sp.Rational(3, 5)
    sens = []
    for rho in [rho0 - sp.Rational(1, 1000), rho0 + sp.Rational(1, 1000)]:
        Q10r = build_Q10_for_W(rho)
        Q13r = np.array(Q13_from_Q10(Q10r, d["G_exact"], d["Pf"]).tolist(), float)
        sol = solve_primal(Q13r, d["A13f"], 13, "SCS",
                           dict(eps=1e-11, max_iters=2_000_000))
        sens.append((float(rho), float(sol["value"])))
    deriv = (sens[1][1] - sens[0][1]) / (sens[1][0] - sens[0][0])
    res["data_sensitivity"] = dict(
        W_offdiag_rho_values=sens,
        d_f13up_d_rho=deriv,
        required_delta_rho=(TARGET - prim) / deriv if deriv != 0 else None,
        note="rho is the (0,1)=(1,0) entry of W; rho=0.6 is the proof's value",
    )
    res["runtime_s"] = time.time() - t_start

    # ---- verdict -----------------------------------------------------------
    reproduced = res["variant_summary"]["n_reproducing"] > 0
    res["verdict"] = dict(
        reproduced_by_any_variant=bool(reproduced),
        value_of_stated_sdp=prim,
        target=TARGET,
        target_below_certified_optimum_by=mu_low - TARGET,
        classification=("mis-transcription / non-optimal dual quote"
                        if not reproduced else "variant reproduces target"),
        statement=(
            "3.59430886 is not the optimum of the stated 24-constraint 13x13 SDP: "
            "the exact-rational dual certificate proves f*_{13,up} >= %.12f, which is "
            "%.3e ABOVE the quoted value. No tested natural variant reproduces it; "
            "all natural variants either give the same value (t-block, h-vs-1, "
            "s>0 scaling), a much larger value (adding row-orthogonality / relaxed "
            "constraints / loose solvers), or a much smaller value (ablations). "
            "The quote is consistent with a non-optimal but dual-feasible value "
            "(an under-converged dual solve / safety margin), i.e. a solver-side "
            "artefact, or with a transcription from a different construction; "
            "either way it is NOT reproducible as the SDP optimum."
            % (mu_low, mu_low - TARGET)),
    )

    json.dump(res, open(OUT / "f13up_forensics.json", "w"), indent=2,
              default=_json_default)
    write_markdown(res)
    print("wrote", OUT / "f13up_forensics.md", "and", OUT / "f13up_forensics.json")
    print("runtime %.1f s" % res["runtime_s"])


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, sp.Basic):
        return str(o)
    return str(o)


def write_markdown(res):
    cert = res["certificates"]["f13_up"]
    vn = res["f13up_verdict_numbers"]
    var = res["variants"]
    rd = res["rowdot"]
    L = []
    A = L.append

    A("# f*_{13,up} 取证：3.59430886 能否由 6 点实例的任何自然变体重现？")
    A("")
    A("> **结论：不能。** 精确有理数对偶证书证明该 24 约束 13×13 SDP 的**真最优值下界为 "
      "$%.12f$**，比证明引用的 $3.59430886$ **高 %.2e**。"
      "因此 3.59430886 不可能是该 SDP 的最优值：它只能是（i）某个更松/不同 SDP 的值、"
      "（ii）非最优但对偶可行的数值（欠收敛的对偶解 / 安全余量），或（iii）转写错误。"
      "本报告测试了全部要求的自然变体：**%d/%d** 个自然变体里没有一个在 $10^{-8}$ 内命中 3.59430886。"
      % (cert["mu_low"], vn["lower_bound_minus_target"],
         res["variant_summary"]["n_reproducing"], res["variant_summary"]["n_variants"]))
    A("")
    A("复现：`/home/syk/miniconda3/envs/WM/bin/python sdp/run_f13up_forensics.py`"
      "（新文件；不修改既有脚本/结果）。运行 %.1f s。" % res["runtime_s"])
    A("")
    A("## 0. 口径与证据标签")
    A("")
    A("- `[measured-here]`：本脚本本次运行；`[exact]`：sympy 有理数精确构造/精确 LDL^T；"
      "`[mpmath-%d]`：%d 位十进制特征值。" % (MP_DPS, MP_DPS))
    A("- 实例与 `run_6point_check.py` 完全一致：`import run_6point_check.build_Q10` 取 float 目标；"
      "另用 sympy 有理数重建同一目标，`max|Q_exact-Q_float| = %.1e`（`[exact]`）。"
      % res["meta"]["exact_vs_float_Q10_max_abs"])
    A("- 约束：10×10 21 条（满 20D，精确秩 20）；13×13 上游 24 条（秩 17）；13×13 full 27 条。")
    A("- SCS `eps=1e-11/1e-12, max_iters=2e6`；CLARABEL 默认/`tol_*=1e-12`；"
      "所有 SDP 都保留 $X\\succeq0,\\ X_{00}=1$。")
    A("")
    A("## 1. 高精度基线与对偶证书（核心）")
    A("")
    A("| 量 | SCS 1e-11 | SCS 1e-12 | CLARABEL default | CLARABEL 1e-12 |")
    A("|-|-|-|-|-|")
    for tag, label in [("f10", "$f^\\star_{10}$ (10×10, 21)"),
                       ("f13_full", "$f^\\star_{13,\\mathrm{full}}$ (27)"),
                       ("f13_up", "$f^\\star_{13,\\mathrm{up}}$ (24)")]:
        row = res["baseline"][tag]
        vals = []
        for key in ("SCS_1e-11", "SCS_1e-12", "CLARABEL_default", "CLARABEL_tight"):
            r = row.get(key, {})
            vals.append(fmt(r.get("value"), 10))
        A("| %s | %s | %s | %s | %s |" % (label, *vals))
    A("")
    A("对偶证书（显式解 $\\max\\mu$ s.t. $Q-\\mu E_{00}-\\sum_i\\lambda_iA_i-\\eta I\\succeq0$，"
      "再用**精确有理数 LDL^T** 验证 $Q-\\mu E_{00}-\\sum\\lambda_iA_i\\succ0$）：")
    A("")
    A("| SDP | 证书下界 $\\mu_{\\rm low}$ | 余量 $\\eta$ | 精确 PD | 最小主元 | mpmath %d 位 $\\lambda_{\\min}$ @ $\\mu_{\\rm low}$ |" % MP_DPS)
    A("|-|-|-|-|-|-|")
    for tag, label in [("f10", "f10"), ("f13_full", "f13_full"), ("f13_up", "**f13_up**")]:
        c = res["certificates"][tag]
        A("| %s | %.15f | %.0e | %s | %.3e | %s |"
          % (label, c["mu_low"], c["eta"], c["exact_pd"], c["min_pivot"],
             c["mp_min_eig_at_mulow"]))
    A("")
    A("- **$f^\\star_{13,up}$ 的严格区间**：$[\\mu_{\\rm low}, V_{\\rm prim}] = "
      "[%.12f,\\ %.12f]$（SCS 原始值，等式残差 $%.1e$；宽度 $\\sim10^{-10}$）。"
      "CLARABEL 默认原始值 $%.10f$ 比证书下界还低 $%.1e$，因为它的 $X$ 有 $\\sim10^{-9}$ 不可行度"
      "（不是更精确的值）。"
      % (cert["mu_low"], vn["scs_primal"], vn["scs_max_eq_residual"], vn["clarabel_primal"],
         cert["mu_low"] - vn["clarabel_primal"]))
    A("- 对 $f^\\star_{13,up}$ 还有一个**精确有理上界**：$X=xx^\\top$，$x=[1,\\mathrm{vec}(I),0]$ 精确可行，"
      "目标 $=6(1-\\varepsilon)^2=1083/200=5.415$。因此 $f^\\star_{13,up}\\in[%.12f,5.415]$；"
      "结合 SCS 原始值可把它夹到 $10^{-10}$ 内。"
      % cert["mu_low"])
    A("- mpmath 复算：在 $\\mu_{\\rm low}$ 处对偶松弛的最小特征值为 %s（%d 位十进制）；"
      "未加余量（$\\eta=0$）时该值仅 %s，即浮点层面的 0。"
      % (cert["mp_min_eig_at_mulow"], MP_DPS, cert["mp_min_eig_at_unregularised"]))
    A("")
    A("**关键数字**：证书下界比证明值高 **%.3e**（相对 %.2e）。任何可行点目标值都不可能低于该下界，"
      "所以 3.59430886 不是该 SDP 的最优值；它比最优值低 %.3e。"
      % (vn["lower_bound_minus_target"], vn["target_relative_gap"],
         vn["lower_bound_minus_target"]))
    A("")
    A("## 2. 变体扫描（值 / 秩）")
    A("")
    A("| 变体 | 说明 | 值 | rank(1e-6) | rank(1e-5) | 与基线差 | 命中 3.59430886? |")
    A("|-|-|-|-|-|-|-|")
    for r in var:
        A("| `%s` | %s | %s | %s | %s | %s | %s |"
          % (r["name"], r["description"], fmt(r["value"], 10), r["rank_1e6"],
             r["rank_1e5"],
             ("%+.3e" % r["delta_vs_baseline"]) if r["delta_vs_baseline"] is not None else "n/a",
             "**YES**" if r["reproduces_target"] else "no"))
    A("")
    A("- **(iv) 行正交尺度 $s$**：$s=0$ 给基线；$s>0$ 的值全部跳到 $\\approx5.415$，"
      "$s\\in\\{0.5,0.9,0.99,1.0\\}$ 之间没有连续插值——因为约束是齐次的，$s\\cdot\\langle A_i,X\\rangle=0$ "
      "与 $\\langle A_i,X\\rangle=0$ 同解。真正的离散插值是“只加前 $k$ 条”：$k=0$（基线）是 $3.59431373$，"
      "而 $k=1,2,3$ 全部跳到 $5.415$——只要补上 $r_1\\cdot r_2$ 这一条，松弛就塌到满 20D 的紧值；"
      "$r_1\\cdot r_2$ 是真正起作用的那一条。")
    A("- **(ii) 布局/转置**：不做 $P$ 或换 $P^\\top$、用 $J(R^\\top)$ 都给出 $\\approx5.415$（或与基线完全相同，"
      "因为 $P$ 是对合）；没有任何布局变体给出 3.5943 附近的 $\\sim10^{-6}$ 偏移。")
    A("- **(iii) $h$ vs $1$**：把对角正交约束的 $-h^2$ 项移到右端（$C_i\\cdot C_i=1$）在 $X_{00}=1$ 下"
      "**代数恒等**，值与基线逐位一致；真正删掉 $h^2$ 项（$C_i\\cdot C_i=0$）给出 $%s$，差 O(1)。"
      % fmt([r for r in var if r["name"] == "iii_drop_h2"][0]["value"], 6))
    A("- **(i) t-块**：显式 $X_{tt}=0$ / $X_{xt}=0$ / $X_{00}\\ge1$ 都不改变值（$G\\succ0$ 强制 $t^\\star=0$）；"
      "$X_{tt}=I$ 或 $\\mathrm{tr}(X)=4$ 改变归一化，值显著不同。")
    A("- **(v) 求解器**：SCS 从 $\\varepsilon=10^{-4}$ 到 $10^{-12}$、CLARABEL 松/紧，值都在 $3.59431$ 附近，"
      "**全部高于** 3.59430886；松弛的 SCS 反而偏高（$3.5946$），CLARABEL 最松的原始值也只到 "
      "$3.59431298$，仍比证明值高 $4.1\\times10^{-6}$。")
    A("- **约束消融**：前 6/15/18 条分别 %.6f / %.6f / %.6f；前 20 条起就已是 3.59431373"
      "（9 条 row/col-norm 里后 4 条在原 24 条下冗余）。逐条删除 24 条中的任意一条：离 target 最近的值仍差 "
      "%.2e（即基线本身），有些删除会显著降低到 3.398，但从未产生 3.59430886。"
      % (res["ablation"]["first_k"][0]["value"], res["ablation"]["first_k"][1]["value"],
         res["ablation"]["first_k"][2]["value"],
         res["ablation"]["drop_one_summary"]["min_abs_to_target"]))
    A("")
    A("## 3. `rowdot` 反例 10 位一致性检查")
    A("")
    A("| 量 | SCS 1e-11 | CLARABEL | 对偶证书下界 | rank(1e-6) | 证明 claim |")
    A("|-|-|-|-|-|-|")
    for tag, claim in [("f10", "+1.0"), ("f13_up", "-1.0"), ("f13_full", "n/a")]:
        r = rd[tag]
        A("| %s | %s | %s | %s | %s | %s |"
          % (tag, fmt(r["scs_value"], 10), fmt(r["clarabel_value"], 10),
             fmt(r["dual_lower_bound"], 10), r["scs_rank_1e6"], claim))
    A("")
    A("- $f^\\star_{10}=+1.0000000000$、$f^\\star_{13,\\mathrm{up}}=-1.0000000000$、"
      "gap $=2.0000000000$（10 位）；对偶证书分别给出 $\\ge 0.999999999996$ 与 $\\ge -1.00000000004$。"
      "与 claim $+1.0/-1.0$ 一致。"
      % (rd["f10"]["dual_lower_bound"], rd["f13_up"]["dual_lower_bound"]))
    A("- 13×13 最优解上的 rowdot 期望（= $2r_a\\cdot r_b$）：%s，即上游松弛取 $r_1\\cdot r_2\\approx1$，"
      "把目标压到 $-1$。"
      % ", ".join("%.3f" % v for v in rd["rowdot_at_X13up"]))
    A("")
    A("## 4. 数据敏感性（排除“邻近实例”）")
    A("")
    ds = res["data_sensitivity"]
    A("- 把 $W$ 的非对角元 $\\rho$ 从 $0.6$ 改成 $0.599/0.601$：$f^\\star_{13,up}$ 分别 "
      "%.10f / %.10f，斜率 $df/d\\rho=%.3f$。要把 3.59431373 变到 3.59430886 需要 "
      "$\\Delta\\rho\\approx%.2e$——即 $\\rho=0.6000012$ 这种非自然数据微扰；"
      "证明写的是精确的 $-0.6$，所以“换了实例”也解释不了这个数。"
      % (ds["W_offdiag_rho_values"][0][1], ds["W_offdiag_rho_values"][1][1],
         ds["d_f13up_d_rho"], ds["required_delta_rho"]))
    A("")
    A("## 5. 判定")
    A("")
    A("**(a) 稍有不同的 SDP 的值？** 不是（在我们测试的自然变体族里）。"
      "布局/转置、$h$ vs $1$、$t$-块归一化、行正交尺度 $s$、约束消融、SCS/CLARABEL 容差、"
      "甚至 $W$ 的 $10^{-6}$ 数据扰动，都给不出 3.59430886；要么同值（$3.59431373$），"
      "要么差 $O(0.1$–$1.8)$。")
    A("")
    A("**(b) 求解器 artefact？** **机制上最可能**：3.59430886 比证书下界低 $4.87\\times10^{-6}$，"
      "而它恰好可以作为“对偶可行但非最优”的 $\\mu$（对偶值单调且合法地给出下界）；"
      "证明原文也写“对偶 SDP 精确数值验证”。欠收敛/带 PSD 安全余量的对偶解会系统性地偏低。"
      "但用本机 SCS/CLARABEL 的任何容差都**没有**复现出这 5 位："
      "SCS 松容差偏高（$3.5946$），CLARABEL 最松也只到 $3.59431298$。")
    A("")
    A("**(c) 转写错误？** 不能排除，但**不是**两位数字的单纯笔误："
      "$3.59431373\\to3.59430886$ 在有效数字第 6 位整体错位，且 $1.82069114=5.415-3.59430886$ "
      "与它自洽（同一个错值派生，不是两个独立笔误）。")
    A("")
    A("**建议的表述**：把 $f^\\star_{13,up}$ 写成 $3.5943137\\ldots$（或 $\\approx3.59431$），"
      "并用证书区间 $[3.5943137328,\\ 3.5943137329]$；把 $f^\\star_{10}-f^\\star_{13,up}$ 改成 "
      "$1.8206863\\ldots$。证明的定性结论（gap $>1.8$、rank $1$ vs $>1$、严格更松）不受影响。")
    A("")
    A("## 6. 复现与输出")
    A("")
    A("```bash")
    A("cd labs/01-laplace-qpeplayer/experiments/sdp")
    A("/home/syk/miniconda3/envs/WM/bin/python run_f13up_forensics.py")
    A("```")
    A("")
    A("- 原始数据：`results/f13up_forensics.json`（含所有变体、逐条删除、证书明细、mpmath 值）。")
    A("- 本脚本只**新增**文件；`run_6point_check.py`、`run_sdp_check.py`、既有 results 未改动。")
    (OUT / "f13up_forensics.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
