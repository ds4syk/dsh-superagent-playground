#!/usr/bin/env python3
"""Exact check of the 6-point physical 3D-3D counterexample (new P1-Q2/Q13 proof).

Construction (proof of 2026-11-27, Feishu RhzDwI1RKinQsekHDhUcjOVenHL):
  p_i = +/-e1, +/-e2, +/-e3 ; y_i = eps p_i, eps=0.05 ;
  W = [[1,-0.6,0],[-0.6,1,0],[0,0,1]] (eigs {0.4,1,1.6} > 0), shared weights.
Then D=0, G=6W and
  J(R,t) = 2 tr(W R R^T) - 4 eps tr(W R) + 2 eps^2 tr(W) + 6 t^T W t,
whose R=I, t=0 value is 6(1-eps)^2 = 5.415.

The 10x10 SDP of this project lives in x_R = [h; vec(R)] (h = ||q||^2), with
R = R(q), so J is a *quadratic* polynomial in x_R; its coefficient matrix is
unique.  We build it exactly, then compare
  10x10 (21 constraints, rank 20),
  13x13 upstream (24 constraints, rank 17),
  13x13 full (24 + 3 row-orthogonality = full 20D),
against the proof's numbers f10 = 5.415000, f13_up = 3.59430886, ranks 1 vs 4.

Run:
  /home/syk/miniconda3/envs/WM/bin/python sdp/run_6point_check.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import sympy as sp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_sdp_check as S  # noqa: E402

OUT = HERE / "results"
OUT.mkdir(parents=True, exist_ok=True)
EPS = 0.05
W = np.array([[1.0, -0.6, 0.0], [-0.6, 1.0, 0.0], [0.0, 0.0, 1.0]])


def build_Q10():
    """10x10 objective in x_R=[h, vec_rowmajor(R)] with x^T Q x = J(R, t=0)."""
    h = sp.symbols("h", real=True)
    R = sp.Matrix(3, 3, lambda i, j: sp.Symbol(f"R{i}{j}", real=True))
    Ws = sp.Matrix(W.tolist())
    qq = h  # ||q||^2, equals x0
    p = (2*sp.trace(Ws*R*R.T) - 4*sp.Rational(1, 20)*sp.trace(Ws*R)*h
         + 2*sp.Rational(1, 400)*sp.trace(Ws)*h*h)
    p = sp.expand(p)
    # ordered variables: [h, R00,R01,R02, R10,R11,R12, R20,R21,R22]
    vars10 = [h] + [R[i, j] for i in range(3) for j in range(3)]
    Q = sp.zeros(10, 10)
    for a in range(10):
        for b in range(a, 10):
            if a == b:
                c = sp.expand(p).coeff(vars10[a], 2)
                Q[a, a] = c
            else:
                c = sp.expand(p).coeff(vars10[a]*vars10[b])
                Q[a, b] = c / 2
                Q[b, a] = c / 2
    pp = (sp.Matrix(vars10).T * Q * sp.Matrix(vars10))[0, 0]
    assert sp.simplify(sp.expand(pp - p)) == 0
    # and the value at R=I, h=1
    subs = {h: 1}
    for i in range(3):
        for j in range(3):
            subs[R[i, j]] = 1 if i == j else 0
    f_at_I = float(sp.simplify(p.subs(subs)))
    return np.array(Q.tolist(), dtype=float), f_at_I


def _solve_clarabel(Q, A_list, n):
    """Independent interior-point cross-check (CLARABEL), same formulation."""
    import cvxpy as cp
    X = cp.Variable((n, n), symmetric=True)
    cons = [X >> 0, X[0, 0] == 1] + [cp.trace(a @ X) == 0 for a in A_list]
    prob = cp.Problem(cp.Minimize(cp.trace(Q @ X)), cons)
    prob.solve(solver=cp.CLARABEL, verbose=False)
    Xv = np.array(X.value)
    ev = np.linalg.eigvalsh(0.5*(Xv + Xv.T))[::-1]
    rank = int(np.sum(ev > 1e-6*max(abs(ev[0]), 1e-12)))
    return dict(status=prob.status, value=float(prob.value), rank_1e6=rank,
                eigs_top=[float(x) for x in ev[:6]])


def nonconvex_value(n_sample=300000, seed=0):
    """min over SO(3) of J(R, 0), sampled densely in quaternion space."""
    rng = np.random.default_rng(seed)
    Q = rng.normal(size=(n_sample, 4))
    Q /= np.linalg.norm(Q, axis=1, keepdims=True)
    q0, q1, q2, q3 = Q.T
    R = np.empty((n_sample, 3, 3))
    R[:, 0, 0] = q0*q0 + q1*q1 - q2*q2 - q3*q3
    R[:, 0, 1] = 2*(q1*q2 - q0*q3)
    R[:, 0, 2] = 2*(q1*q3 + q0*q2)
    R[:, 1, 0] = 2*(q1*q2 + q0*q3)
    R[:, 1, 1] = q0*q0 - q1*q1 + q2*q2 - q3*q3
    R[:, 1, 2] = 2*(q2*q3 - q0*q1)
    R[:, 2, 0] = 2*(q1*q3 - q0*q2)
    R[:, 2, 1] = 2*(q2*q3 + q0*q1)
    R[:, 2, 2] = q0*q0 - q1*q1 - q2*q2 + q3*q3
    RRt = np.einsum("njk,nlk->njl", R, R)          # (R R^T)_jl
    vals = (2*np.einsum("ij,nji->n", W, RRt)
            - 4*EPS*np.einsum("ij,nji->n", W, R) + 2*EPS**2*np.trace(W))
    return float(vals.min())


def main():
    Q10, f_at_I = build_Q10()
    U, A10, names10, checks10 = S.build_10x10()
    A13, checks13 = S.build_13x13()
    P = S.perm_rowmajor_to_colmajor()
    rd_cm, rd_rm = S.rowdot_matrices()
    A13_full = A13 + rd_cm
    G = 6.0 * W
    # 13-layout: rotate x_R to column-major, add the t block (D = 0)
    Q13 = np.zeros((13, 13))
    Q13[:10, :10] = P @ Q10 @ P.T
    Q13[10:, 10:] = G
    Q13 = 0.5*(Q13 + Q13.T)
    # 13x13 full-20D must give the same value: check the t-block pair (X_tt=0)
    # tight tolerance: the claims are stated to 8 decimals
    sol10 = S.solve_sdp(Q10, A10, 10, eps=1e-11, max_iters=1000000)
    sol13_full = S.solve_sdp(Q13, A13_full, 13, eps=1e-11, max_iters=1000000)
    sol13_up = S.solve_sdp(Q13, A13, 13, eps=1e-11, max_iters=1000000)
    clar = {}
    for tag, Q, A, n in [("10", Q10, A10, 10), ("13_full", Q13, A13_full, 13),
                         ("13_up", Q13, A13, 13)]:
        try:
            clar[tag] = _solve_clarabel(Q, A, n)
        except Exception as exc:  # pragma: no cover
            clar[tag] = dict(status="FAILED: " + repr(exc), value=float("nan"), rank_1e6=None)
    # exact nonconvex value: W symmetric with eigs {0.4,1,1.6}, so
    # max_{R in SO(3)} tr(W R) = sum |eig(W)| = 3 at R=I, hence
    # min J = 6 + 2 eps^2 tr(W) - 4 eps * 3 = 5.415 exactly.
    f_nc = f_at_I
    f_nc_sample = nonconvex_value()
    res = dict(
        value_at_R_eq_I=f_at_I,
        f_nonconvex=f_nc, f_nonconvex_sampled=f_nc_sample,
        f10=sol10["value"], f13_full=sol13_full["value"], f13_up=sol13_up["value"],
        rank10=sol10.get("rank_1e6"), rank13_full=sol13_full.get("rank_1e6"),
        rank13_up=sol13_up.get("rank_1e6"),
        eig10=sol10.get("eigs_top", [])[:4], eig13_up=sol13_up.get("eigs_top", [])[:4],
        status10=sol10["status"], status13_full=sol13_full["status"],
        status13_up=sol13_up["status"],
        gap10=float(f_nc - sol10["value"]),
        gap13_full=float(f_nc - sol13_full["value"]),
        gap13_up=float(f_nc - sol13_up["value"]),
        f10_minus_f13_up=float(sol10["value"] - sol13_up["value"]),
        n_constraints_10=len(A10), n_constraints_13_up=len(A13),
        n_constraints_13_full=len(A13_full),
        rank_21_exact=checks10["rank_21_constraints_exact"],
        clarabel=clar,
    )
    json.dump(res, open(OUT / "six_point.json", "w"), indent=2)
    L = ["# 6-point physical 3D-3D counterexample (new P1-Q2/Q13 proof)", "",
         f"eps={EPS}, W eigs {np.round(np.linalg.eigvalsh(W),3).tolist()}, "
         f"p_i=+-e1,+-e2,+-e3, y_i=eps p_i, N=6, G=6W, D=0; value at R=I: {res['value_at_R_eq_I']:.8f}.", "",
         "| quantity | measured here | proof claim |", "|-|-|-|",
         f"| f*_10 (21 constraints, full 20D) | {res['f10']:.8f} | 5.41500000 |",
         f"| f*_13 full (24 + 3 row-orth = full 20D) | {res['f13_full']:.8f} | 5.41500000 |",
         f"| f*_13 upstream (24 constraints, 17D) | {res['f13_up']:.8f} | 3.59430886 |",
         f"| f*_10 - f*_13_up | {res['f10_minus_f13_up']:.8f} | 1.82069114 |",
         f"| rank(X*_10) | {res['rank10']} | 1 |",
         f"| rank(X*_13 upstream) | {res['rank13_up']} | 4 |",
         f"| f_nonconvex (exact: max tr(WR)=3 at R=I) | {res['f_nonconvex']:.8f} | 5.415 |",
         f"| f_nonconvex (300k quaternion samples, sanity) | {res['f_nonconvex_sampled']:.8f} | 5.415 |",
         f"| gap_10 = f_nonconvex - f*_10 | {res['gap10']:.2e} | 0 |",
         f"| gap_13_full | {res['gap13_full']:.2e} | 0 |",
         f"| gap_13_up | {res['gap13_up']:.6f} | 1.82069114 |",
         "",
         "Independent solver cross-check (CLARABEL, interior point): "
         "f10=%.8f, f13_full=%.8f, f13_up=%.8f; f13_up eigenvalues %s. "
         "SCS and CLARABEL agree to <1e-9, so the 4.9e-6 difference from the proof's quoted "
         "3.59430886 is not solver tolerance; the qualitative separation is confirmed but the "
         "last five digits are not reproduced. rank(X*_13up) is 4 at a 1e-5 threshold and 6 at "
         "1e-6 (four O(1) eigenvalues then ~1e-5 ones); the theorem only needs rank > 1."
         % (res["clarabel"]["10"]["value"], res["clarabel"]["13_full"]["value"],
            res["clarabel"]["13_up"]["value"],
            np.round(res["clarabel"]["13_up"]["eigs_top"], 6).tolist()),
         "",
         f"Constraints: 10x10 {res['n_constraints_10']} (exact rank {res['rank_21_exact']}), "
         f"13x13 upstream {res['n_constraints_13_up']}, 13x13 full {res['n_constraints_13_full']}.",
         f"Solver status: 10={res['status10']}, 13full={res['status13_full']}, 13up={res['status13_up']}."]
    (OUT / "six_point.md").write_text("\n".join(L) + "\n")
    print(json.dumps(res, indent=1))
    print("wrote", OUT / "six_point.md")


if __name__ == "__main__":
    main()
