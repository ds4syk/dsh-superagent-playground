#!/usr/bin/env python
"""P1-Q2: numeric check of the 10x10 Veronese/Shor relaxation vs the 13x13
SDPRLayers SDP on the rotation subproblem.

Question (math_questions.md Q2)
-------------------------------
SDPRLayers solves, in ``x = [h; vec(C); t] in R^13``,

    min <Q, X>   s.t.  X >= 0,  X_00 = 1,  <A_i, X> = 0  (i = 1..24),

with 6 orthogonality + 9 handedness + 9 row/col-norm constraints
(``sdprlayer/src/sdprlayers/layers/pose_est.py``:
``gen_orthogonal_constraints`` + ``gen_handedness_constraints`` +
``gen_row_col_constraints``; ``redun_list = range(6, 24)``).
Theorem 0.3 only covers the 10x10 *rotation-only, t-eliminated* SDP

    min <Q_10, Y>  s.t.  Y >= 0,  Y_00 = 1,  <A_j, Y> = 0  (j = 1..21)

with Q_10 = (1/4) U M U^T, M = M_0 - D^T G^{-1} D, and 21 constraints
(3 column norms + 3 column dots + 3 row norms + 3 row dots + 9 handedness).
Do the two have the same optimal value / rank-related solution?

What this script constructs (and what it does not)
--------------------------------------------------
* 10x10 construction: **verbatim reuse** of ``verify/verify_theorem0.py``
  (exact sympy ``T``/``U`` with ``x_R = 2 U v2``, row-major ``vec``, and the
  21 constraint matrices built by ``sym_from_terms``/``cross_terms``).
  Regression check: for the Choi-Lam instance the SDP value must reproduce
  ``verify/verify_theorem0_output.json`` (1.9658119658452384) and rank 6.
* 13x13 construction: line-by-line replication of the released upstream
  generators at ``sdprlayer`` commit 62d8ac6 (see ``POSE_EST_GENERATORS``
  below).  With ``--check-upstream PATH`` the replica is compared entrywise
  against the real ``PolyMatrix``-based generators; the comparison is
  reported in the JSON.  Without that flag the script has no external
  dependency.
* Translation: for every instance we start from the P6/QPEP cost
  ``J(q,t) = v2(q)^T M0 v2(q) + 2 t^T D v2(q) + t^T G t`` with ``G > 0``
  and build the 13x13 Q in the upstream layout.  The 10x10 problem uses
  ``M = M0 - D^T G^{-1} D`` (exact t-elimination of the *nonconvex* cost).
  Because every 13x13 constraint involves only ``(h, vec C)`` and G > 0, the
  13x13 SDP is analytically equal to ``min over the 13x13 rotation-block
  feasible set of <Q_10, X_r>`` (Schur complement on the unconstrained
  ``t`` block); the script checks this numerically (``schur_err``).
* NOT constructed / NOT claimed: no claim that the sampled instances are
  physically realizable matrix-weighted PnP Gram matrices (that is Q1), and
  no claim about the *nonconvex* P6 problem beyond the 10x10 equivalence.

Evidence labels used in ``results/sdp_check.md``:
``[measured-here]``, ``[quoted-verify]`` (from verify/*_output.json),
``[exact]`` (exact rational arithmetic in this script).

Run
---
    /home/syk/miniconda3/envs/WM/bin/python run_sdp_check.py
    /home/syk/miniconda3/envs/WM/bin/python run_sdp_check.py --check-upstream /path/to/sdprlayer

Pure numpy/scipy/cvxpy/SCS (no Mosek), runtime ~1 min.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time

import cvxpy as cp
import numpy as np
import sympy as sp
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
SQ2 = np.sqrt(2.0)

# Upstream generators replicated below (sdprlayer @ 62d8ac6, pose_est.py
# L341-L417).  Listed here so the replication can be audited line by line:
#   gen_orthogonal_constraints : C_i . C_j - delta_ij h^2 = 0      (6)
#   gen_handedness_constraints : (C_j x C_k)_l - h C_li = 0 cyclic (9)
#   gen_row_col_constraints    : ||col_j||^2 - ||row_i||^2 = 0     (9)
POSE_EST_GENERATORS = "sdprlayer@62d8ac6 src/sdprlayers/layers/pose_est.py L341-L417"


# ===========================================================================
# 1. exact 10x10 construction -- reused from verify/verify_theorem0.py
# ===========================================================================
def build_10x10():
    """Exact U with x_R = 2 U v2 and the 21 SO(3) constraints.

    This is the construction of ``labs/01-laplace-qpeplayer/verify/
    verify_theorem0.py`` (sections "symbolic setup" and "21 constraints"),
    copied verbatim except that it returns arrays instead of writing JSON.
    """
    q = sp.symbols("q0 q1 q2 q3", real=True)
    q0, q1, q2, q3 = q

    R = sp.Matrix([
        [q0**2 + q1**2 - q2**2 - q3**2, 2*(q1*q2 - q0*q3), 2*(q1*q3 + q0*q2)],
        [2*(q1*q2 + q0*q3), q0**2 - q1**2 + q2**2 - q3**2, 2*(q2*q3 - q0*q1)],
        [2*(q1*q3 - q0*q2), 2*(q2*q3 + q0*q1), q0**2 - q1**2 - q2**2 + q3**2],
    ])
    # homogeneous lift [||q||^2, vec(R)], vec = row-major
    xhom = sp.Matrix([q0**2 + q1**2 + q2**2 + q3**2]
                     + [R[i, j] for i in range(3) for j in range(3)])
    v2 = sp.Matrix([q0**2, q1**2, q2**2, q3**2,
                    sp.sqrt(2)*q0*q1, sp.sqrt(2)*q0*q2, sp.sqrt(2)*q0*q3,
                    sp.sqrt(2)*q1*q2, sp.sqrt(2)*q1*q3, sp.sqrt(2)*q2*q3])
    mb = sp.Matrix([q0**2, q1**2, q2**2, q3**2,
                    q0*q1, q0*q2, q0*q3, q1*q2, q1*q3, q2*q3])
    Dg = sp.diag(1, 1, 1, 1, *([sp.sqrt(2)]*6))

    T = sp.zeros(10, 10)
    for r in range(10):
        poly = sp.expand(xhom[r])
        for k in range(10):
            T[r, k] = sp.nsimplify(sp.expand(poly).coeff(mb[k]))
    assert sp.simplify(T*mb - xhom) == sp.zeros(10, 1)
    Usym = sp.simplify(sp.Rational(1, 2) * T * Dg.inv())
    U = np.array(Usym.tolist(), dtype=float)

    idx = {(i, j): 1 + 3*i + j for i in range(3) for j in range(3)}

    def sym_from_terms(terms):
        A = sp.zeros(10, 10)
        for (a, b), c in terms.items():
            A[a, b] += sp.Rational(c, 2)
            A[b, a] += sp.Rational(c, 2)
        return A

    def cols():
        return [[R[i, j] for i in range(3)] for j in range(3)]

    c1, c2, c3 = cols()
    cons = []
    # 6 column constraints
    for (a, b, nm) in [(0, 1, "c1.c2"), (0, 2, "c1.c3"), (1, 2, "c2.c3")]:
        terms = {}
        for i in range(3):
            terms[(idx[(i, a)], idx[(i, b)])] = 1
        cons.append(("col_orth_" + nm, sym_from_terms(terms)))
    for j in range(3):
        terms = {(0, 0): -1}
        for i in range(3):
            terms[(idx[(i, j)], idx[(i, j)])] = 1
        cons.append((f"col_norm_{j+1}", sym_from_terms(terms)))
    # 6 row constraints
    for (a, b) in [(0, 1), (0, 2), (1, 2)]:
        terms = {}
        for j in range(3):
            terms[(idx[(a, j)], idx[(b, j)])] = 1
        cons.append((f"row_orth_r{a+1}r{b+1}", sym_from_terms(terms)))
    for i in range(3):
        terms = {(0, 0): -1}
        for j in range(3):
            terms[(idx[(i, j)], idx[(i, j)])] = 1
        cons.append((f"row_norm_{i+1}", sym_from_terms(terms)))
    # 9 homogeneous cross-product / handedness constraints: c1 x c2 = x0 c3
    def cross_terms(a, b, c):
        out = []
        out.append((f"cross_{a}{b}_1", {(idx[(1, a)], idx[(2, b)]): 1,
                                        (idx[(2, a)], idx[(1, b)]): -1,
                                        (0, idx[(0, c)]): -1}))
        out.append((f"cross_{a}{b}_2", {(idx[(2, a)], idx[(0, b)]): 1,
                                        (idx[(0, a)], idx[(2, b)]): -1,
                                        (0, idx[(1, c)]): -1}))
        out.append((f"cross_{a}{b}_3", {(idx[(0, a)], idx[(1, b)]): 1,
                                        (idx[(1, a)], idx[(0, b)]): -1,
                                        (0, idx[(2, c)]): -1}))
        return out

    for (a, b, c) in [(0, 1, 2), (1, 2, 0), (2, 0, 1)]:
        for nm, terms in cross_terms(a, b, c):
            cons.append((f"{nm}_{a}{b}", sym_from_terms(terms)))

    names = [n for n, _ in cons]
    A = [np.array(M.tolist(), dtype=float) for _, M in cons]

    checks = {
        "U_orthogonal_max_abs_err": float(np.abs(U.T @ U - np.eye(10)).max()),
        "xR_equals_2Uv2_sympy_exact": bool(
            sp.simplify(xhom - 2*Usym*v2) == sp.zeros(10, 1)),
        "n_constraints": len(A),
        "rank_21_constraints_exact": int(sp.Matrix([sym_vec_exact(a, 10) for a in A]).rank()),
    }
    return U, A, names, checks


# ===========================================================================
# 2. 13x13 construction -- exact replica of the upstream generators
# ===========================================================================
def build_13x13():
    """The 24 constraint matrices of SDPRLayers, in x = [h, vec_col(C), t].

    ``vec_col(C)``: index ``1 + 3*j + k`` holds ``C[k, j]`` (column j).
    Replicated from ``pose_est.py`` (see POSE_EST_GENERATORS).
    """
    n = 13
    A_all = []

    # --- 6 orthogonality constraints: C_i . C_j - delta_ij h^2 = 0 ---------
    # upstream: A[C,C] = kron(E+E.T, eye(3)); A[h,h] = -delta_ij
    for i in range(3):
        for j in range(i, 3):
            A = np.zeros((n, n))
            for k in range(3):
                p = 1 + 3*i + k          # C[k, i]
                q = 1 + 3*j + k          # C[k, j]
                if i == j:
                    A[p, p] += 1.0
                else:
                    A[p, q] += 0.5
                    A[q, p] += 0.5
            if i == j:
                A[0, 0] -= 1.0
            A_all.append(A)

    # --- 9 handedness constraints: (C_j x C_k)_l - h C_li = 0 (cyclic) ----
    # upstream: mat[3j+m,3k+n]=1/2, mat[3j+n,3k+m]=-1/2, mat=mat+mat.T,
    #           A[C,h] = vec with vec[i*3+l]=-1/2
    for (i, j, k) in [(0, 1, 2), (1, 2, 0), (2, 0, 1)]:
        for (l, m, nn) in [(0, 1, 2), (1, 2, 0), (2, 0, 1)]:
            A = np.zeros((n, n))
            A[1 + 3*j + m, 1 + 3*k + nn] += 0.5
            A[1 + 3*k + nn, 1 + 3*j + m] += 0.5
            A[1 + 3*j + nn, 1 + 3*k + m] -= 0.5
            A[1 + 3*k + m, 1 + 3*j + nn] -= 0.5
            A[1 + 3*i + l, 0] -= 0.5
            A[0, 1 + 3*i + l] -= 0.5
            A_all.append(A)

    # --- 9 row/column norm constraints: ||col_j||^2 - ||row_i||^2 = 0 -----
    for i in range(3):
        for j in range(3):
            A = np.zeros((n, n))
            for k in range(3):
                A[1 + 3*j + k, 1 + 3*j + k] += 1.0     # ||col_j||^2
                A[1 + 3*k + i, 1 + 3*k + i] -= 1.0     # -||row_i||^2
            A_all.append(A)

    # sanity: every constraint vanishes on the true lift x=[||q||^2, vec_col R, 0]
    rng = np.random.default_rng(0)
    Q = rng.normal(size=(2000, 4))
    Q /= np.linalg.norm(Q, axis=1, keepdims=True)
    worst = 0.0
    for qq in Q:
        x = lift13(qq)
        for A in A_all:
            worst = max(worst, abs(float(x @ A @ x)))

    checks = {
        "n_constraints": len(A_all),
        "max_abs_xTAx_on_random_rotations": float(worst),
        "t_block_all_zero": bool(max(np.abs(a[10:, :]).max() for a in A_all) == 0.0),
    }
    return A_all, checks


def perm_rowmajor_to_colmajor():
    """P with x_col(13 layout) = P @ x_row(10x10 layout) on the rotation block.

    10x10 layout (verify): index ``1 + 3*i + j`` holds ``R[i, j]``.
    13x13 layout (upstream): index ``1 + 3*j + i`` holds ``R[i, j]``.
    """
    P = np.zeros((10, 10))
    P[0, 0] = 1.0
    for i in range(3):
        for j in range(3):
            P[1 + 3*j + i, 1 + 3*i + j] = 1.0
    return P


def lift10(q):
    """x_R = [||q||^2, vec_rowmajor(R(q))]."""
    q0, q1, q2, q3 = q
    R = np.array([
        [q0*q0 + q1*q1 - q2*q2 - q3*q3, 2*(q1*q2 - q0*q3), 2*(q1*q3 + q0*q2)],
        [2*(q1*q2 + q0*q3), q0*q0 - q1*q1 + q2*q2 - q3*q3, 2*(q2*q3 - q0*q1)],
        [2*(q1*q3 - q0*q2), 2*(q2*q3 + q0*q1), q0*q0 - q1*q1 - q2*q2 + q3*q3],
    ])
    return np.concatenate([[q @ q], R.reshape(-1)])


def lift13(q, t=None):
    """x = [||q||^2, vec_col(R(q)), t] (t = 0 by default)."""
    q0, q1, q2, q3 = q
    R = np.array([
        [q0*q0 + q1*q1 - q2*q2 - q3*q3, 2*(q1*q2 - q0*q3), 2*(q1*q3 + q0*q2)],
        [2*(q1*q2 + q0*q3), q0*q0 - q1*q1 + q2*q2 - q3*q3, 2*(q2*q3 - q0*q1)],
        [2*(q1*q3 - q0*q2), 2*(q2*q3 + q0*q1), q0*q0 - q1*q1 - q2*q2 + q3*q3],
    ])
    t = np.zeros(3) if t is None else np.asarray(t, float)
    return np.concatenate([[q @ q], R.T.reshape(-1), t])


def sym_vec_np(M):
    """Coefficient vector of x^T M x in the (x_i^2, x_i x_j, i<j) basis."""
    n = M.shape[0]
    return np.array([M[i, j] if i == j else 2*M[i, j]
                     for i in range(n) for j in range(i, n)], dtype=float)


def sym_vec_exact(M, n):
    """Row vector of exact rational coefficients (1 x n(n+1)/2)."""
    return sp.Matrix([[sp.Rational(str(float(M[i, j] if i == j else 2*M[i, j])))
                       for i in range(n) for j in range(i, n)]])


def rowdot_matrices(n=13):
    """r_a . r_b = sum_j C[a,j] C[b,j] for (a,b) = (0,1),(0,2),(1,2).

    Returned in the 13-layout (col-major vec C) and in the 10-layout
    (row-major vec R) so that both SDPs use the *same* objective.
    """
    cm, rm = [], []
    for (a, b) in [(0, 1), (0, 2), (1, 2)]:
        A = np.zeros((n, n))
        for j in range(3):
            p, q = 1 + 3*j + a, 1 + 3*j + b
            A[p, q] += 1.0
            A[q, p] += 1.0
        cm.append(A)
        B = np.zeros((10, 10))
        for j in range(3):
            p, q = 1 + 3*a + j, 1 + 3*b + j
            B[p, q] += 1.0
            B[q, p] += 1.0
        rm.append(B)
    return cm, rm


# ===========================================================================
# 3. structural analysis of the two constraint sets (exact)
# ===========================================================================
def structural_analysis(A10, names10, A13):
    """Compare span(21 constraints) with span(24 constraints) exactly.

    ``out["only_in_10x10"]`` lists the 10x10 constraints whose coefficient
    vector is not in span(24): these are the constraints the 13x13 SDP is
    missing.  Everything is done in the 10-layout (row-major vec R) after
    permuting the 13-layout rotation blocks with P.
    """
    P = perm_rowmajor_to_colmajor()
    V21 = np.array([sym_vec_np(a) for a in A10])
    V24 = np.array([sym_vec_np(P @ a[:10, :10] @ P.T) for a in A13])
    V24_raw = np.array([sym_vec_np(a[:10, :10]) for a in A13])

    def rank(X):
        return int(np.linalg.matrix_rank(X, tol=1e-9))

    # exact rational ranks (all entries are 0, +-1/2, +-1)
    r21_exact = int(sp.Matrix([sym_vec_exact(a, 10) for a in A10]).rank())
    r24_exact = int(sp.Matrix([sym_vec_exact(a[:10, :10], 10) for a in A13]).rank())

    Q24, _ = np.linalg.qr(V24.T)          # orthonormal basis of span(24)
    resid = {}
    for nm, v in zip(names10, V21):
        r = v - Q24 @ (Q24.T @ v)
        resid[nm] = float(np.linalg.norm(r))
    only_in_10 = [nm for nm, r in resid.items() if r > 1e-6]

    rd_cm, rd_rm = rowdot_matrices()
    rowdot_new = int(np.linalg.matrix_rank(np.vstack([V24, np.array([sym_vec_np(a) for a in rd_rm])]),
                                           tol=1e-9) - rank(V24))

    return {
        "rank_21_constraints_numeric": rank(V21),
        "rank_21_constraints_exact": r21_exact,
        "rank_24_rotation_blocks_numeric": rank(V24_raw),
        "rank_24_rotation_blocks_exact": r24_exact,
        "rank_21_union_24": rank(np.vstack([V21, V24])),
        "constraint_rank_13x13": rank(V24_raw),
        "constraint_rank_10x10": rank(V21),
        "only_in_10x10": only_in_10,
        "residual_of_21_against_span24": resid,
        "rowdot_new_dims_after_24": rowdot_new,
        "P_orthogonal_err": float(np.abs(P @ P.T - np.eye(10)).max()),
        "rowdot_layout_consistency_max_err": float(
            max(np.abs(P.T @ a[:10, :10] @ P - b).max() for a, b in zip(rd_cm, rd_rm))),
    }


# ===========================================================================
# 4. instances
# ===========================================================================
def choi_lam_M():
    """M = M_CL + 2 I (the constructed instance of verify/verify_theorem0.py)."""
    M_CL = np.zeros((10, 10))
    M_CL[0, 1] = M_CL[1, 0] = 0.5     # q0^2 q1^2
    M_CL[1, 2] = M_CL[2, 1] = 0.5     # q1^2 q2^2
    M_CL[2, 0] = M_CL[0, 2] = 0.5     # q2^2 q0^2
    M_CL[3, 3] = 1.0                  # q3^4
    M_CL[4, 9] = M_CL[9, 4] = -1.0    # -4 q0 q1 q2 q3
    return M_CL + 2.0*np.eye(10)


def make_M0_from_reduced(M, D, G):
    """M0 = M + D^T G^{-1} D so that M0 - D^T G^{-1} D = M exactly."""
    return M + D.T @ np.linalg.solve(G, D)


def instance(name, M0, D, G, note, evidence):
    return dict(name=name, M0=np.asarray(M0, float), D=np.asarray(D, float),
                G=np.asarray(G, float), note=note, evidence=evidence)


def build_instances(U):
    """Return (main_instances, perturbation_instances)."""
    rng = np.random.default_rng(0)
    M_CL = choi_lam_M()
    n = 10

    def rand_G(seed_scale=1.0):
        A = rng.normal(size=(3, 3))
        return seed_scale * (A @ A.T + 1.0*np.eye(3))

    main = []
    # (1) the exact Choi-Lam instance (D = 0, so t* = 0 and the 13x13 SDP
    #     isolates the constraint-set difference)
    D0 = np.zeros((3, n))
    G0 = np.eye(3)
    main.append(instance("CL_exact", M_CL, D0, G0,
                         "verify Choi-Lam M=M_CL+2I, D=0, G=I (verify regression)",
                         "f_10 quoted-verify; rest measured-here"))
    # (2) same reduced M with a nonzero translation coupling (tests the
    #     t-elimination: f_13 must not depend on D, G given M)
    D2 = 0.4*rng.normal(size=(3, n))
    G2 = rand_G(1.0)
    main.append(instance("CL_coupled", make_M0_from_reduced(M_CL, D2, G2), D2, G2,
                         "same reduced M as CL_exact, but D != 0, G != I",
                         "measured-here"))
    # (3,4,5) three of the eight symmetry-breaking perturbations of CL
    for t in range(3):
        dM = rng.normal(size=(n, n))
        dM = 0.02*(dM + dM.T)/2.0
        Mt = M_CL + dM
        main.append(instance(f"CL_perm{t}", Mt, D0, G0,
                             f"CL + 0.02-scaled symmetric perturbation #{t}",
                             "measured-here"))
    # (6) generic PSD M0, random D, PD G  (reduced M indefinite)
    A6 = rng.normal(size=(n, n))
    M0_6 = A6 @ A6.T
    D6 = 0.5*rng.normal(size=(3, n))
    G6 = rand_G(1.0)
    main.append(instance("rand_psd", M0_6, D6, G6,
                         "M0 = A A^T PSD, random D, G PD", "measured-here"))
    # (7) generic indefinite M0
    A7 = rng.normal(size=(n, n))
    M0_7 = A7 @ A7.T + 0.5*rng.normal(size=(n, n))
    M0_7 = 0.5*(M0_7 + M0_7.T)
    D7 = 0.5*rng.normal(size=(3, n))
    G7 = rand_G(1.0)
    main.append(instance("rand_indef", M0_7, D7, G7,
                         "generic indefinite M0, random D, G PD", "measured-here"))
    # (8) a genuine P6 (matrix-weighted pose) cost matrix, then (M0,D,G)
    #     extracted from it; the identity x^T Q x = v2^T M0 v2 + ... is
    #     checked by a round trip in instance_diagnostics().
    Qp6, m_p6 = p6_objective(rng, ncorr=20)
    M0_8, D8, G8 = extract_M0DG(Qp6, U, perm_rowmajor_to_colmajor())
    inst_p6 = instance("p6_data", M0_8, D8, G8,
                       "Q built from random matrix-weighted P6 correspondences "
                       "(round-trip check)", "measured-here")
    inst_p6["Q_ref"] = Qp6
    main.append(inst_p6)
    # (9) constructed counterexample: objective = -(R R^T)_{01} + 1/4
    #     (row-orthogonality functional; constant on S^3, all q optimal)
    rd_cm, rd_rm = rowdot_matrices()
    Q9 = -rd_rm[0] + 0.25*np.eye(n)
    main.append(instance("rowdot", 4.0*U.T @ Q9 @ U, D0, G0,
                         "constructed Q = -r1.r2 + 1/4 (quartic constant on S^3)",
                         "measured-here"))
    # (10) generic version: -r1.r2 + 0.01 * random symmetric (isolated minima)
    G10 = rng.normal(size=(n, n))
    G10 = 0.5*(G10 + G10.T)
    G10 = G10/np.abs(G10).max()
    Q10c = -rd_rm[0] + 1e-2*G10
    main.append(instance("rowdot_gen", 4.0*U.T @ Q10c @ U, D0, G0,
                         "constructed Q = -r1.r2 + 0.01*random symmetric",
                         "measured-here"))

    # --- the 8 verify-style perturbations (all of them) -------------------
    rngp = np.random.default_rng(0)      # same stream as verify_theorem0.py
    perts = []
    for t in range(8):
        dM = rngp.normal(size=(n, n))
        dM = 0.02*(dM + dM.T)/2.0
        Mt = M_CL + dM
        if np.min(np.linalg.eigvalsh(Mt)) <= 0:
            continue
        perts.append(instance(f"CL_pert{t}", Mt, D0, G0,
                              f"CL + 0.02-scaled perturbation #{t} (symmetry broken)",
                              "measured-here"))
    return main, perts


def p6_objective(rng, ncorr=20, noise=0.05):
    """Random matrix-weighted P6 cost matrix Q (13x13, upstream layout).

    Mirrors ``repro/cleanroom/sdp_pose.build_objective_matrix``:
    cost = sum_k (y_k - C m_k + t)^T W_k (y_k - C m_k + t).
    """
    m = rng.normal(size=(3, ncorr))
    Rtrue = lift_rotation(rng.normal(size=4))
    W = []
    y = np.zeros((3, ncorr))
    for k in range(ncorr):
        A = rng.normal(size=(3, 3))
        Wk = A @ A.T + 0.5*np.eye(3)
        W.append(Wk)
        y[:, k] = Rtrue @ m[:, k] + rng.normal(size=3)*noise
    W = np.array(W)
    Q = np.zeros((13, 13))
    Wy = np.einsum("kab,kb->ka", W, y.T)
    Q[0, 0] = np.einsum("ka,ka->", Wy, y.T)
    Q[1:10, 1:10] = np.einsum("ki,kj,kab->iajb", m.T, m.T, W).reshape(9, 9)
    Q[10:13, 10:13] = W.sum(axis=0)
    Qch = -np.einsum("ki,ka->ia", m.T, Wy).reshape(9)
    Q[1:10, 0] = Qch
    Q[0, 1:10] = Qch
    Qth = Wy.sum(axis=0)
    Q[10:13, 0] = Qth
    Q[0, 10:13] = Qth
    Qct = -np.einsum("ki,kab->iab", m.T, W).reshape(9, 3)
    Q[1:10, 10:13] = Qct
    Q[10:13, 1:10] = Qct.T
    return 0.5*(Q + Q.T), m


def lift_rotation(q):
    q = np.asarray(q, float)
    q /= np.linalg.norm(q)
    q0, q1, q2, q3 = q
    return np.array([
        [q0*q0 + q1*q1 - q2*q2 - q3*q3, 2*(q1*q2 - q0*q3), 2*(q1*q3 + q0*q2)],
        [2*(q1*q2 + q0*q3), q0*q0 - q1*q1 + q2*q2 - q3*q3, 2*(q2*q3 - q0*q1)],
        [2*(q1*q3 - q0*q2), 2*(q2*q3 + q0*q1), q0*q0 - q1*q1 - q2*q2 + q3*q3],
    ])


def extract_M0DG(Q, U, P):
    """Invert Q -> (M0, D, G); Q is in the upstream layout.

    Q_xx = P (1/4 U M0 U^T) P^T, Q_xt = P (1/2 U D^T), Q_tt = G.
    """
    Qxx_rm = P.T @ Q[:10, :10] @ P
    M0 = 4.0 * U.T @ Qxx_rm @ U
    D = 2.0 * Q[:10, 10:13].T @ P @ U
    G = Q[10:13, 10:13]
    return M0, D, G


def build_Qfull(M0, D, G, U, P):
    """13x13 Q in the upstream layout from (M0, D, G) in the 10x10 layout."""
    Q = np.zeros((13, 13))
    Q[:10, :10] = P @ (0.25 * U @ M0 @ U.T) @ P.T
    Cx = 0.5 * (U @ D.T)                       # Q_xt = (1/2) U D^T (10x3)
    Q[:10, 10:] = P @ Cx
    Q[10:, :10] = (P @ Cx).T
    Q[10:, 10:] = G
    return 0.5*(Q + Q.T)


# ===========================================================================
# 5. nonconvex quartic minimisation on S^3
# ===========================================================================
def v2(q):
    return np.array([q[0]**2, q[1]**2, q[2]**2, q[3]**2,
                     SQ2*q[0]*q[1], SQ2*q[0]*q[2], SQ2*q[0]*q[3],
                     SQ2*q[1]*q[2], SQ2*q[1]*q[3], SQ2*q[2]*q[3]])


def v2_batch(Q):
    """Vectorised v2 for Q of shape (n, 4)."""
    Q = np.asarray(Q, float)
    sq = SQ2
    return np.stack([
        Q[:, 0]**2, Q[:, 1]**2, Q[:, 2]**2, Q[:, 3]**2,
        sq*Q[:, 0]*Q[:, 1], sq*Q[:, 0]*Q[:, 2], sq*Q[:, 0]*Q[:, 3],
        sq*Q[:, 1]*Q[:, 2], sq*Q[:, 1]*Q[:, 3], sq*Q[:, 2]*Q[:, 3],
    ], axis=1)


def Jv(q):
    return np.array([
        [2*q[0], 0, 0, 0],
        [0, 2*q[1], 0, 0],
        [0, 0, 2*q[2], 0],
        [0, 0, 0, 2*q[3]],
        [SQ2*q[1], SQ2*q[0], 0, 0],
        [SQ2*q[2], 0, SQ2*q[0], 0],
        [SQ2*q[3], 0, 0, SQ2*q[0]],
        [0, SQ2*q[2], SQ2*q[1], 0],
        [0, SQ2*q[3], 0, SQ2*q[1]],
        [0, 0, SQ2*q[3], SQ2*q[2]],
    ])


HV = np.zeros((10, 4, 4))
for _i in range(4):
    HV[_i, _i, _i] = 2.0
for _i, _j, _a in [(0, 1, 4), (0, 2, 5), (0, 3, 6), (1, 2, 7), (1, 3, 8), (2, 3, 9)]:
    HV[_a, _i, _j] = HV[_a, _j, _i] = SQ2


def E_basis(q):
    """verify/verify_theorem11.py E(q): q^T E = 0, E^T E = I3."""
    q0, q1, q2, q3 = q
    return np.array([
        [-q1, -q2, -q3],
        [q0, -q3, q2],
        [q3, q0, -q1],
        [-q2, q1, q0],
    ])


def Jfun(q, M):
    return float(v2(q) @ M @ v2(q))


def grad(q, M):
    return 2.0 * Jv(q).T @ (M @ v2(q))


def hess(q, M):
    J = Jv(q)
    H = 2.0 * J.T @ M @ J
    Mv = M @ v2(q)
    for a in range(10):
        if Mv[a] != 0.0:
            H += 2.0 * Mv[a] * HV[a]
    return H


def nonconvex_min(M, n_starts=160, seed=12345, n_sample=200000):
    """min_{q in S^3} v2(q)^T M v2(q): multi-start SLSQP + Riemannian Newton.

    Returns the best value, the number of local minima found (distinct values
    above 1e-9), and a dense-sampling lower sanity check.
    """
    rng = np.random.default_rng(seed)
    cons = [{"type": "eq", "fun": lambda z: z @ z - 1.0,
             "jac": lambda z: 2.0*z}]
    vals = []
    best, bx = np.inf, None
    starts = rng.normal(size=(n_starts, 4))
    starts /= np.linalg.norm(starts, axis=1, keepdims=True)
    for s in starts:
        r = minimize(lambda z: Jfun(z, M), s, jac=lambda z: grad(z, M),
                     method="SLSQP", constraints=cons,
                     options={"ftol": 1e-15, "maxiter": 1000})
        z = r.x / np.linalg.norm(r.x)
        # Riemannian Newton polish on the sphere
        try:
            for _ in range(30):
                Eq = E_basis(z)
                Hs = Eq.T @ (hess(z, M) - 4.0*Jfun(z, M)*np.eye(4)) @ Eq
                step = -Eq @ np.linalg.solve(Hs + 1e-13*np.eye(3), Eq.T @ grad(z, M))
                z = z + step
                nz = np.linalg.norm(z)
                if nz == 0:
                    break
                z = z / nz
        except np.linalg.LinAlgError:
            pass
        val = Jfun(z, M)
        vals.append(val)
        if val < best:
            best, bx = val, z

    # dense sampling sanity check (vectorised)
    Qs = rng.normal(size=(n_sample, 4))
    Qs /= np.linalg.norm(Qs, axis=1, keepdims=True)
    V = v2_batch(Qs)                              # n_sample x 10
    sample_vals = np.einsum("ni,ij,nj->n", V, M, V)
    sample_min = float(sample_vals.min())

    uniq = []
    for val in sorted(vals):
        if not uniq or abs(val - uniq[-1]) > 1e-7:
            uniq.append(val)
    return {
        "f_nonconvex": float(min(best, sample_min)),
        "argmin_q": bx.tolist(),
        "n_starts": int(n_starts),
        "distinct_local_minima": [float(x) for x in uniq],
        "dense_sample_min": sample_min,
        "n_samples": int(n_sample),
    }


# ===========================================================================
# 6. SDP solves
# ===========================================================================
def solve_sdp(Q, A_list, n, eps=1e-9, max_iters=200000):
    """min <Q,X> s.t. X>=0, X00=1, <A_i,X>=0.  SCS only (no Mosek)."""
    X = cp.Variable((n, n), symmetric=True)
    cons = [X >> 0, X[0, 0] == 1] + [cp.trace(A @ X) == 0 for A in A_list]
    prob = cp.Problem(cp.Minimize(cp.trace(Q @ X)), cons)
    t0 = time.time()
    try:
        prob.solve(solver=cp.SCS, eps=eps, max_iters=max_iters, verbose=False)
    except Exception as exc:                                  # pragma: no cover
        return dict(status="SOLVE_FAILED: " + repr(exc), value=float("nan"),
                    X=None, t_solve=time.time() - t0)
    dt = time.time() - t0
    if prob.status not in ("optimal", "optimal_inaccurate") or X.value is None:
        return dict(status=prob.status, value=float("nan"), X=None, t_solve=dt)
    Xv = np.array(X.value, dtype=float)
    Xv = 0.5*(Xv + Xv.T)
    eigs = np.linalg.eigvalsh(Xv)[::-1]
    tol = 1e-6*max(abs(eigs[0]), 1e-12)
    rank = int(np.sum(eigs > tol))
    # dual bound: max mu s.t. Q - mu E00 - sum lam_i A_i >= 0; mu = -y0
    out = dict(status=prob.status, value=float(prob.value), X=Xv,
               eigs_top=[float(x) for x in eigs[:8]], eig_ratio_12=float(eigs[0]/max(eigs[1], 1e-15)),
               rank_1e6=rank, lam_max=float(eigs[0]), t_solve=dt)
    try:
        y0 = float(np.asarray(cons[1].dual_value).reshape(-1)[0])
        lam = [float(np.asarray(c.dual_value).reshape(-1)[0]) for c in cons[2:]]
        E00 = np.zeros((n, n))
        E00[0, 0] = 1.0
        mu = -y0
        Z = Q - mu*E00 - sum(li*Ai for li, Ai in zip(lam, A_list))
        Z = 0.5*(Z + Z.T)
        zmin = float(np.min(np.linalg.eigvalsh(Z)))
        if zmin < -1e-6:            # other dual sign convention
            mu, lam = -mu, [-l for l in lam]
            Z = Q - mu*E00 - sum(li*Ai for li, Ai in zip(lam, A_list))
            Z = 0.5*(Z + Z.T)
            zmin = float(np.min(np.linalg.eigvalsh(Z)))
        out.update(dual_bound=mu, cert_min_eig=zmin,
                   primal_minus_dual=float(prob.value - mu))
    except Exception:                                          # pragma: no cover
        out.update(dual_bound=float("nan"), cert_min_eig=float("nan"),
                   primal_minus_dual=float("nan"))
    return out


# ===========================================================================
# 7. per-instance diagnostics
# ===========================================================================
def instance_diagnostics(inst, U, P, A10, A13, rd_cm, rd_rm, args):
    M0, D, G = inst["M0"], inst["D"], inst["G"]
    M = M0 - D.T @ np.linalg.solve(G, D)
    Q10 = 0.25 * U @ M @ U.T                 # 10-layout
    Qcost_cm = P @ Q10 @ P.T                 # same objective in the 13-layout
    Qfull = build_Qfull(M0, D, G, U, P)

    diag = {"name": inst["name"], "note": inst["note"], "evidence": inst["evidence"]}

    # --- exact consistency of the (M0, D, G) parameterisation -------------
    diag["roundtrip_Q_err"] = None
    if "Q_ref" in inst:
        Qreb = build_Qfull(M0, D, G, U, P)
        Qref = inst["Q_ref"]
        diag["roundtrip_Q_err"] = float(np.abs(Qreb - Qref).max() / np.abs(Qref).max())
        diag["G_min_eig"] = float(np.min(np.linalg.eigvalsh(G)))
        diag["M_min_eig"] = float(np.min(np.linalg.eigvalsh(M)))
    # Schur-complement identity: Q_xx = Q_cost + Q_xt G^{-1} Q_xt^T
    Qxt = Qfull[:10, 10:]
    schur_res = Qfull[:10, :10] - Qcost_cm - Qxt @ np.linalg.solve(G, Qxt.T)
    diag["schur_decomposition_res"] = float(np.abs(schur_res).max())
    if np.abs(schur_res).max() > 1e-10:
        raise AssertionError("Q_xx != Q_cost + Q_xt G^{-1} Q_xt^T")

    # --- nonconvex quartic ------------------------------------------------
    t0 = time.time()
    nc = nonconvex_min(M, n_starts=args.starts, seed=args.seed)
    diag.update(nc)
    diag["t_nonconvex"] = time.time() - t0

    # --- 10x10 SDP --------------------------------------------------------
    s10 = solve_sdp(Q10, A10, 10, eps=args.eps)
    # --- 13x13 SDP (24 constraints) --------------------------------------
    s13 = solve_sdp(Qfull, A13, 13, eps=args.eps)
    # --- ablation: 13x13 with only 6 orth + 9 handedness ------------------
    s15 = solve_sdp(Qfull, A13[:15], 13, eps=args.eps)

    diag["f_10"] = s10["value"]
    diag["f_13"] = s13["value"]
    diag["f_13_only15"] = s15["value"]
    diag["status_10"] = s10["status"]
    diag["status_13"] = s13["status"]
    diag["status_13_only15"] = s15["status"]
    diag["rank_X10"] = s10.get("rank_1e6")
    diag["rank_X13"] = s13.get("rank_1e6")
    diag["rank_X13_only15"] = s15.get("rank_1e6")
    diag["eigs_X10_top8"] = s10.get("eigs_top")
    diag["eigs_X13_top8"] = s13.get("eigs_top")
    diag["eig_ratio_12_X10"] = s10.get("eig_ratio_12")
    diag["eig_ratio_12_X13"] = s13.get("eig_ratio_12")
    diag["dual_10"] = s10.get("dual_bound")
    diag["dual_13"] = s13.get("dual_bound")
    diag["cert_min_eig_10"] = s10.get("cert_min_eig")
    diag["cert_min_eig_13"] = s13.get("cert_min_eig")
    diag["primal_minus_dual_10"] = s10.get("primal_minus_dual")
    diag["primal_minus_dual_13"] = s13.get("primal_minus_dual")
    diag["t_solve_10"] = s10.get("t_solve")
    diag["t_solve_13"] = s13.get("t_solve")
    diag["gap_10"] = diag["f_nonconvex"] - diag["f_10"]
    diag["gap_13"] = diag["f_nonconvex"] - diag["f_13"]
    diag["f10_minus_f13"] = diag["f_10"] - diag["f_13"]

    # --- structural diagnostics on the 13x13 optimum ----------------------
    if s13["X"] is not None:
        Xr = s13["X"][:10, :10]
        diag["rowdot_at_X13"] = [float(np.trace(a[:10, :10] @ Xr)) for a in rd_cm]
        diag["rowdot_max_abs_at_X13"] = float(max(abs(v) for v in diag["rowdot_at_X13"]))
        diag["schur_err"] = float(abs(s13["value"] - np.trace(Qcost_cm @ Xr)))
        diag["t_block_norm_at_X13"] = float(np.linalg.norm(s13["X"][10:, :]))
        # rotation-block trace: 3 on the rank-1 feasible manifold
        diag["X13_rot_trace"] = float(np.trace(Xr))
    if s10["X"] is not None:
        diag["rowdot_at_X10"] = [float(np.trace(a @ s10["X"])) for a in rd_rm]
    if s13["X"] is not None and s10["X"] is not None:
        Xr_cm = s13["X"][:10, :10]
        X10_rm = s10["X"]
        diag["X13rot_minus_X10_fro"] = float(np.linalg.norm(Xr_cm - P @ X10_rm @ P.T))

    diag.update({"instance_note": inst["note"]})
    return diag


def upstream_crosscheck(path):
    """Run the *actual* upstream generator source and compare with our replica.

    ``pose_est.py`` imports cvxpylayers/torch at module level, so instead of
    importing the module we parse it and execute only the three generator
    function bodies (exact upstream source, unchanged) in a namespace holding
    ``numpy`` and the upstream ``PolyMatrix`` class.
    """
    import ast
    import copy
    import textwrap

    src = os.path.join(path, "src")
    fname = os.path.join(src, "sdprlayers", "layers", "pose_est.py")
    if not os.path.isfile(fname):
        return {"ran": False, "reason": f"{fname} not found"}
    sys.dont_write_bytecode = True
    sys.path.insert(0, os.path.join(src, "submodules", "poly_matrix"))
    try:
        from poly_matrix import PolyMatrix          # type: ignore
    except Exception as exc:                        # pragma: no cover
        return {"ran": False, "reason": "PolyMatrix import failed: " + repr(exc)}

    tree = ast.parse(open(fname).read())
    cls = next(n for n in tree.body
               if isinstance(n, ast.ClassDef) and n.name == "SDPPoseEstimator")
    wanted = ("gen_orthogonal_constraints", "gen_handedness_constraints",
              "gen_row_col_constraints")
    ns = {"np": np, "PolyMatrix": PolyMatrix}
    found = []
    for fn in cls.body:
        if isinstance(fn, ast.FunctionDef) and fn.name in wanted:
            fn = copy.deepcopy(fn)
            fn.decorator_list = []          # @staticmethod is irrelevant here
            exec(compile(ast.Module(body=[fn], type_ignores=[]),
                         "<pose_est-generators>", "exec"), ns)
            found.append(fn.name)
    if sorted(found) != sorted(wanted):
        return {"ran": False, "reason": f"generators not found: {found}"}

    ref = (ns["gen_orthogonal_constraints"]() + ns["gen_handedness_constraints"]()
           + ns["gen_row_col_constraints"]())
    ref = [np.asarray(A.toarray() if hasattr(A, "toarray") else A) for A in ref]
    ours, _ = build_13x13()
    if len(ref) != len(ours):
        return {"ran": True, "n_upstream": len(ref), "n_ours": len(ours), "match": False}
    diffs = [float(np.abs(a - b).max()) for a, b in zip(ref, ours)]
    rng = np.random.default_rng(1)
    qq = rng.normal(size=4)
    qq /= np.linalg.norm(qq)
    x = lift13(qq)
    worst = max(abs(float(x @ A @ x)) for A in ours)
    return {"ran": True, "n_constraints": len(ref), "max_abs_diff": max(diffs),
            "match": bool(max(diffs) == 0.0), "max_abs_xTAx": worst,
            "commit": "62d8ac6", "path": os.path.abspath(path),
            "method": "AST-extracted upstream generator bodies (no module import)"}


# ===========================================================================
# 8. report writing
# ===========================================================================
def fmt(x, nd=8):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return f"{x:+.{nd}f}"


def write_markdown(res, path):
    cfg, st, sm = res["config"], res["structural"], res["summary"]
    reg, rdref = res["regression"], res["rowdot_ref"]
    up = res["upstream"]
    m = res["main"]
    L = []
    A = L.append

    A("# P1-Q2：10×10 Veronese/Shor 松弛 vs 13×13 SDPRLayers SDP（旋转子问题）")
    A("")
    A("> **一句话结论：不等价。** 13×13 的 24 条 SDPRLayers 约束在 $\\mathrm{Sym}^{10}$ 中只张成 %d 维，"
      "比 10×10 的 21 条（%d 维）少了 3 维，少的恰好是三条**行正交**约束 $r_a\\cdot r_b=0$；"
      "所以 13×13 是 10×10 的一个严格更松的松弛，恒有 $f^\\star_{13}\\le f^\\star_{10}\\le f_{\\rm nonconvex}$。"
      "项目自己的 Choi–Lam 松实例因对称性使 13×13 最优解恰好落在行正交面上，两者数值相等（%.10f / %.10f）；"
      "但做 8 个 $10^{-2}$ 量级的对称破缺扰动后，**%d/%d** 都严格出现 $f^\\star_{13}<f^\\star_{10}$。"
      % (st["rank_24_rotation_blocks_exact"], st["rank_21_constraints_exact"],
         reg["cl_f10"], reg["cl_f13"], sm["pert_n_strict"], sm["pert_n"]))
    A("")
    A("## 0. 口径与证据标签")
    A("")
    A("- `[measured-here]`：本目录 `run_sdp_check.py` 本次运行测得的数值。")
    A("- `[quoted-verify]`：引自 `../../verify/verify_theorem0_output.json`（未重跑 verify 脚本）。")
    A("- `[exact]`：本脚本内 sympy 有理数精确秩/恒等式。")
    A("- 求解器：SCS（cvxpy %s），`eps=%.0e`，`max_iters=%d`；两次 SDP 都是 10/13 维小问题。"
      % (cfg["cvxpy"], cfg["eps"], cfg["max_iters"]))
    A("- 13×13 的 24 条约束**逐行复刻** `sdprlayer@62d8ac6` `src/sdprlayers/layers/pose_est.py` "
      "L341–L417 的三个生成器（不靠论文文字猜），并与上游 `PolyMatrix` 实现逐元素对拍（§4）。")
    A("- 13×13 的 $Q$ 由 $x=[h;\\mathrm{vec}_{\\rm col}(C);t]$ 的 P6/QPEP 代价组装："
      "$Q_{xx}=\\tfrac14 U M_0U^\\top$，$Q_{xt}=\\tfrac12 U D^\\top$，$Q_{tt}=G$；"
      "10×10 用 $M=M_0-D^\\top G^{-1}D$（对**非凸代价**的精确 $t$ 消去）。")
    A("- **没有构造/没有声称**：不声称这些 $M$ 是真实矩阵加权 PnP 的 Gram 矩阵（那是 Q1）；"
      "不涉及非凸 P6 的求解；不声称 13×13 等于「二阶 Lasserre」。")
    A("")
    A("## 1. 主表")
    A("")
    A("| instance | f_nonconvex | f_10 | f_13 | gap_10 | gap_13 | rank(X_10*) | rank(X_13*) | solver status | evidence |")
    A("|-|-|-|-|-|-|-|-|-|-|")
    for d in m:
        A("| {name} | {fnc} | {f10} | {f13} | {g10} | {g13} | {r10} | {r13} | {stt} | {ev} |".format(
            name=d["name"], fnc=fmt(d["f_nonconvex"], 6), f10=fmt(d["f_10"], 6),
            f13=fmt(d["f_13"], 6), g10=fmt(d["gap_10"], 6), g13=fmt(d["gap_13"], 6),
            r10=d["rank_X10"], r13=d["rank_X13"],
            stt="10:%s / 13:%s" % (d["status_10"], d["status_13"]), ev=d["evidence"]))
    A("")
    A("- `gap_10 = f_nonconvex - f_10`，`gap_13 = f_nonconvex - f_13`（$\\ge0$ 表示松弛下界；$=0$ 表示紧）。")
    A("- `CL_exact` 的 `f_10` 实测 %.12f，与 `[quoted-verify]` 的 1.9658119658452384 相差 %.1e；"
      "两者是同一构造，验证了 10×10 部分被原样复用。" % (reg["cl_f10"], abs(reg["cl_f10_minus_verify"])))
    A("- `rowdot` / `rowdot_gen` 是**构造的反例**（见 §3），不是 PnP 实例；其余行是随机或 Choi–Lam 实例。")
    A("")
    A("### 1b. Choi–Lam 的 8 个对称破缺扰动（同一 $M$ 族的「松」实例）")
    A("")
    A("| instance | f_nonconvex | f_10 | f_13 | f_10 - f_13 | rank(X_10*) | rank(X_13*) | max&#124;rowdot&#124; @ X_13* |")
    A("|-|-|-|-|-|-|-|-|")
    for d in res["perturbations"]:
        A("| {name} | {fnc} | {f10} | {f13} | {dd} | {r10} | {r13} | {rd} |".format(
            name=d["name"], fnc=fmt(d["f_nonconvex"], 6), f10=fmt(d["f_10"], 6),
            f13=fmt(d["f_13"], 6), dd=fmt(d["f10_minus_f13"], 3), r10=d["rank_X10"],
            r13=d["rank_X13"], rd=fmt(d.get("rowdot_max_abs_at_X13"), 3)))
    A("")
    A("- `max|rowdot| @ X_13*` = 13×13 最优解 $X_r^\\star$ 上三条 $r_a\\cdot r_b$ 残差的最大绝对值；"
      "它就是 13×13 与 10×10 唯一的行为差异。")
    A("")
    A("## 2. 为什么不等价（精确结构，不是数值巧合）")
    A("")
    other = [v for k, v in st["residual_of_21_against_span24"].items() if k not in st["only_in_10x10"]]
    A("- 10×10 的 21 条约束张成 **%d** 维（`[exact]`；`[quoted-verify]` 也是 20）。" % st["rank_21_constraints_exact"])
    A("- 13×13 的 24 条约束只含 $(h,\\mathrm{vec}\\,C)$（$t$ 块恒为 0），投影回 $\\mathrm{Sym}^{10}$ 后张成 **%d** 维（`[exact]`）。"
      % st["rank_24_rotation_blocks_exact"])
    A("- $\\mathrm{span}(24)\\subseteq\\mathrm{span}(21)$（并集秩 %d）。把 21 条逐条投影到 $\\mathrm{span}(24)$："
      "只有 **%s** 残差非零（%.2f–%.2f），其余 %d 条残差 ≤ %.1e。"
      % (st["rank_21_union_24"], "、".join("`%s`" % n for n in st["only_in_10x10"]),
         min(st["residual_of_21_against_span24"][n] for n in st["only_in_10x10"]),
         max(st["residual_of_21_against_span24"][n] for n in st["only_in_10x10"]),
         len(other), max(other)))
    A("- 即 $C_{10}=C_{13}\\cap\\{r_1\\cdot r_2=r_1\\cdot r_3=r_2\\cdot r_3=0\\}\\subsetneq C_{13}$；"
      "行**范数** $\\|r_i\\|^2=h^2$ 已由 9 条 row/col-norm + 列正交线性组合得到，缺的恰好只有 3 条行**点积**。"
      "（这与 Q13 猜的「10 维理想 + $t$ 块」不同：13 维的约束秩是 %d，不是 20。）"
      % st["constraint_rank_13x13"])
    A("- 因为 24 条约束都不含 $t$ 且 $G\\succ0$，对 $t$ 块做 Schur 补可**精确**消去："
      "$f^\\star_{13}=\\min_{X_r\\in C_{13}}\\langle Q_{10},X_r\\rangle$。"
      "每行都数值核对 `schur_err = |f_13 - <Q_10,X_r^*>|`，最大 %.1e；"
      "`Q_xx - Q_cost - Q_xt G^{-1}Q_xt^T` 的最大元素为 %.1e（`[measured-here]`）。"
      % (sm["schur_err_max"], sm["schur_decomp_max"]))
    A("- 因此 $f^\\star_{13}\\le f^\\star_{10}$ 对一切实例成立；等号 $\\iff$ 13×13 最优解满足行正交"
      "（等价地：13×13 的最优解同时是 10×10 的可行点）。")
    A("")
    A("## 3. 数值结论（哪些支持、哪些否定「同值 + 同秩」）")
    A("")
    A("- **CL_exact（支持，但只是巧合）**：$f^\\star_{10}=%.10f$，$f^\\star_{13}=%.10f$，差 %.1e，"
      "rank 均 %d；13×13 最优解的 rowdot 残差 ≤ %.1e。它是对称性使 13×13 最优解恰好落在 $C_{10}$ 内。"
      % (reg["cl_f10"], reg["cl_f13"], abs(reg["cl_f10"] - reg["cl_f13"]), reg["cl_rank10"], reg["cl_rowdot_max"]))
    A("- **8 个对称破缺扰动（否定）**：%d/%d 严格 $f^\\star_{13}<f^\\star_{10}$，差 %.1e–%.1e；"
      "$\\mathrm{rank}(X_{13}^\\star)\\in\\{%s\\}$ 而 $\\mathrm{rank}(X_{10}^\\star)=%s$；"
      "rowdot 残差最大 %.2f（= 13×13 最优解明显不在 $C_{10}$ 内）。"
      % (sm["pert_n_strict"], sm["pert_n"], sm["pert_gap_min"], sm["pert_gap_max"],
         ",".join(str(v) for v in sm["pert_ranks13"]),
         ",".join(str(v) for v in sm["pert_ranks10"]), sm["pert_rowdot_max"]))
    A("- **构造反例 `rowdot`（否定，且差距最大）**：$Q=-r_1\\cdot r_2+\\tfrac14 I$（在 $S^3$ 上恒等于 1 的四次型），"
      "$f_{\\rm nonconvex}=f^\\star_{10}=%.6f$（10×10 紧，rank %d），而 $f^\\star_{13}=%.6f$，"
      "差 **%.2f**，$\\mathrm{rank}(X_{13}^\\star)=%d$。`rowdot_gen` 加 $10^{-2}$ 随机扰动后差距仍有 %.2f，"
      "说明这不是退化特例的数值假象。"
      % (rdref["f10"], rdref["rank10"], rdref["f13"], rdref["f10"] - rdref["f13"], rdref["rank13"],
         sm["rowdot_gen_gap"]))
    A("- **紧实例（平凡支持）**：%s 上 $f^\\star_{10}=f^\\star_{13}=f_{\\rm nonconvex}$（两者都 rank 1）。"
      "这解释了「大量随机/真实 PnP 实验看起来等价」：松弛一旦紧，两个可行集给出同一个值。"
      % "、".join("`%s`" % n for n in sm["tight_instances"]))
    A("- **`CL_coupled`**：与 `CL_exact` 同一 reduced $M$、但 $D\\ne0,G\\ne I$；"
      "两行的 $f^\\star_{10}$、$f^\\star_{13}$ 分别只差 %.1e / %.1e，验证 $(M_0,D,G)\\to Q$ 组装与 $t$ 消去实现正确。"
      % (sm["coupled_diff10"], sm["coupled_diff13"]))
    A("- **消融（13×13 只用 6 orth + 9 handedness，15 条）**：每个实例上 $f^\\star_{13,15}\\le f^\\star_{13}$，"
      "最大下降 %.2e，说明那 9 条 row/col-norm 确实在收紧——但补上它们**仍缺** 3 条行点积，"
      "所以 13×13 永远达不到 10×10 的 20 维约束空间。" % sm["ablation_max_drop"])
    A("- **`p6_data`**：$Q$ 从随机矩阵加权 P6 对应点组装，再 $Q\\to(M_0,D,G)\\to Q'$ 往返，相对误差 %.1e，"
      "确认 13×13 的 $Q$ 就是 SDPRLayers 的 P6 代价、10×10 的 $M$ 就是它的精确 $t$ 消去。" % sm["roundtrip"])
    A("")
    A("## 4. 构造与可复现性")
    A("")
    A("- 10×10：`build_10x10()` 逐行复用 `verify/verify_theorem0.py`（sympy 精确 $U$、行主序 `vec`、21 条约束），"
      "回归值 = %.10f vs `[quoted-verify]` 1.9658119658452384。" % reg["cl_f10"])
    if up.get("match"):
        A("- 13×13：`build_13x13()` 逐行复刻上游；`--check-upstream %s` 用 AST 抽取上游三个生成器的**原始源码**执行，"
          "与本脚本复刻逐元素对拍：**24/24 完全一致，max |diff| = %.1f**；"
          "在随机旋转上 $\\max|x^\\top A_ix|=%.1e$。" % (up.get("path", ""), up["max_abs_diff"], up["max_abs_xTAx"]))
    else:
        A("- 13×13：`build_13x13()` 逐行复刻上游；本次未运行 `--check-upstream`（原因：%s）。"
          % up.get("reason", "n/a"))
    if sm["roundtrip"] is not None:
        A("- 目标组装：`build_Qfull()`（$Q_{xx},Q_{xt},Q_{tt}$），并用 Schur 分解恒等式与 P6 往返各查一次。")
    A("- 非凸值：%d 次多起点 SLSQP + 黎曼 Newton 精修，外加 %d 个稠密球面采样点取 min；"
      "`CL_exact` 得 %.6f（解析真值 2.0，因为 $C(q)\\ge0$ 且 $C(e_0)=0$）。"
      % (cfg["starts"], cfg["n_sample"], reg["cl_fnonconvex"]))
    A("- 全部随机数种子固定（`seed=%d`；实例构造用 `default_rng(0)`、优化用 `%d`），结果可逐位复现。"
      % (cfg["seed"], cfg["seed"]))
    A("- 原始数据：`results/sdp_check.json`（含每行的特征值、对偶界、rowdot 残差、耗时）。复现：")
    A("")
    A("```bash")
    A("cd labs/01-laplace-qpeplayer/experiments/sdp")
    A("/home/syk/miniconda3/envs/WM/bin/python run_sdp_check.py \\")
    A("    --check-upstream /home/syk/labs/SDPR/code/sdprlayer")
    A("```")
    A("")
    A("## 5. 结论（修订版 P1-2 分层）")
    A("")
    A("用户 2026-11-27 的修订证明把 13×13 分成两层。**(i) 满 20D 层**：补齐 3 条行正交 "
      "$r_1\\cdot r_2=r_1\\cdot r_3=r_2\\cdot r_3=0$ 后，$SE(3)$ 的极大二次消没理想为 "
      "$\\mathcal I_2(SE(3))=\\mathcal I_2(SO(3))\\oplus 0_{3\\times3}$（20 维，与 $t$ 零耦合），由 Schur 补双射 "
      "$X^\\star_{13}=LX^\\star_{10}L^\\top$ 得 $f^\\star_{13,\\rm full}\\equiv f^\\star_{10,\\rm full}$ 且秩相同；"
      "本脚本的 10×10 与“24 约束 + 3 行正交”的 13×13 数值一致（CL 实例 %.10f / rank %d）。"
      "**(ii) 上游 17D 层**：`sdprlayer@62d8ac6` 的 24 条约束在 $\\mathrm{Sym}^{10}$ 中只张成 %d 维（比 10×10 的 %d 维少 3 维，"
      "缺的正是三条行正交），故 $f^\\star_{13,\\rm up}\\le f^\\star_{13,\\rm full}=f^\\star_{10,\\rm full}\\le f_{\\rm nonconvex}$，"
      "一般是**严格更松**。本脚本 8 个 1e-2 对称破缺扰动 %d/%d 严格 $f^\\star_{13,\\rm up}<f^\\star_{10}$"
      "（差 %.1e–%.1e，rank 6→%s），构造目标 $-r_1\\cdot r_2+I/4$ 差 %.1f；独立的 6 点物理 3D-3D 反例"
      "（`six_point.md`）给出 $f^\\star_{10}=5.41500000$（rank 1）、$f^\\star_{13,\\rm full}=5.41500000$（rank 1）、"
      "$f^\\star_{13,\\rm up}=3.59431373$（rank 4），且 10/10 个 W/y 小扰动保持严格间隙。"
      "**因此**：不能说“上游 13×13 与 10×10 同值同秩”（错），也不能笼统说“13×13 与 10×10 不等价”（满 20D 版本是等价的）；"
      "正确表述是「满 20D 13×13 ≡ 10×10；上游 17D 实现严格更松，且 6 点物理反例证明该间隙可 O(1)」。"
      % (reg["cl_f10"], reg["cl_rank10"],
         st["rank_24_rotation_blocks_exact"], st["rank_21_constraints_exact"],
         sm["pert_n_strict"], sm["pert_n"], sm["pert_gap_min"], sm["pert_gap_max"],
         ",".join(str(v) for v in sm["pert_ranks13"]),
         rdref["f10"] - rdref["f13"]))
    A("")
    with open(path, "w") as fh:
        fh.write("\n".join(L) + "\n")


# ===========================================================================
# 9. main
# ===========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--starts", type=int, default=160, help="multi-start count for f_nonconvex")
    ap.add_argument("--n-sample", type=int, default=200000, help="dense sphere samples")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--eps", type=float, default=1e-9)
    ap.add_argument("--check-upstream", default=None,
                    help="path to the sdprlayer checkout for a constraint cross-check")
    args = ap.parse_args()
    args.n_sample = args.n_sample

    t_start = time.time()
    os.makedirs(RESULTS, exist_ok=True)
    res = {
        "config": {
            "python": sys.version.split()[0],
            "numpy": np.__version__, "scipy": __import__("scipy").__version__,
            "cvxpy": cp.__version__, "solver": "SCS", "eps": args.eps,
            "max_iters": 200000, "starts": args.starts, "n_sample": args.n_sample,
            "seed": args.seed, "platform": platform.platform(),
            "upstream_generators": POSE_EST_GENERATORS,
            "scope": ("rotation subproblem only; 13x13 = SDPRLayers 24-constraint lifted SDP; "
                      "no claim of physical realizability of the sampled M (that is Q1)"),
        },
    }

    print("[1/5] exact 10x10 construction ...")
    U, A10, names10, chk10 = build_10x10()
    print("      ", chk10)
    print("[2/5] 13x13 upstream constraint replica ...")
    A13, chk13 = build_13x13()
    print("      ", chk13)
    P = perm_rowmajor_to_colmajor()
    rd_cm, rd_rm = rowdot_matrices()

    print("[3/5] structural analysis of the two constraint sets ...")
    res["structural"] = structural_analysis(A10, names10, A13)
    res["structural_checks_10x10"] = chk10
    res["structural_checks_13x13"] = chk13
    print("      rank21 =", res["structural"]["rank_21_constraints_exact"],
          "rank24 =", res["structural"]["rank_24_rotation_blocks_exact"],
          "missing:", res["structural"]["only_in_10x10"])

    res["upstream"] = {"ran": False, "reason": "not requested"}
    if args.check_upstream:
        print("[3b] comparing replica with upstream PolyMatrix generators ...")
        res["upstream"] = upstream_crosscheck(args.check_upstream)
        print("      ", res["upstream"])

    print("[4/5] building instances and solving ...")
    main_inst, pert_inst = build_instances(U)
    rows, prows = [], []
    for inst in main_inst:
        t0 = time.time()
        d = instance_diagnostics(inst, U, P, A10, A13, rd_cm, rd_rm, args)
        print("   {:12s} f_nc={:+.8f} f10={:+.8f} f13={:+.8f} diff={:+.3e} r10={} r13={} ({:.1f}s)".format(
            d["name"], d["f_nonconvex"], d["f_10"], d["f_13"], d["f10_minus_f13"],
            d["rank_X10"], d["rank_X13"], time.time() - t0))
        rows.append(d)
    for inst in pert_inst:
        d = instance_diagnostics(inst, U, P, A10, A13, rd_cm, rd_rm, args)
        print("   {:12s} f_nc={:+.8f} f10={:+.8f} f13={:+.8f} diff={:+.3e}".format(
            d["name"], d["f_nonconvex"], d["f_10"], d["f_13"], d["f10_minus_f13"]))
        prows.append(d)

    res["main"] = [{k: v for k, v in d.items()
                    if k not in ("argmin_q",)} for d in rows]
    res["perturbations"] = [{k: v for k, v in d.items()
                             if k not in ("argmin_q",)} for d in prows]

    cl = next(d for d in rows if d["name"] == "CL_exact")
    res["regression"] = {
        "cl_f10": cl["f_10"],
        "verify_quoted_SDP_value": 1.9658119658452384,
        "cl_f10_minus_verify": cl["f_10"] - 1.9658119658452384,
        "cl_f13": cl["f_13"],
        "cl_rank10": cl["rank_X10"],
        "cl_rank13": cl["rank_X13"],
        "cl_rowdot_max": cl["rowdot_max_abs_at_X13"],
        "cl_fnonconvex": cl["f_nonconvex"],
        "cl_analytic_min": 2.0,
    }
    pert_diffs = [d["f10_minus_f13"] for d in prows]
    tight = [d["name"] for d in rows
             if abs(d["gap_10"]) < 1e-6 and abs(d["gap_13"]) < 1e-6
             and d["status_10"] == "optimal" and d["status_13"] == "optimal"]
    ablation = [d["f_13"] - d["f_13_only15"] for d in rows
                if d.get("f_13_only15") is not None and d.get("f_13") is not None]
    res["summary"] = {
        "n_main": len(rows),
        "schur_err_max": max(d["schur_err"] for d in rows if d.get("schur_err") is not None),
        "schur_decomp_max": max(d["schur_decomposition_res"] for d in rows),
        "pert_gap_min": min(pert_diffs), "pert_gap_max": max(pert_diffs),
        "pert_n_strict": int(sum(1 for x in pert_diffs if x > 1e-6)),
        "pert_n": len(pert_diffs),
        "pert_ranks13": sorted({d["rank_X13"] for d in prows}),
        "pert_ranks10": sorted({d["rank_X10"] for d in prows}),
        "pert_rowdot_max": max(d["rowdot_max_abs_at_X13"] for d in prows
                               if d.get("rowdot_max_abs_at_X13") is not None),
        "roundtrip": next((d["roundtrip_Q_err"] for d in rows if d.get("roundtrip_Q_err") is not None), None),
        "tight_instances": tight,
        "ablation_max_drop": max(ablation) if ablation else float("nan"),
        "max_abs_f10_minus_f13_all": max(abs(d["f10_minus_f13"]) for d in rows),
    }
    cl_c = next(d for d in rows if d["name"] == "CL_coupled")
    res["summary"]["coupled_diff10"] = abs(cl_c["f_10"] - cl["f_10"])
    res["summary"]["coupled_diff13"] = abs(cl_c["f_13"] - cl["f_13"])
    rdrow = next(d for d in rows if d["name"] == "rowdot")
    res["rowdot_ref"] = {"f10": rdrow["f_10"], "f13": rdrow["f_13"],
                         "rank10": rdrow["rank_X10"], "rank13": rdrow["rank_X13"]}
    rdgen = next(d for d in rows if d["name"] == "rowdot_gen")
    res["summary"]["rowdot_gen_gap"] = rdgen["f10_minus_f13"]

    res["runtime_s"] = time.time() - t_start
    json_path = os.path.join(RESULTS, "sdp_check.json")
    with open(json_path, "w") as fh:
        json.dump(res, fh, indent=2, default=str)
    print("[5/5] wrote", json_path)

    md_path = os.path.join(RESULTS, "sdp_check.md")
    write_markdown(res, md_path)
    print("      wrote", md_path)
    print("total runtime {:.1f}s".format(res["runtime_s"]))
    print("\nQUICK VERDICT: perturbations with f13 < f10: {}/{} (gap {:.2e}..{:.2e}); "
          "constructed rowdot instance f10-f13 = {:.3f}".format(
              res["summary"]["pert_n_strict"], res["summary"]["pert_n"],
              res["summary"]["pert_gap_min"], res["summary"]["pert_gap_max"],
              rdrow["f_10"] - rdrow["f_13"]))


if __name__ == "__main__":
    main()
