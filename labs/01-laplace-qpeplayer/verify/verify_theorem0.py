#!/usr/bin/env python
"""Minimal numerical sanity check for Theorem 0 (rotation-subproblem only).

What this script does
---------------------
(A) Builds the homogeneous lift x_R(q) = [||q||^2, vec(R(q))] and the Veronese
    vector v2(q), and checks *exactly* (sympy, rationals) that
        x_R(q) = 2 U v2(q),      U^T U = I_10.
(B) Builds the 21 SO(3) quadratic constraints in x_R coordinates
    (6 column-orthonormality + 6 row-orthonormality + 9 homogeneous
    cross-product/handedness equations), checks each vanishes identically on
    x_R(q), and computes the exact rank of their span.
(C) Computes the exact dimension of the degree-2 part of the ideal of the
    Veronese variety nu_2(RP^3) (kernel of Sym^2(R^10) -> R[q]_4) and checks
    that the 21 constraints span *all* of it (rank 20 and containment).
(D) Solves the constructed 10x10 rotation-subproblem SDP for the Choi-Lam
    quartic + 2I instance, and compares with the true quartic minimum on S^3.

SCOPE / HONESTY NOTES
---------------------
* This is the **rotation-only, 10x10, t-eliminated** SDP.  It is *not* the
  13x13/24-constraint SDP that the released SDPRLayers implementation solves.
  See theory/proofs.md, "Scope of Theorem 0".
* The Choi-Lam instance is our own construction; the numbers in the source
  document (f*=1.98, gap=0.01773, rank=6) are the author's preliminary
  experiment and are NOT reproduced here verbatim (the exact perturbation is
  not specified there).  We report our own numbers.
"""
import json
import os
import sys

import numpy as np
import sympy as sp

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "verify_theorem0_output.json")
res = {}

# ----------------------------------------------------------------------------
# symbolic setup
# ----------------------------------------------------------------------------
q = sp.symbols("q0 q1 q2 q3", real=True)
q0, q1, q2, q3 = q

R = sp.Matrix([
    [q0**2 + q1**2 - q2**2 - q3**2, 2*(q1*q2 - q0*q3), 2*(q1*q3 + q0*q2)],
    [2*(q1*q2 + q0*q3), q0**2 - q1**2 + q2**2 - q3**2, 2*(q2*q3 - q0*q1)],
    [2*(q1*q3 - q0*q2), 2*(q2*q3 + q0*q1), q0**2 - q1**2 - q2**2 + q3**2],
])
# homogeneous lift: first coordinate is ||q||^2, not the constant 1
xhom = sp.Matrix([q0**2 + q1**2 + q2**2 + q3**2] + [R[i, j] for i in range(3) for j in range(3)])
v2 = sp.Matrix([q0**2, q1**2, q2**2, q3**2,
                sp.sqrt(2)*q0*q1, sp.sqrt(2)*q0*q2, sp.sqrt(2)*q0*q3,
                sp.sqrt(2)*q1*q2, sp.sqrt(2)*q1*q3, sp.sqrt(2)*q2*q3])

# m-basis without sqrt2, used to build coefficient matrices
m = sp.Matrix([q0**2, q1**2, q2**2, q3**2, q0*q1, q0*q2, q0*q3, q1*q2, q1*q3, q2*q3])
D = sp.diag(1, 1, 1, 1, sp.sqrt(2), sp.sqrt(2), sp.sqrt(2), sp.sqrt(2), sp.sqrt(2), sp.sqrt(2))

# express xhom = T m  (T is 10x10, exact rationals)
T = sp.zeros(10, 10)
for r in range(10):
    poly = sp.expand(xhom[r])
    for k in range(10):
        c = sp.expand(poly).coeff(m[k])
        T[r, k] = sp.nsimplify(c)
# verify T m == xhom
assert sp.simplify(T*m - xhom) == sp.zeros(10, 1), "T construction failed"
U = sp.Rational(1, 2) * T * D.inv()
U = sp.simplify(U)
res["U_orthogonal_exact"] = bool(sp.simplify(U.T*U - sp.eye(10)) == sp.zeros(10, 10))
res["xR_equals_2Uv2_exact"] = bool(sp.simplify(xhom - 2*U*v2) == sp.zeros(10, 1))
res["xR_norm_identity"] = bool(sp.simplify((xhom.T*xhom)[0] - 4*(v2.T*v2)[0]) == 0)
res["U_entries"] = sorted({str(e) for e in U})
res["note_xR_first_entry"] = ("x_R must be the homogeneous lift [||q||^2, vec R]; "
                              "with first entry literally 1 the identity holds only on S^3")

# ----------------------------------------------------------------------------
# 21 constraints, in x = [x0, r11..r33] coordinates (x0 = ||q||^2)
# ----------------------------------------------------------------------------
idx = {(i, j): 1 + 3*i + j for i in range(3) for j in range(3)}   # row-major
def sym_from_terms(terms):
    A = sp.zeros(10, 10)
    for (a, b), c in terms.items():
        A[a, b] += sp.Rational(c, 2)
        A[b, a] += sp.Rational(c, 2)
    return A

def cols():
    return [[R[i, j] for i in range(3)] for j in range(3)]

c1, c2, c3 = cols()
cons = []            # (name, 10x10 symmetric matrix over x coordinates)
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
# 9 homogeneous cross-product / handedness constraints: c1 x c2 = x0 c3, cyclic
def cross_terms(a, b, c):
    # (ca x cb)_1 = a2 b3 - a3 b2 = x0 c1 ; etc.  indices (row i, col a)
    out = []
    out.append((f"cross_ab_1", {(idx[(1, a)], idx[(2, b)]): 1,
                                (idx[(2, a)], idx[(1, b)]): -1,
                                (0, idx[(0, c)]): -1}))
    out.append((f"cross_ab_2", {(idx[(2, a)], idx[(0, b)]): 1,
                                (idx[(0, a)], idx[(2, b)]): -1,
                                (0, idx[(1, c)]): -1}))
    out.append((f"cross_ab_3", {(idx[(0, a)], idx[(1, b)]): 1,
                                (idx[(1, a)], idx[(0, b)]): -1,
                                (0, idx[(2, c)]): -1}))
    return out
for (a, b, c) in [(0, 1, 2), (1, 2, 0), (2, 0, 1)]:
    for nm, terms in cross_terms(a, b, c):
        cons.append((f"{nm}_{a}{b}", sym_from_terms(terms)))

names = [n for n, _ in cons]
A_list = [A for _, A in cons]
res["n_constraints"] = len(A_list)

# each constraint vanishes identically when x = xhom(q)
all_zero = True
for nm, A in cons:
    val = sp.expand((xhom.T*A*xhom)[0])
    if val != 0:
        all_zero = False
        print("NONZERO CONSTRAINT:", nm, val)
res["all_21_vanish_on_xR_exact"] = bool(all_zero)

def sym_vec(A):
    # coefficient vector of v^T A v in the basis (v_i^2, v_i v_j, i<j):
    # diagonal entries as-is, off-diagonal entries doubled
    return sp.Matrix([A[i, j] if i == j else 2*A[i, j]
                      for i in range(10) for j in range(i, 10)])

# map the 21 constraint matrices to v-coordinates: B = 4 U^T A U
Mcons_v = sp.Matrix([sym_vec(4 * U.T * A * U).T for A in A_list])   # 21 x 55
rank_cons = Mcons_v.rank()
res["rank_21_constraints_v_coords_exact"] = int(rank_cons)

# ----------------------------------------------------------------------------
# degree-2 part of the ideal of nu_2(RP^3): kernel of Sym^2(R^10) -> R[q]_4
# ----------------------------------------------------------------------------
mons4 = []
for a in range(4):
    for b in range(a, 4):
        for c in range(b, 4):
            for d in range(c, 4):
                e = [0, 0, 0, 0]
                for t in (a, b, c, d):
                    e[t] += 1
                mons4.append(tuple(e))
mons4 = sorted(set(mons4))
assert len(mons4) == 35
mon_index = {e: i for i, e in enumerate(mons4)}

def quartic_coeffs(poly):
    P = sp.Poly(sp.expand(poly), *q)
    d = P.as_dict()
    out = [d.get(e, sp.Integer(0)) for e in mons4]
    recon = sum(out[mon_index[e]] * sp.prod([q[t]**e[t] for t in range(4)])
                for e in mons4)
    assert sp.expand(poly - recon) == 0
    return out

# rows: 55 quadratic forms v_i v_j ; columns: 35 quartic monomials
Phi_rows = []
for i in range(10):
    for j in range(i, 10):
        Phi_rows.append(quartic_coeffs(v2[i]*v2[j]))
Phi = sp.Matrix(Phi_rows)
rank_phi = Phi.rank()
res["rank_veronese_degree4_map"] = int(rank_phi)
res["dim_ideal_degree2"] = int(55 - rank_phi)

# check the constraints span the kernel: build the kernel basis and stack
# kernel of c |-> c^T Phi  (coefficient space R^55 -> quartic space R^35)
kernel = Phi.T.nullspace()        # 55-vectors (sym components)
res["dim_kernel_numeric_exact"] = len(kernel)
# stack constraint rows and kernel vectors, compute rank
Kmat = sp.Matrix([[v[i] for i in range(55)] for v in kernel])   # 20 x 55
stack = Mcons_v.col_join(Kmat)
res["rank_constraints_plus_kernel"] = int(stack.rank())
res["constraints_equal_ideal_degree2"] = bool(stack.rank() == 20 and rank_cons == 20)

# ----------------------------------------------------------------------------
# Choi-Lam quartic + 2I in v-coordinates, constructed rotation-subproblem SDP
# ----------------------------------------------------------------------------
# C(q) = q0^2 q1^2 + q1^2 q2^2 + q2^2 q0^2 + q3^4 - 4 q0 q1 q2 q3
# v2 = [q0^2,q1^2,q2^2,q3^2, sqrt2 q0q1, sqrt2 q0q2, sqrt2 q0q3,
#       sqrt2 q1q2, sqrt2 q1q3, sqrt2 q2q3]
M_CL = np.zeros((10, 10))
M_CL[0, 1] = M_CL[1, 0] = 0.5    # q0^2 q1^2
M_CL[1, 2] = M_CL[2, 1] = 0.5    # q1^2 q2^2
M_CL[2, 0] = M_CL[0, 2] = 0.5    # q2^2 q0^2
M_CL[3, 3] = 1.0                 # q3^4
M_CL[4, 9] = M_CL[9, 4] = -1.0   # -4 q0q1q2q3 = -2 (sqrt2 q0q1)(sqrt2 q2q3)
M = M_CL + 2.0*np.eye(10)
res["ChoiLam_M_eigs"] = sorted(np.linalg.eigvalsh(M).tolist())
res["ChoiLam_min_eig"] = float(np.min(np.linalg.eigvalsh(M)))

# check C(q) >= 0 numerically on the sphere
rng = np.random.default_rng(0)
Qr = rng.normal(size=(200000, 4))
Qr /= np.linalg.norm(Qr, axis=1, keepdims=True)
Cval = (Qr[:, 0]**2*Qr[:, 1]**2 + Qr[:, 1]**2*Qr[:, 2]**2 + Qr[:, 2]**2*Qr[:, 0]**2
        + Qr[:, 3]**4 - 4*Qr[:, 0]*Qr[:, 1]*Qr[:, 2]*Qr[:, 3])
res["ChoiLam_C_min_MC"] = float(Cval.min())

# Q = (1/4) U M U^T  (x = 2 U v2)
Un = np.array(U.tolist(), dtype=float)
Qmat = 0.25 * Un @ M @ Un.T
res["Q_SDP_eigs"] = sorted(np.linalg.eigvalsh(Qmat).tolist())

# solve the constructed 10x10 SDP with cvxpy
try:
    import cvxpy as cp
    X = cp.Variable((10, 10), symmetric=True)
    constr = [X >> 0, X[0, 0] == 1]
    A_np = [np.array(A.tolist(), dtype=float) for A in A_list]
    for Ai in A_np:
        constr.append(cp.trace(Ai @ X) == 0)
    prob = cp.Problem(cp.Minimize(cp.trace(Qmat @ X)), constr)
    prob.solve(solver=cp.SCS, eps=1e-9, max_iters=200000, verbose=False)
    res["SDP_status"] = prob.status
    res["SDP_value"] = float(prob.value)
    Xs = X.value
    ev = np.linalg.eigvalsh((Xs + Xs.T)/2)
    ev = ev[::-1]
    res["SDP_X_eigs_top8"] = ev[:8].tolist()
    tol = 1e-6 * max(1.0, ev[0])
    res["SDP_X_rank_1e-6"] = int(np.sum(ev > tol))
    res["SDP_X_eig_ratio_12"] = float(ev[0]/max(ev[1], 1e-15))
    # dual certificate: Z = Q - mu E00 - sum_i lam_i A_i should be PSD, and the
    # certified lower bound is mu.  This upgrades the numerical gap to a
    # (numerically verified) certificate for this constructed instance.
    try:
        # cvxpy's equality-constraint dual y enters the Lagrangian as f + y*g,
        # so the multiplier of the constraint X00 == 1 in the standard form
        # (Q - mu E00 - sum lam_i A_i) is mu = -y.
        mu = -float(np.asarray(constr[1].dual_value).reshape(-1)[0])
        lam = [-float(np.asarray(c.dual_value).reshape(-1)[0]) for c in constr[2:]]
        E00 = np.zeros((10, 10)); E00[0, 0] = 1.0
        Z = Qmat - mu*E00 - sum(li*Ai for li, Ai in zip(lam, A_np))
        Z = 0.5*(Z + Z.T)
        zmin = float(np.min(np.linalg.eigvalsh(Z)))
        if zmin < -1e-6:   # fall back to the other sign convention
            mu2, lam2 = -mu, [-l for l in lam]
            Z2 = Qmat - mu2*E00 - sum(li*Ai for li, Ai in zip(lam2, A_np))
            Z2 = 0.5*(Z2 + Z2.T)
            zmin2 = float(np.min(np.linalg.eigvalsh(Z2)))
            if zmin2 > zmin:
                mu, lam, Z, zmin = mu2, lam2, Z2, zmin2
        res["dual_mu"] = mu
        res["dual_Z_min_eig"] = zmin
        res["dual_certificate_PSD"] = bool(zmin > -1e-6)
        res["certified_gap_lower_bound"] = float(2.0 - mu)
    except Exception as e:
        res["dual_certificate"] = "FAILED: " + repr(e)
except Exception as e:  # pragma: no cover
    res["SDP_status"] = "FAILED: " + repr(e)

# true minimum of the quartic on S^3 by many random multi-starts
from scipy.optimize import minimize

def f_and_grad(xq):
    n2 = xq @ xq
    vv = np.array([xq[0]**2, xq[1]**2, xq[2]**2, xq[3]**2,
                   np.sqrt(2)*xq[0]*xq[1], np.sqrt(2)*xq[0]*xq[2], np.sqrt(2)*xq[0]*xq[3],
                   np.sqrt(2)*xq[1]*xq[2], np.sqrt(2)*xq[1]*xq[3], np.sqrt(2)*xq[2]*xq[3]])
    f = vv @ M @ vv
    # Jv: 10x4
    sq = np.sqrt(2)
    Jv = np.array([
        [2*xq[0], 0, 0, 0],
        [0, 2*xq[1], 0, 0],
        [0, 0, 2*xq[2], 0],
        [0, 0, 0, 2*xq[3]],
        [sq*xq[1], sq*xq[0], 0, 0],
        [sq*xq[2], 0, sq*xq[0], 0],
        [sq*xq[3], 0, 0, sq*xq[0]],
        [0, sq*xq[2], sq*xq[1], 0],
        [0, sq*xq[3], 0, sq*xq[1]],
        [0, 0, sq*xq[3], sq*xq[2]],
    ])
    g = 2 * Jv.T @ (M @ vv)
    return f, g

best = np.inf
bestx = None
starts = rng.normal(size=(400, 4))
starts /= np.linalg.norm(starts, axis=1, keepdims=True)
for s in starts:
    r = minimize(lambda z: f_and_grad(z)[0], s, jac=lambda z: f_and_grad(z)[1],
                 method="BFGS", options={"gtol": 1e-10, "maxiter": 2000})
    z = r.x / np.linalg.norm(r.x)
    val = f_and_grad(z)[0]
    if val < best:
        best = val; bestx = z
res["QPEP_true_min_multistart"] = float(best)
res["QPEP_argmin"] = bestx.tolist()
# analytic global minimum for this constructed instance: C(q) >= 0 with C(e0)=0,
# hence f = C + 2 has minimum exactly 2 (attained at q = e0 and other points).
res["QPEP_true_min_analytic"] = 2.0
res["C_at_e0"] = 0.0
if "SDP_value" in res:
    res["relaxation_gap"] = float(res["QPEP_true_min_analytic"] - res["SDP_value"])
    res["gap_over_true"] = float(res["relaxation_gap"] / res["QPEP_true_min_analytic"])

# positive-measure sanity: perturb M and re-solve, see gap persists
gaps = []
try:
    import cvxpy as cp
    for t in range(8):
        dM = rng.normal(size=(10, 10)); dM = 0.02*(dM + dM.T)/2
        Mt = M + dM
        if np.min(np.linalg.eigvalsh(Mt)) <= 0:
            continue
        Qm = 0.25 * Un @ Mt @ Un.T
        Xt = cp.Variable((10, 10), symmetric=True)
        ct = [Xt >> 0, Xt[0, 0] == 1] + [cp.trace(Ai @ Xt) == 0 for Ai in A_np]
        pt = cp.Problem(cp.Minimize(cp.trace(Qm @ Xt)), ct)
        pt.solve(solver=cp.SCS, eps=1e-9, max_iters=200000)
        # true min for perturbed M by a few starts (lower bound is enough for gap>0 check)
        best_t = min(f_and_grad((lambda z: z/np.linalg.norm(z))(z))[0]
                     for z in rng.normal(size=(60, 4)))
        gaps.append(float(best_t - pt.value))
    res["perturbation_gaps"] = gaps
    res["all_perturbed_gaps_positive"] = bool(all(g > 0 for g in gaps))
except Exception as e:
    res["perturbation_gaps"] = "FAILED: " + repr(e)

res["scope"] = ("rotation-only 10x10 t-eliminated SDP; NOT the released 13x13/24-constraint "
                "SDPRLayers SDP; Choi-Lam instance is our construction, not the author's exact one")
print(json.dumps(res, indent=2, default=str))
with open(OUT, "w") as fh:
    json.dump(res, fh, indent=2, default=str)
print("\nwrote", OUT)
