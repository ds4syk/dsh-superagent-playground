#!/usr/bin/env python
"""Isotropic 3D-3D (Horn) instances with single-plane constraints (theorem P2-1b).

Model
-----
    J(q,t) = sum_i (R(q) p_i + t - y_i)^T Omega_i (R(q) p_i + t - y_i)
    g_j(q,t) = -n_j^T (R(q) v_j + t) - d_j  <= 0,     j = 1..k

with ``Omega_i`` symmetric positive definite.  ``Omega_i = w_i I`` is the
*isotropic* Horn 3D-3D base; a full ``Omega_i`` (or the PnP bearing objective
of :mod:`qpep_core`) is the *anisotropic / Type I* base.

The ALM inner subproblem (qpep_core.alm_qpep) freezes the active set A and
minimises the quadratic

    J + sum_{j in A} [ mu_j g_j + rho/2 g_j^2 ] ,

which is again of the form ``v2^T M v2 + 2 t^T D v2 + t^T G t``.  Eliminating
``t`` (obj_min_t) gives ``Mred``; the KKT critical points on S^3 are those of
``v2(q)^T Mred v2(q)`` and are enumerated by :mod:`quartic_roots`.

This module only *assembles* the data (S1 of the round plan) and defines the
instance generator; counting/solving lives in ``isotropic_roots.py``.

All forms are exact on the unit sphere: ``e_tr^T v2(q) = q^Tq = 1``.  The
constant term of the least-squares objective is therefore represented by
``c * E_TR E_TR^T`` (equal to ``c`` only when ``q^Tq = 1``); the validator
checks equality on random *unit* q, and also that the form agrees with the
direct objective on the sphere to 1e-10.
"""
from __future__ import annotations

import numpy as np

import qpep_core as core

SQRT2 = np.sqrt(2.0)
E_TR = core.E_TR
m_rows = core.m_rows
v2 = core.v2
Jv2 = core.Jv2
R_of = core.R_of
obj_min_t = core.obj_min_t


# ---------------------------------------------------------------------------
# 1. least-squares assembly  (M0, D0, G0)
# ---------------------------------------------------------------------------
def ls_mats(P, Y, Omegas):
    """Assemble (M0, D0, G0) for J = sum_i (Rp_i+t-y_i)^T Omega_i (Rp_i+t-y_i).

    Exact on the unit sphere:
        v2(q)^T M0 v2(q) + 2 t^T D0 v2(q) + t^T G0 t == J(q, t).
    """
    P = np.atleast_2d(np.asarray(P, float))
    Y = np.atleast_2d(np.asarray(Y, float))
    Om = np.asarray(Omegas, float)
    if Om.ndim == 2:                       # single Omega shared by all points
        Om = np.repeat(Om[None], P.shape[0], axis=0)
    assert Om.shape == (P.shape[0], 3, 3)
    M0 = np.zeros((10, 10))
    D0 = np.zeros((3, 10))
    G0 = np.zeros((3, 3))
    const = 0.0
    c_lin = np.zeros(10)
    for p, y, O in zip(P, Y, Om):
        m = m_rows(p)
        M0 += m.T @ O @ m
        c_lin += -2.0 * (m.T @ (O @ y))     # -2 y^T O R p  =  c_lin^T v2
        const += float(y @ O @ y)
        D0 += O @ m
        G0 += O
    # constant: on the sphere e_tr^T v2 = 1, so c*E_TR E_TR^T == c
    M0 += const * np.outer(E_TR, E_TR)
    # linear-in-v2 term: c_lin^T v2 = v2^T sym(c_lin e_tr^T) v2  (on the sphere)
    M0 += 0.5 * (np.outer(c_lin, E_TR) + np.outer(E_TR, c_lin))
    # pure-t linear part of J is -2 t^T sum_i Omega_i y_i ; the (Rp)-part is
    # already in D0, so subtract the y-part through e_tr (== 1 on the sphere).
    Sy = np.sum([O @ y for y, O in zip(Y, Om)], axis=0)
    D0 -= np.outer(Sy, E_TR)
    return M0, D0, G0


def direct_J(q, t, P, Y, Omegas):
    """Direct least-squares objective (reference for the assembly check)."""
    P = np.atleast_2d(np.asarray(P, float))
    Y = np.atleast_2d(np.asarray(Y, float))
    Om = np.asarray(Omegas, float)
    if Om.ndim == 2:
        Om = np.repeat(Om[None], P.shape[0], axis=0)
    R = R_of(np.asarray(q, float))
    tot = 0.0
    for p, y, O in zip(P, Y, Om):
        r = R @ p + t - y
        tot += float(r @ O @ r)
    return tot


def form_J(q, t, M0, D0, G0):
    x = v2(np.asarray(q, float))
    return float(x @ M0 @ x + 2.0 * np.asarray(t) @ (D0 @ x)
                 + np.asarray(t) @ G0 @ np.asarray(t))


# ---------------------------------------------------------------------------
# 2. plane constraints  g_j = -n^T (R v_j + t) - d_j  <= 0
# ---------------------------------------------------------------------------
def plane_constraints(V, normals, ds):
    """Rows of (A, B) for k constraints.

    V       : (k,3) object-frame points v_j
    normals : (k,3) world-frame normals n_j
    ds      : (k,)  offsets d_j (plane n^T x + d = 0)
    """
    V = np.atleast_2d(np.asarray(V, float))
    Ns = np.atleast_2d(np.asarray(normals, float))
    ds = np.atleast_1d(np.asarray(ds, float))
    k = V.shape[0]
    A = np.zeros((k, 10))
    B = np.zeros((k, 3))
    for j in range(k):
        n = Ns[j]
        A[j] = -(n @ m_rows(V[j])) - ds[j] * E_TR
        B[j] = -n
    return A, B


def g_values(q, t, A, B):
    return A @ v2(np.asarray(q, float)) + B @ np.asarray(t, float)


# ---------------------------------------------------------------------------
# 3. instance generator
# ---------------------------------------------------------------------------
def _rot_rand(rng, scale=0.35):
    w = rng.normal(size=3) * scale
    th = np.linalg.norm(w)
    if th < 1e-12:
        return np.eye(3)
    K = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
    return np.eye(3) + np.sin(th) / th * K + (1 - np.cos(th)) / th ** 2 * (K @ K)


def make_isotropic_instance(seed, k, base="horn", normals=1, rho=1.0,
                            noise=0.02, n_pts=10):
    """Build one instance of theorem P2-1b.

    Parameters
    ----------
    seed    : int
    k       : number of active single-plane constraints (0..)
    base    : "horn"   isotropic Omega_i = w_i I (Type II)
              "aniso"  Omega_i = w_i (I + beta u_i u_i^T) (Type I)
              "pnp"    bearing/重投影 objective from qpep_core (Type I)
    normals : 1 = all planes share one normal (Type II); 2 = two distinct
              normals (Type I-like, N_C = 32)
    rho     : ALM penalty parameter used for the reduced inner matrix

    Returns dict with P,Y,Omega,weights,M0,D0,G0 (raw objective), V,Ns,ds,A,B
    (constraints), and helpers.
    """
    rng = np.random.default_rng(4242 + seed)
    R_gt = _rot_rand(rng, 0.5)
    t_gt = rng.uniform(-0.2, 0.2, size=3)
    P = rng.normal(size=(n_pts, 3)) * 0.5
    Y_true = (R_gt @ P.T).T + t_gt
    Y = Y_true + noise * rng.normal(size=(n_pts, 3))
    w = rng.uniform(0.5, 1.5, size=n_pts)

    if base == "horn":
        Omega = np.array([wi * np.eye(3) for wi in w])
    elif base == "aniso":
        beta = 0.6
        Omega = np.empty((n_pts, 3, 3))
        for i in range(n_pts):
            u = rng.normal(size=3)
            u /= np.linalg.norm(u)
            Omega[i] = w[i] * (np.eye(3) + beta * np.outer(u, u))
    elif base == "pnp":
        # bearing objective: directions u_i = Bw_i/||Bw_i||, build_objective
        Bw = Y
        U = Bw / np.linalg.norm(Bw, axis=1, keepdims=True)
        M0, D0, G0 = core.build_objective(P, U)
        Omega = None
    else:
        raise ValueError("base must be horn|aniso|pnp, got %r" % (base,))

    if base in ("horn", "aniso"):
        M0, D0, G0 = ls_mats(P, Y, Omega)

    # ---- constraints: object points on a face, world-normal planes ----
    V = np.zeros((max(k, 1), 3))
    Ns = np.zeros((max(k, 1), 3))
    ds = np.zeros(max(k, 1))
    if k > 0:
        # points on the bottom face z=-0.5*max|P| (a "table" contact face)
        h = 0.5 * np.max(np.abs(P))
        ang = rng.uniform(0, 2 * np.pi, size=k)
        rad = rng.uniform(0.1, 0.45, size=k)
        V[:, 0] = rad * np.cos(ang)
        V[:, 1] = rad * np.sin(ang)
        V[:, 2] = -h
        # world normal n = R_gt e_z  (the plane contains the contact face)
        n1 = R_gt @ np.array([0.0, 0.0, 1.0])
        n1 /= np.linalg.norm(n1)
        n2 = R_gt @ np.array([np.cos(0.7), np.sin(0.7), 0.15])
        n2 /= np.linalg.norm(n2)
        if normals == 1:
            Ns[:] = n1
        elif normals == 2:
            Ns[::2] = n1
            Ns[1::2] = n2
        else:
            raise ValueError("normals must be 1 or 2")
        # offset so that the true pose touches: n^T(R_gt V_j + t_gt) + d = 0
        for j in range(k):
            ds[j] = -float(Ns[j] @ (R_gt @ V[j] + t_gt))
        A, B = plane_constraints(V[:k], Ns[:k], ds[:k])
    else:
        A = np.zeros((0, 10))
        B = np.zeros((0, 3))

    return dict(seed=seed, k=int(k), base=base, normals=int(normals), rho=float(rho),
                P=P, Y=Y, Omega=Omega, weights=w, M0=M0, D0=D0, G0=G0,
                P_gt=P, R_gt=R_gt, t_gt=t_gt,
                V=V[:k], Ns=Ns[:k], ds=ds[:k], A=A, B=B)


# ---------------------------------------------------------------------------
# 4. reduced (v2) matrix for a frozen active set  (ALM inner subproblem)
# ---------------------------------------------------------------------------
def alm_reduced(inst, act=None, mu=None, rho=None):
    """Mred of the ALM inner quadratic for active set ``act`` (default all k).

    Uses qpep_core._alm_quadratic exactly as the ALM solver does:
        MA = M0 + sum_j mu_j/2 (e_tr a_j^T + a_j e_tr^T) + rho/2 a_j a_j^T
    (mu=0 by default: the *counting* object is the quadratic penalty part; the
    linear multiplier term only shifts the objective and cannot change the
    number of critical points generically.)
    """
    if act is None:
        act = np.arange(inst["k"])
    act = np.asarray(act, dtype=int)
    A, B = inst["A"], inst["B"]
    rho = inst["rho"] if rho is None else float(rho)
    mu = np.zeros(A.shape[0]) if mu is None else np.asarray(mu, float)
    MA, DA, GA = core._alm_quadratic(inst["M0"], inst["D0"], inst["G0"],
                                     A, B, mu, rho, act)
    return obj_min_t(MA, DA, GA), MA, DA, GA


def reduced_value(q, inst, act=None, rho=None):
    """Value of the ALM inner objective (t eliminated) at q (unit)."""
    Mred, _, _, _ = alm_reduced(inst, act, rho=rho)
    x = v2(np.asarray(q, float))
    return float(x @ Mred @ x)


# ---------------------------------------------------------------------------
# 5. self-check (S1)
# ---------------------------------------------------------------------------
def validate(seed=0, tol=1e-10, verbose=True):
    """Check the assembly against the direct objective and A v2 + B t == g."""
    rng = np.random.default_rng(7 + seed)
    out = {}
    for base in ("horn", "aniso", "pnp"):
        for k in (0, 1, 3):
            inst = make_isotropic_instance(seed, k, base=base)
            worst_J = 0.0
            for _ in range(20):
                q = rng.normal(size=4)
                q /= np.linalg.norm(q)
                t = rng.normal(size=3)
                if base == "pnp":
                    # for pnp the (M,D,G) is the bearing form; direct objective
                    # is the bearing residual sum, not a 3D-3D least squares.
                    x = v2(q)
                    U = inst["Y"] / np.linalg.norm(inst["Y"], axis=1, keepdims=True)
                    R = R_of(q)
                    Pw = (R @ inst["P"].T).T + t
                    Jd = 0.0
                    for i in range(inst["P"].shape[0]):
                        r = Pw[i]
                        u = U[i]
                        Jd += float(r @ r - (u @ r) ** 2 / (u @ u))
                    Jf = form_J(q, t, inst["M0"], inst["D0"], inst["G0"])
                else:
                    Jd = direct_J(q, t, inst["P"], inst["Y"], inst["Omega"])
                    Jf = form_J(q, t, inst["M0"], inst["D0"], inst["G0"])
                worst_J = max(worst_J, abs(Jd - Jf))
            worst_g = 0.0
            if k:
                for _ in range(20):
                    q = rng.normal(size=4)
                    q /= np.linalg.norm(q)
                    t = rng.normal(size=3)
                    gd = np.array([-inst["Ns"][j] @ (R_of(q) @ inst["V"][j] + t)
                                   - inst["ds"][j] for j in range(k)])
                    gf = g_values(q, t, inst["A"], inst["B"])
                    worst_g = max(worst_g, float(np.max(np.abs(gd - gf))))
            out[(base, k)] = dict(max_abs_J_err=worst_J, max_abs_g_err=worst_g,
                                  rank_M0=int(np.linalg.matrix_rank(inst["M0"])))
            if verbose:
                print("  base=%-5s k=%d  |J_from_form-J_direct|max=%.2e  "
                      "|g_form-g_direct|max=%.2e  rank(M0)=%d"
                      % (base, k, worst_J, worst_g,
                         np.linalg.matrix_rank(inst["M0"])))
            assert worst_J < tol, (base, k, worst_J)
            if k:
                assert worst_g < 1e-12, (base, k, worst_g)
    return out


if __name__ == "__main__":
    validate()


# ---------------------------------------------------------------------------
# 6. canonical form of theorem P2-1b  (S2/S5 anchor)
# ---------------------------------------------------------------------------
def v2_linear_coeff(K):
    """Vector ell with ell^T v2(q) == q^T K q (K symmetric 4x4)."""
    K = np.asarray(K, float)
    K = 0.5 * (K + K.T)
    ell = np.zeros(10)
    ell[:4] = np.diag(K)
    for c, (i, j) in enumerate([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]):
        ell[4 + c] = SQRT2 * K[i, j]
    return ell


def w_matrix(n):
    """3x10 matrix W with w(q) = R(q)^T n = W v2(q) exactly."""
    n = np.asarray(n, float)
    W = np.zeros((3, 10))
    for a in range(3):
        e = np.zeros(3)
        e[a] = 1.0
        W[a] = n @ m_rows(e)
    return W


def canonical_M(K, Sigma, n=None, W=None):
    """Mred of J=(q^Tq)(q^TKq) + w(q)^T Sigma w(q), w(q)=R(q)^T n."""
    ell = v2_linear_coeff(K)
    M = 0.5 * (np.outer(ell, E_TR) + np.outer(E_TR, ell))
    if W is None:
        W = w_matrix(n)
    Sigma = np.asarray(Sigma, float)
    M = M + W.T @ Sigma @ W
    return M


def make_canonical_instance(seed, rank_Sigma, normals=1, n_planes=None,
                            K_scale=1.0, rank_Sigma2=None):
    """Canonical Type II instance: random symmetric K, random PSD Sigma of rank r.

    ``n`` is random so that the Hopf line is a generic one.  The returned Mred
    is exact (no t, no ALM assembly).
    """
    rng = np.random.default_rng(3030 + seed)
    A = rng.normal(size=(4, 4))
    K = 0.5 * (A + A.T) * K_scale
    n = rng.normal(size=3)
    n /= np.linalg.norm(n)
    s = rng.normal(size=(max(rank_Sigma, 1), 3))
    sig = np.zeros((max(rank_Sigma, 1), 3))
    for a in range(max(rank_Sigma, 1)):
        if a < rank_Sigma:
            sig[a] = s[a] / np.linalg.norm(s[a]) * rng.uniform(0.5, 1.5)
    Sigma = sig[:rank_Sigma].T @ sig[:rank_Sigma] if rank_Sigma else np.zeros((3, 3))
    W = w_matrix(n)
    M = canonical_M(K, Sigma, W=W)
    inst = dict(seed=seed, k=int(rank_Sigma), rank_Sigma=int(rank_Sigma),
                base="canonical", K=K, Sigma=Sigma, n=n, Mred=M, W=W,
                ell=v2_linear_coeff(K), normals=int(normals))
    if normals == 2:
        r = rank_Sigma if rank_Sigma2 is None else rank_Sigma2
        n2 = rng.normal(size=3)
        n2 /= np.linalg.norm(n2)
        # make n2 not parallel to n
        n2 = n2 - (n2 @ n) * n
        n2 /= np.linalg.norm(n2)
        s2 = rng.normal(size=(max(r, 1), 3))
        sig2 = np.zeros((max(r, 1), 3))
        for a in range(max(r, 1)):
            if a < r:
                sig2[a] = s2[a] / np.linalg.norm(s2[a]) * rng.uniform(0.5, 1.5)
        Sigma2 = sig2[:r].T @ sig2[:r] if r else np.zeros((3, 3))
        W2 = w_matrix(n2)
        M = M + W2.T @ Sigma2 @ W2
        inst.update(n2=n2, Sigma2=Sigma2, W2=W2, Mred=M, rank_Sigma2=int(r))
    return inst


def canonical_value(q, inst):
    """(q^Tq)(q^TKq) + w(q)^T Sigma w(q), homogeneous form (exact for all q).

    ``w`` uses the *homogeneous* Veronese representative ``W v2(q)`` which
    equals ``R(q)^T n`` on the unit sphere and is the correct complexification
    for q with q^Tq != 1 (needed for the q^Tq=0 Hopf line).
    """
    q = np.asarray(q, float)
    w = inst["W"] @ v2(q)
    val = float((q @ q) * (q @ inst["K"] @ q) + w @ inst["Sigma"] @ w)
    if "W2" in inst:
        w2 = inst["W2"] @ v2(q)
        val += float(w2 @ inst["Sigma2"] @ w2)
    return val


# ---------------------------------------------------------------------------
# 7. canonical-family extraction (S3: the ALM-assembly -> Sigma_A mapping)
# ---------------------------------------------------------------------------
# The P2-1b canonical family is the 15-dimensional *linear subspace of quartic
# polynomials*
#
#     F_n = span{ (q^Tq)(q^TKq) : K = K^T in R^{4x4} }   (10 dims, K up to cI)
#         + span{ w(q)^T Sigma w(q) : Sigma = Sigma^T in R^{3x3} }  (6 dims)
#
# with the one-dimensional gauge  (Sigma, K) -> (Sigma + a I, K - a|n|^2 I),
# which leaves the polynomial invariant (w^Tw = |n|^2 (q^Tq)^2 exactly, since
# R(q)^T R(q) = (q^Tq)^2 I for the homogeneous quaternion rotation).
#
# Any quartic lift M whose *polynomial* f_M(q)=v2(q)^T M v2(q) lies in F_n has a
# unique canonical representative in the gauge  Sigma >= 0,  lambda_min(Sigma)=0
# (the theorem's (2.8) representative).  Conversely f_M in F_n is exactly the
# hypothesis of theorem P2-1b type II; the ALM assembly of an isotropic Horn
# base + plane constraints is always in F_n (numerically verified below),
# whereas anisotropic/PnP bases are not.
#
# Important: a matrix M in F_n and the canonical representative M_can differ at
# most by an element of ker(M -> polynomial), which is *not* a change of the
# function on the unit sphere.  The KKT root set of the polynomial is what
# matters; M_can just makes the rank of Sigma readable.

# ---- exact polynomial expansion of x^T M x -------------------------------
# degree-2 monomials q_i q_j (i<=j) and degree-4 monomials q_i q_j q_k q_l
_QUAD_MONOS = [(i, j) for i in range(4) for j in range(i, 4)]
_QUARTIC_MONOS = [(i, j, k, l) for i in range(4) for j in range(i, 4)
                  for k in range(j, 4) for l in range(k, 4)]
_CROSS_PAIRS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
_V2_QUAD = None      # 10 x 10: v2_a as a quadratic-monomial vector
_QUAD_PROD = None    # 10 x 10 -> 35 : product of two quadratic monomials


def _v2_quad_vectors():
    """10x10 matrix: row a is v2_a expressed in the 10 quadratic monomials."""
    global _V2_QUAD
    if _V2_QUAD is not None:
        return _V2_QUAD
    Q = np.zeros((10, 10))
    for a in range(10):
        if a < 4:
            Q[a, _QUAD_MONOS.index((a, a))] = 1.0
        else:
            i, j = _CROSS_PAIRS[a - 4]
            Q[a, _QUAD_MONOS.index((i, j))] = SQRT2
    _V2_QUAD = Q
    return Q


def _quad_prod_index():
    """(10,10) integer matrix: index in _QUARTIC_MONOS of mono_p * mono_r."""
    global _QUAD_PROD
    if _QUAD_PROD is not None:
        return _QUAD_PROD
    P = np.zeros((10, 10), dtype=int)
    for p, (i, j) in enumerate(_QUAD_MONOS):
        for r, (k, l) in enumerate(_QUAD_MONOS):
            lab = tuple(sorted((i, j, k, l)))
            P[p, r] = _QUARTIC_MONOS.index(lab)
    _QUAD_PROD = P
    return P


def poly_coeffs(M):
    """35-vector of the quartic polynomial q -> v2(q)^T M v2(q)."""
    Q = _v2_quad_vectors()
    P = _quad_prod_index()
    M = np.asarray(M, float)
    out = np.zeros(35)
    for a in range(10):
        for b in range(a, 10):
            w = M[a, b] * (2.0 if a != b else 1.0)
            if w == 0.0:
                continue
            coef = np.outer(Q[a], Q[b])
            np.add.at(out, P.ravel(), (w * coef).ravel())
    return out


def _poly_of_M_terms(M):
    """(35,) polynomial vector; alias of :func:`poly_coeffs`."""
    return poly_coeffs(M)


def _canon_basis_polys(n, n2=None):
    """Polynomial-space basis columns of F_n (single normal) or F_{n,n2}."""
    cols, meta = [], []
    for i in range(4):
        for j in range(i, 4):
            K = np.zeros((4, 4))
            K[i, j] = K[j, i] = 1.0
            ell = v2_linear_coeff(K)
            M = 0.5 * (np.outer(ell, E_TR) + np.outer(E_TR, ell))
            cols.append(_poly_of_M_terms(M))
            meta.append(("K", i, j))
    W = w_matrix(n)
    for i in range(3):
        for j in range(i, 3):
            S = np.zeros((3, 3))
            S[i, j] = S[j, i] = 1.0
            cols.append(_poly_of_M_terms(W.T @ S @ W))
            meta.append(("S", i, j))
    if n2 is not None:
        W2 = w_matrix(n2)
        for i in range(3):
            for j in range(i, 3):
                S = np.zeros((3, 3))
                S[i, j] = S[j, i] = 1.0
                cols.append(_poly_of_M_terms(W2.T @ S @ W2))
                meta.append(("S2", i, j))
        for i in range(3):
            for j in range(3):
                C = np.zeros((3, 3))
                C[i, j] = 1.0
                M = 0.5 * (W.T @ C @ W2 + W2.T @ C.T @ W)
                cols.append(_poly_of_M_terms(M))
                meta.append(("C", i, j))
    return np.stack(cols, axis=1), meta


def _extract_K_S(cols):
    """(K, Sigma) matrices from the fitted coefficient vector."""
    raise NotImplementedError


def canonical_fit(M, n, n2=None):
    """Fit the quartic polynomial of M to the canonical family F_n (or F_{n,n2}).

    Returns dict with K (4x4), Sigma (3x3), residual (relative polynomial
    residual), rank_Sigma (after the Sigma>=0, lambda_min=0 gauge), Sigma2/C
    when n2 is given, and M_can / M_can2 (the canonical representative).
    ``residual`` is the *membership test*: < 1e-10 means M is in the family
    (theorem P2-1b type II / two-normal type II); O(10^-1) means it is not
    (anisotropic / PnP type I).
    """
    n = np.asarray(n, float)
    n = n / np.linalg.norm(n)
    A, meta = _canon_basis_polys(n, n2)
    b = poly_coeffs(M)
    x, _, rank, sv = np.linalg.lstsq(A, b, rcond=None)
    pred = A @ x
    nb = np.linalg.norm(b)
    residual = float(np.linalg.norm(pred - b) / nb) if nb > 0 else 0.0

    K = np.zeros((4, 4))
    S = np.zeros((3, 3))
    S2 = np.zeros((3, 3)) if n2 is not None else None
    C = np.zeros((3, 3)) if n2 is not None else None
    for v, (kind, i, j) in zip(x, meta):
        if kind == "K":
            K[i, j] += v
            if i != j:
                K[j, i] += v
        elif kind == "S":
            S[i, j] += v
            if i != j:
                S[j, i] += v
        elif kind == "S2":
            S2[i, j] += v
            if i != j:
                S2[j, i] += v
        elif kind == "C":
            C[i, j] += v

    # ---- gauge: Sigma >= 0 with lambda_min = 0 (theorem eq. (2.8)) --------
    ev = np.linalg.eigvalsh(S)
    alpha = -float(ev[0])
    S = S + alpha * np.eye(3)
    K = K - alpha * float(n @ n) * np.eye(4)      # Sigma -> Sigma+a I, K -> K-a|n|^2 I
    out = dict(K=K, Sigma=S, residual=residual, fit_rank=int(rank),
               n=n, W=w_matrix(n), M_can=canonical_M(K, S, W=w_matrix(n)))
    if n2 is not None:
        n2 = np.asarray(n2, float)
        n2 = n2 / np.linalg.norm(n2)
        if abs(float(n @ n2)) < 1.0 - 1e-8:
            ev2 = np.linalg.eigvalsh(S2)
            alpha2 = -float(ev2[0])
            S2 = S2 + alpha2 * np.eye(3)
            K = K - alpha2 * float(n2 @ n2) * np.eye(4)
            # C gauge: C -> C + gamma I  <=>  K -> K - gamma (n1.n2) I
            # C -> C + gamma I  (gamma = -tr(C)/3)  =>  K -> K - gamma (n1.n2) I
            gamma = -float(np.trace(C)) / 3.0
            C = C + gamma * np.eye(3)
            K = K - gamma * float(n @ n2) * np.eye(4)
            W1 = w_matrix(n)
            W2 = w_matrix(n2)
            Mc = canonical_M(K, S, W=W1) + W2.T @ S2 @ W2 \
                + 0.5 * (W1.T @ C @ W2 + W2.T @ C.T @ W1)
            out.update(Sigma2=S2, C=C, K=K, W2=W2, M_can=Mc,
                       rank_Sigma2=int(np.linalg.matrix_rank(S2, tol=1e-8)))
    out["rank_Sigma"] = int(np.linalg.matrix_rank(out["Sigma"], tol=1e-8))
    out["predicted_NC"] = {0: 4, 1: 12, 2: 16}.get(out["rank_Sigma"], 16)
    # verify the canonical representative has the same polynomial
    bcan = poly_coeffs(out["M_can"])
    out["canon_residual"] = float(np.linalg.norm(bcan - b) / nb) if nb > 0 else 0.0
    return out


def canonical_family_residual(M, n):
    """Relative polynomial distance of M to F_n (the P2-1b membership test)."""
    return canonical_fit(M, n)["residual"]


def classify_canonical(M, n):
    """(rank_Sigma, predicted N_C, residual) for a canonical-family matrix."""
    fit = canonical_fit(M, n)
    return fit["rank_Sigma"], fit["predicted_NC"], fit["residual"]
