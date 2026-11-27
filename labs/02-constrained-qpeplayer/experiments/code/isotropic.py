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
