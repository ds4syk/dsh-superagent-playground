#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Constrained-QPEPLayer: CPU end-to-end toy + Fischer--Burmeister backward.

What this is
------------
A small but honest *end-to-end training* experiment for the constrained QPEP
pose layer, plus a finite-difference gradient check of the FB-reduced
implicit backward from ``theory/proofs.md`` §3 (the saddle system (3.2)-(3.3)).

Pipeline (all CPU, float64):
    noisy 2D keypoints (14 points, 28-d)  ->  tiny MLP keypoint refiner
        ->  differentiable QPEP head (one of four variants)  ->  pose (q, t)
        ->  GT-pose loss.

The four heads share the backbone / data / seed budget and differ only in how
the physical inequalities ``g_j(q,t) = a_j^T v2(q) + b_j^T t <= 0`` are handled:

* ``unc``  : unconstrained (C-QPEP) global-ish solve (multi-start Newton on S^3)
* ``alm``  : ALM-QPEP forward (frozen active set, Thm 2.1 assembly) -- OURS
* ``proj`` : unconstrained solve then a post-hoc SQP projection to feasibility
* ``soft`` : quadratic penalty ``J + lam * sum [g]_+^2`` (lambda = 1e3)

Backward:
* ``unc``  : exact implicit differentiation of the sphere KKT (6x6 Hessian)
* ``alm``  : FB-reduced saddle system ``S = [[H_eff, J_A^T], [J_A, -D_a]]``
             (theory §3.2-3.3), with ``eps -> 0`` regularisation
* ``proj`` : implicit differentiation of the projection KKT (H = I + sum mu H_g)
* ``soft`` : implicit differentiation of the smooth penalised objective

The forward solves are batched torch code (CPU); the ALM forward matches the
NumPy reference ``qpep_core.alm_qpep`` on the same instances (see
``--phase selftest``).  Numbers produced here are a prototype, not a paper
result; see the scope note in the generated report.

Interpreter: /home/syk/miniconda3/envs/sdpr/bin/python  (torch 2.5 CPU)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import qpep_core as qc  # noqa: E402

torch.set_num_threads(1)          # single-thread CPU (login node etiquette)
DT = torch.float64

# ---------------------------------------------------------------------------
# 0. constants shared with qpep_core / theory
# ---------------------------------------------------------------------------
P_DEF = 16                        # 9 pen + 4 depth + 1 fov pair + 1 tilt
N_PTS = qc.N_PTS                  # 14 object points
H_TAB = qc.H_TAB
E_TR = torch.tensor(qc.E_TR, dtype=DT)
M_ROWS = torch.tensor(np.stack([qc.m_rows(p) for p in qc.P_OBJ]), dtype=DT)   # (14,3,10)

# camera: 32x32 image, f = 60 px  (object ~0.18 m at ~0.75 m -> ~14 px)
IMG = 32
F_PX = 60.0
C_PX = 16.0

# fixed constraints: A, B are *independent of the observation* (model geometry),
# so they are shared by every sample (unlike M0,D0,G0 which depend on keypoints)
_A0, _B0 = qc.make_instance(0, P_DEF)["A"], qc.make_instance(0, P_DEF)["B"]
A_T = torch.tensor(_A0, dtype=DT)      # (p,10)
B_T = torch.tensor(_B0, dtype=DT)      # (p,3)
P_CON = A_T.shape[0]


# ---------------------------------------------------------------------------
# 1. torch geometry (exact v2 algebra, same conventions as qpep_core)
# ---------------------------------------------------------------------------
def v2_t(q: torch.Tensor) -> torch.Tensor:
    """(M,4) -> (M,10)."""
    q0, q1, q2, q3 = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    s2 = math.sqrt(2.0)
    return torch.stack([
        q0 * q0, q1 * q1, q2 * q2, q3 * q3,
        s2 * q0 * q1, s2 * q0 * q2, s2 * q0 * q3,
        s2 * q1 * q2, s2 * q1 * q3, s2 * q2 * q3,
    ], dim=1)


def Jv2_t(q: torch.Tensor) -> torch.Tensor:
    """(M,4) -> (M,10,4) analytic dv2/dq."""
    M = q.shape[0]
    J = q.new_zeros((M, 10, 4))
    s2 = math.sqrt(2.0)
    J[:, 0, 0] = 2 * q[:, 0]
    J[:, 1, 1] = 2 * q[:, 1]
    J[:, 2, 2] = 2 * q[:, 2]
    J[:, 3, 3] = 2 * q[:, 3]
    J[:, 4] = torch.stack([s2 * q[:, 1], s2 * q[:, 0], q.new_zeros(M), q.new_zeros(M)], 1)
    J[:, 5] = torch.stack([s2 * q[:, 2], q.new_zeros(M), s2 * q[:, 0], q.new_zeros(M)], 1)
    J[:, 6] = torch.stack([s2 * q[:, 3], q.new_zeros(M), q.new_zeros(M), s2 * q[:, 0]], 1)
    J[:, 7] = torch.stack([q.new_zeros(M), s2 * q[:, 2], s2 * q[:, 1], q.new_zeros(M)], 1)
    J[:, 8] = torch.stack([q.new_zeros(M), s2 * q[:, 3], q.new_zeros(M), s2 * q[:, 1]], 1)
    J[:, 9] = torch.stack([q.new_zeros(M), q.new_zeros(M), s2 * q[:, 3], s2 * q[:, 2]], 1)
    return J


def _Hv2() -> torch.Tensor:
    """(10,4,4) constant Hessians of the v2 components."""
    H = torch.zeros(10, 4, 4, dtype=DT)
    for k in range(4):
        H[k, k, k] = 2.0
    s2 = math.sqrt(2.0)
    cross = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    for k, (i, j) in enumerate(cross, start=4):
        H[k, i, j] = s2
        H[k, j, i] = s2
    return H


HV2 = _Hv2()


def R_of_t(q: torch.Tensor) -> torch.Tensor:
    """(M,4) -> (M,3,3), sign-invariant rotation."""
    M = q.shape[0]
    q0, q1, q2, q3 = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = q.new_zeros((M, 3, 3))
    R[:, 0, 0] = 1 - 2 * (q2 * q2 + q3 * q3)
    R[:, 0, 1] = 2 * (q1 * q2 - q0 * q3)
    R[:, 0, 2] = 2 * (q1 * q3 + q0 * q2)
    R[:, 1, 0] = 2 * (q1 * q2 + q0 * q3)
    R[:, 1, 1] = 1 - 2 * (q1 * q1 + q3 * q3)
    R[:, 1, 2] = 2 * (q2 * q3 - q0 * q1)
    R[:, 2, 0] = 2 * (q1 * q3 - q0 * q2)
    R[:, 2, 1] = 2 * (q2 * q3 + q0 * q1)
    R[:, 2, 2] = 1 - 2 * (q1 * q1 + q2 * q2)
    return R


def normalize(q: torch.Tensor) -> torch.Tensor:
    return q / torch.linalg.norm(q, dim=-1, keepdim=True).clamp_min(1e-300)


def tangent_basis(q: torch.Tensor) -> torch.Tensor:
    """(M,4) -> (M,4,3) orthonormal basis of the tangent space at q.

    Householder reflection mapping q to +/- e0; the last 3 columns are the
    tangent basis.  Smooth except at q0 = 0 (measure zero)."""
    M = q.shape[0]
    e0 = q.new_zeros(M, 4)
    e0[:, 0] = torch.where(q[:, 0] >= 0, 1.0, -1.0)
    v = q - e0
    vv = (v * v).sum(1, keepdim=True)
    eye = torch.eye(4, dtype=q.dtype).expand(M, 4, 4)
    H = eye - 2.0 * (v.unsqueeze(2) @ v.unsqueeze(1)) / vv.clamp_min(1e-30).unsqueeze(2)
    return H[:, :, 1:]


# ---------------------------------------------------------------------------
# 2. observation -> (M0,D0,G0) assembly, differentiable in the keypoints
# ---------------------------------------------------------------------------
def keypoints_to_bearings(K: torch.Tensor) -> torch.Tensor:
    """K (B,14,2) normalized image coords -> unit bearings (B,14,3)."""
    ones = K.new_ones(K.shape[0], K.shape[1], 1)
    u = torch.cat([K, ones], dim=2)
    return u / torch.linalg.norm(u, dim=2, keepdim=True).clamp_min(1e-300)


def bearings_to_keypoints(U: np.ndarray) -> np.ndarray:
    """unit bearings (14,3) -> pixel keypoints (14,2) in the 32x32 image."""
    return F_PX * (U[:, :2] / U[:, 2:3]) + C_PX


def assemble(K: torch.Tensor):
    """K (B,14,2) normalized image coords -> M0 (B,10,10), D0 (B,3,10), G0 (B,3,3)."""
    u = keypoints_to_bearings(K)
    P = torch.eye(3, dtype=K.dtype).expand(K.shape[0], N_PTS, 3, 3) - \
        u.unsqueeze(-1) @ u.unsqueeze(-2)                      # (B,14,3,3)
    M0 = torch.einsum("nka,bnkl,nlc->bac", M_ROWS, P, M_ROWS)
    D0 = torch.einsum("bnkl,nlc->bkc", P, M_ROWS)
    G0 = P.sum(dim=1)
    return M0, D0, G0


def reduced_objective(K: torch.Tensor, M0, D0, G0):
    """Return Mred (B,10,10) with t eliminated: t* = -G^{-1} D v2(q)."""
    Ginv_D = torch.linalg.solve(G0, D0)                        # (B,3,10)
    return M0 - D0.transpose(1, 2) @ Ginv_D, Ginv_D


def obj_value(M, D, G, q, t):
    x = v2_t(q)
    Mx = (M @ x.unsqueeze(-1)).squeeze(-1)
    return (x * Mx).sum(1) + 2.0 * (t * (D @ x.unsqueeze(-1)).squeeze(-1)).sum(1) + \
        (t * (G @ t.unsqueeze(-1)).squeeze(-1)).sum(1)


# ---------------------------------------------------------------------------
# 3. forward solvers (batched, float64, deterministic)
# ---------------------------------------------------------------------------
def inner_solve(Mred: torch.Tensor, q0: torch.Tensor, iters: int = 20,
                damp: float = 1e-10):
    """min_{q in S^3} v2(q)^T Mred v2(q), batched Riemannian Levenberg-Newton.

    Mred (M,10,10), q0 (M,4) -> q (M,4).  The chart is the normalised linear
    chart q(xi) = normalize(q + B xi) with B the orthonormal tangent basis;
    its exact Hessian is  B^T H B - (grad . q) I  (see report)."""
    q = normalize(q0)
    # scale-invariance: minimisers are unchanged by Mred -> Mred / s; this makes
    # the Levenberg damping schedule independent of rho (which can reach 1e7)
    scale = Mred.diagonal(dim1=-2, dim2=-1).abs().sum(-1).clamp_min(1e-12)
    Mred = Mred / scale.view(-1, 1, 1)
    for _ in range(iters):
        x = v2_t(q)
        Mx = (Mred @ x.unsqueeze(-1)).squeeze(-1)              # (M,10)
        gq = 2.0 * torch.einsum("mki,mk->mi", Jv2_t(q), Mx)    # (M,4)
        Bq = tangent_basis(q)                                  # (M,4,3)
        h = torch.einsum("mki,mk->mi", Bq, gq)                 # (M,3)
        Gq = torch.einsum("mik,mkj->mij", Jv2_t(q), Bq)        # (M,10,3)
        Wmat = torch.einsum("mk,kij->mij", Mx, HV2)           # (M,4,4)
        Hxx = 2.0 * torch.einsum("mki,mkl,mlj->mij", Gq, Mred, Gq) + \
            2.0 * torch.einsum("mki,mkl,mlj->mij", Bq, Wmat, Bq) - \
            4.0 * (Mx * x).sum(1).view(-1, 1, 1) * torch.eye(3, dtype=q.dtype)
        f0 = (x * Mx).sum(1)
        f_best = f0.clone()
        if bool((h.norm(dim=1) < 1e-11).all()):
            break
        eye3 = torch.eye(3, dtype=q.dtype)
        q_best = q.clone()
        accepted = torch.zeros(q.shape[0], dtype=torch.bool)
        lam = torch.full((q.shape[0],), damp, dtype=q.dtype)
        for _ls in range(14):
            delta = torch.linalg.solve(
                Hxx + lam.view(-1, 1, 1) * eye3, -h.unsqueeze(-1)).squeeze(-1)
            q_try = normalize(q + torch.einsum("mij,mj->mi", Bq, delta))
            x_try = v2_t(q_try)
            f_try = (x_try * (Mred @ x_try.unsqueeze(-1)).squeeze(-1)).sum(1)
            better = f_try < f_best
            if better.any():
                idx = better.nonzero(as_tuple=True)[0]
                q_best[idx] = q_try[idx]
                f_best[idx] = f_try[idx]
                accepted[idx] = True
            # Levenberg: increase damping where the step was uphill
            lam = torch.where(better, torch.clamp(lam * 0.3, min=damp), lam * 10.0)
            lam = torch.clamp(lam, max=1e6)
            if accepted.all():
                break
        q = q_best
    return q, f_best


def unc_solve(K: torch.Tensor, q_starts: torch.Tensor, iters: int = 22):
    """Unconstrained QPEP: multi-start inner solve + t elimination.

    q_starts (NST,4) fixed.  Returns q (B,4), t (B,3), J (B,), Mred/G etc."""
    M0, D0, G0 = assemble(K)
    Mred, GinvD = reduced_objective(K, M0, D0, G0)
    B = K.shape[0]
    NS = q_starts.shape[0]
    q_all = q_starts.unsqueeze(0).expand(B, NS, 4)
    Mred_all = Mred.unsqueeze(1).expand(B, NS, 10, 10)
    q_flat, f_flat = inner_solve(Mred_all.reshape(B * NS, 10, 10),
                                 q_all.reshape(B * NS, 4), iters=iters)
    q_flat = q_flat.reshape(B, NS, 4)
    f = f_flat.reshape(B, NS)
    best = f.argmin(dim=1)
    idx = torch.arange(B)
    q = q_flat[idx, best]
    x = v2_t(q)
    t = -torch.einsum("bki,bi->bk", GinvD, x)
    return q, t, f[idx, best], M0, D0, G0, Mred, GinvD


def _alm_quadratics(M0, D0, G0, mu, rho, act):
    """Thm 2.1 assembly, batched with per-sample active masks.
    mu (B,p), rho (B,), act (B,p) bool."""
    coef = mu * act.to(mu.dtype)                                # (B,p)
    s_a = torch.einsum("bp,pi->bi", coef, A_T)                 # (B,10)
    s_b = torch.einsum("bp,pk->bk", coef, B_T)                 # (B,3)
    MA = M0 + 0.5 * (E_TR.view(1, 10, 1) * s_a.unsqueeze(1) +
                     s_a.unsqueeze(2) * E_TR.view(1, 1, 10))
    DA = D0 + 0.5 * (s_b.unsqueeze(2) * E_TR.view(1, 1, 10))
    GA = G0.clone()
    coef2 = rho.unsqueeze(1) * act.to(mu.dtype)                 # (B,p)
    MA = MA + 0.5 * torch.einsum("bp,pi,pj->bij", coef2, A_T, A_T)
    DA = DA + 0.5 * torch.einsum("bp,pk,pi->bki", coef2, B_T, A_T)
    GA = GA + 0.5 * torch.einsum("bp,pk,pl->bkl", coef2, B_T, B_T)
    return MA, DA, GA


def alm_solve(K: torch.Tensor, q_init: torch.Tensor, t_init: torch.Tensor,
              max_outer: int = 22, inner_iters: int = 7, cons_rounds: int = 2,
              rho0: float = 10.0, rho_max: float = 1e7, tol: float = 1e-10):
    """Batched ALM-QPEP forward (frozen active set + consistency checks).
    Mirrors qpep_core.alm_qpep (same mu/rho schedule, same quadratic assembly)."""
    B = K.shape[0]
    M0, D0, G0 = assemble(K)
    q = normalize(q_init)
    t = t_init.clone()
    mu = q.new_zeros(B, P_CON)
    rho = q.new_full((B,), rho0)
    prev_viol = q.new_full((B,), float("inf"))
    hist = []
    for it in range(max_outer):
        x = v2_t(q)
        g = x @ A_T.T + t @ B_T.T                               # (B,p)
        viol = torch.clamp(g, min=0).max(dim=1).values
        hist.append(float(viol.max()))
        if it > 0 and bool((viol <= tol).all()):
            break
        act = (mu + rho.unsqueeze(1) * g) > 0
        for _cr in range(cons_rounds):
            MA, DA, GA = _alm_quadratics(M0, D0, G0, mu, rho, act)
            Mred = MA - DA.transpose(1, 2) @ torch.linalg.solve(GA, DA)
            q_new, _ = inner_solve(Mred, q, iters=inner_iters)
            x_new = v2_t(q_new)
            t_new = -torch.einsum("bki,bi->bk", torch.linalg.solve(GA, DA), x_new)
            g_new = x_new @ A_T.T + t_new @ B_T.T
            act_new = (mu + rho.unsqueeze(1) * g_new) > 0
            q, t = q_new, t_new
            if bool((act_new == act).all()):
                break
            act = act_new
        x = v2_t(q)
        g = x @ A_T.T + t @ B_T.T
        mu = torch.clamp(mu + rho.unsqueeze(1) * g, min=0.0)
        viol = torch.clamp(g, min=0).max(dim=1).values
        grow = (it >= 1) & (viol > 0.5 * prev_viol) & (viol > tol)
        rho = torch.where(grow, torch.clamp(rho * 5.0, max=rho_max), rho)
        prev_viol = torch.minimum(prev_viol, viol)
    x = v2_t(q)
    g = x @ A_T.T + t @ B_T.T
    viol = torch.clamp(g, min=0).max(dim=1).values
    J = obj_value(M0, D0, G0, q, t)
    return dict(q=q, t=t, mu=mu, act=mu > 1e-9, viol=viol, J=J,
                iters=it + 1, rho=rho, hist=hist)


def soft_solve(K: torch.Tensor, q_init: torch.Tensor, t_init: torch.Tensor,
               lam: float = 1e3, iters: int = 20, inner_iters: int = 16):
    """min J + lam * sum_j [g_j]_+^2   (quadratic penalty, no hard constraint).

    Fixed-active-set t-elimination: for a frozen set P = {j: g_j > 0} the
    penalty is quadratic in t, so t* = -G_P^{-1} D_P v2(q) and the reduced
    problem is again a quartic on S^3 (inner_solve).  Iterate on P."""
    B = K.shape[0]
    M0, D0, G0 = assemble(K)
    q = normalize(q_init)
    t = t_init.clone()
    pos = torch.zeros(B, P_CON, dtype=torch.bool)
    for it in range(iters):
        x = v2_t(q)
        g = x @ A_T.T + t @ B_T.T
        pos_new = g > 0
        coef = pos_new.to(q.dtype) * lam                      # (B,p)
        Mp = M0 + lam * torch.einsum("bp,pi,pj->bij", coef / lam, A_T, A_T)
        Dp = D0 + lam * torch.einsum("bp,pk,pi->bki", coef / lam, B_T, A_T)
        Gp = G0 + lam * torch.einsum("bp,pk,pl->bkl", coef / lam, B_T, B_T)
        Mred = Mp - Dp.transpose(1, 2) @ torch.linalg.solve(Gp, Dp)
        q_new, _ = inner_solve(Mred, q, iters=inner_iters)
        x_new = v2_t(q_new)
        t_new = -torch.einsum("bki,bi->bk", torch.linalg.solve(Gp, Dp), x_new)
        q, t = q_new, t_new
        if bool((pos_new == pos).all()) and it > 0:
            break
        pos = pos_new
    return dict(q=q, t=t, J=obj_value(M0, D0, G0, q, t), pos=pos)


def project_solve(qu: torch.Tensor, tu: torch.Tensor, iters: int = 15,
                  tol: float = 1e-12):
    """Post-hoc projection: min 1/2||q-q_u||^2 + 1/2||t-t_u||^2  s.t.  g <= 0.

    Batched SQP / active-set projection.  At each iteration the working-set
    constraints are linearised and the equality-constrained Newton step for the
    distance objective is solved (H = I on the tangent space).  The working set
    only grows: the redundant contact rows (9 coplanar points, rank 3) all stay
    in, which is exactly the LICQ-degenerate case the FB backward regularises.
    """
    B = qu.shape[0]
    quu = normalize(qu)
    q = quu.clone()
    t = tu.clone()
    eye6 = torch.eye(6, dtype=qu.dtype)
    nu = qu.new_zeros(B, P_CON)
    working = torch.zeros(B, P_CON, dtype=torch.bool)
    best_merit = qu.new_full((B,), float("inf"))
    for it in range(iters):
        x = v2_t(q)
        g = x @ A_T.T + t @ B_T.T
        working = working | (g > -1e-6)
        Bq = tangent_basis(q)
        Gq = torch.einsum("mik,mkj->mij", Jv2_t(q), Bq)
        Jg = torch.cat([torch.einsum("pi,bik->bpk", A_T, Gq),
                        B_T.unsqueeze(0).expand(B, P_CON, 3)], dim=2)   # (B,p,6)
        JgA = Jg * working.to(g.dtype).unsqueeze(-1)
        grad = torch.cat([torch.einsum("bki,bk->bi", Bq, q - quu), t - tu], dim=1)
        gA = g * working.to(g.dtype)
        eps = 1e-10
        S = torch.zeros(B, 6 + P_CON, 6 + P_CON, dtype=qu.dtype)
        S[:, :6, :6] = eye6
        S[:, :6, 6:] = JgA.transpose(1, 2)
        S[:, 6:, :6] = JgA
        S[:, 6:, 6:] = -eps * torch.eye(P_CON, dtype=qu.dtype)
        rhs = torch.cat([-grad, -gA], dim=1)
        try:
            sol = torch.linalg.solve(S, rhs.unsqueeze(-1)).squeeze(-1)
        except Exception:
            sol = torch.linalg.lstsq(S, rhs.unsqueeze(-1)).solution.squeeze(-1)
        d = sol[:, :6]
        nu = sol[:, 6:]
        if bool((d.abs().max(dim=1).values < 1e-13).all()):
            break
        cur_dist = ((q - quu) ** 2).sum(1) + ((t - tu) ** 2).sum(1)
        cur_v = torch.clamp(g, min=0).max(dim=1).values
        cur_merit = cur_dist + 1e4 * cur_v
        q_best, t_best = q.clone(), t.clone()
        merit_best = cur_merit.clone()
        alpha = torch.ones(B, 1, dtype=qu.dtype)
        accepted = torch.zeros(B, dtype=torch.bool)
        for _ls in range(14):
            q_try = normalize(q + torch.einsum("mij,mj->mi", Bq, alpha * d[:, :3]))
            t_try = t + alpha * d[:, 3:]
            g_try = v2_t(q_try) @ A_T.T + t_try @ B_T.T
            v_try = torch.clamp(g_try, min=0).max(dim=1).values
            dist_try = ((q_try - quu) ** 2).sum(1) + ((t_try - tu) ** 2).sum(1)
            merit_try = dist_try + 1e4 * v_try
            better = merit_try < merit_best
            if better.any():
                idx = better.nonzero(as_tuple=True)[0]
                q_best[idx], t_best[idx] = q_try[idx], t_try[idx]
                merit_best[idx] = merit_try[idx]
                accepted[idx] = True
            alpha = torch.where(better.unsqueeze(1), alpha, 0.5 * alpha)
            if accepted.all():
                break
        q, t = q_best, t_best
        g = v2_t(q) @ A_T.T + t @ B_T.T
        if bool((torch.clamp(g, min=0).max(dim=1).values <= tol).all() and it > 2):
            break
    x = v2_t(q)
    g = x @ A_T.T + t @ B_T.T
    return dict(q=q, t=t, nu=nu, act=working, viol=torch.clamp(g, min=0).max(dim=1).values)


# ---------------------------------------------------------------------------
# 4. implicit backward (FB-reduced saddle system) and autograd Functions
# ---------------------------------------------------------------------------
def _fb_coeffs(mu, g, eps):
    r = torch.sqrt(mu * mu + g * g + eps * eps)
    a = 1.0 - mu / r
    b = 1.0 + g / r
    return a, b


def _lagrangian_parts(K, q, t, mu_eff, want_grad=True):
    """Return (M0,D0,G0,x,w,Gq,Bq,grad_z,Htan) at fixed q,t.

    grad_z is d L / d(xi,t) (6-vector) in the normalised tangent chart."""
    B = K.shape[0]
    x = v2_t(q)
    Bq = tangent_basis(q)
    M0, D0, G0 = assemble(K)
    w = (M0 @ x.unsqueeze(-1)).squeeze(-1) + \
        (D0.transpose(1, 2) @ t.unsqueeze(-1)).squeeze(-1)
    if mu_eff is not None:
        w = w + 0.5 * torch.einsum("bp,pi->bi", mu_eff, A_T)
    Jq = Jv2_t(q)
    gq = 2.0 * torch.einsum("bki,bk->bi", Jq, w)               # (B,4)
    gt = 2.0 * ((D0 @ x.unsqueeze(-1)).squeeze(-1) +
                (G0 @ t.unsqueeze(-1)).squeeze(-1))
    if mu_eff is not None:
        gt = gt + torch.einsum("bp,pk->bk", mu_eff, B_T)
    gxi = torch.einsum("bki,bk->bi", Bq, gq)                   # (B,3)
    Gq = torch.einsum("bik,bkj->bij", Jq, Bq)                  # (B,10,3)
    Wmat = torch.einsum("bk,kij->bij", w, HV2)                # (B,4,4)
    Hxx = 2.0 * torch.einsum("bki,bkl,blj->bij", Gq, M0, Gq) + \
        2.0 * torch.einsum("bki,bkl,blj->bij", Bq, Wmat, Bq) - \
        4.0 * (w * x).sum(1).view(-1, 1, 1) * torch.eye(3, dtype=q.dtype)
    Hxt = 2.0 * torch.einsum("bki,bkl->bil", Gq, D0.transpose(1, 2))
    Htan = torch.zeros(B, 6, 6, dtype=q.dtype)
    Htan[:, :3, :3] = Hxx
    Htan[:, :3, 3:] = Hxt
    Htan[:, 3:, :3] = Hxt.transpose(1, 2)
    Htan[:, 3:, 3:] = 2.0 * G0
    grad_z = torch.cat([gxi, gt], dim=1)
    return M0, D0, G0, x, w, Gq, Bq, grad_z, Htan


def _constraint_jac(Jv2q, Bq):
    """(B,10,4),(B,4,3) -> Jg (B,p,6) = [A Gq, B]."""
    Gq = torch.einsum("bik,bkj->bij", Jv2q, Bq)
    Jgx = torch.einsum("pi,bik->bpk", A_T, Gq)
    Jgt = B_T.unsqueeze(0).expand(Jgx.shape[0], P_CON, 3)
    return torch.cat([Jgx, Jgt], dim=2), Gq


def fb_vjp(K, q, t, mu, act, grad_q, grad_t, mode, eps_fb, lam_soft):
    """VJP of the implicit layer dL/dK.

    The forward is the exact ALM / unconstrained / soft solution; the backward
    uses the FB-regularised tangent KKT system (theory (3.2)-(3.3)).  A,B do
    not depend on K, so there is no w = -d_theta g_A term."""
    B = K.shape[0]
    with torch.enable_grad():
        K_ = K.detach().clone().requires_grad_(True)
        x = v2_t(q)
        Bq = tangent_basis(q)
        g = x @ A_T.T + t @ B_T.T
        if mode == "unc":
            mu_eff = None
        elif mode == "alm":
            mu_eff = mu * act.to(mu.dtype)
        elif mode == "soft":
            mu_eff = 2.0 * lam_soft * torch.clamp(g, min=0.0)
        else:
            raise ValueError(mode)
        M0, D0, G0, _, _, _, _, grad_z, Htan = _lagrangian_parts(K_, q, t, mu_eff)
        # d grad_z / dK  (6,B,14,2)
        V = 6
        go = torch.eye(V, dtype=K.dtype).view(V, 1, V).expand(V, B, V).contiguous()
        try:
            Jz = torch.autograd.grad(grad_z, K_, grad_outputs=go, is_grads_batched=True)[0]
        except Exception:
            Jz = torch.stack([torch.autograd.grad(grad_z, K_, grad_outputs=go[i],
                                                  retain_graph=True)[0]
                              for i in range(V)], dim=0)
    Jv2q = Jv2_t(q)
    Jg, Gq = _constraint_jac(Jv2q, Bq)
    gz = torch.cat([torch.einsum("bki,bk->bi", Bq, grad_q), grad_t], dim=1)
    if mode == "unc":
        S = Htan
        rhs = gz
        lam6 = torch.linalg.solve(S + 1e-14 * torch.eye(6, dtype=K.dtype), rhs.unsqueeze(-1)).squeeze(-1)
    elif mode == "soft":
        pos = (g > 0).to(K.dtype)
        Heff = Htan + 2.0 * lam_soft * torch.einsum("bp,bpi,bpj->bij", pos, Jg, Jg)
        lam6 = torch.linalg.solve(Heff + 1e-12 * torch.eye(6, dtype=K.dtype),
                                  gz.unsqueeze(-1)).squeeze(-1)
    else:  # alm
        a, b = _fb_coeffs(mu, g, eps_fb)
        strong = a < 0.5
        coef = (~strong).to(K.dtype) * (b / a.clamp_min(1e-300))
        Heff = Htan + torch.einsum("bp,bpi,bpj->bij", coef, Jg, Jg)
        JgA = Jg * strong.to(K.dtype).unsqueeze(-1)
        Da = torch.where(strong, a / b.clamp_min(1e-300), torch.ones_like(a))
        S = torch.zeros(B, 6 + P_CON, 6 + P_CON, dtype=K.dtype)
        S[:, :6, :6] = Heff
        S[:, :6, 6:] = JgA.transpose(1, 2)
        S[:, 6:, :6] = JgA
        S[:, 6:, 6:] = -torch.diag_embed(Da)
        rhs = torch.cat([gz, gz.new_zeros(B, P_CON)], dim=1)
        try:
            sol = torch.linalg.solve(S, rhs.unsqueeze(-1)).squeeze(-1)
        except Exception:
            sol = torch.linalg.lstsq(S, rhs.unsqueeze(-1)).solution.squeeze(-1)
        lam6 = sol[:, :6]
    dK = -torch.einsum("bi,ibk->bk", lam6, Jz.reshape(V, B, -1)).reshape(K.shape)
    return dK


class PoseImplicitGrad(torch.autograd.Function):
    """Identity forward; FB-reduced implicit backward w.r.t. the keypoints."""
    @staticmethod
    def forward(ctx, K, q, t, mu, act, mode, eps_fb, lam_soft):
        ctx.save_for_backward(K, q, t, mu, act)
        ctx.mode = mode
        ctx.eps = eps_fb
        ctx.lam = lam_soft
        return q.clone(), t.clone()

    @staticmethod
    def backward(ctx, grad_q, grad_t):
        K, q, t, mu, act = ctx.saved_tensors
        dK = fb_vjp(K, q, t, mu, act, grad_q, grad_t, ctx.mode, ctx.eps, ctx.lam)
        return dK, None, None, None, None, None, None, None


class ProjectionImplicitGrad(torch.autograd.Function):
    """Identity forward for the post-hoc projection; implicit backward to z_u."""
    @staticmethod
    def forward(ctx, qu, tu, qp, tp, nu, act, eps_reg):
        ctx.save_for_backward(qu, tu, qp, tp, nu, act)
        ctx.eps = eps_reg
        return qp.clone(), tp.clone()

    @staticmethod
    def backward(ctx, grad_q, grad_t):
        qu, tu, qp, tp, nu, act = ctx.saved_tensors
        B = qp.shape[0]
        Bq = tangent_basis(qp)
        gz = torch.cat([torch.einsum("bki,bk->bi", Bq, grad_q), grad_t], dim=1)
        x = v2_t(qp)
        Wnu = torch.einsum("bp,pi->bi", nu * act.to(qp.dtype), A_T)
        Hxx = torch.eye(3, dtype=qp.dtype).expand(B, 3, 3) + \
            torch.einsum("bki,bkl,blj->bij", Bq, torch.einsum("bk,kij->bij", Wnu, HV2), Bq) - \
            2.0 * (Wnu * x).sum(1).view(-1, 1, 1) * torch.eye(3, dtype=qp.dtype)
        Jv2q = Jv2_t(qp)
        Gq = torch.einsum("bik,bkj->bij", Jv2q, Bq)
        Jgx = torch.einsum("pi,bik->bpk", A_T, Gq)
        Jgt = B_T.unsqueeze(0).expand(B, P_CON, 3)
        Jg = torch.cat([Jgx, Jgt], dim=2) * act.to(qp.dtype).unsqueeze(-1)
        H = torch.zeros(B, 6, 6, dtype=qp.dtype)
        H[:, :3, :3] = Hxx
        H[:, 3:, 3:] = torch.eye(3, dtype=qp.dtype).expand(B, 3, 3)
        S = torch.zeros(B, 6 + P_CON, 6 + P_CON, dtype=qp.dtype)
        S[:, :6, :6] = H
        S[:, :6, 6:] = Jg.transpose(1, 2)
        S[:, 6:, :6] = Jg
        S[:, 6:, 6:] = -ctx.eps * torch.eye(P_CON, dtype=qp.dtype)
        rhs = torch.cat([gz, gz.new_zeros(B, P_CON)], dim=1)
        try:
            sol = torch.linalg.solve(S, rhs.unsqueeze(-1)).squeeze(-1)
        except Exception:
            sol = torch.linalg.lstsq(S, rhs.unsqueeze(-1)).solution.squeeze(-1)
        lam6 = sol[:, :6]          # dL/dz_u (F_theta = -I)
        gq_u = torch.einsum("bij,bj->bi", Bq, lam6[:, :3])
        gt_u = lam6[:, 3:]
        return gq_u, gt_u, None, None, None, None, None


# ---------------------------------------------------------------------------
# 5. head module: the four variants, all returning a pose with gradients to K
# ---------------------------------------------------------------------------
Q_STARTS = None


def get_q_starts(n: int = 8, seed: int = 0) -> torch.Tensor:
    """Fixed, data-independent multistarts for the unconstrained solve."""
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(n, 4, generator=g, dtype=DT)
    return normalize(q)


class QPEPHead:
    """Four differentiable pose heads sharing the same interface.

    ``solve``  : forward only (detached).
    ``forward``: differentiable (implicit backward).
    ``solve_fixed`` / ``forward_fixed``: same but from an externally supplied
    initial pose (needed for a clean finite-difference gradient check: the
    init is then constant under the perturbation).
    """

    def __init__(self, mode: str, eps_fb: float = 1e-4, lam_soft: float = 1e3,
                 q_starts: torch.Tensor | None = None,
                 unc_iters: int = 22, alm_outer: int = 20, alm_inner: int = 6,
                 soft_iters: int = 30, proj_iters: int = 12):
        assert mode in ("unc", "alm", "proj", "soft")
        self.mode = mode
        self.eps_fb = eps_fb
        self.lam_soft = lam_soft
        self.q_starts = get_q_starts(8, 0) if q_starts is None else q_starts
        self.unc_iters = unc_iters
        self.alm_outer = alm_outer
        self.alm_inner = alm_inner
        self.soft_iters = soft_iters
        self.proj_iters = proj_iters

    def _unc(self, K, iters=None, q0=None):
        if q0 is None:
            q, t, J, M0, D0, G0, Mred, GinvD = unc_solve(
                K, self.q_starts, iters=self.unc_iters if iters is None else iters)
        else:
            M0, D0, G0 = assemble(K)
            Mred, GinvD = reduced_objective(K, M0, D0, G0)
            q, _ = inner_solve(Mred, normalize(q0), iters=self.unc_iters)
            t = -torch.einsum("bki,bi->bk", GinvD, v2_t(q))
        return q, t

    def solve(self, K: torch.Tensor, q_init=None, t_init=None):
        """Detached forward solve; returns q,t plus multipliers/active set."""
        with torch.no_grad():
            q0, t0 = self._unc(K, q0=q_init)
            if self.mode == "unc":
                q, t = q0, t0
                mu = K.new_zeros(K.shape[0], P_CON)
                act = torch.zeros(K.shape[0], P_CON, dtype=torch.bool)
            elif self.mode == "alm":
                r = alm_solve(K, q0, t0, max_outer=self.alm_outer,
                              inner_iters=self.alm_inner)
                q, t, mu, act = r["q"], r["t"], r["mu"], r["act"]
            elif self.mode == "soft":
                r = soft_solve(K, q0, t0, lam=self.lam_soft, iters=self.soft_iters)
                q, t = r["q"], r["t"]
                mu = K.new_zeros(K.shape[0], P_CON)
                act = torch.zeros(K.shape[0], P_CON, dtype=torch.bool)
            else:  # proj
                r = project_solve(q0, t0, iters=self.proj_iters)
                q, t, mu, act = r["q"], r["t"], r["nu"], r["act"]
        return dict(q=q, t=t, mu=mu, act=act, q0=q0, t0=t0)

    def forward(self, K: torch.Tensor):
        """Differentiable forward.  For ``proj`` the unconstrained layer is
        differentiated first and the projection VJP is chained on top."""
        if self.mode == "proj":
            with torch.no_grad():
                q0, t0 = self._unc(K)
                r = project_solve(q0, t0, iters=self.proj_iters)
            q0d, t0d = PoseImplicitGrad.apply(
                K, q0, t0, K.new_zeros(K.shape[0], P_CON),
                torch.zeros(K.shape[0], P_CON, dtype=torch.bool), "unc", self.eps_fb, 0.0)
            qd, td = ProjectionImplicitGrad.apply(
                q0d, t0d, r["q"], r["t"], r["nu"], r["act"], 1e-8)
            return qd, td
        sol = self.solve(K)
        qd, td = PoseImplicitGrad.apply(
            K, sol["q"], sol["t"], sol["mu"], sol["act"], self.mode,
            self.eps_fb, self.lam_soft)
        return qd, td

    def solve_fixed(self, K, q0, t0):
        with torch.no_grad():
            if self.mode == "proj":
                r = project_solve(q0, t0, iters=self.proj_iters)
                return dict(q=r["q"], t=r["t"], mu=r["nu"], act=r["act"])
            if self.mode == "unc":
                M0, D0, G0 = assemble(K)
                Mred, GinvD = reduced_objective(K, M0, D0, G0)
                q, _ = inner_solve(Mred, normalize(q0), iters=self.unc_iters)
                t = -torch.einsum("bki,bi->bk", GinvD, v2_t(q))
                return dict(q=q, t=t, mu=K.new_zeros(K.shape[0], P_CON),
                            act=torch.zeros(K.shape[0], P_CON, dtype=torch.bool))
            if self.mode == "alm":
                r = alm_solve(K, q0, t0, max_outer=self.alm_outer,
                              inner_iters=self.alm_inner)
                return dict(q=r["q"], t=r["t"], mu=r["mu"], act=r["act"])
            r = soft_solve(K, q0, t0, lam=self.lam_soft, iters=self.soft_iters)
            return dict(q=r["q"], t=r["t"], mu=K.new_zeros(K.shape[0], P_CON),
                        act=torch.zeros(K.shape[0], P_CON, dtype=torch.bool))

    def forward_fixed(self, K, q0, t0):
        if self.mode == "proj":
            with torch.no_grad():
                M0, D0, G0 = assemble(K)
                Mred, GinvD = reduced_objective(K, M0, D0, G0)
                qu, _ = inner_solve(Mred, normalize(q0), iters=self.unc_iters)
                tu = -torch.einsum("bki,bi->bk", GinvD, v2_t(qu))
                r = project_solve(qu, tu, iters=self.proj_iters)
            qud, tud = PoseImplicitGrad.apply(
                K, qu, tu, K.new_zeros(K.shape[0], P_CON),
                torch.zeros(K.shape[0], P_CON, dtype=torch.bool), "unc", self.eps_fb, 0.0)
            qd, td = ProjectionImplicitGrad.apply(
                qud, tud, r["q"], r["t"], r["nu"], r["act"], 1e-8)
            return qd, td
        sol = self.solve_fixed(K, q0, t0)
        qd, td = PoseImplicitGrad.apply(
            K, sol["q"], sol["t"], sol["mu"], sol["act"], self.mode,
            self.eps_fb, self.lam_soft)
        return qd, td


# ---------------------------------------------------------------------------
# 6. synthetic dataset: biased/noisy desk scenes -> pixel keypoints
# ---------------------------------------------------------------------------
P_OBJ_T = torch.tensor(qc.P_OBJ, dtype=DT)          # (14,3)
V_DENSE_T = torch.tensor(qc.dense_bottom(21), dtype=DT)   # (441,3) object-frame bottom


def make_split(seed_base: int, n: int, p: int = P_DEF):
    """n scenes with seeds seed_base..seed_base+n-1 (same generator as the
    main table: qpep_core.make_instance, bias_z=0.08, bias_xy=0.005,
    per-point noise 0.01 m, GT exactly touching the table)."""
    Ks, qs, ts, Rs = [], [], [], []
    for i in range(n):
        inst = qc.make_instance(seed_base + i, p)
        Ks.append(bearings_to_keypoints(inst["U"]))
        qs.append(inst["q_gt"])
        ts.append(inst["t_gt"])
        Rs.append(inst["R_gt"])
    K = (np.stack(Ks) - C_PX) / F_PX
    return dict(K=torch.tensor(K, dtype=DT),
                q=torch.tensor(np.stack(qs), dtype=DT),
                t=torch.tensor(np.stack(ts), dtype=DT),
                R=torch.tensor(np.stack(Rs), dtype=DT))


# ---------------------------------------------------------------------------
# 7. backbone (keypoint refiner) and pose loss
# ---------------------------------------------------------------------------
class KeypointRefiner(nn.Module):
    """28 -> 64 -> 64 -> 28 residual MLP; zero-initialised last layer so the
    initial head input is exactly the observed keypoints."""

    def __init__(self, hidden: int = 64, bound: float = 0.03):
        super().__init__()
        self.bound = bound
        self.fc1 = nn.Linear(2 * N_PTS, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, 2 * N_PTS)
        nn.init.zeros_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, K: torch.Tensor) -> torch.Tensor:
        z = torch.tanh(self.fc1(K))
        z = torch.tanh(self.fc2(z))
        return K + self.bound * torch.tanh(self.fc3(z))


def pose_loss(q, t, R_gt, t_gt):
    """GT-pose loss (no constraint penalty): squared ADD / (2 cm)^2 plus a
    chordal rotation term (weight 400).  Only pose supervision."""
    R = R_of_t(q)
    dR = R - R_gt
    dp = torch.einsum("bij,nj->bni", dR, P_OBJ_T) + (t - t_gt).unsqueeze(1)
    add2 = (dp * dp).sum(dim=(1, 2)) / N_PTS
    rot2 = (dR * dR).sum(dim=(1, 2))
    return (add2 / 0.02 ** 2 + 400.0 * rot2).mean()


@torch.no_grad()
def eval_metrics(q, t, R_gt, t_gt):
    """ADD (mm), rotation (deg), translation (mm), penetration (mm),
    constraint violation, feasibility."""
    q = q.detach(); t = t.detach()
    R = R_of_t(q)
    dR = R - R_gt
    dp = torch.einsum("bij,nj->bni", dR, P_OBJ_T) + (t - t_gt).unsqueeze(1)
    add = torch.sqrt((dp * dp).sum(dim=2)).mean(dim=1) * 1000.0
    tr = ((R * R_gt).sum(dim=(1, 2)) - 1.0) / 2.0
    rot = torch.arccos(torch.clamp(tr, -1.0, 1.0)) * 180.0 / math.pi
    trans = torch.linalg.norm(t - t_gt, dim=1) * 1000.0
    Rv = torch.einsum("bij,nj->bni", R, V_DENSE_T)              # (B,441,3)
    zmin = Rv[:, :, 2].min(dim=1).values + t[:, 2]
    pen = torch.clamp(H_TAB - zmin, min=0.0) * 1000.0
    x = v2_t(q)
    g = x @ A_T.T + t @ B_T.T
    viol = torch.clamp(g, min=0.0).max(dim=1).values
    return dict(add=add, rot=rot, trans=trans, pen=pen, viol=viol,
                feas=(viol <= 1e-6))


def stat(t: torch.Tensor):
    a = t.detach().cpu().numpy().astype(float)
    return dict(mean=float(np.mean(a)), std=float(np.std(a)),
                median=float(np.median(a)), p90=float(np.percentile(a, 90)),
                max=float(np.max(a)), n=int(a.size))


# ---------------------------------------------------------------------------
# 8. training
# ---------------------------------------------------------------------------
def _seed_all(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def train_one(mode: str, seed: int, tr, va, cfg):
    _seed_all(seed)
    model = KeypointRefiner(hidden=cfg["hidden"], bound=cfg["bound"]).to(DT)
    head = QPEPHead(mode, eps_fb=cfg["eps_fb"], lam_soft=cfg["lam_soft"],
                    q_starts=get_q_starts(cfg["n_starts"], 0),
                    unc_iters=cfg["unc_iters"], alm_outer=cfg["alm_outer"],
                    alm_inner=cfg["alm_inner"], soft_iters=cfg["soft_iters"],
                    proj_iters=cfg["proj_iters"])
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    N = tr["K"].shape[0]
    curves = dict(epoch=[], train_loss=[], val_add=[], val_rot=[], val_pen=[], val_feas=[])
    t_start = time.perf_counter()
    fwd_time = 0.0
    for ep in range(cfg["epochs"]):
        perm = torch.randperm(N)
        tot = 0.0
        for b in range(0, N, cfg["batch"]):
            idx = perm[b:b + cfg["batch"]]
            K = model(tr["K"][idx].reshape(len(idx), -1))
            q, t = head.forward(K.reshape(-1, N_PTS, 2))
            loss = pose_loss(q, t, tr["R"][idx], tr["t"][idx])
            opt.zero_grad()
            t0 = time.perf_counter()
            loss.backward()
            if b == 0:
                fwd_time += time.perf_counter() - t0
            opt.step()
            tot += float(loss.detach()) * len(idx)
        if ep % cfg["eval_every"] == 0 or ep == cfg["epochs"] - 1:
            solv = head.solve(va["K"])
            m = eval_metrics(solv["q"], solv["t"], va["R"], va["t"])
            curves["epoch"].append(ep)
            curves["train_loss"].append(tot / N)
            curves["val_add"].append(float(m["add"].mean()))
            curves["val_rot"].append(float(m["rot"].mean()))
            curves["val_pen"].append(float(m["pen"].mean()))
            curves["val_feas"].append(float(m["feas"].to(DT).mean()))
    wall = time.perf_counter() - t_start
    solv = head.solve(va["K"])
    m = eval_metrics(solv["q"], solv["t"], va["R"], va["t"])
    solt = head.solve(tr["K"])
    mt = eval_metrics(solt["q"], solt["t"], tr["R"], tr["t"])
    # per-sample timing of the differentiable training step at B=32
    Bt = min(32, va["K"].shape[0])
    Kb = va["K"][:Bt].reshape(Bt, -1)
    for _ in range(2):
        qq, tt = head.forward(model(Kb).reshape(Bt, N_PTS, 2))
        loss = pose_loss(qq, tt, va["R"][:Bt], va["t"][:Bt])
        opt.zero_grad(); loss.backward()
    t0 = time.perf_counter()
    for _ in range(3):
        qq, tt = head.forward(model(Kb).reshape(Bt, N_PTS, 2))
        loss = pose_loss(qq, tt, va["R"][:Bt], va["t"][:Bt])
        opt.zero_grad(); loss.backward()
    step_ms = (time.perf_counter() - t0) / 3.0 * 1000.0
    return dict(mode=mode, seed=seed, wall_s=wall, curves=curves,
                val=dict(add=stat(m["add"]), rot=stat(m["rot"]), trans=stat(m["trans"]),
                         pen=stat(m["pen"]), viol=stat(m["viol"]),
                         feas=float(m["feas"].to(DT).mean())),
                train=dict(add=stat(mt["add"]), rot=stat(mt["rot"]),
                           pen=stat(mt["pen"]), feas=float(mt["feas"].to(DT).mean())),
                step_ms_per_batch=step_ms,
                ms_per_pose=step_ms / Bt)


def identity_layer_metrics(va, cfg):
    """Layer-level reference: run each head on the *observed* keypoints with no
    learned refiner (the backbone is the identity)."""
    out = {}
    for mode in ("unc", "alm", "proj", "soft"):
        head = QPEPHead(mode, eps_fb=cfg["eps_fb"], lam_soft=cfg["lam_soft"],
                        q_starts=get_q_starts(cfg["n_starts"], 0),
                        unc_iters=18, alm_outer=20, alm_inner=6,
                        soft_iters=18, proj_iters=12)
        t0 = time.perf_counter()
        sol = head.solve(va["K"])
        dt = time.perf_counter() - t0
        m = eval_metrics(sol["q"], sol["t"], va["R"], va["t"])
        out[mode] = dict(add=stat(m["add"]), rot=stat(m["rot"]), trans=stat(m["trans"]),
                         pen=stat(m["pen"]), viol=stat(m["viol"]),
                         feas=float(m["feas"].to(DT).mean()),
                         ms_per_pose=dt / va["K"].shape[0] * 1000.0)
    return out


# ---------------------------------------------------------------------------
# 9. port self-test against the NumPy reference (qpep_core)
# ---------------------------------------------------------------------------
def run_selftest(n: int = 6, p: int = P_DEF):
    rows = []
    for s in range(n):
        inst = qc.make_instance(s, p)
        K = torch.tensor((bearings_to_keypoints(inst["U"]) - C_PX) / F_PX, dtype=DT)[None]
        row = dict(seed=s)
        # unconstrained
        q_t, t_t, J_t, *_ = unc_solve(K, get_q_starts(8, 0), iters=22)
        J_n, q_n, t_n = qc.solve_qp_global(inst["M0"], inst["D0"], inst["G0"],
                                           n_starts=48, q0=inst["q_gt"])
        row["unc_dJ"] = abs(float(J_t) - float(J_n))
        row["unc_dq"] = float(min(np.linalg.norm(q_t[0].numpy() - q_n),
                                  np.linalg.norm(q_t[0].numpy() + q_n)))
        row["unc_dt"] = float(np.linalg.norm(t_t[0].numpy() - t_n))
        # ALM
        r = alm_solve(K, q_t, t_t, max_outer=24, inner_iters=7)
        rn = qc.alm_qpep(inst["M0"], inst["D0"], inst["G0"], inst["A"], inst["B"],
                         inst["q_gt"], inst["t_gt"], n_starts=6)
        row["alm_dJ"] = abs(float(r["J"]) - float(rn["J"]))
        row["alm_dq"] = float(min(np.linalg.norm(r["q"][0].numpy() - rn["q"]),
                                  np.linalg.norm(r["q"][0].numpy() + rn["q"])))
        row["alm_dt"] = float(np.linalg.norm(r["t"][0].numpy() - rn["t"]))
        row["alm_viol_torch"] = float(r["viol"])
        row["alm_viol_numpy"] = float(rn["viol"])
        # projection
        rp = project_solve(q_t, t_t)
        rpn = qc.project_feasible(inst["M0"], inst["D0"], inst["G0"], inst["A"], inst["B"],
                                  q_t[0].numpy(), t_t[0].numpy(), n_starts=4)
        row["proj_dq"] = float(min(np.linalg.norm(rp["q"][0].numpy() - rpn["q"]),
                                   np.linalg.norm(rp["q"][0].numpy() + rpn["q"])))
        row["proj_dt"] = float(np.linalg.norm(rp["t"][0].numpy() - rpn["t"]))
        row["proj_viol"] = float(rp["viol"])
        # soft (penalised objective value, same lambda)
        rs = soft_solve(K, q_t, t_t, lam=1e3)
        rsn = qc.soft_penalty(inst["M0"], inst["D0"], inst["G0"], inst["A"], inst["B"],
                              1e3, q_t[0].numpy(), t_t[0].numpy(), n_starts=6)
        g_s = E_g = (v2_t(rs["q"]) @ A_T.T + rs["t"] @ B_T.T)
        g_n = inst["A"] @ qc.v2(rsn["q"]) + inst["B"] @ rsn["t"]
        L_s = float(rs["J"] + 1e3 * (torch.clamp(g_s, min=0) ** 2).sum())
        L_n = float(rsn["J"] + 1e3 * np.maximum(g_n, 0).dot(np.maximum(g_n, 0)))
        row["soft_dL"] = abs(L_s - L_n)
        row["soft_viol_torch"] = float(torch.clamp(g_s, min=0).max())
        row["soft_viol_numpy"] = float(np.maximum(g_n, 0).max())
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# 10. finite-difference gradient check of the implicit backward
# ---------------------------------------------------------------------------
def _linear_loss(q, t, Wr, Wt):
    R = R_of_t(q)
    return (R * Wr).sum(dim=(1, 2)) + (t * Wt).sum(dim=1)


def _rel_stats(fd: np.ndarray, an: np.ndarray, floor: float = 1e-8):
    diff = np.abs(fd - an)
    rel = diff / np.maximum(np.abs(fd), floor)
    return dict(max_abs=float(diff.max()),
                max_rel=float(rel.max()),
                median_rel=float(np.median(rel)),
                rel_norm=float(diff.max() / max(np.abs(fd).max(), floor)))


def gradcheck(cfg, seeds=(800000, 800001, 800002), h: float = 1e-6,
              eps_list=(1e-2, 1e-3, 1e-4, 1e-6)):
    torch.manual_seed(0)
    Wr = torch.randn(3, 3, dtype=DT)
    Wt = torch.randn(3, dtype=DT)
    results = []
    for s in seeds:
        inst = qc.make_instance(s, P_DEF)
        Kpix = bearings_to_keypoints(inst["U"])
        Kn = torch.tensor((Kpix - C_PX) / F_PX, dtype=DT)[None]
        head_unc = QPEPHead("unc")
        sol0 = head_unc.solve(Kn)
        q_init, t_init = sol0["q"].clone(), sol0["t"].clone()
        # finite differences (same fixed init)
        fd = {}
        for mode in ("unc", "alm", "proj", "soft"):
            head = QPEPHead(mode, eps_fb=1e-6, unc_iters=18, alm_outer=20,
                            alm_inner=6, soft_iters=18, proj_iters=12)
            vals = np.zeros((2, P_CON * 0 + 2 * N_PTS))
            for ci in range(N_PTS):
                for cj in range(2):
                    for si, sgn in enumerate((+1.0, -1.0)):
                        Kx = Kpix.copy()
                        Kx[ci, cj] += sgn * h
                        Kx = torch.tensor((Kx - C_PX) / F_PX, dtype=DT)[None]
                        with torch.no_grad():
                            qq, tt = head.forward_fixed(Kx, q_init, t_init)
                        vals[si, 2 * ci + cj] = float(_linear_loss(qq, tt, Wr, Wt))
            fd[mode] = (vals[0] - vals[1]) / (2 * h)
        # analytic gradients
        for mode in ("unc", "alm", "proj", "soft"):
            Kt = Kn.clone().requires_grad_(True)
            head = QPEPHead(mode, eps_fb=1e-6, unc_iters=18, alm_outer=20,
                            alm_inner=6, soft_iters=18, proj_iters=12)
            q, t = head.forward_fixed(Kt, q_init, t_init)
            g = torch.autograd.grad(_linear_loss(q, t, Wr, Wt).sum(), Kt)[0]
            an = g[0].numpy().reshape(-1)
            st = _rel_stats(fd[mode], an)
            # straight-through control: zero keypoint gradient
            st_st = _rel_stats(fd[mode], np.zeros_like(an))
            results.append(dict(seed=s, mode=mode, eps_fb=None, fd=fd[mode].tolist(),
                                analytic=an.tolist(), **st, straight_through_max_rel=st_st["max_rel"]))
        # ALM with different FB epsilons (analytic only; same FD)
        for eps in eps_list:
            Kt = Kn.clone().requires_grad_(True)
            head = QPEPHead("alm", eps_fb=eps, unc_iters=18, alm_outer=20, alm_inner=6)
            q, t = head.forward_fixed(Kt, q_init, t_init)
            g = torch.autograd.grad(_linear_loss(q, t, Wr, Wt).sum(), Kt)[0]
            an = g[0].numpy().reshape(-1)
            st = _rel_stats(fd["alm"], an)
            results.append(dict(seed=s, mode="alm", eps_fb=eps, fd=fd["alm"].tolist(),
                                analytic=an.tolist(), **st))
    # summary per (mode, eps)
    summary = {}
    for r in results:
        key = r["mode"] if r["eps_fb"] is None else "alm_eps_%.0e" % r["eps_fb"]
        summary.setdefault(key, []).append(r)
    out = {}
    for k, rs in summary.items():
        out[k] = dict(n=len(rs),
                      max_abs=max(r["max_abs"] for r in rs),
                      max_rel=max(r["max_rel"] for r in rs),
                      median_rel=float(np.median([r["median_rel"] for r in rs])),
                      rel_norm=max(r["rel_norm"] for r in rs))
    return dict(config=dict(seeds=list(seeds), h=h, eps_list=list(eps_list),
                            loss="sum(R*Wr)+t.Wt, fixed init, float64"),
                rows=results, summary=out)


# ---------------------------------------------------------------------------
# 11. report generation (JSON + Markdown + PNG)
# ---------------------------------------------------------------------------
def _mean_std(vals):
    a = np.asarray(vals, dtype=float)
    return float(a.mean()), float(a.std(ddof=0))


def summarise_runs(runs):
    modes = []
    for r in runs:
        if r["mode"] not in modes:
            modes.append(r["mode"])
    out = {}
    for mode in modes:
        rs = [r for r in runs if r["mode"] == mode]
        d = dict(n=len(rs))
        for key in ("add", "rot", "trans", "pen"):
            m, s = _mean_std([r["val"][key]["mean"] for r in rs])
            d[key + "_mean"] = m
            d[key + "_std"] = s
            d[key + "_median"] = float(np.median([r["val"][key]["median"] for r in rs]))
            d[key + "_p90"] = float(np.median([r["val"][key]["p90"] for r in rs]))
        d["feas_mean"] = float(np.mean([r["val"]["feas"] for r in rs]))
        d["feas_std"] = float(np.std([r["val"]["feas"] for r in rs]))
        d["wall_s_mean"] = float(np.mean([r["wall_s"] for r in rs]))
        d["ms_per_pose_mean"] = float(np.mean([r["ms_per_pose"] for r in rs]))
        d["step_ms_mean"] = float(np.mean([r["step_ms_per_batch"] for r in rs]))
        out[mode] = d
    return out


MODE_LABEL = {"unc": "Unconstrained QPEP", "alm": "ALM-QPEP (ours)",
              "proj": "Post-hoc projection", "soft": "Soft penalty ($\\lambda$=1e3)"}


def write_report(data, md_path: Path, png_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs, summ = data["runs"], data["summary"]
    modes = [m for m in ("unc", "alm", "proj", "soft") if m in summ]
    # ---------------- figure ----------------
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.6))
    ax = axes[0, 0]
    for mode in modes:
        rs = [r for r in runs if r["mode"] == mode]
        ep = rs[0]["curves"]["epoch"]
        Y = np.array([r["curves"]["val_add"] for r in rs])
        ax.plot(ep, Y.mean(0), label=MODE_LABEL[mode])
        ax.fill_between(ep, Y.mean(0) - Y.std(0), Y.mean(0) + Y.std(0), alpha=0.18)
    ax.set_xlabel("epoch"); ax.set_ylabel("val ADD (mm)"); ax.set_yscale("log")
    ax.set_title("held-out ADD during training"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax = axes[0, 1]
    xs = np.arange(len(modes))
    vals = [summ[m]["add_mean"] for m in modes]
    errs = [summ[m]["add_std"] for m in modes]
    ax.bar(xs, vals, yerr=errs, capsize=4, color=["#888", "#1f77b4", "#2ca02c", "#ff7f0e"])
    ax.set_xticks(xs); ax.set_xticklabels([m for m in modes])
    ax.set_ylabel("final val ADD (mm)"); ax.set_yscale("log")
    ax.set_title("final ADD (mean $\\pm$ std over seeds)"); ax.grid(alpha=0.3, axis="y")
    ax = axes[1, 0]
    vals = [summ[m]["pen_mean"] for m in modes]
    errs = [summ[m]["pen_std"] for m in modes]
    ax.bar(xs, vals, yerr=errs, capsize=4, color=["#888", "#1f77b4", "#2ca02c", "#ff7f0e"])
    ax.set_xticks(xs); ax.set_xticklabels(modes)
    ax.set_ylabel("penetration (mm)"); ax.set_title("final penetration (mean $\\pm$ std)")
    ax.grid(alpha=0.3, axis="y")
    ax = axes[1, 1]
    gs = (data.get("gradcheck") or {}).get("summary", {})
    lab = [k for k in ("unc", "alm", "alm_eps_1e-03", "alm_eps_1e-04",
                       "alm_eps_1e-06", "proj", "soft") if k in gs]
    if lab:
        val = [max(gs[k]["max_rel"], 1e-12) for k in lab]
        ax.bar(np.arange(len(lab)), val, color="#1f77b4")
        ax.set_yscale("log"); ax.set_xticks(np.arange(len(lab)))
        ax.set_xticklabels(lab, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("max relative gradient error")
        ax.set_title("FB backward vs central differences"); ax.grid(alpha=0.3, axis="y")
        for i, v in enumerate(val):
            ax.text(i, v * 1.5, "%.1e" % v, ha="center", fontsize=7)
    else:
        ax.text(0.5, 0.5, "gradcheck not in this run", ha="center", transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)

    # ---------------- markdown ----------------
    L = []
    A = L.append
    A("# 端到端训练玩具 + FB 反向梯度检查（CPU）\n")
    A("> 由 `experiments/code/e2e_toy.py` 自动生成；数字均来自本文件 `--phase all` 的一次运行。\n")
    A("## 0. 范围（先说不做什么）\n")
    A("- **不是 VLA / 不是检测器 / 不是 GPU 规模实验**：输入是合成的 14 点 2D 关键点（含 "
      "qpep_core 的 0.08 m 系统性外参偏差 + σ=0.01 m 逐点噪声），前端是一个 28→64→64→28 的残差 MLP "
      "（零初始化末层，初始等价于恒等）。没有图像/检测器/真实数据。\n")
    A("- **层是主角**：四个位姿头共享数据/骨干/种子/预算，只换约束处理方式；监督只有 GT 位姿损失，"
      "不加穿透惩罚。\n")
    A("- 前向 ALM 用 `qpep_core` 同一套乘子/ρ 调度与定理 2.1 装配（批量化 torch 实现），"
      "反向我用 `theory/proofs.md` §3 的 FB 约化鞍点系统，**前向精确 KKT、反向借 FB 正则化微分**。\n")
    A("- 这里是 prototype 级 sanity；不可当作论文主表或「学习到的 VLA 管线」的证据。\n")
    A("## 1. 设置\n")
    A("| 项 | 值 |\n|---|---|")
    A("| 场景 | 合成桌面 PnP，N=14 点，p=16（9 非穿透 + 4 正深度 + 1 FoV 对 + 1 倾角） |")
    A("| 观测 | 14 个 2D 关键点（32×32 图像、f=60 px 的透视投影；偏差 0.08 m + σ=0.01 m） |")
    A("| 骨干 | 28→64→64→28 tanh MLP，残差界 0.03（归一化像素），零初始化末层 |")
    A("| 训练 | Adam lr=%.0e，batch=%d，%d epochs，train=%d / val=%d，%d seeds |"
      % (data["meta"]["cfg"]["lr"], data["meta"]["cfg"]["batch"], data["meta"]["cfg"]["epochs"],
         data["meta"]["n_train"], data["meta"]["n_val"], len(data["meta"]["seeds"])))
    A("| 损失 | 仅位姿：mean_i ||ΔR p_i+Δt||²/(0.02 m)² + 400·||R-R_gt||_F² |")
    A("| 硬件 | CPU（单线程，login node） |")
    A("## 2. 四头前向/反向\n")
    A("| head | forward | backward |\n|---|---|---|")
    A("| unc | S³ 上多起点 Levenberg-Newton（%d starts），t 解析消去 | 球面 KKT 隐微分（6×6） |" % data["meta"]["cfg"]["n_starts"])
    A("| alm | 冻结活跃集 ALM，定理 2.1 装配 + 一致性检查 | **FB 鞍点系统** (3.2)-(3.3)，ε=%.0e |" % data["meta"]["cfg"]["eps_fb"])
    A("| proj | 无约束解 → SQP 投影（工作集只增） | 投影 KKT 隐微分（H=I+Σμ∇²g） |")
    A("| soft | 冻结 g>0 集消去 t → S³ 四次型，迭代 | 罚函数光滑 KKT 隐微分 |")
    A("## 3. 梯度检查（主体证据）\n")
    A("固定初值（不做 warm-start 的参数依赖），中心差分 h=%.0e，损失 L=sum(R*Wr)+t·Wt，float64。" % (data["gradcheck"]["config"]["h"] if data.get("gradcheck") else 0.0))
    A("")
    A("| 模式 | n | max abs err | max rel err | median rel err | max|Δg|/max|g| |")
    A("|---|---|---|---|---|---|")
    for k in ("unc", "alm", "alm_eps_1e-02", "alm_eps_1e-03", "alm_eps_1e-04",
              "alm_eps_1e-06", "proj", "soft"):
        if k in gs:
            g = gs[k]
            A("| %s | %d | %.2e | %.2e | %.2e | %.2e |" % (k, g["n"], g["max_abs"],
              g["max_rel"], g["median_rel"], g["rel_norm"]))
    A("")
    st = [r for r in (data.get("gradcheck") or {}).get("rows", []) if r["mode"] == "alm" and r["eps_fb"] is None]
    if st:
        A("直通（straight-through，∂L/∂K=0）在同一差分上的 max relative error = **%.2f**（即完全错）。"
          % max(r["straight_through_max_rel"] for r in st))
    A("")
    A("## 4. 层级别（无学习前端；直接在观测关键点上跑四头）\n")
    A("| head | ADD mm (median) | rot deg | trans mm | pen mm | feasible | ms/pose |")
    A("|---|---|---|---|---|---|---|")
    for m in modes:
        if m in data["layer_metrics"]:
            d = data["layer_metrics"][m]
            A("| %s | %.2f | %.2f | %.2f | %.2f | %.1f%% | %.1f |" % (
                MODE_LABEL[m], d["add"]["median"], d["rot"]["median"], d["trans"]["median"],
                d["pen"]["median"], 100 * d["feas"], d["ms_per_pose"]))
    A("")
    A("## 5. 端到端训练结果（val，mean±std over seeds）\n")
    A("| head | ADD mm | rot deg | trans mm | pen mm | feasible | wall s | ms/pose |")
    A("|---|---|---|---|---|---|---|---|")
    for m in modes:
        d = summ[m]
        A("| %s | %.2f ± %.2f | %.2f ± %.2f | %.2f ± %.2f | %.2f | %.1f%% | %.0f | %.1f |" % (
            MODE_LABEL[m], d["add_mean"], d["add_std"], d["rot_mean"], d["rot_std"],
            d["trans_mean"], d["trans_std"], d["pen_mean"], 100 * d["feas_mean"],
            d["wall_s_mean"], d["ms_per_pose_mean"]))
    A("")
    A("| head | ADD median | ADD p90 | pen median | pen p90 |\n|---|---|---|---|---|")
    for m in modes:
        d = summ[m]
        A("| %s | %.2f | %.2f | %.2f | %.2f |" % (MODE_LABEL[m], d["add_median"],
          d["add_p90"], d["pen_median"], d["pen_p90"]))
    A("")
    A("![curves](%s)\n" % png_path.name)
    A("## 6. 逐种子结果\n")
    A("| head | seed | ADD mm | rot deg | trans mm | pen mm | feasible | wall s |")
    A("|---|---|---|---|---|---|---|---|")
    for r in runs:
        A("| %s | %d | %.2f | %.2f | %.2f | %.2f | %.1f%% | %.0f |" % (
            r["mode"], r["seed"], r["val"]["add"]["mean"], r["val"]["rot"]["mean"],
            r["val"]["trans"]["mean"], r["val"]["pen"]["mean"],
            100 * r["val"]["feas"], r["wall_s"]))
    A("")
    A("## 7. ALM 前向在环（收敛/速度）\n")
    A(data["meta"]["alm_stats_text"])
    A("")
    A("## 8. 诚实的边界\n")
    A("- CPU 玩具、合成数据、**关键点输入**（不是图像/检测器），14 点、p=16、单物体；"
      "不构成 VLA 或真实操作管线的主张。\n")
    A("- 前向是数值局部解（多起点），不声称全局最优，也没有跑 QPEP Gröbner 枚举。\n")
    A("- FB 反向在**固定初值**下与中心差分一致（见上表）；训练时 warm-start 依赖输入但被 detach，"
      "这是显式的近似；活跃集切换/多起点 argmin 跳变处梯度不建模（与 lab 03 的 IFT 玩具同样的 caveat）。\n")
    A("- 软惩罚在 λ=1e3 下用改进的固定活跃集消元求解，数值上很接近可行；它没有硬约束和证书，"
      "故它在这里是 baseline 而非结论。\n")
    md_path.write_text("\n".join(L) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# 12. CLI
# ---------------------------------------------------------------------------
def _log(msg):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", default="all",
                    choices=["selftest", "gradcheck", "train", "quick", "all", "report"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--n-train", type=int, default=192)
    ap.add_argument("--n-val", type=int, default=48)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--bound", type=float, default=0.03)
    ap.add_argument("--n-starts", type=int, default=6)
    ap.add_argument("--eps-fb", type=float, default=1e-6)
    ap.add_argument("--lam-soft", type=float, default=1e3)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--modes", nargs="+", default=["unc", "alm", "proj", "soft"])
    ap.add_argument("--out-results", default=str(HERE.parent / "results"))
    ap.add_argument("--out-root", default=str(HERE.parent.parent / "results"))
    args = ap.parse_args(argv)

    out_res = Path(args.out_results)
    out_root = Path(args.out_root)
    out_res.mkdir(parents=True, exist_ok=True)
    out_root.mkdir(parents=True, exist_ok=True)

    cfg = dict(seeds=args.seeds, epochs=args.epochs, batch=args.batch, lr=args.lr,
               hidden=args.hidden, bound=args.bound, n_starts=args.n_starts,
               eps_fb=args.eps_fb, lam_soft=args.lam_soft, eval_every=args.eval_every,
               unc_iters=12, alm_outer=12, alm_inner=3, soft_iters=14, proj_iters=10)
    if args.phase == "quick":
        cfg.update(epochs=3, eval_every=1)
        args.seeds = [0]
        args.n_train = 32
        args.n_val = 8
        args.modes = ["unc", "alm"]

    selftest_rows = []
    grad = None
    runs = []
    layer_metrics = {}
    alm_stats_text = "not measured"

    t_all = time.perf_counter()
    if args.phase in ("selftest", "all"):
        _log("selftest: torch forward vs NumPy reference (qpep_core) ...")
        selftest_rows = run_selftest(6)
        for r in selftest_rows:
            _log("  seed %d: unc dJ=%.1e dq=%.1e dt=%.1e | alm dJ=%.1e dq=%.1e dt=%.1e "
                 "(viol %.1e vs %.1e) | proj dq=%.1e dt=%.1e | soft dL=%.1e"
                 % (r["seed"], r["unc_dJ"], r["unc_dq"], r["unc_dt"], r["alm_dJ"],
                    r["alm_dq"], r["alm_dt"], r["alm_viol_torch"], r["alm_viol_numpy"],
                    r["proj_dq"], r["proj_dt"], r["soft_dL"]))
        (out_res / "e2e_selftest.json").write_text(json.dumps(selftest_rows, indent=1))

    if args.phase in ("gradcheck", "all"):
        _log("gradcheck: FB backward vs central finite differences ...")
        t0 = time.perf_counter()
        grad = gradcheck(cfg)
        _log("  gradcheck done in %.0f s" % (time.perf_counter() - t0))
        for k, v in grad["summary"].items():
            _log("  %-14s n=%d max_rel=%.2e median_rel=%.2e max_abs=%.2e"
                 % (k, v["n"], v["max_rel"], v["median_rel"], v["max_abs"]))
        (out_res / "e2e_gradcheck.json").write_text(json.dumps(grad, indent=1))

    if args.phase in ("train", "quick", "all"):
        _log("dataset: %d train / %d val scenes per seed ..." % (args.n_train, args.n_val))
        va = make_split(800000, args.n_val)
        layer_metrics = identity_layer_metrics(va, cfg)
        for m, d in layer_metrics.items():
            _log("  layer %-4s (no learning): ADD %.1f mm  pen %.1f mm  feas %.0f%%"
                 % (m, d["add"]["median"], d["pen"]["median"], 100 * d["feas"]))
        # ALM forward-in-the-loop stats on the val set
        head = QPEPHead("alm", eps_fb=cfg["eps_fb"], q_starts=get_q_starts(cfg["n_starts"], 0),
                        unc_iters=cfg["unc_iters"], alm_outer=cfg["alm_outer"],
                        alm_inner=cfg["alm_inner"])
        t0 = time.perf_counter()
        sol_a = head.solve(va["K"])
        ms = (time.perf_counter() - t0) / va["K"].shape[0] * 1000.0
        g = v2_t(sol_a["q"]) @ A_T.T + sol_a["t"] @ B_T.T
        viol = torch.clamp(g, min=0).max(dim=1).values
        alm_stats_text = ("val batch %d: ALM forward **%.1f ms/pose** (CPU single thread), "
                          "max violation %.2e, feasible %.0f%%; "
                          "the remaining `e2e_%s` runs use the same forward inside the loop."
                          % (va["K"].shape[0], ms, float(viol.max()),
                             100.0 * float((viol <= 1e-6).to(DT).mean()),
                             "alm"))
        _log("  " + alm_stats_text)
        for seed in args.seeds:
            tr = make_split(10000 + 1000 * seed, args.n_train)
            for mode in args.modes:
                _log("train mode=%s seed=%d ..." % (mode, seed))
                r = train_one(mode, seed, tr, va, cfg)
                runs.append(r)
                _log("  mode=%s seed=%d val ADD %.2f mm rot %.2f deg trans %.2f mm "
                     "pen %.2f mm feas %.0f%% wall %.0f s"
                     % (mode, seed, r["val"]["add"]["mean"], r["val"]["rot"]["mean"],
                        r["val"]["trans"]["mean"], r["val"]["pen"]["mean"],
                        100 * r["val"]["feas"], r["wall_s"]))
        (out_res / "e2e_main.json").write_text(json.dumps(runs, indent=1))
        (out_res / "e2e_layer.json").write_text(json.dumps(layer_metrics, indent=1))

    if args.phase == "report":
        grad = json.loads((out_res / "e2e_gradcheck.json").read_text())
        runs = json.loads((out_res / "e2e_main.json").read_text())
        layer_metrics = json.loads((out_res / "e2e_layer.json").read_text())
        selftest_rows = json.loads((out_res / "e2e_selftest.json").read_text()) \
            if (out_res / "e2e_selftest.json").exists() else []

    if args.phase in ("train", "quick", "all", "report") and runs:
        summary = summarise_runs(runs)
        meta = dict(cfg=cfg, seeds=args.seeds, n_train=args.n_train, n_val=args.n_val,
                    p=P_DEF, image="32x32, f=60 px", n_points=N_PTS,
                    torch=torch.__version__, numpy=np.__version__,
                    seconds=time.perf_counter() - t_all,
                    alm_stats_text=alm_stats_text,
                    scope="CPU toy; 2D keypoint input (no image/detector); synthetic desk PnP; p=16")
        data = dict(meta=meta, selftest=selftest_rows, gradcheck=grad,
                    layer_metrics=layer_metrics, runs=runs, summary=summary)
        (out_root / "e2e_toy.json").write_text(json.dumps(data, indent=1))
        write_report(data, out_root / "e2e_toy.md", out_root / "e2e_toy_curves.png")
        _log("wrote %s and %s" % (out_root / "e2e_toy.json", out_root / "e2e_toy.md"))
        for m, d in summary.items():
            _log("SUMMARY %-4s ADD %.2f±%.2f mm  rot %.2f±%.2f deg  trans %.2f±%.2f mm  "
                 "pen %.2f mm  feas %.0f%%  %.0f s"
                 % (m, d["add_mean"], d["add_std"], d["rot_mean"], d["rot_std"],
                    d["trans_mean"], d["trans_std"], d["pen_mean"], 100 * d["feas_mean"],
                    d["wall_s_mean"]))
    _log("total %.0f s" % (time.perf_counter() - t_all))
    return 0


if __name__ == "__main__":
    sys.exit(main())
