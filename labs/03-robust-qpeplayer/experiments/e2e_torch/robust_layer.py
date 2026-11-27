#!/usr/bin/env python3
"""Minimal real PyTorch ``autograd.Function`` for the 3D-3D robust pose layer.

Forward
-------
GNC-GM (graduated non-convexity, Geman-McClure kernel) with a weighted-Kabsch
inner solve, followed by Newton polish on the tangent space of S^3 x R^3 so
that the returned pose is a stationary point of the exact robust cost

    F(z; y) = sum_i rho(e_i(z; y)),   rho(e) = c^2 e / (c^2 + e),
    e_i = ||R(q) x_i + t - y_i||^2

to machine precision.  All arithmetic is float64.

Backward
--------
The exact implicit-function-theorem (IFT) derivative of the fixed point
``Phi(z; y) = 0``:

    dz*/dy_i = -H_rob^{-1} dPhi/dy_i,
    H_rob    = B^T [sum_i rho'(e_i) H_i + rho''(e_i) g_i g_i^T] B
               - lambda_rob diag(I_3, 0_3),
    lambda_rob = sum_i rho'(e_i) (q . g_{i,q}),
    dPhi/dy_i  = B^T [ rho'(e_i) dg_i/dy_i + rho''(e_i) g_i (de_i/dy_i)^T ].

These are exactly the formulas implemented in
``experiments/synthetic/common.py`` (``robust_hessian_exact``,
``dphi_dy_exact``, ``dphi_dy_naive``); this file is a float64 torch port.

Three backward modes are provided:
  * ``exact``    : exact IFT Hessian and exact dPhi/dy (both rho', rho'' terms);
  * ``naive``    : frozen-weight / IRLS-StopGrad (rho'' terms dropped in both
                   the Hessian and dPhi/dy), i.e. what you get by treating the
                   GM weights as constants;
  * ``stopgrad`` : no gradient through the layer at all (zero grad wrt y).

The layer backpropagates *only* to the observations ``y``; ``x`` and ``A``
are constants of the problem.

Public API
----------
``make_A(x)``                       -> (N,3,4,4) constant tensors
``forward_pose(y, x, A, w=1.0)``    -> (q, t, R) numpy-free torch tensors
``robust_pose(y, x, A, mode=...)``  -> autograd.Function output (R, t)
``pose_loss(R, t, Rgt, tgt, w=1.0)``
``backward_y(...)``                 -> analytic dL/dy (N,3), used by tests

Conventions match ``common.py``: q = (w,x,y,z) with ||q||=1, R = Rmat(q)
(unit-quaternion formula), A_k from the *homogeneous* quartic Rhom(q).
"""
from __future__ import annotations

import math

import numpy as np
import torch

DT = torch.float64

# Problem defaults (same as experiments/synthetic/common.py).
C_DEFAULT = 0.05
GAMMA = 1.4
MAX_MU_ITERS = 40


# ---------------------------------------------------------------------------
# Small geometry helpers
# ---------------------------------------------------------------------------
def _t(x):
    return torch.as_tensor(x, dtype=DT)


def Rhom(q):
    """Homogeneous (degree-2) quaternion rotation map; Rmat(q) for ||q||=1."""
    w, x, y, z = q
    return torch.stack([
        torch.stack([w * w + x * x - y * y - z * z, 2 * (x * y - w * z), 2 * (x * z + w * y)]),
        torch.stack([2 * (x * y + w * z), w * w - x * x + y * y - z * z, 2 * (y * z - w * x)]),
        torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), w * w - x * x - y * y + z * z]),
    ])


def Rmat(q):
    return Rhom(q)


def build_A(p):
    """Constant symmetric A_k with (Rmat(q) p)_k = q^T A_k q (torch port)."""
    p = torch.as_tensor(p, dtype=DT)
    E = torch.eye(4, dtype=DT)
    A = torch.zeros(3, 4, 4, dtype=DT)
    for k in range(3):
        diag = [(Rhom(E[i]) @ p)[k] for i in range(4)]
        for i in range(4):
            A[k, i, i] = diag[i]
        for i in range(4):
            for j in range(i + 1, 4):
                val = ((Rhom(E[i] + E[j]) @ p)[k] - diag[i] - diag[j]) / 2.0
                A[k, i, j] = val
                A[k, j, i] = val
    return A


def make_A(x):
    """(N,3,4,4) A-tensors for a batch of model points x (N,3)."""
    x = torch.as_tensor(x, dtype=DT)
    return torch.stack([build_A(x[i]) for i in range(x.shape[0])])


def quat_from_R(R):
    """Rotation matrix -> unit quaternion (w,x,y,z), w >= 0."""
    Rn = np.asarray(R.detach().cpu().numpy(), dtype=float)
    tr = np.trace(Rn)
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (Rn[2, 1] - Rn[1, 2]) / s
        y = (Rn[0, 2] - Rn[2, 0]) / s
        z = (Rn[1, 0] - Rn[0, 1]) / s
    elif Rn[0, 0] > Rn[1, 1] and Rn[0, 0] > Rn[2, 2]:
        s = math.sqrt(1.0 + Rn[0, 0] - Rn[1, 1] - Rn[2, 2]) * 2
        w = (Rn[2, 1] - Rn[1, 2]) / s
        x = 0.25 * s
        y = (Rn[0, 1] + Rn[1, 0]) / s
        z = (Rn[0, 2] + Rn[2, 0]) / s
    elif Rn[1, 1] > Rn[2, 2]:
        s = math.sqrt(1.0 + Rn[1, 1] - Rn[0, 0] - Rn[2, 2]) * 2
        w = (Rn[0, 2] - Rn[2, 0]) / s
        x = (Rn[0, 1] + Rn[1, 0]) / s
        y = 0.25 * s
        z = (Rn[1, 2] + Rn[2, 1]) / s
    else:
        s = math.sqrt(1.0 + Rn[2, 2] - Rn[0, 0] - Rn[1, 1]) * 2
        w = (Rn[1, 0] - Rn[0, 1]) / s
        x = (Rn[0, 2] + Rn[2, 0]) / s
        y = (Rn[1, 2] + Rn[2, 1]) / s
        z = 0.25 * s
    q = _t([w, x, y, z])
    q = q / torch.linalg.norm(q)
    return -q if q[0] < 0 else q


def tangent_basis(q):
    """4x3 orthonormal basis of T_q S^3 (deterministic SVD, as in common.py)."""
    q = torch.as_tensor(q, dtype=DT)
    vh = torch.linalg.svd(q.reshape(1, 4))[2]
    return vh[1:].T.contiguous()


def exp_map(q, t, xi, Bq):
    """Geodesic exponential on S^3 x R^3 (same retraction as common.py)."""
    xq = Bq @ xi[:3]
    nq = torch.linalg.norm(xq)
    if float(nq) < 1e-14:
        q2 = q
    else:
        q2 = torch.cos(nq) * q + torch.sin(nq) * xq / nq
    return q2 / torch.linalg.norm(q2), t + xi[3:]


# ---------------------------------------------------------------------------
# Weighted Kabsch / GNC-GM forward
# ---------------------------------------------------------------------------
def kabsch(x, y, w):
    """Weighted Kabsch (torch float64).  Returns R (3,3), t (3,)."""
    x = torch.as_tensor(x, dtype=DT)
    y = torch.as_tensor(y, dtype=DT)
    w = torch.as_tensor(w, dtype=DT)
    s = w.sum()
    if not torch.isfinite(s) or float(s) <= 1e-12:
        w = torch.ones_like(w)
        s = w.sum()
    w = w / s
    cx = (w[:, None] * x).sum(0)
    cy = (w[:, None] * y).sum(0)
    H = (w[:, None] * (x - cx)).T @ (y - cy)
    U, _, Vt = torch.linalg.svd(H)
    d = torch.sign(torch.linalg.det(Vt.T @ U.T))
    D = torch.diag(_t([1.0, 1.0, d]))
    R = Vt.T @ D @ U.T
    t = cy - R @ cx
    return R, t


def residuals_sq(x, y, R, t):
    return ((x @ R.T + t - y) ** 2).sum(1)


def gm_rho(e, c=C_DEFAULT):
    c2 = c * c
    c4 = c2 * c2
    rho = c2 * e / (c2 + e)
    rho1 = c4 / (c2 + e) ** 2
    rho2 = -2.0 * c4 / (c2 + e) ** 3
    return rho, rho1, rho2


def gnc_solve(x, y, c=C_DEFAULT, gamma=GAMMA, max_mu_iters=MAX_MU_ITERS):
    """GNC-GM with weighted-Kabsch inner solves (no gradient, forward only)."""
    n = x.shape[0]
    R, t = kabsch(x, y, torch.ones(n, dtype=DT))
    e = residuals_sq(x, y, R, t)
    c2 = c * c
    mu = max(3.0 * float(e.max()) / c2, 1.0)
    n_mu = 1
    for _ in range(max_mu_iters):
        w = (mu * c2 / (mu * c2 + e)) ** 2
        R, t = kabsch(x, y, w)
        e = residuals_sq(x, y, R, t)
        if mu <= 1.0 + 1e-12:
            break
        mu = max(1.0, mu / gamma)
        n_mu += 1
    return R, t, dict(n_mu=n_mu, e=e)


# ---------------------------------------------------------------------------
# Robust terms / Hessians (torch port of common.py §"Exact robust cost")
# ---------------------------------------------------------------------------
def robust_terms(x, y, q, t, A):
    """Per-observation e_i, g_i (7,), H_i (7,7), J_i (3,4), u_i, r_i."""
    u = torch.einsum("a,nkab,b->nk", q, A, q)          # (N,3) = R(q) x_i
    J = 2.0 * torch.einsum("nkab,b->nka", A, q)        # (N,3,4) = du_i/dq
    r = u + t[None, :] - y                             # (N,3)
    e = (r * r).sum(1)                                 # (N,)
    gq = 2.0 * torch.einsum("nk,nka->na", r, J)        # (N,4)
    gt = 2.0 * r                                       # (N,3)
    g = torch.cat([gq, gt], dim=1)                     # (N,7)
    Hqq = (2.0 * torch.einsum("nka,nkb->nab", J, J)
           + 4.0 * torch.einsum("nk,nkab->nab", r, A))  # (N,4,4)
    Hqt = 2.0 * J.transpose(1, 2)                       # (N,4,3)
    H = torch.zeros(x.shape[0], 7, 7, dtype=DT)
    H[:, :4, :4] = Hqq
    H[:, :4, 4:] = Hqt
    H[:, 4:, :4] = Hqt.transpose(1, 2)
    H[:, 4:, 4:] = 2.0 * torch.eye(3, dtype=DT)
    return dict(e=e, g=g, H=H, J=J, u=u, r=r)


def _Bfull(Bq):
    B = torch.zeros(7, 6, dtype=DT)
    B[:4, :3] = Bq
    B[4:, 3:] = torch.eye(3, dtype=DT)
    return B


def hessian_exact(terms, q, Bq, c=C_DEFAULT):
    _, rho1, rho2 = gm_rho(terms["e"], c)
    B = _Bfull(Bq)
    Htot = (torch.einsum("n,nab->ab", rho1, terms["H"])
            + torch.einsum("n,na,nb->ab", rho2, terms["g"], terms["g"]))
    gtot = torch.einsum("n,na->a", rho1, terms["g"])
    lam = torch.dot(q, gtot[:4])
    D6 = torch.diag(_t([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]))
    H = B.T @ Htot @ B - lam * D6
    return H, lam, gtot, B


def hessian_naive(terms, q, Bq, c=C_DEFAULT):
    _, rho1, _ = gm_rho(terms["e"], c)
    B = _Bfull(Bq)
    Htot = torch.einsum("n,nab->ab", rho1, terms["H"])
    lam = torch.einsum("n,n->", rho1, terms["g"][:, :4] @ q)
    D6 = torch.diag(_t([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]))
    H = B.T @ Htot @ B - lam * D6
    return H, lam


def newton_polish(x, y, q, t, A, c=C_DEFAULT, max_iters=60, tol=1e-15):
    """Newton on Phi(z)=0 in tangent coordinates; returns a true fixed point."""
    q = q / torch.linalg.norm(q)
    pn = float("inf")
    it = 0
    for it in range(max_iters):
        Bq = tangent_basis(q)
        terms = robust_terms(x, y, q, t, A)
        H, lam, gtot, B = hessian_exact(terms, q, Bq, c)
        phi = B.T @ gtot
        pn = float(torch.linalg.norm(phi))
        if pn < tol:
            break
        xi = torch.linalg.solve(H, -phi)
        q, t = exp_map(q, t, xi, Bq)
    return q, t, dict(pn=pn, n_newton=it + 1)


def solve_robust(x, y, A, c=C_DEFAULT, polish=True):
    """Full forward: GNC-GM then (optionally) Newton polish."""
    x = torch.as_tensor(x, dtype=DT)
    y = torch.as_tensor(y, dtype=DT)
    R, t, info = gnc_solve(x, y, c=c)
    q = quat_from_R(R)
    if polish:
        q, t, ninfo = newton_polish(x, y, q, t, A, c=c)
        info.update(ninfo)
    else:
        info.update(pn=float("nan"), n_newton=0)
    return q, t, info


def forward_pose(y, x, A, c=C_DEFAULT, polish=True):
    """Convenience wrapper: (q, t, R, info) with R = Rmat(q)."""
    q, t, info = solve_robust(x, y, A, c=c, polish=polish)
    return q, t, Rmat(q), info


# ---------------------------------------------------------------------------
# Exact IFT backward  (dL/dy given dL/dR, dL/dt)
# ---------------------------------------------------------------------------
def dR_dq(q):
    """(3,3,4) tensor with dR_dq[:,:,k] = dRmat/dq_k (matches common.dL_dz)."""
    w, x, y, z = q
    cols = [
        torch.stack([torch.stack([_t(0.0), -2 * z, 2 * y]),
                     torch.stack([2 * z, _t(0.0), -2 * x]),
                     torch.stack([-2 * y, 2 * x, _t(0.0)])]),
        torch.stack([torch.stack([_t(0.0), 2 * y, 2 * z]),
                     torch.stack([2 * y, -4 * x, -2 * w]),
                     torch.stack([2 * z, 2 * w, -4 * x])]),
        torch.stack([torch.stack([-4 * y, 2 * x, 2 * w]),
                     torch.stack([2 * x, _t(0.0), 2 * z]),
                     torch.stack([-2 * w, 2 * z, -4 * y])]),
        torch.stack([torch.stack([-4 * z, -2 * w, 2 * x]),
                     torch.stack([2 * w, -4 * z, 2 * y]),
                     torch.stack([2 * x, 2 * y, _t(0.0)])]),
    ]
    return torch.stack(cols, dim=2)


def backward_y(q, t, x, y, A, grad_R, grad_t, mode="exact", c=C_DEFAULT):
    """Analytic dL/dy (N,3) for the IFT stationarity map.

    ``mode='exact'`` uses the exact Hessian + exact dPhi/dy; ``mode='naive'``
    freezes rho'(e); ``mode='stopgrad'`` returns zeros.
    """
    if mode == "stopgrad":
        return torch.zeros_like(y)
    if grad_R is None:
        grad_R = torch.zeros(3, 3, dtype=DT)
    if grad_t is None:
        grad_t = torch.zeros(3, dtype=DT)
    Bq = tangent_basis(q)
    terms = robust_terms(x, y, q, t, A)
    if mode == "exact":
        H, lam, gtot, B = hessian_exact(terms, q, Bq, c)
    elif mode == "naive":
        H, lam = hessian_naive(terms, q, Bq, c)
        B = _Bfull(Bq)
    else:
        raise ValueError(mode)
    dLdq = torch.einsum("ab,abk->k", grad_R, dR_dq(q))
    dz = torch.cat([Bq.T @ dLdq, grad_t])
    w = torch.linalg.solve(H, dz)          # H^{-1} dL/dz
    BW = B @ w                             # (7,)
    rho1, rho2 = gm_rho(terms["e"], c)[1:]
    if mode == "naive":
        rho2 = torch.zeros_like(rho2)
    J = terms["J"]
    r = terms["r"]
    g = terms["g"]
    # dL/dy_i = -(dPhi/dy_i)^T H^{-1} dL/dz
    #         = 2 rho' (J_i w_q + w_t) + 2 rho'' (g_i . w) r_i
    grad_y = (2.0 * rho1[:, None] * (torch.einsum("nka,a->nk", J, BW[:4]) + BW[4:][None, :])
              + 2.0 * rho2[:, None] * r * (g @ BW)[:, None])
    return grad_y


# ---------------------------------------------------------------------------
# autograd.Function
# ---------------------------------------------------------------------------
class RobustPose(torch.autograd.Function):
    """y -> (R, t) where (R,t) solves the GM-robust registration problem.

    Backward modes: 'exact' (IFT), 'naive' (frozen weights), 'stopgrad'.
    """

    @staticmethod
    def forward(ctx, y, x, A, c, mode):
        q, t, info = solve_robust(x, y, A, c=c, polish=True)
        if info["pn"] > 1e-8:
            raise RuntimeError(f"Newton polish did not converge: pn={info['pn']:.2e}")
        ctx.save_for_backward(y)
        ctx.q, ctx.t, ctx.x, ctx.A = q, t, x, A
        ctx.c, ctx.mode = c, mode
        return Rmat(q), t

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_R, grad_t):
        (y,) = ctx.saved_tensors
        grad_y = backward_y(ctx.q, ctx.t, ctx.x, y, ctx.A, grad_R, grad_t,
                            mode=ctx.mode, c=ctx.c)
        return grad_y, None, None, None, None


def robust_pose(y, x, A, c=C_DEFAULT, mode="exact"):
    """Differentiable robust pose layer: y (N,3) -> (R (3,3), t (3,))."""
    return RobustPose.apply(torch.as_tensor(y, dtype=DT), torch.as_tensor(x, dtype=DT),
                            torch.as_tensor(A, dtype=DT), float(c), mode)


def pose_loss(R, t, Rgt, tgt, trans_weight=1.0):
    """||R - Rgt||_F^2 + w ||t - tgt||^2."""
    return ((R - Rgt) ** 2).sum() + trans_weight * ((t - tgt) ** 2).sum()


def rot_err_deg(R, Rgt):
    c = (torch.trace(R.T @ Rgt) - 1.0) / 2.0
    return float(torch.arccos(torch.clamp(c, -1.0, 1.0)).detach()) * 180.0 / math.pi
