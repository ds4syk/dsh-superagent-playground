"""Torch wrapper for a branch-aware differentiable QPEP layer.

Forward: collect a small set of certified branches by local solves from
deterministic + random starts; return the best branch (and the runner-up).
Backward: implicit KKT derivative of each branch; the certified mode returns
a branch-gap-weighted generalized gradient (Clarke/mixture selection).
"""
import numpy as np
import torch
from torch.autograd import Function

from .poly import qpep_residuals, J_from_res, peval, pgrad, q2R
from .local import local_solve
from .implicit import kkt_matrix

_IDENTITY = np.array([1.0, 0.0, 0.0, 0.0])
_QX = np.array([0.0, 1.0, 0.0, 0.0])
_QY = np.array([0.0, 0.0, 1.0, 0.0])
_QZ = np.array([0.0, 0.0, 0.0, 1.0])


def branch_starts(seed=0):
    rng = np.random.default_rng(seed)
    return [_IDENTITY, _QX, _QY, _QZ]


def _dedup(sols):
    out = []
    for f, q in sorted(sols, key=lambda z: z[0]):
        if all(min(np.linalg.norm(q - u), np.linalg.norm(q + u)) > 1e-4
               for _, u in out):
            out.append((f, q))
    return out


def solve_branches(J, starts, keep=2):
    sols = []
    for q0 in starts:
        q, f, _ = local_solve(J, q0, maxiter=250, ftol=1e-10)
        sols.append((f, q))
    sols = _dedup(sols)
    return sols[:keep]


def dJ_duv_fd(p, uv, q, h=1e-6):
    """Finite-difference derivative of J(q; uv) and grad_q J(q; uv) w.r.t. uv."""
    n = uv.shape[0]
    dJ = np.zeros((n, 2))
    dg = np.zeros((4, n, 2))
    for i in range(n):
        for j in range(2):
            up = uv.copy(); um = uv.copy()
            up[i, j] += h; um[i, j] -= h
            Jp = J_from_res(qpep_residuals(p, up))
            Jm = J_from_res(qpep_residuals(p, um))
            dJ[i, j] = (peval(Jp, q) - peval(Jm, q)) / (2 * h)
            gp = np.array([peval(gi, q) for gi in pgrad(Jp)])
            gm = np.array([peval(gi, q) for gi in pgrad(Jm)])
            dg[:, i, j] = (gp - gm) / (2 * h)
    return dJ, dg


def branch_jacobian(J, q, p, uv, h=1e-6):
    """Return dq/duv and the branch-cost derivative dJ/duv."""
    g = np.array([peval(gi, q) for gi in pgrad(J)])
    lam = float(q @ g)
    K, _, _ = kkt_matrix(J, q, lam)
    n = uv.shape[0]
    dq = np.zeros((4, n, 2))
    dJ, dg = dJ_duv_fd(p, uv, q, h)
    for i in range(n):
        for j in range(2):
            rhs = -np.concatenate([dg[:, i, j], [0.0]])
            try:
                sol = np.linalg.solve(K, rhs)
            except np.linalg.LinAlgError:
                sol = np.linalg.lstsq(K, rhs, rcond=None)[0]
            dq[:, i, j] = sol[:4]
    return dq, dJ


class QPEPBranchLayer(Function):
    @staticmethod
    def forward(ctx, uv, p, mode, temp):
        uv_np = uv.detach().cpu().numpy().astype(np.float64)
        p_np = p.detach().cpu().numpy().astype(np.float64)
        B, n, _ = uv_np.shape
        q1 = np.zeros((B, 4)); q2 = np.zeros((B, 4)); w = np.zeros((B, 2))
        Js = []; f1s = []; f2s = []
        starts = branch_starts(0)
        for b in range(B):
            J = J_from_res(qpep_residuals(p_np, uv_np[b]))
            if mode == 'local':
                q, f, _ = local_solve(J, _IDENTITY, maxiter=250, ftol=1e-10)
                sols = [(f, q)]
            else:
                sols = solve_branches(J, starts, keep=2)
            f1, qq1 = sols[0]
            if len(sols) > 1:
                f2, qq2 = sols[1]
            else:
                f2, qq2 = f1, qq1
            q1[b] = qq1; q2[b] = qq2
            if mode == 'certified':
                gap = max(float(f2 - f1), 0.0)
                z = np.array([0.0, -gap / max(temp, 1e-12)])
                z -= z.max()
                ww = np.exp(z); ww /= ww.sum()
            else:
                ww = np.array([1.0, 0.0])
            w[b] = ww
            Js.append(J); f1s.append(f1); f2s.append(f2)
        ctx.save_for_backward(uv, p)
        ctx.uv_np = uv_np; ctx.p_np = p_np
        ctx.q1 = q1; ctx.q2 = q2; ctx.Js = Js
        ctx.mode = mode; ctx.w = w; ctx.temp = float(temp)
        return (torch.tensor(q1, dtype=torch.float32),
                torch.tensor(q2, dtype=torch.float32),
                torch.tensor(w, dtype=torch.float32))

    @staticmethod
    def backward(ctx, grad_q1, grad_q2, grad_w):
        uv, p = ctx.saved_tensors
        g1 = grad_q1.detach().cpu().numpy().astype(np.float64)
        g2 = grad_q2.detach().cpu().numpy().astype(np.float64)
        B, n, _ = ctx.uv_np.shape
        grad_uv = np.zeros((B, n, 2))
        gw = None if grad_w is None else grad_w.detach().cpu().numpy().astype(np.float64)
        for b in range(B):
            J = ctx.Js[b]
            dq1, dJ1 = branch_jacobian(J, ctx.q1[b], ctx.p_np, ctx.uv_np[b])
            w1, w2 = ctx.w[b, 0], ctx.w[b, 1]
            for i in range(n):
                for j in range(2):
                    val = w1 * (g1[b] @ dq1[:, i, j])
                    if ctx.mode == 'certified' and w2 > 1e-8:
                        dq2, dJ2 = branch_jacobian(J, ctx.q2[b], ctx.p_np, ctx.uv_np[b])
                        val += w2 * (g2[b] @ dq2[:, i, j])
                        # selection term: (L1-L2) * dw1/duv
                        if gw is not None:
                            coef = (gw[b, 0] - gw[b, 1]) * w1 * w2 / max(ctx.temp, 1e-12)
                            val += coef * (dJ2[i, j] - dJ1[i, j])
                    grad_uv[b, i, j] = val
        return torch.tensor(grad_uv, dtype=torch.float32), None, None, None


def q2R_torch(q):
    q0, q1, q2, q3 = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = torch.stack([
        torch.stack([q0*q0+q1*q1-q2*q2-q3*q3, 2*(q0*q3+q1*q2), 2*(q1*q3-q0*q2)], dim=-1),
        torch.stack([2*(q1*q2-q0*q3), q0*q0-q1*q1+q2*q2-q3*q3, 2*(q0*q1+q2*q3)], dim=-1),
        torch.stack([2*(q0*q2+q1*q3), 2*(q2*q3-q0*q1), q0*q0-q1*q1-q2*q2+q3*q3], dim=-1),
    ], dim=-2)
    return R


def pose_loss(q, q_true):
    R = q2R_torch(q)
    Rt = q2R_torch(q_true)
    return ((R - Rt) ** 2).sum(dim=(-2, -1)).mean()
