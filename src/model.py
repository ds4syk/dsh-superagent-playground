"""Torch model pieces: measurement head, direct head and the SDPR bridge."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from rotsync import RotSyncProblem, SDPRLayer


# ---------------------------------------------------------------------------
# SO(3) helpers (differentiable)
# ---------------------------------------------------------------------------


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """Zhou et al. 6D representation -> rotation matrices (columns)."""
    a1, a2 = d6[..., :3], d6[..., 3:6]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def project_so3_torch(R: torch.Tensor) -> torch.Tensor:
    U, _, Vt = torch.linalg.svd(R)
    D = torch.ones(R.shape[:-2] + (3,), dtype=R.dtype, device=R.device)
    D[..., 2] = torch.sign(torch.det(U @ Vt))
    return (U * D.unsqueeze(-2)) @ Vt


def geodesic_loss(Rp: torch.Tensor, Rg: torch.Tensor) -> torch.Tensor:
    tr = (Rp * Rg).sum(dim=(-2, -1))  # tr(Rp^T Rg)
    cos = ((tr - 1.0) / 2.0).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    return torch.acos(cos).mean()


# ---------------------------------------------------------------------------
# heads
# ---------------------------------------------------------------------------


class EdgeHead(nn.Module):
    """Predict a relative rotation and a weight for every graph edge."""

    def __init__(self, d: int = 1024, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3 * d, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 7),
        )

    def forward(self, feats: torch.Tensor, edges: list[tuple[int, int]]):
        B = feats.shape[0]
        pairs = torch.stack(
            [torch.cat([feats[:, i], feats[:, j], feats[:, j] - feats[:, i]], dim=-1) for i, j in edges],
            dim=1,
        )  # [B,E,3D]
        out = self.net(pairs)
        meas = rotation_6d_to_matrix(out[..., :6])
        weight = F.softplus(out[..., 6]) + 0.01
        return meas, weight


class NodeHead(nn.Module):
    """Direct per-node rotation regression (no structured layer)."""

    def __init__(self, d: int = 1024, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 6),
        )

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return rotation_6d_to_matrix(self.net(feats))


# ---------------------------------------------------------------------------
# cost matrix from measurements (differentiable)
# ---------------------------------------------------------------------------


def build_Q_torch(
    prob: RotSyncProblem,
    meas: torch.Tensor,
    weights: torch.Tensor,
    edges: list[tuple[int, int]],
    prior_weight: float = 10.0,
) -> torch.Tensor:
    """``meas[e] = R_ij``, ``weights[e] = w_ij`` -> Q, differentiably."""
    n = prob.n
    Q = torch.zeros((n, n), dtype=meas.dtype, device=meas.device)
    eye3 = torch.eye(3, dtype=meas.dtype, device=meas.device)
    eye9 = torch.eye(9, dtype=meas.dtype, device=meas.device)
    for e, (i, j) in enumerate(edges):
        w = weights[e]
        si, sj = prob.node_slice(i), prob.node_slice(j)
        Q = Q + _scatter(si, si, 2.0 * w * eye9, n)
        Q = Q + _scatter(sj, sj, 2.0 * w * eye9, n)
        cross = -w * torch.kron(eye3, meas[e])
        Q = Q + _scatter(si, sj, cross, n)
        Q = Q + _scatter(sj, si, cross.transpose(-1, -2), n)
    for k in range(3):
        idx = 1 + 3 * k + k
        Q = Q + _scatter(
            0, idx, torch.as_tensor(-prior_weight, dtype=meas.dtype, device=meas.device).reshape(1, 1), n
        )
        Q = Q + _scatter(
            idx, 0, torch.as_tensor(-prior_weight, dtype=meas.dtype, device=meas.device).reshape(1, 1), n
        )
    return Q


def _scatter(i: int | slice, j: int | slice, block: torch.Tensor, n: int) -> torch.Tensor:
    M = torch.zeros((n, n), dtype=block.dtype, device=block.device)
    target = M[i, j]
    if target.shape != block.shape:
        block = block.reshape(target.shape)
    M[i, j] = block
    return M


def chordal_sync(
    meas: torch.Tensor,
    weights: torch.Tensor,
    edges: list[tuple[int, int]],
    n_nodes: int,
) -> torch.Tensor:
    """Spectral (chordal) rotation synchronization baseline.

    Builds the 3N x 3N block Laplacian and projects the top-3 eigenvectors'
    blocks onto SO(3).  Gauge is free; callers should align node 0.
    """
    B = meas.shape[0]
    out = []
    eye = torch.eye(3, dtype=meas.dtype, device=meas.device)
    for b in range(B):
        L = torch.zeros((3 * n_nodes, 3 * n_nodes), dtype=meas.dtype, device=meas.device)
        for e, (i, j) in enumerate(edges):
            w = weights[b, e]
            Rm = meas[b, e]
            si, sj = slice(3 * i, 3 * i + 3), slice(3 * j, 3 * j + 3)
            L[si, sj] = L[si, sj] - w * Rm
            L[sj, si] = L[sj, si] - w * Rm.transpose(-1, -2)
            L[si, si] = L[si, si] + w * eye
            L[sj, sj] = L[sj, sj] + w * eye
        _, V = torch.linalg.eigh(L)
        V = V[:, :3]  # null space (smallest eigenvalues): 3N x 3
        # The eigenbasis is V = [R_1 G; ...; R_N G] for an unknown invertible G.
        # Fix the gauge with node 0 (R_0 = I): G = V_0, so R_i = V_i G^{-1}.
        G = V[:3, :]
        Ginv = torch.linalg.inv(G)
        Rs = torch.stack(
            [project_so3_torch(V[3 * i : 3 * i + 3, :] @ Ginv) for i in range(n_nodes)]
        )
        out.append(Rs)
    return torch.stack(out)


def solve_rotations(
    prob: RotSyncProblem,
    Qs: torch.Tensor,
    runner,
) -> tuple[torch.Tensor, dict]:
    """Run the SDPR layer per batch element and return rotations + diagnostics."""
    squeeze = Qs.dim() == 2
    if squeeze:
        Qs = Qs.unsqueeze(0)
    xs = []
    diags = {"rank_gap": [], "eig_min": [], "hx_norm": []}
    for b in range(Qs.shape[0]):
        x, d = SDPRLayer(Qs[b], prob, runner)
        xs.append(x)
        for k in diags:
            diags[k].append(float(d[k]))
    x = torch.stack(xs, dim=0)  # [B,n]
    Rs = []
    for i in range(prob.N):
        c = x[:, 1 + 9 * i : 1 + 9 * (i + 1)]
        Rs.append(project_so3_torch(c.reshape(-1, 3, 3).transpose(-1, -2)))
    Rs = torch.stack(Rs, dim=1)
    return (Rs.squeeze(0) if squeeze else Rs), {k: np.asarray(v) for k, v in diags.items()}
