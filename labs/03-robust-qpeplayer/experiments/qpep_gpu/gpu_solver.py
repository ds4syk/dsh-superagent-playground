#!/usr/bin/env python3
"""GPU batched GNC-GM / Horn 4x4 for scalar-weight 3D-3D: eager, static-mask,
CUDA-graph and analytic-Jacobi variants.

The reference algorithm is exactly ``experiments/systems/batch_gnc_torch.py``
(which in turn mirrors ``experiments/synthetic/common.py::solve_gnc``), so the
only things that differ here are (i) the execution strategy and (ii) the
batched symmetric 4x4 eigen inner.

Two solvers are exposed:

* ``solve_gnc_dynamic`` -- the existing eager active-set loop (imported from
  ``systems/batch_gnc_torch.py``).  It drops finished instances from the batch
  and therefore has data-dependent tensor shapes, which cannot be captured in
  a CUDA graph.
* ``solve_gnc_static`` -- a *static-shape* rewrite of the same schedule: every
  instance is carried through all ``max_mu_iters + n_polish`` iterations, and
  finished instances are frozen with ``torch.where(active, new, old)``.  Each
  instance sees exactly the same arithmetic and the same number of GNC
  iterations as in the dynamic loop (finished instances are bitwise copied),
  but the shapes are fixed, so the whole solve can be captured in a CUDA graph
  or compiled.  The price is that finished instances are still recomputed
  (FLOPs are not saved), which is the honest cost of graph capture.

The 4x4 symmetric eigenproblem (Horn/Davenport) can be solved by

* ``eigen="eigh"``  -- ``torch.linalg.eigh`` (cuSOLVER batched; LAPACK path),
* ``eigen="jacobi"`` -- a fixed 6-sweep batched cyclic-Jacobi eigensolver
  written entirely in tensor ops (no cuSOLVER, no host sync, capture-safe).

For scalar weights the QPEP stationary-point problem of 3D-3D is exactly this
4x4 eigenproblem; see ``qpep_scope.md``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_SYSTEMS = _HERE.parent / "systems"
if str(_SYSTEMS) not in sys.path:
    sys.path.insert(0, str(_SYSTEMS))

from batch_gnc_torch import (  # noqa: E402
    C,
    GNC_GAMMA,
    GNC_MAX_MU_ITERS,
    GNC_POLISH_STEPS,
    _residuals_sq_T,
    solve_gnc_t,
)

JACOBI_SWEEPS = 6
# Cyclic Jacobi pair order (p, q), p < q.
_JACOBI_PAIRS = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))


# ---------------------------------------------------------------------------
# Batched symmetric 4x4 eigen: Horn matrix construction (shared)
# ---------------------------------------------------------------------------
def horn_K(H):
    """Horn/Davenport 4x4 matrix from H = sum_i w_i (x-cx)(y-cy)^T, (B,3,3)."""
    B = H.shape[0]
    K = torch.empty((B, 4, 4), dtype=H.dtype, device=H.device)
    H00, H01, H02 = H[:, 0, 0], H[:, 0, 1], H[:, 0, 2]
    H10, H11, H12 = H[:, 1, 0], H[:, 1, 1], H[:, 1, 2]
    H20, H21, H22 = H[:, 2, 0], H[:, 2, 1], H[:, 2, 2]
    K[:, 0, 0] = H00 + H11 + H22
    K[:, 1, 1] = H00 - H11 - H22
    K[:, 2, 2] = -H00 + H11 - H22
    K[:, 3, 3] = -H00 - H11 + H22
    K[:, 0, 1] = K[:, 1, 0] = H12 - H21
    K[:, 0, 2] = K[:, 2, 0] = H20 - H02
    K[:, 0, 3] = K[:, 3, 0] = H01 - H10
    K[:, 1, 2] = K[:, 2, 1] = H01 + H10
    K[:, 1, 3] = K[:, 3, 1] = H20 + H02
    K[:, 2, 3] = K[:, 3, 2] = H12 + H21
    return K


def eigh4_jacobi(K, sweeps=JACOBI_SWEEPS):
    """Batched 4x4 symmetric eigensolver, fixed-sweep cyclic Jacobi.

    Pure tensor ops; no cuSOLVER, no host sync, capture-safe.  Returns
    eigenvalues ``(B,4)`` and eigenvectors ``(B,4,4)`` (columns) sorted by
    *descending* eigenvalue (so column 0 is Horn's optimal quaternion).
    """
    B = K.shape[0]
    A = K.clone()
    V = torch.eye(4, dtype=K.dtype, device=K.device).expand(B, 4, 4).clone()
    for _ in range(int(sweeps)):
        for p, q in _JACOBI_PAIRS:
            apq = A[:, p, q]
            app = A[:, p, p]
            aqq = A[:, q, q]
            # Givens rotation that zeroes A[p,q]; guard the 0/0 case with
            # atan2's branch (theta = pi/4 when app == aqq and apq == 0).
            theta = 0.5 * torch.atan2(2.0 * apq, app - aqq)
            c = torch.cos(theta)
            s = torch.sin(theta)
            G = torch.eye(4, dtype=K.dtype, device=K.device).expand(B, 4, 4).clone()
            G[:, p, p] = c
            G[:, q, q] = c
            G[:, p, q] = s
            G[:, q, p] = -s
            A = G.transpose(1, 2) @ A @ G
            V = V @ G
    # Sort eigenvalues descending (4 values: a full sort is cheap and keeps
    # the graph static).
    vals, order = torch.sort(torch.diagonal(A, dim1=1, dim2=2), dim=1, descending=True)
    idx = order[:, 0]
    vecs = V.gather(2, idx.view(B, 1, 1).expand(B, 4, 1))[:, :, 0]
    return vals, vecs


# ---------------------------------------------------------------------------
# Weighted Kabsch with a selectable 4x4 eigen backend
# ---------------------------------------------------------------------------
def kabsch_T(xT, yT, w, eigen="eigh"):
    """Weighted Kabsch on (B,3,N) with weights (B,N); same math as systems."""
    B = xT.shape[0]
    s = w.sum(dim=1)
    bad = (~torch.isfinite(s)) | (s <= 1e-12)
    w = torch.where(bad[:, None], torch.ones_like(w), w)
    s = torch.where(bad, w.sum(dim=1), s)
    wn = w / s[:, None]

    cx = torch.matmul(xT, wn[:, :, None])[:, :, 0]
    cy = torch.matmul(yT, wn[:, :, None])[:, :, 0]
    xc = xT - cx[:, :, None]
    yc = yT - cy[:, :, None]
    H = torch.matmul(xc * wn[:, None, :], yc.transpose(1, 2))

    K = horn_K(H)
    if eigen == "eigh":
        _, V = torch.linalg.eigh(K)
        q = V[:, :, -1]
    elif eigen == "jacobi":
        _, q = eigh4_jacobi(K)
    else:
        raise ValueError(f"unknown eigen backend {eigen!r}")
    q = q / torch.linalg.norm(q, dim=1, keepdim=True)
    q0, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = torch.empty((B, 3, 3), dtype=H.dtype, device=H.device)
    R[:, 0, 0] = q0 * q0 + qx * qx - qy * qy - qz * qz
    R[:, 1, 1] = q0 * q0 - qx * qx + qy * qy - qz * qz
    R[:, 2, 2] = q0 * q0 - qx * qx - qy * qy + qz * qz
    R[:, 0, 1] = 2.0 * (qx * qy - q0 * qz)
    R[:, 1, 0] = 2.0 * (qx * qy + q0 * qz)
    R[:, 0, 2] = 2.0 * (qx * qz + q0 * qy)
    R[:, 2, 0] = 2.0 * (qx * qz - q0 * qy)
    R[:, 1, 2] = 2.0 * (qy * qz - q0 * qx)
    R[:, 2, 1] = 2.0 * (qy * qz + q0 * qx)
    t = cy - torch.matmul(R, cx[:, :, None])[:, :, 0]
    return R, t


# ---------------------------------------------------------------------------
# Static-shape GNC-GM (graph/compile friendly)
# ---------------------------------------------------------------------------
def solve_gnc_static(xT, yT, c=C, gamma=GNC_GAMMA,
                     max_mu_iters=GNC_MAX_MU_ITERS,
                     n_polish=GNC_POLISH_STEPS, eigen="eigh"):
    """Static-shape rewrite of ``solve_gnc_t``.

    Every instance is advanced through ``max_mu_iters`` masked mu iterations
    plus ``n_polish`` exact-kernel steps; once an instance's mu reaches 1 it
    runs the mu=1 iteration and is then frozen (bitwise carried), exactly as
    in the dynamic loop.  ``n_mu`` equals the dynamic loop's count.
    """
    B, _, N = xT.shape
    c2 = float(c) * float(c)
    w0 = torch.ones((B, N), dtype=xT.dtype, device=xT.device)
    R, t = kabsch_T(xT, yT, w0, eigen=eigen)
    e = _residuals_sq_T(xT, yT, R, t)

    mu = torch.clamp(3.0 * e.max(dim=1).values / c2, min=1.0)
    mu0 = mu.clone()
    n_mu = torch.zeros(B, dtype=torch.long, device=xT.device)
    active = torch.ones(B, dtype=torch.bool, device=xT.device)

    for _ in range(int(max_mu_iters)):
        a = mu[:, None] * c2
        w = (a / (a + e)) ** 2
        R_new, t_new = kabsch_T(xT, yT, w, eigen=eigen)
        e_new = _residuals_sq_T(xT, yT, R_new, t_new)
        R = torch.where(active[:, None, None], R_new, R)
        t = torch.where(active[:, None], t_new, t)
        e = torch.where(active[:, None], e_new, e)
        n_mu = n_mu + active.to(torch.long)
        done = active & (mu <= 1.0 + 1e-12)
        active = active & (~done)
        mu = torch.clamp(mu / gamma, min=1.0)

    for _ in range(int(n_polish)):
        w = (c2 / (c2 + e)) ** 2
        R, t = kabsch_T(xT, yT, w, eigen=eigen)
        e = _residuals_sq_T(xT, yT, R, t)

    return R, t, dict(mu0=mu0, n_mu=n_mu)


# ---------------------------------------------------------------------------
# CUDA graph wrapper
# ---------------------------------------------------------------------------
class CudaGraphSolver:
    """Capture ``solver(x, y)`` for fixed shapes; replay on new inputs.

    ``x``/``y`` are static device buffers copied into before each replay, so
    the graph has no Python or allocator work in the timed region.
    """

    def __init__(self, solver, xT, yT, warmup=3):
        assert xT.is_cuda and yT.is_cuda
        self.solver = solver
        self.x = xT.clone()
        self.y = yT.clone()
        # Torch requires capture to happen on a non-default stream; warm up on
        # a side stream first so cuSOLVER/cuBLAS workspaces are allocated.
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(int(warmup)):
                self._out = solver(self.x, self.y)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self._out = solver(self.x, self.y)
        torch.cuda.synchronize()

    def replay(self, xT, yT):
        self.x.copy_(xT, non_blocking=True)
        self.y.copy_(yT, non_blocking=True)
        self.graph.replay()
        return self._out


# ---------------------------------------------------------------------------
# Correctness helpers
# ---------------------------------------------------------------------------
def rot_err(Ra, Rb):
    """Per-instance Frobenius norm of Ra - Rb, numpy or torch tensors."""
    if isinstance(Ra, torch.Tensor):
        d = Ra - Rb
        return torch.linalg.norm(d.reshape(d.shape[0], -1), dim=1)
    import numpy as np
    d = Ra - Rb
    return np.linalg.norm(d.reshape(d.shape[0], -1), axis=1)


__all__ = [
    "C", "GNC_GAMMA", "GNC_MAX_MU_ITERS", "GNC_POLISH_STEPS",
    "solve_gnc_t", "solve_gnc_static", "kabsch_T", "horn_K",
    "eigh4_jacobi", "CudaGraphSolver", "rot_err",
]
