#!/usr/bin/env python3
"""Torch (CPU/CUDA) mirror of ``batch_gnc.batch_solve_gnc``.

Same batched GNC-GM schedule as ``common.solve_gnc`` (mu0 = max(3 max e /
c^2, 1), gamma = 1.4, <= 40 mu iterations, 5 exact-kernel polish steps) and
the same math as ``batch_gnc``: (B,3,N) internal layout, weighted Kabsch via
Horn/Davenport 4x4 eigendecomposition (or batched 3x3 SVD), residual via
batched matmul.

The only reason this file exists is the optional GPU measurement required by
the P3 validation task (``srun --partition=gpu --gres=gpu:1``).  Default
dtype is float64 so that the GPU output can be compared with the CPU loop at
the requested 1e-8 / 1e-10 tolerance; fp32 timing is available via
``--dtype float32`` for throughput-only comparison.
"""
from __future__ import annotations

import torch

C = 0.05
GNC_GAMMA = 1.4
GNC_MAX_MU_ITERS = 40
GNC_POLISH_STEPS = 5


def _kabsch_T(xT, yT, w, method="eigh"):
    """Weighted Kabsch on (B,3,N) torch tensors, weights (B,N)."""
    B = xT.shape[0]
    s = w.sum(dim=1)
    # No host sync here: the fallback is applied unconditionally with
    # torch.where.  A ``bool(bad.any())`` guard would force a GPU->CPU sync
    # on every GNC iteration (~40 syncs per solve), which dominates the
    # GPU latency for small batches.
    bad = (~torch.isfinite(s)) | (s <= 1e-12)
    w = torch.where(bad[:, None], torch.ones_like(w), w)
    s = torch.where(bad, w.sum(dim=1), s)
    wn = w / s[:, None]

    cx = torch.matmul(xT, wn[:, :, None])[:, :, 0]
    cy = torch.matmul(yT, wn[:, :, None])[:, :, 0]
    xc = xT - cx[:, :, None]
    yc = yT - cy[:, :, None]
    H = torch.matmul(xc * wn[:, None, :], yc.transpose(1, 2))

    if method == "svd":
        U, _, Vt = torch.linalg.svd(H)
        Ruv = torch.matmul(Vt.transpose(1, 2), U.transpose(1, 2))
        d = torch.sign(torch.linalg.det(Ruv))
        D = torch.zeros_like(H)
        D[:, 0, 0] = 1.0
        D[:, 1, 1] = 1.0
        D[:, 2, 2] = d
        R = torch.matmul(torch.matmul(Vt.transpose(1, 2), D), U.transpose(1, 2))
    elif method == "eigh":
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
        _, V = torch.linalg.eigh(K)
        q = V[:, :, -1]
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
    else:
        raise ValueError(f"unknown method {method!r}")

    t = cy - torch.matmul(R, cx[:, :, None])[:, :, 0]
    return R, t


def _residuals_sq_T(xT, yT, R, t):
    r = torch.matmul(R, xT) + t[:, :, None] - yT
    return torch.einsum("bin,bin->bn", r, r)


def solve_gnc_t(xT, yT, c=C, gamma=GNC_GAMMA,
                max_mu_iters=GNC_MAX_MU_ITERS,
                n_polish=GNC_POLISH_STEPS, method="eigh"):
    """Batched GNC-GM on torch tensors already laid out as (B,3,N) float tensors."""
    B, _, N = xT.shape
    c2 = float(c) * float(c)
    R, t = _kabsch_T(xT, yT, torch.ones((B, N), dtype=xT.dtype, device=xT.device),
                     method=method)
    e = _residuals_sq_T(xT, yT, R, t)

    mu = torch.clamp(3.0 * e.max(dim=1).values / c2, min=1.0)
    mu0 = mu.clone()
    n_mu = torch.zeros(B, dtype=torch.long, device=xT.device)

    active = torch.arange(B, device=xT.device)
    for _ in range(int(max_mu_iters)):
        if active.numel() == 0:
            break
        a = mu[active][:, None] * c2
        w = (a / (a + e[active])) ** 2
        R[active], t[active] = _kabsch_T(xT[active], yT[active], w, method=method)
        e[active] = _residuals_sq_T(xT[active], yT[active], R[active], t[active])
        n_mu[active] += 1
        done = mu[active] <= 1.0 + 1e-12
        active = active[~done]
        mu[active] = torch.clamp(mu[active] / gamma, min=1.0)

    for _ in range(int(n_polish)):
        w = (c2 / (c2 + e)) ** 2
        R, t = _kabsch_T(xT, yT, w, method=method)
        e = _residuals_sq_T(xT, yT, R, t)

    return R, t, dict(mu0=mu0, n_mu=n_mu)
