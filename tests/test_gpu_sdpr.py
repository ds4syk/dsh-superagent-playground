"""Tests for the GPU-parallel SDPR layer.

Run with:  python -m pytest tests/test_gpu_sdpr.py -q
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "repro", "cleanroom"))

from gpu_sdpr import SDPRLayer, build_p6_sdp  # noqa: E402
from gpu_sdpr.p6 import build_objective_matrix, objective_scale, scale_objective  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _make_pose_batch(n=4, seed=0):
    import data_gen
    from camera_model import StereoCamera

    cam = StereoCamera(fu=484.5, fv=484.5, cu=0.0, cv=0.0, sigma_u=0.5, sigma_v=0.5)
    rng = np.random.default_rng(seed)
    data = data_gen.get_calibration_data(
        n_batch=n, board_dims=(1.0, 1.0), n_squares=(8, 8), radius=3.0,
        beta=np.pi / 4, alpha=np.pi / 4, cam=cam, b_true=0.24, rng=rng,
        noise_mode="independent")
    Qs = []
    for k in range(n):
        meas, W = cam.inverse(data["pixels"][k], 0.243)
        Q = build_objective_matrix(data["r_l"], meas, W)
        Qs.append(scale_objective(Q, objective_scale(Q)))
    return data, np.stack(Qs)


def test_forward_matches_cpu_scs():
    import sdp_pose
    data, Q = _make_pose_batch(4)
    sdp = build_p6_sdp()
    layer = SDPRLayer(sdp["A"], sdp["b"], sdp["c0"], device="cuda")
    X, x = layer(torch.tensor(Q, device="cuda"))
    sv = sdp_pose.SDPPoseSolver(eps=1e-9)
    xr = np.stack([sv.solve(Q[k])["x"] for k in range(len(Q))])
    err = np.linalg.norm(x.cpu().numpy() - xr, axis=1) / np.linalg.norm(xr, axis=1)
    assert err.max() < 5e-4, err
    cert = layer.certificate()
    assert float(cert["H_min_eig"].min()) > -1e-7
    assert cert["corank1"] == len(Q)
    assert float(cert["rank_ratio"].min()) > 1e4


def test_gradient_direction_matches_cpu():
    import implicit_diff as idiff
    import sdp_pose
    data, Q = _make_pose_batch(4)
    sdp = build_p6_sdp()
    layer = SDPRLayer(sdp["A"], sdp["b"], sdp["c0"], device="cuda")
    Qd = torch.tensor(Q, device="cuda", requires_grad=True)
    X, x = layer(Qd)
    rng = np.random.default_rng(1)
    dQ = rng.normal(size=Q.shape)
    dQ = 0.5 * (dQ + dQ.transpose(0, 2, 1))
    grad_x = rng.normal(size=x.shape) * 1e-2
    (x * torch.tensor(grad_x, device=x.device)).sum().backward()
    gpu = float((Qd.grad.detach().cpu().numpy() * dQ).sum())

    sv = sdp_pose.SDPPoseSolver(eps=1e-9)
    ref = 0.0
    for k in range(len(Q)):
        rec = sv.solve(Q[k])
        G_r, _ = sv.reduced_gradient_rows(rec["x"])
        ref += idiff.qcqp_backward(rec["H"], G_r, rec["x"], dQ[k], grad_x[k])["grad_theta"]
    assert abs(gpu - ref) / max(abs(ref), 1e-12) < 1e-2, (gpu, ref)


def test_autograd_matches_solver_gradient():
    data, Q = _make_pose_batch(3)
    sdp = build_p6_sdp()
    layer = SDPRLayer(sdp["A"], sdp["b"], sdp["c0"], device="cuda")
    Qd = torch.tensor(Q, device="cuda", requires_grad=True)
    X, x = layer(Qd)
    g = torch.randn_like(x) * 1e-3
    (x * g).sum().backward()
    ref, _ = layer.solver.gradient(x.detach(), layer.last_out["H"].detach(), g)
    err = (Qd.grad.detach() - ref).abs().max()
    assert float(err) < 1e-9, float(err)
