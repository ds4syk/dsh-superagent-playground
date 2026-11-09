"""Verify the GPU SDPR layer against the CPU SCS/diffcp reference on (P6).

For each seed this script builds the 20-pose stereo calibration batch of
Holmes et al. (T-RO 2025, Sec. VI-B4), solves the Shor SDP with

* the reference CPU path: CVXPY + SCS (eps=1e-9) and implicit differentiation
  from ``repro/cleanroom``;
* the GPU path: batched dual log-det barrier (``src/gpu_sdpr``), with the
  same Theorem-2 KKT backward on the GPU,

and records solution error, gradient error, certificate quality and timing.

Usage
-----
    python repro/gpu_sdpr/verify_p6.py --seeds 10 --batch 20 \
        --out repro/gpu_sdpr/results/verify_p6.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "repro", "cleanroom"))

import bilevel  # noqa: E402
import implicit_diff as idiff  # noqa: E402
import sdp_pose  # noqa: E402
from gpu_sdpr import BarrierSolver, build_p6_sdp  # noqa: E402


def _dQ_db(cam, pixels, m, b, hc=1e-20):
    mc, Wc = cam.inverse(pixels, b + 1j * hc)
    Qc = sdp_pose.build_objective_matrix(m, mc, Wc)
    dQ = np.imag(Qc) / hc
    return 0.5 * (dQ + dQ.T)


def run_seed(seed: int, n_batch: int, device: str) -> dict:
    cfg = bilevel.ExpConfig(n_batch=n_batch, seed0=seed)
    data = bilevel.make_dataset(cfg, 0)
    cam = bilevel.make_camera(cfg)
    b = cfg.b_true + cfg.b_offset
    m = data["r_l"]

    Qs, dQs, scales = [], [], []
    for k in range(n_batch):
        meas, W = cam.inverse(data["pixels"][k], b)
        Q = sdp_pose.build_objective_matrix(m, meas, W)
        sc = sdp_pose.objective_scale(Q)
        scales.append(sc)
        Qs.append(sdp_pose.scale_objective(Q, sc))
        dQ = sc * _dQ_db(cam, data["pixels"][k], m, b)
        dQ[0, 0] = 0.0
        dQs.append(dQ)
    Q = np.stack(Qs)
    dQ = np.stack(dQs)

    # ---- CPU reference -------------------------------------------------
    sv = sdp_pose.SDPPoseSolver(eps=1e-9)
    t0 = time.time()
    rec = [sv.solve(Q[k]) for k in range(n_batch)]
    t_cpu_solve = time.time() - t0
    x_ref = np.stack([r["x"] for r in rec])
    H_ref = np.stack([r["H"] for r in rec])

    grad_x = np.zeros_like(x_ref)
    for k in range(n_batch):
        _, grad_x[k] = idiff.loss_from_x_and_grad(
            x_ref[k], data["C_p0s"][k], data["r_p0s"][k])
    t0 = time.time()
    grad_ref = np.zeros(n_batch)
    for k in range(n_batch):
        G_r, _ = sv.reduced_gradient_rows(x_ref[k])
        grad_ref[k] = idiff.qcqp_backward(H_ref[k], G_r, x_ref[k], dQ[k], grad_x[k])["grad_theta"]
    t_cpu_bwd = time.time() - t0

    # ---- GPU -----------------------------------------------------------
    sdp = build_p6_sdp()
    solver = BarrierSolver(sdp["A"], sdp["b"], sdp["c0"], device=device,
                           mu_final=1e-8)
    Qd = torch.tensor(Q, device=device)
    solver.solve(Qd[: min(4, n_batch)])  # warm-up / JIT
    torch.cuda.synchronize()
    t0 = time.time()
    out = solver.solve(Qd)
    torch.cuda.synchronize()
    t_gpu_fwd = time.time() - t0
    t0 = time.time()
    grad_Q, _ = solver.gradient(out["x"], out["H"], torch.tensor(grad_x, device=device))
    torch.cuda.synchronize()
    t_gpu_bwd = time.time() - t0
    grad_gpu = torch.einsum("bij,bij->b", grad_Q, Qd.new_tensor(dQ)).detach().cpu().numpy()

    x_gpu = out["x"].detach().cpu().numpy()
    H_gpu = out["H"].detach().cpu().numpy()
    x_err = np.linalg.norm(x_gpu - x_ref, axis=1) / np.linalg.norm(x_ref, axis=1)
    grad_abs = np.abs(grad_gpu - grad_ref)
    grad_err = grad_abs / np.maximum(np.abs(grad_ref), 1e-300)
    grad_sum_err = abs(float(grad_gpu.sum() - grad_ref.sum())) / max(abs(float(grad_ref.sum())), 1e-300)
    wH = np.linalg.eigvalsh(H_gpu)
    wX = np.linalg.eigvalsh(out["X"].detach().cpu().numpy())
    rank_ratio = wX[:, -1] / np.maximum(wX[:, -2], 1e-300)

    return dict(
        seed=seed,
        x_err_mean=float(x_err.mean()), x_err_max=float(x_err.max()),
        grad_err_mean=float(grad_err.mean()), grad_err_max=float(grad_err.max()),
        grad_abs_max=float(grad_abs.max()),
        grad_sum_err=float(grad_sum_err),
        H_min_eig_max=float(wH[:, 0].max()),
        corank1=int((wH[:, 1] > 1e-6).sum()),
        rank_ratio_min=float(rank_ratio.min()),
        gap=float(out["gap"]),
        t_cpu_solve=t_cpu_solve, t_cpu_bwd=t_cpu_bwd,
        t_gpu_fwd=t_gpu_fwd, t_gpu_bwd=t_gpu_bwd,
        n_stages=len(out["stats"]),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--batch", type=int, default=20)
    ap.add_argument("--out", default=os.path.join(ROOT, "repro/gpu_sdpr/results/verify_p6.json"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    rows = []
    for seed in range(args.seeds):
        row = run_seed(seed, args.batch, args.device)
        rows.append(row)
        print("seed %2d  xerr %.2e  grad_sum_err %.2e  t_gpu %.3fs  t_cpu %.3fs"
              % (seed, row["x_err_max"], row["grad_sum_err"],
                 row["t_gpu_fwd"], row["t_cpu_solve"]), flush=True)

    def agg(key):
        vals = np.array([r[key] for r in rows], dtype=float)
        return dict(mean=float(np.mean(vals)), max=float(np.max(vals)),
                    min=float(np.min(vals)))

    summary = dict(
        seeds=args.seeds, batch=args.batch, n_poses=args.seeds * args.batch,
        x_err_rel=agg("x_err_max"),
        grad_sum_err_rel=agg("grad_sum_err"),
        grad_abs_max=agg("grad_abs_max"),
        grad_err_per_pose_rel=agg("grad_err_max"),
        H_min_eig_max=agg("H_min_eig_max"),
        corank1_total=int(sum(r["corank1"] for r in rows)),
        rank_ratio_min=float(min(r["rank_ratio_min"] for r in rows)),
        gap=float(rows[0]["gap"]),
        t_gpu_fwd=agg("t_gpu_fwd"),
        t_gpu_bwd=agg("t_gpu_bwd"),
        t_cpu_solve=agg("t_cpu_solve"),
        t_cpu_bwd=agg("t_cpu_bwd"),
        speedup_solve=float(np.mean([r["t_cpu_solve"] / r["t_gpu_fwd"] for r in rows])),
        speedup_backward=float(np.mean([r["t_cpu_bwd"] / r["t_gpu_bwd"] for r in rows])),
    )
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(dict(summary=summary, runs=rows), f, indent=2)
    print(json.dumps(summary, indent=2))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
