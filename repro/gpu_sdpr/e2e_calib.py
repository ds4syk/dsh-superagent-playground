"""Mini end-to-end stereo baseline calibration with the GPU SDPR layer.

Runs the outer SGD of Holmes et al. (T-RO 2025, Sec. VI-B4 / P7) for a few
runs and iterations with (a) the reference CPU SCS layer and (b) the batched
GPU layer, and compares the baseline trajectories.

Usage
-----
    python repro/gpu_sdpr/e2e_calib.py --runs 3 --iters 30
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


def run_gpu(data, cfg, solver, max_iter):
    cam = bilevel.make_camera(cfg)
    m = data["r_l"]
    b = cfg.b_true + cfg.b_offset
    scales = []
    for k in range(cfg.n_batch):
        meas, W = cam.inverse(data["pixels"][k], b)
        Q = sdp_pose.build_objective_matrix(m, meas, W)
        scales.append(sdp_pose.objective_scale(Q))
    hist = dict(b=[], err=[], loss=[], grad=[], grad_sq=[], time_iter=[])
    grad_sq = np.inf
    loss = np.inf
    for _ in range(max_iter):
        t0 = time.perf_counter()
        Qs, dQs = [], []
        for k in range(cfg.n_batch):
            meas, W = cam.inverse(data["pixels"][k], b)
            Q = sdp_pose.build_objective_matrix(m, meas, W)
            Qs.append(sdp_pose.scale_objective(Q, scales[k]))
            dQ = scales[k] * _dQ_db(cam, data["pixels"][k], m, b)
            dQ[0, 0] = 0.0
            dQs.append(dQ)
        Qd = torch.tensor(np.stack(Qs), device="cuda")
        out = solver.solve(Qd)
        grad_x = np.zeros((cfg.n_batch, sdp_pose.N_VARS))
        losses = np.zeros(cfg.n_batch)
        for k in range(cfg.n_batch):
            losses[k], grad_x[k] = idiff.loss_from_x_and_grad(
                out["x"][k].detach().cpu().numpy(),
                data["C_p0s"][k], data["r_p0s"][k])
        gQ, _ = solver.gradient(out["x"], out["H"], torch.tensor(grad_x, device="cuda"))
        grads = torch.einsum("bij,bij->b", gQ, torch.tensor(np.stack(dQs), device="cuda"))
        loss = float(losses.sum())
        grad = float(grads.sum().item())
        grad_sq = grad ** 2
        b = b - cfg.lr * grad
        hist["b"].append(float(b))
        hist["err"].append(float(b - cfg.b_true))
        hist["loss"].append(loss)
        hist["grad"].append(grad)
        hist["grad_sq"].append(grad_sq)
        hist["time_iter"].append(time.perf_counter() - t0)
        if grad_sq <= cfg.tol_grad_sq or loss <= cfg.tol_loss:
            break
    return hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--batch", type=int, default=20)
    ap.add_argument("--out", default=os.path.join(ROOT, "repro/gpu_sdpr/results/e2e_calib.json"))
    args = ap.parse_args()

    sdp = build_p6_sdp()
    solver = BarrierSolver(sdp["A"], sdp["b"], sdp["c0"], device="cuda", mu_final=1e-8)
    cfg0 = bilevel.ExpConfig(n_batch=2, seed0=0)
    data0 = bilevel.make_dataset(cfg0, 0)
    cam0 = bilevel.make_camera(cfg0)
    Q0 = []
    for k in range(2):
        meas0, W0 = cam0.inverse(data0["pixels"][k], cfg0.b_true + cfg0.b_offset)
        Q0.append(sdp_pose.scale_objective(
            sdp_pose.build_objective_matrix(data0["r_l"], meas0, W0),
            sdp_pose.objective_scale(sdp_pose.build_objective_matrix(data0["r_l"], meas0, W0))))
    solver.solve(torch.tensor(np.stack(Q0), device="cuda"))  # warm-up

    out = dict(runs=[])
    for run in range(args.runs):
        cfg = bilevel.ExpConfig(n_batch=args.batch, seed0=run, max_iter=args.iters)
        data = bilevel.make_dataset(cfg, 0)
        cam = bilevel.make_camera(cfg)

        t0 = time.time()
        cpu = bilevel.run_trial(data, "sdpr", cfg)
        t_cpu = time.time() - t0

        t0 = time.time()
        gpu = run_gpu(data, cfg, solver, args.iters)
        t_gpu = time.time() - t0

        n = min(len(cpu["b"]), len(gpu["b"]))
        err_diff = np.abs(np.array(cpu["err"][:n]) - np.array(gpu["err"][:n]))
        row = dict(
            run=run,
            n_iter_cpu=cpu["n_iter"], n_iter_gpu=len(gpu["b"]),
            b_final_cpu=cpu["b"][-1], b_final_gpu=gpu["b"][-1],
            err_final_cpu=cpu["err"][-1], err_final_gpu=gpu["err"][-1],
            max_abs_traj_diff=float(err_diff.max()) if n else None,
            t_cpu=t_cpu, t_gpu=t_gpu,
            status_cpu=cpu["status"], status_gpu="max_iter",
        )
        out["runs"].append(row)
        print(json.dumps(row, indent=2), flush=True)
        out.setdefault("trajectories", {})[str(run)] = dict(
            b_cpu=cpu["b"], err_cpu=cpu["err"], loss_cpu=cpu["loss"],
            b_gpu=gpu["b"], err_gpu=gpu["err"], loss_gpu=gpu["loss"])

    out["summary"] = dict(
        runs=args.runs, iters=args.iters,
        max_abs_traj_diff=float(max(r["max_abs_traj_diff"] for r in out["runs"]
                                    if r["max_abs_traj_diff"] is not None)),
        err_final_cpu_mean=float(np.mean([r["err_final_cpu"] for r in out["runs"]])),
        err_final_gpu_mean=float(np.mean([r["err_final_gpu"] for r in out["runs"]])),
        speedup=float(np.mean([r["t_cpu"] / r["t_gpu"] for r in out["runs"]])),
        t_cpu_mean=float(np.mean([r["t_cpu"] for r in out["runs"]])),
        t_gpu_mean=float(np.mean([r["t_gpu"] for r in out["runs"]])),
    )
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out["summary"], indent=2))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
