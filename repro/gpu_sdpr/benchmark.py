"""Batch-scaling benchmark: GPU batched barrier vs CPU SCS (single process).

Usage
-----
    python repro/gpu_sdpr/benchmark.py --batches 20 80 200
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
import sdp_pose  # noqa: E402
from gpu_sdpr import BarrierSolver, build_p6_sdp  # noqa: E402


def make_batch(n_batch: int, seed: int = 0):
    cfg = bilevel.ExpConfig(n_batch=n_batch, seed0=seed)
    data = bilevel.make_dataset(cfg, 0)
    cam = bilevel.make_camera(cfg)
    b = cfg.b_true + cfg.b_offset
    Qs = []
    for k in range(n_batch):
        meas, W = cam.inverse(data["pixels"][k], b)
        Q = sdp_pose.build_objective_matrix(data["r_l"], meas, W)
        Qs.append(sdp_pose.scale_objective(Q, sdp_pose.objective_scale(Q)))
    return np.stack(Qs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, nargs="+", default=[20, 80, 200])
    ap.add_argument("--out", default=os.path.join(ROOT, "repro/gpu_sdpr/results/benchmark.json"))
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    sdp = build_p6_sdp()
    solver = BarrierSolver(sdp["A"], sdp["b"], sdp["c0"], device="cuda", mu_final=1e-8)
    rows = []
    for B in args.batches:
        Q = make_batch(B)
        Qd = torch.tensor(Q, device="cuda")
        solver.solve(Qd[: min(20, B)])  # warm-up
        torch.cuda.synchronize()
        ts = []
        for _ in range(args.repeats):
            t0 = time.time()
            out = solver.solve(Qd)
            torch.cuda.synchronize()
            ts.append(time.time() - t0)
        t_fwd = min(ts)
        gx = torch.randn(B, 13, dtype=torch.float64, device="cuda") * 1e-6
        t0 = time.time()
        solver.gradient(out["x"], out["H"], gx)
        torch.cuda.synchronize()
        t_bwd = time.time() - t0

        sv = sdp_pose.SDPPoseSolver(eps=1e-9)
        t0 = time.time()
        sv.solve(Q[0])
        t_cold = time.time() - t0
        t0 = time.time()
        for k in range(1, B):
            sv.solve(Q[k])
        t_warm = (time.time() - t0) / max(B - 1, 1)
        t_cpu = t_cold + t_warm * (B - 1)
        row = dict(
            batch=B,
            gpu_fwd_s=t_fwd, gpu_bwd_s=t_bwd,
            cpu_cold_s=t_cold, cpu_warm_s=t_warm, cpu_total_s=t_cpu,
            speedup_fwd=t_cpu / t_fwd,
        )
        rows.append(row)
        print("B=%3d  GPU fwd %.3fs  bwd %.4fs | CPU fwd %.3fs (cold %.3fs + %.4fs/pose)"
              " | speedup %.1fx" % (B, t_fwd, t_bwd, t_cpu, t_cold, t_warm,
                                    row["speedup_fwd"]), flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(dict(rows=rows), f, indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
