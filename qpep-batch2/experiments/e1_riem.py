#!/usr/bin/env python
"""E1: RiemQPEP-Layer — 前向全局解、3x3 隐式 VJP、双覆盖与计时。"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from core import algebra as A, modes, problems as P  # noqa: E402
from p1_riem import layer  # noqa: E402


def main():
    rng = np.random.default_rng(20261128)
    rows = []
    for seed in range(12):
        prob = P.random_instance(seed=seed, N=10, noise=0.02)
        q, f, _, _ = layer.forward(prob, n_starts=48)
        gq = rng.normal(size=4)
        dS = rng.normal(size=(10, 10))
        dQ = 0.5 * (dS + dS.T)
        h = 1e-5
        qp, _, _ = modes.newton(q, prob.Qm + h * dQ)
        qm, _, _ = modes.newton(q, prob.Qm - h * dQ)
        fd_rot = float((gq @ qp - gq @ qm) / (2 * h))
        GQ, _ = layer.vjp(prob.Qm, q, gq)
        an_rot = float(np.sum(GQ * dQ))
        rel_rot = abs(an_rot - fd_rot) / max(1e-14, abs(fd_rot))

        gt = rng.normal(size=3)
        fd_tr = float((gt @ prob.t_star(qp) - gt @ prob.t_star(qm)) / (2 * h))
        GQ2, _ = layer.vjp_with_translation(prob, q, np.zeros(4), gt)
        an_tr = float(np.sum(GQ2 * dQ))
        rel_tr = abs(an_tr - fd_tr) / max(1e-14, abs(fd_tr))

        GQc, _ = layer.vjp(prob.Qm, -q, -gq)
        cover = float(np.max(np.abs(GQ - GQc)))

        rot_err = float(np.rad2deg(np.arccos(np.clip(abs(q @ prob.meta["q_true"]), -1, 1))))
        rows.append(dict(seed=seed, f_star=f, rot_err_deg=rot_err,
                         vjp_rot_rel=rel_rot, vjp_trans_rel=rel_tr,
                         cover_max_abs=cover))

    # 计时：只计反向（3x3 解 + 外积），前向另计
    prob = P.random_instance(seed=0)
    q, _, _, _ = layer.forward(prob, n_starts=48)
    gq = rng.normal(size=4)
    GQ, _ = layer.vjp(prob.Qm, q, gq)
    t0 = time.perf_counter()
    n_rep = 2000
    for _ in range(n_rep):
        layer.vjp(prob.Qm, q, gq)
    t_back = (time.perf_counter() - t0) / n_rep
    t0 = time.perf_counter()
    for _ in range(20):
        layer.forward(prob, n_starts=32)
    t_fwd = (time.perf_counter() - t0) / 20
    # 局部精化：从扰动 1e-3 的解出发做一次黎曼牛顿
    qp = q + 1e-3 * (A.E(q) @ rng.normal(size=3))
    qp /= np.linalg.norm(qp)
    t0 = time.perf_counter()
    for _ in range(2000):
        modes.newton(qp, prob.Qm, max_iter=50)
    t_ref = (time.perf_counter() - t0) / 2000

    out = dict(
        instances=rows,
        vjp_rot_rel_max=float(max(r["vjp_rot_rel"] for r in rows)),
        vjp_trans_rel_max=float(max(r["vjp_trans_rel"] for r in rows)),
        cover_max_abs=float(max(r["cover_max_abs"] for r in rows)),
        rot_err_deg_median=float(np.median([r["rot_err_deg"] for r in rows])),
        backward_seconds=t_back,
        backward_us=t_back * 1e6,
        forward_seconds=t_fwd,
        forward_ms=t_fwd * 1e3,
        local_refine_us=t_ref * 1e6,
    )
    with open(os.path.join(ROOT, "results", "e1_riem.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
