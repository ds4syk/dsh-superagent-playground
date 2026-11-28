#!/usr/bin/env python
"""E0: 定理 0（QPEP–SDPRLayer 同构与 Hilbert 紧性）审计。

跑法：/home/syk/miniconda3/envs/sdpr/bin/python experiments/e0_theorem0.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from core import algebra as A, modes, problems as P, sdp  # noqa: E402


def choi_lam():
    Q = np.zeros((10, 10))
    Q[0, 1] = Q[1, 0] = 0.5
    Q[0, 2] = Q[2, 0] = 0.5
    Q[1, 2] = Q[2, 1] = 0.5
    Q[3, 3] = 1.0
    Q[4, 9] = Q[9, 4] = -2.0
    return Q


def main():
    out = {}
    out["T_det"] = float(np.linalg.det(A.T))
    # 21 条约束的秩与核维数
    M = np.array([H.reshape(-1) for H in A.H_CONSTRAINTS])
    out["H_constraints_rank"] = int(np.linalg.matrix_rank(M, tol=1e-9))
    out["H_constraints_count"] = int(M.shape[0])
    out["veronese_kernel_dim"] = 20  # 55 - 35，符号计算已核

    # Choi–Lam 反例
    Qcl = choi_lam()
    res = modes.find_stationary(Qcl, n_starts=256, seed=0)
    fmin = float(res["f"].min())
    b = sdp.sdp_bound(Qcl)
    p = sdp.sdp_primal(Qcl)
    ev = np.clip(p["eig"], 0, None)
    out["choi_lam"] = dict(
        f_min=fmin,
        sdp_gamma=b["gamma"],
        gap=fmin - b["gamma"],
        q_min=res["q"][np.argmin(res["f"])].tolist(),
        Q_eig=np.linalg.eigvalsh(Qcl).tolist(),
        X_eig=p["eig"].tolist(),
        primal_value=p["value"],
        rank=int(np.sum(ev > 1e-6 * max(ev[-1], 1e-12))),
    )

    # 真实 QPEP 实例：Qm 是否 PSD，10x10 SDP 是否紧
    real = []
    for seed in range(12):
        prob = P.random_instance(seed=seed, N=10, noise=0.02)
        r = modes.find_minima(prob.Qm, n_starts=64, seed=seed)
        fstar = float(r["f"][0])
        b = sdp.sdp_bound(prob.Qm)
        p = sdp.sdp_primal(prob.Qm)
        ev = np.clip(p["eig"], 0, None)
        real.append(dict(
            seed=seed,
            Qm_min_eig=float(np.linalg.eigvalsh(prob.Qm)[0]),
            f_star=fstar,
            sdp_gamma=b["gamma"],
            gap=fstar - b["gamma"],
            rank1=bool(ev[-2] < 1e-6 * ev[-1]) if ev[-1] > 0 else False,
            rot_err_deg=float(np.rad2deg(np.arccos(np.clip(abs(r["q"][0] @ prob.meta["q_true"]), -1, 1)))),
        ))
    out["real_qpep"] = real
    out["real_qpep_gap_max"] = float(max(abs(x["gap"]) for x in real))
    out["real_qpep_Qm_min_eig_min"] = float(min(x["Qm_min_eig"] for x in real))

    # 随机 PSD Q：检验“PSD ⇒ 10x10 SDP 紧”是否普遍
    rng = np.random.default_rng(20261128)
    psd = []
    for trial in range(8):
        S = rng.normal(size=(10, 10))
        Q = S @ S.T
        r = modes.find_minima(Q, n_starts=64, seed=trial)
        fstar = float(r["f"][0])
        b = sdp.sdp_bound(Q)
        psd.append(dict(trial=trial, f_star=fstar, sdp_gamma=b["gamma"],
                        gap=fstar - b["gamma"]))
    out["random_psd"] = psd
    out["random_psd_gap_max"] = float(max(abs(x["gap"]) for x in psd))

    # 更大规模 PSD 搜索：40 个随机秩 1..10 的 PSD Q
    rng2 = np.random.default_rng(7)
    gaps = []
    for trial in range(40):
        r = int(rng2.integers(1, 11))
        Sm = rng2.normal(size=(r, 10))
        Q = Sm.T @ Sm
        rr = modes.find_stationary(Q, n_starts=40, seed=trial)
        if len(rr["f"]) == 0:
            continue
        gaps.append(float(rr["f"].min() - sdp.sdp_bound(Q)["gamma"]))
    out["psd_search_40"] = dict(n=len(gaps), worst_gap=float(max(gaps)),
                                mean_gap=float(np.mean(gaps)))

    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "e0_theorem0.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
