#!/usr/bin/env python
"""E3: SpecQPEP-Layer — 代数谱配分、Maxwell 交叉、秩 2 坍缩与采样方差。"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from core import algebra as A, sdp  # noqa: E402
from p3_spec import layer as S  # noqa: E402


def main():
    qA, qB = S.two_mode_data()
    rng = np.random.default_rng(20261128)
    quats = rng.normal(size=(400000, 4))
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)
    beta = 0.05
    h = 1e-4
    thetas = [0.05, 0.10, 0.15, 0.19, 0.20, 0.21, 0.25, 0.30, 0.35]

    def loss(theta, kind):
        sp = S.spectrum(S.two_mode_Q(theta), beta, extra=[qA, qB], n_starts=6)
        f_gt = theta  # q_gt = q_B, f(q_B) = theta
        if kind == "Z0":
            return f_gt / beta + np.log(sp["Z0"])
        if kind == "ZML":
            return f_gt / beta + np.log(sp["ZML"])
        if kind == "true":
            return f_gt / beta + np.log(S.Z_true(theta, beta, quats))
        if kind == "hard":
            return f_gt / beta + min(sp["f"]) / beta
        raise ValueError(kind)

    rows = []
    B = np.outer(S.uB, S.uB)
    for theta in thetas:
        sp = S.spectrum(S.two_mode_Q(theta), beta, extra=[qA, qB], n_starts=6)
        fd = {k: (loss(theta + h, k) - loss(theta - h, k)) / (2 * h)
              for k in ("Z0", "ZML", "true", "hard")}
        an_z0 = 1.0 / beta + S.grad_Z0_theta(sp, B, beta)
        naive_ml = 1.0 / beta + S.grad_ZML_shape_only(sp, B, beta)
        rows.append(dict(theta=theta, f=sp["f"].tolist(), pi0=sp["pi0"].tolist(),
                         piL=sp["piL"].tolist(),
                         fd=fd, an_z0=an_z0, naive_ml=naive_ml,
                         Z0=sp["Z0"], ZML=sp["ZML"],
                         Ztrue=S.Z_true(theta, beta, quats)))

    # 蒙特卡洛：按 Boltzmann 权重采样一个模式，估计 d/dθ log Z0
    mc = []
    for theta in [0.10, 0.20, 0.30]:
        sp = S.spectrum(S.two_mode_Q(theta), beta, extra=[qA, qB], n_starts=6)
        fp = np.array([A.m2(q) @ B @ A.m2(q) for q in sp["q"]])
        est = []
        for seed in range(200):
            r = np.random.default_rng(seed)
            k = r.choice(len(sp["q"]), p=sp["pi0"])
            est.append(-fp[k] / beta)
        mc.append(dict(theta=theta, mean=float(np.mean(est)), std=float(np.std(est))))

    # SDP 秩：tie 处应为秩 2
    sdp_rows = []
    for theta in [0.15, 0.20, 0.25]:
        Q = S.two_mode_Q(theta)
        b = sdp.sdp_bound(Q)
        p = sdp.sdp_primal(Q)
        ev = np.clip(p["eig"], 0, None)
        sdp_rows.append(dict(theta=theta, dual=b["gamma"], primal=p["value"],
                             X_eig=ev[-4:].tolist(),
                             lambda1_over_lambda2=float(ev[-1] / ev[-2]) if ev[-2] > 0 else None))

    out = dict(beta=beta, rows=rows, mc=mc, sdp=sdp_rows,
               Z0_fd_vs_an_max=float(max(abs(r["fd"]["Z0"] - r["an_z0"]) for r in rows)),
               ZML_vs_true_max=float(max(abs(r["fd"]["ZML"] - r["fd"]["true"]) for r in rows)),
               Z0_vs_true_max=float(max(abs(r["fd"]["Z0"] - r["fd"]["true"]) for r in rows)),
               naiveML_vs_ZML_max=float(max(abs(r["naive_ml"] - r["fd"]["ZML"]) for r in rows)))
    with open(os.path.join(ROOT, "results", "e3_spec.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
