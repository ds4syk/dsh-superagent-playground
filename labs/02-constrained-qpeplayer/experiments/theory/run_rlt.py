#!/usr/bin/env python3
"""P2-Q4 validation: zero-dimensional RLT cuts g_i g_j >= 0 on the 14x14 SDP.

Compares the Shor lower bound with/without RLT on synthetic C-QPEP instances
at several noise levels.  RLT cuts are valid for every feasible y (g <= 0),
so the RLT bound must be >= the base bound; a gap inspects tightness.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CODE = HERE.parent / "code"
sys.path.insert(0, str(CODE))
import qpep_core as core  # noqa: E402
from certificate import shor_bound  # noqa: E402

OUT = HERE / "results"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    rows = []
    for noise, bias in [(0.02, 0.1), (0.05, 0.2), (0.10, 0.3), (0.20, 0.4)]:
        for seed in range(4):
            inst = core.make_instance(seed, 16, noise=noise, bias_z=bias)
            M0, D0, G0, A, B = inst["M0"], inst["D0"], inst["G0"], inst["A"], inst["B"]
            alm = core.alm_qpep(M0, D0, G0, A, B, inst["q_gt"], inst["t_gt"],
                                rho0=10.0, n_starts=6, rng=np.random.default_rng(seed))
            J = core.obj_value(alm["q"], alm["t"], M0, D0, G0)
            feas = core.max_viol(A, B, alm["q"], alm["t"]) <= 1e-6
            base = shor_bound(M0, D0, G0, A, B, add_rlt=False,
                               solver="SCS", eps=1e-9, max_iters=50000)
            rlt = shor_bound(M0, D0, G0, A, B, add_rlt=True,
                              solver="SCS", eps=1e-9, max_iters=50000)
            rows.append(dict(noise=noise, bias=bias, seed=seed, J=float(J), feasible=bool(feas),
                              base=float(base["value"]), rlt=float(rlt["value"]),
                              base_gap=float(J - base["value"]),
                              rlt_gap=float(J - rlt["value"]),
                              base_cert=bool(feas and J - base["value"] <= 1e-6),
                              rlt_cert=bool(feas and J - rlt["value"] <= 1e-6),
                              base_time=float(base["time_s"]), rlt_time=float(rlt["time_s"])))
            print(rows[-1], flush=True)
    json.dump(rows, open(OUT / "rlt.json", "w"), indent=2)
    lines = ["# P2-Q4: 14x14 Shor + zero-dimensional RLT cuts\n",
             "| noise | bias | seed | J | base gap | RLT gap | base cert | RLT cert | base time | RLT time |",
             "|-|-|-|-|-|-|-|-|-|-|"]
    for r in rows:
        lines.append(f"| {r['noise']:.2f} | {r['bias']:.2f} | {r['seed']} | {r['J']:.4e} | {r['base_gap']:.2e} | "
                     f"{r['rlt_gap']:.2e} | {r['base_cert']} | {r['rlt_cert']} | {r['base_time']:.2f} | {r['rlt_time']:.2f} |")
    (OUT / "rlt.md").write_text("\n".join(lines) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
