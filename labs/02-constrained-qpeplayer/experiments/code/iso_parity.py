#!/usr/bin/env python
"""Parity test: structure-aware reduced solver vs the committed 40-root solver (S6/S7).

For N canonical rank-2 instances (the authoritative 16-root template case) and
M isotropic Horn + k=2 plane ALM assemblies, compare

  * ``isotropic_roots.solve_isotropic_plane``  (census + endgame polish + 64-start BFGS)
  * ``quartic_roots.solve_inner_exact``        (committed 162-path solver)

on the global minimum (f, q up to sign), the census, fallback/exact flags and
wall time.  Writes ``results/isotropic_parity.json`` and a markdown section
consumed by ``results/round_roots.md``.

Run:  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
          /home/syk/miniconda3/envs/sdpr/bin/python code/iso_parity.py
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import isotropic as ISO          # noqa: E402
import isotropic_roots as IR     # noqa: E402
import quartic_roots as QR       # noqa: E402

RES = os.path.join(HERE, "..", "results")
# solve_isotropic_plane's predicted census for a rank-2 canonical instance
N_GAMMA = 2
NORM_CAP = 1e5


def _qdist(q1, q2):
    if q1 is None or q2 is None:
        return float("nan")
    q1 = np.asarray(q1, float)
    q2 = np.asarray(q2, float)
    return float(min(np.linalg.norm(q1 - q2), np.linalg.norm(q1 + q2)))


def _run_instance(inst, tag):
    M = np.asarray(inst["Mred"], float)
    n = inst.get("n", None)
    if n is None and inst.get("Ns") is not None and len(inst["Ns"]):
        n = inst["Ns"][0]
    t0 = time.perf_counter()
    red = IR.solve_isotropic_plane(inst, n_gamma=N_GAMMA, multistart=64,
                                   norm_cap=NORM_CAP, structure=True)
    t_red = time.perf_counter() - t0
    t0 = time.perf_counter()
    com = QR.solve_inner_exact(M, seed=0, n_gamma=N_GAMMA, fallback_starts=64)
    t_com = time.perf_counter() - t0
    rec = dict(tag=tag,
               f_reduced=red["f"], f_committed=com["f"],
               dq=_qdist(red["q"], com["q"]),
               n_finite_reduced=red["n_finite_projective"],
               n_finite_committed=int(com["n_finite_projective"]),
               n_unique_reduced=red["n_real_geometric"],
               n_unique_committed=int(com["n_unique"]),
               fallback_reduced=bool(red["fallback"]),
               fallback_committed=bool(com["fallback"]),
               exact_committed=bool(com["exact"]),
               rank_Sigma=red.get("rank_Sigma"),
               predicted_NC=red.get("predicted_NC"),
               n_lost=red["n_lost"],
               t_reduced=t_red, t_committed=t_com)
    print("%-28s f_red=%.12f f_com=%.12f dq=%.1e | NC %s/%s | fallback %s/%s "
          "| %.2fs/%.2fs" %
          (tag, rec["f_reduced"], rec["f_committed"], rec["dq"],
           rec["n_finite_reduced"], rec["n_finite_committed"],
           rec["fallback_reduced"], rec["fallback_committed"],
           rec["t_reduced"], rec["t_committed"]))
    return rec


def main():
    rows = []
    for seed in range(10):
        inst = ISO.make_canonical_instance(seed, 2)
        rows.append(_run_instance(inst, "canonical-r2-seed%d" % seed))
    for seed in range(5):
        inst = ISO.make_isotropic_instance(seed, 2, base="horn", normals=1)
        M = ISO.alm_reduced(inst, act=np.arange(2), rho=inst["rho"])[0]
        inst = dict(inst)
        inst["Mred"] = M
        rows.append(_run_instance(inst, "alm-horn-k2-seed%d" % seed))
    ok_f = max(abs(r["f_reduced"] - r["f_committed"]) for r in rows)
    ok_q = max(r["dq"] for r in rows)
    out = dict(n_gamma=N_GAMMA, norm_cap=NORM_CAP, rows=rows,
               max_abs_delta_f=float(ok_f), max_dq=float(ok_q),
               n_instances=len(rows))
    with open(os.path.join(RES, "isotropic_parity.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("\nmax |Delta f| = %.2e   max dq = %.2e  (n=%d)"
          % (ok_f, ok_q, len(rows)))


if __name__ == "__main__":
    main()
