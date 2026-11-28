#!/usr/bin/env python3
"""GNC schedule ablation: gamma, mu0 multiplier, and final polish steps.

The forward GNC-GM loop has three knobs that are usually left at defaults:
gamma (annealing rate), mu0 (initial surrogate width), and the number of
exact-kernel IRLS polish steps at mu=1.  This script measures whether the
defaults are defensible on random outliers, and whether a slower/faster
schedule changes success or runtime.

Run:
  srun --partition=cpu --time=00:20:00 --ntasks=1 --cpus-per-task=2 \
    /home/syk/miniconda3/envs/WM/bin/python run_gnc_schedule.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SYN = HERE.parents[1] / "synthetic"
sys.path.insert(0, str(SYN))
import common as cm  # noqa: E402

OUT = HERE / "results"
(OUT / "logs").mkdir(parents=True, exist_ok=True)
N = 100
C = 0.05
TRIALS = 50
P_LIST = [0.3, 0.5, 0.7]
GAMMAS = [1.2, 1.4, 1.7, 2.0]
MU0_MULTS = [0.5, 1.0, 4.0]
POLISH = [0, 5, 10]


def solve(x, y, gamma=1.4, mu0_mult=1.0, polish=5, max_mu_iters=60):
    R, t = cm.kabsch(x, y, np.ones(len(x)))
    e = cm.residuals(x, y, R, t) ** 2
    mu = max(3.0 * e.max() / (C * C) * mu0_mult, 1.0)
    for _ in range(max_mu_iters):
        w = (mu * C * C / (mu * C * C + e)) ** 2
        R, t = cm.kabsch(x, y, w)
        e = cm.residuals(x, y, R, t) ** 2
        if mu <= 1.0 + 1e-12:
            break
        mu = max(1.0, mu / gamma)
    for _ in range(polish):
        w = (C * C / (C * C + e)) ** 2
        R, t = cm.kabsch(x, y, w)
        e = cm.residuals(x, y, R, t) ** 2
    return R, t


def main():
    t_start = time.time()
    rows = []
    configs = [("default", dict(gamma=1.4, mu0_mult=1.0, polish=5))]
    configs += [(f"gamma={g}", dict(gamma=g, mu0_mult=1.0, polish=5)) for g in GAMMAS]
    configs += [(f"mu0x={m}", dict(gamma=1.4, mu0_mult=m, polish=5)) for m in MU0_MULTS]
    configs += [(f"polish={q}", dict(gamma=1.4, mu0_mult=1.0, polish=q)) for q in POLISH]
    # de-duplicate
    seen, uniq = set(), []
    for name, kw in configs:
        key = tuple(sorted(kw.items()))
        if key in seen:
            continue
        seen.add(key)
        uniq.append((name, kw))
    for p in P_LIST:
        for name, kw in uniq:
            re, te, tt = [], [], []
            for trial in range(TRIALS):
                rng = np.random.default_rng([20261127, 41, int(round(100 * p)), trial])
                x, y, Rgt, tgt = cm.make_instance(rng, p, n=N)
                t0 = time.perf_counter()
                R, t = solve(x, y, **kw)
                dt = time.perf_counter() - t0
                r, tr = cm.pose_errors(R, t, Rgt, tgt)
                re.append(r)
                te.append(tr)
                tt.append(dt)
            s = cm.summarize_errors(re, te, tt)
            rows.append(dict(p=p, config=name, **kw, **s))
            print(f"p={p:.1f} {name:12s} succ={100*s['success']:.0f}% "
                  f"rot={s['rot_median']:.3f} t={s['time_ms_p50']:.2f}ms "
                  f"({time.time()-t_start:.1f}s)", flush=True)
    cm.json_dump(OUT / "gnc_schedule.json", dict(
        config=dict(n=N, c=C, trials=TRIALS, p=P_LIST, gammas=GAMMAS,
                    mu0_mults=MU0_MULTS, polish=POLISH), rows=rows))
    L = ["# GNC schedule ablation", "",
         f"N={N}, c={C}, {TRIALS} trials/cell, random outliers. Default: gamma=1.4, "
         "mu0 multiplier=1.0, 5 polish steps.", "",
         "| config | p | success | rot med (deg) | p50 (ms) | p95 (ms) |",
         "|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['config']} | {r['p']:.1f} | {100*r['success']:.0f}% | "
                 f"{r['rot_median']:.3f} | {r['time_ms_p50']:.2f} | {r['time_ms_p95']:.2f} |")
    L += ["", "## Reading", "",
          "* The default is only defensible if no schedule variant gives a material "
          "success/accuracy win at comparable or lower runtime.",
          "* mu0 multiplier below 1 can start too narrow and miss the inlier basin; "
          "above 1 is safer but slower. polish=0 leaves the first-order residual larger."]
    (OUT / "gnc_schedule.md").write_text("\n".join(L) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
