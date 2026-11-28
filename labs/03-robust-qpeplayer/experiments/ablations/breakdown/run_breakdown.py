#!/usr/bin/env python3
"""Breakdown sweep: random 70-90% and coherent 40-60%.

The main table stops at 70% random and only samples 50/60/70% coherent.
This script zooms into the transition:
* random p in {0.70,0.75,0.80,0.85,0.90}, 100 trials, 5 methods;
* coherent p in {0.40,0.45,0.50,0.55,0.60}, 200 trials, 5 methods,
  plus the fraction that recovers the outlier transform.

Run:
  srun --partition=cpu --time=00:20:00 --ntasks=1 --cpus-per-task=2 \
    /home/syk/miniconda3/envs/WM/bin/python run_breakdown.py
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
RANDOM_P = [0.70, 0.75, 0.80, 0.85, 0.90]
COHERENT_P = [0.40, 0.45, 0.50, 0.55, 0.60]
RANDOM_TRIALS = 100
COHERENT_TRIALS = 200
METHODS = ["Kabsch", "RANSAC-1000", "RANSAC-adaptive", "IRLS-GM", "GNC-GM"]


def methods():
    def kabsch(x, y, rng):
        return (*cm.kabsch(x, y, np.ones(len(x))), {})

    def rf(x, y, rng):
        return cm.solve_ransac(x, y, rng, c=C, iters=1000, adaptive=False, return_info=True)

    def ra(x, y, rng):
        return cm.solve_ransac(x, y, rng, c=C, iters=1000, adaptive=True, return_info=True)

    def irls(x, y, rng):
        return (*cm.solve_irls(x, y, c=C), {})

    def gnc(x, y, rng):
        return (*cm.solve_gnc(x, y, c=C), {})

    return {"Kabsch": kabsch, "RANSAC-1000": rf, "RANSAC-adaptive": ra,
            "IRLS-GM": irls, "GNC-GM": gnc}


def run_cell(fns, p, otype, trials, seed0):
    rows = []
    recs = {m: [] for m in METHODS}
    for trial in range(trials):
        rng = np.random.default_rng([20261127, 17, {"random": 1, "coherent": 2}[otype],
                                     int(round(100 * p)), trial])
        x, y, Rgt, tgt, meta = cm.make_instance(rng, p, n=N, outlier_type=otype,
                                                return_meta=True)
        for m in METHODS:
            rrng = np.random.default_rng([20261127, 18, int(round(100 * p)), trial])
            R, t, info = fns[m](x, y, rrng)
            re, te = cm.pose_errors(R, t, Rgt, tgt)
            r = dict(p=p, outlier_type=otype, trial=trial, method=m,
                     success=cm.success(re, te), rot=re, trans=te)
            if otype == "coherent" and meta.get("Rout") is not None:
                re_o = cm.rot_err_deg(R, meta["Rout"])
                te_o = float(np.linalg.norm(t - meta["tout"]))
                r["picked_outlier"] = bool(re_o < re and te_o < te)
            recs[m].append(r)
    for m in METHODS:
        r = recs[m]
        re = np.array([z["rot"] for z in r])
        te = np.array([z["trans"] for z in r])
        row = dict(p=p, outlier_type=otype, method=m, n=len(r),
                   success=float(np.mean([z["success"] for z in r])),
                   rot_median=float(np.median(re)),
                   rot_p90=float(np.percentile(re, 90)),
                   trans_median=float(np.median(te)))
        if otype == "coherent":
            row["frac_picked_outlier"] = float(np.mean([z.get("picked_outlier", False) for z in r]))
        rows.append(row)
    return rows


def main():
    fns = methods()
    rows = []
    for otype, ps, trials in [("random", RANDOM_P, RANDOM_TRIALS),
                              ("coherent", COHERENT_P, COHERENT_TRIALS)]:
        for p in ps:
            t0 = time.time()
            rr = run_cell(fns, p, otype, trials, 0)
            rows += rr
            print(f"{otype} p={p:.2f} done in {time.time()-t0:.1f}s: " +
                  ", ".join(f"{z['method']}={100*z['success']:.0f}%" for z in rr), flush=True)
    cm.json_dump(OUT / "breakdown.json", dict(
        config=dict(n=N, c=C, random_p=RANDOM_P, coherent_p=COHERENT_P,
                    random_trials=RANDOM_TRIALS, coherent_trials=COHERENT_TRIALS,
                    methods=METHODS), rows=rows))
    L = ["# Breakdown sweep", "",
         f"N={N}, c={C}; random {RANDOM_TRIALS} trials/cell, coherent {COHERENT_TRIALS}.",
         "", "## Random outliers: success rate", "",
         "| method | " + " | ".join(f"{int(100*p)}%" for p in RANDOM_P) + " |",
         "|---|" + "|".join(["---"] * len(RANDOM_P)) + "|"]
    for m in METHODS:
        cells = [f"{100*next(z for z in rows if z['outlier_type']=='random' and z['method']==m and abs(z['p']-p)<1e-9)['success']:.0f}"
                 for p in RANDOM_P]
        L.append(f"| {m} | " + " | ".join(cells) + " |")
    L += ["", "## Coherent outliers: success rate", "",
          "| method | " + " | ".join(f"{int(100*p)}%" for p in COHERENT_P) + " |",
          "|---|" + "|".join(["---"] * len(COHERENT_P)) + "|"]
    for m in METHODS:
        cells = [f"{100*next(z for z in rows if z['outlier_type']=='coherent' and z['method']==m and abs(z['p']-p)<1e-9)['success']:.0f}"
                 for p in COHERENT_P]
        L.append(f"| {m} | " + " | ".join(cells) + " |")
    L += ["", "## Coherent outliers: fraction that recovers the outlier transform", "",
          "| method | " + " | ".join(f"{int(100*p)}%" for p in COHERENT_P) + " |",
          "|---|" + "|".join(["---"] * len(COHERENT_P)) + "|"]
    for m in METHODS:
        cells = [f"{100*next(z for z in rows if z['outlier_type']=='coherent' and z['method']==m and abs(z['p']-p)<1e-9)['frac_picked_outlier']:.0f}"
                 for p in COHERENT_P]
        L.append(f"| {m} | " + " | ".join(cells) + " |")
    L += ["", "## Reading", "",
          "* Random 80–90% is the fixed-RANSAC failure regime; GNC may still work because its "
          "cost does not depend on the clean-sample probability.",
          "* Coherent 45–55% is a coin-flip / identifiability transition, not a solver bug; "
          "the recovered-outlier column makes that explicit."]
    (OUT / "breakdown.md").write_text("\n".join(L) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
