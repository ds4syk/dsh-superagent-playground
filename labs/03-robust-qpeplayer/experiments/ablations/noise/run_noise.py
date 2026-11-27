#!/usr/bin/env python3
"""Inlier-noise sweep at fixed kernel scale c=0.05.

The main tables fix sigma=0.01.  Real sensors vary; this sweep measures
whether GNC's advantage survives when inlier noise approaches the kernel
scale.  sigma in {0.005, 0.01, 0.02, 0.03, 0.05}, p in {0.3,0.5,0.7},
50 trials/cell, random outliers.
"""
from __future__ import annotations

import json
import sys
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
SIGMAS = [0.005, 0.01, 0.02, 0.03, 0.05]
P_LIST = [0.3, 0.5, 0.7]
TRIALS = 50
METHODS = ["Kabsch", "RANSAC-1000", "IRLS-GM", "GNC-GM"]


def main():
    rows = []
    old_sigma = cm.SIGMA
    for sigma in SIGMAS:
        cm.SIGMA = sigma
        for p in P_LIST:
            acc = {m: dict(re=[], te=[], tt=[]) for m in METHODS}
            for trial in range(TRIALS):
                rng = np.random.default_rng([20261127, 61, int(round(1000 * sigma)),
                                             int(round(100 * p)), trial])
                x, y, Rgt, tgt = cm.make_instance(rng, p, n=N)
                for name in METHODS:
                    rrng = np.random.default_rng([20261127, 62, trial])
                    if name == "Kabsch":
                        R, t = cm.kabsch(x, y, np.ones(len(x)))
                    elif name == "RANSAC-1000":
                        R, t = cm.solve_ransac(x, y, rrng, c=C, iters=1000, adaptive=False)
                    elif name == "IRLS-GM":
                        R, t = cm.solve_irls(x, y, c=C)
                    else:
                        R, t = cm.solve_gnc(x, y, c=C)
                    re, te = cm.pose_errors(R, t, Rgt, tgt)
                    acc[name]["re"].append(re)
                    acc[name]["te"].append(te)
            for name in METHODS:
                a = acc[name]
                s = cm.summarize_errors(a["re"], a["te"], [0.0] * len(a["re"]))
                rows.append(dict(sigma=sigma, p=p, method=name, **s))
            print(f"sigma={sigma} p={p} done", flush=True)
    cm.SIGMA = old_sigma
    cm.json_dump(OUT / "noise.json", dict(
        config=dict(n=N, c=C, sigmas=SIGMAS, p=P_LIST, trials=TRIALS, methods=METHODS),
        rows=rows))
    L = ["# Inlier-noise sweep", "",
         f"N={N}, c={C}, {TRIALS} trials/cell, random outliers. Fixed kernel scale; "
         "sigma is the Gaussian inlier noise.", ""]
    for p in P_LIST:
        L += [f"## p={p:.1f}: success rate", "",
              "| method | " + " | ".join(f"σ={s}" for s in SIGMAS) + " |",
              "|---|" + "|".join(["---"] * len(SIGMAS)) + "|"]
        for m in METHODS:
            cells = [f"{100*next(z for z in rows if z['sigma']==s and z['p']==p and z['method']==m)['success']:.0f}"
                     for s in SIGMAS]
            L.append(f"| {m} | " + " | ".join(cells) + " |")
        L += ["", f"## p={p:.1f}: median rotation error (deg)", "",
              "| method | " + " | ".join(f"σ={s}" for s in SIGMAS) + " |",
              "|---|" + "|".join(["---"] * len(SIGMAS)) + "|"]
        for m in METHODS:
            cells = [f"{next(z for z in rows if z['sigma']==s and z['p']==p and z['method']==m)['rot_median']:.2f}"
                     for s in SIGMAS]
            L.append(f"| {m} | " + " | ".join(cells) + " |")
        L.append("")
    L += ["## Reading", "",
          "* The success threshold is 5 deg / 0.1 m, so high sigma can fail even at 0% outliers; "
          "the interesting question is whether the robust ranking changes.",
          "* If GNC only wins when sigma << c, that is an important scope restriction."]
    (OUT / "noise.md").write_text("\n".join(L) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
