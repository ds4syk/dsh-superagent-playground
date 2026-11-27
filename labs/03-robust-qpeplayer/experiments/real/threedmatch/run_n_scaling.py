#!/usr/bin/env python3
"""3DMatch real-fragment scaling with N (GNC vs fixed-budget RANSAC).

One high-overlap pair family per scene is sampled at N=50/100/300/1000,
p=0.3 and p=0.7 random outliers.  The point is to check the claim that GNC
is O(N) per iteration while fixed 1000-iteration RANSAC is dominated by
its iteration count.  Base correspondences are GT-mapped NN (oracle), as
in run_threedmatch.py; this isolates the solver scaling.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SYN = HERE.parents[1] / "synthetic"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(SYN))
import load_3dmatch as L  # noqa: E402
import common as cm  # noqa: E402
import run_threedmatch as RT  # noqa: E402

OUT = HERE / "results"
(OUT / "logs").mkdir(parents=True, exist_ok=True)
N_LIST = [50, 100, 300, 1000]
P_LIST = [0.3, 0.7]
TRIALS = 20
C = 0.05
TAU = 0.06
METHODS = ["Kabsch", "RANSAC-1000", "IRLS-GM", "GNC-GM"]
BASE_SEED = 20261127


def main():
    t_start = time.time()
    rows, raw = [], []
    for scene in RT.SCENES:
        clouds = L.load_scene(RT.ROOT / scene, voxel=RT.VOXEL)
        gt_pairs = L.parse_gt_log(RT.ROOT / f"{scene}-evaluation" / "gt.log")
        chosen, _ = RT.select_pairs(clouds, gt_pairs, max_pairs=4)
        # Precompute a large base per pair.
        bases = []
        for k, pair in enumerate(chosen):
            rng = np.random.default_rng([BASE_SEED, 55, k])
            b = RT.clean_pair_sample(clouds, pair, rng, n=1500)
            if b is not None:
                bases.append((pair, b))
        print(f"[{scene}] {len(bases)} pairs, base sizes "
              f"{[len(b[1][0]) for b in bases]}", flush=True)
        for n in N_LIST:
            for p in P_LIST:
                for trial in range(TRIALS):
                    rng = np.random.default_rng([BASE_SEED, 56, RT.SCENES.index(scene),
                                                 n, int(round(100 * p)), trial])
                    pair, b = bases[int(rng.integers(len(bases)))]
                    x0, y0, d0, Rgt, tgt = b
                    take = rng.choice(len(x0), min(n, len(x0)), replace=False)
                    x, y, d = x0[take], y0[take], d0[take]
                    if len(x) < n:
                        pad = rng.choice(len(x), n - len(x), replace=True)
                        x, y, d = np.concatenate([x, x[pad]]), np.concatenate([y, y[pad]]), np.concatenate([d, d[pad]])
                    if p > 0:
                        idx = rng.choice(n, int(round(p * n)), replace=False)
                        y = y.copy()
                        y[idx] = clouds[pair["id2"]][rng.choice(len(clouds[pair["id2"]]), len(idx), replace=True)]
                    for name in METHODS:
                        rrng = np.random.default_rng([BASE_SEED, 57, n, trial])
                        t0 = time.perf_counter()
                        if name == "RANSAC-1000":
                            R, t = cm.solve_ransac(x, y, rrng, c=C, iters=1000, adaptive=False)
                        elif name == "Kabsch":
                            R, t = cm.kabsch(x, y, np.ones(len(x)))
                        elif name == "IRLS-GM":
                            R, t = cm.solve_irls(x, y, c=C)
                        else:
                            R, t = cm.solve_gnc(x, y, c=C)
                        dt = time.perf_counter() - t0
                        re, te = cm.pose_errors(R, t, Rgt, tgt)
                        rec = dict(scene=scene, p=p, n=n, trial=trial, method=name,
                                   success=cm.success(re, te), rot=re, trans=te,
                                   time_ms=1000 * dt)
                        raw.append(rec)
                for name in METHODS:
                    recs = [r for r in raw if r["scene"] == scene and r["p"] == p
                            and r["n"] == n and r["method"] == name]
                    re = np.array([r["rot"] for r in recs])
                    tt = np.array([r["time_ms"] for r in recs])
                    rows.append(dict(scene=scene, p=p, n=n, method=name,
                                     success=float(np.mean([r["success"] for r in recs])),
                                     rot_median=float(np.median(re)),
                                     time_ms_p50=float(np.median(tt)),
                                     time_ms_p95=float(np.percentile(tt, 95))))
                print(f"[{scene}] n={n} p={p} done ({time.time()-t_start:.1f}s)", flush=True)
    cm.json_dump(OUT / "n_scaling.json", dict(
        config=dict(n_list=N_LIST, p=P_LIST, trials=TRIALS, methods=METHODS),
        rows=rows))
    with open(OUT / "n_scaling_raw.jsonl", "w") as f:
        for r in raw:
            f.write(json.dumps(r) + "\n")
    Ls = ["# 3DMatch real fragments: N scaling", "",
          f"20 trials/cell, p in {P_LIST}, GT-NN oracle base. Runtime p50/p95 per call.", "",
          "| scene | p | N | method | success | rot med (deg) | p50 (ms) | p95 (ms) |",
          "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        Ls.append(f"| {RT.SCENE_SHORT[r['scene']]} | {r['p']:.1f} | {r['n']} | {r['method']} | "
                  f"{100*r['success']:.0f}% | {r['rot_median']:.2f} | {r['time_ms_p50']:.2f} | "
                  f"{r['time_ms_p95']:.2f} |")
    Ls += ["", "## Reading", "",
           "* If GNC p50 grows approximately linearly in N while RANSAC-1000 stays flat, "
           "the fixed-budget RANSAC advantage is only at small N; at large N its per-iteration "
           "scoring cost dominates.",
           "* Kabsch is the non-robust lower bound on runtime and a failure baseline at p>0."]
    (OUT / "n_scaling.md").write_text("\n".join(Ls) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
