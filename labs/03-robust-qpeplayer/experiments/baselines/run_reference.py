#!/usr/bin/env python3
"""Reference methods (GNC / IRLS / RANSAC / Kabsch) on the baseline instance set.

This re-runs the *same* instance generator and the *same* methods as
``experiments/synthetic/common.py`` so that the Theseus numbers are compared
against numbers measured in the same session, same machine, same seed
scheme -- not against numbers copied from the main table.

Outputs
-------
results/reference_raw.jsonl   one JSON record per (p, trial, method)
results/reference.json        per-cell summary
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "synthetic"))
import common  # noqa: E402

P_LIST = [0.0, 0.3, 0.5, 0.7]
N_TRIALS_DEFAULT = 25
N_POINTS = 100
OUT = HERE / "results"

METHODS = ["Kabsch", "RANSAC-1000", "RANSAC-adaptive", "IRLS-GM", "GNC-GM"]


def instance_rng(p: float, trial: int) -> np.random.Generator:
    """Same seed stream as the Theseus run (identical instances)."""
    return np.random.default_rng(
        np.random.SeedSequence([20261127, 103, 0, round(100 * p), trial])
    )


def solve(method: str, x, y, rng):
    if method == "Kabsch":
        return common.kabsch(x, y, np.ones(len(x)))
    if method == "RANSAC-1000":
        return common.solve_ransac(x, y, rng, adaptive=False)
    if method == "RANSAC-adaptive":
        return common.solve_ransac(x, y, rng, adaptive=True)
    if method == "IRLS-GM":
        return common.solve_irls(x, y)
    if method == "GNC-GM":
        return common.solve_gnc(x, y)
    raise ValueError(method)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=N_TRIALS_DEFAULT)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    n_trials = args.trials
    out_json = args.out or (OUT / "reference.json")
    OUT.mkdir(parents=True, exist_ok=True)
    raw = open(out_json.with_name(out_json.stem + "_raw.jsonl"), "w")
    summary = {}
    t_start = time.time()
    for p in P_LIST:
        cells = {m: dict(re=[], te=[], t=[], succ=[]) for m in METHODS}
        for trial in range(n_trials):
            rng = instance_rng(p, trial)
            x, y, Rgt, tgt = common.make_instance(rng, p, n=N_POINTS,
                                                  outlier_type="random")
            for m in METHODS:
                rng_s = np.random.default_rng(
                    np.random.SeedSequence([20261127, 103, 1, round(100 * p), trial])
                )
                t0 = time.perf_counter()
                R, t = solve(m, x, y, rng_s)
                dt = time.perf_counter() - t0
                re, te = common.pose_errors(R, t, Rgt, tgt)
                s = common.success(re, te)
                c = cells[m]
                c["re"].append(re); c["te"].append(te); c["t"].append(dt)
                c["succ"].append(bool(s))
                raw.write(json.dumps(dict(p=p, trial=trial, method=m,
                                          rot_err_deg=re, trans_err=te,
                                          success=bool(s), time_s=dt)) + "\n")
        summary[str(p)] = {m: common.summarize_errors(c["re"], c["te"], c["t"])
                           for m, c in cells.items()}
        print(f"p={p}: " + " | ".join(
            f"{m} succ={summary[str(p)][m]['success']:.2f} "
            f"p50={summary[str(p)][m]['time_ms_p50']:.2f}ms" for m in METHODS),
            flush=True)
    raw.close()
    common.json_dump(out_json, dict(
        protocol=dict(n_points=N_POINTS, n_trials=n_trials, p_list=P_LIST,
                      outlier_type="random", seed="SeedSequence([20261127,103,0,100p,trial])",
                      success="rot<5deg and trans<0.1m", c=common.C,
                      wall_time_s=time.time() - t_start),
        cells=summary,
    ))
    print(f"done in {time.time() - t_start:.1f}s -> {out_json}")


if __name__ == "__main__":
    main()
