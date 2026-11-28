#!/usr/bin/env python3
"""Theseus-GM baseline on the same synthetic 3D-3D instances.

Object of study: the *same* weighted 3D-3D / Geman-McClure objective as the
GNC-GM reference, optimised with Theseus' Levenberg-Marquardt on SE(3).
Each correspondence contributes its own GM-wrapped residual, so the kernel
is applied per point exactly as in ``common.solve_gnc`` (verified: see the
``self_check`` block in ``theseus_cost.py``).

Initialisations (protocol.md §2):
  * ``gt``   -- strong baseline: start at the ground-truth pose;
  * ``rand`` -- weak baseline: random rotation + translation in [-1,1]^3.

Metrics: success (rot < 5 deg and trans < 0.1 m), rot/trans error, solve
wall-clock (objective construction excluded), LM iterations, status.

Run with the isolated venv::

  /mnt/storage1/syk/venvs/theseus/bin/python run_theseus.py
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import theseus as th

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "synthetic"))
import common  # noqa: E402
from theseus_cost import build_gm_objective, errors_from_pose, self_check  # noqa: E402

P_LIST = [0.0, 0.3, 0.5, 0.7]
N_TRIALS = int(__import__("os").environ.get("THESEUS_TRIALS", "20"))
N_POINTS = 100
MAX_ITERS = 100
OUT = HERE / "results"


def instance_rng(p: float, trial: int) -> np.random.Generator:
    return np.random.default_rng(
        np.random.SeedSequence([20261127, 103, 0, round(100 * p), trial])
    )


def init_rng(p: float, trial: int) -> np.random.Generator:
    return np.random.default_rng(
        np.random.SeedSequence([20261127, 103, 2, round(100 * p), trial])
    )


def random_pose(rng):
    R = common.random_rotation(rng)
    t = rng.uniform(-1.0, 1.0, size=3)
    return R, t


def solve_theseus(x, y, R0, t0, c):
    obj, pose = build_gm_objective(x, y, R0, t0, c)
    solver = th.LevenbergMarquardt(
        obj, linear_solver_cls=th.CholeskyDenseSolver,
        max_iterations=MAX_ITERS, vectorize=False,
        abs_err_tolerance=1e-10, rel_err_tolerance=1e-8,
    )
    t0c = time.perf_counter()
    info = solver.optimize()
    dt = time.perf_counter() - t0c
    R, t = errors_from_pose(pose)
    iters = int(info.converged_iter.flatten()[0])
    status = int(info.status.flatten()[0].value)
    last_err = float(info.last_err.flatten()[0])
    return R, t, dt, iters, status, last_err


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    # Jacobian / objective self-check on a separate instance before measuring.
    rng = np.random.default_rng(0)
    xc = rng.uniform(-1, 1, size=(20, 3))
    yc = xc + 0.01 * rng.normal(size=(20, 3))
    chk = self_check(xc, yc, common.C)
    common.json_dump(OUT / "theseus_selfcheck.json", chk)
    print("self-check:", chk, flush=True)

    raw = open(OUT / "theseus_raw.jsonl", "w")
    summary = {}
    t_start = time.time()
    for p in P_LIST:
        cells = {m: dict(re=[], te=[], t=[], succ=[], iters=[], status=[])
                 for m in ("gt", "rand")}
        for trial in range(N_TRIALS):
            rng = instance_rng(p, trial)
            x, y, Rgt, tgt = common.make_instance(rng, p, n=N_POINTS,
                                                  outlier_type="random")
            rng_i = init_rng(p, trial)
            Rr, tr = random_pose(rng_i)
            for name, (R0, t0) in (("gt", (Rgt, tgt)), ("rand", (Rr, tr))):
                R, t, dt, iters, status, last_err = solve_theseus(
                    x, y, R0, t0, common.C)
                re, te = common.pose_errors(R, t, Rgt, tgt)
                s = common.success(re, te)
                c = cells[name]
                c["re"].append(re); c["te"].append(te); c["t"].append(dt)
                c["succ"].append(bool(s)); c["iters"].append(iters)
                c["status"].append(status)
                raw.write(json.dumps(dict(
                    p=p, trial=trial, init=name, rot_err_deg=re,
                    trans_err=te, success=bool(s), time_s=dt, iters=iters,
                    status=status, last_err=last_err,
                    init_rot_err_deg=common.rot_err_deg(R0, Rgt),
                    init_trans_err=float(np.linalg.norm(t0 - tgt)),
                )) + "\n")
        summary[str(p)] = {}
        for name, c in cells.items():
            s = common.summarize_errors(c["re"], c["te"], c["t"])
            s["iters_median"] = float(np.median(c["iters"]))
            s["n_max_iters"] = int(sum(1 for k in c["iters"] if k >= MAX_ITERS))
            s["n_fail_status"] = int(sum(1 for k in c["status"] if k < 0))
            s["time_s"] = dict(p50=float(np.median(c["t"])),
                               mean=float(np.mean(c["t"])),
                               p95=float(np.percentile(c["t"], 95)))
            summary[str(p)][name] = s
        print(f"p={p}: " + " | ".join(
            f"{k} succ={summary[str(p)][k]['success']:.2f} "
            f"med={summary[str(p)][k]['time_ms_p50']:.0f}ms" for k in cells),
            flush=True)
    raw.close()
    common.json_dump(OUT / "theseus.json", dict(
        protocol=dict(
            n_points=N_POINTS, n_trials=N_TRIALS, p_list=P_LIST,
            outlier_type="random", c=common.C, max_iters=MAX_ITERS,
            solver="LevenbergMarquardt + CholeskyDenseSolver, float64, CPU",
            theseus_version=th.__version__, torch_version=torch.__version__,
            init="gt | rand (uniform random rotation, t ~ U(-1,1)^3)",
            seed="data SeedSequence([20261127,103,0,100p,trial]); "
                 "init SeedSequence([20261127,103,2,100p,trial])",
            wall_time_s=time.time() - t_start,
            self_check=chk,
        ),
        cells=summary,
    ))
    print(f"done in {time.time() - t_start:.1f}s -> {OUT/'theseus.json'}")


if __name__ == "__main__":
    main()
