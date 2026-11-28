#!/usr/bin/env python3
"""SDPRLayers forward-only baseline on the same 3D-3D instances.

SDPRLayers (Holmes et al., arXiv:2405.19309) solves the matrix-weighted
3D-3D registration QCQP through its Shor SDP relaxation.  This script uses
the *upstream* ``SDPPoseEstimator`` (13-dim QCQP) with the licence-free SCS
backend (paper Sec. VI-B4 used SCS eps=1e-9) on exactly the same instance
generator / seeds as GNC / RANSAC / Theseus.

Important scope note: this is a **forward-only, non-robust** least-squares
objective (no GM/Cauchy kernel, weight = 1 for every correspondence).  It is
the right baseline for the "SDPR without RANSAC is outlier-sensitive" claim,
not a robust-kernel baseline.

Run with the prior SDPR env (numpy 1.26 / torch 2.5.1 / SCS 3.3.1) and the
patched fresh clone on PYTHONPATH::

  PYTHONPATH=/mnt/storage1/syk/third_party/sdprlayer/src \
    /home/syk/miniconda3/envs/sdpr/bin/python run_sdpr_forward.py

Outputs
-------
results/sdpr_forward_raw.jsonl  one record per (N,p,trial)
results/sdpr_forward.json       per-cell summary incl. SCS failure rate
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch

torch.set_default_dtype(torch.float64)
warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "synthetic"))
import common  # noqa: E402

from sdprlayers import SDPPoseEstimator  # noqa: E402

P_LIST = [0.0, 0.3, 0.5, 0.7]
N_LIST = [20, 50, 100]
TRIALS = {20: 20, 50: 12, 100: 20}
OUT = HERE / "results"
SCS_EPS = 1e-9


def instance_rng(p: float, trial: int, n: int) -> np.random.Generator:
    # Same data stream as the GNC/RANSAC/Theseus runs for N=100; for N=20/50
    # the N is added to the seed so the streams are distinct and reproducible.
    # N=100 must be bit-identical to the GNC/RANSAC/Theseus instance stream;
    # smaller N gets an extra stream key so seeds do not collide.
    key = [20261127, 103, 0, round(100 * p), trial]
    if n != 100:
        key.append(n)
    return np.random.default_rng(np.random.SeedSequence(key))


def to_tensor(x: np.ndarray, y: np.ndarray):
    def hom(a):
        t = torch.as_tensor(a, dtype=torch.float64).T  # (3,N)
        return torch.cat([t, torch.ones(1, t.shape[1], dtype=torch.float64)], 0)[None]
    return hom(x), hom(y)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    estimator = SDPPoseEstimator(
        T_s_v=torch.eye(4, dtype=torch.float64),
        diff_qcqp=False,
        solver="SCS",
        scs_eps=SCS_EPS,
    )
    raw = open(OUT / "sdpr_forward_raw.jsonl", "w")
    summary = {}
    t_start = time.time()
    with torch.no_grad():
        for n in N_LIST:
            cells = {p: dict(re=[], te=[], t=[], ok=[]) for p in P_LIST}
            for p in P_LIST:
                for trial in range(TRIALS[n]):
                    rng = instance_rng(p, trial, n)
                    x, y, Rgt, tgt = common.make_instance(
                        rng, p, n=n, outlier_type="random")
                    xb, yb = to_tensor(x, y)
                    wb = torch.ones(1, 1, n, dtype=torch.float64)
                    t0 = time.perf_counter()
                    err = None
                    try:
                        T = estimator(
                            keypoints_3D_src=xb, keypoints_3D_trg=yb, weights=wb)
                        dt = time.perf_counter() - t0
                        R = T[0, :3, :3].cpu().numpy()
                        t = T[0, :3, 3].cpu().numpy()
                        re, te = common.pose_errors(R, t, Rgt, tgt)
                        ok = True
                    except Exception as exc:  # non-tight / SCS failure
                        dt = time.perf_counter() - t0
                        re = te = float("nan")
                        ok = False
                        err = f"{type(exc).__name__}: {exc}"[:300]
                    rec = dict(n=n, p=p, trial=trial, ok=ok,
                               rot_err_deg=re, trans_err=te, time_s=dt,
                               error=err)
                    raw.write(json.dumps(rec) + "\n")
                    c = cells[p]
                    c["ok"].append(ok)
                    c["t"].append(dt)
                    if ok:
                        c["re"].append(re)
                        c["te"].append(te)
            summary[str(n)] = {}
            for p in P_LIST:
                c = cells[p]
                n_ok = int(sum(c["ok"]))
                n_tot = len(c["ok"])
                tt = np.asarray(c["t"], float)
                if n_ok:
                    re = np.asarray(c["re"], float)
                    te = np.asarray(c["te"], float)
                    cell = dict(
                        n_trials=n_tot, n_ok=n_ok,
                        scs_failure_rate=1.0 - n_ok / n_tot,
                        success=float(np.mean(
                            (re < common.SUCCESS_ROT_DEG)
                            & (te < common.SUCCESS_TRANS))),
                        success_over_attempts=float(
                            np.sum((re < common.SUCCESS_ROT_DEG)
                                   & (te < common.SUCCESS_TRANS)) / n_tot),
                        rot_median=float(np.median(re)),
                        rot_p90=float(np.percentile(re, 90)),
                        trans_median=float(np.median(te)),
                        trans_p90=float(np.percentile(te, 90)),
                        time_ms_p50=float(1000 * np.median(tt)),
                        time_ms_p95=float(1000 * np.percentile(tt, 95)),
                    )
                else:
                    cell = dict(n_trials=n_tot, n_ok=0,
                                scs_failure_rate=1.0, success=0.0,
                                success_over_attempts=0.0,
                                rot_median=float("nan"),
                                trans_median=float("nan"),
                                time_ms_p50=float(1000 * np.median(tt)),
                                time_ms_p95=float(1000 * np.percentile(tt, 95)))
                summary[str(n)][str(p)] = cell
                print(f"N={n} p={p}: ok={n_ok}/{n_tot} "
                      f"succ={cell['success']:.2f} "
                      f"p50={cell['time_ms_p50']:.1f}ms", flush=True)
    raw.close()
    common.json_dump(OUT / "sdpr_forward.json", dict(
        protocol=dict(
            method="SDPRLayers SDPPoseEstimator (13-dim QCQP), forward only",
            solver="SCS", scs_eps=SCS_EPS,
            diff_qcqp=False, weights="all ones (non-robust least squares)",
            n_points=N_LIST, trials=TRIALS, p_list=P_LIST,
            outlier_type="random", success="rot<5deg and trans<0.1m",
            sdpr_source=str(Path(__import__("sdprlayers").__file__).resolve()),
            sdpr_version=getattr(
                __import__("sdprlayers"), "__version__", "unknown"),
            torch_version=torch.__version__,
            scs_version=__import__("scs").__version__,
            wall_time_s=time.time() - t_start,
            note="failure = exception from forward (non-tight rank-1 recovery "
                 "or SCS solver error); success computed on returned poses "
                 "only, success_over_attempts also counts failures as misses",
        ),
        cells=summary,
    ))
    print(f"done in {time.time() - t_start:.1f}s -> {OUT/'sdpr_forward.json'}")


if __name__ == "__main__":
    main()
