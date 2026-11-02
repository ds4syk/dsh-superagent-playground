"""Run the full baseline-calibration experiment (paper Sec. VI-B4, Fig. 4 / Table V).

Usage examples
--------------
Single process (debugging)::

    python run_experiment.py --methods sdpr local_gt local_rnd --runs 2 --workers 1

Full experiment (50 runs x 20 poses x 3 methods, both noise models)::

    python run_experiment.py --runs 50 --workers 12

Outputs (under ``repro/results_clean``):

* ``raw/{method}_{noise}.pkl``   full histories (trajectories + diagnostics)
* ``tables/tableV_{noise}.csv``  Table-V style statistics
* ``trajectories_{noise}.csv``   long-format parameter-error trajectories (Fig. 4)
* ``run_summary_{noise}.csv``    one row per run
* ``timing_{noise}.json``        aggregated timings / failure counts
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from dataclasses import asdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bilevel  # noqa: E402

METHODS = ["sdpr", "local_gt", "local_rnd"]


# ----------------------------------------------------------------------
# worker
# ----------------------------------------------------------------------
def _run_one(args):
    """Run one trial; executed inside a worker process."""
    cfg_dict, method, run_idx = args
    cfg = bilevel.ExpConfig(**cfg_dict)
    t0 = time.perf_counter()
    data = bilevel.make_dataset(cfg, run_idx)
    t_data = time.perf_counter() - t0
    t0 = time.perf_counter()
    hist = bilevel.run_trial(data, method, cfg)
    t_trial = time.perf_counter() - t0
    hist["run"] = run_idx
    hist["method"] = method
    hist["wall_time"] = t_trial
    hist["data_time"] = t_data
    return run_idx, hist


def _copy_cfg(cfg: bilevel.ExpConfig, **kwargs) -> bilevel.ExpConfig:
    d = asdict(cfg)
    d.update(kwargs)
    return bilevel.ExpConfig(**d)


# ----------------------------------------------------------------------
# aggregation
# ----------------------------------------------------------------------
def summarise_run(hist, cfg):
    err = np.array(hist["err"])
    loss = np.array(hist["loss"])
    grad_sq = np.array(hist["grad_sq"])
    b_final = hist["b"][-1]
    out = dict(
        run=hist["run"],
        method=hist["method"],
        n_iter=hist["n_iter"],
        b_final=b_final,
        err_final=b_final - cfg.b_true,
        outer_loss_final=loss[-1],
        grad_sq_final=grad_sq[-1],
        wall_time=hist["wall_time"],
        data_time=hist["data_time"],
        time_per_iter=float(np.mean(hist["time_iter"])),
        converged=int(grad_sq[-1] <= cfg.tol_grad_sq),
    )
    if len(hist["diag"]) > 0:
        diag = hist["diag"]
        keys = diag[0].keys()
        for kk in keys:
            try:
                vals = np.array([np.mean(d[kk]) for d in diag])
            except (ValueError, TypeError):
                continue
            if vals.ndim == 1:
                out[f"diag_{kk}"] = float(np.mean(vals))
    return out


def table_v(histories, cfg):
    """Table-V style statistics (mean/std over runs), as in ``get_statistics``."""
    rows = []
    for method in METHODS:
        hs = [h for h in histories if h["method"] == method]
        if not hs:
            continue
        errs = np.array([h["b"][-1] - cfg.b_true for h in hs])
        iters = np.array([h["n_iter"] for h in hs])
        losses = np.array([h["loss"][-1] for h in hs])
        walls = np.array([h["wall_time"] for h in hs])
        rows.append(
            dict(
                method=method,
                n_runs=len(hs),
                final_baseline_error_mean=float(np.mean(errs)),
                final_baseline_error_std=float(np.std(errs)),
                final_baseline_error_abs_mean=float(np.mean(np.abs(errs))),
                n_iter_mean=float(np.mean(iters)),
                n_iter_std=float(np.std(iters)),
                outer_loss_mean=float(np.mean(losses)),
                outer_loss_std=float(np.std(losses)),
                wall_time_mean=float(np.mean(walls)),
                wall_time_std=float(np.std(walls)),
            )
        )
    return rows


def trajectory_rows(histories, cfg):
    rows = []
    for h in histories:
        rows.append(
            dict(
                method=h["method"],
                run=h["run"],
                iteration=0,
                b=cfg.b_true + cfg.b_offset,
                err=cfg.b_offset,
                loss="",
                grad_sq="",
            )
        )
        for i in range(h["n_iter"]):
            rows.append(
                dict(
                    method=h["method"],
                    run=h["run"],
                    iteration=i,
                    b=h["b"][i],
                    err=h["err"][i],
                    loss=h["loss"][i],
                    grad_sq=h["grad_sq"][i],
                )
            )
    return rows


def timing_summary(histories, cfg):
    out = {"config": asdict(cfg), "methods": {}}
    for method in METHODS:
        hs = [h for h in histories if h["method"] == method]
        if not hs:
            continue
        walls = np.array([h["wall_time"] for h in hs])
        iters = np.array([h["n_iter"] for h in hs])
        per_iter = np.array([np.mean(h["time_iter"]) for h in hs])
        d = dict(
            n_runs=len(hs),
            wall_time_total_s=float(np.sum(walls)),
            wall_time_mean_s=float(np.mean(walls)),
            wall_time_std_s=float(np.std(walls)),
            n_iter_mean=float(np.mean(iters)),
            time_per_outer_iter_mean_s=float(np.mean(per_iter)),
            n_nonconverged=int(np.sum([h["grad_sq"][-1] > cfg.tol_grad_sq for h in hs])),
        )
        diag = [d_ for h in hs for d_ in h["diag"]]
        if diag:
            for kk in [
                "t_solve", "t_backward", "r_tight", "corank_H",
                "corank_H_strict", "eig2_eig1", "H_min_eig",
                "Hx_res", "orth_err", "primal", "dual", "n_retry",
                "fallback_H", "n_gn_iter", "cond", "g_norm", "converged",
                "cost",
            ]:
                if kk not in diag[0]:
                    continue
                vals = np.concatenate([np.ravel(d_[kk]) for d_ in diag])
                d[f"{kk}_mean"] = float(np.mean(vals))
                d[f"{kk}_std"] = float(np.std(vals))
                if kk in ("n_retry", "fallback_H", "corank_H", "corank_H_strict"):
                    d[f"{kk}_total"] = int(np.sum(vals))
                    d[f"{kk}_max"] = int(np.max(vals))
                if kk == "r_tight":
                    d["r_tight_min"] = float(np.min(vals))
                    d["r_tight_frac_above_1e5"] = float(np.mean(vals > 1e5))
        out["methods"][method] = d
    return out


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=20)
    ap.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    ap.add_argument("--noise-modes", nargs="+", default=["independent", "upstream"])
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--max-iter", type=int, default=150)
    ap.add_argument("--tol-grad-sq", type=float, default=1e-5)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--sdp-eps", type=float, default=1e-9)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--tag", default="")
    args = ap.parse_args(argv)

    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = args.out_dir or os.path.join(here, "..", "results_clean")
    out_dir = os.path.abspath(out_dir)
    for sub in ("raw", "tables", "logs"):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)

    base_cfg = bilevel.ExpConfig(
        n_batch=args.batch,
        n_runs=args.runs,
        seed0=args.seed0,
        max_iter=args.max_iter,
        tol_grad_sq=args.tol_grad_sq,
        lr=args.lr,
        sdp_eps=args.sdp_eps,
    )

    all_histories = []
    for noise_mode in args.noise_modes:
        cfg = _copy_cfg(base_cfg, noise_mode=noise_mode)
        print(f"=== noise_mode={noise_mode}: {args.runs} runs x {args.batch} poses "
              f"x {args.methods}, {args.workers} workers ===", flush=True)
        method_histories = []
        for method in args.methods:
            t0 = time.perf_counter()
            tasks = [(asdict(cfg), method, r) for r in range(args.runs)]
            results = []
            if args.workers <= 1:
                for task in tasks:
                    results.append(_run_one(task))
                    h = results[-1][1]
                    print(f"  {method} run {h['run']:3d}: iters={h['n_iter']:3d} "
                          f"err={h['err'][-1]:+.3e} loss={h['loss'][-1]:.3e} "
                          f"wall={h['wall_time']:.2f}s", flush=True)
            else:
                import multiprocessing as mp
                ctx = mp.get_context("fork")
                with ctx.Pool(args.workers) as pool:
                    for idx, (_r, h) in enumerate(pool.imap_unordered(_run_one, tasks)):
                        method_histories.append(h)
                        print(f"  {method} run {h['run']:3d} done "
                              f"({idx+1}/{args.runs}): iters={h['n_iter']:3d} "
                              f"err={h['err'][-1]:+.3e} wall={h['wall_time']:.2f}s",
                              flush=True)
                    results = None
            if results is not None:
                method_histories.extend([h for _, h in results])
            t_elapsed = time.perf_counter() - t0
            print(f"  -> {method}: {t_elapsed:.1f}s wall "
                  f"(sum of trial times {sum(h['wall_time'] for h in method_histories):.1f}s)",
                  flush=True)
        # sort for reproducibility
        method_histories.sort(key=lambda h: (h["method"], h["run"]))
        all_histories.extend(method_histories)

        tag = f"_{args.tag}" if args.tag else ""
        with open(os.path.join(out_dir, "raw", f"histories_{noise_mode}{tag}.pkl"), "wb") as f:
            pickle.dump(dict(histories=method_histories, config=asdict(cfg)), f)

        # tables -----------------------------------------------------
        import csv

        tv = table_v(method_histories, cfg)
        with open(os.path.join(out_dir, "tables", f"tableV_{noise_mode}{tag}.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(tv[0].keys()))
            w.writeheader()
            w.writerows(tv)
        with open(os.path.join(out_dir, "tables", f"tableV_{noise_mode}{tag}.json"), "w") as f:
            json.dump(tv, f, indent=2)

        runs = [summarise_run(h, cfg) for h in method_histories]
        with open(os.path.join(out_dir, f"run_summary_{noise_mode}{tag}.csv"), "w", newline="") as f:
            keys = sorted({k for r in runs for k in r})
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(runs)

        rows = trajectory_rows(method_histories, cfg)
        with open(os.path.join(out_dir, f"trajectories_{noise_mode}{tag}.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["method", "run", "iteration", "b", "err", "loss", "grad_sq"])
            w.writeheader()
            w.writerows(rows)

        timing = timing_summary(method_histories, cfg)
        with open(os.path.join(out_dir, f"timing_{noise_mode}{tag}.json"), "w") as f:
            json.dump(timing, f, indent=2)
        print(json.dumps({k: {kk: vv for kk, vv in v.items() if not isinstance(vv, str)}
                          for k, v in timing["methods"].items()}, indent=2)[:4000])
    print("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
