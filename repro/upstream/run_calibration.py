"""Run the Sec. VI-B4 baseline-calibration experiment (Fig. 4 / Table V).

Mirrors ``_scripts/stereo_cal.py::compare_tune_baseline`` exactly (same data
generator, same hyper-parameters, same tuner entry point ``tune_baseline``) but
saves into ``repro/results_upstream``, records resource/tightness statistics and
supports run sharding for parallel execution:

  # 1) generate the 50 problem instances once
  python run_calibration.py --mode prob_data --n-runs 50
  # 2) run e.g. 5 shards of 10 runs each (indices i with i % n_shards == shard)
  python run_calibration.py --mode sdpr-dq --n-runs 50 --shard 0 --n-shards 5
  ...
  # 3) merge shards into the upstream-compatible pkl layout
  python merge_shards.py --mode sdpr-dq --n-runs 50

Modes (upstream names):
  prob_data    generate the N_runs problem instances (N_batch poses each)
  spdr         SDPR-SDP   (differentiate through the SDP, CVXPYLayers)
  sdpr-dq      SDPR-IS    (implicit selections / Thm.2 + CIFT fallback)
  theseus_gt   Theseus, Gauss-Newton initialised at ground truth
  theseus_rand Theseus, random initialisation on a 3 m ball
"""

import argparse
import os
import pickle
import time

import numpy as np
import torch

import repro_common as rc

# exactly upstream compare_tune_baseline()
TERM_CRIT = {"max_iter": 150, "tol_grad_sq": 1e-5, "tol_loss": 1e-10}
OFFSET = 0.003  # initial baseline error (m), paper Sec. VI-B4
RADIUS = 3.0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--mode",
        default="sdpr-dq",
        choices=["prob_data", "spdr", "sdpr-dq", "theseus_gt", "theseus_rand"],
    )
    p.add_argument("--n-batch", type=int, default=20, help="poses per instance")
    p.add_argument("--n-runs", type=int, default=1, help="total runs (all shards)")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--n-shards", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default=None, help="override result folder name")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument(
        "--solver",
        default="SCS",
        help="SDP backend for the SDPR modes (SCS; MOSEK unavailable w/o license)",
    )
    return p.parse_args()


def folder_name(n_batch, n_runs):
    return f"str_tune_b0p003_sgd_{n_batch}b_{n_runs}r"


def _scalar(x):
    a = np.asarray(x).ravel()
    return float(a[0])


def run_indices(n_runs, shard, n_shards):
    return [i for i in range(n_runs) if i % n_shards == shard]


def run_prob_data(args, folder):
    rc.setup_env(args.threads)
    cal = rc.load_stereo_cal()
    cal.set_seed(args.seed)
    info = []
    for i in range(args.n_runs):
        t0 = time.perf_counter()
        prob_data = cal.get_cal_data(
            radius=RADIUS,
            board_dims=[1.0, 1.0],
            N_squares=[8, 8],
            N_batch=args.n_batch,
            setup="cone",
            plot=False,
        )
        info.append(tuple(prob_data))
        print(
            f"[prob_data] run {i + 1}/{args.n_runs} generated in "
            f"{time.perf_counter() - t0:.2f}s",
            flush=True,
        )
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "stereo_tune_prob_data.pkl"), "wb") as f:
        pickle.dump(info, f)
    rc.write_json(
        os.path.join(folder, "prob_data_meta.json"),
        dict(
            n_batch=args.n_batch,
            n_runs=args.n_runs,
            seed=args.seed,
            radius=RADIUS,
            board_dims=[1.0, 1.0],
            N_squares=[8, 8],
            setup="cone",
            n_features=int(np.asarray(info[0][2]).shape[2]),
            host=rc.host_info(),
        ),
    )


MODE_TO_TUNER = {
    "spdr": ("spdr", dict(gt_init=False)),
    "sdpr-dq": ("sdpr-dq", dict(gt_init=False)),
    "theseus_gt": ("theseus", dict(gt_init=True)),
    "theseus_rand": ("theseus", dict(gt_init=False)),
}


def run_tuner(args, folder):
    rc.setup_env(args.threads)
    rc.install_hooks()
    if args.mode.startswith("theseus"):
        # theseus-ai 0.2.1 needs an import-only sksparse stub and a
        # torch>=2.4 broadcasting fix (see theseus_compat.py)
        import theseus_compat

        theseus_compat.install()
    cal = rc.load_stereo_cal()
    tuner, kwargs = MODE_TO_TUNER[args.mode]
    prob_path = os.path.join(folder, "stereo_tune_prob_data.pkl")
    with open(prob_path, "rb") as f:
        info_p = pickle.load(f)
    assert len(info_p) >= args.n_runs, (
        f"prob_data has {len(info_p)} instances but N_runs={args.n_runs}"
    )

    shard_dir = os.path.join(folder, "shards")
    os.makedirs(shard_dir, exist_ok=True)
    idxs = run_indices(args.n_runs, args.shard, args.n_shards)
    shard_summary_path = os.path.join(
        shard_dir, f"shard_{args.shard}_{args.mode}.json"
    )
    records, t_all = [], time.perf_counter()
    for i in idxs:
        # deterministic per-run seed (upstream is unseeded; documented deviation)
        cal.set_seed(1000 * args.seed + i)
        rc.reset_recorder()
        t0 = time.perf_counter()
        err = None
        df = None
        relaxed_tol = False

        def _run():
            return cal.tune_baseline(
                tuner,
                opt_select="sgd",
                b_offs=OFFSET,
                N_batch=args.n_batch,
                radius=RADIUS,
                prob_data=info_p[i],
                term_crit=TERM_CRIT,
                **kwargs,
            )

        try:
            df = _run()
        except Exception as exc:  # theseus_rand can diverge (upstream try/except)
            err = f"{type(exc).__name__}: {exc}"
            # ---- adaptive fallback (solve-failure fix, report §5.6) --------------
            # The strict KKT residual acceptance (1e-5) fails on ~28% of SDPR-IS
            # instances with the SCS backend. Retry once with the validated
            # relaxed tolerance 1e-3 (FD gradient error 0.2-1.7%).  The faithful
            # default stays strict, so successful strict runs are unaffected.
            if args.mode in ("sdpr-dq", "spdr") and os.environ.get("SDPR_KKT_TOL") is None:
                rc.setup_env(threads=int(os.environ.get("SLURM_CPUS_PER_TASK", "2")))
                cal.set_seed(1000 * args.seed + i)
                rc.reset_recorder()
                os.environ["SDPR_KKT_TOL"] = "1e-3"
                try:
                    df = _run()
                    relaxed_tol = True
                    err = None
                except Exception as exc2:  # noqa: BLE001
                    err = f"{err} | relaxed-retry: {type(exc2).__name__}: {exc2}"
                finally:
                    os.environ.pop("SDPR_KKT_TOL", None)
        wall = time.perf_counter() - t0
        snap = rc.snapshot_recorder()
        rec = dict(
            mode=args.mode,
            run=i,
            shard=args.shard,
            ok=df is not None,
            error=err,
            relaxed_tol=relaxed_tol,
            wall_s=wall,
            rss_peak_mb=rc.rss_mb(),
            tightness=rc.summarise_tightness(snap),
        )
        if df is not None:
            params = np.asarray([_scalar(v) for v in df["params"].values])
            rec.update(
                n_outer_iters=int(len(df)),
                final_param=float(params[-1]),
                final_baseline_error=float(params[-1] - cal.cam_gt.b),
                init_baseline_error=float(params[0] - cal.cam_gt.b),
                final_loss=float(df["loss"].values[-1]),
                final_grad_sq=float(df["grad_sq"].values[-1]),
                mean_time_inner_s=float(np.mean(df["time_inner"].values)),
                sum_time_inner_s=float(np.sum(df["time_inner"].values)),
                max_grad_sq=float(np.max(df["grad_sq"].values)),
            )
            with open(
                os.path.join(shard_dir, f"run_{i:03d}_{args.mode}.pkl"), "wb"
            ) as f:
                pickle.dump(df, f)
        records.append(rec)
        # incremental dump (survive crashes)
        rc.write_json(
            shard_summary_path,
            dict(
                mode=args.mode,
                shard=args.shard,
                n_shards=args.n_shards,
                n_runs=args.n_runs,
                solver=args.solver,
                term_crit=TERM_CRIT,
                host=rc.host_info(),
                records=records,
            ),
        )
        msg = (
            f"[{args.mode} s{args.shard}] run {i} "
            f"{'OK' if df is not None else 'FAILED: ' + str(err)} "
            f"wall={wall:.1f}s"
        )
        if df is not None:
            msg += (
                f" iters={len(df)} fin_err={rec['final_baseline_error']:.3e}"
                f" loss={rec['final_loss']:.3e}"
            )
        print(msg, flush=True)

    print(
        f"[{args.mode} s{args.shard}] done {len(idxs)} runs in "
        f"{time.perf_counter() - t_all:.1f}s",
        flush=True,
    )


def main():
    args = parse_args()
    folder = os.path.join(rc.RESULTS, args.tag or folder_name(args.n_batch, args.n_runs))
    os.makedirs(folder, exist_ok=True)
    if args.mode == "prob_data":
        run_prob_data(args, folder)
    else:
        run_tuner(args, folder)


if __name__ == "__main__":
    main()
