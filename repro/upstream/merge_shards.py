"""Merge sharded calibration runs into the upstream-compatible result layout.

After running ``run_calibration.py --mode M --n-runs K --shard i --n-shards S``
for every shard, this script rebuilds

    <folder>/stereo_tune_<mode>.pkl     # list[DataFrame|None], run order 0..K-1
    <folder>/curves_<mode>.csv          # long-format trajectories (Fig. 4)
    <folder>/summary_<mode>.json        # aggregate statistics

so that ``postprocess.py`` (and the upstream get_statistics /
baseline_param_plots) can consume them unchanged.
"""

import argparse
import json
import os
import pickle

import numpy as np
import pandas as pd

import repro_common as rc


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True)
    p.add_argument("--n-runs", type=int, default=50)
    p.add_argument("--fname", default=None)
    p.add_argument("--b-gt", type=float, default=0.24)
    return p.parse_args()


def main():
    args = parse_args()
    folder = os.path.join(
        rc.RESULTS, args.fname or f"str_tune_b0p003_sgd_20b_{args.n_runs}r"
    )
    shard_dir = os.path.join(folder, "shards")
    frames = [None] * args.n_runs
    records = []
    shard_files = sorted(
        f for f in os.listdir(shard_dir) if f.startswith("shard_") and f.endswith(f"_{args.mode}.json")
    )
    for sf in shard_files:
        with open(os.path.join(shard_dir, sf)) as f:
            shard = json.load(f)
        records.extend(shard["records"])
    records.sort(key=lambda r: r["run"])
    # effective number of runs = one past the largest recorded run index (a
    # sharded run may have been stopped early; do not pad with fake failures)
    if records:
        n_runs = max(r["run"] for r in records) + 1
    else:
        n_runs = args.n_runs
    frames = frames[:n_runs]
    for r in records:
        i = r["run"]
        path = os.path.join(shard_dir, f"run_{i:03d}_{args.mode}.pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                frames[i] = pickle.load(f)
    with open(os.path.join(folder, f"stereo_tune_{args.mode}.pkl"), "wb") as f:
        pickle.dump(frames, f)

    rows = []
    for i, df in enumerate(frames):
        if df is None:
            continue
        params = np.asarray([float(np.asarray(v).ravel()[0]) for v in df["params"].values])
        for k in range(len(df)):
            rows.append(
                dict(
                    run=i,
                    iter=k,
                    param=float(params[k]),
                    param_err=float(params[k] - args.b_gt),
                    loss=float(df["loss"].values[k]),
                    grad_sq=float(df["grad_sq"].values[k]),
                    time_inner=float(df["time_inner"].values[k]),
                )
            )
    pd.DataFrame(rows).to_csv(
        os.path.join(folder, f"curves_{args.mode}.csv"), index=False
    )

    ok = [r for r in records if r["ok"]]
    n_attempts = len(records)
    n_fail = n_attempts - len(ok)
    summary = dict(
        mode=args.mode,
        n_runs=n_attempts,
        n_slots=n_runs,
        n_ok=len(ok),
        n_failed=n_fail,
        fail_rate=n_fail / max(n_attempts, 1),
        total_wall_s=float(sum(r["wall_s"] for r in ok)),
        mean=dict(
            final_baseline_error=float(np.mean([r["final_baseline_error"] for r in ok]))
            if ok
            else None,
            n_outer_iters=float(np.mean([r["n_outer_iters"] for r in ok])) if ok else None,
            final_outer_loss=float(np.mean([r["final_loss"] for r in ok])) if ok else None,
            mean_time_inner_s=float(np.mean([r["mean_time_inner_s"] for r in ok]))
            if ok
            else None,
            wall_s=float(np.mean([r["wall_s"] for r in ok])) if ok else None,
        ),
        std=dict(
            final_baseline_error=float(np.std([r["final_baseline_error"] for r in ok]))
            if ok
            else None,
        ),
        rss_peak_mb=float(max(r["rss_peak_mb"] for r in records)) if records else None,
        records=records,
    )
    tight = [r["tightness"] for r in records if r.get("tightness", {}).get("n")]
    if tight:
        summary["tightness_agg"] = dict(
            n_solves=int(sum(t["n"] for t in tight)),
            eig_ratio_median=float(np.median([t["eig_ratio_median"] for t in tight])),
            rank1_frac_1e5=float(np.mean([t["rank1_frac_1e5"] for t in tight])),
            rank1_frac_1e8=float(np.mean([t["rank1_frac_1e8"] for t in tight])),
            rank1_frac_1e10=float(np.mean([t["rank1_frac_1e10"] for t in tight])),
            spe_median=float(np.median([t["spe_median"] for t in tight])),
        )
    rc.write_json(os.path.join(folder, f"summary_{args.mode}.json"), summary)
    print(
        f"[merge {args.mode}] {len(ok)}/{n_attempts} ok; "
        f"mean err={summary['mean']['final_baseline_error']:.4e} "
        f"std={summary['std']['final_baseline_error']:.4e} "
        f"iters={summary['mean']['n_outer_iters']:.2f} "
        f"loss={summary['mean']['final_outer_loss']:.4e}"
    )


if __name__ == "__main__":
    main()
