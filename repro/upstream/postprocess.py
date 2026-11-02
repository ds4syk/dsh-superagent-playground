"""Post-processing of the Sec. VI-B4 baseline-calibration runs.

Produces
  * Table V-style statistics (final baseline error avg/std, number of outer
    iterations, final outer loss) comparing Theseus-RND / Theseus-GT / SDPR-IS
    (and, if available, SDPR-SDP)  -> tableV.csv / tableV.json
  * Fig. 4-style parameter-error trajectories (50 runs per method)
    -> figures/fig4_baseline_param_err{,_zoom}.png
  * loss / gradient / inner-time diagnostics -> figures/fig4_*.png

The upstream plotting/statistics routines are reused where they are hard-coded
to the upstream output layout: a symlink ``_scripts/outputs/<fname>`` is created
so that ``stereo_cal.get_statistics`` and ``stereo_cal.baseline_param_plots`` run
unmodified and produce the exact figures the authors would have produced.
"""

import argparse
import json
import os
import pickle
import shutil

import numpy as np
import pandas as pd

import repro_common as rc


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--fname", default="str_tune_b0p003_sgd_20b_50r")
    p.add_argument("--b-gt", type=float, default=0.24)
    p.add_argument("--err-init", type=float, default=0.003)
    return p.parse_args()


def load_frames(folder, mode):
    path = os.path.join(folder, f"stereo_tune_{mode}.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def _param(df):
    return np.asarray([float(np.asarray(v).ravel()[0]) for v in df["params"].values])


def stats_for(frames, b_gt, n_failed_hint=None):
    """Same statistics as upstream `get_statistics` (which crashes on None)."""
    ok = [f for f in frames if f is not None]
    n_none = (len(frames) - len(ok)) if n_failed_hint is not None else 0
    if n_failed_hint is not None:
        n_none = n_failed_hint
    params = np.concatenate([_param(f)[-1:] - b_gt for f in ok])
    n_iters = np.asarray([len(f) for f in ok], dtype=float)
    loss = np.asarray([float(f["loss"].values[-1]) for f in ok])
    return dict(
        n_runs=len(ok) + n_none,
        n_ok=len(ok),
        n_failed=n_none,
        final_baseline_error_avg=float(np.mean(params)),
        final_baseline_error_std=float(np.std(params)),
        final_baseline_error_absavg=float(np.mean(np.abs(params))),
        n_outer_iters_avg=float(np.mean(n_iters)),
        n_outer_iters_std=float(np.std(n_iters)),
        outer_loss_avg=float(np.mean(loss)),
        mean_time_inner_s=float(np.mean([f["time_inner"].mean() for f in ok])),
        per_run_final_error=[float(x) for x in params],
    )


def make_upstream_symlink(folder, fname):
    """Point ``_scripts/outputs/<fname>`` at our result folder (read-only reuse)."""
    out_root = os.path.join(rc.REQ, "_scripts", "outputs")
    os.makedirs(out_root, exist_ok=True)
    link = os.path.join(out_root, fname)
    if os.path.islink(link) or os.path.exists(link):
        if os.path.islink(link):
            os.unlink(link)
        elif os.path.isfile(link):
            os.unlink(link)
        else:
            shutil.rmtree(link)
    os.symlink(folder, link)
    return link


def plot_fig4(folder, fname, b_gt, err_init):
    """Fig. 4: 50 parameter-error trajectories per method."""
    import matplotlib.pyplot as plt

    modes = [
        ("sdpr-dq", "SDPR-IS", "g"),
        ("theseus_gt", "Theseus-GT", "b"),
        ("theseus_rand", "Theseus-RND", "r"),
    ]
    data = {m: load_frames(folder, m) for m, _, _ in modes}

    def draw(ax, xlim=None, ylim=None):
        for mode, label, color in modes:
            frames = data[mode]
            if frames is None:
                continue
            for i, df in enumerate(frames):
                if df is None:
                    continue
                p = np.concatenate([[err_init], _param(df) - b_gt])
                ax.plot(
                    np.arange(len(p)),
                    p,
                    color=color,
                    alpha=0.35,
                    linewidth=0.6,
                    label=label if i == 0 else None,
                )
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Outer iteration")
        ax.set_ylabel("Baseline parameter error [m]")
        if xlim is not None:
            ax.set_xlim(*xlim)
        if ylim is not None:
            ax.set_ylim(*ylim)

    fig, ax = plt.subplots(figsize=(6, 4))
    draw(ax)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(rc.FIGS, "fig4_baseline_param_err.png"), dpi=300)
    fig, ax = plt.subplots(figsize=(6, 2.4))
    draw(ax, xlim=(0, 60), ylim=(-2e-3, 4e-3))
    ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(rc.FIGS, "fig4_baseline_param_err_zoom.png"), dpi=300)
    plt.close("all")


def plot_diagnostics(folder):
    """Loss / |grad| / inner-time curves (median + quartiles across runs)."""
    import matplotlib.pyplot as plt

    modes = [
        ("sdpr-dq", "SDPR-IS", "g"),
        ("theseus_gt", "Theseus-GT", "b"),
        ("theseus_rand", "Theseus-RND", "r"),
        ("spdr", "SDPR-SDP", "m"),
    ]
    for col, ylabel, scale, fname in [
        ("loss", "Outer loss", "log", "fig4_outer_loss"),
        ("grad_sq", "Gradient squared", "log", "fig4_grad_sq"),
        ("time_inner", "Inner time [s]", "log", "fig4_inner_time"),
    ]:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        for mode, label, color in modes:
            path = os.path.join(folder, f"curves_{mode}.csv")
            if not os.path.exists(path):
                continue
            df = pd.read_csv(path)
            g = df.groupby("iter")[col]
            med = g.median()
            lo = g.quantile(0.25)
            hi = g.quantile(0.75)
            ax.plot(med.index, med.values, color=color, label=label)
            ax.fill_between(med.index, lo.values, hi.values, color=color, alpha=0.2)
        ax.set_yscale(scale)
        ax.set_xlabel("Outer iteration")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(rc.FIGS, f"{fname}.png"), dpi=300)
        plt.close(fig)


def main():
    args = parse_args()
    os.makedirs(rc.FIGS, exist_ok=True)
    folder = os.path.join(rc.RESULTS, args.fname)
    cal = rc.load_stereo_cal()
    # upstream savefig writes into plot_tools.fig_dir (captured at import time)
    from sdprlayers.utils import plot_tools

    plot_tools.fig_dir = rc.FIGS

    # ---------------------------------------------------------------- Table V
    # `sdpr-dq` is SDPR-IS (Thm.2 implicit selections, compute_multipliers=False);
    # `spdr` is SDPR-SDP (differentiate through the SDP).  论文 Table V 的
    # "SDPR-IS" 对应 `sdpr-dq`；上游 get_statistics 也读的是 sdpr-dq。
    modes = {
        "SDPR-IS": "sdpr-dq",
        "SDPR-SDP": "spdr",
        "Theseus-RND": "theseus_rand",
        "Theseus-GT": "theseus_gt",
    }
    paper = {
        "Theseus-RND": dict(err_mean=1.132e-1, err_std=3.191e-2, iter_mean=8.034e1, loss_mean=1.384e2),
        "Theseus-GT": dict(err_mean=3.463e-4, err_std=4.901e-4, iter_mean=3.540e1, loss_mean=9.854e-3),
        "SDPR-IS": dict(err_mean=3.454e-4, err_std=4.907e-4, iter_mean=3.422e1, loss_mean=9.857e-3),
    }
    rows = []
    table = {}
    for label, mode in modes.items():
        frames = load_frames(folder, mode)
        if frames is None:
            print(f"[skip] no results for {mode}")
            continue
        summary_path = os.path.join(folder, f"summary_{mode}.json")
        n_failed_hint = None
        if os.path.exists(summary_path):
            import json as _json

            with open(summary_path) as f:
                _summ = _json.load(f)
            n_failed_hint = _summ.get("n_failed")
        st = stats_for(frames, args.b_gt, n_failed_hint=n_failed_hint)
        table[label] = st
        # per-run wall / rss / tightness from the merged summary json
        summary_path = os.path.join(folder, f"summary_{mode}.json")
        wall_mean = rss_peak = tw_frac = None
        if os.path.exists(summary_path):
            import json as _json

            with open(summary_path) as f:
                summ = _json.load(f)
            wall_mean = summ["mean"].get("wall_s")
            rss_peak = summ.get("rss_peak_mb")
            ta = summ.get("tightness_agg") or {}
            tw_frac = ta.get("rank1_frac_1e5")
        row = dict(
            method=label,
            mode=mode,
            n_runs=st["n_runs"],
            failures=st["n_failed"],
            err_mean=st["final_baseline_error_avg"],
            err_std=st["final_baseline_error_std"],
            iter_mean=st["n_outer_iters_avg"],
            iter_std=st["n_outer_iters_std"],
            loss_mean=st["outer_loss_avg"],
            loss_std=float(np.std([float(f["loss"].values[-1]) for f in frames if f is not None])),
            wall_mean_s=wall_mean,
            rss_peak_mb=rss_peak,
            tightness_frac_rank1=tw_frac,
            corank1_frac=None,  # filled from tightness.json when available
        )
        if label in paper:
            row.update(
                paper_err_mean=paper[label]["err_mean"],
                paper_err_std=paper[label]["err_std"],
                paper_iter_mean=paper[label]["iter_mean"],
                paper_loss_mean=paper[label]["loss_mean"],
            )
        rows.append(row)
    df = pd.DataFrame(rows).set_index("method")
    odir = folder
    os.makedirs(odir, exist_ok=True)
    df.to_csv(os.path.join(odir, "tableV_faithful.csv"))
    rc.write_json(os.path.join(odir, "tableV_faithful.json"), table)
    print("\n=== Table V (reproduction) ===")
    print(df.to_string(float_format=lambda x: f"{x:0.4e}"))

    # combined per-run trajectories: method,run,iteration,b,err,loss,grad_sq
    traj = []
    for label, mode in modes.items():
        path = os.path.join(folder, f"curves_{mode}.csv")
        if not os.path.exists(path):
            continue
        c = pd.read_csv(path)
        c.insert(0, "method", label)
        c = c.rename(columns={"iter": "iteration", "param": "b"})
        traj.append(c[["method", "run", "iteration", "b", "param_err", "loss", "grad_sq"]])
    if traj:
        pd.concat(traj).to_csv(os.path.join(odir, "trajectories_all.csv"), index=False)
        print(f"[info] wrote {os.path.join(odir, 'trajectories_all.csv')}")

    # ------------------------------------------- upstream get_statistics/plots
    link = make_upstream_symlink(folder, args.fname)
    print(f"[info] symlinked {link} -> {folder}")
    try:
        cal.get_statistics(fname=args.fname)
        cal.baseline_param_plots(fname=args.fname)
    except Exception as exc:  # theseus_rand may contain None frames
        print(f"[warn] upstream post-processing failed: {type(exc).__name__}: {exc}")
        print("[warn] our own figures are still produced below")

    # --------------------------------------------------------------- figures
    plot_fig4(folder, args.fname, args.b_gt, args.err_init)
    plot_diagnostics(folder)
    print(f"[done] figures in {rc.FIGS}")


if __name__ == "__main__":
    main()
