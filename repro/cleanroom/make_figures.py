"""Figures for the clean-room reproduction of Sec. VI-B4 (Fig. 4 / Table V).

Produces, for each noise model:

* ``fig4_baseline_trajectories_{noise}.png`` -- baseline parameter error vs
  outer iteration for all 50 trials of the three methods (paper Fig. 4), with
  a zoomed panel matching the upstream plotting range,
* ``fig4_zoom_{noise}.png`` -- zoom only,
* ``fig_tableV_bars_{noise}.png`` -- final baseline error / iterations /
  outer loss per method,
* ``fig_tightness_{noise}.png`` -- SDP tightness statistics (eigenvalue
  ratio, certificate corank, minimum certificate eigenvalue, solve and
  backward times),
* ``fig3b_local_minima.png`` -- the paper's Fig. 3b analog: 100 random
  initialisations of the local solver, colour coded by convergence to the
  global optimum or to a local minimum.

Usage::

    python make_figures.py
"""

from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.abspath(os.path.join(HERE, "..", "results_clean"))
FIG = os.path.join(RES, "figures")

COLORS = {"sdpr": "tab:green", "local_gt": "tab:blue", "local_rnd": "tab:red"}
LABELS = {
    "sdpr": "SDPR-IS",
    "local_gt": "local-GT (Theseus-GT analog)",
    "local_rnd": "local-RND (Theseus-RND analog)",
}
ORDER = ["local_rnd", "local_gt", "sdpr"]
SHORT = {"sdpr": "SDPR-IS", "local_gt": "local-GT", "local_rnd": "local-RND"}


def _style(method):
    """Line style; local-GT is drawn dashed so that its near-perfect overlap
    with SDPR-IS (identical trajectories to ~1e-6 m) remains visible."""
    if method == "sdpr":
        return dict(alpha=0.45, lw=2.2, ls="-")
    if method == "local_gt":
        return dict(alpha=0.9, lw=0.8, ls="--")
    return dict(alpha=0.45, lw=0.9, ls="-")


def ensure_dirs():
    os.makedirs(FIG, exist_ok=True)


# ----------------------------------------------------------------------
# Fig. 4: parameter-error trajectories
# ----------------------------------------------------------------------
def fig4(noise="independent"):
    path = os.path.join(RES, f"trajectories_{noise}.csv")
    if not os.path.exists(path):
        print(f"skip fig4 {noise}: {path} missing")
        return
    df = pd.read_csv(path)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, (xlim, ylim, title) in zip(
        axs,
        [
            (None, None, "baseline parameter error (all trials)"),
            ((0, 70), (-2e-3, 4e-3), "zoom (paper Fig. 4 range)"),
        ],
    ):
        for method in ORDER:
            sub = df[df["method"] == method]
            for run, g in sub.groupby("run"):
                g = g.sort_values("iteration")
                style = _style(method)
                ax.plot(
                    g["iteration"].values,
                    g["err"].values,
                    color=COLORS[method],
                    alpha=style["alpha"],
                    linewidth=style["lw"],
                    linestyle=style["ls"],
                    label=LABELS[method] if run == sub["run"].min() else "_nolegend_",
                )
        ax.axhline(0.0, color="k", linestyle="--", linewidth=0.8)
        ax.set_xlabel("outer iteration")
        ax.set_ylabel(r"$b - b_{\mathrm{true}}$  [m]")
        ax.set_title(f"{title} ({noise} noise)", fontsize=10)
        ax.grid(True, alpha=0.3)
        if xlim is not None:
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
        ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    out = os.path.join(FIG, f"fig4_baseline_trajectories_{noise}.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("wrote", out)

    # zoom-only figure
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for method in ORDER:
        sub = df[df["method"] == method]
        for run, g in sub.groupby("run"):
            g = g.sort_values("iteration")
            style = _style(method)
            ax.plot(g["iteration"].values, g["err"].values,
                    color=COLORS[method], alpha=style["alpha"],
                    linewidth=style["lw"], linestyle=style["ls"],
                    label=LABELS[method] if run == sub["run"].min() else "_nolegend_")
    ax.axhline(0.0, color="k", linestyle="--", linewidth=0.8)
    ax.set_xlim(0, 70)
    ax.set_ylim(-2e-3, 4e-3)
    ax.set_xlabel("outer iteration")
    ax.set_ylabel(r"$b - b_{\mathrm{true}}$  [m]")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    out = os.path.join(FIG, f"fig4_zoom_{noise}.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("wrote", out)


# ----------------------------------------------------------------------
# Table V bar chart
# ----------------------------------------------------------------------
def fig_tableV(noise="independent"):
    path = os.path.join(RES, "tables", f"tableV_{noise}.json")
    if not os.path.exists(path):
        print(f"skip tableV fig {noise}: {path} missing")
        return
    tv = json.load(open(path))
    by = {r["method"]: r for r in tv}
    methods = [m for m in ["sdpr", "local_gt", "local_rnd"] if m in by]
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.4))
    for ax, key, name, logy in [
        (axs[0], "final_baseline_error_mean", "final |b - b_true| (mean)", True),
        (axs[1], "n_iter_mean", "outer iterations (mean)", False),
        (axs[2], "outer_loss_mean", "outer loss (mean)", True),
    ]:
        vals = [by[m][key] for m in methods]
        errs = [by[m][key.replace("_mean", "_std")] for m in methods]
        ax.bar([LABELS[m] for m in methods], vals, yerr=errs,
               color=[COLORS[m] for m in methods], alpha=0.8, capsize=4)
        ax.set_title(name, fontsize=9)
        if logy:
            ax.set_yscale("log")
        ax.tick_params(axis="x", labelrotation=20, labelsize=7)
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle(f"Table V analog ({noise} noise model)", fontsize=10)
    fig.tight_layout()
    out = os.path.join(FIG, f"fig_tableV_bars_{noise}.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("wrote", out)


# ----------------------------------------------------------------------
# tightness / timing diagnostics
# ----------------------------------------------------------------------
def fig_tightness(noise="independent"):
    path = os.path.join(RES, "raw", f"histories_{noise}.pkl")
    if not os.path.exists(path):
        print(f"skip tightness fig {noise}: {path} missing")
        return
    import pickle

    d = pickle.load(open(path, "rb"))
    hist = [h for h in d["histories"] if h["method"] == "sdpr"]
    if not hist:
        return
    r_tight, corank, hmin, tsolve, tback, eig2 = [], [], [], [], [], []
    for h in hist:
        for diag in h["diag"]:
            r_tight.append(np.ravel(diag["r_tight"]))
            corank.append(np.ravel(diag["corank_H"]))
            hmin.append(np.ravel(diag["H_min_eig"]))
            tsolve.append(np.ravel(diag["t_solve"]))
            tback.append(np.ravel(diag["t_backward"]))
            eig2.append(np.ravel(diag["eig2_eig1"]))
    r_tight = np.concatenate(r_tight)
    corank = np.concatenate(corank)
    hmin = np.concatenate(hmin)
    tsolve = np.concatenate(tsolve) * 1e3
    tback = np.concatenate(tback) * 1e3
    eig2 = np.concatenate(eig2)
    # the per-iteration diagnostic stored by the main run used an older
    # (incorrect) index for the second eigenvalue; use the dedicated
    # certificate sample when available
    cert_path = os.path.join(RES, "certificate_sample.json")
    if os.path.exists(cert_path):
        _cs = json.load(open(cert_path))
        eig2 = np.array([r["eig2_ratio"] for r in _cs["rows"]])

    fig, axs = plt.subplots(2, 3, figsize=(13, 6.5))
    axs = axs.ravel()
    axs[0].hist(np.log10(r_tight), bins=60, color="tab:green")
    axs[0].set_xlabel(r"$\log_{10}(\lambda_1/\lambda_2)$ of $X^*$")
    axs[0].set_ylabel("solves")
    axs[0].set_title(f"rank-1 tightness (frac>1e5: {np.mean(r_tight > 1e5):.3f})",
                     fontsize=9)
    axs[1].hist(corank, bins=np.arange(-0.5, 5.5), color="tab:orange")
    axs[1].set_xlabel("corank(H) @1e-6 relative")
    axs[1].set_title(f"corank=1 in {np.mean(corank == 1)*100:.2f}% of solves",
                     fontsize=9)
    axs[2].hist(np.log10(np.maximum(eig2, 1e-16)), bins=40, color="tab:purple")
    axs[2].set_xlabel(r"$\log_{10}(|\lambda_2(H)|/|\lambda_1(H)|)$")
    axs[2].set_title(f"2nd-smallest eigenvalue of H (median "
                     f"{np.median(eig2):.1e})", fontsize=9)
    axs[3].hist(np.clip(hmin, -1e-6, 1e-6) * 1e9, bins=60, color="tab:red")
    axs[3].set_xlabel(r"$\lambda_{\min}(H)$  [1e-9]")
    axs[3].set_title("certificate PSD", fontsize=9)
    axs[4].hist(np.clip(tsolve, 0, np.percentile(tsolve, 99.5)), bins=60,
                color="tab:blue")
    axs[4].set_xlabel("SDP solve + recovery [ms]")
    axs[4].set_title(f"median {np.median(tsolve):.2f} ms", fontsize=9)
    axs[5].hist(np.clip(tback, 0, np.percentile(tback, 99.5)), bins=60,
                color="tab:cyan")
    axs[5].set_xlabel("implicit backward [ms]")
    axs[5].set_title(f"median {np.median(tback):.3f} ms", fontsize=9)
    for ax in axs:
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"SDPR-IS inner-solve diagnostics ({noise} noise, "
                 f"{len(r_tight)} solves)")
    fig.tight_layout()
    out = os.path.join(FIG, f"fig_tightness_{noise}.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("wrote", out)


# ----------------------------------------------------------------------
# Fig. 3b: local minima visualisation
# ----------------------------------------------------------------------
def _plot_poses(R_cw, t_cw_w, ax, color="k", alpha=1.0, linewidth=1.0, scale=0.35):
    """Plot camera frames (rows of R_cw are the camera axes in the world)."""
    for i in range(len(R_cw)):
        origin = np.asarray(t_cw_w[i]).reshape(3)
        dirs = np.asarray(R_cw[i]).T       # columns = camera axes in world
        for j in range(3):
            ax.quiver(*origin, *(scale * dirs[:, j]), color=color, alpha=alpha,
                      linewidth=linewidth)


def fig3b(noise="independent"):
    path = os.path.join(RES, f"local_minima_dataset0_{noise}.npz")
    if not os.path.exists(path):
        print(f"skip fig3b: {path} missing")
        return
    d = np.load(path)
    r_init, C_init = d["r_init"], d["C_init"]
    r_final, C_final = d["r_final"], d["C_final"]
    is_local, is_global = d["is_local"].astype(bool), d["is_global"].astype(bool)
    C_gt, r_gt = d["C_gt"], d["r_gt"].reshape(3)

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(*d["m"], marker="*", color="k", s=8, label="features")
    _plot_poses(C_gt[None], r_gt[None], ax, color="magenta", linewidth=2.5, scale=0.5)
    _plot_poses(C_init[is_local], r_init[is_local], ax, color="tab:red", alpha=0.35)
    _plot_poses(C_init[is_global], r_init[is_global], ax, color="tab:green", alpha=0.35)
    _plot_poses(C_final[is_global], r_final[is_global], ax, color="tab:green",
                alpha=0.8, scale=0.5)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    r = 3.4
    ax.set_xlim(-r, r)
    ax.set_ylim(-r, r)
    ax.set_zlim(-r, r)
    frac = float(np.mean(is_local))
    ax.set_title(
        f"Fig. 3b analog: {len(r_init)} random initialisations "
        f"({int(is_local.sum())} local minima, {100*frac:.0f}%)")
    from matplotlib.lines import Line2D

    handles = [
        Line2D([0], [0], color="tab:green", label="init -> global minimum"),
        Line2D([0], [0], color="tab:red", label="init -> local minimum"),
        Line2D([0], [0], color="magenta", lw=2.5, label="ground-truth pose"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=8)
    fig.tight_layout()
    out = os.path.join(FIG, "fig3b_local_minima.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("wrote", out)

    # cost histogram
    fig, ax = plt.subplots(figsize=(6, 3.2))
    costs = d["costs"]
    ax.hist(costs, bins=40, color="tab:blue", alpha=0.8)
    ax.axvline(float(d["cost_global"]), color="k", linestyle="--",
               label=f"global cost {float(d['cost_global']):.2f}")
    ax.set_xlabel("inner cost (matrix-weighted squared error)")
    ax.set_ylabel("initialisations")
    ax.set_title("local solver: final cost distribution")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = os.path.join(FIG, "fig3b_cost_histogram.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("wrote", out)


def threshold_sensitivity(noise="independent"):
    """Iteration counts implied by alternative gradient thresholds.

    The reported number of outer iterations is sensitive to the stopping
    threshold: the paper text quotes ``|grad| < 1e-3`` while the released
    script uses ``tol_grad_sq = 1e-5`` (``|grad| < 3.2e-3``).  Because the
    full gradient trajectories are stored, the iteration count for any
    threshold can be recomputed without re-running the experiment.
    """
    path = os.path.join(RES, f"trajectories_{noise}.csv")
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)
    thresholds = [1e-2, 3e-3, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
    rows = []
    for method in ["sdpr", "local_gt", "local_rnd"]:
        sub = df[(df["method"] == method) & df["grad_sq"].notna()]
        if sub.empty:
            continue
        for thr in thresholds:
            counts = []
            for run, g in sub.groupby("run"):
                g = g.sort_values("iteration")
                hit = g[g["grad_sq"] <= thr]
                counts.append(int(hit["iteration"].iloc[0]) + 1 if len(hit) else 150)
            rows.append(dict(method=method, tol_grad_sq=thr,
                             n_iter_mean=float(np.mean(counts)),
                             n_iter_std=float(np.std(counts)),
                             frac_hit=float(np.mean([c < 150 for c in counts]))))
    out = pd.DataFrame(rows)
    path_out = os.path.join(RES, "tables", f"threshold_sensitivity_{noise}.csv")
    out.to_csv(path_out, index=False)
    print("wrote", path_out)
    print(out.pivot(index="tol_grad_sq", columns="method", values="n_iter_mean"))


def fig_final_error_box(noise="independent"):
    path = os.path.join(RES, f"trajectories_{noise}.csv")
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)
    finals = []
    for method in ORDER:
        sub = df[df["method"] == method]
        for run, g in sub.groupby("run"):
            g = g.sort_values("iteration")
            finals.append(dict(method=method, err=float(g["err"].values[-1])))
    fdf = pd.DataFrame(finals)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    data = [fdf[fdf.method == m]["err"].abs().values for m in ORDER]
    bp = ax.boxplot(data, tick_labels=[SHORT[m] for m in ORDER], patch_artist=True,
                    showfliers=True)
    for patch, m in zip(bp["boxes"], ORDER):
        patch.set_facecolor(COLORS[m])
        patch.set_alpha(0.6)
    ax.set_yscale("log")
    ax.set_ylabel(r"final $|b - b_{\mathrm{true}}|$  [m]")
    ax.set_title(f"final baseline error over 50 trials ({noise} noise)")
    ax.tick_params(axis="x", labelrotation=15, labelsize=7)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out = os.path.join(FIG, f"fig_final_error_box_{noise}.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("wrote", out)


def fig_jacobian_checks():
    path = os.path.join(RES, "jacobians.csv")
    if not os.path.exists(path):
        print("skip jacobian figure: jacobians.csv missing")
        return
    df = pd.read_csv(path)
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.6))
    # (a) histograms of the Eq.(29) relative Jacobian error against FD
    for key, label, color in [
        ("sdpr_jac_err_vs_fd", "SDPR-IS vs FD", "tab:green"),
        ("local_jac_err_vs_fd", "local-GT vs FD", "tab:blue"),
    ]:
        v = df[key].values
        axs[0].hist(np.log10(np.maximum(v, 1e-10)), bins=25, alpha=0.6,
                    color=color, label=label)
    axs[0].set_xlabel(r"$\log_{10}(\Delta J)$ vs finite differences")
    axs[0].set_ylabel("trials")
    axs[0].set_title("(a) Jacobian vs FD (baseline b)", fontsize=9)
    axs[0].legend(fontsize=7)
    # (b) scatter: SDPR vs local Jacobian error, per trial
    axs[1].loglog(df["sdpr_jac_err_vs_fd"], df["local_jac_err_vs_fd"], "o",
                  ms=4, alpha=0.7, color="tab:purple")
    lim = [1e-8, 1e-2]
    axs[1].plot(lim, lim, "k--", lw=0.8)
    axs[1].set_xlim(lim)
    axs[1].set_ylim(lim)
    axs[1].set_xlabel("SDPR-IS  $\Delta J$")
    axs[1].set_ylabel("local-GT  $\Delta J$")
    axs[1].set_title("(b) both methods vs FD reference", fontsize=9)
    axs[1].grid(True, which="both", alpha=0.3)
    # (c) m-Jacobian (feature-location) difference, Table IV analog
    v = df["jac_diff_m_linf"].values
    axs[2].hist(np.log10(np.maximum(v, 1e-12)), bins=25, color="tab:orange")
    axs[2].set_xlabel(r"$\log_{10}$ relative $\Delta J$ w.r.t. feature locations $m_k$")
    axs[2].set_title("(c) SDPR-IS vs local-GT (Table IV analog)", fontsize=9)
    axs[2].set_ylabel("trials")
    for ax in axs:
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = os.path.join(FIG, "fig_jacobian_checks.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("wrote", out)


def write_tableV_markdown():
    """Write the Table-V analog as markdown for the report."""
    lines = []
    for noise in ["independent", "upstream"]:
        path = os.path.join(RES, "tables", f"tableV_{noise}.json")
        if not os.path.exists(path):
            continue
        tv = json.load(open(path))
        lines.append(f"### noise model: `{noise}`\n")
        lines.append("| Method | Final baseline error (mean) | Final baseline error (std) | "
                     "Iterations (mean) | Outer loss (mean) | Wall time / run [s] |")
        lines.append("|---|---|---|---|---|---|")
        for r in tv:
            lines.append(
                f"| {r['method']} | {r['final_baseline_error_mean']:.3e} | "
                f"{r['final_baseline_error_std']:.3e} | {r['n_iter_mean']:.2f} | "
                f"{r['outer_loss_mean']:.4e} | {r['wall_time_mean']:.2f} |")
        lines.append("")
    out = os.path.join(RES, "tables", "tableV.md")
    with open(out, "w") as f:
        f.write("\n".join(lines))
    print("wrote", out)


def main():
    ensure_dirs()
    for noise in ["independent", "upstream"]:
        fig4(noise)
        fig_tableV(noise)
        fig_tightness(noise)
        fig3b(noise)
        threshold_sensitivity(noise)
        fig_final_error_box(noise)
    fig_jacobian_checks()
    write_tableV_markdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
