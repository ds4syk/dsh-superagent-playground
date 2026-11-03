"""Build cleaned figures for the SDPR reproduction atlas.

Data come from the cleanroom run (`repro/results_clean/`) and the faithful
upstream run (`repro/results_upstream/`).  Only our reproduced data are
plotted; paper values are not drawn.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle, FancyArrowPatch, Polygon

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Noto Sans CJK TC", "AR PL UMing CN"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 11
plt.rcParams["axes.titlesize"] = 13
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["legend.fontsize"] = 10

GREEN, BLUE, RED = "#2e8b57", "#2f6fb0", "#d9534f"
CLEAN = Path("/home/syk/labs/SDPR/repro/results_clean")
UP = Path("/home/syk/labs/SDPR/repro/results_upstream")
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------- setup
def fig_setup():
    fig, ax = plt.subplots(figsize=(10, 4.0))
    ax.set_xlim(-0.9, 4.6); ax.set_ylim(-1.15, 1.55); ax.set_aspect("equal"); ax.axis("off")

    # board 1m x 1m at origin (8x8 dots)
    ax.add_patch(Rectangle((-0.5, -0.5), 1.0, 1.0, fill=False, lw=1.5, ec="#555"))
    xs = np.linspace(-0.44, 0.44, 8); ys = np.linspace(-0.44, 0.44, 8)
    for x in xs:
        for y in ys:
            ax.plot(x, y, "k.", ms=4)
    ax.text(0, -0.78, "8×8 标定板\n1 m × 1 m", ha="center", va="top", fontsize=11)

    # camera at 3 m, baseline b (exaggerated)
    cx, cy = 3.0, 0.0; b = 0.18
    ax.add_patch(Rectangle((cx-0.08, cy-b/2-0.06), 0.16, b+0.12, fc="#eaf2fb", ec=BLUE, lw=1.5))
    ax.plot([cx-0.03, cx-0.03], [cy-b/2-0.04, cy+b/2+0.04], color=BLUE, lw=3)
    ax.plot([cx+0.03, cx+0.03], [cy-b/2-0.04, cy+b/2+0.04], color=BLUE, lw=3)
    ax.text(cx, cy+b/2+0.28, "双目相机\nb = 0.24 m", ha="center", va="bottom", fontsize=11, color=BLUE)

    # 90 deg cone to board corners
    for sy in (-1, 1):
        ax.plot([cx-0.08, 0.0], [cy, sy*0.5], color=BLUE, lw=1.0, ls="--", alpha=0.7)
    ax.add_patch(FancyArrowPatch((0.2, cy), (cx-0.15, cy), arrowstyle="<->", mutation_scale=12, color="#555"))
    ax.text(1.6, 0.12, "3 m", ha="center", fontsize=11, color="#555")
    ax.text(2.25, 0.72, "90° 锥", fontsize=11, color=BLUE)
    ax.text(0.4, 1.35, "相机在半径 3 m、90° 锥内随机；光轴指向板中心", ha="center", fontsize=11, color="#333")
    save(fig, "01_setup.png")


# ---------------------------------------------------------------- trajectories
def load_traj():
    df = pd.read_csv(CLEAN / "trajectories_independent.csv")
    grid = np.arange(0, 151)
    out = {}
    for method in ["sdpr", "local_gt", "local_rnd"]:
        sub = df[df.method == method]
        curves = {}
        for col, tf in [("err", lambda v: np.maximum(np.abs(v), 1e-8)),
                        ("loss", lambda v: np.maximum(v, 1e-4))]:
            piv = sub.pivot_table(index="iteration", columns="run", values=col, aggfunc="mean")
            arr = piv.reindex(grid).interpolate(limit_direction="both").to_numpy()
            arr = tf(arr)
            curves[col] = (np.nanmedian(arr, axis=1), np.nanpercentile(arr, 25, axis=1),
                            np.nanpercentile(arr, 75, axis=1))
        out[method] = curves
    return out


def fig_trajectories():
    data = load_traj()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    colors = {"sdpr": GREEN, "local_gt": BLUE, "local_rnd": RED}
    labels = {"sdpr": "SDPR-IS", "local_gt": "局部解（真值初始化）", "local_rnd": "局部解（随机初始化）"}
    x = np.arange(0, 151)
    plot_order = ["sdpr", "local_gt", "local_rnd"]
    style = {"sdpr": dict(lw=3.0, ls="-", alpha=0.85), "local_gt": dict(lw=2.2, ls="--"),
             "local_rnd": dict(lw=2.2, ls="-")}
    alpha = {"sdpr": 0.10, "local_gt": 0.18, "local_rnd": 0.14}
    for method in plot_order:
        med, q1, q3 = data[method]["err"]
        axes[0].plot(x, med, color=colors[method], label=labels[method], **style[method])
        axes[0].fill_between(x, q1, q3, color=colors[method], alpha=alpha[method], lw=0)
    axes[0].set_yscale("log"); axes[0].set_xlim(0, 150)
    axes[0].set_xlabel("外层迭代"); axes[0].set_ylabel("|baseline 误差| [m]（对数）")
    axes[0].set_title("(a) baseline 误差：中位数 + 50% 分位带（50 runs）")
    axes[0].grid(alpha=0.25, which="both")
    handles, labs = axes[0].get_legend_handles_labels()
    order = [labs.index(labels[m]) for m in ["sdpr", "local_gt", "local_rnd"]]
    axes[0].legend([handles[i] for i in order], [labs[i] for i in order], loc="center right")

    for method in plot_order:
        med, q1, q3 = data[method]["loss"]
        axes[1].plot(x, med, color=colors[method], label=labels[method], **style[method])
        axes[1].fill_between(x, q1, q3, color=colors[method], alpha=alpha[method], lw=0)
    axes[1].set_yscale("log"); axes[1].set_xlim(0, 150)
    axes[1].set_xlabel("外层迭代"); axes[1].set_ylabel("outer loss（对数）")
    axes[1].set_title("(b) 外层损失：中位数 + 50% 分位带")
    axes[1].grid(alpha=0.25, which="both")
    handles, labs = axes[1].get_legend_handles_labels()
    order = [labs.index(labels[m]) for m in ["sdpr", "local_gt", "local_rnd"]]
    axes[1].legend([handles[i] for i in order], [labs[i] for i in order], loc="center right")
    save(fig, "02_trajectories.png")


# ---------------------------------------------------------------- final
def fig_final():
    df = pd.read_csv(CLEAN / "run_summary_independent.csv")
    methods = ["sdpr", "local_gt", "local_rnd"]
    labels = ["SDPR-IS", "局部解\n(真值初始化)", "局部解\n(随机初始化)"]
    colors = [GREEN, BLUE, RED]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
    rng = np.random.default_rng(0)

    data = [np.abs(df[df.method == m]["err_final"].to_numpy()) for m in methods]
    bp = axes[0].boxplot(data, widths=0.55, patch_artist=True, showfliers=False)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.18)
    for med, c in zip(bp["medians"], colors):
        med.set_color(c); med.set_linewidth(2)
    for i, (vals, c) in enumerate(zip(data, colors), 1):
        xj = i + rng.uniform(-0.12, 0.12, len(vals))
        axes[0].scatter(xj, vals, s=12, color=c, alpha=0.55, edgecolors="none")
    axes[0].set_yscale("log"); axes[0].set_xticks(range(1, 4)); axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("最终 |baseline 误差| [m]（对数）")
    axes[0].set_title("(a) 最终误差（50 runs）"); axes[0].grid(alpha=0.25, which="both")

    it = [df[df.method == m]["n_iter"].to_numpy() for m in methods]
    means = [v.mean() for v in it]; stds = [v.std() for v in it]
    bars = axes[1].bar(range(3), means, yerr=stds, color=colors, alpha=0.8, capsize=4)
    axes[1].bar_label(bars, fmt="%.1f", padding=2)
    axes[1].set_xticks(range(3)); axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("外层迭代次数"); axes[1].set_title("(b) 收敛所需迭代")
    axes[1].grid(axis="y", alpha=0.25)

    wt = [df[df.method == m]["wall_time"].to_numpy() for m in methods]
    means = [v.mean() for v in wt]; stds = [v.std() for v in wt]
    bars = axes[2].bar(range(3), means, yerr=stds, color=colors, alpha=0.8, capsize=4)
    axes[2].bar_label(bars, fmt="%.1f s", padding=2)
    axes[2].set_xticks(range(3)); axes[2].set_xticklabels(labels)
    axes[2].set_ylabel("单 run 墙钟时间 [s]"); axes[2].set_title("(c) 资源开销（12 workers）")
    axes[2].grid(axis="y", alpha=0.25)
    save(fig, "03_final.png")


# ---------------------------------------------------------------- local minima
def fig_local_minima():
    df = pd.read_csv(CLEAN / "local_minima_independent.csv")
    names = ["数据集 0", "数据集 1", "数据集 2"]
    local = df["frac_local"].to_numpy() * 100
    glob = df["frac_global"].to_numpy() * 100
    overall = local.mean()
    fig, ax = plt.subplots(figsize=(9, 4.6))
    x = np.arange(len(names))
    ax.bar(x, glob, color=GREEN, alpha=0.85, label="收敛到全局最优")
    ax.bar(x, local, bottom=glob, color=RED, alpha=0.85, label="收敛到局部极小")
    for i, (g, l) in enumerate(zip(glob, local)):
        ax.text(i, g/2, f"全局\n{g:.0f}%", ha="center", va="center", color="white", fontsize=10)
        ax.text(i, g + l/2, f"局部\n{l:.0f}%", ha="center", va="center", color="white", fontsize=10)
    ax.axhline(overall, color="#333", ls="--", lw=1.2)
    ax.set_xticks(x); ax.set_xticklabels([f"{n}（n=100）" for n in names])
    ax.set_ylim(0, 105); ax.set_ylabel("随机初始化的结果 [%]")
    ax.set_title(f"随机初始化：多少比例掉进局部极小？（三组均值 {overall:.0f}%）")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, frameon=False); ax.grid(axis="y", alpha=0.25)
    save(fig, "04_local_minima.png")


# ---------------------------------------------------------------- tightness
def fig_tightness():
    df = pd.read_csv(CLEAN / "run_summary_independent.csv")
    d = df[df.method == "sdpr"]
    r = d["diag_r_tight"].to_numpy()
    hmin = d["diag_H_min_eig"].to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    axes[0].hist(np.log10(r), bins=18, color=GREEN, alpha=0.8, edgecolor="white")
    axes[0].axvline(5, color=RED, ls="--", lw=1.5)
    axes[0].text(5.1, axes[0].get_ylim()[1]*0.85, "判据 1e5", color=RED, fontsize=10)
    axes[0].set_xlabel("log10(λ1/λ2)（秩 1 判据）"); axes[0].set_ylabel("run 数")
    axes[0].set_title("(a) 50 runs：全部远高于 1e5"); axes[0].grid(alpha=0.25)

    axes[1].hist(hmin * 1e10, bins=18, color=BLUE, alpha=0.8, edgecolor="white")
    axes[1].axvline(0, color=RED, ls="--", lw=1.5)
    axes[1].set_xlabel("H 的最小特征值（×1e-10）"); axes[1].set_ylabel("run 数")
    axes[1].set_title("(b) 证书矩阵 H 半正定（数值零附近）"); axes[1].grid(alpha=0.25)
    fig.suptitle("紧性与证书：10.8 万次 SDP 求解中 rank-1 比例 100%，corank=1 占 99.97%", fontsize=12)
    save(fig, "05_tightness.png")


# ---------------------------------------------------------------- jacobian
def fig_jacobian():
    df = pd.read_csv(CLEAN / "jacobians.csv")
    cols = ["sdpr_jac_err_vs_fd", "local_jac_err_vs_fd", "sdpr_vs_local_b"]
    labels = ["SDPR-IS\nvs 有限差分", "局部解\nvs 有限差分", "SDPR-IS\nvs 局部解"]
    colors = [GREEN, BLUE, "#8a6d3b"]
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    data = [df[c].to_numpy() for c in cols]
    bp = ax.boxplot(data, widths=0.55, patch_artist=True, showfliers=False)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.18)
    for med, c in zip(bp["medians"], colors):
        med.set_color(c); med.set_linewidth(2)
    rng = np.random.default_rng(1)
    for i, (vals, c) in enumerate(zip(data, colors), 1):
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(vals)), vals, s=12, color=c, alpha=0.55, edgecolors="none")
    ax.set_yscale("log"); ax.set_xticks(range(1, 4)); ax.set_xticklabels(labels)
    ax.set_ylabel("相对误差（对数）"); ax.set_title("Jacobian 校验：解析导数 vs 有限差分（50 trials）")
    ax.grid(alpha=0.25, which="both")
    ax.text(0.02, 0.03, "三条误差均在 1e-6 量级 → 反向实现与有限差分一致", transform=ax.transAxes, fontsize=10, color="#555")
    save(fig, "06_jacobian.png")


# ---------------------------------------------------------------- faithful
def fig_faithful():
    df = pd.read_csv(UP / "final_adaptive/tableV_faithful.csv")
    order = ["SDPR-IS", "Theseus-GT", "Theseus-RND"]
    labels = ["SDPR-IS", "Theseus-GT\n(真值初始化)", "Theseus-RND\n(随机初始化)"]
    colors = [GREEN, BLUE, RED]
    df = df.set_index("method").loc[order]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    x = np.arange(3)
    err = df["err_mean"].to_numpy()
    bars = axes[0].bar(x, err, color=colors, alpha=0.85)
    axes[0].set_yscale("log")
    for i, (b, e, n, f) in enumerate(zip(bars, err, df["n_runs"], df["failures"])):
        axes[0].text(i, e*1.15, f"{e:.3g} m\n成功 {int(n-f)}/{int(n)}", ha="center", va="bottom", fontsize=9)
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("最终 |baseline 误差| [m]（对数）"); axes[0].set_title("(a) 最终误差")
    axes[0].grid(axis="y", alpha=0.25, which="both")

    it = df["iter_mean"].to_numpy()
    bars = axes[1].bar(x, it, color=colors, alpha=0.85)
    axes[1].bar_label(bars, fmt="%.1f", padding=2)
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("外层迭代次数"); axes[1].set_title("(b) 收敛所需迭代")
    axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle("官方代码修复版：50 runs × 20 poses", fontsize=12)
    save(fig, "07_faithful.png")




def fig_trajectories_vertical():
    data = load_traj()
    fig, axes = plt.subplots(2, 1, figsize=(9, 9.2))
    colors = {"sdpr": GREEN, "local_gt": BLUE, "local_rnd": RED}
    labels = {"sdpr": "SDPR-IS", "local_gt": "局部解（真值初始化）", "local_rnd": "局部解（随机初始化）"}
    x = np.arange(0, 151)
    plot_order = ["sdpr", "local_gt", "local_rnd"]
    style = {"sdpr": dict(lw=3.0, ls="-", alpha=0.85), "local_gt": dict(lw=2.2, ls="--"),
             "local_rnd": dict(lw=2.2, ls="-")}
    alpha = {"sdpr": 0.10, "local_gt": 0.18, "local_rnd": 0.14}
    for ax, key, ylabel, title in [
        (axes[0], "err", "|baseline 误差| [m]（对数）", "(a) baseline 误差：中位数 + 50% 分位带（50 runs）"),
        (axes[1], "loss", "outer loss（对数）", "(b) 外层损失：中位数 + 50% 分位带"),
    ]:
        for method in plot_order:
            med, q1, q3 = data[method][key]
            ax.plot(x, med, color=colors[method], label=labels[method], **style[method])
            ax.fill_between(x, q1, q3, color=colors[method], alpha=alpha[method], lw=0)
        ax.set_yscale("log"); ax.set_xlim(0, 150)
        ax.set_xlabel("外层迭代"); ax.set_ylabel(ylabel); ax.set_title(title)
        ax.grid(alpha=0.25, which="both")
        handles, labs = ax.get_legend_handles_labels()
        order = [labs.index(labels[m]) for m in ["sdpr", "local_gt", "local_rnd"]]
        ax.legend([handles[i] for i in order], [labs[i] for i in order], loc="center right")
    save(fig, "02_trajectories_vertical.png")


if __name__ == "__main__":
    fig_setup()
    fig_trajectories()
    fig_trajectories_vertical()
    fig_final()
    fig_local_minima()
    fig_tightness()
    fig_jacobian()
    fig_faithful()
