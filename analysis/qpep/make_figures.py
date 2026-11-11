"""生成 04 页与一页图所需的图表。

依赖 results/ptop_results.json、results/ptop_grads.json、results/sdp_scaling.json、
libqpep_bench/*.csv。输出 figures/*.png。
"""

import csv
import json
import math
import os
import statistics
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 160,
})

CB = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]

roots = json.load(open(os.path.join(RES, "ptop_results.json")))["roots"]
grads = json.load(open(os.path.join(RES, "ptop_grads.json")))


def fig1():
    noises = sorted(set(r["noise"] for r in roots))
    lam_by = defaultdict(list)
    rec_by = defaultdict(list)
    ov_by = defaultdict(list)
    for r in roots:
        lam_by[r["noise"]].append(abs(r["lam_star"]))
        rec_by[r["noise"]].append(any(b and b["contains_global"] for b in r["branch"].values()))
        ov_by[r["noise"]].append(r["n_branches_containing_global"])
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9))
    xs = np.arange(len(noises))
    med = [statistics.median(lam_by[n]) for n in noises]
    ax[0].semilogy(xs, [max(v, 1e-12) for v in med], "o-", color=CB[0], label="median |λ*|")
    for i, n in enumerate(noises):
        ax[0].scatter([i] * len(lam_by[n]), [max(v, 1e-12) for v in lam_by[n]],
                      s=12, color=CB[1], alpha=0.7)
    ax[0].set_xticks(xs)
    ax[0].set_xticklabels([f"{n:g}" for n in noises])
    ax[0].set_xlabel("noise σ (units of b)")
    ax[0].set_ylabel("|λ*|  (log)")
    ax[0].set_title("KKT multiplier grows with noise")
    ax[0].grid(alpha=0.25, which="both")

    rec = [sum(rec_by[n]) / len(rec_by[n]) for n in noises]
    ax[1].bar(xs, rec, color=[CB[2] if v > 0 else CB[3] for v in rec], width=0.6)
    ax[1].set_xticks(xs)
    ax[1].set_xticklabels([f"{n:g}" for n in noises])
    ax[1].set_ylim(0, 1.05)
    ax[1].set_ylabel("fraction with global opt. in union")
    ax[1].set_xlabel("noise σ")
    ax[1].set_title("branch recall: only λ=0 succeeds (4/4; else 0/20)")
    for i, v in enumerate(rec):
        ax[1].text(i, v + 0.03, f"{int(round(v*len(rec_by[noises[i]])))}/{len(rec_by[noises[i]])}",
                   ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig1_lambda_recall.png"), bbox_inches="tight")
    plt.close(fig)


def fig2():
    labels, groups = [], []
    for r in grads:
        labels.append(f"σ={r['noise']:g}\ns{r['seed']}")
        groups.append([b["diff_vs_full"] for b in r["branch"].values() if b.get("converged")])
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.9))
    # 左：每个实例的四个分支相对差
    x = np.arange(len(labels))
    for i, vals in enumerate(groups):
        for j, v in enumerate(vals):
            ax[0].scatter(i + (j - 1.5) * 0.12, max(v, 1e-3), s=16,
                          color=CB[j], zorder=3)
    ax[0].axhline(1.0, color="k", ls="--", lw=0.8)
    ax[0].set_yscale("log")
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(labels, fontsize=6.5)
    ax[0].set_ylabel("relative difference from full-KKT dq/dtheta")
    ax[0].set_title("branch gradient vs full-KKT gradient")
    ax[0].grid(alpha=0.25, which="both")

    # 右：full 隐式 vs FD 的误差
    errs = [max(r["full_impl_vs_fd"], 1e-12) for r in grads]
    ax[1].semilogy(np.arange(len(errs)), errs, "o-", color=CB[0])
    ax[1].set_xticks(np.arange(len(labels)))
    ax[1].set_xticklabels(labels, fontsize=6.5)
    ax[1].set_ylabel("relative error vs finite difference")
    ax[1].set_title("full-KKT implicit diff matches FD (1e-9–1e-8)")
    ax[1].grid(alpha=0.25, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig2_gradients.png"), bbox_inches="tight")
    plt.close(fig)


def fig3():
    sdp = json.load(open(os.path.join(RES, "sdp_scaling.json")))
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9))
    ns = [d["n"] for d in sdp]
    ts = [d["sec"] * 1e3 for d in sdp]
    ax[0].plot(ns, ts, "o-", color=CB[0])
    for n, t in zip(ns, ts):
        ax[0].annotate(f"{t:.1f}", (n, t), textcoords="offset points", xytext=(3, 4), fontsize=7)
    ax[0].set_xlabel("QCQP variable dimension n  (SDP vars = n(n+1)/2)")
    ax[0].set_ylabel("SCS solve time (ms)")
    ax[0].set_title("SDPR forward: SDP cost grows with n")
    ax[0].grid(alpha=0.25)

    x = np.arange(3)
    vals = [18.26, 1.91, 5.3]
    labs = ["full template\n596×596", "approx template\n249×249", "approx end-to-end\n(249 + LM)"]
    ax[1].bar(x, vals, color=[CB[3], CB[2], CB[2]], width=0.6)
    for i, v in enumerate(vals):
        ax[1].text(i, v * 1.05, f"{v:.1f} ms", ha="center", fontsize=8)
    ax[1].set_yscale("log")
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(labs, fontsize=7.5)
    ax[1].set_ylabel("time (ms, log)")
    ax[1].set_title("QPEP: fixed template cost (ratio ≈ 9.6×)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig3_cost.png"), bbox_inches="tight")
    plt.close(fig)


def pnp_summary():
    out = {}
    base = os.path.join(HERE, "libqpep_bench")
    for tag in ["4pt", "5pt", "50pt", "500pt"]:
        f = os.path.join(base, f"pnp_data-{tag}-1_approx.csv")
        by = defaultdict(list)
        for line in open(f):
            parts = line.strip().split(",")
            if len(parts) == 13 and parts[0].isdigit():
                by[parts[1]].append(dict(rot=float(parts[4]), trans=float(parts[5]),
                                         loss=float(parts[6]), total=float(parts[11])))
        out[tag] = {}
        for m, rs in by.items():
            ok = [r for r in rs if math.isfinite(r["rot"]) and math.isfinite(r["loss"])]
            out[tag][m] = dict(
                rot_med=statistics.median([r["rot"] for r in ok]) if ok else float("nan"),
                trans_med=statistics.median([r["trans"] for r in ok]) if ok else float("nan"),
                t_med_ms=statistics.median([r["total"] for r in rs]) * 1e3,
                fail=sum(1 for r in rs if not math.isfinite(r["rot"]) or r["rot"] > 1.0),
            )
    return out


def fig4(pnp):
    """LibQPEP 实测：精度 + 时间。"""
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9))
    tags = ["4pt", "5pt", "50pt", "500pt"]
    for i, m in enumerate(["approx_sum", "full"]):
        rots = [pnp[t][m]["rot_med"] for t in tags]
        ts = [pnp[t][m]["t_med_ms"] for t in tags]
        ax[0].plot(range(4), rots, "o-", color=CB[i], label=m)
        ax[1].plot(range(4), ts, "o-", color=CB[i], label=m)
    ax[0].set_yscale("log")
    ax[0].set_xticks(range(4))
    ax[0].set_xticklabels(tags)
    ax[0].set_ylabel("median rotation error (deg)")
    ax[0].set_title("accuracy (1 px noise, 30 trials)")
    ax[0].legend(fontsize=7)
    ax[0].grid(alpha=0.25, which="both")
    ax[1].set_yscale("log")
    ax[1].set_xticks(range(4))
    ax[1].set_xticklabels(tags)
    ax[1].set_ylabel("median time (ms)")
    ax[1].set_title("cost per solve")
    ax[1].legend(fontsize=7)
    ax[1].grid(alpha=0.25, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig4_libqpep.png"), bbox_inches="tight")
    plt.close(fig)


def one_page(pnp):
    """一页可视化：4 个面板。"""
    fig = plt.figure(figsize=(10, 6.2))
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.28)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    # (a) lambda / recall
    noises = sorted(set(r["noise"] for r in roots))
    lam_by = defaultdict(list)
    for r in roots:
        lam_by[r["noise"]].append(abs(r["lam_star"]))
    ax1.semilogy(range(len(noises)), [max(statistics.median(lam_by[n]), 1e-12) for n in noises],
                 "o-", color=CB[0])
    ax1.set_xticks(range(len(noises)))
    ax1.set_xticklabels([f"{n:g}" for n in noises], fontsize=7)
    ax1.set_xlabel("noise σ")
    ax1.set_ylabel("median |λ*| (log)")
    ax1.set_title("(a) λ grows with noise; branch recall=0 for λ≠0", fontsize=9)
    ax1.grid(alpha=0.25, which="both")

    # (b) branch vs full gradient
    for i, r in enumerate(grads):
        for k in range(4):
            b = r["branch"].get(str(k), r["branch"].get(k))
            if b and b.get("converged"):
                ax2.scatter(i + (k - 1.5) * 0.12, max(b["diff_vs_full"], 1e-3),
                            s=18, color=CB[k], zorder=3)
    ax2.axhline(1.0, color="k", ls="--", lw=0.8)
    ax2.set_ylabel("relative difference (branch vs full)")
    ax2.set_title("(b) branch derivatives ≠ correct derivative (0.12–25.7×)", fontsize=9)
    ax2.set_xticks([])
    ax2.grid(alpha=0.25, which="both")

    # (c) pnp accuracy/time
    tags = ["4pt", "5pt", "50pt", "500pt"]
    ax3.plot(range(4), [pnp[t]["approx_sum"]["rot_med"] for t in tags], "o-", color=CB[2], label="approx+LM")
    ax3.plot(range(4), [pnp[t]["full"]["rot_med"] for t in tags], "s--", color=CB[3], label="full (596)")
    ax3.set_yscale("log")
    ax3.set_xticks(range(4))
    ax3.set_xticklabels(tags)
    ax3.set_ylabel("median rot. error (deg)")
    ax3.set_title("(c) LibQPEP: approx 0.01–0.19°; full returns wrong root", fontsize=9)
    ax3.legend(fontsize=7)
    ax3.grid(alpha=0.25, which="both")

    # (d) cost bars
    vals = [2.66, 5.3, 18.3, 55.0]
    labs = ["SDP n=13\n(SDPR repro)", "QPEP\napprox", "QPEP full\n(template LU)", "QPEP full\n(end-to-end)"]
    ax4.bar(range(4), vals, color=[CB[0], CB[2], CB[3], CB[3]], width=0.6)
    for i, v in enumerate(vals):
        ax4.text(i, v * 1.06, f"{v:.1f}", ha="center", fontsize=8)
    ax4.set_yscale("log")
    ax4.set_xticks(range(4))
    ax4.set_xticklabels(labs, fontsize=7)
    ax4.set_ylabel("ms (log)")
    ax4.set_title("(d) fixed template vs SDP: different scaling laws", fontsize=9)
    fig.suptitle("QPEP: full vs approximate (3-of-4), complexity, and differentiability  (2026-11)", fontsize=11)
    out = os.path.join(HERE, "..", "..", "docs", "visuals", "2026-11-11-qpep-diff-layer.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, bbox_inches="tight", dpi=170)
    fig.savefig(os.path.join(FIG, "onepage.png"), bbox_inches="tight", dpi=170)
    plt.close(fig)
    print("wrote", os.path.abspath(out))


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
    pnp = pnp_summary()
    json.dump(pnp, open(os.path.join(RES, "pnp_summary.json"), "w"), indent=1)
    fig4(pnp)
    one_page(pnp)
    print("figures done")
