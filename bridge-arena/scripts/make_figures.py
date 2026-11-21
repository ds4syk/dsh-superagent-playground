"""生成报告图表（需先完成 evaluate.py）。"""
from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from arena_bridge.design import LEVEL_GRID  # noqa: E402
from arena_bridge.render import gallery, save_bridge_png  # noqa: E402
from arena_bridge.sim import Evaluator  # noqa: E402

PG = ["transformer", "diffusion", "cnn"]
CN = {"transformer": "Transformer", "diffusion": "Diffusion", "cnn": "CNN-VAE",
      "random": "随机基线", "expert": "专家数据"}
COL = {"transformer": "#1971c2", "diffusion": "#7048e8", "cnn": "#e8590c",
       "random": "#868e96", "expert": "#2f9e44"}
OUT = "report/figures"
G = LEVEL_GRID[1]


def load_json(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def fig_score_dist(samples, met):
    groups = ["random", "transformer", "diffusion", "cnn", "expert"]
    data = [samples[f"{g}_score"] for g in groups]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    parts = ax.violinplot(data, showextrema=False, widths=0.75)
    for b, g in zip(parts["bodies"], groups):
        b.set_facecolor(COL[g]); b.set_alpha(0.25); b.set_edgecolor(COL[g])
    bp = ax.boxplot(data, widths=0.16, patch_artist=True, showfliers=False,
                    medianprops=dict(color="black", lw=1.2))
    for patch, g in zip(bp["boxes"], groups):
        patch.set_facecolor("white"); patch.set_edgecolor(COL[g])
    rng = np.random.default_rng(0)
    for i, (d, g) in enumerate(zip(data, groups)):
        x = i + 1 + rng.uniform(-0.13, 0.13, len(d))
        ax.scatter(x, d, s=4, color=COL[g], alpha=0.35, linewidths=0)
    ax.set_xticks(range(1, len(groups) + 1))
    ax.set_xticklabels([f"{CN[g]}\n通过率 {met[g]['pass_rate']*100:.0f}%"
                        if g in met else CN[g] for g in groups], fontsize=9)
    ax.set_ylabel("官方关卡分")
    ax.set_title("512 个样本的关卡分分布（每个点是一座桥）", fontsize=11)
    ax.grid(axis="y", ls=":", alpha=0.5)
    ax.set_ylim(0, 110)
    fig.tight_layout()
    fig.savefig(f"{OUT}/score_distribution.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_metrics(met):
    groups = ["random", "transformer", "diffusion", "cnn", "expert"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    ax = axes[0]
    vals = [met[g]["pass_rate"] * 100 for g in groups]
    ax.bar(range(len(groups)), vals, color=[COL[g] for g in groups])
    for i, v in enumerate(vals):
        ax.text(i, v + 1.5, f"{v:.0f}%", ha="center", fontsize=9)
    ax.set_xticks(range(len(groups))); ax.set_xticklabels([CN[g] for g in groups], rotation=20, fontsize=8.5)
    ax.set_ylabel("通关率 (%)"); ax.set_ylim(0, 108); ax.set_title("三场景全部通过的比例")
    ax.grid(axis="y", ls=":", alpha=0.5)

    ax = axes[1]
    vals = [met[g]["score_mean"] for g in groups]
    errs = [met[g]["score_std"] / np.sqrt(met[g]["n"]) for g in groups]
    ax.bar(range(len(groups)), vals, yerr=errs, capsize=3, color=[COL[g] for g in groups])
    for i, v in enumerate(vals):
        ax.text(i, v + 1.5, f"{v:.1f}", ha="center", fontsize=9)
    ax.set_xticks(range(len(groups))); ax.set_xticklabels([CN[g] for g in groups], rotation=20, fontsize=8.5)
    ax.set_ylabel("平均关卡分"); ax.set_ylim(0, 80); ax.set_title("平均关卡分（±标准误）")
    ax.grid(axis="y", ls=":", alpha=0.5)

    ax = axes[2]
    vals = [met[g]["cost_mean"] for g in groups]
    ax.bar(range(len(groups)), vals, color=[COL[g] for g in groups])
    ax.axhline(10000, color="#c92a2a", ls="--", lw=1)
    ax.text(len(groups) - 0.45, 10300, "预算 10000", color="#c92a2a", fontsize=8)
    for i, v in enumerate(vals):
        ax.text(i, v + 120, f"{v:.0f}", ha="center", fontsize=9)
    ax.set_xticks(range(len(groups))); ax.set_xticklabels([CN[g] for g in groups], rotation=20, fontsize=8.5)
    ax.set_ylabel("平均造价"); ax.set_ylim(0, 11500); ax.set_title("平均造价（预算 10000）")
    ax.grid(axis="y", ls=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(f"{OUT}/metrics_bars.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_training(meta):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    tf = meta.get("transformer", [])
    if tf:
        ep = [r["epoch"] for r in tf]
        axes[0].plot(ep, [r["train"] for r in tf], label="train", color=COL["transformer"])
        axes[0].plot(ep, [r["val"] for r in tf], label="val", color="#0b7285", ls="--")
        axes[0].set_title(f"Transformer 交叉熵（{meta['transformer_time']:.0f}s）")
        axes[0].legend(fontsize=8)
    dm = meta.get("diffusion", [])
    if dm:
        ep = [r["epoch"] for r in dm]
        axes[1].plot(ep, [r["train"] for r in dm], color=COL["diffusion"])
        axes[1].set_title(f"Diffusion 噪声预测 MSE（{meta['diffusion_time']:.0f}s）")
    cv = meta.get("cnn", [])
    if cv:
        ep = [r["epoch"] for r in cv]
        axes[2].plot(ep, [r["bce"] for r in cv], color=COL["cnn"])
        axes[2].set_title(f"CNN-VAE 重建 BCE（{meta['cnn_time']:.0f}s）")
    for ax in axes:
        ax.set_xlabel("epoch"); ax.grid(ls=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(f"{OUT}/training_curves.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_dataset():
    d = np.load("results/dataset_l1.npz")
    sc, passed = d["score"], d["passed"].astype(bool)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    ax = axes[0]
    ax.hist(sc[~passed], bins=60, color="#adb5bd", alpha=0.8, label="未通过")
    ax.hist(sc[passed], bins=60, color="#2f9e44", alpha=0.8, label="通过")
    ax.set_xlabel("官方关卡分"); ax.set_ylabel("设计数")
    ax.set_title(f"30,000 个变异设计（通过 {passed.sum()}，{passed.mean()*100:.1f}%）")
    ax.legend(fontsize=8); ax.grid(ls=":", alpha=0.4)
    ax = axes[1]
    top = np.sort(sc[passed])[::-1][:6000]
    ax.hist(top, bins=50, color="#1971c2", alpha=0.85)
    ax.set_xlabel("官方关卡分"); ax.set_ylabel("设计数")
    ax.set_title(f"训练集：通过设计里分数最高的 {len(top)} 个")
    ax.grid(ls=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(f"{OUT}/dataset.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_best(samples):
    ev = Evaluator(1)
    items = []
    for m in PG:
        d = samples[f"{m}_designs"].astype(np.int64)
        sc = samples[f"{m}_score"]
        order = np.argsort(-sc)
        picked, seen = [], set()
        for i in order:
            key = d[i].tobytes()
            if key in seen:
                continue
            seen.add(key); picked.append(i)
            if len(picked) == 3:
                break
        while len(picked) < 3:          # CNN 只有 1 个不同样本：重复展示，突出塌缩
            picked.append(picked[0])
        for rank, i in enumerate(picked, 1):
            design = G.build_json(d[i])
            r = ev.evaluate(design)
            items.append((design, r, f"{CN[m]} · #{rank}", ""))
            save_bridge_png(f"{OUT}/best_{m}_{rank}.png", design, 1, r,
                            title=f"{CN[m]} · #{rank}  {r.score:.1f} 分", figsize=(6.2, 2.4))
    gallery(f"{OUT}/best_bridges.png", items, 1, ncols=3,
            title="三个模型各自最好的三座桥（官方引擎评分）")
    strip = [it for it in items if it[2].endswith("#1")]
    gallery(f"{OUT}/cover_strip.png", strip, 1, ncols=3, figsize=(13, 3.2))


def fig_density(samples):
    groups = ["random", "transformer", "diffusion", "cnn", "expert"]
    fig, ax = plt.subplots(figsize=(8.4, 3.3))
    bins = np.arange(12, 48, 1.5)
    for g in groups:
        d = samples[f"{g}_designs"]
        members = (d > 0).sum(1)
        ax.hist(members, bins=bins, histtype="step", lw=1.8, color=COL[g],
                label=f"{CN[g]}（均值 {members.mean():.1f}）", density=True)
    ax.set_xlabel("上部杆件数"); ax.set_ylabel("密度")
    ax.set_title("512 个样本的上部杆件数分布")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(ls=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(f"{OUT}/density_score.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_quality_diversity(met):
    groups = ["random"] + PG + ["expert"]
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    for g in groups:
        x, y = met[g]["diversity"], met[g]["score_mean"]
        ax.scatter([x], [y], s=120, color=COL[g], edgecolor="white", zorder=3)
        ax.annotate(CN[g], (x, y), textcoords="offset points", xytext=(7, 6), fontsize=9,
                    color=COL[g])
    ax.set_xlabel("多样性（1 − 平均成对 Jaccard，越大越多样）")
    ax.set_ylabel("平均关卡分")
    ax.set_title("质量 vs 多样性")
    ax.grid(ls=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(f"{OUT}/quality_diversity.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_failures(met):
    groups = ["transformer", "diffusion", "cnn"]
    cats = ["通过", "杆件断裂/车辆失败", "车辆未通过", "物理异常", "超预算"]
    fig, ax = plt.subplots(figsize=(8, 3.2))
    bottom = np.zeros(len(groups))
    colors = ["#2f9e44", "#e8590c", "#f08c00", "#c92a2a", "#7048e8"]
    for c, col in zip(cats, colors):
        vals = np.array([met[g]["failures"].get(c, 0) for g in groups])
        ax.bar(range(len(groups)), vals, bottom=bottom, color=col, label=c, width=0.55)
        bottom += vals
    ax.set_xticks(range(len(groups))); ax.set_xticklabels([CN[g] for g in groups])
    ax.set_ylabel("样本数"); ax.set_title("512 个样本的结局构成")
    ax.legend(fontsize=7.5, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    ax.grid(axis="y", ls=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(f"{OUT}/failure_types.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    met = {m["name"]: m for m in load_json("results/eval/metrics.json")["metrics"]}
    samples = dict(np.load("results/eval/samples.npz"))
    meta = load_json("results/checkpoints/train_meta.json")
    fig_score_dist(samples, met)
    fig_metrics(met)
    fig_training(meta)
    fig_dataset()
    fig_best(samples)
    fig_density(samples)
    fig_failures(met)
    fig_quality_diversity(met)
    print("figures saved")


if __name__ == "__main__":
    main()
