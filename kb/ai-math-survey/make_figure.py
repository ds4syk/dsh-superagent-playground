#!/usr/bin/env python3
"""One-page visual for the AI-math survey / blind reproduction ticket."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "visuals" / "2026-11-21-ai-math-survey.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Noto Sans CJK TC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

CASES = [
    ("007", "相关"), ("017", "μ(π)"), ("087", "Mahler"),
    ("102", "SDP"), ("159", "Erdos"), ("197", "Kapl."),
    ("221", "MP"), ("271", "Heis."), ("287", "L(F)"), ("362", "VM"),
]
MODELS = ["deepseek", "kimi"]
LABELS = {"deepseek": "DeepSeek-V4-Pro", "kimi": "Kimi-K3"}


def load_scores():
    scores = {m: [0.0] * len(CASES) for m in MODELS}
    idx = {c[0]: i for i, c in enumerate(CASES)}
    gdir = ROOT / "repro" / "openai-10" / "results" / "grades"
    acc = {m: {} for m in MODELS}
    if gdir.exists():
        for f in gdir.glob("*.json"):
            try:
                o = json.loads(f.read_text())
                sid = o.get("session", "")
                m, c = sid.split("/")
                cid = c.split("_")[0]
                sc = o.get("score")
                if m in acc and cid in idx and isinstance(sc, int):
                    acc[m].setdefault(cid, []).append(sc)
            except Exception:
                pass
    for m in MODELS:
        for cid, vals in acc[m].items():
            scores[m][idx[cid]] = sum(vals) / len(vals)
    return scores


def panel_pipeline(ax):
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    ax.set_title("A. 从题面到可信证明：四层流水线", fontsize=13, loc="left", fontweight="bold")
    boxes = [
        (0.3, 7.2, 2.2, 1.5, "题面/定义\n（自然语言）", "#E8F1FB"),
        (2.9, 7.2, 2.2, 1.5, "模型探索\n长思维链", "#E8F1FB"),
        (5.5, 7.2, 2.2, 1.5, "工具\nPython/CAS/检索", "#FFF3D6"),
        (8.1, 7.2, 1.6, 1.5, "Lean\n形式化", "#E4F4E4"),
        (2.9, 4.5, 2.2, 1.5, "敌意审稿\n找缺口", "#F3E8FB"),
        (5.5, 4.5, 2.2, 1.5, "证明修复\n/子目标", "#F3E8FB"),
        (8.1, 4.5, 1.6, 1.5, "Comparator\n/审计", "#E4F4E4"),
        (5.5, 1.8, 2.2, 1.5, "可信交付\n证据分级", "#DFF0D8"),
    ]
    for x, y, w, h, t, c in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06",
                                    fc=c, ec="#444", lw=1.0))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=10)
    arrows = [
        ((2.5, 7.95), (2.9, 7.95)), ((5.1, 7.95), (5.5, 7.95)),
        ((7.7, 7.95), (8.1, 7.95)), ((4.0, 7.2), (4.0, 6.0)),
        ((5.1, 5.25), (5.5, 5.25)), ((7.7, 5.25), (8.1, 5.25)),
        ((8.9, 4.5), (7.7, 3.3)), ((7.7, 2.55), (7.7, 2.55)),
    ]
    for (x1, y1), (x2, y2) in arrows:
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=14, lw=1.2, color="#555"))
    ax.text(0.3, 2.8, "探索流水线：快、便宜、可错", fontsize=10, color="#1f4e79")
    ax.text(0.3, 2.1, "验证流水线：慢、贵、可信", fontsize=10, color="#2e7d32")
    ax.text(0.3, 1.0, "关键：两条流水线分开；不要用一条同时追求快和可信。",
            fontsize=10, color="#7a4b00")


def panel_frontiermath(ax):
    models = ["GPT-5.5 Pro", "Claude Opus 4.7", "Kimi K2.6", "Gemini 3 Pro",
              "GPT-5", "DeepSeek-V3.2", "o3"]
    vals = [52.4, 43.79, 38.97, 37.6, 32.41, 22.1, 18.69]
    colors = ["#c0392b" if "GPT" in m else ("#e67e22" if "Claude" in m else
              ("#27ae60" if "Kimi" in m or "DeepSeek" in m else "#2980b9")) for m in models]
    ax.barh(range(len(models)), vals, color=colors, alpha=0.9)
    ax.set_yticks(range(len(models))); ax.set_yticklabels(models, fontsize=9)
    ax.invert_yaxis(); ax.set_xlim(0, 60)
    ax.set_xlabel("FrontierMath 准确率 (%)")
    ax.set_title("B. 研究级数学 benchmark（2026-11 快照）", fontsize=13, loc="left", fontweight="bold")
    for i, v in enumerate(vals):
        ax.text(v + 1, i, f"{v:.1f}", va="center", fontsize=9)
    ax.text(0.98, 0.02, "来源：evals.report/frontiermath；不同 scaffold 不可直接横比",
            fontsize=8, color="#666", transform=ax.transAxes, va="bottom", ha="right",
            bbox=dict(fc="white", ec="none", alpha=0.85))


def panel_heatmap(ax):
    scores = load_scores()
    data = np.array([scores[m] for m in MODELS])
    im = ax.imshow(data, cmap="YlGnBu", vmin=0, vmax=5, aspect="auto")
    ax.set_xticks(range(len(CASES)))
    ax.set_xticklabels([f"{c[0]}\n{c[1]}" for c in CASES], fontsize=7, rotation=35, ha="right")
    ax.set_yticks(range(len(MODELS)))
    ax.set_yticklabels([LABELS[m] for m in MODELS], fontsize=9)
    ax.set_title("C. 10 个公开案例的盲复现评分（0–5）", fontsize=13, loc="left", fontweight="bold")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            ax.text(j, i, f"{v:.1f}" if v else "0", ha="center", va="center",
                    fontsize=8, color="white" if v > 3 else "#222")
    ax.axhline(0.5, color="white", lw=2)
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)


def panel_conclusions(ax):
    ax.axis("off"); ax.set_title("D. 结论与数字", fontsize=13, loc="left", fontweight="bold")
    lines = [
        ("722 → 721", "初始 722 篇/372 family；当前 721/373"),
        ("10 / 721", "公开推理摘要的 family 数"),
        ("~42%", "top-line 结果有 Lean 形式化"),
        ("10/10", "公开案例都有 Lean 文档"),
        ("0–5", "盲复现评分：见 C 面板"),
        ("", ""),
        ("主流方案", "模型探索 + 工具 + 验证"),
        ("工具需求", "探索：Python 即可；可信：Lean/Comparator"),
        ("模型差距", "FrontierMath 52% vs 22%；研究级更大"),
        ("最大瓶颈", "提出关键引理 + 诚实自审"),
    ]
    y = 0.95
    for a, b in lines:
        if a:
            ax.text(0.02, y, a, fontsize=11, fontweight="bold", color="#1f4e79", va="top")
            ax.text(0.30, y, b, fontsize=10, va="top")
        y -= 0.085
    ax.text(0.02, 0.02, "证据分级：论文声称 / 独立复现 / 未验证；本图 C 为独立复现，B 为官方 benchmark。",
            fontsize=8, color="#666", va="bottom")


def main():
    fig = plt.figure(figsize=(16, 10), dpi=200)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0], hspace=0.28, wspace=0.22)
    ax1 = fig.add_subplot(gs[0, 0]); panel_pipeline(ax1)
    ax2 = fig.add_subplot(gs[0, 1]); panel_frontiermath(ax2)
    ax3 = fig.add_subplot(gs[1, 0]); panel_heatmap(ax3)
    ax4 = fig.add_subplot(gs[1, 1]); panel_conclusions(ax4)
    fig.suptitle("AI 数学证明：方案、工具与模型差距（2026-11）", fontsize=18, fontweight="bold", y=0.98)
    fig.text(0.5, 0.005, "OpenAI 722 篇 / 10 个公开案例 / DeepSeek-V4-Pro vs Kimi-K3 信息隔离复现",
             ha="center", fontsize=9, color="#555")
    fig.savefig(OUT, bbox_inches="tight", facecolor="white")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
