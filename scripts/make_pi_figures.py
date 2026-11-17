#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成第一阶段的两张图：
1) 一页速览：目标、已证、未证、障碍
2) π 的连分数前缀数值实验
"""
import csv
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib import font_manager as fm

FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
fm.fontManager.addfont(FONT)
plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"] = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(ROOT, "research", "pi-exponent", "assets")
os.makedirs(ASSETS, exist_ok=True)

GREEN = "#E6F4EA"
GREEN_E = "#137333"
RED = "#FCE8E6"
RED_E = "#A50E0E"
GRAY = "#F1F3F4"
GRAY_E = "#5F6368"
BLUE = "#E8F0FE"
BLUE_E = "#1A73E8"
YELLOW = "#FEF7E0"
YELLOW_E = "#B06000"
BG = "#F7F9FC"


def box(ax, x, y, w, h, text, fc, ec, fs=11, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.05,rounding_size=0.10",
                                fc=fc, ec=ec, lw=1.3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color="#202124",
            fontweight="bold" if bold else "normal", linespacing=1.45)


def arrow(ax, x1, y1, x2, y2, color="#5F6368", style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 mutation_scale=14, lw=1.2, color=color,
                                 linestyle=ls, shrinkA=2, shrinkB=2))


def overview():
    fig, ax = plt.subplots(figsize=(16, 10.5))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10.5)
    ax.axis("off")
    fig.patch.set_facecolor(BG)

    ax.text(8, 10.05, "π 的无理性指数：第一阶段独立探索（严格盲测）",
            ha="center", va="center", fontsize=19, fontweight="bold", color="#202124")
    ax.text(8, 9.55, "任务 A/B/D 已完成；C 已作三条路线探索并定位障碍",
            ha="center", va="center", fontsize=12, color="#5F6368")

    # ---------------- left ----------------
    box(ax, 0.4, 7.75, 7.4, 1.55,
        "结论\n"
        "①  $\\mu(\\pi)\\geq 2$：已完整证明（Dirichlet + $\\pi$ 无理性）\n"
        "②  $\\mu(\\pi)=2\\Leftrightarrow \\mathrm{limsup}_n \\frac{\\log q_{n+1}}{\\log q_n}=1$（连分数）\n"
        "③  $\\mu(\\pi)\\leq 2$：据训练知识为公开问题，详见右栏",
        BLUE, BLUE_E, 10.5, False)

    box(ax, 0.4, 6.05, 7.4, 1.45,
        "题目目标（T）\n"
        "$\\forall\\varepsilon>0\\ \\exists Q\\ \\forall q\\geq Q\\ \\forall p:\\ "
        "|\\pi-p/q|\\geq q^{-2-\\varepsilon}$\n"
        "$\\Leftrightarrow \\|q\\pi\\|\\geq q^{-1-\\varepsilon}$（对所有大 $q$）",
        YELLOW, YELLOW_E, 10.5, False)

    # number line
    ax.text(0.7, 5.35, "已知范围", fontsize=12, fontweight="bold", color="#202124")
    x0, x1 = 1.2, 7.6
    y0 = 4.55
    ax.plot([x0, x1], [y0, y0], color="#5F6368", lw=2)
    # map 2..8 to x0..x1
    def X(v):
        return x0 + (v - 2) / 6 * (x1 - x0)
    ax.add_patch(Rectangle((X(2), y0 - 0.13), X(7.6063) - X(2), 0.26,
                           fc="#D2E3FC", ec="none"))
    for v in [2, 3, 4, 5, 6, 7, 8]:
        ax.plot([X(v), X(v)], [y0 - 0.18, y0 + 0.18], color="#5F6368", lw=1)
        ax.text(X(v), y0 - 0.38, str(v), ha="center", va="top", fontsize=8.5, color="#5F6368")
    ax.plot([X(2)], [y0], marker="o", ms=9, color=GREEN_E)
    ax.plot([X(7.6063)], [y0], marker="o", ms=9, color=RED_E)
    ax.text(X(2) + 0.08, y0 + 0.30, "2（Dirichlet）", fontsize=9, color=GREEN_E, ha="left")
    ax.text(X(7.6063) - 0.05, y0 + 0.30, "7.6063（最好已知上界，待核验）",
            fontsize=9, color=RED_E, ha="right")
    ax.text(4.0, y0 - 0.85, "目标 $\\mu(\\pi)=2$ 在区间左端；一致的 $c/q^2$ 下界不可能，$\\varepsilon$ 不可去",
            fontsize=9.5, color="#202124", ha="center")

    box(ax, 0.4, 1.45, 7.4, 2.15,
        "为什么难：三条路线落在同一障碍\n"
        "连分数 $\\Leftrightarrow$ 线性型双见证 $\\Leftrightarrow$ 所有部分商次指数增长\n"
        "• 任何“对所有无理数成立”的论证都不够：Liouville 数 $\\mu=+\\infty$\n"
        "• 单族线性型在例外有理数 $p/q=-a_n/b_n$ 处失效\n"
        "• 第二个独立见证 $\\Leftrightarrow q_{m+1}\\leq 2q_m^{1+\\varepsilon}$：仍是公开问题",
        RED, RED_E, 9.8, False)

    # ---------------- right ----------------
    ax.text(11.95, 8.95, "状态面板", fontsize=13, fontweight="bold", color="#202124", ha="center")

    box(ax, 8.25, 7.65, 7.4, 1.05,
        "已证明（可逐行检查）\n"
        "S1 定义/$\\mathrm{limsup}$   S2 $\\mu(\\pi)\\geq2$   S3 连分数公式\n"
        "S5 线性型转移   S6 双见证$\\Leftrightarrow$连分数   $\\mu(e)=2$",
        GREEN, GREEN_E, 9.3, False)

    box(ax, 8.25, 6.35, 7.4, 1.0,
        "未证明 / 公开问题\n"
        "$\\mu(\\pi)\\leq 2$（主定理）；$\\pi$ 是否 badly approximable",
        RED, RED_E, 10.5, True)

    box(ax, 8.25, 5.05, 7.4, 1.0,
        "待核验（阶段 II/III）\n"
        "$\\mu(\\pi)\\leq7.6063$ 的出处与机制；Family 017 的正式命题",
        GRAY, GRAY_E, 9.8, False)

    box(ax, 8.25, 3.55, 7.4, 1.2,
        "下一步\n"
        "阶段 II 核对 Family 017 命题 $\\to$ 阶段 III 论文对照\n"
        "$\\to$ 阶段 IV 用 Lean 4 形式化已证部分（零 sorry）",
        BLUE, BLUE_E, 9.8, False)

    box(ax, 8.25, 1.45, 7.4, 1.7,
        "关键量化\n"
        "已知 $2\\leq\\mu(\\pi)\\leq 7.6063$，目标 2\n"
        "指数差距 $\\approx 5.6$；这是“质的跳跃”，不是常数优化",
        YELLOW, YELLOW_E, 10.5, False)

    fig.savefig(os.path.join(ASSETS, "2026-11-17-pi-exponent-overview.png"),
                dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


def cf_prefix():
    rows = list(csv.DictReader(open(os.path.join(ROOT, "research", "pi-exponent",
                                                 "data", "pi_cf_prefix.csv"), encoding="utf-8")))
    n = [int(r["n"]) for r in rows]
    lq = [float(r["log10_q_n"]) for r in rows]
    a = [int(r["a_n"]) for r in rows]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True,
                                   gridspec_kw={"height_ratios": [1.15, 1]})
    fig.patch.set_facecolor("white")

    ax1.plot(n, lq, color="#1A73E8", lw=1.2, label="$\\log_{10} q_n$")
    levy = math.pi ** 2 / (12 * math.log(2)) / math.log(10)
    ax1.plot([n[0], n[-1]], [levy * n[0], levy * n[-1]], "--", color="#A50E0E",
             lw=1.0, label=f"Lévy: {levy:.4f}·n")
    ax1.set_ylabel("$\\log_{10} q_n$", fontsize=11)
    ax1.set_title("$\\pi$ 的连分数分母增长（前 5856 项，$10^{3000}$ 以内）", fontsize=13)
    ax1.legend(loc="upper left", fontsize=10)
    ax1.grid(alpha=0.25)
    ax1.text(0.99, 0.05, f"平均 $\\log q_n/n$ = {lq[-1]*math.log(10)/n[-1]:.4f}",
             transform=ax1.transAxes, ha="right", fontsize=9, color="#5F6368")

    ax2.scatter(n, a, s=3, color="#B06000", alpha=0.55, linewidths=0)
    ax2.set_yscale("log")
    ax2.set_xlabel("$n$", fontsize=11)
    ax2.set_ylabel("$a_n$（对数刻度）", fontsize=11)
    ax2.set_title("部分商 $a_n$：最大 20776（索引 431），但不改变 $\\mathrm{limsup}$", fontsize=13)
    ax2.grid(alpha=0.25)
    imax = a.index(max(a))
    ax2.annotate(f"$a_{{{imax}}}=20776$", xy=(n[imax], a[imax]),
                 xytext=(n[imax] + 300, a[imax] * 1.5), fontsize=9,
                 arrowprops=dict(arrowstyle="-", color="#5F6368", lw=0.8))
    ax2.text(0.99, 0.05,
             "该处 $\\log q_{n+1}/\\log q_n$ = 1.000000095",
             transform=ax2.transAxes, ha="right", fontsize=9, color="#5F6368")

    fig.tight_layout()
    fig.savefig(os.path.join(ASSETS, "2026-11-17-pi-cf-prefix.png"),
                dpi=180, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    overview()
    cf_prefix()
    print("ok")
