# -*- coding: utf-8 -*-
"""Generate the one-page visual for the FourColor Lean 4 report.

Run with the venv that has matplotlib::

    /mnt/storage1/syk/.venvs/fourcolor/bin/python scripts/make_fourcolor_figure.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "visuals" / "2026-11-17-fourcolor-lean.png"

SANS = "Noto Sans CJK SC"
SERIF = "Noto Serif CJK SC"
for name in (SANS, SERIF):
    try:
        font_manager.findfont(name, fallback_to_default=False)
    except Exception:
        pass

plt.rcParams["font.sans-serif"] = [SANS, "DejaVu Sans"]
plt.rcParams["font.serif"] = [SERIF, "DejaVu Serif"]
plt.rcParams["axes.unicode_minus"] = False

GREEN = "#2E6B3A"
BLUE = "#1F4E79"
GRAY = "#4A4A4A"
LIGHT = "#F4F7F4"
LIGHT_BLUE = "#EEF3F9"
ACCENT = "#B03A2E"
GOLD = "#8A6D1D"


def box(ax, x, y, w, h, text, face=LIGHT, edge=GREEN, size=11, weight="normal",
        color="#1A1A1A", lw=1.4):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                       linewidth=lw, edgecolor=edge, facecolor=face, zorder=3)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size,
            fontweight=weight, color=color, zorder=4, linespacing=1.35)


def arrow(ax, x1, y1, x2, y2, text="", color=GRAY, size=9, style="-|>", lw=1.3,
          text_off=(0, 0), text_ha="left"):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=13,
                        linewidth=lw, color=color, zorder=2,
                        shrinkA=2, shrinkB=2)
    ax.add_patch(a)
    if text:
        ax.text((x1 + x2) / 2 + text_off[0], (y1 + y2) / 2 + text_off[1], text,
                ha=text_ha, va="center", fontsize=size, color=color, zorder=5,
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.85))


def main():
    fig = plt.figure(figsize=(16, 10), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(0.55, 9.55, "四色定理的 Lean 4 形式化", fontsize=25, fontweight="bold",
            color="#14281A", va="top")
    ax.text(0.55, 9.10, "从连续平面到内核可检查的证明：一次可复现的独立验证",
            fontsize=13.5, color=GRAY, va="top")
    ax.plot([0.55, 15.45], [8.88, 8.88], color=GREEN, lw=2.2, alpha=0.55)

    # ---------------- left: proof pipeline ----------------
    ax.text(0.55, 8.55, "证明流水线", fontsize=13, fontweight="bold", color=GREEN)
    x, w = 0.7, 5.1
    cx = x + w / 2

    box(ax, x, 7.55, w, 0.72, "任意简单平面地图\n（区域开、连通；可无限多）", size=11.2)
    arrow(ax, cx, 7.53, cx, 7.02, "Finitize：紧性 + 网格点 + 超滤子", size=9.2,
          text_off=(0.12, 0), text_ha="left")
    box(ax, x, 6.25, w, 0.72, "有限简单平面地图", size=11.2)
    arrow(ax, cx, 6.23, cx, 5.72, "Discretize：横截点 · 邻接盒 · matte 插值 · 网格地图",
          size=8.6, text_off=(0.12, 0), text_ha="left")
    box(ax, x, 4.95, w, 0.72, "平面无桥超地图（hypermap）\n三置换 · Euler 公式 · Jordan 性质", size=10.6)
    arrow(ax, cx, 4.93, cx, 4.42, "Combinatorial：立方化 + 对 dart 数归纳", size=9.2,
          text_off=(0.12, 0), text_ha="left")
    box(ax, x, 3.55, w, 0.82, "极小反例不存在\n⇒ 每个平面无桥超地图四色可着色", size=10.8,
        face="#EAF2EA")
    arrow(ax, cx, 3.53, cx, 3.02, "Complete：接上两块计算", size=9.2,
          text_off=(0.12, 0), text_ha="left")
    box(ax, x, 2.10, w, 0.88, "FourColor.fourColorTheorem : FourColorTheorem\n"
        "∀ m, SimpleMap m → ColorableWith 4 m", size=10.4, face="#E4EFE4",
        edge=GREEN, weight="bold")

    # side boxes feeding the minimal-counterexample step
    sx, sw = 6.15, 3.35
    box(ax, sx, 4.62, sw, 1.10,
        "Unavoidability\n放电 ⇒ 正电荷面，元数 5–11\n七组展示 exclude5 … exclude11",
        size=9.6, face=LIGHT_BLUE, edge=BLUE)
    arrow(ax, sx, 5.17, x + w, 4.30, "", color=BLUE)
    box(ax, sx, 3.10, sw, 1.12,
        "Reducibility\n633 个可约构型\nKempe 链证书（内核检查）",
        size=9.6, face=LIGHT_BLUE, edge=BLUE)
    arrow(ax, sx, 3.66, x + w, 3.96, "", color=BLUE)
    arrow(ax, sx + sw / 2, 4.60, sx + sw / 2, 4.24, "", color=BLUE, lw=1.1)

    # ---------------- right: verification panel ----------------
    px, pw = 9.95, 5.45
    ax.text(px, 8.55, "本次独立验证", fontsize=13, fontweight="bold", color=GREEN)
    box(ax, px, 7.25, pw, 1.05,
        "固定 commit 3db71e0 · Lean v4.34.0-rc2\nMathlib 141f6b6 · 干净源码重建",
        size=9.8, face="#F7F7F2", edge=GOLD)
    box(ax, px, 6.10, pw, 0.95,
        "lake exe cache get  →  regenerate 633 certificates\n→  build 821 modules  →  check.sh",
        size=9.8, face=LIGHT)
    box(ax, px, 4.95, pw, 0.95,
        "#print axioms FourColor.fourColorTheorem\n[propext, Classical.choice, Quot.sound]",
        size=9.8, face="#EAF2EA", edge=GREEN)
    box(ax, px, 3.72, pw, 1.05,
        "独立 Audit.lean：类型 = FourColorTheorem\n公理闭包 · 无 sorryAx · 声明计数\n→ AUDIT OK",
        size=9.6, face="#EAF2EA", edge=GREEN)
    box(ax, px, 2.10, pw, 1.42,
        "反 vacuity 负对照通过\n非空性：4 个两两相邻区域不可 3 着色\n无 sorry / native_decide / 自定义公理",
        size=9.6, face="#F7F7F2", edge=GOLD)

    # big number strip
    nums = [("633", "可约构型"), ("7", "组放电展示"), ("3", "标准公理"),
            ("0", "sorry / native_decide")]
    for i, (n, label) in enumerate(nums):
        nx = px + i * (pw / 4)
        ax.text(nx + pw / 8, 1.52, n, ha="center", va="center", fontsize=21,
                fontweight="bold", color=ACCENT if n == "0" else GREEN)
        ax.text(nx + pw / 8, 1.20, label, ha="center", va="center", fontsize=8.2, color=GRAY)
    ax.plot([px, px + pw], [1.00, 1.00], color=GRAY, lw=0.8, alpha=0.4)

    # ---------------- bottom timeline ----------------
    ty = 0.55
    ax.plot([0.7, 15.3], [ty, ty], color=GRAY, lw=1.0, alpha=0.5)
    events = [
        (0.8, "1852", "Guthrie 提出"),
        (3.0, "1879", "Kempe 的证明"),
        (5.2, "1890", "Heawood 指出漏洞"),
        (7.4, "1976", "Appel–Haken"),
        (9.6, "1997", "633 构型证明"),
        (11.8, "2005", "Gonthier Coq"),
        (14.0, "2026", "Lean 4 × 2"),
    ]
    for ex, year, label in events:
        ax.plot([ex], [ty], marker="o", markersize=5, color=GREEN, zorder=4)
        ax.text(ex, ty + 0.16, year, ha="center", va="bottom", fontsize=8.6,
                fontweight="bold", color="#14281A")
        ax.text(ex, ty - 0.16, label, ha="center", va="top", fontsize=7.4, color=GRAY)

    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.12, facecolor="white")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
