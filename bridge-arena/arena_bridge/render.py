"""把比赛 JSON 设计画成报告用 2D 侧视图。"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle, Circle
import numpy as np

from .sim import Evaluator

matplotlib.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Noto Sans CJK JP", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

COL = {"road": "#343a40", "wood": "#d99a4e", "steel": "#d9480f", "cable": "#1c3f60"}
LW = {"road": 5.0, "wood": 2.4, "steel": 2.4, "cable": 1.5}


def draw(ax, lv, design, result, title=None, show_car=True):
    span = float(lv["anchors"][0][0]) if False else None
    xs = [a[0] for a in lv["anchors"]]
    x0, x1 = min(xs), max(xs)
    if len(xs) > 2 and max(xs) - min(xs) > 30:
        x0, x1 = 0.0, max(xs)
    ax.set_facecolor("#eaf4fb")
    water = lv.get("water_y", -1e8)
    if water is not None and water > -1e7:
        ax.add_patch(Rectangle((x0 - 3, water), (x1 - x0) + 6,
                               max(8, -10 - water), color="#bcd7ee", zorder=0))
    for poly in lv.get("solid", []):
        ax.add_patch(Polygon(poly, closed=True, facecolor="#d9b98c",
                             edgecolor="#b08355", lw=0.8, zorder=1))
    for line in lv.get("terrain", []):
        p = np.array(line)
        ax.plot(p[:, 0], p[:, 1], color="#7a9a45", lw=2.0, zorder=2)

    anchors = {f"A{i}": tuple(a) for i, a in enumerate(lv["anchors"])}
    nodes = [tuple(n) for n in design.get("nodes", [])]

    def pos(ref):
        if isinstance(ref, str):
            return anchors[ref]
        return nodes[int(ref)]

    used = set()
    for m in design.get("members", []):
        p, q, mat = pos(m[0]), pos(m[1]), m[2]
        ax.plot([p[0], q[0]], [p[1], q[1]], color=COL.get(mat, "#495057"),
                lw=LW.get(mat, 2.0), solid_capstyle="round", zorder=4)
        used.add(m[0]); used.add(m[1])
    for ref in used:
        if isinstance(ref, str):
            continue
        x, y = nodes[int(ref)]
        ax.scatter([x], [y], s=8, color="#212529", zorder=5)
    for name, (ax_, ay) in anchors.items():
        ax.add_patch(Polygon([[ax_ - 0.35, ay - 0.35], [ax_ + 0.35, ay - 0.35], [ax_, ay]],
                             closed=True, facecolor="#495057", zorder=6))

    if show_car:
        cx = (x0 - 2.4) if x0 > -1 else (x1 - 2.0)
        ax.add_patch(Rectangle((cx, 0.05), 1.5, 0.36, facecolor="#1971c2", zorder=7))
        ax.add_patch(Rectangle((cx + 1.0, 0.05), 0.42, 0.55, facecolor="#1c7ed6", zorder=7))
        for wx in (cx + 0.32, cx + 1.22):
            ax.add_patch(Circle((wx, 0.02), 0.11, color="#212529", zorder=8))

    ymax = max([n[1] for n in nodes] + [a[1] for a in lv["anchors"]] + [3]) + 2
    ymin = water if (water is not None and water > -1e7) else -6
    ymin = min(ymin, min(n[1] for n in nodes) - 2)
    ax.set_xlim(x0 - 3.2, x1 + 3.2)
    ax.set_ylim(ymin - 0.8, ymax)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if title:
        ax.set_title(title, fontsize=11, pad=4)
    if result is not None:
        state = "通过" if result.passed else "未通过"
        ax.text(0.012, 0.025,
                f"{state} · 关卡分 {result.score:.1f} · 造价 {result.cost:.0f}\n"
                f"杆件 {len(design.get('members', []))} 根 · {result.reason or '全部车辆到达'}",
                transform=ax.transAxes, fontsize=8.2, va="bottom", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#dee2e6", alpha=0.92))


def save_bridge_png(path, design, level, result=None, title=None, figsize=(8.4, 3.2)):
    ev = Evaluator(level)
    if result is None:
        result = ev.evaluate(design)
    fig, ax = plt.subplots(figsize=figsize)
    draw(ax, ev.lv, design, result, title=title)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def gallery(path, items, level, ncols=3, title=None, figsize=None):
    """items: list of (design, result, title, subtitle)"""
    ev = Evaluator(level)
    n = len(items)
    nrows = (n + ncols - 1) // ncols
    if figsize is None:
        figsize = (5.4 * ncols, 2.6 * nrows + 0.4)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.set_axis_off()
    for ax, (design, result, t, sub) in zip(axes, items):
        ax.set_axis_on()
        draw(ax, ev.lv, design, result, title=t)
        if sub:
            ax.text(0.012, 0.025, sub, transform=ax.transAxes, fontsize=8,
                    va="bottom", ha="left",
                    bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="#dee2e6", alpha=0.92))
    if title:
        fig.suptitle(title, fontsize=14, y=0.995)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
