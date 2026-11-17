# -*- coding: utf-8 -*-
"""Bar chart: where the 7.5 core-hours of the FourColor build go."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[1] / "docs" / "visuals" / "2026-11-17-fourcolor-corehours.png"
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

labels = ["数学（65 模块）", "放电展示（406 模块）", "可约性证书（349 模块）"]
hours = [0.1, 3.9, 3.5]
colors = ["#2E6B3A", "#1F4E79", "#8A6D1D"]

fig, ax = plt.subplots(figsize=(10, 4.2), dpi=200)
bars = ax.barh(labels, hours, color=colors, height=0.55)
for b, h in zip(bars, hours):
    ax.text(h + 0.06, b.get_y() + b.get_height() / 2, f"{h:g} core-hour",
            va="center", fontsize=11, color="#1A1A1A")
ax.set_xlim(0, 4.6)
ax.set_xlabel("core-hour（上游 README 的测量）", fontsize=10)
ax.set_title("四色定理 Lean 4 构建：7.5 core-hour 的去向", fontsize=15, fontweight="bold",
             loc="left", color="#14281A")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="x", color="#DDDDDD", linewidth=0.8)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(OUT, bbox_inches="tight", facecolor="white")
print(OUT)
