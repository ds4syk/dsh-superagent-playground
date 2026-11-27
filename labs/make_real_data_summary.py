"""Two-panel real-data summary: ModelNet40 (P3) and LINEMOD (P2)."""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

for _f in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",):
    try:
        fm.fontManager.addfont(_f)
    except Exception:
        pass
plt.rcParams["font.family"] = ["Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "visuals" / "2026-11-27-real-data.png"

fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))

# P3 ModelNet40
p3 = json.load(open(ROOT / "labs/03-robust-qpeplayer/experiments/real/modelnet/results/main_table.json"))
methods = ["Kabsch", "RANSAC+Kabsch", "IRLS-GM", "GNC-GM (ours)"]
ax = axes[0]
for m in methods:
    rs = [r for r in p3 if r["method"] == m]
    ax.plot([100 * r["p"] for r in rs], [100 * r["success"] for r in rs], "o-", ms=3.5, label=m)
ax.set_xlabel("外点率 (%)"); ax.set_ylabel("成功率 (%)")
ax.set_ylim(-2, 102); ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="lower left")
ax.set_title("P3 ModelNet40：真实物体点云 3D-3D")
ax.text(0.03, 0.55, "p=70%: GNC 100%, IRLS 98%, Kabsch 0%\nGNC 1.7 ms vs RANSAC 54 ms (~30×)",
        transform=ax.transAxes, fontsize=8)

# P2 LINEMOD
rows = json.load(open(ROOT / "labs/02-constrained-qpeplayer/experiments/real/linemod/results/raw.json"))
names = [("Unconstrained", "Unconstrained"), ("Soft penalty", "Soft"), ("Projection", "Projection"), ("ALM-QPEP", "ALM-QPEP")]
pen, feas, cert = [], [], []
for key, _ in names:
    rs = [r for r in rows if r["method"] == key]
    pen.append(np.median([r["pen_mm"] for r in rs]))
    feas.append(100 * np.mean([r["feas"] for r in rs]))
    cert.append(100 * np.mean([bool(r.get("cert")) for r in rs]))
ax = axes[1]
x = np.arange(len(names))
ax.bar(x - 0.2, feas, 0.4, color="#A8C8E8", label="可行率")
ax.bar(x + 0.2, cert, 0.4, color="#2E6B3A", label="认证率")
for i, (pe, n) in enumerate(zip(pen, names)):
    ax.text(i, 106, f"pen {pe:.1f} mm", ha="center", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels([n for _, n in names], fontsize=8)
ax.set_ylabel("百分比 (%)"); ax.set_ylim(0, 118)
ax.set_title("P2 LINEMOD：真实网格 + 真实桌面平面")
ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=0.3, axis="y")

fig.suptitle("真实数据实验（2026-11-27）—— P3 ModelNet40 / P2 BOP LINEMOD；P1 真实对称性尝试未产生双模式，未纳入主表",
             fontsize=13, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(OUT, dpi=180, facecolor="white", bbox_inches="tight", pad_inches=0.15)
print("wrote", OUT, OUT.stat().st_size)
