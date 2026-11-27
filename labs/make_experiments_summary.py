"""Three-panel summary of the three prototype experiments."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

for _f in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
           "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"):
    try:
        fm.fontManager.addfont(_f)
    except Exception:
        pass
plt.rcParams["font.family"] = ["Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "visuals" / "2026-11-27-experiments.png"

fig, axes = plt.subplots(1, 3, figsize=(16.2, 4.8))

# ---------------- P1 ----------------
p1 = json.load(open(ROOT / "labs/01-laplace-qpeplayer/experiments/results/raw_grid.json"))
th = [r["theta"] for r in p1]
ax = axes[0]
ax.semilogy(th, [r["laplace_grad_rel"] for r in p1], label="Laplace", lw=2)
ax.semilogy(th, [r["mc_grad_rel"] for r in p1], label="MC (200×20)", lw=1.5)
hard = [r["hard_grad_rel"] if r["hard_grad_rel"] is not None else float("nan") for r in p1]
ax.semilogy(th, hard, ".", ms=3, label="Hard argmin（交叉处 N/A）")
ax.axvline(0, color="k", ls=":", lw=1)
ax.set_xlabel(r"$\theta$（分支参数）")
ax.set_ylabel("梯度相对误差 vs FD")
ax.set_title("P1 分支交叉：Laplace 仍可微、MC 方差大")
ax.grid(alpha=0.3, which="both")
ax.legend(fontsize=8)
ax.text(0.02, 0.70, r"$\theta=0$: Laplace $4.3\times10^{-4}$" + "\n" + r"MC $3.3\times10^{-2}$, std $0.11$",
        transform=ax.transAxes, fontsize=8, va="top")

# ---------------- P2 ----------------
p2 = json.load(open(ROOT / "labs/02-constrained-qpeplayer/experiments/results/main_table.json"))
agg = p2["agg_main"]["16"]
names = [("unc", "Unconstrained"), ("soft", "Soft penalty"), ("proj", "Projection"), ("alm", "ALM-QPEP")]
pens = [agg[k]["pen"]["median"] for k, _ in names]
cert = [100 * agg[k]["certified"] for k, _ in names]
feas = [100 * agg[k]["feasible"] for k, _ in names]
ax = axes[1]
colors = ["#C0504D", "#E6A23C", "#7F7F7F", "#2E6B3A"]
x = range(len(names))
ax.bar([i - 0.18 for i in x], feas, width=0.36, color="#A8C8E8", label="可行率")
ax.bar([i + 0.18 for i in x], cert, width=0.36, color="#2E6B3A", label="认证率")
ax.set_xticks(list(x)); ax.set_xticklabels([n for _, n in names], fontsize=8)
ax.set_ylabel("百分比 (%)")
ax.set_ylim(0, 118)
ax.set_title("P2 桌面 PnP：硬可行 + 14×14 证书（p=16）")
for i, pen in enumerate(pens):
    ax.text(i, 105, f"pen {pen:.1f} mm", ha="center", fontsize=7.5)
ax.legend(fontsize=8, loc="lower right")
ax.grid(alpha=0.3, axis="y")

# ---------------- P3 ----------------
p3 = json.load(open(ROOT / "labs/03-robust-qpeplayer/experiments/synthetic/results/main_table_extended.json"))
rows = [r for r in p3["rows"] if r["outlier_type"] == "random"]
methods = ["Kabsch", "RANSAC-1000", "IRLS-GM", "GNC-GM"]
ax = axes[2]
for m in methods:
    rs = [r for r in rows if r["method"] == m]
    ax.plot([100 * r["p"] for r in rs], [100 * r["success"] for r in rs], "o-", ms=3, label=m)
ax.set_xlabel("外点率 (%)")
ax.set_ylabel("成功率 (%)")
ax.set_ylim(-2, 102)
ax.set_title("P3 3D-3D：同精度 + 约 32× 加速（vs 固定 RANSAC）")
ax.grid(alpha=0.3)
ax.legend(fontsize=8, loc="lower left")
ax.text(0.30, 0.20, "p=50%: GNC 1.5 ms, RANSAC-1000 47 ms\n（自适应 RANSAC 在低外点率更快）",
        transform=ax.transAxes, fontsize=8, va="bottom")

fig.suptitle("三个 QPEP prototype 的合成实验主表（2026-11-27）", fontsize=15, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(OUT, dpi=180, facecolor="white", bbox_inches="tight", pad_inches=0.15)
print("wrote", OUT, OUT.stat().st_size)
