"""P1 post-supplement theory + ablations summary."""
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
BASE = ROOT / "labs/01-laplace-qpeplayer/experiments"
OUT = ROOT / "docs/visuals/2026-11-27-p1-theory.png"

fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.0))

# A: 1D exact crossing
rows = json.load(open(BASE / "theory/results/uniform_unequal.json"))
ax = axes[0, 0]
for label, name in [("crossing", "交叉 ΔJ=0"), ("separated", "分离 ΔJ>0")]:
    rs = [r for r in rows if r["regime"] == label]
    t = np.array([r["tau"] for r in rs]); e = np.array([r["errE"] for r in rs])
    slope = np.polyfit(np.log(t), np.log(e), 1)[0]
    ax.loglog(t, e, "o-", ms=4, label=f"{name}（斜率 {slope:.2f}）")
ax.set_xlabel(r"$\tau$"); ax.set_ylabel("$|E_{\\rm Lap}-E_{\\rm exact}|$")
ax.set_title("A. 1D 精确后验：交叉处仍为 $O(\\tau)$")
ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)

# B: 6D QPEP gradient error
p1 = json.load(open(BASE / "results/raw_grid.json"))
th = [r["theta"] for r in p1]
ax = axes[0, 1]
ax.semilogy(th, [r["laplace_grad_rel"] for r in p1], lw=2, label="Laplace")
ax.semilogy(th, [r["mc_grad_rel"] for r in p1], lw=1.3, label="MC (200×20)")
ax.semilogy(th, [r["hard_grad_rel"] if r["hard_grad_rel"] is not None else np.nan for r in p1], ".", ms=3, label="Hard argmin")
ax.axvline(0, color="k", ls=":", lw=1)
ax.set_xlabel(r"$\theta$"); ax.set_ylabel("梯度相对误差 vs FD")
ax.set_title("B. 6D QPEP：交叉处 Laplace 4.3e-4 / MC 3.3e-2")
ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)

# C: top-K
rows = json.load(open(BASE / "results/ablation.json"))
rs = [r for r in rows if r["exp"] == "topk" and r["theta"] == 0.0]
ks = [r["k"] for r in rs]; pose = [max(r["pose_diff"], 1e-18) for r in rs]; grad = [max(r["grad_diff"], 1e-18) for r in rs]
ax = axes[1, 0]
ax.semilogy(ks, pose, "o-", label="位姿误差")
ax.semilogy(ks, grad, "s-", label="梯度误差")
ax.set_xlabel("保留模式数 K"); ax.set_ylabel("相对全模式的误差")
ax.set_title("C. 6D 消融：top-K 模式（交叉处 θ=0）")
ax.set_xticks(ks); ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)
ax.text(0.03, 0.55, "K=1 位姿差 1.44\nK=2 仍差 0.11\nK=4 全模式", transform=ax.transAxes, fontsize=8)

# D: fold
rows = json.load(open(BASE / "theory/results/fold.json"))
rs = [r for r in rows if abs(r["tau"] - 0.01) < 1e-12]
a = [r["a"] for r in rs]; el = [r["err_lap"] for r in rs]; er = [r["err_reg"] for r in rs]
ax = axes[1, 1]
ax.loglog(a, el, "o-", label="标准 Laplace")
ax.loglog(a, er, "s-", label=r"$\tau^{1/3}$ 正则化")
ax.set_xlabel("$a$（$H=2a$，$a\\to0$ 为 Fold）"); ax.set_ylabel("配分函数相对误差")
ax.set_title(r"D. Fold 合并：标准 Laplace 发散，正则化有界")
ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)

fig.suptitle("P1 补充证明后的主表/消融（2026-11-27）", fontsize=15, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(OUT, dpi=180, facecolor="white", bbox_inches="tight", pad_inches=0.15)
print("wrote", OUT, OUT.stat().st_size)
