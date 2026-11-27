import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
for _f in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",):
    try: fm.fontManager.addfont(_f)
    except Exception: pass
plt.rcParams["font.family"] = ["Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs/visuals/2026-11-27-p2-rlt.png"
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

# real
names = ["LINEMOD", "LM-O", "YCB-Video"]
base = [50, 22, 83]; rlt = [50, 39, np.nan]
x = np.arange(3)
ax = axes[0]
ax.bar(x - 0.2, base, 0.4, color="#7F7F7F", label="14×14 Shor")
ax.bar(x + 0.2, [0 if np.isnan(v) else v for v in rlt], 0.4, color="#2E6B3A", label="+ RLT（0 维）")
ax.set_xticks(x); ax.set_xticklabels(names)
ax.set_ylabel("ALM 认证率 (%)"); ax.set_ylim(0, 110)
ax.set_title("真实数据：RLT 对 ALM 证书")
ax.legend(fontsize=8, loc="upper right"); ax.grid(alpha=0.3, axis="y")
ax.text(2, 86, "未跑", ha="center", fontsize=8)

# synthetic
rows = json.load(open(ROOT / "labs/02-constrained-qpeplayer/experiments/theory/results/rlt.json"))
noises = [0.02, 0.05, 0.10, 0.20]
b = [100*np.mean([r["base_cert"] for r in rows if abs(r["noise"]-n)<1e-12]) for n in noises]
rr = [100*np.mean([r["rlt_cert"] for r in rows if abs(r["noise"]-n)<1e-12]) for n in noises]
x = np.arange(4); ax = axes[1]
ax.bar(x - 0.2, b, 0.4, color="#7F7F7F", label="14×14 Shor")
ax.bar(x + 0.2, rr, 0.4, color="#2E6B3A", label="+ RLT")
ax.set_xticks(x); ax.set_xticklabels([f"σ={n:.2f}" for n in noises])
ax.set_ylabel("认证率 (%)"); ax.set_ylim(0, 110)
ax.set_title("合成噪声：低噪紧 / 高噪仍不紧")
ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
fig.suptitle("P2 补充证明后的 RLT 紧性消融（2026-11-27）", fontsize=14, fontweight="bold")
fig.tight_layout(rect=(0,0,1,0.93))
fig.savefig(OUT, dpi=180, facecolor="white", bbox_inches="tight", pad_inches=0.15)
print("wrote", OUT, OUT.stat().st_size)
