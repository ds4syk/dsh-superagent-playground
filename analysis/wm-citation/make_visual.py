# -*- coding: utf-8 -*-
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.font_manager as fm

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(ROOT, "../.."))
E = json.load(open(os.path.join(ROOT, "data/entries.json")))
main = [e for e in E if not e.get("appendix")]

plt.rcParams["font.family"] = "Noto Sans CJK SC"
plt.rcParams["axes.unicode_minus"] = False

BG = "#f7f6f2"; INK = "#1d1d1f"; MUTED = "#666"; ACC = "#0b6e99"
C1, C2, C3, C4, C5, C6 = "#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#7f7f7f"

fig = plt.figure(figsize=(16, 11.2), dpi=150)
fig.patch.set_facecolor(BG)

fig.text(0.045, 0.955, "引用 Dreamer / JEPA / RWM 的高被引世界模型论文（2023-11 — 2026-11）",
         fontsize=23, fontweight="bold", color=INK)
fig.text(0.045, 0.922, "46 篇主表 + 4 篇附录 · OpenAlex 引用 ≥ 90 · 数据截至 2026-11-11 · 世界模型三系引用调研",
         fontsize=12.5, color=MUTED)

# ---------- A: lineage converge (top-left) ----------
axA = fig.add_axes([0.045, 0.60, 0.44, 0.29]); axA.axis("off")
axA.set_xlim(0, 10); axA.set_ylim(0, 10)
axA.text(0, 9.5, "A. 三条谱系的合流", fontsize=15, fontweight="bold", color=INK)
cols = [
    ("Dreamer（2020→2026）", ["RSSM + 想象训练", "DreamerV2 离散潜变量", "DreamerV3 跨域固定超参", "Dreamer 4 离线想象"], C1),
    ("JEPA（2022→2026）", ["联合嵌入预测", "I-JEPA → V-JEPA", "V-JEPA 2 理解+规划", "LeWM / DINO-WM / VLA-JEPA"], C2),
    ("RWM（2018→2026）", ["VAE + RNN 世界模型", "PlaNet / Dreamer 继承", "视频生成式世界模型", "Cosmos / Genie 类"], C3),
]
for i, (name, items, col) in enumerate(cols):
    x = 0.1 + i * 3.25
    axA.add_patch(FancyBboxPatch((x, 7.0), 2.9, 1.5, boxstyle="round,pad=0.08", fc=col, ec="none", alpha=0.92))
    axA.text(x + 1.45, 7.75, name, ha="center", va="center", color="white", fontsize=10.5, fontweight="bold")
    for j, it in enumerate(items):
        axA.text(x + 0.15, 5.9 - j * 0.75, "· " + it, fontsize=8.6, color=INK)
axA.add_patch(FancyBboxPatch((0.3, 0.15), 9.4, 1.35, boxstyle="round,pad=0.1", fc="#e9f2f7", ec=ACC, lw=1.2))
axA.text(5.0, 1.12, "合流：视频生成式世界模型 + 动作条件（世界-动作模型）", ha="center", fontsize=11.5, color=ACC, fontweight="bold")
axA.text(5.0, 0.58, "32/46 篇 · 约 74% 的引用量", ha="center", fontsize=9, color=MUTED)
for x in (1.55, 4.8, 8.05):
    axA.add_patch(FancyArrowPatch((x, 2.05), (x, 1.62), arrowstyle="-|>", mutation_scale=14, color=ACC, lw=1.4))

# ---------- B: method distribution (bottom-left) ----------
axB = fig.add_axes([0.055, 0.10, 0.42, 0.40])
methods = ["视频生成式", "机器人 VLA", "JEPA", "评测/理论/综述", "潜空间/结构化", "3D/占用"]
cnt = [20, 12, 6, 6, 1, 1]
cites = [4347, 3265, 1120, 1207, 165, 152]
bars = axB.barh(range(len(methods)), cnt, color=[C1, C2, C3, C4, C5, C6], height=0.62, alpha=0.88)
axB.set_yticks(range(len(methods))); axB.set_yticklabels(methods, fontsize=10.5)
axB.invert_yaxis(); axB.set_xlim(0, 24)
axB.set_xlabel("论文数（括号内为该类引用合计）", fontsize=9.5, color=MUTED)
axB.set_title("B. 按方法类型：篇数与引用", fontsize=15, fontweight="bold", loc="left", color=INK)
for i, (b, n, c) in enumerate(zip(bars, cnt, cites)):
    axB.text(b.get_width() + 0.35, i, f"{n} 篇 · {c} 引用", va="center", fontsize=9.2, color=INK)
for s in ["top", "right"]:
    axB.spines[s].set_visible(False)
axB.tick_params(axis="x", labelsize=8.5, colors=MUTED)
axB.grid(axis="x", color="#ddd", lw=0.6, alpha=0.6); axB.set_axisbelow(True)

# ---------- C: top 10 (top-right) ----------
axC = fig.add_axes([0.565, 0.55, 0.40, 0.34])
tops = sorted(main, key=lambda e: -e["cites"])[:10]
short = {"cosmos": "Cosmos", "dreamzero": "DreamZero", "platonic": "柏拉图表征", "worldvla": "WorldVLA",
         "gr2": "GR-2", "univla": "UniVLA", "drivedreamer": "DriveDreamer", "lewm": "LeWorldModel",
         "cosmospolicy": "Cosmos Policy", "dinowm": "DINO-WM"}
names = [short.get(e["id"], e["zh"][:10]) for e in tops][::-1]
vals = [e["cites"] for e in tops][::-1]
cols = {"cosmos": C1, "dreamzero": C2, "platonic": C4, "worldvla": C2, "gr2": C2, "univla": C2,
        "drivedreamer": C1, "lewm": C3, "cosmospolicy": C2, "dinowm": C3}
colors = [cols.get(e["id"], ACC) for e in tops][::-1]
axC.barh(range(len(names)), vals, color=colors, height=0.62, alpha=0.9)
axC.set_yticks(range(len(names))); axC.set_yticklabels(names, fontsize=10)
axC.set_xlim(0, 1180)
axC.set_title("C. 引用数 Top 10（OpenAlex）", fontsize=15, fontweight="bold", loc="left", color=INK)
for i, v in enumerate(vals):
    axC.text(v + 18, i, str(v), va="center", fontsize=9, color=INK)
for s in ["top", "right"]:
    axC.spines[s].set_visible(False)
axC.tick_params(axis="x", labelsize=8.5, colors=MUTED)
axC.grid(axis="x", color="#ddd", lw=0.6, alpha=0.6); axC.set_axisbelow(True)

# ---------- D: hardware tiers (bottom-right) ----------
axD = fig.add_axes([0.565, 0.10, 0.40, 0.30])
tiers = ["万卡级", "数百卡", "数十卡", "单卡/数卡", "未报告", "不适用"]
tv = [3, 8, 12, 4, 15, 4]
tc = ["#08306b", "#2171b5", "#6baed6", "#c6dbef", "#bdbdbd", "#e0e0e0"]
axD.bar(range(len(tiers)), tv, color=tc, width=0.62)
axD.set_xticks(range(len(tiers))); axD.set_xticklabels(tiers, fontsize=9.5)
axD.set_ylim(0, 18)
axD.set_title("D. 训练硬件档位（篇数）", fontsize=15, fontweight="bold", loc="left", color=INK)
for i, v in enumerate(tv):
    axD.text(i, v + 0.35, str(v), ha="center", fontsize=10, color=INK, fontweight="bold")
for s in ["top", "right"]:
    axD.spines[s].set_visible(False)
axD.tick_params(axis="y", labelsize=8.5, colors=MUTED)
axD.grid(axis="y", color="#ddd", lw=0.6, alpha=0.6); axD.set_axisbelow(True)

# ---------- E: conclusions strip ----------
fig.text(0.045, 0.028,
         "三个结论：① 视频生成 + 机器人 VLA 占主表 74% 引用；② 纯潜空间想象训练只剩 Dreamer 4，但 DreamerV3/TD-MPC2 仍是通用基线；"
         "③ 硬件跨四个数量级，15/46 篇未报告训练卡数。",
         fontsize=9.8, color=MUTED)

out = os.path.join(REPO, "docs/visuals/2026-11-11-wm-citation-survey.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
plt.savefig(out, dpi=150, facecolor=BG, bbox_inches="tight")
print("wrote", out)
