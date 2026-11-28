"""Generate the one-page visual for the three-paper workspace (labs/01-03).

Run: /home/syk/miniconda3/bin/python labs/make_overview.py
Output: docs/visuals/2026-11-27-three-paper-tracks.png
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

for _f in (
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
):
    try:
        fm.fontManager.addfont(_f)
    except Exception:
        pass

plt.rcParams["font.family"] = ["Noto Sans CJK SC", "Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

HERE = Path(__file__).resolve().parent.parent
OUT = HERE / "docs" / "visuals" / "2026-11-27-three-paper-tracks.png"

EDGE = "#333333"
SHARED_FACE = "#EAF3E7"
SHARED_EDGE = "#2E6B3A"
FOOT_FACE = "#F7F7F7"

CARDS = [
    dict(
        x=0.35,
        title="P1  Laplace-QPEPLayer",
        sub="歧义 / 分支交叉 · 多峰 Laplace",
        face="#FDE8E8",
        head="#C0504D",
        pain="argmin 在分支交叉处不可微；SDPR 升维且可能非紧",
        method="枚举全部代数极小 + 平移精确积分 + 旋转 Laplace 平滑",
        thm="3×3 黎曼隐微分；平滑期望损失的解析梯度",
        gap="模式重叠的一致 Laplace；Choi–Lam 反例的物理可实现性",
        exp="LINEMOD / UTIAS In-The-Dark；对比 EPro-PnP、SDPRLayers",
        venue="T-RO / CVPR / RSS",
        status="理论：初稿｜verify：2｜P0：5",
    ),
    dict(
        x=5.80,
        title="P2  Constrained-QPEPLayer",
        sub="物理不等式 · FB 约化微分",
        face="#E8F0FA",
        head="#2F5597",
        pain="SDP 侧 p 个松弛变量升维；互补非光滑；四脚桌 LICQ 失效",
        method="ALM 把 p 个约束压回 10×10 QPEP；FB 消到 (6+|A|) 系统",
        thm="常数维归约；SDP 下界证书；FB 极限的相容条件",
        gap="ALM 只到 KKT；H_tan 精确 vs GN；弱活跃 Clarke 导数",
        exp="YCB-Video/BOP 非穿透；KITTI/TUM 倾角与视锥约束",
        venue="RSS / ICRA / RA-L",
        status="理论：初稿｜verify：2｜P0：3",
    ),
    dict(
        x=11.25,
        title="P3  Robust-QPEPLayer",
        sub="高外点 · GNC + 曲率修正",
        face="#EAF3E7",
        head="#548235",
        pain="鲁棒 SDP 随 N 升维；冻结权重 IRLS 丢二阶项、梯度出错",
        method="BR 对偶 + GNC，每轮仍是 4 变量 QPEP；6×6 精确不动点 IFT",
        thm="黎曼修正 H_rob；1D 拐点变号；O(‖r‖⁻⁴) 逐观测界",
        gap="GNC 只到局部稳定点；6D 变号需联合谱条件",
        exp="UTIAS 无 RANSAC；3DMatch/KITTI 30–70% 外点",
        venue="T-RO / ICRA / CVPR",
        status="理论：初稿｜verify：2｜P0：4",
    ),
]


def box(ax, x, y, w, h, face, edge=EDGE, lw=1.4, radius=0.08):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle=f"round,pad=0.02,rounding_size={radius}",
            linewidth=lw, edgecolor=edge, facecolor=face,
        )
    )


def arrow(ax, p0, p1, color=EDGE, lw=1.8, ls="-"):
    ax.add_patch(
        FancyArrowPatch(
            p0, p1, arrowstyle="-|>", mutation_scale=15,
            linewidth=lw, color=color, linestyle=ls,
        )
    )


fig, ax = plt.subplots(figsize=(16.8, 9.6), dpi=200)
ax.set_xlim(0, 16.8)
ax.set_ylim(0, 9.6)
ax.axis("off")

# Header
ax.text(0.35, 9.18, "从三个 QPEP prototype 到三篇论文 · 一页速览",
        fontsize=23, fontweight="bold", ha="left", va="center", color="#1a1a1a")
ax.text(0.35, 8.76,
        "labs/01–03 并行工作区｜共享 QPEP 零升维内核｜2026-11-27｜理论初稿 + 实验方案 + 数学问题",
        fontsize=11.5, ha="left", va="center", color="#444444")

# Shared base
box(ax, 0.35, 7.48, 16.1, 0.95, SHARED_FACE, SHARED_EDGE, lw=1.2)
ax.text(0.55, 8.16,
        "共享内核：QPEP 消去 t → S³ 上 4 变量、10×10 M（与观测数 N 无关）；"
        "x_R = 2U·v2(q)，21 约束 / 20 维，对应四元数四次型的二阶松弛",
        fontsize=11.8, ha="left", va="center", color="#1a1a1a", fontweight="bold")
ax.text(0.55, 7.72,
        "Hilbert 非紧：四元四次非负 ≠ SOS，SDPRLayers 存在结构性非紧风险；"
        "QPEP 在 S³ 上零松弛，但是穷举式代数全局解，不是凸证书",
        fontsize=11.0, ha="left", va="center", color="#333333")

# Cards
TOP = 7.10
BOT = 1.95
H = TOP - BOT


def wrap_cjk(text, width=31):
    lines, line = [], ""
    for ch in text:
        line += ch
        if len(line) >= width and ch in "，。；：、） ":
            lines.append(line.strip())
            line = ""
    if line:
        lines.append(line.strip())
    return lines or [text]


for c in CARDS:
    box(ax, c["x"], BOT, 5.2, H, c["face"])
    # header band
    ax.add_patch(
        FancyBboxPatch(
            (c["x"], TOP - 0.62), 5.2, 0.62,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            linewidth=1.2, edgecolor=c["head"], facecolor=c["head"],
        )
    )
    ax.text(c["x"] + 0.18, TOP - 0.31, c["title"], fontsize=13.2, fontweight="bold",
            ha="left", va="center", color="white")
    ax.text(c["x"] + 0.18, TOP - 0.82, c["sub"], fontsize=10.5, fontweight="bold",
            ha="left", va="center", color=c["head"])

    rows = [
        ("痛点", c["pain"]),
        ("方法", c["method"]),
        ("定理", c["thm"]),
        ("GAP", c["gap"]),
        ("实验", c["exp"]),
        ("目标", c["venue"]),
    ]
    y = TOP - 1.08
    for label, text in rows:
        lines = wrap_cjk(text)
        ax.text(c["x"] + 0.18, y, label, fontsize=9.6, fontweight="bold",
                ha="left", va="top", color=c["head"])
        for i, ln in enumerate(lines):
            ax.text(c["x"] + 0.78, y - i * 0.255, ln, fontsize=9.3,
                    ha="left", va="top", color="#222222")
        y -= 0.255 * len(lines) + 0.17
    # status pill at the bottom
    ax.text(c["x"] + 0.18, BOT + 0.16, c["status"], fontsize=8.8,
            ha="left", va="center", color=c["head"], fontweight="bold")

# Arrows from shared base to cards
for c in CARDS:
    cx = c["x"] + 2.6
    arrow(ax, (cx, 7.48), (cx, TOP + 0.01), color=c["head"], lw=2.0)

# Bottom band
box(ax, 0.35, 0.55, 16.1, 1.18, FOOT_FACE, "#666666", lw=1.2)
ax.text(0.55, 1.50, "数学问题汇总：labs/MATH_QUESTIONS.md", fontsize=12.0,
        fontweight="bold", ha="left", va="center", color="#1a1a1a")
ax.text(0.55, 1.18,
        "P0 13 条（CC 1 · P1 5 · P2 3 · P3 4）· P1 16 条 · 完整清单见各目录 math_questions.md",
        fontsize=9.8, ha="left", va="center", color="#333333")
ax.text(0.55, 0.93,
        "最需作者判断：批量全局 QPEP 前向；P1 模式重叠；P2 ALM 定位/Clarke 导数；P3 GNC 全局性/6D 变号",
        fontsize=9.6, ha="left", va="center", color="#333333")
ax.text(0.55, 0.68,
        "证据分级：已证明 / 数值验证 / 作者预实验（待复现）/ 待验证假设。本页是研究初稿，不是投稿版；"
        "定理状态与数字以各目录 README 为准。",
        fontsize=9.3, ha="left", va="center", color="#555555")

ax.text(0.35, 0.22,
        "ds4syk/dsh-superagent-playground · agent/three-paper-tracks-95244014fb3c · "
        "生成脚本 labs/make_overview.py",
        fontsize=8.5, ha="left", va="center", color="#777777")

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=200, facecolor="white", bbox_inches="tight", pad_inches=0.15)
print("wrote", OUT, OUT.stat().st_size)
