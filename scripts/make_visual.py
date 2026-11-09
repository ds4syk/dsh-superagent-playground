"""One-page visual for the SDPR main-table experiment design note."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import matplotlib.font_manager as fm

FONT = "Noto Sans CJK SC"
plt.rcParams["font.family"] = FONT
plt.rcParams["axes.unicode_minus"] = False

fig = plt.figure(figsize=(16, 9), dpi=200)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 160)
ax.set_ylim(0, 90)
ax.axis("off")

INK = "#1f2937"
MUTED = "#4b5563"
BLUE = "#1d4ed8"
GREEN = "#047857"
ORANGE = "#b45309"
LIGHT = "#f3f4f6"
GRID = "#d1d5db"


def rbox(x, y, w, h, fc=LIGHT, ec=GRID, lw=1.0, label=None, lc=None, lwbar=0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.2",
                                facecolor=fc, edgecolor=ec, linewidth=lw))
    if lwbar and lc:
        ax.add_patch(Rectangle((x + 0.4, y + 0.8), 1.0, h - 1.6, facecolor=lc, edgecolor="none"))


def txt(x, y, s, size=10, color=INK, weight="normal", ha="left", va="center", wrap=False):
    ax.text(x, y, s, fontsize=size, color=color, fontweight=weight, ha=ha, va=va, linespacing=1.35)


# ---------- title ----------
txt(3, 87.2, "基于 SDPR 写新方法：主表实验怎么设计", 19, INK, "bold")
txt(3, 83.4, "先定创新点类型 → 再定数据集与 baseline → 最后套四张表；单一主表不足以支撑“可认证”卖点", 10.5, MUTED)

# ---------- left column ----------
txt(3, 79.6, "① 先定创新点类型", 12.5, INK, "bold")

cards = [
    (64.0, 14.4, BLUE, "A  层/求解器改进",
     "tightness、效率、鲁棒核、非紧 fallback、稀疏/GPU",
     "数据：合成 PnP、位姿图/旋转同步、标定、BAL/Replica",
     "主 baseline：SDPRLayers、PnP*、Theseus、EPro-PnP"),
    (47.0, 14.4, GREEN, "B  新任务 / 新前端",
     "spacecraft、事件相机、时序/视频、多模态",
     "数据：SPEED+（主）+ SPEED / prisma25 / SPADES",
     "主 baseline：SPNv2、KRN、UDA、EPro-PnP、PnP*"),
    (30.0, 14.4, ORANGE, "C  新表征桥接",
     "V-JEPA2 / 视觉基础模型 → 几何输出",
     "数据：合成旋转视频 + KITTI / ETH3D / TUM",
     "主 baseline：direct、two-stage、chordal、SDPR"),
]
for (y, h, col, head, l1, l2, l3) in cards:
    rbox(3, y, 47, h, fc="#ffffff", ec=GRID)
    ax.add_patch(Rectangle((3.4, y + 1.0), 1.1, h - 2.0, facecolor=col, edgecolor="none"))
    txt(6.4, y + h - 3.0, head, 11, col, "bold")
    txt(6.4, y + h - 6.5, l1, 8.6, MUTED)
    txt(6.4, y + h - 9.6, l2, 8.6, MUTED)
    txt(6.4, y + h - 12.6, l3, 8.6, MUTED)

txt(3, 26.4, "默认建议：已有 SPEED+ 草稿与 SDPR 复现代码，优先走 B，", 8.8, MUTED)
txt(3, 24.4, "用 A 的表格证明“可认证/效率”不是空话。", 8.8, MUTED)

# ---------- middle column ----------
txt(54, 79.6, "② 主表四件套", 12.5, INK, "bold")

tables = [
    (64.0, 14.4, "T1  主比较",
     "行：方法（按类分组）  列：Lightbox / Sunlamp",
     "指标：ET (m) ↓、ER (°) ↓、Epose ↓，mean ± std",
     "同训练集 · 同 split · 无 crop · 标注 best/second"),
    (47.0, 14.4, "T2  尾部与鲁棒",
     "RMSE、max、fail>30° / >90°；P90/P95/P99",
     "噪声 σ、外点率、遮挡、距离、初始化 sweep",
     "回答：平均好之外，灾难性失败是否更少"),
    (30.0, 14.4, "T3  证书与效率",
     "tight rate、certificate rate、suboptimality",
     "solver ms（median/P95）、FPS、params、显存",
     "回答：可认证性的代价是否可接受"),
    (13.0, 14.4, "T4  消融",
     "local solver ↔ SDPR、random init、no precision",
     "no depth、first-stage、chordal/sparse、层数",
     "每个设计点单独关闭，报 Δ 与 tight rate"),
]
for (y, h, head, l1, l2, l3) in tables:
    rbox(54, y, 47, h, fc="#ffffff", ec=GRID)
    txt(56, y + h - 3.0, head, 11, INK, "bold")
    txt(56, y + h - 6.4, l1, 8.6, MUTED)
    txt(56, y + h - 9.4, l2, 8.6, MUTED)
    txt(56, y + h - 12.4, l3, 8.6, MUTED)

# ---------- right column ----------
txt(106, 79.6, "③ baseline 三层 + ④ 数据集", 12.5, INK, "bold")

# pyramid-ish three tiers
tiers = [
    (64.0, 14.0, "#eef2ff", BLUE, "领域 SOTA（必须 3–5 个）",
     "SPNv2-H / SPNv2-E、KRN、UDA、StructGCN、竞赛前排"),
    (49.0, 13.0, "#ecfdf5", GREEN, "可微 / 可认证几何",
     "SDPRLayers、PnP*、EPro-PnP、BPnP、Theseus"),
    (35.0, 12.0, "#fff7ed", ORANGE, "经典几何（强基线）",
     "EPnP + LM、SQPnP、P3P + RANSAC、SVD/ICP"),
]
for (y, h, fc, ec, head, body) in tiers:
    rbox(106, y, 51, h, fc=fc, ec=ec)
    txt(108, y + h - 3.2, head, 10.5, ec, "bold")
    txt(108, y + h - 7.2, body, 8.4, MUTED)
    txt(108, y + h - 10.4, "同一训练数据、同一 split、同一预处理", 8.0, "#6b7280")

# dataset table
txt(106, 30.0, "数据集矩阵", 10.5, INK, "bold")
rows = [
    ("SPEED+", "spacecraft", "主表：HIL 双域 + Epose", "必选"),
    ("SPEED / prisma25", "跨数据集", "泛化/真实飞行", "推荐"),
    ("SPADES / 事件", "光照鲁棒", "可选第二域", "按方法定"),
    ("In-The-Dark", "SDPR 原实验", "层改进的对照", "A 类必选"),
    ("BAL / Replica", "全局 SfM", "可扩展性", "A 类可选"),
]
x0, y0 = 106, 28.2
colw = [17, 14, 15, 5]
for i, r in enumerate(rows):
    yy = y0 - i * 3.9
    if i == 0:
        for j, v in enumerate(r):
            txt(x0 + sum(colw[:j]) + 0.4, yy, v, 8.2, "#6b7280", "bold")
    else:
        for j, v in enumerate(r):
            txt(x0 + sum(colw[:j]) + 0.4, yy, v, 8.2, MUTED)

# ---------- bottom band ----------
rbox(3, 1.2, 154, 7.0, fc="#111827", ec="#111827")
txt(5, 5.7, "主表 =  领域指标证明“有效”  +  证书/效率证明“独特”  +  消融证明“设计”", 11.5, "#ffffff", "bold")
txt(5, 3.2, "只报一张精度表，审稿人会问：可认证性到底买到了什么？", 9.0, "#d1d5db")

fig.savefig("docs/visuals/2026-11-09-sdpr-main-table-design.png",
            facecolor="white", bbox_inches="tight", pad_inches=0.15)
print("saved")
