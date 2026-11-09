"""Draw the SDPR forward pipeline diagram used in the report."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch

plt.rcParams["font.family"] = ["Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "report" / "figures" / "pipeline.png"

# Colors follow the paper's Fig. 5 convention loosely.
TRAIN = "#F6C28B"   # trainable network
NOLEARN = "#A8C8E8" # no learnable parameters
TENSOR = "#F2A6A6"  # key tensors
LOSS = "#A9D6A0"    # loss
EDGE = "#333333"


def box(ax, x, y, w, h, title, body="", face=TRAIN, title_size=12.5, body_size=9.5):
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.015,rounding_size=0.06",
        linewidth=1.4, edgecolor=EDGE, facecolor=face,
    )
    ax.add_patch(p)
    if body:
        ax.text(x + w / 2, y + h * 0.64, title, ha="center", va="center",
                fontsize=title_size, fontweight="bold", color="#1a1a1a")
        ax.text(x + w / 2, y + h * 0.28, body, ha="center", va="center",
                fontsize=body_size, color="#222222", linespacing=1.35)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center",
                fontsize=title_size, fontweight="bold", color="#1a1a1a",
                linespacing=1.3)


def arrow(ax, p0, p1, style="-|>", ls="-", color=EDGE, lw=1.8, rad=0.0):
    a = FancyArrowPatch(
        p0, p1, arrowstyle=style, mutation_scale=16, linewidth=lw,
        color=color, linestyle=ls, connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(a)


fig, ax = plt.subplots(figsize=(16.2, 8.6), dpi=200)
ax.set_xlim(0, 17.3)
ax.set_ylim(0, 8.9)
ax.axis("off")

# Row 1
box(ax, 0.35, 6.35, 3.25, 1.55, "双目图像（4 张）", "B×12×384×512\n源/目标 × 左/右", TENSOR)
box(ax, 4.05, 6.35, 3.25, 1.55, "VGG16 编码器", "ImageNet 预训练\n截到 conv5_3", TRAIN)
box(ax, 7.75, 6.35, 3.55, 1.55, "2× UNet 式解码器", "关键点 logits + 得分 (sigmoid)\n960-d 逐像素描述子", TRAIN)
box(ax, 11.75, 6.35, 3.85, 1.55, "KeypointBlock", "16×16 窗口空间 softmax\n→ 768 个 2D 关键点", NOLEARN)

for x0, x1 in [(3.60, 4.05), (7.30, 7.75), (11.30, 11.75)]:
    arrow(ax, (x0, 7.12), (x1, 7.12))

# Row 2
box(ax, 0.35, 3.65, 3.25, 1.55, "MatcherBlock", "dense 描述子 softmax\n→ 伪目标点", NOLEARN)
box(ax, 4.05, 3.65, 3.25, 1.55, "双目逆投影", "3D 点 + 3×3 协方差\nW = Σ⁻¹（σ = 0.5 px）", NOLEARN)
box(ax, 7.75, 3.65, 3.55, 1.55, "WeightBlock", "ZNCC × 两帧得分\n→ 标量 w ∈ [0,1]", NOLEARN)
box(ax, 11.75, 3.65, 3.85, 1.55, "组装 Q（13×13）", "x = [h; vec(C); t]\nΣ wₖ eₖᵀ Wₖ eₖ", NOLEARN)

for x0, x1 in [(3.60, 4.05), (7.30, 7.75), (11.30, 11.75)]:
    arrow(ax, (x0, 4.42), (x1, 4.42))

# Connect row 1 to row 2 (Keypoints -> Matcher)
arrow(ax, (13.68, 6.35), (13.68, 5.25), style="-|>", ls="--", color="#555555")
arrow(ax, (13.68, 5.25), (3.60, 5.25), style="-", ls="--", color="#555555")
arrow(ax, (3.60, 5.25), (3.60, 5.20), style="-|>", ls="--", color="#555555")

# Row 3
box(ax, 4.05, 0.85, 3.70, 1.45, "SDPRLayer", "Shor SDP（Mosek）\n前向全局解 + 证书 H ⪰ 0", NOLEARN)
box(ax, 8.30, 0.85, 3.10, 1.45, "位姿 T_trg_src", "R = reshape(x[1:10])ᵀ\nt = −x[10:13]", TENSOR)
box(ax, 11.95, 0.85, 3.65, 1.45, "损失 + 反向", "pose_plane + keypoint_plane\n隐函数微分 → 回传前端", LOSS)

arrow(ax, (13.68, 3.65), (13.68, 2.55), style="-|>")
arrow(ax, (13.68, 2.55), (5.90, 2.55), style="-")
arrow(ax, (5.90, 2.55), (5.90, 2.30), style="-|>")
arrow(ax, (7.75, 1.575), (8.30, 1.575))
arrow(ax, (11.40, 1.575), (11.95, 1.575))

# Backward dashed arrow from loss to the trainable front end, routed outside the boxes
arrow(ax, (15.60, 1.575), (16.35, 1.575), style="-", ls="--", color="#B03A2E", lw=1.5)
arrow(ax, (16.35, 1.575), (16.35, 8.03), style="-", ls="--", color="#B03A2E", lw=1.5)
arrow(ax, (16.35, 8.03), (5.67, 8.03), style="-", ls="--", color="#B03A2E", lw=1.5)
arrow(ax, (5.67, 8.03), (5.67, 7.92), style="-|>", ls="--", color="#B03A2E", lw=1.5)
ax.text(16.62, 4.75, "梯度回传\n（穿 SDPR 层）", fontsize=9.5, color="#B03A2E",
        ha="center", va="center", rotation=90, linespacing=1.3)

# Legend
handles = [
    Patch(facecolor=TRAIN, edgecolor=EDGE, linewidth=1.0),
    Patch(facecolor=NOLEARN, edgecolor=EDGE, linewidth=1.0),
    Patch(facecolor=TENSOR, edgecolor=EDGE, linewidth=1.0),
    Patch(facecolor=LOSS, edgecolor=EDGE, linewidth=1.0),
]
labels = ["可训练网络", "无可学习参数", "关键张量", "损失 / 反向"]
ax.legend(handles, labels, loc="lower left", bbox_to_anchor=(0.012, 0.005),
          ncol=4, frameon=False, fontsize=10.5, handlelength=1.1,
          handleheight=1.0, columnspacing=1.4, handletextpad=0.5)

ax.text(0.35, 8.66, "SDPR 论文真实实验：可微优化层的前向接法", fontsize=16,
        fontweight="bold", ha="left", va="center", color="#1a1a1a")
ax.text(0.35, 8.28, "图像 → 关键点/描述子/得分 → 匹配 → 3D + 协方差 → 权重 → 13 维 QCQP → SDPRLayer → 位姿",
        fontsize=10.5, ha="left", va="center", color="#444444")

fig.savefig(OUT, bbox_inches="tight", facecolor="white")
print(f"saved {OUT}")
