"""One-page visual summary for the SDPR forward-network/dataset ticket.

Repo AGENTS.md asks for a one-page visual in docs/visuals/ and embedding it in the PR body.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams["font.family"] = ["Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

HERE = Path(__file__).resolve().parent.parent
OUT = HERE / "docs" / "visuals" / "2026-11-09-sdpr-forward-net.png"

TRAIN = "#F6C28B"
NOLEARN = "#A8C8E8"
TENSOR = "#F2A6A6"
LOSS = "#A9D6A0"
EDGE = "#333333"


def box(ax, x, y, w, h, title, body="", face=NOLEARN, title_size=12.5, body_size=9.4):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            linewidth=1.4, edgecolor=EDGE, facecolor=face,
        )
    )
    if body:
        ax.text(x + w / 2, y + h * 0.64, title, ha="center", va="center",
                fontsize=title_size, fontweight="bold", color="#1a1a1a")
        ax.text(x + w / 2, y + h * 0.30, body, ha="center", va="center",
                fontsize=body_size, color="#222222", linespacing=1.3)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center",
                fontsize=title_size, fontweight="bold", color="#1a1a1a")


def arrow(ax, p0, p1, style="-|>", ls="-", color=EDGE, lw=1.8):
    ax.add_patch(
        FancyArrowPatch(
            p0, p1, arrowstyle=style, mutation_scale=16, linewidth=lw,
            color=color, linestyle=ls,
        )
    )


fig, ax = plt.subplots(figsize=(16.8, 9.6), dpi=200)
ax.set_xlim(0, 16.8)
ax.set_ylim(0, 9.6)
ax.axis("off")

# Header
ax.text(0.35, 9.15, "SDPR 可微优化层：前向网络与数据集 · 一页速览",
        fontsize=23, fontweight="bold", ha="left", va="center", color="#1a1a1a")
ax.text(0.35, 8.72, "SDPRLayers (IEEE T-RO 2025) 真实关键点定位实验｜读论文 + 读代码，未跑数值复现",
        fontsize=11.5, ha="left", va="center", color="#444444")

# Callout
ax.add_patch(FancyBboxPatch((0.35, 7.78), 16.1, 0.62,
                            boxstyle="round,pad=0.02,rounding_size=0.08",
                            linewidth=1.2, edgecolor="#2E6B3A", facecolor="#EAF3E7"))
ax.text(0.55, 8.09,
        "SDPRLayer 本身没有可学习参数；前端是 VGG16 + 2×UNet 解码器；数据是 UTIAS In-The-Dark。",
        fontsize=12.2, ha="left", va="center", color="#1a1a1a")

# Flow row 1
box(ax, 0.35, 6.35, 4.55, 1.10, "双目图像（4 张）", "B×12×384×512\n源/目标 × 左/右", TENSOR)
box(ax, 5.35, 6.35, 5.15, 1.10, "VGG16 编码器", "ImageNet 预训练，截到 conv5_3\n五级特征图", TRAIN)
box(ax, 10.95, 6.35, 5.50, 1.10, "2×UNet 解码器 + 描述子", "关键点 logits / 得分（sigmoid）\n960 维逐像素描述子", TRAIN)
arrow(ax, (4.90, 6.90), (5.35, 6.90))
arrow(ax, (10.50, 6.90), (10.95, 6.90))

# Flow row 2
box(ax, 0.35, 4.85, 4.55, 1.10, "KeypointBlock", "16×16 窗口空间 softmax\n→ 768 个 2D 关键点", NOLEARN)
box(ax, 5.35, 4.85, 5.15, 1.10, "MatcherBlock", "dense 描述子 softmax\n→ 伪目标点", NOLEARN)
box(ax, 10.95, 4.85, 5.50, 1.10, "双目逆投影", "3D 点 + 3×3 协方差\nW = Σ⁻¹（σ = 0.5 px）", NOLEARN)
# Connect row 1 to row 2
arrow(ax, (13.70, 6.35), (13.70, 6.10), style="-")
arrow(ax, (13.70, 6.10), (2.62, 6.10), style="-", color="#555555")
arrow(ax, (2.62, 6.10), (2.62, 5.95), style="-|>", color="#555555")
arrow(ax, (4.90, 5.40), (5.35, 5.40))
arrow(ax, (10.50, 5.40), (10.95, 5.40))

# Flow row 3
box(ax, 0.35, 3.35, 4.55, 1.10, "标量 × 矩阵权重", "ZNCC × 两端得分 → w ∈ [0,1]\n再乘 W = Σ⁻¹", NOLEARN)
box(ax, 5.35, 3.35, 5.15, 1.10, "13 维 QCQP", "x = [h; vec(R); t]\nΣ wₖ eₖᵀ Wₖ eₖ", NOLEARN)
box(ax, 10.95, 3.35, 5.50, 1.10, "SDPRLayer → 位姿 T", "Shor SDP（Mosek）\n隐函数微分反向", face=LOSS)
arrow(ax, (13.70, 4.85), (13.70, 4.60), style="-")
arrow(ax, (13.70, 4.60), (2.62, 4.60), style="-", color="#555555")
arrow(ax, (2.62, 4.60), (2.62, 4.45), style="-|>", color="#555555")
arrow(ax, (4.90, 3.90), (5.35, 3.90))
arrow(ax, (10.50, 3.90), (10.95, 3.90))

# Backward dashed arrow: loss/pose -> VGG16 encoder
arrow(ax, (16.45, 3.90), (16.45, 7.58), style="-", ls="--", color="#B03A2E", lw=1.5)
arrow(ax, (16.45, 7.58), (7.92, 7.58), style="-", ls="--", color="#B03A2E", lw=1.5)
arrow(ax, (7.92, 7.58), (7.92, 7.45), style="-|>", ls="--", color="#B03A2E", lw=1.5)
ax.text(16.62, 5.55, "位姿/关键点损失\n经 SDPR 层回传", fontsize=9.2, color="#B03A2E",
        ha="center", va="center", rotation=90, linespacing=1.25)

# Bottom fact panels
ax.add_patch(FancyBboxPatch((0.35, 0.55), 8.05, 2.35,
                            boxstyle="round,pad=0.02,rounding_size=0.08",
                            linewidth=1.2, edgecolor="#1F4E79", facecolor="#F4F7FB"))
ax.text(0.60, 2.62, "网络是什么", fontsize=14, fontweight="bold", ha="left", va="center", color="#1F4E79")
ax.text(0.60, 2.20,
        "• 编码器：VGG16（ImageNet），截到 conv5_3，整段 fine-tune\n"
        "• 解码器：两个 UNet 式解码器（关键点热图 + 得分），从零训练\n"
        "• 描述子：conv1_2/conv2_2/conv3_3/conv4_3 特征拼接 = 960 维\n"
        "• 关键点：每帧 16×16 窗口 → 768 个；dense 匹配 196,608 个目标像素",
        fontsize=10.4, ha="left", va="top", color="#222222", linespacing=1.45)

ax.add_patch(FancyBboxPatch((8.65, 0.55), 7.80, 2.35,
                            boxstyle="round,pad=0.02,rounding_size=0.08",
                            linewidth=1.2, edgecolor="#2E6B3A", facecolor="#F4FAF2"))
ax.text(8.90, 2.62, "数据从哪来", fontsize=14, fontweight="bold", ha="left", va="center", color="#2E6B3A")
ax.text(8.90, 2.20,
        "• UTIAS In-The-Dark（Long-Term Localization and Mapping Dataset）\n"
        "• 39 条 260 m 路线，约 30 小时每小时一条；夜间开前灯\n"
        "• Grizzly 机器人 + Bumblebee XB3 双目：0.24 m 基线，512×384\n"
        "• 论文：100k 训练 / 20k 验证；测试 runs 2/11/16/17/23/28/35",
        fontsize=10.4, ha="left", va="top", color="#222222", linespacing=1.45)

ax.text(0.35, 0.22,
        "证据等级：论文声称 + 代码核对（未跑数值）｜源码：utiasASRL/sdprlayer；deep_learned_visual_features@ba66e408",
        fontsize=8.6, ha="left", va="center", color="#666666")

fig.savefig(OUT, bbox_inches="tight", facecolor="white")
print(f"saved {OUT}")
