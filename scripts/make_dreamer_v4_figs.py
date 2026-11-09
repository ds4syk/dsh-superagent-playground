#!/usr/bin/env python3
"""Generate the DreamerV4-vs-V3 comparison figures.

Run with the repo's plotting environment, e.g.:
  ~/miniconda3/envs/jev/bin/python scripts/make_dreamer_v4_figs.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib import font_manager as fm

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "kb", "dreamer-v4-vs-v3", "assets")
VIS = os.path.join(ROOT, "docs", "visuals")
os.makedirs(OUT, exist_ok=True)
os.makedirs(VIS, exist_ok=True)

FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
fm.fontManager.addfont(FONT)
plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"] = False

BG = "#F7F9FC"
INK = "#202124"
MUTED = "#5F6368"
V3 = "#1A73E8"
V3L = "#E8F0FE"
V4 = "#E8710A"
V4L = "#FEF1E6"
GREEN = "#188038"
GREENL = "#E6F4EA"
RED = "#C5221F"
REDL = "#FCE8E6"
GREY = "#9AA0A6"
GREYL = "#F1F3F4"


def canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    fig.patch.set_facecolor(BG)
    return fig, ax


def box(ax, x, y, w, h, text, fc, ec=MUTED, fs=12, bold=False, tc=INK, lw=1.3, radius=0.10):
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.04,rounding_size={radius}",
        fc=fc, ec=ec, lw=lw,
    )
    ax.add_patch(p)
    ax.text(
        x + w / 2, y + h / 2, text,
        ha="center", va="center", fontsize=fs, color=tc,
        fontweight="bold" if bold else "normal", linespacing=1.35,
    )
    return p


def arrow(ax, x1, y1, x2, y2, color=MUTED, lw=1.4, style="-|>", rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle=style, mutation_scale=14,
        lw=lw, color=color, shrinkA=2, shrinkB=2, linestyle=ls,
        connectionstyle=f"arc3,rad={rad}",
    ))


def title(ax, x, y, text, size=19, color=INK, weight="bold", ha="center"):
    ax.text(x, y, text, ha=ha, va="center", fontsize=size, color=color, fontweight=weight)


def chip(ax, x, y, w, h, text, fc, ec=MUTED, fs=10.5, tc=INK):
    box(ax, x, y, w, h, text, fc, ec=ec, fs=fs, tc=tc, radius=0.16)


# ----------------------------------------------------------------------------
# Figure 1: one-page paradigm shift
# ----------------------------------------------------------------------------
def fig1():
    W, H = 16, 10
    fig, ax = canvas(W, H)
    title(ax, W / 2, 9.55, "DreamerV4 vs DreamerV3：换的是范式，不是超参", 21)
    ax.text(W / 2, 9.05,
            "V3 = 在线交互 + RSSM + 从零 RL      |      V4 = 离线视频 + 生成式世界模型 + 先 BC 后想象 RL",
            ha="center", va="center", fontsize=12, color=MUTED)

    ax.text(3.85, 8.45, "DreamerV3（2023 / Nature 2025）", ha="center", va="center",
            fontsize=15, color=V3, fontweight="bold")
    ax.text(12.15, 8.45, "DreamerV4（2025）", ha="center", va="center",
            fontsize=15, color=V4, fontweight="bold")

    x0, w0 = 0.45, 6.8
    box(ax, x0, 7.45, w0, 0.75, "在线环境（Minecraft）\n64×64 图像 + 库存/装备/生命向量", V3L, ec=V3, fs=11.5)
    arrow(ax, x0 + w0 / 2, 7.45, x0 + w0 / 2, 7.15)
    box(ax, x0, 6.35, w0, 0.8, "Replay Buffer\n1.4K 小时在线经验 · 100M 环境步", V3L, ec=V3, fs=11.5)
    arrow(ax, x0 + w0 / 2, 6.35, x0 + w0 / 2, 6.05)
    box(ax, x0, 5.05, w0, 1.0, "RSSM 世界模型\nh 8192（block GRU） · z 32×64 categorical · KL/free bits", V3L, ec=V3, fs=11.5)
    arrow(ax, x0 + w0 / 2, 5.05, x0 + w0 / 2, 4.75)
    box(ax, x0, 3.95, w0, 0.8, "在想象中 rollout 15 步\nλ-return（γ=0.997，λ=0.95）", V3L, ec=V3, fs=11.5)
    arrow(ax, x0 + w0 / 2, 3.95, x0 + w0 / 2, 3.65)
    box(ax, x0, 2.80, w0, 0.85, "Actor-Critic 从零学\nREINFORCE + return norm + entropy", V3L, ec=V3, fs=11.5)
    chip(ax, x0, 2.15, w0, 0.42, "200M 参数 · 1×A100 · Minecraft 8.9 GPU 天", GREYL, ec=GREY, fs=10.5)

    x1, w1 = 8.75, 6.8
    box(ax, x1, 7.45, w1, 0.75, "离线 VPT 数据集（2541 小时）\n360×640 像素 · 23 键 + 121 鼠标类", V4L, ec=V4, fs=11.5)
    arrow(ax, x1 + w1 / 2, 7.45, x1 + w1 / 2, 7.15)
    box(ax, x1, 6.35, w1, 0.8, "Causal Tokenizer（0.4B）\n960 patch → 512×16 → 256×32 token", V4L, ec=V4, fs=11.5)
    arrow(ax, x1 + w1 / 2, 6.35, x1 + w1 / 2, 6.05)
    box(ax, x1, 4.95, w1, 1.05, "Dynamics Transformer（1.6B）\nshortcut forcing · 每帧 4 步 · 192 帧上下文", V4L, ec=V4, fs=11.5)
    arrow(ax, x1 + w1 / 2, 4.95, x1 + w1 / 2, 4.65)
    box(ax, x1, 3.80, w1, 0.85, "Agent Tokens：BC + reward\n20 任务 embedding · MTP 长度 8", V4L, ec=V4, fs=11.5)
    arrow(ax, x1 + w1 / 2, 3.80, x1 + w1 / 2, 3.50)
    box(ax, x1, 2.65, w1, 0.85, "PMPO 想象 RL\n冻结 transformer · reverse KL 到 BC prior", V4L, ec=V4, fs=11.5)
    chip(ax, x1, 2.15, w1, 0.42, "2B 参数 · 256–1024 TPU-v5p · 0 环境交互 · 21 FPS", GREYL, ec=GREY, fs=10.5)

    ax.text(W / 2, 8.0, "主要\n改动", ha="center", va="center", fontsize=11, color=RED, fontweight="bold")
    changes = [
        "RSSM →\ntokenizer + transformer",
        "KL/重建 →\nshortcut forcing",
        "在线 → 离线\n+ BC 先验",
        "200M / 1 GPU →\n2B / TPU 集群",
    ]
    ys = [7.35, 6.25, 5.15, 4.05]
    for txt, y in zip(changes, ys):
        ax.text(W / 2, y, txt, ha="center", va="center", fontsize=8.8, color=RED,
                bbox=dict(boxstyle="round,pad=0.28", fc=REDL, ec=RED, lw=0.8), linespacing=1.25)

    by, bh = 0.30, 1.55
    box(ax, 0.35, by, 7.55, bh, "", GREENL, ec=GREEN, fs=1)
    ax.text(0.55, by + bh - 0.25, "不变的部分", fontsize=12.5, color=GREEN, fontweight="bold", va="center")
    ax.text(0.55, by + bh - 0.62,
            "• 都在想象中训练行为      • λ-return、γ=0.997\n"
            "• reward/value 用 symexp twohot      • 策略不反传世界模型\n"
            "• 测试时都不做前瞻规划      • Minecraft 钻石作为试金石",
            fontsize=10.3, color=INK, va="top", linespacing=1.45)

    box(ax, 8.25, by, 7.4, bh, "", REDL, ec=RED, fs=1)
    ax.text(8.45, by + bh - 0.25, "代价与边界", fontsize=12.5, color=RED, fontweight="bold", va="center")
    ax.text(8.45, by + bh - 0.62,
            "• 2B 预训练 + TPU 集群：不是同算力对比\n"
            "• 上下文仅 9.6 秒，库存预测仍不精确\n"
            "• 论文 v1，无官方代码，未独立复现",
            fontsize=10.3, color=INK, va="top", linespacing=1.45)

    ax.text(0.35, 0.10, "来源：DreamerV3 arXiv:2301.04104v2 / Nature 2025；DreamerV4 arXiv:2509.24527v1；V3 官方代码。"
                        "  V4 数字为论文声称，未独立复现。",
            fontsize=7.8, color=GREY, va="center")
    fig.savefig(os.path.join(OUT, "fig1_overview.png"), dpi=200, bbox_inches="tight", facecolor=BG)
    fig.savefig(os.path.join(VIS, "2026-11-09-dreamer-v4-vs-v3.png"), dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Figure 2: world models side by side
# ----------------------------------------------------------------------------
def fig2():
    W, H = 16, 10
    fig, ax = canvas(W, H)
    title(ax, W / 2, 9.55, "世界模型对照：RSSM vs causal tokenizer + interactive dynamics", 19)
    ax.text(W / 2, 9.10, "上半：V3 每步压成小状态，单步推演；下半：V4 先压成连续 token，再用 shortcut forcing 多步生成",
            ha="center", va="center", fontsize=10.5, color=MUTED)

    # V3
    ax.text(0.35, 8.55, "DreamerV3 · RSSM（200M）", fontsize=14, color=V3, fontweight="bold")
    box(ax, 0.35, 7.55, 1.85, 0.80, "x_t\n64×64×3\n+ 库存向量", V3L, ec=V3, fs=10.2)
    box(ax, 2.45, 7.55, 1.75, 0.80, "CNN\nEncoder", V3L, ec=V3, fs=10.5)
    box(ax, 4.45, 7.55, 1.65, 0.80, "h_t\n8192\n(block GRU)", V3L, ec=V3, fs=10.0)
    box(ax, 6.35, 7.55, 1.65, 0.80, "z_t\n32×64\ncategorical", V3L, ec=V3, fs=10.0)
    box(ax, 8.30, 7.95, 2.15, 0.55, "reward head  r_hat_t", V3L, ec=V3, fs=9.6)
    box(ax, 8.30, 7.25, 2.15, 0.55, "continue head  c_hat_t", V3L, ec=V3, fs=9.6)
    box(ax, 8.30, 6.55, 2.15, 0.55, "CNN Decoder  x_hat_t", V3L, ec=V3, fs=9.6)
    box(ax, 10.75, 7.05, 4.40, 0.95, "损失\n重建 + reward + continue + KL", V3L, ec=V3, fs=9.6)

    for x in [2.20, 4.20, 6.10]:
        arrow(ax, x, 7.95, x + 0.25, 7.95, color=V3)
    for yy in [8.22, 7.52, 6.82]:
        arrow(ax, 8.00, 7.95, 8.30, yy, color=V3)
    for yy in [8.22, 7.52, 6.82]:
        arrow(ax, 10.45, yy, 10.75, 7.52, color=V3)

    box(ax, 4.45, 5.85, 3.8, 0.85, "posterior q(z|h,x) ↔ prior p(z|h)\nKL balancing + free bits 1 nat + 1% unimix",
        GREYL, ec=GREY, fs=9.2)
    arrow(ax, 5.27, 7.55, 5.82, 6.70, color=GREY, ls="--")
    arrow(ax, 7.17, 7.55, 6.88, 6.70, color=GREY, ls="--")
    ax.text(8.55, 5.95, "训练时：重建项保留像素信息，KL 项让 latent 可预测；\n测试时只走 prior 一步，不解码像素。",
            fontsize=9.2, color=MUTED, va="center", linespacing=1.35)

    ax.plot([0.35, 15.65], [5.35, 5.35], color="#DADCE0", lw=1.2)

    # V4
    ax.text(0.35, 4.95, "DreamerV4 · causal tokenizer + interactive dynamics（2B）", fontsize=14, color=V4, fontweight="bold")
    box(ax, 0.35, 3.55, 1.95, 0.85, "视频帧\n384×640\n20 FPS", V4L, ec=V4, fs=9.8)
    box(ax, 2.60, 3.55, 1.95, 0.85, "patch 16×16\n960 tokens", V4L, ec=V4, fs=9.8)
    box(ax, 4.85, 3.55, 2.15, 0.85, "Causal Tokenizer\n512×16 → 256×32", V4L, ec=V4, fs=9.6)
    box(ax, 7.30, 3.55, 2.55, 0.85, "Dynamics Transformer\nblock-causal · GQA\n时间层每 4 层", V4L, ec=V4, fs=9.4)
    box(ax, 10.15, 3.55, 1.85, 0.85, "z_1（预测）\n256×32", V4L, ec=V4, fs=10.0)
    box(ax, 12.30, 3.55, 1.95, 0.85, "Decoder\n（仅训练/可视化）", GREYL, ec=GREY, fs=9.2)

    box(ax, 7.30, 2.20, 2.55, 0.80, "动作 a_t\n23 键 + 121 鼠标类\n→ S_a tokens", V4L, ec=V4, fs=9.2)
    box(ax, 10.15, 2.20, 1.85, 0.80, "τ, d\nK=4 步\nτ_ctx=0.1", V4L, ec=V4, fs=9.8)
    arrow(ax, 7.30 + 1.275, 3.00, 7.30 + 1.275, 3.55, color=V4)
    arrow(ax, 10.15 + 0.925, 3.00, 10.15 + 0.925, 3.55, color=V4)
    for x in [2.30, 4.55, 7.00, 10.00]:
        arrow(ax, x, 3.97, x + 0.30, 3.97, color=V4)
    arrow(ax, 12.00, 3.97, 12.30, 3.97, color=GREY)

    box(ax, 2.25, 1.05, 6.3, 0.75,
        "tokenizer：MSE + 0.2×LPIPS + MAE（p~U(0,0.9)）\n各损失项按 running RMS 归一化",
        V4L, ec=V4, fs=9.0)
    box(ax, 8.85, 1.05, 6.4, 0.75,
        "dynamics：shortcut forcing（flow + bootstrap）\nx-prediction，ramp w(τ)=0.9τ+0.1，K=4",
        V4L, ec=V4, fs=9.0)
    ax.text(0.35, 0.55,
            "关键差别：V4 的 dynamics 不经过 decoder、不在像素空间算损失；decoder 只服务 tokenizer 与可视化。",
            fontsize=9.4, color=MUTED, va="center")
    fig.savefig(os.path.join(OUT, "fig2_world_model.png"), dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Figure 3: training pipelines
# ----------------------------------------------------------------------------
def fig3():
    W, H = 15, 8.6
    fig, ax = canvas(W, H)
    title(ax, W / 2, 8.15, "训练流程对照：V3 单阶段在线闭环 vs V4 三阶段离线流程", 18)
    ax.text(W / 2, 7.72, "两者都在想象里学行为；差别在数据来源、世界模型是否预训练、策略是否先模仿",
            ha="center", va="center", fontsize=10.5, color=MUTED)

    ax.text(0.35, 7.20, "DreamerV3", fontsize=13.5, color=V3, fontweight="bold")
    box(ax, 0.35, 5.95, 3.1, 0.70, "环境交互\n64×64 + 库存", V3L, ec=V3, fs=10.2)
    box(ax, 0.35, 4.80, 3.1, 0.70, "Replay Buffer\n在线、随策略变化", V3L, ec=V3, fs=10.2)
    box(ax, 0.35, 3.65, 3.1, 0.70, "世界模型更新\n重建 + reward + continue + KL", V3L, ec=V3, fs=9.4)
    box(ax, 0.35, 2.50, 3.1, 0.70, "想象 15 步\n从 replay 状态出发", V3L, ec=V3, fs=10.2)
    box(ax, 0.35, 1.35, 3.1, 0.70, "Actor-Critic 更新\nreturn norm + entropy", V3L, ec=V3, fs=9.6)
    for y in [5.95, 4.80, 3.65, 2.50]:
        arrow(ax, 1.90, y, 1.90, y - 0.45, color=V3)
    arrow(ax, 1.90, 1.35, 1.90, 1.05, color=V3)
    arrow(ax, 1.90, 1.05, 4.10, 1.05, color=V3, style="-")
    arrow(ax, 4.10, 1.05, 4.10, 6.30, color=V3, style="-")
    arrow(ax, 4.10, 6.30, 3.45, 6.30, color=V3)
    ax.text(4.25, 3.6, "数\n据\n回\n流", fontsize=8.5, color=V3, va="center", linespacing=1.2)
    ax.text(0.35, 0.55, "单阶段：数据、世界模型、策略一起进步；策略从零开始。",
            fontsize=9.6, color=MUTED)

    ax.plot([5.05, 5.05], [0.40, 7.45], color="#DADCE0", lw=1.2)

    ax.text(5.45, 7.20, "DreamerV4", fontsize=13.5, color=V4, fontweight="bold")
    sx = [5.45, 8.55, 11.65]
    sw = 2.85
    headers = ["① WM 预训练", "② Agent finetuning", "③ Imagination RL"]
    for x, h in zip(sx, headers):
        ax.text(x + sw / 2, 6.75, h, ha="center", va="center", fontsize=12, color=V4, fontweight="bold")

    box(ax, sx[0], 4.55, sw, 1.65,
        "离线视频 2.5K h\n+ 少量动作标签\n\ntokenizer（MSE+LPIPS+MAE）\n+ dynamics（shortcut）",
        V4L, ec=V4, fs=9.3)
    box(ax, sx[1], 4.55, sw, 1.65,
        "插入 agent tokens\n+ task embedding\n\nBC + reward（MTP 8）\n+ dynamics loss",
        V4L, ec=V4, fs=9.3)
    box(ax, sx[2], 4.55, sw, 1.65,
        "从数据上下文 rollout\nreward/value/λ-return\n\nPMPO + reverse KL 到 BC prior\n只更新 policy/value 头",
        V4L, ec=V4, fs=9.0)
    for i in range(2):
        arrow(ax, sx[i] + sw, 5.37, sx[i + 1] - 0.10, 5.37, color=V4, lw=1.8)

    box(ax, 5.45, 2.95, 2.85, 1.05, "更新：tokenizer\n+ dynamics\n（全量）", GREENL, ec=GREEN, fs=9.4)
    box(ax, 8.55, 2.95, 2.85, 1.05, "更新：agent 头\n+ 主干继续训\n（保持视频损失）", GREENL, ec=GREEN, fs=9.4)
    box(ax, 11.65, 2.95, 2.85, 1.05, "更新：policy/value\n冻结：transformer", REDL, ec=RED, fs=9.4)
    arrow(ax, 6.87, 4.55, 6.87, 4.00, color=GREY, ls="--")
    arrow(ax, 9.97, 4.55, 9.97, 4.00, color=GREY, ls="--")
    arrow(ax, 13.07, 4.55, 13.07, 4.00, color=GREY, ls="--")

    box(ax, 5.45, 1.55, 8.95, 0.85,
        "数据：20 任务 + 50% 均匀 / 50% 相关序列；BC 损失只加在相关序列，dynamics 损失只加在均匀序列\n"
        "奖励：VPT 事件标注的稀疏二值奖励 → task-conditioned reward head",
        GREYL, ec=GREY, fs=9.1)
    ax.text(5.45, 1.12, "阶段传递：①→② 预训练权重；②→③ 行为克隆先验。",
            fontsize=8.8, color=MUTED)
    ax.text(5.45, 0.55, "三阶段串行：先有世界模型，再有策略先验，最后才在想象里做 RL。",
            fontsize=9.6, color=MUTED)

    ax.text(W - 0.35, 0.12, "V4 结构按 arXiv:2509.24527v1 Algorithm 1；V3 按官方代码 agent.py / configs.yaml。",
            fontsize=7.6, color=GREY, ha="right")
    fig.savefig(os.path.join(OUT, "fig3_training.png"), dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
    print("wrote figures to", OUT)
