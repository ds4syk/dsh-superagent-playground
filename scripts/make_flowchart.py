import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib import font_manager as fm

FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
fm.fontManager.addfont(FONT)
plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(figsize=(16, 11))
ax.set_xlim(0, 16); ax.set_ylim(0, 11); ax.axis("off")

C_BG = "#F7F9FC"; C_SRC = "#E8F0FE"; C_JEV = "#FFF1E0"; C_OUT = "#E6F4EA"; C_WARN = "#FCE8E6"
fig.patch.set_facecolor(C_BG)

def box(x, y, w, h, text, fc, fs=13, bold=False, ec="#5F6368"):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.12",
                       fc=fc, ec=ec, lw=1.4)
    ax.add_patch(p)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", linespacing=1.35)
    return (x, y, w, h)

def arrow(x1, y1, x2, y2, text="", color="#5F6368"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=18,
                                 lw=1.5, color=color, shrinkA=0, shrinkB=0))
    if text:
        ax.text((x1+x2)/2, (y1+y2)/2 + 0.12, text, ha="center", va="bottom",
                fontsize=10, color=color)

# title
ax.text(8, 10.55, "JEV 如何“生成”候选动作：它不生成，只对候选打分/选择",
        ha="center", va="center", fontsize=18, fontweight="bold", color="#202124")

# top: state + candidate sources
box(0.5, 8.9, 4.2, 1.0, "环境状态 $s_t$\n（文本/结构化字段）", C_SRC, 14, True)
box(5.6, 8.9, 4.4, 1.2, "候选动作来源（外部）\n• 当前可执行命令（本实验）\n• 策略/LLM 提议\n• 技能库/规划器", C_SRC, 12)
box(11.0, 8.9, 4.4, 1.0, "候选集 $A_t=\\{a_1,\\dots,a_k\\}$\n动作文本 + 选项", C_SRC, 13, True)

arrow(4.7, 9.4, 5.6, 9.4)
arrow(10.0, 9.4, 11.0, 9.4)
arrow(12.2, 8.9, 12.2, 8.2)

# three branches
ax.text(2.5, 7.95, "① 直接 choice", fontsize=14, fontweight="bold", color="#B06000")
ax.text(7.2, 7.95, "② 逐动作 score", fontsize=14, fontweight="bold", color="#B06000")
ax.text(12.2, 7.95, "③ 世界模型前瞻", fontsize=14, fontweight="bold", color="#B06000")

box(0.5, 6.6, 3.8, 1.15, "JEV：state + 问题\noptions = 全部候选动作", C_JEV, 12)
box(0.5, 5.35, 3.8, 0.95, "输出：每个动作的概率\n$p(a\\mid s_t)$", C_OUT, 12)
box(0.5, 4.15, 3.8, 0.85, "取 argmax / top-k", C_OUT, 12, True)

box(4.9, 6.6, 3.9, 1.15, "对每个 $a_i$：JEV\n“执行它是否推进目标？”\n（yes/no）", C_JEV, 12)
box(4.9, 5.35, 3.9, 0.95, "输出：$P(\\mathrm{yes}\\mid s_t,a_i)$\n作为 $\\mathrm{score}(a_i)$", C_OUT, 12)
box(4.9, 4.15, 3.9, 0.85, "按分数排序 / 取最高", C_OUT, 12, True)

box(9.9, 6.6, 5.2, 1.15, "对每个 $a_i$：JEV 问 8 个选择题\n是否执行 + 7 个状态变量", C_JEV, 12)
box(9.9, 5.25, 5.2, 1.05, "拼出预测状态 $\\hat s_{t+1}$\n（执行失败则状态不变）", C_OUT, 12)
box(9.9, 4.05, 5.2, 0.95, "手写效用 $U(\\hat s_{t+1})$\n按效用排序 / 取最高", C_OUT, 12, True)

arrow(2.4, 6.6, 2.4, 6.3)
arrow(2.4, 5.35, 2.4, 5.0)
arrow(6.85, 6.6, 6.85, 6.3)
arrow(6.85, 5.35, 6.85, 5.0)
arrow(12.5, 6.6, 12.5, 6.3)
arrow(12.5, 5.25, 12.5, 5.0)

# merge
box(5.6, 2.95, 5.0, 0.75, "选定动作 $a_t$（三选一）", "#D2E3FC", 14, True)
for x in (2.4, 6.85, 12.5):
    arrow(x, 4.15, 8.1, 3.7)

arrow(8.1, 2.95, 8.1, 2.55)
box(4.9, 1.75, 6.4, 0.8, "执行 $a_t$ → 观测真实 $s_{t+1}$", C_OUT, 13, True)
arrow(11.3, 2.15, 13.5, 2.15)
box(13.2, 1.7, 2.5, 0.9, "记录 $(s_t,a_t)$\n重规划", C_OUT, 11)
arrow(13.2, 2.15, 13.2, 3.6)
arrow(13.2, 3.6, 12.2, 3.6)

# distillation
box(0.5, 0.35, 7.3, 1.0, "离线蒸馏：用 (s_t,a_t) 训练 ~1M 学生\n部署时 0 次 JEV 调用", "#EADCF8", 12, True)
arrow(4.9, 1.75, 4.1, 1.35)

# side note
box(8.3, 0.35, 7.2, 1.0, "关键：JEV 的选项必须预先枚举；\n候选从哪来是工程问题，不是 JEV 自己“想”出来的。", C_WARN, 11)

plt.tight_layout()
out = "kb/assets/fig5_jev_action_flow.png"
plt.savefig(out, dpi=170, bbox_inches="tight", facecolor=fig.get_facecolor())
print("wrote", out)
