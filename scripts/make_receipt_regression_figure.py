"""One-page visual: why the 👌 Feishu receipt reaction disappeared, and how it was restored."""
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
ax.set_xlim(0, 16)
ax.set_ylim(0, 11)
ax.axis("off")

BG = "#F7F9FC"
GOOD = "#E6F4EA"
BAD = "#FCE8E6"
FIX = "#E8F0FE"
GREY = "#F1F3F4"
fig.patch.set_facecolor(BG)


def box(x, y, w, h, text, fc, fs=12, bold=False, ec="#5F6368"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.12",
                                fc=fc, ec=ec, lw=1.4))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", linespacing=1.4)


def arrow(x1, y1, x2, y2, color="#5F6368"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=16,
                                 lw=1.6, color=color, shrinkA=0, shrinkB=0))


ax.text(8, 10.55, "飞书 OK 回执为什么消失：一次「换构建」的事故链",
        ha="center", va="center", fontsize=19, fontweight="bold", color="#202124")
ax.text(8, 10.12, "结论：不是权限或开关，而是 11-26 切换 AO 时换上的构建缺少回执代码；已把补丁移植回 AO 分支并重建",
        ha="center", va="center", fontsize=11.5, color="#5F6368")

# ---- three builds
ax.text(0.6, 9.35, "三个构建", fontsize=13, fontweight="bold", color="#202124")
box(0.5, 7.55, 4.8, 1.55,
    "11-12 构建（原 live）\n\n回执 reactFeishu 有　AO 执行器 无\n结果：回执正常，最后一条 11-26 01:35:31", GOOD, 11.5)
box(5.6, 7.55, 4.8, 1.55,
    "11-26 构建（AO 分支）\n\n回执 reactFeishu 无　AO 执行器 有\n结果：回执消失，0 条新回执", BAD, 11.5)
box(10.7, 7.55, 4.8, 1.55,
    "11-27 构建（本修复）\n\n回执 reactFeishu 有　AO 执行器 有\n结果：typecheck ✓ · 回执测试 129/129 ✓", FIX, 11.5)

# ---- timeline
ax.text(0.6, 6.7, "时间线（+08:00）", fontsize=13, fontweight="bold", color="#202124")

steps = [
    (0.6, "11-02 19:02", "回执功能实现\n（旧工作树 + 补丁）", GOOD),
    (4.0, "11-12 13:25", "构建 v11-12\n含回执", GOOD),
    (7.4, "11-25 02:41", "feat/ao-executor\n基于 10-31 原型分叉\n不含回执", BAD),
    (10.8, "11-26 01:35", "用 AO 分支构建\n替换 live lib", BAD),
]
for x, t, text, fc in steps:
    box(x, 5.25, 3.1, 0.95, f"{t}\n{text}", fc, 10.5)
for x in (3.7, 7.1, 10.5):
    arrow(x, 5.72, x + 0.3, 5.72)

box(0.6, 3.85, 5.0, 0.95, "11-26 01:47  回滚 AO profile\n但 lib 没有一起回滚", BAD, 10.5)
box(5.9, 3.85, 4.8, 0.95, "11-27 19:29  用户反馈\nOK 回执没了", GREY, 10.5)
box(11.0, 3.85, 4.5, 0.95, "11-27 19:36  移植补丁\n重建并部署", FIX, 10.5)
arrow(3.1, 5.25, 3.1, 4.8)
arrow(8.3, 5.25, 8.3, 4.8)

# ---- mechanism
ax.text(0.6, 3.35, "正常路径", fontsize=13, fontweight="bold", color="#202124")
mech = ["新鲜人类消息\n（5 分钟内）", "Intake\n准入 + @/私聊/工单线程", "Outbox\nkind: reaction", "SourceRegistry.react\n→ 飞书 reaction API", "消息上出现 OK 表情"]
x = 0.6
for i, text in enumerate(mech):
    box(x, 2.15, 2.55, 0.9, text, "#D2E3FC" if i in (1, 2, 3) else GOOD, 9.8)
    if i < len(mech) - 1:
        arrow(x + 2.55, 2.6, x + 2.8, 2.6)
    x += 2.8

ax.text(0.6, 1.45, "断点：11-26 构建的 feishuKind.channel 只有 { allowed, send, link }，没有 react；Outbox 因此直接跳过回执，连失败的 receipt 都不留。",
        fontsize=10.5, color="#A50E0E")

# ---- evidence strip
ax.text(0.6, 0.95, "证据", fontsize=12, fontweight="bold", color="#202124")
ax.text(1.6, 0.98, "outbox 153 条 reaction 全部 sent；最后一条 2026-11-25T17:35:31Z（= 11-26 01:35:31 +08）；此后 0 条。",
        fontsize=9.8, color="#3C4043")
ax.text(1.6, 0.62, "旧构建：reactFeishu + emoji OK 有　|　坏构建：reactFeishu 无、index 中 reaction 0 次　|　新构建：两者都有",
        fontsize=9.8, color="#3C4043")

fig.savefig("/mnt/storage1/syk/.superagent-work/receipt-regression-96f00a1df4c7/docs/visuals/2026-11-27-receipt-regression.png",
            dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
print("ok")
