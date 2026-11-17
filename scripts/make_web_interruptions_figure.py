#!/usr/bin/env python3
"""One-page visual for the 2026-11-17 Web-side interruption diagnosis."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D

FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
FONT_BOLD = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
zh = fm.FontProperties(fname=FONT, size=11)
zh_small = fm.FontProperties(fname=FONT, size=9)
zh_tiny = fm.FontProperties(fname=FONT, size=8)
zh_bold = fm.FontProperties(fname=FONT_BOLD, size=11)
zh_title = fm.FontProperties(fname=FONT_BOLD, size=20)
zh_sub = fm.FontProperties(fname=FONT, size=11)
for p in (FONT, FONT_BOLD):
    fm.fontManager.addfont(p)
plt.rcParams["axes.unicode_minus"] = False

GREEN = "#2e8b57"
BLUE = "#2f6db3"
RED = "#c0392b"
ORANGE = "#d68910"
GRAY = "#6b7280"
LIGHT = "#f4f6f8"

fig = plt.figure(figsize=(16, 10.2), facecolor="white")
gs = fig.add_gridspec(3, 2, height_ratios=[1.35, 1.05, 0.95], hspace=0.42, wspace=0.18,
                      left=0.055, right=0.965, top=0.865, bottom=0.055)

fig.text(0.055, 0.955, "Web 端「中断」的真相：冷读误报，不是工单中断", fontproperties=zh_title, color="#111827")
fig.text(0.055, 0.923, "2026-11-17 · DSH 0.2.1-rc.2 · 个人 web 冷读 superagent 正在写的会话 → 合成的 turn/end interrupted",
         fontproperties=zh_sub, color="#4b5563")

# ---------------- A. timeline ----------------
ax = fig.add_subplot(gs[0, :])
ax.set_facecolor("white")
ax.set_xlim(17.8, 19.25)
ax.set_ylim(-0.85, 3.35)
ax.set_yticks([])
ax.set_xticks([18.0, 18.25, 18.5, 18.75, 19.0, 19.25])
ax.set_xticklabels(["18:00", "18:15", "18:30", "18:45", "19:00", "19:15"], fontproperties=zh_small)
ax.grid(axis="x", color="#e5e7eb", lw=0.8)
for s_ in ("top", "right", "left"):
    ax.spines[s_].set_visible(False)
ax.spines["bottom"].set_color("#9ca3af")

def bar(y, start, end, color, label, sub, textcolor="white"):
    ax.add_patch(FancyBboxPatch((start, y), end - start, 0.36, boxstyle="round,pad=0.0,rounding_size=0.02",
                                fc=color, ec="none", alpha=0.92))
    ax.text((start + end) / 2, y + 0.18, label, ha="center", va="center",
            fontproperties=zh_bold, color=textcolor)
    ax.text((start + end) / 2, y - 0.12, sub, ha="center", va="top", fontproperties=zh_tiny, color=GRAY)

bar(2.25, 18.109, 19.163, BLUE, "四色 worker · turn 1", "18:06:34 → 19:09:50 · completed（1322 步）")
bar(1.35, 18.976, 19.176, GREEN, "π 工单的子 agent · turn 1", "18:58:33 → 19:10:33 · completed")

# observation window
ax.add_patch(Rectangle((19.103, 0.95), 19.152 - 19.103, 1.80, fc=RED, ec="none", alpha=0.08))
ax.axvline(19.103, color=RED, lw=1.2, ls="--")
ax.text(19.128, 3.12, "用户看到「中断」的窗口（19:06–19:09）", ha="center", fontproperties=zh_small, color=RED)
ax.text(19.165, 2.78, "19:09:50 completed", ha="right", fontproperties=zh_tiny, color=GREEN)
ax.text(19.183, 1.88, "19:10:33 completed", ha="right", fontproperties=zh_tiny, color=GREEN)
ax.annotate("两个回合在 19:06 仍开着；\nresume 消息被并入运行中的回合，随后正常完成",
            xy=(19.095, 2.45), xytext=(18.30, 3.00), fontproperties=zh_small, color="#111827",
            arrowprops=dict(arrowstyle="->", color=GRAY, lw=1.0))
ax.text(17.85, -0.62, "19:06:16 个人 web 启动   ·   19:07:57 resume 四色   ·   19:08:09 resume π   ·   两个回合随后 completed",
        fontproperties=zh_small, color=GRAY)

ax.set_title("A · 时间线：用户以为中断的两个回合，实际上一直在运行", loc="left",
             fontproperties=zh_bold, color="#111827", pad=10)

# ---------------- B. mechanism ----------------
axb = fig.add_subplot(gs[1, :])
axb.set_axis_off()
axb.set_xlim(0, 1)
axb.set_ylim(0, 1)
axb.set_title("B · 误报机制：冷读把「未闭合的回合」一律当成「写进程已崩溃」", loc="left",
              fontproperties=zh_bold, color="#111827", pad=10)

boxes = [
    (0.015, "superagent host 76796\n正在写 worker 会话\n（turn 尚未结束）", BLUE),
    (0.255, "个人 web host 2039546\n冷读 readColdSessionLog\n（读者不碰 session.lock）", ORANGE),
    (0.495, "追加合成的\nturn/end {kind: interrupted}\n（磁盘日志本身没有这条）", RED),
    (0.735, "聊天 UI\n「任务已停止」\n「DSH 进程在本轮完成前退出…」", RED),
]
box_w, box_h = 0.205, 0.42
for x, text, color in boxes:
    axb.add_patch(FancyBboxPatch((x, 0.28), box_w, box_h, boxstyle="round,pad=0.012,rounding_size=0.02",
                                 fc="white", ec=color, lw=1.8))
    axb.text(x + box_w / 2, 0.28 + box_h / 2, text, ha="center", va="center",
             fontproperties=zh_small, color="#111827")
for x in (0.22, 0.46, 0.70):
    axb.annotate("", xy=(x + 0.035, 0.49), xytext=(x, 0.49),
                 arrowprops=dict(arrowstyle="->", color=GRAY, lw=1.6))
axb.text(0.5, 0.12, "根因：冷读路径无法区分「写者已崩溃」与「写者仍活着、只是被另一个 DSH 进程读取」。"
                   "于是任何在回合结束前打开会话的只读查看，都会看到一条假的「进程退出」恢复记录。",
         ha="center", fontproperties=zh_small, color="#374151")
axb.text(0.5, 0.035, "源码：dsh-session-query:readColdSessionLog → interruptedTurnClosers；dsh-client-ui-chat:endFailure → message.failure.interrupted",
         ha="center", fontproperties=zh_tiny, color=GRAY)

# ---------------- C. verdict ----------------
axc = fig.add_subplot(gs[2, 0])
axc.set_axis_off()
axc.set_xlim(0, 1)
axc.set_ylim(0, 1)
axc.set_title("C · 结论", loc="left", fontproperties=zh_bold, color="#111827", pad=8)
rows = [
    ("今天真实中断的回合", "0", GREEN),
    ("四色 worker · turn 1", "completed", GREEN),
    ("π 工单的子 agent · turn 1", "completed", GREEN),
    ("跨进程冷读误报", "是，可复现", RED),
    ("数据丢失", "无（工作区/日志/PR 均在）", GREEN),
]
axc.add_patch(FancyBboxPatch((0.0, 0.0), 1.0, 0.92, boxstyle="round,pad=0.01,rounding_size=0.02",
                             fc=LIGHT, ec="#e5e7eb"))
for i, (k, v, color) in enumerate(rows):
    y = 0.80 - i * 0.17
    axc.text(0.03, y, k, fontproperties=zh_small, color="#374151")
    axc.text(0.97, y, v, ha="right", fontproperties=zh_bold, color=color)

# ---------------- D. history ----------------
axd = fig.add_subplot(gs[2, 1])
axd.set_axis_off()
axd.set_xlim(0, 1)
axd.set_ylim(0, 1)
axd.set_title("D · 真正的中断发生在 11-13（对照）", loc="left", fontproperties=zh_bold, color="#111827", pad=8)
axd.add_patch(FancyBboxPatch((0.0, 0.0), 1.0, 0.92, boxstyle="round,pad=0.01,rounding_size=0.02",
                             fc=LIGHT, ec="#e5e7eb"))
lines = [
    ("11-01 / 11-02", "Dreamer 个人会话 + worker", "个人 web 随连接退出"),
    ("11-13 14:56:53", "3 个 worker（notebooks / 日报 / resume）", "host 重启"),
    ("11-13 14:58:13", "2 个 worker + orchestrator", "host 重启"),
    ("11-17", "0 个 interrupted turn", "本次为冷读误报"),
]
axd.text(0.03, 0.84, "时间", fontproperties=zh_tiny, color=GRAY)
axd.text(0.28, 0.84, "受影响会话", fontproperties=zh_tiny, color=GRAY)
axd.text(0.72, 0.84, "原因", fontproperties=zh_tiny, color=GRAY)
for i, (t, s, c) in enumerate(lines):
    y = 0.66 - i * 0.17
    axd.text(0.03, y, t, fontproperties=zh_tiny, color="#111827")
    axd.text(0.28, y, s, fontproperties=zh_tiny, color="#111827")
    axd.text(0.72, y, c, fontproperties=zh_tiny, color=GRAY)

fig.text(0.055, 0.018, "证据：~/.dsh/sessions 会话日志、进程启动时间、dsh 0.2.1-rc.2 安装包源码；复现方法与代码路径见报告 §3.3。",
         fontproperties=zh_tiny, color=GRAY)

out = "/mnt/storage1/syk/.superagent-work/diagnose-web-interruptions-69610e43782d/docs/visuals/2026-11-17-web-interruptions.png"
fig.savefig(out, dpi=180, facecolor="white")
print("wrote", out)
