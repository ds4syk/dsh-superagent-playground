"""Render the one-page visual for the merge CLI / approval rules plan."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib import font_manager as fm

FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
fm.fontManager.addfont(FONT)
plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"] = False

BG = "#F7F9FC"
INK = "#111827"
MUTED = "#4B5563"
BORDER = "#CBD5E1"
BLUE, BLUE_L = "#1D4ED8", "#DBEAFE"
GREEN, GREEN_L = "#047857", "#D1FAE5"
AMBER, AMBER_L = "#B45309", "#FEF3C7"
RED, RED_L = "#B91C1C", "#FEE2E2"
PURPLE, PURPLE_L = "#6D28D9", "#EDE9FE"

fig, ax = plt.subplots(figsize=(16, 10))
fig.patch.set_facecolor(BG)
ax.set_xlim(0, 16)
ax.set_ylim(0, 10)
ax.axis("off")


def box(x, y, w, h, fc, ec=BORDER, lw=1.2, radius=0.12):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
        boxstyle=f"round,pad=0.02,rounding_size={radius}", fc=fc, ec=ec, lw=lw, zorder=1))


def text(x, y, s, size=11, color=INK, weight="normal", ha="left", va="center", **kw):
    ax.text(x, y, s, fontsize=size, color=color, fontweight=weight, ha=ha, va=va, zorder=2, **kw)


def arrow(x1, y1, x2, y2, color=MUTED):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=15,
        lw=1.6, color=color, shrinkA=0, shrinkB=0, zorder=1))


text(0.5, 9.55, "merge CLI 与合并审批规则：你批准一次，其余交给 agent", 21, INK, "bold")
text(0.5, 9.18, "部署 ds4syk/dsh-superagent-playground · 2026-11-11 · 依据 dsh-superagent docs/003 §Merging、docs/006 §Accounts/§Settings", 10.5, MUTED)

# top strip: current state
box(0.5, 8.28, 15.0, 0.66, "#FFFFFF")
text(0.75, 8.61, "现场核对", 11, INK, "bold")
text(1.95, 8.61, "（23:55）", 10, MUTED)
chips = [
    ("prompt revision 0", "未命名 merge CLI", AMBER_L, AMBER),
    ("无 ~/.local/bin/gh-merge", "第二账号未登录", RED_L, RED),
    ("allow_auto_merge = false", "面板也开不了", RED_L, RED),
    ("main 无 ruleset", "批准/强推都没有强制", RED_L, RED),
]
x = 4.1
for title, sub, fc, ec in chips:
    box(x, 8.36, 2.72, 0.5, fc, ec, 1.0, 0.06)
    text(x + 0.12, 8.69, title, 8.6, ec, "bold")
    text(x + 0.12, 8.48, sub, 8.0, MUTED)
    x += 2.82

# left panel: three layers
box(0.5, 1.2, 5.55, 6.75, "#FFFFFF")
text(0.8, 7.62, "要配的三层（缺一层都不算完成）", 13, INK, "bold")

cards = [
    (5.75, "① 本机 CLI", "~/.local/bin/gh-merge", "以 @YingkeSu 登录 · fine-grained PAT", "需要你本人登录一次", AMBER_L, AMBER),
    (3.85, "② 面板文字规则", "设置 → 提示词 → Delivery requirements", "命名 merge CLI + 1 个批准 + 文档豁免 200 行", "agent 执行", BLUE_L, BLUE),
    (1.95, "③ GitHub 强制规则", "ruleset：必须 PR + 1 approval + dismiss stale", "allow_auto_merge = true · bypass 为空", "agent 执行", GREEN_L, GREEN),
]
for y, title, line1, line2, badge, fc, ec in cards:
    box(0.8, y, 4.95, 1.55, fc, ec, 1.1, 0.1)
    text(1.0, y + 1.28, title, 12, ec, "bold")
    text(1.0, y + 0.93, line1, 9.5, INK)
    text(1.0, y + 0.58, line2, 9.5, MUTED)
    text(5.5, y + 0.16, badge, 8.2, ec, "bold", ha="right")

# right panel: flow after approval
box(6.35, 1.2, 9.15, 6.75, "#FFFFFF")
text(6.65, 7.62, "批准之后的流程（配置完成后）", 13, INK, "bold")

steps = [
    (6.42, 1.0, "1  agent 开 PR（@ds4syk）", "提交已带 @YingkeSu 的 Co-authored-by；请求 review", BLUE_L, BLUE),
    (5.12, 1.0, "2  你点 Approve（唯一人工动作）", "覆盖当前 head；GitHub 记 1 个 approval", AMBER_L, AMBER),
    (3.82, 1.0, "3  review 事件唤醒 worker", "核对 head、批准状态、行数阈值", PURPLE_L, PURPLE),
    (2.35, 1.25, "4  worker 打开 auto-merge", "gh-merge pr merge <n> --auto --merge --match-head-commit <sha>", GREEN_L, GREEN),
    (1.10, 0.95, "5  GitHub 合并 · merged by @YingkeSu", "ticket 结束；不需要再点 Merge", GREEN_L, GREEN),
]
for y, h, title, sub, fc, ec in steps:
    box(6.65, y, 8.55, h, fc, ec, 1.0, 0.1)
    text(6.85, y + h - 0.28, title, 11.2, ec, "bold")
    text(6.85, y + (0.55 if h > 1.1 else 0.28), sub, 8.8, MUTED)
text(6.85, 2.50, "push 新提交前：gh-merge pr merge <n> --disable-auto，再请求 review", 8.2, AMBER)
for i in range(len(steps) - 1):
    y, h, *_ = steps[i]
    next_y, next_h, *_ = steps[i + 1]
    arrow(10.93, y, 10.93, next_y + next_h, MUTED)

# bottom band
box(0.5, 0.25, 15.0, 0.72, "#0F172A")
text(0.75, 0.68, "顺序：先登录 gh-merge → 再保存 prompt / 创建 ruleset。", 10.5, "#FFFFFF", "bold")
text(0.75, 0.42, "否则批准后 worker 找不到 CLI。回滚：--disable-auto → 删 ruleset → 恢复 prompt revision。", 9.2, "#CBD5E1")
text(15.25, 0.68, "不是硬安全边界：worker 同一 OS 用户可读 token；owner 仍是 agent。", 8.6, "#FCA5A5", ha="right")

fig.savefig("docs/visuals/2026-11-11-merge-cli-flow.png", dpi=220, bbox_inches="tight", facecolor=BG)
print("saved docs/visuals/2026-11-11-merge-cli-flow.png")
