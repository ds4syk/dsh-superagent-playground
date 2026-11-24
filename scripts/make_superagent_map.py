"""Draw the one-page feature map of SuperAgent (dsh-superagent).

Run with the base conda Python, which has matplotlib and Noto CJK fonts:

    ~/miniconda3/bin/python scripts/make_superagent_map.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib import font_manager as fm

FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
fm.fontManager.addfont(FONT)
plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"] = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "visuals", "2026-11-25-superagent-features.png")

W, H = 16.0, 10.2
fig = plt.figure(figsize=(W, H), dpi=220)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")
fig.patch.set_facecolor("#F6F7FB")

INK = "#0F172A"
MUTED = "#64748B"
RULE = "#E2E8F0"


def box(x, y, w, h, fc, ec="none", lw=0, rad=0.10, z=2):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad=0.0,rounding_size={rad}",
        fc=fc, ec=ec, lw=lw, zorder=z))


def text(x, y, s, size=9, weight="normal", color=INK, ha="left", va="top",
         linespacing=1.32, z=5):
    ax.text(x, y, s, fontsize=size, fontweight=weight, color=color,
            ha=ha, va=va, linespacing=linespacing, zorder=z)


# ---------------------------------------------------------------- header
text(0.42, 10.02, "SuperAgent 功能地图", 25, "bold", INK)
text(0.42, 9.62,
     "一个守着 GitHub 仓库和群聊的主动式编码代理：主 Session 只做准入研判、路由与转达，"
     "每个工单在自己的执行器 Session 和目录里实现与验证，再回到原会话交付。",
     10.2, "normal", "#334155")
text(15.55, 9.98, "dsh-superagent 0.1 · research prototype", 8.2, "normal", MUTED, ha="right")

# principle strip
box(0.42, 8.98, 15.16, 0.36, "#0F172A", rad=0.08)
text(0.62, 9.19,
     "原则：以最少的人类注意力换最大的价值 —— 只被请求或书面规则允许时行动；"
     "只在人需要行动或能使用结果时说话。",
     8.8, "normal", "#E2E8F0", va="center")

# ---------------------------------------------------------------- layers
LAYERS = [
    ("01", "来源接入", "Source", "#2563EB", "#EFF6FF", [
        ("GitHub",
         "agent 账号通知；订阅的 Issue / PR、评论与\nreview、默认分支 CI；PR 合并或关闭"),
        ("飞书",
         "群（仅 @ / 全部消息）、单聊、其他群仅 @；\n图片读取、历史恢复、话题轮询、OK 回执"),
        ("频道 Channel",
         "GitHub 评论与 review 回复；飞书话题回复。\n消息持久化、退避重试、丢弃可审计"),
        ("可扩展",
         "SourceKind 插件：配置表单、身份、频道、\n线程上下文；新来源不改内核"),
    ]),
    ("02", "准入研判", "Eval", "#7C3AED", "#F5F3FF", [
        ("Gate 准入",
         "按来源的允许名单、飞书群成员同步、bot 名单；\n@ / 单聊 / 后续消息 / 准入表判定强度"),
        ("准入表",
         "名单内外 × 是否 @ × 接收范围 → 明确请求 /\n推断候选 / 忽略；每条规则可查"),
        ("接受模式",
         "off（不接推断）· observe（只记录“会接”）·\nauto（规则匹配才接）；Policy 版本化"),
        ("模型研判",
         "有界线程窗口 + 图片 + 只读仓库；逐条判定规则；\n超时重试、unavailable 进面板"),
    ]),
    ("03", "工单与订阅", "Ticket", "#0EA5E9", "#F0F9FF", [
        ("授权 Mandate",
         "记录策略版本、范围、强度、命中的规则；\n只有人能 stop，停止后新工作需重新接单"),
        ("工单阶段",
         "待回答 / 待审阅 / 阻塞 · 失败 / 排队 /\n等待触发 / 工作中 / 输入未送达"),
        ("订阅扇出",
         "线程后续消息自动到达工单；人工跟进由该\n工单 worker 判断归属，越界则交回重路由"),
        ("恢复与结束",
         "结束工单保留订阅，人消息可恢复；bot/CI 不\n唤醒。结果：完成 / 合并 / 关闭 / 停止 / 无需处理"),
    ]),
    ("04", "执行", "Executor", "#059669", "#ECFDF5", [
        ("一工单一环境",
         "worker-<id> Session + 独立目录；prepare 命令\n准备 worktree；环境变量注入仓库与账号"),
        ("并发与后台",
         "maxWorkers 个槽 + 人的消息 1 个额外槽；\n前台命令 5 秒转后台，子代理可嵌套"),
        ("失败与恢复",
         "瞬时错误三次重试；retry 重开 Assignment；\n重启带恢复说明；全局暂停不打断进行中回合"),
        ("工具与边界",
         "bash、文件、搜索、子代理、report/ack；\n无 ask_user；命令 shim + blocked commands"),
    ]),
    ("05", "交付", "Delivery", "#D97706", "#FFFBEB", [
        ("四种发言",
         "确认 / 回答 / 结果 / 待决策；每条请求只确认\n一次，IM 硬上限 100 词，结果回原 thread"),
        ("PR 交付",
         "分支、作者、评审信用、CI 跟进；不自行 merge；\n自动合并仅在明确条件与批准满足时启用"),
        ("评审处理",
         "已修复静默解决；不采纳给一句理由；\n自动化评审每个 PR 最多三轮"),
        ("可见证据",
         "UI 改动必须 Before/After；本仓库另要求\n一页可视化 + DOCX 知识库 + 原图直链"),
    ]),
    ("06", "面板运维", "Panel & Ops", "#475569", "#F8FAFC", [
        ("六个页签",
         "概览 / 来源 / 触发 / 评估 / 执行 / 设置；\n先列需要人的事，健康状态不占地方"),
        ("操作",
         "暂停处理、重试、停止、重新评估、丢回 Eval、\n丢弃消息审计、搜索、术语表"),
        ("Prompt 与技能",
         "团队三段可编辑，内置部分可覆盖，技能可加；\n保存即版本，升级保留团队文本并标出差异"),
        ("运维",
         "start/status/stop/restart；四档退出码；健康分项；\n管理 API；stateDir 单实例锁、备份与迁移"),
    ]),
]

top = 8.62
row_h = 1.19
gap = 0.105
label_w = 1.62
x0 = 0.42
feat_x = x0 + label_w + 0.22
feat_w = (15.58 - feat_x - 3 * 0.12) / 4

for i, (num, name, en, accent, tint, feats) in enumerate(LAYERS):
    y = top - row_h - i * (row_h + gap)
    # label
    box(x0, y, label_w, row_h, accent, rad=0.10)
    text(x0 + 0.14, y + row_h - 0.13, num, 8.2, "bold", "#FFFFFF")
    text(x0 + 0.14, y + row_h - 0.36, name, 12.8, "bold", "#FFFFFF")
    text(x0 + 0.14, y + row_h - 0.68, en, 8.0, "normal", "#DBEAFE")
    # features
    for j, (title, body) in enumerate(feats):
        fx = feat_x + j * (feat_w + 0.12)
        box(fx, y, feat_w, row_h, "#FFFFFF", ec="#E5EAF3", lw=0.8, rad=0.08)
        ax.add_patch(Rectangle((fx + 0.045, y + 0.10), 0.055, row_h - 0.20,
                               fc=accent, ec="none", zorder=4, alpha=0.9))
        text(fx + 0.17, y + row_h - 0.11, title, 9.1, "bold", "#1E293B")
        text(fx + 0.17, y + row_h - 0.36, body, 7.9, "normal", "#475569", linespacing=1.42)
    # connector arrow to the next label
    if i < len(LAYERS) - 1:
        ax.add_patch(FancyArrowPatch(
            (x0 + label_w / 2, y - 0.005), (x0 + label_w / 2, y - gap + 0.005),
            arrowstyle="-|>", mutation_scale=9, lw=1.1, color="#CBD5E1", zorder=1))

# ---------------------------------------------------------------- flow line
flow_y = 0.55
box(0.42, 0.28, 15.16, 0.50, "#FFFFFF", ec="#E2E8F0", lw=0.8, rad=0.08)
flow = "请求 → 来源过滤 → 准入（gate + 研判）→ 工单与授权 → 执行器 Session → 交付回复 / PR → 面板观察"
text(0.60, flow_y, flow, 8.4, "bold", "#334155", va="center")
text(15.40, flow_y, "本机实例：GitHub + 飞书已接入 · observe 模式 · 3 个 worker 槽",
     7.8, "normal", MUTED, ha="right", va="center")

fig.savefig(OUT, dpi=220, facecolor=fig.get_facecolor())
print("wrote", OUT, os.path.getsize(OUT))
