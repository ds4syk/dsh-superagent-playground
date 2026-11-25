#!/usr/bin/env python3
"""Render the lifetime token-usage one-pager from aggregate.py output."""
import json
import re
import sys
from PIL import Image, ImageDraw, ImageFont

FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
FONT_B = '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'

INK = (25, 32, 44)
MUTED = (110, 120, 136)
GRID = (228, 232, 238)
PANEL = (247, 249, 252)
BLUE = (54, 110, 235)
TEAL = (16, 160, 150)
ORANGE = (240, 140, 40)
RED = (220, 80, 90)
PURPLE = (125, 92, 210)
COLORS = {'deepseek-flash': BLUE, 'deepseek-v4-flash': TEAL, 'kimi-k3': ORANGE, 'glm-5.4': PURPLE}


def f(size, bold=False, idx=2):
    return ImageFont.truetype(FONT_B if bold else FONT, size, index=idx)


def fmt_tok(n):
    if n >= 1e8:
        return f'{n/1e8:.2f} 亿'
    if n >= 1e4:
        return f'{n/1e4:.0f} 万'
    return str(n)


def money(usd):
    return f'${usd:,.2f}'


src = sys.argv[1] if len(sys.argv) > 1 else 'usage.json'
out = sys.argv[2] if len(sys.argv) > 2 else 'lifetime-token-usage.png'
data = json.load(open(src))
t = data['totals']
cov = data['coverage']
dedup = data['dedup']
fx = data['prices']['fxUsdCny']
days = data['byDay']

W, H = 1600, 1300
img = Image.new('RGB', (W, H), 'white')
dr = ImageDraw.Draw(img)


def text(x, y, s, size=22, bold=False, fill=INK, anchor='la'):
    dr.text((x, y), s, font=f(size, bold), fill=fill, anchor=anchor)


def clean_ws(name):
    return re.sub(r'-[0-9a-f]{8,}$', '', name)


# ---------- header ----------
text(70, 42, 'DSH 历史全部 Token 用量与花费', 40, True)
text(70, 96, f"2026-10-21 ~ 11-26 · {cov['sessionFiles']} 个会话日志 · 按 models.dev 定价估算（列表价，非账单）", 19, False, MUTED)
text(1530, 40, f"{fmt_tok(t['tokens'])} token", 34, True, BLUE, 'ra')
text(1530, 88, f"{money(t['cost'])} ≈ ¥{t['costCNY']:,.0f}", 25, True, ORANGE, 'ra')
text(1530, 122, f"{t['requests']:,} 次请求 · {cov['sessionsWithUsage']} 个会话 · {cov['daysWithUsage']} 个活跃日", 18, False, MUTED, 'ra')

# ---------- panel A: daily bars ----------
ax0, ay0, ax1, ay1 = 70, 160, 950, 630
dr.rounded_rectangle((ax0, ay0, ax1, ay1), 16, fill=PANEL)
text(ax0 + 24, ay0 + 18, '每日计费 token', 24, True)
peak_day = max(days, key=lambda d: d['tokens'])
text(ax1 - 24, ay0 + 22, f"单日峰值 {peak_day['date'][5:]}：{peak_day['tokens']/1e8:.2f} 亿", 17, False, MUTED, 'ra')
max_tok = max(d['tokens'] for d in days)
plot_l, plot_r = ax0 + 70, ax1 - 30
plot_t, plot_b = ay0 + 78, ay1 - 62
n = len(days)
slot = (plot_r - plot_l) / n
for i in range(0, 5):
    y = plot_b - (plot_b - plot_t) * i / 4
    dr.line((plot_l, y, plot_r, y), fill=GRID, width=1)
    text(plot_l - 10, y, f'{max_tok * i / 4 / 1e8:.0f}亿', 15, False, MUTED, 'rm')
for i, d in enumerate(days):
    cx = plot_l + slot * (i + 0.5)
    bw = min(40, slot * 0.58)
    h = (plot_b - plot_t) * d['tokens'] / max_tok
    color = RED if d is peak_day else BLUE
    dr.rounded_rectangle((cx - bw / 2, plot_b - h, cx + bw / 2, plot_b), 4, fill=color)
    text(cx, plot_b + 8, d['date'][5:], 14, False, MUTED, 'ma')
    if d['tokens'] > 3e8:
        text(cx, plot_b - h - 8, f"{d['tokens']/1e8:.1f}", 14, True, INK, 'mb')
text(ax0 + 24, ay1 - 26, f"缓存读取 {fmt_tok(t['cacheRead'])} · 占总 token {t['cacheRead']/t['tokens']*100:.2f}%", 16, False, MUTED)

# ---------- panel B: models ----------
bx0, by0, bx1, by1 = 980, 160, 1530, 630
dr.rounded_rectangle((bx0, by0, bx1, by1), 16, fill=PANEL)
text(bx0 + 24, by0 + 18, '模型与费用', 24, True)
groups = {}
for m in data['byModel']:
    key = 'glm-5.3' if m['model'].startswith('glm-') else m['model']
    g = groups.setdefault(key, {'model': key, 'tokens': 0, 'requests': 0, 'cost': 0.0})
    for k in ('tokens', 'requests', 'cost'):
        g[k] += m[k]
order = ['deepseek-flash', 'deepseek-v4-flash', 'kimi-k3', 'glm-5.3']
rows = [groups[k] for k in order if k in groups]
row_y = by0 + 74
for m in rows:
    label = {
        'deepseek-flash': 'DeepSeek V4.1 Flash',
        'deepseek-v4-flash': 'DeepSeek V4 Flash（旧名）',
        'kimi-k3': 'Kimi K3',
        'glm-5.3': 'GLM-5.3 / 5.4',
    }.get(m['model'], m['model'])
    text(bx0 + 24, row_y, label, 17, True)
    text(bx1 - 24, row_y, f"{money(m['cost'])} · 费用 {m['cost']/t['cost']*100:.1f}%", 15, False, MUTED, 'ra')
    bar_y = row_y + 30
    dr.rounded_rectangle((bx0 + 24, bar_y, bx1 - 24, bar_y + 10), 5, fill=(234, 238, 246))
    w = (bx1 - bx0 - 48) * (m['cost'] / t['cost'])
    dr.rounded_rectangle((bx0 + 24, bar_y, bx0 + 24 + max(4, w), bar_y + 10), 5, fill=COLORS.get(m['model'], PURPLE))
    text(bx0 + 24, bar_y + 16, f"{fmt_tok(m['tokens'])} token · 占 {m['tokens']/t['tokens']*100:.2f}%", 14, False, MUTED)
    row_y += 92
out_share = t['output'] / t['tokens'] * 100
out_cost_share = t['baseCostOutput'] / t['cost'] * 100
peak_cost_share = t['peakSurcharge'] / t['cost'] * 100
text(bx0 + 24, by1 - 52, f'输出 token 只占 {out_share:.2f}%，却贡献约 {out_cost_share:.0f}% 花费；', 16, False, INK)
text(bx0 + 24, by1 - 28, f'DeepSeek 峰时加价另占 {peak_cost_share:.0f}%。', 16, False, INK)

# ---------- panel C: cost composition ----------
cx0, cy0, cx1, cy1 = 70, 660, 760, 1080
dr.rounded_rectangle((cx0, cy0, cx1, cy1), 16, fill=PANEL)
text(cx0 + 24, cy0 + 18, '花费构成（估算）', 24, True)
rows = [
    ('缓存读取', t['baseCostCacheRead'], TEAL),
    ('输出', t['baseCostOutput'], ORANGE),
    ('未缓存输入', t['baseCostUncached'], BLUE),
    ('缓存写入', t['baseCostCacheWrite'], PURPLE),
    ('高峰加价', t['peakSurcharge'], RED),
]
rows = [r for r in rows if r[1] > 0]
row_y = cy0 + 80
maxv = max(r[1] for r in rows)
for label, v, color in rows:
    text(cx0 + 24, row_y, label, 18, True)
    text(cx1 - 24, row_y, f'{money(v)} · {v/t["cost"]*100:.0f}%', 16, False, MUTED, 'ra')
    by = row_y + 28
    dr.rounded_rectangle((cx0 + 24, by, cx1 - 24, by + 12), 6, fill=(234, 238, 246))
    w = (cx1 - cx0 - 48) * v / maxv
    dr.rounded_rectangle((cx0 + 24, by, cx0 + 24 + w, by + 12), 6, fill=color)
    row_y += 62
text(cx0 + 24, cy1 - 40, f"高峰时段 token 占 {t['peakTokens']/t['tokens']*100:.1f}%，按 2 倍价计。", 16, False, MUTED)

# ---------- panel D: months + workspaces ----------
dx0, dy0, dx1, dy1 = 790, 660, 1530, 1080
dr.rounded_rectangle((dx0, dy0, dx1, dy1), 16, fill=PANEL)
text(dx0 + 24, dy0 + 18, '按月与按工作区', 24, True)
text(dx0 + 24, dy0 + 62, '月份', 18, True, BLUE)
text(dx0 + 120, dy0 + 62, 'token', 16, False, MUTED)
text(dx1 - 200, dy0 + 62, '请求', 16, False, MUTED, 'ra')
text(dx1 - 24, dy0 + 62, '估算费用', 16, False, MUTED, 'ra')
ry = dy0 + 92
for m in data['byMonth']:
    text(dx0 + 24, ry, m['month'], 17, True)
    text(dx0 + 120, ry, fmt_tok(m['tokens']), 17)
    text(dx1 - 200, ry, f"{m['requests']:,}", 17, False, INK, 'ra')
    text(dx1 - 24, ry, money(m['cost']), 17, False, INK, 'ra')
    ry += 32
ry += 18
text(dx0 + 24, ry, 'Top 工作区', 18, True, BLUE)
ry += 34
for w in data['byWorkspace'][:6]:
    name = clean_ws(w['workspace'])
    if len(name) > 25:
        name = name[:24] + '…'
    text(dx0 + 24, ry, name, 16)
    text(dx1 - 24, ry, f"{fmt_tok(w['tokens'])} · {money(w['cost'])}", 15, False, MUTED, 'ra')
    ry += 30

# ---------- footer ----------
fy = 1106
dr.line((70, fy, 1530, fy), fill=GRID, width=1)
notes = [
    f"数据源：~/.dsh/sessions 全部 {cov['sessionFiles']} 个会话日志，按 assistant message 计费并去重（剔除 fork 复制 {fmt_tok(dedup['removedTokens'])} token；原始合计 {fmt_tok(dedup['rawTokens'])}）；含 compaction 摘要调用。",
    f"定价：models.dev 快照（2026-11-26）；DeepSeek 采用官方列表价，高峰（北京时间工作日 9:00-12:00 / 14:00-18:00）乘 2；1 USD = {fx:.3f} CNY。",
    "不含：会话标题生成、web search LLM 调用（本地只有请求体、无 usage 记录；搜索自 11-18 起免费）；Claude Code / Codex / OpenCode 等其它 CLI 不在 DSH 日志内。",
    "以上为列表价估算，不是 DeepSeek / Moonshot / Z.AI 平台账单；完整明细与复算脚本见仓库报告。",
]
for i, note in enumerate(notes):
    text(70, fy + 12 + i * 24, note, 14, False, MUTED)

img.save(out)
print('saved', out, img.size)
