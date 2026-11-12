#!/usr/bin/env python3
"""Render the monthly token-usage one-pager from monthly-usage.mjs JSON.

usage: python3 monthly-usage-chart.py <usage.json> <out.png>

Prices mirror automation/monthly-usage.mjs (DeepSeek CNY list price, peak bands);
the figure shows list-price estimates, not a platform bill.
"""
import json
import sys
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo
from PIL import Image, ImageDraw, ImageFont

FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
FONT_B = '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'


def f(size, bold=False, idx=2):
    return ImageFont.truetype(FONT_B if bold else FONT, size, index=idx)


INK = (25, 32, 44)
MUTED = (110, 120, 136)
GRID = (228, 232, 238)
PANEL = (247, 249, 252)
BLUE = (54, 110, 235)
TEAL = (16, 160, 150)
ORANGE = (240, 140, 40)
RED = (220, 80, 90)
GREEN = (52, 168, 92)

src = sys.argv[1] if len(sys.argv) > 1 else 'usage-30d.json'
out = sys.argv[2] if len(sys.argv) > 2 else 'monthly-token-cost.png'
data = json.load(open(src))
totals = data['totals']
bands = data['bands']
peak = bands['peak']
off = bands['off']

# Base cost at off-peak price plus the peak surcharge, so the four rows add up to the total.
base_cache = totals['cacheRead'] / 1e6 * 0.02
base_out = totals['output'] / 1e6 * 4
base_unc = (totals['uncached'] + totals['cacheWrite']) / 1e6 * 1
base_total = base_cache + base_out + base_unc
peak_surcharge = totals['cost'] - base_total
parts = [
    ('缓存读取', base_cache, f"{totals['cacheRead']:,} tok × ¥0.02/M", TEAL),
    ('输出', base_out, f"{totals['output']:,} tok × ¥4/M", ORANGE),
    ('高峰加价', peak_surcharge, f"峰值 {peak['tokens']/1e8:.2f} 亿 tok，单价 ×2 的增量", RED),
    ('未缓存输入', base_unc, f"{totals['uncached'] + totals['cacheWrite']:,} tok × ¥1/M", BLUE),
]

day_map = {d['date']: d for d in data['byDay']}
start = date.fromisoformat(data['window']['start'])
days = [(start + timedelta(days=i)).isoformat() for i in range(data['window']['days'])]
rows = [(d, day_map.get(d)) for d in days]
peak_share = peak['tokens'] / totals['tokens'] * 100 if totals['tokens'] else 0

W, H = 1600, 1280
img = Image.new('RGB', (W, H), 'white')
dr = ImageDraw.Draw(img)


def fmt_tok(n):
    if n >= 1e8:
        return f'{n/1e8:.2f} 亿'
    if n >= 1e4:
        return f'{n/1e4:.1f} 万'
    return f'{n}'


def text(x, y, s, size=24, bold=False, fill=INK, anchor='la'):
    dr.text((x, y), s, font=f(size, bold), fill=fill, anchor=anchor)


def short_workdir(cwd):
    if cwd.startswith('/mnt/storage1/syk/.superagent-work/'):
        return '工单·' + cwd.rstrip('/').split('/')[-1]
    if cwd == '/mnt/storage1/syk/.dsh-state/superagent':
        return 'orchestrator 主会话'
    if cwd.startswith('/home/syk/labs/'):
        return 'labs/' + cwd.rstrip('/').split('/')[-1]
    return cwd.replace('/mnt/storage1/syk/', '~/').replace('/home/syk/', '~/')


SHORT = {
    'ppo-minecraft-f942b789b4ce': 'ppo-minecraft',
    'resume-dreamer-session-91165e4a84d2': 'resume-dreamer',
    'jamba-world-model-fdd0e082a6b1': 'jamba-world-model',
    'sdpr-qpep-study-6fe49f598db5': 'sdpr-qpep-study',
    'sdpr-jepa-bridge-c46ae10a1a2f': 'sdpr-jepa-bridge',
    'dreamer-training-deepdive-3f780f0d2e38': 'dreamer-deepdive',
    'sdpr-dreamer-notebooks-927183da853a': 'sdpr-dreamer-ipynb',
}

# ---------- header ----------
text(80, 50, '过去一个月 DSH Token 用量与花费', 40, True)
text(80, 102, f"{data['window']['start']} ~ {data['window']['end']}（30 天窗口；DSH 有记录：10-21 起，{data['coverage']['daysWithUsage']} 天有用量）", 21, False, MUTED)
text(1520, 54, f"总计 {totals['tokens']/1e8:.2f} 亿 token", 30, True, BLUE, 'ra')
text(1520, 102, f"估算 ¥{totals['cost']:.2f} · {totals['requests']:,} 次请求 · {totals['sessions']} 个会话", 21, False, MUTED, 'ra')

# ---------- panel A: daily tokens ----------
ax0, ay0, ax1, ay1 = 80, 170, 1520, 590
dr.rounded_rectangle((ax0, ay0, ax1, ay1), 16, fill=PANEL)
text(ax0 + 24, ay0 + 18, '每日计费 token（柱）与估算花费（标注）', 24, True)
text(ax1 - 24, ay0 + 20, f"峰值时段占比 {peak_share:.1f}%", 18, False, MUTED, 'ra')
max_tok = max((r['tokens'] for _, r in rows if r), default=0) or 1
bar_top = ay0 + 80
bar_bottom = ay1 - 50
usable_h = bar_bottom - bar_top
n = len(rows)
slot = (ax1 - ax0 - 80) / n
for i, (d, r) in enumerate(rows):
    cx = ax0 + 40 + slot * (i + 0.5)
    bw = min(26, slot * 0.52)
    tok = (r or {}).get('tokens', 0)
    h = usable_h * (tok / max_tok) if tok else 0
    x0, x1 = cx - bw / 2, cx + bw / 2
    if tok:
        color = (32, 84, 200) if d == '2026-11-03' else BLUE
        dr.rounded_rectangle((x0, bar_bottom - h, x1, bar_bottom), 5, fill=color)
        if tok >= 1.5e8:
            text(cx, bar_bottom - h - 36, fmt_tok(tok), 13, True, INK, 'ma')
            text(cx, bar_bottom - h - 20, f"¥{r['cost']:.0f}", 12, False, MUTED, 'ma')
    else:
        dr.rectangle((x0, bar_bottom - 2, x1, bar_bottom), fill=(214, 220, 230))
    day_label = f"{d[5:7]}/{d[8:10]}"
    text(cx, bar_bottom + 8, day_label, 13, False, MUTED if tok else (180, 186, 198), 'ma')

# ---------- panel B: cost composition ----------
bx0, by0, bx1, by1 = 80, 620, 790, 1070
dr.rounded_rectangle((bx0, by0, bx1, by1), 16, fill=PANEL)
text(bx0 + 24, by0 + 18, f"一个月花费构成（估算 ¥{totals['cost']:.2f}）", 24, True)
max_part = max(p[1] for p in parts)
row_h = 82
for i, (name, cny, detail, color) in enumerate(parts):
    y = by0 + 78 + i * row_h
    text(bx0 + 24, y, name, 21, True)
    text(bx1 - 24, y, f"¥{cny:.2f}   {cny/totals['cost']*100:.1f}%", 20, True, color, 'ra')
    bw = int((bx1 - bx0 - 48) * (cny / max_part) * 0.40)
    dr.rounded_rectangle((bx0 + 24, y + 30, bx0 + 24 + max(bw, 6), y + 44), 6, fill=color)
    text(bx0 + 24, y + 48, detail, 15, False, MUTED)

# ---------- panel C: top consumers ----------
cx0, cy0, cx1, cy1 = 820, 620, 1520, 1070
dr.rounded_rectangle((cx0, cy0, cx1, cy1), 16, fill=PANEL)
text(cx0 + 24, cy0 + 18, '谁在用：按会话目录 Top 8', 24, True)

agg = {}
for w in data['byWorkspace']:
    label = short_workdir(w['workspace'])
    base = label.split('·')[-1].split('/')[-1]
    label = label.replace(base, SHORT.get(base, base))
    a = agg.setdefault(label, {'tokens': 0, 'requests': 0})
    a['tokens'] += w['tokens']
    a['requests'] += w['requests']
top = sorted(agg.items(), key=lambda kv: -kv[1]['tokens'])[:8]
maxv = top[0][1]['tokens'] if top else 1
bar_x0 = cx0 + 24
bar_max = cx1 - cx0 - 48
for i, (name, a) in enumerate(top):
    y = cy0 + 66 + i * 48
    text(bar_x0, y, name, 18, False, INK)
    text(cx1 - 24, y, f"{a['tokens']/1e8:.2f} 亿 · {a['requests']} 次", 17, True, INK, 'ra')
    bw = int(bar_max * (a['tokens'] / maxv))
    dr.rounded_rectangle((bar_x0, y + 30, bar_x0 + max(bw, 4), y + 36), 3, fill=GREEN if i == 0 else TEAL)

# ---------- footer ----------
local_end = datetime.fromisoformat(data['generatedAt'].replace('Z', '+00:00')).astimezone(ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M')
text(80, 1096, '口径：本机 DSH 全部会话原始日志（session.v4.jsonl.zstd），逐条 assistant/message 的 usage 汇总；', 15, False, MUTED)
text(80, 1120, '计费 token = 未缓存输入 + 缓存读取 + 缓存写入 + 输出，按 step 开始时间归入 Asia/Shanghai 日期，与 Context Insights 同口径；', 15, False, MUTED)
text(80, 1144, '继承种子会话的种子事件不计入。花费按 DeepSeek 官方 CNY 列表价估算（deepseek-flash：缓存命中 ¥0.04 / 未命中 ¥2 / 输出 ¥8 每百万 token，', 15, False, MUTED)
text(80, 1168, '工作日 9-12 / 14-18 为高峰、其余半价），不是 DeepSeek 平台账单；未计法定节假日与 Search/Fetch。', 15, False, MUTED)
text(80, 1204, f"数据截至 {local_end}（含本期统计工单自身用量）。", 15, False, MUTED)
text(1520, 1204, 'dsh-superagent · 复算脚本 automation/monthly-usage.mjs', 15, False, MUTED, 'ra')

img.save(out)
print('saved', out)
