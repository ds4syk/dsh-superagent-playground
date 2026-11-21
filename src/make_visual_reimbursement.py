# -*- coding: utf-8 -*-
"""One-page decision graphic for the reimbursement knowledge base.

Run:  ~/miniconda3/envs/sdpr-clean/bin/python src/make_visual_reimbursement.py
Out:  docs/visuals/2026-11-21-ai-reimbursement.png
"""
import os
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "docs", "visuals", "2026-11-21-ai-reimbursement.png")

W, H = 2200, 1600
GREEN = (46, 107, 58)
DARKGREEN = (33, 78, 42)
BLUE = (31, 78, 121)
LIGHTBLUE = (234, 241, 248)
LIGHTGREEN = (234, 243, 231)
LIGHTAMBER = (255, 247, 230)
AMBER = (176, 112, 26)
RED = (176, 58, 46)
LIGHTRED = (253, 235, 232)
GRAY = (110, 110, 110)
LIGHTGRAY = (245, 245, 245)
DARK = (30, 30, 30)
WHITE = (255, 255, 255)

FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
FONT_BOLD_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"


def load_font(size, bold=False):
    path = FONT_BOLD_PATH if bold else FONT_PATH
    for idx in range(5):
        try:
            f = ImageFont.truetype(path, size, index=idx)
            if "SC" in f.getname()[0]:
                return f
        except Exception:
            continue
    return ImageFont.truetype(path, size, index=2)


# cache fonts by (size, bold)
_font_cache = {}


def font(size, bold=False):
    key = (size, bold)
    if key not in _font_cache:
        _font_cache[key] = load_font(size, bold)
    return _font_cache[key]


def text_width(draw, s, f):
    return draw.textlength(s, font=f)


def wrap(draw, s, f, max_w):
    """Wrap CJK/Latin text to max_w. Breaks at spaces, CJK chars anywhere."""
    lines = []
    for para in s.split("\n"):
        if not para:
            lines.append("")
            continue
        cur = ""
        i = 0
        while i < len(para):
            ch = para[i]
            # group ASCII word
            if ch.isascii() and (ch.isalnum() or ch in "._-/%+$"):
                j = i + 1
                while j < len(para) and para[j].isascii() and (para[j].isalnum() or para[j] in "._-/%+$"):
                    j += 1
                token = para[i:j]
                trial = cur + token
                if cur and text_width(draw, trial, f) > max_w:
                    lines.append(cur)
                    cur = token
                else:
                    cur = trial
                i = j
            else:
                trial = cur + ch
                if cur and text_width(draw, trial, f) > max_w:
                    lines.append(cur)
                    cur = ch
                else:
                    cur = trial
                i += 1
        if cur:
            lines.append(cur)
    return lines


def draw_text_block(draw, x, y, s, size, color=DARK, bold=False,
                    max_w=None, line_gap=1.45, align="left"):
    f = font(size, bold)
    lines = wrap(draw, s, f, max_w) if max_w else s.split("\n")
    lh = int(size * line_gap)
    for line in lines:
        if align == "center":
            tw = text_width(draw, line, f)
            draw.text((x - tw / 2, y), line, font=f, fill=color)
        else:
            draw.text((x, y), line, font=f, fill=color)
        y += lh
    return y


def box(draw, xy, fill=WHITE, outline=None, radius=16, width=2):
    draw.rounded_rectangle(xy, radius=radius, fill=fill,
                           outline=outline, width=width if outline else 0)


def arrow(draw, x1, y1, x2, y2, color=GREEN, width=4, head=12):
    draw.line((x1, y1, x2, y2), fill=color, width=width)
    if y2 > y1:
        draw.polygon([(x2, y2), (x2 - head, y2 - head * 1.5),
                      (x2 + head, y2 - head * 1.5)], fill=color)
    elif y2 < y1:
        draw.polygon([(x2, y2), (x2 - head, y2 + head * 1.5),
                      (x2 + head, y2 + head * 1.5)], fill=color)


def pill(draw, x, y, label, size=20, fill=GREEN, fg=WHITE, pad=12):
    f = font(size, True)
    tw = text_width(draw, label, f)
    draw.rounded_rectangle((x, y, x + tw + pad * 2, y + size + pad * 1.2),
                           radius=(size + pad) // 2, fill=fill)
    draw.text((x + pad, y + pad * 0.4), label, font=f, fill=fg)
    return x + tw + pad * 2


img = Image.new("RGB", (W, H), WHITE)
d = ImageDraw.Draw(img)

# Header band
d.rectangle((0, 0, W, 150), fill=GREEN)
draw_text_block(d, 70, 34, "境外 AI 订阅，能不能报销？", 52, WHITE, True)
draw_text_block(d, 72, 106, "一张图看懂：高校/课题组的经费、票据与合规路径", 25, (220, 240, 220), False)
draw_text_block(d, 1850, 48, "2026-11", 30, (220, 240, 220), False)
draw_text_block(d, 1850, 90, "研究整理", 22, (200, 225, 200), False)

# Left section label
draw_text_block(d, 80, 176, "决策路径", 28, GREEN, True)

LEFT_X, LEFT_W = 80, 1080
spine_x = LEFT_X + 62
y = 220
step_h = 118
gap = 26

steps = [
    ("0", "先看经费和预算科目", "纵向：设备费 / 业务费 / 劳务费 ｜ 横向：按合同 ｜ 包干：自主但需真实相关",
     LIGHTBLUE, BLUE),
    ("1", "预算 / 科目里有这一项吗？", "有 → 进入下一步\n没有 → 先调剂 / 补预算，不要先买后报",
     LIGHTAMBER, AMBER),
    ("2", "能取得境内发票吗？", "能 → 按真实品名报销：云服务 / 数据服务 / 技术服务；留合同 + 用量记录\n不能 → 进入下一步",
     LIGHTGREEN, GREEN),
    ("3", "学校能对公采购境外服务吗？", "能 → 合同 + 境外 invoice + 付款 + 代扣税 + 备案 + 数据评估（门槛高）\n不能 → 进入下一步",
     LIGHTBLUE, BLUE),
    ("4", "有境内合规替代 / 学校统一采购吗？", "有 → 国内模型 / 云厂商 / 学校集中账号（推荐）\n没有 → 个人垫付（高风险，见下）",
     LIGHTGREEN, GREEN),
]

for idx, (num, title, body, fill, accent) in enumerate(steps):
    top = y + idx * (step_h + gap)
    box(d, (LEFT_X, top, LEFT_X + LEFT_W, top + step_h), fill=fill, outline=accent, width=3)
    # number circle
    r = 26
    cx, cy = LEFT_X + 40, top + 38
    d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=accent)
    nf = font(28, True)
    tw = text_width(d, num, nf)
    d.text((cx - tw / 2, cy - 19), num, font=nf, fill=WHITE)
    # title
    draw_text_block(d, LEFT_X + 82, top + 18, title, 27, DARK, True, max_w=LEFT_W - 100)
    # body
    draw_text_block(d, LEFT_X + 82, top + 58, body, 20, DARK, False,
                    max_w=LEFT_W - 100, line_gap=1.35)
    # arrow to next
    if idx < len(steps) - 1:
        arrow(d, LEFT_X + LEFT_W // 2, top + step_h + 4,
              LEFT_X + LEFT_W // 2, top + step_h + gap - 4)

# Red caution box (left, bottom)
red_top = y + len(steps) * (step_h + gap) + 4
box(d, (LEFT_X, red_top, LEFT_X + LEFT_W, red_top + 210), fill=LIGHTRED, outline=RED, width=3)
draw_text_block(d, LEFT_X + 28, red_top + 18, "个人垫付前的三条红线", 27, RED, True)
lines = [
    "1. 票货不符 / 变名开票（“技术服务费”代开）= 虚开风险",
    "2. 个人消费 / 与项目无关 = 不得列支",
    "3. 涉密 / 个人信息 / 重要数据出境 = 先过数据合规",
]
yy = red_top + 62
for s in lines:
    draw_text_block(d, LEFT_X + 28, yy, s, 20, DARK, False, max_w=LEFT_W - 56)
    yy += 34
draw_text_block(d, LEFT_X + 28, red_top + 166,
                "先书面问财务：科目、凭证、个人垫付、税务处理；不要先买后报。",
                19, RED, True, max_w=LEFT_W - 56)

# Right panel
RX, RW = 1240, 880

def panel(title, top, h, fill=WHITE, outline=(210, 210, 210)):
    box(d, (RX, top, RX + RW, top + h), fill=fill, outline=outline, width=2)
    d.rounded_rectangle((RX, top, RX + RW, top + 46), radius=12, fill=GREEN)
    draw_text_block(d, RX + 22, top + 8, title, 24, WHITE, True)

panel("关键事实：境外订阅的硬约束", 220, 420)
facts = [
    "• OpenAI / Anthropic 官方支持地区不含中国大陆",
    "• 通常只有英文 invoice / receipt，不是中国发票",
    "• 个人 Plus/Pro 不能对公付款、不能抵扣",
    "• 境内消费的境外服务：增值税 6% 由购买方扣缴",
    "• 特许权使用费：预提所得税 10%（税收协定可降）",
    "• 单笔 > 5 万美元：对外支付税务备案",
    "• 涉密 / 个人信息 / 重要数据：先过出境评估",
]
yy = 282
for s in facts:
    draw_text_block(d, RX + 22, yy, s, 20, DARK, False, max_w=RW - 44)
    yy += 44

panel("各校口径差异很大（样本）", 668, 336, fill=(250, 250, 250))
schools = [
    ("新疆大学", "纵向不报“各类软件服务费 / 会员年费”"),
    ("山西工商", "数据库权限费可报；非科研会员费不报"),
    ("北信科", "信息传播费限出版 / 文献 / 检索"),
    ("东南 / 中科大", "境外 Invoice/Receipt + 付款记录可作凭证"),
]
yy = 730
for name, desc in schools:
    pill(d, RX + 22, yy, name, size=17, fill=GREEN)
    draw_text_block(d, RX + 185, yy + 2, desc, 19, DARK, False, max_w=RW - 210)
    yy += 62

panel("一句话", 1018, 210, fill=(240, 247, 240), outline=GREEN)
draw_text_block(d, RX + 26, 1078,
                "能不能报，不取决于发票叫什么，\n而取决于业务真不真、科目有没有、\n票据配不配、数据能不能出境。",
                23, DARKGREEN, True, max_w=RW - 52, line_gap=1.5)
draw_text_block(d, RX + 26, 1190, "—— 先问财务，再花钱", 19, GREEN, False)

# Bottom recommendation strip
strip_top = 1250
box(d, (80, strip_top, W - 80, strip_top + 200), fill=(248, 250, 247), outline=(205, 215, 205), width=2)
draw_text_block(d, 108, strip_top + 16, "推荐顺序（从稳到险）", 26, GREEN, True)
paths = [
    ("A", "境内模型 / 云服务", "有发票、可对公、数据境内", LIGHTGREEN, GREEN),
    ("B", "学校统一采购", "一次采购、统一开票", LIGHTGREEN, GREEN),
    ("C", "横向合作采购", "合同约定、真实服务", LIGHTBLUE, BLUE),
    ("D", "学校对公境外", "合同 + 扣税 + 备案 + 评估", LIGHTAMBER, AMBER),
    ("E", "个人垫付 + 找票", "合同主体是个人；不建议", LIGHTRED, RED),
]
pw, ph, pgap = 388, 118, 24
px = 108
for i, (letter, name, desc, fill, accent) in enumerate(paths):
    top = strip_top + 56
    box(d, (px, top, px + pw, top + ph), fill=fill, outline=accent, width=2)
    d.ellipse((px + 18, top + 18, px + 66, top + 66), fill=accent)
    lf = font(26, True)
    tw = text_width(d, letter, lf)
    d.text((px + 42 - tw / 2, top + 23), letter, font=lf, fill=WHITE)
    draw_text_block(d, px + 80, top + 16, name, 22, DARK, True, max_w=pw - 92)
    draw_text_block(d, px + 80, top + 54, desc, 17, DARK, False, max_w=pw - 92, line_gap=1.3)
    if i < len(paths) - 1:
        ax = px + pw + 4
        arrow(d, ax, top + ph // 2, ax + pgap - 8, top + ph // 2, color=GRAY, width=3, head=7)
    px += pw + pgap


# Footer
d.rectangle((0, 1520, W, H), fill=(242, 242, 242))
draw_text_block(d, 70, 1538,
                "依据：国自然 / 重点研发资金管理办法、增值税法、企业所得税法、发票公告、生成式 AI 暂行办法、数据出境办法、各校财务通知 ｜ 2026-11 ｜ 研究整理，不构成法律 / 税务意见",
                17, GRAY, False, max_w=W - 140)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
img.save(OUT, "PNG")
print("wrote", OUT, img.size)
