#!/usr/bin/env python3
"""Build the DOCX version of the 2026-11-17 Web-interruption diagnosis report."""
import os
import re
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, RGBColor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "docs/reports/2026-11-17-web-interruptions.md")
VISUAL = os.path.join(ROOT, "docs/visuals/2026-11-17-web-interruptions.png")
OUT = os.path.join(ROOT, "deliverables/Web端中断误报诊断_2026-11-17.docx")

CJK = "Noto Sans CJK SC"
MONO = "Consolas"
INK = RGBColor(0x11, 0x18, 0x27)
MUTED = RGBColor(0x4B, 0x55, 0x63)
ACCENT = RGBColor(0x1F, 0x4E, 0x79)
RED = RGBColor(0xC0, 0x39, 0x2B)

doc = Document()
sec = doc.sections[0]
sec.page_width = Cm(21.0)
sec.page_height = Cm(29.7)
sec.top_margin = Cm(1.9)
sec.bottom_margin = Cm(1.9)
sec.left_margin = Cm(2.1)
sec.right_margin = Cm(2.1)

normal = doc.styles["Normal"]
normal.font.name = "Arial"
normal.font.size = Pt(10.5)
normal.font.color.rgb = INK
normal.element.rPr.rFonts.set(qn("w:eastAsia"), CJK)
normal.paragraph_format.space_after = Pt(6)
normal.paragraph_format.line_spacing = 1.22

for name, size, color in (("Title", 21, INK), ("Heading 1", 15, ACCENT), ("Heading 2", 12.5, INK), ("Heading 3", 11, INK)):
    style = doc.styles[name]
    style.font.name = "Arial"
    style.font.size = Pt(size)
    style.font.color.rgb = color
    style.font.bold = True
    style.element.rPr.rFonts.set(qn("w:eastAsia"), CJK)
    style.paragraph_format.space_before = Pt(14 if name != "Title" else 0)
    style.paragraph_format.space_after = Pt(6)

def set_cjk(run, font=CJK):
    run.font.name = "Arial"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), font)

def shade(paragraph, fill="F3F4F6"):
    pPr = paragraph._p.get_or_add_pPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill)
    pPr.append(shd)

INLINE = re.compile(r"(\*\*.+?\*\*|`[^`]+`)")

def add_inline(paragraph, text, size=None, color=None, bold=False, italic=False):
    for part in INLINE.split(text):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**"):
            run = paragraph.add_run(part[2:-2])
            run.bold = True
        elif part.startswith("`") and part.endswith("`"):
            run = paragraph.add_run(part[1:-1])
            run.font.name = MONO
            run.font.size = Pt((size or 10.5) - 0.5)
            run.font.color.rgb = RGBColor(0x37, 0x41, 0x51)
        else:
            run = paragraph.add_run(part)
            if bold:
                run.bold = True
        if size:
            run.font.size = Pt(size)
        if color is not None and not (part.startswith("`") and part.endswith("`")):
            run.font.color.rgb = color
        if italic:
            run.italic = True
        set_cjk(run)

lines = open(SRC, encoding="utf-8").read().splitlines()
i = 0
first_h1 = True
num_counter = 0
in_code = False
code_buf = []
table_buf = []
visual_added = False

def flush_table():
    global table_buf
    if not table_buf:
        return
    rows = []
    for ln in table_buf:
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        if all(re.fullmatch(r":?-+:?", c) for c in cells if c):
            continue
        rows.append(cells)
    table_buf = []
    if not rows:
        return
    width = max(len(r) for r in rows)
    table = doc.add_table(rows=len(rows), cols=width)
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for r, cells in enumerate(rows):
        for c in range(width):
            cell = table.cell(r, c)
            text = cells[c] if c < len(cells) else ""
            p = cell.paragraphs[0]
            p.paragraph_format.space_after = Pt(2)
            add_inline(p, text, size=8.5, color=MUTED if r == 0 else INK, bold=(r == 0))
    doc.add_paragraph().paragraph_format.space_after = Pt(2)

def flush_code():
    global code_buf
    if not code_buf:
        return
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Cm(0.4)
    p.paragraph_format.space_after = Pt(8)
    p.paragraph_format.keep_together = True
    shade(p, "F3F4F6")
    run = p.add_run("\n".join(code_buf))
    run.font.name = MONO
    run.font.size = Pt(8.5)
    run.font.color.rgb = RGBColor(0x37, 0x41, 0x51)
    set_cjk(run, MONO)
    code_buf = []

while i < len(lines):
    line = lines[i]
    if in_code:
        if line.strip() == "```":
            in_code = False
            flush_code()
        else:
            code_buf.append(line)
        i += 1
        continue
    if line.strip().startswith("```"):
        flush_table()
        in_code = True
        i += 1
        continue
    if line.startswith("|"):
        table_buf.append(line)
        i += 1
        continue
    flush_table()
    if not line.strip():
        i += 1
        continue
    if (not visual_added) and line.startswith("## 0."):
        visual_added = True
        doc.add_page_break()
        pic = doc.add_picture(VISUAL, width=Inches(6.3))
        cap = doc.add_paragraph()
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = cap.add_run("一页可视化：时间线、误报机制、结论与历史对照")
        run.font.size = Pt(8.5)
        run.font.color.rgb = MUTED
        set_cjk(run)
    if line.startswith("# "):
        num_counter = 0
        p = doc.add_paragraph(style="Title")
        add_inline(p, line[2:], size=21, color=INK, bold=True)
        first_h1 = False
    elif line.startswith("## "):
        num_counter = 0
        p = doc.add_paragraph(style="Heading 1")
        add_inline(p, line[3:], size=15, color=ACCENT, bold=True)
    elif line.startswith("### "):
        num_counter = 0
        p = doc.add_paragraph(style="Heading 2")
        add_inline(p, line[4:], size=12.5, color=INK, bold=True)
    elif line.startswith(">"):
        num_counter = 0
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Cm(0.35)
        p.paragraph_format.space_before = Pt(2)
        p.paragraph_format.space_after = Pt(8)
        shade(p, "F3F4F6")
        add_inline(p, line.lstrip("> ").strip(), size=10.2, color=INK, italic=False)
    elif re.match(r"^\d+\.\s", line):
        num_counter += 1
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Cm(0.75)
        p.paragraph_format.first_line_indent = Cm(-0.5)
        run = p.add_run(f"{num_counter}. ")
        run.font.size = Pt(10.5)
        set_cjk(run)
        add_inline(p, re.sub(r"^\d+\.\s", "", line), size=10.5, color=INK)
    elif line.startswith("- "):
        indent = len(line) - len(line.lstrip())
        p = doc.add_paragraph(style="List Bullet")
        if indent >= 2:
            p.paragraph_format.left_indent = Cm(1.0)
        add_inline(p, line.lstrip()[2:], size=10.5, color=INK)
    elif line.strip() == "---":
        pass
    else:
        num_counter = 0
        p = doc.add_paragraph()
        add_inline(p, line.strip(), size=10.5, color=INK)
    i += 1
flush_table()
if in_code:
    flush_code()
if not visual_added:
    doc.add_picture(VISUAL, width=Inches(6.25))

# footer
footer = sec.footer.paragraphs[0]
footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = footer.add_run("2026-11-17 · Web 端中断误报诊断 · DSH 0.2.1-rc.2")
run.font.size = Pt(8)
run.font.color.rgb = MUTED
set_cjk(run)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
doc.save(OUT)
print("wrote", OUT, os.path.getsize(OUT))
