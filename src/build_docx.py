# -*- coding: utf-8 -*-
"""Build the DOCX knowledge base from kb/*.md.

Run with the env that has python-docx::

    ~/miniconda3/envs/sdpr-clean/bin/python src/build_docx.py
"""

import os
import re
import sys

from docx import Document
from docx.enum.section import WD_ORIENT, WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FILES = [os.path.join(ROOT, "kb", f"{i:02d}.md") for i in range(5)]
FIG_DIR = os.path.join(ROOT, "report", "figures")
OUT = os.path.join(ROOT, "deliverables", "SDPR_VJEPA2_桥接输出层_2026-11.docx")
FONT = "Noto Sans CJK SC"


def latex_simplify(s: str) -> str:
    repl = [
        (r"\\mathbb\{R\}", "R"),
        (r"\\mathbb\{S\}", "S"),
        (r"\\top", "ᵀ"),
        (r"\\succeq", "⪰"),
        (r"\\preceq", "⪯"),
        (r"\\in", "∈"),
        (r"\\times", "×"),
        (r"\\cdot", "·"),
        (r"\\lambda", "λ"),
        (r"\\theta", "θ"),
        (r"\\phi", "φ"),
        (r"\\bar\{θ\}", "θ̄"),
        (r"\\hat", ""),
        (r"\\mathrm\{vec\}", "vec"),
        (r"\\mathrm\{sg\}", "sg"),
        (r"\\sum", "Σ"),
        (r"\\min", "min"),
        (r"\\max", "max"),
        (r"\\quad", "  "),
        (r"\\;", " "),
        (r"\\,", " "),
        (r"\\left", ""),
        (r"\\right", ""),
        (r"\\begin\{[a-z]+\}", ""),
        (r"\\end\{[a-z]+\}", ""),
        (r"\\", ""),
        ("$", ""),
    ]
    for a, b in repl:
        s = s.replace(a, b)
    s = re.sub(r"\s{2,}", " ", s)
    return s


def set_font(run, size=None, bold=None, color=None, mono=False, italic=None):
    run.font.name = "Consolas" if mono else FONT
    run._element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
    if size:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic
    if color:
        run.font.color.rgb = RGBColor(*color)


def shade(el, fill):
    tcPr = el.get_or_add_tcPr() if hasattr(el, "get_or_add_tcPr") else el
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill)
    tcPr.append(shd)


def clean_inline(text: str) -> str:
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    return text


def add_inline(p, text, size=10.5, color=None):
    text = clean_inline(text)
    for part in re.split(r"(\*\*.*?\*\*)", text):
        bold = part.startswith("**") and part.endswith("**") and len(part) > 4
        if bold:
            part = part[2:-2]
        for part2 in re.split(r"(`[^`]*`)", part):
            if not part2:
                continue
            if part2.startswith("`") and part2.endswith("`") and len(part2) > 2:
                r = p.add_run(latex_simplify(part2[1:-1]))
                set_font(r, size - 0.5, bold, (0x1A, 0x1A, 0x1A), mono=True)
            else:
                r = p.add_run(latex_simplify(part2))
                set_font(r, size, bold, color)


def add_equation(text):
    text = (
        text.replace(r"\begin{bmatrix}", "[")
        .replace(r"\end{bmatrix}", "]")
        .replace(r"\\", "; ")
        .replace("&", " ")
    )
    text = latex_simplify(text)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(2)
    r = p.add_run(text)
    set_font(r, 9.6, False, (0x1A, 0x1A, 0x1A))
    r.italic = True
    shade(p._p.get_or_add_pPr(), "F7F9FC")


def add_caption(text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(latex_simplify(clean_inline(text)))
    set_font(r, 8.5, False, (0x60, 0x60, 0x60))


def add_table(rows):
    header = [c.strip() for c in rows[0].strip().strip("|").split("|")]
    body = []
    for line in rows[2:]:
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        body.append(cells)
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = True
    width = 17.0 / len(header)
    for i in range(len(header)):
        t.columns[i].width = Cm(width)
    hdr = t.rows[0].cells
    for i, h in enumerate(header):
        hdr[i].text = ""
        p = hdr[i].paragraphs[0]
        add_inline(p, h, size=8.8)
        for r in p.runs:
            r.bold = True
        shade(hdr[i]._tc, "E8EEF7")
    for row in body:
        cells = t.add_row().cells
        for i, c in enumerate(row[: len(header)]):
            cells[i].text = ""
            p = cells[i].paragraphs[0]
            add_inline(p, c, size=8.4)
    for row in t.rows:
        for cell in row.cells:
            for p in cell.paragraphs:
                p.paragraph_format.space_after = Pt(1.2)
                p.paragraph_format.line_spacing = 1.1
    doc.add_paragraph().paragraph_format.space_after = Pt(0)


def add_image(path, alt):
    path = os.path.join(FIG_DIR, os.path.basename(path))
    if not os.path.exists(path):
        add_caption(f"[缺图: {path}]")
        return
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(path, width=Cm(16.2))
    add_caption(alt)


doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
sec.left_margin = sec.right_margin = Cm(1.9)
sec.top_margin = sec.bottom_margin = Cm(1.9)
st = doc.styles["Normal"]
st.font.name = FONT
st.font.size = Pt(10.5)
st.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
st.paragraph_format.line_spacing = 1.35
st.paragraph_format.space_after = Pt(4)

# ---------------- cover ----------------
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_before = Pt(80)
r = p.add_run("SDPR 能作为 V-JEPA 2 的\n桥接输出层吗？")
set_font(r, 24, True, (0x0D, 0x47, 0xA1))
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = p.add_run("机制分析 + 受控实验（旋转同步）")
set_font(r, 15, False, (0x37, 0x47, 0x4F))
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_before = Pt(24)
r = p.add_run("V-JEPA 2（ViT-L）· SDPRLayers · 隐式微分 · 可认证全局最优")
set_font(r, 11, False, (0x54, 0x6E, 0x7A))
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_before = Pt(30)
r = p.add_run("2026-11 · 依据论文原文 + 官方代码 + 本机 Slurm 实验")
set_font(r, 9.5, False, (0x78, 0x90, 0x9C))
doc.add_page_break()

# ---------------- body ----------------
for idx, path in enumerate(FILES):
    if not os.path.exists(path):
        continue
    lines = open(path, encoding="utf-8").read().splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        s = line.strip()
        if not s:
            i += 1
            continue
        if s.startswith("```"):
            i += 1
            code = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code.append(lines[i])
                i += 1
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Cm(0.4)
            r = p.add_run("\n".join(code))
            set_font(r, 8.4, False, (0x1A, 0x1A, 0x1A), mono=True)
            shade(p._p.get_or_add_pPr(), "F5F5F5")
        elif s.startswith("# "):
            p = doc.add_heading(level=1)
            add_inline(p, s[2:], size=17, color=(0x0D, 0x47, 0xA1))
            for r in p.runs:
                r.bold = True
        elif s.startswith("## "):
            p = doc.add_heading(level=2)
            add_inline(p, s[3:], size=13, color=(0x15, 0x65, 0xC0))
            for r in p.runs:
                r.bold = True
        elif s.startswith("### "):
            p = doc.add_heading(level=3)
            add_inline(p, s[4:], size=11.2, color=(0x2E, 0x7D, 0x32))
            for r in p.runs:
                r.bold = True
        elif s.startswith("> "):
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Cm(0.4)
            p.paragraph_format.space_before = Pt(3)
            add_inline(p, s[2:], size=10.2, color=(0x37, 0x47, 0x4F))
            for r in p.runs:
                r.italic = True
            shade(p._p.get_or_add_pPr(), "F1F8E9")
        elif s.startswith("$$"):
            if s.strip() == "$$":
                i += 1
                eq = []
                while i < len(lines) and lines[i].strip() != "$$":
                    eq.append(lines[i].strip())
                    i += 1
                add_equation(" ".join(eq))
            else:
                add_equation(s.strip().strip("$"))
        elif s.startswith("!["):
            m = re.match(r"!\[(.*?)\]\((.*?)\)", s)
            if m:
                add_image(m.group(2), m.group(1))
        elif s.startswith("- "):
            p = doc.add_paragraph(style="List Bullet")
            add_inline(p, s[2:], size=10.2)
        elif re.match(r"^\d+\.\s+", s):
            num = re.match(r"^(\d+)\.\s+", s).group(1)
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Cm(0.75)
            p.paragraph_format.first_line_indent = Cm(-0.45)
            r = p.add_run(num + ". ")
            set_font(r, 10.2, True)
            add_inline(p, re.sub(r"^\d+\.\s+", "", s), size=10.2)
        elif s.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(lines[i].strip())
                i += 1
            if len(rows) >= 3:
                add_table(rows)
            continue
        elif s.startswith("---"):
            p = doc.add_paragraph()
            p.paragraph_format.space_after = Pt(2)
            pPr = p._p.get_or_add_pPr()
            pbdr = OxmlElement("w:pBdr")
            bottom = OxmlElement("w:bottom")
            bottom.set(qn("w:val"), "single")
            bottom.set(qn("w:sz"), "4")
            bottom.set(qn("w:color"), "B0B0B0")
            pbdr.append(bottom)
            pPr.append(pbdr)
        else:
            p = doc.add_paragraph()
            add_inline(p, s, size=10.5)
        i += 1
    if idx < len(FILES) - 1:
        doc.add_page_break()

os.makedirs(os.path.dirname(OUT), exist_ok=True)
doc.save(OUT)
print(OUT, os.path.getsize(OUT))
