"""A small Markdown -> DOCX renderer for the SDPR report.

Supports the subset used by report/sdpr_forward_network_dataset.md:
headings, paragraphs, callout blockquotes, bullets, numbered steps,
fenced code, GFM tables, local images, bold/code/links.
"""
from __future__ import annotations

import re
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

FONT_SERIF = "Noto Serif CJK SC"
FONT_SANS = "Noto Sans CJK SC"
FONT_MONO = "DejaVu Sans Mono"

GREEN = RGBColor(0x2E, 0x6B, 0x3A)
BLUE = RGBColor(0x1F, 0x4E, 0x79)
GRAY = RGBColor(0x59, 0x59, 0x59)
DARK = RGBColor(0x1A, 0x1A, 0x1A)
RED = RGBColor(0xB0, 0x3A, 0x2E)

CJK = re.compile(r"[\u3400-\u9fff\uf900-\ufaff]")


def set_run_font(run, name=FONT_SERIF, size=10.5, bold=None, italic=None, color=None):
    run.font.name = name
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    rfonts.set(qn("w:ascii"), name)
    rfonts.set(qn("w:hAnsi"), name)
    rfonts.set(qn("w:eastAsia"), name)
    run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic
    if color is not None:
        run.font.color.rgb = color


def shade(el, fill):
    tcPr = el.get_or_add_tcPr() if hasattr(el, "get_or_add_tcPr") else el
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill)
    tcPr.append(shd)


def para_shade(p, fill):
    pPr = p._p.get_or_add_pPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill)
    pPr.append(shd)


def para_border(p, color="2E6B3A", size=18, side="left"):
    pPr = p._p.get_or_add_pPr()
    pbdr = OxmlElement("w:pBdr")
    b = OxmlElement(f"w:{side}")
    b.set(qn("w:val"), "single")
    b.set(qn("w:sz"), str(size))
    b.set(qn("w:space"), "6")
    b.set(qn("w:color"), color)
    pbdr.append(b)
    pPr.append(pbdr)


def heading(doc, text, level=1):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(16 if level == 1 else 11)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.keep_with_next = True
    if text.strip() == "参考":
        p.paragraph_format.page_break_before = True
    r = p.add_run(text)
    if level == 1:
        set_run_font(r, FONT_SANS, 17, bold=True, color=GREEN)
    elif level == 2:
        set_run_font(r, FONT_SANS, 13.5, bold=True, color=BLUE)
    else:
        set_run_font(r, FONT_SANS, 11.5, bold=True, color=BLUE)
    outline = OxmlElement("w:outlineLvl")
    outline.set(qn("w:val"), str(level - 1))
    p._p.get_or_add_pPr().append(outline)
    return p


def add_hyperlink(paragraph, url, text):
    part = paragraph.part
    r_id = part.relate_to(
        url,
        "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink",
        is_external=True,
    )
    hyperlink = OxmlElement("w:hyperlink")
    hyperlink.set(qn("r:id"), r_id)
    new_run = OxmlElement("w:r")
    rPr = OxmlElement("w:rPr")
    color = OxmlElement("w:color")
    color.set(qn("w:val"), "1F4E79")
    underline = OxmlElement("w:u")
    underline.set(qn("w:val"), "single")
    rPr.append(color)
    rPr.append(underline)
    new_run.append(rPr)
    t = OxmlElement("w:t")
    t.text = text
    new_run.append(t)
    hyperlink.append(new_run)
    paragraph._p.append(hyperlink)


INLINE = re.compile(r"(\*\*.+?\*\*|`[^`]+`|\[[^\]]+\]\([^)]+\)|\*[^*\n]+?\*|<https?://[^>]+>)")


def add_inline(paragraph, text, size=10.5, color=DARK, base_name=FONT_SERIF):
    for token in INLINE.split(text):
        if not token:
            continue
        if token.startswith("**") and token.endswith("**") and len(token) > 4:
            r = paragraph.add_run(token[2:-2])
            set_run_font(r, base_name, size, bold=True, color=color)
        elif token.startswith("`") and token.endswith("`") and len(token) > 2:
            r = paragraph.add_run(token[1:-1])
            set_run_font(r, FONT_MONO, size - 0.8, color=RGBColor(0x8B, 0x1A, 0x1A))
        elif token.startswith("[") and "](" in token and token.endswith(")"):
            m = re.match(r"\[([^\]]+)\]\(([^)]+)\)", token)
            add_hyperlink(paragraph, m.group(2), m.group(1))
        elif token.startswith("<http") and token.endswith(">"):
            add_hyperlink(paragraph, token[1:-1], token[1:-1])
        elif token.startswith("*") and token.endswith("*") and len(token) > 2:
            r = paragraph.add_run(token[1:-1])
            set_run_font(r, base_name, size, italic=True, color=color)
        else:
            r = paragraph.add_run(token)
            set_run_font(r, base_name, size, color=color)


def smart_join(lines):
    """Join wrapped Markdown lines without inserting spaces between CJK chars."""
    out = ""
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if not out:
            out = line
            continue
        if CJK.search(out[-1]) and CJK.search(line[0]):
            out += line
        else:
            out += " " + line
    return out


def add_callout(doc, text):
    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = True
    cell = table.cell(0, 0)
    cell.width = Cm(16.5)
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), "EAF3E7")
    tcPr.append(shd)
    # left accent border
    tcBorders = OxmlElement("w:tcBorders")
    left = OxmlElement("w:left")
    left.set(qn("w:val"), "single")
    left.set(qn("w:sz"), "24")
    left.set(qn("w:space"), "0")
    left.set(qn("w:color"), "2E6B3A")
    tcBorders.append(left)
    tcPr.append(tcBorders)
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.line_spacing = 1.3
    add_inline(p, text, size=10.5, color=DARK)
    doc.add_paragraph().paragraph_format.space_after = Pt(0)


def add_code(doc, lines):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(8)
    p.paragraph_format.left_indent = Cm(0.4)
    p.paragraph_format.line_spacing = 1.15
    para_shade(p, "F2F2F2")
    para_border(p, color="BFBFBF", size=6, side="left")
    for i, line in enumerate(lines):
        run = p.add_run(line)
        set_run_font(run, FONT_MONO, 8.8, color=RGBColor(0x33, 0x33, 0x33))
        if i < len(lines) - 1:
            run.add_break()


def add_table(doc, rows):
    header = rows[0]
    body = rows[1:]
    table = doc.add_table(rows=1 + len(body), cols=len(header))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = True
    for j, text in enumerate(header):
        cell = table.cell(0, j)
        cell.paragraphs[0].paragraph_format.space_before = Pt(2)
        cell.paragraphs[0].paragraph_format.space_after = Pt(2)
        shade(cell._tc, "D9E2F3")
        add_inline(cell.paragraphs[0], text, size=9.2, color=DARK, base_name=FONT_SANS)
        for run in cell.paragraphs[0].runs:
            run.bold = True
    # repeat header row and prevent rows from splitting across pages
    trPr = table.rows[0]._tr.get_or_add_trPr()
    tblHeader = OxmlElement("w:tblHeader")
    tblHeader.set(qn("w:val"), "true")
    trPr.append(tblHeader)
    for row in table.rows:
        rp = row._tr.get_or_add_trPr()
        cs = OxmlElement("w:cantSplit")
        rp.append(cs)
    for i, row in enumerate(body, start=1):
        for j in range(len(header)):
            cell = table.cell(i, j)
            cell.paragraphs[0].paragraph_format.space_before = Pt(1.5)
            cell.paragraphs[0].paragraph_format.space_after = Pt(1.5)
            add_inline(cell.paragraphs[0], row[j] if j < len(row) else "", size=9.2,
                       color=DARK)
    doc.add_paragraph().paragraph_format.space_after = Pt(0)


def add_image(doc, path: Path, caption):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(4)
    p.add_run().add_picture(str(path), width=Cm(16.4))
    if caption:
        c = doc.add_paragraph()
        c.alignment = WD_ALIGN_PARAGRAPH.CENTER
        c.paragraph_format.space_after = Pt(8)
        add_inline(c, caption, size=9.2, color=GRAY)


def add_footer(section, text):
    footer = section.footer
    p = footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(text + "    第 ")
    set_run_font(run, FONT_SANS, 8.5, color=GRAY)
    fld = OxmlElement("w:fldSimple")
    fld.set(qn("w:instr"), "PAGE")
    r = OxmlElement("w:r")
    t = OxmlElement("w:t")
    t.text = "1"
    r.append(t)
    fld.append(r)
    p._p.append(fld)
    run2 = p.add_run(" 页")
    set_run_font(run2, FONT_SANS, 8.5, color=GRAY)


def is_block_start(s):
    return (
        re.match(r"^(#{1,3})\s+", s)
        or s.startswith("|")
        or s.startswith("> ")
        or re.match(r"^[-*]\s+", s)
        or re.match(r"^\d+\.\s+", s)
        or s.startswith("```")
        or s.startswith("![")
    )


def collect_item(lines, i):
    item = lines[i].strip()
    i += 1
    cont = []
    while i < len(lines):
        nxt = lines[i].strip()
        if not nxt or is_block_start(nxt):
            break
        cont.append(nxt)
        i += 1
    return item, cont, i


def parse_table(lines, i):
    rows = []
    while i < len(lines) and lines[i].strip().startswith("|"):
        raw = lines[i].strip().strip("|")
        cells = [c.strip() for c in raw.split("|")]
        if not all(re.fullmatch(r":?-+:?", c) for c in cells):
            rows.append(cells)
        i += 1
    return rows, i


def render(md_path: Path, out_path: Path, title: str, subtitle: str, meta: str,
           footer: str = "SDPR 前向网络与数据集 · 2026-11-09"):
    text = md_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    md_dir = md_path.parent

    doc = Document()
    section = doc.sections[0]
    section.page_height = Cm(29.7)
    section.page_width = Cm(21.0)
    section.top_margin = Cm(2.1)
    section.bottom_margin = Cm(1.9)
    section.left_margin = Cm(2.2)
    section.right_margin = Cm(2.2)

    normal = doc.styles["Normal"]
    normal.font.name = FONT_SERIF
    normal.font.size = Pt(10.5)
    normal.element.rPr.rFonts.set(qn("w:eastAsia"), FONT_SERIF)

    # Title block
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(2)
    r = p.add_run(title)
    set_run_font(r, FONT_SANS, 20, bold=True, color=GREEN)

    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(2)
    r = p.add_run(subtitle)
    set_run_font(r, FONT_SANS, 11, color=BLUE)

    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(10)
    r = p.add_run(meta)
    set_run_font(r, FONT_SANS, 9, color=GRAY)

    # thin rule
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(4)
    para_border(p, color="2E6B3A", size=8, side="bottom")

    add_footer(section, footer)

    i = 0
    skip_title = True
    while i < len(lines):
        line = lines[i]
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
            i += 1
            add_code(doc, code)
            continue

        m = re.match(r"^(#{1,3})\s+(.*)$", s)
        if m:
            level = len(m.group(1))
            heading_text = m.group(2)
            if level == 1 and skip_title:
                skip_title = False
                i += 1
                continue
            heading(doc, heading_text, level=min(level, 3))
            i += 1
            continue

        if s.startswith("|"):
            rows, i = parse_table(lines, i)
            if rows:
                add_table(doc, rows)
            continue

        if s.startswith("!["):
            m = re.match(r"!\[([^\]]*)\]\(([^)]+)\)", s)
            if m:
                caption, rel = m.group(1), m.group(2)
                add_image(doc, (md_dir / rel).resolve(), caption)
            i += 1
            continue

        if s.startswith("> "):
            buf = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                buf.append(lines[i].strip()[1:].strip())
                i += 1
            add_callout(doc, smart_join(buf))
            continue

        if re.match(r"^[-*]\s+", s):
            while i < len(lines) and re.match(r"^[-*]\s+", lines[i].strip()):
                item, cont, i = collect_item(lines, i)
                text = re.sub(r"^[-*]\s+", "", item)
                p = doc.add_paragraph(style="List Bullet")
                p.paragraph_format.space_after = Pt(2)
                p.paragraph_format.line_spacing = 1.28
                add_inline(p, smart_join([text] + cont), size=10.5)
            continue

        if re.match(r"^\d+\.\s+", s):
            while i < len(lines) and re.match(r"^\d+\.\s+", lines[i].strip()):
                item, cont, i = collect_item(lines, i)
                num, rest = item.split(".", 1)
                rest = rest.strip()
                p = doc.add_paragraph()
                p.paragraph_format.left_indent = Cm(0.75)
                p.paragraph_format.first_line_indent = Cm(-0.55)
                p.paragraph_format.space_after = Pt(3)
                p.paragraph_format.line_spacing = 1.28
                r = p.add_run(num + ". ")
                set_run_font(r, FONT_SANS, 10.5, bold=True, color=BLUE)
                add_inline(p, smart_join([rest] + cont), size=10.5)
            continue

        # paragraph: collect until blank / block start
        buf = []
        while i < len(lines):
            nxt = lines[i].strip()
            if not nxt:
                break
            if re.match(r"^(#{1,3})\s+", nxt) or nxt.startswith("|") or nxt.startswith("> ") \
               or re.match(r"^[-*]\s+", nxt) or re.match(r"^\d+\.\s+", nxt) \
               or nxt.startswith("```") or nxt.startswith("!["):
                break
            buf.append(nxt)
            i += 1
        if buf:
            p = doc.add_paragraph()
            p.paragraph_format.space_after = Pt(5)
            p.paragraph_format.line_spacing = 1.32
            add_inline(p, smart_join(buf), size=10.5)

    doc.save(out_path)
    return out_path
