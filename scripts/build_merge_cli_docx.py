"""Build the DOCX reading copy of docs/merge-cli-and-approval.md."""
from __future__ import annotations

import re
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "docs" / "merge-cli-and-approval.md"
OUTPUT = ROOT / "deliverables" / "merge-cli-setup-2026-11.docx"
IMAGE = ROOT / "docs" / "visuals" / "2026-11-11-merge-cli-flow.png"

CN = "Noto Sans CJK SC"
MONO = "DejaVu Sans Mono"
BLUE = RGBColor(0x1F, 0x4E, 0x79)
INK = RGBColor(0x11, 0x18, 0x27)
MUTED = RGBColor(0x4B, 0x55, 0x63)


def set_east_asia(run_or_style, name: str) -> None:
    rpr = run_or_style._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    for attr in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
        rfonts.set(qn(attr), name)


def shade(element, color: str) -> None:
    pr = element.get_or_add_pPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), color)
    pr.append(shd)


def shade_cell(cell, color: str) -> None:
    pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), color)
    pr.append(shd)


def border(paragraph, color: str = "1D4ED8", size: int = 18) -> None:
    pr = paragraph._element.get_or_add_pPr()
    borders = OxmlElement("w:pBdr")
    left = OxmlElement("w:left")
    left.set(qn("w:val"), "single")
    left.set(qn("w:sz"), str(size))
    left.set(qn("w:space"), "8")
    left.set(qn("w:color"), color)
    borders.append(left)
    pr.append(borders)


def add_runs(paragraph, text: str) -> None:
    pattern = re.compile(r"(\*\*.+?\*\*|`[^`]+`|\[[^\]]+\]\([^)]+\))")
    for token in pattern.split(text):
        if not token:
            continue
        if token.startswith("**") and token.endswith("**"):
            run = paragraph.add_run(token[2:-2])
            run.bold = True
        elif token.startswith("`") and token.endswith("`"):
            run = paragraph.add_run(token[1:-1])
            run.font.name = MONO
            run.font.size = Pt(9)
            run.font.color.rgb = RGBColor(0xB9, 0x1C, 0x1C)
        elif token.startswith("[") and "](" in token:
            label, url = token[1:-1].split("](", 1)
            run = paragraph.add_run(label)
            run.underline = True
            run.font.color.rgb = RGBColor(0x1D, 0x4E, 0xD8)
            paragraph.add_run(f"（{url}）").font.size = Pt(8)
        else:
            paragraph.add_run(token)


def page_number(paragraph) -> None:
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run("第 ")
    fld = OxmlElement("w:fldChar")
    fld.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = "PAGE"
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    run._r.append(fld)
    run._r.append(instr)
    run._r.append(end)
    paragraph.add_run(" 页")


def add_table(doc: Document, rows: list[list[str]]) -> None:
    if not rows:
        return
    cols = max(len(row) for row in rows)
    table = doc.add_table(rows=0, cols=cols)
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for index, row in enumerate(rows):
        cells = table.add_row().cells
        for j in range(cols):
            value = row[j] if j < len(row) else ""
            paragraph = cells[j].paragraphs[0]
            paragraph.paragraph_format.space_after = Pt(2)
            if index == 0:
                run = paragraph.add_run(value)
                run.bold = True
                run.font.size = Pt(9.5)
                run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                shade_cell(cells[j], "1F4E79")
            else:
                add_runs(paragraph, value)
                for run in paragraph.runs:
                    run.font.size = Pt(9)
                if index % 2 == 0:
                    shade_cell(cells[j], "F1F5F9")
    # Repeat the header row when the table continues on a later page.
    trpr = table.rows[0]._tr.get_or_add_trPr()
    header_flag = OxmlElement("w:tblHeader")
    header_flag.set(qn("w:val"), "true")
    trpr.append(header_flag)
    doc.add_paragraph()


def build() -> None:
    markdown = SOURCE.read_text(encoding="utf-8").splitlines()
    doc = Document()

    normal = doc.styles["Normal"]
    normal.font.name = CN
    normal.font.size = Pt(10.5)
    normal.font.color.rgb = INK
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.35
    set_east_asia(normal, CN)

    section = doc.sections[0]
    section.top_margin = Cm(1.8)
    section.bottom_margin = Cm(1.8)
    section.left_margin = Cm(2.2)
    section.right_margin = Cm(2.2)

    header = section.header.paragraphs[0]
    header.text = "配置 merge CLI 与合并审批规则"
    header.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    for run in header.runs:
        run.font.size = Pt(8.5)
        run.font.color.rgb = MUTED
    page_number(section.footer.paragraphs[0])

    # Cover
    title = doc.add_paragraph()
    title.paragraph_format.space_before = Pt(90)
    run = title.add_run("配置 merge CLI 与合并审批规则")
    run.bold = True
    run.font.size = Pt(26)
    run.font.color.rgb = BLUE
    run.font.name = CN
    set_east_asia(run, CN)
    subtitle = doc.add_paragraph()
    run = subtitle.add_run("dsh-superagent-playground · 2026-11")
    run.font.size = Pt(12)
    run.font.color.rgb = MUTED
    note = doc.add_paragraph()
    note.paragraph_format.space_before = Pt(24)
    shade(note._p, "EFF6FF")
    border(note)
    add_runs(note, "**阅读提示**：这是一份可执行的方案，不是已经生效的配置。当前部署仍是默认状态（无 merge CLI、未开 auto-merge、main 无保护）；第 3 节是执行顺序，第 4 节列出需要你决定或操作的事项。")
    for run in note.runs:
        run.font.size = Pt(10)

    doc.add_page_break()

    i = 0
    in_code = False
    code: list[str] = []
    paragraph: list[str] = []
    quote: list[str] = []
    table: list[list[str]] = []

    def flush_paragraph() -> None:
        nonlocal paragraph
        if paragraph:
            p = doc.add_paragraph()
            add_runs(p, " ".join(paragraph))
            paragraph = []

    def flush_quote() -> None:
        nonlocal quote
        if quote:
            p = doc.add_paragraph()
            shade(p._p, "EFF6FF")
            border(p)
            p.paragraph_format.left_indent = Cm(0.3)
            add_runs(p, "\n".join(quote))
            quote = []

    def flush_table() -> None:
        nonlocal table
        if table:
            add_table(doc, table)
            table = []

    while i < len(markdown):
        line = markdown[i]
        if in_code:
            if line.strip().startswith("```"):
                in_code = False
                p = doc.add_paragraph()
                shade(p._p, "F1F5F9")
                p.paragraph_format.left_indent = Cm(0.25)
                p.paragraph_format.space_after = Pt(8)
                run = p.add_run("\n".join(code))
                run.font.name = MONO
                run.font.size = Pt(8.5)
                code = []
            else:
                code.append(line)
            i += 1
            continue

        if line.startswith("```"):
            flush_paragraph(); flush_quote(); flush_table()
            in_code = True
            i += 1
            continue

        if line.startswith("|"):
            flush_paragraph(); flush_quote()
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            if not all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells):
                table.append(cells)
            i += 1
            continue
        if table:
            flush_table()

        if line.startswith("# "):
            flush_paragraph(); flush_quote()
            # The cover already carries the title.
        elif line.startswith("## "):
            flush_paragraph(); flush_quote()
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(14)
            run = p.add_run(line[3:])
            run.bold = True
            run.font.size = Pt(15)
            run.font.color.rgb = BLUE
        elif line.startswith("### "):
            flush_paragraph(); flush_quote()
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(8)
            run = p.add_run(line[4:])
            run.bold = True
            run.font.size = Pt(12)
            run.font.color.rgb = RGBColor(0x0F, 0x17, 0x2A)
        elif line.startswith(">"):
            flush_paragraph()
            quote.append(line.lstrip("> ").strip())
        elif line.startswith("!["):
            flush_paragraph(); flush_quote()
            match = re.search(r"\]\(([^)]+)\)", line)
            if match:
                path = (SOURCE.parent / match.group(1)).resolve()
                doc.add_picture(str(path), width=Inches(6.2))
                doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        elif re.match(r"^- \[[ xX]\] ", line):
            flush_paragraph(); flush_quote()
            checked = line[3] in "xX"
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Cm(0.45)
            p.paragraph_format.first_line_indent = Cm(-0.45)
            p.add_run("\u2611 " if checked else "\u2610 ")
            add_runs(p, line[6:])
            for run in p.runs:
                run.font.size = Pt(10.5)
        elif re.match(r"^- ", line):
            flush_paragraph(); flush_quote()
            p = doc.add_paragraph(style="List Bullet")
            add_runs(p, line[2:])
            for run in p.runs:
                run.font.size = Pt(10.5)
        elif re.match(r"^\d+\. ", line):
            flush_paragraph(); flush_quote()
            match = re.match(r"^(\d+)\. (.*)", line)
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Cm(0.75)
            p.paragraph_format.first_line_indent = Cm(-0.75)
            number = p.add_run(match.group(1) + ". ")
            number.bold = True
            add_runs(p, match.group(2))
            for run in p.runs:
                run.font.size = Pt(10.5)
        elif not line.strip():
            flush_paragraph(); flush_quote()
        else:
            flush_quote()
            paragraph.append(line.strip())
        i += 1

    flush_paragraph(); flush_quote(); flush_table()

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUTPUT)
    print(f"saved {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    build()
