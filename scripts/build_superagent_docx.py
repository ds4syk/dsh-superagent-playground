"""Build the SuperAgent feature-list DOCX from kb/superagent-features/*.md."""
import glob
import os
import re

import pypandoc
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KB = os.path.join(ROOT, "kb", "superagent-features")
OUT = os.path.join(ROOT, "deliverables", "SuperAgent_功能清单_2026-11.docx")
FONT = "Noto Sans CJK SC"
INK = RGBColor(0x1F, 0x29, 0x37)
ACCENT = RGBColor(0x25, 0x63, 0xEB)
MUTED = RGBColor(0x64, 0x74, 0x8B)


def merge_markdown():
    parts = []
    for path in sorted(glob.glob(os.path.join(KB, "*.md"))):
        text = open(path, encoding="utf-8").read()
        # Local cross-page links are useful in the repo; in a printed DOCX they
        # would point at nonexistent paths, so keep only the label.
        text = re.sub(r"\[([^\]]+)\]\((?!https?://)[^)]+\.md\)", r"\1", text)
        text = text.replace("](assets/2026-11-25-superagent-features.png)",
                            "](assets/2026-11-25-superagent-features.png){width=16cm}")
        parts.append(text.strip())
    return "\n\n".join(parts)


def set_east_asian(run, name=FONT):
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    rfonts.set(qn("w:eastAsia"), name)
    rfonts.set(qn("w:ascii"), name)
    rfonts.set(qn("w:hAnsi"), name)


def style_run(run, size=None, bold=None, color=None):
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if color is not None:
        run.font.color.rgb = color
    set_east_asian(run)


def shade(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill)
    tc_pr.append(shd)


def borders(table):
    tbl_pr = table._tbl.tblPr
    borders_el = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), "4")
        el.set(qn("w:space"), "0")
        el.set(qn("w:color"), "D9E2F3")
        borders_el.append(el)
    tbl_pr.append(borders_el)


def add_page_number(section):
    footer = section.footer
    p = footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run("SuperAgent 功能清单（2026-11）  ·  ")
    style_run(run, 8.5, color=MUTED)
    run = p.add_run()
    style_run(run, 8.5, color=MUTED)
    fld_begin = OxmlElement("w:fldChar")
    fld_begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = "PAGE"
    fld_end = OxmlElement("w:fldChar")
    fld_end.set(qn("w:fldCharType"), "end")
    run._r.append(fld_begin)
    run._r.append(instr)
    run._r.append(fld_end)


def main():
    md = merge_markdown()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    pypandoc.convert_text(
        md, "docx", format="markdown", outputfile=OUT,
        extra_args=["--resource-path", KB, "-V", "lang=zh-CN", "--wrap=none"])

    doc = Document(OUT)
    for section in doc.sections:
        section.top_margin = Cm(2.1)
        section.bottom_margin = Cm(2.0)
        section.left_margin = Cm(2.2)
        section.right_margin = Cm(2.2)
        add_page_number(section)

    styles = doc.styles
    normal = styles["Normal"]
    normal.font.name = FONT
    normal.font.size = Pt(10.5)
    normal.font.color.rgb = INK
    normal.paragraph_format.line_spacing = 1.45
    normal.paragraph_format.space_after = Pt(5)
    normal.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)

    for name, size, color, before, after in (
        ("Heading 1", 19, ACCENT, 0, 10),
        ("Heading 2", 14, INK, 14, 6),
        ("Heading 3", 11.5, INK, 10, 4),
    ):
        st = styles[name]
        st.font.name = FONT
        st.font.size = Pt(size)
        st.font.bold = True
        st.font.color.rgb = color
        st.paragraph_format.space_before = Pt(before)
        st.paragraph_format.space_after = Pt(after)
        st.paragraph_format.keep_with_next = True
        st.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)

    # Page break before every H1 except the first; normalise inline runs.
    first_h1 = True
    for p in list(doc.paragraphs):
        if p.style.name == "Heading 1":
            if not first_h1:
                p.paragraph_format.page_break_before = True
            first_h1 = False
        for run in p.runs:
            set_east_asian(run)
            if p.style.name.startswith("Heading"):
                run.font.color.rgb = p.style.font.color.rgb
                run.bold = True

    for table in doc.tables:
        borders(table)
        table.alignment = 1
        for row in table.rows:
            tr_pr = row._tr.get_or_add_trPr()
            cant = OxmlElement("w:cantSplit")
            tr_pr.append(cant)
        for r_idx, row in enumerate(table.rows):
            for cell in row.cells:
                if r_idx == 0:
                    shade(cell, "EFF3FB")
                for p in cell.paragraphs:
                    p.paragraph_format.space_after = Pt(1)
                    p.paragraph_format.line_spacing = 1.2
                    for run in p.runs:
                        style_run(run, 8.6, bold=(r_idx == 0))

    doc.save(OUT)
    print("wrote", OUT, os.path.getsize(OUT))


if __name__ == "__main__":
    main()
