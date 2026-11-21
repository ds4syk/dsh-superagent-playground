# -*- coding: utf-8 -*-
"""Build the DOCX knowledge base from kb/reimbursement-ai/*.md.

Run with the env that has python-docx::

    ~/miniconda3/envs/sdpr-clean/bin/python src/build_docx_reimbursement.py
"""
import os
import re
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Cm, Pt, RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

import md2docx as M

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
KB = os.path.join(ROOT, "kb", "reimbursement-ai")
OUT = os.path.join(ROOT, "deliverables", "高校报销与境外AI订阅_2026-11.docx")
VISUAL = os.path.join(ROOT, "docs", "visuals", "2026-11-21-ai-reimbursement.png")

FILES = [os.path.join(KB, f"{i:02d}.md") for i in range(7)]


def clean_links(text: str) -> str:
    """Turn relative KB links into plain labels (DOCX has no page navigation)."""
    return re.sub(r"\[([^\]]+)\]\(0[0-6]\.md\)", r"\1", text)


def body(doc, md_path: str):
    text = clean_links(Path(md_path).read_text(encoding="utf-8"))
    text = re.sub(r"(?m)^\s*---\s*$", "", text)
    lines = text.splitlines()
    md_dir = Path(md_path).parent

    i = 0
    skip_title = False
    first_h1 = True
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
            M.add_code(doc, code)
            continue

        m = re.match(r"^(#{1,3})\s+(.*)$", s)
        if m:
            level = len(m.group(1))
            heading_text = m.group(2)
            if level == 1 and skip_title:
                skip_title = False
                i += 1
                continue
            h = M.heading(doc, heading_text, level=min(level, 3))
            if level == 1 and first_h1:
                h.paragraph_format.page_break_before = True
                first_h1 = False
            i += 1
            continue

        if s.startswith("|"):
            rows, i = M.parse_table(lines, i)
            if rows:
                M.add_table(doc, rows)
            continue

        if s.startswith("!["):
            m = re.match(r"!\[([^\]]*)\]\(([^)]+)\)", s)
            if m:
                caption, rel = m.group(1), m.group(2)
                M.add_image(doc, (md_dir / rel).resolve(), caption)
            i += 1
            continue

        if s.startswith("> "):
            buf = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                buf.append(lines[i].strip()[1:].strip())
                i += 1
            M.add_callout(doc, M.smart_join(buf))
            continue

        if re.match(r"^[-*]\s+", s):
            while i < len(lines) and re.match(r"^[-*]\s+", lines[i].strip()):
                item, cont, i = M.collect_item(lines, i)
                t = re.sub(r"^[-*]\s+", "", item)
                p = doc.add_paragraph(style="List Bullet")
                p.paragraph_format.space_after = Pt(2)
                p.paragraph_format.line_spacing = 1.28
                M.add_inline(p, M.smart_join([t] + cont), size=10.5)
            continue

        if re.match(r"^\d+\.\s+", s):
            while i < len(lines) and re.match(r"^\d+\.\s+", lines[i].strip()):
                item, cont, i = M.collect_item(lines, i)
                num, rest = item.split(".", 1)
                rest = rest.strip()
                p = doc.add_paragraph()
                p.paragraph_format.left_indent = Cm(0.75)
                p.paragraph_format.first_line_indent = Cm(-0.55)
                p.paragraph_format.space_after = Pt(3)
                p.paragraph_format.line_spacing = 1.28
                r = p.add_run(num + ". ")
                M.set_run_font(r, M.FONT_SANS, 10.5, bold=True, color=M.BLUE)
                M.add_inline(p, M.smart_join([rest] + cont), size=10.5)
            continue

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
            M.add_inline(p, M.smart_join(buf), size=10.5)


def cover(doc):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(90)
    p.paragraph_format.space_after = Pt(4)
    r = p.add_run("高校报销与境外 AI 订阅")
    M.set_run_font(r, M.FONT_SANS, 28, bold=True, color=M.GREEN)

    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(4)
    r = p.add_run("境外 AI 订阅（ChatGPT / Claude）能否按“技术服务费”报销？")
    M.set_run_font(r, M.FONT_SANS, 14, bold=True, color=M.BLUE)

    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(18)
    r = p.add_run("知识库 · 2026-11 · 研究整理，不构成法律 / 税务意见")
    M.set_run_font(r, M.FONT_SANS, 10, color=M.GRAY)

    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(24)
    M.para_border(p, color="2E6B3A", size=10, side="bottom")

    p = doc.add_paragraph()
    p.paragraph_format.line_spacing = 1.7
    r = p.add_run("一句话：")
    M.set_run_font(r, M.FONT_SANS, 11, bold=True, color=M.GREEN)
    r = p.add_run("能不能报，不取决于发票叫什么，而取决于业务真不真、科目有没有、票据配不配、数据能不能出境。")
    M.set_run_font(r, M.FONT_SERIF, 11, color=M.DARK)

    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.line_spacing = 1.6
    r = p.add_run("本册覆盖：高校/课题组的经费与科目、境外 AI 订阅的票据/税务/外汇、"
                  "“技术服务费”的合规边界、各校口径样本、五条路径与风险清单。")
    M.set_run_font(r, M.FONT_SERIF, 10.5, color=M.GRAY)

    # A small index box
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(22)
    M.add_callout(
        doc,
        "阅读建议：先看第 00 页的六条结论，再看第 05 页的决策树与“问财务的 6 个问题”；"
        "需要判断“技术服务费”风险时看第 03 页；需要核对政策原文时看各页“依据”链接。",
    )


def main():
    doc = Document()
    section = doc.sections[0]
    section.page_height = Cm(29.7)
    section.page_width = Cm(21.0)
    section.top_margin = Cm(2.1)
    section.bottom_margin = Cm(1.9)
    section.left_margin = Cm(2.2)
    section.right_margin = Cm(2.2)

    normal = doc.styles["Normal"]
    normal.font.name = M.FONT_SERIF
    normal.font.size = Pt(10.5)
    normal.element.rPr.rFonts.set(qn("w:eastAsia"), M.FONT_SERIF)

    M.add_footer(section, "高校报销与境外 AI 订阅 · 2026-11")

    cover(doc)

    # Decision graphic on its own page
    ip = M.add_image(doc, VISUAL, "图 0：境外 AI 订阅报销决策图（2026-11）")
    ip.paragraph_format.page_break_before = True

    for idx, f in enumerate(FILES):
        if not os.path.exists(f):
            raise SystemExit(f"missing {f}")
        body(doc, f)

    doc.save(OUT)
    print("wrote", OUT, os.path.getsize(OUT))


if __name__ == "__main__":
    main()
