"""Build the SDPR main-table design knowledge base DOCX from markdown pages."""
import glob
import os
import pypandoc
from docx import Document
from docx.shared import Cm, Pt, RGBColor
from docx.oxml.ns import qn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KB = os.path.join(ROOT, "kb", "sdpr-main-table")
OUT = os.path.join(ROOT, "deliverables", "SDPR新方法主表实验设计知识库_2026-11.docx")
VISUAL = os.path.join(ROOT, "docs", "visuals",
                      "2026-11-09-sdpr-main-table-design.png")

FONT = "Noto Sans CJK SC"

TOC = """# 目录

- **00 总览**：结论、创新点决策表、证据分级
- **01 数据集选择与评测协议**：SPEED+ 与第二数据集、指标、防泄漏
- **02 Baseline 分层与可复现性**：经典几何 / 可微与可认证 / 领域 SOTA
- **03 主表模板**：六张表的列、单位与自检清单
- **04 文献范式**：七篇论文的主表拆解与两类范式
- **05 风险、审稿问题与执行清单**：P0–P2 行动项
"""

os.makedirs(os.path.dirname(OUT), exist_ok=True)

pages = sorted(glob.glob(os.path.join(KB, "0*.md")))
parts = [TOC]
for i, path in enumerate(pages):
    with open(path, encoding="utf-8") as f:
        body = f.read()
    if i == 0:
        body = body.replace(
            "## 资料索引",
            "![一页速览：主表设计决策图](" + VISUAL + "){width=6.3in}\n\n## 资料索引",
            1,
        )
    parts.append(body)

text = "\n\n\\newpage\n\n".join(parts)
pypandoc.convert_text(
    text,
    "docx",
    format="markdown",
    outputfile=OUT,
    extra_args=[
        "-V", "lang=zh-CN",
        f"--resource-path={ROOT}",
    ],
)

# ---- post-process: A4, margins, Chinese font, compact tables ----
doc = Document(OUT)
for sec in doc.sections:
    sec.page_width = Cm(21.0)
    sec.page_height = Cm(29.7)
    sec.left_margin = sec.right_margin = Cm(2.2)
    sec.top_margin = sec.bottom_margin = Cm(2.2)

normal = doc.styles["Normal"]
normal.font.name = FONT
normal.font.size = Pt(10.5)
normal.element.get_or_add_rPr().get_or_add_rFonts().set(qn("w:eastAsia"), FONT)
normal.paragraph_format.line_spacing = 1.3
normal.paragraph_format.space_after = Pt(4)

for name, size in (("Heading 1", 18), ("Heading 2", 13), ("Heading 3", 11)):
    st = doc.styles[name]
    st.font.name = FONT
    st.font.size = Pt(size)
    st.font.color.rgb = RGBColor(0x1F, 0x38, 0x64)
    st.element.get_or_add_rPr().get_or_add_rFonts().set(qn("w:eastAsia"), FONT)
    st.paragraph_format.keep_with_next = True

from docx.oxml import OxmlElement
from docx.shared import Cm as _Cm

for table in doc.tables:
    table.autofit = False
    tblPr = table._tbl.tblPr
    layout = OxmlElement("w:tblLayout")
    layout.set(qn("w:type"), "fixed")
    tblPr.append(layout)
    # column widths weighted by the longest line in each column
    ncol = len(table.columns)
    weights = []
    for j in range(ncol):
        best = 4
        for cell in table.columns[j].cells:
            for para in cell.paragraphs:
                best = max(best, len(para.text))
        weights.append(best)
    if weights:
        weights[0] *= 1.5  # method-name column
    avail = 16.6  # cm between margins
    total = sum(weights)
    widths = [max(2.0, min(7.6, avail * w / total)) for w in weights]
    scale = avail / sum(widths)
    widths = [w * scale for w in widths]
    for j, w in enumerate(widths):
        for cell in table.columns[j].cells:
            cell.width = _Cm(w)
    for ri, row in enumerate(table.rows):
        trPr = row._tr.get_or_add_trPr()
        trPr.append(OxmlElement("w:cantSplit"))
        if ri == 0:
            trPr.append(OxmlElement("w:tblHeader"))
        for cell in row.cells:
            for para in cell.paragraphs:
                para.paragraph_format.space_after = Pt(1)
                para.paragraph_format.line_spacing = 1.1
                for run in para.runs:
                    run.font.size = Pt(8.5)

doc.save(OUT)
print("wrote", OUT, os.path.getsize(OUT))
