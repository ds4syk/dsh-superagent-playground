"""极简 Markdown -> DOCX（标题/段落/列表/表格/粗体/行内代码/图片）。"""
from __future__ import annotations

import re
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

FONT = "Noto Sans CJK SC"
CLEAN_MATH = False

_GREEK = {
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ε",
    "varepsilon": "ε", "theta": "θ", "lambda": "λ", "mu": "μ", "nu": "ν",
    "pi": "π", "rho": "ρ", "sigma": "σ", "tau": "τ", "phi": "φ", "varphi": "φ",
    "omega": "ω", "Gamma": "Γ", "Delta": "Δ", "Theta": "Θ", "Lambda": "Λ",
    "Sigma": "Σ", "Phi": "Φ", "Omega": "Ω",
}


def _clean_math(text):
    """把 $...$ 内的 LaTeX 粗略转成可读 Unicode（不追求排版）。"""
    def repl(m):
        s = m.group(1)
        s = re.sub(r"\\bm\s*", "", s)
        s = re.sub(r"\\mathbb\s*([SR])", r"\1", s)
        s = re.sub(r"\\mathrm\s*\{([^}]*)\}", r"\1", s)
        s = re.sub(r"\\operatorname\s*\{([^}]*)\}", r"\1", s)
        s = re.sub(r"\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"(\1)/(\2)", s)
        s = re.sub(r"\\sqrt\s*\{([^{}]*)\}", r"√(\1)", s)
        for k, v in _GREEK.items():
            s = s.replace("\\" + k, v)
        s = s.replace("\\top", "ᵀ").replace("\\le", "≤").replace("\\leq", "≤")
        s = s.replace("\\ge", "≥").replace("\\geq", "≥").replace("\\times", "×")
        s = s.replace("\\cdot", "·").replace("\\to", "→").replace("\\mapsto", "↦")
        s = s.replace("\\infty", "∞").replace("\\pm", "±").replace("\\approx", "≈")
        s = s.replace("\\nabla", "∇").replace("\\partial", "∂")
        s = s.replace("\\sum", "Σ").replace("\\prod", "∏").replace("\\int", "∫")
        s = s.replace("\\left", "").replace("\\right", "")
        s = s.replace("\\,", " ").replace("\\;", " ").replace("\\!", "")
        s = s.replace("\\max", "max").replace("\\min", "min").replace("\\log", "log")
        s = s.replace("\\det", "det ").replace("\\exp", "exp ")
        s = s.replace("\\frac12", "1/2").replace("\\frac{1}{2}", "1/2")
        s = s.replace("\\succeq", "≽").replace("\\preceq", "≼")
        s = s.replace("\\succ", "≻").replace("\\prec", "≺")
        s = s.replace("\\rm", "")
        s = s.replace("\\star", "⋆").replace("\\ast", "∗")
        s = s.replace("\\supsetneq", "⊋").replace("\\subsetneq", "⊊")
        s = s.replace("\\setminus", "∖").replace("\\in", "∈")
        s = s.replace("\\subset", "⊂").replace("\\subseteq", "⊆")
        s = s.replace("\\cup", "∪").replace("\\cap", "∩")
        s = s.replace("\\forall", "∀").replace("\\exists", "∃")
        s = s.replace("\\Rightarrow", "⇒").replace("\\Leftrightarrow", "⇔")
        s = s.replace("\\equiv", "≡").replace("\\sim", "∼").replace("\\propto", "∝")
        s = s.replace("\\langle", "⟨").replace("\\rangle", "⟩")
        s = s.replace("\\lVert", "‖").replace("\\rVert", "‖")
        s = s.replace("\\quad", "  ").replace("\\qquad", "   ")
        s = s.replace("\\dots", "…").replace("\\cdots", "⋯")
        s = s.replace("\\otimes", "⊗").replace("\\odot", "⊙")
        s = s.replace("\\dagger", "†").replace("\\prime", "′")
        s = s.replace("\\begin", "").replace("\\end", "")
        s = s.replace("\\\\", "; ")
        s = re.sub(r"\\mathcal\s*\{([^}]*)\}", r"\1", s)
        s = re.sub(r"\\([A-Za-z]+)", r"\1", s)  # 未知命令去反斜杠
        s = s.replace("{", "").replace("}", "")
        s = re.sub(r"_\s+", "_", s)
        s = re.sub(r"\^\s+", "^", s)
        return s
    return re.sub(r"\$([^$]+)\$", repl, text)


def _set_font(run, size=None, bold=None, color=None):
    run.font.name = FONT
    run._element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if color is not None:
        run.font.color.rgb = color


def _add_inline(p, text, size=10.5):
    """处理 **bold**、`code`、$math$。"""
    if CLEAN_MATH:
        text = _clean_math(text)
    tokens = re.split(r"(\*\*[^*]+\*\*|`[^`]+`)", text)
    for tok in tokens:
        if not tok:
            continue
        if tok.startswith("**") and tok.endswith("**"):
            r = p.add_run(tok[2:-2]); _set_font(r, size, bold=True)
        elif tok.startswith("`") and tok.endswith("`"):
            r = p.add_run(tok[1:-1]); _set_font(r, size - 0.5)
            r.font.color.rgb = RGBColor(0x8B, 0x1A, 0x1A)
        else:
            r = p.add_run(tok); _set_font(r, size)


def _is_table_sep(line):
    return bool(re.match(r"^\s*\|?[\s:|-]+\|[\s:|-]*$", line)) and "-" in line


def _split_row(line):
    line = line.strip().strip("|")
    return [c.strip() for c in line.split("|")]


def md_to_docx(md_path, out_path, title=None, subtitle=None, image=None):
    md_path = Path(md_path)
    lines = md_path.read_text(encoding="utf-8").splitlines()
    doc = Document()
    style = doc.styles["Normal"]
    style.font.name = FONT
    style.font.size = Pt(10.5)
    style._element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
    for section in doc.sections:
        section.top_margin = Pt(54)
        section.bottom_margin = Pt(54)
        section.left_margin = Pt(56)
        section.right_margin = Pt(56)

    if title:
        h = doc.add_heading(title, level=0)
        for r in h.runs:
            _set_font(r, 18, bold=True, color=RGBColor(0x1A, 0x1A, 0x1A))
    if subtitle:
        p = doc.add_paragraph()
        r = p.add_run(subtitle); _set_font(r, 9, color=RGBColor(0x66, 0x66, 0x66))
        r.italic = True
    if image and Path(image).exists():
        doc.add_picture(str(image), width=Inches(6.6))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER

    i = 0
    n = len(lines)
    while i < n:
        line = lines[i].rstrip()
        if not line.strip():
            i += 1
            continue
        # heading
        m = re.match(r"^(#{1,4})\s+(.*)$", line)
        if m:
            level = len(m.group(1))
            h = doc.add_heading(m.group(2), level=min(level, 4))
            for r in h.runs:
                _set_font(r, {1: 15, 2: 13, 3: 11.5, 4: 11}[min(level, 4)],
                          bold=True, color=RGBColor(0x1A, 0x1A, 0x1A))
            i += 1
            continue
        # code fence
        if line.strip().startswith("```"):
            j = i + 1
            block = []
            while j < n and not lines[j].strip().startswith("```"):
                block.append(lines[j])
                j += 1
            for bl in block:
                p = doc.add_paragraph()
                r = p.add_run(bl if bl else " ")
                r.font.name = "Noto Sans Mono"
                r._element.rPr.rFonts.set(qn("w:eastAsia"), "Noto Sans CJK SC")
                r.font.size = Pt(8.5)
                p.paragraph_format.space_after = Pt(0)
                p.paragraph_format.left_indent = Pt(6)
            doc.add_paragraph()
            i = j + 1
            continue
        # table
        if line.strip().startswith("|") and i + 1 < n and _is_table_sep(lines[i + 1]):
            header = _split_row(line)
            rows = []
            j = i + 2
            while j < n and lines[j].strip().startswith("|"):
                rows.append(_split_row(lines[j]))
                j += 1
            table = doc.add_table(rows=1, cols=len(header))
            table.style = "Light Grid Accent 1"
            for k, text in enumerate(header):
                cell = table.rows[0].cells[k]
                cell.text = ""
                p = cell.paragraphs[0]
                _add_inline(p, text, 9)
                for r in p.runs:
                    r.bold = True
            for row in rows:
                cells = table.add_row().cells
                for k in range(len(header)):
                    text = row[k] if k < len(row) else ""
                    cells[k].text = ""
                    _add_inline(cells[k].paragraphs[0], text, 9)
            doc.add_paragraph()
            i = j
            continue
        # image
        m = re.match(r"^!\[[^\]]*\]\(([^)]+)\)\s*$", line)
        if m:
            img = (md_path.parent / m.group(1)).resolve()
            if img.exists():
                doc.add_picture(str(img), width=Inches(6.2))
                doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
            i += 1
            continue
        # bullet
        m = re.match(r"^\s*[-*]\s+(.*)$", line)
        if m:
            p = doc.add_paragraph(style="List Bullet")
            _add_inline(p, m.group(1))
            p.paragraph_format.space_after = Pt(2)
            i += 1
            continue
        # numbered
        m = re.match(r"^\s*\d+\.\s+(.*)$", line)
        if m:
            p = doc.add_paragraph(style="List Number")
            _add_inline(p, m.group(1))
            p.paragraph_format.space_after = Pt(2)
            i += 1
            continue
        # blockquote
        if line.strip().startswith(">"):
            text = line.strip().lstrip(">").strip()
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Pt(12)
            _add_inline(p, text, 9.5)
            for r in p.runs:
                r.font.color.rgb = RGBColor(0x33, 0x44, 0x55)
            i += 1
            continue
        # paragraph (join consecutive non-empty lines)
        buf = [line]
        j = i + 1
        while j < n and lines[j].strip() and not re.match(
                r"^(#{1,4}\s|[-*]\s|\d+\.\s|>|\|)", lines[j].strip()):
            buf.append(lines[j].strip())
            j += 1
        p = doc.add_paragraph()
        _add_inline(p, " ".join(buf))
        p.paragraph_format.space_after = Pt(4)
        i = j

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(out_path))
    return out_path
