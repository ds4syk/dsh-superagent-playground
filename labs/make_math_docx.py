"""Build the Chinese reading copy (DOCX) of the consolidated math questions.

Run: /home/syk/miniconda3/bin/python labs/make_math_docx.py
Output: deliverables/三篇论文数学问题汇总_2026-11.docx
"""
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor

HERE = Path(__file__).resolve().parent.parent
OUT = HERE / "deliverables" / "三篇论文数学问题汇总_2026-11.docx"
OUT.parent.mkdir(parents=True, exist_ok=True)

doc = Document()

# Base font: Chinese + Latin
style = doc.styles["Normal"]
style.font.name = "Noto Sans CJK SC"
style.font.size = Pt(10.5)
style._element.rPr.rFonts.set(qn("w:eastAsia"), "Noto Sans CJK SC")

for section in doc.sections:
    section.top_margin = Pt(54)
    section.bottom_margin = Pt(54)
    section.left_margin = Pt(60)
    section.right_margin = Pt(60)


def h(text, level=1):
    p = doc.add_heading(text, level=level)
    for r in p.runs:
        r.font.name = "Noto Sans CJK SC"
        r._element.rPr.rFonts.set(qn("w:eastAsia"), "Noto Sans CJK SC")
        r.font.color.rgb = RGBColor(0x1A, 0x1A, 0x1A)
    return p


def para(label, text, bold_label=True):
    p = doc.add_paragraph()
    if label:
        r = p.add_run(label + "：")
        r.bold = bold_label
        r.font.name = "Noto Sans CJK SC"
        r._element.rPr.rFonts.set(qn("w:eastAsia"), "Noto Sans CJK SC")
    r = p.add_run(text)
    r.font.name = "Noto Sans CJK SC"
    r._element.rPr.rFonts.set(qn("w:eastAsia"), "Noto Sans CJK SC")
    p.paragraph_format.space_after = Pt(3)
    return p


def bullet(text):
    p = doc.add_paragraph(style="List Bullet")
    r = p.add_run(text)
    r.font.name = "Noto Sans CJK SC"
    r._element.rPr.rFonts.set(qn("w:eastAsia"), "Noto Sans CJK SC")
    p.paragraph_format.space_after = Pt(2)
    return p


title = doc.add_heading("三篇论文数学问题汇总（2026-11-27）", level=0)
for r in title.runs:
    r.font.name = "Noto Sans CJK SC"
    r._element.rPr.rFonts.set(qn("w:eastAsia"), "Noto Sans CJK SC")

p = doc.add_paragraph()
r = p.add_run("三条 QPEP 论文线工作区 · GitHub PR #39 阅读副本 · 完整证明见仓库 labs/MATH_QUESTIONS.md")
r.italic = True
r.font.size = Pt(9)
r.font.color.rgb = RGBColor(0x66, 0x66, 0x66)
r.font.name = "Noto Sans CJK SC"
r._element.rPr.rFonts.set(qn("w:eastAsia"), "Noto Sans CJK SC")

h("一句话结论", 1)
bullet("P1 Laplace-QPEPLayer：零升维 + 3×3 黎曼隐微分 + 多峰 Laplace。定理 0 的 21 约束 = Veronese 二次理想已用符号+秩证明；Choi–Lam 构造实例 gap 0.0342（限定为“构造的旋转子问题”，不是 SDPRLayers 实际 13 维 SDP）。")
bullet("P2 Constrained-QPEPLayer：常数维 ALM + 14×14 Shor 证书 + FB 约化隐微分。实例上 SDP tight 到 2×10^-11；FB 的 ε→0 极限 = H-加权最小范数解，单约束 kink 精确给出 Clarke 中点 1/2。")
bullet("P3 Robust-QPEPLayer：BR 对偶 + GNC + 6×6 曲率修正。发现并修正源文档 H_rob 缺少 S³ 约束乘子项 −λ_rob（FD 误差 29.7% → 1.5×10^-7）；一维变号定理严格成立。")

h("最需要你判断的 5 条", 1)
items = [
    ("CC-1 [P0] 批量全局 QPEP 前向", "训练规模下 CPU Gröbner 单样本不可行。是否接受“推理全局、训练局部/多起点”的折中？是否有现成 GPU 实现？"),
    ("P1-Q1 [P0] Choi–Lam 可实现性", "构造的 10×10 代价矩阵是否落在真实 PnP/配准的 Gram 锥内？这决定“真实问题必然非紧”能否成立。"),
    ("P1-Q5 [P0] Laplace 模式重叠", "ΔJ→0 时用一致渐近/Airy、数值积分，还是只声明“远离分岔”？"),
    ("P2-Q2 [P0] Clarke 广义导数", "弱活跃点 FB 选出的广义导数如何严格刻画（单约束已证为 Clarke 中点，多约束 open）。"),
    ("P2-Q3 [P0] GN vs 精确 Hessian", "主定理用可证的 GN 版本 + 精确版 Remark，是否接受？"),
]
for label, text in items:
    para(label, text)

h("其余 P0（8 条）", 1)
for text in [
    "P1-Q2：13×13 SDP 与 10×10 旋转子问题是否同值（决定能否声称等价于 SDPRLayers 实际实现）。",
    "P1-Q3：QPEP 根数 27 还是 40；枚举完整性证书。",
    "P1-Q9：Hilbert ⇒ 正测度非紧还缺哪几步（可实现性、开性、值间隙 ⇒ rank>1）。",
    "P2-Q1：内层 KKT 根数是否仍 40；QPEP 模板能否复用（Bézout 162）。",
    "P3-Q1：6D 位姿层变号定理或明确反例（1D 已严格）。",
    "P3-Q2：GNC μ-同伦分岔的显式反例（当前只能陈述机制）。",
    "P3-Q3：O(‖r‖^-4) 的一致 H_rob^-1 界。",
    "P3-Q4：相干外点下的 3D 崩溃点/临界比例。",
]:
    bullet(text)

h("三条线状态", 1)
table = doc.add_table(rows=1, cols=5)
table.style = "Light Grid Accent 1"
hdr = table.rows[0].cells
for i, text in enumerate(["线", "英文 draft", "定理/证明", "verify", "P0"]):
    hdr[i].text = text
for row in [
    ["P1", "有（实验 TODO）", "定理 0.1/0.2/0.3、1.1、1.2、1.3", "2 个脚本", "5"],
    ["P2", "有（实验 TODO）", "定理 2.1/2.2/2.3 + 新 Lemma 2.2", "2 个脚本 / 31 项", "3"],
    ["P3", "有（实验 TODO）", "3 Lemma + 3 Theorem + 4 Prop", "2 个脚本", "4"],
]:
    cells = table.add_row().cells
    for i, text in enumerate(row):
        cells[i].text = text
for row in table.rows:
    for cell in row.cells:
        for p in cell.paragraphs:
            for r in p.runs:
                r.font.size = Pt(9)
                r.font.name = "Noto Sans CJK SC"
                r._element.rPr.rFonts.set(qn("w:eastAsia"), "Noto Sans CJK SC")

h("证据与边界", 1)
bullet("证据分级：已证明 / 数值验证 / 作者预实验（待独立复现）/ 待验证。源文档的预实验数字均未当作本仓库结果。")
bullet("未做：真实数据实验、端到端训练、QPEP C++/GPU 工程。论文是 v0.1 研究初稿，不是投稿版。")
bullet("仓库是 public；合并 PR 会把三篇未发表论文的想法与定理公开。若需保密，请先转私有仓库或暂不合并。")
bullet("完整清单与证明：labs/MATH_QUESTIONS.md 与各目录 math_questions.md / theory/proofs.md。")

doc.save(OUT)
print("wrote", OUT, OUT.stat().st_size)
