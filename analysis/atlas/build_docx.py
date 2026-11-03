"""Build the SDPR reproduction figure atlas as a DOCX."""
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
OUT = HERE / "SDPR_复现图表集.docx"

GREEN = RGBColor(0x2E, 0x6B, 0x3A)
BLUE = RGBColor(0x1F, 0x4E, 0x79)
GRAY = RGBColor(0x59, 0x59, 0x59)
FONT_CN = "Noto Serif CJK SC"
FONT_CN_SANS = "Noto Sans CJK SC"


def set_run_font(run, name=FONT_CN, size=10.5, bold=None, color=None):
    run.font.name = name
    run._element.rPr.rFonts.set(qn("w:eastAsia"), name)
    run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if color is not None:
        run.font.color.rgb = color


def para(doc, text="", size=10.5, bold=False, color=None, align=None, space_after=6, name=FONT_CN):
    p = doc.add_paragraph()
    if align is not None:
        p.alignment = align
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = 1.25
    r = p.add_run(text)
    set_run_font(r, name=name, size=size, bold=bold, color=color)
    return p


def heading(doc, text, level=1, new_page=False):
    if new_page:
        doc.add_page_break()
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(14 if level == 1 else 10)
    p.paragraph_format.space_after = Pt(6)
    r = p.add_run(text)
    set_run_font(r, name=FONT_CN_SANS, size=15 if level == 1 else 12, bold=True,
                 color=GREEN if level == 1 else BLUE)
    # outline level for navigation
    pPr = p._p.get_or_add_pPr()
    ol = OxmlElement("w:outlineLvl")
    ol.set(qn("w:val"), str(level - 1))
    pPr.append(ol)
    return p


def bullet(doc, text, size=10.5):
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.line_spacing = 1.2
    r = p.add_run(text)
    set_run_font(r, size=size)
    return p


def figure(doc, name, width_cm=16.0, caption=""):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(4)
    p.add_run().add_picture(str(FIG / name), width=Cm(width_cm))
    if caption:
        c = doc.add_paragraph()
        c.alignment = WD_ALIGN_PARAGRAPH.CENTER
        c.paragraph_format.space_after = Pt(8)
        r = c.add_run(caption)
        set_run_font(r, size=9, color=GRAY)


def table(doc, header, rows, widths=None):
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr = t.rows[0].cells
    for i, h in enumerate(header):
        hdr[i].text = ""
        p = hdr[i].paragraphs[0]
        r = p.add_run(h)
        set_run_font(r, name=FONT_CN_SANS, size=9.5, bold=True, color=RGBColor(0xFF, 0xFF, 0xFF))
        shd = OxmlElement("w:shd")
        shd.set(qn("w:fill"), "2E6B3A")
        hdr[i]._tc.get_or_add_tcPr().append(shd)
    for row in rows:
        cells = t.add_row().cells
        for i, val in enumerate(row):
            cells[i].text = ""
            p = cells[i].paragraphs[0]
            r = p.add_run(str(val))
            set_run_font(r, size=9)
    if widths:
        for i, w in enumerate(widths):
            for row in t.rows:
                row.cells[i].width = Cm(w)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return t


def add_page_number_footer(section):
    p = section.footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    fld1 = OxmlElement("w:fldChar"); fld1.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText"); instr.set(qn("xml:space"), "preserve"); instr.text = " PAGE "
    fld2 = OxmlElement("w:fldChar"); fld2.set(qn("w:fldCharType"), "end")
    run._r.append(fld1); run._r.append(instr); run._r.append(fld2)
    set_run_font(run, size=8, color=GRAY)


def main():
    doc = Document()
    sec = doc.sections[0]
    sec.page_height = Cm(29.7); sec.page_width = Cm(21.0)
    sec.top_margin = Cm(2.2); sec.bottom_margin = Cm(2.0)
    sec.left_margin = Cm(2.4); sec.right_margin = Cm(2.4)
    add_page_number_footer(sec)

    # normal style
    st = doc.styles["Normal"]
    st.font.name = FONT_CN
    st.font.size = Pt(10.5)
    st.element.rPr.rFonts.set(qn("w:eastAsia"), FONT_CN)

    # cover-ish header
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(18); p.paragraph_format.space_after = Pt(2)
    r = p.add_run("SDPR 复现图表集")
    set_run_font(r, name=FONT_CN_SANS, size=26, bold=True, color=GREEN)

    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(2)
    r = p.add_run("Clean-room 独立实现 + 官方代码修复版 · 2026-11")
    set_run_font(r, size=12, color=GRAY)

    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(16)
    r = p.add_run("只放我们复现的数据；曲线做了中位数 + 50% 分位带清洗，便于观看。")
    set_run_font(r, size=10, color=GRAY)

    para(doc, "这份图集把 SDPRLayers 相机 baseline 标定复现中的图重新画了一遍："
              "同一套实验数据，去掉论文报告值，只保留我们自己的 50 次实验；"
              "轨迹类图改成“中位数 + 50% 分位带”，避免 50 条线糊在一起。"
              "每张图下面都写了“怎么读”和“结论”，可以直接放进汇报/论文附录。",
         size=10.5)

    # data scope
    heading(doc, "0. 数据与口径")
    table(doc,
          ["数据", "规模", "说明"],
          [["clean-room 主实验", "50 runs × 20 poses", "独立实现；SDPR-IS / 局部解(真值) / 局部解(随机)"],
           ["官方代码修复版", "50 runs × 20 poses", "SCS 3.3.1；自适应 KKT 容差重试后 49/50 成功"],
           ["局部极小实验", "3 组 × 100 次随机初始化", "统计落入局部极小的比例"],
           ["Jacobian 校验", "50 trials", "解析导数 vs 中心差分"],
           ["紧性/证书", "约 10.8 万次 SDP 求解", "rank-1、corank(H)、H 半正定、强对偶"]],
          widths=[4.0, 4.2, 7.0])
    para(doc, "清洗口径：轨迹图按 run 插值到 0–150 次外层迭代，取中位数与 25/75 分位数；"
              "误差取绝对值后画对数轴；损失画对数轴。图中不画论文值。",
         size=9.5, color=GRAY)

    # figure list
    heading(doc, "图表清单")
    table(doc,
          ["图", "内容", "一句话"],
          [["图 1", "实验设置", "8×8 标定板、双目 baseline、3 m/90° 采样"],
           ["图 2", "baseline 误差与 outer loss", "SDPR/真值初始化收敛；随机初始化发散"],
           ["图 3", "最终误差、迭代、耗时", "SDPR 与真值初始化同量级；随机初始化差 3 个数量级"],
           ["图 4", "随机初始化落局部极小比例", "三组 40%/67%/70%，均值 59%"],
           ["图 5", "紧性与证书", "rank-1 100%、corank=1 99.97%、H 半正定"],
           ["图 6", "Jacobian 校验", "解析导数与有限差分 1e-6 量级一致"],
           ["图 7", "官方代码修复版", "SDPR≈Theseus-GT；RND 失败；49/50 成功"]],
          widths=[1.6, 5.2, 8.4])

    # 1 setup
    heading(doc, "1. 实验设置：在标定板前估计位姿，再标定 baseline", new_page=True)
    figure(doc, "01_setup.png", 14.5, "图 1：双目相机在 3 m 半径、90° 锥内随机拍摄 8×8 标定板；外层只调 baseline b。")
    para(doc, "怎么读：内层是“给定 b，估计相机位姿”；外层是“调 b，让位姿更接近真值”。"
              "b 的初值故意偏 0.003 m，看能不能收敛回 0.24 m。", size=10)
    para(doc, "结论：这是一个很小的标定问题，但足以暴露“局部解梯度”与“全局解梯度”的差异。",
         size=10, bold=True, color=GREEN)

    # 2 trajectories
    heading(doc, "2. 轨迹：SDPR 与真值初始化几乎重合，随机初始化跑飞", new_page=True)
    figure(doc, "02_trajectories_vertical.png", 15.0,
           "图 2：50 runs 的 baseline 误差（上）与 outer loss（下）；实线为中位数，阴影为 25–75% 分位带。")
    bullet(doc, "SDPR-IS（绿实线）与“真值初始化局部解”（蓝虚线）几乎完全重合：最终误差都在 4×10⁻⁴ m 量级。")
    bullet(doc, "“随机初始化局部解”（红实线）的误差一路上升到 0.3 m 量级，outer loss 高 4 个数量级。")
    bullet(doc, "中位数/分位带比 50 条原始线更干净；原始逐 run 曲线仍保留在复现数据里。")
    para(doc, "结论：SDPR 的梯度与“全局解附近的局部解”一致；随机初始化局部解的梯度会把参数带向错误方向。",
         size=10, bold=True, color=GREEN)

    # 3 final
    heading(doc, "3. 最终误差、迭代次数与资源开销", new_page=True)
    figure(doc, "03_final.png", 16.0,
           "图 3：最终 baseline 误差箱线图（对数）、外层迭代次数、单 run 墙钟时间（12 workers）。")
    bullet(doc, "最终误差：SDPR 与真值初始化局部解的中位数都在 3.9×10⁻⁴ m；随机初始化局部解在 3.3×10⁻¹ m。")
    bullet(doc, "迭代次数：SDPR 54.5、真值初始化 54.6、随机初始化 143.3（后者经常跑到 150 上限）。")
    bullet(doc, "资源：SDPR 单 run 约 37 s（12 workers 争用 CPU 时的上界）；真值初始化约 7.8 s。")
    para(doc, "结论：SDPR 的代价是前向解 SDP；换来的是一致、可信的梯度。", size=10, bold=True, color=GREEN)

    # 4 local minima
    heading(doc, "4. 随机初始化：约 59% 掉进局部极小")
    figure(doc, "04_local_minima.png", 15.5,
           "图 4：三组各 100 次随机初始化，收敛到全局最优/局部极小的比例。")
    para(doc, "怎么读：绿色是收敛到全局最优，红色是收敛到局部极小；三组分别有 40%、67%、70% 落入局部极小，"
              "均值 59%，与论文“约 60%”一致。", size=10)
    para(doc, "结论：局部求解器“从随机初始化出发”并不可靠；这正是 SDPR 要解决的问题。",
         size=10, bold=True, color=GREEN)

    # 5 tightness
    heading(doc, "5. 紧性与证书：论文的假设在我们数据上成立", new_page=True)
    figure(doc, "05_tightness.png", 16.0,
           "图 5：50 runs 的秩 1 判据 λ1/λ2（左）与证书矩阵 H 的最小特征值（右）。")
    bullet(doc, "全部 50 runs 的 λ1/λ2 都远高于 1e5（实际在 10¹⁴ 量级），即 SDP 解是秩 1。")
    bullet(doc, "H 的最小特征值在 0 附近（数值误差内），即 H 半正定；corank=1 占 99.97%。")
    bullet(doc, "约 10.8 万次 SDP 求解中，rank-1 比例 100%，原问题与对偶目标差约 10⁻¹⁰。")
    para(doc, "结论：§VI-B4 实验“松弛紧、证书成立”的声明，在独立实现下成立。",
         size=10, bold=True, color=GREEN)

    # 6 jacobian
    heading(doc, "6. Jacobian 校验：解析导数与有限差分一致")
    figure(doc, "06_jacobian.png", 15.0,
           "图 6：50 trials 上，SDPR-IS、局部解、以及两者之间的 Jacobian 相对差异。")
    para(doc, "怎么读：纵轴是对数相对误差；三组的中位数都在 10⁻⁶ 量级，说明解析反向与中心差分一致。",
         size=10)
    para(doc, "结论：SDPR 的反向实现不是“看起来能跑”，而是数值上对得上。",
         size=10, bold=True, color=GREEN)

    # 7 faithful
    heading(doc, "7. 官方代码修复版：SDPR ≈ Theseus-GT，RND 失败", new_page=True)
    figure(doc, "07_faithful.png", 15.5,
           "图 7：官方代码修复版（SCS 3.3.1）的最终误差与迭代次数；SDPR-IS 自适应重试后 49/50 成功。")
    bullet(doc, "最终误差：SDPR-IS 4.62×10⁻⁴ m、Theseus-GT 4.62×10⁻⁴ m、Theseus-RND 1.22×10⁻¹ m。")
    bullet(doc, "迭代次数：34.5 / 34.7 / 82.9，与论文报告的 34.22 / 35.40 / 80.34 基本一致。")
    bullet(doc, "工程发现：strict 配置下 SDPR-IS 有 28% 失败；自适应 KKT 容差重试降到 2%。")
    para(doc, "结论：官方代码路径与 clean-room 路径给出一致结论；失败率问题已定位并修复。",
         size=10, bold=True, color=GREEN)

    # summary
    heading(doc, "8. 一句话总结")
    bullet(doc, "SDPR 与真值初始化局部解几乎重合，随机初始化局部解差 3 个数量级。")
    bullet(doc, "10.8 万次 SDP 求解 100% rank-1、H 半正定、corank=1 占 99.97%。")
    bullet(doc, "局部极小比例 59%，与论文约 60% 一致。")
    bullet(doc, "官方代码 28% 失败已用自适应重试降到 2%。")
    para(doc, "证据边界：全量 50-run 数字来自已保存的复现产物；本次整理未重跑全量。"
              "SDPR 复现没有 episode/reward 指标（那是 RL 训练），对应的曲线是 baseline error、outer loss、"
              "迭代次数、紧性与 Jacobian。", size=9.5, color=GRAY)

    # appendix
    heading(doc, "附录：数据与脚本")
    table(doc,
          ["文件", "位置"],
          [["清洗后的图", "analysis/atlas/figures/*.png"],
           ["画图脚本", "analysis/atlas/make_atlas.py"],
           ["原始轨迹", "repro/results_clean/trajectories_independent.csv"],
           ["run 汇总", "repro/results_clean/run_summary_independent.csv"],
           ["官方代码结果", "repro/results_upstream/final_adaptive/tableV_faithful.csv"]],
          widths=[5.0, 10.5])

    doc.save(OUT)
    print("saved", OUT, OUT.stat().st_size)


if __name__ == "__main__":
    main()
