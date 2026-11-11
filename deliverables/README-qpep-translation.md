# QPEP 论文中文译本（TRO 2022）

- 源论文：J. Wu, Y. Zheng, Z. Gao, Y. Jiang, X. Hu, Y. Zhu, J. Jiao, M. Liu,
  “Quadratic Pose Estimation Problems: Globally Optimal Solutions, Solvability/Observability
  Analysis, and Uncertainty Description,” *IEEE Transactions on Robotics*, 2022.
  DOI: 10.1109/TRO.2022.3155880
- 成品：`QPEP_论文中文译本_TRO2022_2026-11.pdf`（22 页，9.3 MB）
- 版面：与原文逐页一致（A4/Letter 原尺寸 594×792 pt、IEEE 双栏、页码与页眉页脚位置不变）；
  图 1–24、表 I–VI、式 (1)–(54) 均原位保留。

## 做法

1. 从 Feishu 消息附件取源 PDF，用 PyMuPDF 做字符级版面分析：按基线聚类视觉行、按列切分、
   合并跨行段落，识别正文/标题/图注/表格/公式区域。
2. 正文逐段中译；占位符 `⟦i⟧` 对应原文行内公式，按 600 dpi 从原页裁剪为图片，
   按原基线嵌入中文段落，保证符号、上下标、矩阵不变形。
3. 用自研 CJK 排版器（Noto Serif CJK）在每段原包围盒内重排中文：按行宽断行、
   两端对齐（可控拉伸）、按可用高度自适应字号与行距，避免与相邻段落/公式重叠。
4. 公式块、图、表不重排：仅对译段做白底遮盖后写入中文，其余内容保持矢量原样。
5. 删除原文中的批注高亮（10 处），避免中文段落被误标黄。

## 覆盖范围

- 已译：标题、作者、摘要、关键词、正文全部章节、图注、表题、脚注、结论、致谢、页眉页脚。
- 保留英文：参考文献条目；表格单元格（原表为矢量路径，非文本对象）；图内标注。
- 例外的 7 段：第 3、7、9、10、11 页中公式极密集的段落保留英文原文（见下文限制）。

## 已知限制

1. 7 个公式极密集段落保留英文原文：`p3_4`、`p3_5`、`p7_6`、`p9_27`、`p10_21`、`p10_25`、`p11_26`。
   这些段落含跨行大公式（行内嵌入会导致与相邻行重叠），重排风险高于收益。
2. 译文字数一般短于英文，个别列尾留白；行距与字号为自适应结果，与英文原行距不完全一致。
3. 表格单元格与图内文字保持英文；公式编号、文献编号保持原样。

## 复现

```bash
python3 work/units2.py          # 版面分析 → analysis/units.json
python3 work/prepare2.py        # 生成行内公式裁剪与待译清单
python3 work/sheet.py 1,2,3     # 生成翻译工作页 work/sheet/pNN.md
# 译文：analysis/zh_pages/zh_pNN.json（人工/模型翻译）
python3 work/normalize.py       # 页眉页脚、上下标、占位符归一化
python3 work/build.py all       # 生成 source/translated.pdf
gs -sDEVICE=pdfwrite -dPDFSETTINGS=/prepress -dSubsetFonts=true \
   -dCompressFonts=true -dAutoRotatePages=/None \
   -sOutputFile=source/final.pdf source/translated.pdf
```
