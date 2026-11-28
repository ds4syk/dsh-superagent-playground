# 01-laplace-qpeplayer — LaTeX manuscript (zh，人工翻译)

- **Target venue:** 《自动化学报》式中文双栏期刊（自动化学报 style）
- **Document class:** `ctexart, twocolumn`
- **Compile command:**

  ```bash
  export HTTPS_PROXY=http://127.0.0.1:7899 HTTP_PROXY=http://127.0.0.1:7899
  /mnt/storage1/syk/tools/tectonic/tectonic -X compile main.tex
  ```

- **Engine:** Tectonic 0.17.0 (`/mnt/storage1/syk/tools/tectonic/tectonic`).
- **Source of truth:** `../../draft.md` (Markdown; do not modify). All numbers in this
  tree must be traceable to that file and its `experiments/` result files.

## Build status

| item | value |
|---|---|
| last compile | Tectonic exit 0, 无未定义引用, 无 >3pt 的 overfull（见 `build.log`） |
| total pages | 21（双栏；含附录 A--B） |
| tables / figures | 17 表 / 5 图 |
| 说明 | 由英文终稿逐节人工翻译（非机器翻译），结构、公式、表格、数字与引用一一对应（2026-11-28） |

## Files

`main.tex` (documentclass, title, abstract, keywords, \input lines), `preamble.tex`,
`sections/*.tex`, `figures/*.png`, `refs.bib` (EN only), `build.log`.

## 论文中陈述的局限性（第 6 节）

1. 评估为层级评估：81 点合成交叉、一维标准型、15 帧 LINEMOD；无检测器级 ADD(-S)/ADD-S AUC，无学习式前端，无端到端训练。
2. 实根枚举完备性是假设 (A2)；全次数同伦认证 4/4 个实例上的 40 个复射影根，而非枚举器的漏检率。
3. 真实近打平处 Laplace 后验均值被拉向竞争模态（44.24 deg / 13.9 mm ADD-S），硬 argmin 为 2.06 deg / 3.7 mm。
4. 构造的 Choi–Lam 代价作为对应 Gram 矩阵的可实现性仍开放。
5. 学习式前端、检测器级基准、UTIAS In-The-Dark、SDPR/EPro-PnP 对比、逐实例认证实根数与 GPU 批量前向列为未来工作。
