# 02-constrained-qpeplayer — LaTeX 稿（zh）

- **目标格式：** 中文双栏会议格式（ICRA 式）
- **文档类：** `ctexart, twocolumn`
- **编译命令：**

  ```bash
  export HTTPS_PROXY=http://127.0.0.1:7899 HTTP_PROXY=http://127.0.0.1:7899
  /mnt/storage1/syk/tools/tectonic/tectonic -X compile main.tex
  ```

- **引擎：** Tectonic 0.17.0（`/mnt/storage1/syk/tools/tectonic/tectonic`）。
- **事实来源：** `../../draft.md`（Markdown，请勿修改）。本目录中所有数字均可追溯到该文件及其 `experiments/` 结果文件。

## 构建状态

| 项目 | 值 |
|---|---|
| 最近编译 | 退出码 0，无未定义引用，无 overfull 盒 |
| 正文页数（第 1–7 节） | 7 |
| 总页数 | 17 |
| 表 / 图 | 18 / 4 |

## 文件

`main.tex`（文档类、标题、摘要、关键词、`\input` 行）、`preamble.tex`、
`sections/*.tex`、`figures/*.png`、`refs.bib`、`build.log`。

## 结果范围

1. 逐格明细表、定位比较表与实现验证位于 `\appendix` 之后的附录；中文稿不受六页限制约束，正文 7 页。
2. 真实数据为 GT 投影对应（30 个随机模型顶点按 GT 位姿投影，加 1 px 噪声与 10/20/30 px 偏置），`n=90/72/84`，研究为层级别。
3. 单边 Shor 标志（`J - Shor <= 1e-6`）只是上界；记录在案的证书是 P2-Cert 的 Schur 修复：基础割 24.4/15.3/13.1%，可靠 bound-RLT 族 51.1/41.7/23.8%。
4. 交付的枚举模板是一般/PnP 情形的 40 根未饱和模板；4/12/16/32/40 谱在标准形下数值验证。

## 编辑规则

- 表格：`booktabs`，宽表用 `table*` 与 `\scriptsize`，不用竖线。
- 图：`figure` 中用 `\includegraphics[width=\columnwidth]{...}`；宽总览图用 `figure*` 与 `width=\textwidth`。
- 作者、单位与基金为占位符，不得杜撰。
