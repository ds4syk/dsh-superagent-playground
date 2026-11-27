# 01-laplace-qpeplayer — LaTeX draft (zh)

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
| last compile | 2026-11-28, Tectonic 0.17.0, exit 0 |
| main-text pages | 22 |
| total pages | 24 |
| tables / figures | 19 / 5 |
| undefined refs / cites | 0 |
| bibliography | `refs.bib` via BibTeX, `plain` style, 15 entries |

## Translation status

All eight section files plus the abstract were hand-translated from
`../en/sections/*.tex`. Equations, table rows, figure files and every numeric
value are carried over verbatim (audited: 552/552 measurement-like numeric
tokens and 98/98 multi-digit integers appear unchanged; 35/35 `\cite` keys,
100/100 labels, 19 tables, 5 figures, 25 numbered equations match the English
tree). Proper nouns and acronyms (QPEP, SDP, Veronese, Choi–Lam, LINEMOD,
Theseus, EPnP, …) are kept in Latin script. Author names and affiliations
remain placeholders.

## Files

`main.tex` (documentclass, title, abstract, keywords, \input lines), `preamble.tex`,
`sections/*.tex`, `figures/*.png`, `refs.bib`, `build.log`.

## Known pending claims / gaps (must stay visible in the paper)

1. 没有检测器级 ADD(-S)/ADD-S AUC，也没有学习前端：真实数据部分是受控双模态层实验，不是检测器基准。
2. UTIAS In-The-Dark 未运行，不对该流程作任何声明。
3. 未运行学习型/可微分层次 baseline（SDPR-IS、SDPR-SDP、EPro-PnP、BPnP）；Theseus 与 cv2 经典解法只是同一批对应上的前向/局部参照。
4. mode-level MC 只对离散模态权重采样，不是 EPro-PnP 的 SE(3) AMIS 采样器。
5. 已报告运行假定根枚举完备（A2）；total-degree homotopy 在 4/4 实例上确认 40 个复射影根，但不证明多起点枚举器的漏根率。
6. SDP 证书是数值证书（SCS 容差），不是符号证书；Choi–Lam 实例的物理可实现性（Q1）仍开放；逐实例认证实根数是未来工作。
7. GPU 批处理 / 端到端训练 / 峰值内存 / 吞吐未测量。
8. 必须保留的负结果：在 15 帧 LINEMOD 上，Laplace 后验均值被拉向翻转模态（中位 44.24° / 13.9 mm ADD-S），而 hard argmin 为 2.06° / 3.7 mm。

## Editorial rules

- Tables: `booktabs`, wide tables as `table*` with `\scriptsize`; no vertical rules.
- Figures: `\\includegraphics[width=\\columnwidth]{...}` inside `figure`; wide overview
  figures in `figure*` with `width=\\textwidth`.
- Evidence tags stay honest: *paper claim* / *independent reproduction (this repo)* /
  *unverified*; Chinese: 论文声明 / 独立复现（本仓库） / 未验证 / 待核实.
- Author names, affiliations and funding are placeholders; do not invent them.
