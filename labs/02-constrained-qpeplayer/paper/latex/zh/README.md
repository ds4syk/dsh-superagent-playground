# 02-constrained-qpeplayer — LaTeX draft (zh)

- **Target venue:** 中文双栏会议格式（ICRA 式）
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
| last compile | see `build.log` |
| main-text pages | TBD |
| total pages | TBD |
| tables / figures | TBD |

## Files

`main.tex` (documentclass, title, abstract, keywords, \input lines), `preamble.tex`,
`sections/*.tex`, `figures/*.png`, `refs.bib` (EN only), `build.log`.

## Known pending claims / gaps (must stay visible in the paper)

1. 正文须控制在 6 页以内；逐格明细表放入附录，并明确标注不计入页数限制。
2. 真实数据是 GT 投影对应（30 个随机模型顶点按 GT 位姿投影，加 1 px 噪声与 10/20/30 px 偏置），不是检测器/匹配器输出；n=90/72/84。
3. 未运行学习型层 baseline（约束版 SDPRLayers、dQP、CVXPYLayers）；SCP 是局部线性化+SOCP 组合，不是 dQP。
4. 单边 Shor 标志只是上界；记录的严格界是 P2-Cert 的 Schur 修复（base 24.4/15.3/13.1%，sound bound-RLT 51.1/41.7/23.8%）。
5. 端到端训练（T1-network、T2 KITTI/TUM/UTIAS、T3 手眼）只陈述未运行；YCB-Video 仅层实验。
6. FB 在弱活跃点的 ε→0 选择（G2）仍开放：只能说“一个选定的广义导数”，不能说“该导数”。
7. 各向同性单法向族的 16 根饱和模板未交付；4/12/16 谱只在标准形下数值验证。
8. Gauss–Newton 与精确 Hessian 的消融、反向传播计时、装配/内层/证书延迟拆分均未测量。
9. 真实数据上的零维 RLT 是负结果；正结果是 sound bound-RLT 族。

## Editorial rules

- Tables: `booktabs`, wide tables as `table*` with `\scriptsize`; no vertical rules.
- Figures: `\\includegraphics[width=\\columnwidth]{...}` inside `figure`; wide overview
  figures in `figure*` with `width=\\textwidth`.
- Evidence tags stay honest: *paper claim* / *independent reproduction (this repo)* /
  *unverified*; Chinese: 论文声明 / 独立复现（本仓库） / 未验证 / 待核实.
- Author names, affiliations and funding are placeholders; do not invent them.
