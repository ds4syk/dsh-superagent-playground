# 01-laplace-qpeplayer — LaTeX manuscript (en)

- **Target venue:** IEEE Transactions on Robotics (T-RO), journal article
- **Document class:** `IEEEtran [journal]`
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
| last compile | Tectonic exit 0, no undefined refs, no overfull boxes > 3 pt (see `build.log`) |
| total pages | 16 (journal, two-column) |
| tables / figures | 17 tables / 5 figures |
| style | IEEE T-RO submission style; internal draft/status/evidence-level scaffolding removed (2026-11-28 rewrite) |

## Files

`main.tex` (documentclass, title, abstract, keywords, \input lines), `preamble.tex`,
`sections/*.tex`, `figures/*.png`, `refs.bib` (EN only), `build.log`.

## What the paper states as limitations (Section VI)

1. The evaluation is layer-level: 81-point synthetic crossing, 1D normal forms, and 15 LINEMOD frames; no detector-level ADD(-S)/ADD-S AUC, no learned front-end, no end-to-end training.
2. Root-enumeration completeness is assumption (A2); the total-degree homotopy certifies 40 complex projective roots on 4/4 instances, not the enumerator's missed-root rate.
3. The Laplace posterior mean is pulled toward the competing mode at real near-ties (44.24 deg / 13.9 mm ADD-S) versus the hard argmin (2.06 deg / 3.7 mm).
4. The constructed Choi–Lam cost's realizability as a correspondence Gram matrix is open.
5. Extending to learned front-ends, detector-level benchmarks, UTIAS In-The-Dark, SDPR/EPro-PnP comparisons, a certified per-instance real-root count and a batched GPU forward is stated as future work.

## Editorial rules

- Tables: `booktabs`, wide tables as `table*` with `\scriptsize`; no vertical rules.
- Figures: `\\includegraphics[width=\\columnwidth]{...}` inside `figure`; wide overview
  figures in `figure*` with `width=\\textwidth`.
- Evidence tags stay honest: *paper claim* / *independent reproduction (this repo)* /
  *unverified*; Chinese: 论文声明 / 独立复现（本仓库） / 未验证 / 待核实.
- Author names, affiliations and funding are placeholders; do not invent them.
