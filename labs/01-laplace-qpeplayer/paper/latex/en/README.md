# 01-laplace-qpeplayer — LaTeX draft (en)

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
| last compile | see `build.log` |
| main-text pages | TBD |
| total pages | TBD |
| tables / figures | TBD |

## Files

`main.tex` (documentclass, title, abstract, keywords, \input lines), `preamble.tex`,
`sections/*.tex`, `figures/*.png`, `refs.bib` (EN only), `build.log`.

## Known pending claims / gaps (must stay visible in the paper)

1. No detector-level ADD(-S)/ADD-S AUC and no learned front-end: the real-data study is a controlled two-mode layer experiment, not a detector benchmark.
2. UTIAS In-The-Dark was not run; no claim about that pipeline.
3. Learned/differentiable-layer baselines (SDPR-IS, SDPR-SDP, EPro-PnP, BPnP) were not run; Theseus and the cv2 classical solvers are forward/local references on the same correspondences only.
4. The mode-level MC baseline samples discrete mode weights; it is not the EPro-PnP SE(3) AMIS sampler.
5. Root enumeration completeness (A2) is assumed for the reported runs; the total-degree homotopy certifies 40 complex projective roots on 4/4 instances but not the multi-start enumerator's missed-root rate.
6. The SDP certificate is numerical (SCS tolerance), not symbolic; the Choi–Lam instance's physical realizability (Q1) is open; a per-instance certified real-root count is future work.
7. GPU batching / end-to-end training / peak memory / throughput were not measured.
8. Negative result that must stay: Laplace posterior mean is biased toward the flip (median 44.24 deg / 13.9 mm ADD-S) versus the hard argmin (2.06 deg / 3.7 mm) on the 15 LINEMOD frames.

## Editorial rules

- Tables: `booktabs`, wide tables as `table*` with `\scriptsize`; no vertical rules.
- Figures: `\\includegraphics[width=\\columnwidth]{...}` inside `figure`; wide overview
  figures in `figure*` with `width=\\textwidth`.
- Evidence tags stay honest: *paper claim* / *independent reproduction (this repo)* /
  *unverified*; Chinese: 论文声明 / 独立复现（本仓库） / 未验证 / 待核实.
- Author names, affiliations and funding are placeholders; do not invent them.
