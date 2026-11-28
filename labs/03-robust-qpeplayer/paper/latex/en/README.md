# 03-robust-qpeplayer — LaTeX paper (en)

- **Target venue:** IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), conference paper
- **Document class:** `IEEEtran [conference]`
- **Compile command:**

  ```bash
  export HTTPS_PROXY=http://127.0.0.1:7899 HTTP_PROXY=http://127.0.0.1:7899
  /mnt/storage1/syk/tools/tectonic/tectonic -X compile main.tex
  ```

- **Engine:** Tectonic 0.17.0 (`/mnt/storage1/syk/tools/tectonic/tectonic`).
- **Source of truth:** `../../draft.md` v0.4 (Markdown; do not modify). All numbers in this
  tree are traceable to that file and its `experiments/` result files.

## Build status

| item | value |
|---|---|
| last compile | 2026-11-28, exit 0 |
| main-text pages | 7 (Sections 1–7; references start on p. 8) |
| total pages | 13 |
| tables / figures | 10 / 3 |
| warnings | no undefined references; max overfull hbox 2.76 pt |

## Files

`main.tex` (documentclass, title, abstract, keywords, \input lines), `preamble.tex`,
`sections/*.tex`, `figures/*.png`, `refs.bib` (EN only), `build.log`.

## Evaluation scope (v0.4)

1. Main text is 7 pages; the appendix is not counted toward the limit.
2. The inner solver is weighted Kabsch, the closed-form 3D–3D specialization of QPEP; a
   4×4 Davenport/Horn variant is benchmarked in the appendix. A general 40-root QPEP inner
   solver and GPU batching are future work.
3. In-The-Dark evidence is two daylight runs (15 pairs each) and two 10-pair night windows,
   with a hand-written SIFT+StereoSGBM front end and no training.
4. 3DMatch covers all eight official test scenes plus a strict 0.20–0.50 low-overlap band;
   KITTI odometry covers 00–10 with 33 selected frames per sequence.
5. Theseus-GM and SDPRLayers+SCS are external baselines on synthetic 3D–3D.
6. The 6D sign flip is a numerical result; the formal statement is the 1D proposition.
7. The robust cost is solved locally; coherent-majority outliers can make the global optimum
   itself wrong.
8. The KITTI stereo front end solves rotation but leaves translation front-end-limited.

## Editorial rules

- Tables: `booktabs`, wide tables as `table*` with `\scriptsize`; no vertical rules.
- Figures: `\\includegraphics[width=\\columnwidth]{...}` inside `figure`; wide overview
  figures in `figure*` with `width=\\textwidth`.
- Author names, affiliations, and funding are intentionally generic; do not invent them.
