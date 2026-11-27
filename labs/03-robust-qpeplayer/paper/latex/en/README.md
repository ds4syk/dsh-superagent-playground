# 03-robust-qpeplayer — LaTeX draft (en)

- **Target venue:** IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), conference paper
- **Document class:** `IEEEtran [conference]`
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

1. Main text must stay within 6 pages; kernel-dual / ablation detail tables go to the appendix and are clearly marked as not counted toward the limit.
2. No 40-root QPEP solver is implemented: the inner solver is weighted Kabsch (closed-form 3D-3D specialization) and a 4x4 Davenport/Horn variant is benchmarked separately.
3. In-The-Dark evidence is two daylight runs, 15 consecutive pairs per run, with a hand-written SIFT+StereoSGBM front-end and no training; night-time, glare, and the SDPR VGG/UNet front-end are untested.
4. 3DMatch covers 8 official scenes (not the full 50); KITTI odometry 00/05/07 uses 33 selected frames per sequence (not full 00–10).
5. BPnP/Pyatov/POLI are out of scope for 3D-3D; Theseus-GM and SDPRLayers+SCS are synthetic forward baselines only.
6. The GPU/CUDA-graph leg of the 4x4 inner-solver probe is blocked by node occupancy; only the CPU validation ran.
7. The 6D sign-flip result is numerical, not a theorem; the rigorous statement is the 1D Proposition.
8. No global optimality, breakdown-point, or adversarial-robustness guarantee; coherent-majority outliers can make the robust global optimum wrong.
9. The KITTI stereo fallback is rotation-positive but translation-negative (0% success at 0.1 m) and is not a substitute for a trained front-end.

## Editorial rules

- Tables: `booktabs`, wide tables as `table*` with `\scriptsize`; no vertical rules.
- Figures: `\\includegraphics[width=\\columnwidth]{...}` inside `figure`; wide overview
  figures in `figure*` with `width=\\textwidth`.
- Evidence tags stay honest: *paper claim* / *independent reproduction (this repo)* /
  *unverified*; Chinese: 论文声明 / 独立复现（本仓库） / 未验证 / 待核实.
- Author names, affiliations and funding are placeholders; do not invent them.
