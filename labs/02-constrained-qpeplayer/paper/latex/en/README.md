# 02-constrained-qpeplayer — LaTeX draft (en)

- **Target venue:** IEEE International Conference on Robotics and Automation (ICRA), conference paper
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

1. Main text must stay within 6 pages; detailed per-cell tables go to the appendix and are clearly marked as not counted toward the limit.
2. Real data are GT-projected correspondences (30 random model vertices projected with the GT pose plus 1 px noise and 10/20/30 px bias), not detector/matcher output; n=90/72/84.
3. Learned-layer baselines (SDPRLayers in a constrained form, dQP, CVXPYLayers) were not run; the SCP baseline is a local linearisation + SOCP combination, not dQP.
4. The one-sided Shor flag is an upper bound only; the rigorous bound of record is the P2-Cert Schur repair (24.4/15.3/13.1% base, 51.1/41.7/23.8% with sound bound-RLT).
5. End-to-end training (T1-network, T2 KITTI/TUM/UTIAS, T3 hand–eye) is stated but not run; YCB-Video is layer-level only.
6. The FB epsilon->0 selection at weakly active points (G2) remains open: the paper may say 'a selected generalized derivative', never 'the derivative'.
7. The 16-root saturated template for the isotropic single-normal family is not shipped; the 4/12/16 spectrum is validated in canonical form only.
8. Gauss–Newton vs exact-Hessian ablation, backward-pass timing, and assembly/inner/certificate latency decomposition are not measured.
9. Zero-dimensional RLT on real data is a negative result; the positive result is the sound bound-RLT family.

## Editorial rules

- Tables: `booktabs`, wide tables as `table*` with `\scriptsize`; no vertical rules.
- Figures: `\\includegraphics[width=\\columnwidth]{...}` inside `figure`; wide overview
  figures in `figure*` with `width=\\textwidth`.
- Evidence tags stay honest: *paper claim* / *independent reproduction (this repo)* /
  *unverified*; Chinese: 论文声明 / 独立复现（本仓库） / 未验证 / 待核实.
- Author names, affiliations and funding are placeholders; do not invent them.
