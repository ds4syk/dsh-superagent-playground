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
  tree are traceable to that file and its `experiments/` result files.

## Build status

| item | value |
|---|---|
| last compile | exit 0, no undefined references, no overfull box |
| main-text pages (Secs. 1–7) | 6 |
| total pages | 16 |
| tables / figures | 18 / 4 |

## Files

`main.tex` (documentclass, title, abstract, keywords, `\input` lines), `preamble.tex`,
`sections/*.tex`, `figures/*.png`, `refs.bib` (EN only), `build.log`.

## Scope of the reported results

1. Main text is 6 pages; per-cell tables, the positioning table and the implementation
   verification live in the appendix after `\appendix`, which is excluded from the limit.
2. Real data are GT-projected correspondences (30 random model vertices projected with the
   GT pose plus 1 px noise and 10/20/30 px bias), `n=90/72/84`; the study is layer-level.
3. The one-sided Shor flag (`J - Shor <= 1e-6`) is an upper bound only. The certificate of
   record is the P2-Cert Schur repair: 24.4/15.3/13.1% with base cuts and 51.1/41.7/23.8%
   with the sound bound-RLT family.
4. The shipped enumeration template is the 40-root unsaturated template for the generic and
   PnP cases; the 4/12/16/32/40 spectrum is validated numerically in canonical form.

## Editorial rules

- Tables: `booktabs`, wide tables as `table*` with `\scriptsize`; no vertical rules.
- Figures: `\includegraphics[width=\columnwidth]{...}` inside `figure`; wide overview
  figures in `figure*` with `width=\textwidth`.
- Author names, affiliations and funding are placeholders; do not invent them.
