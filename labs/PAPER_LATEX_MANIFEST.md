# Paper LaTeX manifest (2026-11-28)

Six submission-style drafts, English and Chinese, all compiled with Tectonic 0.17.0.
Venues: P1 IEEE T-RO (`IEEEtran` journal), P2 IEEE ICRA (`IEEEtran` conference),
P3 IEEE/RSJ IROS (`IEEEtran` conference); Chinese versions are `ctexart` two-column
mirrors (《自动化学报》-style for P1, Chinese conference-style for P2/P3).

| ID | Paper | Lang | PDF | Pages | Main text | Tables | Figures |
|---|---|---|---|---|---|---|---|
| P1-EN | Laplace-QPEPLayer | EN | `labs/01-laplace-qpeplayer/paper/latex/en/main.pdf` | 16 | 15 + refs | 17 | 5 |
| P1-ZH | Laplace-QPEPLayer | ZH | `labs/01-laplace-qpeplayer/paper/latex/zh/main.pdf` | 18 | 16 + refs | 17 | 5 |
| P2-EN | Constrained-QPEPLayer | EN | `labs/02-constrained-qpeplayer/paper/latex/en/main.pdf` | 16 | 6 (pp. 1–6) + appendix/refs | 18 | 4 |
| P2-ZH | Constrained-QPEPLayer | ZH | `labs/02-constrained-qpeplayer/paper/latex/zh/main.pdf` | 17 | 7 (pp. 1–7) + appendix/refs | 18 | 4 |
| P3-EN | Robust-QPEPLayer | EN | `labs/03-robust-qpeplayer/paper/latex/en/main.pdf` | 13 | 7 (pp. 1–7) + refs/appendix | 10 | 3 |
| P3-ZH | Robust-QPEPLayer | ZH | `labs/03-robust-qpeplayer/paper/latex/zh/main.pdf` | 13 | 7 + refs/appendix | 10 | 3 |

Each source tree is self-contained: `main.tex`, `preamble.tex`, `sections/*.tex`,
`refs.bib` (BibTeX for EN; ZH also uses BibTeX `refs.bib` with `plain`/`unsrt`),
`figures/*.png`, `README.md`, and `build.log` from the last successful compile.

## Compile

```bash
export HTTPS_PROXY=http://127.0.0.1:7899 HTTP_PROXY=http://127.0.0.1:7899
cd labs/01-laplace-qpeplayer/paper/latex/en
/mnt/storage1/syk/tools/tectonic/tectonic -X compile main.tex
```

All six at once:

```bash
./labs/build_paper_latex.sh
```

## Sources of truth

- P1: `labs/01-laplace-qpeplayer/paper/draft.md` (v4) and `experiments/`.
- P2: `labs/02-constrained-qpeplayer/paper/draft.md` (v0.3) and `experiments/`.
- P3: `labs/03-robust-qpeplayer/paper/draft.md` (**v0.4**, 2026-11-28, from `final/p3` @ e796a93)
  and `experiments/`. P3 numbers include the night-window In-The-Dark runs and the
  low-overlap 3DMatch study; the general 40-root QPEP solver and GPU run are future work.
