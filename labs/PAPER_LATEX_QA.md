# Paper rewrite QA report (2026-11-28)

Final deliverable: six submission-style LaTeX drafts (3 papers × EN/ZH), compiled, with
normal academic structure, no draft/status/defensive markers, and measured numbers kept.

## 1. Toolchain and build status

Tectonic 0.17.0 (`/mnt/storage1/syk/tools/tectonic/tectonic`); IEEEtran for EN,
`ctexart` + Noto Serif/Sans CJK SC for ZH. All six were rebuilt from scratch in
`wt-paper` with `labs/build_paper_latex.sh`:

| Tree | Exit | Pages | Overfull hbox | Undefined refs | PDF |
|---|---|---|---|---|---|
| P1 EN | 0 | 16 | 0 | 0 | 780 KB |
| P1 ZH | 0 | 18 | 0 | 0 | 1.1 MB |
| P2 EN | 0 | 16 | 0 | 0 | 536 KB |
| P2 ZH | 0 | 17 | 0 | 0 | 788 KB |
| P3 EN | 0 | 13 | 6 ≤ 2.76 pt | 0 | 612 KB |
| P3 ZH | 0 | 13 | 4 ≤ 2.59 pt | 0 | 840 KB |

No errors, no missing citations. P3's residual overfull boxes are all below the 3 pt
threshold (max 2.76 pt EN / 2.59 pt ZH), invisible in print. Visual spot-checks of the
first page of each PDF and table-heavy pages show correct layout; the ZH PDFs render CJK
with no missing glyphs.

## 2. Banned-phrase audit — 0 hits

Script `labs/tools/audit_banned.py` (EN and ZH lists). Final result: **0 hits in all six
trees**, including `main.tex`, all `sections/*.tex`, abstracts, captions and appendices.

- EN list: TODO, pending, unverified, blocked, not run, not measured, we do not claim,
  do not claim, not a detector, honest/honesty, caveat, evidence level/grade,
  claim-to-evidence, falsif*, anti-overclaim, draft, status, `Source: results/`, `.json`,
  placeholder.
- ZH list: 草稿, 待定, 未测量, 未运行, 待核实, 证据等级, 不声称, 诚实地, 坦白, 状态,
  负结果 (as a label), 占位, 待撰写.

Removed content: the two non-measurement status tables in P1 (what-was-run inventory and
claim-to-evidence map), all evidence-level/source lines under tables, the P2 “what we do
not claim” subsection, the P3 “what is not measured” list, and every `\pend{}`/red marker.
Where data is genuinely absent, cells are blank or `--`; no TODO text remains.

## 3. Numeric consistency

- **EN ↔ ZH (same paper)**: `labs/tools/compare_trees.py` normalises the LaTeX number
  spellings and compares the sets. P1: 527 distinct numbers, **identical sets**.
  P2/P3: the only set differences are tabular column widths (`p{3.85cm}` etc.) and
  thousands-separator/formatting variants, not measured values.
- **Against the Markdown drafts**: key values checked in the PDFs/text, e.g. P1
  2.06°/3.7 mm (hard) vs 44.24°/13.9 mm (projected Laplace mean), +36.9°, 58.28°
  (unprojected mean, explicitly not a rotation error), 0.034188 gap with rank 6, 40
  complex roots on 4/4; P2 24.4→51.1%, 15.3→41.7%, 13.1→23.8%, one-sided 33.3/30.6/33.3%,
  two-sided 18.9/12.5/9.5%, n=90/72/84, 103.02 mm, 14×14, 4/12/16/32/40 spectrum;
  P3 0.078–0.150°/0.007–0.020 m (day + night windows), sparse night 0.197°/0.015 m,
  motion-matched control 0.121°/0.007 m, exact-vs-FD 4.9e-8/3.1e-8/1.25e-4, naive
  0.212/1.07 with boundary cosine 0.234, 22–32× vs fixed-budget RANSAC.
- P3 uses the **v0.4** draft from `final/p3` @ e796a93 (night windows, all 8 official
  3DMatch test scenes + low-overlap study, KITTI 00–10, no general 40-root solver, GPU
  run left as future work). `wt-paper` P3 draft/contributions/summary were synced to v0.4.
- The only P1 content deletion is the two non-measurement status tables; every measured
  table row and figure was retained.

## 4. Structure and style

All six follow Abstract, Introduction, Related Work, Method/Theory, Experiments,
Discussion and Limitations, Conclusion, References. Limitations are one concise scholarly
paragraph; unmeasured items are phrased as future work. No “draft v0.x” in titles or
abstracts. EN abstracts are 220–250 words. P1 remains the long journal paper (16/18 pp.,
17 tables, 5 figures); P2 main text is 6 pp. EN / 7 pp. ZH; P3 is 7 pp. main text + refs +
appendix.

## 5. Accepted-round ledger (rewrite phase)

Accounting rule: one accepted rewrite unit = one paper-language with a compiled artifact
and a clean audit. Failed attempts and the earlier machine-translation experiment count 0.

| Round | Unit | Owner (subagent) | Artifact | Pages | Audit | Accepted |
|---|---|---|---|---|---|---|
| R1 | P1 EN (T-RO) | 80972cdd (EN rewrite) | `01.../latex/en/main.pdf` | 16 | 0 banned, exit 0 | yes |
| R2 | P1 ZH (ctexart) | 80972cdd (hand translation) | `01.../latex/zh/main.pdf` | 18 | 0 banned, exit 0 | yes |
| R3 | P2 EN (ICRA) | 7102a3fb | `02.../latex/en/main.pdf` | 16 (6 main) | 0 banned, exit 0 | yes |
| R4 | P2 ZH | 7102a3fb (ZH sync, hand-edited) | `02.../latex/zh/main.pdf` | 17 (7 main) | 0 banned, exit 0 | yes |
| R5 | P3 EN (IROS) | 7aecf226 | `03.../latex/en/main.pdf` | 13 (7 main) | 0 banned, exit 0 | yes |
| R6 | P3 ZH | 7aecf226 (ZH sync, hand-edited) | `03.../latex/zh/main.pdf` | 13 (7 main) | 0 banned, exit 0 | yes |

Total accepted rounds: **6**. Failed/stuck attempts in this phase: 0 accepted; the earlier
Google-MT pass was discarded and is not counted.

## 6. Residual issues / notes

- P1 EN is 16 pages (T-RO has no hard page limit; 17 tables / 5 figures justify the
  length). If a shorter version is required, the per-frame LINEMOD tables are the natural
  candidates for supplementary material.
- P2 ZH main text is 7 pp. vs 6 pp. EN (Chinese typesetting); appendix and references are
  outside the limit. P3 ZH main text is 7–8 pp. depending on how the appendix boundary is
  counted; both are within the “6–7 + refs/appendix” target or one page over.
- P3 has 6/4 overfull hboxes at ≤2.76 pt; no action needed at submission scale.
- P2/P3 appendix tables are inside the PDF after `\appendix` and are not counted toward
  the conference page limits.
- Author names, affiliations and funding remain placeholders (`First Author`, `[Institution]`,
  `[单位名称]`), to be filled at submission.

## 7. Reproduction

```bash
cd /mnt/storage1/syk/.superagent-work/wt-paper
./labs/build_paper_latex.sh                     # all six, prints exit/pages/overfull
python3 labs/tools/audit_banned.py labs/01-laplace-qpeplayer/paper/latex/en en
python3 labs/tools/audit_banned.py labs/01-laplace-qpeplayer/paper/latex/zh zh
python3 labs/tools/compare_trees.py labs/01-laplace-qpeplayer/paper/latex/en \
        labs/01-laplace-qpeplayer/paper/latex/zh
```
