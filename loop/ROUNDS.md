# 10-round self-improvement loop

Each round: goal → action → result → validation → next.

| Round | Goal | Status |
|---|---|---|
| R1 | Verified dual / face certificate (SDP dual PSD projection, gap, face) | done (numerical, not strict) |
| R2 | 3D non-planar / outlier benchmark (atomic vs local/global/cue/OpenCV) | done (negative, 299) |
| R3 | Paper theorem/proof/references polish (6 pages) | done, revised in R9 |
| R4 | Report + DOCX knowledge base update | done |
| R5 | One-pager update | done |
| R6 | Lean additional lemma (PSD or 1-D derivative) | done (1-D + rank) |
| R7 | Code/tests/README hygiene | partial: code + README, no test dir |
| R8 | Fresh-context adversarial review (paper/report) | done (15 findings, 2 fatal) |
| R9 | Apply R8 + P3P/same-start benchmark + PR | done |
| R10 | Final packaging + Clash stop | in progress |

## R1 — Verified dual / face certificate

Goal: turn the numerical face diagnosis into a dual certificate check: project the
SCS dual `H` to PSD, check `tr(HX)=0`, report the certified bound and face rank
for the 12x12 R-form and the q-form moment SDP.

**Result (R1)**: 12x12 R-form planar face rank 4; SCS equality-dual H has min eig ~ -1.6e-7 (not strictly PSD), `tr(HX) ~ 3.9e-6`; projected H is PSD and `tr(H_psd X) ~ 3.9e-6`. So the numerical certificate is consistent with a rank-4 face but is **not** a strict verified dual (SCS tolerance); paper must state this honestly. JSON: `diffqpep/results/verified_dual.json`.



## R2 — 3D non-planar/outlier (negative)

8-point non-planar PnP, 299 instances, noise 1e-3, with/without 10% outlier:

| method | no outlier | 10% outlier |
|---|---|---|
| local KKT | 0.431 | 0.361 |
| global-naive | 0.993 | 0.860 |
| cue-init local | 0.993 | 0.863 |
| OpenCV (best of 4) | 1.000 | 0.896 |
| atomic | 0.967 | 0.849 |

Atomic is **not** better; the cue can select a wrong branch when the cost already separates branches. Positive case is the exact 3-point tie (R3/R5). JSON: `diffqpep/results/benchmark3d.json`.

## R3 — Paper

Added the 3D negative control and the SCS-dual caveat (min eig -1.6e-7, tr(HX) 3.9e-6; projected PSD but numerical, not strict). Paper is 6 pages content + 1 page references.

## R4/R5 — Report/DOCX + one-pager

Report adds §5.7 3D negative, §5.8 verified dual (numerical), §5.9 Lean/loop; DOCX rebuilt (374873 bytes). One-pager panel (c) notes the 3D negative.

## R6/R7 — Lean + code

Lean: 1-D tie + value-face rank, `lake build` 8766 jobs; PSD/general SDP/KKT paper. Code: `exp_verified_dual.py`, `exp_3d_benchmark.py`, README boundary section; `rform.recover` rejects rank>1.

## R8 — adversarial self-review

Fresh-context subagent, 15 findings, 2 fatal:
- F1 Theorem 2(ii)/3 false without a transversal-switch hypothesis; "simple tie" undefined; set-valued map category error.
- F2 broken refs (`Lemma ??`, "Proposition 3").
- F3 R-form "double cover / 2 atoms" wrong: X=xx^T identifies ±x; it is one atom plus a two-to-one section.
- F4-F7 3-point experiment did not test the certificate layer; 8.8/10.4 decomposition backwards; cue leaks truth; false P3P claim.
- F8 Table 2 rank/gap inconsistency; F9 overclaimed non-necessity and Ω(1/m); F10 dual framing; F11 3D protocol; F12 status sync; F13-F15 wording/Lean.

## R9 — fixes + benchmark

Applied all 15. Paper: transversal-tie theorem, corrected R-form section, both directions of value/face separation, smoothing annealing corollary, fixed refs, consistent Table 2 tolerance, scoped claims. Benchmark: added `cv2.solveP3P` (3-point, up to 4 solutions), same-start ablation (4 vs 8), random-cue control, translation/pose scoring. Result: atomic-8 80.8, cue-init 64.4, multi-start 27.0, P3P+cue 86.6, P3P oracle 98.8, QPEP oracle 89.6; selection error 8.8, recall gap 10.4. No SOTA claim.

