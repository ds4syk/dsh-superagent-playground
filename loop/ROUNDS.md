# 10-round self-improvement loop

Each round: goal → action → result → validation → next.

| Round | Goal | Status |
|---|---|---|
| R1 | Verified dual / face certificate (SDP dual PSD projection, gap, face) | done |
| R2 | 3D non-planar / outlier benchmark (atomic vs local/global/cue/OpenCV) | pending |
| R3 | Paper theorem/proof/references polish (6 pages) | pending |
| R4 | Report + DOCX knowledge base update | pending |
| R5 | One-pager update | pending |
| R6 | Lean additional lemma (PSD or 1-D derivative) | pending |
| R7 | Code/tests/README hygiene | pending |
| R8 | Fresh-context self-review (paper/report) | pending |
| R9 | PR/CI update | pending |
| R10 | Final packaging + Clash stop | pending |

## R1 — Verified dual / face certificate

Goal: turn the numerical face diagnosis into a dual certificate check: project the
SCS dual `H` to PSD, check `tr(HX)=0`, report the certified bound and face rank
for the 12x12 R-form and the q-form moment SDP.

**Result (R1)**: 12x12 R-form planar face rank 4; SCS equality-dual H has min eig ~ -1.6e-7 (not strictly PSD), `tr(HX) ~ 3.9e-6`; projected H is PSD and `tr(H_psd X) ~ 3.9e-6`. So the numerical certificate is consistent with a rank-4 face but is **not** a strict verified dual (SCS tolerance); paper must state this honestly. JSON: `diffqpep/results/verified_dual.json`.

