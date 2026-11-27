# P2 final push — measured results summary (70-round budget)

All numbers below are produced by the subagent code in this working tree; see the per-line JSON/MD for raw rows.

## 1. SDPRLayers-style baselines (our reimplementation; official PyTorch layer not used)

| cell | variant | n | pen mm | rot deg | trans mm | feasible | certified (sound) | ms | λ2/λ1 |
|---|---|---|---|---|---|---|---|---|---|
| synth p=8 | unc_ref | 50 | 103.018 | 5.296 | 97.90 | 0% | 0% | 18.8 | nan |
| synth p=8 | unc | 50 | 103.018 | 5.296 | 97.90 | 0% | 0% | 30.7 | 2.6e-11 |
| synth p=8 | hard | 50 | 0.000 | 4.169 | 10.63 | 96% | 96% | 29.6 | 7.1e-11 |
| synth p=8 | slack | 50 | 0.000 | 4.169 | 10.63 | 98% | 98% | 62.7 | 2.1e-01 |
| synth p=16 | unc_ref | 12 | 96.318 | 6.029 | 96.66 | 0% | 0% | 17.2 | nan |
| synth p=16 | unc | 12 | 96.318 | 6.029 | 96.66 | 0% | 0% | 37.2 | 3.7e-11 |
| synth p=16 | hard | 12 | 0.000 | 3.923 | 10.30 | 100% | 100% | 33.8 | 1.1e-10 |
| synth p=16 | slack | 12 | 0.000 | 3.923 | 10.30 | 92% | 92% | 194.5 | 1.5e-01 |
| synth p=32 | unc_ref | 50 | 103.018 | 5.296 | 97.90 | 0% | 0% | 20.0 | nan |
| synth p=32 | unc | 50 | 103.018 | 5.296 | 97.90 | 0% | 0% | 35.1 | 2.6e-11 |
| synth p=32 | hard | 50 | 0.000 | 4.169 | 10.63 | 98% | 98% | 40.5 | 4.4e-11 |
| synth p=32 | slack | 50 | 0.000 | 4.169 | 10.63 | 100% | 100% | 522.8 | 1.2e-01 |
| synth p=64 | unc_ref | 50 | 103.018 | 5.296 | 97.90 | 0% | 0% | 19.4 | nan |
| synth p=64 | unc | 50 | 103.018 | 5.296 | 97.90 | 0% | 0% | 31.9 | 2.6e-11 |
| synth p=64 | hard | 50 | 0.000 | 4.169 | 10.63 | 92% | 92% | 52.4 | 1.3e-10 |
| synth p=64 | slack | 50 | 0.000 | 4.169 | 10.63 | 92% | 92% | 2259.4 | 8.7e-02 |
| LM | unc_ref | 3 | 10.124 | 2.250 | 37.08 | 33% | 33% | 20.6 | nan |
| LM | unc | 3 | 13.088 | 2.275 | 37.08 | 0% | 0% | 584.6 | 8.6e-02 |
| LM | hard | 3 | 10.052 | 3.101 | 51.42 | 33% | 0% | 642.0 | 8.8e-02 |
| LM | slack | 3 | 6.009 | 2.204 | 35.59 | 33% | 33% | 3130.2 | 6.4e-02 |

- `unc` = SDPRLayers-style unconstrained Shor relaxation + pose extraction; `hard` = same relaxation with the hard `g<=0` rows (the 14×14 constrained Shor); `slack` = lifted slack version; `unc_ref` = the multistart reference on the same instance.
- `certified (sound)` uses the rigorous Schur/dual bound when present (`certified_sound`), otherwise the one-sided flag (`certified`).

## 2. dQP / CVXPYLayers-style differentiable QP (from `baseline_dqp.md`)

- p=8, n=2/trial: dQP pen 0.000 mm, rot 4.87°, trans 11.50 mm, feasible 100%, **cert 50%**, 192 ms; SCP 100%/100%, 127 ms; ALM 100%/100%, 175 ms.
- Implicit differentiation vs finite differences: median rel. error ~1e-6 on random well-conditioned QPs; **max 3.3** on LICQ-degenerate active sets (planar multi-contact) where the QP derivative is non-unique — structural, not an implementation bug.
- CVXPYLayers cross-check: contracted-gradient rel. diff <2e-5. `theseus-ai` is installed but `import theseus` fails (missing CHOLMOD); Theseus-style LM/Huber baseline is in `baselines_theseus.py` but **was not run** (blocked).

## 3. Real data: BOP ADD/ADD-S + KITTI + UTIAS

- **BOP ADD(-S) (LM, n=90)** — ADD(-S) median [IQR] and ADD(-S)<2cm: Unconstrained 31.5 [18.8,38.6]/31.1%; Soft 37.1 [22.6,51.9]/16.7%; Projection 28.1 [16.3,36.4]/32.2%; **ALM 35.4 [21.7,50.0]/18.9%**, with ALM 0 mm penetration / 100% feasible. Honest reading: constraints trade ADD accuracy for admissibility; projection has the best ADD but cert 15.6% (one-sided). Full tables (LM/LM-O/YCB) in `real_bop_add.md`; pose dump replays `raw.json` exactly (`max|diff|=0`).
- **KITTI odometry (layer-level, LiDAR→2D, sequences 00/05/07)**: pooled depth+tilt, ALM vs unconstrained — tilt>10° **0.0% vs 8.3%**, rot RMSE **26.45° vs 51.03°**, cert 79.4% vs 76.2%; clean setting both recover GT (rot 0.05°). Full tables in `kitti_layer.md`.
- **UTIAS In-The-Dark**: `real_utias.json` (114 pairs on run_000000, stereo/GT transforms) — see file; the loader is present, the constraint table may be partial.

## 4. End-to-end CPU toy (partial)

- `e2e_main.json` contains only a 3-epoch smoke (val ADD ~106.9 mm, rot ~4.77°, feasible 0% for the unconstrained head) — **not a trained comparison**. `e2e_layer.json` has per-layer forward timings. Treat as **partial/blocked**: the FB-backward training loop and four-head comparison were not completed before the time cap.

## 5. Paper

- `paper/draft.md` +100/−24 vs base (some new baseline text integrated), but **5 pending/TODO mentions remain** and the SDPRLayers/dQP/E2E/KITTI/UTIAS tables are not yet in the draft. `theory/proofs.md` unchanged in this phase.

## Blockers / not delivered

- Theseus baseline not run (`import theseus` fails: missing CHOLMOD); Theseus-style code exists.
- SDPR slack/large-p tables only partially measured (p=16 n=12; p=8/32/64 n=50; LM n=3).
- E2E training incomplete (smoke only).
- UTIAS constraint table partial; KITTI counts need dedup verification (frames list appeared 4× in the table).
- Paper integration incomplete (SDPR/dQP/E2E/real tables not yet inserted; TODO 5).
- No commit/push; all artifacts are in this working tree.
