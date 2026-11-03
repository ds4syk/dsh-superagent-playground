This is by an automated agent, not a human.

## What this is

An analysis + controlled experiment answering the Feishu question: **can SDPR be a
bridge/output layer for V-JEPA 2?**

The short answer: **yes as a structured geometric output head, no as a replacement for the
missing pixel/video decoder.** SDPR solves a homogeneous QCQP and returns a certifiably
global geometric solution; JEPA deliberately has no decoder. The report separates the two
readings and tests the only one that is actually well-posed.

## Deliverables

- `report/sdpr_vjepa2_bridge.md` — main technical report (Chinese).
- `deliverables/SDPR_VJEPA2_桥接输出层_2026-11.docx` — DOCX knowledge base (default delivery).
- `kb/00.md` … `kb/04.md` — KB source pages.
- `src/` — panorama/data generation, V-JEPA 2 ViT-L feature extraction, SDPR rotation
  synchronization (QCQP + Shor SDP + certificate + implicit-diff backward), adapter/training,
  bridge gradient check, figures/analysis, DOCX builder.
- `report/repro.md` — environment, assets, Slurm commands.

## Experiment

Synthetic rotating-camera benchmark: a 2:1 natural image as an equirectangular environment
map; a camera rotates in place; 16 frames → 8 nodes with exact GT rotations. Frozen V-JEPA 2
ViT-L features (303.9M, `target_encoder`, 292 tensors, missing=0) → edge adapter predicts
relative rotations + weights → SDPR solves rotation synchronization (8 nodes, 13 edges) →
global rotations. Comparisons: direct MLP, two-stage (relative loss + frozen SDPR), end-to-end
through SDPR, chordal spectral synchronization, oracle (GT measurements + SDPR).

## Results (to be filled from `artifacts/results/metrics_*.json`)

| method | test_id | test_ood | rel meas. (id/ood) | cert |
|---|---|---|---|---|
| direct MLP | 1.87° | **12.67°** | — | — |
| two-stage + SDPR | **1.06°** | 15.32° | 0.98° / 10.34° | 1.00 |
| two-stage + chordal | 1.79° | 27.41° | 0.98° / 10.34° | — |
| end-to-end + SDPR | 1.18° | 15.34° | 1.12° / 10.27° | 1.00 |
| end-to-end + chordal | 1.97° | 27.55° | 1.12° / 10.27° | — |
| oracle (GT measurements) | 0.08° | 0.08° | 0.08° / 0.08° | 1.00 |

In-distribution, the structured SDPR head beats direct regression and chordal synchronization;
on the unseen panorama the direct absolute head transfers better because the edge adapter's
relative measurements degrade (0.98°→10.3°), which SDPR cannot fix. End-to-end fine-tuning
through SDPR did not improve over the two-stage pipeline.

## Verification

- 810 sequences × 16 frames encoded with the frozen V-JEPA 2 ViT-L (`target_encoder`, 292
  tensors, missing=0); features `[810,8,1024]`.
- 40-epoch two-stage head; 10-epoch end-to-end fine-tune (lr 1e-5); direct MLP; oracle; chordal
  baselines; `report/results/tables.md` and metrics JSON are committed.
- SDPR certificate: rank gap max 2.3e-8, H min eigenvalue ~-8e-7, numerical cert rate 1.00
  (strict 0.83–0.89); oracle 0.08°.
- Implicit gradient: finite-difference absolute error ~5e-6 on the well-conditioned N=4 problem;
  N=8 learned problems are ill-conditioned and the end-to-end gradient is noisy (documented).
- Gradient path to V-JEPA 2: last-block/patch/feature norms nonzero; one encoder SGD step
  changes the loss by +0.006°.
- Supplementary noise sweep (N=8, independent Gaussian measurement noise, σ=0–20°): SDPR,
  chordal and a local least-squares solver coincide; the structured-error advantage is specific
  to the learned measurements.

## Limitations

- Synthetic controlled benchmark, not real video/robot data.
- 8 nodes / 13 edges; SDPR on CPU (SCS); large-N scaling untested.
- Certificate checks are numerical (SCS tolerance), not formal proofs.
- V-JEPA 2 features are not trained for metric pose; the adapter learns a coarse mapping.
