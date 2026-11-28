# Final P3 scope audit（2026-11-28）

This file fixes the vocabulary for the final paper so "full" and "50/00–10/39" are not overclaimed.

| dataset | official total | measured here | why not more | claim allowed |
|---|---|---|---|---|
| 3DMatch geometric registration | **8 scene sets** (the official scene-fragment benchmark; 50-scene figure refers to the *training* split of RGB-D scenes, which has no per-pair GT for this table) | 8/8 scene sets, 20 trials/cell, high-overlap + low/mid/high overlap bins | no more official GT scene sets exist for this benchmark | "all 8 official geometric-registration scene sets" |
| KITTI odometry | 11 sequences (00–10), Velodyne ~84.8 GB | all 11 sequences represented, 32–33 selected frames/sequence, exact GT | full 84.8 GB download exceeds budget; range-fetch only protocol frames | "all 11 sequences represented by 33 frames each" — **not** full 00–10 streams |
| KITTI raw | 100+ drives | 0001, 0002 (2 drives), 10 pairs each | raw drives are 320–900 MB each; 2 drives already give independent OXTS motion | "two raw drives" |
| UTIAS In-The-Dark | 39 runs (30 h day/night cycle) | 000002/000012 (daylight) + night run(s) in phase-4 night worker | 590 MB–2.76 GB/run; range extraction gives GT + selected images cheaply | "2 daylight runs; night subset if measured" |
| QPEP | full 40-root Gröbner/action-matrix solver | scalar-weight 4×4 eigen specialization + scope note | 40-root solver is for generalized weighted problems; no implementation | "no 40-root solver; scalar-W 4×4" |
| GPU | 3× RTX 5090 on node | CPU validation only; GPU jobs blocked by cyr occupancy | no free GPU for the session | "CPU-only; GPU blocked by occupancy" |
| external baselines | SDPR/Theseus/BPnP/Pyatov/POLI | Theseus-GM + SDPR-SCS forward on synthetic; BPnP/Pyatov/POLI out of scope | SDPR forward W=I equals Kabsch; no Mosek/backward; BPnP/Pyatov are different tasks; POLI no pose code | "Theseus/SDPR forward synthetic; no real-data external baseline" |

## Rules

1. Never write "50-scene 3DMatch" unless the text explicitly says "training split without GT".
2. Never write "KITTI 00–10"; write "00/05/07 subset".
3. In-The-Dark day/night must be split; a daylight run is not night-time robustness.
4. SDPR with W=I is not an independent robust baseline; state the Kabsch equivalence.
5. The GPU block is an environment fact, not a negative result.
