# Night UTIAS In-The-Dark: range-fetched prefix results

The full night zips are 2.6–2.76 GB; we range-fetched only the metadata plus 7–11 consecutive left/right frames per run. GT is the same composed protocol (repeat spatial × teach temporal × camera-frame conjugation), validated on day runs; night rows are **prefix subsets**, not full runs.

| run | local time | frames | pairs | image mean intensity | natural inliers | GNC rot | GNC trans | success@0.1 m |
|---|---|---|---|---|---|---|---|---|
| 000017 | 21:58, nighttime/dark | 0–8 | 7 | 96.6 | 76.8% | 0.440° | 0.0259 m | 100% |
| 000018 | 22:28, nighttime/dark | 2529–2539 | 7 | 98.4 | 97.0% | 0.137° | 0.0113 m | 100% |
| 000002 (day) | 09:46, morning/sunny | 200–214* | 15 | 133 | 79.4% | 0.078° | 0.020 m | 100% |
| 000012 (day) | — | 15 pairs | 15 | 131 | 88.5% | 0.092° | 0.010 m | 100% |

\*Day table from `inthedark_main.md`.

| method | night17 rot / trans / succ@0.1 | night18 rot / trans / succ@0.1 |
|---|---|---|
| Kabsch | 0.917° / 0.212 m / 29% | 0.280° / 0.020 m / 100% |
| RANSAC-1000 | 0.509° / 0.034 m / 100% | 0.141° / 0.015 m / 100% |
| IRLS-GM | 0.440° / 0.026 m / 100% | 0.137° / 0.011 m / 100% |
| **GNC-GM** | **0.440° / 0.026 m / 100%** | **0.137° / 0.011 m / 100%** |

## Reading

- Night is not uniformly harder: run_000017's prefix is dark/low-overlap (77% inliers, 0.44°), run_000018's prefix is easy (97%, 0.14°). Both are within 0.1 m.
- The robust layer is doing real work on night17: Kabsch only 29%, GNC 100%.
- These are **prefix subsets** (7 pairs), not full-run averages; do not present them as full-night benchmark. They do close the "no night data at all" gap.
- The front-end is the hand-written SIFT+SGBM, not SDPR VGG/UNet; no training.
