> **Superseded 2026-11-28:** the phase-3 'still open' items were addressed in phase 4 — the 8-scene 3DMatch aggregate, KITTI odometry 00/05/07, two daylight In-The-Dark runs, and the GPU retry (job 3184, still blocked) are recorded in `phase4_consistency.md` and the paper v0.4. This file is kept as the phase-3 history; its 'still open' section is not the current status.

# Phase-3 consistency fixes (P3b, 2026-11-27)

Scope: `paper/draft.md` (§5, abstract, limitations/conclusion), `experiments/README.md`,
`experiments/repro_audit.py`, plus the QPEP-GPU engineering note. No theory/verify edits,
no commit/push. All before→after strings below are grep/JSON evidence, not new numbers.

## Fixed contradictions

| # | Contradiction (before) | Fix (after) | Evidence |
|---|---|---|---|
| 1 | Draft status line said **two 3DMatch scenes, KITTI drive 0001** while §5 reported four scenes and two drives. | Status line now says four 3DMatch scenes (two with FPFH), two KITTI raw drives, stereo fallback. | `grep -n "two 3DMatch scenes\|KITTI drive 0001" paper/draft.md` → 0 hits; §5 matches `threedmatch.json` + `extra_scenes.json`, `kitti.json` + `kitti2.json`. |
| 2 | Abstract/conclusion claimed **22–32× faster** without naming the RANSAC budget; LO-RANSAC/adaptive results were only in §5.7. | Abstract, §5.7, §6 and §7 now say "fixed 1000-iteration RANSAC", note adaptive RANSAC is faster at low outlier fractions and GNC is ~8× at 70%; LO-RANSAC caveat kept. | `grep -n "fixed 1000-iteration\|adaptive-budget" paper/draft.md`; §5.7 LO-RANSAC bullet unchanged. |
| 3 | In-The-Dark was described as "not measured / future work" (abstract, README §10) without the measured download block. | Now consistently **download-blocked** (14 KB/s single / ~34 KB/s 8-way, ≥590 MB, ETA 11.7–17.4 h), explicitly not a scientific negative; pointer to `real/inthedark/README.md`. | abstract, §5.5, §5.8(ii), §6, §7; README §0/§10/§15. |
| 4 | The KITTI stereo fallback was only a paragraph in §5.5/README; the **translation-negative** result was not in abstract/conclusion. | Labelled rotation-positive / translation-negative (0.052° rotation, 0.70 m median translation, 0% @0.1 m, 0.70 vs floor 0.239 m). | abstract, §5.5, §6, §7; `real/kitti_stereo/results/kitti_stereo_main.json` + README. |
| 5 | Experiments README §0 said real data / QPEP / GPU were **"未做"**; §10 said In-The-Dark "未做"; KITTI row said "单 drive"; §12.1 fallback was misplaced after §15. | Status table now lists measured subsets, blocked/scaffolded items and the GPU probe; KITTI row is two drives; fallback is §16 with a pointer from §12; §15 gaps updated. | `grep -n "未做\|单 drive\|GPU 批处理未做" experiments/README.md` → 0 stale hits. |
| 6 | New phase-3 JSONs were not in the reproducibility audit; a partial run could pass. | `repro_audit.py` now checks **38 JSON + 11 JSONL** (was 27 + 11), including stereo, odometry direction/motion, 8-scene manifest/verify/4 pair files, GPU probe and this phase's rounds; PASS. | `REPRO_AUDIT.md` overall **PASS**; 49 `| ok` rows; hashes + HEAD recorded. |

## Engineering finding kept as a round

`gpu_solver.eigh4_jacobi` had the wrong Givens sign (`atan2(2apq, app-aqq)` instead of
`atan2(2apq, aqq-app)`): the static-jacobi path disagreed with the NumPy reference by
max |dR|_F ≈ 1.80 with only 50% of μ-schedules matching. After the fix, the full CPU grid
(float64/float32, B={1,8,32,128,512}, N={100,500}) matches to ≤3.9e-16 (float64) /
≤3.7e-7 (float32), with 100% n_mu match and 0 failures (70 rows). The CUDA-graph path is
built but **not measured** (see below).

## Still open / deliberately not claimed

1. **GPU p50/p95 and CUDA-graph error: BLOCKED.** Node `ls` has 3 GPUs, all allocated to
   user cyr's jobs (3151/3094/3086); probe job 3171 was PENDING with StartTime
   2026-11-28T21:34 and was cancelled after recording the Slurm state; duplicate job
   3170 is still PENDING at 21:23. No graph was captured, so the CUDA-graph error is
   **not available** (not "no error"). `results/qpep_gpu.*` is the CPU validation
   fallback and is labelled as such.
2. **8-scene 3DMatch / low–mid–high overlap**: data, direction checks and pair manifests
   exist (`real/threedmatch_full/`), aggregate tables not run → not claimed.
3. **KITTI odometry 00/05/07**: loader + 18/18 direction check exist
   (`real/kitti_odom/verify_kitti_odom.json`), main table not run → not claimed.
4. **40-root QPEP solver** remains unimplemented; the 4×4 results are only the
   scalar-weight special case (`qpep_gpu/qpep_scope.md`).
5. Lab-level `README.md` still says "In-The-Dark 端到端训练未做" in its "not done" list;
   it is directionally consistent (not measured) but does not yet say "download-blocked".
   Left untouched because this round's declared scope is paper/experiments/repro.
