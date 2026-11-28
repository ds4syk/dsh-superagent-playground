# P2 final push — accepted-round ledger (`paper/round_paper_final2.md`)

Worktree: `/mnt/storage1/syk/.superagent-work/wt-final-p2-paper`, branch of `labs/02-constrained-qpeplayer`.
Owner of this line: `paper/draft.md`, `theory/proofs.md` (additive only), this file,
`experiments/results/fig_p2_overview.png`, `experiments/results/make_p2_overview.py`.
No commit / no push. Budget: up to 20 accepted rounds. CPU only. Every number in
`paper/draft.md` must be traceable to a JSON/MD file with an explicit `n`.

Format: `E<n> | unit | command | result | artifact | accept`.

---

## E0 · Baseline snapshot (before any edit)

| item | value |
|---|---|
| draft | `paper/draft.md`, 830 lines, status line v0.5 (2026-11-28) |
| TODO/TBD | 0 / 0 |
| `[verify]` reference entries | 2 (entry 6, NeurIPS page range; entry 4 was already resolved in v0.5) |
| tables | 5.1–5.12, 5.9b, 5.9c, 5.16, 5.17 (no 5.13–5.15 exist; §5.11 nevertheless cites "Table 5.13") |
| stale markers | 8 hits of `pending`/`smoke`/`not completed`/`stated but not run`, 3 of them superseded by the results below |
| sibling result files present at E0 | real: `kitti_layer.*`, `utias_layer.*`, `real_bop_add.*` (mtime 00:37–00:43); e2e: `e2e_main.json`, `e2e_layer.json` (00:40); dqp: `baseline_dqp.*` (00:33); sdpr: synthetic p8/32/64 (00:37–00:40), LM n=3 (00:36), p16 n=12 archive (00:58) |

`accept`: baseline recorded. The 830-line file was restored from the read-only
sibling snapshot `wt-final-p2/.../paper/draft.md` after the partially-applied
edit in this worktree (792 lines, missing §5.7/§5.8) was detected; the restoration
is a working-tree write only and changes no sibling worktree.

---

## E1 · Claim → source inventory (every table, every headline claim)

All paths below are relative to `labs/02-constrained-qpeplayer/`. "n" is the
number of independent instances behind the cell, not the number of rows printed.
`[R]` = independently reproduced in this repository; `[P]` = prototype sanity
check; `[X]` = re-implementation of another paper's layer (not the official code).

| table | claim (unit) | source file(s) | n | aggregation |
|---|---|---|---|---|
| 5.1 | constraint composition per $p$ | `results/raw_main.json` (`config.split`, `recs[].actypes`) | 50/cell | counts of scalar rows; FoV recorded as pairs |
| 5.2 | metric definitions / pass rule | `experiments/protocol.md` §3 | — | definitions only |
| 5.3 | penetration, 4 methods × $p$ | `results/raw_main.json`; reproduced by `results/main_table.json/.md` | 50/cell | median [IQR], rate $d_{\rm pen}>1$ mm |
| 5.4 | rotation / translation error, 4 × $p$ | `results/raw_main.json` | 50/cell | median [IQR] |
| 5.5 | feasibility, certification, runtime | `results/raw_main.json` | 50/cell | rate; median wall-clock |
| 5.6 | noise × bias, 4 cells × 4 methods | `results/raw_main.json` (`agg_noise`) | 50/cell | median [IQR], rate |
| 5.7 | soft-penalty Pareto, 5 weights | `results/raw_main.json` (`pareto`) | 250 (5×50) | median, rate |
| 5.8 | certificate margins, 8 cells | `results/cert_margins.json` (from `raw_main.json`) | 400 | max/median $\lambda_2/\lambda_1$, Shor$-J$ range, mean SDP time |
| 5.8b | exact Schur vs $P$-shift dual repair | `results/p2b_schur_check.json/.md` | 6 | per-instance gaps, case A |
| 5.9 | zero-dim RLT ablation, 4 noise levels | `experiments/theory/results/rlt.json` | 16 (4/cell) | counts, median abs gap, time |
| 5.10 | real layer table (BOP), 4 methods × 3 datasets | `experiments/real/linemod/results/raw.json`; `experiments/real/bop/results/lmo/raw.json`; `experiments/real/bop/results/ycbv/raw.json`; Wilson CIs in each `main_table.md` | 90 / 72 / 84 | median, rate + Wilson |
| 5.11 | rigorous Schur / $P$-shift certification, 3 cut families × 3 datasets | `results/rigorous_real.json/.md`; per-dataset `results/rigorous_{lm,lmo,ycbv}.json` | 90 / 72 / 84; 738 replay rows | count + Wilson, paired McNemar |
| 5.12 | LM-O depth+tilt augmentation, 6 worst rows | `results/rigorous_lmo_depth_tilt.json`; row-level gaps in `experiments/real/bop/results/lmo/rigorous/rigorous_rows.json` | 6 of 61 uncertified | per-row gap before→after |
| 5.9b | SDPRLayers-style forward relaxation | `results/baseline_sdpr_synth_p{8,16,32,64}.json`, `results/baseline_sdpr_lm.json` **[X]** | see §5.7 (n per cell in table) | median, rate, ms |
| 5.9c | CPU end-to-end toy, 4 heads | `results/e2e_main.json`, `results/e2e_layer.json` **[R]** | 8 validation scenes | forward metrics; training budget |
| 5.16 | KITTI odometry layer, 4 settings | `results/kitti_layer.json/.md` **[R]** | 63/cell (3 seq × 21 pairs) | median [IQR], RMSE, rate + Wilson |
| 5.16b | UTIAS In-The-Dark layer, 3 settings | `results/utias_layer.json/.md` **[R]** | 38/cell (run_000000) | median [IQR], RMSE, rate + Wilson |
| 5.17 | BOP ADD / ADD-S, 4 methods × 3 datasets | `results/real_bop_add.json/.md` **[R]** | 90 / 72 / 84 | median [IQR], bootstrap CI, Wilson |

Headline claims and their sources:

| claim (location) | source | n |
|---|---|---|
| ALM 100% feasible / certified at $p=8$–$64$ (abstract, §5.2, Table 5.5) | `results/raw_main.json` | 200 |
| certificate tight to $2.2\times10^{-12}$ on the controlled instance (abstract, §5.1.5 C5) | `verify/` outputs | 1 |
| rigorous real rates 24.4/15.3/13.1% base → 51.1/41.7/23.8% with bound-RLT (abstract, §5.5(v), §6, §7) | `results/rigorous_real.json` | 90/72/84 |
| synthetic exact inner = multistart to $1.8\times10^{-12}$ (§5.6) | `results/exact_alm.json/.md` | 8 ALM trials, 114 enumerations |
| SCP 98–100% certificate, 130–142 ms (§5.6) | `results/scp_baseline.json/.md` | 150 (3×50) |
| neighbouring-layer comparison (§5.7) | `results/baseline_sdpr_*`, `results/baseline_dqp.*`, `results/baseline_theseus_final.*` (when present) | per row |
| end-to-end toy + FB gradient check (§5.8) | `results/e2e_final.*` (when present) | per row |
| KITTI rotation RMSE 4.68°→ replaced by final dedup table (§5.9) | `results/kitti_layer.json/.md` | 63/cell |
| UTIAS depth/FoV layer (§5.9) | `results/utias_layer.json/.md` | 38/cell |
| ADD(-S) 28.1/18.1/8.9 mm projection best, ALM 35.4/25.4/25.2 (§5.10) | `results/real_bop_add.json/.md` | 90/72/84 |
| all 24 references | Crossref / arXiv / publisher pages, checked 2026-11-28 | 24 |

`accept`: inventory complete for every table and headline claim; the only rows
whose sources were not yet final at E1 are 5.9b, 5.9c and the Theseus paragraph,
which are exactly the units E2–E4 replace. Cells with no measurement are left
blank in the draft rather than estimated.

---

## E2 · SDPRLayers table: awaiting final `baseline_sdpr_final.md`
`pending` — the sibling's p=16 n=50 file landed at 00:59, LM n=90 / LM-O n=72 / YCB n=84 were
still running at the time of writing. No number was copied into §5.7 from an unfinished run.

## E3 · dQP / Theseus: awaiting final `baseline_dqp_final.md`
`pending` — `baseline_dqp.json/.md` (n=2 per cell) is in the sibling worktree; the final file and
`baseline_theseus_final.*` were not present. The §5.7 dQP paragraph is therefore still the n=2
version (traceable to `baseline_dqp.md`), and Theseus remains "not run (CHOLMOD)".

## E4 · CPU end-to-end toy: awaiting final `e2e_final.md`
`pending` — only `e2e_main.json` / `e2e_layer.json` (3 epochs, no FD check) exist. §5.8 is
untouched until the trained four-head run and the FD check land.

## E5 · KITTI table de-duplicated; UTIAS table added (accepted)

| | |
|---|---|
| unit | replace stale KITTI Table 5.16 with the de-duplicated final run; add the missing UTIAS table (the §5.9 heading promised UTIAS but the section had no UTIAS content) |
| command | read `wt-final-p2-real/.../results/{kitti_layer,utias_layer}.{json,md}`; copy them into this worktree's `experiments/results/`; recompute every cell from the JSON |
| result | KITTI n=63 per cell (3 sequences × 21 distinct pairs, repeated pair list removed); the 15%-outlier cell changed materially vs the old table (rot RMSE 22.91→22.99 unconstrained, 4.68→5.06 ALM; trans median 2245→2327.95; cert 98.4% unchanged) and the 50%-outlier ALM cell changed (rot RMSE 52.68→16.96; depth violation 88.9%→0.0%; cert 19.0%→14.3%/12.7%; 10 of 25 flips repaired). The change is a protocol fix (depth now on all 80 landmarks, not the 24 closest) plus de-duplication. UTIAS: run_000000, n=38 per cell, 3 settings, Table 5.16b; at 20 px+5% wrong matches ALM rot RMSE 10.87° vs 57.57°, trans 603.0 vs 1375.3 mm, depth violations 0.0% vs 10.5%. `ms` uses the JSON median (the `.md` prints the mean; this is stated in the source line). |
| artifact | `paper/draft.md` §5.9 (Tables 5.16 + 5.16b, 8 readings); `experiments/results/{kitti_layer,utias_layer}.{json,md,png}` |
| accept | yes |

## E6 · ADD table: certification mapping corrected (accepted)

| | |
|---|---|
| unit | map Table 5.17 to the rigorous certificate counts on the same poses |
| command | compare `results/real_bop_add.md` (ADD counts) with `results/rigorous_*.json` and Table 5.11 |
| result | ADD values and counts unchanged (28/90, 31/72, 54/84 unconstrained; 17/90, 26/72, 34/84 ALM); the table now states the rigorous ALM rates on the same replay (base 22/90→46/90, 11/72→30/72, 11/84→20/84 with bound-RLT) and that the table's own `cert` column is the one-sided SCS upper bound for *every* method |
| artifact | `paper/draft.md` §5.10 source line |
| accept | yes |

## E7 · Stale "pending"/smoke wording removed (accepted, except §5.8/§5.11 which depend on E4)

| | |
|---|---|
| unit | remove wording superseded by the new results |
| command | `grep -n "pending\|smoke\|not completed\|Table 5.13" paper/draft.md` |
| result | abstract no longer says KITTI/TUM/UTIAS "stated but not run" (T2 is run at layer level, TUM is not); §5.1.3 no longer says "pending for the items in §5.6"; §5.1.4 distinguishes protocols run at layer level (KITTI, UTIAS) from T1/TUM/T3 not run; §5.11's dangling "Table 5.13" reference replaced by §5.7; §5.11 opening no longer refers to a "pending list". `grep` now returns no `pending` and no `Table 5.13`; the only remaining `smoke`/`not completed` hits are the §5.8 toy paragraph and its §5.11 bullet, which E4 replaces |
| artifact | `paper/draft.md` abstract, §4.2, §4.3, §5.1.3, §5.1.4, §5.11 |
| accept | yes (E4 closes the remaining two hits) |

## E8 · References verified; two fixes (accepted)

| | |
|---|---|
| unit | verify all 24 entries against Crossref / PMLR / NeurIPS proceedings / publisher / arXiv |
| command | `api.crossref.org/works/<DOI>` for 1,2,3,4,7,8,10,11,18,19,20,21,22,23,24; `proceedings.mlr.press/v70/amos17a.html`; NeurIPS 2019 metadata JSON; arXiv 2104.12225 and 2106.04350; ASRL dataset page; book catalogue records for 12–15,17 |
| result | two corrections: **entry 5 OptNet pages 146–155 → 136–145** (PMLR page), **entry 6 NeurIPS 2019 pages added 9562–9574** (proceedings metadata) and its `[verify]` removed. Entry 1 (T-RO 41:4120–4138), 2 (38(5):3314–3335), 3 (NeurIPS 38:120073–120104), 4 (NeurIPS 35:3801–3818), 7 (CVPR 2020:8097–8106), 8 (CVPR 2022:2771–2780), 10, 11, 18–24 all confirmed. Dataset refs present: BOP (19), LINEMOD (20), LINEMOD-Occlusion (21), YCB-Video/PoseCNN (22), KITTI (23), UTIAS (24 + official URL). `[verify]` count 2 → 0. |
| artifact | `paper/draft.md` References; `theory/proofs.md` untouched |
| accept | yes |

## E2 · SDPRLayers Table 5.9b replaced at full n (accepted)

| | |
|---|---|
| unit | replace the n=12/n=3 SDPR table and paragraph with the final `baseline_sdpr_final.md` at synthetic n=50 for every p and real n=90/72/84 |
| command | read `wt-final-p2-sdpr/.../results/baseline_sdpr_final.{json,md}` + per-cell JSONs; copy them into this worktree; recompute every median/rate from the JSON records |
| result | synthetic hard/slack unchanged in the aggregate (0.00 mm, 4.17°, 10.63 mm; 92–100% feasible; 30–52 / 63–2259 ms), but p=16 is now n=50 (was 12). **Real data reverses the earlier n=3 reading**: SDPR-hard is feasible only 14/12/10% with 9.19/12.70/11.28 mm penetration; SDPR-slack 42/32/46% feasible at 2.40/1.96/0.28 mm penetration, sound cert 12/3/2%; ALM 100% feasible, cert 24.4/15.3/13.1%. Text and table state this; the old "hard relaxation reaches 0.00 mm and 96–100% feasible" sentence is now scoped to synthetic only. |
| artifact | `paper/draft.md` §5.7 + Table 5.9b; `experiments/results/baseline_sdpr_final.*`, `baseline_sdpr_{synth_p8,synth_p16,synth_p32,synth_p64,lm,lmo,ycbv}.json`, `baseline_sdpr_status.*`, `fig_sdpr_*.png` |
| accept | yes |

## E3 · dQP and Theseus replaced with final runs (accepted)

| | |
|---|---|
| unit | replace the n=2 dQP paragraph and the "Theseus not run" statement with the final `baseline_dqp.md` (p=8/16/32, n=50) and the official Theseus λ-Pareto (`baseline_theseus.md`, p=8/16/32, n=30) |
| command | read `wt-final-p2-dqp/.../results/{baseline_dqp,baseline_theseus}.{json,md}` after the final runs (logs `baseline_dqp_final.log`, `baseline_theseus_final.log`); copy them into this worktree |
| result | dQP is no longer a weak baseline: 100/98/98% feasible and certified at p=8/16/32, 0.00 mm / 4.17° / 10.63–10.85 mm, 127.6/108.6/102.9 ms (faster than SCP 154/128/132 and ALM 233/226/200); real 30-instance subset 100% feasible, cert 37/57/20% (base 17/20/10%); gradient check max rel 3.1e-9/3.0e-10/8.0e-9/4.2e-7 on random QPs, CVXPYLayers ≤1.9e-5, degenerate on 2/8 PnP seeds. Theseus runs via an isolated scikit-sparse/CHOLMOD overlay: no finite λ reaches 100% feasibility (best 93%, quad λ=1e5 p=32), cert 7–13% at λ≥1e4 vs ALM 100%, Huber 93–97% feasible but 58–85 mm translation and 0% cert. The old "CHOLMOD missing, not run" paragraph is deleted. |
| artifact | `paper/draft.md` §5.7 (dQP paragraph + Table 5.9d Theseus), §5.6 SCP timing reconciled (128–154 vs 200–233 ms), §5.11 Theseus bullet, references header; `experiments/results/{baseline_dqp,baseline_theseus}.{json,md}` |
| accept | yes |

## E5b · KITTI rigorous certificate subset added (accepted)

| | |
|---|---|
| unit | add the sound P2-Cert column to Table 5.16 where it was measured |
| command | read `wt-final-p2-real/.../results/kitti_cert.{json,md}` (full run: seqs 00/05/07, settings clean+bias6, cfg depth+tilt, base cuts, n=126) |
| result | 63/63 and 63/63 certify with base cuts; Schur gaps 4.9e-9–6.2e-8; median SDP 0.09–0.10 s. Table gains a `cert rigorous` column (100% (63/63) on the two cells, `–` elsewhere); reading (iv) rewritten; §5.11 bullet changed from "not computed on KITTI/UTIAS" to "partial on localisation" (126/126 on KITTI clean+bias6, not on the outlier cells or UTIAS) |
| artifact | `paper/draft.md` §5.9 Table 5.16 + reading (iv), §5.11; `experiments/results/kitti_cert.{json,md}` |
| accept | yes |

## E9 · Figure panel (d) regenerated with the final SDPR data (accepted)

| | |
|---|---|
| unit | regenerate `fig_p2_overview.png` now that `results/baseline_sdpr_synth_p{8,16,32,64}.json` are in this worktree; fix log-axis tick labels |
| command | `nice -n 10 /home/syk/miniconda3/bin/python make_p2_overview.py` |
| result | panel (d) now has four groups (p=8/16/32/64) with hard 30/33/41/52 ms vs slack 63/178/523/2259 ms; y ticks printed as plain numbers (no `6×10²` minor labels); source line unchanged and still points to `results/baseline_sdpr_synth_p*.json`; determinism unchanged (no RNG in the figure) |
| artifact | `experiments/results/fig_p2_overview.png` (322,325 bytes), `experiments/results/make_p2_overview.py` |
| accept | yes |
