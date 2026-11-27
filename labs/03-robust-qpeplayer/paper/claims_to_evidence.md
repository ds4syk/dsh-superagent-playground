# P3 claims → evidence map (final paper consistency, 2026-11-28)

Scope: `labs/03-robust-qpeplayer/paper/draft.md` v0.4. Every headline claim in the
abstract, §1 contributions, §5, and §7 must point to a committed JSON/MD artifact
and a regeneration command. `Theory` rows point to the proof documents, not to an
experiment. `Blocked` rows say exactly what is missing; they are not results.

Notation: `$PY=/home/syk/miniconda3/envs/WM/bin/python`; commands are run from
`labs/03-robust-qpeplayer/experiments/` unless noted. Slurm is used on the lab
server; a local run is acceptable only when the command is explicitly marked
"local".

## 1. Theory claims

| # | Claim (paper location) | Status | Artifact / exact location | Regeneration / check |
|---|---|---|---|---|
| T1 | Exact BR dual for GM; optimal weight $w^*=\rho'_\mu(e)$ (§4.1, Lemma 1) | Theory | `theory/proofs.md` (Lemma 1), `theory/derivations.md` | proof review; scalar check in `synthetic/verify_smoke.py` |
| T2 | GNC-QPEP forward is monotone block descent; fixed point is stationary (§4.1, Thm 1) | Theory | `theory/proofs.md` (Thm 1) | proof review |
| T3 | Curvature correction = Schur complement of the BR weight block (§4.2, Lemma 3) | Theory | `theory/proofs.md` (Lemma 3) | proof review |
| T4 | Exact Riemannian Hessian on S³×R³ with $-\lambda_{\rm rob}\operatorname{diag}(I_3,0_3)$; omission gives 25.9% vs FD, corrected 9.9e-8 (§4.2, Lemma 4) | Theory + numerical check | `verify/riemann_hessian_check_output.txt`, `verify/riemann_hessian_check.py`; `synthetic/results/gradient_table.json` → `hessian_check` | `$PY verify/riemann_hessian_check.py` (not edited in this pass) |
| T5 | 1D sign flip at $e>c^2/3$ (§4.4, Prop. 1) | Theory + numerical check | `theory/proofs.md` (Prop. 1); `verify/ift_1d_signflip_output.txt` | `$PY verify/ift_1d_signflip.py` (not edited in this pass) |
| T6 | Existence of the two-observation sign-flip configuration (§4.4, Prop. 2) | Theory | `theory/proofs.md` (Prop. 2) | proof review |
| T7 | Per-observation $O(\lVert\mathbf r_i\rVert^{-4})$ bound, distinct from $O(\lVert\mathbf r_i\rVert^{-3})$ forward bias (§4.4, Prop. 3) | Theory | `theory/proofs.md` (Prop. 3) | proof review; no numerical artifact claimed |

## 2. Measured claims

| # | Claim (paper location) | Status | Artifact | Regeneration command |
|---|---|---|---|---|
| M1 | Exact IFT vs central FD: 4.9e-8 (inlier), 3.1e-8 (boundary), 1.25e-4 (gross, FD-limited), cosine 1.000000 (§5.3) | Measured | `synthetic/results/gradient_table.json` → `bins` | `$PY run_gradient.py` (in `synthetic/`) |
| M2 | Frozen-weight naive error 0.212 (inlier), 1.07 (boundary); boundary cosine 0.234, min −0.358; no-λ 9.0e-4–9.8e-4 (§5.3) | Measured | same `gradient_table.json` → `bins` | same |
| M3 | Radial sign flip in 122/122 boundary and 64/64 gross observations; pose-translation layer flips in 100% of the 6D sample (§5.3) | Measured (6D numerical, not theorem) | `gradient_table.json` → `sign_flips` | same |
| M4 | GNC-GM equals fixed-1000 RANSAC success through 70% random outliers; IRLS 98% at 70%; Kabsch collapses at 10% (§5.2) | Measured | `synthetic/results/main_table_extended.json` → `rows` (`outlier_type=random`) | `$PY run_main_extended.py` |
| M5 | GNC 31.6–32.5× faster than fixed RANSAC for $p\ge10\%$; adaptive RANSAC comparison 3.9× slower at 10%, parity at 40%, 8.3× faster at 70% (§5.2) | Measured | same `main_table_extended.json` → `rows` time fields | same |
| M6 | RANSAC budget: 90% outliers → RANSAC-1000 46%, adaptive 46%, theory 52.4%; 80% → 100% (§5.2 note / README §3.4) | Measured | `main_table_extended.json` → `budget` | same |
| M7 | Coherent majority: at 50% GNC 52%, RANSAC 50%; at 55% GNC 2.5%, RANSAC 0%; at 60% all methods return the outlier transform (100%) (§5.2) | Measured | `ablations/breakdown/results/breakdown.json` → `rows` (coherent) | `$PY run_breakdown.py` (in `ablations/breakdown/`) |
| M8 | 80–90% random: 90% RANSAC 51%, GNC 62%, IRLS 32%; 80% GNC 89%, RANSAC 100% (§5.2) | Measured | same `breakdown.json` | same |
| M9 | Boundary outliers: forward success 100% for all; 70% GNC 0.27° vs RANSAC 0.62° vs Kabsch 0.33° (§5.2) | Measured | `main_table_extended.json` (`outlier_type=boundary`, `p=0.7`) | `$PY run_main_extended.py` |
| M10 | E2E boundary refinement: p=0.5/lr 1e-3 exact loss 1.90e-6 vs naive 1.66e-5, paired median ratio 0.071 (14×), exact 10/10; p=0.7 ratio 0.059 (17×), exact 9/10; all-points 4–9/10 (§5.4) | Measured toy diagnostic | `synthetic/results/e2e_table.json` → `aggregate`, `paired` | `$PY run_e2e.py` (in `synthetic/`) |
| M11 | Torch autograd layer: gradcheck exact PASS at N=6/8/10 (max abs 6–8e-10), naive FAIL (rel 1.46–1.84); port fidelity 1e-16/3e-16 (§5.4) | Measured | `e2e_torch/results/gradcheck.json`, `e2e_torch/results/e2e_torch.json` | `$PY gradcheck_layer.py`; `$PY learned_refine.py` (in `e2e_torch/`) |
| M12 | Learned refinement 0.635°→0.068° exact vs 0.106° naive, 4/5 seeds, median ratio 0.56; held-out protocol negative (§5.4) | Measured toy diagnostic | `e2e_torch/results/e2e_torch.json` → `protocols`; `e2e_torch/results/e2e_torch.md` | same |
| M13 | ModelNet40/10: GNC 100% at 0–70%, 1.7–1.9 ms vs 53.5–61.5 ms (30–32.5×); robust-method deviation ≤2 pp / ≤0.038° (§5.5) | Measured | `real/modelnet/results/main_table.json`, `real/modelnet10/results/main_table.json`, `real/modelnet10/results/meta.json` | `$PY run_modelnet.py`; `$PY run_modelnet10.py` |
| M14 | 3DMatch: all 8 official test scenes; pooled random GNC/RANSAC 100% through 70%, IRLS 99%; GNC 2.26 ms vs 62.6 ms at 70% (27.7×); boundary 0.27° vs 0.52°; coherent 50% GNC 46% vs RANSAC 3% (§5.5) | Measured | `real/threedmatch_full/results/full_scenes.md`; per-scene JSON `real/threedmatch_full/results/{manifest,verify_gt_direction_full,pairs_*}.json`; `extra_scenes.json` for the 4-scene subset | `$PY run_full_scenes.py`; `$PY verify_gt_direction_full.py` (in `real/threedmatch_full/`) |
| M15 | 3DMatch overlap study: 24/21/20 pairs, clean rotation 0.52/0.24/0.18°, p=0.7 success 100% in all bins, speedup 27.7–30.4× for p≥30% (§5.5) | Measured | `real/threedmatch_full/results/overlap_bins.md`, `overlap_bins.json`, `overlap_dependence.json` | `$PY run_overlap_bins.py` (in `real/threedmatch_full/`) |
| M16 | FPFH front-end: 80.0%/67.8% natural inliers; GNC 100/100, RANSAC-1000 100/97, Kabsch 100/83 (§5.5) | Measured | `real/threedmatch/results/descriptor.json` | `$PY run_descriptor.py` (Open3D venv) |
| M17 | KITTI raw 2 drives: random 100% through 70%, GNC 2.7–2.8 ms vs 61–64 ms (~22×); boundary GNC 0.07–0.09° vs RANSAC 0.14–0.16°; drive 0002 random/boundary identical, coherent 50% GNC 40 vs 35 / IRLS 30 vs 45 (§5.5) | Measured | `real/kitti/results/kitti.json`, `real/kitti2/results/kitti2.json`, `real/kitti2/results/kitti2_compare.json` | `$PY run_kitti.py`; `$PY run_kitti2.py`; `$PY compare_kitti2.py` |
| M18 | KITTI odometry 00/05/07: 18/18 direction checks; random 100% through 70%; pooled 2.8 vs 73.7 ms (26.3×); coherent 50% GNC 53/IRLS 38/RANSAC 0; boundary 0.067° vs 0.143°; 171 motion cells GNC never below RANSAC (§5.5) | Measured | `real/kitti_odom/results/kitti_odom.json`, `verify_kitti_odom.json`, `raw_drive_motion.json` | `$PY run_kitti_odom.py`; `$PY verify_kitti_odom.py` |
| M19 | In-The-Dark: composed official GT; reprojection 0.65/0.52 px vs 9.5–31.3 px alternatives; 79.4%/88.5% inliers; GNC 0.078°/0.020 m and 0.092°/0.010 m; 100% @0.1 m; 3.5–4.1 s/run; both daylight (§5.5) | Measured protocol on 2 daylight runs | `real/inthedark/results/inthedark_dense.json`, `inthedark_run12_dense.json`, `verify_run_000002.json`, `verify_run_000012.json`, `inthedark_main.md`, `README.md` | `$PY run_inthedark.py`; `$PY verify_inthedark.py` (in `real/inthedark/`) |
| M20 | KITTI stereo fallback: 17.8% natural inliers, GNC 0.052°, translation 0.70 m vs 1.92 m, 0% @0.1/0.25 m, 19.3× faster than RANSAC-1000; depth ratio 1.000–1.061 (§5.5) | Measured negative result | `real/kitti_stereo/results/kitti_stereo_main.json`, `kitti_stereo_main.md`, `real/kitti_stereo/README.md` (depth check + 0.63–0.73 ratio) | `$PY run_kitti_stereo.py` (in `real/kitti_stereo/`) |
| M21 | Batch GNC: B=32/N=100 56.2→8.5 ms (6.60×), B=8 2.95×/3.27×, B=1 0.54× (§5.6) | Measured | `systems/results/batch_gnc.json`, `batch_gnc.md` | `$PY run_batch.py --mode cpu` (in `systems/`) |
| M22 | GPU float64 eager: launch floor 42.4–45.7 ms; loses at B≤8 (0.04–0.57×), tied at B=32/N=100 (1.33×), wins at B=32/N=500 (2.23×) and B=128 (4.96×/8.24×) (§5.6) | Measured on RTX 5090 (earlier allocation) | `systems/results/batch_gnc.json` → `gpu` | `$PY run_batch.py --mode gpu` under Slurm |
| M23 | Davenport/Horn 4×4 inner: max $\lVert R_d-R_k\rVert_F=1.07e-14$, cost excess 9.4e-16; p50 0.94/1.02/1.28×, geomean 1.07× (§5.6) | Measured | `inner/results/inner_solver.json`, `inner_solver.md` | `$PY run_inner_solver.py` (in `inner/`) |
| M24 | Theseus-GM 0.2.3: Jacobian FD 1.09e-10, cost exact; GT-init 100%, random-init 100/80/55/25%, 358–670 ms; GNC 100% at 1.7 ms; Wilson overlap at p=0.3 (§5.7) | Measured on synthetic only | `baselines/results/theseus.json`, `theseus_selfcheck.json`, `baselines.json`, `baselines.md` | `THESEUS_TRIALS=20 $PY run_theseus.py` (Theseus venv) |
| M25 | SDPRLayers+SCS: 0/208 exceptions, 8.5–13.4 ms; succeeds p=0, fails p≥30%; 80 paired trials max ΔR 1.5e-5°, Δt 1.3e-9 m at W=I (§5.7) | Measured forward only; not a robust baseline | `baselines/results/sdpr_forward.json`, `sdpr_kabsch_equiv.json`, `baselines.json` | `$PY run_sdpr_forward.py` (sdpr env); `$PY check_sdpr_kabsch_equivalence.py` |
| M26 | LO-RANSAC ties accuracy (≤0.006°) but is 1.03–1.52× slower than adaptive RANSAC (§5.8) | Measured | `ablations/ransac_lo/results/ransac_lo.json`, `ransac_lo.md` | `$PY run_ransac_lo.py` |
| M27 | Kernel ablation: GM lowest boundary median rotation 0.199/0.229/0.340° vs Cauchy/Welsch (§5.8) | Measured | `ablations/kernels/results/kernel_ablation.json` | `$PY run_kernel_ablation.py` |
| M28 | GNC schedule: all 24 configs 100%, same median rotation; γ=2.0 1.04–1.79× faster, γ=1.2 1.47–2.49× slower (§5.8) | Measured | `ablations/gnc_schedule/results/gnc_schedule.json` | `$PY run_gnc_schedule.py` |
| M29 | Anisotropic $W_i$: W=I matches Kabsch 5.3e-13; scalar inner not exact (ratio 1.035–1.088, max 3.80); no accuracy gain under isotropic noise (Δrot +0.055..+0.124°, 92% at p=0.7); 4.9–23× slower (§5.8) | Measured correctness/negative result | `ablations/anisotropic/results/anisotropic.json`, `anisotropic.md` | `$PY run_anisotropic.py` |
| M30 | Wilson 95% bounds: 96.3% (100 trials), 97.7% (160 3DMatch-8), 94.0% (60 odom), 91.2% (40 KITTI-2) (§5.1) | Measured/derived from trials | `analysis/wilson.json`, `analysis/wilson.md` | `$PY analysis_wilson.py` |

## 3. Blocked / future (never to be read as results)

| # | Item | Status | Artifact | Why blocked / next step |
|---|---|---|---|---|
| B1 | GPU p50/p95 and CUDA-graph capture of the 4×4 inner | **Blocked** | `qpep_gpu/results/qpep_gpu.json` (`meta.device=cpu`), `qpep_gpu/results/qpep_gpu.md`, `qpep_gpu/results/logs/slurm_gpu_3184_state.txt` | Job 3184 PENDING (`Reason=Resources`, `StartTime=2026-11-28T21:23:07`) while 3151/3180/3181 held the 3 GPUs; cancelled after 5 min; no graph captured, so **no CUDA-graph error is claimed**. Re-run `sbatch sbatch_probe_phase4.sbatch` when a GPU is free. |
| B2 | General 40-root QPEP polynomial solver, root completeness/failure rate, GPU batching of it | **Not implemented** | `qpep_gpu/qpep_scope.md`, §5.1/§5.6/§5.9 | All 3D-3D tables use scalar weights; the QPEP stationary problem reduces to the 4×4 eigenproblem. Implementing LibQPEP-style Gröbner elimination is a separate project. |
| B3 | Real-data external baselines (SDPR+RANSAC, Theseus on 3DMatch/KITTI, BPnP/Pyatov/POLI on 3D-3D) | **Not measured / out of scope** | `baselines/results/availability.md`, §5.7/§5.9 | BPnP is 3D–2D PnP, Pyatov is two-view 2D–2D, POLI is a geometry estimator; no 3D-3D pose code was found. SDPR needs non-identity $W_i$ or RANSAC to be a robust baseline. |
| B4 | Night-time / glare / trained In-The-Dark | **Not measured** | `real/inthedark/README.md`, `inthedark_main.md` §Limitations | Both measured runs are daylight (mean intensity 133/131); front-end is hand-written SIFT+SGBM; no training. |
| B5 | Learned descriptors (FCGF/Predator) and the 54-scene 3DMatch training corpus | **Not measured** | `real/threedmatch_full/README.md` caveats | Only hand-crafted FPFH; main tables use GT-NN oracle matches. |
| B6 | 6D sign-flip theorem | **Not proved** | §4.4, §5.3 | Only the 1D proposition is a theorem; 6D is a numerical observation (122/122, 64/64). |
| B7 | Global optimality / breakdown point / adversarial robustness | **Not claimed** | §4.6, §6 | GNC is local; coherent majority defeats every method (60–70% all 0%). |

## 4. Reference verification record (2026-11-28)

All 16 references were checked against Crossref/arXiv/publisher pages. The
following corrections were applied to `paper/draft.md`:

| Ref | Before | After | Primary source |
|---|---|---|---|
| [4] GNC | DOI `10.1109/LRA.2020.2969178` | DOI `10.1109/LRA.2020.2965893` | Crossref `api.crossref.org/works/10.1109/LRA.2020.2965893` |
| [5] TEASER | "Fast and Accurate 3D Registration with Extreme Outlier Rates", CVPR 2020 | "TEASER: Fast and Certifiable Point Cloud Registration", IEEE T-RO 37(2):314–333, 2021, DOI `10.1109/TRO.2020.3033695` | Crossref + MIT DSpace `hdl.handle.net/1721.1/134163` |
| [7] BPnP | no pages/DOI | pp. 8097–8106, DOI `10.1109/CVPR42600.2020.00812` | Crossref |
| [8] EPro-PnP | no pages/DOI | pp. 2771–2780, DOI `10.1109/CVPR52688.2022.00280` | Crossref |
| [9] Pyatov et al. | no pages/DOI | pp. 11924–11931, DOI `10.1109/IROS58592.2024.10801617` | Crossref |
| [10] POLI | arXiv only | added DOI `10.48550/arXiv.2605.10456` | arXiv:2605.10456 (title/authors match) |
| [12] RANSAC | no DOI | added DOI `10.1145/358669.358692`; now cited in §1 | Crossref |
| [14] JFB | wrong authors "S. Fung, H. Heaton, Q. Li, D. Wu, K. He, Y. Zhang" | S. W. Fung, H. Heaton, Q. Li, D. McKenzie, S. Osher, W. Yin; AAAI 36(6):6648–6656, DOI `10.1609/aaai.v36i6.20619` | AAAI OJS 20619 |
| [16] Horn | missing | added Horn 1987, JOSA A 4(4):629–642, DOI `10.1364/JOSAA.4.000629`; cited in §5.6 | Crossref |

Unchanged and verified: [1] SDPRLayers (T-RO 41:4120–4138, 2025, DOI
`10.1109/TRO.2025.3578228`), [2] QPEP (T-RO 38(5):3314–3335, 2022, DOI
`10.1109/TRO.2022.3155880`), [3] Black–Rangarajan (IJCV 19(1):57–91, 1996,
DOI `10.1007/BF00131148`), [6] Theseus (NeurIPS 2022, arXiv:2207.09442),
[11] Geman–McClure (Proc. ASA 1985, pp. 12–18), [13] Absil et al. (Princeton,
2008), [15] UTIAS dataset (https URL reachable). No DOI is guessed: [6], [11],
[13] have no DOI and are cited without one.

## 5. Audit status

`experiments/repro_audit.py` (run 2026-11-28) parses **60 result JSON** entries,
checks **13 raw JSONL** line counts, hashes the committed key results, and runs
exact-text checks for the blocked-GPU record and the daylight In-The-Dark note.
Gitignored re-generatable artifacts (`full_scenes.json`, `*raw.jsonl`) are
reported as `absent-ignored` and do not fail the audit; `--strict` requires
them. Overall: **PASS**. See `experiments/REPRO_AUDIT.md`.

Audit command:

```bash
cd labs/03-robust-qpeplayer/experiments
/home/syk/miniconda3/envs/WM/bin/python repro_audit.py
```
