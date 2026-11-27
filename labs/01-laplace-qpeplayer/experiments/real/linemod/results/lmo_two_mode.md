# LMO extension of the real two-mode experiment (eggbox 10 / glue 11)

**Date:** 2026-11-28 · **Data:** `/mnt/storage1/syk/data/bop/lmo` test sequence `000002` · **Frames:** 37 usable of 40 attempted (20 attempts/object) · **Runtime:** 31.9 s.

> **结论先行**
>
> 1. LMO 只有一条 test 序列 `000002`（1214 张图，8 个对象同框）。obj 10/11 都在：GT 实例 1175 / 903 个，可见像素 ≥1200 的候选 1031 / 726 个；按数据可用性等距抽 20+20 帧，37 帧通过 K≥2 物理模式过滤（eggbox 20/20，glue 17/20）。
> 2. 两模结构本身在 LMO 上仍然成立：K_phys=2 在 37/37 帧，中位 ΔJ=1.37e-03，翻转模与 GT 夹角中位 100.5°，错误模 Laplace 权重中位 0.474。
> 3. **但 hard argmin 在 LMO 上明显变差**：旋转误差中位 2.95°（LINEMOD 2.06°）、平移中位 19.5 mm（LINEMOD 6.2 mm）、ADD-S 中位 10.2 mm（LINEMOD 3.7 mm）。2× starts 复算 37/37 帧 K 与 hard 完全不变，排除模式枚举漏解；约化代价的最小点本身在 15/37 帧距 GT >5°（$J_{gt}/J_{\min}$ 中位 1.17），这是可见平面片小（中位 61 mm）叠加深度-模型残差（p50 1.4–9.3 mm）把最小点推离 GT，而不是模式排序错误；逐帧 diagnostic 见 §2–§3。
> 4. Laplace/MC 仍被翻转模拉偏：旋转误差中位 43.7° / 43.5°，ADD-S 中位 19.0 / 19.0 mm；Laplace 旋转误差与 LINEMOD 相当（LMO 43.7° vs LINEMOD 44.2°），ADD-S 偏大（LMO 19.0 mm vs LINEMOD 13.9 mm），方向一致。
> 5. ADD-S AUC(0–0.1 m)：hard 0.862 [0.812, 0.903]，Laplace 0.766，MC 0.766；n=37 时 AUC 只有 ≤37 个台阶，CI 很宽，只作描述性统计。

**Evidence level:** independent reproduction/measurement in this repo; no image front-end, no claim against published BOP LMO pipelines.

## 0. Dataset inventory and the occlusion adaptation

| item | value |
|-|-|
| LMO root | `/mnt/storage1/syk/data/bop/lmo` |
| test sequences | 000002 (only one) |
| images in sequence | 1214 |
| object ids in models/ | [1, 5, 6, 8, 9, 10, 11, 12] |
| obj 10 / 11 present | True / True |
| obj 10 / 11 GT instances | 1175 / 903 |
| obj 10 / 11 instances with px_count_visib ≥ 1200 | 1031 / 726 |
| max instances of the same object per image | 1 / 1 |
| unique cam_K / depth_scale | 1 / [1.0] |
| depth image size | [480, 640] |

**Occlusion adaptation.** LMO is LINEMOD-Occlusion: every instance is cut by occluders. The pipeline is unchanged from LINEMOD and uses `mask_visib` (visible mask) as the only source of depth pixels — the amodal `mask/` is *not* used, because occluded pixels have no object depth (the depth there belongs to the occluder). The consequence is documented rather than compensated: the RANSAC plane is fitted to the *visible* part, so the dominant planar patch is often much smaller than in LINEMOD (median in-plane extent 61 mm; n_plane 186–694 vs 347–558 in LINEMOD). No metric or estimator was changed.

## 1. Pipeline and metrics (same as LINEMOD)

Real depth pixels inside `mask_visib` → bearings $U_i$ (real intrinsics, depth in mm, `depth_scale=1.0`); nearest real mesh vertex → 3-D model point $P_i$; deterministic RANSAC (band 2.5 mm, 900 points, seed 0) selects the dominant planar face; the reduced cost eliminates the translation on the inlier set; modes with any transformed inlier point at $z\le0.05$ m are rejected. `n_starts=384`, $\tau=0.01$. The three estimators are exactly those of `run_pose_error.py`: hard = lowest-$J$ physical mode; Laplace = projected chordal mean with weights $w_k\propto\exp(-J_k/\tau-\tfrac12\log\det H_k)$; MC = 400 mode samples from $w$, 20 seeds, per-frame mean.

Metrics vs BOP GT: rotation error (geodesic, deg), translation error (m), **ADD-S** = mean over 1000 mesh vertices of the nearest-neighbour distance to the full GT mesh (KD-tree), in mm; **ADD** is the same-index mean (reference only, both objects have a ~180° discrete symmetry); **ADD-S AUC** = area under the recall-vs-threshold curve over 0–0.1 m, normalised by 0.1 m. The primary mesh is `models/`, byte-identical to the LINEMOD meshes (sha256 check: True); `models_eval` ADD-S is a separate secondary column. Using a 1000-vertex target instead of the full mesh adds **+0.86 mm** on average (the LINEMOD run reported +1.1 mm); the primary table uses the dense target.

## 2. Per-frame table

Mode structure and data regime (why LMO is harder):

| frame | obj | visib | px_visib | n_valid | n_plane | plane ext (mm) | depth p50 (mm) | depth core p50 (mm) | K_phys | K_all | ΔJ | flip° | w_wrong | closest mode° | J_gt/J_min |
|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|
| 000002/000000/5 | 10 | 0.61 | 1779 | 900 | 252 | 48 | 2.2 | 2.1 | 2 | 4 | 1.10e-04 | 18.3 | 0.532 | 18.3 | 1.17 |
| 000002/000057/5 | 10 | 0.59 | 1533 | 900 | 231 | 61 | 5.8 | 5.5 | 2 | 4 | 3.06e-04 | 32.8 | 0.550 | 8.1 | 1.03 |
| 000002/000131/5 | 10 | 0.74 | 2204 | 900 | 252 | 59 | 3.4 | 3.4 | 2 | 4 | 2.10e-03 | 109.6 | 0.462 | 4.1 | 1.21 |
| 000002/000205/5 | 10 | 0.63 | 3269 | 900 | 214 | 51 | 6.7 | 7.1 | 2 | 4 | 6.19e-04 | 48.7 | 0.512 | 22.6 | 1.20 |
| 000002/000275/5 | 10 | 0.75 | 3010 | 900 | 211 | 53 | 4.7 | 4.7 | 2 | 4 | 4.04e-04 | 45.0 | 0.524 | 16.8 | 1.18 |
| 000002/000346/4 | 10 | 0.31 | 2488 | 900 | 393 | 25 | 3.2 | 3.2 | 2 | 4 | 1.99e-04 | 65.4 | 0.599 | 17.5 | 1.38 |
| 000002/000408/5 | 10 | 0.98 | 3297 | 900 | 475 | 63 | 3.9 | 3.9 | 2 | 4 | 2.88e-03 | 98.9 | 0.435 | 1.1 | 1.04 |
| 000002/000468/5 | 10 | 0.70 | 1289 | 900 | 225 | 44 | 2.9 | 2.7 | 2 | 4 | 1.51e-04 | 48.6 | 0.528 | 9.0 | 1.16 |
| 000002/000553/4 | 10 | 0.78 | 3043 | 900 | 351 | 68 | 4.2 | 4.1 | 2 | 4 | 3.15e-03 | 124.0 | 0.427 | 2.8 | 1.17 |
| 000002/000613/5 | 10 | 0.98 | 3240 | 900 | 502 | 63 | 2.6 | 2.7 | 2 | 4 | 1.37e-03 | 88.1 | 0.471 | 1.1 | 1.07 |
| 000002/000685/5 | 10 | 0.81 | 4594 | 900 | 404 | 58 | 2.9 | 3.0 | 2 | 4 | 1.80e-03 | 109.3 | 0.454 | 0.0 | 1.00 |
| 000002/000739/4 | 10 | 0.97 | 2378 | 900 | 197 | 63 | 2.2 | 2.1 | 2 | 4 | 6.74e-04 | 118.7 | 0.489 | 5.0 | 1.28 |
| 000002/000794/5 | 10 | 0.80 | 1667 | 900 | 585 | 61 | 6.0 | 5.8 | 2 | 4 | 3.54e-03 | 103.2 | 0.417 | 1.7 | 1.06 |
| 000002/000848/5 | 10 | 0.83 | 1816 | 900 | 536 | 61 | 4.0 | 3.8 | 2 | 4 | 3.07e-03 | 114.2 | 0.427 | 0.8 | 1.04 |
| 000002/000913/5 | 10 | 0.97 | 5574 | 900 | 429 | 63 | 1.6 | 1.6 | 2 | 4 | 2.29e-03 | 107.4 | 0.449 | 1.8 | 1.33 |
| 000002/000981/5 | 10 | 0.21 | 1942 | 900 | 543 | 55 | 1.5 | 1.5 | 2 | 4 | 1.70e-03 | 125.2 | 0.471 | 2.2 | 1.13 |
| 000002/001040/5 | 10 | 0.91 | 2135 | 900 | 355 | 64 | 3.3 | 3.1 | 2 | 4 | 3.09e-03 | 135.6 | 0.426 | 1.5 | 1.08 |
| 000002/001095/5 | 10 | 0.94 | 3254 | 900 | 186 | 60 | 3.9 | 4.2 | 2 | 4 | 1.15e-03 | 111.8 | 0.485 | 6.8 | 1.33 |
| 000002/001152/5 | 10 | 0.93 | 4471 | 900 | 395 | 62 | 3.0 | 3.1 | 2 | 4 | 1.24e-03 | 116.2 | 0.474 | 0.0 | 1.00 |
| 000002/001213/5 | 10 | 0.88 | 3149 | 900 | 391 | 62 | 3.7 | 3.6 | 2 | 4 | 3.63e-03 | 128.1 | 0.413 | 2.2 | 1.25 |
| 000002/000002/6 | 11 | 0.70 | 1261 | 900 | 467 | 56 | 2.0 | 2.0 | 2 | 4 | 8.93e-04 | 87.4 | 0.492 | 2.2 | 1.07 |
| 000002/000054/6 | 11 | 0.78 | 1533 | 900 | 584 | 70 | 3.0 | 2.8 | 2 | 4 | 1.02e-03 | 61.5 | 0.515 | 10.1 | 1.44 |
| 000002/000128/6 | 11 | 0.78 | 1861 | 900 | 509 | 69 | 1.9 | 1.9 | 2 | 4 | 1.67e-03 | 86.7 | 0.474 | 2.8 | 1.14 |
| 000002/000180/6 | 11 | 0.92 | 1355 | 900 | 446 | 56 | 1.5 | 1.5 | 2 | 4 | 2.58e-03 | 109.9 | 0.456 | 2.5 | 1.06 |
| 000002/000269/6 | 11 | 0.80 | 3196 | 900 | 501 | 63 | 1.4 | 1.3 | 2 | 4 | 2.49e-03 | 97.0 | 0.460 | 0.0 | 1.02 |
| 000002/000312/6 | 11 | 0.92 | 3481 | 900 | 487 | 70 | 1.9 | 1.9 | 2 | 4 | 1.71e-03 | 112.1 | 0.455 | 0.0 | 1.07 |
| 000002/000353/6 | 11 | 0.62 | 2745 | 900 | 552 | 56 | 3.3 | 3.1 | 2 | 4 | 3.23e-04 | 35.2 | 0.548 | 19.7 | 1.28 |
| 000002/000396/6 | 11 | 1.00 | 1942 | 900 | 540 | 66 | 2.3 | 2.4 | 2 | 4 | 2.00e-03 | 116.1 | 0.458 | 1.0 | 1.03 |
| 000002/000435/5 | 11 | 0.97 | 2193 | 900 | 599 | 61 | 4.1 | 4.2 | 2 | 4 | 1.08e-04 | 7.2 | 0.530 | 7.2 | 1.17 |
| 000002/000473/6 | 11 | 1.00 | 3552 | 900 | 467 | 65 | 6.2 | 6.5 | 2 | 4 | 1.33e-04 | 84.1 | 0.486 | 2.3 | 1.06 |
| 000002/000520/6 | 11 | 0.85 | 3704 | 900 | 581 | 62 | 2.4 | 2.3 | 2 | 4 | 3.59e-03 | 124.4 | 0.413 | 4.6 | 1.61 |
| 000002/000591/6 | 11 | 0.62 | 3147 | 900 | 694 | 56 | 4.7 | 4.7 | 2 | 4 | 1.34e-03 | 100.5 | 0.470 | 9.4 | 3.17 |
| 000002/000715/6 | 11 | 0.99 | 2404 | 900 | 629 | 65 | 3.1 | 3.0 | 2 | 4 | 1.67e-03 | 78.8 | 0.475 | 4.7 | 1.32 |
| 000002/000817/6 | 11 | 0.69 | 2089 | 900 | 663 | 56 | 3.7 | 3.6 | 2 | 4 | 6.70e-04 | 118.8 | 0.487 | 9.7 | 2.16 |
| 000002/000864/6 | 11 | 0.97 | 2713 | 900 | 524 | 65 | 3.4 | 3.3 | 2 | 4 | 2.11e-03 | 115.5 | 0.456 | 8.4 | 2.02 |
| 000002/001020/6 | 11 | 0.52 | 1639 | 900 | 600 | 42 | 7.3 | 7.8 | 2 | 4 | 4.31e-05 | 96.8 | 0.496 | 2.9 | 1.04 |
| 000002/001071/6 | 11 | 0.86 | 2519 | 900 | 238 | 105 | 9.3 | 10.5 | 2 | 4 | 3.95e-04 | 14.3 | 0.536 | 14.3 | 1.28 |

Pose metrics per frame (rot deg / trans mm / ADD-S mm / AUC):

| frame | obj | hard rot | Laplace rot | MC rot | hard trans | Laplace trans | MC trans | hard ADD-S | Laplace ADD-S | MC ADD-S | hard AUC | Laplace AUC | MC AUC |
|-|-|-|-|-|-|-|-|-|-|-|-|-|-|
| 000002/000000/5 | 10 | 20.22 | 5.53 | 5.58 | 70.8 | 62.8 | 62.8 | 29.2 | 25.1 | 25.1 | 0.708 | 0.749 | 0.749 |
| 000002/000057/5 | 10 | 8.10 | 14.89 | 15.30 | 80.6 | 72.8 | 72.7 | 33.5 | 29.4 | 29.3 | 0.665 | 0.706 | 0.707 |
| 000002/000131/5 | 10 | 4.07 | 47.79 | 47.00 | 72.2 | 98.9 | 98.6 | 29.9 | 47.6 | 47.4 | 0.701 | 0.524 | 0.526 |
| 000002/000205/5 | 10 | 22.56 | 14.27 | 13.68 | 30.8 | 33.6 | 33.4 | 10.7 | 15.5 | 15.3 | 0.893 | 0.845 | 0.847 |
| 000002/000275/5 | 10 | 16.82 | 19.07 | 18.97 | 32.6 | 38.3 | 38.2 | 11.4 | 16.2 | 16.2 | 0.886 | 0.838 | 0.838 |
| 000002/000346/4 | 10 | 17.51 | 46.45 | 46.10 | 19.0 | 42.4 | 42.1 | 9.7 | 22.4 | 22.2 | 0.903 | 0.776 | 0.778 |
| 000002/000408/5 | 10 | 1.12 | 41.28 | 41.38 | 14.2 | 31.7 | 31.8 | 6.8 | 17.1 | 17.2 | 0.932 | 0.829 | 0.828 |
| 000002/000468/5 | 10 | 8.98 | 23.00 | 22.72 | 123.5 | 122.4 | 122.3 | 62.1 | 62.1 | 62.0 | 0.379 | 0.379 | 0.380 |
| 000002/000553/4 | 10 | 2.75 | 47.84 | 48.96 | 19.5 | 42.7 | 43.0 | 8.9 | 21.6 | 21.9 | 0.911 | 0.784 | 0.781 |
| 000002/000613/5 | 10 | 1.14 | 39.88 | 39.24 | 2.9 | 20.0 | 19.8 | 1.7 | 12.5 | 12.4 | 0.983 | 0.875 | 0.876 |
| 000002/000685/5 | 10 | 0.00 | 47.29 | 47.08 | 2.8 | 24.5 | 24.5 | 1.7 | 14.6 | 14.6 | 0.983 | 0.854 | 0.854 |
| 000002/000739/4 | 10 | 5.02 | 56.68 | 55.74 | 47.0 | 68.2 | 67.9 | 16.3 | 30.0 | 29.8 | 0.837 | 0.700 | 0.702 |
| 000002/000794/5 | 10 | 1.69 | 38.59 | 39.10 | 42.8 | 55.8 | 55.9 | 17.8 | 28.5 | 28.7 | 0.822 | 0.715 | 0.713 |
| 000002/000848/5 | 10 | 0.85 | 43.73 | 42.67 | 31.2 | 46.8 | 46.5 | 12.5 | 23.5 | 23.2 | 0.875 | 0.765 | 0.768 |
| 000002/000913/5 | 10 | 1.85 | 45.37 | 44.66 | 4.3 | 25.3 | 25.1 | 2.4 | 14.1 | 14.0 | 0.976 | 0.859 | 0.860 |
| 000002/000981/5 | 10 | 2.24 | 56.43 | 55.16 | 7.7 | 25.7 | 25.3 | 3.7 | 14.9 | 14.6 | 0.963 | 0.851 | 0.854 |
| 000002/001040/5 | 10 | 1.51 | 47.30 | 48.02 | 33.1 | 54.2 | 54.3 | 12.3 | 26.2 | 26.2 | 0.877 | 0.738 | 0.738 |
| 000002/001095/5 | 10 | 6.81 | 52.13 | 51.15 | 62.9 | 83.3 | 83.0 | 23.1 | 38.9 | 38.6 | 0.769 | 0.611 | 0.614 |
| 000002/001152/5 | 10 | 0.00 | 53.09 | 54.44 | 4.4 | 24.4 | 24.8 | 2.4 | 14.6 | 14.9 | 0.976 | 0.854 | 0.851 |
| 000002/001213/5 | 10 | 2.18 | 44.00 | 44.09 | 19.1 | 43.3 | 43.3 | 8.4 | 20.9 | 20.9 | 0.916 | 0.791 | 0.791 |
| 000002/000002/6 | 11 | 2.19 | 42.90 | 42.60 | 2.9 | 17.7 | 17.6 | 2.0 | 17.0 | 16.9 | 0.980 | 0.830 | 0.831 |
| 000002/000054/6 | 11 | 10.14 | 32.74 | 32.42 | 55.6 | 40.7 | 40.9 | 32.9 | 33.7 | 33.7 | 0.671 | 0.663 | 0.663 |
| 000002/000128/6 | 11 | 2.81 | 40.57 | 40.94 | 13.9 | 12.1 | 12.2 | 7.5 | 19.9 | 20.1 | 0.925 | 0.801 | 0.799 |
| 000002/000180/6 | 11 | 2.53 | 48.56 | 48.06 | 9.9 | 17.7 | 17.6 | 3.9 | 17.6 | 17.4 | 0.961 | 0.824 | 0.826 |
| 000002/000269/6 | 11 | 0.00 | 43.85 | 43.47 | 1.7 | 19.2 | 19.1 | 1.1 | 17.6 | 17.5 | 0.989 | 0.824 | 0.825 |
| 000002/000312/6 | 11 | 0.00 | 48.01 | 46.78 | 0.9 | 14.9 | 14.7 | 1.0 | 17.3 | 16.9 | 0.990 | 0.827 | 0.831 |
| 000002/000353/6 | 11 | 19.67 | 12.64 | 12.55 | 79.8 | 83.8 | 83.8 | 50.9 | 67.9 | 67.8 | 0.491 | 0.321 | 0.322 |
| 000002/000396/6 | 11 | 0.98 | 50.70 | 51.24 | 13.4 | 41.2 | 41.3 | 6.4 | 19.0 | 19.1 | 0.936 | 0.810 | 0.809 |
| 000002/000435/5 | 11 | 33.94 | 19.10 | 18.91 | 34.0 | 29.1 | 29.1 | 13.9 | 12.6 | 12.6 | 0.861 | 0.874 | 0.874 |
| 000002/000473/6 | 11 | 2.35 | 39.96 | 39.43 | 8.2 | 19.1 | 19.0 | 5.3 | 12.7 | 12.5 | 0.947 | 0.873 | 0.875 |
| 000002/000520/6 | 11 | 4.61 | 41.70 | 40.17 | 15.7 | 10.0 | 9.9 | 6.6 | 19.2 | 18.5 | 0.934 | 0.808 | 0.815 |
| 000002/000591/6 | 11 | 9.37 | 43.26 | 44.72 | 12.9 | 18.4 | 19.0 | 5.1 | 16.8 | 17.3 | 0.949 | 0.832 | 0.827 |
| 000002/000715/6 | 11 | 4.69 | 35.59 | 36.29 | 27.9 | 10.1 | 10.1 | 13.7 | 18.8 | 19.0 | 0.863 | 0.812 | 0.810 |
| 000002/000817/6 | 11 | 9.69 | 56.54 | 57.73 | 42.7 | 16.6 | 16.5 | 15.7 | 31.1 | 31.4 | 0.843 | 0.689 | 0.686 |
| 000002/000864/6 | 11 | 8.44 | 50.09 | 49.64 | 30.8 | 13.0 | 13.1 | 11.4 | 26.4 | 26.2 | 0.886 | 0.736 | 0.738 |
| 000002/001020/6 | 11 | 2.95 | 48.25 | 47.70 | 25.6 | 14.3 | 14.0 | 10.2 | 17.6 | 17.4 | 0.898 | 0.824 | 0.826 |
| 000002/001071/6 | 11 | 71.00 | 24.78 | 25.40 | 7.4 | 3.9 | 3.8 | 17.1 | 6.3 | 6.4 | 0.829 | 0.937 | 0.936 |

## 3. Aggregate (median / mean with 95% frame-bootstrap CI)

| metric | hard | Laplace | MC | Laplace − hard | MC − hard |
|-|-|-|-|-|-|
| rotation (deg) | 2.95 [2.19, 8.10] | 43.73 [39.96, 47.30] | 43.47 [39.43, 46.78] | 31.16 [22.94, 38.91] | 31.03 [23.36, 38.00] |
| translation (mm) | 19.5 [13.4, 32.6] | 29.1 [20.0, 42.4] | 29.1 [19.1, 42.1] | 8.0 [2.9, 12.6] | 7.9 [3.1, 12.4] |
| ADD-S (mm) | 10.2 [6.6, 13.7] | 19.0 [17.1, 23.5] | 19.0 [17.2, 23.2] | 9.7 [7.4, 11.8] | 9.6 [7.4, 11.7] |
| ADD-S models_eval (mm) | 10.4 [6.4, 14.0] | 19.9 [18.6, 21.7] | 19.2 [18.4, 21.7] | 9.7 [7.3, 11.9] | 9.6 [7.3, 11.9] |
| ADD (mm, reference) | 26.0 [13.6, 34.9] | 43.7 [38.9, 50.2] | 43.9 [35.7, 51.0] | 18.8 [13.3, 23.9] | 18.7 [13.5, 23.8] |
| ADD-S AUC (0–0.1 m) | 0.898 [0.863, 0.934] | 0.810 [0.765, 0.829] | 0.810 [0.768, 0.828] | -0.097 [-0.117, -0.074] | -0.096 [-0.115, -0.074] |

Mean (same CIs are in the JSON):

| metric | hard | Laplace | MC |
|-|-|-|-|
| rotation (deg) | 8.40 [4.64, 12.79] | 39.56 [34.76, 43.76] | 39.43 [34.92, 43.34] |
| translation (mm) | 29.9 [21.4, 38.8] | 37.8 [29.5, 47.8] | 37.8 [28.8, 46.8] |
| ADD-S (mm) | 13.8 [9.7, 18.0] | 23.4 [19.7, 28.1] | 23.4 [19.4, 27.8] |
| ADD-S models_eval (mm) | 13.8 [9.8, 18.0] | 23.5 [19.9, 28.2] | 23.4 [19.5, 27.8] |
| ADD (mm, reference) | 30.6 [22.5, 39.4] | 49.4 [42.6, 57.3] | 49.3 [42.2, 56.7] |
| ADD-S AUC | 0.862 [0.812, 0.903] | 0.766 [0.718, 0.802] | 0.766 [0.725, 0.805] |

MC seed diagnostics (frame bootstrap vs combined frame+seed):

| metric | between-seed SD | frame CI | combined frame+seed CI |
|-|-|-|-|
| rotation (deg) | 0.542 | [34.916, 43.343] | [35.053, 43.659] |
| translation (mm) | 0.192 | [28.824, 46.833] | [29.887, 47.029] |
| ADD-S (mm) | 0.149 | [19.431, 27.785] | [19.726, 27.671] |

**Post-hoc stratification by plane size** (descriptive; n_plane is a visibility covariate, not an outcome). n_plane ≥ 350: n=28, < 350: n=9. Median hard rotation 2.29° vs 8.98°; median hard ADD-S 7.1 mm vs 23.1 mm. On the larger-patch stratum hard rotation median 2.29° is close to LINEMOD's 2.06°, while ADD-S remains about 1.9× the LINEMOD value because the depth-model residual itself is larger in LMO. This supports the patch-size/depth-noise mechanism rather than a new bug.

## 4. Comparison with LINEMOD (cited from `results/pose_error.json`, not recomputed)

| metric | LINEMOD hard | LMO hard | LINEMOD Laplace | LMO Laplace |
|-|-|-|-|-|
| rotation median (deg) | 2.06 [1.47, 3.49] | 2.95 | 44.24 [33.90, 46.87] | 43.73 |
| translation median (mm) | 6.2 [5.5, 11.1] | 19.5 | 20.2 [16.2, 23.1] | 29.1 |
| ADD-S median (mm) | 3.7 [3.0, 4.3] | 10.2 | 13.9 [10.3, 20.2] | 19.0 |

LINEMOD source: `results/pose_error.json` (date 2026-11-27, 15 frames). That file contains no ADD-S AUC, so no LINEMOD AUC is quoted.

## 5. What the LMO run says (and what it does not)

- **The two-mode structure is not LINEMOD-specific.** K_phys=2 in 37/37 usable LMO frames; median ΔJ=1.37e-03; median wrong-mode weight 0.474.
- **Hard argmin degrades under occlusion, but mostly for one reason.** Median rotation error 2.95° vs 2.06° on LINEMOD and ADD-S 10.2 mm vs 3.7 mm. Hard argmin returns the physical mode closest to GT in 34/37 frames, so mode *ranking* is usually right; but that closest mode is itself >5° from GT in 15/37 frames and the reduced cost at the GT rotation (translation re-eliminated) exceeds its minimum by a median factor 1.17. The error is therefore mostly a displaced cost minimum driven by the small visible patch and the depth-model residual, not a failure of the two-mode enumeration.
- **Mode enumeration is not the explanation.** `find_modes` was re-run with 2× starts (768) on the identical reduced costs: neither K_all/K_phys nor the hard argmin changed in 37/37 frames (max rotation difference 8.6e-13 deg).
- **Laplace/MC inherit the flip.** With two near-tied modes ~101° apart, the posterior mean sits between them; the paired bias vs hard is +31.2° [22.9, 38.9] and +9.7 mm ADD-S. This reproduces the LINEMOD direction.
- **ADD-S AUC is computed but coarse.** The recall curve over 0–0.1 m has ≤37 steps; the frame-bootstrap CI spans [0.812, 0.903] for hard. It is reported because the task asked for it, not as a leaderboard-comparable number.
- **Not a detector claim.** Correspondences still use the GT pose; this measures estimator behaviour on real cost functions. No front-end, no appearance model, no comparison with published BOP LMO results.

## 6. Caveats specific to LMO

1. **Depth-model residual has a heavier tail than LINEMOD.** Per-frame p50 3.3 mm (1.4–9.3 mm; LINEMOD 2–5 mm), p90 up to 56.7 mm. A mask-boundary split gives median core p50 3.1 mm, so boundary mixed pixels are *not* the explanation; the residual is in the depth itself. A 2-D shift spot check on 6 frames found best shifts up to 4 px, and for the worst frame (000002/001071/6, hard 71°) the best-shift residual is 8.6 mm vs 9.3 mm at zero shift, i.e. the shift does not explain it. This is a data-quality fact, not a pipeline change, and it is the main driver of the degraded hard accuracy.
2. **Selection covers moderate-to-good visibility only.** The 1200-visible-pixel threshold is a data-availability proxy for being able to sample 900 valid depth pixels; selected frames have visib_fract median 0.81 (range 0.21–1.00). Heavily occluded instances are under-represented because they often cannot supply 900 valid depth pixels (2 of the 20 glue attempts), so the LMO table is an *optimistic* view of the hardest LMO occlusion for this pipeline.
3. **Meshes.** Primary ADD-S uses `models/` exactly like the cited LINEMOD table; BOP-official evaluation uses the coarser `models_eval/` meshes, whose 1000-vs-full floor is larger. `adds_eval_m` is provided so the difference is visible, but the primary numbers should not be compared with published BOP LMO ADD-S values.
4. **Failures are data-driven, not cherry-picked.** Failures: 3 (obj11/222: K_phys_lt_2; obj11/761: n_valid_depth_lt_900; obj11/1175: n_valid_depth_lt_900). Selection used `px_count_visib` only, so K is not selected on. The no-depth candidates (obj11/761, obj11/1175) have essentially empty depth maps (6 and 65 valid pixels), an LMO capture quirk rather than a pipeline failure.
5. **AUC semantics.** The BOP ADD-S AUC uses the *same* 0–0.1 m normalisation; with n≈37 the estimate is a step function and its CI is wide. Paired differences vs hard are the more informative quantity.

## 7. Reproduce

```bash
cd labs/01-laplace-qpeplayer/experiments/real/linemod
/home/syk/miniconda3/envs/WM/bin/python run_two_mode_lmo.py            # full, < 1 min
/home/syk/miniconda3/envs/WM/bin/python run_two_mode_lmo.py --quick    # smoke
```

Data: `/mnt/storage1/syk/data/bop/lmo` (BOP LMO; single test sequence 000002). Constants and helpers are imported from `run_two_mode.py` / `run_pose_error.py`; those files are unmodified. `checks.constants` in the JSON records the exact values used.

