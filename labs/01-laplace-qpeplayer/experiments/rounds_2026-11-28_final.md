# P1 最终 60 轮日志（2026-11-28，final/p1）

工作树：`/mnt/storage1/syk/.superagent-work/wt-final-p1`（branch `final/p1`，起点 `2ca1ec3`）。子 agent 各自在 `wt-final-p1-<task>`（branch `final/p1-<task>`）。
约束：不 commit/push；不改 `deliverables/`、`AGENTS.md`、顶层 `labs/EXPERIMENTS.md`。GPU 被 cyr/wyx 占用，全部 CPU。

| Round | hypothesis | change | command | metric before → after | keep/revert | artifact |
|-|-|-|-|-|-|-|
| 1 | 最终工作树与论文现状可审计 | 建立本日志；审计 paper TODO/引用/19 表；确认 HEAD 2ca1ec3 含 phase-4 | — | 19 表；N1–N3 TODO 明确 | keep | 本日志 |
| 2 | sQPEP 引文可经 Crossref 验证并补入；POLI/Pyatov/dQP 仍不可验证 | 用 Crossref API 验证 DOI 10.1109/TIM.2025.3540135（B. Xue 等，IEEE TIM 74:1-11, 2025），加 [18] 并在 QPEP 段引用；TODO note 改为“sQPEP 已加、其余省略” | `curl api.crossref.org` | 论文引用 17→18；sQPEP 元数据完整；POLI/Pyatov/dQP 保持省略 | keep | `paper/draft.md` |
| 3 | LMO 主表可复用 LINEMOD 构造（mask_visib 可见面片） | 子 agent lmo：`run_two_mode_lmo.py` + `results/lmo_two_mode.{json,md,png}` + README §8 | WM python，32 s | **37/40 帧 K=2**；hard 2.95° [2.19,8.10] / ADD-S 10.2 mm [6.6,13.7] / AUC 0.862；Laplace 43.73°/19.0 mm/AUC 0.766；两模结构在 LMO 保持（ΔJ 1.37e-3、flip 100.5°）；退化机制=可见面片小（中位 61 mm）+深度-模型残差，不是漏枚举（2× starts 37/37 不变） | keep | `real/linemod/run_two_mode_lmo.py`、`results/lmo_two_mode.*`、README §8 |
| 4 | SDPRLayers 同数据前向可解且 10×10 紧 | 子 agent sdpr：`baselines/run_sdpr_p1.py` + `results/sdpr_p1.json` | sdpr env（torch/scs），147 s | 合成 3 个 θ：10×10 SDP 值 gap ≤3.3e-10（θ=0 rank 2 简并，其余 rank 1=hard）；真实 5 帧（15 帧中解出 5 帧）：全 tight rank-1=hard；上游 13×13 值低 ~1e-6、非紧（1/5 rank 2） | keep | `baselines/run_sdpr_p1.py`、`results/sdpr_p1.json` |
| 5 | EPro-PnP-style 连续采样/BPnP-style IFT 可作同数据参照 | 子 agent epro：`baselines/run_epro_bpnp_style.py` + `results/epro_bpnp_style.{md,json}` | WM python | θ=0.1：EPro IS 4.06°（ESS 1598/4000，grad rel 0.43）、BPnP IFT 8.5e-5（0.7 ms）；θ=0：EPro ESS 1399 且破坏 P-对称 → **inconclusive**；BPnP 单支在 θ=-0.1 已非 argmin（gap 1.19） | keep | `baselines/run_epro_bpnp_style.py`、`results/epro_bpnp_style.*` |
| 6 | 检测器噪声可作前端代理（GPU 禁） | 子 agent detect：`real/linemod/run_detector_noise.py` + `results/detector_noise.{json,md,png}` + README §9；完整 n=15、σ∈{0,0.5,1,2,5}、f∈{0.05,0.1,0.2}、10 seeds（461 s） | WM python | σ≤1 px 无退化；σ=5 hard 20.49°/164.2 mm/66% 失败（沿平坦双模谷滑动，非 flip 切换），Laplace 0.9×；粗差 f=0.05 时 K≥1 仅 5%、f≥0.1 为 0（L2 无鲁棒核的 O(1) breakdown） | keep | `run_detector_noise.py`、`results/detector_noise.*`、README §9 |
| 7 | 端到端 toy 训练/梯度方差/吞吐可在 CPU 做 | 子 agent e2e：`code/run_e2e_fit.py` + smoke 结果；完整 20-seed/120-step 运行被时间上限杀掉，只保留 smoke（4 seeds/40 steps） | WM python，78 s（smoke） | mix 目标有限 τ 不可表示（Laplace 下限 1.108e-2，成功率 0%）；梯度方差 Laplace≈0、MC 相对 5.08e-2、hard 跨 tie 两支相差 1.87 | keep（smoke，标 unverified full） | `code/run_e2e_fit.py`、`results/_smoke_e2e.*` |
| 8 | 新结果需写入论文并更新 N1/N3/TODO | 新增 §5.7（Table 5.20 LMO、5.21 检测器噪声、5.22 SDPR、5.23 EPro/BPnP-style）；N1→partially run、N3→smoke only；not-run 段改为“官方可微基线仍未跑，前向/局部参照已测”；删除论文 TODO note 中的 sQPEP 项 | — | 论文 19→23 表；引用 18 条；TODO 只剩官方可微训练/GPU/In-The-Dark | keep | `paper/draft.md` |
| 9 | 600 轮预算下应把 LMO/检测器/e2e 做完，并尽可能跑官方 SDPR/BPnP（本地 third_party 有 clone） | 建立 wave-2 五个 worktree（lmo2/detect2/sdpr2/bnp2/e2e2），各自复制最新脚本/结果；40 根扩展到更多实例（自建 `roots/run_rootcount_many.py`） | — | 待记录 | 待 | 本日志 |
| 10 | 40 根计数应在更多随机实例上稳健 | 新增 `roots/run_rootcount_many.py`（10 个随机对称 M + instance M(0)），复用 `run_rootcount_fast.run_case` | `python roots/run_rootcount_many.py` | 结果见 `results/rootcount_many.json`（11 例，预期 40/40；若个别≠40 如实记录） | 待 | `roots/run_rootcount_many.py`、`results/rootcount_many.json` |
| 11 | 官方 SDPRLayers 能否在 CPU 上跑 P1 10×10 前向 | 子 agent sdpr2：`baselines/run_official_sdpr.py`，加载 released cb3aed2 + 本地 fork（62d8ac6），diffcp/SCS 前向 | sdpr env，3.3 s | 官方层跑通；10×10 值差 6.7e-14（θ=0.01）/5.4e-12（θ=0.1）；θ=0 rank-2 → xs=None（值仍对 +3.7e-10）；tight 行 pose 与 hard <1e-6°；6–9 ms vs 我们 12–14 ms | keep | `baselines/run_official_sdpr.py`、`results/official_sdpr.*` |
| 12 | 官方 BPnP 能否在 CPU 上跑并适配 P1 代价 | 子 agent bnp2：`baselines/run_official_bpnp.py`，released f1f4f6a + 一行 shim（kornia/.contiguous） | sdpr env，60 s | toy 反向 vs FD 17.1%（forward 几何 LS vs backward 代数 normal equations，结构性）；same-data：BPnP 代数残差 ≡ P1 bearing 残差 1.8e-15，用 P1 forward + released backward 与我们的 IFT/FD 一致 | keep | `baselines/run_official_bpnp.py`、`results/official_bpnp.*` |
| 13 | 真实图像上的经典特征前端能否喂给层 | 子 agent detect2：`real/linemod/run_feature_frontend.py` + `results/feature_frontend.*` | sdpr env cv2，短 | 严格门 0/15；放宽门 Shi-Tomasi 12/15、ORB 11/15、Harris 0/15；匹配点数下 Shi-Tomasi 2.77° vs dense@n 2.18°（CI 重叠）；hard 翻模 17%/27% vs dense 0%；ORB 选错平面 64% | keep | `run_feature_frontend.py`、`results/feature_frontend.*` |
| 14 | LMO 应从 37 帧扩到全序列量级 | 子 agent lmo2（修正后）：`run_two_mode_lmo_full.py --per-obj 100`，slurm 3202，177/200 可用 | slurm cpu，143 s | **177 帧 K=2**；hard 3.33°/19.8 mm/ADD-S 9.4 mm/AUC 0.857 [0.831,0.881]；Laplace 42.5°/18.7 mm/AUC 0.776；n_plane≥350 vs <350：2.57° vs 8.97°（p=0.0004），最近模式同向→代价面位移；SQPNP 同平面内点与 hard 等价（|Δrot| 中位 0.03°） | keep | `run_two_mode_lmo_full.py`、`results/lmo_full.*` |
| 15 | 端到端全量（20 seeds/120 steps）应跑完 | 子 agent e2e2：`code/run_e2e_fit.py` 全量，664 s | WM python | mix 不可表示（Laplace 下限 1.1e-2、hard 2.05、成功率 0%）；star 成功率 25%（63 [26,66] 步）；梯度方差 Laplace≈0、MC 5.08e-2、hard 两支差 1.87；替换之前的 smoke 文本 | keep | `results/e2e_fit.*` |

## 并行 wave 1（每完成 1 个 = 1 轮）

| Round | task | worktree | 状态 |
|-|-|-|-|
| P1 | LMO 主表（hard/Laplace/MC，ADD-S/AUC，CI） | `wt-final-p1-lmo` (`final/p1-lmo`) | running |
| P2 | SDPRLayers 同数据前向基线（10×10/13×13 SDP） | `wt-final-p1-sdpr` (`final/p1-sdpr`) | running |
| P3 | EPro-PnP-style 连续采样 + BPnP-style IFT 梯度 | `wt-final-p1-epro` (`final/p1-epro`) | running |
| P4 | 检测器噪声鲁棒性（σ/外点扫，代理 front-end） | `wt-final-p1-detect` (`final/p1-detect`) | running |
| P5 | 端到端 toy 训练 + 梯度方差 + 吞吐 | `wt-final-p1-e2e` (`final/p1-e2e`) | running |
