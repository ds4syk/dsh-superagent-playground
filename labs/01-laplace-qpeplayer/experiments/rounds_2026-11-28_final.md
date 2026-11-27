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

## 并行 wave 1（每完成 1 个 = 1 轮）

| Round | task | worktree | 状态 |
|-|-|-|-|
| P1 | LMO 主表（hard/Laplace/MC，ADD-S/AUC，CI） | `wt-final-p1-lmo` (`final/p1-lmo`) | running |
| P2 | SDPRLayers 同数据前向基线（10×10/13×13 SDP） | `wt-final-p1-sdpr` (`final/p1-sdpr`) | running |
| P3 | EPro-PnP-style 连续采样 + BPnP-style IFT 梯度 | `wt-final-p1-epro` (`final/p1-epro`) | running |
| P4 | 检测器噪声鲁棒性（σ/外点扫，代理 front-end） | `wt-final-p1-detect` (`final/p1-detect`) | running |
| P5 | 端到端 toy 训练 + 梯度方差 + 吞吐 | `wt-final-p1-e2e` (`final/p1-e2e`) | running |
