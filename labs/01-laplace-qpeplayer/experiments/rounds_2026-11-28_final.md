# P1 最终 60 轮日志（2026-11-28，final/p1）

工作树：`/mnt/storage1/syk/.superagent-work/wt-final-p1`（branch `final/p1`，起点 `2ca1ec3`）。子 agent 各自在 `wt-final-p1-<task>`（branch `final/p1-<task>`）。
约束：不 commit/push；不改 `deliverables/`、`AGENTS.md`、顶层 `labs/EXPERIMENTS.md`。GPU 被 cyr/wyx 占用，全部 CPU。

| Round | hypothesis | change | command | metric before → after | keep/revert | artifact |
|-|-|-|-|-|-|-|
| 1 | 最终工作树与论文现状可审计 | 建立本日志；审计 paper TODO/引用/19 表；确认 HEAD 2ca1ec3 含 phase-4 | — | 19 表；N1–N3 TODO 明确 | keep | 本日志 |
| 2 | sQPEP 引文可经 Crossref 验证并补入；POLI/Pyatov/dQP 仍不可验证 | 用 Crossref API 验证 DOI 10.1109/TIM.2025.3540135（B. Xue 等，IEEE TIM 74:1-11, 2025），加 [18] 并在 QPEP 段引用；TODO note 改为“sQPEP 已加、其余省略” | `curl api.crossref.org` | 论文引用 17→18；sQPEP 元数据完整；POLI/Pyatov/dQP 保持省略 | keep | `paper/draft.md` |

## 并行 wave 1（每完成 1 个 = 1 轮）

| Round | task | worktree | 状态 |
|-|-|-|-|
| P1 | LMO 主表（hard/Laplace/MC，ADD-S/AUC，CI） | `wt-final-p1-lmo` (`final/p1-lmo`) | running |
| P2 | SDPRLayers 同数据前向基线（10×10/13×13 SDP） | `wt-final-p1-sdpr` (`final/p1-sdpr`) | running |
| P3 | EPro-PnP-style 连续采样 + BPnP-style IFT 梯度 | `wt-final-p1-epro` (`final/p1-epro`) | running |
| P4 | 检测器噪声鲁棒性（σ/外点扫，代理 front-end） | `wt-final-p1-detect` (`final/p1-detect`) | running |
| P5 | 端到端 toy 训练 + 梯度方差 + 吞吐 | `wt-final-p1-e2e` (`final/p1-e2e`) | running |
