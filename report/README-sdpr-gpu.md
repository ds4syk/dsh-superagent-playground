# SDPR 的 GPU 并行实现（2026-11）

本目录是“SDPR 是否有 GPU 并行版本、没有则实现并验证”的交付入口。

| 文档 | 内容 |
|---|---|
| [sdpr_gpu.md](sdpr_gpu.md) | **主报告**：原文 CPU-only 引用、公开实现检索、batch-parallel GPU 层、P6 验证、速度、局限 |
| [../docs/visuals/2026-11-09-sdpr-gpu.png](../docs/visuals/2026-11-09-sdpr-gpu.png) | 一页速览图 |
| [../deliverables/SDPR_GPU并行实现_2026-11.docx](../deliverables/SDPR_GPU并行实现_2026-11.docx) | Word 版（默认交付格式） |
| [../src/gpu_sdpr/](../src/gpu_sdpr/) | 实现：`BarrierSolver`、`SDPRLayer`、(P6) 构造 |
| [../repro/gpu_sdpr/](../repro/gpu_sdpr/) | 复现：`verify_p6.py`、`benchmark.py`、`e2e_calib.py`、Slurm 脚本 |
| [../research/gpu_sdpr/README.md](../research/gpu_sdpr/README.md) | 负结果与调参记录（原始 ADMM 为什么不行） |

## 一句话结论

原文 §VII 明确把 forward/backward 都在 CPU 列为局限；截至 2026-11-09 没有公开的 GPU 并行 SDPR。本实现把 (P6) 的 200 个 SDP 在单卡 RTX 5090 上批量求解，x 误差 ≤7.8e-6、batch 梯度误差 ≤6.7e-5，证书全部 corank=1；B=20 前向 13×、B=200 前向 150×（对单进程 CPU SCS）。

## 证据分级

- **论文声称**：原文的 CPU-only 与 future-work 表述。
- **独立复现**：`repro/gpu_sdpr/results/*.json`（Slurm job 2699）。
- **未验证**：cuADMM/CARDAL/SCS-GPU 的性能只做 README/摘要级检索，未集成。
