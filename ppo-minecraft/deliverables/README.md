# 交付物：PPO × Minecraft 知识库（DOCX）

本目录是 Feishu 知识库的离线 DOCX 副本，按「00 总览 + NN 子页」组织：

| 文件 | 内容 |
|-|-|
| `kb/00 总览：PPO × Minecraft 训练系统（2026-11）.docx` | 范围、结论、资料索引、证据分级 |
| `kb/01 环境与基准：Dreamer 的 MC 设置.docx` | DreamerV3 的 MC 任务、25 动作、20 seeds 步数 |
| `kb/02 PPO 方案设计.docx` | 网络、PPO 超参、并行采样、分块 BPTT |
| `kb/03 手写奖励函数.docx` | 奖励公式、物品价值表、效率、生存、反 hacking |
| `kb/04 训练系统与实验.docx` | 本次真实训练的吞吐、曲线、里程碑、诊断 |
| `kb/05 到钻石的可行性与路线.docx` | 算力外推、BC/分层/课程三条路线 |

源码与数据：

- 代码：仓库 `ppo-minecraft/`（PR #11）
- 运行数据：`/mnt/storage1/syk/dreamer/runs/mcppo/diamond_s0`（metrics/episodes/TB/ckpt/curves）
- 可读副本：Feishu 文件夹「PPO × Minecraft（2026-11）」

维护约定：文档结论随 run 更新；任何"到钻石"的声称必须附 `episodes.jsonl` 里程碑与录像。
