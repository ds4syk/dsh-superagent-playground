# SDPR 作为 V-JEPA 2 的桥接输出层

本目录回答 2026-11-03 Feishu 问题：“SDPR 能不能作为 V-JEPA 2 的一个层 / 输出桥接层？”
并给出一个受控实验。

| 文档 | 内容 |
|---|---|
| [**sdpr_vjepa2_bridge.md**](sdpr_vjepa2_bridge.md) | **主报告**：两篇论文事实、接口判定、实验、边界、下一步 |
| [repro.md](repro.md) | 环境、资产、Slurm 命令、产物 |
| [../kb/sdpr-vjepa2/00.md](../kb/sdpr-vjepa2/00.md) | DOCX 知识库总览 |
| [../deliverables/](../deliverables/) | Word 版（默认交付） |
| [figures/](figures/) | 架构/样本/误差/证书/梯度图 |

## 一句话结论

SDPR 可以作为 V-JEPA 2 的**几何结构化输出头**：V-JEPA 2 latent → 适配器 → 齐次 QCQP →
SDPR 的全局最优解，输出损失经 KKT/证书隐函数微分回传到适配器（以及解冻时的编码器）。
但它**不能**替代 JEPA 缺失的像素/视频解码器；JEPA 的“无输出”是设计选择，不是缺陷。

## 实验速览

- 任务：8 节点相机旋转同步；V-JEPA 2 ViT-L 冻结特征 + 边适配器 + SDPR。
- 对照：直接回归、两阶段（相对损失 + 离线 SDPR）、端到端（全局损失穿 SDPR）、
  chordal 谱同步、oracle（GT 测量）。
- 关键证据：输出损失对编码器/适配器的梯度、SDP tightness/证书、端到端 vs 两阶段。
- 结果：见主报告 §4 与 `artifacts/results/metrics_*.json`。
