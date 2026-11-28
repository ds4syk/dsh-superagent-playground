# dsh-superagent-playground
Sandbox repo for dsh-superagent

## Hello

Hello! This is a sandbox repository for dsh-superagent testing. Feel free to poke around here without worrying about breaking anything.

## Lab

This sandbox is also used to test the agent running on the lab server.

## Research reports

- **SDPR & QPEP 研读与复现**（五个问题的回答 + 相机 baseline 标定复现）：
  [report](report/README-sdpr-qpep.md) · [answers](report/answers.md) ·
  [QPEP notes](report/qpep_reading_notes.md) · [reproduction](report/reproduction.md) ·
  [DOCX 知识库](deliverables/README.md) · [本科简明版](deliverables_simple/README.md) ·
  [SDPR 复现图表集](analysis/atlas/SDPR_复现图表集.docx) · [KB 约定](docs/KB_CONVENTION.md)
- **SDPR 能否作为 V-JEPA 2 的桥接输出层？** 机制分析 + 受控实验（旋转同步）：
  [report](report/README-sdpr-vjepa2.md) · [主报告](report/sdpr_vjepa2_bridge.md) ·
  [DOCX](deliverables/SDPR_VJEPA2_桥接输出层_2026-11.docx)
- **SDPR 前向网络与数据集**（2026-11）：
  [report](report/README-sdpr-forward-net.md) · [主报告](report/sdpr_forward_network_dataset.md) ·
  [DOCX](deliverables/SDPR_前向网络与数据集_2026-11.docx)

## 论文工作区（labs，2026-11）

从三个 QPEP 可微位姿层 prototype 并行推进为三篇论文：

- [总览与共享骨架](labs/README.md) · [数学问题汇总](labs/MATH_QUESTIONS.md)
- [P1 Laplace-QPEPLayer](labs/01-laplace-qpeplayer/)（歧义/分支交叉、多峰 Laplace 平滑）
- [P2 Constrained-QPEPLayer](labs/02-constrained-qpeplayer/)（物理不等式约束、FB 约化隐微分）
- [P3 Robust-QPEPLayer](labs/03-robust-qpeplayer/)（高外点、GNC + 曲率修正不动点微分）
