# 盲复现结果（中间记录）

> 数据快照：2026-11-21。完整评分见 `results/grades/` 与 `results/grades_summary.json`；
> 本文件是便于 PR 阅读的摘要，最终以知识库 05 页为准。

## 运行

- 模型：DeepSeek-V4-Pro（`deepseek-v4-pro`）、Kimi-K3（`kimi-k3`）。
- 10 个案例，每个独立会话；4 轮（独立求解 → 敌意审稿 → 工具 → 终稿）。
- 预算：每次最多 24,576 输出 token；S1/S2 若 reasoning 占满，则用 `thinking disabled`
  把 reasoning 摘要合并成终稿（记录为 `*_force_final`）。
- Slurm：`cpu` 分区；日志 `logs/`；transcripts `transcripts/`。

## 最终结果（两名评分者均值；0–5）

| 案例 | DeepSeek-V4-Pro | Kimi-K3 |
|---|---:|---:|
| 007 两点相关 | 1.5 | 1.0 |
| 017 μ(π)=2 | 1.5 | 2.0 |
| 087 Mahler | 2.0 | 1.5 |
| 102 BasicSDP | 1.0 | 1.0 |
| 159 Erdős | 1.5 | 1.5 |
| 197 Kaplansky | 1.5 | 1.0 |
| 221 Mézard–Parisi | 1.5 | 2.0 |
| 271 Heisenberg | 2.5 | 1.0 |
| 287 自由群因子 | 1.5 | 1.0 |
| 362 Vlasov–Maxwell | 2.5 | 1.5 |

**完整复现（≥4）：0/10（两个模型都是）。** 最高均分 2.5（DeepSeek 271/362）；
Kimi 最高 2.0（017/221）。

- DeepSeek：2 阶段（24k + 16k）+ 强制终稿；10/10 有终稿。
- Kimi：K3 thinking 模式在同预算下 7–8 分钟/次，票内时间不可行；改用**缩减 direct 协议**
  （单次调用、thinking disabled、16k 上限），10/10 有终稿；与 DeepSeek 不完全可比。

## 失败模式

1. 诚实 NO_RESOLUTION + 已知定理归约（007, 102, 159, 197）。
2. 条件性证明依赖未证黑箱（271 DLS；087 特殊情形）。
3. 过度声称/方向错误（287, 221）。
4. 预算耗尽：多数案例的 24k 预算全花在 reasoning 上，需强制终稿步骤。

## 污染

- 20/20 探针（DeepSeek 10 + Kimi 10）回答 “NO RELIABLE RECOLLECTION”；Kimi 的 102/221
  提到 2026 前的背景结果，但没有 2026-10 的官方结论。
- 没有证据表明模型直接复述官方结论；预训练污染无法完全排除。

## 限制

- 单次采样、非算力对齐、10 个案例有选择偏差。
- 研究级证明评分困难；双评分 + 人工复核仍可能误判。
- 未复现 Lean 证明。
