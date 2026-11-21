# 一页速览：AI 数学证明（2026-11-21）

文件：`2026-11-21-ai-math-survey.png`（2757×1996，200 dpi）。

四个面板：

- **A. 四层流水线**：题面/定义 → 模型探索 → 工具（Python/CAS/检索） → Lean 形式化；
  另一条是敌意审稿 → 证明修复/子目标 → Comparator/审计 → 可信交付。核心信息：探索流水线
  （快、便宜、可错）与验证流水线（慢、贵、可信）必须分开。
- **B. FrontierMath**：2026-11-15 快照，GPT-5.5 Pro 52.4%、Claude Opus 4.7 43.79%、
  Kimi K2.6 38.97%、Gemini 3 Pro 37.6%、GPT-5 32.41%、DeepSeek-V3.2 22.1%、o3 18.69%。
- **C. 盲复现评分**：10 个公开案例 × DeepSeek-V4-Pro / Kimi-K3 的 0–5 分；数据来自
  `repro/openai-10/results/grades/`。
- **D. 结论与数字**：722→721、10/721、~42%、10/10、0–5；主流方案/工具需求/模型差距/最大瓶颈。

数据源：
- `openai/math` README / overview.tex（2026-11-21 快照）
- `evals.report/benchmarks/frontiermath`（2026-11-15 数据快照）
- 本票 `repro/openai-10/`（独立复现）
