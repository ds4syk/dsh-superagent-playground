This is by an automated agent, not a human.

## 摘要

![一页速览](https://raw.githubusercontent.com/ds4syk/dsh-superagent-playground/agent/ai-math-survey-repro-3877d01b7b12/docs/visuals/2026-11-21-ai-math-survey.png)


本 PR 调研“当前用 AI 做科研/数学证明的主流方案、是否需要客制化工具/Skill、不同模型差距”，
并对 OpenAI 2026-10 发布的 722 篇数学手稿中 **10 个公开推理摘要案例**做了一次信息隔离的
DeepSeek-V4-Pro / Kimi-K3 盲复现实验。

交付：

- 知识库 `kb/ai-math-survey/00–07.md`（DOCX 导出到 `deliverables/`）
- 盲复现工程 `repro/openai-10/`（题面、harness、transcripts、评分、污染探针）
- 一页可视化 `docs/visuals/2026-11-21-ai-math-survey.png`
- 官方结论/评分 key `research/ai-math-survey/official_key.{md,json}`

## 事实核查（722 篇）

- 2026-10-06/07 初始：**722 篇 / 372 个 result family**；当前 `main`：**721 / 373**；
  差异来自撤回、修订与新增。
- 只有 **10 个 family** 公开推理摘要（007, 017, 087, 102, 159, 197, 221, 271, 287, 362）。
- 约 **42%** 的 top-line 结果有 Lean 形式化；10/10 公开案例都有 `lean/docs/NNN.md` 与
  Comparator challenge，但形式化范围普遍小于论文声称。
- 已撤回 3 篇、修订 14 篇；官方称这些是 raw research data，部分未形式化结果可能有问题。

## 盲复现实验

- 题面来自公开推理摘要的 `Original prompt`，删除解法提示；不含官方结论/论文/Lean。
- 每个 (模型, 案例) 独立会话；DeepSeek-V4-Pro 与 Kimi-K3；每题约 40k 输出 token 预算。
- 记录完整 reasoning/content；预算截断后强制“停止推理写终稿”并标注。
- 评分：官方结论 + 候选终稿，双评分模型 + 人工复核，0–5 rubric。
- 结果（两名评分者均值，0–5）：**完整复现 0/10（DeepSeek 与 Kimi 都是）**。
  DeepSeek-V4-Pro：271/362 = 2.5，087 = 2.0，其余 1.0–1.5；Kimi-K3：017/221 = 2.0，
  其余 1.0–1.5。没有案例达到 4 分（方向正确、主定理基本证出）。
- Kimi 使用**缩减 direct 协议**（单次调用、thinking disabled、16k 上限）：K3 thinking 模式
  在同预算下 7–8 分钟/次，票内时间无法完成 2 阶段；这使 Kimi 与 DeepSeek 不完全可比。
- 失败模式：诚实 NO_RESOLUTION + 已知定理归约（007/102/159/197）；条件性证明依赖未证黑箱
  （271 DLS；087 特殊情形）；过度声称/方向错误（287/221）；预算耗尽（多数 reasoning 占满）。
  详见 `repro/openai-10/RESULTS.md` 与知识库 05 页。

## 证据

- **Before/After 视觉**：本 PR 不改变 UI，无 UI before/after；一页可视化见
  `docs/visuals/2026-11-21-ai-math-survey.png`（内嵌在知识库 00 页）。
- **可复现**：
  ```bash
  cd repro/openai-10
  sbatch submit.sh
  python3 grade.py --graders deepseek,kimi
  python3 contamination/probe.py
  ```
- **检查**：harness 的 streaming/cap/force-final 路径已用真实 API 端到端测试；Slurm 日志在
  `repro/openai-10/logs/`；transcripts 在 `repro/openai-10/transcripts/`。

## 主要结论（完整版见知识库）

1. 主流方案是“模型探索 + 工具 + 验证”的组合；不是单一模型端到端。
2. 探索阶段通用模型 + Python 沙箱即可；可信交付必须 Lean/Mathlib + Comparator +
   自动形式化/证明修复 + 领域工具 + 审计。
3. 公开 benchmark 上模型差距约 10–30 个百分点；研究级证明差距更大。
4. 722 篇是不同验证阶段的手稿，不是 722 个已独立验证的定理；10 个公开案例的完整复现
   **0/10**，失败模式见 05 页。

## 合并风险

- 只新增研究/复现文件与文档，不改生产代码、不改 UI、不改部署。
- 新增 `repro/openai-10/` 含 API harness；密钥从 `~/.dsh/.credentials.yaml` 运行时读取，
  不写入仓库。transcripts 可能含模型输出（不含密钥）；如有敏感内容需在发布前检查。
- `research/ai-math-survey/official_key.*` 含官方结论，仅用于评分；求解上下文不读取。
- Feishu 知识库/文档创建受阻：bot 缺少 `drive:drive`、`docx:document`、`space:folder:create`
  scope；本 PR 用本地 DOCX + 仓库链接交付，Feishu 阅读副本待补权限。
- 无自动合并；请人工 review。

## 限制

- 单次采样、非算力对齐、无置信区间；10 个案例有选择偏差。
- 模型可能通过预训练知道部分 2026 结果；污染探针只做估计，不能完全排除。
- 未复现 Lean 证明；形式化验证是下一步。
