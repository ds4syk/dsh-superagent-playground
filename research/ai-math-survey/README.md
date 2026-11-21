# AI 数学证明调研：工作区索引

本目录是知识库 `kb/ai-math-survey/` 的研究底稿与评分 key。

| 文件 | 内容 |
|---|---|
| `survey.md` | 调研长文：主流方案、工具需求、模型差距、722 篇事实核查 |
| `official_key.md` | 10 个公开案例的官方结论（人读版，仅用于评分） |
| `official_key.json` | 同上，机器可读 |
| `sources.md` | 主要来源与链接（可选） |

## 复现工程

- `repro/openai-10/`：题面、harness、transcripts、评分、污染探针、设计说明。
- 运行：`cd repro/openai-10 && sbatch submit.sh`（全量）或 `submit_kimi.sh` / `submit_ds362.sh`。
- 评分：`python3 grade.py --graders deepseek,kimi --force`。
- 汇总：`python3 make_summary.py`、`python3 update_kb.py`。

## 证据分级

- **论文/官方声称**：`openai/math` 的 README、overview、论文摘要、Lean docs；厂商 benchmark 数字。
- **独立复现**：本票的 10 案例盲测、评分、污染探针；`repro/openai-10/` 可追溯。
- **未验证**：官方论文的完整证明细节；未重跑的 Lean 形式化。

## 重要限制

- 10 个案例是 OpenAI 选出的公开样例，存在选择偏差。
- 盲测预算（约 40k 输出 token/案例）远小于官方平均 3 小时 ChatGPT Pro thinking。
- 模型可能通过预训练知道部分 2026 结果；污染探针只做估计。
- 没有复现 Lean 证明；形式化验证是下一步。
- Feishu 知识库创建受 bot scope 限制（`drive:drive`、`docx:document` 等未开通），
  本票以本地 DOCX + 仓库链接交付；详见 PR 的“限制”一节。
