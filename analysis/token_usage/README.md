# DSH 全量 Token 用量统计

从 `~/.dsh/sessions` 的原始会话日志复算本机 DSH 的**全部历史** token 用量与估算花费。

## 用法

```bash
python3 analysis/token_usage/aggregate.py \
  --prices analysis/token_usage/prices-2026-11-26.json \
  --out /tmp/usage.json \
  --cutoff-ms 1795629600000     # 可选：只统计该时刻之前的请求（2026-11-26 02:00 +08:00）
python3 analysis/token_usage/render.py /tmp/usage.json docs/visuals/2026-11-26-lifetime-token-usage.png
```

## 口径

- 数据源：`~/.dsh/sessions/**/session.v4.jsonl.zstd`，按 `assistant/message.usage` 汇总；`compaction/summary` 的 usage 也计入。
- 计费 token = `inputTokens + cacheReadTokens + cacheWriteTokens + outputTokens`。
- 同一 assistant message id 在多个会话出现时只计一次（seeded fork 会复制父会话历史）。
- 按 `step/start` 的 Asia/Shanghai 时间归日、归月。
- 价格：`prices-2026-11-26.json`（models.dev 快照）。DeepSeek 的 models.dev 价是官方 off-peak 价，峰时（北京时间工作日 09:00–12:00、14:00–18:00）乘 2；汇率 1 USD = 6.678 CNY。
- 不含：会话标题生成、web search LLM 调用（本地只有请求体、没有 usage 事件）、其它 CLI（Claude Code / Codex / OpenCode）的用量。
