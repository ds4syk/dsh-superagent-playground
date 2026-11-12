# Daily DSH report

`daily-summary.mjs` builds a one-page Chinese work report and posts it to a Feishu group as an
image:

- **今日重点** — a short, result-first brief of the day: the main threads of work, completed
  outcomes and open pull requests.
- **Token 用量** — per-day billed tokens from the dsh-context *Context Insights* projection
  cache (`~/.dsh/storages/session_projcache/sessions/*.json`): total, cache reads/cache-hit
  rate, uncached input and output.
- **风险 / 阻塞** — taken from blocked work items, or a sentence in a report that asks for a
  decision.
- **明日重点** — generated from the open superagent tickets and open pull requests.

The report is rendered as HTML, converted to a one-page A4 PDF with WeasyPrint, rasterized with
`pdftoppm`, and sent with `lark-cli im +messages-send --image`. The narrative (headline and
highlights) is curated by a small DeepSeek call; if the model or the render fails, the script
falls back to a deterministic brief and then to a short text message, so the group still gets
the day's summary.

## Requirements

- Node.js 22+ (uses `node:sqlite` and the built-in `fetch`)
- Python 3.10+ with `venv` (the installer creates a private virtualenv with WeasyPrint 70)
- `pdftoppm` and `pdfinfo` (poppler-utils)
- Noto Sans CJK SC fonts
- `lark-cli` as the Feishu bot (`~/.local/bin/lark-bot`)
- Optional: `gh` for the open-PR list, and `~/.dsh/.credentials.yaml` with `DEEPSEEK_API_KEY`
  for the narrated brief

## Install

```sh
./install-daily-summary.sh --chat-id oc_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx \
  --repo ds4syk/dsh-superagent-playground --time 20:00
```

The installer copies the script to `~/.local/share/dsh-daily-summary/`, creates the WeasyPrint
virtualenv on first use, writes a `run.sh` launcher that carries the chat id and environment (so
the repository never stores one), and replaces the `dsh-daily-summary` line in the user's
crontab. It is idempotent; re-run it after changing the script.

## Run by hand

```sh
~/.local/share/dsh-daily-summary/run.sh --dry-run     # render the image, send nothing
~/.local/share/dsh-daily-summary/run.sh              # render and send now
~/.local/share/dsh-daily-summary/run.sh --date 2026-11-01 --dry-run
~/.local/share/dsh-daily-summary/run.sh --no-llm     # deterministic brief, no API call
```

The last HTML, PDF and PNG are kept in `~/.local/share/dsh-daily-summary/last/` for inspection;
cron redirects output to `~/.local/share/dsh-daily-summary/run.log`.

# Monthly token usage

`monthly-usage.mjs` tallies token usage and estimated CNY cost over a rolling window and
`monthly-usage-chart.py` renders the one-page summary. It reuses the same 口径 as the daily
report, but reads the durable session logs directly
(`~/.dsh/sessions/<workspace>/<session>/session.v4.jsonl.zstd`), so it also works when the
Context Insights projection cache is stale or only partially backfilled.

```sh
node automation/monthly-usage.mjs --days 30 --json usage-30d.json
python3 automation/monthly-usage-chart.py usage-30d.json docs/visuals/2026-11-12-monthly-token-cost.png
```

- `--days N` sets the window length (default 30), `--end YYYY-MM-DD` fixes its last day
  (default today in Asia/Shanghai). Without `--json` the JSON is printed to stdout.
- Billed tokens = uncached input + cache read + cache write + output; a step is billed to the
  day it started, seeded sessions do not double-count inherited parent history.
- Prices live in `MODEL_PRICES` in `monthly-usage.mjs` (DeepSeek CNY list price, checked
  2026-11-11: `deepseek-flash` cache hit ¥0.04 / miss ¥2 / output ¥8 per 1M tokens, weekday
  09:00–12:00 and 14:00–18:00 at peak, everything else half price). The figure is a
  list-price estimate, not a platform bill.
- Output JSON: window, coverage, totals, `byDay`, `byModel`, `byWorkspace`, `bands`
  (peak/off-peak), `byKind` (main vs subagent), per-session totals with `kind`, and the
  price table used. Subagent child sessions are separate logs under the same workspace
  directory and are counted like any other session.
