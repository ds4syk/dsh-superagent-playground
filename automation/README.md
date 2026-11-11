# Weekly DSH report

`daily-summary.mjs` builds a one-page Chinese **weekly** work report and posts it to a Feishu
group as an image. The file and the installed launcher keep the historical `daily` name so the
existing crontab marker keeps working.

- **本周重点** — a short, result-first brief of the week: the main threads of work, completed
  outcomes and open pull requests.
- **任务 Token 占比** — a pie chart of the week's billed tokens grouped by superagent task. A
  task is a ticket; sessions are attributed to it through the ticket's execution directory
  (`~/.dsh-state/superagent/orchestrator.sqlite`). The top 7 tasks are drawn separately and the
  rest is folded into **其他** together with non-task sessions (main session, project
  workspaces). The legend shows each slice's share and token count.
- **Token 用量** — billed tokens folded from the DSH session logs
  (`~/.dsh/sessions/<workspace>/<session>/session.v4.jsonl.zstd`) over the window: total, cache
  reads/cache-hit rate, uncached input and output. A step is billed to the day it started, and
  inherited side-conversation history is not counted twice. The dsh-context *Context Insights*
  projection cache (`~/.dsh/storages/session_projcache/sessions/*.json`) is only a fallback:
  the web UI refreshes it on demand, so a cron run can read it stale or empty.
- **风险 / 阻塞** — taken from blocked work items, or a sentence in a report that asks for a
  decision.
- **下周重点** — generated from the open superagent tickets and open pull requests.

The report is rendered as HTML, converted to a one-page A4 PDF with WeasyPrint, rasterized with
`pdftoppm`, and sent with `lark-cli im +messages-send --image`. The narrative (headline and
highlights) is curated by a small DeepSeek call; if the model or the render fails, the script
falls back to a deterministic brief and then to a short text message, so the group still gets
the week's summary.

The window ends on the run day and defaults to the last 7 days (`--days`), so the Sunday 20:00
cron covers Monday through Sunday.

## Requirements

- Node.js 22+ (uses `node:sqlite` and the built-in `fetch`)
- `zstdcat` (zstd) to read the durable session logs
- Python 3.10+ with `venv` (the installer creates a private virtualenv with WeasyPrint 70)
- `pdftoppm` and `pdfinfo` (poppler-utils)
- Noto Sans CJK SC fonts
- `lark-cli` as the Feishu bot (`~/.local/bin/lark-bot`)
- Optional: `gh` for the open-PR list, and `~/.dsh/.credentials.yaml` with `DEEPSEEK_API_KEY`
  for the narrated brief

## Install

```sh
./install-daily-summary.sh --chat-id oc_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx \
  --repo ds4syk/dsh-superagent-playground --time 20:00 --weekday 0 --days 7
```

`--weekday` is `0`–`6` (`0` = Sunday, the default), `--days` is the window length (default 7).
The installer copies the script to `~/.local/share/dsh-daily-summary/`, creates the WeasyPrint
virtualenv on first use, writes a `run.sh` launcher that carries the chat id and environment (so
the repository never stores one), and replaces the `dsh-daily-summary` line in the user's
crontab (including an older daily line). It is idempotent; re-run it after changing the script.

## Run by hand

```sh
~/.local/share/dsh-daily-summary/run.sh --dry-run     # render the image, send nothing
~/.local/share/dsh-daily-summary/run.sh              # render and send now
~/.local/share/dsh-daily-summary/run.sh --date 2026-11-11 --days 7 --dry-run
~/.local/share/dsh-daily-summary/run.sh --no-llm     # deterministic brief, no API call
~/.local/share/dsh-daily-summary/run.sh --date 2026-11-11 --resend  # re-send the same week
```

`--date` is the last day of the window. The last HTML, PDF and PNG are kept in
`~/.local/share/dsh-daily-summary/last/` for inspection; cron redirects output to
`~/.local/share/dsh-daily-summary/run.log`.
