# Daily DSH summary

`daily-summary.mjs` builds a Chinese daily summary and posts it to a Feishu group with the
`lark-cli` bot:

- **今日工作** — the day's non-superagent DSH sessions grouped by workspace, plus the
  superagent tickets touched today and their state.
- **Token 用量** — per-day billed tokens from the dsh-context *Context Insights* projection
  cache (`~/.dsh/storages/session_projcache/sessions/*.json`), split into cache reads,
  uncached input and output.
- **明日规划** — open superagent tickets and open pull requests of the configured repository.

The script reads local state only. The only network calls are the optional `gh api` PR list
(skipped when it fails) and the final Feishu send.

## Install

```sh
./install-daily-summary.sh --chat-id oc_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx \
  --repo ds4syk/dsh-superagent-playground --time 20:00
```

The installer copies the script to `~/.local/share/dsh-daily-summary/`, writes a `run.sh`
launcher that carries the chat id and environment (so the repository never stores one), and
replaces the `dsh-daily-summary` line in the user's crontab. It is idempotent; re-run it after
changing the script.

## Run by hand

```sh
~/.local/share/dsh-daily-summary/run.sh --dry-run     # print the message, send nothing
~/.local/share/dsh-daily-summary/run.sh              # send now
~/.local/share/dsh-daily-summary/run.sh --date 2026-11-01 --dry-run
```

Cron redirects its output to `~/.local/share/dsh-daily-summary/run.log`.
