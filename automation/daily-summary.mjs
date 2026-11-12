#!/usr/bin/env node
/**
 * dsh-daily-summary: build a one-page weekly work report image and post it to a Feishu chat.
 *
 * The file keeps its historical name so the installed launcher and the crontab marker keep
 * working; the report itself is a weekly summary with a per-task token pie chart.
 *
 * Pipeline: collect local state -> curate the brief -> render HTML -> WeasyPrint PDF -> PNG
 * -> lark-cli image message. If the image pipeline fails, a short text fallback is sent.
 *
 * Data sources:
 *  - DSH session logs: <home>/.dsh/sessions/<workspace>/<session>/session.v4.jsonl.zstd
 *    (the durable record that carries each step's provider-reported usage). The Context Insights
 *    projection cache is only a fallback: it is refreshed on demand by the web UI, so a cron run
 *    can see it stale or empty.
 *  - dsh-superagent tickets: <stateDir>/orchestrator.sqlite. Each ticket's execution directory is
 *    matched against the session workspace to attribute the week's tokens to a task.
 *  - Open pull requests of the configured repository (optional; skipped when gh fails)
 *
 * Usage:
 *   node daily-summary.mjs [--date YYYY-MM-DD] [--days 7] [--dry-run] [--no-llm] [--resend]
 *
 * --date is the last day of the report window; --days is its length (default 7).
 *
 * Environment overrides: DSH_DAILY_SUMMARY_CHAT, DSH_DAILY_SUMMARY_REPO,
 * DSH_DAILY_SUMMARY_LARK_CLI, DSH_DAILY_SUMMARY_GH, DSH_DAILY_SUMMARY_STATE_DIR,
 * DSH_DAILY_SUMMARY_WEASYPRINT, DSH_DAILY_SUMMARY_PDFTOPPM, DSH_DAILY_SUMMARY_API_KEY.
 */
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { execFileSync, spawn } from 'node:child_process'
import readline from 'node:readline'
import { DatabaseSync } from 'node:sqlite'

const TZ = 'Asia/Shanghai'
const HOME = os.homedir()
const APP_DIR = path.join(HOME, '.local/share/dsh-daily-summary')

function parseArgs(argv) {
  const args = { dryRun: false, llm: true, days: 7 }
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i]
    if (a === '--dry-run') args.dryRun = true
    else if (a === '--resend') args.resend = true
    else if (a === '--no-llm') args.llm = false
    else if (a === '--date') args.date = argv[++i]
    else if (a === '--days') args.days = Number(argv[++i])
    else if (a === '--chat-id') args.chat = argv[++i]
    else if (a === '--repo') args.repo = argv[++i]
    else if (a === '--lark-cli') args.lark = argv[++i]
    else if (a === '--gh') args.gh = argv[++i]
    else if (a === '--state-dir') args.stateDir = argv[++i]
    else if (a === '--content') args.content = argv[++i]
    else if (a === '--help' || a === '-h') { console.log(fs.readFileSync(process.argv[1], 'utf8').split('*/')[0]); process.exit(0) }
    else throw new Error(`unknown argument: ${a}`)
  }
  return args
}

const args = parseArgs(process.argv.slice(2))
const chatId = args.chat || process.env.DSH_DAILY_SUMMARY_CHAT || ''
const repo = args.repo || process.env.DSH_DAILY_SUMMARY_REPO || ''
const larkCli = args.lark || process.env.DSH_DAILY_SUMMARY_LARK_CLI || path.join(HOME, '.local/bin/lark-bot')
const ghCli = args.gh || process.env.DSH_DAILY_SUMMARY_GH || path.join(HOME, '.local/bin/gh-agent')
const stateDir = args.stateDir || process.env.DSH_DAILY_SUMMARY_STATE_DIR || '/mnt/storage1/syk/.dsh-state/superagent'
const weasyprint = process.env.DSH_DAILY_SUMMARY_WEASYPRINT || path.join(APP_DIR, 'venv/bin/weasyprint')
const pdftoppm = process.env.DSH_DAILY_SUMMARY_PDFTOPPM || '/usr/bin/pdftoppm'
const pdfinfo = process.env.DSH_DAILY_SUMMARY_PDFINFO || '/usr/bin/pdfinfo'
const model = process.env.DSH_DAILY_SUMMARY_MODEL || 'deepseek-flash'
const sessionsDir = path.join(HOME, '.dsh/storages/session_projcache/sessions')
const ticketDb = path.join(stateDir, 'orchestrator.sqlite')

const dateFormatter = new Intl.DateTimeFormat('en-CA', { timeZone: TZ, year: 'numeric', month: '2-digit', day: '2-digit' })
const timeFormatter = new Intl.DateTimeFormat('en-GB', { timeZone: TZ, hour: '2-digit', minute: '2-digit', hour12: false })
function localDate(value = new Date()) {
  return dateFormatter.format(value)
}
function localTime(value = new Date()) {
  return timeFormatter.format(value)
}
function weekdayText(date) {
  const [y, m, d] = date.split('-').map(Number)
  return new Intl.DateTimeFormat('zh-CN', { timeZone: TZ, weekday: 'short' }).format(new Date(Date.UTC(y, m - 1, d, 4)))
}
/** Add days to a YYYY-MM-DD string in the report timezone (noon UTC+8 avoids DST edges). */
function addDays(date, days) {
  const [y, m, d] = date.split('-').map(Number)
  const value = new Date(Date.UTC(y, m - 1, d, 4))
  value.setUTCDate(value.getUTCDate() + days)
  return localDate(value)
}
/** The inclusive window ending at `end`, newest last. */
function reportWindow(end, days) {
  const dates = []
  for (let i = days - 1; i >= 0; i--) dates.push(addDays(end, -i))
  return { start: dates[0], end, days: dates.length, dates }
}
function shortDate(date) {
  const [, month, day] = date.split('-')
  return `${Number(month)}.${String(day).padStart(2, '0')}`
}
function rangeText(start, end) {
  return start === end ? shortDate(end) : `${shortDate(start)} – ${shortDate(end)}`
}
function hyphenDate(date) {
  const [, month, day] = date.split('-')
  return `${month}-${day}`
}
function clean(value, max = 0) {
  if (typeof value !== 'string') return ''
  let text = value.replace(/\r?\n+/g, ' ').replace(/\s+/g, ' ').trim()
  if (max > 0 && text.length > max) text = `${text.slice(0, max)}…`
  return text
}
function fmtTokens(n) {
  if (!Number.isFinite(n)) return '0'
  if (n >= 1e8) return `${(n / 1e8).toFixed(2).replace(/\.?0+$/, '')} 亿`
  if (n >= 1e4) return `${(n / 1e4).toFixed(1).replace(/\.0$/, '')} 万`
  return String(n)
}
function fmtInt(n) { return Number(n || 0).toLocaleString('en-US') }
function esc(value) {
  return String(value ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;')
}

// DeepSeek API list prices in CNY per 1M tokens (api-docs.deepseek.com/zh-cn/quick_start/pricing,
// checked 2026-11-11). Peak is Beijing-time Mon-Fri 09:00-12:00 / 14:00-18:00; off-peak is half.
const MODEL_PRICES = {
  'deepseek-flash': {
    hit: { peak: 0.04, off: 0.02 },
    miss: { peak: 2, off: 1 },
    out: { peak: 8, off: 4 },
  },
  'deepseek-v4-pro': {
    hit: { peak: 0.3, off: 0.15 },
    miss: { peak: 9, off: 4.5 },
    out: { peak: 27, off: 13.5 },
  },
}
const DEFAULT_MODEL_PRICE = MODEL_PRICES['deepseek-flash']
function modelPrice(model) {
  return MODEL_PRICES[model] || DEFAULT_MODEL_PRICE
}
function isPeakTime(value) {
  const weekday = weekdayText(localDate(value))
  if (weekday === '周六' || weekday === '周日') return false
  const [hour, minute] = localTime(value).split(':').map(Number)
  const minutes = hour * 60 + minute
  return (minutes >= 540 && minutes < 720) || (minutes >= 840 && minutes < 1080)
}
/** Estimated CNY list-price cost for one usage record. cacheWrite has no price row; bill it as miss. */
function costOfUsage({ uncached = 0, cacheRead = 0, cacheWrite = 0, output = 0 }, model, band) {
  const price = modelPrice(model)
  const key = band === 'peak' ? 'peak' : 'off'
  return ((Number(uncached) + Number(cacheWrite)) / 1e6) * price.miss[key]
    + (Number(cacheRead) / 1e6) * price.hit[key]
    + (Number(output) / 1e6) * price.out[key]
}
function fmtCost(value) {
  const n = Number(value || 0)
  return `¥${n.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`
}
function fmtCostShort(value) {
  const n = Number(value || 0)
  return `¥${n >= 100 ? Math.round(n).toLocaleString('en-US') : n.toFixed(2)}`
}
function emptyModelUsage(provider, model) {
  return { provider, model, requests: 0, tokens: 0, cost: 0, peakRequests: 0, offRequests: 0 }
}

const sessionsRoot = path.join(HOME, '.dsh/sessions')
const zstdcat = process.env.DSH_DAILY_SUMMARY_ZSTD || (fs.existsSync('/usr/bin/zstdcat') ? '/usr/bin/zstdcat' : 'zstdcat')
// Only these events carry what the report needs; the rest (mostly tool output) is skipped before
// JSON.parse, which keeps the window scan of the largest logs cheap.
const trackedEvent = /"type":\s*"(?:assistant\/message|step\/start|session\/end-seed|session\/title|session)"/

/** Sessions with the per-day figures Context Insights computes, summed over the window. */
function collectSessionsFromCache(dates) {
  const sessions = []
  let files = []
  try { files = fs.readdirSync(sessionsDir).filter(f => f.endsWith('.json')) } catch { return sessions }
  for (const file of files) {
    let record
    try { record = JSON.parse(fs.readFileSync(path.join(sessionsDir, file), 'utf8')).record } catch { continue }
    const rows = record?.rows ?? {}
    let tokens = 0
    let requests = 0
    const breakdown = { uncached: 0, cacheRead: 0, cacheWrite: 0, output: 0 }
    const models = new Map()
    for (const date of dates) {
      const day = rows?.contextActivity?.val?.days?.[date]
      if (!day || !Number.isFinite(day.tokens) || day.tokens <= 0) continue
      tokens += day.tokens
      requests += Number(day.requests || 0)
      for (const [provider, providerModels] of Object.entries(day.cost ?? {})) {
        for (const [model, bands] of Object.entries(providerModels ?? {})) {
          const key = `${provider}/${model}`
          const aggregate = models.get(key) || emptyModelUsage(provider, model)
          for (const [band, bucket] of Object.entries(bands ?? {})) {
            const usage = {
              uncached: Number(bucket?.uncached || 0),
              cacheRead: Number(bucket?.cacheRead || 0),
              cacheWrite: Number(bucket?.cacheWrite || 0),
              output: Number(bucket?.output || 0),
            }
            for (const name of Object.keys(breakdown)) breakdown[name] += usage[name]
            aggregate.tokens += usage.uncached + usage.cacheRead + usage.cacheWrite + usage.output
            aggregate.cost += costOfUsage(usage, model, band)
          }
          models.set(key, aggregate)
        }
      }
    }
    if (tokens <= 0) continue
    const first = clean(rows?.titleInput?.val?.first?.text)
    const rawTitle = clean(rows?.title?.val)
    let title = rawTitle || first || '未命名会话'
    // The UI truncates long first prompts to a short title; use the first sentence then.
    if (rawTitle && rawTitle.length >= 18 && first.startsWith(rawTitle)) title = first.split(/[。\n]/)[0] || first
    // Generic subagent/worker prompts are not useful work labels.
    if (/^You are working on\b/i.test(title)) title = ''
    sessions.push({
      id: file.replace(/\.json$/, ''),
      tokens,
      requests,
      breakdown,
      models: [...models.values()],
      title,
      workspace: rows?.workingDirectory?.val || record?.identity?.cwd || '',
    })
  }
  return sessions
}

/** Session logs that can hold events in the window; a log's mtime is its last appended event. */
function sessionLogFiles(startDate) {
  const since = Date.parse(`${startDate}T00:00:00+08:00`)
  const files = []
  for (const slug of fs.readdirSync(sessionsRoot)) {
    const dir = path.join(sessionsRoot, slug)
    let ids
    try {
      if (!fs.statSync(dir).isDirectory()) continue
      ids = fs.readdirSync(dir)
    } catch { continue }
    for (const id of ids) {
      const file = path.join(dir, id, 'session.v4.jsonl.zstd')
      try {
        if (fs.statSync(file).mtimeMs >= since) files.push({ id, file })
      } catch { /* not a session directory */ }
    }
  }
  return files
}

/**
 * Fold one session log into the requested window.
 *
 * A step is billed to the day it started, like Context Insights. `session/end-seed` with
 * `inherited: true` closes the copied parent history of a seeded side conversation: those events
 * were already billed in the parent session, so only later events are counted.
 */
async function scanSessionLog({ id, file }, start, end) {
  const child = spawn(zstdcat, [file], { stdio: ['ignore', 'pipe', 'ignore'] })
  await new Promise((resolve, reject) => {
    child.once('spawn', resolve)
    child.once('error', reject)
  })
  const lines = readline.createInterface({ input: child.stdout, crlfDelay: Infinity })
  let header = null
  let title = ''
  let inheritedThrough = -1
  const stepStarts = new Map()
  const usages = []
  let readError = null
  try {
    for await (const line of lines) {
      if (!trackedEvent.test(line)) continue
      let event
      try { event = JSON.parse(line) } catch { continue }
      if (event.type === 'session') {
        if (header === null) header = event
        continue
      }
      if (event.type === 'session/title') {
        title = clean(event.data?.title || title)
        continue
      }
      if (event.type === 'session/end-seed') {
        // The inherited marker closes the copied parent history; collect first, filter after.
        if (event.data?.inherited === true && event.seq > inheritedThrough) inheritedThrough = event.seq
        continue
      }
      if (event.type === 'step/start') {
        stepStarts.set(`${event.data?.turn}:${event.data?.step}`, event.time)
        continue
      }
      const usage = event.data?.usage
      if (event.type !== 'assistant/message' || !usage || !Number.isFinite(event.time)) continue
      const source = event.data.message?.source
      usages.push({
        seq: event.seq,
        time: event.time,
        turn: event.data.turn,
        step: event.data.step,
        usage,
        model: source?.model || 'unknown',
        provider: source?.provider || '',
      })
    }
  } catch (error) {
    readError = error
  } finally {
    lines.close()
  }
  const code = await new Promise(resolve => child.once('close', resolve))
  if (code !== 0 && usages.length === 0) {
    throw readError instanceof Error ? readError : new Error(`${path.basename(zstdcat)} exited ${code}`)
  }
  if (code !== 0) {
    // A live session may be mid-frame; keep the steps that were fully read.
    console.error(`session log ${id} ended early (${path.basename(zstdcat)} exited ${code}); using ${usages.length} step(s)`)
  }
  const day = { tokens: 0, requests: 0, breakdown: { uncached: 0, cacheRead: 0, cacheWrite: 0, output: 0 } }
  const models = new Map()
  for (const item of usages) {
    if (item.seq <= inheritedThrough) continue
    const started = stepStarts.get(`${item.turn}:${item.step}`) ?? item.time
    const startedDate = localDate(new Date(started))
    if (startedDate < start || startedDate > end) continue
    const usage = item.usage
    const uncached = Number(usage.inputTokens || 0)
    const cacheRead = Number(usage.cacheReadTokens || 0)
    const cacheWrite = Number(usage.cacheWriteTokens || 0)
    const output = Number(usage.outputTokens || 0)
    day.requests += 1
    day.breakdown.uncached += uncached
    day.breakdown.cacheRead += cacheRead
    day.breakdown.cacheWrite += cacheWrite
    day.breakdown.output += output
    const stepTokens = uncached + cacheRead + cacheWrite + output
    day.tokens += stepTokens

    const band = isPeakTime(started) ? 'peak' : 'off'
    const key = `${item.provider}/${item.model}`
    const aggregate = models.get(key) || emptyModelUsage(item.provider, item.model)
    aggregate.requests += 1
    aggregate.tokens += stepTokens
    aggregate.cost += costOfUsage({ uncached, cacheRead, cacheWrite, output }, item.model, band)
    if (band === 'peak') aggregate.peakRequests += 1
    else aggregate.offRequests += 1
    models.set(key, aggregate)
  }
  if (day.tokens <= 0) return null
  if (/^You are working on\b/i.test(title)) title = ''
  return { id, tokens: day.tokens, requests: day.requests, breakdown: day.breakdown, models: [...models.values()], title, workspace: header?.cwd || '' }
}

/** Sessions with the window's per-day figures, read from the durable session logs. */
async function collectSessions(dates, start, end) {
  try {
    execFileSync(zstdcat, ['--version'], { stdio: 'ignore', timeout: 10_000 })
    const sessions = []
    let failures = 0
    for (const entry of sessionLogFiles(start)) {
      try {
        const session = await scanSessionLog(entry, start, end)
        if (session) sessions.push(session)
      } catch (error) {
        failures += 1
        console.error(`session log ${entry.id} skipped: ${clean(error instanceof Error ? error.message : String(error), 120)}`)
      }
    }
    if (failures > 0 && sessions.length === 0) throw new Error(`all ${failures} candidate session log(s) failed`)
    if (sessions.length === 0) {
      // A log-format change would also look empty; the cache is the only other source then.
      const cached = collectSessionsFromCache(dates)
      if (cached.length > 0) {
        console.error(`no usage found in session logs for ${start}..${end}; using the Context Insights projection cache`)
        return cached
      }
    }
    return sessions
  } catch (error) {
    const reason = clean(error instanceof Error ? error.message : String(error), 120)
    console.error(`session log scan failed (${reason}); falling back to the Context Insights projection cache`)
    return collectSessionsFromCache(dates)
  }
}

/** Tickets the superagent touched in the window, plus the ones still open. */
function readTickets(start, end) {
  if (!fs.existsSync(ticketDb)) return { week: [], open: [], byDir: new Map() }
  const db = new DatabaseSync(ticketDb, { readOnly: true })
  const week = []
  const open = []
  const byDir = new Map()
  const inWindow = (ms) => {
    if (!Number.isFinite(ms)) return false
    const day = localDate(new Date(ms))
    return day >= start && day <= end
  }
  try {
    for (const row of db.prepare('select value from ticket').all()) {
      let ticket
      try { ticket = JSON.parse(row.value) } catch { continue }
      if (typeof ticket?.title !== 'string') continue
      const created = Date.parse(ticket.createdAt || '')
      const updated = Date.parse(ticket.updatedAt || ticket.createdAt || '')
      const assignment = (ticket.assignments || []).at(-1)
      const report = assignment?.report
      const prs = new Set()
      for (const match of String(report?.summary || '').matchAll(/https:\/\/github\.com\/[^\s)]+\/pull\/\d+/g)) prs.add(match[0].split('/').pop())
      for (const follow of ticket.follows || []) {
        const m = /^github:(\d+)$/.exec(follow?.thread || '')
        if (m) prs.add(m[1])
      }
      const item = {
        id: ticket.id,
        title: clean(ticket.title, 90),
        created,
        updated,
        ended: ticket.ended || null,
        reportState: report?.state || '',
        result: clean(report?.summary, 260),
        prs: [...prs],
        directory: typeof ticket.execution?.directory === 'string' ? ticket.execution.directory : '',
      }
      // A ticket's execution directory is the worker session's cwd, i.e. the task key.
      if (item.directory) byDir.set(item.directory, item)
      if (!Number.isFinite(updated)) continue
      if (inWindow(updated) || inWindow(created)) week.push(item)
      if (!item.ended) open.push(item)
    }
  } finally { db.close() }
  week.sort((a, b) => a.updated - b.updated)
  open.sort((a, b) => a.updated - b.updated)
  return { week, open, byDir }
}

function collectTickets(start, end) {
  // The host writes this SQLite database continuously; a busy read gets retried.
  for (let attempt = 1; attempt <= 3; attempt++) {
    try { return readTickets(start, end) } catch {
      if (attempt === 3) return { week: [], open: [], byDir: new Map() }
      execFileSync('/bin/sleep', ['1'])
    }
  }
  return { week: [], open: [], byDir: new Map() }
}

function openPullRequests() {
  if (!repo) return []
  for (let attempt = 1; attempt <= 2; attempt++) {
    try {
      const output = execFileSync(ghCli, ['api', `repos/${repo}/pulls?state=open&per_page=20`, '--jq', '[.[] | {number, title, url: .html_url}]'], {
        encoding: 'utf8', timeout: 30_000, stdio: ['ignore', 'pipe', 'pipe'],
      })
      const list = JSON.parse(output)
      return Array.isArray(list) ? list : []
    } catch {
      if (attempt === 1) execFileSync('/bin/sleep', ['2'])
    }
  }
  return []
}

function ticketStatus(ticket) {
  if (ticket.ended) {
    if (ticket.ended.outcome === 'merged') return '已合并'
    if (ticket.ended.outcome === 'closed') return '已关闭'
    return '已完成'
  }
  if (ticket.reportState === 'blocked' || ticket.reportState === 'failed') return '受阻'
  if (ticket.reportState === 'ready') return '待审'
  return '进行中'
}

/** Normalized data handed to the curator and the renderer. */
async function buildReportData(range) {
  const sessions = await collectSessions(range.dates, range.start, range.end)
  const tickets = collectTickets(range.start, range.end)
  const openPrs = openPullRequests()

  const taskMap = new Map()
  const other = { tokens: 0, requests: 0, sessions: 0 }
  for (const s of sessions) {
    const ticket = tickets.byDir.get(s.workspace)
    if (ticket) {
      const task = taskMap.get(ticket.id) || {
        id: ticket.id,
        title: clean(ticket.title, 70),
        status: ticketStatus(ticket),
        prs: ticket.prs,
        tokens: 0,
        requests: 0,
        sessions: 0,
      }
      task.tokens += s.tokens
      task.requests += s.requests
      task.sessions += 1
      taskMap.set(ticket.id, task)
    } else {
      other.tokens += s.tokens
      other.requests += s.requests
      other.sessions += 1
    }
  }
  const totals = { tokens: 0, requests: 0, sessions: sessions.length, uncached: 0, cacheRead: 0, cacheWrite: 0, output: 0 }
  for (const s of sessions) {
    totals.tokens += s.tokens
    totals.requests += s.requests
    for (const key of ['uncached', 'cacheRead', 'cacheWrite', 'output']) totals[key] += s.breakdown[key]
  }
  const denom = totals.uncached + totals.cacheRead + totals.cacheWrite
  totals.cacheHit = denom > 0 ? Number(((totals.cacheRead / denom) * 100).toFixed(1)) : 0

  const modelMap = new Map()
  for (const s of sessions) {
    for (const usage of s.models || []) {
      const key = `${usage.provider}/${usage.model}`
      const aggregate = modelMap.get(key) || emptyModelUsage(usage.provider, usage.model)
      aggregate.requests += usage.requests
      aggregate.tokens += usage.tokens
      aggregate.cost += usage.cost
      aggregate.peakRequests += usage.peakRequests || 0
      aggregate.offRequests += usage.offRequests || 0
      modelMap.set(key, aggregate)
    }
  }
  const models = [...modelMap.values()].sort((a, b) => b.tokens - a.tokens)
  totals.cost = models.reduce((sum, model) => sum + model.cost, 0)
  totals.peakRequests = models.reduce((sum, model) => sum + model.peakRequests, 0)
  totals.offRequests = models.reduce((sum, model) => sum + model.offRequests, 0)

  const tasks = [...taskMap.values()].sort((a, b) => b.tokens - a.tokens)
  for (const task of tasks) task.share = totals.tokens > 0 ? Number(((task.tokens / totals.tokens) * 100).toFixed(1)) : 0

  return {
    start: range.start,
    end: range.end,
    days: range.days,
    weekday: weekdayText(range.end),
    totals,
    tasks,
    models,
    other,
    tickets: tickets.week.map(t => ({ title: t.title, status: ticketStatus(t), prs: t.prs, result: t.result })),
    open: tickets.open.map(t => ({ title: t.title, status: ticketStatus(t), prs: t.prs, result: t.result })),
    openPrs: openPrs.map(p => ({ number: p.number, title: clean(p.title, 60) })),
  }
}

function apiKey() {
  if (process.env.DSH_DAILY_SUMMARY_API_KEY) return process.env.DSH_DAILY_SUMMARY_API_KEY
  try {
    const file = process.env.DSH_DAILY_SUMMARY_CREDENTIALS || path.join(HOME, '.dsh/.credentials.yaml')
    const match = fs.readFileSync(file, 'utf8').match(/DEEPSEEK_API_KEY:\s*(\S+)/)
    return match ? match[1].replace(/^["']|["']$/g, '') : ''
  } catch { return '' }
}

/** Ask the model for a concise, result-first weekly brief; return undefined on any failure. */
async function curate(data) {
  if (!args.llm) return undefined
  const key = apiKey()
  if (!key) return undefined
  const cadence = data.days === 1 ? '每日' : '每周'
  const word = data.days === 1 ? '今日' : '本周'
  const next = data.days === 1 ? '明日' : '下周'
  const system = [
    `你是资深技术负责人，为团队写${cadence}一页简报。遵循这些原则：`,
    `- 倒金字塔：先给结论，再给支撑细节；读者 10 秒能看懂${word}最重要的事。`,
    '- 只写结果、影响和关键进展；合并同类项；不写过程流水账，不罗列所有会话。',
    '- 亮点 3 条以内；每条主题不超过 14 个字，说明不超过 45 个字。',
    '- 只使用输入数据中的事实；不编造数字或结论。API 费用只能引用 totals.costCNY，不要自行按价格表计算。',
    `- 任务标题只代表${word}在做的主题，不能据此写「已产出 / 已完成 / 已跑出 / 成功 / 已交付」；只有 windowTickets 中 status 为「已完成」或「已合并」的事项才能写完成。`,
    '- 「待审」表示 PR/交付物还没合并或验收，不能写成已交付；「进行中」只能写推进、提交、排队或等待。',
    '- 进行中的事项用「推进 / 恢复 / 提交 / 排队 / 等待」等状态词。',
    '- 不要自行推断风险、CI 缺失或流程问题。',
    '- 中文，短句，避免术语堆砌和空话；正文不要出现输入 JSON 的字段名（如 windowTickets、totals、openItems、openPullRequests），用中文表述。',
    '- 完成/已合并数量以 completedTasks 为准，待审 PR 数以 openPrCount 为准，不要自行清点 windowTickets。',
    `- 简报口径是「${cadence}」；正文使用「${word} / ${next}」措辞，不要混用其他周期。`,
    `- 不需要输出风险或${next}重点：系统会从工作项状态自动生成。`,
    '只返回 JSON：{"headline":"一句话结论，不超过40字","highlights":[{"title":"主题","detail":"结果或影响，不超过45字"}]}',
  ].join('\n')
  const payload = {
    period: { start: data.start, end: data.end, days: data.days, cadence: data.days === 1 ? 'daily' : 'weekly' },
    taskCount: data.tickets.length,
    completedTasks: data.tickets.filter(t => t.status === '已完成' || t.status === '已合并').length,
    openPrCount: data.openPrs.length,
    totals: { tokens: data.totals.tokens, requests: data.totals.requests, sessions: data.totals.sessions, cacheHitPercent: data.totals.cacheHit, costCNY: Number(data.totals.cost.toFixed(2)) },
    models: data.models.map(m => ({ model: m.model, tokens: m.tokens, requests: m.requests, costCNY: Number(m.cost.toFixed(2)) })),
    tasks: data.tasks.slice(0, 12).map(t => ({ title: t.title, tokens: t.tokens, sharePercent: t.share, sessions: t.sessions, requests: t.requests, status: t.status, prs: t.prs })),
    nonTaskSessions: { tokens: data.other.tokens, sessions: data.other.sessions },
    windowTickets: data.tickets.map(t => ({ title: t.title, status: t.status, prs: t.prs })),
    openItems: data.open.map(t => ({ title: t.title, status: t.status, prs: t.prs })),
    openPullRequests: data.openPrs.map(p => `#${p.number} ${p.title}`),
  }
  let lastError = ''
  for (let attempt = 1; attempt <= 2; attempt++) {
    try {
      const response = await fetch('https://api.deepseek.com/chat/completions', {
        method: 'POST',
        headers: { 'content-type': 'application/json', authorization: `Bearer ${key}` },
        body: JSON.stringify({
          model,
          messages: [{ role: 'system', content: system }, { role: 'user', content: JSON.stringify(payload) }],
          temperature: 0.3,
          max_tokens: 3000,
          response_format: { type: 'json_object' },
        }),
        signal: AbortSignal.timeout(60_000),
      })
      if (!response.ok) { lastError = `HTTP ${response.status}`; throw new Error(lastError) }
      const body = await response.json()
      const text = body?.choices?.[0]?.message?.content ?? ''
      const json = JSON.parse(String(text).replace(/^```(?:json)?\s*|\s*```$/g, ''))
      if (!validContent(json)) { lastError = 'invalid content'; throw new Error(lastError) }
      return normalizeContent(json)
    } catch (error) {
      lastError = error instanceof Error ? error.message : String(error)
      if (attempt === 1) await new Promise(resolve => setTimeout(resolve, 3000))
    }
  }
  console.error(`curation failed: ${lastError}`)
  return undefined
}

function validContent(c) {
  return c && typeof c === 'object'
    && typeof c.headline === 'string' && c.headline.trim().length > 0
    && Array.isArray(c.highlights) && c.highlights.length >= 1 && c.highlights.length <= 4
    && c.highlights.every(h => h && typeof h.title === 'string' && typeof h.detail === 'string')
    && (c.risk === undefined || c.risk === null || typeof c.risk === 'string')
}

function scrubFields(value) {
  return String(value ?? '')
    .replace(/weekTickets\s*中/g, '窗口内工单中')
    .replace(/\bweekTickets\b/g, '窗口内工单')
    .replace(/windowTickets\s*中/g, '窗口内工单中')
    .replace(/\bwindowTickets\b/g, '窗口内工单')
    .replace(/\bopenItems\b/g, '进行中事项')
    .replace(/\bopenPullRequests\b/g, '开放 PR')
    .replace(/\btotals\b/g, '总计')
}
function normalizeContent(c) {
  const text = (value, max) => clean(scrubFields(value), max)
  return {
    headline: text(c.headline, 60),
    highlights: c.highlights.slice(0, 3).map(h => ({ title: text(h.title, 14), detail: text(h.detail, 58) })),
    risk: text(c.risk || '', 46),
  }
}

/** Deterministic fallback when the model is unavailable or answers badly. */
function localContent(data) {
  const completed = data.tickets.filter(t => t.status === '已完成' || t.status === '已合并').length
  const waiting = data.tickets.filter(t => t.status === '待审')
  const blocked = data.tickets.find(t => t.status === '受阻')
  const period = data.days === 1 ? '今日' : `${data.days} 天`
  const headline = `${period} ${data.totals.sessions} 个会话、${fmtInt(data.totals.requests)} 次请求；API 费用约 ${fmtCost(data.totals.cost)}。`
  const highlights = data.tasks.slice(0, 2).map(t => ({
    title: clean(t.title, 14),
    detail: `${t.share}% token · ${t.sessions} 个会话 · ${fmtInt(t.requests)} 次请求`,
  }))
  if (data.tickets.length > 0) {
    highlights.push({
      title: 'superagent',
      detail: [
        completed ? `${completed} 项已完成` : '',
        waiting.length ? `PR ${waiting.flatMap(t => t.prs).slice(0, 2).map(p => `#${p}`).join('、')} 待审` : '',
        data.open.filter(t => t.status === '进行中').length ? `${data.open.filter(t => t.status === '进行中').length} 项进行中` : '',
      ].filter(Boolean).join('；') || '进行中',
    })
  }
  return {
    headline: clean(headline, 60),
    highlights: highlights.slice(0, 3),
    risk: blocked ? `受阻：${blocked.title}` : '',
  }
}

function planItems(data) {
  const items = []
  const add = (text) => {
    if (!text || items.length >= 3 || items.includes(text)) return
    items.push(text)
  }
  for (const t of data.open) {
    if (t.status === '待审' && t.prs.length > 0) add(`审阅并合并 PR ${t.prs.slice(0, 2).map(p => `#${p}`).join('、')}`)
  }
  for (const p of data.openPrs) {
    if (data.open.some(t => t.prs.includes(String(p.number)))) continue
    add(`审阅 PR #${p.number}：${clean(p.title, 26)}`)
  }
  for (const t of data.open) {
    if (/定时(整理|生成).{0,4}总结|定时总结/.test(t.title)) continue
    if (t.status === '受阻') add(`处理阻塞：${clean(t.title, 24)}`)
    else if (t.status === '进行中') add(`继续：${clean(t.title, 24)}`)
  }
  if (items.length === 0) items.push('暂无进行中的工作项。')
  return items.slice(0, 3)
}

function riskFromData(data) {
  // Only active (open) work adds a current risk; a merged ticket's old blocker is history.
  const all = data.open
  const blocked = all.find(t => t.status === '受阻')
  if (blocked) return clean(`受阻：${blocked.title}`, 72)
  const pattern = /拍板|需要(你|用户|人工)|等待(你|用户)|待(你|用户)确认|需要决策/
  // Deployment tooling notes (merge CLI / auto-merge rules) are not a per-week task risk.
  const tooling = /merge CLI|auto-merge|自动合并|审批规则/
  for (const item of all) {
    for (const sentence of String(item.result || '').split(/[。\n]/)) {
      if (tooling.test(sentence)) continue
      if (pattern.test(sentence)) return clean(sentence.replace(/\*\*/g, ''), 72)
    }
  }
  return ''
}

const PIE_COLORS = ['#4d6bfe', '#f59e0b', '#10b981', '#ef4444', '#8b5cf6', '#06b6d4', '#f97316']
const PIE_OTHER_COLOR = '#94a3b8'
const PIE_NON_TASK_COLOR = '#cbd5e1'
const PIE_MAX_TASKS = 6

/** Slices for the per-task token donut: top tasks, other tasks and non-task sessions. */
function pieData(data) {
  const total = data.totals.tokens
  if (!(total > 0)) return { items: [], total: 0, taskShare: 0 }
  const top = data.tasks.slice(0, PIE_MAX_TASKS)
  const restTokens = data.tasks.slice(PIE_MAX_TASKS).reduce((sum, task) => sum + task.tokens, 0)
  const items = top.map((task, i) => ({ label: task.title, tokens: task.tokens, color: PIE_COLORS[i] }))
  if (restTokens > 0) items.push({ label: '其他任务', tokens: restTokens, color: PIE_OTHER_COLOR })
  if (data.other.tokens > 0) items.push({ label: '非任务会话', tokens: data.other.tokens, color: PIE_NON_TASK_COLOR })
  const taskShare = total > 0 ? ((total - data.other.tokens) / total) * 100 : 0
  return { items, total, taskShare }
}

function polar(cx, cy, r, degrees) {
  const radians = ((degrees - 90) * Math.PI) / 180
  return [cx + r * Math.cos(radians), cy + r * Math.sin(radians)]
}

function annulusPath(cx, cy, outer, inner, start, end) {
  const [x1, y1] = polar(cx, cy, outer, start)
  const [x2, y2] = polar(cx, cy, outer, end)
  const [x3, y3] = polar(cx, cy, inner, end)
  const [x4, y4] = polar(cx, cy, inner, start)
  const large = end - start > 180 ? 1 : 0
  return `M${x1.toFixed(2)},${y1.toFixed(2)} A${outer},${outer} 0 ${large} 1 ${x2.toFixed(2)},${y2.toFixed(2)} L${x3.toFixed(2)},${y3.toFixed(2)} A${inner},${inner} 0 ${large} 0 ${x4.toFixed(2)},${y4.toFixed(2)} Z`
}

function donutSvg(items, total, taskShare) {
  if (items.length === 0 || total <= 0) {
    return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" width="40mm" height="40mm"><circle cx="50" cy="50" r="48" fill="#eef2f8"/><circle cx="50" cy="50" r="25" fill="#ffffff"/><text x="50" y="52" text-anchor="middle" font-size="10" fill="#94a3b8">无数据</text></svg>`
  }
  let angle = 0
  const slices = items.map(item => {
    const sweep = (item.tokens / total) * 360
    const start = angle
    angle += sweep
    if (sweep >= 359.999) {
      return `<circle cx="50" cy="50" r="48" fill="${item.color}"/><circle cx="50" cy="50" r="25" fill="#ffffff"/>`
    }
    return `<path d="${annulusPath(50, 50, 48, 25, start, angle)}" fill="${item.color}" stroke="#ffffff" stroke-width="0.5"/>`
  }).join('')
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" width="40mm" height="40mm" role="img" aria-label="任务 token 占比">${slices}<text x="50" y="49" text-anchor="middle" font-size="17" font-weight="700" fill="#0f172a">${taskShare.toFixed(1)}%</text><text x="50" y="60" text-anchor="middle" font-size="7" fill="#64748b">任务占比</text></svg>`
}

/** Open PRs still need a human review/merge; this is the reference report's risk strip. */
function prRisk(data) {
  const numbers = [...new Set(data.openPrs.map(p => String(p.number)))]
  if (numbers.length === 0) return ''
  const shown = numbers.slice(0, 3).map(n => `#${n}`).join('、')
  return `${numbers.length} 个 PR 待人工评审/合并（${shown}${numbers.length > 3 ? ' 等' : ''}）`
}

function reportHtml(data, content) {
  const t = data.totals
  const daily = data.days === 1
  const period = daily ? '今日' : '本周'
  const nextPeriod = daily ? '明日' : '下周'
  const completed = data.tickets.filter(item => item.status === '已完成' || item.status === '已合并').length
  const subtitle = `${period} DSH 处理 ${data.tickets.length} 项任务，${completed} 项完成`
  const eyebrowPeriod = daily ? `${data.end} · 日报` : `${data.start} ~ ${hyphenDate(data.end)} · 周报`
  const footerPeriod = daily ? `今日 ${hyphenDate(data.end)}` : `本周 ${hyphenDate(data.start)} ~ ${hyphenDate(data.end)}`
  const kpis = [
    { value: fmtTokens(t.tokens), label: 'Token 用量' },
    { value: fmtInt(t.requests), label: '模型请求' },
    { value: String(t.sessions), label: '会话数' },
    { value: `${t.cacheHit.toFixed(1)}%`, label: '缓存命中' },
    { value: fmtCostShort(t.cost), label: 'API 费用' },
  ]
  const highlightRows = content.highlights.map((h, i) => `
      <div class="hl">
        <div class="num">${String(i + 1).padStart(2, '0')}</div>
        <div class="body"><div class="hl-title">${esc(h.title)}</div><div class="hl-detail">${esc(h.detail)}</div></div>
      </div>`).join('')
  const nextItems = content.next.map((item, i) => `
      <div class="next-item"><span class="next-num">${String(i + 1).padStart(2, '0')}</span><span>${esc(item)}</span></div>`).join('')
  const riskText = content.risk || prRisk(data)
  const risk = riskText
    ? `<div class="section"><div class="risk"><span class="risk-label">风险 / 阻塞</span><span>${esc(riskText)}</span></div></div>`
    : ''
  const pie = pieData(data)
  const pieRows = pie.items.map(item => {
    const share = pie.total > 0 ? (item.tokens / pie.total) * 100 : 0
    const percent = share >= 10 ? share.toFixed(0) : share.toFixed(1)
    return `<div class="task-row"><i style="background:${item.color}"></i><span class="task-name">${esc(clean(item.label, 12))}</span><span class="task-pct">${percent}%</span></div>`
  }).join('')
  const parts = [
    { key: 'cache', label: '缓存读取', value: t.cacheRead, color: '#4d6bfe' },
    { key: 'uncached', label: '未缓存输入', value: t.uncached, color: '#f59e0b' },
    { key: 'output', label: '输出', value: t.output, color: '#94a3b8' },
  ].filter(p => p.value > 0)
  const partTotal = parts.reduce((sum, p) => sum + p.value, 0) || 1
  const tokenBar = parts.map(p => `<span class="seg" style="width:${((p.value / partTotal) * 100).toFixed(2)}%;background:${p.color}"></span>`).join('')
  const tokenLegend = parts.map(p => `<span class="legend-item"><i style="background:${p.color}"></i>${esc(p.label)} ${fmtTokens(p.value)}</span>`).join('')
  const modelRows = data.models.map(model => `<div class="model-row"><span class="model-name">${esc(model.model)}</span><span>${fmtTokens(model.tokens)}</span><span>${model.requests > 0 ? fmtInt(model.requests) : '—'}</span><span>${fmtCost(model.cost)}</span></div>`).join('')
  const costNote = '按 DeepSeek 官方 CNY 价目（2026-11 核对）估算：工作日 9–12 / 14–18 为高峰，空闲价减半；缓存写入并入未命中输入；未计法定节假日与 Search/Fetch，以实际账单为准。'

  return `<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><style>
@page { size: A4; margin: 0; }
* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; }
body { font-family: 'Noto Sans CJK SC', 'Noto Sans CJK SC Medium', sans-serif; color: #0f172a; -weasy-hyphens: none; }
.page { position: relative; width: 210mm; height: 297mm; padding: 14mm 16mm 13mm; background: #f7f8fb; overflow: hidden; }
.eyebrow { display: flex; justify-content: space-between; align-items: baseline; font-size: 8pt; letter-spacing: 1.2pt; color: #64748b; }
.eyebrow .brand { color: #4d6bfe; font-weight: 700; letter-spacing: 1.8pt; }
h1 { font-size: 26pt; line-height: 1.15; letter-spacing: -0.6pt; margin: 4mm 0 2mm; font-weight: 700; }
.subtitle { font-size: 11.5pt; line-height: 1.5; color: #475569; margin: 0 0 4.5mm; }
.kpis { display: flex; gap: 2.5mm; margin-bottom: 4mm; }
.kpi { flex: 1; background: #ffffff; border: 1px solid #e6eaf2; border-radius: 3mm; padding: 3.9mm 3.4mm 3.5mm; }
.kpi .value { font-size: 16.5pt; font-weight: 700; letter-spacing: -0.3pt; white-space: nowrap; }
.kpi .label { font-size: 8pt; color: #64748b; margin-top: 1.2mm; }
.token-cards { display: flex; gap: 4mm; margin-bottom: 3.4mm; }
.token-card { flex: 1; min-width: 0; background: #ffffff; border: 1px solid #e6eaf2; border-radius: 3mm; padding: 3.5mm 4mm 3.2mm; display: flex; flex-direction: column; }
.token-head { display: flex; justify-content: space-between; align-items: baseline; font-size: 8pt; color: #475569; margin-bottom: 2.3mm; }
.token-note { color: #94a3b8; font-size: 7.1pt; }
.token-bar { display: flex; height: 3.2mm; border-radius: 1.6mm; overflow: hidden; background: #eef2f8; }
.token-bar .seg { display: block; height: 3.2mm; }
.token-legend { display: flex; flex-wrap: wrap; gap: 1mm 4mm; margin-top: 2.1mm; font-size: 7.3pt; color: #64748b; }
.legend-item i { display: inline-block; width: 2.2mm; height: 2.2mm; border-radius: 0.6mm; margin-right: 1.2mm; }
.token-summary { margin-top: auto; padding-top: 3mm; font-size: 7.3pt; color: #94a3b8; }
.task-pie-row { display: flex; gap: 3.4mm; align-items: center; }
.donut { flex: none; width: 40mm; }
.donut svg { display: block; }
.task-legend { flex: 1; min-width: 0; display: flex; flex-direction: column; gap: 1.2mm; font-size: 7.7pt; color: #334155; }
.task-row { display: flex; align-items: center; gap: 1.6mm; line-height: 1.35; }
.task-row i { flex: none; width: 2.2mm; height: 2.2mm; border-radius: 0.6mm; }
.task-name { flex: 1; overflow: hidden; white-space: nowrap; text-overflow: ellipsis; color: #1e293b; }
.task-pct { width: 10mm; text-align: right; font-weight: 700; }
.model-card { background: #ffffff; border: 1px solid #e6eaf2; border-radius: 3mm; padding: 3.2mm 4mm 3mm; margin-bottom: 4.6mm; }
.model-table { margin-top: 0.4mm; }
.model-row { display: flex; align-items: baseline; font-size: 8.1pt; line-height: 1.6; color: #334155; border-top: 1px solid #f1f5f9; }
.model-row > span:nth-child(1) { flex: 1; min-width: 0; overflow: hidden; white-space: nowrap; text-overflow: ellipsis; }
.model-row > span:nth-child(2) { width: 30mm; text-align: right; }
.model-row > span:nth-child(3) { width: 20mm; text-align: right; }
.model-row > span:nth-child(4) { width: 22mm; text-align: right; }
.model-head { font-size: 6.9pt; color: #94a3b8; line-height: 1.4; border-top: none; }
.model-head > span { font-weight: 400; }
.model-total { font-weight: 700; border-top: 1px solid #e2e8f0; }
.model-note { margin-top: 1.4mm; font-size: 6.7pt; line-height: 1.4; color: #94a3b8; }
.section { margin-top: 4.6mm; }
.section-head { display: flex; align-items: center; gap: 2.5mm; margin-bottom: 1.4mm; }
.section-head .bar { width: 3mm; height: 3mm; border-radius: 1mm; background: #4d6bfe; }
.section-head .title { font-size: 9pt; font-weight: 700; letter-spacing: 1pt; color: #4d6bfe; }
.hl { display: flex; gap: 4mm; padding: 3mm 0; border-bottom: 1px solid #e9edf5; }
.hl:last-child { border-bottom: none; }
.hl .num { width: 7mm; font-size: 9pt; font-weight: 700; color: #94a3b8; padding-top: 0.6mm; }
.hl .body { flex: 1; }
.hl-title { font-size: 11.5pt; font-weight: 700; margin-bottom: 0.7mm; }
.hl-detail { font-size: 9.4pt; line-height: 1.5; color: #475569; }
.risk { display: flex; gap: 4mm; align-items: baseline; background: #fff7ed; border: 1px solid #fed7aa; border-radius: 2.5mm; padding: 3.6mm 5mm; font-size: 9.4pt; color: #9a3412; line-height: 1.4; }
.risk-label { font-size: 8pt; letter-spacing: 1pt; font-weight: 700; white-space: nowrap; opacity: 0.8; }
.next { display: flex; gap: 3mm; }
.next-item { flex: 1; background: #ffffff; border: 1px solid #e6eaf2; border-radius: 2.5mm; padding: 4mm 4.4mm; font-size: 9.4pt; line-height: 1.5; color: #1e293b; }
.next-num { display: inline-block; color: #4d6bfe; font-weight: 700; margin-right: 1.6mm; }
.footer { position: absolute; left: 16mm; right: 16mm; bottom: 11mm; display: flex; justify-content: space-between; font-size: 7.3pt; color: #94a3b8; border-top: 1px solid #e6eaf2; padding-top: 2.4mm; }
</style></head>
<body><div class="page">
  <div class="eyebrow"><span class="brand">DSH ${daily ? 'DAILY' : 'WEEKLY'} BRIEF</span><span>${esc(eyebrowPeriod)}</span></div>
  <h1>${period}工作总结</h1>
  <p class="subtitle">${esc(subtitle)}</p>
  <div class="kpis">${kpis.map(k => `<div class="kpi"><div class="value">${esc(k.value)}</div><div class="label">${esc(k.label)}</div></div>`).join('')}</div>
  <div class="token-cards">
    <div class="token-card">
      <div class="token-head"><span>Token 构成</span><span class="token-note">provider 计费口径</span></div>
      <div class="token-bar">${tokenBar}</div>
      <div class="token-legend">${tokenLegend}</div>
      <div class="token-summary">总 ${fmtTokens(t.tokens)} token · ${fmtInt(t.requests)} 次请求 · ${t.sessions} 个会话</div>
    </div>
    <div class="token-card">
      <div class="token-head"><span>任务 Token 占比</span><span class="token-note">任务合计 ${pie.taskShare.toFixed(1)}%</span></div>
      <div class="task-pie-row">
        <div class="donut">${donutSvg(pie.items, pie.total, pie.taskShare)}</div>
        <div class="task-legend">${pieRows}</div>
      </div>
    </div>
  </div>
  <div class="model-card">
    <div class="token-head"><span>模型 Token / 费用</span><span class="token-note">API 费用 ≈ ${fmtCost(t.cost)}</span></div>
    <div class="model-table">
      <div class="model-row model-head"><span>模型</span><span>Token</span><span>请求</span><span>费用</span></div>
      ${modelRows}
      <div class="model-row model-total"><span>合计</span><span>${fmtTokens(t.tokens)}</span><span>${fmtInt(t.requests)}</span><span>${fmtCost(t.cost)}</span></div>
    </div>
    <div class="model-note">${esc(costNote)}</div>
  </div>
  <div>
    <div class="section-head"><span class="bar"></span><span class="title">${period}重点</span></div>
    ${highlightRows}
  </div>
  ${risk}
  <div class="section">
    <div class="section-head"><span class="bar"></span><span class="title">${nextPeriod}重点</span></div>
    <div class="next">${nextItems}</div>
  </div>
  <div class="footer"><span>数据：DSH 会话日志（provider usage）· superagent 工单</span><span>${esc(footerPeriod)} · 数据截至 ${localTime()} · DSH 自动生成</span></div>
</div></body></html>`
}

function renderImage(html, base) {
  const dir = path.join(APP_DIR, 'last')
  fs.mkdirSync(dir, { recursive: true })
  const htmlPath = path.join(dir, `${base}.html`)
  const pdfPath = path.join(dir, `${base}.pdf`)
  const pngPrefix = path.join(dir, base)
  const pngPath = `${pngPrefix}.png`
  fs.writeFileSync(htmlPath, html)
  execFileSync(weasyprint, [htmlPath, pdfPath], { timeout: 120_000, stdio: ['ignore', 'pipe', 'pipe'] })
  const info = execFileSync(pdfinfo, [pdfPath], { encoding: 'utf8', timeout: 20_000 })
  const pages = Number((info.match(/^Pages:\s+(\d+)/m) || [])[1] || 0)
  if (pages !== 1) throw new Error(`report rendered to ${pages} pages; expected 1`)
  execFileSync(pdftoppm, ['-png', '-r', '150', '-singlefile', pdfPath, pngPrefix], { timeout: 60_000, stdio: ['ignore', 'pipe', 'pipe'] })
  if (!fs.existsSync(pngPath)) throw new Error('pdftoppm produced no image')
  return { pngPath, htmlPath, pdfPath }
}

function larkEnv() {
  return { ...process.env, PATH: [path.join(HOME, '.local/bin'), '/usr/local/bin', '/usr/bin', '/bin'].join(':') }
}

function sendImage(imagePath, end, mode) {
  if (!chatId) throw new Error('missing chat id (pass --chat-id or set DSH_DAILY_SUMMARY_CHAT)')
  const key = `dsh-${mode}-img-${end.replace(/-/g, '')}${args.resend ? `-${Date.now()}` : ''}`
  const output = execFileSync(larkCli, ['im', '+messages-send', '--chat-id', chatId, '--image', path.basename(imagePath), '--idempotency-key', key], {
    cwd: path.dirname(imagePath), encoding: 'utf8', timeout: 120_000, env: larkEnv(),
  })
  const parsed = JSON.parse(output)
  if (parsed?.ok !== true) throw new Error(JSON.stringify(parsed?.error ?? parsed))
  return parsed
}

function sendTextFallback(data, content, end, mode, reason) {
  const nextLabel = data.days === 1 ? '明日' : '下周'
  const lines = [
    `📊 ${data.days === 1 ? '今日' : '本周'}工作总结 · ${rangeText(data.start, end)}`,
    '',
    content.headline,
    '',
    ...content.highlights.map((h, i) => `${i + 1}. **${h.title}** — ${h.detail}`),
    '',
    content.risk ? `⚠️ ${content.risk}` : '✅ 无阻塞。',
    '',
    `Token ${fmtTokens(data.totals.tokens)} · API 费用约 ${fmtCost(data.totals.cost)}`,
    '',
    `**${nextLabel}重点**`,
    ...content.next.map((n, i) => `${i + 1}. ${n}`),
    '',
    `（图片版生成失败：${reason}）`,
  ]
  const key = `dsh-${mode}-text-${end.replace(/-/g, '')}${args.resend ? `-${Date.now()}` : ''}`
  const output = execFileSync(larkCli, ['im', '+messages-send', '--chat-id', chatId, '--markdown', '-', '--idempotency-key', key], {
    input: lines.join('\n'), encoding: 'utf8', timeout: 120_000, env: larkEnv(),
  })
  const parsed = JSON.parse(output)
  if (parsed?.ok !== true) throw new Error(JSON.stringify(parsed?.error ?? parsed))
  return parsed
}

async function main() {
  const end = args.date || localDate()
  if (!/^\d{4}-\d{2}-\d{2}$/.test(end)) throw new Error(`invalid date: ${end}`)
  const days = Number(args.days || 7)
  if (!Number.isInteger(days) || days < 1 || days > 31) throw new Error(`invalid --days: ${args.days} (1-31)`)
  const range = reportWindow(end, days)
  const mode = days === 1 ? 'daily' : 'weekly'
  const data = await buildReportData(range)
  const curated = args.content ? JSON.parse(fs.readFileSync(args.content, 'utf8')) : await curate(data)
  const content = validContent(curated) ? normalizeContent(curated) : localContent(data)
  content.next = planItems(data)
  content.risk = riskFromData(data) || prRisk(data)
  fs.mkdirSync(path.join(APP_DIR, 'last'), { recursive: true })
  fs.writeFileSync(path.join(APP_DIR, 'last', `content-${mode}.json`), JSON.stringify(content, null, 2))
  fs.writeFileSync(path.join(APP_DIR, 'last', 'content.json'), JSON.stringify(content, null, 2))
  const html = reportHtml(data, content)
  const base = `${mode}-summary-${range.end}`

  if (args.dryRun) {
    const rendered = renderImage(html, base)
    console.log(JSON.stringify(content, null, 2))
    console.log(`\n[dry-run] ${mode} ${range.start}..${range.end} tokens=${data.totals.tokens} cost=${data.totals.cost.toFixed(2)} requests=${data.totals.requests} sessions=${data.totals.sessions} tasks=${data.tasks.length} otherSessions=${data.other.sessions} tickets=${data.tickets.length} open=${data.open.length} openPrs=${data.openPrs.length}`)
    console.log(`models: ${data.models.map(m => `${m.model} ${m.requests}req ${m.tokens}tok ¥${m.cost.toFixed(2)}`).join(' | ')}`)
    console.log(`image: ${rendered.pngPath}`)
    return
  }

  try {
    const rendered = renderImage(html, base)
    const sent = sendImage(rendered.pngPath, range.end, mode)
    console.log(`[${new Date().toISOString()}] sent ${mode} report for ${range.start}..${range.end}: ${JSON.stringify(sent?.data ?? sent)} image=${rendered.pngPath}`)
  } catch (error) {
    const reason = clean(error instanceof Error ? error.message : String(error), 80)
    console.error(`[${new Date().toISOString()}] image report failed for ${range.start}..${range.end}: ${reason}`)
    if (!chatId) throw error
    const sent = sendTextFallback(data, content, range.end, mode, reason)
    console.error(`[${new Date().toISOString()}] text fallback sent: ${JSON.stringify(sent?.data ?? sent)}`)
    process.exitCode = 1
  }
}

await main()
