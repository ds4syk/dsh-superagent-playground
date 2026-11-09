#!/usr/bin/env node
/**
 * dsh-daily-summary: build a one-page daily work report image and post it to a Feishu chat.
 *
 * Pipeline: collect local state -> curate the brief -> render HTML -> WeasyPrint PDF -> PNG
 * -> lark-cli image message. If the image pipeline fails, a short text fallback is sent.
 *
 * Data sources:
 *  - DSH session logs: <home>/.dsh/sessions/<workspace>/<session>/session.v4.jsonl.zstd
 *    (the durable record that carries each step's provider-reported usage). The Context Insights
 *    projection cache is only a fallback: it is refreshed on demand by the web UI, so a cron run
 *    can see it stale or empty.
 *  - dsh-superagent tickets: <stateDir>/orchestrator.sqlite
 *  - Open pull requests of the configured repository (optional; skipped when gh fails)
 *
 * Usage:
 *   node daily-summary.mjs [--date YYYY-MM-DD] [--dry-run] [--no-llm] [--resend] [--chat-id oc_xxx]
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
  const args = { dryRun: false, llm: true }
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i]
    if (a === '--dry-run') args.dryRun = true
    else if (a === '--resend') args.resend = true
    else if (a === '--no-llm') args.llm = false
    else if (a === '--date') args.date = argv[++i]
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

const sessionsRoot = path.join(HOME, '.dsh/sessions')
const zstdcat = process.env.DSH_DAILY_SUMMARY_ZSTD || (fs.existsSync('/usr/bin/zstdcat') ? '/usr/bin/zstdcat' : 'zstdcat')
// Only these events carry what the report needs; the rest (mostly tool output) is skipped before
// JSON.parse, which keeps the daily scan of the largest logs cheap.
const trackedEvent = /"type":\s*"(?:assistant\/message|step\/start|session\/end-seed|session\/title|session)"/

/** Sessions from the Context Insights projection cache; kept as a fallback for the durable logs. */
function collectSessionsFromCache(date) {
  const sessions = []
  let files = []
  try { files = fs.readdirSync(sessionsDir).filter(f => f.endsWith('.json')) } catch { return sessions }
  for (const file of files) {
    let record
    try { record = JSON.parse(fs.readFileSync(path.join(sessionsDir, file), 'utf8')).record } catch { continue }
    const rows = record?.rows ?? {}
    const day = rows?.contextActivity?.val?.days?.[date]
    if (!day || !Number.isFinite(day.tokens) || day.tokens <= 0) continue
    const breakdown = { uncached: 0, cacheRead: 0, cacheWrite: 0, output: 0 }
    for (const models of Object.values(day.cost ?? {})) {
      for (const buckets of Object.values(models ?? {})) {
        for (const bucket of Object.values(buckets ?? {})) {
          for (const key of Object.keys(breakdown)) breakdown[key] += Number(bucket?.[key] || 0)
        }
      }
    }
    const first = clean(rows?.titleInput?.val?.first?.text)
    const rawTitle = clean(rows?.title?.val)
    let title = rawTitle || first || '未命名会话'
    // The UI truncates long first prompts to a short title; use the first sentence then.
    if (rawTitle && rawTitle.length >= 18 && first.startsWith(rawTitle)) title = first.split(/[。\n]/)[0] || first
    // Generic subagent/worker prompts are not useful work labels.
    if (/^You are working on\b/i.test(title)) title = ''
    sessions.push({
      id: file.replace(/\.json$/, ''),
      tokens: day.tokens,
      requests: Number(day.requests || 0),
      breakdown,
      title,
      workspace: rows?.workingDirectory?.val || record?.identity?.cwd || '',
    })
  }
  return sessions
}

/** Session logs that can hold the given day; a log's mtime is the time of its last appended event. */
function sessionLogFiles(date) {
  const since = Date.parse(`${date}T00:00:00+08:00`)
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
 * Fold one session log into the requested day.
 *
 * A step is billed to the day it started, like Context Insights. `session/end-seed` with
 * `inherited: true` closes the copied parent history of a seeded side conversation: those events
 * were already billed in the parent session, so only later events are counted.
 */
async function scanSessionLog({ id, file }, date) {
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
      usages.push({ seq: event.seq, time: event.time, turn: event.data.turn, step: event.data.step, usage })
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
  for (const item of usages) {
    if (item.seq <= inheritedThrough) continue
    const started = stepStarts.get(`${item.turn}:${item.step}`) ?? item.time
    if (localDate(new Date(started)) !== date) continue
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
    day.tokens += uncached + cacheRead + cacheWrite + output
  }
  if (day.tokens <= 0) return null
  if (/^You are working on\b/i.test(title)) title = ''
  return { id, tokens: day.tokens, requests: day.requests, breakdown: day.breakdown, title, workspace: header?.cwd || '' }
}

/** Sessions with the per-day figures, read from the durable session logs. */
async function collectSessions(date) {
  try {
    execFileSync(zstdcat, ['--version'], { stdio: 'ignore', timeout: 10_000 })
    const sessions = []
    let failures = 0
    for (const entry of sessionLogFiles(date)) {
      try {
        const session = await scanSessionLog(entry, date)
        if (session) sessions.push(session)
      } catch (error) {
        failures += 1
        console.error(`session log ${entry.id} skipped: ${clean(error instanceof Error ? error.message : String(error), 120)}`)
      }
    }
    if (failures > 0 && sessions.length === 0) throw new Error(`all ${failures} candidate session log(s) failed`)
    if (sessions.length === 0) {
      // A log-format change would also look empty; the cache is the only other source then.
      const cached = collectSessionsFromCache(date)
      if (cached.length > 0) {
        console.error(`no usage found in session logs for ${date}; using the Context Insights projection cache`)
        return cached
      }
    }
    return sessions
  } catch (error) {
    const reason = clean(error instanceof Error ? error.message : String(error), 120)
    console.error(`session log scan failed (${reason}); falling back to the Context Insights projection cache`)
    return collectSessionsFromCache(date)
  }
}

function workspaceLabel(dir) {
  if (!dir) return '未知工作区'
  if (dir.includes('/.superagent-work/')) return 'superagent 工作区'
  return dir.replace(/^\/home\/syk\//, '~/').replace(/^\/mnt\/storage1\/syk\//, '~/')
}

/** Tickets the superagent touched today, plus the ones still open. */
function readTickets(date) {
  if (!fs.existsSync(ticketDb)) return { today: [], open: [] }
  const db = new DatabaseSync(ticketDb, { readOnly: true })
  const today = []
  const open = []
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
      }
      if (!Number.isFinite(updated)) continue
      if (localDate(new Date(updated)) === date || localDate(new Date(created || updated)) === date) today.push(item)
      if (!item.ended) open.push(item)
    }
  } finally { db.close() }
  today.sort((a, b) => a.updated - b.updated)
  open.sort((a, b) => a.updated - b.updated)
  return { today, open }
}

function collectTickets(date) {
  // The host writes this SQLite database continuously; a busy read gets retried.
  for (let attempt = 1; attempt <= 3; attempt++) {
    try { return readTickets(date) } catch {
      if (attempt === 3) return { today: [], open: [] }
      execFileSync('/bin/sleep', ['1'])
    }
  }
  return { today: [], open: [] }
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
async function buildReportData(date) {
  const sessions = await collectSessions(date)
  const tickets = collectTickets(date)
  const openPrs = openPullRequests()

  const groups = new Map()
  for (const s of sessions) {
    const label = workspaceLabel(s.workspace)
    const g = groups.get(label) || { name: label, sessions: 0, requests: 0, tokens: 0, titleEntries: [] }
    g.sessions += 1
    g.requests += s.requests
    g.tokens += s.tokens
    if (s.title) g.titleEntries.push({ title: s.title, tokens: s.tokens })
    groups.set(label, g)
  }
  const totals = { tokens: 0, requests: 0, sessions: sessions.length, uncached: 0, cacheRead: 0, cacheWrite: 0, output: 0 }
  for (const s of sessions) {
    totals.tokens += s.tokens
    totals.requests += s.requests
    for (const key of ['uncached', 'cacheRead', 'cacheWrite', 'output']) totals[key] += s.breakdown[key]
  }
  const denom = totals.uncached + totals.cacheRead + totals.cacheWrite
  totals.cacheHit = denom > 0 ? Number(((totals.cacheRead / denom) * 100).toFixed(1)) : 0

  return {
    date,
    weekday: weekdayText(date),
    totals,
    workspaces: [...groups.values()]
      .filter(g => g.name !== 'superagent 工作区')
      .sort((a, b) => b.tokens - a.tokens)
      .map(g => {
        const byTitle = new Map()
        for (const entry of g.titleEntries) byTitle.set(entry.title, Math.max(entry.tokens, byTitle.get(entry.title) || 0))
        const titles = [...byTitle.entries()].sort((a, b) => b[1] - a[1]).slice(0, 5).map(([title]) => title)
        return { name: g.name, sessions: g.sessions, requests: g.requests, tokens: g.tokens, titles }
      }),
    tickets: tickets.today.map(t => ({ title: t.title, status: ticketStatus(t), prs: t.prs, result: t.result })),
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

/** Ask the model for a concise, result-first brief; return undefined on any failure. */
async function curate(data) {
  if (!args.llm) return undefined
  const key = apiKey()
  if (!key) return undefined
  const system = [
    '你是资深技术负责人，为团队写每日一页简报。遵循这些原则：',
    '- 倒金字塔：先给结论，再给支撑细节；读者 10 秒能看懂今天最重要的事。',
    '- 只写结果、影响和关键进展；合并同类项；不写过程流水账，不罗列所有会话。',
    '- 亮点 3 条以内；每条主题不超过 10 个字，说明不超过 38 个字。',
    '- 只使用输入数据中的事实；不编造数字或结论。',
    '- 会话标题只代表今天在做的主题，不能据此写「已产出 / 已完成 / 已跑出 / 成功」；只有 todayTickets 中 status 为「已完成」或「已合并」的事项才能写完成。',
    '- 进行中的事项用「推进 / 恢复 / 提交 / 排队 / 等待」等状态词。',
    '- 不要自行推断风险、CI 缺失或流程问题。',
    '- 中文，短句，避免术语堆砌和空话。',
    '- 不需要输出风险或明日重点：系统会从工作项状态自动生成。',
    '只返回 JSON：{"headline":"一句话结论，不超过40字","highlights":[{"title":"主题","detail":"结果或影响，不超过45字"}]}',
  ].join('\n')
  const payload = {
    date: data.date,
    weekday: data.weekday,
    totals: { tokens: data.totals.tokens, requests: data.totals.requests, sessions: data.totals.sessions, cacheHitPercent: data.totals.cacheHit },
    workspaces: data.workspaces.map(g => ({ name: g.name, sessions: g.sessions, requests: g.requests, tokens: g.tokens, titles: g.titles })),
    todayTickets: data.tickets.map(t => ({ title: t.title, status: t.status, prs: t.prs, result: t.result })),
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

function normalizeContent(c) {
  return {
    headline: clean(c.headline, 60),
    highlights: c.highlights.slice(0, 3).map(h => ({ title: clean(h.title, 14), detail: clean(h.detail, 58) })),
    risk: clean(c.risk || '', 46),
  }
}

/** Deterministic fallback when the model is unavailable or answers badly. */
function localContent(data) {
  const completed = data.tickets.filter(t => t.status === '已完成' || t.status === '已合并').length
  const waiting = data.tickets.filter(t => t.status === '待审')
  const blocked = data.tickets.find(t => t.status === '受阻')
  const top = data.workspaces[0]
  const headline = [
    `${data.totals.sessions} 个会话、${fmtInt(data.totals.requests)} 次请求`,
    top ? `主要精力在 ${top.name.replace(/^~\//, '')}` : '',
    `superagent 完成 ${completed} 项${waiting.length ? `，PR ${waiting.flatMap(t => t.prs).slice(0, 2).map(p => `#${p}`).join('、')} 待审` : ''}`,
  ].filter(Boolean).join('；') + '。'
  const highlights = data.workspaces.slice(0, 2).map(g => ({
    title: g.name.replace(/^~\//, '').replace(/^labs\//, ''),
    detail: `${g.sessions} 个会话 / ${fmtInt(g.requests)} 次请求：${g.titles.slice(0, 2).map(t => clean(t, 16)).join('、')}${g.titles.length > 2 ? ' 等' : ''}`,
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
  const all = [...data.open, ...data.tickets]
  const blocked = all.find(t => t.status === '受阻')
  if (blocked) return clean(`受阻：${blocked.title}`, 52)
  const pattern = /拍板|需要(你|用户|人工)|等待(你|用户)|待(你|用户)确认|需要决策/
  for (const item of all) {
    for (const sentence of String(item.result || '').split(/[。\n]/)) {
      if (pattern.test(sentence)) return clean(sentence.replace(/\*\*/g, ''), 52)
    }
  }
  return ''
}

function reportHtml(data, content) {
  const t = data.totals
  const kpis = [
    { value: fmtTokens(t.tokens), label: 'Token 用量' },
    { value: fmtInt(t.requests), label: '模型请求' },
    { value: String(t.sessions), label: '会话数' },
    { value: `${t.cacheHit.toFixed(1)}%`, label: '缓存命中' },
  ]
  const highlightRows = content.highlights.map((h, i) => `
      <div class="hl">
        <div class="num">${String(i + 1).padStart(2, '0')}</div>
        <div class="body"><div class="hl-title">${esc(h.title)}</div><div class="hl-detail">${esc(h.detail)}</div></div>
      </div>`).join('')
  const nextItems = content.next.map((item, i) => `
      <div class="next-item"><span class="next-num">${i + 1}</span><span>${esc(item)}</span></div>`).join('')
  const risk = content.risk
    ? `<div class="section"><div class="risk"><span class="risk-label">风险 / 阻塞</span><span>${esc(content.risk)}</span></div></div>`
    : ''
  const parts = [
    { key: 'cache', label: '缓存读取', value: t.cacheRead, color: '#4d6bfe' },
    { key: 'uncached', label: '未缓存输入', value: t.uncached, color: '#f59e0b' },
    { key: 'output', label: '输出', value: t.output, color: '#94a3b8' },
  ].filter(p => p.value > 0)
  const partTotal = parts.reduce((sum, p) => sum + p.value, 0) || 1
  const tokenBar = parts.map(p => `<span class="seg" style="width:${((p.value / partTotal) * 100).toFixed(2)}%;background:${p.color}"></span>`).join('')
  const legend = parts.map(p => `<span class="legend-item"><i style="background:${p.color}"></i>${esc(p.label)} ${fmtTokens(p.value)}</span>`).join('')

  return `<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><style>
@page { size: A4; margin: 0; }
* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; }
body { font-family: 'Noto Sans CJK SC', 'Noto Sans CJK SC Medium', sans-serif; color: #0f172a; -weasy-hyphens: none; }
.page { position: relative; width: 210mm; height: 297mm; padding: 15mm 16mm 13mm; background: #f7f8fb; overflow: hidden; }
.eyebrow { display: flex; justify-content: space-between; align-items: baseline; font-size: 8pt; letter-spacing: 1.4pt; color: #64748b; }
.eyebrow .brand { color: #4d6bfe; font-weight: 700; letter-spacing: 1.8pt; }
h1 { font-size: 29pt; line-height: 1.16; letter-spacing: -0.6pt; margin: 8mm 0 4.5mm; font-weight: 700; }
.headline { font-size: 12pt; line-height: 1.65; color: #334155; margin: 0 0 9mm; max-width: 174mm; }
.kpis { display: flex; gap: 3mm; margin-bottom: 5mm; }
.kpi { flex: 1; background: #ffffff; border: 1px solid #e6eaf2; border-radius: 3mm; padding: 5mm 4.5mm 4.4mm; }
.kpi .value { font-size: 19pt; font-weight: 700; letter-spacing: -0.3pt; white-space: nowrap; }
.kpi .label { font-size: 8pt; color: #64748b; margin-top: 1.4mm; }
.token-block { background: #ffffff; border: 1px solid #e6eaf2; border-radius: 3mm; padding: 4.4mm 4.5mm; margin-bottom: 10mm; }
.token-head { display: flex; justify-content: space-between; align-items: baseline; font-size: 8pt; color: #475569; margin-bottom: 3mm; }
.token-note { color: #94a3b8; font-size: 7.5pt; }
.token-bar { display: flex; height: 3.2mm; border-radius: 1.6mm; overflow: hidden; background: #eef2f8; }
.token-bar .seg { display: block; height: 3.2mm; }
.legend { display: flex; gap: 5mm; margin-top: 2.6mm; font-size: 8pt; color: #64748b; }
.legend-item i { display: inline-block; width: 2.2mm; height: 2.2mm; border-radius: 0.6mm; margin-right: 1.2mm; }
.section { margin-top: 10mm; }
.section-head { display: flex; align-items: center; gap: 2.5mm; margin-bottom: 2mm; }
.section-head .bar { width: 3mm; height: 3mm; border-radius: 1mm; background: #4d6bfe; }
.section-head .title { font-size: 9pt; font-weight: 700; letter-spacing: 1pt; color: #4d6bfe; }
.hl { display: flex; gap: 4mm; padding: 5mm 0; border-bottom: 1px solid #e9edf5; }
.hl:last-child { border-bottom: none; }
.hl .num { width: 7mm; font-size: 9pt; font-weight: 700; color: #94a3b8; padding-top: 0.6mm; }
.hl .body { flex: 1; }
.hl-title { font-size: 11pt; font-weight: 700; margin-bottom: 0.8mm; }
.hl-detail { font-size: 9.5pt; line-height: 1.55; color: #475569; }
.risk { display: flex; gap: 4mm; align-items: baseline; background: #fff7ed; border: 1px solid #fed7aa; border-radius: 2.5mm; padding: 4.5mm 5mm; font-size: 9.5pt; color: #9a3412; line-height: 1.5; }
.risk-label { font-size: 8pt; letter-spacing: 1pt; font-weight: 700; white-space: nowrap; opacity: 0.8; }
.next { display: flex; gap: 3mm; }
.next-item { flex: 1; background: #ffffff; border: 1px solid #e6eaf2; border-radius: 2.5mm; padding: 5.4mm 4.5mm; font-size: 9.5pt; line-height: 1.55; color: #1e293b; }
.next-num { display: inline-block; color: #4d6bfe; font-weight: 700; margin-right: 1.5mm; }
.footer { position: absolute; left: 16mm; right: 16mm; bottom: 11mm; display: flex; justify-content: space-between; font-size: 7.5pt; color: #94a3b8; border-top: 1px solid #e6eaf2; padding-top: 2.5mm; }
</style></head>
<body><div class="page">
  <div class="eyebrow"><span class="brand">DSH DAILY BRIEF</span><span>${esc(data.date)} · ${esc(data.weekday)}</span></div>
  <h1>今日工作总结</h1>
  <p class="headline">${esc(content.headline)}</p>
  <div class="kpis">${kpis.map(k => `<div class="kpi"><div class="value">${esc(k.value)}</div><div class="label">${esc(k.label)}</div></div>`).join('')}</div>
  <div class="token-block">
    <div class="token-head"><span>Token 构成</span><span class="token-note">provider 计费口径（同 Context Insights）</span></div>
    <div class="token-bar">${tokenBar}</div>
    <div class="legend">${legend}</div>
  </div>
  <div>
    <div class="section-head"><span class="bar"></span><span class="title">今日重点</span></div>
    ${highlightRows}
  </div>
  ${risk}
  <div class="section">
    <div class="section-head"><span class="bar"></span><span class="title">明日重点</span></div>
    <div class="next">${nextItems}</div>
  </div>
  <div class="footer"><span>数据：DSH 会话日志（provider usage）</span><span>数据截至 ${localTime()} · DSH 自动生成</span></div>
</div></body></html>`
}

function renderImage(html, date) {
  const dir = path.join(APP_DIR, 'last')
  fs.mkdirSync(dir, { recursive: true })
  const base = `daily-summary-${date}`
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

function sendImage(imagePath, date) {
  if (!chatId) throw new Error('missing chat id (pass --chat-id or set DSH_DAILY_SUMMARY_CHAT)')
  const key = `dsh-daily-img-${date.replace(/-/g, '')}${args.resend ? `-${Date.now()}` : ''}`
  const output = execFileSync(larkCli, ['im', '+messages-send', '--chat-id', chatId, '--image', path.basename(imagePath), '--idempotency-key', key], {
    cwd: path.dirname(imagePath), encoding: 'utf8', timeout: 120_000, env: larkEnv(),
  })
  const parsed = JSON.parse(output)
  if (parsed?.ok !== true) throw new Error(JSON.stringify(parsed?.error ?? parsed))
  return parsed
}

function sendTextFallback(data, content, date, reason) {
  const lines = [
    `📊 今日工作总结 · ${data.date}（${data.weekday}）`,
    '',
    content.headline,
    '',
    ...content.highlights.map((h, i) => `${i + 1}. **${h.title}** — ${h.detail}`),
    '',
    content.risk ? `⚠️ ${content.risk}` : '✅ 无阻塞。',
    '',
    '**明日重点**',
    ...content.next.map((n, i) => `${i + 1}. ${n}`),
    '',
    `（图片版生成失败：${reason}）`,
  ]
  const key = `dsh-daily-text-${date.replace(/-/g, '')}${args.resend ? `-${Date.now()}` : ''}`
  const output = execFileSync(larkCli, ['im', '+messages-send', '--chat-id', chatId, '--markdown', '-', '--idempotency-key', key], {
    input: lines.join('\n'), encoding: 'utf8', timeout: 120_000, env: larkEnv(),
  })
  const parsed = JSON.parse(output)
  if (parsed?.ok !== true) throw new Error(JSON.stringify(parsed?.error ?? parsed))
  return parsed
}

async function main() {
  const date = args.date || localDate()
  if (!/^\d{4}-\d{2}-\d{2}$/.test(date)) throw new Error(`invalid date: ${date}`)
  const data = await buildReportData(date)
  const curated = args.content ? JSON.parse(fs.readFileSync(args.content, 'utf8')) : await curate(data)
  const content = validContent(curated) ? normalizeContent(curated) : localContent(data)
  content.next = planItems(data)
  content.risk = riskFromData(data)
  fs.mkdirSync(path.join(APP_DIR, 'last'), { recursive: true })
  fs.writeFileSync(path.join(APP_DIR, 'last', 'content.json'), JSON.stringify(content, null, 2))
  const html = reportHtml(data, content)

  if (args.dryRun) {
    const rendered = renderImage(html, date)
    console.log(JSON.stringify(content, null, 2))
    console.log(`\n[dry-run] ${date} tokens=${data.totals.tokens} requests=${data.totals.requests} sessions=${data.totals.sessions} tickets=${data.tickets.length} open=${data.open.length} openPrs=${data.openPrs.length}`)
    console.log(`image: ${rendered.pngPath}`)
    return
  }

  try {
    const rendered = renderImage(html, date)
    const sent = sendImage(rendered.pngPath, date)
    console.log(`[${new Date().toISOString()}] sent daily report for ${date}: ${JSON.stringify(sent?.data ?? sent)} image=${rendered.pngPath}`)
  } catch (error) {
    const reason = clean(error instanceof Error ? error.message : String(error), 80)
    console.error(`[${new Date().toISOString()}] image report failed for ${date}: ${reason}`)
    if (!chatId) throw error
    const sent = sendTextFallback(data, content, date, reason)
    console.error(`[${new Date().toISOString()}] text fallback sent: ${JSON.stringify(sent?.data ?? sent)}`)
    process.exitCode = 1
  }
}

await main()
