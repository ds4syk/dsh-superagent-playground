#!/usr/bin/env node
/**
 * dsh-daily-summary: build a daily DeepSeek Harness work summary and post it to a Feishu chat.
 *
 * Data sources:
 *  - Context Insights session projections: <home>/.dsh/storages/session_projcache/sessions/*.json
 *    (the same per-day billed-token figures the dsh-context "Context Insights" page shows)
 *  - dsh-superagent tickets: <stateDir>/orchestrator.sqlite
 *  - Open pull requests of the configured repository (optional; skipped when gh fails)
 *
 * Usage:
 *   node daily-summary.mjs [--date YYYY-MM-DD] [--dry-run] [--chat-id oc_xxx]
 *
 * Environment overrides: DSH_DAILY_SUMMARY_CHAT, DSH_DAILY_SUMMARY_REPO,
 * DSH_DAILY_SUMMARY_LARK_CLI, DSH_DAILY_SUMMARY_GH, DSH_DAILY_SUMMARY_STATE_DIR.
 */
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { execFileSync } from 'node:child_process'
import { DatabaseSync } from 'node:sqlite'

const TZ = 'Asia/Shanghai'
const HOME = os.homedir()

function parseArgs(argv) {
  const args = { dryRun: false }
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i]
    if (a === '--dry-run') args.dryRun = true
    else if (a === '--date') args.date = argv[++i]
    else if (a === '--chat-id') args.chat = argv[++i]
    else if (a === '--repo') args.repo = argv[++i]
    else if (a === '--lark-cli') args.lark = argv[++i]
    else if (a === '--gh') args.gh = argv[++i]
    else if (a === '--state-dir') args.stateDir = argv[++i]
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
const sessionsDir = path.join(HOME, '.dsh/storages/session_projcache/sessions')
const ticketDb = path.join(stateDir, 'orchestrator.sqlite')

function localDate(value = new Date()) {
  return new Intl.DateTimeFormat('en-CA', { timeZone: TZ, year: 'numeric', month: '2-digit', day: '2-digit' }).format(value)
}
function localTime(value = new Date()) {
  return new Intl.DateTimeFormat('en-GB', { timeZone: TZ, hour: '2-digit', minute: '2-digit', hour12: false }).format(value)
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

/** Sessions, with the per-day figures Context Insights computes. */
function collectSessions(date) {
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
    if (rawTitle && rawTitle.length >= 18 && first.startsWith(rawTitle)) {
      title = first.split(/[。\n]/)[0] || first
    }
    // Generic subagent/worker prompts are not useful work labels.
    if (/^You are working on\b/i.test(title)) title = ''
    sessions.push({
      id: file.replace(/\.json$/, ''),
      tokens: day.tokens,
      requests: Number(day.requests || 0),
      breakdown,
      title,
      turns: Number(rows?.sessionStats?.val?.turns || 0),
      workspace: rows?.workingDirectory?.val || record?.identity?.cwd || '',
    })
  }
  return sessions
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
      for (const match of String(report?.summary || '').matchAll(/https:\/\/github\.com\/[^\s)]+\/pull\/\d+/g)) prs.add(match[0])
      for (const follow of ticket.follows || []) {
        const m = /^github:(\d+)$/.exec(follow?.thread || '')
        if (m && repo) prs.add(`https://github.com/${repo}/pull/${m[1]}`)
      }
      const item = {
        id: ticket.id,
        title: clean(ticket.title, 80),
        created,
        updated,
        ended: ticket.ended || null,
        assignment: assignment?.state || '',
        reportState: report?.state || '',
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
    if (ticket.ended.outcome === 'merged') return '✅ 已合并'
    if (ticket.ended.outcome === 'closed') return '⛔ 已关闭'
    return '✅ 已完成'
  }
  if (ticket.reportState === 'blocked' || ticket.reportState === 'failed') return '⚠️ 受阻'
  if (ticket.reportState === 'ready') return '🔍 待审'
  return '🔄 进行中'
}
function prNumbers(urls) { return urls.map(u => `#${u.split('/').pop()}`) }
function prLinks(urls) { return urls.map(u => `[PR ${prNumbers([u])[0]}](${u})`).join('、') }

function buildMessage(date, sessions, tickets, openPrs) {
  const lines = []
  const [, month, day] = date.split('-').map(Number)
  lines.push(`📊 **今日工作总结 · ${month}月${day}日（${weekdayText(date)}）**`)
  lines.push('')

  const groups = new Map()
  for (const s of sessions) {
    const label = workspaceLabel(s.workspace)
    const g = groups.get(label) || { tokens: 0, requests: 0, sessions: [] }
    g.tokens += s.tokens
    g.requests += s.requests
    g.sessions.push(s)
    groups.set(label, g)
  }
  const total = { tokens: 0, requests: 0, uncached: 0, cacheRead: 0, cacheWrite: 0, output: 0, sessions: sessions.length }
  for (const s of sessions) {
    total.tokens += s.tokens
    total.requests += s.requests
    for (const key of ['uncached', 'cacheRead', 'cacheWrite', 'output']) total[key] += s.breakdown[key]
  }

  lines.push('**一、今日工作**')
  const workLines = []
  for (const [label, group] of [...groups.entries()].filter(([label]) => label !== 'superagent 工作区').sort((a, b) => b[1].tokens - a[1].tokens)) {
    workLines.push(`- **${label}**（${group.sessions.length} 个会话 / ${fmtInt(group.requests)} 次请求）`)
    const shown = [...group.sessions].filter(s => s.title).sort((a, b) => b.tokens - a.tokens).slice(0, 5)
    for (const s of shown) {
      workLines.push(`  - ${clean(s.title, 48)}（${fmtTokens(s.tokens)} tokens）`)
    }
    if (group.sessions.length > shown.length) workLines.push(`  - 另有 ${group.sessions.length - shown.length} 个会话`)
  }
  if (tickets.today.length > 0) {
    workLines.push('- **superagent 工作项**')
    for (const t of tickets.today.slice(0, 8)) {
      const pr = t.prs.length > 0 ? ` → ${prLinks(t.prs.slice(0, 2))}` : ''
      workLines.push(`  - ${ticketStatus(t)} ${t.title}${pr}`)
    }
    if (tickets.today.length > 8) workLines.push(`  - 另有 ${tickets.today.length - 8} 个工作项`)
  }
  if (workLines.length === 0) workLines.push('- 今天没有记录到 DSH 活动。')
  lines.push(...workLines)
  lines.push('')

  lines.push('**二、Token 用量**（Context Insights 口径）')
  if (total.tokens > 0) {
    const denom = total.uncached + total.cacheRead + total.cacheWrite
    const hit = denom > 0 ? ((total.cacheRead / denom) * 100).toFixed(1) : '0.0'
    lines.push(`- 总计 **${fmtInt(total.tokens)} tokens**（${fmtTokens(total.tokens)}），${fmtInt(total.requests)} 次请求，${total.sessions} 个会话`)
    if (total.uncached + total.cacheRead + total.cacheWrite + total.output > 0) {
      lines.push(`- 构成：缓存读取 ${fmtTokens(total.cacheRead)}（命中率 ${hit}%）、未缓存输入 ${fmtTokens(total.uncached)}、输出 ${fmtTokens(total.output)}`)
    }
    const ws = [...groups.entries()].sort((a, b) => b[1].tokens - a[1].tokens)
    const shown = ws.slice(0, 4)
    const rest = ws.slice(4).reduce((sum, [, g]) => sum + g.tokens, 0)
    lines.push(`- 按工作区：${shown.map(([label, g]) => `${label} ${fmtTokens(g.tokens)}`).join('｜')}${rest > 0 ? `｜其他 ${fmtTokens(rest)}` : ''}`)
    const busiest = [...sessions].sort((a, b) => b.tokens - a.tokens)[0]
    if (busiest) lines.push(`- 单会话最高：${clean(busiest.title, 36)}（${fmtTokens(busiest.tokens)} tokens）`)
  } else {
    lines.push('- 今天没有记录到 token 用量。')
  }
  lines.push('')

  lines.push('**三、明日规划**')
  const plans = []
  const mentionedPrs = new Set()
  for (const t of tickets.open) {
    for (const pr of t.prs) mentionedPrs.add(pr.split('/').pop())
    if (t.reportState === 'ready' && t.prs.length > 0) plans.push(`审阅并处理 ${prLinks(t.prs.slice(0, 2))}（${t.title}）`)
    else if (t.reportState === 'blocked' || t.reportState === 'failed') plans.push(`处理阻塞：${t.title}`)
    else plans.push(`继续：${t.title}`)
  }
  for (const pr of openPrs) {
    if (mentionedPrs.has(String(pr.number))) continue
    plans.push(`审阅待审 ${prLinks([pr.url])}：${clean(pr.title, 50)}`)
  }
  if (plans.length === 0) plans.push('暂无进行中的工作项。')
  lines.push(...plans.slice(0, 6).map((p, i) => `${i + 1}. ${p}`))
  lines.push('')
  lines.push(`—— 数据截至 ${localTime()} · DSH 自动生成`)

  let message = lines.join('\n')
  if (message.length > 8000) message = `${message.slice(0, 7980)}\n…（内容过长已截断）`
  return message
}

function send(message, date) {
  if (!chatId) throw new Error('missing chat id (pass --chat-id or set DSH_DAILY_SUMMARY_CHAT)')
  const key = `dsh-daily-${date.replace(/-/g, '')}`
  let lastError
  for (let attempt = 1; attempt <= 3; attempt++) {
    try {
      const output = execFileSync(larkCli, ['im', '+messages-send', '--chat-id', chatId, '--markdown', '-', '--idempotency-key', key], {
        input: message, encoding: 'utf8', timeout: 90_000,
        env: { ...process.env, PATH: [path.join(HOME, '.local/bin'), '/usr/local/bin', '/usr/bin', '/bin'].join(':') },
      })
      const parsed = JSON.parse(output)
      if (parsed?.ok !== true) throw new Error(JSON.stringify(parsed?.error ?? parsed))
      return parsed
    } catch (error) {
      lastError = error
      if (attempt < 3) execFileSync('/bin/sleep', [attempt === 1 ? '5' : '15'])
    }
  }
  throw lastError
}

const date = args.date || localDate()
if (!/^\d{4}-\d{2}-\d{2}$/.test(date)) throw new Error(`invalid date: ${date}`)
const sessions = collectSessions(date)
const tickets = collectTickets(date)
const openPrs = openPullRequests()
const message = buildMessage(date, sessions, tickets, openPrs)

if (args.dryRun) {
  console.log(message)
  console.log(`\n[dry-run] date=${date} sessions=${sessions.length} tickets=${tickets.today.length} openPrs=${openPrs.length} chat=${chatId || '(unset)'}`)
} else {
  const sent = send(message, date)
  console.log(`[${new Date().toISOString()}] sent daily summary for ${date}: ${JSON.stringify(sent?.data ?? sent)}`)
}
