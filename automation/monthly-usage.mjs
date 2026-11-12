#!/usr/bin/env node
/**
 * dsh-monthly-usage: tally DSH token usage and CNY list-price cost over a rolling window.
 *
 * Data source: durable session logs
 *   <home>/.dsh/sessions/<workspace>/<session>/session.v4.jsonl.zstd
 * which carry each step's provider-reported usage. This is the same 口径 as
 * automation/daily-summary.mjs (Context Insights): a step is billed to the day it started,
 * tokens = uncached input + cache read + cache write + output, cache write is billed as a
 * cache miss, and the price band follows the Beijing-time peak rules.
 *
 * Usage:
 *   node monthly-usage.mjs [--days 30] [--end YYYY-MM-DD] [--json out.json] [--verbose]
 *
 * --end is the last day of the window (inclusive), default today in Asia/Shanghai.
 */
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import readline from 'node:readline'
import { spawn } from 'node:child_process'

const TZ = 'Asia/Shanghai'
const HOME = os.homedir()
const sessionsRoot = path.join(HOME, '.dsh/sessions')
const zstdcat = process.env.DSH_MONTHLY_USAGE_ZSTD || (fs.existsSync('/usr/bin/zstdcat') ? '/usr/bin/zstdcat' : 'zstdcat')

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
function modelPrice(model) { return MODEL_PRICES[model] || DEFAULT_MODEL_PRICE }

const dateFormatter = new Intl.DateTimeFormat('en-CA', { timeZone: TZ, year: 'numeric', month: '2-digit', day: '2-digit' })
const timeFormatter = new Intl.DateTimeFormat('en-GB', { timeZone: TZ, hour: '2-digit', minute: '2-digit', hour12: false })
const weekdayFormatter = new Intl.DateTimeFormat('zh-CN', { timeZone: TZ, weekday: 'short' })
function localDate(value = new Date()) { return dateFormatter.format(value) }
function localTime(value = new Date()) { return timeFormatter.format(value) }
function weekdayText(date) {
  const [y, m, d] = date.split('-').map(Number)
  return weekdayFormatter.format(new Date(Date.UTC(y, m - 1, d, 4)))
}
function addDays(date, days) {
  const [y, m, d] = date.split('-').map(Number)
  const value = new Date(Date.UTC(y, m - 1, d, 4))
  value.setUTCDate(value.getUTCDate() + days)
  return localDate(value)
}
function windowDates(end, days) {
  const dates = []
  for (let i = days - 1; i >= 0; i--) dates.push(addDays(end, -i))
  return dates
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

function parseArgs(argv) {
  const args = { days: 30, verbose: false }
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i]
    if (a === '--days') args.days = Number(argv[++i])
    else if (a === '--end') args.end = argv[++i]
    else if (a === '--json') args.json = argv[++i]
    else if (a === '--verbose') args.verbose = true
    else if (a === '--help' || a === '-h') { console.log(fs.readFileSync(process.argv[1], 'utf8').split('*/')[0]); process.exit(0) }
    else throw new Error(`unknown argument: ${a}`)
  }
  if (!Number.isInteger(args.days) || args.days <= 0) throw new Error('--days must be a positive integer')
  return args
}

const trackedEvent = /"type":\s*"(?:assistant\/message|step\/start|session\/end-seed|session\/title|subagent\/descriptor|session)"/

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
        const stat = fs.statSync(file)
        files.push({ id, file, mtime: stat.mtimeMs })
      } catch { /* not a session directory */ }
    }
  }
  return { files: files.filter(f => f.mtime >= since), all: files }
}

function newBucket() {
  return { requests: 0, uncached: 0, cacheRead: 0, cacheWrite: 0, output: 0, tokens: 0, cost: 0, peakRequests: 0, offRequests: 0 }
}
function addUsage(bucket, usage, model, band) {
  const uncached = Number(usage.inputTokens || 0)
  const cacheRead = Number(usage.cacheReadTokens || 0)
  const cacheWrite = Number(usage.cacheWriteTokens || 0)
  const output = Number(usage.outputTokens || 0)
  bucket.requests += 1
  bucket.uncached += uncached
  bucket.cacheRead += cacheRead
  bucket.cacheWrite += cacheWrite
  bucket.output += output
  bucket.tokens += uncached + cacheRead + cacheWrite + output
  bucket.cost += costOfUsage({ uncached, cacheRead, cacheWrite, output }, model, band)
  if (band === 'peak') bucket.peakRequests += 1
  else bucket.offRequests += 1
}

async function scanSessionLog({ id, file }, start, end, verbose) {
  const child = spawn(zstdcat, [file], { stdio: ['ignore', 'pipe', 'ignore'] })
  await new Promise((resolve, reject) => {
    child.once('spawn', resolve)
    child.once('error', reject)
  })
  const lines = readline.createInterface({ input: child.stdout, crlfDelay: Infinity })
  let header = null
  let title = ''
  let isSubagent = false
  let subagentLabel = ''
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
      if (event.type === 'session/title') { title = String(event.data?.title || title); continue }
      if (event.type === 'subagent/descriptor') { isSubagent = true; subagentLabel = String(event.data?.label || ''); continue }
      if (event.type === 'session/end-seed') {
        if (event.data?.inherited === true && event.seq > inheritedThrough) inheritedThrough = event.seq
        continue
      }
      if (event.type === 'step/start') { stepStarts.set(`${event.data?.turn}:${event.data?.step}`, event.time); continue }
      const usage = event.data?.usage
      if (event.type !== 'assistant/message' || !usage || !Number.isFinite(event.time)) continue
      const source = event.data.message?.source
      usages.push({ seq: event.seq, time: event.time, turn: event.data.turn, step: event.data.step, usage, model: source?.model || 'unknown', provider: source?.provider || '' })
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
  if (code !== 0) console.error(`session log ${id} ended early (${path.basename(zstdcat)} exited ${code}); using ${usages.length} step(s)`)

  const days = new Map()
  const bands = { peak: newBucket(), off: newBucket() }
  const models = new Map()
  let counted = 0
  for (const item of usages) {
    if (item.seq <= inheritedThrough) continue
    const started = stepStarts.get(`${item.turn}:${item.step}`) ?? item.time
    const startedDate = localDate(new Date(started))
    if (startedDate < start || startedDate > end) continue
    // The off-peak discount is a DeepSeek pricing rule; other providers have no band.
    const isDeepSeek = item.provider.includes('deepseek')
    const band = isDeepSeek && !isPeakTime(started) ? 'off' : 'peak'
    addUsage(band === 'peak' ? bands.peak : bands.off, item.usage, item.model, band)
    const key = `${item.provider}/${item.model}`
    const modelBucket = models.get(key) || newBucket()
    addUsage(modelBucket, item.usage, item.model, band)
    models.set(key, modelBucket)
    const dayBucket = days.get(startedDate) || newBucket()
    addUsage(dayBucket, item.usage, item.model, band)
    days.set(startedDate, dayBucket)
    counted += 1
  }
  if (verbose && counted > 0) console.error(`  ${id}: ${counted} steps, ${title || '(untitled)'}`)
  if (counted === 0) return null
  return {
    id,
    bands,
    kind: isSubagent ? 'subagent' : 'main',
    label: subagentLabel,
    title,
    workspace: header?.cwd || '',
    createdAt: header?.createdAt,
    days: [...days.entries()].map(([date, bucket]) => ({ date, ...bucket })),
    models: [...models.entries()].map(([key, bucket]) => {
      const [provider, model] = key.split('/')
      return { provider, model, ...bucket }
    }),
  }
}

async function main() {
  const args = parseArgs(process.argv.slice(2))
  const end = args.end || localDate()
  const dates = windowDates(end, args.days)
  const start = dates[0]
  if (end > localDate()) throw new Error('--end is in the future')

  const { files, all } = sessionLogFiles(start)
  const totals = newBucket()
  const bands = { peak: newBucket(), off: newBucket() }
  const byKind = { main: newBucket(), subagent: newBucket() }
  const perDay = new Map(dates.map(d => [d, newBucket()]))
  const perDaySessions = new Map(dates.map(d => [d, new Set()]))
  const perModel = new Map()
  const perWorkspace = new Map()
  const sessions = []
  const coverage = { earliestLog: null, firstDayWithUsage: null, lastDayWithUsage: null }
  for (const entry of all) {
    const stat = fs.statSync(entry.file)
    if (coverage.earliestLog === null || stat.mtimeMs < coverage.earliestLog) coverage.earliestLog = stat.mtimeMs
  }

  let failures = 0
  for (const entry of files) {
    let session
    try {
      session = await scanSessionLog(entry, start, end, args.verbose)
    } catch (error) {
      failures += 1
      console.error(`session log ${entry.id} skipped: ${String(error?.message || error).slice(0, 160)}`)
      continue
    }
    if (!session) continue
    sessions.push({ id: session.id, kind: session.kind, label: session.label, title: session.title, workspace: session.workspace, tokens: session.days.reduce((n, d) => n + d.tokens, 0) })
    for (const day of session.days) {
      perDaySessions.get(day.date)?.add(session.id)
      const bucket = perDay.get(day.date) || newBucket()
      for (const key of ['requests', 'uncached', 'cacheRead', 'cacheWrite', 'output', 'tokens', 'cost', 'peakRequests', 'offRequests']) bucket[key] += day[key]
      perDay.set(day.date, bucket)
      for (const key of ['requests', 'uncached', 'cacheRead', 'cacheWrite', 'output', 'tokens', 'cost', 'peakRequests', 'offRequests']) totals[key] += day[key]
      for (const key of ['requests', 'uncached', 'cacheRead', 'cacheWrite', 'output', 'tokens', 'cost', 'peakRequests', 'offRequests']) byKind[session.kind][key] += day[key]
      if (!coverage.firstDayWithUsage || day.date < coverage.firstDayWithUsage) coverage.firstDayWithUsage = day.date
      if (!coverage.lastDayWithUsage || day.date > coverage.lastDayWithUsage) coverage.lastDayWithUsage = day.date
    }
    for (const band of ['peak', 'off']) {
      for (const key of ['requests', 'uncached', 'cacheRead', 'cacheWrite', 'output', 'tokens', 'cost']) bands[band][key] += session.bands[band][key]
    }
    for (const model of session.models) {
      const key = `${model.provider}/${model.model}`
      const bucket = perModel.get(key) || { ...newBucket(), provider: model.provider, model: model.model }
      for (const k of ['requests', 'uncached', 'cacheRead', 'cacheWrite', 'output', 'tokens', 'cost', 'peakRequests', 'offRequests']) bucket[k] += model[k]
      perModel.set(key, bucket)
    }
    const ws = session.workspace || '(unknown)'
    const bucket = perWorkspace.get(ws) || newBucket()
    for (const day of session.days) for (const key of ['requests', 'tokens', 'cost']) bucket[key] += day[key]
    perWorkspace.set(ws, bucket)
  }

  const result = {
    generatedAt: new Date().toISOString(),
    timezone: TZ,
    window: { start, end, days: dates.length },
    coverage: {
      earliestSessionLog: coverage.earliestLog === null ? null : new Date(coverage.earliestLog).toISOString(),
      firstDayWithUsage: coverage.firstDayWithUsage,
      lastDayWithUsage: coverage.lastDayWithUsage,
      daysWithUsage: [...perDay.values()].filter(b => b.tokens > 0).length,
    },
    totals: { sessions: sessions.length, ...totals },
    bands,
    byKind,
    byDay: [...perDay.entries()].map(([date, bucket]) => ({ date, sessions: perDaySessions.get(date)?.size ?? 0, ...bucket })),
    byModel: [...perModel.values()].sort((a, b) => b.tokens - a.tokens),
    byWorkspace: [...perWorkspace.entries()].map(([workspace, bucket]) => ({ workspace, ...bucket })).sort((a, b) => b.tokens - a.tokens),
    pricing: { checkedAt: '2026-11-11', currency: 'CNY', source: 'DeepSeek API 官方价目（按 1M tokens，工作日 9-12/14-18 高峰，其余半价）', prices: MODEL_PRICES },
    failures,
    sessions: sessions.sort((a, b) => b.tokens - a.tokens),
  }
  const json = JSON.stringify(result, null, 2)
  if (args.json) {
    fs.writeFileSync(args.json, json + '\n')
    const { totals: t } = result
    console.log(`${start}..${end} (${result.window.days}d): ${t.tokens.toLocaleString()} tokens, ${t.requests} requests, ${sessions.length} sessions, ¥${t.cost.toFixed(2)}`)
  } else {
    console.log(json)
  }
  if (failures > 0) console.error(`${failures} session log(s) failed to scan`)
}

main().catch(error => { console.error(error); process.exit(1) })
