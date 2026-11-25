#!/usr/bin/env python3
"""Aggregate every DSH session log into lifetime token usage and estimated cost.

Source of truth: ~/.dsh/sessions/**/session.v4.jsonl.zstd (all DSH profiles).
Unit: one assistant/message response = one request; usage = input / cacheRead /
cacheWrite / output tokens. Compaction-summary LLM calls are included too.
Records are de-duplicated by assistant message id, so a forked/seeded session
that replays its parent's history is not counted twice.

Pricing: models.dev list price (USD per 1M tokens). DeepSeek's models.dev entry
equals the official off-peak rate; the official peak rate is 2x (Beijing weekday
09:00-12:00 and 14:00-18:00), which we apply to DeepSeek requests.
"""
import argparse, json, os, subprocess
from datetime import datetime, timezone, timedelta
from collections import defaultdict

CST = timezone(timedelta(hours=8))

PROVIDER_MAP = {
    'deepseek-official': 'deepseek',
    'deepseek-account': 'deepseek',
    'moonshotai-cn': 'moonshotai-cn',
    'kimi-code-plan-cn': 'moonshotai-cn',
    'zai-coding-cn': 'zhipuai',
    'zai': 'zai',
    'zhipuai': 'zhipuai',
}


def norm_provider(p):
    return PROVIDER_MAP.get(p, p)


def is_peak(ms):
    t = datetime.fromtimestamp(ms / 1000, CST)
    if t.weekday() >= 5:
        return False
    h = t.hour
    return 9 <= h < 12 or 14 <= h < 18


def parse_usage(u):
    if not isinstance(u, dict):
        return None
    def n(k):
        v = u.get(k)
        if isinstance(v, (int, float)) and v == v:
            return max(0, int(round(v)))
        return 0
    return {k: n(k) for k in ('inputTokens', 'cacheReadTokens', 'cacheWriteTokens', 'outputTokens')}


def walk(root):
    for dirpath, _, names in os.walk(root):
        for name in names:
            if name == 'session.v4.jsonl.zstd':
                yield os.path.join(dirpath, name)


def workspace_of(cwd):
    if not cwd:
        return '(unknown)'
    if cwd.startswith('/mnt/storage1/syk/.superagent-work/'):
        return '工单 ' + cwd[len('/mnt/storage1/syk/.superagent-work/'):]
    if cwd == '/mnt/storage1/syk/.dsh-state/superagent':
        return 'orchestrator 主会话'
    if cwd.startswith('/home/syk/labs/'):
        return 'labs/' + cwd[len('/home/syk/labs/'):].split('/')[0]
    if cwd == '/home/syk/labs':
        return 'labs'
    return cwd.replace('/mnt/storage1/syk/', '~/').replace('/home/syk/', '~/')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sessions-root', default=os.path.expanduser('~/.dsh/sessions'))
    ap.add_argument('--prices', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'prices-2026-11-26.json'))
    ap.add_argument('--out', default=None)
    ap.add_argument('--cutoff-ms', type=int, default=None)
    args = ap.parse_args()

    price = json.load(open(args.prices))
    rates = {}
    for prov, models in price['providers'].items():
        for model, cost in models.items():
            rates[(prov, model)] = cost

    files = sorted(walk(args.sessions_root))
    records = {}
    raw = {'tokens': 0, 'requests': 0}
    session_meta = {}
    parse_errors = []
    for f in files:
        try:
            raw_bytes = subprocess.run(['zstd', '-dc', f], capture_output=True, check=True).stdout
        except Exception as e:
            parse_errors.append((f, str(e)))
            continue
        events = []
        for line in raw_bytes.decode('utf-8', 'replace').split('\n'):
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except Exception:
                continue
        if not events or events[0].get('type') != 'session':
            parse_errors.append((f, 'no header'))
            continue
        hdr = events[0]
        sid = hdr['id']
        session_meta[sid] = {
            'id': sid, 'cwd': hdr.get('cwd', ''), 'createdAt': hdr.get('createdAt'),
            'parentSession': hdr.get('parentSession'), 'origin': hdr.get('origin'),
            'isSeeded': bool(hdr.get('isSeeded')), 'depth': hdr.get('delegationDepth', 0),
            'file': f, 'lastTime': 0,
        }
        step_start = None
        provider = model = None
        for ev in events:
            typ = ev.get('type')
            t = ev.get('time')
            if isinstance(t, (int, float)) and t > session_meta[sid]['lastTime']:
                session_meta[sid]['lastTime'] = int(t)
            if typ == 'request/header':
                cfg = (ev.get('data') or {}).get('header', {}).get('config') or {}
                if isinstance(cfg.get('provider'), str):
                    provider = cfg['provider']
                if isinstance(cfg.get('model'), str):
                    model = cfg['model']
            elif typ == 'request/context':
                d = ev.get('data') or {}
                if isinstance(d.get('provider'), str):
                    provider = d['provider']
                if isinstance(d.get('model'), str):
                    model = d['model']
            elif typ == 'step/start':
                if isinstance(t, (int, float)):
                    step_start = int(t)
            elif typ == 'step/end':
                step_start = None
            elif typ == 'assistant/message':
                d = ev.get('data') or {}
                u = parse_usage(d.get('usage'))
                if not u:
                    continue
                msg = d.get('message') or {}
                src = msg.get('source') or {}
                p = src.get('provider') or provider or 'unknown'
                m = src.get('model') or model or 'unknown'
                ts = step_start if isinstance(step_start, (int, float)) else t
                if not isinstance(ts, (int, float)):
                    continue
                if args.cutoff_ms and int(ts) > args.cutoff_ms:
                    continue
                mid = msg.get('id') or f'{sid}:{ev.get("seq")}'
                rec = {'id': mid, 'session': sid, 'time': int(ts),
                       'provider': p, 'model': m,
                       'uncached': u['inputTokens'], 'cacheRead': u['cacheReadTokens'],
                       'cacheWrite': u['cacheWriteTokens'], 'output': u['outputTokens'],
                       'kind': 'message'}
                raw['tokens'] += sum(rec[k] for k in ('uncached', 'cacheRead', 'cacheWrite', 'output'))
                raw['requests'] += 1
                old = records.get(mid)
                if old is None or rec['time'] < old['time']:
                    records[mid] = rec
            elif typ == 'compaction/summary':
                d = ev.get('data') or {}
                u = parse_usage(d.get('usage'))
                if not u:
                    continue
                if not isinstance(t, (int, float)):
                    continue
                if args.cutoff_ms and int(t) > args.cutoff_ms:
                    continue
                mid = f'{sid}:compaction:{ev.get("seq")}'
                rec = {'id': mid, 'session': sid, 'time': int(t),
                       'provider': d.get('provider') or provider or 'unknown',
                       'model': d.get('model') or model or 'unknown',
                       'uncached': u['inputTokens'], 'cacheRead': u['cacheReadTokens'],
                       'cacheWrite': u['cacheWriteTokens'], 'output': u['outputTokens'],
                       'kind': 'compaction'}
                raw['tokens'] += sum(rec[k] for k in ('uncached', 'cacheRead', 'cacheWrite', 'output'))
                raw['requests'] += 1
                records[mid] = rec

    def rate_of(p, m):
        key = (norm_provider(p), m)
        if key in rates:
            return key, rates[key]
        for (prov, model), r in rates.items():
            if model == m:
                return (prov, model), r
        return None, None

    def zero():
        return defaultdict(float)

    totals = zero()
    by_day = defaultdict(zero)
    by_month = defaultdict(zero)
    by_model = defaultdict(zero)
    by_session = defaultdict(zero)
    by_workspace = defaultdict(zero)
    unpriced = set()
    for rec in records.values():
        tok = sum(rec[k] for k in ('uncached', 'cacheRead', 'cacheWrite', 'output'))
        key, rate = rate_of(rec['provider'], rec['model'])
        if rate is None:
            unpriced.add(f"{rec['provider']}/{rec['model']}")
            base = 0.0
        else:
            hit = rate.get('cache_read', rate.get('input', 0))
            wr = rate.get('cache_write', rate.get('input', 0))
            base = (rec['cacheRead'] * hit + rec['uncached'] * rate.get('input', 0)
                    + rec['cacheWrite'] * wr + rec['output'] * rate.get('output', 0)) / 1e6
        mult = 2 if norm_provider(rec['provider']) == 'deepseek' and is_peak(rec['time']) else 1
        usd = base * mult
        peak = mult > 1
        t = datetime.fromtimestamp(rec['time'] / 1000, CST)
        day, month = t.strftime('%Y-%m-%d'), t.strftime('%Y-%m')
        ws = workspace_of(session_meta[rec['session']]['cwd'])
        for target in (totals, by_day[day], by_month[month],
                       by_model[(norm_provider(rec['provider']), rec['model'])],
                       by_session[rec['session']], by_workspace[ws]):
            for k in ('uncached', 'cacheRead', 'cacheWrite', 'output'):
                target[k] += rec[k]
            target['tokens'] += tok
            target['requests'] += 1
            target['cost'] += usd
        if mult > 1:
            totals['peakSurcharge'] += usd - base
            totals['peakTokens'] += tok
        else:
            totals['offTokens'] += tok
        if rate:
            hit = rate.get('cache_read', rate.get('input', 0)) / 1e6
            wr = rate.get('cache_write', rate.get('input', 0)) / 1e6
            parts = {
                'CacheRead': rec['cacheRead'] * hit,
                'Uncached': rec['uncached'] * rate.get('input', 0) / 1e6,
                'CacheWrite': rec['cacheWrite'] * wr,
                'Output': rec['output'] * rate.get('output', 0) / 1e6,
            }
            for name, amount in parts.items():
                totals['cost' + name] += amount * mult
                totals['baseCost' + name] += amount
        if norm_provider(rec['provider']) == 'deepseek':
            totals['deepseekTokens'] += tok

    sessions = {r['session'] for r in records.values()}
    times = [r['time'] for r in records.values()]
    first, last = (min(times), max(times)) if times else (None, None)
    days = sorted(by_day)

    def pack(d, is_total=False):
        out = {}
        for k, v in d.items():
            out[k] = round(v, 6) if (k.startswith('cost') or k.startswith('baseCost') or k in ('peakSurcharge', 'usd')) else int(v)
        if 'cost' in d:
            out['cost'] = round(d['cost'], 6)
            out['costCNY'] = round(d['cost'] * price.get('fxUsdCny', 6.678), 2)
        return out

    by_month_rows = [dict(month=m, **pack(v)) for m, v in sorted(by_month.items())]
    by_day_rows = [dict(date=d, **pack(v)) for d, v in sorted(by_day.items())]
    by_model_rows = [dict(provider=p, model=mo, **pack(v)) for (p, mo), v in sorted(by_model.items(), key=lambda kv: -kv[1]['tokens'])]
    by_ws_rows = [dict(workspace=w, **pack(v)) for w, v in sorted(by_workspace.items(), key=lambda kv: -kv[1]['tokens'])]
    total_row = pack(totals, True)
    total_row['costCNY'] = round(totals['cost'] * price.get('fxUsdCny', 6.678), 2)
    total_row['cacheHitRate'] = round(totals['cacheRead'] / max(1, totals['cacheRead'] + totals['uncached'] + totals['cacheWrite']), 6)

    report = {
        'generatedAt': datetime.now(timezone.utc).isoformat(),
        'coverage': {
            'sessionFiles': len(files),
            'sessionsWithUsage': len(sessions),
            'sessionsWithoutUsage': len(session_meta) - len(sessions),
            'requests': len(records),
            'firstRequest': datetime.fromtimestamp(first / 1000, CST).isoformat() if first else None,
            'lastRequest': datetime.fromtimestamp(last / 1000, CST).isoformat() if last else None,
            'daysWithUsage': len(days),
            'days': days,
            'parseErrors': parse_errors,
        },
        'dedup': {
            'rawRequests': raw['requests'], 'rawTokens': raw['tokens'],
            'removedRequests': raw['requests'] - len(records),
            'removedTokens': raw['tokens'] - int(totals['tokens']),
        },
        'prices': {
            'source': price.get('source'), 'fetchedAt': price.get('fetchedAt'),
            'fxUsdCny': price.get('fxUsdCny'), 'note': price.get('note'),
        },
        'totals': total_row,
        'byDay': by_day_rows,
        'byMonth': by_month_rows,
        'byModel': by_model_rows,
        'byWorkspace': by_ws_rows,
        'bySession': [dict(session=s, **pack(v), cwd=session_meta[s]['cwd'],
                            kind=('subagent' if session_meta[s]['depth'] else 'main'))
                      for s, v in sorted(by_session.items(), key=lambda kv: -kv[1]['tokens'])],
        'sessions': session_meta,
        'unpriced': sorted(unpriced),
    }
    js = json.dumps(report, ensure_ascii=False, indent=1)
    if args.out:
        open(args.out, 'w').write(js)
    t = report['totals']
    print(f"files={len(files)} sessions={len(sessions)} requests={len(records)} days={len(days)}")
    print(f"range={report['coverage']['firstRequest']} .. {report['coverage']['lastRequest']}")
    print(f"tokens={t['tokens']:,} raw={report['dedup']['rawTokens']:,} removed={report['dedup']['removedTokens']:,}")
    print(f"components uncached={t['uncached']:,} cacheRead={t['cacheRead']:,} cacheWrite={t['cacheWrite']:,} output={t['output']:,}")
    print(f"cost=${t['cost']:.2f} ~ ¥{t['costCNY']:.2f} peakTokens={t.get('peakTokens',0):,} surcharge=${t.get('peakSurcharge',0):.2f}")
    print(f"cost components: cacheRead=${t.get('costCacheRead',0):.2f} uncached=${t.get('costUncached',0):.2f} output=${t.get('costOutput',0):.2f}")
    for m in report['byMonth']:
        print(f"  {m['month']}: {m['tokens']:,} tok  {m['requests']:,} req  ${m['cost']:.2f}")
    for m in report['byModel']:
        print(f"  {m['provider']}/{m['model']}: {m['tokens']:,} tok  {m['requests']:,} req  ${m['cost']:.2f}")
    for w in report['byWorkspace'][:6]:
        print(f"  ws {w['workspace']}: {w['tokens']:,} tok ${w['cost']:.2f}")
    if unpriced:
        print('UNPRICED', unpriced)


if __name__ == '__main__':
    main()
