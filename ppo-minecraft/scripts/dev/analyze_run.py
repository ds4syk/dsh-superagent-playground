"""Quantitative summary for the KB results page."""
import json, os, sys
from collections import Counter

run = sys.argv[1] if len(sys.argv) > 1 else '/mnt/storage1/syk/dreamer/runs/mcppo/diamond_s0'

def rows(name):
    p = os.path.join(run, name); out = []
    if os.path.exists(p):
        for line in open(p):
            try: out.append(json.loads(line))
            except Exception: pass
    return out

m, e = rows('metrics.jsonl'), rows('episodes.jsonl')
if not m:
    print('no metrics yet'); raise SystemExit
last = m[-1]
print('samples', last['step'], 'updates', len(m))
print('sps_mean %.1f sps_last %.1f' % (sum(x['sps'] for x in m)/len(m), last['sps']))
print('collect_mean %.1f learn_mean %.1f' % (
    sum(x['collect_s'] for x in m)/len(m), sum(x['learn_s'] for x in m)/len(m)))
half = max(1, len(m)//2)
print('milestone_events first_half', sum(x.get('milestone_events',0) for x in m[:half]),
      'second_half', sum(x.get('milestone_events',0) for x in m[half:]))
inv = Counter()
for x in m:
    for k, v in (x.get('max_inventory') or {}).items():
        inv[k] = max(inv[k], v)
print('max_inventory_ever', dict(inv))
print('episodes', len(e))
if e:
    print('episode_return mean %.2f min %.2f max %.2f' % (
        sum(x['return'] for x in e)/len(e), min(x['return'] for x in e), max(x['return'] for x in e)))
    print('score histogram', Counter(x['score'] for x in e))
    print('best', max(e, key=lambda x: x['score']))
