"""Print a compact summary of a run directory (metrics.jsonl + episodes.jsonl)."""
import json, os, sys

run = sys.argv[1] if len(sys.argv) > 1 else '/mnt/storage1/syk/dreamer/runs/mcppo/diamond_s0'

def rows(name):
    p = os.path.join(run, name)
    out = []
    if os.path.exists(p):
        for line in open(p):
            try: out.append(json.loads(line))
            except Exception: pass
    return out

m = rows('metrics.jsonl'); e = rows('episodes.jsonl')
print('run', run)
if m:
    last = m[-1]
    sps = [x['sps'] for x in m if x.get('sps')]
    print(f"updates={len(m)} samples={last.get('step')} "
          f"sps_mean={sum(sps)/len(sps):.1f} sps_last={last.get('sps'):.1f} "
          f"ticks/s={last.get('ticks_per_s'):.1f}")
    print(f"collect_s={last.get('collect_s'):.1f} learn_s={last.get('learn_s'):.1f}")
    print(f"loss_policy={last.get('loss_policy_loss'):.4f} loss_value={last.get('loss_value_loss'):.4f} "
          f"entropy={last.get('loss_entropy'):.3f} kl={last.get('loss_approx_kl'):.5f} clip={last.get('loss_clip_frac'):.3f}")
    print('max_inventory_last', last.get('max_inventory'))
    print('milestone_events_total', sum(x.get('milestone_events', 0) for x in m))
    comps = {}
    for x in m:
        for k, v in (x.get('reward_components') or {}).items():
            comps[k] = comps.get(k, 0.0) + v
    print('reward_components_total', {k: round(v, 1) for k, v in comps.items()})
if e:
    print(f"episodes={len(e)}")
    for x in e[-5:]:
        print({k: x.get(k) for k in ('step', 'return', 'score', 'milestones', 'length')})
else:
    print('episodes=0 (no 18k-step episode finished yet)')
