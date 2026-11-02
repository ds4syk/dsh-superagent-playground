"""Fill kb/00 and kb/04 from templates using a run directory."""
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
    raise SystemExit('no metrics')
last = m[-1]
samples = last['step']
sps = sum(x['sps'] for x in m) / len(m)
updates = len(m)
max_score = max((x.get('score', 0) for x in e), default=0)
reached = {}
for x in sorted(e, key=lambda r: r.get('step', 0)):
    for ms in x.get('milestones', []):
        reached.setdefault(ms, x['step'])
ms_events = sum(x.get('milestone_events', 0) for x in m)
inv = Counter()
for x in m:
    for k, v in (x.get('max_inventory') or {}).items():
        inv[k] = max(inv[k], v)
half = max(1, len(m) // 2)
first_half = sum(x.get('milestone_events', 0) for x in m[:half])
second_half = sum(x.get('milestone_events', 0) for x in m[half:])
loss_pol = last['loss_policy_loss']; loss_val = last['loss_value_loss']

ms_order = ['log','planks','stick','crafting_table','wooden_pickaxe','cobblestone',
            'stone_pickaxe','iron_ore','furnace','iron_ingot','iron_pickaxe','diamond']
ms_reached = [x for x in ms_order if inv.get(x, 0) > 0]
ep_reached = [x for x in ms_order if x in reached]
k = f"{samples/10000:.1f} 万"
diagnosis = (
    f"里程碑事件在前半程 {first_half} 次、后半程 {second_half} 次；"
    + ("后半程更多，说明策略在向里程碑靠拢。" if second_half > first_half else
       "后半程没有增加，说明本次样本量内 PPO 还没有形成稳定的任务行为，主要是在探索项上获得信号。")
)
if ep_reached:
    milestone_summary = "episode 中到达 " + " / ".join(ep_reached)
elif ms_reached:
    milestone_summary = "在线出现过 " + " / ".join(ms_reached) + "（未在完整 episode 的里程碑里复现）"
else:
    milestone_summary = "未到达任何里程碑"
learning_note = f"episode 数 {len(e)}，最高 score {max_score}，最深里程碑 {milestone_summary}。"

targets = [
    ('kb/templates/00 总览.md', 'kb/00 总览：PPO × Minecraft 训练系统（2026-11）.md', {
        '{{samples}}': f"{samples:,}", '{{sps}}': f"{sps:.1f}",
        '{{episodes}}': str(len(e)), '{{milestones}}': milestone_summary,
        '{{learning_note}}': learning_note,
    }),
    ('kb/templates/04 实验.md', 'kb/04 训练系统与实验.md', {
        '{{k_samples}}': k, '{{samples}}': f"{samples:,}", '{{sps}}': f"{sps:.1f}",
        '{{updates}}': str(updates), '{{episodes}}': str(len(e)),
        '{{max_score}}': str(max_score), '{{milestones}}': milestone_summary,
        '{{ms_events}}': str(ms_events), '{{max_inv}}': str(dict(inv)),
        '{{losses}}': f"{loss_pol:.4f} / {loss_val:.4f}",
        '{{diagnosis}}': diagnosis, '{{milestone_summary}}': milestone_summary,
    }),
]
for tpl, out, mapping in targets:
    s = open(tpl).read()
    for a, b in mapping.items():
        s = s.replace(a, b)
    open(out, 'w').write(s)
print('filled kb from', run, 'samples', samples, 'episodes', len(e), 'max_score', max_score)
