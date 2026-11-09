"""Simulate an ideal diamond tech-tree run through the reward engine."""
import copy, json
from mcppo.config import DEFAULTS
from mcppo.rewards.reward import RewardEngine

cfg = copy.deepcopy(DEFAULTS["reward"])
eng = RewardEngine(cfg, episode_steps=18000)
inv = {}
events = [
    (100, {"log": 1}), (150, {"log": 3, "planks": 4}), (200, {"planks": 4, "stick": 4}),
    (250, {"crafting_table": 1}), (300, {"wooden_pickaxe": 1}),
    (1500, {"cobblestone": 10}), (2000, {"stone_pickaxe": 1}),
    (5000, {"iron_ore": 3}), (5500, {"furnace": 1}),
    (6000, {"iron_ingot": 3}), (7000, {"iron_pickaxe": 1}),
    (17000, {"diamond": 1}),
]
ev = dict(events)
rows = []
eng.step({}, 20, 20, 300, (0, 64, 0), is_first=True)
for t in range(1, 18001):
    if t in ev:
        inv.update(ev[t])
    r, comp = eng.step(inv, 20, 20, 300, (0, 64, 0))
    if t in ev:
        rows.append({"t": t, "items": sorted(ev[t]), "reward": round(r, 3),
                     "milestone": round(comp["milestone"], 3), "shape": round(comp["shape"], 4),
                     "score": eng.score()})
print(json.dumps({"events": rows, "components": {k: round(v, 3) for k, v in eng.components.items()},
                  "episode_return": round(eng.episode_return, 3),
                  "diamond_milestone": rows[-1]["milestone"]}, indent=1))
