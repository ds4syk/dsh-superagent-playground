"""试玩：在官方模拟器里造几座桥，记录分数并出图。

- 关卡 1：光板桥 → 上承式桁架 → 下承式桁架 → 优化后的下承式桁架（自己设计）
- 关卡 2/3/6：直线桥面 + 桁架的朴素尝试（失败），以及官方参考答案 / 选手设计（展示上限）
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from arena_bridge.dataset import base_above, base_under  # noqa: E402
from arena_bridge.design import LEVEL_GRID  # noqa: E402
from arena_bridge.render import gallery, save_bridge_png  # noqa: E402
from arena_bridge.sim import Evaluator  # noqa: E402

REPO = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "third_party", "meow-ai-arena", "bridge")


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def naive_l2_above():
    g = LEVEL_GRID[2]
    c = np.zeros(g.n_cand, np.int64)
    for i, (k, r, col) in enumerate(g.cand):
        if k == "h" and r == 2:
            c[i] = 2
        if k == "v" and r == 1:
            c[i] = 1
        if k in ("d1", "d2") and r == 1 and ((k == "d1" and col % 2 == 0) or
                                             (k == "d2" and col % 2 == 1)):
            c[i] = 2
    return c


def main():
    out = "results/play"
    os.makedirs(os.path.join(out, "bridges"), exist_ok=True)
    g1 = LEVEL_GRID[1]
    seed = np.load("results/seed_l1.npy")
    items, rows = [], []

    def add(key, name, level, design, note, result=None):
        ev = Evaluator(level)
        r = result or ev.evaluate(design)
        rows.append(dict(key=key, name=name, level=level, note=note,
                         cost=round(r.cost, 2), score=r.score, passed=r.passed,
                         reason=r.reason, members=len(design.get("members", []))))
        title = f"{name}"
        save_bridge_png(os.path.join(out, "bridges", f"{key}.png"), design, level, r, title=title)
        items.append((design, r, title, ""))
        print(f"{key} {name}: pass={r.passed} score={r.score} cost={r.cost:.0f} {r.reason}")

    add("A", "A 光板桥（L1）", 1, g1.build_json(np.zeros(g1.n_cand, np.int64)),
        "只有 12 根桥面梁，不加上部结构")
    add("B", "B 上承式桁架（L1）", 1, g1.build_json(base_above(g1)),
        "桥面上方 2 层钢/木桁架")
    add("C", "C 下承式桁架（L1）", 1, g1.build_json(base_under(g1)),
        "桥面下方 2 米深木/钢桁架")
    add("D", "D 优化下承式（L1）", 1, g1.build_json(seed),
        "从 C 出发逐根删减/换材，保持三场景通过")

    add("E", "E 光板桥（L2）", 2, LEVEL_GRID[2].build_json(np.zeros(LEVEL_GRID[2].n_cand, np.int64)),
        "30 米跨径只铺直线桥面")
    add("F", "F 直线桁架（L2）", 2, LEVEL_GRID[2].build_json(naive_l2_above()),
        "把 L1 的上承式桁架直接放大到 30 米")
    add("G", "G 官方参考（L2）", 2, load_json(os.path.join(REPO, "reference", "level2.json")),
        "载入参考答案，展示正式 100 分基准")
    add("H", "H 选手 Sol（L2）", 2, load_json(os.path.join(REPO, "designs", "sol62", "level2.json")),
        "载入 GPT-6.2 Sol 的 30 米悬索桥（137.85 分）")
    add("I", "I 官方参考（L3）", 3, load_json(os.path.join(REPO, "reference", "level3.json")),
        "28 米河 + 中间礁石，双向车流")
    add("J", "J 官方参考（L6）", 6, load_json(os.path.join(REPO, "reference", "level6.json")),
        "36 米、两块礁石、航道与卡车的地狱关")

    gallery(os.path.join(out, "gallery_l1.png"),
            [it for it in items[:4]], 1, ncols=2, title="关卡 1：从光板桥到 61.5 分")
    gallery(os.path.join(out, "gallery_l2.png"),
            [it for it in items[4:8]], 2, ncols=2,
            title="关卡 2：把 L1 的桁架直接放大到 30 米会失败，悬索/参考设计才能过")
    with open(os.path.join(out, "play_log.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print("saved", out)


if __name__ == "__main__":
    main()
