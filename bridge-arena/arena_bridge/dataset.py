"""关卡 1 的设计数据：种子桥、变异、进化搜索、数据集生成。

设计向量长度 = GridLevel.n_cand，每个分量取 0..3。
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from .design import LEVEL_GRID
from .sim import evaluate_many


def base_under(g) -> np.ndarray:
    c = np.zeros(g.n_cand, np.int64)
    for i, (k, r, col) in enumerate(g.cand):
        if k == "h" and r == 0:
            c[i] = 1
        if k == "v" and r == 0 and 0 < col < g.ncol - 1:
            c[i] = 1
        if k in ("d1", "d2") and r == 0 and ((k == "d1" and col % 2 == 0) or
                                             (k == "d2" and col % 2 == 1)):
            c[i] = 2
    return c


def base_above(g) -> np.ndarray:
    c = np.zeros(g.n_cand, np.int64)
    for i, (k, r, col) in enumerate(g.cand):
        if k == "h" and r == 2:
            c[i] = 2
        if k == "v" and 1 <= r < 2:
            c[i] = 1
        if k in ("d1", "d2") and 1 <= r < 2 and ((k == "d1" and col % 2 == 0) or
                                                 (k == "d2" and col % 2 == 1)):
            c[i] = 2
    return c


def seed_designs(level: int):
    g = LEVEL_GRID[level]
    seeds = [base_under(g), base_above(g)]
    p = f"results/seed_l{level}.npy"
    if os.path.exists(p):
        seeds.append(np.load(p))
    return g, seeds


def mutate(c: np.ndarray, rng: np.random.Generator, k: int | None = None) -> np.ndarray:
    c = c.copy()
    k = int(rng.integers(1, 4)) if k is None else k
    for _ in range(k):
        i = int(rng.integers(0, c.shape[0]))
        c[i] = int(rng.choice([0, 0, 1, 2, 3]))
    return c


def _build_all(g, designs):
    return [g.build_json(c) for c in designs]


def evolve(level: int = 1, pop: int = 32, gens: int = 40, children: int = 6,
           workers: int = 16, seed: int = 0, out: str | None = None):
    g, seeds = seed_designs(level)
    rng = np.random.default_rng(seed)
    pool = list(seeds)
    while len(pool) < pop:
        pool.append(mutate(seeds[rng.integers(len(seeds))], rng, k=int(rng.integers(1, 3))))
    pool = [c for c in pool if g.cost_estimate(c) <= g.budget]
    res = evaluate_many(level, _build_all(g, pool), workers=workers)
    passing = [(r["score"], r["cost"], c) for c, r in zip(pool, res) if r["passed"]]
    passing.sort(key=lambda x: -x[0])
    if not passing:
        raise SystemExit("初始种群没有通过的设计")
    print(f"init: {len(passing)} passing, best {passing[0][0]:.2f}", flush=True)
    t0 = time.time()
    for gen in range(gens):
        parents = [c for _, _, c in passing[: max(2, pop // 2)]]
        designs = []
        for c in parents:
            for _ in range(children):
                designs.append(mutate(c, rng, k=int(rng.integers(1, 4))))
        designs = [c for c in designs if g.cost_estimate(c) <= g.budget]
        res = evaluate_many(level, _build_all(g, designs), workers=workers)
        newpass = [(r["score"], r["cost"], c) for c, r in zip(designs, res) if r["passed"]]
        merged = passing + newpass
        merged.sort(key=lambda x: -x[0])
        seen = set(); passing = []
        for s, cost, c in merged:
            key = c.tobytes()
            if key in seen:
                continue
            seen.add(key); passing.append((s, cost, c))
            if len(passing) >= pop:
                break
        print(f"gen {gen:02d} best_score={passing[0][0]:.2f} best_cost={passing[0][1]:.0f} "
              f"new_pass={len(newpass)}/{len(designs)} ({time.time()-t0:.0f}s)", flush=True)
    best = passing[0][2]
    bc = g.cost_estimate(best)
    print(f"final best cost={bc:.0f} score={100*3509.84/bc:.2f}")
    if out:
        np.save(out, best)
        print("saved", out)
    return best

def generate(level: int, n: int, out: str, workers: int = 16, seed: int = 0):
    g, seeds = seed_designs(level)
    rng = np.random.default_rng(seed)
    designs, seen = [], set()
    t0 = time.time()
    while len(designs) < n:
        c = mutate(seeds[rng.integers(len(seeds))], rng,
                   k=int(rng.choice([1, 1, 2, 2, 3, 4])))
        if g.cost_estimate(c) > g.budget:
            continue
        key = c.tobytes()
        if key in seen:
            continue
        seen.add(key); designs.append(c)
        if len(designs) % 5000 == 0:
            print(f"  gen {len(designs)}/{n} ({time.time()-t0:.0f}s)", flush=True)
    print("evaluating", len(designs), flush=True)
    res = evaluate_many(level, _build_all(g, designs), workers=workers)
    arr = np.stack(designs).astype(np.uint8)
    cost = np.array([r["cost"] for r in res])
    score = np.array([r["score"] for r in res])
    passed = np.array([r["passed"] for r in res], bool)
    reasons = np.array([r["reason"] for r in res])
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(out, designs=arr, cost=cost, score=score, passed=passed,
                        reason=reasons, level=level)
    print(f"saved {out} n={len(arr)} pass_rate={passed.mean():.3f} "
          f"score_top={np.sort(score)[-5:]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["evolve", "dataset"], default="evolve")
    ap.add_argument("--level", type=int, default=1)
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--pop", type=int, default=32)
    ap.add_argument("--gens", type=int, default=30)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/dataset_l1.npz")
    args = ap.parse_args()
    if args.mode == "evolve":
        evolve(args.level, args.pop, args.gens, workers=args.workers, seed=args.seed,
               out=f"results/seed_l{args.level}.npy")
    else:
        generate(args.level, args.n, args.out, workers=args.workers, seed=args.seed)


if __name__ == "__main__":
    main()
