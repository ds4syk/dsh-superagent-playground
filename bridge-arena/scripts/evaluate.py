"""统一评测：三个模型各采样 N 个设计，用官方引擎打分。

python scripts/evaluate.py --n 512 --out results/eval
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from arena_bridge.design import LEVEL_GRID  # noqa: E402
from arena_bridge.sim import evaluate_many  # noqa: E402
from models.models import (BridgeTransformer, ConvVAE, EpsMLP, GaussianDiffusion,  # noqa: E402
                           N_CAND)

G = LEVEL_GRID[1]
CN = {"transformer": "Transformer", "diffusion": "Diffusion", "cnn": "CNN-VAE",
      "random": "随机基线", "expert": "专家数据"}


def metrics(name, designs, res):
    sc = np.array([r["score"] for r in res])
    passed = np.array([r["passed"] for r in res])
    cost = np.array([r["cost"] for r in res])
    order = np.argsort(-sc)
    best5 = [sc[i:i + 5].max() for i in range(0, len(sc) - 4, 5)]
    uniq = len({d.tobytes() for d in designs}) / len(designs)
    rng = np.random.default_rng(0)
    ii = rng.integers(0, len(designs), 200)
    jj = rng.integers(0, len(designs), 200)
    jac = []
    for a, b in zip(ii, jj):
        da, db = designs[a] > 0, designs[b] > 0
        jac.append(1.0 - np.logical_and(da, db).sum() / max(np.logical_or(da, db).sum(), 1))
    fails = {}
    for r in res:
        k = r["reason"] or "通过"
        fails[k] = fails.get(k, 0) + 1
    return dict(name=name, n=len(designs),
                pass_rate=float(passed.mean()),
                score_mean=float(sc.mean()), score_std=float(sc.std()),
                score_max=float(sc.max()),
                score_pass_mean=float(sc[passed].mean()) if passed.any() else 0.0,
                best_of_5=float(np.mean(best5)),
                cost_mean=float(cost.mean()),
                members_mean=float((designs > 0).sum(1).mean()),
                unique=float(uniq), diversity=float(np.mean(jac)),
                failures=fails,
                top_score=[int(i) for i in order[:10]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="results/dataset_l1.npz")
    ap.add_argument("--seed-design", default="results/seed_l1.npy")
    ap.add_argument("--ckpt", default="results/checkpoints")
    ap.add_argument("--out", default="results/eval")
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    g = torch.Generator(device=dev); g.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    def evaluate(name, designs, store):
        t = time.time()
        res = evaluate_many(1, [G.build_json(d) for d in designs], workers=args.workers)
        m = metrics(name, designs, res)
        m["eval_time"] = time.time() - t
        print(json.dumps({k: m[k] for k in ("name", "pass_rate", "score_mean", "score_max",
                                           "cost_mean", "members_mean")}, ensure_ascii=False))
        store[name] = dict(designs=designs.astype(np.uint8),
                           score=np.array([r["score"] for r in res], np.float32),
                           cost=np.array([r["cost"] for r in res], np.float32),
                           passed=np.array([r["passed"] for r in res], bool),
                           reason=np.array([r["reason"] for r in res]))
        return m

    all_metrics, store = [], {}
    seed = np.load(args.seed_design).astype(np.int64)

    # 随机基线：种子设计 + 1~3 处随机改动
    rand = []
    while len(rand) < args.n:
        c = seed.copy()
        for _ in range(int(rng.integers(1, 4))):
            c[int(rng.integers(0, G.n_cand))] = int(rng.choice([0, 0, 1, 2, 3]))
        if G.cost_estimate(c) <= G.budget:
            rand.append(c)
    all_metrics.append(evaluate("random", np.stack(rand), store))

    # 专家基线：训练集里分数最高的一批
    d = np.load(args.data)
    idx = np.where(d["passed"].astype(bool))[0]
    idx = idx[np.argsort(-d["score"][idx])][: args.n]
    all_metrics.append(evaluate("expert", d["designs"][idx].astype(np.int64), store))

    # Transformer
    t0 = time.time()
    tf = BridgeTransformer().to(dev)
    tf.load_state_dict(torch.load(os.path.join(args.ckpt, "transformer.pt"), map_location=dev))
    tf.eval()
    out = tf.sample(args.n, dev, temperature=args.temperature, generator=g).cpu().numpy()
    tf_time = time.time() - t0
    all_metrics.append(evaluate("transformer", out, store))

    # Diffusion
    t0 = time.time()
    dm = EpsMLP().to(dev)
    dm.load_state_dict(torch.load(os.path.join(args.ckpt, "diffusion.pt"), map_location=dev))
    diff = GaussianDiffusion(1000, device=dev)
    out = diff.sample(dm, args.n, dev, steps=100, generator=g).cpu().numpy()
    dm_time = time.time() - t0
    all_metrics.append(evaluate("diffusion", out, store))

    # CNN
    t0 = time.time()
    cv = ConvVAE().to(dev)
    cv.load_state_dict(torch.load(os.path.join(args.ckpt, "cnn.pt"), map_location=dev))
    grids = cv.sample(args.n, dev, generator=g).cpu().numpy()
    out = np.stack([G.from_grid(grids[i]) for i in range(len(grids))])
    cv_time = time.time() - t0
    all_metrics.append(evaluate("cnn", out, store))

    timing = dict(transformer=tf_time, diffusion=dm_time, cnn=cv_time)
    out_json = dict(metrics=all_metrics, timing=timing, n=args.n, seed=args.seed,
                    temperature=args.temperature)
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(out_json, f, ensure_ascii=False, indent=2)
    np.savez_compressed(os.path.join(args.out, "samples.npz"),
                        **{f"{k}_designs": v["designs"] for k, v in store.items()},
                        **{f"{k}_{m}": v[m] for k, v in store.items()
                           for m in ("score", "cost", "passed", "reason")})
    print("saved", args.out)


if __name__ == "__main__":
    main()
