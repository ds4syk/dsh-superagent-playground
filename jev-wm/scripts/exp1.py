"""Experiment 1: JEV-style decision model as a factored world model.

Measures one-step state-variable accuracy, exact-match accuracy, execution
validity, four-step open-loop drift, and serving cost against baselines.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time

from jevwm.experiments import (
    evaluate_predictions,
    load_jsonl,
    make_backend,
    rollout_eval,
    save_json,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True,
                    choices=["persistence", "tacit", "decider", "generative"])
    ap.add_argument("--model-path", default="")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--level", default="raw")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=800)
    ap.add_argument("--rollouts", type=int, default=80)
    ap.add_argument("--horizon", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    records = load_jsonl(os.path.join("data", f"{args.split}.jsonl"))
    rng = random.Random(args.seed)
    if args.limit and args.limit < len(records):
        records = rng.sample(records, args.limit)

    tag = args.backend
    if args.model_path:
        tag += "-" + os.path.basename(args.model_path.rstrip("/"))
    if args.backend == "decider":
        tag += "-" + args.level

    backend = make_backend(args.backend, args.model_path, args.device, args.level)
    t0 = time.perf_counter()
    preds = backend.predict_transitions(records)
    wall = time.perf_counter() - t0

    metrics = evaluate_predictions(records, preds)
    metrics["wall_s"] = wall
    metrics["backend"] = tag
    metrics["sample_per_s"] = len(records) / wall
    metrics["decisions_per_s"] = metrics["decisions"] / wall if wall else None

    # a rough hosted-JEV cost estimate: 8 questions x state+question text
    total_tokens = 0
    for rec in records:
        total_tokens += 8 * int(len(rec["state_text"].split()) * 1.3 + 40)
    metrics["est_input_tokens"] = total_tokens
    metrics["est_usd_per_1k_transitions_jev_api"] = (
        total_tokens / len(records) * 1000 * 0.042 / 1e6
    )

    print(json.dumps({k: v for k, v in metrics.items() if k != "variable_accuracy"}, indent=1))
    print("variable accuracy:", json.dumps(metrics["variable_accuracy"], indent=1))

    if args.rollouts > 0:
        roll = rollout_eval(backend, records, limit=args.rollouts, horizon=args.horizon,
                            seed=args.seed + 1)
        metrics["rollout"] = roll
        print("rollout exact_by_k:", roll["exact_by_k"])

    save_json(os.path.join(args.out, f"exp1_{tag}.json"), metrics)
    with open(os.path.join(args.out, f"exp1_{tag}_preds.jsonl"), "w") as f:
        for rec, p in zip(records, preds):
            f.write(json.dumps({
                "id": rec["id"],
                "world_id": rec["world"]["world_id"],
                "action": rec["action"],
                "executes_pred": p.executes,
                "executes_gt": rec["executes"],
                "values_pred": p.values,
                "values_gt": rec["next_values"],
                "parse_fail": bool(getattr(p, "parse_fail", False)),
            }, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
