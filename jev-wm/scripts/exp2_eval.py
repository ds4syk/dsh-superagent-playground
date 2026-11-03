"""Closed-loop evaluation of the distilled learner vs its JEV teacher."""

from __future__ import annotations

import argparse
import json
import os
import time

from jevwm.experiments import evaluate_policy, save_json
from jevwm.policies import OraclePolicy, RandomPolicy, StudentPolicy
from jevwm.student import load_student
from jevwm.world import make_specs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", default="results/student.pt")
    ap.add_argument("--students", default="", help="comma-separated list; overrides --student")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--worlds", type=int, default=48)
    ap.add_argument("--starts-per-world", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--teacher-json", default="")
    ap.add_argument("--out", default="results/exp2_eval.json")
    args = ap.parse_args()

    specs = make_specs("testsame", args.worlds)
    results = {}
    for name, pol in (("random", RandomPolicy(seed=args.seed)), ("oracle", OraclePolicy())):
        res, _ = evaluate_policy(pol, specs, args.starts_per_world, args.max_steps, args.seed)
        results[name] = res
        print(f"{name}: success {res['success_rate']:.3f} invalid {res['invalid_rate']:.3f}")

    paths = [p.strip() for p in args.students.split(",") if p.strip()] or [args.student]
    for path in paths:
        model, meta = load_student(path, device=args.device)
        t0 = time.perf_counter()
        res, _ = evaluate_policy(StudentPolicy(model, meta["vocab"], device=args.device), specs,
                                 args.starts_per_world, args.max_steps, args.seed)
        res["latency_per_step_s"] = (time.perf_counter() - t0) / max(1, sum(r["steps"] for r in res["rows"]))
        res["val_acc"] = meta.get("val_acc")
        key = os.path.basename(path).replace(".pt", "")
        results[key] = res
        print(f"{key}: success {res['success_rate']:.3f} invalid {res['invalid_rate']:.3f} "
              f"val_acc {res['val_acc']:.3f}")

    if args.teacher_json and os.path.exists(args.teacher_json):
        teachers = json.load(open(args.teacher_json))
        for name, r in teachers.items():
            results[name] = r

    save_json(args.out, results)
    print("saved", args.out)


if __name__ == "__main__":
    main()
