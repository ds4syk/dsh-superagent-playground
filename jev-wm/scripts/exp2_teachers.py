"""Evaluate zero-shot action generators built on a JEV-style decision model:
direct choice, per-action progress scores, and one-step world-model lookahead.
Also collects teacher labels for distillation."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import defaultdict
from typing import Dict, List

from jevwm.experiments import evaluate_policy, save_json
from jevwm.policies import (
    ChoicePolicy,
    OraclePolicy,
    RandomPolicy,
    ScorePolicy,
    StudentPolicy,
    WorldModelPolicy,
    make_starts,
    run_episode,
)
from jevwm.world import State, WorldSpec, initial_state, make_specs, render_state


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--backend", default="tacit", choices=["tacit", "decider"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--teachers", default="choice,score,wm")
    ap.add_argument("--worlds", type=int, default=48)
    ap.add_argument("--starts-per-world", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results")
    ap.add_argument("--labels-out", default="")
    ap.add_argument("--label-teacher", default="wm")
    ap.add_argument("--label-worlds", type=int, default=30)
    ap.add_argument("--label-starts", type=int, default=2)
    ap.add_argument("--label-offpolicy", type=int, default=6)
    args = ap.parse_args()

    from jevwm.experiments import make_backend

    backend = make_backend(args.backend, args.model_path, args.device, "raw")
    specs = make_specs("testsame", args.worlds)

    results: Dict[str, dict] = {}
    baselines = {"random": RandomPolicy(seed=args.seed), "oracle": OraclePolicy()}
    for name, pol in baselines.items():
        res, _ = evaluate_policy(pol, specs, args.starts_per_world, args.max_steps, args.seed)
        results[name] = res
        print(f"{name}: success {res['success_rate']:.3f} steps {res['mean_steps']:.1f} "
              f"invalid {res['invalid_rate']:.3f} decisions/ep {res['decisions_per_episode']:.1f} "
              f"wall {res['wall_s']:.1f}s")

    teachers = [t.strip() for t in args.teachers.split(",") if t.strip()]
    policy_map = {"choice": ChoicePolicy, "score": ScorePolicy, "wm": WorldModelPolicy}
    for name in teachers:
        pol = policy_map[name](backend)
        res, _ = evaluate_policy(pol, specs, args.starts_per_world, args.max_steps, args.seed)
        results[name] = res
        print(f"{name}: success {res['success_rate']:.3f} steps {res['mean_steps']:.1f} "
              f"invalid {res['invalid_rate']:.3f} decisions/ep {res['decisions_per_episode']:.1f} "
              f"wall {res['wall_s']:.1f}s")

    tag = os.path.basename(args.model_path.rstrip("/"))
    save_json(os.path.join(args.out, f"exp2_teachers_{tag}.json"), results)

    if args.labels_out:
        label_specs = make_specs("train", args.label_worlds)
        pol = policy_map[args.label_teacher](backend)
        res, labels = evaluate_policy(pol, label_specs, args.label_starts, args.max_steps, args.seed + 1,
                                      collect_labels=True, teacher_name=args.label_teacher)
        # off-policy coverage: random-walk / solution-branch states that the
        # teacher would not visit on its own but the student might.
        if args.label_offpolicy > 0:
            from gen_data import _collect_states
            for spec in label_specs:
                for st in _collect_states(spec, random.Random(args.seed + 7), n_walk=12, n_branch=8)[: args.label_offpolicy]:
                    a = pol.act(st, spec)
                    labels.append({"state": st.to_dict(), "world": spec.to_dict(), "action": a,
                                   "state_text": render_state(st, spec), "teacher": args.label_teacher,
                                   "world_id": spec.world_id})
        # de-duplicate on (state, action)
        seen = set()
        uniq = []
        for rec in labels:
            key = (rec["world_id"], tuple(sorted(rec["state"].items())), rec["action"])
            if key in seen:
                continue
            seen.add(key)
            uniq.append(rec)
        os.makedirs(os.path.dirname(args.labels_out) or ".", exist_ok=True)
        with open(args.labels_out, "w") as f:
            for rec in uniq:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"labels: {len(uniq)} unique from {res['n_episodes']} episodes, "
              f"success {res['success_rate']:.3f}")


if __name__ == "__main__":
    main()
