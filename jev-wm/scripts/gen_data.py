"""Generate offline transitions for the JEV world-model experiments."""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Dict, List

from jevwm.world import (
    ACTIONS,
    State,
    WorldSpec,
    admissible_actions,
    initial_state,
    make_specs,
    random_action,
    render_state,
    solution_path,
    step,
    variable_values,
)


def _collect_states(spec: WorldSpec, rng: random.Random, n_walk: int = 24, n_branch: int = 16) -> List[State]:
    states: List[State] = [initial_state(spec)]
    s = initial_state(spec)
    for _ in range(n_walk):
        states.append(s)
        a = random_action(s, rng)
        s, _ = step(s, a, spec)
    sol = solution_path(spec)
    for _ in range(n_branch):
        s = initial_state(spec)
        k = rng.randint(0, len(sol))
        for a in sol[:k]:
            s, _ = step(s, a, spec)
        for _ in range(rng.randint(0, 3)):
            s, _ = step(s, random_action(s, rng), spec)
        states.append(s)
    # de-duplicate
    seen = set()
    uniq = []
    for st in states:
        key = tuple(sorted(st.to_dict().items()))
        if key not in seen:
            seen.add(key)
            uniq.append(st)
    return uniq


def _sample_actions(state: State, rng: random.Random, n_change: int = 3, n_invalid: int = 4) -> List[str]:
    adm = set(admissible_actions(state))
    change = [a for a in ACTIONS if a in adm and a != "look"]
    invalid = [a for a in ACTIONS if a not in adm]
    pick = rng.sample(change, min(n_change, len(change)))
    pick.append("look")
    pick += rng.sample(invalid, min(n_invalid, len(invalid)))
    rng.shuffle(pick)
    return pick


def transitions_for_world(spec: WorldSpec, rng: random.Random, n: int) -> List[dict]:
    states = _collect_states(spec, rng)
    rng.shuffle(states)
    records: List[dict] = []
    seen = set()
    for state in states:
        for action in _sample_actions(state, rng):
            key = (tuple(sorted(state.to_dict().items())), action)
            if key in seen:
                continue
            seen.add(key)
            nxt, executed = step(state, action, spec)
            records.append(
                {
                    "id": f"{spec.world_id}-{len(records):03d}",
                    "world": spec.to_dict(),
                    "state": state.to_dict(),
                    "action": action,
                    "next_state": nxt.to_dict(),
                    "executes": bool(executed),
                    "state_text": render_state(state, spec),
                    "next_state_text": render_state(nxt, spec),
                    "values": variable_values(state, spec),
                    "next_values": variable_values(nxt, spec),
                }
            )
            if len(records) >= n:
                return records
    return records


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-worlds", type=int, default=40)
    ap.add_argument("--val-worlds", type=int, default=4)
    ap.add_argument("--test-worlds", type=int, default=24)
    ap.add_argument("--per-world", type=int, default=80)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = random.Random(args.seed)

    train_specs = make_specs("train", args.train_worlds + args.val_worlds)
    test_specs = make_specs("test", args.test_worlds)
    splits = {
        "train": train_specs[: args.train_worlds],
        "val": train_specs[args.train_worlds:],
        "test": test_specs,
    }

    counts: Dict[str, int] = {}
    for split, specs in splits.items():
        recs: List[dict] = []
        for spec in specs:
            recs.extend(transitions_for_world(spec, rng, args.per_world))
        with open(os.path.join(args.out, f"{split}.jsonl"), "w") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        counts[split] = len(recs)
        print(f"{split}: {len(specs)} worlds, {len(recs)} transitions")

    with open(os.path.join(args.out, "worlds.json"), "w") as f:
        json.dump({s: [w.to_dict() for w in specs] for s, specs in splits.items()}, f, indent=1)
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump({"args": vars(args), "counts": counts}, f, indent=1)


if __name__ == "__main__":
    main()
