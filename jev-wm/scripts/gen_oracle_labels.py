"""Oracle action labels on a broad state distribution (BC upper bound)."""

import argparse
import json
import random
import os
from jevwm.world import ACTIONS, make_specs, oracle_action, render_state, initial_state, step, random_action
from gen_data import _collect_states


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/labels_oracle.jsonl")
    ap.add_argument("--worlds", type=int, default=40)
    ap.add_argument("--per-world", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    rows = []
    for spec in make_specs("train", args.worlds):
        states = _collect_states(spec, rng, n_walk=200, n_branch=100)
        rng.shuffle(states)
        for s in states[: args.per_world]:
            rows.append({"state": s.to_dict(), "world": spec.to_dict(), "action": oracle_action(s, spec),
                         "state_text": render_state(s, spec), "teacher": "oracle", "world_id": spec.world_id})
        # on-policy trajectory states
        s = initial_state(spec)
        for _ in range(20):
            rows.append({"state": s.to_dict(), "world": spec.to_dict(), "action": oracle_action(s, spec),
                         "state_text": render_state(s, spec), "teacher": "oracle", "world_id": spec.world_id})
            s, _ = step(s, oracle_action(s, spec), spec)
    seen, uniq = set(), []
    for r in rows:
        key = (r["world_id"], tuple(sorted(r["state"].items())), r["action"])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(r)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        for r in uniq:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("wrote", len(uniq), "labels to", args.out)


if __name__ == "__main__":
    main()
