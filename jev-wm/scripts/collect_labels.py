"""Collect action labels from one JEV teacher on train worlds."""

import argparse, json, os, random
from jevwm.backends import TacitBackend
from jevwm.policies import ChoicePolicy, ScorePolicy
from jevwm.experiments import evaluate_policy
from jevwm.world import make_specs, render_state
from gen_data import _collect_states


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="/mnt/storage1/syk/jev-wm/models/Tacit-4B")
    ap.add_argument("--teacher", default="score", choices=["choice", "score"])
    ap.add_argument("--worlds", type=int, default=30)
    ap.add_argument("--starts", type=int, default=2)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--offpolicy", type=int, default=8)
    ap.add_argument("--out", default="data/labels_score.jsonl")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    b = TacitBackend(args.model_path, device=args.device)
    pol = (ScorePolicy if args.teacher == "score" else ChoicePolicy)(b)
    specs = make_specs("train", args.worlds)
    res, labels = evaluate_policy(pol, specs, args.starts, args.steps, 1, collect_labels=True,
                                  teacher_name=args.teacher)
    print("on-policy", res["success_rate"], "labels", len(labels), flush=True)
    rng = random.Random(7)
    for spec in specs:
        for st in _collect_states(spec, rng, n_walk=12, n_branch=8)[: args.offpolicy]:
            a = pol.act(st, spec)
            labels.append({"state": st.to_dict(), "world": spec.to_dict(), "action": a,
                           "state_text": render_state(st, spec), "teacher": args.teacher,
                           "world_id": spec.world_id})
    seen, uniq = set(), []
    for r in labels:
        key = (r["world_id"], tuple(sorted(r["state"].items())), r["action"])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(r)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        for r in uniq:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("wrote", len(uniq), "to", args.out, flush=True)


if __name__ == "__main__":
    main()
