"""Pilot: compare two open JEV-style models on the same handful of cases."""

import json
import os
import random
import time

from jevwm.experiments import evaluate_policy, evaluate_predictions, load_jsonl, make_backend
from jevwm.policies import ChoicePolicy, ScorePolicy
from jevwm.world import make_specs

TACIT = "/mnt/storage1/syk/jev-wm/models/Tacit-4B"
QWEN = "/mnt/storage1/syk/jev-wm/models/Qwen3-4B-Instruct-2507"


def run(name, backend, records, worlds, starts, max_steps=8):
    t0 = time.perf_counter()
    preds = backend.predict_transitions(records)
    m = evaluate_predictions(records, preds)
    m["wall_s"] = time.perf_counter() - t0
    print(f"== {name} ==")
    print(json.dumps({k: m[k] for k in ("exact_match", "exec_accuracy", "macro_variable_accuracy",
                                        "exact_by_action_kind", "parse_fail_rate", "mean_latency_s", "wall_s")},
                     ensure_ascii=False, indent=1))
    return m


def main():
    recs = load_jsonl("data/test.jsonl")
    rng = random.Random(0)
    sub = rng.sample(recs, 120)
    specs = make_specs("test", 3)
    seen, probe6 = set(), []
    for r in recs:
        if r["action"] in seen:
            continue
        seen.add(r["action"])
        probe6.append(r)
        if len(probe6) == 9:
            break
    out = {}

    from jevwm.backends import TacitBackend
    b = TacitBackend(TACIT, device="cuda:0", batch_size=8)
    out["tacit4b_pred"] = run("tacit4b_pred", b, sub, specs, 2)
    out["tacit4b_probe9"] = run("tacit4b_probe9", b, probe6, specs, 1)
    for pol_cls, nm in ((ChoicePolicy, "choice"), (ScorePolicy, "score")):
        pol = pol_cls(b)
        res, _ = evaluate_policy(pol, specs, starts_per_world=1, max_steps=8, seed=0)
        print(nm, res["success_rate"], res["mean_steps"], res["invalid_rate"])
        out[f"tacit4b_{nm}"] = res
    del b
    import torch
    torch.cuda.empty_cache()

    b2 = make_backend("decider", QWEN, "cuda:0", "raw")
    out["qwen_typed_pred"] = run("qwen_typed_pred", b2, sub[:60], specs, 1)
    print(json.dumps(out, ensure_ascii=False, indent=1)[:2000])
    with open("results/pilot.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
