"""Shared helpers for the JEV experiments."""

from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Sequence

from .world import ACTION_KIND, WorldSpec, State, state_from_values, step, variable_values


def load_jsonl(path: str) -> List[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def save_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)


def make_backend(kind: str, model_path: str = "", device: str = "cuda:0", level: str = "raw"):
    if kind == "persistence":
        from .backends import PersistenceBackend
        return PersistenceBackend()
    if kind == "tacit":
        from .backends import TacitBackend
        return TacitBackend(model_path, device=device)
    if kind == "decider":
        from .backends import AnyJevDeciderBackend
        return AnyJevDeciderBackend(model_path, device=device, level=level)
    if kind == "generative":
        from .backends import GenerativeBackend
        return GenerativeBackend(model_path, device=device)
    raise ValueError(kind)


VARIABLES = (
    "player_room",
    "key_location",
    "apple_location",
    "mushroom_location",
    "door_mode",
    "box_mode",
    "holding",
)


def evaluate_predictions(records: Sequence[dict], preds: Sequence) -> Dict:
    n = len(records)
    var_correct = {v: 0 for v in VARIABLES}
    exact = 0
    exec_correct = 0
    parse_fail = 0
    by_exec = {"true": [0, 0], "false": [0, 0]}  # correct, total
    by_kind = {"observe": [0, 0], "simple": [0, 0], "derive": [0, 0]}
    for rec, p in zip(records, preds):
        vals = p.values
        ok_all = True
        for v in VARIABLES:
            if vals.get(v) == rec["next_values"][v]:
                var_correct[v] += 1
            else:
                ok_all = False
        if p.executes == rec["executes"]:
            exec_correct += 1
        if ok_all:
            exact += 1
        key = "true" if rec["executes"] else "false"
        by_exec[key][1] += 1
        if ok_all:
            by_exec[key][0] += 1
        parse_fail += int(bool(getattr(p, "parse_fail", False)))
        kind = ACTION_KIND.get(rec["action"], "derive")
        by_kind[kind][1] += 1
        if ok_all:
            by_kind[kind][0] += 1
    return {
        "n": n,
        "exact_match": exact / n,
        "exec_accuracy": exec_correct / n,
        "variable_accuracy": {v: var_correct[v] / n for v in VARIABLES},
        "macro_variable_accuracy": sum(var_correct.values()) / (n * len(VARIABLES)),
        "exact_by_exec": {k: (v[0] / v[1] if v[1] else None) for k, v in by_exec.items()},
        "count_by_exec": {k: v[1] for k, v in by_exec.items()},
        "parse_fail_rate": parse_fail / n,
        "exact_by_action_kind": {k: (v[0] / v[1] if v[1] else None) for k, v in by_kind.items()},
        "count_by_action_kind": {k: v[1] for k, v in by_kind.items()},
        "mean_latency_s": sum(p.latency_s for p in preds) / n,
        "decisions": sum(getattr(p, "decisions", 0) for p in preds),
    }


def unique_start_states(records: Sequence[dict], limit: int = 80, seed: int = 0) -> List[dict]:
    import random

    rng = random.Random(seed)
    seen = set()
    chosen = []
    for rec in records:
        key = tuple(sorted(rec["state"].items()))
        if key in seen:
            continue
        seen.add(key)
        chosen.append(rec)
    rng.shuffle(chosen)
    return chosen[:limit]


def make_rollout_actions(spec: WorldSpec, state: State, rng, horizon: int = 4) -> List[str]:
    from .world import ACTIONS, admissible_actions

    acts = []
    for _ in range(horizon):
        if rng.random() < 0.7:
            adm = [a for a in admissible_actions(state) if a != "look"]
            a = rng.choice(adm) if adm else "look"
        else:
            a = rng.choice(list(ACTIONS))
        acts.append(a)
        state, _ = step(state, a, spec)
    return acts


def rollout_eval(backend, records: Sequence[dict], limit: int = 80, horizon: int = 4,
                 seed: int = 1) -> Dict:
    """Open-loop k-step prediction: feed the model's own predicted state back."""
    import random

    rng = random.Random(seed)
    starts = unique_start_states(records, limit=limit, seed=seed)
    plans = []
    for rec in starts:
        spec = WorldSpec.from_dict(rec["world"])
        s0 = State.from_dict(rec["state"])
        actions = make_rollout_actions(spec, s0, rng, horizon)
        s = s0
        engine_states = [s]
        for a in actions:
            s, _ = step(s, a, spec)
            engine_states.append(s)
        plans.append({"spec": spec, "s0": s0, "actions": actions, "engine": engine_states})

    cur = [p["s0"] for p in plans]
    exact_by_k = {k: 0 for k in range(1, horizon + 1)}
    var_by_k = {k: {v: 0 for v in VARIABLES} for k in range(1, horizon + 1)}
    for t in range(horizon):
        recs = []
        for p, s in zip(plans, cur):
            recs.append({
                "world": p["spec"].to_dict(),
                "state": s.to_dict(),
                "action": p["actions"][t],
            })
        preds = backend.predict_transitions(recs)
        for i, p in enumerate(plans):
            s_next = preds[i].state(p["spec"])
            cur[i] = s_next
            gt = p["engine"][t + 1]
            ok = True
            for v in VARIABLES:
                pred_v = variable_values(s_next, p["spec"])[v]
                gt_v = variable_values(gt, p["spec"])[v]
                if pred_v == gt_v:
                    var_by_k[t + 1][v] += 1
                else:
                    ok = False
            exact_by_k[t + 1] += int(ok)
    n = len(plans)
    return {
        "n": n,
        "exact_by_k": {k: v / n for k, v in exact_by_k.items()},
        "var_by_k": {k: {v: c / n for v, c in d.items()} for k, d in var_by_k.items()},
    }


def evaluate_policy(policy, specs, starts_per_world: int = 4, max_steps: int = 20,
                    seed: int = 0, collect_labels: bool = False, teacher_name: str = ""):
    """Run a policy on a fixed set of worlds/starts.  Returns (summary, labels)."""
    import random
    from .policies import make_starts, run_episode
    from .world import State, render_state

    rng = random.Random(seed)
    rows, labels = [], []
    t0 = time.perf_counter()
    for spec in specs:
        starts = make_starts(spec, starts_per_world, rng)
        for si, start in enumerate(starts):
            res = run_episode(policy, spec, start, max_steps=max_steps)
            rows.append({"world_id": spec.world_id, "start": si, "success": res.success,
                         "steps": res.steps, "invalid": res.invalid, "decisions": res.decisions})
            if collect_labels:
                for k, (sdict, a) in enumerate(zip(res.states, res.actions)):
                    labels.append({
                        "state": sdict, "world": spec.to_dict(), "action": a,
                        "state_text": render_state(State.from_dict(res.states[k]), spec),
                        "teacher": teacher_name, "world_id": spec.world_id,
                    })
    wall = time.perf_counter() - t0
    n = len(rows)
    by_start = {}
    for i in sorted({r["start"] for r in rows}):
        sel = [r for r in rows if r["start"] == i]
        by_start[str(i)] = sum(r["success"] for r in sel) / len(sel)
    return {
        "n_episodes": n,
        "success_by_start": by_start,
        "success_rate": sum(r["success"] for r in rows) / n,
        "mean_steps": sum(r["steps"] for r in rows) / n,
        "mean_invalid": sum(r["invalid"] for r in rows) / n,
        "invalid_rate": sum(r["invalid"] for r in rows) / max(1, sum(r["steps"] for r in rows)),
        "decisions": policy.stats.decisions,
        "decisions_per_episode": policy.stats.decisions / n,
        "wall_s": wall,
        "episodes_per_s": n / wall,
        "rows": rows,
    }, labels
