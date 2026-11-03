"""Action-generation policies used to (a) compare what JEV can do zero-shot and
(b) produce labels for distillation."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .questions import action_text, policy_question, progress_question
from .world import (
    ACTIONS,
    State,
    WorldSpec,
    action_text as _action_text,
    admissible_actions,
    goal_reached,
    holding,
    initial_state,
    random_action,
    render_goal,
    render_state,
    step,
    variable_values,
)


def _score_state(spec: WorldSpec, s: State, executed: bool, action: str,
                 cur: State) -> float:
    """Heuristic utility used by the world-model planner.  It is intentionally
    simple and identical for every model, so differences come from prediction."""
    if goal_reached(s):
        return 100.0
    if not executed:
        return -20.0
    sc = 0.0
    if s.apple == "i":
        sc += 10.0
    if s.player == "b":
        sc += 5.0
    if s.door == "open":
        sc += 4.0
    if s.key == "i":
        sc += 2.0
    if s.box == "open":
        sc += 1.0
    # prefer progress relative to the current state
    if s.apple == "i" and cur.apple != "i":
        sc += 2.0
    if s.player == "b" and cur.player != "b":
        sc += 2.0
    if s.door == "open" and cur.door != "open":
        sc += 2.0
    return sc


@dataclass
class PolicyStats:
    decisions: int = 0
    forward_calls: int = 0


class BasePolicy:
    name = "base"

    def __init__(self, mask_invalid: bool = True):
        self.stats = PolicyStats()
        self.mask_invalid = mask_invalid

    def candidates(self, state: State) -> List[str]:
        if self.mask_invalid:
            acts = [a for a in admissible_actions(state) if a != "look"]
            return acts or ["look"]
        return list(ACTIONS)

    def act(self, state: State, spec: WorldSpec) -> str:
        raise NotImplementedError


class OraclePolicy(BasePolicy):
    name = "oracle"

    def act(self, state: State, spec: WorldSpec) -> str:
        from .world import oracle_action
        return oracle_action(state, spec)


class RandomPolicy(BasePolicy):
    name = "random"

    def __init__(self, seed: int = 0, mask_invalid: bool = True):
        super().__init__(mask_invalid=mask_invalid)
        self.rng = random.Random(seed)

    def act(self, state: State, spec: WorldSpec) -> str:
        return random_action(state, self.rng)


class ChoicePolicy(BasePolicy):
    """Ask JEV directly for the next command (a 9-way decision)."""

    name = "jev_choice"

    def __init__(self, backend, mask_invalid: bool = True):
        super().__init__(mask_invalid=mask_invalid)
        self.backend = backend

    def act(self, state: State, spec: WorldSpec) -> str:
        cands = self.candidates(state)
        if len(cands) == 1:
            return cands[0]
        text = render_state(state, spec)
        q = policy_question(state, spec, cands)
        item = {"state": text, "question": q["question"], "options": q["options"], "kind": "choice"}
        out = self.backend.decide_items([item])[0]
        self.stats.decisions += 1
        self.stats.forward_calls += 1
        ans = out["answer"]
        for a in cands:
            if action_text(a, spec) == ans:
                return a
        return random.choice(cands)


class ScorePolicy(BasePolicy):
    """Ask yes/no ``does this action make progress?`` for each candidate and
    take the highest probability (a JEV-as-value-function baseline)."""

    name = "jev_score"

    def __init__(self, backend, mask_invalid: bool = True):
        super().__init__(mask_invalid=mask_invalid)
        self.backend = backend

    def act(self, state: State, spec: WorldSpec) -> str:
        cands = self.candidates(state)
        if len(cands) == 1:
            return cands[0]
        text = render_state(state, spec) + " Available commands: " + ", ".join(action_text(a, spec) for a in cands) + ". Goal: " + render_goal(spec)
        items = []
        for a in cands:
            q = progress_question(state, spec, a)
            items.append({"state": text, "question": q["question"], "options": q["options"], "kind": "noul"})
        outs = self.backend.decide_items(items)
        self.stats.decisions += len(items)
        self.stats.forward_calls += 1
        best_a, best_p = cands[0], -1.0
        for a, out in zip(cands, outs):
            p = out.get("probs", {}).get("Yes", out.get("probs", {}).get("yes", 0.0))
            if p > best_p:
                best_a, best_p = a, p
        return best_a


class WorldModelPolicy(BasePolicy):
    """One-step lookahead with the decision model as a factored world model:
    predict the next state for every candidate action and rank by utility."""

    name = "jev_wm_lookahead"

    def __init__(self, backend, mask_invalid: bool = True):
        super().__init__(mask_invalid=mask_invalid)
        self.backend = backend

    def act(self, state: State, spec: WorldSpec) -> str:
        cands = self.candidates(state)
        if len(cands) == 1:
            return cands[0]
        records = []
        for a in cands:
            records.append({
                "world": spec.to_dict(),
                "state": state.to_dict(),
                "action": a,
            })
        preds = self.backend.predict_transitions(records)
        self.stats.decisions += sum(getattr(p, "decisions", 8) for p in preds)
        self.stats.forward_calls += 1
        best_a, best_s = cands[0], -1e9
        for a, p in zip(cands, preds):
            s_pred = p.state(spec)
            s = _score_state(spec, s_pred, p.executes, a, state)
            if not self.mask_invalid and a not in admissible_actions(state):
                s -= 10.0
            if s > best_s:
                best_a, best_s = a, s
        return best_a


class StudentPolicy(BasePolicy):
    name = "student"

    def __init__(self, student, vocab, device: str = "cpu", mask_invalid: bool = True):
        super().__init__(mask_invalid=mask_invalid)
        self.student = student
        self.vocab = vocab
        self.device = device

    def act(self, state: State, spec: WorldSpec) -> str:
        import torch
        from .student import encode_text

        text = render_state(state, spec)
        ids = encode_text(text, self.vocab)
        x = torch.tensor([ids], dtype=torch.long, device=self.device)
        with torch.no_grad():
            logits = self.student(x)[0]
        if self.mask_invalid:
            allowed = set(self.candidates(state))
            for i, a in enumerate(ACTIONS[: logits.numel()]):
                if a not in allowed:
                    logits[i] = -1e9
        a_idx = int(logits.argmax(-1).item())
        self.stats.forward_calls += 1
        return ACTIONS[a_idx]


@dataclass
class EpisodeResult:
    success: bool
    steps: int
    invalid: int
    decisions: int
    states: List[dict] = field(default_factory=list)
    actions: List[str] = field(default_factory=list)


def run_episode(policy: BasePolicy, spec: WorldSpec, start: Optional[State] = None,
                max_steps: int = 20) -> EpisodeResult:
    state = start if start is not None else initial_state(spec)
    states = [state.to_dict()]
    actions: List[str] = []
    invalid = 0
    for _ in range(max_steps):
        if goal_reached(state):
            break
        a = policy.act(state, spec)
        nxt, executed = step(state, a, spec)
        invalid += int(not executed)
        actions.append(a)
        state = nxt
        states.append(state.to_dict())
    return EpisodeResult(
        success=goal_reached(state),
        steps=len(actions),
        invalid=invalid,
        decisions=policy.stats.decisions,
        states=states,
        actions=actions,
    )


def make_starts(spec: WorldSpec, n: int, rng: random.Random) -> List[State]:
    starts = [initial_state(spec)]
    while len(starts) < n:
        s = initial_state(spec)
        for _ in range(rng.randint(1, 5)):
            s, _ = step(s, random_action(s, rng), spec)
        starts.append(s)
    return starts
