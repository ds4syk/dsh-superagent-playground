"""Backends: JEV-style decision models (Tacit / AnyJev readout), a generative
LLM baseline, and the trivial persistence baseline.

All backends expose ``predict_transitions(records)`` returning one dict per
record with the predicted variable values, an ``executes`` flag, and timing.
Model code is imported lazily so the CPU-side data pipeline stays light.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .questions import transition_questions
from .world import (
    ITEMS,
    State,
    WorldSpec,
    action_text,
    render_state,
    state_from_values,
    variable_options,
    variable_values,
)


# ---------------------------------------------------------------------------
# Disk cache


class Cache:
    def __init__(self, path: Optional[str] = None):
        self.conn = None
        if path:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self.conn = sqlite3.connect(path)
            self.conn.execute("CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT)")
            self.conn.commit()

    def get(self, key: str):
        if self.conn is None:
            return None
        row = self.conn.execute("SELECT v FROM kv WHERE k=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key: str, value: Any) -> None:
        if self.conn is None:
            return
        self.conn.execute("INSERT OR REPLACE INTO kv (k,v) VALUES (?,?)", (key, json.dumps(value)))
        self.conn.commit()

    def close(self):
        if self.conn is not None:
            self.conn.close()


def _cache_key(model: str, state_text: str, question: str, options: Sequence[str], kind: str) -> str:
    payload = json.dumps([model, state_text, question, list(options), kind], sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Result container


@dataclass
class TransitionPrediction:
    executes: bool
    values: Dict[str, str]
    latency_s: float = 0.0
    decisions: int = 0
    parse_fail: bool = False

    def state(self, spec: WorldSpec) -> State:
        if not self.executes:
            # convention used by the paper: a failed command leaves state intact
            return state_from_values(self.values, spec)
        return state_from_values(self.values, spec)


def _apply_answers(record: dict, answers: Dict[str, str]) -> TransitionPrediction:
    spec = WorldSpec.from_dict(record["world"])
    state = State.from_dict(record["state"])
    gt = variable_values(state, spec)
    # For a failed command the next state is the current state; JEV is still
    # asked about all variables, but we override with the engine convention.
    exec_ans = answers.get("executes", "Yes")
    executes = exec_ans.lower() in ("yes", "true", "1") if isinstance(exec_ans, str) else bool(exec_ans)
    vals = {k: answers.get(k, gt[k]) for k in gt}
    if not executes:
        vals = gt
    pred = TransitionPrediction(executes=executes, values=vals)
    return pred


# ---------------------------------------------------------------------------
# JEV-style models


class TacitBackend:
    """A self-distilled open JEV-style decision model (AnyJev Tacit)."""

    def __init__(self, model_path: str, device: str = "cuda:0", batch_size: int = 16,
                 cache_path: Optional[str] = None, dtype: str = "bfloat16"):
        from anyjev import Tacit

        self.name = f"tacit:{os.path.basename(model_path.rstrip('/'))}"
        self.tacit = Tacit.from_pretrained(model_path, dtype=dtype, device_map=device,
                                           batch_size=batch_size)
        self.cache = Cache(cache_path)

    def _items(self, record: dict):
        spec = WorldSpec.from_dict(record["world"])
        state = State.from_dict(record["state"])
        text = render_state(state, spec)
        qs = transition_questions(state, record["action"], spec)
        items = []
        for q in qs:
            item = {
                "id": q["id"],
                "state": text,
                "question": q["question"],
                "options": q["options"],
                "kind": q["kind"],
            }
            items.append(item)
        return items

    def predict_transitions(self, records: Sequence[dict]) -> List[TransitionPrediction]:
        all_items: List[dict] = []
        owners: List[Tuple[int, str]] = []
        for i, rec in enumerate(records):
            for it in self._items(rec):
                all_items.append(it)
                owners.append((i, it["id"]))
        t0 = time.perf_counter()
        raw = self.tacit.decide_batch(all_items)
        elapsed = time.perf_counter() - t0
        answers: List[Dict[str, str]] = [{} for _ in records]
        for (i, qid), r in zip(owners, raw):
            answers[i][qid] = r["answer"]
        out = []
        for rec, ans in zip(records, answers):
            pred = _apply_answers(rec, ans)
            pred.latency_s = elapsed / max(1, len(records))
            pred.decisions = len(ans)
            out.append(pred)
        return out

    def decide_items(self, items: Sequence[dict]) -> List[dict]:
        t0 = time.perf_counter()
        out = self.tacit.decide_batch(list(items))
        elapsed = time.perf_counter() - t0
        for r in out:
            r["_elapsed"] = elapsed / max(1, len(out))
        return out


class AnyJevDeciderBackend:
    """Training-free JEV-style readout on a stock LLM (AnyJev Decider)."""

    def __init__(self, model_path: str, device: str = "cuda:0", level: str = "raw",
                 cache_path: Optional[str] = None):
        from anyjev import Decider
        from anyjev.backends.hf import HFBackend

        self.name = f"decider:{os.path.basename(model_path.rstrip('/'))}:{level}"
        self.level = level
        self.decider = Decider(HFBackend(model_path, device=device), level=level)
        self.cache = Cache(cache_path)

    def decide_items(self, items: Sequence[dict]) -> List[dict]:
        from anyjev import Question

        out = []
        for it in items:
            if it.get("kind") == "noul":
                q = Question.noul(it["question"], name=it.get("id"))
            else:
                q = Question.choice(it["question"], it["options"], name=it.get("id"))
            r = self.decider.decide(it["state"], [q], level=self.level)[q.id]
            dist = r.distribution
            out.append({"answer": max(dist, key=dist.get), "probs": dist})
        return out

    def predict_transitions(self, records: Sequence[dict]) -> List[TransitionPrediction]:
        from anyjev import Question

        out: List[TransitionPrediction] = []
        t_all = time.perf_counter()
        for rec in records:
            spec = WorldSpec.from_dict(rec["world"])
            state = State.from_dict(rec["state"])
            text = render_state(state, spec)
            qs = transition_questions(state, rec["action"], spec)
            questions = []
            for q in qs:
                if q["kind"] == "noul":
                    questions.append(Question.noul(q["question"], name=q["id"]))
                else:
                    questions.append(Question.choice(q["question"], q["options"], name=q["id"]))
            ds = self.decider.decide(text, questions, level=self.level)
            ans = {}
            for q in qs:
                d = ds[q["id"]]
                dist = d.distribution
                ans[q["id"]] = max(dist, key=dist.get)
            pred = _apply_answers(rec, ans)
            out.append(pred)
        elapsed = time.perf_counter() - t_all
        per = elapsed / max(1, len(records))
        for p in out:
            p.latency_s = per
            p.decisions = 8
        return out


# ---------------------------------------------------------------------------
# Generative baseline


GEN_SCHEMA_INSTRUCTION = """You predict the next state of a deterministic text adventure.

Current state:
{state}

The agent executes: {action}

Reply with a single JSON object and nothing else. Use exactly these keys and values:
{{
  "executes": true or false,
  "player_room": one of {rooms},
  "key_location": one of {locs},
  "apple_location": one of {locs},
  "mushroom_location": one of {locs},
  "door_mode": one of {door_modes},
  "box_mode": one of {box_modes},
  "holding": one of {holdings}
}}
If the command does not execute, the state does not change."""


class GenerativeBackend:
    def __init__(self, model_path: str, device: str = "cuda:0", max_new_tokens: int = 220,
                 batch_size: int = 8):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = f"gen:{os.path.basename(model_path.rstrip('/'))}"
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.bfloat16, device_map=device
        )
        self.model.eval()
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token

    def _prompt(self, record: dict) -> str:
        spec = WorldSpec.from_dict(record["world"])
        state = State.from_dict(record["state"])
        opts = variable_options(spec)
        instruction = GEN_SCHEMA_INSTRUCTION.format(
            state=render_state(state, spec),
            action=action_text(record["action"], spec),
            rooms=json.dumps(opts["player_room"]),
            locs=json.dumps(opts["key_location"]),
            door_modes=json.dumps(opts["door_mode"]),
            box_modes=json.dumps(opts["box_mode"]),
            holdings=json.dumps(opts["holding"]),
        )
        msgs = [{"role": "user", "content": instruction}]
        try:
            return self.tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    @staticmethod
    def _extract_json(text: str) -> Optional[dict]:
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None

    def _parse(self, text: str, record: dict) -> Tuple[Dict[str, str], bool, bool]:
        spec = WorldSpec.from_dict(record["world"])
        state = State.from_dict(record["state"])
        opts = variable_options(spec)
        gt = variable_values(state, spec)
        obj = self._extract_json(text)
        if obj is None:
            return gt, True, False
        vals = {}
        for key in ("player_room", "key_location", "apple_location", "mushroom_location",
                    "door_mode", "box_mode", "holding"):
            raw = str(obj.get(key, "")).strip().lower()
            options = opts[key]
            hit = None
            for o in options:
                if raw == o.lower():
                    hit = o
                    break
            if hit is None:
                for o in options:
                    if raw and (raw in o.lower() or o.lower() in raw):
                        hit = o
                        break
            vals[key] = hit if hit is not None else gt[key]
        exec_raw = obj.get("executes", True)
        if isinstance(exec_raw, str):
            exec_bool = exec_raw.strip().lower() in ("true", "yes", "1")
        else:
            exec_bool = bool(exec_raw)
        if not exec_bool:
            vals = gt
        return vals, False, exec_bool

    def predict_transitions(self, records: Sequence[dict]) -> List[TransitionPrediction]:
        out: List[TransitionPrediction] = []
        t_all = time.perf_counter()
        for start in range(0, len(records), self.batch_size):
            chunk = records[start:start + self.batch_size]
            prompts = [self._prompt(r) for r in chunk]
            enc = self.tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(self.model.device)
            with self.torch.no_grad():
                gen = self.model.generate(
                    **enc, max_new_tokens=self.max_new_tokens, do_sample=False,
                    pad_token_id=self.tok.pad_token_id,
                )
            texts = self.tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            for rec, text in zip(chunk, texts):
                vals, fail, executes = self._parse(text, rec)
                pred = TransitionPrediction(executes=executes, values=vals, parse_fail=fail)
                pred.decisions = 1
                out.append(pred)
        elapsed = time.perf_counter() - t_all
        per = elapsed / max(1, len(records))
        for p in out:
            p.latency_s = per
        return out


# ---------------------------------------------------------------------------
# Baseline


class PersistenceBackend:
    name = "persistence"

    def predict_transitions(self, records: Sequence[dict]) -> List[TransitionPrediction]:
        out = []
        for rec in records:
            state = State.from_dict(rec["state"])
            spec = WorldSpec.from_dict(rec["world"])
            # the floor: nothing changes and we do not know whether it ran
            pred = TransitionPrediction(executes=False, values=variable_values(state, spec))
            pred.decisions = 0
            out.append(pred)
        return out
