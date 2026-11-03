"""Tiny GPU smoke test: load Tacit, ask transition/policy questions."""

import json
import os
import sys

from jevwm.backends import TacitBackend
from jevwm.experiments import load_jsonl
from jevwm.questions import transition_questions
from jevwm.world import State, WorldSpec, render_state


def main():
    model = os.environ.get("TACIT_PATH", "/mnt/storage1/syk/jev-wm/models/Tacit-4B")
    backend = TacitBackend(model, device="cuda:0", batch_size=8)
    recs = load_jsonl("data/test.jsonl")[:2]
    items = []
    for rec in recs:
        spec = WorldSpec.from_dict(rec["world"])
        state = State.from_dict(rec["state"])
        text = render_state(state, spec)
        for q in transition_questions(state, rec["action"], spec):
            items.append({"state": text, "question": q["question"], "options": q["options"], "kind": q["kind"]})
    out = backend.decide_items(items)
    for it, r in zip(items, out):
        print(json.dumps({"q": it["question"][:90], "answer": r["answer"],
                          "probs": {k: round(v, 3) for k, v in r["probs"].items()}}, ensure_ascii=False))
    print("OK", len(out))


if __name__ == "__main__":
    main()
