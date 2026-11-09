import json, torch, time
torch.set_num_threads(3)
from jevwm.backends import TacitBackend
from jevwm.experiments import load_jsonl
from jevwm.questions import transition_questions
from jevwm.world import State, WorldSpec, render_state
b = TacitBackend("/mnt/storage1/syk/jev-wm/models/Tacit-1.7B", device="cpu", batch_size=2)
rec = load_jsonl("data/test.jsonl")[0]
spec = WorldSpec.from_dict(rec["world"]); state = State.from_dict(rec["state"])
text = render_state(state, spec)
items = [{"state": text, "question": q["question"], "options": q["options"], "kind": q["kind"]}
         for q in transition_questions(state, rec["action"], spec)]
t0 = time.time(); out = b.decide_items(items); print("elapsed", time.time()-t0)
for it, r in zip(items, out):
    print(it["question"][:80], "=>", r["answer"], {k: round(v,2) for k,v in r["probs"].items()})
print("GT executes", rec["executes"], "next", rec["next_values"])
