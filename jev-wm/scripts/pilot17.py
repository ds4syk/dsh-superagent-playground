import json, random
from jevwm.experiments import evaluate_policy, evaluate_predictions, load_jsonl
from jevwm.backends import TacitBackend
from jevwm.policies import ChoicePolicy, ScorePolicy
from jevwm.world import make_specs

b = TacitBackend("/mnt/storage1/syk/jev-wm/models/Tacit-1.7B", device="cuda:0", batch_size=8)
recs = random.Random(0).sample(load_jsonl("data/test.jsonl"), 120)
preds = b.predict_transitions(recs)
m = evaluate_predictions(recs, preds)
print("PRED", json.dumps({k: m[k] for k in ("exact_match","exec_accuracy","macro_variable_accuracy","exact_by_action_kind","mean_latency_s")}, ensure_ascii=False))
for cls, nm in ((ChoicePolicy,"choice"),(ScorePolicy,"score")):
    pol=cls(b); res,_=evaluate_policy(pol, make_specs("test",3), 1, 8, 0)
    print(nm, res["success_rate"], res["mean_steps"], res["invalid_rate"], res["decisions_per_episode"])
