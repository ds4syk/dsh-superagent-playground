"""Distil JEV-generated actions into the tiny transformer learner."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter

import torch
torch.set_num_threads(2)

from jevwm.experiments import load_jsonl, save_json
from jevwm.student import save_student, train_student
from jevwm.world import ACTIONS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", default="results/student.pt")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    recs = load_jsonl(args.labels)
    texts = [r["state_text"] for r in recs]
    labels = [ACTIONS.index(r["action"]) for r in recs]
    print("labels:", len(recs), Counter(r["action"] for r in recs))
    model, meta, stats = train_student(texts, labels, epochs=args.epochs, seed=args.seed,
                                       device=args.device)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    save_student(args.out, model, meta)
    save_json(os.path.join(os.path.dirname(args.out) or ".", "exp2_student_meta.json"),
              {"n_labels": len(recs), **stats, "label_counts": dict(Counter(r["action"] for r in recs))})
    print("saved", args.out, "val_acc", meta["val_acc"])


if __name__ == "__main__":
    main()
