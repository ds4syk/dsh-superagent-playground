"""One GPU allocation: Exp1 + Exp2 teachers/labels + student train + eval."""

import argparse
import os
import subprocess
import sys

PY = sys.executable


def run(cmd):
    print("\n$", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, env={**os.environ, "PYTHONPATH": "."})
    if r.returncode != 0:
        raise SystemExit(r.returncode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="/mnt/storage1/syk/jev-wm/models/Tacit-4B")
    ap.add_argument("--backend", default="tacit", choices=["tacit", "decider"])
    ap.add_argument("--exp1-limit", type=int, default=500)
    ap.add_argument("--exp1-rollouts", type=int, default=50)
    ap.add_argument("--worlds", type=int, default=48)
    ap.add_argument("--starts", type=int, default=1)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--skip-exp1", action="store_true")
    ap.add_argument("--skip-teachers", action="store_true")
    args = ap.parse_args()

    if not args.skip_exp1:
        if args.backend == "tacit":
            run([PY, "scripts/exp1.py", "--backend", "tacit", "--model-path", args.model_path,
                 "--limit", str(args.exp1_limit), "--rollouts", str(args.exp1_rollouts)])
        else:
            run([PY, "scripts/exp1.py", "--backend", "decider", "--model-path",
                 "/mnt/storage1/syk/jev-wm/models/Qwen3-4B-Instruct-2507",
                 "--level", "raw", "--limit", str(max(100, args.exp1_limit // 2)),
                 "--rollouts", str(max(20, args.exp1_rollouts // 2))])

    if not args.skip_teachers:
        run([PY, "scripts/exp2_teachers.py", "--model-path", args.model_path,
             "--teachers", "choice,score,wm", "--worlds", str(args.worlds),
             "--starts-per-world", str(args.starts), "--max-steps", str(args.steps),
             "--labels-out", "data/labels_wm.jsonl", "--label-teacher", "wm",
             "--label-worlds", "30", "--label-starts", "2", "--label-offpolicy", "8"])

    if os.path.exists("data/labels_oracle.jsonl"):
        run([PY, "scripts/exp2_train.py", "--labels", "data/labels_oracle.jsonl",
             "--out", "results/student_oracle.pt", "--device", "cpu", "--epochs", "60"])
    if os.path.exists("data/labels_wm.jsonl"):
        run([PY, "scripts/exp2_train.py", "--labels", "data/labels_wm.jsonl",
             "--out", "results/student_wm.pt", "--device", "cpu", "--epochs", "60"])
    students = ",".join(p for p in ["results/student_wm.pt", "results/student_oracle.pt"] if os.path.exists(p))
    if students:
        run([PY, "scripts/exp2_eval.py", "--students", students,
             "--teacher-json", "results/exp2_teachers_Tacit-4B.json",
             "--out", "results/exp2_eval.json"])
    run([PY, "scripts/analyze.py"])


if __name__ == "__main__":
    main()
