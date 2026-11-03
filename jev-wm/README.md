# JEV world-model pre-experiments

Code for the two pre-experiments in the report:

- **Exp1**: a frozen JEV-style decision model (Tacit-4B) predicts the next state of a
  deterministic text gridworld as typed multiple-choice answers; compared with a
  Qwen3-4B typed readout, a generative Qwen3-4B, and persistence.
- **Exp2**: three ways to produce actions with the same model (direct choice,
  per-action progress score, one-step world-model lookahead), then distill the best
  teacher into a ~1M-parameter transformer student and evaluate closed-loop.

## Layout

```
jevwm/world.py        deterministic world, state variables, transitions, oracle
jevwm/questions.py    typed query compiler
jevwm/backends.py     Tacit / AnyJev typed readout / generative / persistence
jevwm/policies.py     choice / score / world-model lookahead / student / oracle
jevwm/student.py      tiny transformer policy + training
jevwm/experiments.py  evaluation and rollout helpers
scripts/              data generation, Exp1, Exp2, figures
sbatch/               Slurm jobs
data/                 generated transitions and labels (gitignored)
results/              JSON results, student checkpoints, figures
```

## Quick start

```bash
conda activate jev
export PYTHONPATH=.
python scripts/gen_data.py --out data --per-world 80
python scripts/exp1.py --backend persistence --limit 200
python scripts/exp2_train.py --labels data/labels_oracle.jsonl --out results/student_oracle.pt
python scripts/exp2_eval.py --students results/student_oracle.pt --out results/exp2_eval.json
```

GPU jobs: `sbatch sbatch/exp1_tacit.sbatch`, `sbatch sbatch/exp2_teachers.sbatch`.

## Models

- `Tacit-4B`: self-distilled AnyJev model, `morriszjm/Tacit-4B` (Qwen3-4B base).
- `Qwen3-4B-Instruct-2507`: typed readout via AnyJev `Decider(level="raw")`, and
  generative baseline.
- All local weights under `/mnt/storage1/syk/jev-wm/models/`.

## Notes

- The test split (`test`) uses unseen object names; Exp2 uses `testsame` (same
  vocabulary, new layouts) so distillation is not confounded with lexical
  generalization.
- Policies mask non-admissible commands (an action interface is assumed); the
  underlying transitions still include invalid commands for Exp1.
- `scripts/analyze.py` writes the report tables and figures.

See `../RESULTS.md` for setup, numbers and limitations.
