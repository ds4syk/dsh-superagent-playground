#!/bin/bash
set -e
cd /mnt/storage1/syk/.superagent-work/jamba-world-model-fdd0e082a6b1/jev-wm
export PYTHONPATH=.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate jev
for pair in "labels_oracle.jsonl:student_oracle.pt" "labels_wm.jsonl:student_wm.pt" "labels_score.jsonl:student_score.pt"; do
  labels="${pair%%:*}"; out="${pair##*:}"
  if [ -f "data/$labels" ]; then
    python scripts/exp2_train.py --labels "data/$labels" --out "results/$out" --device cpu --epochs 30
  fi
done
STUDENTS=""
for s in student_wm student_oracle student_score; do
  [ -f "results/$s.pt" ] && STUDENTS="$STUDENTS,results/$s.pt"
done
STUDENTS=${STUDENTS#,}
python scripts/exp2_eval.py --students "$STUDENTS" \
  --teacher-json results/exp2_teachers_Tacit-4B.json --out results/exp2_eval.json
python scripts/analyze.py
