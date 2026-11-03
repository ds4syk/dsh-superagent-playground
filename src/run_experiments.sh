#!/bin/bash
# Run the full V-JEPA2 -> SDPR experiment on CPU (SDPR uses SCS on CPU).
# Submit after the GPU feature-extraction job has produced vjepa2_vitl.npz:
#
#   sbatch --job-name=sdpr-jepa-train --cpus-per-task=8 --mem=32G --time=03:00:00 \
#     tools/slurm/run_cpu.sbatch /path/to/src/run_experiments.sh
set -uo pipefail

ROOT=/mnt/storage1/syk/.superagent-work/sdpr-jepa-bridge-c46ae10a1a2f
PY=/home/syk/miniconda3/envs/jepa/bin/python
F=$ROOT/artifacts/features/vjepa2_vitl.npz

# Wait for the GPU feature job (up to 1 hour) if it is still running.
for _ in $(seq 1 360); do
  [ -f "$F" ] && break
  sleep 10
done
if [ ! -f "$F" ]; then
  echo "missing features after waiting: $F" >&2
  exit 1
fi

echo "===== twostage $(date '+%F %T') ====="
$PY -u "$ROOT/src/train_eval.py" --features "$F" --method twostage \
  --epochs 40 --batch-size 64 --tag v1 || exit 1

echo "===== e2e $(date '+%F %T') ====="
$PY -u "$ROOT/src/train_eval.py" --features "$F" --method e2e \
  --init "$ROOT/artifacts/results/edgehead_twostage_v1.pt" \
  --epochs 10 --batch-size 8 --lr 1e-5 --tag v1 || exit 1

echo "===== direct $(date '+%F %T') ====="
$PY -u "$ROOT/src/train_eval.py" --features "$F" --method direct \
  --epochs 40 --batch-size 64 --tag v1 || exit 1

echo "===== oracle $(date '+%F %T') ====="
$PY -u "$ROOT/src/train_eval.py" --features "$F" --method oracle --tag v1 || exit 1

echo "===== done $(date '+%F %T') ====="
