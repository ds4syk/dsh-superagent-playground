#!/bin/bash
# One GPU allocation: feature extraction (GPU) followed by CPU-only SDPR training.
set -euo pipefail
ROOT=/mnt/storage1/syk/.superagent-work/sdpr-jepa-bridge-c46ae10a1a2f
PY=/home/syk/miniconda3/envs/jepa/bin/python
F=$ROOT/artifacts/features/vjepa2_vitl.npz

echo "===== features $(date '+%F %T') ====="
$PY -u "$ROOT/src/extract_features.py" --device cuda --out "$F"

echo "===== train/eval $(date '+%F %T') ====="
bash "$ROOT/src/run_experiments.sh"

echo "===== bridge check $(date '+%F %T') ====="
$PY -u "$ROOT/src/bridge_check.py" \
  --features "$F" \
  --head "$ROOT/artifacts/results/edgehead_e2e_v1.pt" \
  --samples "$ROOT/artifacts/data/samples.npz" \
  --device cuda || echo "bridge check failed (non-fatal)"

echo "===== analyze $(date '+%F %T') ====="
$PY -u "$ROOT/src/analyze.py" --tag v1 || echo "analyze failed (non-fatal)"
mkdir -p "$ROOT/report/figures" "$ROOT/report/results"
cp -f "$ROOT/artifacts/results/"fig_*.png "$ROOT/report/figures/" 2>/dev/null || true
cp -f "$ROOT/artifacts/results/"metrics_*_v1.json "$ROOT/report/results/" 2>/dev/null || true
cp -f "$ROOT/artifacts/results/"pred_*_v1.npz "$ROOT/report/results/" 2>/dev/null || true
cp -f "$ROOT/artifacts/results/bridge_check.json" "$ROOT/report/results/" 2>/dev/null || true
cp -f "$ROOT/artifacts/results/noise_experiment.json" "$ROOT/report/results/" 2>/dev/null || true
cp -f "$ROOT/artifacts/results/summary.json" "$ROOT/artifacts/results/tables.md" "$ROOT/report/results/" 2>/dev/null || true

echo "===== pipeline done $(date '+%F %T') ====="
