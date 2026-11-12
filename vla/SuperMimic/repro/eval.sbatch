#!/bin/bash
#SBATCH --job-name=sm-eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/storage1/syk/VLA/SuperMimic/logs/slurm/sm-eval-%j.log

# Evaluate a local SuperMimic checkpoint (Appendix-F protocol).
# Overridable: CHECKPOINT_DIR, MOTION_GLOB, OUTPUT.
set -euo pipefail
ROOT=/mnt/storage1/syk/VLA/SuperMimic
cd "$ROOT/supermimic"

export PATH=/home/syk/.local/bin:$PATH
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
# mujoco 3.16 + mujoco_core 4.0.1 ship mixed plugin libs in one package dir.
export LD_LIBRARY_PATH="$ROOT/supermimic/.venv/lib/python3.13/site-packages/mujoco${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

CHECKPOINT_DIR=${CHECKPOINT_DIR:?set CHECKPOINT_DIR}
MOTION_GLOB=${MOTION_GLOB:-"$ROOT/motions/lafan1_npz/*.npz"}
OUTPUT=${OUTPUT:-"$ROOT/results/eval_supermimic.json"}
ACTOR_OBS=${ACTOR_OBS:-proprio}
HISTORY_LENGTH=${HISTORY_LENGTH:-10}

echo "[sbatch] job=$SLURM_JOB_ID gpu=$CUDA_VISIBLE_DEVICES ckpt=$CHECKPOINT_DIR"
uv run --frozen python "$ROOT/repro/eval_local.py" \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --motion-glob "$MOTION_GLOB" \
  --output "$OUTPUT" \
  --actor-obs "$ACTOR_OBS" \
  --history-length "$HISTORY_LENGTH"
