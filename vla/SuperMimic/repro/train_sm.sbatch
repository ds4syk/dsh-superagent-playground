#!/bin/bash
#SBATCH --job-name=sm-train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/mnt/storage1/syk/VLA/SuperMimic/logs/slurm/sm-train-%j.log

# SuperMimic training on one GPU (official implementation, LAFAN1 motions).
# Overridable: ITERS, NUM_ENVS, RUN_NAME, MOTION_GLOB, TASK.
set -euo pipefail
ROOT=/mnt/storage1/syk/VLA/SuperMimic
cd "$ROOT/supermimic"

export PATH=/home/syk/.local/bin:$PATH
export WANDB_MODE=offline
export WANDB_DIR="$ROOT/logs/wandb"
export WANDB_SILENT=true
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
# Optional: TORCHDYNAMO_DISABLE=1 runs the compiled WM/actor eagerly (skips long first-compile).
if [ -n "${TORCHDYNAMO_DISABLE:-}" ]; then export TORCHDYNAMO_DISABLE; fi
# mujoco 3.16 + mujoco_core 4.0.1 ship mixed plugin libs in one package dir.
export LD_LIBRARY_PATH="$ROOT/supermimic/.venv/lib/python3.13/site-packages/mujoco${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

MOTION_GLOB=${MOTION_GLOB:-"$ROOT/motions/lafan1_npz/*.npz"}
NUM_ENVS=${NUM_ENVS:-4096}
ITERS=${ITERS:-60000}
RUN_NAME=${RUN_NAME:-lafan1-4096}
TASK=${TASK:-SuperMimic-GenCoord-G1}
SAVE_INTERVAL=${SAVE_INTERVAL:-2000}
HISTORY_LENGTH=${HISTORY_LENGTH:-10}
ACTOR_OBS=${ACTOR_OBS:-proprio}

echo "[sbatch] job=$SLURM_JOB_ID gpu=$CUDA_VISIBLE_DEVICES task=$TASK envs=$NUM_ENVS iters=$ITERS"
echo "[sbatch] motion=$MOTION_GLOB"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

uv run --frozen src/supermimic/scripts/train.py "$TASK" \
  --runner.run_name "$RUN_NAME" \
  --num_envs "$NUM_ENVS" \
  --runner.max_iterations "$ITERS" \
  --runner.save_interval "$SAVE_INTERVAL" \
  --runner.log_interval 10 \
  --runner.history_length "$HISTORY_LENGTH" \
  --runner.actor_obs "$ACTOR_OBS" \
  --motion_file "$MOTION_GLOB"
