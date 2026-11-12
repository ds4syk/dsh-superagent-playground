#!/bin/bash
#SBATCH --job-name=ppo-sim
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/mnt/storage1/syk/VLA/SuperMimic/logs/slurm/ppo-sim-%j.log

# PPO-Sim baseline: mjlab's rsl_rl PPO on the same tracking task, motions,
# observation, reward, termination and domain randomization as SuperMimic.
# Overridable: ITERS, NUM_ENVS, RUN_NAME, MOTION_GLOB.
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
ITERS=${ITERS:-2000}
RUN_NAME=${RUN_NAME:-ppo-lafan1-4096}
SAVE_INTERVAL=${SAVE_INTERVAL:-250}

echo "[sbatch] job=$SLURM_JOB_ID gpu=$CUDA_VISIBLE_DEVICES envs=$NUM_ENVS iters=$ITERS"
echo "[sbatch] motion=$MOTION_GLOB"

uv run --frozen train Mjlab-Tracking-Flat-Unitree-G1 \
  --env.commands.motion.motion-file "$MOTION_GLOB" \
  --env.scene.num_envs "$NUM_ENVS" \
  --agent.max_iterations "$ITERS" \
  --agent.save_interval "$SAVE_INTERVAL" \
  --agent.run_name "$RUN_NAME"
