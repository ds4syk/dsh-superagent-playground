#!/bin/bash
#SBATCH --job-name=sm-eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/storage1/syk/VLA/SuperMimic/logs/slurm/eval-%j.log

# Evaluate both trained policies (SuperMimic and PPO-Sim) on the 40 LAFAN1 clips
# with the paper's Appendix-F protocol, one after the other on one GPU.
set -euo pipefail
ROOT=/mnt/storage1/syk/VLA/SuperMimic
cd "$ROOT/supermimic"

export PATH=/home/syk/.local/bin:$PATH
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH="$ROOT/supermimic/.venv/lib/python3.13/site-packages/mujoco${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

SM_CKPT=${SM_CKPT:-$(ls -dt "$ROOT"/supermimic/logs/SuperMimic-GenCoord-G1/*/ | head -1)}
PPO_CKPT=${PPO_CKPT:-$(ls -t "$ROOT"/supermimic/logs/rsl_rl/g1_tracking/*/model_*.pt 2>/dev/null | head -1)}
MOTION_GLOB=${MOTION_GLOB:-"$ROOT/motions/lafan1_npz/*.npz"}

echo "[eval] SM checkpoint:  $SM_CKPT"
echo "[eval] PPO checkpoint: $PPO_CKPT"

uv run --frozen python "$ROOT/repro/eval_local.py" \
  --checkpoint-dir "$SM_CKPT" \
  --motion-glob "$MOTION_GLOB" \
  --output "$ROOT/results/eval_supermimic.json" \
  --actor-obs proprio --history-length 10

if [ -n "$PPO_CKPT" ]; then
  uv run --frozen python "$ROOT/repro/eval_ppo.py" \
    --checkpoint "$PPO_CKPT" \
    --motion-glob "$MOTION_GLOB" \
    --output "$ROOT/results/eval_ppo.json"
fi
