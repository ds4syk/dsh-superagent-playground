#!/bin/bash
#SBATCH --job-name=mcppo-diamond
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=10
#SBATCH --mem=26G
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/storage1/syk/.superagent-work/ppo-minecraft-f942b789b4ce/ppo-minecraft/logs/train-%j.out
#
# PPO on real Minecraft (DreamerV3 diamond task), hand-written dense reward.
# Re-run with `--resume` (default) to continue from ckpt/latest.pt.
set -uo pipefail
cd /mnt/storage1/syk/.superagent-work/ppo-minecraft-f942b789b4ce/ppo-minecraft
source scripts/mc_env.sh
export OMP_NUM_THREADS=3
RUN=${RUN:-/mnt/storage1/syk/dreamer/runs/mcppo/diamond_s0}
mkdir -p "$RUN" logs
rm -rf "$TMPDIR"/* 2>/dev/null || true
mkdir -p "$TMPDIR"
DISP=:90
rm -f /tmp/.X90-lock /tmp/.X11-unix/X90
$XP/root/usr/bin/Xvfb $DISP -screen 0 1024x768x24 +extension GLX +render >/tmp/xvfb$DISP.log 2>&1 &
XVFB=$!
sleep 2
export DISPLAY=$DISP
python -u -m mcppo.train --config configs/diamond_ppo.json --out-dir "$RUN" --resume "$@"
RC=$?
kill $XVFB 2>/dev/null
exit $RC
