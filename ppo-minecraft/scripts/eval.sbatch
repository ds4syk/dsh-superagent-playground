#!/bin/bash
#SBATCH --job-name=mcppo-eval
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/storage1/syk/.superagent-work/ppo-minecraft-f942b789b4ce/ppo-minecraft/logs/eval-%j.out
set -uo pipefail
cd /mnt/storage1/syk/.superagent-work/ppo-minecraft-f942b789b4ce/ppo-minecraft
source scripts/mc_env.sh
RUN=${RUN:-/mnt/storage1/syk/dreamer/runs/mcppo/diamond_s0}
DISP=:91
rm -f /tmp/.X91-lock /tmp/.X11-unix/X91
$XP/root/usr/bin/Xvfb $DISP -screen 0 1024x768x24 +extension GLX +render >/tmp/xvfb$DISP.log 2>&1 &
XVFB=$!
sleep 2
export DISPLAY=$DISP
python -u -m mcppo.evaluate --run "$RUN" --episodes 1 --max-steps 4000 --record "$RUN/eval_4000.mp4"
RC=$?
kill $XVFB 2>/dev/null
exit $RC
