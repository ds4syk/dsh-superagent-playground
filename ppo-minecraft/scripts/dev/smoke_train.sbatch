#!/bin/bash
#SBATCH --job-name=mcppo-smoke
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/storage1/syk/.superagent-work/ppo-minecraft-f942b789b4ce/ppo-minecraft/logs/smoke-train-%j.out
set -uo pipefail
cd /mnt/storage1/syk/.superagent-work/ppo-minecraft-f942b789b4ce/ppo-minecraft
source scripts/mc_env.sh
DISP=:87
rm -f /tmp/.X87-lock /tmp/.X11-unix/X87
$XP/root/usr/bin/Xvfb $DISP -screen 0 1024x768x24 +extension GLX +render >/tmp/xvfb$DISP.log 2>&1 &
XVFB=$!
sleep 2
export DISPLAY=$DISP
python -u -m mcppo.train --config configs/smoke.json --out-dir runs/smoke --steps 128
RC=$?
kill $XVFB 2>/dev/null
exit $RC
