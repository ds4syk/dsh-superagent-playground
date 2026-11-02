#!/bin/bash
#SBATCH --job-name=mcppo-env
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/mnt/storage1/syk/.superagent-work/ppo-minecraft-f942b789b4ce/ppo-minecraft/logs/env-own-%j.out
set -uo pipefail
cd /mnt/storage1/syk/.superagent-work/ppo-minecraft-f942b789b4ce/ppo-minecraft
source scripts/mc_env.sh
DISP=:86
rm -f /tmp/.X86-lock /tmp/.X11-unix/X86
$XP/root/usr/bin/Xvfb $DISP -screen 0 1024x768x24 +extension GLX +render >/tmp/xvfb$DISP.log 2>&1 &
XVFB=$!
sleep 2
export DISPLAY=$DISP
export PYTHONPATH=$PWD
python -u scripts/dev/env_own.py
RC=$?
kill $XVFB 2>/dev/null
exit $RC
