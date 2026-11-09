#!/bin/bash
#SBATCH --job-name=mcppo-smoke
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/storage1/syk/.superagent-work/ppo-minecraft-f942b789b4ce/ppo-minecraft/logs/smoke-%j.out
set -uo pipefail
MC=/mnt/storage1/syk/dreamer/envs/mcppo
XP=/mnt/storage1/syk/dreamer/xvfb
export JAVA_HOME=$MC
export PATH=$MC/bin:$PATH
export LIBGL_ALWAYS_SOFTWARE=1
export TMPDIR=/mnt/storage1/syk/dreamer/tmp
export LD_LIBRARY_PATH=$XP/root/usr/lib/x86_64-linux-gnu
mkdir -p /tmp/.X11-unix "$TMPDIR"
DISP=:88
rm -f /tmp/.X88-lock /tmp/.X11-unix/X88
$XP/root/usr/bin/Xvfb $DISP -screen 0 1024x768x24 +extension GLX +render >/tmp/xvfb$DISP.log 2>&1 &
XVFB=$!
sleep 2
export DISPLAY=$DISP
export PYTHONPATH=/home/syk/labs/Dreamer/repos/dreamerv3
cd /mnt/storage1/syk/.superagent-work/ppo-minecraft-f942b789b4ce/ppo-minecraft
mkdir -p logs
python -u scripts/dev/smoke_env.py
RC=$?
kill $XVFB 2>/dev/null
exit $RC
