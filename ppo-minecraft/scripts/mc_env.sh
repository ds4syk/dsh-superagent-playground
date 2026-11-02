#!/bin/bash
# Common environment for MC-PPO Slurm jobs (Xvfb + MineRL patched wheel + Java 8).
set -uo pipefail
MC=/mnt/storage1/syk/dreamer/envs/mcppo
XP=/mnt/storage1/syk/dreamer/xvfb
export JAVA_HOME=$MC
export PATH=$MC/bin:$PATH
export LIBGL_ALWAYS_SOFTWARE=1
# Minecraft instances copy themselves into TMPDIR on first reset; keep those
# ephemeral copies on NVMe (see scripts/prepare_mc.sh), artifacts stay on the
# data disk.
export MCPPO_MINECRAFT_DIR=${MCPPO_MINECRAFT_DIR:-/tmp/mcppo/Minecraft}
export TMPDIR=${MCPPO_TMPDIR:-/tmp/mcppo/tmp}
export LD_LIBRARY_PATH=$XP/root/usr/lib/x86_64-linux-gnu
export PYTHONUNBUFFERED=1
mkdir -p /tmp/.X11-unix "$TMPDIR"
