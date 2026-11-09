#!/bin/bash
# Stage the patched MineRL Minecraft tree on NVMe once per node, so the 8 env
# workers do not all read the busy /mnt/storage1 HDD on startup.
set -euo pipefail
SRC=/mnt/storage1/syk/dreamer/envs/mcppo/lib/python3.11/site-packages/minerl/Malmo/Minecraft
DST=/tmp/mcppo/Minecraft
if [ -f "$DST/launchClient.sh" ]; then
  echo "already staged: $DST"
  exit 0
fi
mkdir -p /tmp/mcppo
rsync -a "$SRC/" "$DST/"
echo "staged $DST"
