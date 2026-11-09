#!/bin/bash
# Append a compact status line every 5 minutes; refresh curves every hour.
RUN=${1:-/mnt/storage1/syk/dreamer/runs/mcppo/diamond_s0}
cd /mnt/storage1/syk/.superagent-work/ppo-minecraft-f942b789b4ce/ppo-minecraft
export PYTHONPATH=.
PY=/mnt/storage1/syk/dreamer/envs/mcppo/bin/python
for i in $(seq 1 200); do
  {
    date '+%F %T'
    $PY scripts/dev/summarize.py "$RUN" 2>/dev/null | head -6
    squeue -j $(squeue -u syk --name=mcppo-diamond -h -o %i | tail -1) -o 'job=%i state=%T time=%M' -h 2>/dev/null
    echo '---'
  } >> "$RUN/status.log"
  if [ $((i % 12)) -eq 0 ]; then
    $PY -m mcppo.plot_curves --run "$RUN" --out "$RUN/curves" >/dev/null 2>&1 || true
  fi
  sleep 300
done
