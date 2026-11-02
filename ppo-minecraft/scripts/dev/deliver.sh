#!/bin/bash
# Final delivery: summary -> fill/publish KB -> collect results.
set -uo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH=.
PY=/mnt/storage1/syk/dreamer/envs/mcppo/bin/python
RUN=${RUN:-/mnt/storage1/syk/dreamer/runs/mcppo/diamond_s0}
echo "== summary =="
$PY scripts/dev/analyze_run.py "$RUN"
echo "== publish KB =="
bash scripts/dev/publish_final.sh
echo "== collect results =="
bash scripts/collect_results.sh "$RUN"
echo "== leftover placeholders =="
grep -rn "{{" kb/*.md || echo none
