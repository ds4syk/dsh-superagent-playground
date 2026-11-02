#!/bin/bash
# Self-check: unit tests, reward walkthrough, run artifacts, curves.
set -uo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH=.
PY=/mnt/storage1/syk/dreamer/envs/mcppo/bin/python
echo "== unit tests =="
$PY -m pytest tests -q || exit 1
echo "== reward walkthrough =="
$PY scripts/dev/reward_walkthrough.py | tail -8
RUN=${1:-/mnt/storage1/syk/dreamer/runs/mcppo/diamond_s0}
echo "== run summary =="
$PY scripts/dev/summarize.py "$RUN"
echo "== curves =="
$PY -m mcppo.plot_curves --run "$RUN" --out "$RUN/curves"
ls -la "$RUN/curves"
