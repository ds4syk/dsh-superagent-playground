#!/bin/bash
# Copy small, reviewable artifacts from a run into the repo results/ directory.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
RUN=${1:-/mnt/storage1/syk/dreamer/runs/mcppo/diamond_s0}
OUT="$ROOT/results"
mkdir -p "$OUT/curves"
cp "$RUN/config.json" "$OUT/config.json" 2>/dev/null || true
cp "$RUN/code_info.json" "$OUT/code_info.json" 2>/dev/null || true
cp "$RUN/metrics.jsonl" "$OUT/metrics.jsonl" 2>/dev/null || true
cp "$RUN/episodes.jsonl" "$OUT/episodes.jsonl" 2>/dev/null || true
cp "$RUN/curves/curves.png" "$OUT/curves/curves.png" 2>/dev/null || true
cp "$RUN/curves/curves.pdf" "$OUT/curves/curves.pdf" 2>/dev/null || true
echo "results copied to $OUT"
ls -la "$OUT"
