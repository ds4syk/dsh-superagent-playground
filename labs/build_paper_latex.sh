#!/usr/bin/env bash
# Build all six paper LaTeX trees and print a status table + overfull summary.
# Usage: labs/build_paper_latex.sh
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)
TECTONIC=/mnt/storage1/syk/tools/tectonic/tectonic
export HTTPS_PROXY=${HTTPS_PROXY:-http://127.0.0.1:7899}
export HTTP_PROXY=${HTTP_PROXY:-http://127.0.0.1:7899}

printf "%-52s %-5s %-6s %-9s %-11s %s\n" "tree" "exit" "pages" "overfull" "undef-refs" "pdf-size"
fail=0
for d in "$ROOT"/labs/*/paper/latex/en "$ROOT"/labs/*/paper/latex/zh; do
  [ -f "$d/main.tex" ] || continue
  rel=${d#"$ROOT"/}
  ( cd "$d" && rm -f main.pdf && "$TECTONIC" -X compile main.tex > build.log 2>&1; echo $? > .exit )
  code=$(cat "$d/.exit" 2>/dev/null || echo 1)
  pages=$(pdfinfo "$d/main.pdf" 2>/dev/null | awk '/^Pages:/{print $2}')
  over=$(grep -c "Overfull \\\\hbox" "$d/build.log" 2>/dev/null || true)
  undef=$(grep -ciE "undefined (references|citation)|Citation .* undefined|Reference .* undefined" "$d/build.log" 2>/dev/null || true)
  size=$(du -h "$d/main.pdf" 2>/dev/null | cut -f1)
  [ "$code" = "0" ] || fail=$((fail+1))
  printf "%-52s %-5s %-6s %-9s %-11s %s\n" "$rel" "$code" "${pages:-NA}" "$over" "$undef" "${size:-NA}"
done
echo
echo "trees with non-zero exit: $fail"
