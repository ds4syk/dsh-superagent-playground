#!/usr/bin/env bash
# 独立复现上游 FourColor 形式化，并运行本仓库的审计文件。
#
# 用法：
#   ./run.sh                 # 默认 JOBS=16 MEMORY=60
#   JOBS=24 MEMORY=80 ./run.sh
#   UPSTREAM_DIR=/path/to/4ct ./run.sh   # 复用已有 checkout
#
# 依赖：elan、git、curl、gcc、python3、zstd，约 25 GB 磁盘。
# 网络受限时，可先给 shell 设置 http_proxy/https_proxy。
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)
LOCK="$ROOT/upstream.lock"
EV="$ROOT/evidence"
mkdir -p "$EV"

repo=$(sed -n 's/^repo = "\(.*\)"/\1/p' "$LOCK" | head -1)
commit=$(sed -n 's/^commit = "\(.*\)"/\1/p' "$LOCK" | head -1)
UP=${UPSTREAM_DIR:-"$HERE/.upstream/4ct"}
JOBS=${JOBS:-16}
MEMORY=${MEMORY:-60}

export PATH="$HOME/.elan/bin:$PATH"

if ! python3 -c 'import numpy' >/dev/null 2>&1; then
  echo "error: python3 需要 numpy（生成 mask 证书的 scripts/bulkspace.py 使用）" >&2
  echo "提示：python3 -m pip install numpy，或使用带 numpy 的虚拟环境并把它的 bin 放到 PATH 前面" >&2
  exit 1
fi

{
  echo "date: $(date -Is)"
  echo "host: $(hostname)"
  echo "kernel: $(uname -srm)"
  echo "cpu: $(nproc) threads"
  echo "memory: $(awk '/MemTotal/ {printf "%.1f GiB", $2/1024/1024}' /proc/meminfo)"
  echo "upstream: $repo @ $commit"
  echo "lean: $(lean --version 2>&1 || true)"
  echo "lake: $(lake --version 2>&1 || true)"
  echo "git: $(git --version)"
  echo "gcc: $(gcc --version | head -1)"
  echo "python3: $(python3 --version)"
  echo "numpy: $(python3 -c 'import numpy; print(numpy.__version__)')"
} | tee "$EV/environment.txt"

if [ ! -d "$UP/.git" ]; then
  mkdir -p "$(dirname "$UP")"
  git clone "$repo" "$UP"
fi
git -C "$UP" fetch --all --tags
git -C "$UP" checkout --detach "$commit"

cd "$UP"
lake exe cache get 2>&1 | tee "$EV/cache-get.log"
JOBS="$JOBS" MEMORY="$MEMORY" bash build.sh 2>&1 | tee "$EV/build.log"

lake env lean "$HERE/audit/Audit.lean" 2>&1 | tee "$EV/audit.txt"

{
  echo "=== #print axioms ==="
  lake env lean "$HERE/audit/Audit.lean" 2>&1 | grep -E "depends on|AUDIT|axioms" || true
  echo
  echo "=== source scan ==="
  echo -n "sorry (code): "
  grep -rn --include='*.lean' -E '\bsorry\b' FourColor/ FourColor.lean | grep -v '`sorry`' | wc -l
  echo -n "native_decide: "
  grep -rn --include='*.lean' -E '\bnative_decide\b' FourColor/ FourColor.lean | wc -l
  echo -n "axiom declarations: "
  grep -rn --include='*.lean' -E '^[[:space:]]*(private |protected |noncomputable )*axiom ' FourColor/ FourColor.lean | wc -l
  echo -n "Lean modules: "
  find FourColor -name '*.lean' | wc -l
  echo -n "Lean source lines: "
  find FourColor -name '*.lean' -print0 | xargs -0 cat | wc -l
} | tee "$EV/summary.txt"

echo "DONE: evidence in $EV"
