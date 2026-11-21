#!/bin/bash
# 拉取官方开源引擎并编译（本仓库不提交 third_party/）
set -e
cd "$(dirname "$0")"
mkdir -p third_party
if [ ! -d third_party/meow-ai-arena ]; then
  git clone --depth 1 --filter=blob:none --sparse \
    https://github.com/chen-006/meow-ai-arena third_party/meow-ai-arena
  cd third_party/meow-ai-arena
  git sparse-checkout set bridge
  cd ../..
fi
cd third_party/meow-ai-arena/bridge
python3 build_engine.py
echo "engine: third_party/meow-ai-arena/bridge/workspace/bridge_sim.exe"
