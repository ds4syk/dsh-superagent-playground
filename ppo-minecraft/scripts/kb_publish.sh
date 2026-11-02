#!/bin/bash
# Publish the Chinese knowledge base to Feishu and export DOCX (run from repo root).
# Usage: LARKSUITE_CLI_CONFIG_DIR=~/.lark-cli-dsh-superagent bash scripts/kb_publish.sh
set -uo pipefail
export LARKSUITE_CLI_CONFIG_DIR=${LARKSUITE_CLI_CONFIG_DIR:-$HOME/.lark-cli-dsh-superagent}
OWNER_OPEN_ID=${OWNER_OPEN_ID:-ou_d8336eaf6b330fd6a3f0cd3163b5b337}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
mkdir -p deliverables/kb

echo "== create folder =="
FOLDER_JSON=$(lark-cli drive +create-folder --as bot --name "PPO × Minecraft（2026-11）" --format json)
echo "$FOLDER_JSON"
FOLDER=$(echo "$FOLDER_JSON" | python3 -c 'import json,sys; d=json.load(sys.stdin)["data"]; print(d.get("token") or d.get("folder_token") or "")')
if [ -z "$FOLDER" ]; then echo "folder token missing"; exit 1; fi

echo "== create pages =="
declare -A TOKENS
for f in kb/*.md; do
  title=$(basename "$f" .md)
  out=$(lark-cli docs +create --as bot --parent-token "$FOLDER" \
        --doc-format markdown --title "$title" --content "@./$f" --format json)
  echo "$out"
  url=$(echo "$out" | python3 -c 'import json,sys; d=json.load(sys.stdin)["data"]; print(d.get("url") or d.get("document",{}).get("url") or "")')
  echo "$title -> $url"
  TOKENS["$title"]=$url
done

echo "== insert curves into 04 =="
RUN=${RUN:-/mnt/storage1/syk/dreamer/runs/mcppo/diamond_s0}
if [ -f "$RUN/curves/curves.png" ]; then
  cp "$RUN/curves/curves.png" kb/assets/curves.png
  lark-cli docs +media-insert --as bot --doc "${TOKENS[04 训练系统与实验]}" \
    --file ./kb/assets/curves.png --type image --caption '图 1：本次 PPO 训练曲线（返回 / score / loss / 吞吐 / 奖励分量）' --width 900
fi

echo "== grant owner =="
lark-cli drive +member-add --as bot --token "$FOLDER" --type folder \
  --member-type openid --member-id "$OWNER_OPEN_ID" --perm full_access --yes || true

echo "== export docx =="
for title in "${!TOKENS[@]}"; do
  lark-cli drive +export --as bot --url "${TOKENS[$title]}" --file-extension docx \
    --output-dir ./deliverables/kb --overwrite || true
done

echo "folder=$FOLDER"
