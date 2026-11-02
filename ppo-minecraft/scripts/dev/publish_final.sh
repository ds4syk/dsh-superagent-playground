#!/bin/bash
# Final Feishu publish: fill KB numbers, create 00/04, insert images, export DOCX.
set -uo pipefail
export LARKSUITE_CLI_CONFIG_DIR=${LARKSUITE_CLI_CONFIG_DIR:-$HOME/.lark-cli-dsh-superagent}
ROOT=$(cd "$(dirname "$0")/../.." && pwd); cd "$ROOT"
RUN=${RUN:-/mnt/storage1/syk/dreamer/runs/mcppo/diamond_s0}
FOLDER=BUFCfJffNl2MR0dBmCCcjzaenFb
OWNER=ou_d8336eaf6b330fd6a3f0cd3163b5b337
export PYTHONPATH=.
PY=/mnt/storage1/syk/dreamer/envs/mcppo/bin/python

$PY scripts/dev/fill_kb.py "$RUN"
$PY -m mcppo.plot_curves --run "$RUN" --out "$RUN/curves"
cp "$RUN/curves/curves.png" kb/assets/curves.png
cp results/v0/curves/curves.png kb/assets/v0_curves.png 2>/dev/null || true

# refresh static pages from the latest local Markdown
for pair in "01 环境与基准：Dreamer 的 MC 设置|https://yide4vukqxv.feishu.cn/docx/RNEMdZJL6oAwRUx9sP3cjexBnYc" \
            "02 PPO 方案设计|https://yide4vukqxv.feishu.cn/docx/XIeWdywp9ohI6MxE4sPcoqBMnrb" \
            "03 手写奖励函数|https://yide4vukqxv.feishu.cn/docx/BhmIdJ7P5omLGsxPNDWcZiALnid" \
            "05 到钻石的可行性与路线|https://yide4vukqxv.feishu.cn/docx/ZKzidTQt9odL8fxfqvhcUWHBncc"; do
  f="${pair%%|*}"; u="${pair##*|}"
  lark-cli docs +update --as bot --doc "$u" --command overwrite --doc-format markdown \
    --content "@./kb/$f.md" --format json >/dev/null
done

mk() { # title, file
  lark-cli docs +create --as bot --parent-token "$FOLDER" --doc-format markdown \
    --title "$1" --content "@./$2" --format json | \
    $PY -c 'import json,sys; print(json.load(sys.stdin)["data"]["document"]["url"])'
}
URL00=$(mk "00 总览：PPO × Minecraft 训练系统（2026-11）" "kb/00 总览：PPO × Minecraft 训练系统（2026-11）.md")
URL04=$(mk "04 训练系统与实验" "kb/04 训练系统与实验.md")
echo "00=$URL00"; echo "04=$URL04"

lark-cli docs +media-insert --as bot --doc "$URL04" --file ./kb/assets/curves.png \
  --type image --caption '图 1：本次 PPO 训练曲线（返回 / score / 里程碑事件 / loss / 吞吐 / 奖励分量）' --width 900
lark-cli docs +media-insert --as bot --doc https://yide4vukqxv.feishu.cn/docx/BhmIdJ7P5omLGsxPNDWcZiALnid --file ./kb/assets/reward_milestones.png \
  --type image --caption '图 1：理想跑通的里程碑奖励（钻石 41.1，其余合计 29.8）' --width 900

lark-cli drive +member-add --as bot --token "$FOLDER" --type folder --member-type openid \
  --member-id "$OWNER" --perm full_access --yes || true

for u in "$URL00" "$URL04" https://yide4vukqxv.feishu.cn/docx/RNEMdZJL6oAwRUx9sP3cjexBnYc \
         https://yide4vukqxv.feishu.cn/docx/XIeWdywp9ohI6MxE4sPcoqBMnrb \
         https://yide4vukqxv.feishu.cn/docx/BhmIdJ7P5omLGsxPNDWcZiALnid \
         https://yide4vukqxv.feishu.cn/docx/ZKzidTQt9odL8fxfqvhcUWHBncc; do
  lark-cli drive +export --as bot --url "$u" --file-extension docx \
    --output-dir ./deliverables/kb --overwrite --format json | \
    $PY -c 'import json,sys; print(json.load(sys.stdin)["data"]["saved_path"])'
done
echo "KB folder: https://yide4vukqxv.feishu.cn/drive/folder/$FOLDER"
