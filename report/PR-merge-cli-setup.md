This is by an automated agent, not a human.

## 摘要

把 superagent 的 **merge CLI 与合并审批规则**真正启用的具体方案：以 @YingkeSu 身份登录的 `gh-merge`、面板 Delivery requirements 的文字规则、GitHub ruleset 三层一起配。**本 PR 只交付方案、脚本和文档，不改变线上配置**；唯一需要人做的是第二账号的一次性登录（token 不进聊天）。

![图 1：merge CLI 与合并审批规则](https://github.com/ds4syk/dsh-superagent-playground/blob/agent/merge-cli-setup-b4e5759d4c05/docs/visuals/2026-11-11-merge-cli-flow.png?raw=true)

## 现状（实测）

- Prompt 还是 revision 0（defaults），Delivery requirements 未命名 merge CLI；
- 本机没有 `~/.local/bin/gh-merge`，也没有 `~/.config/gh-merge`；
- 仓库 `allow_auto_merge=false`，main 没有 branch protection / ruleset；
- 10 个未合并 PR（#16–#25），按默认 200 行阈值只有 #17 能自动合并。

所以现在是“所有 PR 都由人合并”。详细证据见 `docs/merge-cli-and-approval.md` §1。

## 方案

1. **本机 CLI**：`~/.local/bin/gh-merge`，`GH_CONFIG_DIR=~/.config/gh-merge`，以 @YingkeSu 登录；fine-grained PAT 只给这个仓库的 Contents / Pull requests 读写。
2. **面板文字规则**：`设置 → 提示词 → 团队约定 → Delivery requirements`，把 `<merge CLI>` 换成绝对路径，并追加：1 个 @YingkeSu 对当前 head 的批准；开 PR 前加 reviewer 的 `Co-authored-by`；文档类 PR 豁免 200 行。
3. **GitHub 强制规则**：`allow_auto_merge=true`；ruleset 要求 PR + 1 approval + dismiss stale approvals + 禁止强推/删 main + 无 bypass。
4. **顺序**：先登录 `gh-merge`，再保存 prompt 与 ruleset；否则批准后 worker 会找不到 CLI。

## 交付物

- `docs/merge-cli-and-approval.md` — 主方案（中文）：现状、设计、执行顺序、风险、验收清单。
- `docs/visuals/2026-11-11-merge-cli-flow.png` — 一页流程图（上方内嵌；源码 `scripts/make_merge_cli_visual.py`）。
- `scripts/merge-cli/setup-gh-merge.sh` — 创建包装器 + 检查登录身份（不替你登录，`--check` 只检查）。
- `scripts/merge-cli/apply-prompt.py` — 默认 dry-run，调 `POST /prompts/preview` 校验；`--apply` 才保存新 revision。
- `scripts/merge-cli/ruleset.json` — GitHub ruleset 的完整 JSON。
- `deliverables/merge-cli-setup-2026-11.docx` — DOCX 阅读副本（8 页，内嵌上图）。
- `scripts/build_merge_cli_docx.py` — 由 Markdown 构建 DOCX 的脚本。

## 验证

- `apply-prompt.py` 的 dry-run 已对运行中的面板执行：`GET /prompts` → 替换 + 追加 → `POST /prompts/preview`，preview 的 worker 文本包含 “Merge rule for this deployment”；**没有 PUT，线上仍是 revision 0**。
- `setup-gh-merge.sh --check` 按预期报 “missing / not signed in”，说明身份检查路径生效。
- `ruleset.json` 通过 JSON 解析；**没有对真实仓库创建 ruleset**，方案阶段不改线上设置。
- DOCX 已用 LibreOffice 转 PDF、逐页目检（8 页，表格/代码块/复选框/图片均正常）。

## 合并风险

- 文档 + 脚本 PR，不改线上配置、不改产品 UI；合并本 PR 不会启用 auto-merge。
- 一旦按方案执行，会改变所有未来 PR 的合并方式；回滚步骤见文档 §3。
- 方案里的“现状”是 2026-11-11 23:55 的实测；如果之后有人改了配置，以执行时的实测为准。
- 方案不是硬安全边界：worker 以同一 OS 用户运行，可读取 `~/.config/gh-merge` 的 token；仓库 owner 仍是 agent。要硬边界需把仓库转给人类/组织。
