# 撰写约定：PR、提交与文档用中文

> 用户 2026-11-09 设定：**之后的 PR 与交付文档一律用中文撰写**，不回溯改写历史 PR。
> 本文件是可版本化的详细约定；全局 `~/.dsh/AGENTS.md` 只保留简短规则。

## 1. 适用范围

| 对象 | 约定 |
| --- | --- |
| PR 标题 | 中文摘要；沿用仓库习惯时可保留 `<type>:` 前缀，例如 `docs: 约定 PR 与文档用中文撰写` |
| PR 正文 | 中文；段落标题也用中文（`## 摘要`、`## 证据`、`## 合并风险`） |
| Issue / GitHub 评论 / review 回复 | 中文 |
| commit message | 保留 conventional 前缀 `<type>(<scope>):`，摘要与正文用中文 |
| 交付文档 | 知识库、报告、README、AGENTS.md、本文件等正文用中文 |
| Feishu / IM | 跟随来源会话语言（现有规则）；中文会话用中文 |

## 2. 保持原样的内容

- 代码、标识符、API、命令、配置键、文件路径。
- 日志、堆栈、报错、测试输出原文。
- 论文、项目、产品、模型的原名与直接引用原文；首次出现可加中文括注。
- 每个 GitHub 消息首行的固定声明：`This is by an automated agent, not a human.`
- conventional commit 的 type/scope，以及 GitHub 关键字（`Fixes #123`、`Co-authored-by:`）。

## 3. 写法要求

- 不逐字机翻。术语用团队常用中文译名，首次出现给出英文原文。
- PR 结构仍按 `pr` skill：摘要（最小视图）、证据（前后对比）、合并风险（单向/双向门与影响面）；只把标题与说明改成中文。
- 历史英文 PR 不改写；引用时保留原标题，并附一句中文说明。
- 中文 PR 也要保留机器人声明首行；不要把首行翻译成中文。

## 4. 为什么以前是英文

这是缺省行为，不是有意的英文规范：

1. **部署提示词与 skills 用英文写成**：`dsh-superagent` 的 `docs/004-prompts.md` 明确 “Prompt text is written in English”；内置 worker / orchestrator 提示词、GitHub 与 Feishu skills 也全是英文。
2. **语言规则只覆盖会话回复**：`Reply in the language of the source conversation` 管的是 Feishu 回复，没有覆盖 PR 标题/正文、commit message 和仓库文档。
3. **通用 PR 模板是英文**：用户级 `pr` skill 提供 `Summary / Evidence / Merge Danger` 英文骨架，agent 直接照抄。
4. **仓库没有语言约定，先例是英文**：`main` 只有英文 README；截至 2026-11-09，此前 59 条 commit 摘要均不含中文，早期 PR（#3、#5、#6）正文全英文，后续 agent 跟随先例。

因此，过去的 PR 语言由「模型默认 + 英文上下文 + 英文模板」决定，而不是规则要求。

## 5. 检查清单

- [ ] PR 标题与正文是中文，首行英文声明保留。
- [ ] 段落标题是中文。
- [ ] commit 摘要与正文是中文，`type(scope):` 前缀保留。
- [ ] 文档正文是中文；代码、命令、日志、外部引用保持原样。
