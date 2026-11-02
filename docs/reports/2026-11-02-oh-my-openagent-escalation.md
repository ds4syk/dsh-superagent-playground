# 研读报告：执行 Session 遇难题时，向更强的 Agent 求助

> **调研对象**：`code-yeongyu/oh-my-openagent`（早期名 `oh-my-opencode`，下称 OmO），`dev` 分支，2026-11-02 快照。
> **对照对象**：本机运行的 `dsh-superagent`（DSH 0.2.1-rc.2 上的 bundle）与 `@deepseek-ai/dsh-tool-subagent`。
> **结论**：这个想法可行，而且**不需要新的运行时**——DSH 已经有全部原语；缺的是配置和提示词。可以先零代码上线，再决定是否产品化。

---

## 1. TL;DR

OmO 的关键不在"给难题换一个更聪明的模型"，而在三件事的组合：

1. **只读顾问**（Oracle / architect）：不能写文件、不能继续委派，只输出建议；
2. **显式触发规则**：什么情况该问、什么情况不该问，写进提示词，而不是靠模型临场判断；
3. **模型分档路由**：难题路由到强模型，机械活路由到快模型，且调用方不能绕过配置自选模型。

当前 superagent 已经具备：每 Ticket 一个独立 Session、`subagent`（进程内 spawn）、Agent preset、`agentOptions` 可以把子 Agent 的 provider/model/reasoningEffort 钉死。

**缺的只有三样**：一个只读顾问 preset、一个固定强模型的委派工具、一段 worker 提示词规则。前两样是配置，第三样是提示词覆盖。

建议：先做 **L1（配置级）**，挑一两个真实难题做 A/B，比较"flash 单独"与"flash + 顾问"的解决率、耗时和费用；有效再做 **L2（产品化）**。

---

## 2. OmO 的架构主线

来源：[docs/guide/orchestration.md](https://github.com/code-yeongyu/oh-my-openagent/blob/dev/docs/guide/orchestration.md)、[README.zh-cn.md](https://github.com/code-yeongyu/oh-my-openagent/blob/dev/README.zh-cn.md)。

> 版本说明：该仓库正在从 OpenCode 插件重构为多 harness 的 agent OS（README 顶部有提示）。当前 `docs/guide/orchestration.md` 描述的是 senpi 底座的设计，而 Oracle 的实现仍保留在 OpenCode adapter（`packages/omo-opencode/src/agents/oracle.ts`）。两者共享同一套设计语言；本报告只引用与"遇难题求助"直接相关的部分。

- 只有一个 orchestrator（主 Agent，即你正在对话的模型）。它**不把会话交给别的 Agent**，而是通过 `task` 委派工作单元，读回结果，维护计划、todo 和证据。
- 委派分两类：
  - `task(category=...)`：按任务类别路由到一个全新 worker 会话，**模型由配置决定**（`categories.<name>.models`）。调用方同时传 `model` 会被拒绝——防止模型自己挑贵模型。
  - `task(subagent_type=...)`：直接点名 Agent。内置只读研究/咨询 Agent（`explore`、`librarian`、`plan-consultant`、`plan-reviewer`），以及负责写报告的 reviewer。
- 子任务可后台运行、并行成 wave，用 `task_send` / `task_output` / `task_cancel` 控制。
- 配置里每个 Agent 可覆盖 `model(s)`、`reasoning`、`tools`、`allowed_subagents`、`max_turns`、`temperature`；category 也有一组 `models` 链。
- 模型分档：`quick` / `unspecified-low` 用便宜模型；`deep` / `ultrabrain` 用旗舰模型（部分类别要求注册表里存在特定模型才开放，缺了就明确报错，而不是偷偷降级）；`architect` 是只读的架构咨询通道。

---

## 3. 最值得借鉴的四点

### 3.1 Oracle：按需、只读、昂贵的顾问

源码：[agents/oracle.ts](https://github.com/code-yeongyu/oh-my-openagent/blob/dev/packages/omo-opencode/src/agents/oracle.ts)。

元数据里直接写了**什么时候用**：

- 架构决策：多系统取舍、不熟悉的模式；
- 自查：完成重要实现之后；
- 硬调试：**连续 2 次以上修复失败**。

也写了**什么时候别用**（同样重要）：

- 简单文件操作；第一次尝试；读代码就能回答的问题；变量命名/格式等琐碎决定；能从现有代码模式推断的东西。

实现细节：

- `mode: subagent`，禁止 `write` / `edit` / `apply_patch` / `task`——**不能改代码，也不能继续委派**；
- `temperature: 0.1`，GPT 系列给 `xhigh` / `medium` 的 reasoning effort；
- 提示词把它定义成"战略技术顾问"：Bottom line（2–3 句）+ ≤7 步 action plan + 工作量估计；倾向最简单可行方案；不确定时先澄清或声明解释。

关键点：**顾问的产物是"建议"，不是"改动"**。执行者仍然自己验证、自己落代码。

### 3.2 只读与可写分离

只读 Agent 不能写文件；reviewer 独立验证 worker 的完成声明；orchestrator 只负责合并。这避免了"执行者自己判断自己"的偏差。

### 3.3 模型由配置决定，而不是由调用方决定

category 路由拒绝同时传 `model`；Agent 的模型来自 `omo.json`。好处是成本可控、行为可预测；代价是灵活性降低。对"执行器随手挑一个贵模型"这类问题，这个约束很有效。

### 3.4 上下文与生命周期

子 Agent 是全新会话，主 Agent 必须给出自包含的任务描述；子 Agent 可以继续对话（DSH 里用 `send_message` 追问，结束后还能 resume）、后台运行、取消。**咨询按次进行，不是长期驻留**。

---

## 4. 映射到当前 superagent

已核对本地代码与正在运行的实例：

- 结构：主 Session（研判/路由，preset `dsh-superagent-router`）+ 每个 Ticket 一个执行 Session（preset `dsh-superagent-worker`），独立 worktree、分支和 PR。
- 执行 Session 已经有 `subagent` 工具：provider `spawn`，`agentPreset: standard`，**子 Agent 继承父会话的模型**（当前是 `deepseek-flash`）。
- worker preset 里写了 `modelSelection: true` / `modelDiscovery: true`，但当前安装的 `@deepseek-ai/dsh-tool-subagent@0.2.1-rc.2` 的配置类型里只有 `modelSelectionSettings` / `agentOptions`，没有这两个键；实际 worker 会话里也没有 `list_subagent_models` 工具。→ 结论：**目前没有"按次选择子 Agent 模型"的能力**。
- 主 Session 帮不上忙：worker → 主 Session 只有单向的报告/通知通道，**没有"提问并等待回答"的往返**；也不该让路由者替执行者读代码。所以求助对象应是一个**新的只读顾问节点**，而不是主 Session。
- 遇难题的现状：worker 只能自己硬扛，或 `report blocked` + 向人提问（人的注意力很贵，这条路径应留给"需要人做决定"的事）；没有 Agent 级的求助路径。

**差距总结**：运行时原语齐全，缺一个顾问角色（preset + 工具 + 提示词规则）和一套触发/成本策略。

---

## 5. 落地方案

### L1：配置级（不改 dsh-superagent 代码）

**1）在 bundle 的 `cordis.patch.yml` 增加只读顾问 preset：**

```yaml
- id: dsh-superagent-escalation-preset
  name: '@deepseek-ai/dsh-agent-preset'
  config:
    id: dsh-superagent-escalation
    name: dsh-superagent 顾问
    description: 只读技术顾问；执行器遇到架构/硬调试难题时调用，不供手动选择。
    order: 92
    plugins:
      - id: persona
        name: '@deepseek-ai/dsh-persona'
        config:
          prefix: |
            You are a read-only technical advisor. You cannot change files.
            Answer with: bottom line (2-3 sentences), a numbered action plan (up to 7 steps),
            an effort estimate, and the main risks. Prefer the simplest workable option.
      - id: agent-instructions
        name: '@deepseek-ai/dsh-agent-instructions'
        config: { maxBytes: 65536 }
      - id: tool-fs
        name: '@deepseek-ai/dsh-tool-fs'
        config: { write: false, edit: false }
      - id: tool-fs-search
        name: '@deepseek-ai/dsh-tool-fs-search'
```

**2）在 worker preset 的 delegation group 里加第二个委派工具，把模型钉死：**

```yaml
- id: tool-oracle
  name: '@deepseek-ai/dsh-tool-subagent'
  config:
    provider: spawn
    toolName: ask_oracle
    agentPreset: dsh-superagent-escalation
    agentOptions:
      provider: deepseek-official
      model: deepseek-v4-pro
      reasoningEffort: max
```

**3）在 worker 提示词里加触发/禁止规则（照抄 OmO 的 triggers / avoidWhen）：**

- 先自己试；同一问题连续 2 次修复失败，或涉及模块边界/架构取舍/安全与性能权衡，且读代码无法定论时，调用 `ask_oracle`。
- 提供自包含 brief：目标、已尝试方案与结果、关键文件、复现命令、你的假设。
- 把回答当建议，自己验证后再改；回答与代码/测试冲突时，以代码和测试为准。
- 不要问：读代码能回答的、命名/格式、第一次尝试。

**4）预算与边界：**

- 顾问只有 `read` / `glob` / `grep`，没有 write/bash/subagent，不能改代码、不能继续委派；
- 与 worker 同一个 worktree、同一权限范围；不要把 secrets 放进 brief；
- 建议每 Ticket 最多 1–2 次咨询（目前只能靠提示词约束，没有硬限额）；
- 顾问是 subagent：handoff 时被释放，并受 30 分钟后台上限约束，不能当常驻服务。

**5）验证方法：**

- 选一个已知难题（例如某次 2 次修复失败的 bug），分别在"只用 flash"和"flash + ask_oracle"下跑；
- 记录：是否解决、轮次、耗时、token/费用、顾问建议是否被采纳；
- 结论写回 Ticket/文档，再决定是否 L2。

### L2：产品化（改 dsh-superagent）

- settings 增加 `oracle` 角色（provider/model/reasoningEffort），面板加"顾问模型"，与 eval/executor 并列；
- 工具与 preset 由设置驱动，而不是写死在 bundle patch；
- 增加按角色 / 按 Ticket 的 token 成本统计；
- 可选：把"卡住"做成显式状态（连续失败计数），由系统触发，而不是靠模型自觉；
- 涉及 sessions / executor / panel / prompts 多个模块，属于独立工单。

### 与"直接把 worker 换成强模型"的区别

把 `executor.model` 设成 `deepseek-v4-pro` 是"所有活都用强模型"：简单但贵，TPS 也更低。顾问方案是"默认快模型 + 难题按需升级"，更接近 OmO 的设计。

---

## 6. 风险与取舍

| 风险 | 说明 | 缓解 |
| --- | --- | --- |
| 成本 | pro 按次计费，提示词失效会滥用 | 触发规则 + avoid 列表 + 每单上限 |
| 上下文 | 子 Agent 空对话起步，brief 不全就会答偏 | 要求自包含 brief；顾问可自己读工作区 |
| 生命周期 | subagent 在 handoff 释放，且有 30 分钟上限 | 咨询放在一次 turn 内完成 |
| 权威性 | 顾问只读，回答可能错 | worker 验证后再改；不替代测试 |
| 观测 | 暂无按角色汇总的成本 | L1 看会话用量；L2 加统计 |
| 安全 | 与 worker 同权限、同 worktree | 不传 secrets；只读工具 |

---

## 7. 建议

1. 先做 L1（约半天配置 + 提示词），在一个真实难题上做 A/B；
2. 有效再 L2；无效就删掉，不留维护负担；
3. 把 `eval.model` 设成强模型是另一件事（主 Session 研判），不要和"顾问"混在一起。

---

## 附录：源码与文档

- Oracle 定义：<https://github.com/code-yeongyu/oh-my-openagent/blob/dev/packages/omo-opencode/src/agents/oracle.ts>
- 编排指南：<https://github.com/code-yeongyu/oh-my-openagent/blob/dev/docs/guide/orchestration.md>
- 模型匹配：<https://github.com/code-yeongyu/oh-my-openagent/blob/dev/docs/guide/agent-model-matching.md>
- 中文 README：<https://github.com/code-yeongyu/oh-my-openagent/blob/dev/README.zh-cn.md>
- DSH `tool-subagent`：本地 `@deepseek-ai/dsh-tool-subagent` 的 README 与 `lib/types/index.d.ts`
- dsh-superagent：`src/app/sessions.ts`（preset 创建与 Session 装饰）、`src/app/role-models.ts`（角色模型）、`cordis.patch.yml`（worker/router preset）
