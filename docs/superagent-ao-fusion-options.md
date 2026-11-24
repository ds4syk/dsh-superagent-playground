# SuperAgent × Agent Orchestrator：两套融合方案详解

> 结论先行：A 是“把 SuperAgent 当 AO 的执行后端”，改动小、1 个 PR、可先跑通；B 是“把 AO 当 SuperAgent 的执行层”，更符合你最后一句，也更能发挥两边强项，但要 AO + SuperAgent 两侧改、2 个 PR。**Kimi K3 与 GLM-5.4 独立给出的方案都判定 B**；最终形态是 B，A 的 `dshacp` 只是 B 里 worker 的一种 harness，放 Phase 2。

## 0. 两个系统各自是什么

- **SuperAgent（dsh-superagent-prototype）**：DSH bundle，工厂层/Router 层。它管 Source（GitHub/飞书）、Trigger 去重、Policy/Authorization、Eval/Decision、Ticket/Mandate、Assignment/Input/Report、人的注意力门控。它的 `Executor` 接口只有 `start/deliver/stop`，本来就是给外部执行层预留的缝。
- **AO（agent-orchestrator）**：Go daemon + Electron UI。它管 project、隔离 worktree、23+ harness 的进程/会话、Chat（ACP）或 TUI 控制器、activity 事实、PR/CI/review 观察、Kanban 派生状态。它的强项是“真实执行环境”和“给人和编排器看的执行状态”。

两边最强的部分不重叠：SuperAgent 强在“该不该做、给谁做、做到什么程度算完”；AO 强在“在哪里做、用什么 agent 做、做完的 PR/CI 长什么样”。融合的关键就是选对缝。

## 1. 方案 A：AO 作中控，SuperAgent/DSH 作执行后端

### 1.1 架构

```
人 / AO 桌面 / ao CLI
        │
        ▼
AO Project Orchestrator（AO 自己的 Router 提示词 + ao CLI）
        │  ao spawn --harness dsh
        ▼
AO Session Manager ──► 隔离 worktree
        │
        ▼
dshacp ChatDriver ──► dsh --profile acp（DSH/SuperAgent agent loop）
        │
        ▼
AO 观察 activity / PR / CI ──► Kanban
```

AO 的 orchestrator 继续用现有 `using-ao` skill 调度 worker；只是 worker（或 orchestrator 自己）可以选择新的 `dsh` harness。AO 的会话、worktree、PR/CI 观察全部保留。

### 1.2 具体改动（都在 AO 一个仓库）

1. `domain/harness.go`：加 `HarnessDSH = "dsh"`，加入 `AllHarnesses`；加一条 SQLite migration 重建 `sessions.harness` 的 CHECK（现有迁移 0083 的枚举要加 `dsh`）。
2. `backend/internal/adapters/agent/dsh/`：最小 Agent adapter：
   - `Manifest`：id `dsh`，name `DeepSeek Harness`，capability `agent`；
   - `ResolveBinary`：PATH + `~/.local/bin/dsh` + npm 全局 bin；版本探针 `dsh --version`；
   - `GetConfigSpec`：model 覆盖；
   - TUI 模式如果 DSH 的 `dsh tui` 不稳定，可以只声明 Chat 支持，`GetLaunchCommand` 返回“Chat only”错误。
3. `backend/internal/adapters/chatdriver/dshacp/`：复用 `nativeacp`/`acp` 通用传输，约 50–100 行：
   - `Launch`：`dsh --profile acp`，按需 `--patch` 注入 AO 的 system prompt / model provider；
   - `Capabilities`：streaming/tools/approvals/interrupt/resume/usage 为 true；`history` 为 false（DSH ACP 不支持 `session/load`，只支持 `session/resume`）；diffs/plans 视 DSH 实际更新而定；
   - `Probe`：`dsh --version` + profile 可加载（`dsh --profile acp --dump-config` 或 `--help`），不探测模型凭据；
   - `ValidateInitialize`：校验 `agentInfo.name`/version，拒绝不兼容版本。
4. `chatdriver/registry.Build` 注册 `dshacp.New(...)`；`agent/registry.Constructors` 注册 `dsh.New()`。
5. 前端 harness 列表/agent catalog 自动从后端读取，通常无需改 UI；若 agent catalog 需要图标/描述，补一个条目。
6. 文档：`docs/harnesses/dsh.md`，写安装 `dsh`、profile 准备、Chat-only 限制、模型 provider 配置、认证与故障排查。
7. 测试：`dsh` adapter 的二进制解析/版本探针；`dshacp` 的 launch args、capabilities、initialize 校验；registry 的 shipped driver 列表；SQLite CHECK migration 测试。ACP 协议用 fake ACP server 或 `dsh --profile acp` 的 headless 冒烟测试。

### 1.3 验收

- `ao spawn --project <p> --harness dsh --mode chat --prompt "..."` 能创建一个 DSH 会话；AO UI 能看到 activity、聊天记录、worktree、PR/CI。
- `go test ./...`（除两个环境相关的既有失败）通过；前端 typecheck 通过。
- 用 `dsh --profile acp` 的 fake/真实 server 跑一次 ACP 握手、prompt、cancel、resume。

### 1.4 优点

- 自包含，1 个 PR，改动面小（估计 500–800 行 + 测试），最容易 review。
- 立刻让 AO 能跑 DeepSeek/DSH 模型与 DSH 工具链。
- 不碰 SuperAgent 的 Router/Engine/Policy，风险低。

### 1.5 局限

- 只用到 DSH 的 agent loop；SuperAgent 的 Router、Ticket、Mandate、Subscription、Report、注意力门控全部不参与。
- AO 的 orchestrator 仍是 AO 自己的提示词，不是 SuperAgent 的 Router；谈不上“SuperAgent 的中间层架构”。
- “把 AO 作为 SuperAgent 的执行层”这句话没有兑现。

## 2. 方案 B：SuperAgent 作中间层，AO 作执行层

### 2.1 架构

```
人 / GitHub / 飞书
        │
        ▼
SuperAgent Router（Source→Policy→Ticket→Assignment→Report）
        │  Executor.start / deliver / stop
        ▼
AO Execution Contract（幂等 ensure / 状态投影 / stop）
        │
        ▼
AO Session Manager ──► 隔离 worktree + 任意 harness（Claude/Codex/DSH…）
        │
        ▼
PR / CI / review 事实 ──► AO 派生状态 ──► SuperAgent Report
```

SuperAgent 继续做它擅长的“工厂层”；AO 继续做它擅长的“执行层”。两边不互相读库，只通过 HTTP + 幂等键连接。

### 2.2 具体改动

**AO 侧（PR 1）**

1. 新表 `external_sessions`（external_ref PK、project_id、session_id UNIQUE、assignment_ref、request_fingerprint、created_at、updated_at）+ 迁移 + sqlc 查询。
2. 新 service `backend/internal/service/execution`：
   - `Ensure(ctx, ref, assignmentRef, fingerprint, spawnCfg)`：同 ref 同 fingerprint 返回已有会话；同 ref 不同 fingerprint 返回 409；并发 ensure 用唯一约束 + singleflight，失败方回滚新会话。
   - `Get(ctx, ref)`：返回 AO 的 `domain.Session` 读模型（status/activity/PR/CI/needsPerson）。
   - `List(ctx, projectId)`；`Stop(ctx, ref, reason)`（幂等 kill，不删 dirty worktree）。
3. 新 controller `backend/internal/httpd/controllers/execution.go` + DTO：
   - `PUT /api/v1/execution/sessions/{externalRef}`（ensure）
   - `GET /api/v1/execution/sessions/{externalRef}`（state）
   - `POST /api/v1/execution/sessions/{externalRef}/stop`
   - `GET /api/v1/execution/sessions?projectId=`
   - 路由注册 + OpenAPI specgen 条目 + `openapi.yaml`/`frontend/src/api/schema.ts` 重新生成。
4. 投递不新增路径：复用 `POST /api/v1/sessions/{id}/conversation/messages` 的 `clientMessageId` 幂等和 `duplicate` 标记；执行契约的 state 返回 `sessionId`。
5. 观察不新增路径：`Get` 复用 `service/session` 的 `Session` 读模型和 PR facts，不重新实现 Kanban 派生规则。
6. 测试：service 的幂等/冲突/回滚/停止幂等/未知 ref；controller 的表驱动契约；migration 测试；用 fake session service 避免真实 harness。
7. 文档：`docs/integrations/superagent.md`，写执行契约、状态映射、幂等语义、恢复流程。

**SuperAgent 侧（PR 2，patch/薄适配）**

1. 新增 `src/executor/ao.ts` 实现 `Executor`：
   - `start`：`PUT /execution/sessions/{ref}`，fingerprint 用 assignment 的不可变启动参数；失败走 `engine.report(..., 'failed')`。
   - `deliver`：解析 `sessionId` 后调用 AO 的 conversation messages（`clientMessageId` 幂等），成功 `engine.inputsDelivered`，失败 `engine.deliveryFailed`。
   - `stop`：`POST .../stop`；先写 SuperAgent 的 ending 事实，再发请求。
   - poller：按 `externalRef` 轮询 AO state，映射 `ready/blocked/failed/waiting/dropped` 并调用 `engine.turnEnded`/`report`；用 `updatedAt` 去重。
2. `runtime.ts` 组合根：把硬编码的 `new DshExecutor(...)` 改成可配置的 executor factory（`dsh` | `ao`），默认 `dsh`，不改变现有部署。
3. `policy.ts` / 配置：加 `execution.executor: 'dsh' | 'ao'`、`ao.baseUrl`、`ao.projectId`、超时/重试。
4. 测试：fake AO HTTP server（httptest/vitest msw）测 ensure/deliver/stop/状态映射/重试；`runtime` 组合根选择测试；不依赖真实 AO daemon。

### 2.3 验收

- SuperAgent 接受一个 Ticket → 通过 `AoExecutor` 在 AO 创建会话/worktree → 投递 assignment → AO worker 干活并开 PR → SuperAgent 收到 `ready` + PR link → 人在原 thread 收到结果。
- AO daemon 重启 / SuperAgent 重启后，按 `externalRef` 能找回同一会话，不重复 spawn。
- 同 `externalRef` 不同 fingerprint 被 409 拒绝；重复 `clientMessageId` 不产生第二个 turn。
- `go test`（AO）+ `pnpm test`/`typecheck`（SuperAgent fork）通过。

### 2.4 优点

- 完整复用两边强项，真正“有机融合”。
- 符合“把 AO 作为 SuperAgent 的执行层”。
- 执行契约是通用的，未来接别的 orchestrator 也能用。
- 不碰 AO 的 OBSERVE→UPDATE→DERIVE，也不碰 SuperAgent 的 Policy/Engine。

### 2.5 风险

- 两侧改动，联调成本高；两个 PR 需要一起验收。
- SuperAgent 原型是研究代码（单 commit、peer 依赖 alpha），组合根/配置可能要 patch。
- 状态映射（AO 派生状态 ↔ SuperAgent Report）容易漂移，需要表驱动测试和边界文档。
- AO 现有 API 已经能覆盖大部分；新增执行契约要避免变成“第二套 session API”，只做投影和幂等。

## 3. 并排对比

| 维度 | A | B |
|---|---|---|
| 谁在上 | AO | SuperAgent |
| 复用 SuperAgent 中间层 | 否 | 是（Router/Ticket/Mandate/Report） |
| 复用 AO 执行层 | 是 | 是 |
| 改动仓库 | AO | AO + SuperAgent |
| PR 数 | 1 | 2 |
| 估计工作量 | 1–2 天 | 3–5 天 |
| 风险 | 低 | 中 |
| 可立即验证 | 是（ao spawn --harness dsh） | 需要两侧联调 |
| 最终形态 | 子能力 | 目标形态 |

## 4. 两个模型的独立意见

### Kimi K3（直接 API，`kimi-k3`）

- 方向：SuperAgent 在上，AO 作为可插拔 `Executor` 执行层；用户末句为准。
- AO 侧：新增 `adapters/executor/superagentexec/`，提供 start/deliver/stop 三个 HTTP 端点；持久化 assignment 状态；start 以 `assignment.ID` 幂等；并发槽、崩溃恢复、Report 回调。
- SuperAgent 侧：薄 `AoExecutor` + 组合根 patch。
- 它明确说：把 Router 塞进 AO 等于重写 AO 核心；AO 的价值在执行侧。

### GLM-5.4（直接 API，`glm-5.4`）

- 方向：同样判定 B，并指出“魔改 AO”应解释为“把 AO 改造成合格的 Executor”，不是把 Router 搬进 AO。
- AO 侧缺口只有约 20%：session 创建接受 `metadata` + `Idempotency-Key`；新增持久化 report 流（SSE，`since=seq` 可回放）；stop 携带 reason；chat 投递已有 `client_message_id` 去重；新增纯函数 `DeriveWorkerState(session facts, approve, PR/CI)`。
- SuperAgent 侧：`AoExecutor` + ReportPump，core 零改动。
- 它明确说：`dshacp` 只是“又一个 harness”，丢失 Router/Worker 语义，不构成融合，放 Phase 2。

### 综合结论

两者在方向、缝、SuperAgent 侧薄适配、AO 侧只做执行契约上一致。差异在 AO 侧实现：
- Kimi：`external_sessions` 映射表 + 三个端点 + Report 回调；
- GLM：session metadata + Idempotency-Key + 持久化 report 流（SSE）。

MVP 取两者交集、先做最稳的：**`external_sessions` 幂等映射 + `GET execution` 状态投影 + `stop`**；Report 流先用轮询（`updatedAt` 去重），SSE 作为下一步。这样不需要一次引入事件日志和双端共享映射表，仍能跑通端到端。`dshacp` 放 Phase 2。

## 5. 推荐落地顺序

1. **B 的第一刀（AO PR 1）**：`external_sessions` 幂等映射 + `PUT/GET/POST stop` 执行契约 + `GET execution` 状态投影；SuperAgent 侧 `AoExecutor` 轮询这个视图。
2. **B 的第二刀（SuperAgent patch）**：`AoExecutor` + ReportPump + 组合根选择器，用 AO 的契约跑通一张 Ticket。
3. **B 的增强**：Report SSE（`since=seq`）、审批动作透传、断点续传；再评估 `dshacp` 作为 AO worker 的可选 harness。

这样既符合两个模型的 B 判定，又用一个可 review 的 AO PR 先落地执行契约。

## 5. 需要 owner 确认的点

1. 生产部署：SuperAgent Router 与 AO daemon 是否在同一台机器？跨机时需要 base URL、认证和网络策略。
2. SuperAgent 侧是否可以接受一个 fork/patch（`AoExecutor` + 组合根选择器）？还是只交付 AO 侧契约 + 适配说明。
3. 是否同意“先 A 后 B”的顺序；如果只想先看 B，我可以直接做 B 的 AO 执行契约（PR 1），但联调要等 SuperAgent 侧 patch。

## 6. 关键澄清：桥接层（2026-11-25 owner）

Owner 补充了真正的目标：

> SuperAgent 的要点在于人类入口与工作区的解耦，工作区不必成为人类的认知负担，从而人可以通过一个对话 agent 去管理多个工作区。在单个管理的工作区中，AO 的架构更具优势。问题在于怎么处理好其中的桥接层。

这确认了 B，并把设计重心从“谁在上”转到“桥接层怎么切”：

```
一个人类对话入口（SuperAgent Router，Feishu/GitHub）
        │  人只说“在 workspace X 做 Y”
        ▼
Workspace Registry（bridge 配置：workspaceRef → AO project/repo）
        │
        ▼
SuperAgent Ticket（一个 Ticket 只属于一个 workspace）
        │  Executor.start/deliver/stop
        ▼
AO Execution Contract（external_ref + project_id + session_id）
        │
        ▼
AO 单工作区执行：project → worktree → harness session → PR/CI
        │
        ▼
Report 回到同一对话（带 workspace 标签）
```

桥接层的四条不变量：

1. **一个 Ticket 只属于一个 workspace**；跨 workspace 的请求拆成多个 Ticket，不共享 session。
2. **workspaceRef → AO projectId 的解析是桥接层职责**：优先按 repo URL 匹配 AO project；未注册时返回 `blocked`，并给出“注册仓库/授权”的具体动作，不在 Router 里猜。
3. **AO session 永远属于一个 project**：worktree、branch、PR 都在该 project 内；SuperAgent 不直接读 worktree 路径。
4. **人的注意力只回入口**：所有 workspace 的 ready/blocked/failed 都回到最初的人类对话，带 workspace 标签；人不管理 worktree、分支、CI 页面。

桥接层的最小实现（也是 MVP 的第一刀）：

- AO 侧：`external_sessions` 以 `external_ref` 为幂等键，记录 `project_id` + `session_id`；`PUT/GET/stop` 执行契约；`GET execution` 返回 status/needsPerson/PR/CI。
- SuperAgent 侧：`AoExecutor` 的 `start` 从 Ticket 上取 `workspaceRef`，经 Workspace Registry 解析为 `projectId`，再调 AO ensure；`deliver` 用 `clientMessageId`；`stop` 透传 reason；poller 把状态映射为 Report，回同一对话。
- Workspace Registry 先放桥接层配置（`workspaceRef`、repo URL、AO projectId、默认 harness/model），不放进 SuperAgent 的 core；后续可做成 AO project 的自动发现。

这样“人类入口与工作区解耦”和“单工作区内用 AO”同时成立：SuperAgent 只承担入口、授权、Ticket、注意力；AO 只承担单工作区的执行与 PR/CI 事实；桥接层只做解析、幂等和状态翻译。
