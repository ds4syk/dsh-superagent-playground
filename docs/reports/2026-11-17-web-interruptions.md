# Web 端显示「中断」的诊断：冷读误报，不是工单真的中断（2026-11-17）

> **结论先行**：今天在 Web 端看到的「任务已停止 / DSH 进程在本轮完成前退出」**不是工单真的中断**。它是 DSH Web 端在读取一个**仍由另一个 DSH 进程写入**的会话时，把「尚未结束的回合」当成「写进程已崩溃」而补出的恢复事件。四色定理和 π 指数两个工单的执行进程在 19:06–19:08 一直活着，用户 19:07/19:08 发出的 `resume` 消息被并入运行中的回合，随后正常完成；两个会话日志里**没有任何** `interrupted` 事件。历史上有过真实中断（11-13 14:56/14:58 的 host 重启），今天为 0。

**如果只记三件事**：

1. **误报机制**：`readColdSessionLog` 对任何「最后回合未结束」的冷读日志追加合成的 `turn/end {kind:'interrupted'}`；聊天 UI 据此显示「DSH 进程在本轮完成前退出…」。读者不检查写租约（`session.lock`），所以另一个进程正在写的会话也会被这样标注。
2. **事实层**：superagent host 从 11-13 18:29 起连续运行；今天 0 个 interrupted turn；四色 worker 的 turn 1 是 18:06:34–19:09:50 `completed`，π 子 agent 的 turn 1 是 18:58:33–19:10:33 `completed`。
3. **建议**：冷读前探测写租约；写进程还活着时标成「另一进程正在写入（只读）」，而不是「已中断」。短期从 superagent 面板（3092）的 Open Session 打开 worker 会话，或在看到该提示时先核对最后事件时间与工单状态。

## 0. 口径

| 项目 | 说明 |
|-|-|
| 时间 | Asia/Shanghai（+08:00）；会话事件时间取自 `session.v4.jsonl.zstd` 的 `time` 字段 |
| 对象 | `ds4syk/dsh-superagent-playground` 部署；DSH `0.2.1-rc.2`；superagent host PID 76796；个人 web host PID 2039546 |
| 证据 | 会话日志（zstd JSONL）、进程启动时间、`/proc`、superagent 管理 API、DSH 包源码与 UI 文案 |
| 证据分级 | **已核实**：时间点、会话事件、代码路径、复现输出；**推断**：用户当时具体打开的是哪个会话（日志不记录 Web 端浏览行为） |

## 1. 用户看到的到底是什么

Web 端会话视图里，红色错误行由 `turn/end` 的 `reason.kind = "interrupted"` 渲染出来。`dsh-client-ui-chat` 把该状态映射为：

- 标题：`任务已停止`（`message.taskStopped`）
- 正文：`DSH 进程在本轮完成前退出（例如被强制结束、崩溃或关闭超时），重新启动后本轮已中止。未完成的工具调用结果未知。`（`message.failure.interrupted`）

所以从用户视角看，它确实写着「进程可能被强制结束、崩溃或关闭超时」——**看起来像不明原因的中断**。但这条状态只说明「读到的日志里有一个没有闭合的回合」，并不等价于「写这个回合的进程真的死了」。第 3 节给出误报路径，第 4 节给出反证。

## 2. 今天的两个工单：没有实际中断

### 2.1 时间线

| 时间（11-17） | 事件 | 证据 |
|-|-|-|
| 18:06:34 | 四色工单 worker 的 turn 1 开始 | session `worker-09ced558…` `turn/start turn=1` |
| 18:58:33 | π 工单的独立子 agent（`独立推导 μ(π)=2`）turn 1 开始 | session `f0040578…` `turn/start turn=1` |
| 19:06:16 | 个人 `dsh web --no-open --port 0` 启动（用户从远端打开 Web UI） | `/proc/2039546` 启动时间 |
| 19:06:26 | 个人 web 在 π 工单目录下新建空会话 `session-ded22bb1…` | 该会话目录与文件时间 |
| 19:07:57 | 用户在四色工单 thread 发「resume一下这份工作，疑似中断了」 | Feishu trigger `1c31ac4b…` |
| 19:07:58 | 该消息被并入**仍在运行**的四色 worker turn 1（`agent/inbox/spliced`） | 四色会话 seq 10403 |
| 19:08:05 | 四色 worker 在 turn 1 内明确处理该 resume 消息 | 四色会话 seq 10411 |
| 19:08:09 | 用户在 π 工单 thread 发同样的 resume | Feishu trigger `710abd4a…` |
| 19:08:11 | 该消息进入 π worker 的 turn 5 | π worker 会话 seq 1195/1199 |
| 19:08:57 | π worker 的 turn 5 `completed` | π worker 会话 `turn/end` |
| 19:09:50 | 四色 worker 的 turn 1 `completed`（共 1322 步） | 四色会话 `turn/end` |
| 19:10:33 | π 子 agent 的 turn 1 `completed` | 子 agent 会话 `turn/end` |

### 2.2 关键反证

- 四色 worker 的 turn 1 从 18:06:34 一直开到 19:09:50；在 19:06–19:08 用户「看到中断」的窗口内，它处于**打开状态**，且仍在每约 10 秒轮询构建日志。
- 用户在 19:07:57 发的 resume 被 **spliced 进运行中的 turn 1**（19:07:58），worker 在 19:08:05 读取并回应；如果 turn 真的在 19:06 中断，这条消息只会开启新回合，而不会出现在 turn 1 的 step 1300。
- 两个会话的 `turn/end` 原因全部是 `completed`；对全部 `~/.dsh/sessions` 的扫描显示 **11-17 当天 interrupted turn = 0**。
- superagent host 76796 自 **11-13 18:29:13** 起持续运行（11-17 19:14 时已 4 天 0 小时），期间没有重启；不存在「今天 host 崩了导致回合被截断」的路径。

## 3. 误报是怎么产生的

### 3.1 冷读路径

用户使用的 Web 端**最可能是个人 web host**（PID 2039546，19:06:16 启动），不是写这些 worker 会话的 superagent host（PID 76796）。个人 web 没有这些 worker 会话的 in-process Agent，打开会话时走**冷读**。反过来说，如果从 superagent host 自己的 Web UI（3092）打开，这些会话是 live 的，不会出现该误报；而今天的日志里没有任何 interrupted 事件，所以今天显示的「中断」只可能来自跨进程冷读。冷读路径是：

1. `dsh-session-query` 的 `readColdSessionLog` 读取持久化日志；
2. 读完后执行 `events: [...events, ...interruptedTurnClosers(events)]`；
3. `interruptedTurnClosers` 对日志尾部**任何未闭合的 turn** 追加合成的 `turn/end`，原因为 `{ kind: "interrupted" }`；
4. 聊天 UI 把这条合成事件渲染成第 1 节的「任务已停止」错误行。

源码位置（本机安装包）：

- `…/@deepseek-ai/dsh-session-query/lib/index.js`：`readColdSessionLog`（第 26–54 行）、`loadSource`（第 413–419 行）；
- `…/@deepseek-ai/dsh-session/lib/index.js`：`interruptedTurnClosers`（第 907–908 行）、`openTurnClosers`（第 750 行起）；
- `…/@deepseek-ai/dsh-client-ui-chat/lib/client.js`：`endFailure` 把 `interrupted` 映射为 `{ message: "The DSH process exited before this turn finished.", code: "INTERRUPTED" }`（约第 10475–10484 行）；中文文案见第 5881 行附近。

### 3.2 为什么另一个进程正在写也会被误判

会话目录里的 `session.lock` 是**写租约**，只有写者持有；持久化层的设计明确写着 **“Readers never touch the lock”**（`dsh-session-persistence-jsonl/lib/index.js` 的 lease 注释）。因此个人 web 可以无阻塞地冷读一个**正在被 superagent host 写入**的会话；它看到的最后一个 turn 当然是「未闭合」的，于是被补成 `interrupted`。冷读层无法区分：

- 写者已崩溃（真实中断，应补 closer）；
- 写者仍活着，只是被另一个 DSH 进程从外部读取（误报）。

### 3.3 可复现的验证

把四色会话在 19:09:50 之前的日志前缀（seq < 10617）交给 `interruptedTurnClosers`，输出正好是一条合成的中断：

```json
{
  "events": 10617,
  "turnsStarted": 1,
  "turnsEnded": 0,
  "closers": [
    { "type": "turn/end", "seq": 10617, "time": 1794913790463,
      "data": { "turn": 1, "reason": { "kind": "interrupted" } } }
  ]
}
```

同一脚本对 π 子 agent 的 19:10:33 前前缀（seq < 551）也返回一条 `interrupted` closer。也就是说：**只要在回合结束前冷读，就必然显示「中断」**，与写者是否存活无关。

## 4. 真正的中断历史（对照）

扫描全部 `~/.dsh/sessions`，历史上共 11 个 `interrupted` turn，集中在两个时段：

| 时间 | 会话 | 原因 |
|-|-|-|
| 11-01 15:55 / 20:28 / 21:40；11-02 18:10 | Dreamer 个人会话（`session-9e5fc772…`、`session-40f1da1c…`） | 个人 `dsh web` 随远端连接退出，回合被截断（真实中断） |
| 11-01 22:25 | resume-dreamer worker | host 重启/会话恢复链（真实中断） |
| **11-13 14:56:45 / 14:56:48 / 14:56:52** | sdpr-dreamer-notebooks、daily-summary-zero、resume-dreamer 三个 worker | superagent host 重启（启动日志 14:56:53） |
| **11-13 14:58:09 / 14:58:11 / 14:58:12** | dreamer-version-diff、sdpr-gpu-parallel 两个 worker + orchestrator | 第二次 host 重启（启动日志 14:58:13） |

对照今天的两个会话：没有任何 `interrupted`；worker 的回合都自然闭合。**同一句「中断」文案，在 11-13 是真实的，在今天这次是冷读误报。**

## 5. 结论：是否存在该问题

| 问题 | 结论 |
|-|-|
| 今天这两个工单真的中断了吗？ | **没有**。四色 worker 在 19:06–19:08 持续运行；π 工单的 resume 进入 turn 5 并完成。两个工单的工作都没有被中断。 |
| Web 端显示「中断」是界面 bug 吗？ | **是，存在可复现的误报**。跨 DSH 进程冷读一个仍在写入的会话时，未闭合回合被无条件当成本地崩溃恢复。 |
| 有数据丢失吗？ | 没有。工作区、git、PR 与会话日志都在；误报只影响 Web 端的显示与用户判断。 |
| 当前系统是否异常？ | 没有。superagent 管理 API `problems = 0`，16 个 open 工单无 `failed`/`blocked`；运行 2 / 槽位 3。 |
| 影响面 | 任何「A 进程写、B 进程读」的场景：个人 `dsh web`/`dsh-remote` 查看 superagent worker 会话、多个 DSH 实例共享 `~/.dsh/sessions`、崩溃后由另一进程打开旧会话。 |
| 责任位置 | DSH core：`dsh-session-query` 的冷读恢复 + `dsh-client-ui-chat` 的文案；不是 `dsh-superagent` 插件本身。本机只有安装包（无源码 checkout），因此本工单只做诊断，不在这里改 DSH 代码。 |

## 6. 建议

### P0（可直接消除误报）

1. **冷读前探测写租约**：对 `session.lock` 做一次非阻塞 `flock(LOCK_EX|LOCK_NB)` 探测（成功即无写者，立即释放；`EAGAIN` 表示写者仍存活）。
   - 写者存活：**不追加** `interruptedTurnClosers`，改为返回一个 `writing-elsewhere`/`openTurn` 标记；
   - 写者已死：维持现状，补 `interrupted` closer（真实崩溃恢复）。
2. UI 层区分 **持久化恢复** 与 **显示期合成**：合成的 closer 带 `synthetic: true`（或单独字段），聊天视图对合成的显示为「另一进程正在写入，当前为只读快照」，不要复用「DSH 进程退出」文案。

### P1（过渡）

3. 在 superagent 面板（3092）的工单详情里，用 **Open Session** 打开 worker 会话（同一 host，live 读取），不要从个人 web 打开另一个 host 正在写的 worker 会话。
4. 看到该提示时，先用两个廉价信号核对：会话最后事件时间是否仍在推进；工单面板的 stage 是否仍为「执行中」。两者任一成立，就不要据此判断中断。

### P2（体验）

5. 个人 web 的会话列表对「另一个进程持有写租约」的会话显示「另一实例运行中」的只读标记，避免用户点进去后看到误导性的结束状态。

## 7. 限制与未做

- **无法从日志确定用户具体打开了哪个会话**：Web 端不记录浏览行为。按时间与机制，四色 worker（turn 1 在 18:06:34–19:09:50 开着）最符合；π 工单的独立子 agent（18:58:33–19:10:33）也处于打开状态，若用户点开的是父会话下的子会话，同样会看到误报。这是**推断**。
- **没有截图**：本机没有可用的浏览器自动化；本次验证是代码级 + 日志级复现，不是对用户浏览器画面的回放。
- **未修改 DSH 代码，未向上游提 issue**：本机 `@deepseek-ai/dsh` 是 `0.2.1-rc.2` 的安装包（只有 `lib/`，无 `src/`），改不了也不会在这里发布；P0/P1 需要 DSH core 侧修改。若需要，我可以按 P0 起草一个上游 issue/补丁方案。
- **真实根因的上游确认**：个人 web 冷读路径与 `interruptedTurnClosers` 的行为已在本机核实；「读者不碰锁」是持久化层的既定设计，不是本次临时观察。
