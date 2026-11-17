This is by an automated agent, not a human.

## 摘要

诊断 2026-11-17 在 Web 端看到的「工单中断」。结论：**不是工单真的中断，而是 Web 端冷读误报**——个人 web host 读取一个仍由 superagent host 写入的会话时，把未闭合的回合补成 `turn/end {kind:'interrupted'}`，聊天 UI 据此显示「DSH 进程在本轮完成前退出」。

新增的文件：

```text
docs/
├── reports/2026-11-17-web-interruptions.md   # 诊断报告（时间线、机制、复现、建议）
└── visuals/2026-11-17-web-interruptions.png  # 一页可视化
deliverables/
└── Web端中断误报诊断_2026-11-17.docx        # 6 页 DOCX
scripts/
├── make_web_interruptions_figure.py          # 生成一页图
└── build_web_interruptions_docx.py           # 生成 DOCX
```

![Web 端中断误报诊断](https://github.com/ds4syk/dsh-superagent-playground/raw/agent/diagnose-web-interruptions-69610e43782d/docs/visuals/2026-11-17-web-interruptions.png)

## 证据

- **Before（用户看到）**：19:06 左右打开个人 web，看到四色 / π 两个工单「任务已停止 / DSH 进程在本轮完成前退出」；随后在 19:07:57、19:08:09 发出 resume 请求。
- **After（实际状态）**：四色 worker 的 turn 1 从 18:06:34 开到 19:09:50 `completed`（1322 步）；π 子 agent 的 turn 1 从 18:58:33 到 19:10:33 `completed`；resume 消息在 19:07:58 / 19:08:11 被并入**仍在运行**的回合。11-17 当天全部会话的 `interrupted` turn 数为 **0**；superagent host 自 11-13 18:29 起未重启。
- **复现**：把四色会话在 19:09:50 之前的日志前缀交给 `interruptedTurnClosers`，输出：

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

同一脚本对 π 子 agent 的前缀也返回一条 `interrupted` closer。只要有回合未闭合，冷读就会显示「中断」，与写者是否存活无关。

- **代码路径**：`dsh-session-query` 的 `readColdSessionLog` → `interruptedTurnClosers`；持久化层明确「Readers never touch the lock」（`dsh-session-persistence-jsonl` 的 lease 注释），因此读者无法区分「写者崩溃」与「写者仍活着」。
- **历史对照**：11-13 14:56/14:58 的 host 重启确实造成 5 个 worker + orchestrator 的真实 `interrupted`；11-01/11-02 的 Dreamer 个人会话也有真实中断。今天没有。

完整证据、源码行号、限制与建议见 [诊断报告](https://github.com/ds4syk/dsh-superagent-playground/blob/agent/diagnose-web-interruptions-69610e43782d/docs/reports/2026-11-17-web-interruptions.md)；DOCX 见 [deliverables/Web端中断误报诊断_2026-11-17.docx](https://github.com/ds4syk/dsh-superagent-playground/blob/agent/diagnose-web-interruptions-69610e43782d/deliverables/Web%E7%AB%AF%E4%B8%AD%E6%96%AD%E8%AF%AF%E6%8A%A5%E8%AF%8A%E6%96%AD_2026-11-17.docx)。

## 合并风险

**Door:** two-way（仅文档与图片，无代码、无部署行为变化；删除即可回退）

**Blast Radius:** docs（不影响运行时、工单数据或 DSH 行为）

说明：

- 本 PR 只记录诊断，不修改 DSH core。P0 修复建议（冷读前探测 `session.lock` 写租约、区分显示期合成 closer）需要在上游 DSH 源码侧实施；本机只有 `0.2.1-rc.2` 安装包。
- 报告未包含用户浏览器截图（本机无浏览器自动化），误报的「打开的是哪个会话」是时间线推断；冷读机制、复现与日志证据均已核实。
