# AO 执行桥接补丁

`ao-execution-bridge.patch` 是 `ds4syk/agent-orchestrator` 分支
`agent/orchestrator-fork-adapt-9d591cc8a18c` 相对 fork `main`（72c68b59）的补丁，
包含 external_sessions 幂等映射、execution API、revision、OpenAPI/前端类型与测试。

对应的可 review PR：https://github.com/YingkeSu/agent-orchestrator/pull/34
应用：在 AO 仓库 `git am patches/ao-execution-bridge.patch`。
