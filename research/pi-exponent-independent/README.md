# 第二独立窗口：μ(π)=2 盲测记录（2026-11-17）

本目录是第二个独立窗口在严格盲测条件下的完整尝试。该窗口：

- 未读取 `openai/math` 的 Family 017 论文、推理摘要、Lean 源码；
- 未读取 `research/pi-exponent/pi_exponent_phase1.md`、`docs/` 或 PR #30；
- 只写本目录，未修改其他文件/分支/PR。

**主报告**：[`independent_attempt.md`](independent_attempt.md)（约 27k 字符）。

**结论**：完整证明 $\mu(\pi)\ge2$；证明等价刻画
$\mu(x)=2+\limsup \log a_{n+1}/\log q_n$，从而 $\mu(\pi)=2$ 等价于 $\pi$ 的连分数部分商次指数增长；
**未能证明** $\mu(\pi)\le2$，并给出两个等价障碍形式（连分数 / 线性型）。

**数据与图**：`cf_d50000.csv`、`cf_d100000.csv`、`*.svg` 为本地大文件，不纳入 git；
运行 `python3 compute_cf.py` 可重现。`summary_*.json/txt`、`run_*.log` 保留作为数值结果记录。

**状态说明**：本报告写于盲测条件，其中“$\mu=2$ 是公开问题”“上界约 7.6”等背景判断是基于 2025 年前的训练知识，
已被同日的官方过程核对更正（见 [`../pi-exponent/pi_exponent_phase1.md`](../pi-exponent/pi_exponent_phase1.md) §9）：
OpenAI 2026-09-24 论文声称证明 $\mu(\pi)=2$，已有约 869 个文件的 Lean 形式化，但独立编译/审计尚未完成。
本目录保留原始盲测记录，不做事后改写。
