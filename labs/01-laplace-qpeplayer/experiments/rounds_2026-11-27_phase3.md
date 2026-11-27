# P1 优化轮次日志 · 第三阶段（2026-11-27 晚，30 轮预算）

起点：HEAD `17d7ffc`（phase-2 产物已提交）。约束：不改 `deliverables/`、`AGENTS.md`、顶层 `labs/EXPERIMENTS.md`；不 commit/push；只做非数学工程/实验/论文一致性，数学口径以修订版 P1-2、P1-Q5b/Q5c 为准，不超出 10×10 与满 20D 13×13 的范围。

| Round | hypothesis | change | command | metric before → after | keep/revert | artifact |
|-|-|-|-|-|-|-|
| 1 | HEAD 的 phase-2 状态完整，`run_all.sh --with-sdp` 应从干净 worktree 通过 | 复核 HEAD 跟踪文件 + 跑集成 harness | `./run_all.sh --with-sdp` | 7 脚本 + sdp 全 rc=0，**34.307 s PASS**；已提交 results hash 不变 | keep | `results/logs/20261127-220057/` |
| 2 | Scorer 脚本应纳入一键复现组，且结果可复现 | `run_all.sh` 新增 `--with-scorer`（preflight + 期望输出 + 数值比对 + 归档） | `./run_all.sh --with-sdp --with-scorer` | 8 脚本全 rc=0，**55.352 s PASS**；scorer 脚本 **20.807 s**；`scorer_fold.json` 非计时字段与已提交一致 | keep | `run_all.sh`、`results/logs/20261127-220143/` |
| 3 | REPRO 文档应记录 `--with-scorer` 的墙钟与用法 | 更新 `REPRO.md`（范围/墙钟表/命令/实测） | — | `--with-sdp --with-scorer` 55.4 s；scorer 20.8 s；默认 35.1 s | keep | `REPRO.md` |
| 4 | lab README 的「下一步 3 条」仍停留在 Q5 未结案、LINEMOD 负结果口径 | 在 `labs/01-laplace-qpeplayer/README.md` 增加 2026-11-27 晚状态块并改写 3 条下一步 | — | 明确 Q2/Q13 与 Q5 分层结案、真实双峰正向但 Laplace 偏差、未做项与工程 harness | keep | `labs/01-laplace-qpeplayer/README.md` |
| 5 | 论文数字应与结果文件可机器对照，防止重跑后漂移 | 新增 `code/check_paper_numbers.py`（16 条 claim 的 paper pattern ↔ 结果文件值） | `python code/check_paper_numbers.py` | **16/16 patterns found**（presence/rounding 检查，非语义审查） | keep | `code/check_paper_numbers.py` |
| 6 | Scorer 报告的 1D 原始权重可由解析式独立复算（不是只对 fraction） | 用 $w=\sqrt{2\pi\tau/\lambda_1}e^{-J/\tau}$、$\lambda_1=c_3x_+$、$\eta=\tau^{1/3}$ 独立计算 | 独立 Python 计算 | $w_{\rm std}=4.4575$、$w_{\rm reg}=0.6726$、fraction 0.867032/0.495950（与 scorer 报告逐位一致）；Scorer raw 0.349624 + $Z_{\rm rest}=0.6836$ → 0.338382 | keep（支撑 Round P3） | 本轮日志 |
| 7 | τ 幂次可按文档自身 3D Laplace 约定严格重算 | 并入 theory3 的 `theory/proofs.md` §7.5.3：$\det(H+\eta I)	o(\lambda_1+\eta)\det H_\perp$，$\eta=c_0	au^{1/3}$ | 复制（+58/−11） | 折叠权重 **Θ(τ^{4/3}e^{−J0/τ})**；跳变分数 **Θ(τ^{−1/6}e^{−ΔJ/τ})**（ΔJ=O(τ) 时 Θ(1)）；ζ≪−1 权重 **O(τ²/|α|)**；原文 τ^{1/3} 是 1D 软方向因子；定性“跳变非零”不变 | keep | `theory/proofs.md` §7.5.3 |
| 8 | 6D Scorer 的位姿跳变应随网格细化线性下降（真连续，不是数值假象） | 子 agent `wt-p1-scorerscale`（运行中） | — | 待 | pending | — |
| 9 | 并入 paper3：全文 Q2/Q13 三层限定、缺失数值表、claim→evidence 表、τ 幂次与 6D 状态同步 | 复制 `paper/draft.md`（+85/−33，17 表） | — | 无 stale "Q2 open/pending/reopened"；τ 指数改为权重 τ^{4/3}/分数 τ^{−1/6}；6D Scorer 标为已实现 | keep | `paper/draft.md` |
| 10 | sdpfig 的 Q2 一页图可关闭 sdp2 blocked | 并入 `sdp/make_q2_summary.py`、`results/q2_summary.{png,json}` | `python sdp/make_q2_summary.py` | 两面板：10 实例 f10 vs f13up + 6 点星标；6 点柱状（5.415000×3 / 3.594314，rank 1/1/1/4，20D vs 17D） | keep | `sdp/...q2_summary.*` |
| 11 | 6 点 SDP 应有数值证书区间，且不能把 proof 值误报为被包含 | 并入 `sdp/run_six_point_cert.py`、`results/six_point_cert.{md,json}`；修正其一句错误（区间不含 1.82069114） | `python sdp/run_six_point_cert.py` | f10∈[5.41499990,5.41500000]、f13full 同、f13up∈[3.59431363,3.59431383]、gap∈[1.82068607,1.82068637]；PSD 裕度 1e-8、Cholesky OK | keep | `sdp/...six_point_cert.*` |
| 12 | 论文与结果文件的数字应机器可查，且新表格/修正后仍 16/16 | 更新 `code/check_paper_numbers.py` 并运行 | `python code/check_paper_numbers.py` | 16/16 patterns found（含新 Table 5.8/5.16/5.17 数字） | keep | `code/check_paper_numbers.py` |
| 13 | 6 点证书与 Q2 图应纳入 `--with-sdp` 一键复现 | `run_all.sh` 的 `--with-sdp` 追加 `run_six_point_cert.py` 与 `make_q2_summary.py` | `./run_all.sh --with-sdp` | 9 脚本全 rc=0，**34.877 s PASS**；证书 1.38 s、图 1.11 s；已提交 results 不变 | keep | `run_all.sh`、`results/logs/20261127-220948/` |
| 14 | 论文应引用 Q2 图与证书区间，且区间不含证明引用值 | 在 §5.3.4 加图引用；6 点段加证书区间与“proof 值落在区间外” | — | 论文现引 `q2_summary.png` 与 `six_point_cert.md`；明确 proof 的 3.59430886/1.82069114 在区间外 | keep | `paper/draft.md` |

## 并行子 agent（phase-3 wave 1）

| Round | task | worktree | 状态 |
|-|-|-|-|
| P1 | 论文全文 Q2/Q13/Q5 一致性 + 缺失数值表 + claim→evidence 表 | `wt-p1-paper3` (`opt/p1-paper3`) | running |
| P2 | 理论 §7.5 τ 幂次推导 + 一致性 + 状态表 | `wt-p1-theory3` (`opt/p1-theory3`) | running |
| P3 | Scorer 网格细化 scaling + 位姿梯度 + 浅 fold 定量限制 | `wt-p1-scorerscale` (`opt/p1-scorerscale`) | running |
| P4 | f13_up 末 5 位取证：高精度/对偶证书/构造变体 | `wt-p1-forensics` (`opt/p1-forensics`) | running |
| P5 | 关闭 sdp2：Q2 一页图 + 6 点数值证书 + sdp/README | `wt-p1-sdpfig` (`opt/p1-sdpfig`) | running |
