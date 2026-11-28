# P1 优化轮次日志 · 第二阶段（2026-11-27 晚，新数学证明后）

预算：新增 36 轮。约束（本阶段）：可改 `labs/01-laplace-qpeplayer/`（含 `theory/proofs.md`，按用户新证明整合）；不改 `deliverables/`、`AGENTS.md`、顶层 `labs/EXPERIMENTS.md`；不 commit/push。
新证明来源：Feishu `RhzDwI1RKinQsekHDhUcjOVenHL`（P1-Q2/Q13 修订定理 + P1-Q5b/Q5c）。

| Round | hypothesis | change | command | metric before → after | keep/revert | artifact |
|-|-|-|-|-|-|-|
| 1 | 新证明的 6 点物理反例可被独立复现，且与证明数值一致 | 新增 `experiments/sdp/run_6point_check.py`：在 x_R=[h;vec R] 坐标直接构造 3D-3D 二次型，构 10×10(21 约束) / 13×13 上游(24) / 13×13 满 20D(27) | `python sdp/run_6point_check.py` | f10 = **5.41500000**（证明 5.41500000）；f13_full = **5.41500000**；f13_up = **3.59431373**（证明 3.59430886，差 4.9e-6，SCS 级）；rank **1 / 1 / 4**；f10−f13up = **1.82068627**（证明 1.82069114）；exact nonconvex = 5.415 | keep | `sdp/run_6point_check.py`、`sdp/results/six_point.{json,md}` |
| 2 | 反例的严格间隙在 W/y 小扰动（含 D≠0）的邻域内保持 | 新增 `sdp/run_six_point_perturb.py`：一般 3D-3D 的 13×13 二次型构造 + 10 个扰动实例 | `python sdp/run_six_point_perturb.py` | 10/10：`|f10−f13full| = 0`（≤1e-9）；`f10−f13up` ∈ **[0.106, 1.821]** 全部严格正；rank10=1、rank13up=4–5 | keep | `sdp/run_six_point_perturb.py`、`sdp/results/six_point_perturb.{json,md}` |
| 3 | 一页图 D 面板的“13×13 strictly looser”已过时，应改为分层结论并加入 6 点反例 | 改 `code/make_summary_figure.py` 面板 D：10 实例 + 6 点星标；标题改为「upstream 13×13 is looser; full 20D = 10×10」 | `python code/make_summary_figure.py` | 图更新；6 点 (5.415, 3.594) 明确标注 | keep | `results/p1_summary.png` |
| 4 | `sdp_check.md` 的结论应改成修订版分层，避免与满 20D 等价冲突 | 改 `run_sdp_check.py` 结论段并重跑 | `python sdp/run_sdp_check.py` | 结论现明确「满 20D ≡ 10×10；上游 17D 严格更松；6 点反例 O(1) 间隙」；主表数值不变 | keep | `sdp/results/sdp_check.md` |
| 5 | f13_up 与证明引用的 3.59430886 差 4.9e-6，需判断是 SCS 容差还是构造差异 | `run_6point_check.py` 增加 CLARABEL 内点法独立交叉验证 + rank 阈值说明 | `python sdp/run_6point_check.py` | SCS 3.59431373 vs **CLARABEL 3.59431371**（<1e-9）⇒ 不是求解器容差；证明引用的末 5 位无法复现，但 1.82 间隙与 rank 1 vs >1 确认；rank13up 在 1e-5 阈值下 4、1e-6 下 6（~1e-5 小特征值） | keep | `sdp/results/six_point.{json,md}` |
| 6 | Q5c 的 Scorer 函数性质（整函数、S(0)、S′>0、ζ≪−1 渐近）可独立数值核验 | 独立 quad 计算（不依赖 scorer 子 agent 代码） | `python - <<...`（见 `rounds` 附注） | S(0)=**1.28789932** = Γ(1/3)/3^{2/3}；S′(ζ)>0 实测；S(−10)=0.0998 vs 1/|ζ|=0.1 | keep（支撑 §7.5） | 本轮日志 |
| 7 | 论文需按修订版 Q2/Q13 与 Q5b/Q5c 重写，且不夸大 Scorer 实现 | 并入 paper2 的 `draft.md`（§5.3.4 三段式 + Table 5.7 + Discussion 1/4 + §5.5），再修正两处：4.9e-6 不是求解器容差（SCS/CLARABEL 一致到 1e-9）；Q5b/Q5c 的指针改到 `theory/proofs.md` §7.5 | 复制 + 两处 edit | 论文 Q2/Q13 状态=「满 20D 等价 + 上游严格更松 + 物理反例」，Q5 明确 Scorer 为理论结果、6D 实现未做 | keep | `paper/draft.md`（+75/−32 + 2 处修正） |
| 8 | 复现 harness 应覆盖 6 点反例，且 clean-cwd/确定性结论更新 | 并入 repro2：`run_all.sh` 新增 `--with-sdp` / `--with-sdp-upstream`；重写 `REPRO.md`（8 脚本 clean-cwd、三次 harness PASS、6 点 1.5 s、确定性）；更新两份 README | `./run_all.sh [--with-sdp]` | 默认 35.1 s PASS；`--with-sdp` 39.0 s PASS；`--with-linemod --with-sdp` 43.0 s PASS；6 点脚本独立 1.50 s；已提交 results hash 不变 | keep | `run_all.sh`、`REPRO.md`、两份 README |
| 9 | 修订版 P1-2 + Q5b/Q5c 应进入 `theory/proofs.md` 并修正证据分级 | 并入 q2theory 的 `proofs.md`（§4.3 定理 P1-2 三段式 + 数值表 + 证明要点；§7.5 Q5b/Q5c；GAP-0.3-a 结案；差异表 #13–#15），再修 4.9e-6 的 solver 表述 | 复制 + 1 处 edit | Q2/Q13 从「GAP/被推翻」→「分层结案」；Q5 增加 P1-Q5b 必然跳变 + P1-Q5c Scorer C^∞ 延拓；证据标签区分 paper claim / 本仓库复现 | keep | `theory/proofs.md`（+117/−5） |
| 10 | Q2 一页图/数值证书应由 sdp2 子 agent 交付 | 子 agent 在 `wt-p1-sdp2` 运行 20+ min 无产物，按时间上限中断 | — | **blocked（无产物）**：Q2 可视化已由 Round 3 的 `p1_summary.png` 面板 D 覆盖；数值证书已由 Round 5 的 SCS/CLARABEL 交叉验证部分覆盖 | revert（无新文件） | — |
| 11 | P1-Q5c 的 6D Scorer 延拓可实际实现，并在 fold 处给出连续位姿 | 并入 scorer 子 agent：`theory/run_scorer_fold.py`（1D 三路 + 6D 复共轭根 Newton 追踪 + Scorer 权重 + 位姿跳变网格） | `cd theory && python run_scorer_fold.py`（~50 s） | **1D 复现**：0.86703/0.49595/0.33838→0.33834（证明 0.867/0.497/0.339113/0.339069，≤1.6e-4）；**6D**：$\theta_f=1.3185485e-3$、$c_3=2.55981$、复根追踪成功（$|\alpha'|$ 两侧一致 0.02%）；位姿最大跳变 **1.206e-2**（标准 3.041e-1、$\tau^{1/3}$ 1.068e-1），jump/step 6.0e2 vs 1.5e4/5.3e3；$K$ 恒 4 | keep | `theory/run_scorer_fold.py`、`theory/results/scorer_fold.{json,md,png}` |
| 12 | 论文/理论应写入 6D Scorer 实测而非“待实现” | 论文 §5.1 加 E14、§5.5 加 Table 5.14、改两处“待实现”为实测；`theory/proofs.md` §7.5 实现状态改为 6D 已实现；更新两份 README | 复制 + edit | 论文现报「Scorer 权重连续 + 位姿跳变 1.21e-2（标准 3.04e-1）」，并明确浅 fold/首阶规范形限制；理论 §7.5 同步 | keep | `paper/draft.md`、`theory/proofs.md`、`experiments/README.md`、`theory/README.md` |

## 并行子 agent（第二阶段 wave 1）

| Round | task | worktree | 状态 |
|-|-|-|-|
| P1 | 修订版 P1-2 定理 + Q5b/Q5c 写入 `theory/proofs.md` | `wt-p1-q2theory` (`opt/p1-q2theory`) | **done (Round 9)** |
| P2 | P1-Q5c Scorer/半线 Airy 延拓实现 + 6D fold 零跳变测试 | `wt-p1-scorer` (`opt/p1-scorer`) | **done (Round 11)** |
| P3 | 论文 Q2/Q13 修订 + 6 点表 + Q5b/Q5c | `wt-p1-paper2` (`opt/p1-paper2`) | **done (Round 7)** |
| P4 | Q2 一页图 + 6 点数值证书 | `wt-p1-sdp2` (`opt/p1-sdp2`) | **blocked (Round 10)** |
| P5 | repro harness / README 更新（6 点、Scorer、Q2 状态） | `wt-p1-repro2` (`opt/p1-repro2`) | **done (Round 8)** |

（子 agent 完成后并入主 worktree 并在此表更新为 done/kept/revert。）
