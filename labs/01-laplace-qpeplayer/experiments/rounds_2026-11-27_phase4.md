# P1 优化轮次日志 · 第四阶段（2026-11-27 晚，13 轮预算）

起点：HEAD `fb717d8`。约束：不改 `deliverables/`、`AGENTS.md`、顶层 `labs/EXPERIMENTS.md`；不 commit/push。
重点：真实 LINEMOD 的 ADD-S/位姿误差代理（hard/Laplace/MC，含 CI）、外部/经典基线（cv2 EPnP/SQPNP、Theseus、局部 GN/LM）、40 根快速计数、论文/复现一致性。

| Round | hypothesis | change | command | metric before → after | keep/revert | artifact |
|-|-|-|-|-|-|-|
| 1 | HEAD 已含 phase-3 的 Scorer 细化与论文/理论；真实 two-mode 结果可作为新误差表的基线 | 复核 HEAD 跟踪文件与 `real/linemod/results/two_mode.md` | — | HEAD `fb717d8`；15 帧 two-mode 表完整 | keep | 本轮日志 |
| 2 | 真实 LINEMOD 需要 ADD-S/位姿误差代理与 CI | 子 agent poseerr：`real/linemod/run_pose_error.py` + `results/pose_error.{json,md,png}`（hard/Laplace/MC，15 帧，bootstrap CI） | `python real/linemod/run_pose_error.py` | hard 中位 2.06° [1.47,3.49]、ADD-S 4.6 mm [4.0,5.9]；Laplace（**投影 chordal mean**）中位 **44.24°** [33.90,46.87]、ADD-S 15.1 mm；配对偏差 +36.9° [+27.2,+44.9]；MC 同 estimand、seed SD 1.011°；hard 失败 1/15。**修正**：旧 58.28° 是未投影 9-vector mean，不是 geodesic 误差 | keep | `real/linemod/run_pose_error.py`、`results/pose_error.*` |
| 3 | 经典/外部 PnP 基线可在同一真实对应上跑 | 子 agent classical：`run_classical_baselines.py`（cv2 5.0 EPnP/ITERATIVE/SQPNP/IPPE + scipy LM；sdpr env） | `sdpr/bin/python run_classical_baselines.py` | SQPNP 中位 2.04° / ADD-S 4.6 mm（15/15 真模式）；EPnP 2.05° 但 12/15；IPPE 2.21° / 5.5 mm；LM(identity) 106°（3/15）；LM(GT) 2.06°；重投影中位 1.47 px | keep | `run_classical_baselines.py`、`results/classical_baselines.*` |
| 4 | Theseus 可作为同一代价的外部局部基线（复用 P3） | 子 agent theseus：`run_theseus_baseline.py`（theseus venv；复用 P3 `theseus_cost/run_theseus` 模式） | `/mnt/storage1/syk/venvs/theseus/bin/python run_theseus_baseline.py` | 目标值一致 1.1e-13、Jacobian 1e-10；GT init 回最近 P1 模式；identity 1/15 GT；random 100/300 GT（33.3%）、57 flip、134 非物理；8-init multistart 11/40 GT。**Theseus 是局部优化器，无枚举/物理过滤；不是与 Laplace 同 estimand 的比较** | keep | `run_theseus_baseline.py`、`results/theseus_baseline.*` |
| 5 | 40 根计数应能用快速同伦在分钟级确认 | 子 agent roots2：`roots/run_rootcount_fast.py`（total-degree homotopy，4 个仿射 chart 取并） | `python roots/run_rootcount_fast.py`（26 s） | **4/4 实例 40 个复射影根**（80 仿射根），残差 ≤1.2e-13，0 发散路径；实射影 26/12/12/20、极小 5/2/2/4；**数值完整性（4 卡并饱和）而非符号商环上界** | keep | `roots/run_rootcount_fast.py`、`results/rootcount_fast.{json,md}` |
| 6 | 集成后脚本应可独立复现，且旧 58.3° 引用需在新文档中纠正 | 主 worktree 重跑 roots（40/40/40/40，counts 一致）；在 classical/theseus 文档加“58.28 是未投影值、有效值 44.24”注 | `python roots/run_rootcount_fast.py` | 重跑 26 s，4/4 counts 与子 agent 一致；两份基线文档已加注 | keep | 本轮日志 + 两处 note |
| 7 | 真实数据结果需要一页可视化（P1 vs 经典/局部基线，含 CI） | 新增 `real/linemod/make_real_summary.py`：读三个 JSON，画旋转误差与 ADD-S 中位数+CI | `python real/linemod/make_real_summary.py` | `results/real_summary.png`（P1 hard/Laplace/MC、SQPNP/IPPE/EPnP/LM(GT)、Theseus(GT)） | keep | `real/linemod/make_real_summary.py`、`results/real_summary.png` |
| 8 | pose_error 的 ADD-S 需用 dense GT mesh 作目标（避免 1000-vs-1000 的 +1.1 mm 离散地板）；旧副本需替换 | 用子 agent 最终版覆盖主 worktree 与 paper4 worktree 的 `pose_error.*`，重跑 `make_real_summary.py` | `python real/linemod/make_real_summary.py` | 主 worktree 现为最终值：hard ADD-S 中位 **3.7 mm [3.0,4.3]**、mean 6.0；Laplace 中位 **13.9 mm [10.3,20.2]**、mean 15.5；rot 44.24° | keep | `results/pose_error.*`、`results/real_summary.png` |
| 9 | Theseus 基线的说明/README 应并入，且其表里的 P1 Laplace 引用需改为 44.24° | 复制 `THESEUS_BASELINE.md`、`README.md` §7；把表行与注改为 projected 44.24°（58.28 作为未投影值保留说明） | — | Theseus 表：GT init 2.06°/15-15；identity 178.82°/1-15；random 100/300 GT；multistart 11/40；目标对账 1.1e-13 | keep | `THESEUS_BASELINE.md`、`real/linemod/README.md`、`results/theseus_baseline.md` |
| 10 | 论文需吸收 pose-error/ADD-S、经典/Theseus 基线、40 根，并把 58.3° 全面改为 44.24° | 并入 paper4 的 `draft.md`（+147/−42；Table 5.13/5.14/5.15/5.16/5.18；E15–E18；abstract/§5.2/§5.5/Discussion/Conclusion） | — | 全表 5.1–5.19 顺序、逐格核对；44.24° 为唯一有效 Laplace 旋转值，58.28° 仅作为“未投影 9-vector 值”出现在更正句 | keep | `paper/draft.md` |
| 11 | 结果文件中的旧 58.28 引用需加注，避免与论文冲突；论文数字检查需覆盖新值 | 在 `two_mode.md` 顶部加更正块、`classical_baselines.md` finding 1 改 44.24、`rootcount_fast.md` 路径改相对；`check_paper_numbers.py` 新增 4 条事实 | `python code/check_paper_numbers.py` | **20/20 patterns found**（含 44.24、ADD-S 3.7/13.9、40 roots） | keep | 上述 md + `code/check_paper_numbers.py` |
| 12 | 新脚本应纳入一键复现，且各组可从空 cwd 通过 | 并入 repro4 的 `run_all.sh`（`--with-roots/--with-pose-error/--with-classical/--with-theseus`，含 per-script 解释器与 preflight）；更新 REPRO/README | `./run_all.sh --with-roots --with-pose-error --with-classical` | 9 脚本全 rc=0，**64.648 s PASS**；roots 23.96 s、pose 6.64 s、classical 3.39 s；已提交 results hash 不变 | keep | `run_all.sh`、`REPRO.md`、`experiments/README.md`、`labs/01-laplace-qpeplayer/README.md` |
| 13 | `--with-theseus` 组需实测通过，且论文数字/文件最终一致 | 主 worktree 跑 `--with-theseus`；`check_paper_numbers.py` 20/20；核对论文 44.24/ADD-S/40 roots 与文件 | `./run_all.sh --with-theseus`；`python code/check_paper_numbers.py` | Theseus 组 **10.748 s**、总 40.079 s PASS；论文数字 **20/20**；所有新结果文件在 main | keep | `results/logs/20261127-223924/`、`code/check_paper_numbers.py` |

## 并行子 agent（phase-4 wave 1）

| Round | task | worktree | 状态 |
|-|-|-|-|
| P1 | 真实 LINEMOD ADD-S/位姿误差代理：hard/Laplace/MC + bootstrap CI | `wt-p1-poseerr` (`opt/p1-poseerr`) | running |
| P2 | 经典基线：cv2 EPnP/SQPNP/ITERATIVE + 局部 GN/LM（同真实对应） | `wt-p1-classical` (`opt/p1-classical`) | running |
| P3 | Theseus LM/GN 基线（复用 P3 的 theseus_cost/run_theseus 与 venv） | `wt-p1-theseus` (`opt/p1-theseus`) | running |
| P4 | 40 根快速计数：Macaulay/action-matrix/homotopy，15 min 时间盒 | `wt-p1-roots2` (`opt/p1-roots2`) | running |
