# P2 final push — SDPRLayers 线 accepted-round 账本（final/p2-sdpr）

工作树 `/mnt/storage1/syk/.superagent-work/wt-final-p2-sdpr`，分支 `final/p2-sdpr`，
无 commit/push。CPU 登录节点、`nice -n 10`、单线程 BLAS；GPU 分区开头 `squeue -p gpu`
查过一次（cyr/wyx 占用），未等待、未使用。预算 25 round，实际用 13。
所有数字均为实测；没有测量到的格子留空/标 `–`，不估计。

| # | unit | command | result (numbers) | artifact | accept |
|---|---|---|---|---|---|
| A1 | synth p=16 全变量，n=12 → **n=50**（seed 0–49，与 `raw_main.json` 配对） | `QPE_SDPR_P=16 QPE_TRIALS=50 nice -n 10 $PY code/run_sdpr_synth.py` | 50/50 完成，wall 16.1 s；unc 103.018/5.296/97.901 mm、0% feas、31 ms；hard 0.00/4.169/10.630、98%、33 ms；slack 0.00/4.169/10.630、98%、178 ms；cert = feas（relaxation tight）；PSD 30×30 | `results/baseline_sdpr_synth_p16.json`（旧 n=12 存档 `..._n12_archive.json`） | accept |
| A2 | LINEMOD 全量 **n=3 → n=90**，unc/unc_ref/hard/slack + 独立 Schur cert | `nice -n 10 $PY code/run_sdpr_bop.py lm` | 90/90 完成，wall 359.9 s；join published 90/90，`max|ΔJ|=0`；unc 11.473/2.191/36.023、12% feas、9% cert、198 ms；hard 9.188/2.136/35.947、14%、13%、675 ms；slack 2.401/2.496/38.272、42%、12%、3169 ms；unc_ref = published unc 11.261/2.080/35.634 精确复现 | `results/baseline_sdpr_lm.json`（旧 n=3 存档 `..._n3_archive.json`） | accept |
| A3 | LM-O 首次全量 **n=72** | `nice -n 10 $PY code/run_sdpr_bop.py lmo` | 72/72，wall 280.6 s；join 72/72 `max|ΔJ|=0`；unc 14.926/2.069/34.694、7%、7%、132 ms；hard 12.702/2.075/34.895、12%、7%、669 ms；slack 1.959/2.528/47.068、32%、3%、3150 ms | `results/baseline_sdpr_lmo.json` | accept |
| A4 | YCB-Video 首次全量 **n=84** | `nice -n 10 $PY code/run_sdpr_bop.py ycbv` | 84/84，wall 340.3 s；join 84/84 `max|ΔJ|=0`；unc 11.399/1.124/19.061、13%、6%、238 ms；hard 11.283/1.176/17.269、10%、6%、667 ms；slack 0.277/1.426/28.308、46%、2%、3185 ms | `results/baseline_sdpr_ycbv.json` | accept |
| A5 | slack 罚形式 M sweep，p=8/16 × M∈{1e2,1e3,1e4} × n=20 | `QPE_SWEEP_P=8,16 QPE_SWEEP_TRIALS=20 nice -n 10 $PY code/run_sdpr_sweep.py` | 6 格全 20/20 解出，wall 252 s；最佳 M=1e2：p=8 feas 90%、p=16 85%；feas 随 M **下降**（p=8: 90/40/20%；p=16: 85/65/20%）；M≥1e3 多数 `optimal_inaccurate`，value−hard 中位 +8.7e-7…+6.4e-5（本应 ≤0）；s_max ~1e-13 说明罚项在 SDP 最优点不激活 | `results/baseline_sdpr_sweep.{json,md}` | accept |
| A6 | slack PSD 块大小 + SCS 状态随 p 的普查（含直接读回 X） | `nice -n 10 $PY code/analyze_sdpr_status.py` | 直接 probe：X = **22×22 / 30×30 / 46×46 / 78×78**（p=8/16/32/64，=14+p），n_eq = 47/55/71/103（=39+p），n_ineq = 8/16/32/64（=p）；p=32 slack 4/50 行经 retry ladder 后 `optimal`（attempts=2），其余 196/200 `optimal`；真实数据 slack `optimal_inaccurate` 79/90、54/72、71/84；`slack≥hard` 序被违反 13/11/14 行（最多 5.7e-5） | `results/baseline_sdpr_status.{json,md}` | accept |
| A7 | 汇总：每 dataset / 每 p 一张表（pen/rot/trans/feasible/cert/time/tightness/rank） | `nice -n 10 $PY code/make_sdpr_results.py` | `baseline_sdpr_final.json`（synthetic 4 格 + real 3 格 + sweep + status）、`baseline_sdpr_final.md`（Table 1–7）、`fig_sdpr_baselines.png` 更新 | `results/baseline_sdpr_final.{json,md}` | accept |
| A8 | §5.7 替换正文：reimplementation vs official SDPRLayers（无官方 PyTorch 层、仅 forward、无 dual-certificate backward）+ 新数字 + Table 5.9b | 写作（无命令） | 英文正文 4 段 + 21 行 Table 5.9b（n=50/90/72/84）+ A5/A6 短段；明确列出未复现项 | `paper/round_paper_sdpr_final.md` §2 | accept |
| A9 | `results/final_push_summary.md` SDPR 节更新 | 写作（无命令） | 20 行 → 28 行（4 p × 4 variant + 3 dataset × 4 variant），n 全部标出；A5/A6 结论并入；关闭旧 blocker；记录 p=32 slack 100%→92% 的更正 | `results/final_push_summary.md` §1、§7 | accept |
| A10 | 每格精确命令与 n | 写作（无命令） | `baseline_sdpr_final.md` 的 "Exact commands, n, and provenance" 表：7 类 cell、n、seed/实例、原始 JSON、解释器与线程设置 | `results/baseline_sdpr_final.md` | accept |
| A11 | 统一代码代际：p=8/32/64 重跑 n=50（旧文件来自更早代码，p=32/64 缺 `solver_used`） | `QPE_SDPR_P=8,32,64 QPE_TRIALS=50 nice -n 10 $PY code/run_sdpr_synth.py` | p=8、p=64 与旧文件**逐位一致**（max|Δ|=0）；p=32 的 4 个 slack 行旧解为 `optimal_inaccurate` 且值非法（1 行低 2.7×，3 行高于自身提取位姿的 J），新解经 retry 后 `optimal`（attempts=2），slack feas/cert **100% → 92%** | `results/baseline_sdpr_synth_p{8,32,64}.json` | accept |
| A12 | 成本图（hard 14×14 vs slack 14+p） | 由 A7 的 `make_cost_figure` 生成 | medians：hard 30/33/42/59 ms，slack 63/178/532/2405 ms（p=8/16/32/64），PSD 22/30/46/78 标注；slack 在 p=16 起超过 published ALM pipeline（~277 ms 中位） | `results/fig_sdpr_cost.png` | accept |
| A13 | 真实数据 `slack<hard` 行的 tighter 重解（敏感性检查） | `SDPR_RECHECK_EPS=1e-10 SDPR_RECHECK_ITERS=200000 nice -n 10 $PY code/recheck_sdpr_real.py` | 见下（A13 完成后补） | `results/baseline_sdpr_recheck.{json,md}` | pending |

## 说明

- 一次 accepted round = 一个完成且实测的单元（一组表 / 一个修好并验证的 loader / 一节文字）。
  失败或重试不计：A2 之前的 4 实例计时 probe、A11 中的旧数据都没有计为 round。
- 所有 "certified" 一律用 sound 证书：`certificate.shor_bound(..., repair_dual=True,
  schur=True)` 的 `rigorous_schur`（fallback `rigorous`），不用 SCS 的 `value`。
- 耗时是共享登录节点上的 wall clock，受同时运行的 3 个真实数据 job 影响（A5），
  只作相对比较。
- 没有 commit/push；未触碰 `wt-final-p2` 及其他 sub-worktree、`deliverables/`、
  `AGENTS.md`、顶层 `labs/EXPERIMENTS.md`、`theory/`。
