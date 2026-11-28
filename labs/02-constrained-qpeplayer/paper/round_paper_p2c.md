# P2 论文线轮次记录（paper 子代理 P2C，2026-11-27 22:00–22:20）

> **任务**：把两条并行线的实测结果接进论文：严格真实证书（`wt-p2c-rigorous`）与根结构/16 根模板状态（`wt-p2c-roots`），并按测量修正 §4.2/§4.7/§5.4/§5.5/§5.6 与证明文档的口径。
> **分支/wt**：`opt/p2c-paper`，base HEAD `17d7ffc`，worktree `/mnt/storage1/syk/.superagent-work/wt-p2c-paper`。**不 commit/push**。
> **约束**：不改 `deliverables/`、`AGENTS.md`、顶层 `labs/EXPERIMENTS.md`、实验代码/结果 JSON；`theory/proofs.md` 只做**加性**编辑，不改既有定理语句。
> **硬规则**：0 个编造数字。每个数字可追到 `experiments/` 下的 JSON/MD；代数数字（$4/12/16/32/40$、Gröbner 阶梯）标为 "user-supplied proof"。
> **环境**：登录节点直接跑（`nice -n 10`），无 Slurm；计时只作相对比较。

## 轮次表

| Round | 假设 | 改动 | 命令 | 指标 before → after | 判定 | 产物 |
|---|---|---|---|---|---|---|
| **S1** | 严格真实证书（P2-Cert Schur）落地后，§5.5 的 one-sided/two-sided 两列不足以支撑"certified"，必须换成带 Wilson CI 的严格表；且 bound-RLT 在 sound 判据下可能是**正**结果（旧结论被 unsound value 污染） | §5.5 新增 `(v)` + **Table 5.11**（3 数据集 × base/zero-dim RLT/bound-RLT：Schur 与 $P$-shift 计数+CI、Schur gap min/med/max、naive value）；新增 `(vi)` + **Table 5.12**（LM-O 6 个最差行的 depth+tilt before→after）；§5.4 真实段、§5.5 Table 5.10 source、§5.6 bullet、状态行、Abstract、§6、§7 同步改口径 | `python -c` 读 `results/rigorous_real.json` 汇总；`P2_RIG_DIR=…python make_p2_overview.py`；`git diff --numstat` | 真实数据严格数 0 → **Table 5.11 9 行**：base Schur **22/90=24.4% [16.7,34.2]**、**11/72=15.3% [8.8,25.3]**、**11/84=13.1% [7.5,21.9]**；bound-RLT **46/90=51.1%**、**30/72=41.7%**、**20/84=23.8%**；paired McNemar $p=8.4\times10^{-6}/1.6\times10^{-4}/3.5\times10^{-2}$；zero-dim RLT 中性→负（22→19、11→6、11→6）；LM-O depth+tilt base 0/6 →+brlt 6/6 | keep | `paper/draft.md` §5.5(v)(vi)、§5.4、§5.6、状态行/Abstract/§6/§7 |
| **S2** | 证明文档需要把"canonical form 的 4/12/16 已数值确认、但 ALM 组装到 $\Sigma_{\mathcal A}$ 的映射未对齐"写成可审计的一节；roots 子代理落盘后还必须记录"12 vs 16 是 census 伪影"的判定 | `theory/proofs.md` **加性**新增 `### 2.6.5`（canonical form 表 4/12/16/16、Type I 40/80、$\lambda$ 平移 $2\alpha$、$\rho$-only 组装 12–24 的开放点）与 `### 2.6.6`（roots 测量：canonical 4/12/16/16；ALM 组装 corrected 4/12/16/16/16；$k=2$ 的 12 是 `div_tol` census 伪影；双法向 1+1→32/64、2+2→40/80；$F_n$ 残差判别器；16×16 模板**未**实现） | `grep -n '^#### 2.6' theory/proofs.md`；`git diff --numstat -- theory/proofs.md` | proofs.md 643 → 702 行；**+59 / −0**；既有定理陈述 0 改动；新增两张数值表 + 15 行结论 | keep | `theory/proofs.md` §2.6.5（345–372 行）、§2.6.6（373–402 行） |
| **S3** | §4.7/§4.2 的"ships two templates"是过度声称：仓库只 ship 40 根模板；16 根饱和模板只有定理代数陈述 + canonical 数值确认，没有可发货数值模板 | §4.7 "Implementation consequence" 重写为**按实现状态分级**；§5.3 "Which inner root count applies" 末段删掉"two templates"；§4.7 新增 corrected-census 段；统一口径为 "40-root implemented+validated；4/12/16 spectrum 数值确认；16-root saturated template **partial/absent**" | `git show 17d7ffc:…/draft.md \| grep -c 'two templates'`；`grep -c` 现文件 | "two templates" **1 → 0**；新增 status 句 1 处；16 根模板明确标 partial/absent（不再说 offline symbolic reference） | keep | `paper/draft.md` §4.7（316–331）、§5.3 末段 |
| **S4** | exact-inner 的 37–64× 是旧枚举器测的；优化后枚举器 2.92× 更快，但不能把 37–64× 直接除以 2.92（未端到端重测） | §5.6 exact-inner bullet 增一段"Enumeration-cost update"：0.488 s → 0.168 s（2.92×）、356→1077 paths/s、15 case 根数逐项相同、$\max\|\Delta f\|=3.3\times10^{-16}$、批评估次数不变（median 470）；**明确不重标定** 37–64×；记录被否决的 `div_tol` 早剔除 | `python -c` 读 `results/perf_quartic.json` 的 `summary`；`grep -n 'Enumeration-cost update'` | 37–64×（median 47×）保留；新增 2.92× 段落 1 段；`summary.speedup_med=2.923`、`max_abs_df_all=3.33e-16`、`n_parity=15` 均逐项核对 | keep | `paper/draft.md` §5.6 exact-inner bullet |
| **S5** | 图 5.1 panel (c) 只有 one-sided/two-sided + pending 第三柱；需要 one-sided(upper bound)/two-sided(necessary)/rigorous base/+bound-RLT 四柱 + Wilson CI，并把数据源改成严格 JSON | 重写 `make_p2_overview.py` panel (c)：`_load_rig()` 支持 merged `rigorous_real.json` 或 per-dataset `rigorous_{lm,lmo,ycbv}.json`；`P2_RIG_DIR` 环境变量允许从并行 worktree 读取；四柱 + CI + k/n 标签；panel (a) 图例移到轴下避免遮挡 100% 柱；script docstring 记录确定性与用法 | `P2_RIG_DIR=… nice -n 10 python make_p2_overview.py`（连跑两次比对 md5） | PNG 223423 → **238506 bytes**；panel (c) bars 2+1(pending) → **4**（22/90、19/90、17/90、46/90 等）；两次运行 md5 完全一致（`8b59e3b…`）；merged 与 per-dataset 两条数据路径产出同一 md5 | keep | `experiments/results/fig_p2_overview.png`、`make_p2_overview.py` |
| **S6** | 一致性审计能抓出残留 TODO、旧 50/22/83、"rigorous rerun pending"、缺 source 的表 | 修 §5.5(iv) 旧三元组、§5.4 真实段结尾、状态行、Abstract、§7 "tightness rate on real data 仍开放"；Table 5.10 source 改为指向 Table 5.11；逐表核对 source line；§5.6 增 rigorous real bullet | `grep -nE 'TODO\|TBD\|待补\|FIXME'`；`grep -c '^\*\*Table 5\.'` vs `grep -c 'Source:'`；`python -c` 复算 Wilson CI 与 raw.json 计数 | TODO/TBD **0**；13 个数据表 **全部有 source line**（19 处 Source）；旧 `50%/22%/83%` **0**（只留 "earlier $n=18/12$ superseded"）；§5.5 Wilson CI 复算一致；真实数据 one-sided 30/22/28 与 two-sided 17/9/8 由 raw.json 复算一致 | keep | 本文件 + `paper/draft.md` 审计修正 |

## S1 的严格数字与来源（供核对）

**Table 5.11（Schur 为主界，$P$-shift 为保守回退；$n=90/72/84$）**

| dataset | cuts | Schur cert (95% CI) | dual-repair | naive(value) | Schur gap min/med/max |
|---|---|---|---|---|---|
| LM | base | 22/90 = 24.4% [16.7,34.2] | 23 (25.6%) | 25 | 3.98e-12 / 9.11e-6 / 3.99e-4 |
| LM | +rlt | 19/90 = 21.1% [14.0,30.6] | 18 (20.0%) | 35 | 9.26e-12 / 1.64e-5 / 2.27e-1 |
| LM | +brlt | **46/90 = 51.1% [41.0,61.2]** | 43 (47.8%) | 61 | 3.99e-12 / 3.37e-7 / 2.77e-1 |
| LM-O | base | 11/72 = 15.3% [8.8,25.3] | 11 (15.3%) | 19 | 1.65e-11 / 1.55e-5 / 2.56e-4 |
| LM-O | +rlt | 6/72 = 8.3% [3.9,17.0] | 6 (8.3%) | 19 | 3.18e-11 / 6.40e-5 / 3.28 |
| LM-O | +brlt | **30/72 = 41.7% [31.0,53.2]** | 24 (33.3%) | 45 | 5.60e-12 / 2.13e-6 / 9.37e-2 |
| YCB-V | base | 11/84 = 13.1% [7.5,21.9] | 10 (11.9%) | 17 | 1.00e-11 / 2.10e-5 / 4.02e-4 |
| YCB-V | +rlt | 6/84 = 7.1% [3.3,14.7] | 6 (7.1%) | 11 | 1.00e-11 / 3.81e-5 / 1.07 |
| YCB-V | +brlt | **20/84 = 23.8% [16.0,33.9]** | 19 (22.6%) | 31 | 4.10e-12 / 1.25e-5 / 3.01 |

- Soundness：`Schur>J` 0、`dual>J` 0、`Schur>value` 0、invalid shift 0、NaN 0、`B_noRt` 0；**738 行全部 case A**（$\tilde{\mathbf H}_{tt}\succ0$，max cond 1668）；ALM 可行 90/90、72/72、84/84，max violation 8.6e-11–9.1e-11；replay 的 ALM $J$ 与 `raw.json` **bit-identical**（$\max|\Delta J|=0$）。
- Paired（Schur flags）：LM base→brlt 22→46（27 up / 3 down，$p=8.4\times10^{-6}$）；LM-O 11→30（22/3，$p=1.57\times10^{-4}$）；YCB 11→20（12/3，$p=0.035$）。+rlt：22→19（$p=0.45$）、11→6（$p=0.0625$）、11→6（$p=0.0625$）。
- **口径修正（本轮最重要）**：旧稿把"RLT 在真实数据上是负结果"写死；严格界下应分成两句——**zero-dim RLT 是负/中性**；**sound bound-RLT 是正结果**（+26.7/+26.4/+10.7 pp）。旧结论来自 unsound 的 solver value（zero-dim RLT 在 22/72 行上把 value 抬到 $J$ 之上）。
- **LM-O depth+tilt（Table 5.12）**：base Schur 未认证 61/72，取 Schur gap 最大的 6 行（全部 frame 000002，obj 5/8/12，bias 20/30 px）。加 4 条 positive-depth + 1 条 upright cone（$\alpha=12^\circ$，$\cos\alpha=0.978$，$n_{\rm ineq}=16\to21$）：**aug+base 仍 0/6**（gap 9.9e-5–2.5e-4），**aug+brlt 6/6**（gap $9.0\times10^{-12}$–$9.9\times10^{-11}$），aug+full RLT 6/6。**不带增广的 brlt 只认证 1/6**（另一 5 行 gap 3.5e-5–2.0e-4），所以是 cross-direction 乘积在起作用。增广不改变答案：$|\Delta J|\le7.4\times10^{-15}$，ALM 位姿已满足新增行（depth 0.85–0.88 m $\gg z_{\min}=0.02$；upright 0.998–0.999 $\ge\cos12^\circ$）——但证书只对**增广后的物理问题**成立。

## S2 的 roots 测量（供核对）

`results/isotropic_spectrum.{json,md}`（`n_gamma=2`，`norm_cap=1e5`，单线程 BLAS）：

- canonical：rank Σ=0/1/2/3（seed 0–4）→ **4/12/16/16 复射影（8/24/32/32 仿射）**，`n_lost=0`，$F$-residual 1.0e-15–2.6e-15。
- ALM 组装（`isotropic.alm_reduced`，Horn + 同法向 $k$）：committed census 在 $k=2$ 读 12（seed 0）/16/14；**corrected census 全部 16/32**。原因：4 个额外复根在 $\|q\|\approx29$–34，跟踪越过 `div_tol=1e4` 判 diverged，committed 只对 status∈{1,2,4} 做 endgame polish。KKT 残差 ~1e-16 ⇒ 真解。
- 双法向：**1+1（$k=2$）→ 32/64**（定理）；**2+2（$k=4$）→ 40/80**（说明定理的 32 是"每条法向一条活跃约束"）。
- Type I：aniso / PnP → 40/80；到 $F_n$ 的残差 7e-2 / 6e-1（数量级拒绝），可作结构判别器。
- **16×16 饱和作用矩阵仍未实现为可发货数值模板**；`solve_isotropic_plane` 仍是 full enumeration + endgame polish + 64-start 交叉验证。

## 并行依赖与产物位置（本轮）

| 产物 | 实际位置（本 worktree 尚无） | 状态 |
|---|---|---|
| `results/rigorous_real.{json,md}`（merged，2026-11-27 22:12） | `wt-p2c-rigorous/labs/02-constrained-qpeplayer/experiments/results/` | 已落盘并读取；§5.5/§5.6 引用 |
| `results/rigorous_{lm,lmo,ycbv}.json`、`rigorous_lmo_depth_tilt.json` | 同上 | 已落盘；Table 5.11/5.12 的直接来源 |
| `results/isotropic_spectrum.{json,md}`（22:13） | `wt-p2c-roots/…/experiments/results/` | 已落盘；§2.6.6/§4.7 引用 |
| `results/p2b_schur_check.{json,md}`（Table 5.8b） | **主 worktree** `three-paper-tracks-…/labs/02-constrained-qpeplayer/experiments/results/` | 主 worker 产物；本 worktree 尚未合入，引用后需 **merge 后复核路径存在** |

> **图的复现**：本 worktree 的 `results/` 里还没有严格 JSON，所以本轮 PNG 用
> `P2_RIG_DIR=/…/wt-p2c-rigorous/…/experiments/results nice -n 10 python make_p2_overview.py` 生成；
> merge 之后普通 `python make_p2_overview.py` 会读 `results/rigorous_real.json`，产出**同一 md5**（已交叉验证 merged 与 per-dataset 两条路径同 md5）。

## 未能核验 / 需要 merge 后复核的数字

1. **Table 5.8b（`p2b_schur_check.*`）**：6 行合成对照（Schur gap 5.6e-12–4.2e-10 vs dual $2\times10^{-8}$，一行 $-6.7\times10^{-13}$）来自主 worktree 的主 worker 产物，本 worktree 无该文件；数字逐项读过主 worktree 的 JSON/MD，但**路径要在 merge 后才存在于本分支**。
2. **`rigorous_real.*` / `isotropic_spectrum.*`**：同上，位于并行分支；本文件与正文按"merge 后存在"引用。
3. **枚举器 0.488 s vs exact_alm 的 649 ms/call**：两个测量语境不同（perf_quartic 是单次 `critical_points`，exact_alm 含 polish/fallback 的每次枚举均值），正文两者都按各自来源引用，**没有**互相换算。
4. **用户证明的代数数字**（$4/12/16/32/40$、15/12/17 元 Gröbner、多余交度 36/28/24/8/0、$16\times16$ 作用矩阵）：仅有 Feishu 证明，本仓库未复跑 Rabinowitsch/$\mathbb F_{32003}$；正文标 "user-supplied proof"。
5. **§5.1.5 prototype sanity 数字**（5.57e-1 / 3.55e-15 等）：`verify/` 旧轮产物，本轮未重跑，正文已标 "prototype sanity check"。
6. **真实数据 RLT 的 unsound 计数**（21/90、22/72、15/84，max 1.13e-3/7.42e-2/4.91e-2）：已由 `raw.json` 的 `shor_rlt_minus_J` 复算核对一致（`>0` 的行数与最大值）。

## 最终状态

- `paper/draft.md`：655 → **716 行**（+73 / −12）；新增 Table 5.11（9 行）、Table 5.12（6 行）、§5.5(v)(vi)、§4.7 corrected-census 段、Table 5.8b、§5.6 enumeration-cost 段；状态行/Abstract/§6/§7 改为严格证书口径。
- `theory/proofs.md`：643 → **702 行**（+59 / −0）；新增 §2.6.5、§2.6.6；既有定理语句未改。
- `make_p2_overview.py`：panel (c) 四柱 + Wilson CI + merged/per-dataset 双数据源 + `P2_RIG_DIR`；panel (a) 图例移出轴外；两次运行 md5 相同。
- `fig_p2_overview.png`：223423 → **238506 bytes**。
- **仍未做**（与上轮一致，不属本轮范围）：learned baselines（SDPRLayers/Theseus）、end-to-end T1/T2/T3、backward/latency 分解、16 根饱和模板的数值实现。
