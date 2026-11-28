# P2 论文线轮次记录（paper 子代理，2026-11-27）

> **依据**：用户新证明（Feishu 文档 `RhzDwI1RKinQsekHDhUcjOVenHL`）定理 **P2-1b**（复/实根结构二分 + $4/12/16/32/40$ 谱系）与定理 **P2-Cert**（精确平移 Schur 补的 `repair_dual` 严格下界）。
> **约束**：不 commit/push；不改 `deliverables/`、`AGENTS.md`、顶层 `labs/EXPERIMENTS.md`、实验代码/结果 JSON。本文件只记录 paper 线（S25–S31，7 轮）。
> **硬规则**：0 个编造数字。每个数字要么可追到 `experiments/` 下的 JSON/MD，要么在文中显式标注 "user-supplied proof, 2026-11-27"（P2-1b/P2-Cert 的代数数字）。
> **环境**：Slurm `cpu` 分区被其他用户占满，本轮所有命令都在登录节点直接跑（`nice -n 10`），无 `srun`/`sbatch`；计时只作相对比较。

## 轮次表

| Round | 假设 | 改动 | 命令 | 指标 before → after | 判定 | 产物 |
|---|---|---|---|---|---|---|
| S25 | 用户证明 P2-1b 可以在不触碰既有定理的前提下独立成节，且"秩 9 仍 40 根 / 各向同性+单平面坍缩到 4/12/16"能写成可审计的二分 | `theory/proofs.md` 新增 `## 2A` 与 `### 2.6 定理 P2-1b`：陈述（类型 I/II）、$4/12/16/32/40$ 谱系表、证明梗概（平移不变 + Hopf 直纹线参数化）、Rabinowitsch/$\mathbb F_{32003}$ 验证说明、A2 修正、本仓库数值对照（`quartic_validation`） | `git diff --numstat -- labs/02-constrained-qpeplayer/theory/proofs.md`；`grep -n '2.6 定理 P2-1b'` | proofs.md 427 → 643 行；**新增 216 行 / 删除 0 行**；既有 §1–§3 定理语句 0 改动 | keep | `theory/proofs.md` §2.6 |
| S26 | P2-Cert 的两种情形（$H_{tt}\succ0$ 的 Schur 补 / 非 PD 的 $R_t$ 球界）与 Rump 检验可以写成一条可实现的定理，并直接替换 §4.4 的"数值 sound"叙事 | `theory/proofs.md` 新增 `### 2.7 定理 P2-Cert`（F 折叠、$\tilde\mu/\tilde\Gamma$ 投影、$d_{\rm sound}$ 两式、证明、Rump 条件、LM-O 共线诊断）；`paper/draft.md` §4.4 增 "A rigorous dual repair" 段；§4.6 G1/G3 状态更新；新增 §4.7 根结构小节 | `grep -n 'P2-Cert\|P2-1b' paper/draft.md`；`git diff --numstat -- paper/draft.md` | draft 619 → 655 行；P2-Cert 在 §4.4/§4.6/§6 共 5 处引用；"numerical soundness still open" 的表述改为"soundness 已由 repair 解决、tightness 仍经验" | keep | `theory/proofs.md` §2.7；`paper/draft.md` §4.4/§4.6/§4.7 |
| S27 | (a) §5 需要解释"为什么合成主表能用 40 根模板、各向同性+单平面要用 16 根饱和模板"；(b) §5.5 的真实证书率必须换成严格 `repair_dual` 数字，否则只能标 upper bound | §5.3 末尾新增 "Which inner root count applies (Theorem P2-1b)"；§5.4 的真实数据段由 $n=18/12$ 改为 $n=90/72/84$ 的 base/+RLT 双侧数字；§5.5 表保留 one-sided（显式 upper bound）与 two-sided，rigorous 列/说明待 `rigorous_real.json` | `grep -n 'n=18\|n=90\|n=72\|n=84' paper/draft.md` | 删除 §5.4 遗留的 $n=18$ 段落 1 处；新增谱系段 1 段；§5.5 表仍为 4 方法 × 3 数据集，cert 列标注 upper bound / necessary；rigorous 数字见下方"并行依赖" | keep（若 `rigorous_real` 不出现，按 pending 收口） | `paper/draft.md` §5.3/§5.4/§5.5 |
| S28 | exact-inner ALM、SCP baseline、sound 高阶 RLT 已经测出，§5.6 应从"全 pending"改成"measured + pending"两段 | §5.6 重写为 "Measured baselines and what is still pending"：新增 exact ALM（`exact_alm.json`）、SCP（`scp_baseline.md`）、sound bound-RLT（`round_rlt.md`）、YCB RLT 四条 measured；pending 只留 learned baselines、end-to-end、backward、real scale-up | `grep -c '^\* \*\*' paper/draft.md`；读 `exact_alm.json` | §5.6 measured bullets 0 → 4；pending bullets 7 → 4；exact ALM：8 trials，$\max|\Delta J|=1.8\times10^{-12}$，中位 $47\times$ 慢；SCP：98% vs ALM 100%（主表），85% vs 100%（压力档） | keep | `paper/draft.md` §5.6 |
| S29 | 状态行、abstract、§6 还有 "protocol / we will report" 的旧口径，且 v0.2 已不匹配实测范围 | status v0.2 → v0.3；abstract 末句由"full experimental protocol"改为 BOP 实测描述；§6 "We will report the rate honestly" 改为现在时并补 P2-Cert 的 soundness 区分；conclusion 明确 P2-1b/P2-Cert 已结案、G2 仍开放 | `grep -n 'We will\|we will\|v0.2\|full experimental protocol' paper/draft.md` | "We will/we will" 1 → 0；"full experimental protocol" 1 → 0；status v0.2 → v0.3 | keep | `paper/draft.md` 状态行/Abstract/§6/§7 |
| S30 | 图 5.1 的 panel (c) 仍用 $n=18/12$ 的 `real_cert_ci_baseline.json`，与正文 $n=90/72/84$ 不一致；且需要给严格证书留出第三柱 | `make_p2_overview.py`：panel (c) 改为直接从 `real/*/results/raw.json` 计算 one-sided/two-sided 与 Wilson CI，新增 rigorous 第三柱（读 `results/rigorous_real.json`，缺失时画斜线 pending）；重生成 PNG | `nice -n 10 /home/syk/miniconda3/bin/python make_p2_overview.py` | panel (c) 数据源 `real_cert_ci_baseline.json`($n=18/12$) → `real/*/results/raw.json`($n=90/72/84$)；bars 2 → 3（第三柱 pending/rigorous）；PNG 204052 → 223423 bytes | keep | `fig_p2_overview.png`、`make_p2_overview.py` |
| S31 | 一致性审计能抓出残留的 $n=18/12$、旧 50%/22%/83%、"entirely pending" 与缺 source 的表 | 修 §5.1.4 $n=18/18/12$ → 90/72/84 与 30/24/28 poses；修 §5.4 遗留的 $n=18$ 段；§5.5(iv) 负 gap 比例 19–27% → 18–26%（$16/90,17/72,22/84$）；表源覆盖检查；`grep` 全量 TODO/TBD/旧数字 | `grep -nE 'TODO|TBD|待填|n=18|50%|22%|83%|entirely pending' paper/draft.md theory/proofs.md`；`grep -c '^\*\*Table'` / `grep -c 'Source:'` | TODO/TBD/待填 0 → 0；旧 $n=18/12$ 段 2 → 1（仅保留"earlier n=18/12 superseded"说明）；11 个数据/指标表 11 有 source（Table 1 为定位表，不需数据源） | keep | `paper/draft.md` 审计修正 |

## 并行依赖（`wt-p2b-rigorous`）

- 目标文件：`/mnt/storage1/syk/.superagent-work/wt-p2b-rigorous/labs/02-constrained-qpeplayer/experiments/results/rigorous_real.{json,md}`。
- 截至本文件最后一次更新（2026-11-27 21:5x）该文件**未出现**；`wt-p2b-rigorous` 仍在改 `code/certificate.py` / `real/bop/run_bop.py`，没有 `real/*/results/rigorous/rigorous_rows.json`。
- 因此 §5.5 的严格证书率与 §5.6 的 "rigorous real" bullet、图 5.1 panel (c) 第三柱均按 **pending** 处理；§5.5 的 one-sided 明确标为 upper bound、two-sided 标为 necessary-only。若该文件在收口前出现，用 `results/rigorous_real.json` 替换/补齐并在本表 S27 行记实际数字。

## 最终状态

- `theory/proofs.md`：+216 行 / 0 删除，新增 §2A（§2.6 P2-1b、§2.7 P2-Cert）；既有定理/证明未改。
- `paper/draft.md`：v0.3；§4.4 引 P2-Cert、§4.7 引 P2-1b、§5.3/§5.5/§5.6 更新；所有数据表有 source line。
- `fig_p2_overview.png`：panel (c) 已对齐 $n=90/72/84$；rigorous 第三柱在 JSON 缺失时为显式 pending。
- 未修/未做：learned baselines（SDPRLayers/Theseus）、end-to-end T1/T2/T3、backward/latency 分解、真实数据严格证书（依赖并行子代理）。
