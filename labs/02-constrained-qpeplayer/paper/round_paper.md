# P2 paper round report — §5 Experiments 与 P2 总览图

日期：2026-11-27 · worktree：`/mnt/storage1/syk/.superagent-work/wt-p2-paper`（branch `opt/p2-paper`）
角色：paper-experiment track 单轮 validation subagent。未 commit、未 push、未改其他 worktree。

## 1. Round row

| hypothesis | change | command | metric（placeholders before→after / tables / figure） | keep/revert | artifact |
|---|---|---|---|---|---|
| 现有结果 JSON 足以填满 §5 的 TODO：合成主表、p 饱和、噪声/偏差、证书紧性 + RLT、真实 BOP 数据都能在不造数的前提下写成可追溯的论文正文；真实数据应按 fail-safe / 条件认证写，不能写 100%。 | 重写 `paper/draft.md` §5.1–§5.6；保留并降级 §5.1.5 prototype sanity check；新增 3 面板总览图 + 生成脚本；同步状态行与 §4.5 两处交叉引用（原指 §5.3(i)，改为 §5.6 pending）。 | `/home/syk/miniconda3/bin/python experiments/results/make_p2_overview.py`；`/home/syk/miniconda3/bin/python /tmp/audit_tables.py`（逐格回算 raw JSON）；`grep -n 'TODO\|TBD\|FIXME' paper/draft.md` | §5 的 TODO 占位：1 个 `### 5.2 ... (**TODO** results)` + 状态行“results marked TODO” → **0**（全 draft grep 为 0）；§5 表格：1 张（sanity）→ **11 张**（编号表 5.1–5.10 + sanity 表 1）；图：0 → **1 张 3 面板**；表格逐格回算 mismatch **0**；真实 ALM 认证（存储 flag）50% / 22% / 83%（上界，见 §9）。 | **keep** | `paper/draft.md` §5（235+/25−）；`experiments/results/fig_p2_overview.png`；`experiments/results/make_p2_overview.py`；本文件 |

## 2. Files changed

`git diff --stat`（tracked）：

```
 labs/02-constrained-qpeplayer/paper/draft.md | 260 ++++++++++++++++++++++++---
 1 file changed, 235 insertions(+), 25 deletions(-)
```

新增（untracked，需一并提交）：

| 文件 | 归属 | 说明 |
|---|---|---|
| `experiments/results/fig_p2_overview.png` | 本轮生成 | 3 面板总览图（feasible/cert vs p；penetration median[IQR] vs p；真实 ALM 认证 base vs +RLT + 95% Wilson CI） |
| `experiments/results/make_p2_overview.py` | 本轮生成 | 图脚本；所有数值从 raw JSON 现算（panel c 读 `real_cert_ci_baseline.json`），无手写数字 |
| `experiments/results/cert_margins.{md,json}` | 主 worker 中途提供 | 400 个合成实例的 λ₂/λ₁ 与 sound gap 裕度；本轮引用，未修改 |
| `experiments/results/real_cert_ci_baseline.{md,json}` | 主 worker 中途提供 | 真实认证率 Wilson 95% CI；本轮引用，未修改 |
| `experiments/results/inner_baseline.{md,json}` | 主 worker 中途提供 | 内层 multi-start BFGS 起点数基线；本轮引用，未修改 |

未触碰：`theory/`、`verify/`、`labs/EXPERIMENTS.md`、`AGENTS.md`、`deliverables/`、任何实验代码/结果 JSON（只读）。

## 3. Placeholders before → after

| 位置 | before | after |
|---|---|---|
| `draft.md` 状态行 | “Section 5 gives the full experimental design with results marked **TODO**. The only numerical results reported are the synthetic prototype sanity checks...” | 改为 v0.2：§5 已包含合成/噪声/证书/真实 BOP 的测量结果，pending 项在 §5.6 明确列出 |
| `### 5.2 Datasets and tasks (**TODO** results)` | 只有 T1/T2/T3 设计，无结果 | 删除该设计小节；§5.2–§5.5 为测量结果，§5.6 为 pending 清单 |
| `### 5.1 Prototype sanity checks` | §5 的唯一小节 | 保留为 §5.1.5（明确“not paper results”），其数字逐字保留、本轮未重测 |
| `### 5.3 Baselines` / `5.4 Measurements...` / `5.5 Compute budget and risks` / `5.6 Reproducibility` | 设计/计划，含未测量的 GPU 预算数字（60/30/10/20 GPU·h） | 删除；未测量的计划数字不进论文。可复现性改为 §5.1.5/§5.6 的路径说明 |
| §4.5 两处 `(§5.3 (i))` | 指向已不存在的 GN-vs-exact 消融小节 | 改为 `(§5.6)` pending ablation（仅改交叉引用，不改 §4 内容） |

## 4. Numbers added and sources

所有表格逐格从 raw JSON 回算（见 §5 的一致性检查）。论文中每个表下都有一行 `source:`。

| 论文表 | 内容 | 来源 |
|---|---|---|
| Table 5.1 | p 的约束构成（scalar rows）p=8/16/32/64 | `results/raw_main.json` `config.split` + `recs[].actypes`（FoV 按 pair 记录，1/3/9 pair = 2/6/18 rows） |
| Table 5.2 | 指标定义（penetration / feasible / certified / runtime） | `experiments/protocol.md` §3 |
| Table 5.3 | 合成主表：penetration median [IQR] + rate | `results/raw_main.json`（50 trials/cell，与 `main_table.md/json` 一致） |
| Table 5.4 | 合成主表：rot/trans median [IQR] | 同上 |
| Table 5.5 | feasible / certified / runtime；14×14 SDP tight + time | 同上 |
| Table 5.6 | p=16 的 4 个 σ/bias cell × 4 方法 | 同上（`agg_noise`） |
| Table 5.7 | soft-penalty Pareto（5 个 λ） | 同上（`pareto`；与 main soft λ=1e3 是**独立 50-trial sweep**，已在正文注明） |
| Table 5.8 | 400 个合成实例的证书裕度（λ₂/λ₁、Shor−J_ALM、SDP time） | `results/cert_margins.json`（由 raw_main 汇总） |
| Table 5.9 | 14×14 Shor vs 0-dim RLT 消融（4 噪声档 × 4 seeds） | `experiments/theory/results/rlt.json` |
| Table 5.10 | 真实 LM/LM-O/YCB 四方法主表 | `experiments/real/*/results/raw.json`（与三个 `main_table.md` 一致）；CI 来自 `results/real_cert_ci_baseline.json` |

正文新增的派生数字（均可回算）：

- p 饱和：ALM 逐 seed 位姿跨 p 最大变化 5.9e-7° / 5.6e-7 mm；projection 1.7e-5° / 4.1e-5 mm；ALM active set median 2→4.5、max 33（`raw_main.json`）。
- runtime log–log slope：ALM median 0.045、mean 0.031（`raw_main.json`）。
- soft penalty 的 1/50 坏局部解：p16 156.5°、p32 171.6°、p64 84.8°、trans 最大 633.6 mm（`raw_main.json`，seed 4）。
- RLT：σ=0.10 base 1/4→RLT 2/4；σ=0.20 0/4→0/4，gap 10–17%（`rlt.json`）。
- 真实认证严格化：双侧 `|J−Shor|≤1e-6` 下 ALM 7/18 (LM)、1/18 (LM-O)、9/12 (YCB)；+RLT 5/18、0/18（`real/*/raw.json`，本轮计算）。
- 真实 RLT 的负 gap：5/18 (LM)、7/18 (LM-O)，最大 −3.7e-4 / −4.3e-4；两个 bound 崩塌（gap 1.49e-1 / 6.2e-2）（`real/*/raw.json`）。

内层求解器基线（`inner_baseline.json`）：4 starts 在 8/24 个 unconstrained 矩阵上欠解（max gap 1.17e-3）；8 starts 在全部 48 个实例达到 reference（median 26 ms）；256 starts 678 ms（25×）无收益；ALM active-set 矩阵 4 starts 已全部达到 reference（24/24）。

## 5. Consistency check

- 脚本：`/tmp/audit_tables.py`（本轮临时；未入库）。它解析 draft 的 Table 5.3–5.7、5.10，按论文同样的格式从 raw JSON 重算每个单元格的 median/IQR/rate/median-time，逐格字符串比较。
- 结果：**mismatch 0**。Table 5.1/5.8/5.9 的数值来自已汇总的 JSON/MD，另行逐项人工核对。
- 图脚本：`make_p2_overview.py` 从 raw JSON 现算 panel a/b、从 `real_cert_ci_baseline.json` 读 panel c，不手写数字。
- `grep -n 'TODO\|TBD\|FIXME' paper/draft.md` → 无输出。

## 6. Discrepancies / stale numbers found

### 6.1 旧 draft 的状态性 stale

1. 状态行称“Section 5 ... results marked TODO / 只有 verify sanity check 数字”，且 §5.2 标题为 `(**TODO** results)`。这是本轮前的真实状态；已更新。
2. §4.5 两处交叉引用 `§5.3 (i)` 指向一个从未写过、且本轮也不存在的 GN-vs-exact 消融小节。已改为 `§5.6` pending，避免坏引用。

### 6.2 结果/README 文件中的数字不一致（未改源文件，只在论文里用一致值并在此记录）

3. **`experiments/real/README.md` 的跨数据集表与 raw/`main_table.md` 不一致**：
   - LM-O unconstrained penetration：README **23.06 mm** → `main_table.md` 与 raw 都是 **23.08 mm**。
   - LM-O soft-penalty feasible：README **89%** → `main_table.md` 与 raw 是 **83%**（15/18）。
   - README 说 LM-O trans “29–63 mm”；实际四法 median 是 **30.63 / 35.14 / 46.51 / 48.91 mm**（无 63）。
   - YCB trial 构成：README 写“2 帧 × 1 图 × 3 物体 × 3 偏置 = 12”；raw 实际是 **4 组 (frame,img,obj) × 3 偏置 = 12**（000048 的 obj 1/6/14 + 000049 的 obj 6）。n 对，拆分描述错。
4. **`experiments/real/linemod/README.md` 的帧/图描述与 raw 不一致**：README 写“9 帧，每帧前 2 张图；对象 1/5/10”（按字面远超 18 trials）；raw 实际是 6 组 (frame,img,obj) × 3 偏置 = 18。论文按 raw 写“six (frame,image,object) configurations × three biases”。
5. **`main_table.md` 对 soft-penalty 最坏 trial 的描述不完整**：它只写“λ=1e4 有 trial 落到差局部解（trans 最大 203 mm）”。实际情况是同一 seed（4）在 **所有 λ≥10** 都掉进坏局部解；λ=1e3 的 trans 最坏 **504.6 mm**，λ=1e4 的 rot 最坏 **132.9°**（不只 203 mm trans）。论文按完整事实写。
6. **`main_table.md` 的“p>8 只增加冗余接触点，精度/穿透不再变化”只对 unconstrained/projection/ALM 成立**，对 soft penalty 不成立（其 median 随 p 变化 ≤2.1° / ≤2.1 mm，且 1/50 坏局部解可达 171.6° / 633.6 mm）。论文把 soft 列为例外。
7. **`main_table.md` 的 p 构成行 `9/4/1/1`、`17/8/3/1`、`33/12/9/1` 数值上不等于 p**（sum 15/29/55），因为 FoV 以“对”计数。不是错数，但容易被误读成不一致。论文 Table 5.1 直接给 scalar rows 0/2/6/18 并说明 pair 约定。
8. **Pareto sweep 与主表 soft λ=1e3 不是同一次 run**：逐 seed 值不同（例如 rot median 6.61 vs 6.62，worst case 也不同）。`main_table.md` 并排展示时没有注明；论文 §5.3 明确写“separate 50-trial sweep”。

### 6.3 本轮发现的实质性问题（不是 stale，而是会影响结论的 caveat）

9. **真实数据 cert flag 是单侧的**：`raw.json` 只判 `J−Shor ≤ 1e-6`，负 gap（Shor > J，soundness 被违反）也会被记为 certified。真实 ALM 的负 gap 最大到 **−3.7e-4 (LM) / −4.3e-4 (LM-O)**，远超 SCS 容差。改用双侧 `|J−Shor|≤1e-6` 后：ALM **7/18 (LM)、1/18 (LM-O)、9/12 (YCB)**；+RLT **5/18、0/18**。因此 `main_table.md` / README / 本报告的 50/22/83 都是**上界**；论文 §5.5(iv) 明确写了这一点。这是真实数据证书在投稿前必须补的 dual-feasible/interval 验证。
10. **真实 RLT 的 per-instance bound 数值不稳定**：除 9 的负 gap 外，还有 bound 崩塌（LM gap 1.49e-1、LM-O 6.2e-2）。所以“LM-O 22%→39%”不仅是 n=18 不显著（95% CI [9,45] vs [20,61] 重叠），其单实例 bound 也不可信；论文把它写成 numerical evidence，不写成 certification。
11. **§6 仍把真实数据 tightness 失败写成假设**（“If it fails on real data at a non-trivial rate...”），而 §5 已测到 50%/22%/83%（严格化后更低）。§6 不在我的编辑范围，留给 revision 统一。
12. **Abstract / §1.3 仍只承诺“protocol”**，没有提本轮新增的合成/真实结果。也不在 §5 范围，留作下一步（若要把论文改成 results paper，需要同步摘要）。

## 7. Figure

- 路径：`experiments/results/fig_p2_overview.png`（3029×897，200 dpi，约 200 KB / 204,052 B）。
- 生成：`make_p2_overview.py`，无随机性，可重复。
- Panel (a)：4 方法 × p，柱=feasible、黑色菱形=certified；一眼看出只有 ALM 在 100/100，projection 100 feasible/0 cert，soft ~38–50 feasible/0 cert，unconstrained 0/0。
- Panel (b)：penetration median [IQR] vs p，symlog y（0 可见）；projection 与 ALM 在 0.00 重合，unconstrained 恒为 103.02 mm，soft 在 0.00–0.01 mm。
- Panel (c)：LM/LM-O/YCB 的 ALM 认证率 base vs +0-dim RLT，带 95% Wilson CI；YCB +RLT 标 n/a。panel (c) 用的是**存储 flag**，因此继承 §6.3-9 的单侧 caveat；caption 与 §5.5(iv) 已说明。

## 8. Keep/revert

**keep。** 理由：

1. §5 的每个表/图数值都能从 raw JSON 逐格回算（mismatch 0），没有手写数字；
2. TODO 占位清零，pending 项（YCB RLT、exact inner、SCP/learned baselines、higher-order RLT、端到端训练）在 §5.6 显式标出，没有越界声明；
3. 真实数据按 fail-safe / 条件认证写，并额外暴露了单侧 cert flag 与 RLT bound 不稳定两个实质问题——这比“50/22/83 看起来不错”更接近可投稿状态。

不建议 revert；若必须回退，只需 `git checkout -- labs/02-constrained-qpeplayer/paper/draft.md` 并删除新增的 3 个文件（图、脚本、本报告）。

## 9. Caveats（写给 parent / 下一轮）

1. **真实认证率是上界**（§6.3-9）。要写进论文的“真实 cert rate”必须先用双侧/对偶可行验证重跑；当前 50/22/83 只能作为“存储 flag”。
2. **YCB RLT 未跑**；LM-O 的 22→39 在 n=18 下不显著（CI 重叠）；真实数据需要 scale-up（目标 n≥40）。
3. **RLT 只跑了 0-dim cuts**；σ=0.10/0.20 的失败与真实数据的不稳定都说明需要更高阶紧化，未验证。
4. **内层是 multi-start BFGS，不是 exact 40-root enumerator**；forward 时间是上界，“global forward”未测。`inner_baseline` 只说明 8 starts 在该 benchmark 上够用，不构成全局性证明。
5. **runtime 未做硬件归一化**；只用于 p 缩放比较，论文已注明。
6. **§5.1.5 sanity 表的数字是逐字保留的旧 draft 内容**，本轮没有重跑 `verify/`（不在我的编辑范围）；它们不被任何 paper claim 使用。
7. **图 panel (c) 与表 5.10 的 cert 列都用存储 flag**；若下一轮改成双侧验证，图/表/正文三处需同步更新（脚本读 `real_cert_ci_baseline.json`，改该文件即可重生成）。
