# P2 真实数据严格证书汇总轮（S1–S9）

日期：2026-11-27 · worktree `wt-p2c-rigorous`（branch `opt/p2c-rigorous`，base `17d7ffc`）
解释器：`/home/syk/miniconda3/envs/sdpr/bin/python`（cvxpy 1.5.4 / numpy 1.26.3 / scipy 1.13.1）

> 口径：`cert` = ALM 可行（`viol <= 1e-6`）且 `J - bound <= 1e-6`，`bound` 必须是修复后的严格下界。
> 主界 = `rigorous_schur`（定理 P2-Cert，13 维 Schur，仅计 Cholesky 验证通过的 case A/B）；
> 保守界 = `rigorous`（全局 `repair_dual`，PSD 平移）。`naive` = 求解器 `value`，**不是界**，只作对照。
> `n` 为 ALM 行数（pose × bias）：LM 90（30 pose × 3 bias）、LM-O 72（24×3）、YCB-V 84（28×3）。

## 0. 调度前置检查（blocked，未排队）

`sinfo -p cpu` = `mix`（单节点 `ls`，非全空闲），`squeue -u syk` 为空。按任务约定**不提交 Slurm**，
三个进程直接登录节点并行：`nice -n 10`，`OMP/OPENBLAS/MKL/NUMEXPR_NUM_THREADS=1`。实测每进程 7–8 min，低于 15 min 上限。

## 1. 轮次表

| 轮 | 假设 | 改动 | 命令 | 指标 before → after | 判定 | 产物 |
|---|---|---|---|---|---|---|
| S0 | Slurm 空闲才排队 | 只检查 `sinfo`/`squeue` | `sinfo -p cpu; squeue -u syk` | `mix` 非全空闲 → 改登录节点 `nice -n 10` + 单线程 BLAS | blocked（不排队） | 本条记录 |
| S1 | `collect` 能把 Schur/dual-repair 字段写进逐行 raw | 无（smoke 子集：1 frame × 1 img，LM 3 行） | `python code/run_rigorous_real.py collect lm --smoke --max-rows 6` | 3 行 × 3 配置字段齐全（`value/rigorous/rigorous_schur/schur_case/time_s/status`）；bound > J 违例 **0**；3 行 wall 16.9 s；3 配置/pose 1.7–4.8 s | keep | scratch `real/smoke/lm/rigorous/`（运行后清除，不入库） |
| S2 | 全量 LM 可在 15 min 内跑完 | 无（完整重放，`RIGOROUS_ONLY=1`） | `nice -n 10 python code/run_rigorous_real.py collect lm > logs/rigorous_lm.log` | n=90；base Schur **22/90 (24.4%)**、dual **23**、naive **25**；base Schur gap 中位 **9.11e-6**、max 3.99e-4；wall **476.5 s** | keep | `real/linemod/results/rigorous/`、`results/rigorous_lm.json/.md` |
| S3 | LM-O 同样可跑完 | 无 | `... collect lmo > logs/rigorous_lmo.log` | n=72；base Schur **11/72 (15.3%)**、dual 11、naive 19；gap 中位 **1.55e-5**、max 2.56e-4；wall **437.9 s** | keep | `real/bop/results/lmo/rigorous/`、`results/rigorous_lmo.json/.md` |
| S4 | YCB-V 同样可跑完 | 无 | `... collect ycbv > logs/rigorous_ycbv.log` | n=84；base Schur **11/84 (13.1%)**、dual 10、naive 17；gap 中位 **2.10e-5**、max 4.02e-4；wall **425.4 s** | keep | `real/bop/results/ycbv/rigorous/`、`results/rigorous_ycbv.json/.md` |
| S5 | 三数据集可合并为一张严格证书表 | `final()` 重聚合 + 合并 + 生成 md | `python code/run_rigorous_real.py final` | 三数据集全部 `max|ΔJ| = 0`（与已发表 `raw.json` 实例逐位一致）；产出 `rigorous_real.{json,md}`；Schur 修复对 dual 的中位增益 8.5e-8 / 1.6e-7 / 7.2e-8（见 §3） | keep | `results/rigorous_real.json`、`results/rigorous_real.md`、`results/rigorous_soundness.md` |
| S6 | 所有行、所有配置严格 sound | `_soundness` 全量审计 + 单独审计 md；新增 verified-case 计数 | `... final` | **738 次 bound 检查（246 行 × 3 配置）0 违例**；`rigorous_schur > J` 0、`dual > J` 0、`Schur > value` 0、非法平移 0、NaN 0、`A_unverified` 0、`B_noRt` 0；ALM max viol 8.59e-11 | keep | `results/rigorous_soundness.md` |
| S7 | LM-O 纯平面实例的 rank-2 伪混合可用 depth+tilt+跨方向 RLT 切掉 | `depth-tilt --max-rows 6 --select worst`；每次重解 3 个证书配置（base / +bound-RLT / +full RLT），逐行落盘 | `python code/run_rigorous_real.py depth-tilt --max-rows 6 --select worst --alpha 12 --n-depth 4 --zmin 0.02` | 选 base 未认证的 **6 个最差行**（gap 1.00e-4–2.56e-4）；加 4 条深度（`b=-e3`, `zmin=0.02`）+ 1 条 tilt 锥（12°）后：base **0/6**，+bound-RLT **6/6**（gap 5.2e-12–1.95e-11），+full RLT **6/6**（5.1e-11–5.3e-9）；`J` 不变（|ΔJ| ≤ 1.3e-14），ALM 位姿本就满足 depth（min z 0.845–0.883 m）与 upright（0.9983–0.9995 > cos12°） | keep（augmented 口径，见 §5 声明） | `results/rigorous_lmo_depth_tilt.json`、`logs/rigorous_lmo_depth_tilt.log` |
| S8 | 严格列必须由脚本写进已发表表，不能手改 | `update_tables()` 幂等追加块 + `meta.json["rigorous"]` | `... final` | 3 个 `main_table.md` 各追加 `<!-- RIGOROUS_CERT:BEGIN -->` 块；3 个 `meta.json` 写入严格证书字段（含 verified CI、Schur case、soundness）；论文可引用的 n/CI 见 §6 | keep | 各 `real/*/results/main_table.md`、`meta.json` |
| S9 | brlt 的失败是求解器对偶质量而非非法切 | CLARABEL 与 SCS 200k 迭代在 2–3 个最差 LM 行上探测（一次性诊断，不改主配置） | 见 §7 命令 | CLARABEL 在这些 SDP 上 `Solver failed`；SCS 200k 迭代两行中一行 0.277→1.2e-5，另一行 0.156→2.72（更差）；结论：brlt 的 gap 双峰是 SCS 对偶质量问题，界仍合法但偏松 | keep（负结果，记录） | 本文件 §7 |

## 2. 严格证书表（主表；95% Wilson CI）

`base` = 已发表平面实例；`rlt` = 零维 RLT `g_i g_j >= 0`；`brlt` = bound-RLT `g_i(y_k-lo_k) <= 0`。
`brlt` 只加**有效切**，因此它的证书仍然是同一个原问题的证书。

| dataset | n | cuts | Schur cert (95% CI) | dual-repair cert | naive(value) | Schur gap min/median/max | Schur case |
|---|---|---|---|---|---|---|---|
| LM | 90 | base | **22 (24.4%; [16.7,34.2])** | 23 (25.6%) | 25 | 3.98e-12 / 9.11e-6 / 3.99e-4 | A 90 |
| LM | 90 | rlt | 19 (21.1%; [14.0,30.6]) | 18 | 35 | 9.26e-12 / 1.64e-5 / 2.27e-1 | A 90 |
| LM | 90 | brlt | **46 (51.1%; [41.0,61.2])** | 43 | 61 | 3.99e-12 / 3.37e-7 / 2.77e-1 | A 90 |
| LM-O | 72 | base | **11 (15.3%; [8.8,25.3])** | 11 | 19 | 1.65e-11 / 1.55e-5 / 2.56e-4 | A 72 |
| LM-O | 72 | rlt | 6 (8.3%; [3.9,17.0]) | 6 | 19 | 3.18e-11 / 6.40e-5 / 3.28 | A 72 |
| LM-O | 72 | brlt | **30 (41.7%; [31.0,53.2])** | 24 | 45 | 5.60e-12 / 2.13e-6 / 9.37e-2 | A 72 |
| YCB-V | 84 | base | **11 (13.1%; [7.5,21.9])** | 10 | 17 | 1.00e-11 / 2.10e-5 / 4.02e-4 | A 84 |
| YCB-V | 84 | rlt | 6 (7.1%; [3.3,14.7]) | 6 | 11 | 1.00e-11 / 3.81e-5 / 1.07 | A 84 |
| YCB-V | 84 | brlt | **20 (23.8%; [16.0,33.9])** | 19 | 31 | 4.10e-12 / 1.25e-5 / 3.01 | A 84 |

注意：`naive` 用本次 SCS 解（eps=1e-9, 50k iters），与旧 one-sided 表（20k iters）略有差异；
旧表 LM 30/90=33.3%、LM-O 22/72=30.6%、YCB 28/84=33.3% 只是**上界**，均不可作为证书引用。

## 3. dual-repair vs Schur（同一对偶点）

| dataset | cuts | Schur−dual 中位 | min | max | Schur 更紧行数 | 更松行数 | dual 未修复 |
|---|---|---|---|---|---|---|---|
| LM | base | +8.49e-8 | −1.51e-6 | +3.37e-6 | 82/90 | 8/90 | 0 |
| LM-O | base | +1.57e-7 | −2.51e-6 | +1.28e-5 | 68/72 | 4/72 | 0 |
| YCB-V | base | +7.18e-8 | −1.11e-5 | +2.16e-5 | 78/84 | 6/84 | 0 |

Schur 界在绝大多数行更紧（约 4 个数量级更小的修复损耗），且全部 case A（`H_tt ≻ 0`），
没有出现 `B`/`B_noRt`/`A_unverified`——真实数据上不存在 `R_t` 依赖问题。故论文以 Schur 为主界、dual 为保守 fallback。

## 4. 运行时

| dataset | cuts | 证书求解中位 | 最大 | Schur 后处理中位 | 中位不等式数 |
|---|---|---|---|---|---|
| LM | base / rlt / brlt | 1.65 / 0.96 / 1.96 s | 2.72 / 4.45 / 3.82 s | ~2–3.5e-4 s | 16 / 152 / 336 |
| LM-O | base / rlt / brlt | 1.66 / 2.60 / 2.51 s | 2.25 / 3.59 / 3.96 s | ~2–3.5e-4 s | 16 / 152 / 336 |
| YCB-V | base / rlt / brlt | 1.68 / 1.12 / 1.52 s | 2.20 / 3.73 / 3.88 s | ~2–3.5e-4 s | 16 / 152 / 336 |

每数据集 3 配置 × 3 数据集 × 全部行，三进程并行（单线程 BLAS），wall 477 / 438 / 425 s。
逐行 `rigorous_rows.json` + `rigorous_arrays.npz` 已落盘，可离线复算、无需重放。

## 5. LM-O depth+tilt（S7）before → after

选择：base 未认证行中 gap 最大的 6 行（`select=worst`），单 frame `000002`，obj 5/8/12，bias 20/30。
增强：4 条深度约束 `(Rv+t)_z >= 0.02`（GT 最近顶点）+ 1 条 tilt 锥 `n^T R(q) u >= cos 12°`。

| frame | img | obj | bias | J_old | gap_old (base) | gap_aug base | gap_aug +brlt | gap_aug +rlt | cert base/brlt/rlt |
|---|---|---|---|---|---|---|---|---|---|
| 000002 | 1 | 8 | 30 | 4.5417e-4 | 2.56e-4 | 2.54e-4 | **9.02e-12** | **1.20e-9** | F/T/T |
| 000002 | 0 | 8 | 30 | 5.4827e-4 | 2.56e-4 | 2.37e-4 | **5.17e-11** | **5.09e-11** | F/T/T |
| 000002 | 1 | 5 | 30 | 3.0391e-4 | 1.78e-4 | 1.77e-4 | **1.03e-11** | **5.34e-9** | F/T/T |
| 000002 | 0 | 5 | 30 | 2.5315e-4 | 1.15e-4 | 1.15e-4 | **9.90e-11** | **9.78e-11** | F/T/T |
| 000002 | 0 | 8 | 20 | 2.9761e-4 | 1.02e-4 | 1.02e-4 | **1.95e-11** | **1.93e-11** | F/T/T |
| 000002 | 0 | 12 | 30 | 2.3687e-4 | 1.00e-4 | 9.93e-5 | **6.01e-11** | **2.36e-10** | F/T/T |

结论：**rank-2 伪混合被切掉**——只加 depth/tilt 而不加跨方向切时 gap 仍是 ~1e-4；
加 bound-RLT 或 full RLT 后 6/6 达到 ≤5.3e-9。`J_new ≡ J_old`（|ΔJ| ≤ 1.3e-14）、ALM 位姿满足
depth（min z 0.845–0.883 m）与 upright（0.9983–0.9995 > cos12°=0.9781），说明原 ALM 解仍在增广可行域内，
证书下降完全来自松弛收紧。

**口径声明**：depth/tilt 约束**缩小**可行域，因此这里认证的是「增广后的物理问题」，不是原始 16-plane-only 问题；
不能把 6/6 写成对原问题的认证率。这是定理 §2.7.4 明确划出的实验边界。

## 6. 论文应引用的数字（严格口径）

- **平面实例（无额外切，base）**：LM 22/90 = 24.4% [16.7, 34.2]；LM-O 11/72 = 15.3% [8.8, 25.3]；YCB-V 11/84 = 13.1% [7.5, 21.9]。
- **同一问题 + 有效 bound-RLT 切（brlt）**：LM 46/90 = 51.1% [41.0, 61.2]；LM-O 30/72 = 41.7% [31.0, 53.2]；YCB-V 20/84 = 23.8% [16.0, 33.9]。
  这是推荐的严格主表：切只依赖已证变量界，不改变原问题；但 brlt 在部分行上 SCS 对偶很差（gap 双峰），所以这些比率**偏低**而非偏高。
- **不要**再引用旧的 one-sided 33.3% / 30.6% / 33.3%（或更早的 50%/22%/83%）作为 certified；它们是 solver value 上界。
- **LM-O depth+tilt**：只能作为「增广物理问题」的 6/6 认证（J 不变，gap 从 ~1e-4 降到 ≤5.3e-9），不能替代原问题认证率。
- 主界用 `rigorous_schur`；`repair_dual` 只在需要与旧结果衔接时作 fallback（约 1e-7–1e-5 更松）。

## 7. 求解器质量诊断（负结果）

命令（一次性，用已存 `rigorous_arrays.npz`，不重放）：

```bash
python - <<'PY'
import sys, json, numpy as np
sys.path.insert(0, "code")
from certificate import shor_bound
rows = json.load(open("real/linemod/results/rigorous/rigorous_rows.json"))["rows"]
arr = dict(np.load("real/linemod/results/rigorous/rigorous_arrays.npz"))
# 取 brlt gap 最大的行，分别用 CLARABEL / SCS(200k iters) 重解 add_bound_rlt=True
PY
```

结果：CLARABEL 在最差 3 行全部 `Solver failed`（0.14–0.17 s 即失败）；SCS `max_iters=200000`
（eps 1e-9/1e-10 无差别）两行中一行 gap 0.277 → 1.16e-5（仍不认证），另一行 0.156 → 2.72（更差）。
因此 brlt 的大 gap 行不是「切非法」，而是对偶点质量差；这些行上的 `rigorous_schur` 仍是合法下界（S6 已验证），
只是偏松。若要提高 brlt 认证率，需要更好的 SDP 对偶（缩放/活跃集/专门的对偶修复），不在本轮范围。

## 8. 产物与复现

- 代码：`code/run_rigorous_real.py`（新增 `--smoke`、verified-case 统计、`rigorous_soundness.md`、
  depth-tilt 3 配置与 `--select worst`、逐行 progress jsonl）；`code/certificate.py` 未改。
- 数据：`real/linemod/results/rigorous/`、`real/bop/results/lmo/rigorous/`、`real/bop/results/ycbv/rigorous/`
  （`rigorous_rows.json` + `rigorous_arrays.npz`）。
- 结果：`results/rigorous_{lm,lmo,ycbv}.{json,md}`、`results/rigorous_real.{json,md}`、
  `results/rigorous_soundness.md`、`results/rigorous_lmo_depth_tilt.json`。
- 表：`real/*/results/main_table.md` 的 `RIGOROUS_CERT` 块、`meta.json["rigorous"]`，均由 `final()` 生成，不手改。
- 日志：`logs/rigorous_{lm,lmo,ycbv}.log`（逐行 `rig` 行）、`logs/rigorous_lmo_depth_tilt.log`。
- 复现：`python code/run_rigorous_real.py collect {lm,lmo,ycbv}` → `depth-tilt ...` → `final`。
