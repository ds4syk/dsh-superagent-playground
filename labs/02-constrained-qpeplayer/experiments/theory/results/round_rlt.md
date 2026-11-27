# P2 round-rlt：sound 高阶切 + 严格对偶修复（14×14 Veronese-Shor 证书）

日期：2026-11-27 · worktree `wt-p2-rlt`（branch `opt/p2-rlt`）· 解释器
`/home/syk/miniconda3/envs/sdpr/bin/python`（cvxpy 1.5.4 / numpy 1.26.3 / scipy 1.13.1）

## 轮次行

| 假设 | 改动 | 命令 | 指标 before → after | 保留/回退 | 产物 |
|---|---|---|---|---|---|
| 零维 RLT 在高噪声下太弱；**sound** 的高阶/添界切 + **严格对偶可行修复** 能同时收紧下界并让"certified"真正 sound | `certificate.py` 新增 `add_bounds` / `add_bound_rlt` / `add_box_rlt` 切族与 `repair_dual`（严格对偶下界）；新增 `run_round_rlt.py`（4 cell × 4 seed × 5 变体 × 2 求解器）与 `run_round_rlt_lmo.py` | `python .../run_round_rlt.py`（108 s）→ `python .../post_round_rlt.py`（从 JSON 重算 check/表）；`python .../run_round_rlt_lmo.py` | **严格 cert 率（SCS，16 实例）9/16 → 16/16**；最大严格 gap **1.15e-2 → 2.28e-8**；0.10 cell 1/4→4/4；0.20 cell 0/4→4/4；中位耗时 0.036 s→0.180 s | **keep** `add_bound_rlt` + `repair_dual`；**revert** `add_box_rlt`；`add_rlt` 单独保留但不作主变体；`add_bounds` 单独无增益 | `results/round_rlt.json`、`results/round_rlt_tables.md`、`results/round_rlt_lmo.json`、两份 `run_round_rlt*.log` |

一句话：**base 与 base+RLT 在高噪声的"不 certified"是松弛真的松，不是求解器容差**（CLARABEL 独立复现同一 gap）；
而一组 **sound 的二次添界-RLT 切** `g_i·(y_k-lo_k) ≤ 0`（界来自 sound 区间传播）把 0.10/0.20 两个 cell
从 1/4、0/4 直接推到 **4/4**，严格对偶界只比 J 低 **2.0e-8**（即修复下限）。严格对偶修复是"敢说 certified"
的必要条件：naive 求解器值不是下界，在 2/160 行上与严格结论不一致，LM-O 上还抓到 1 个 naive 假证书。

## 0. 口径

* `J`：ALM 前向在可行点（`max_viol ≤ 1e-6`）上的目标值，故 `SDP_opt ≤ J* ≤ J`。
* `value`：求解器报告的 `<Q,X>`。**不是**严格下界（`status=optimal_inaccurate` 时 X 常轻微违反 `g≤0`）。
* `rigorous`：`certificate.py` 的对偶可行修复值（见 §1）。`rig gap = J - rigorous`。
* `cert`（与原 `run_rlt.py` 一致）：`feasible and J - bound ≤ 1e-6`。
* 修正任务书笔误：三次族应是 `g_i g_j g_k ≤ 0`（三个非正数之积非正），不是 `≥ 0`；见 §5.4。

## 1. 关键方法学发现：cvxpy 对偶符号（实测）

严格下界走 Lagrange 对偶。若 `S = Q - Σ ỹ_k A_k + Σ λ_m M_m ⪰ 0`（`λ≥0`），则对任意原始可行 X：
`<Q,X> ≥ Σ ỹ_k b_k`。实测（cvxpy 1.5.4 + SCS/CLARABEL）：

* `cons.dual_value` 的 **y 与标准乘子反号**（`ỹ = -y`），`≤` 约束的 λ 与标准乘子同号；
* 因此正确的重建是 `S = Q + Σ y_k A_k + Σ λ_m M_m`，严格界 `= -Σ y_k b_k`；
* 校验：这样重建的 S 与 `X>>0` 的对偶矩阵 `cons[0].dual_value` 逐元素一致
  （探针实例 `||S-Z||_F = 3.8e-8`，`λmin(S)≈-1e-8`），对偶目标 `-y·b` 与 primal 值一致到 1e-11。
  直接取 `S = Q - Σ y A + …` 会得到 `λmin(S) ≈ -1.4`、界低到 `-0.65`（错误）。

修复（`_dual_repair`）：取求解器对偶 → `λ ← max(λ,0)` → 由 **Schur 补**求最小平移 `c`
（`P = diag(1_{0..9},0,0,0,1)` 恰为两个 `b=1` 等式约束矩阵之和，故 `+cP` 使对偶目标减少 `2c`）→
`rigorous = -y·b - 2c`。`c` 的典型值 1e-8（即 `repair_delta` 下限），所以严格界 ≈ SDP 最优值 − 2e-8。
**任何 (y, λ≥0) 都给出合法下界**，因此该路径不会"伪证"；只是求解器对偶不准时界会偏保守（见 CLARABEL 行）。

## 2. 切族（全部默认关闭，flag 开启）

记 `y = (v2(q), t, 1) ∈ R^14`，`g_j(y) = c_j^T y ≤ 0`。所有切对**任意原始可行点**成立，
且都线性于 `X = E[y y^T]`：

| flag | 切 | 依据 |
|---|---|---|
| `add_rlt` | `g_i g_j ≥ 0`（i≤j） | 两个非正数之积非负 |
| `add_bounds` | `(y_k - lo_k)(y_k - hi_k) ≤ 0`，退化为单侧界 | `box_bounds` 的已证盒 |
| `add_bound_rlt` | `g_i·(y_k - lo_k) ≤ 0`、`g_i·(hi_k - y_k) ≤ 0` | `g_i≤0` 且因子 `≥0` |
| `add_box_rlt` | 两个已证单侧因子之积 `≥ 0`（含 `(lo_k)(hi_l)` 交叉） | 同上；**未保留** |

`box_bounds(A,B)`（sound 区间传播）：
1. 用 `E_TR·v2 = 1` 精确消去 `v2_0 = 1 - v2_1 - v2_2 - v2_3`；
2. 基础盒 `v2_i∈[0,1]`、`|v2_ab|≤1/√2`，并追加 sound 行 `v2_1+v2_2+v2_3 ≤ 1`（因 `v2_0≥0`；
   缺这行会让盒与替换式矛盾、传播推出 infeasible 的 ±inf —— 这是本轮修掉的一个真实 bug）；
3. 交叉项用 `|v2_ab| ≤ sqrt(2·v2_a·v2_b) ≤ sqrt(2 hi_a hi_b)` 收紧（tilt 锥自动从 `A` 里的行推出，
   **不硬编码 alpha**，因此对任意实例 sound）；
4. 对 `A v2 + B t ≤ 0` 反复传播（注意 `c_k<0` 时下界取 `rhi/c_k`，不是 `rlo/c_k` —— 另一个修掉的 bug）。

本 benchmark（p=16）得到的界示例：`v2_1,v2_2 ≤ 0.0109`（tilt），`v2_7∈[±0.0155]`，
`t_z ≥ 0.765`（穿透/深度 + 盒）。`t_x, t_y` 保持 ±inf（约束确实不给它们有限界）。

## 3. 数字

### 3.1 变体总表（16 实例 = 4 noise cell × 4 seed；cert 容差 1e-6）

| solver | variant | cert naive | **cert rigorous** | max rig gap | median rig gap | max dual shift | median n_ineq | median t(s) |
|---|---|---|---|---|---|---|---|---|
| SCS | base | 9/16 | **9/16** | 1.148e-02 | 2.37e-08 | 1.2e-06 | 16 | 0.036 |
| SCS | +rlt | 10/16 | **10/16** | 9.817e-03 | 2.14e-08 | 3.2e-07 | 152 | 0.158 |
| SCS | +bounds | 9/16 | **9/16** | 1.148e-02 | 2.02e-08 | 1.5e-07 | 29 | 0.048 |
| SCS | **+bound_rlt** | **16/16** | **16/16** | **2.282e-08** | 2.01e-08 | 1.1e-08 | 352 | 0.180 |
| SCS | +all | 15/16 | 13/16 | 1.321e-03 | 2.00e-08 | 9.0e-04 | 732 | 0.452 |
| CLARABEL | base | 9/16 | 9/16 | 9.442e-03 | 3.00e-08 | 1.2e-08 | 16 | 0.032 |
| CLARABEL | +rlt | 10/16 | 10/16 | 9.429e-03 | 2.95e-08 | 5.8e-06 | 152 | 0.107 |
| CLARABEL | +bounds | 9/16 | 9/16 | 1.147e-02 | 3.44e-08 | 1.3e-08 | 29 | 0.036 |
| CLARABEL | +bound_rlt | 13/16 | 13/16 | 8.510e-05 | 5.56e-08 | 3.8e-05 | 352 | 0.164 |
| CLARABEL | +all | 16/16 | 16/16 | 1.460e-07 | 4.33e-08 | 8.5e-08 | 732 | 0.364 |

`+rlt` 只从 9/16 提到 10/16（对应原 `rlt.md` 的 0.10 cell seed 1），**与假设一致：零维 RLT 太弱**。

### 3.2 每 cell 严格 cert（SCS；括号为中位严格 gap）

| noise | base | +rlt | +bounds | **+bound_rlt** | +all |
|---|---|---|---|---|---|
| 0.02 | 4/4 (2.0e-8) | 4/4 (2.0e-8) | 4/4 (2.0e-8) | **4/4 (2.0e-8)** | 2/4 (6.5e-6) |
| 0.05 | 4/4 (2.0e-8) | 4/4 (2.0e-8) | 4/4 (2.0e-8) | **4/4 (2.0e-8)** | 3/4 (2.1e-8) |
| 0.10 | 1/4 (4.3e-4) | 2/4 (2.9e-4) | 1/4 (4.2e-4) | **4/4 (2.0e-8)** | 4/4 (2.0e-8) |
| 0.20 | 0/4 (8.6e-3) | 0/4 (7.8e-3) | 0/4 (8.6e-3) | **4/4 (2.0e-8)** | 4/4 (2.0e-8) |

### 3.3 退化 cell 逐实例（SCS，J 由 ALM 给出）

| noise | seed | J | base gap | +rlt gap | +bounds gap | **+bound_rlt gap（严格）** |
|---|---|---|---|---|---|---|
| 0.10 | 0 | 9.0579e-01 | 3.194e-03 | 3.093e-03 | 3.193e-03 | **2.0e-08** |
| 0.10 | 1 | 6.8500e-01 | 2.385e-04 | 7.6e-12 | 2.377e-04 | **2.0e-08** |
| 0.10 | 2 | 4.3643e-01 | 6.093e-04 | 5.794e-04 | 6.098e-04 | **2.0e-08** |
| 0.10 | 3 | 5.7689e-01 | -2.3e-10 | 2.0e-09 | 5.8e-11 | **2.0e-08** |
| 0.20 | 0 | 3.2392e+00 | 9.447e-03 | 8.541e-03 | 9.391e-03 | **2.0e-08** |
| 0.20 | 1 | 3.1128e+00 | 1.148e-02 | 9.826e-03 | 1.148e-02 | **2.0e-08** |
| 0.20 | 2 | 3.0309e+00 | 7.763e-03 | 7.150e-03 | 7.754e-03 | **2.0e-08** |
| 0.20 | 3 | 2.2501e+00 | 6.586e-03 | 5.509e-03 | 6.095e-03 | **2.0e-08** |

`+bound_rlt` 的 gap 全部等于 `2×repair_delta`，即**松弛在这些实例上恰好在 ALM 可行点处精确**
（`SDP_opt ∈ [J−2.3e-8, J]`）。这不是容差伪造：同一 SDP 用 CLARABEL 在 0.02/0.10 两个 cell 也是
4/4（0.05/0.20 的少数失败是 CLARABEL 自己 `optimal_inaccurate`，见 §5.3）。

### 3.4 soundness 检查（全部来自 `round_rlt.json:checks`）

| 检查 | 结果 |
|---|---|
| 切在**采样可行点**上的最大违反（16 实例；GT + 3×ALM + ≤25 个随机 q + LP 可行 t） | `g:+1.0e-10, rlt:+8.2e-11, bounds:-6.8e-26, bound_rlt:+1.1e-10, box_rlt:-3.8e-47` → **通过**（≤1e-8） |
| `naive value`：bound_with_cuts ≥ base − 1e-6 | **3 行违反**，且全是 `status=optimal_inaccurate` 的大 SDP：SCS `all` 0.02/2（−3.3e-4）与 CLARABEL `bound_rlt` 0.05/0、0.05/3（−3.5e-6、−6.9e-6）。属**求解器未收敛**，非切不 sound |
| `rigorous`：bound_with_cuts ≥ base_rigorous − 1e-6 | **5 行违反**：SCS `all` 0.02/2、0.02/3、0.05/2（`dual_shift` 6e-6~9e-4）+ CLARABEL `bound_rlt` 0.05/0、0.05/3（`optimal_inaccurate`，平移 1.9e-6/3.8e-5） |
| 无 NaN | SCS 160/160 有值；CLARABEL **4/160 solver raise**（非切问题） |
| `rank-1 ⇒ bound ≤ J + 1e-6` | 115 个 rank-1 解，**0 违反** |
| 修复后对偶 slack PSD（`λmin(S_rep) ≥ −1e-10`） | 除 4 个 CLARABEL 失败行外全部通过（典型 +7.5e-9） |
| naive cert 与 rigorous cert 不一致 | **2 行**：SCS `all` 0.02/3（naive gap −4.0e-7 判 cert，严格 gap 1.3e-5 判否）、SCS `all` 0.05/2（naive −1.1e-3，严格 1.3e-3）→ **naive cert 的容差依赖被直接抓到** |

### 3.5 LM-O 真实子集（6 实例：2 物体 × bias 20/30/40 px，frame 000002/img 0）

只搬 `run_bop.py` 的 plane 约束（无 tilt/depth），N=30、P_CON=16、noise=1 px。

| obj | bias | J | base gap | base 严格 | **bound_rlt gap（严格）** | 备注 |
|---|---|---|---|---|---|---|
| 5 | 20 | 2.11e-04 | 7.18e-05 | 7.22e-05 | 2.72e-02 | bound_rlt 的 SCS `optimal_inaccurate`，反而更松 |
| 5 | 30 | 3.09e-04 | 1.70e-04 | 1.71e-04 | 4.55e-03 | 同上 |
| 5 | 40 | 6.46e-04 | 4.58e-04 | 4.58e-04 | **1.81e-05** | 改善 25×，仍不 cert |
| 8 | 20 | 2.97e-04 | **−2.27e-05（naive 假 cert）** | **1.04e-04（严格：不 cert）** | 6.95e-05 | **naive 值与严格界冲突的直接证据** |
| 8 | 30 | 3.45e-04 | 1.59e-04 | 1.59e-04 | 1.28e-04 | 略改善 |
| 8 | 40 | 7.21e-04 | 4.78e-04 | 4.79e-04 | 3.78e-04 | 略改善 |

结论：**真实实例上不 certified**（gap 相对 J 为 1e-4~5e-4），bound_rlt 有改善但不稳定
（2/6 因 336 条切导致 SCS 未收敛而更松）。属**部分负结果**：合成 benchmark 的"精确化"不能外推到 LMO。

## 4. 成本

| variant | median n_ineq | median t（SCS） | median t（CLARABEL） |
|---|---|---|---|
| base | 16 | 0.036 s | 0.032 s |
| +rlt | 152 | 0.158 s | 0.107 s |
| +bounds | 29 | 0.048 s | 0.036 s |
| **+bound_rlt** | 352 | **0.180 s** | 0.164 s |
| +all | 732 | 0.452 s | 0.364 s |

## 5. 保留/回退决定

1. **保留 `add_bound_rlt`（首选变体）**：唯一在 0.10/0.20 两个 cell 全 4/4 严格 cert 的族，
   严格 gap 掉到修复下限 2e-8；代价 352 条线性约束、中位 0.18 s。**这是本轮主交付。**
2. **保留 `repair_dual`（必须）**：naive `value` 不是下界；严格对偶修复把 "certified" 变成可审计的结论。
   修复值 ≈ SDP 值 − 2e-8（求解器对偶好时），对偶差时如实变松（CLARABEL 0.05 cell）。
3. **`add_rlt`：保留 flag，但不作为主变体**（9/16→10/16，被 bound_rlt 完全覆盖）。
4. **回退 `add_box_rlt`**：加 732 条后 SCS 在 3/16 行未收敛（严格 cert 15→13，甚至出现 value < base），
   无任何相对 bound_rlt 的增益；CLARABEL 虽 16/16，但同类增益可用 bound_rlt 达到。**判定：不保留。**
5. **`add_bounds` 单独：无增益**（9/16）——但它提供的 `t_z` 等界被 `add_bound_rlt` 内部复用，故保留代码。
6. **三次 lift：未做（明确记录）**。理由：任务书符号写反（应为 `≤0`）；14 变量三次单项式
   `C(17,3)=680` 个，moment 矩阵 680×680，单实例 SDP 规模≈100×、远超本轮 40 min 预算；
   且 bound_rlt 已把 0.10/0.20 cell 压到 2e-8，边际收益上限只剩 1e-3 量级。**建议下一轮再做**（并先做规模/时间预估）。

## 6. Caveats

1. **"精确化"的含义**：`+bound_rlt` 的严格界 = J − 2e-8 表示 `SDP_opt ∈ [J−2e-8, J]`，
   即 ALM 可行点就是全局最优到 2e-8。若 ALM 落在次优局部解（J > J*），界会停在 J* 附近而非 J；
   本轮 16 个合成实例上 ALM 均达到全局最优。
2. **区间传播是"经证明 + 数值审计"**，不是形式化证明：soundness 由每步的区间运算保证，
   数值上在 16×~30 个采样可行点上最大违反 1.1e-10（浮点级）。审计点是"可行点采样"，不是穷举。
3. **求解器鲁棒性**：`bound_rlt` 的 352 条约束对 SCS 友好、对 CLARABEL 在 0.05 cell 反而
   `optimal_inaccurate`（3/16），且 CLARABEL 共 4/160 直接 raise。生产上应固定 SCS 或做 fallback。
4. **`all` 变体的 value < base 不是切不 sound**：切只可能抬高松弛最优值；观测到的降低来自
   `optimal_inaccurate`。这也说明"用 naive value 做 soundness 断言"本身依赖求解器收敛。
5. **LM-O 子集只有 6 个实例、单 frame、单物体 × 3 bias**，且约束只有 plane；不构成真实 benchmark 结论。
6. `repair_delta=1e-8` 是严格界的下限来源；若需更强的浮点保证可调大到 1e-7（界相应再低 ~2e-7）。

## 7. 复现

```bash
cd labs/02-constrained-qpeplayer
/home/syk/miniconda3/envs/sdpr/bin/python experiments/theory/run_round_rlt.py      # ~108 s, 160 SDP
/home/syk/miniconda3/envs/sdpr/bin/python experiments/theory/post_round_rlt.py     # 从 JSON 重算 checks/表
/home/syk/miniconda3/envs/sdpr/bin/python experiments/theory/run_round_rlt_lmo.py  # LM-O 6 实例
```

产物：`results/round_rlt.json`（raw rows + audits + checks + summary）、`results/round_rlt_tables.md`
（逐实例/逐 cell 表）、`results/round_rlt_lmo.json`、`results/run_round_rlt*.log`。
