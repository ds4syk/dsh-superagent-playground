# P1-Q3 根计数快速复核：40 个复射影根（CONFIRMED）

日期：2026-11-27（phase 4，worker `wt-p1-roots2`）
脚本：`labs/01-laplace-qpeplayer/experiments/roots/run_rootcount_fast.py`
数据：`experiments/results/rootcount_fast.json`（相对 `labs/01-laplace-qpeplayer/`）
状态：**已实测**（`independent reproduction (this repo)`），非引文。

---

## 0. 一句话结论

对 3 个随机对称 `M` 和 `instance.Instance` 的 `M(0)`，**复射影平稳根恰为 40 个**（`q ~ -q` 已商掉），对应球面 KKT 系统的 80 个仿射复根（± 成对）；每个根都用残差独立验证到 `≤ 1.2e-13`。**P1-Q3 的「40 个复射影根」在本仓库实例上 CONFIRMED**。

## 1. 外部工具检查（先按任务要求）

```
M2 MISSING   msolve MISSING   sage MISSING   phcpy MISSING
bertini MISSING   Singular MISSING
```

本机没有任何可用的符号/同伦外部工具，因此自实现**一个**快速数值方法（总次数同伦），未安装任何东西。

## 2. 精确多项式系统

记 `q = (q0,q1,q2,q3) ∈ C^4`，`lambda ∈ C`，

```
v2(q) = [q0^2, q1^2, q2^2, q3^2,
         sqrt2 q0 q1, sqrt2 q0 q2, sqrt2 q0 q3,
         sqrt2 q1 q2, sqrt2 q1 q3, sqrt2 q2 q3]            (10 个二次单项式)
Jv    = d v2 / d q                                          (10 x 4，q 的线性函数)
J(q)  = v2(q)^T M v2(q)                                     (4 次齐次)
grad J(q)_i = 2 * sum_{a=1..10} Jv[a,i] * (M v2(q))_a       (q 的 3 次齐次)
```

平稳方程（`grad J = lambda q`，`q^T q = 1`）：

```
F_i(q,lambda) = grad J(q)_i - lambda * q_i = 0,   i = 0,1,2,3     (总次数 3)
F_4(q,lambda) = q0^2 + q1^2 + q2^2 + q3^2 - 1 = 0                (总次数 2)
```

Bezout 数 `3^4 * 2 = 162`。注意该仿射系统在无穷远处有**正维数**解集（见 §5），所以直接用 162 路同伦既慢又不可靠——这正是上一轮 sympy Gröbner + 总次数同伦 40 分钟超时的原因之一。

**射影化**：`J` 是 4 次齐次且为偶函数，`grad J(cq) = c^3 grad J(q) = c^2 lambda (cq)`，故「`grad J(q)` 与 `q` 平行」是尺度不变的射影条件，`q` 与 `-q` 是同一射影点。在坐标卡 `q_k = 1` 中它等价于 3 个 4 次方程

```
P_i(q) = grad J(q)_i - grad J(q)_k * q_i = 0,   i != k,   q_k = 1,
```

Bezout 数 `4^3 = 64`。每个射影点至多在一个卡中「有限性」变差，因此 4 个卡的结果取并即为全部射影根。

## 3. 方法（总次数同伦 + 复 Newton + endgame）

- 起始系统 `G_i(x) = x_i^4 - c_i`（`c_i` 随机复单位数，即 γ-trick），64 个已知起点（单位根组合）。
- 自然参数同伦 `H(x,t) = (1-t) G(x) + t F(x)`，`t: 0 -> 1`；批量复 Newton 预测-校正（全部 64 条路径同时推进），步长自适应（成功 ×1.2，失败 ×0.4）。
- `t > 0.9` 进入 **endgame**：步长按 `0.1(1-t)` 几何收缩，把 `t=1` 附近的接近根分开。
- 路径终点 `t=1` 再做 Newton 精修，残差 `< 1e-8` 才收为根；发散路径判为无穷远（本轮 `diverged = 0`，未能收敛的路径都停在 chart 的无穷远分支附近，`|x| ≲ 3e2`，记为 `failed`）。4 个卡取并、按 `max|Δ| < 1e-5` 去重。
- 每个根做两级独立验证：(a) 射影条件在「它有限的卡」里的残差；(b) 仿射提升 `q / sqrt(q^T q)` 代回 5 方程系统的残差（含 `q^T q = 1`、`lambda = 4J(q)`）。

## 4. 结果

命令（可复现，单进程，总耗时约 26 s）：

```
/home/syk/miniconda3/envs/WM/bin/python \
  labs/01-laplace-qpeplayer/experiments/roots/run_rootcount_fast.py
```

| 实例 | 复射影根 | 仿射复根 (±) | 实射影根 | 复（非实） | 实球面根 (±) | 实局部极小 (`find_modes`) | 卡内收敛/64 | 最大残差 (chart / lift) | 运行时间 |
|---|---|---|---|---|---|---|---|---|---|
| random_symmetric_seed1 | **40** | 80 | 26 | 14 | 52 | 5 | 44,41,40,41 | 5.3e-15 / 1.4e-14 | 5.3 s |
| random_symmetric_seed2 | **40** | 80 | 12 | 28 | 24 | 2 | 40,39,41,46 | 4.8e-15 / 7.6e-15 | 6.0 s |
| random_symmetric_seed3 | **40** | 80 | 12 | 28 | 24 | 2 | 40,42,40,32 | 7.1e-15 / 1.2e-13 | 6.9 s |
| `instance.M(0)`（seed 20261127） | **40** | 80 | 20 | 20 | 40 | 4 | 39,46,45,42 | 2.2e-15 / 2.4e-14 | 8.0 s |

- 每实例 64×4 = 256 条路径，`max_newton_residual` 全部 `< 1.2e-13`；**没有**丢失到无穷远的路径（`diverged = 0`），未收敛路径数 18–32/64，均为 chart 的无穷远分支（见 §5）。
- 实根判别：`q` 的规范代表虚部 `< 1e-7`（实测 `≤ 1e-32`），且所有实根的代数 `q^T q > 0`，因此每个实射影根恰好提升为 2 个实球面根（±）。`n_real_hyperboloid = 0`（没有 `q^T q < 0` 的实根，即没有落在双曲面 `q^T q=-1` 上的实平稳点）。
- `find_modes` 找到的每个实局部极小都出现在 40 根集合里：`min_k |q_k · q_root| = 1.000000`（4 个实例全部成立），即极小是根集合的子集，且 `n_minima ≤ 2 * n_real_projective` 全部满足。

## 5. 收敛性与完整性说明（诚实边界）

1. **为什么单卡只有 39–46 条路径收敛、而不是 40 条整**：chart 系统 `P_i` 不是齐次的，其齐次化在 `s=0`（即 `q_k=0`）处含一条**一维**分支（`grad J(q)_k = 0` 曲线），所以 64 个 Bezout 根中有 64 − 40 = 24 个在无穷远；这些路径在 endgame 中停在 `|x| ~ 30–300` 处无法给出有限根，记为 `failed`。这是方法的已知边界，不是数值发散（`diverged = 0`）。
2. **完整性论证（下界 + 交叉饱和）**：4 个卡各自独立（不同起始系统、不同卡、不同 γ）给出 32–46 个不同根，取并后**恰好饱和在 40**；两套额外随机起始系统（chart seed 101 / 2024）在 `random_symmetric_seed1` 与 `instance.M(0)` 上重复得到 **40**。若存在第 41 个射影根，它在 4 个卡中有 4 次机会被不同路径收敛，但 4 个卡都没有给出新点。因此 40 是**数值上可复现的计数**；严格上界（代数重数/商环维数）本轮未给出，见 §6。
3. **与 LibQPEP 的 40 的关系**：LibQPEP 的 `sols` 是 4×40，即 40 个四元数解，已经把 `q ~ -q` 商掉；本轮得到的正是「模 ± 的 40 个复射影根」，而球面 KKT 系统的复根是 80 个（± 成对）。旧文档里「27」与本轮 40 不冲突：27 是某个坐标卡/消元模板的中间数，不是射影根数（且奇数与 ± 自由对跖作用矛盾）。

## 6. 缺口 / 下一步

- **缺严格上界**：本轮给的是数值同伦计数（每根残差 ≤ 1.2e-13、跨卡/跨起始系统可复现），不是「商环维数 = 40」的符号证明。若要写成定理级证据，需要：(a) 在射影簇 `rank([q | grad J(q)]) ≤ 1` 上给 40 的代数重数证明；或 (b) 对小规模随机特化做有限域 Gröbner 商维数（上一轮 sympy 40 min 超时，可换 FLINT/msolve 容器）；或 (c) 用 monodromy + trace test 给下界并配 Bezout 型上界。
- **没跑 `phcpy`/`msolve`**：本机均未安装，按任务约束不安装。
- 上轮 `run_rootcount.py`（sympy Gröbner + 总次数同伦，40 min 超时）与本轮方法无关，本轮脚本为**新文件**，未修改任何既有 `.py`。

## 7. 环境

`/home/syk/miniconda3/envs/WM/bin/python`（numpy 2.2.6 / scipy 1.15.3 / sympy 1.14.0），Linux `ls`，单进程，无 GPU，总 wall ≈ 26 s（4 实例 × 4 卡 + `find_modes`）。
