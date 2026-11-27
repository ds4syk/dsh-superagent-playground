# Inner solver: weighted Kabsch (current) vs Horn 4x4 (candidate)

> 生成时间：2026-11-27T20:52:37+08:00 ｜ 分支 `agent/three-paper-tracks-95244014fb3c` @ `c4cca04` ｜ Python 3.10.22 / NumPy 2.2.6

## 0. 一句话结论

**保留 Davenport**：两者在 1000×3 个加权问题上数值等价（最大旋转差 1.07e-14，最大代价超出 9.44e-16，成功率完全一致），而 Davenport 的 p50 几何平均加速 **1.07×**（N=20/100/500 分别 0.94×/1.02×/1.28×）。

工程解读：p50 收益集中在 N=500（1.28×），最差档 N=20（0.94×，慢于基线）。小 N 下固定开销（权重校验 + det 检查 + 四元数→旋转矩阵）占主导；若内层主要用于 N≤100，收益在噪声内，是否采用取决于目标 N。

## 1. 范围说明（重要）

> 范围说明：本实验只验证 3D-3D 的标量权重内层加权最小二乘。当 W_i = w_i I 时，QPEP 的 6-DoF 位姿估计退化为“加权质心 + SO(3) 上的正交 Procrustes”，其全局最优由 4x4 对称矩阵的最大特征向量精确给出（Horn 1987）。因此这里比较的是同一 4x4 特征问题的两种实现（Kabsch/SVD vs Horn/4x4），并没有实现、也没有声称实现完整的 40 根 QPEP 多项式求解器（十次/四十次多项式、根筛选、数值完备性、P3P 等均不在本轮范围）。Davenport/Horn 在本文语境中的角色是：一个可证明全局最优的闭式内层，反射/det 修正由四元数参数化自动满足（无需 Kabsch 式的 det 分支，正常路径不用 SVD）。各向异性 W_i、尺度、以及 40 根代数本身均未测试。

## 2. 设置

* 问题：`y_i = R x_i + t + ε_i`，`x_i ~ U(-1,1)^3`，`R_gt` 均匀随机旋转，`t_gt ~ U(-0.5,0.5)^3`，`ε ~ N(0, 0.01²)`；求解 `min Σ w_i ||R x_i + t - y_i||²`。
* 权重：3 种分布（均匀 0.05–1 / 指数 / 均匀 0–1），并强制约 10% 的权重为 0；因此同时覆盖 `w_i ≥ 0`、零权重、强不对称权重。负权重/NaN/Inf 被拒绝（见 §5）。
* 规模：N ∈ {20, 100, 500}，每个 N **1000** 个独立随机问题；种子 `SeedSequence([20261127, N, trial])`。
* 计时：每个问题重复 5 次，单线程 BLAS，两个求解器**交错**调用并交替顺序；每档 5000 个单次调用样本，报告 p50/p95/p99。绝对 µs 含 Python 调用与 NumPy 开销。
* 成功判据与 `synthetic/common.py` 一致：旋转误差 < 5° 且平移误差 < 0.1 m。
* 环境：单节点 CPU（Slurm `cpu` 分区），`OMP/OPENBLAS/MKL_NUM_THREADS=1`；共享节点 wall-clock，p95 有系统噪声。

## 3. 正确性：Davenport vs Kabsch

| N | `‖R_d−R_k‖_F` p50 | p95 | max | `‖t_d−t_k‖` p50 | max | 代价超出 max | 成功率 Kabsch | 成功率 Davenport | 成功一致率 | 近退化 |
|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 1.00e-15 | 2.18e-15 | 1.07e-14 | 1.69e-16 | 2.74e-15 | 1.51e-16 | 100.0% | 100.0% | 100.0% | 0 |
| 100 | 9.91e-16 | 1.95e-15 | 8.49e-15 | 1.42e-16 | 9.04e-16 | 6.38e-16 | 100.0% | 100.0% | 100.0% | 0 |
| 500 | 9.58e-16 | 1.91e-15 | 7.77e-15 | 2.72e-16 | 1.16e-15 | 9.44e-16 | 100.0% | 100.0% | 100.0% | 0 |

逐档相对真值误差（旋转 deg / 平移 m，中位数）：

| N | Kabsch rot | Davenport rot | Kabsch trans | Davenport trans |
|---|---|---|---|---|
| 20 | 0.3180 | 0.3180 | 4.39e-03 | 4.39e-03 |
| 100 | 0.1417 | 0.1417 | 2.00e-03 | 2.00e-03 |
| 500 | 0.0600 | 0.0600 | 8.86e-04 | 8.86e-04 |

> 解读：两者在浮点意义上解的是同一个全局最优问题：旋转差 max 1.07e-14（相对旋转尺度可忽略），加权代价相对超出 max 3.73e-14，成功判定逐题一致。**这不是精度提升，而是等价性验证**；Davenport 的卖点只能是运行时/实现简洁性（正常路径无 SVD，反射分支由四元数参数化自动满足）。

## 4. 运行时：before（当前 Kabsch）/ after（候选 Davenport）

| N | Kabsch p50 (µs) | Kabsch p95 (µs) | Davenport p50 (µs) | Davenport p95 (µs) | p50 节省 (µs) | 加速 p50 | 加速 p95 | 无 det 检查 p50 (µs) |
|---|---|---|---|---|---|---|---|---|
| 20 | 37.46 | 43.19 | 39.81 | 46.99 | -2.35 | 0.94× | 0.92× | 38.10 |
| 100 | 42.68 | 48.28 | 41.88 | 48.98 | 0.79 | 1.02× | 0.99× | 40.10 |
| 500 | 65.72 | 86.75 | 51.16 | 70.15 | 14.56 | 1.28× | 1.24× | 49.42 |

> `davenport_no_detcheck` 是去掉 `det(R)>0` 安全检查的诊断列（本机实测成本 1.71–1.78 µs/次），用于说明安全网的成本；正式候选是带检查的 `davenport`。

## 5. 退化与边界校验

* **平面数据（rank-2）**：200 题、N=100、z≡0。`‖R_d−R_k‖_F` max 2.01e-15，代价超出 max 4.37e-16，成功一致率 100.0%，`det(R_kabsch)` min 1.000000，`det(R_davenport)` min 1.000000。说明在本组数据上 Kabsch 的 det 修正没有触发伪反射，两者仍等价。
* **全零权重**：`sum(w)=0` 未定义；两者都回退到均匀权重（Davenport 报告 `weight_fallback=True`），与均匀权重的解差 8.92e-16。
* **非法权重**：负权重 / NaN / Inf 均抛出 `ValueError`（`negative`: ValueError: weights must be non-negative；`nan`: ValueError: weights must be finite with a finite sum；`inf`: ValueError: weights must be finite with a finite sum）。Kabsch 基线不做该校验（上游若传入负权重会静默给出非最优解），这是候选实现的行为差异。
* **近共线**：`eig_gap=4.26e-06`，标记 `near_degenerate=False`；代价仍相同（差 3.48e-26），但旋转接近不唯一（条件数大），`‖R_d−R_k‖_F=1.25e-11` 是病态放大而非求解错误；`near_degenerate` 阈值（相对间隔 1e-10）只标记严格退化，本例未触发。
* **N=1（欠定）**：任何旋转都能拟合；`eig_gap=0`、`near_degenerate=True`，两种实现给出不同的合法极小点（差 2.83e+00），二者 `det=1`。

## 6. 判断：keep / revert

门槛（脚本内固定）：正确性要求 `max‖R_d−R_k‖_F < 1e-08`、`max|代价差| < 1e-10`、成功一致率 = 1.0、`det>0`；运行时要求 p50 几何平均加速 ≥ 1.05 且单档最差 ≥ 0.9。

| 门槛 | 结果 |
|---|---|
| 正确性 | 通过 |
| 运行时 | 通过 |
| p50 几何平均加速 | 1.072× |
| 单档最小加速 | 0.941× |
| **结论** | **keep** |

保留理由（按上表固定门槛）：数值严格等价、成功率逐题一致；p50 加速 N=20 0.94× / N=100 1.02× / N=500 1.28×；正常路径无 SVD、反射修正自动满足。注意这是**内层**加速，不代表 GNC 端到端同等加速（未测）。

## 7. 复现

```bash
cd labs/03-robust-qpeplayer/experiments/inner
PY=/home/syk/miniconda3/envs/WM/bin/python
srun --partition=cpu --time=00:20:00 --ntasks=1 --cpus-per-task=2 \
  $PY run_inner_solver.py 2>&1 | tee results/logs/inner_solver_$(date +%Y%m%d_%H%M%S).log
```

产物：`results/inner_solver.json`、`results/inner_solver.md`、`results/inner_solver.png`（本文件即由脚本写出）。

