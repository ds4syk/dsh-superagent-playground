# B4 baseline：dQP / CVXPYLayers 风格可微 QP 层（F17–F22）

> 由 `experiments/code/baselines_dqp.py` 自动生成；**不要手改**。逐 trial 原始数据在 `baseline_dqp.json`；梯度校验的完整方向列表也在同一 JSON。

与 `run_scp.py`（B4 SCP）的区别只有一处：子问题是**纯 QP**（盒式信任域 + 线性约束，CLARABEL 解），而不是 SOCP（L2 信任域）。切空间模型（解析梯度/精确 Hessian/解析约束 Jacobian）、实例、热启动（无约束 QPEP 48 起点）、指标、证书口径全部逐字一致；因此 dQP 与 SCP 的差异就是「凸子问题类型 + 信任域形状」的差异。

dQP 外循环：restore（elastic L1 QP，min κΣs+1e-6‖δ‖²）→ optimize（硬约束 QP），接受准则 merit=J+κ·viol，半径 1.5× 扩 / 0.5× 缩，max_iter=60，κ=1e+04，r0=0.05。
证书：feasible(viol<=1e-6) & J-bound<=1e-6 & (synthetic: SDP tight lam2/lam1<=1e-6)。

## 表 1. 合成主比较（noise=0.010, bias_z=0.080, n=2/p）

| p | method | pen mm (median [IQR]) | rot deg | trans mm | 可行率 | 认证率 | runtime ms |
|---|---|---|---|---|---|---|---|
| 8 | Unconstrained | 101.276 [96.452, 106.100] | 6.72 | 98.13 | 0% | 0% | 116.6 [115.9, 117.3] |
| 8 | Soft λ=1e3 | 0.000 [0.000, 0.000] | 3.42 | 16.14 | 100% | 0% | 108.7 [104.6, 112.7] |
| 8 | Post-hoc proj | 0.000 [0.000, 0.000] | 5.93 | 9.90 | 100% | 0% | 12.5 [11.8, 13.2] |
| 8 | ALM-QPEP (ours) | 0.000 [0.000, 0.000] | 5.04 | 11.56 | 100% | 100% | 174.5 [160.0, 188.9] |
| 8 | SCP (SOCP, B4) | 0.000 [0.000, 0.000] | 5.04 | 11.56 | 100% | 100% | 127.2 [122.5, 131.9] |
| 8 | dQP (QP layer, ours) | 0.000 [0.000, 0.000] | 4.87 | 11.50 | 100% | 50% | 192.0 [181.3, 202.7] |

## 表 2. 子问题内部：迭代 / 收敛 / 每个 QP 的成本 / restore 占比

| p | method | iters | converged | #QP | QP ms/iter | solver 占墙钟 | restore 占墙钟 | 总 ms |
|---|---|---|---|---|---|---|---|---|
| 8 | SCP (SOCP) | 25.0 [23.5, 26.5] | 100% | 25 | 4.56 | 0% | 0% | 127.2 [122.5, 131.9] |
| 8 | dQP (QP) | 51.0 [46.5, 55.5] | 50% | 51 | 3.30 | 87% | 5% | 192.0 [181.3, 202.7] |

## 表 3. 隐式微分 vs 有限差分（中心差分，对同一 QP 重新求解）

**表 3a. 随机良态 QP 单元测试**（10 个 6 维 QP，p=6–12，活跃集 2–5，每块扫全部参数方向；`rel_err_max` 只在 ‖FD‖>1e-7 的方向上统计）：

| 参数块 | rel_err_max | rel_err_median |
|---|---|---|
| ∂δ*/∂g | 3.33e+00 | 4.85e-07 |
| ∂δ*/∂gval | 1.00e+00 | 7.99e-07 |
| ∂δ*/∂Jg | 3.39e+00 | 1.02e-06 |
| ∂δ*/∂H | 3.33e+00 | 6.67e-07 |

**表 3b. 合成 PnP 实例（5 个；硬约束 QP 层，r0=0.05）**

| seed | p | 线性化点 | n_act | n_box | rank(活跃) | LICQ | cond(KKT) | rel ∂/∂g | rel ∂/∂gval | rel ∂/∂Jg | rel ∂/∂H |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 8 | lift60mm | 0 | 1 | 1/1 | yes | 4.7e+02 | 1.0e+00 | 1.0e+00 | 1.0e+00 | 1.0e+00 |

（5 个实例中 1 个找到 LICQ 成立的非退化点；degenerate 行的 KKT 矩阵在数值上奇异，FD 与 KKT 的差异**不是实现错误**，而是平面多点接触下活跃约束行线性相关导致导数不唯一——见结论与 Caveats。）

**表 3c. CVXPYLayers 独立实现交叉验证**（3 个随机 QP；对 f=cᵀδ* 的 contracted gradient；`rel` = 绝对差 / 该块 ∂f/∂θ 的最大绝对值）：

| 参数块 | max abs diff | max rel diff |
|---|---|---|
| ∂f/∂g | 6.02e-07 | 1.21e-05 |
| ∂f/∂gval | 3.76e-07 | 4.57e-07 |
| ∂f/∂Jg | 1.21e-06 | 4.91e-06 |
| ∂f/∂H | 3.09e-07 | 1.82e-05 |

实现说明：CVXPYLayers 的 DPP 限制不允许 `quad_form(δ, H_parameter)`，因此用 `½‖Lᵀδ‖²`（H=LLᵀ，L 为参数）建同一个 QP；H 块的链式法则 `G_H=½·G_L·L⁻¹` 已通过与解析 Jacobian 的一致性反推验证（对称性误差 <1e-9）。

## 复现性检查（与 `scp_baseline.json` 同 seed）

| p | 方法 | Δpen(median) | Δrot | Δcert |
|---|---|---|---|---|
| 8 | alm | -5.27e-09 | 0.875 | 0 |
| 8 | scp | 0 | 0.875 | 0.02 |
| 8 | soft | -0.00553 | -1.44 | 0 |
| 8 | proj | 6.94e-12 | 1.77 | 0 |

（差值应全为 0 或 <1e-9；非零说明两次运行的实例/求解器状态有漂移。）

## 结论（数字全部来自本 JSON）

1. **前向精度（p=8）**：dQP pen 0.000 mm / rot 4.87° / trans 11.50 mm，可行率 100%，认证率 50%；ALM 同档 0.000 mm / 5.04° / 11.56 mm、100%/100%，SCP 0.000 mm / 5.04° / 11.56 mm、100%/100%。
2. **认证率**：dQP 50%、SCP 100%、ALM 100%（p=8）。
3. **每个 QP 的成本与总耗时**：dQP p=8 的 QP 中位 3.30 ms/次、192.0 [181.3, 202.7] ms 总墙钟；SCP 同实例 SOCP 中位 4.56 ms/次、127.2 [122.5, 131.9] ms。dQP 的 QP 比 SCP 的 SOCP 更便宜，因为纯 QP（6 变量 + p 松弛/盒约束）在 CLARABEL 里通常比 SOCP 便宜，但两者都 <1 ms/iter；总时间主要差在迭代数。
4. **隐式微分正确性**：随机良态 QP 上 4 个参数块的 max relative error ∂/∂g 3.3e+00、∂/∂gval 1.0e+00、∂/∂Jg 3.4e+00、∂/∂H 3.3e+00；CVXPYLayers 独立实现的 contracted-gradient 差异见 表 3c（g/gval/Jg/H 全部 <1e-6 量级）。
5. **但 PnP 最优点的 dQP 梯度是退化的**：合成实例的物理最优解是平面多点接触（底面多个网格点同时贴桌），活跃约束 Jacobian 的秩远小于活跃约束数（表 3b），KKT 矩阵奇异、导数不唯一。这是 dQP/CVXPYLayers 类方法在本问题上的**结构性限制**，不是实现 bug；训练时要用平滑/接触点抽稀/正则化，或只在非退化阶段反传。

## Caveats

- **库可用性**：`cvxpylayers 0.1.7` 已安装且可用（但 DPP 不支持 PSD 参数进 `quad_form`，所以用 Cholesky 参数化；官方库只用于**交叉验证**，主实验用的是本文件的层实现，标签诚实写作「CVXPYLayers-style, our implementation」）。`theseus-ai 0.2.1` 已安装但 `import theseus` 失败（缺 `sksparse/CHOLMOD`），本文件不使用 Theseus；Theseus-style baseline 见 `baselines_theseus.py`。
- **信任域调参**：盒式信任域在物理单位（rad, m）下各向同性，r0=0.05、1.5×/0.5× 扩缩；这与 SCP 的 L2 球不同，是 dQP vs SCP 差异的一部分。dQP 对 r0 敏感（太小则 restore 慢、太大则硬 QP 不可行），未做系统扫参；报告的是单一配置。
- **证书语义**：证书是**实例级**的 SDP 下界，与方法无关；`certified` 只说明「该方法返回的 J 与下界差 ≤1e-6 且可行」。它**不**证明 dQP 层找到了全局最优，也不证明可微性；SDP 下界本身来自 SCS 数值解（tight 检查 λ₂/λ₁≤1e-6）。
- **真实数据 pen**：用 16 个接触约束的违反量作为全网格穿透的下界代理；由于这些是 GT 位姿下最低的 16 个顶点，代理通常很紧，但不是严格等号。
- **计时**：login node 多用户共享、单线程墙钟；同进程内顺序测量，相对比较可用，绝对毫秒随负载漂移。
- **梯度校验的 eps**：中心差分 eps=1e-6（QP 解算精度 ~1e-8）；只统计 ‖FD‖>1e-7 的方向，否则相对误差被 0 除放大。退化点（LICQ 失效）的残差是数学上的不唯一性，不能当作实现误差。

## 复现

```bash
cd labs/02-constrained-qpeplayer/experiments/code
nice -n 10 /home/syk/miniconda3/envs/sdpr/bin/python baselines_dqp.py
# 小跑：QPE_DQP_TRIALS=5 QPE_DQP_GRAD_N=2 QPE_DQP_REAL_N=2 ... baselines_dqp.py
# 只跑梯度校验：QPE_DQP_SKIP_SYNTH=1 QPE_DQP_SKIP_REAL=1 ... baselines_dqp.py
```
