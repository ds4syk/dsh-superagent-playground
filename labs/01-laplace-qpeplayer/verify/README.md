# verify/ — 最小数值 sanity check

两个脚本，均为**最小可复现检查**，不是 benchmark，也不产生论文结果。所有输出存 JSON，结论写在本文件。

## 环境说明（重要）

任务说明写 `/home/syk/miniconda3/bin/python` 已装 numpy/scipy/sympy/matplotlib。实测（2026-11-27）：

- base `/home/syk/miniconda3/bin/python`：有 numpy 2.5.4、sympy 1.14.0、matplotlib，但**没有 scipy**，也没有 cvxpy；
- `sdpr-clean` 环境 `/home/syk/miniconda3/envs/sdpr-clean/bin/python`：numpy 2.4.6、scipy 1.17.1、sympy 1.14.0、cvxpy 1.9.3（SCS/CLARABEL）。

因此脚本用 `sdpr-clean` 运行。若要在 base 运行，需先 `pip install scipy cvxpy`。

## 运行

```bash
cd labs/01-laplace-qpeplayer/verify
/home/syk/miniconda3/envs/sdpr-clean/bin/python verify_theorem0.py
/home/syk/miniconda3/envs/sdpr-clean/bin/python verify_theorem11.py
```

## `verify_theorem0.py` — 定理 0（旋转子问题）

**一句话结论**：$x_R=2Uv_2$ 与 $U^\top U=I$ 精确成立；21 条 $SO(3)$ 约束秩 20，且其张成空间**等于** $\nu_2(\mathbb{RP}^3)$ 的二次消没理想；构造的 Choi–Lam + $2I$ 实例有**认证的** SDP 间隙 $0.034188$、$\operatorname{rank}(X^\star)=6$。

关键输出（`verify_theorem0_output.json`）：

| 项 | 值 |
|-|-|
| `U_orthogonal_exact` | `true` |
| `xR_equals_2Uv2_exact` | `true` |
| `all_21_vanish_on_xR_exact` | `true` |
| `rank_21_constraints_v_coords_exact` | `20` |
| `rank_veronese_degree4_map` | `35` |
| `dim_ideal_degree2` | `20` |
| `rank_constraints_plus_kernel` | `20` |
| `constraints_equal_ideal_degree2` | `true` |
| `SDP_value` | `1.9658119658452384` |
| `dual_mu` | `1.9658119658123265` |
| `dual_Z_min_eig` | `-1.79e-11`（PSD 容差内） |
| `dual_certificate_PSD` | `true` |
| `certified_gap_lower_bound` | `0.0341880341877` |
| `SDP_X_rank_1e-6` | `6` |
| `SDP_X_eig_ratio_12` | `1.4509` |
| `QPEP_true_min_analytic` | `2.0`（$C\ge0$，$C(e_0)=0$） |
| `all_perturbed_gaps_positive` | `true`（8 个随机小扰动） |

**注意**：
- 这是**构造的 10×10 旋转子问题 SDP**，不是 SDPRLayers 的 13×13 实现；不能标成“SDPRLayer 原实现”。
- Choi–Lam 实例是我们自己的构造（$M_{CL}+2I$），不是源文档的扰动；源文档的 $1.98/0.01773/1.561$ 未复现。
- 间隙 $0.034188$ 是“真值 2.0 − 对偶认证值 1.9658”，不是 primal-dual 差。
- “正测度”只验证了 $M$ 空间的小扰动（8 个），不是可实现 PnP Gram 锥内的正测度；见 `math_questions.md` Q1/Q9。

## `verify_theorem11.py` — 定理 1.1 + Laplace 常数

**一句话结论**：$E(q)$ 正交、Euler 恒等、黎曼 Hessian、隐式 $q/t$ 导数、对跖不变性全部通过；旋转 Laplace 的相对误差随 $\tau$ 线性下降（斜率 1.05），即 $O(\tau)$，**不是精确积分**。

关键输出（`verify_theorem11_output.json`）：

| 项 | 值 |
|-|-|
| `E_orthonormal_err` | $1.1\times10^{-16}$ |
| `E_orthogonal_to_q_err` | $4.4\times10^{-18}$ |
| `E_antipodal_err` | $0$ |
| `projected_grad_norm` | $2.1\times10^{-10}$ |
| `euler_identity_abs_err` | $5.3\times10^{-15}$ |
| `H_fd_vs_analytic_relerr` | $9.6\times10^{-8}$ |
| `dq_implicit_vs_fd_rel` | $1.5\times10^{-7}$ |
| `dt_implicit_vs_fd_rel` | $4.6\times10^{-8}$ |
| `H_antipodal_rel_err` | $0$ |
| `dq_antipodal_relation_rel` | $0$ |
| `dR_antipodal_rel_err` | $0$ |
| `laplace_rel_errors`（$\tau=0.2,0.1,0.05,0.025$） | $8.26\times10^{-2}, 3.81\times10^{-2}, 1.86\times10^{-2}, 9.25\times10^{-3}$ |
| `laplace_error_order_slope` | $1.05$ |

**注意**：
- 有限差分误差受优化器精度限制（$\epsilon=10^{-4}$，中心差分），$10^{-7}$ 已足够确认公式；不是“证明”。
- Laplace 表是在 $S^3$ 的局部 chart 上做的重要性采样；$SO(3)$ 的 Haar 测度：局部 $dR=8dq$（每片），二重覆盖给 $1/2$，故 $\int_{SO(3)}=4\int_{S^3}$，每个 $SO(3)$ 模式合计 $8I$；见 `theory/derivations.md` §8。
- 该表支持“旋转 Laplace 误差 $O(\tau)$”，**不支持**“旋转精确积分”。源文档的“精确”只对平移成立。
