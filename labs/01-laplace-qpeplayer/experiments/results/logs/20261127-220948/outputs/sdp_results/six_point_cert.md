# 6 点物理反例：数值对偶证书（numerical certificate (float), not exact rational）

> **一句话结论：** 对 6 点实例，三个 SDP 的数值区间为 $f^\star_{10}\in[5.41499990,\,5.41500000]$、$f^\star_{13,\rm full}\in[5.41499990,\,5.41500000]$、$f^\star_{13,\rm up}\in[3.59431363,\,3.59431383]$；后者与满 20D 值的间隔落在 $[1.82068607,\,1.82068637]$。这是**浮点数值证书**，不是有理数精确证书。

## 0. 口径、求解器与安全裕度

- 对偶（下界）求解器：**CLARABEL 0.11.1**（内点法，cvxpy 1.7.5）；原始（上界/秩）求解器：**SCS 3.3.1**；原始值另用 CLARABEL 交叉检查。
- 下界：直接求解对偶 $\max\mu$ s.t. $Z=Q-\mu E_{00}+\sum_i y_iA_i\succeq0$，取 $\mu-10^{-7}$。
- PSD 安全裕度：$m_{\rm psd}=10^{-8}$；用 $\lambda_{\min}(Z)$ 与 $Z+m_{\rm psd}I$ 的 Cholesky 分解双重核验。
- 数值安全裕度：$\varepsilon_{\rm safe}=10^{-7}$，取在区间两个端点外侧；它远大于观测到的原始-对偶差（$\le10^{-9}$）与两求解器原始值差（$\le2\times10^{-8}$）。
- 秩阈值：$\lambda_i>10^{-4}\,\lambda_{\rm max}$。
- 上界：$f^\star_{10}$、$f^\star_{13,\rm full}$ 用**精确可行点** $X=xx^\top$，$x=[1,\mathrm{vec}(I),0]$，其目标值为闭式 $6(1-\varepsilon)^2=1083/200=5.415$（sympy 精确核对）；$f^\star_{13,\rm up}$ 的最优解 rank 4，没有构造精确可行点，上界取原始求解值加 $\varepsilon_{\rm safe}$。

## 1. 区间表

| 量 | 约束（独立维数） | rank(X*) | 数值区间 `[lower, upper]`（numerical certificate (float), not exact rational） | 宽度 | 上界来源 |
|-|-|-|-|-|-|
| $f^\star_{10}$ | 21 (20D) | 1 | [5.41499990, 5.41500000] | 1.00e-07 | exact rational 1083/200 at the exactly feasible rank-1 point X = x x^T, x = [1, vec(I), 0] |
| $f^\star_{13,\rm full}$ | 27 (20D) | 1 | [5.41499990, 5.41500000] | 1.00e-07 | exact rational 1083/200 at the exactly feasible rank-1 point X = x x^T, x = [1, vec(I), 0] |
| $f^\star_{13,\rm up}$ | 24 (17D) | 4 | [3.59431363, 3.59431383] | 2.00e-07 | primal solver value + value safety margin (X numerically feasible) |

派生区间（同一证书的区间算术）：

- $f^\star_{10}-f^\star_{13,\rm up}\in[1.82068607,\,1.82068637]$（宽度 3.00e-07）；包含点值 1.82068627；**不包含**证明引用的 1.82069114（相差约 4.8e-6），与正文“末 5 位未由我们的构造复现”一致。
- $f^\star_{10}-f^\star_{13,\rm full}\in[-1.00e-07,\,1.00e-07]$，包含 0，与满 20D 等价的结论一致。

## 2. 对偶松弛的 PSD 核验（裕度已声明）

| 量 | $\lambda_{\min}(Z)$ | $\lambda_{\min}(Z+m_{\rm psd}I)$ | Cholesky | $\langle Z,X^\star\rangle$ | $\lambda_{\min}(X^\star)$ | 最大约束残差 | 原始-对偶差 |
|-|-|-|-|-|-|-|-|
| 10 | -5.87e-12 | +9.99e-09 | OK | -2.26e-11 | +1.23e-13 | 9.88e-15 | -2.27e-11 |
| 13_full | +7.08e-13 | +1.00e-08 | OK | +3.26e-12 | -8.05e-15 | 1.55e-15 | +3.26e-12 |
| 13_up | -4.96e-12 | +1.00e-08 | OK | -4.05e-12 | -4.61e-13 | 6.66e-16 | -4.04e-12 |

读法：$\lambda_{\min}(Z)$ 的负值都在 $m_{\rm psd}=10^{-8}$ 以内，$Z+10^{-8}I$ 的 Cholesky 分解全部成功，故对偶松弛在数值精度内 PSD；$\langle Z,X^\star\rangle$ 与原始-对偶差一致，说明互补松弛在 ~1e-10 内闭合。

## 3. 求解器交叉检查

| 量 | SCS 原始值（主） | CLARABEL 原始值（校验） | 两原始值差 | CLARABEL 对偶 $\mu$ |
|-|-|-|-|-|
| 10 | 5.4150000000 | 5.4150000025 | 2.5e-09 | 5.4150000000 |
| 13_full | 5.4150000000 | 5.4149999989 | 1.1e-09 | 5.4150000000 |
| 13_up | 3.5943137329 | 3.5943137133 | 2.0e-08 | 3.5943137329 |

两个独立求解器的原始值一致到 ≤2e-8（f10 2.5e-9，13_up 2.0e-8），且 SCS 原始值与 `results/six_point.json` 逐位一致；下界来自 CLARABEL 对偶。因此区间端点不是单一求解器或单次运行的伪影。

## 4. 必须保留的 caveat

- This is a floating-point certificate, not exact rational arithmetic: the dual multipliers and the slack matrix are stored as doubles, and the lower endpoint is the solver dual value minus the stated value safety margin.
- The revised proof quotes f*_13up = 3.59430886; both CLARABEL and SCS give 3.59431373 (difference 4.9e-6, above solver tolerance), so that quoted value lies OUTSIDE the interval [3.59431363, 3.59431383]. The O(1) separation from f10 (gap ~1.82) is unaffected.
- rank(X*_13up) = 4 at the stated 1e-4 relative threshold (four O(1) eigenvalues, then numerical dust at ~5e-6); it is 6 at a 1e-6 threshold, but the theorem only needs rank > 1.
- For f10 and f13_full the upper endpoint is the exact rational 1083/200 at the exactly feasible X = x x^T; the float evaluation of the same point with the assembled float Q is 5.414999999999999. The solver primal values for f10/f13_full can sit up to ~3e-9 above 5.415 because the returned X is only numerically feasible; they are reported as a check only and are not used for the upper endpoint.

## 5. 复现

```bash
cd labs/01-laplace-qpeplayer/experiments/sdp
/home/syk/miniconda3/envs/WM/bin/python run_6point_check.py
/home/syk/miniconda3/envs/WM/bin/python run_six_point_cert.py
```

总耗时 0.5 s。原始数据：`results/six_point_cert.json`；复核值：`results/six_point.json`（`run_6point_check.py`）。
