# analysis/theory — SDPRLayers (TRO 2025) Q1–Q3 数值验证

对应笔记：[../../notes/theory_Q1_Q3.md](../../notes/theory_Q1_Q3.md)（中文，含公式与本文所有结果表格）。

## 环境

Python 3.10+，需要 numpy / scipy / cvxpy / matplotlib；SDP 求解器用 SCS（主）与 Clarabel（对照），
**不需要 Mosek license**。

## 运行

```bash
cd analysis/theory && bash run_all.sh
```

单个实验：

| 脚本 | 内容 | 主要输出 |
|---|---|---|
| `exp1_certificate.py` | §VI-A 六次多项式：tight / $H\succeq0$ / corank=1 / SOSC；$\theta_3$ 扫描与全局极小切换；tie（两个全局极小相等）边界；随机 $\theta$ 统计 | `results/exp1_certificate.json`, `figures/fig1_certificate.png` |
| `exp2_jacobian.py` | Thm.2 的 $J=-PM_r^\dagger N_r$ vs 解析真值 / SDP-有限差分 / SDPR-CIFT；$G_r$ 选取不变性；对偶面 268 个证书的 $J$ 不变性；$dX$ 一致性；$N$ 的约束块一致性；tie 分支 Jacobian | `results/exp2_jacobian.json`, `figures/fig2_jacobian.png` |
| `exp3_redundancy.py` | 冗余约束下经典 $M$ 奇异（$\mathrm{cond}\sim10^{17}$–$10^{66}$）、$M_r$ 行满秩与伪逆公式、乘子重算（LS/CIFT）、行选择不变性 | `results/exp3_redundancy.json`, `figures/fig3_redundancy.png` |

## 文件

- `sdpr_poly.py`：核心库。$Q_\theta$ 与 $A_0..A_3$（按论文 §VI-A 的矩阵定义；原工作区曾用论文第 9 页图像核对，图像不随仓库分发）、SDP 求解与对偶证书提取、证书诊断（PSD/corank/SOSC）、Thm.2 与 CIFT Jacobian、解析真值、FD、对偶面采样。
- `results/*.json`：全部数值结果（含中间量），笔记中数字均可在此追溯。
- `figures/*.png`：笔记中嵌入的三张图。

## 关键结论（详见笔记 §4）

- 基准 $\theta_0$：rank-1 比值 $6.6\times10^{14}$、$\mathrm{corank}(H)=1$、$\|Hx\|=2.6\times10^{-14}$、SOSC margin $0.252$。
- Thm.2 Jacobian（SCS 证书）vs 解析真值：$2.7\times10^{-14}$；SDP 有限差分 $h=10^{-5}$ 时 $1.4\times10^{-9}$。
- 冗余约束（1 个冗余）：$\mathrm{cond}(M)=2.4\times10^{17}$ 而 $\mathrm{cond}(M_r)=72$；伪逆公式误差 $2.7\times10^{-14}$；$G$ 行任意独立子集给出同一 $J$（偏差 $2\times10^{-14}$）。
- 对偶最优面上 268 个不同证书给出同一 $x$-Jacobian（偏差 $4.4\times10^{-14}$）。
- tie 点（$\theta_3^*=-0.6659084$）：$X^*$ 秩 2（不 tight），$\mathrm{corank}(H)=2$，两个全局极小都被同一证书认证；给定分支点，Thm.2 仍给出该分支正确 Jacobian（$10^{-13}$）。
