# verify/：最小数值 sanity check

> 环境：`/home/syk/miniconda3/bin/python`（Python 3.14.8；numpy 2.5.4、sympy 1.14.0、matplotlib 3.11.2）。**scipy 未安装**（`ModuleNotFoundError`），与任务说明不符；两个脚本只用 numpy + matplotlib，root finding 用二分，不依赖 scipy。

---

## 脚本 1：`ift_1d_signflip.py`

**检查**：一维 GM 模型 $F(t)=\sum_i\rho((t-y_i)^2)$ 上的 BR 对偶、$\lambda_\parallel$ 闭式、精确 IFT vs 冻结权重 naive vs 中心有限差分、$O(\|r\|^{-4})$ 斜率。

**输出**：`ift_1d_signflip_output.txt`、`ift_1d_signflip.png`。

**一句话结论**：BR 对偶值误差 $10^{-11}$；$\lambda_\parallel$ 闭式误差 $6.7\times10^{-16}$、零点 $c^2/3$、极小 $-1/2$ 在 $c^2$；一维精确导数 $-0.0278147$ 与 FD 一致到 $1.2\times10^{-8}$，naive 为 $+0.0102918$（误差 $137\%$，符号相反）；尾斜率 $-3.9979$，验证 $O(\|r\|^{-4})$。

**关键数字**：

| 项 | 结果 |
|---|---|
| $\min_w[we+\psi(w)]-\rho(e)$（$e=0.01,0.25,1,10,10^3$） | $\le1.5\times10^{-11}$ |
| $\max|\lambda_{\mathrm{num}}-\lambda_{\mathrm{closed}}|$ | $6.66\times10^{-16}$ |
| $t^*$（$y=[0,1.5]$，$c=0.5$） | $0.015437730142$ |
| $e$ | $[2.383\times10^{-4},\,2.203925]$ |
| $H_{\mathrm{rob}}$ | $1.934772552898>0$ |
| exact $\mathrm dt^*/\mathrm dy_2$ | $-0.027814707778$ |
| naive | $+0.010291820095$ |
| FD | $-0.027814708105$；相对误差 $1.18\times10^{-8}$ |
| naive 相对误差 | $1.37$（$137\%$） |
| 尾斜率 | $-3.997905$ |

---

## 脚本 2：`riemann_hessian_check.py`

**检查**：3D-3D、$N=6$、GM、$c=1.0$ 的 $F_{\mathrm{rob}}$ 在 $\mathbb S^3\times\mathbb R^3$ 上的黎曼 Hessian。比较：
- 源文档公式 $H_{\mathrm{doc}}=B^\top[\sum\rho'H_i^E+\sum\rho''g_i^Eg_i^{E,T}]B$；
- 修正公式 $H_{\mathrm{corr}}=H_{\mathrm{doc}}-\lambda_{\mathrm{rob}}\operatorname{diag}(I_3,0)$；
- 测地线二阶差分 $H_{\mathrm{fd}}$；
- 另验证 $\lambda_i=\mathbf q^\top g_{i,q}=4\mathbf u_i^\top W_i\mathbf r_i$。

**输出**：`riemann_hessian_check_output.txt`、`riemann_hessian_check.png`。

**一句话结论**：源公式与 FD 的相对 Frobenius 误差 $2.587\times10^{-1}$，修正公式 $9.88\times10^{-8}$；$\lambda_{\mathrm{rob}}=1.0329$；$\lambda_i$ 与 $4\mathbf u_i^\top W_i\mathbf r_i$ 差 $7.1\times10^{-15}$，与 $4e_i$ 差 $51.05$；常数测试 $f=\frac12\|\mathbf q\|^2$ 的源公式给 $I_3$、修正给 $0$。

**关键数字**：

| 项 | 结果 |
|---|---|
| $c$ / $N$ / $q_0$ | $1.0$ / $6$ / $[-0.018684,0.51724,-0.263015,0.814209]$ |
| $e_i$ | $[0.425759,9.949492,20.131661,3.072509,6.943894,27.301985]$ |
| $\lambda_{\mathrm{rob}}$ | $+1.055853$ |
| $\|H_{\mathrm{doc}}-H_{\mathrm{fd}}\|_F/\|H_{\mathrm{fd}}\|_F$ | $2.587\times10^{-1}$ |
| $\|H_{\mathrm{corr}}-H_{\mathrm{fd}}\|_F/\|H_{\mathrm{fd}}\|_F$ | $1.458\times10^{-7}$ |
| $\|H_{\mathrm{alt}}-H_{\mathrm{corr}}\|_F$ | $2.34\times10^{-15}$ |
| $\max|\lambda_i-4\mathbf u_i^\top W_i\mathbf r_i|$ | $7.11\times10^{-15}$ |
| $\max|\lambda_i-4e_i|$ | $51.05$ |
| 常数测试 $\operatorname{eig}(H_{\mathrm{doc}})$ / $\operatorname{eig}(H_{\mathrm{corr}})$ | $[1,1,1]$ / $[0,0,0]$ |

---

## 失败/未覆盖

- **未复现**源文档的 6D cos$=-0.079$：缺少其精确实验配置（点云、$W_i$、$c$、扰动方向）。本仓库只把它标为“作者预实验（待独立复现）”。
- **未验证** GNC 前向的全局/分岔行为：需要真实 QPEP 求解器与更大规模实验；属 `math_questions.md` P0-2。
- **未验证** $H_{\mathrm{rob}}$ 在真实 3D-3D 数据上的 $\lambda_{\min}$ 分布：属实验 E5。
- 脚本 2 的 FD 在 $c=1.0$ 的配置下条件数较大（$H_{\mathrm{fd}}$ 不定），但相对误差仍到 $10^{-7}$；这说明公式核对有效，不代表该点是局部极小。
- 环境缺 scipy：未安装（避免联网/耗时），用二分替代 `scipy.optimize.brentq`。

---

## 复跑

```bash
cd labs/03-robust-qpeplayer
/home/syk/miniconda3/bin/python verify/ift_1d_signflip.py
/home/syk/miniconda3/bin/python verify/riemann_hessian_check.py
```

两个脚本都会覆盖同名 output/PNG；结果对固定种子/固定参数可复现。
