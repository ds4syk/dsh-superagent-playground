# 文献笔记（一般文献，未使用 Family 017 材料）

## 1. 经典 TSR/Dyson 机器

### E. Croot, *An Outline of the Thue–Siegel Theorem* (2007)

- 清晰给出 Thue–Siegel 的三步架构（正是本报告 §2 的"架构 I"）：
  1. 在 $(\alpha,\alpha)$ 处构造有小系数的整数多项式 $f$，使初始 Taylor 系数消失；
  2. 在有理点 $(p_1/q_1,p_2/q_2)$ 求值，用分母 $q_1^{N_1}q_2^{N_2}$ 给整性下界；
  3. 用 Gauss 引理/Wronskian 证明某个低阶 Taylor 系数在有理点非零。
- 关键观察（对本报告至关重要）：第 1 步用**极小多项式**把 $\alpha$ 的高次幂约化到 $\mathbb Q(\alpha)$ 的有限维空间，从而把消失条件变成**有理**线性方程组；第 3 步用 Gauss 引理（有理根的重数 ⇒ 首项系数被 $q^m$ 整除）证明非零。**两步都本质地使用 $\alpha$ 代数。**
- 对 $\pi$：第 1 步不可行（无极小多项式；$\operatorname{ind}_{(\pi,\dots,\pi)}P=0$），第 3 步的 Gauss 论证也不适用。

### J.-H. Evertse, *Approximation of algebraic numbers by rationals* (Leiden notes, Ch. 6)

- Liouville 定理（$|ξ−α|>cH(ξ)^{−d}$）的证明用**范数/乘积公式**：$F(x,y)$ 是非零整数，$|F(x,y)|\ge1$；同时 $|F(x,y)|\lesssim M(α)H(ξ)^d|ξ−α|$。
- 这正是"算术下界（整数≥1/范数）vs 解析上界"的原型；Thue/Siegel/Dyson/Roth 的改进都在这条线上。
- 与本报告的关系：对代数 α，算术下界的指数是 $d$（或 Roth 优化后的 $k−1$），**与 $\deg Q$ 无关**；对 π 没有这个范数下界。

### M. Nakamaye, *Intersection Theory and Diophantine Approximation* (JAG 1999)

- 定义 Dyson index $\operatorname{ind}(\zeta,s)=\min\{\sum α_i/d_i: b_\alpha\ne0\}$。
- Dyson 引理（Esnault–Viehweg 的改进版）：多个点上高 index 的截面条件"几乎独立"，给出 $\sum \operatorname{Vol}(t_i)\le\dots$。
- 与本报告的关系：index 是"在一点的高阶消失"的几何度量。对 $\pi$，任何非零整系数多项式在 $(\pi,\dots,\pi)$ 的 index 为 0（常数项非零），所以 Dyson/Esnault–Viehweg 的整台机器无法启动。

### E. Bombieri, W. Gubler, *Heights in Diophantine Geometry* (2006)

- Siegel 引理、Roth 定理、Dyson 引理的现代处理；Siegel 引理的标准鸽笼证明与高度估计。
- 本报告附录 A.1 的 Siegel 引理即标准形式；A.2 的 Gauss 可除性是经典事实。

## 2. π 的 irrationality measure：已知记录

| 年份 | 作者 | 上界 |
|---|---|---|
| 1953 | Mahler | 42 |
| 1993 | Hata | 8.0161 |
| 2008 | Salikhov | 7.6063 |
| 2020 | Zeilberger–Zudilin | 7.103205334137… |

- 这些上界都来自**显式线性形式/Padé 构造**（利用 $\pi=4\arctan1$ 的积分表示），不是一般插值机器。
- 下界：Dirichlet 给 $\mu(\pi)\ge2$（所有无理数）；$\mu(\pi)>2$ 未知（等价于存在比 $q^{-2}$ 更好的无穷多逼近）。
- $\mu(e)=2$ 已知（e 的连分数部分商 $1,2,1,1,4,1,1,6,\dots$ 线性增长）。这说明"某些显式超越数确实能达到 2"，但 π 的方法不同。
- 目标 $\mu(\pi)\le2$ 是"典型实数"的行为（Khintchine：a.e. $x$ 有 $\mu(x)=2$），但对 π 未知。

## 3. 与本报告 no-go 的关系

- 架构 I（代数插值）是 TSR 的核心；π 的超越性使其 index 恒 0，机器不能启动。
- 架构 II（有理插值）对任意 α 可启动，但其算术输入（对机器中所有 Q 的值下界）必须与机器自身的 $Q_0=G$ 相容；$Q_0$ 的有效指数 $\ge\kappa-1$，而目标需要 $A\le1$。这是本报告的主要障碍。
- 显式 Padé 路线绕开了"一般插值机器"，但它自己的算术下界也退化为线性形式的指数 $A_{\rm eff}\approx\mu(\pi)-1$；到 2 需要 $A_{\rm eff}=1$，即目标。

## 4. 未采用的材料

- 未读 OpenAI Family 017 的论文、推理摘要、Lean 文件。
- 未读主窗口的 `research/pi-exponent/`、PR #30、§9。
- 未搜索 Family 017 的解法或第三方解读。
- 检索中遇到一份声称 $\mu(\pi)=2$ 的非同行评议预印本（N. A. Carella, arXiv:1902.08817）；本报告**未采用**其任何内容，也不对其作评价。本报告只引用公认的 7.103 上界（Zeilberger–Zudilin 2020）。
