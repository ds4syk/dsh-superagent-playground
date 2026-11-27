# Robust-QPEPLayer 关键推导

> 本文件给 `theory/proofs.md` 的**中间步骤**，面向实现与审稿复核。目标：把每个公式从“定义”推到“可直接写代码”的形式，并标出源文档缺项/符号约定。
>
> 阅读顺序建议：§1（BR 对偶）→ §2（GNC 不动点）→ §3（黎曼 Hessian）→ §4（RHS）→ §5（观测敏感度与 $\lambda_\parallel$）→ §6（$O(r^{-4})$）→ §7（实现配方）。

---

## 1. Black–Rangarajan 对偶的完整配方

### 1.1 问题

给定核 $\rho:\mathbb R_{\ge0}\to\mathbb R_{\ge0}$，考虑
$$
\min_{z\in\mathcal M}\ F(z)=\sum_{i=1}^N\rho(e_i(z)),\qquad e_i(z)\ge0. \tag{1.1}
$$
BR 对偶引入 $w_i\in[0,1]$，把 $\rho(e_i)$ 写成
$$
\rho(e_i)=\min_{w_i\in[0,1]}\big\{w_ie_i+\psi(w_i)\big\}, \tag{1.2}
$$
其中 $\psi$ 是 $\rho$ 的凹共轭（concave conjugate）：
$$
\psi(w)=\inf_{e\ge0}\big\{\rho(e)-we\big\}. \tag{1.3}
$$
于是
$$
\min_z\sum_i\rho(e_i)
=\min_{z,\,w\in[0,1]^N}\sum_i\big\{w_ie_i(z)+\psi(w_i)\big\}. \tag{1.4}
$$
**关键**：这是**值**的精确等价，不是算法等价。交替最小化只是求解 (1.4) 的一种非凸方法。

### 1.2 GM 的闭式

GM：$\rho(e)=c^2e/(c^2+e)$。设
$$
\psi(w)=c^2(\sqrt w-1)^2=c^2(w-2\sqrt w+1). \tag{1.5}
$$
对固定 $e$，$\varphi(w)=we+\psi(w)$：
$$
\varphi'(w)=e+c^2\left(1-\frac{1}{\sqrt w}\right),\qquad
\varphi''(w)=\frac{c^2}{2}w^{-3/2}>0. \tag{1.6}
$$
令 $\varphi'=0$：
$$
\frac{c^2}{\sqrt w}=e+c^2
\Longrightarrow
\sqrt w=\frac{c^2}{c^2+e}
\Longrightarrow
w^*=\left(\frac{c^2}{c^2+e}\right)^2. \tag{1.7}
$$
对 GM，
$$
\rho'(e)=\frac{c^4}{(c^2+e)^2}=w^*. \tag{1.8}
$$
对偶值：
$$
\begin{aligned}
\varphi(w^*)&=w^*e+c^2(\sqrt{w^*}-1)^2\\
&=w^*(e+c^2)-2c^2\sqrt{w^*}+c^2\\
&=\frac{c^4}{(c^2+e)^2}(c^2+e)-\frac{2c^4}{c^2+e}+c^2\\
&=\frac{c^4}{c^2+e}-\frac{2c^4}{c^2+e}+c^2\\
&=c^2-\frac{c^4}{c^2+e}
=\frac{c^2(c^2+e)-c^4}{c^2+e}
=\frac{c^2e}{c^2+e}
=\rho(e). \quad\checkmark
\end{aligned} \tag{1.9}
$$
带 $\mu$：$c^2\to a=\mu c^2$，得
$$
\psi_\mu(w)=a(\sqrt w-1)^2,\qquad
w^*=\left(\frac{a}{a+e}\right)^2,\qquad
\rho_\mu(e)=\frac{ae}{a+e}. \tag{1.10}
$$

### 1.3 其它核的闭式（逐步）

**(a) Cauchy**：$\rho(e)=a\log(1+e/a)$，$\rho'(e)=a/(a+e)=w$。反解
$$
e=a\left(\frac1w-1\right),\qquad
\rho(e)=a\log(1/w)=-a\log w.
$$
$$
\psi(w)=\rho(e(w))-w e(w)
=-a\log w-w\,a\left(\frac1w-1\right)
=-a\log w-a+aw
=a(w-1-\log w). \tag{1.11}
$$
$\psi''(w)=a/w^2>0$。

**(b) Welsch**：$\rho(e)=a(1-e^{-e/a})$，$\rho'(e)=e^{-e/a}=w$。反解
$$
e=-a\log w,\qquad \rho(e)=a(1-w).
$$
$$
\psi(w)=a(1-w)-w(-a\log w)=a(1-w+w\log w). \tag{1.12}
$$
$\psi''(w)=a/w>0$。

**(c) TLS**：$\rho(e)=\min(e,a)$。对 $w\in[0,1]$，
$$
we+a(1-w)=a+w(e-a). \tag{1.13}
$$
若 $e<a$，最小在 $w=1$，值 $e$；若 $e>a$，最小在 $w=0$，值 $a$。故 $\psi(w)=a(1-w)$，$\psi''=0$，$w^*$ 在 $e=a$ 处不唯一（次梯度）。GNC-TLS 用二值权重，且其 $\mu$-代理不是简单的带宽缩放，而是 Yang et al. (2020) 的截断代理。

**(d) GM 的 $\psi''$ 与 $1/\psi''$**：
$$
\psi(w)=a(w-2\sqrt w+1),\qquad
\psi'(w)=a\left(1-\frac1{\sqrt w}\right),\qquad
\psi''(w)=\frac{a}{2}w^{-3/2}. \tag{1.14}
$$
在 $w^*=(a/(a+e))^2$：
$$
\psi''(w^*)=\frac{a}{2}\left(\frac{a+e}{a}\right)^3
=\frac{(a+e)^3}{2a^2}. \tag{1.15}
$$
而
$$
\rho''(e)=-\frac{2a^2}{(a+e)^3}, \tag{1.16}
$$
故
$$
\psi''(w^*)=-\frac{1}{\rho''(e)}. \tag{1.17}
$$
这是 §2 Schur 补的关键。

### 1.4 多观测的可交换性

$$
\min_{w\in[0,1]^N}\sum_i\varphi_{e_i}(w_i)
=\sum_i\min_{w_i\in[0,1]}\varphi_{e_i}(w_i)
=\sum_i\rho(e_i). \tag{1.18}
$$
因为每个 $w_i$ 只出现在自己的项里，约束也是箱约束（笛卡尔积）。因此 (1.4) 成立。

---

## 2. GNC 不动点与原始鲁棒代价稳定点

### 2.1 交替下降的单调性

固定 $\mu$，$J_\mu(z,w)=\sum_i[w_ie_i(z)+\psi_\mu(w_i)]$。交替：
$$
w^{(k)}=\arg\min_w J_\mu(z^{(k)},w),\qquad
z^{(k+1)}=\arg\min_z J_\mu(z,w^{(k)}). \tag{2.1}
$$
则
$$
J_\mu(z^{(k+1)},w^{(k+1)})
\le J_\mu(z^{(k+1)},w^{(k)})
\le J_\mu(z^{(k)},w^{(k)}). \tag{2.2}
$$
第一个 $\le$ 是 $w$-步最优；第二个 $\le$ 是 $z$-步最优。又
$$
J_\mu(z,w^*(z))=\sum_i\rho_\mu(e_i(z))=F_\mu(z), \tag{2.3}
$$
故 $F_\mu(z^{(k)})=J_\mu(z^{(k)},w^{(k)})\downarrow$。$F_\mu\ge0$，收敛。

### 2.2 不动点 $\Rightarrow$ 稳定点

在不动点 $(z^*,w^*)$：
$$
w_i^*=\rho_\mu'(e_i(z^*)),\qquad
0=\nabla_zJ_\mu(z^*,w^*)=\sum_iw_i^*g_i(z^*). \tag{2.4}
$$
$\mu=1$ 时 $w_i^*=\rho'(e_i(z^*))$，故
$$
\Phi(z^*)=\sum_i\rho'(e_i(z^*))g_i(z^*)=0. \tag{2.5}
$$
这就是 $F_{\mathrm{rob}}$ 的切空间一阶条件。

### 2.3 反向：稳定点不一定是 GNC 不动点

$\Phi(z^*)=0$ 只说明 $z^*$ 是加权代价 $\sum_iw_i^*e_i$ 的**稳定点**，不一定是**全局最小**。GNC 的 $z$-步要求全局最小（QPEP 给全局），故 GNC 不动点集合是“稳定点 ∩ 加权全局最小”。这个区别在论文中必须写清楚：**GNC 的输出是比稳定点更强的点，但仍不是 $F_{\mathrm{rob}}$ 的全局最小。**

### 2.4 Schur 补

把 $J(z,w)=\sum_iw_ie_i+\sum_i\psi(w_i)$ 在 $(z^*,w^*)$ 展开。切空间坐标下：
$$
\nabla^2_{zz}J=\sum_iw_i^*H_i=H_{\mathrm{IRLS}}, \tag{2.6}
$$
$$
\nabla^2_{w_iw_j}J=\delta_{ij}\psi''(w_i^*),\qquad
\nabla^2_{w_iz}J=g_i^\top. \tag{2.7}
$$
故
$$
H_J=\begin{bmatrix}H_{\mathrm{IRLS}}&D\\D^\top&\Psi''\end{bmatrix},\qquad
D=[g_1,\dots,g_N]. \tag{2.8}
$$
Schur 补：
$$
H_J/\Psi''
=H_{\mathrm{IRLS}}-D\Psi''^{-1}D^\top
=H_{\mathrm{IRLS}}-\sum_i\frac{g_ig_i^\top}{\psi''(w_i^*)}. \tag{2.9}
$$
用 (1.17)：
$$
\frac{1}{\psi''(w_i^*)}=-\rho''(e_i^*), \tag{2.10}
$$
故
$$
H_J/\Psi''
=H_{\mathrm{IRLS}}+\sum_i\rho''(e_i^*)g_ig_i^\top
=H_{\mathrm{rob}}. \tag{2.11}
$$
这就是“降权曲率修正 = 消去 $w$ 的 Schur 补”的完整推导。

---

## 3. 切空间/黎曼 Hessian 下 $H_{\mathrm{rob}}$ 的精确形式

### 3.1 球面单约束的拉格朗日乘子

$\mathbb S^3$ 上的约束 $c(\mathbf q)=\frac12(\mathbf q^\top\mathbf q-1)=0$，$\nabla c=\mathbf q$。记
$$
\lambda=\mathbf q^\top\nabla_{\mathbf q}F. \tag{3.1}
$$
它是约束的法向系数；在稳定点处就是 Lagrange 乘子，而下面的 Hessian 公式对任意点都用这个 $\lambda$。这个 $\lambda$ 就是源文档里的 $\lambda_{\mathbb S^3}$：
$$
\lambda_{\mathrm{rob}}=\mathbf q^\top G_q
=\mathbf q^\top\sum_i\rho'(e_i)g_{i,q}^E
=\sum_i\rho'(e_i)\lambda_i. \tag{3.2}
$$
**源文档定义对了 $\lambda$，但在 $H_{\mathrm{rob}}$ 公式里没有用它。**

### 3.2 球面 Hessian 的完整推导

设 $f:\mathbb R^4\to\mathbb R$，$g_q=\nabla f$，$H_{qq}=\nabla^2f$。在 $\mathbb S^3$ 上，黎曼梯度
$$
\operatorname{grad}f=Pg_q,\qquad P=I-\mathbf q\mathbf q^\top. \tag{3.3}
$$
黎曼 Hessian 是 $\operatorname{Hess}f[\xi]=P(D\operatorname{grad}f[\xi])$，$\xi\perp\mathbf q$。计算
$$
D(Pg_q)[\xi]
=P H_{qq}\xi+(DP[\xi])g_q.
$$
而
$$
DP[\xi]=\frac{\mathrm d}{\mathrm dt}\Big(I-(\mathbf q+t\xi)(\mathbf q+t\xi)^\top\Big)\Big|_{t=0}
=-\xi\mathbf q^\top-\mathbf q\xi^\top. \tag{3.4}
$$
故
$$
(DP[\xi])g_q=-\xi(\mathbf q^\top g_q)-\mathbf q(\xi^\top g_q). \tag{3.5}
$$
$P$ 作用：
$$
P(DP[\xi])g_q=-\xi(\mathbf q^\top g_q)-P\mathbf q(\xi^\top g_q)
=-\xi(\mathbf q^\top g_q), \tag{3.6}
$$
因为 $P\mathbf q=0$。于是
$$
\operatorname{Hess}f[\xi]
=P H_{qq}\xi-(\mathbf q^\top g_q)\xi
=P\big(H_{qq}-(\mathbf q^\top g_q)I\big)\xi. \tag{3.7}
$$
在切空间上 $P\xi=\xi$，写成算子
$$
\operatorname{Hess}f=P\big(H_{qq}-(\mathbf q^\top g_q)I\big)P. \tag{3.8}
$$
**检查**：$f=\frac12\|\mathbf q\|^2$，$g_q=\mathbf q$，$H_{qq}=I$，$\lambda=1$，故 $\operatorname{Hess}f=P(I-I)P=0$。$\checkmark$

### 3.3 乘积流形 $\mathbb S^3\times\mathbb R^3$

$z=(\mathbf q,\mathbf t)$，$\Pi=\operatorname{diag}(P,I_3)$，$D=\operatorname{diag}(I_4,0_3)$。欧氏梯度 $G=(G_q,G_t)$，欧氏 Hessian
$$
H_E=\begin{bmatrix}H_{qq}&H_{qt}\\H_{tq}&H_{tt}\end{bmatrix}. \tag{3.9}
$$
黎曼 Hessian：
$$
\operatorname{Hess}F
=\Pi H_E\Pi-(\mathbf q^\top G_q)\operatorname{diag}(P,0_3). \tag{3.10}
$$
在切空间正交基 $B=\operatorname{diag}(B_q,I_3)$ 下：
$$
H_F=B^\top H_E B-(\mathbf q^\top G_q)\operatorname{diag}(I_3,0_3). \tag{3.11}
$$
注意 $B^\top\Pi=B^\top$ 与 $B^\top\operatorname{diag}(P,0)B=\operatorname{diag}(I_3,0)$。

### 3.4 对 $F_{\mathrm{rob}}$ 的展开

$$
G=\sum_i\rho'(e_i)g_i^E,\qquad
H_E=\sum_i\Big(\rho'(e_i)H_i^E+\rho''(e_i)g_i^E(g_i^E)^\top\Big). \tag{3.12}
$$
故
$$
H_{\mathrm{rob}}
=B^\top\Big[\sum_i\rho'H_i^E+\sum_i\rho''g_i^E(g_i^E)^\top\Big]B
-\lambda_{\mathrm{rob}}\operatorname{diag}(I_3,0_3). \tag{3.13}
$$
若用黎曼 $H_i,g_i$：
$$
H_{\mathrm{rob}}
=\sum_i\rho'H_i+\sum_i\rho''g_ig_i^\top. \tag{3.14}
$$
两者逐项相同，因为
$$
H_i=B^\top H_i^EB-\lambda_i\operatorname{diag}(I_3,0_3),\qquad
g_i=B^\top g_i^E. \tag{3.15}
$$
**实现建议**：用 (3.13) 更直接（不需要先算每个 $e_i$ 的黎曼 Hessian）；用 (3.14) 更优雅（论文陈述）。数值上两者应一致到机器精度（`verify/riemann_hessian_check.py` 中 `H_alt` 与 `H_corr` 差 $2.3\times10^{-15}$）。

### 3.5 3D-3D 的 $H_i^E$ 显式

$\mathbf r_i=\mathbf u_i+\mathbf t-\mathbf y_i$，$\mathbf u_i=R(\mathbf q)\mathbf p_i$，$J_i=\partial\mathbf u_i/\partial\mathbf q$。

梯度：
$$
g_{i,q}^E=2J_i^\top W_i\mathbf r_i,\qquad
g_{i,t}^E=2W_i\mathbf r_i. \tag{3.16}
$$
Hessian：
$$
H_{i,qq}^E=2J_i^\top W_iJ_i+2\sum_{k=1}^3(W_i\mathbf r_i)_k\,\nabla^2_{\mathbf q}(\mathbf u_i)_k, \tag{3.17}
$$
$$
H_{i,qt}^E=2J_i^\top W_i,\qquad
H_{i,tq}^E=2W_iJ_i,\qquad
H_{i,tt}^E=2W_i. \tag{3.18}
$$
其中 $\nabla^2_{\mathbf q}(\mathbf u_i)_k$ 是常数矩阵（因为 $\mathbf u_i$ 对 $\mathbf q$ 二次）。

**$A_k$ 技巧**：把 $\mathbf u_i$ 写成二次型
$$
(\mathbf u_i)_k=\mathbf q^\top A_{i,k}\mathbf q,\qquad A_{i,k}\in\mathbb R^{4\times4}\text{ 对称}. \tag{3.19}
$$
则
$$
J_{i,k}=2A_{i,k}\mathbf q,\qquad
\nabla^2_{\mathbf q}(\mathbf u_i)_k=2A_{i,k}. \tag{3.20}
$$
于是
$$
H_{i,qq}^E=2J_i^\top W_iJ_i+4\sum_{k=1}^3(W_i\mathbf r_i)_kA_{i,k}. \tag{3.21}
$$
$A_{i,k}$ 可由 $\mathbf u_i$ 在基向量上的取值精确重建：
$$
(A_{i,k})_{ll}=(\mathbf u_i(\mathbf e_l))_k,\qquad
(A_{i,k})_{lm}=\frac{(\mathbf u_i(\mathbf e_l+\mathbf e_m))_k-(\mathbf u_i(\mathbf e_l))_k-(\mathbf u_i(\mathbf e_m))_k}{2}. \tag{3.22}
$$
这是 `verify/riemann_hessian_check.py` 的实现方式。

---

## 4. $\partial\Phi/\partial\theta$ 的两项分解

### 4.1 一般推导

$$
\Phi(z,\theta)=\sum_i\rho'(e_i(z,\theta))\,g_i(z,\theta). \tag{4.1}
$$
固定 $z$，对 $\theta$ 求偏导：
$$
\frac{\partial\Phi}{\partial\theta}
=\sum_i\left[\rho''(e_i)\frac{\partial e_i}{\partial\theta}g_i
+\rho'(e_i)\frac{\partial g_i}{\partial\theta}\right]. \tag{4.2}
$$
写成矩阵：
$$
\frac{\partial\Phi}{\partial\theta}
=\sum_i\rho'(e_i)\frac{\partial g_i}{\partial\theta}
+\sum_i\rho''(e_i)g_i\left(\frac{\partial e_i}{\partial\theta}\right)^\top. \tag{4.3}
$$
维度：$g_i\in\mathbb R^6$，$\partial g_i/\partial\theta\in\mathbb R^{6\times p}$，$\partial e_i/\partial\theta\in\mathbb R^{1\times p}$，故两项都是 $6\times p$。

**第一项**：冻结权重项。它就是 Naive IRLS-StopGrad 会保留的项。
**第二项**：隐式权重敏感度项。它来自 $w_i=\rho'(e_i)$ 随 $\theta$ 变化；冻结权重丢掉它。

### 4.2 特例：$\theta=\mathbf y_i$（3D-3D）

$\partial\mathbf r_i/\partial\mathbf y_i=-I_3$，$\partial e_i/\partial\mathbf y_i=-2\mathbf r_i^\top W_i$，$\partial g_{i,t}/\partial\mathbf y_i=-2W_i$，$\partial g_{i,q}/\partial\mathbf y_i=-2J_i^\top W_i$。代入：
$$
\frac{\partial\Phi_t}{\partial\mathbf y_i}
=\rho''(2W_i\mathbf r_i)(-2\mathbf r_i^\top W_i)+\rho'(-2W_i)
=-2\big(\rho'W_i+2\rho''W_i\mathbf r_i\mathbf r_i^\top W_i\big). \tag{4.4}
$$
$$
\frac{\partial\Phi_q}{\partial\mathbf y_i}
=\rho''(2J_i^\top W_i\mathbf r_i)(-2\mathbf r_i^\top W_i)+\rho'(-2J_i^\top W_i)
=-2J_i^\top W_i\big(\rho'I+2\rho''\mathbf r_i\mathbf r_i^\top W_i\big). \tag{4.5}
$$

### 4.3 如果 $W_i=W_i(\theta)$

$e_i=\mathbf r_i^\top W_i(\theta)\mathbf r_i$，$g_i$ 也依赖 $W_i$。RHS 变为
$$
\frac{\partial\Phi}{\partial\theta}\Big|_{\text{总}}
=\frac{\partial\Phi}{\partial\theta}\Big|_{W\text{ fixed}}
+\sum_i\rho'(e_i)\frac{\partial g_i}{\partial W_i}\frac{\partial W_i}{\partial\theta}
+\sum_i\rho''(e_i)g_i\left(\mathbf r_i^\top\frac{\partial W_i}{\partial\theta}\mathbf r_i\right). \tag{4.6}
$$
实现上，如果 $W_i$ 由网络输出，最简单是把 $W_i$ 也当作 $\theta$ 的一部分，用自动微分算 $\partial g_i/\partial W_i$，或手写 log-Cholesky 的链式。论文主实验建议先固定 $W_i$ 或只学尺度，避免这一项干扰核心贡献。

---

## 5. $\partial\Phi_t/\partial y_i$ 与 $\lambda_\parallel(e_i)$

### 5.1 各向同性情形

$W_i=I_3$：
$$
\frac{\partial\Phi_t}{\partial\mathbf y_i}
=-2\big(\rho'I+2\rho''\mathbf r\mathbf r^\top\big). \tag{5.1}
$$
这是对称矩阵，特征向量为 $\hat{\mathbf r}=\mathbf r/\|\mathbf r\|$ 与两个正交方向。沿 $\hat{\mathbf r}$：
$$
-2(\rho'+2\rho''\|\mathbf r\|^2)=-2(\rho'+2e\rho'')=-\lambda_\parallel(e). \tag{5.2}
$$
正交方向：
$$
-2\rho'. \tag{5.3}
$$
其中
$$
\lambda_\parallel(e)=2(\rho'+2e\rho'')
=2\left[\frac{c^4}{(c^2+e)^2}-\frac{4c^4e}{(c^2+e)^3}\right]
=\frac{2c^4(c^2+e-4e)}{(c^2+e)^3}
=\frac{2c^4(c^2-3e)}{(c^2+e)^3}. \tag{5.4}
$$
**符号**：$\lambda_\parallel>0$ 当 $e<c^2/3$；$=0$ 当 $e=c^2/3$；$<0$ 当 $e>c^2/3$。

### 5.2 各向异性情形

令 $\tilde{\mathbf r}=W_i^{1/2}\mathbf r_i$，$e=\|\tilde{\mathbf r}\|^2$。则
$$
W_i^{1/2}\frac{\partial\Phi_t}{\partial\mathbf y_i}W_i^{-1/2}
=-2\big(\rho'I+2\rho''\tilde{\mathbf r}\tilde{\mathbf r}^\top\big). \tag{5.5}
$$
故在 $W_i$ 度量下，广义特征值同样是 $-\lambda_\parallel(e)$ 与 $-2\rho'$。这验证了源文档“各向异性时在 $W_i^{1/2}$ 度量下完全一致”的说法。

### 5.3 从映射到 6D 位姿：为什么不能只看 $\lambda_\parallel$

$$
\frac{\partial z^*}{\partial\mathbf y_i}
=-H_{\mathrm{rob}}^{-1}\frac{\partial\Phi}{\partial\mathbf y_i}. \tag{5.6}
$$
若 $H_{\mathrm{rob}}$ 与 $\partial\Phi/\partial\mathbf y_i$ 可交换（例如 $H_{\mathrm{rob}}=hI$ 的标量情形，或二者共享特征基），则沿 $\hat{\mathbf r}$ 的位姿敏感度是
$$
\frac{\lambda_\parallel(e)}{h}, \tag{5.7}
$$
此时 $\lambda_\parallel<0$ 直接给出负敏感度。一般 6D 中 $H_{\mathrm{rob}}^{-1}$ 混合方向，符号由
$$
\hat{\mathbf r}^\top\frac{\partial z^*}{\partial\mathbf y_i}\hat{\mathbf r}
=-\hat{\mathbf r}^\top H_{\mathrm{rob}}^{-1}\frac{\partial\Phi}{\partial\mathbf y_i}\hat{\mathbf r} \tag{5.8}
$$
决定，不能只看 $\lambda_\parallel$。**一维标量情形是 (5.7) 的特例**（$H_{\mathrm{rob}}=h$，$\partial\Phi/\partial y=-\lambda_\parallel$）。

### 5.4 一维标量模型的完整推导

$F(t)=\sum_i\rho((t-y_i)^2)$，$e_i=(t-y_i)^2$。
$$
\Phi(t)=\sum_i2\rho'(e_i)(t-y_i). \tag{5.9}
$$
$$
\frac{\partial\Phi}{\partial t}
=\sum_i\left[2\rho'(e_i)+2\rho''(e_i)\cdot2(t-y_i)^2\right]
=\sum_i\left[2\rho'+4e_i\rho''\right]
=\sum_i\lambda_\parallel(e_i). \tag{5.10}
$$
$$
\frac{\partial\Phi}{\partial y_j}
=2\rho''(e_j)\cdot(-2(t-y_j))(t-y_j)+2\rho'(e_j)(-1)
=-2\rho'(e_j)-4e_j\rho''(e_j)
=-\lambda_\parallel(e_j). \tag{5.11}
$$
故
$$
\frac{\mathrm dt^*}{\mathrm dy_j}
=-\frac{\partial\Phi/\partial y_j}{\partial\Phi/\partial t}
=\frac{\lambda_\parallel(e_j)}{\sum_i\lambda_\parallel(e_i)}. \tag{5.12}
$$
Naive（冻结 $w_i=\rho'(e_i^*)$）：
$$
\Phi_{\mathrm{naive}}(t)=\sum_i2w_i(t-y_i),\qquad
\frac{\partial\Phi_{\mathrm{naive}}}{\partial t}=\sum_i2w_i,\qquad
\frac{\partial\Phi_{\mathrm{naive}}}{\partial y_j}=-2w_j, \tag{5.13}
$$
$$
\left(\frac{\mathrm dt^*}{\mathrm dy_j}\right)_{\mathrm{naive}}
=\frac{2w_j}{\sum_i2w_i}
=\frac{2\rho'(e_j)}{\sum_i2\rho'(e_i)}>0. \tag{5.14}
$$
数值：见 `verify/ift_1d_signflip_output.txt`，精确 $-0.0278147$，naive $+0.0102918$，FD $-0.0278147$，相对误差 $1.2\times10^{-8}$。

### 5.5 $\lambda_\parallel$ 的极值与渐近

直接对 (5.4) 的闭式求导：
$$
\frac{\mathrm d}{\mathrm de}\left[\frac{c^2-3e}{(c^2+e)^3}\right]
=\frac{-3(c^2+e)^3-(c^2-3e)\cdot3(c^2+e)^2}{(c^2+e)^6}
=\frac{-3(c^2+e)-3(c^2-3e)}{(c^2+e)^4}
=\frac{6(e-c^2)}{(c^2+e)^4}. \tag{5.15}
$$
故
$$
\lambda_\parallel'(e)=\frac{12c^4(e-c^2)}{(c^2+e)^4}. \tag{5.16}
$$
极小点 $e=c^2$，$\lambda_\parallel(c^2)=-1/2$；极大点在边界 $e=0$，$\lambda_\parallel(0)=2$。渐近：
$$
\lambda_\parallel(e)=\frac{2c^4(-3e)(1+o(1))}{e^3}
=-\frac{6c^4}{e^2}(1+o(1)). \tag{5.17}
$$
所以 log-log 斜率 $-4$（因为 $e=\|\mathbf r\|^2$，$e^{-2}=\|\mathbf r\|^{-4}$）。数值拟合 $-3.9979$（`verify/ift_1d_signflip_output.txt`）。

---

## 6. $O(\|\mathbf r\|^{-4})$ 的严格界

### 6.1 各块的显式界

由 (4.4)：
$$
\left\|\frac{\partial\Phi_t}{\partial\mathbf y_i}\right\|_F
\le2\left(\rho'\|W_i\|+2|\rho''|\|W_i\|^2\|\mathbf r_i\|^2\right). \tag{6.1}
$$
GM：
$$
\rho'=\frac{c^4}{(c^2+e)^2},\qquad
|\rho''|\|\mathbf r\|^2=\frac{2c^4e}{(c^2+e)^3}. \tag{6.2}
$$
故
$$
\left\|\frac{\partial\Phi_t}{\partial\mathbf y_i}\right\|_F
\le2\overline w\frac{c^4}{(c^2+e)^2}
+8\overline w^2\frac{c^4e}{(c^2+e)^3}. \tag{6.3}
$$
用 $e/(c^2+e)\le1$：
$$
\frac{c^4e}{(c^2+e)^3}
=\frac{c^4}{(c^2+e)^2}\cdot\frac{e}{c^2+e}
\le\frac{c^4}{(c^2+e)^2}. \tag{6.4}
$$
故
$$
\left\|\frac{\partial\Phi_t}{\partial\mathbf y_i}\right\|_F
\le(2\overline w+8\overline w^2)\frac{c^4}{(c^2+e)^2}. \tag{6.5}
$$
四元数块：
$$
\left\|\frac{\partial\Phi_q}{\partial\mathbf y_i}\right\|_F
\le2\|J_i\|\left(\overline w\rho'+2\overline w^2|\rho''|e\right)
\le(2\bar J\overline w+4\bar J\overline w^2)\frac{c^4}{(c^2+e)^2}. \tag{6.6}
$$
合并：
$$
\left\|\frac{\partial\Phi}{\partial\mathbf y_i}\right\|_F
\le C_1\frac{c^4}{(c^2+e)^2},\qquad
C_1=2\overline w+8\overline w^2+2\bar J\overline w+4\bar J\overline w^2. \tag{6.7}
$$
因为 $c^2+e\ge e=\|\mathbf r\|^2$，
$$
\frac{c^4}{(c^2+e)^2}\le\frac{c^4}{\|\mathbf r\|^4}. \tag{6.8}
$$
于是
$$
\left\|\frac{\partial z^*}{\partial\mathbf y_i}\right\|_F
\le\|H_{\mathrm{rob}}^{-1}\|\left\|\frac{\partial\Phi}{\partial\mathbf y_i}\right\|_F
\le M C_1\frac{c^4}{\|\mathbf r_i\|^4}. \tag{6.9}
$$

### 6.2 前向偏差的 $O(\|\mathbf r\|^{-3})$

单观测对 $F_{\mathrm{rob}}$ 的梯度贡献：
$$
\|\rho'g_i\|_F
\le\rho'\cdot2\overline w\|\mathbf r_i\|
=\frac{2\overline wc^4\|\mathbf r_i\|}{(c^2+\|\mathbf r_i\|^2)^2}
\le\frac{2\overline wc^4}{\|\mathbf r_i\|^3}. \tag{6.10}
$$
故不动点位移 $\delta z\approx-H_{\mathrm{rob}}^{-1}\rho'g_i$ 是 $O(\|\mathbf r\|^{-3})$。这比梯度敏感度 $O(\|\mathbf r\|^{-4})$ 弱一阶。**所以“梯度免疫”比“前向免疫”更强；不能反过来说前向不受影响。**

### 6.3 一致有界性的条件

(B2) 要求 $H_{\mathrm{rob}}$ 在路径上一致正定。充分条件（可作 P1 问题）：
- 剩余观测（去掉第 $i$ 个）的加权问题在 $K$ 上满足二阶充分条件，且 $\lambda_{\min}(H_{\mathrm{rob}}^{(i)})\ge\mu_0$；
- 第 $i$ 个观测的 $\rho'H_i$ 与 $\rho''g_ig_i^\top$ 在 $\|\mathbf r_i\|\to\infty$ 时趋于 $0$（由 $\rho'\to0$、$\rho''\|\mathbf r\|^2\to0$）；
- 故 $H_{\mathrm{rob}}=H_{\mathrm{rob}}^{(i)}+o(1)$，一致正定保持。
这个论证可以作为 P1-4 的证明草图；严格化需要 $H_i$ 的有界性与 $K$ 的紧性。$\square$

---

## 7. 实现配方（可直接写 PyTorch autograd.Function）

### 7.1 前向

```
输入: p_i, y_i, W_i, c, μ schedule, θ
1. 初始化 q, t (identity / previous / multi-start)
2. μ ← μ0 = 3 max_i e_i / c^2
3. while μ >= 1:
     for inner in range(K):
        e_i ← ||R(q)p_i + t - y_i||^2_{W_i}
        w_i ← (μ c^2 / (μ c^2 + e_i))^2
        (q,t) ← QPEP_global(Σ w_i e_i)   # 代数全局
     μ ← max(1, μ/γ)
4. 最终 μ=1，得到 q*, t*, w* = ρ'(e*)
5. 缓存 q*, t*, w*, e*（以及可选的 H_rob）
```

### 7.2 反向（手写）

```
输入: 上游梯度 ḡ_q, ḡ_t; 缓存 q*, t*, w*, e*
1. 对每个 i:
     u_i = R(q*)p_i
     r_i = u_i + t* - y_i
     J_i = ∂u_i/∂q          # 3x4
     g_i^E = [2 J_i^T W_i r_i; 2 W_i r_i]   # 7
     H_i^E = [[2J^TWJ + 4Σ_k(Wr)_k A_k, 2J^TW],
              [2WJ, 2W]]                    # 7x7
2. ρ'_i, ρ''_i
3. G = Σ ρ'_i g_i^E
   H_E = Σ (ρ'_i H_i^E + ρ''_i g_i^E g_i^{E,T})
4. B_q = orthonormal basis of {q*}^⊥  (4x3)
   B = blkdiag(B_q, I_3)               # 7x6
5. H_rob = B^T H_E B - λ_rob diag(I_3,0)
   λ_rob = q*^T G_q
6. RHS = Σ ρ'_i ∂g_i/∂θ + Σ ρ''_i g_i (∂e_i/∂θ)^T
   （θ=y_i 时用 (4.4)-(4.5)）
7. δz = solve(H_rob, -RHS)            # 6xp
8. 把 δz 映射回 (q,t) 切空间，链式到上游梯度
```

### 7.3 二阶证书与数值保护

```
1. 计算 H_rob 的特征值
2. if λ_min < ε: 记录 warning；减小学习率/加阻尼；或回退到 FD/局部模型
3. if λ_min > 0: 正常反向；可把 λ_min 作为 diagnostic 记录
4. if 条件数 > 1e8: 用 Cholesky 失败 → 伪逆/阻尼；记录
```

### 7.4 复杂度

| 项 | 前向 | 反向 |
|---|---|---|
| 每轮 GNC | $O(N)$ 累加 + 固定维数 QPEP | — |
| 反向 | — | $O(N)$ 组装 + $O(6^3)$ 解 |
| 显存 | $O(N)$（$w,e$） | $O(N)$（$g_i,H_i$ 可重算，不必全存） |
| 迭代展开 | 不展开 | 不展开（$O(1)$ 对 $T$） |

> 注：若为了省显存，反向可以只存 $z^*,w^*$，重新计算 $e_i,g_i,H_i$（$O(N)$ 时间），显存 $O(N)$ 而非 $O(N\cdot\text{feat})$。

---

## 8. 与源文档的符号约定对照

| 源文档 | 本文件 | 说明 |
|---|---|---|
| $\nabla_{\mathbf z}e_i$ “含切空间投影” | $g_i$（黎曼梯度） | 梯度投影对；Hessian 不能只投影 |
| $\nabla^2_{\mathbf z\mathbf z}e_i$ | $H_i$（黎曼 Hessian）或 $H_i^E$ | 必须明确；欧氏版本需加 $-\lambda_i\operatorname{diag}(I_3,0)$ |
| $\lambda_{\mathbb S^3}^*=\mathbf q^{*T}\sum_iw_i^*\nabla_{\mathbf q}e_i^*$ | $\lambda_{\mathrm{rob}}=\mathbf q^\top G_q=\sum_i\rho'\lambda_i$ | 定义一致；源文档未代入 $H_{\mathrm{rob}}$ |
| $\rho''<0$，$\Delta H_{\mathrm{redesc}}\preceq0$ | 同 | 正确（黎曼梯度外积） |
| $\partial\Phi/\partial\theta$ 两项 | 同 (4.3) | 正确；注意第二项是外积转置 |
| $\lambda_\parallel=2(\rho'+2e\rho'')$ | 同 (5.4) | 正确 |
| $O(\|\mathbf r\|^{-4})$ | 同 (6.9) | 需 $H^{-1}$ 一致有界；前向 $O(\|\mathbf r\|^{-3})$ |

---

## 9. 最小数值核对清单（对应 `verify/`）

1. **BR 对偶**：$\min_w[we+\psi(w)]=\rho(e)$，$w^*=\rho'(e)$；`ift_1d_signflip.py` §[0]，误差 $10^{-11}$。
2. **$\lambda_\parallel$ 闭式**：数值 vs 闭式 $6.7\times10^{-16}$；零点 $c^2/3$；极小 $-1/2$ 在 $c^2$。
3. **一维 sign flip**：精确 vs FD $1.2\times10^{-8}$；naive 误差 $137\%$；符号相反。
4. **$O(r^{-4})$**：log-log 斜率 $-3.9979$。
5. **黎曼 Hessian**：源公式 vs FD $2.59\times10^{-1}$；修正公式 $9.9\times10^{-8}$；常数测试 $H_{\mathrm{doc}}=I_3$ vs $H_{\mathrm{corr}}=0$。
6. **$\lambda_i$**：$4\mathbf u_i^\top W_i\mathbf r_i$ 与 $\mathbf q^\top g_{i,q}$ 差 $7.1\times10^{-15}$；与 $4e_i$ 差 $51.0$。

> 结论：源文档的 BR 对偶、$w^*$、$\lambda_\parallel$ 闭式、$O(r^{-4})$ 渐近**正确**；$H_{\mathrm{rob}}$ 公式**缺 $-\lambda_{\mathrm{rob}}$ 项**；6D 变号**未证**；GNC 全局性**过度声明**。
