# Constrained-QPEPLayer：定理、假设、证明与 GAP 清单

> **结论先行**：源文档的三条定理骨架是对的，但三处表述必须改写才能提交：
> (i) 定理 2.1 的"10×10 QPEP"只是**提升形式**（内层在 $q$ 上是四次型，不是二次型），并非常义上的 QPEP；
> (ii) 定理 2.2 的"$\delta_{\rm gap}\approx 0$ ⇒ 100% 认证"不成立——$d(\mu^*)$ 由无证书的代数求解器给出，且源文档预实验中 $\delta=-1.40\times10^{-12}<0$（本文同类实例复现为 $-1.67\times10^{-11}$）已自证其为数值量；正确做法是用 **14×14 Shor 半定下界**做 sound 证书（本文新增 Lemma 2.2）；
> (iii) 定理 2.3 的"$\varepsilon>0$ 时 FB 等价于互补""$\varepsilon\to0$ 精确收敛到真方向导数"都不成立；$\varphi_\varepsilon=0$ 的零点集是双曲线 $\mu g=-\varepsilon^2/2$，极限是 **$H$-加权最小范数解**，只在相容性条件下等于真导数（Proposition 2.1）。
>
> 文中所有数值均来自本目录 `verify/` 的两个脚本，可复跑；不做无来源的断言。

---

## 0. 记号、问题与假设

### 0.1 问题（C-QPEP）

$$
\begin{aligned}
\min_{q\in\mathbb{S}^3,\;t\in\mathbb{R}^3}\quad & J_\theta(q,t) \;=\; \mathbf{v}_2(q)^\top \mathbf{M}_0(\theta)\,\mathbf{v}_2(q) + 2\,\mathbf{t}^\top \mathbf{D}_0(\theta)\,\mathbf{v}_2(q) + \mathbf{t}^\top \mathbf{G}_0(\theta)\,\mathbf{t}\\
\text{s.t.}\quad & g_j(q,t;\theta) \;=\; \tilde{\mathbf a}_j(\theta)^\top \mathbf{v}_2(q) + \mathbf{b}_j(\theta)^\top \mathbf{t} \;\le\; 0,\qquad j=1,\dots,p .
\end{aligned}\tag{C-QPEP}
$$

其中 $\mathbb{S}^3=\{q\in\mathbb R^4:\|q\|_2=1\}$，

$$
\mathbf{v}_2(q)=\big[q_0^2,\,q_1^2,\,q_2^2,\,q_3^2,\,\sqrt2 q_0q_1,\,\sqrt2 q_0q_2,\,\sqrt2 q_0q_3,\,\sqrt2 q_1q_2,\,\sqrt2 q_1q_3,\,\sqrt2 q_2q_3\big]^\top\in\mathbb R^{10},
$$

$\mathbf e_{\rm tr}=[1,1,1,1,0,\dots,0]^\top$ 满足 $\mathbf e_{\rm tr}^\top\mathbf v_2(q)=\|q\|_2^2\equiv1$（在 $\mathbb S^3$ 上）。

**约定（齐次化）**：约束里的常数项 $c_j$ 通过 $\tilde{\mathbf a}_j=\mathbf a_j+c_j\mathbf e_{\rm tr}$ 吸收，即 $c_j=c_j\,\mathbf e_{\rm tr}^\top\mathbf v_2(q)$。这一步**只在 $\mathbb S^3$ 上成立**，因此内层子问题必须保留 $\|q\|=1$ 约束，不能把 $\mathbf v_2$ 当自由变量。

### 0.2 假设

| 编号 | 内容 | 用途 | 状态 |
|---|---|---|---|
| **A1** | $\mathbf M_0,\mathbf D_0,\mathbf G_0$ 连续可微依赖于 $\theta$；$\mathbf G_0\succeq0$，且 $\mathbf G_0\succ0$ 或 $\{\mathbf b_j\}_{j\in\mathcal A}$ 张成 $\mathbb R^3$ | 内层 $t$-消去可逆 | 可检验 |
| **A2** | 内层子问题（固定 $\mu,\rho$、固定活跃集 $\mathcal A$）由 QPEP 型 Gröbner/作用矩阵求解器**返回全部实驻点**（无漏根、无重根误判） | 内层"全局" | **不可证**，只能做残差/多起点交叉验证 |
| **A3** | 活跃集在相邻外层迭代间稳定，或活跃集识别在有限步内终止 | ALM 收敛 | 数值经验，见 Lemma 2.1 |
| **A4** | 存在 $\bar\rho$，使 $\rho\ge\bar\rho$ 时 ALM 的增广拉格朗日内层极小点满足 LICQ/ACQ 或至少 MFCQ | 乘子有界、KKT 收敛 | 非凸 ALM 的标准假设 |
| **A5** | 输出点 $(\hat q,\hat t)$ 的可行性违反 $\nu:=\max_j[g_j(\hat q,\hat t)]_+\le \nu_{\rm tol}$ | 认证中的上界一侧 | 可检验 |
| **A6** | 输出乘子 $\hat\mu\ge0$ | 弱对偶 $d(\hat\mu)\le J^\star$ | ALM 结构自动满足（见 §2.1）|

> **A2 是本线最脆的一环**。它与 QPEP 原文的"全局"是同一类承诺：*枚举完备性*，不是凸证书。知识库 `kb/sdpr-qpep/03.md` §2.4 已明确指出 QPEP 的 "globally optimal" 不等于 SDPR 意义上的 "certifiably optimal"。本文的所有"全局"声明都必须显式引用 A2，或改用 Lemma 2.2 的 SDP 下界。

---

## 1. 定理 2.1（加性保持的常数维归约定理）

### 1.1 陈述

> **定理 2.1（Constant-dimension reduction）**
> 设 (C-QPEP) 满足 A1。对任意 $\mu\in\mathbb R^p_{\ge0}$、$\rho>0$ 与任意活跃集 $\mathcal A\subseteq\{1,\dots,p\}$，定义 Rockafellar–Wets 增广拉格朗日
> $$
> \mathcal L_\rho(q,t,\mu)=J_\theta(q,t)+\frac{1}{2\rho}\sum_{j=1}^p\Big(\big[\mu_j+\rho g_j(q,t)\big]_+^2-\mu_j^2\Big). \tag{1.1}
> $$
> 则
> (a) 在集合 $\{(\mathbf v_2(q),t):\ \mu_j+\rho g_j(q,t)>0\ \forall j\in\mathcal A,\ \mu_j+\rho g_j\le0\ \forall j\notin\mathcal A\}$ 上，
> $$
> \mathcal L_\rho(q,t,\mu)\Big|_{\mathcal A}=\mathbf v_2(q)^\top\mathbf M_{\mathcal A}\mathbf v_2(q)+2\,\mathbf t^\top\mathbf D_{\mathcal A}\mathbf v_2(q)+\mathbf t^\top\mathbf G_{\mathcal A}\mathbf t+\kappa_{\mathcal A}, \tag{1.2}
> $$
> 其中
> $$
> \begin{aligned}
> \mathbf M_{\mathcal A}&=\mathbf M_0+\sum_{j\in\mathcal A}\Big[\tfrac{\mu_j}{2}\big(\mathbf e_{\rm tr}\tilde{\mathbf a}_j^\top+\tilde{\mathbf a}_j\mathbf e_{\rm tr}^\top\big)+\tfrac{\rho}{2}\tilde{\mathbf a}_j\tilde{\mathbf a}_j^\top\Big],\\
> \mathbf D_{\mathcal A}&=\mathbf D_0+\sum_{j\in\mathcal A}\Big[\tfrac{\mu_j}{2}\mathbf b_j\mathbf e_{\rm tr}^\top+\tfrac{\rho}{2}\mathbf b_j\tilde{\mathbf a}_j^\top\Big],\\
> \mathbf G_{\mathcal A}&=\mathbf G_0+\tfrac{\rho}{2}\sum_{j\in\mathcal A}\mathbf b_j\mathbf b_j^\top,\qquad
> \kappa_{\mathcal A}=-\frac{1}{2\rho}\sum_{j\notin\mathcal A}\mu_j^2 .
> \end{aligned}\tag{1.3}
> $$
> 特别地 $\kappa_{\mathcal A}\le0$，且**只有非活跃约束**贡献常数项。
>
> (b) 装配 (1.3) 只需 $|\mathcal A|$ 次 $10\times10$、$3\times10$、$3\times3$ 的秩-1 累加，即 $O(p)$ 时间、$O(1)$ 内存；$\mathbf M_{\mathcal A}\in\mathbb S^{10}$、$\mathbf D_{\mathcal A}\in\mathbb R^{3\times10}$、$\mathbf G_{\mathcal A}\in\mathbb S^3_{++}$ 的尺寸与 $p$ 无关。
>
> (c) 若 $\mathbf G_{\mathcal A}\succ0$，则
> $$
> \min_{t\in\mathbb R^3}\mathcal L_\rho\Big|_{\mathcal A}=\mathbf v_2(q)^\top\big(\mathbf M_{\mathcal A}-\mathbf D_{\mathcal A}^\top\mathbf G_{\mathcal A}^{-1}\mathbf D_{\mathcal A}\big)\mathbf v_2(q)+\kappa_{\mathcal A},\qquad
> t^\star(q)=-\mathbf G_{\mathcal A}^{-1}\mathbf D_{\mathcal A}\mathbf v_2(q). \tag{1.4}
> $$
> 因此**内层子问题等价于 $\mathbb S^3$ 上的一个四次型最小化**
> $$
> \min_{q\in\mathbb S^3}\ \Phi_{\mathcal A}(q):=\mathbf v_2(q)^\top\mathbf M^{\rm red}_{\mathcal A}\mathbf v_2(q)+\kappa_{\mathcal A},\qquad \mathbf M^{\rm red}_{\mathcal A}:=\mathbf M_{\mathcal A}-\mathbf D_{\mathcal A}^\top\mathbf G_{\mathcal A}^{-1}\mathbf D_{\mathcal A}. \tag{1.5}
> $$

### 1.2 证明

**(a)** 对 $j\in\mathcal A$，展开 hinge：
$$
\tfrac{1}{2\rho}\big[(\mu_j+\rho g_j)^2-\mu_j^2\big]=\mu_j g_j+\tfrac{\rho}{2}g_j^2 .
$$
对 $j\notin\mathcal A$，$[\cdot]_+=0$，贡献 $-\mu_j^2/(2\rho)$。故
$$
\mathcal L_\rho|_{\mathcal A}=J_\theta+\sum_{j\in\mathcal A}\Big(\mu_j g_j+\tfrac\rho2 g_j^2\Big)-\frac{1}{2\rho}\sum_{j\notin\mathcal A}\mu_j^2. \tag{1.6}
$$

逐项写成 $(\mathbf v_2,t)$ 的二次型。记 $\mathbf x=\mathbf v_2(q)$。

*线性项 $\mu_j g_j=\mu_j\tilde{\mathbf a}_j^\top\mathbf x+\mu_j\mathbf b_j^\top\mathbf t$*：$\mathbf t$ 部分本身就是线性的，**滞留在 $\mathbf D$ 之外**——它不能进入 $\mathbb S^3$ 上的二次型！正确做法是利用 $\mathbf e_{\rm tr}^\top\mathbf x=1$：
$$
\mu_j\tilde{\mathbf a}_j^\top\mathbf x=\mu_j(\tilde{\mathbf a}_j^\top\mathbf x)(\mathbf e_{\rm tr}^\top\mathbf x)=\mathbf x^\top\Big[\tfrac{\mu_j}{2}\big(\tilde{\mathbf a}_j\mathbf e_{\rm tr}^\top+\mathbf e_{\rm tr}\tilde{\mathbf a}_j^\top\big)\Big]\mathbf x,
$$
$$
\mu_j\mathbf b_j^\top\mathbf t=\mu_j(\mathbf b_j^\top\mathbf t)(\mathbf e_{\rm tr}^\top\mathbf x)=2\,\mathbf t^\top\Big[\tfrac{\mu_j}{2}\mathbf b_j\mathbf e_{\rm tr}^\top\Big]\mathbf x .
$$

*二次项 $\tfrac\rho2 g_j^2=\tfrac\rho2\big[(\tilde{\mathbf a}_j^\top\mathbf x)^2+2(\tilde{\mathbf a}_j^\top\mathbf x)(\mathbf b_j^\top\mathbf t)+(\mathbf b_j^\top\mathbf t)^2\big]$*：
$$
\tfrac\rho2(\tilde{\mathbf a}_j^\top\mathbf x)^2=\mathbf x^\top\Big[\tfrac\rho2\tilde{\mathbf a}_j\tilde{\mathbf a}_j^\top\Big]\mathbf x,\quad
\rho(\tilde{\mathbf a}_j^\top\mathbf x)(\mathbf b_j^\top\mathbf t)=2\,\mathbf t^\top\Big[\tfrac\rho2\mathbf b_j\tilde{\mathbf a}_j^\top\Big]\mathbf x,\quad
\tfrac\rho2(\mathbf b_j^\top\mathbf t)^2=\mathbf t^\top\Big[\tfrac\rho2\mathbf b_j\mathbf b_j^\top\Big]\mathbf t .
$$

求和即得 (1.3)。常数项只来自 (1.6) 的最后一项，即 $\kappa_{\mathcal A}=-\frac1{2\rho}\sum_{j\notin\mathcal A}\mu_j^2$。∎

> **与源文档的差异（关键）**：源文档写"$+\text{const}$"而未给表达式。若误取 $\kappa=-\frac1{2\rho}\sum_{j=1}^p\mu_j^2$，则 (1.2) 与直接计算 (1.1) 相差 $\sum_{j\in\mathcal A}\mu_j^2/(2\rho)$。本仓库 `verify/sanity_constraints_alm.py` 的 C2 检查在错误常数下报 $5.57\times10^{-1}$ 的误差、在正确常数下为 $3.55\times10^{-15}$，说明**源文档公式在活跃集上是对的，但常数项必须按上式区分活跃/非活跃**。这是实现时极易写错、且会让 ALM 收敛到错误乘子的地方。

**(b)** 每个 $j$ 只做 3 次外积累加，显然。尺寸与 $p$ 无关。`verify` C4 实测：$p=256$ 时装配 4.37 ms（单核 numpy，纯 Python 循环），$\mathbf M/\mathbf D/\mathbf G$ 恒为 $(10,10)/(3,10)/(3,3)$。

**(c)** 对 $\mathbf t$ 求驻点：$2\mathbf G_{\mathcal A}\mathbf t+2\mathbf D_{\mathcal A}\mathbf x=0$，得 $\mathbf t^\star=-\mathbf G_{\mathcal A}^{-1}\mathbf D_{\mathcal A}\mathbf x$（线性于 $\mathbf x$，因此**二次于 $q$**）。代回并注意 $\mathbf G_{\mathcal A}$ 对称：
$$
\mathbf t^{\star\top}\mathbf G_{\mathcal A}\mathbf t^\star+2\mathbf t^{\star\top}\mathbf D_{\mathcal A}\mathbf x=-\mathbf x^\top\mathbf D_{\mathcal A}^\top\mathbf G_{\mathcal A}^{-1}\mathbf D_{\mathcal A}\mathbf x .
$$
得 (1.4)。再由 $\mathbf v_2(\lambda q)=\lambda^2\mathbf v_2(q)$ 知 $\Phi_{\mathcal A}$ 是 $q$ 的四次齐次型。∎

### 1.3 推论 2.1.1（"$10\times10$ QPEP"的正确含义与次数账）

> **推论 2.1.1**
> (a) 内层子问题 (1.5) 在 $q$ 上是**四次型**，在 $\mathbf v_2$ 上是**二次型**。它不是 $q$ 上的二次型 QPEP。
> (b) 其 KKT 系统为 4 个三次方程（$\nabla_q\Phi_{\mathcal A}-2\lambda q=0$）加 1 个二次方程（$\|q\|^2=1$），共 5 个未知量 $(q,\lambda)$；Bézout 上界 $3^4\cdot2=162$。
> (c) QPEP 原问题（$\mathbf M_{\mathcal A}=\mathbf M_0$）消 $t$ 后同为四次型、同为上述 KKT 结构；其官方求解器返回至多 40 个实/复根（`solver_WQ`，见 `kb/sdpr-qpep/03.md` §2.3）。40 是**该问题族的通用根数**，不是 Bézout 上界。

**证明**：(a) 由 (1.4)。(b) $\partial_q[\mathbf v_2^\top\mathbf M\mathbf v_2]=4\cdot$（三次），$\partial_q[2\lambda(\|q\|^2-1)]$ 线性；故为三次。未知量 5 个，次数向量 $(3,3,3,3,2)$，Bézout 乘积 $=162$。(c) 引 `kb/sdpr-qpep/03.md` 中 `solution_size=40` 的代码事实。∎

> **GAP 1（P0）**：$\mathcal A\neq\varnothing$ 时 $\mathbf M^{\rm red}_{\mathcal A}$ 的词支撑与 $\mathbf M_0$ 相同（都是全体四次单项式），但**系数的秩结构不同**（$\mathbf M_{\mathcal A}$ 是 $\mathbf M_0$ 加 $|\mathcal A|$ 个秩-1 PSD 项，且 $\mathbf D_{\mathcal A}$ 亦被扰动）。通用根数是否仍为 40、还是升到 162 或其他值，取决于该系统族的 Macaulay/BKK 界。**在验证之前不能声称"复用 QPEP 的 249×249 消元模板"**。可行的验证路径：(i) 用 sympy 对 $(\mu,\rho,\mathcal A)$ 符号化，数 $\mathbb C[q,\lambda]$ 商环的维数；(ii) 数值上对 200 组随机 $(\mathbf M_0,\mathbf D_0,\mathbf G_0,\rho,\mathcal A)$ 跑同伦/作用矩阵，统计实根数上界。见 `math_questions.md` Q2。

### 1.4 推论 2.1.2（四次型下界的两条来源）

> **推论 2.1.2**　设 $\lambda_{\min}(\mathbf M^{\rm red}_{\mathcal A})$ 为最小特征值。则对一切 $q\in\mathbb S^3$，
> $$
> \lambda_{\min}(\mathbf M^{\rm red}_{\mathcal A})\;\le\;\Phi_{\mathcal A}(q)-\kappa_{\mathcal A}.
> $$
> 且 $\min_{\mathbb S^3}\Phi_{\mathcal A}\ge\kappa_{\mathcal A}+\min\{0,\lambda_{\min}(\mathbf M^{\rm red}_{\mathcal A})\}$。

**证明**：$\mathbf x=\mathbf v_2(q)$，$\|\mathbf x\|^2=\|q\|^4=1$，故 $\mathbf x^\top\mathbf M^{\rm red}\mathbf x\ge\lambda_{\min}(\mathbf M^{\rm red})\|\mathbf x\|^2=\lambda_{\min}$。∎

这个下界是**平凡的**（对四次型极松），但它说明：内层"全局"不能靠谱下界，只能靠枚举或 SDP。这正是 Lemma 2.2 的动机。

---

### 1.5 Lemma 2.1（活跃集一致性：冻结活跃集的内层解何时是 (1.1) 的真正极小）

> **Lemma 2.1**　设 $\mathcal A_k$ 是第 $k$ 次外层迭代的活跃集，$\mathbf z_k^\star$ 是 $\mathcal A_k$-片段的极小点（即定理 2.1 的内层解）。
> (a) 若对一切 $j$ 有 $\operatorname{sign}(\mu_j+\rho g_j(\mathbf z_k^\star))$ 与 $\mathcal A_k$ 的指示函数一致（即 $\mathbf z_k^\star$ 仍在同一片内），则 $\mathbf z_k^\star$ 是 $\mathcal L_\rho(\cdot,\mu)$ 的局部极小点，且外层更新就是标准 ALM 更新；
> (b) 若不一致，则 $\mathcal A_k$ 片段不含 $\mathcal L_\rho$ 的极小点，必须重新识别活跃集；若活跃集在两个集合间循环，则 ALM 可能不收敛到 KKT 点（此时需改用 $\rho$ 增大或光滑化）。

**证明**：(a) $\mathcal L_\rho$ 在片 $\mathcal A_k$ 上等于二次型 $\mathcal L_{\mathcal A_k}$（定理 2.1(a)），而 $\mathcal L_{\mathcal A_k}$ 的极小点 $\mathbf z_k^\star$ 若落在片内，则它是 $\mathcal L_\rho$ 在一个邻域内的最小点（因为邻域内 $\mathcal L_\rho$ 或等于 $\mathcal L_{\mathcal A_k}$，或由其他片给出更大的值——后者不成立，需注意：其他片的值可能更小，故 (a) 只保证**局部**极小，且需 $\mathbf z_k^\star$ 的片邻域非空）。标准 ALM 更新 $\mu_j\leftarrow[\mu_j+\rho g_j]_+$ 恰是 $\mathcal L_\rho$ 对 $\mu$ 的上升方向，故外层即 ALM。∎
(b) 若 $\mathbf z_k^\star$ 落在片外，则它不满足片定义，不能是 $\mathcal L_\rho$ 的极小点（$\mathcal L_\rho$ 在该点的次微分不含 0）；重新识别是必要的。循环是分段二次问题的经典病态，需 $\rho$ 增大或对 hinge 做光滑化。∎

> **说明**：Lemma 2.1 是"算法正确性"级别的引理，不是深度结果。它的作用是**明确**：本文的内层不是精确子问题求解，而是"冻结活跃集 + 一致性检查"的 Gauss–Seidel 变体。论文里必须这样写，否则"每步全局求解内层 ⇒ ALM 收敛"的推理链是断的（全局解的是**片段**，不是 $\mathcal L_\rho$）。

## 2. 定理 2.2（条件全局最优性：弱对偶 + sound 下界）

### 2.1 陈述

> **定理 2.2（Conditional global optimality）**
> 设 $\hat{\mathbf z}=(\hat q,\hat t)$ 是 ALM 的输出，$\hat\mu\ge0$ 是其乘子，$\nu=\max_j[g_j(\hat{\mathbf z})]_+$。
> (a) **弱对偶（下界一侧）**：对任意 $\mu\ge0$，定义对偶函数
> $$
> d(\mu)=\min_{q\in\mathbb S^3,\,t\in\mathbb R^3}\Big\{J_\theta(q,t)+\sum_{j=1}^p\mu_j g_j(q,t)\Big\}. \tag{2.1}
> $$
> 则 $d(\mu)\le J^\star$，其中 $J^\star$ 是 (C-QPEP) 的全局最优值。
> (b) **三明治**：若 $J_\theta$ 在 $\hat{\mathbf z}$ 的 $\nu$-邻域内 $L$-Lipschitz，则
> $$
> d(\hat\mu)\;\le\;J^\star\;\le\;J_\theta(\hat{\mathbf z})+L\nu . \tag{2.2}
> $$
> 因此
> $$
> 0\;\le\;J_\theta(\hat{\mathbf z})-J^\star\;\le\;\underbrace{J_\theta(\hat{\mathbf z})-d(\hat\mu)}_{=:\delta}\;+\;L\nu . \tag{2.3}
> $$
> (c) **可靠性**：若 $\delta+L\nu\le\tau$，则 $\hat{\mathbf z}$ 是 $\tau$-全局最优。
> (d) **反面**：$\delta<0$ 在精确算术下不可能（当 A5、A6 成立），一律是数值误差；此时 $\delta$ 只能写成 $\max(0,\delta_{\rm num})+O(\text{求解容差})$。

**证明**：
(a) 对任意可行 $z$ 有 $g_j(z)\le0$ 与 $\mu_j\ge0$，故 $\sum_j\mu_j g_j(z)\le0$，于是 $J(z)+\sum\mu_jg_j(z)\le J(z)$。对可行集取下确界：$d(\mu)=\min_{\mathbb S^3\times\mathbb R^3}(\cdot)\le\min_{\text{可行}}(\cdot)\le\min_{\text{可行}}J=J^\star$。∎
(b) 右半边：由 A5，$\hat{\mathbf z}$ 到可行集的（加权）距离 $\le \nu$（每个约束的违反量以 $\nu$ 为上界，取一阶可行化即可），故存在可行 $\bar{\mathbf z}$ 使 $\|\bar{\mathbf z}-\hat{\mathbf z}\|\le C\nu$；于是 $J^\star\le J(\bar{\mathbf z})\le J(\hat{\mathbf z})+LC\nu$。把 $C$ 吸收进 $L$ 即得。∎
(c) 直接由 (2.3)。
(d) 若 A5、A6 成立，则 $d(\hat\mu)\le J^\star\le J(\hat{\mathbf z})$，故 $\delta=J(\hat{\mathbf z})-d(\hat\mu)\ge0$。矛盾 ⇒ 数值误差。∎

> **对源文档的修正**：源文档写
> $$0\le J(q^*)-J^\star_{\rm global}\le\delta_{\rm gap}:=J(q^*)-d(\mu^*),$$
> 并称 "一旦 $\delta_{\rm gap}\approx0$，即 100% 认证"。三处问题：
> 1. **方向对，但前提漏了**：需要 $\mu^*\ge0$ **且** $q^*$ 可行（A5、A6）。$q^*$ 不可行时 $J(q^*)$ 可以小于 $J^\star$，左侧不等式失效。
> 2. **"100% 认证"不成立**：$d(\mu^*)$ 由 (2.1) 的内层全局极小给出，而该内层"全局"依赖 A2（Gröbner 枚举完备性），本身没有证书。用无证书的量去做证书，逻辑上不闭合。
> 3. **负的 $\delta$ 不是小事**：源文档预实验 $\delta=-1.40\times10^{-12}$，本文 `verify` C5a 复现同类现象 $\delta=-1.67\times10^{-11}$。负值恰好说明 $d(\mu^*)$ 的数值精度不足以支撑"严格证书"。正确写法是"在求解容差 $\tau_{\rm solve}$ 内认证"，并给出 $\tau_{\rm solve}$ 的可验证估计。

### 2.2 新增 Lemma 2.2（14×14 Shor 下界：真正的 sound 证书）

这是本文相对源文档**新增**的核心结果，也是"certifiable"一词唯一站得住的依据。

> **Lemma 2.2（Veronese 增强的 14×14 Shor 下界）**
> 令 $\mathbf x=(\mathbf v_2(q),\,\mathbf t)\in\mathbb R^{13}$，$\hat{\mathbf x}=(\mathbf x,1)\in\mathbb R^{14}$，$\mathbf X=\hat{\mathbf x}\hat{\mathbf x}^\top\in\mathbb S^{14}$。考虑 SDP
> $$
> \begin{aligned}
> \mathrm{Shor}_{\rm ineq}\;=\;\min_{\mathbf X}\quad&\big\langle \hat{\mathbf Q}_\theta,\mathbf X\big\rangle\\
> \text{s.t.}\quad&\mathbf X\succeq0,\quad X_{14,14}=1,\quad \big\langle \operatorname{sym}(\hat{\mathbf e}_{\rm tr}\mathbf e_{14}^\top),\mathbf X\big\rangle=1\ (\Leftrightarrow\ \mathbf e_{\rm tr}^\top\mathbf v_2=1),\quad \operatorname{tr}(\mathbf X_{[10]})=1\ (\Leftrightarrow\|\mathbf v_2\|^2=1),\\
> &\big\langle \operatorname{sym}(\hat{\mathbf c}_j\mathbf e_{14}^\top),\mathbf X\big\rangle\le0,\ j=1..p,\\
> &\text{Veronese 二次关系（见下）},
> \end{aligned}\tag{2.4}
> $$
> 其中 $\hat{\mathbf Q}_\theta$ 是 $J_\theta$ 在 $\hat{\mathbf x}$ 上的齐次二次型矩阵（$Q$ 块加零行零列），$\hat{\mathbf c}_j=(\tilde{\mathbf a}_j,\mathbf b_j,0)$，Veronese 关系指 $\mathbf S=qq^\top$ 的全部 $2\times2$ 子式 $S_{ik}S_{jl}=S_{il}S_{jk}$ 在 $\mathbf X$ 上的线性化（例如 $2X_{01}=X_{44}$）。则
> $$
> \mathrm{Shor}_{\rm ineq}\;\le\;J^\star . \tag{2.5}
> $$
> 且若最优 $\mathbf X^\star$ 的秩为 1（$\lambda_2/\lambda_1\le\kappa$），则 $\mathbf X^\star=\hat{\mathbf x}^\star\hat{\mathbf x}^{\star\top}$ 给出 (C-QPEP) 的全局最优解与**可独立检验的证书**。

**证明**：设 $z^\star=(q^\star,t^\star)$ 是 (C-QPEP) 的任一可行点。取 $\mathbf X=\hat{\mathbf x}\hat{\mathbf x}^\top$。逐条验证可行性：$\mathbf X\succeq0$、$X_{14,14}=1$；$\operatorname{tr}(\mathbf X_{[10]})=\|\mathbf v_2(q^\star)\|^2=\|q^\star\|^4=1$；$\mathbf e_{\rm tr}^\top\mathbf v_2(q^\star)=1$；$\langle\operatorname{sym}(\hat{\mathbf c}_j\mathbf e_{14}^\top),\mathbf X\rangle=\hat{\mathbf c}_j^\top\hat{\mathbf x}=g_j(z^\star)\le0$；Veronese 关系对任何 $\mathbf v_2(q)$ 成立。目标值 $=\hat{\mathbf x}^\top\hat{\mathbf Q}\hat{\mathbf x}=J(z^\star)$。故 (2.4) 的可行集包含所有可行点的秩-1 提升，取下确界得 $\mathrm{Shor}_{\rm ineq}\le J^\star$。秩-1 时取 $\hat{\mathbf x}^\star$ 即有可行点达到 $\mathrm{Shor}_{\rm ineq}$，结合 (2.5) 得 $J(\hat{\mathbf x}^\star)=\mathrm{Shor}_{\rm ineq}\le J^\star$，故相等且 $\hat{\mathbf x}^\star$ 全局最优。∎

**数值证据**（`verify/sanity_constraints_alm.py` C5，$p=12$）：

| SDP | 值 | $\lambda_2/\lambda_1$ | 解释 |
|---|---|---|---|
| 无 Veronese、无对偶项 | $-1\times10^{-10}$ | 0.33 | 平凡界，无用 |
| $+$Veronese | **0.0030334692** | $1.4\times10^{-9}$ | 恰等于**无约束** QPEP 数值全局值 0.0030334724，且紧 |
| $+$Veronese $+$不等式 | **0.0680069355** | $2.2\times10^{-12}$ | 恰等于 ALM 输出 $J(\hat z)=0.06800693548$，**紧** |
| 用 $d(\hat\mu)$ 的下界代替 | 0.0034784073 | — | 只给 0.0645 的 sound gap，比直接 SDP 差 3 个数量级 |

三条结论：
1. **Veronese 关系是紧性的开关**：不加它，SDP 完全无效；加上它，本实例的松弛是紧的。
2. **直接对不等式问题做 SDP 优于对偶函数**：$\mathrm{Shor}_{\rm ineq}$ 与 $J(\hat z)$ 相差 $2.3\times10^{-11}$（SCS 容差量级），而 $\delta=J(\hat z)-d(\hat\mu)=-1.67\times10^{-11}$ 为负、不可用。这与源文档的路线（定理 2.2 用 $d(\mu^*)$）相反，是本线应主推的认证路径。
3. **成本与 $p$ 的关系**：PSD 块恒为 $14\times14$，不等式以 $p$ 条线性约束进入（每条 $O(14^2)$）；内点法/一阶法的总代价 $\approx O(p\cdot14^3)$ 每迭代，$p$ 大时仍远优于把 PSD 块升到 $(10+p)\times(10+p)$。实测 $p=12$ 时 SCS 0.09 s（CPU）。

> **GAP 2（P1）**：Lemma 2.2 的"sound"依赖 SDP 求解器的数值可行性，严格证书需要 (i) 对偶可行解（14×14 PSD + 乘子）显式保存并用区间算术验证 $\mathbf S\succeq0$；(ii) 目标值的舍入误差界。当前只用 SCS 默认容差，因此是"数值 sound"，应写成 $\mathrm{Shor}_{\rm ineq}\le J^\star+\tau_{\rm sdp}$，$\tau_{\rm sdp}=2.3\times10^{-11}$。
> **GAP 3（P1）**：$p$ 很大时 Veronese 关系（36 条）与不等式（$p$ 条）都作用在同一个 14×14 块上，SDP 的总约束数 $\approx p+40$。若 $p>10^4$，SDP 的约束处理成本可能超过 QPEP 内层，需实测。

---

## 3. 定理 2.3（FB 约化的隐微分：可逆性 / 极限 / 相容性）

### 3.1 FB 函数：先把"等价性"说清楚

> **Lemma 2.3（$\varphi_\varepsilon$ 的准确性质）**
> 定义 $\varphi_\varepsilon:\mathbb R^2\to\mathbb R$，
> $$
> \varphi_\varepsilon(\mu,g)=\mu-g-\sqrt{\mu^2+g^2+\varepsilon^2}. \tag{3.1}
> $$
> (a) **不是互补函数的等价重述**：$\varphi_\varepsilon(\mu,g)=0\iff \mu g=-\varepsilon^2/2$（且 $\mu-g>0$）。因此当 $\varepsilon>0$ 时零点集是双曲线，完全落在开象限内，**不是** $\{0\le\mu\perp(-g)\ge0\}$。
> (b) 在互补点上的残差：强活跃 $(\mu>0,g=0)$ 处 $\varphi_\varepsilon=-\sqrt{\mu^2+\varepsilon^2}+\mu=-\varepsilon^2/(2\mu)+O(\varepsilon^4)$；弱活跃 $(\mu,g)=(0,0)$ 处 $\varphi_\varepsilon=-\varepsilon$；非活跃 $(\mu=0,g<0)$ 处 $\varphi_\varepsilon=-\varepsilon^2/(2|g|)+O(\varepsilon^4)$。
> (c) 偏导：$\partial_\mu\varphi_\varepsilon=1-\mu/r=a$，$\partial_g\varphi_\varepsilon=-(1+g/r)=-b$，$r=\sqrt{\mu^2+g^2+\varepsilon^2}$。
> (d) 取值范围：$\mu\ge0$ 时 $a\in(0,1]$；**$g\le0$（可行）时 $b\in[0,1]$；$g>0$（迭代中不可行）时 $b\in(1,2)$**。$b<2$ 恒成立（因 $g<r$）。
> (e) 极限（固定互补点附近）：$r_j\to|\mu_j|$（强活跃）或 $|g_j|$（非活跃）或 $0$（弱活跃），故
> $$
> a_j\sim\frac{\varepsilon^2}{2\mu_j^2},\quad b_j\to1\ (\text{强活跃});\qquad a_j=1,\ b_j\sim\frac{\varepsilon^2}{2g_j^2}\to0\ (\text{非活跃});\qquad a_j=b_j=1\ (\text{弱活跃}).
> $$

**证明**：(a) $\varphi_\varepsilon=0\Rightarrow\mu-g=r>0$，平方得 $\mu^2-2\mu g+g^2=\mu^2+g^2+\varepsilon^2$，即 $\mu g=-\varepsilon^2/2$。反之若 $\mu g=-\varepsilon^2/2$ 且 $\mu-g>0$，则 $r^2=(\mu-g)^2$ 且 $r>0$，故 $r=\mu-g$，$\varphi_\varepsilon=0$。∎
(b) 强活跃：$\varphi_\varepsilon(\mu,0)=\mu-\sqrt{\mu^2+\varepsilon^2}=-\varepsilon^2/(\mu+\sqrt{\mu^2+\varepsilon^2})$。∎
(c)(d) 直接求导与放缩：$0\le\mu\le r$；$g\le0\Rightarrow -r\le g\le0\Rightarrow b\in[0,1]$；$0<g<r$ 时 $b\in(1,2)$。∎
(e) 泰勒展开。∎

**数值验证**（`verify/sanity_fb_saddle.py`，$\varepsilon=10^{-4}$）：$a\in[0,1]$、可行点 $b\le1$、不可行点 $b<2$ 全部 PASS；强活跃 $\varphi=-1.0000\times10^{-9}$ vs $-\varepsilon^2/(2\cdot5)=-1.0000\times10^{-9}$（相对差 $8\times10^{-8}$）；弱活跃 $-1.0000\times10^{-4}$；非活跃 $-1.6667\times10^{-9}$ vs $-\varepsilon^2/6$；零点集满足 $\mu g=-\varepsilon^2/2$ 到 $1.8\times10^{-15}$。

> **因此**：$\varphi_\varepsilon$ 是**对精确互补映射的 $\varepsilon$-光滑化**，不是等价重述。用它求微分，得到的是**光滑化解映射**的导数；要与真导数建立联系，必须再取 $\varepsilon\to0$ 的极限并证明该极限等于真导数（Proposition 2.1）。源文档"$\varphi_\varepsilon=0$（仅用于反向，前向保持精确无偏解）"这句话的合法性只在此意义下成立：**前向不解 $\varphi_\varepsilon$，反向只借用它的微分作为 KKT 系统的一种 $\varepsilon$-正则化**。这不是"精确"，而是"正则化 + 极限"。

### 3.2 定理 2.3（可逆性与正则化系统）

> **定理 2.3（Regularized reduced system）**
> 设 $\mathbf z=(q,t)$，切空间 $\delta\mathbf z=(\delta\boldsymbol\xi,\delta\mathbf t)\in\mathbb R^6$，$\mathbf J_{g,j}=\nabla_{\mathbf z}g_j\in\mathbb R^6$（对 $\delta\boldsymbol\xi$ 取右扰动 $\mathbf R\exp(\delta\boldsymbol\xi^\wedge)$ 的一阶系数）。
> 设活跃集 $\mathcal A=\mathcal A(\varepsilon)=\{j:a_j<\delta_{\rm act}\}$，$\mathcal R=\{1..p\}\setminus\mathcal A$。定义
> $$
> \mathbf H_{\rm tan}:=\nabla^2_{\mathbf z}\mathcal L=\nabla^2_{\mathbf z}J+\sum_{j\in\mathcal A}\mu_j\nabla^2_{\mathbf z}g_j,\qquad
> \mathbf H_{\rm eff}:=\mathbf H_{\rm tan}+\sum_{j\in\mathcal R}\frac{b_j}{a_j}\mathbf J_{g,j}^\top\mathbf J_{g,j},\qquad
> \mathbf D_a:=\operatorname{diag}_{j\in\mathcal A}\Big(\frac{a_j}{b_j}\Big). \tag{3.2}
> $$
> 则 (i) 若 $\mathbf H_{\rm eff}\succ0$ 且 $\mathbf D_a\succ0$，则鞍点矩阵
> $$
> \mathbf S=\begin{bmatrix}\mathbf H_{\rm eff}&\mathbf J_{\mathcal A}^\top\\ \mathbf J_{\mathcal A}&-\mathbf D_a\end{bmatrix}\in\mathbb R^{(6+|\mathcal A|)\times(6+|\mathcal A|)} \tag{3.3}
> $$
> **恒可逆，与 $\mathbf J_{\mathcal A}$ 的秩无关**；
> (ii) 其逆的显式分解为 $\mathbf S^{-1}$ 的 (2,2) 块 $=-(\mathbf D_a+\mathbf J_{\mathcal A}\mathbf H_{\rm eff}^{-1}\mathbf J_{\mathcal A}^\top)^{-1}\prec0$；
> (iii) $\varepsilon\to0^+$ 时 $\mathbf D_a\to0$（强活跃块），$\mathbf S\to\begin{bmatrix}\mathbf H_{\rm eff}&\mathbf J_{\mathcal A}^\top\\ \mathbf J_{\mathcal A}&0\end{bmatrix}$，后者可逆 **iff** $\mathbf J_{\mathcal A}$ 行满秩且 $\mathbf H_{\rm eff}\succ0$ 在 $\ker\mathbf J_{\mathcal A}$ 上；
> (iv) 条件数 $\kappa(\mathbf S)\asymp\|\mathbf J_{\mathcal A}\|^2/\lambda_{\min}(\mathbf D_a)\asymp 1/\varepsilon^2$。

**证明**：
(i) 用 Schur 补。$\mathbf H_{\rm eff}\succ0$ 可逆，$\mathbf S$ 可逆 $\iff$ 其 Schur 补 $\mathbf S/\mathbf H_{\rm eff}=-\mathbf D_a-\mathbf J_{\mathcal A}\mathbf H_{\rm eff}^{-1}\mathbf J_{\mathcal A}^\top$ 可逆。而 $-\mathbf S/\mathbf H_{\rm eff}=\mathbf D_a+\mathbf J_{\mathcal A}\mathbf H_{\rm eff}^{-1}\mathbf J_{\mathcal A}^\top$。$\mathbf D_a\succ0$，$\mathbf J_{\mathcal A}\mathbf H_{\rm eff}^{-1}\mathbf J_{\mathcal A}^\top\succeq0$，故和 $\succ0$，可逆。∎
(ii) 标准块逆公式。
(iii) $\mathbf D_a\to0$ 时 Schur 补变成 $-\mathbf J_{\mathcal A}\mathbf H_{\rm eff}^{-1}\mathbf J_{\mathcal A}^\top$，它可逆 iff $\mathbf J_{\mathcal A}$ 行满秩（且 $\mathbf H_{\rm eff}$ 在 $\ker\mathbf J_{\mathcal A}$ 上非奇异）。等价地，$\mathbf S$ 在 $\mathbf D_a=0$ 时可逆 iff $\mathbf J_{\mathcal A}$ 行满秩且 $\mathbf H_{\rm eff}$ 在 $\ker\mathbf J_{\mathcal A}$ 上正定——即 LICQ + SOSC。∎
(iv) $\mathbf S$ 的最大奇异值 $\asymp\|\mathbf J_{\mathcal A}\|$；最小奇异值由 Schur 补给出 $\asymp\lambda_{\min}(\mathbf D_a+\mathbf J_{\mathcal A}\mathbf H_{\rm eff}^{-1}\mathbf J_{\mathcal A}^\top)$。当 $\mathbf J_{\mathcal A}$ 行满秩时，$\mathbf J_{\mathcal A}\mathbf H^{-1}\mathbf J_{\mathcal A}^\top\succ0$ 给出与 $\varepsilon$ 无关的下界 ⇒ **条件数不爆破**；当 $\mathbf J_{\mathcal A}$ 秩亏时，$\mathbf J_{\mathcal A}\mathbf H^{-1}\mathbf J_{\mathcal A}^\top$ 奇异于 $\ker\mathbf J_{\mathcal A}^\top$，此时最小奇异值 $\asymp\lambda_{\min}(\mathbf D_a)\asymp\varepsilon^2/\mu^2$，故 $\kappa\asymp1/\varepsilon^2$。∎

**数值验证**（`verify/sanity_fb_saddle.py`）：
* F4a：200 组随机 $(\mathbf H\succ0,\mathbf J$ 任意秩$,\mathbf D\succ0)$，最小/最大奇异值比 $4.4\times10^{-3}>0$ ⇒ 恒可逆（含 $\mathbf J$ 秩亏）。PASS。
* F4b：四脚桌 $\mathbf J_{\mathcal A}\in\mathbb R^{4\times3}$ 秩 3、$\mathbf H=\mathbf I_3$、$\mathbf D_a=d\mathbf I$：$d=4.96\times10^{-3}\to5.0\times10^{-13}$ 时 $\kappa=5.16\times10^{2}\to5.12\times10^{12}$，$\log\kappa/\log\varepsilon$ 斜率 $-2.00$ ⇒ $\kappa\asymp1/\varepsilon^2$。PASS。
* F6：$\mathbf J_{\mathcal A}=\begin{bmatrix}1&0.15&0.1\\1&0.15&-0.1\\1&-0.15&0.1\\1&-0.15&-0.1\end{bmatrix}$，$\operatorname{rank}=3<4$；$\mathbf D_a=0$ 时 $\mathbf S$ 秩 $6/7$ ⇒ 奇异。PASS。

### 3.3 新增 Proposition 2.1（$\varepsilon\to0$ 的极限：闭式与相容性）

> **Proposition 2.1（$\varepsilon\to0$ limit）**
> 记 $\mathbf r=-\partial_\theta\nabla_{\mathbf z}\mathcal L$、$\mathbf w=-\partial_\theta\mathbf g_{\mathcal A}$，考虑正则化系统
> $$
> \begin{bmatrix}\mathbf H_{\rm eff}&\mathbf J_{\mathcal A}^\top\\ \mathbf J_{\mathcal A}&-\mathbf D_a\end{bmatrix}
> \begin{bmatrix}d\mathbf z\\ d\mu_{\mathcal A}\end{bmatrix}
> =\begin{bmatrix}\mathbf r\\ \mathbf w\end{bmatrix},\qquad \mathbf D_a=D(\varepsilon)\to0\ \text{正定}. \tag{3.4}
> $$
> 则
> (a) 对每个 $\varepsilon>0$，解 $(d\mathbf z(\varepsilon),d\mu_{\mathcal A}(\varepsilon))$ 存在唯一；
> (b) $d\mathbf z(\varepsilon)\to d\mathbf z^\star$，其中
> $$
> d\mathbf z^\star=\mathbf H_{\rm eff}^{-1}\mathbf J_{\mathcal A}^\top\big(\mathbf J_{\mathcal A}\mathbf H_{\rm eff}^{-1}\mathbf J_{\mathcal A}^\top\big)^{+}\mathbf w+\Big[\mathbf I-\mathbf H_{\rm eff}^{-1}\mathbf J_{\mathcal A}^\top\big(\mathbf J_{\mathcal A}\mathbf H_{\rm eff}^{-1}\mathbf J_{\mathcal A}^\top\big)^{+}\mathbf J_{\mathcal A}\Big]\mathbf H_{\rm eff}^{-1}\mathbf r, \tag{3.5}
> $$
> 即 $\mathbf H_{\rm eff}$-加权最小范数解；
> (c) **相容性条件**：$\mathbf w\in\operatorname{range}(\mathbf J_{\mathcal A})$ 时 $d\mathbf z^\star$ 与 $\mathbf D_a$ 的分布（即乘子的选择）无关，且等于真方向导数；若 $\mathbf r=0$ 则退化为 $d\mathbf z^\star=\mathbf H_{\rm eff}^{-1}\mathbf J_{\mathcal A}^\top(\mathbf J_{\mathcal A}\mathbf H_{\rm eff}^{-1}\mathbf J_{\mathcal A}^\top)^{+}\mathbf w$；
> (d) 不相容时 $d\mu_{\mathcal A}(\varepsilon)\sim\mathbf D_a^{-1}(\mathbf J_{\mathcal A}d\mathbf z^\star-\mathbf w)\to\infty$，$d\mathbf z$ 仍有限但**不是**任何可微选择的导数；
> (e) 乘子极限：相容时 $d\mu_{\mathcal A}(\varepsilon)$ 有界；不相容时 $\|d\mu_{\mathcal A}\|\asymp 1/\lambda_{\min}(\mathbf D_a)\asymp1/\varepsilon^2$。

**证明**：
(a) 定理 2.3(i)。
(b) 由 (3.4) 消去 $d\mu$：$d\mu=\mathbf D_a^{-1}(\mathbf J_{\mathcal A}d\mathbf z-\mathbf w)$（注意第二行 $\mathbf J_{\mathcal A}d\mathbf z-\mathbf D_a d\mu=\mathbf w$）。代入第一行：
$$
\big(\mathbf H_{\rm eff}+\mathbf J_{\mathcal A}^\top\mathbf D_a^{-1}\mathbf J_{\mathcal A}\big)d\mathbf z=\mathbf r+\mathbf J_{\mathcal A}^\top\mathbf D_a^{-1}\mathbf w. \tag{3.6}
$$
当 $\mathbf D_a\to0$ 且 $\mathbf J_{\mathcal A}$ 行满秩时，$\mathbf J_{\mathcal A}^\top\mathbf D_a^{-1}\mathbf J_{\mathcal A}$ 的零空间是 $\ker\mathbf J_{\mathcal A}$（维数 $6-|\mathcal A|$），其上 $\mathbf H_{\rm eff}$ 起作用；在 $\operatorname{range}(\mathbf J_{\mathcal A}^\top)$ 上该项发散。将 $d\mathbf z$ 分解为 $\mathbf P d\mathbf z\oplus(\mathbf I-\mathbf P)d\mathbf z$，其中 $\mathbf P=\mathbf H_{\rm eff}^{-1}\mathbf J_{\mathcal A}^\top(\mathbf J_{\mathcal A}\mathbf H_{\rm eff}^{-1}\mathbf J_{\mathcal A}^\top)^{-1}\mathbf J_{\mathcal A}$ 是 $\mathbf H_{\rm eff}$-正交投影到 $\operatorname{range}(\mathbf H_{\rm eff}^{-1}\mathbf J_{\mathcal A}^\top)$ 上，取极限即得 (3.5)。$\mathbf J_{\mathcal A}$ 秩亏时把 $(\cdot)^{-1}$ 换成 Moore–Penrose 伪逆。∎
(c) 若 $\mathbf w=\mathbf J_{\mathcal A}\mathbf v$，则 $(\mathbf J_{\mathcal A}\mathbf H^{-1}\mathbf J_{\mathcal A}^\top)^{+}\mathbf J_{\mathcal A}\mathbf v$ 与 $\mathbf D_a$ 无关（$\mathbf J_{\mathcal A}\mathbf H^{-1}\mathbf J_{\mathcal A}^\top$ 与 $\mathbf D_a$ 无耦合），故 $d\mathbf z^\star$ 与 $d$ 的分布无关。真方向导数由 KKT 微分方程唯一确定，与 (3.5) 在 $\mathbf r=0$ 时一致。∎
(d)(e) 由 $d\mu=\mathbf D_a^{-1}(\mathbf J_{\mathcal A}d\mathbf z-\mathbf w)$，在相容情形 $\mathbf J_{\mathcal A}d\mathbf z^\star-\mathbf w=0$，故 $d\mu$ 的极限取决于差值的收敛速率；由 (3.6) 两端乘以 $\mathbf D_a^{1/2}$ 可证差值 $=O(\lambda_{\min}(\mathbf D_a))$，故 $d\mu$ 有界。在不相容情形，$\mathbf J_{\mathcal A}d\mathbf z^\star-\mathbf w=-(\mathbf I-\mathbf P_{\rm range})\mathbf w=:\boldsymbol\eta\ne0$，于是 $d\mu=\mathbf D_a^{-1}(\boldsymbol\eta+o(1))$，取 $\mathbf D_a=d\mathbf I$ 得 $\|d\mu\|\asymp\|\boldsymbol\eta\|/d\to\infty$。∎

> **数值验证**（`verify/sanity_fb_saddle.py` F5）：
> * 相容（$\mathbf w=\mathbf J_{\mathcal A}\mathbf v$，$\mathbf H=10^{-6}\mathbf I$，$\mathbf J_{\mathcal A}$ 列满秩）：$\mathbf D_a$ 从 $1$ 降到 $10^{-6}$ 时 $\|d\mathbf z-\mathbf v\|$ 从 $1.33\times10^{-5}$ 线性降到 $1.33\times10^{-11}$ ⇒ 收敛到 $\mathbf v$。$d\mu$ 恒为 $2.8\times10^{-6}$（有界）。
> * $\mathbf D_a$ 分布不均匀（$[1,1,1,1]$ vs $[1,2,5,10]$ vs $[0,1,1,1]$）时 $d\mathbf z$ 的差 $2.3\times10^{-9}$ ⇒ **与乘子选择无关**。
> * 不相容（$\mathbf w=\mathbf w_{\rm con}+10^{-2}\,\nu$，$\nu$ 为 $\mathbf J_{\mathcal A}^\top$ 的零空间向量）：$\mathbf D_a=d\mathbf I$，$d$ 从 $10^{-1}$ 到 $10^{-8}$ 时 $d\mathbf z$ 恒为 $\|d\mathbf z\|=0.707$（有限），$\|d\mu\|=10^{-1}\to10^{6}$，恰好 $\asymp1/d$。**$d\mu$ 爆破，$d\mathbf z$ 有限但非真导数。**
> * 闭式验证（非各向同性 $\mathbf H=\operatorname{diag}(1,1,10)$，$\mathbf J$ 秩 2，$\mathbf w=\mathbf J\mathbf v$）：$d\mathbf z(\varepsilon)$ 收敛到 $\mathbf H$-加权闭式 $[1.1667,2.1667,0.3333]$（误差 $7.0\times10^{-7}$），而欧氏最小范数解是 $[0.1667,1.1667,1.3333]$，两者差 $1.73$。**证明极限是 $\mathbf H$-加权最小范数解，不是欧氏最小范数解**。

> **对源文档的修正**：源文档称"在 $\mathbf J_{\mathcal S_{\rm act}}$ 秩亏时依然严格可逆，并在 $\varepsilon\to0^+$ 时精确收敛到真方向导数"。正确的拆法是：
> 1. **$\varepsilon>0$ 时恒可逆**——成立（定理 2.3(i)），且与秩无关；
> 2. **$\varepsilon\to0$ 的极限**——只在相容性条件 $\mathbf w\in\operatorname{range}(\mathbf J_{\mathcal A})$ 下等于（$\mathbf H$-加权意义下的）真方向导数；秩亏本身**不**破坏可逆性，但**破坏极限的良定性**；
> 3. 不相容时，原问题的解映射在 $\theta=0$ 处不可微（活跃集必须变化），FB 给出的是**一个选定的广义导数**，不是真导数。
>
> **四脚桌实例的相容性**：线性化接触约束为 $g_j=(1,a_j,b_j)^\top(z,\text{roll},\text{pitch})-h_j(\theta)$，$\mathbf J_{\mathcal A}$ 的列恰为 $(1,a_j,b_j)$。若 $\theta$ 只改变接触平面的**刚体位姿**（高度/滚转/俯仰），则 $\partial_\theta g_{\mathcal A}=-(\partial_\theta h,\partial_\theta\text{roll},\partial_\theta\text{pitch})$ 是 $\mathbf J_{\mathcal A}$ 三列的组合 ⇒ $\mathbf w\in\operatorname{range}(\mathbf J_{\mathcal A})$ **自动成立**。若四条腿的高度被独立参数化（非刚体桌面），则 $\mathbf w$ 一般不在 $\operatorname{range}$，此时问题本身不可微。**论文里必须写明这一区分**，否则"四脚桌 LICQ 失效"这一招牌例子会被审稿人攻破。

### 3.4 新增 Lemma 2.4（冗余活跃集下与乘子选择无关）

> **Lemma 2.4**　在 Proposition 2.1 的相容情形且 $\mathbf J_{\mathcal A}$ 行不满秩时，$d\mathbf z^\star$ 不依赖 $\mathcal A$ 中冗余约束的乘子分配，只依赖 $\operatorname{range}(\mathbf J_{\mathcal A})$ 与 $\mathbf w$ 的投影。

**证明**：由 (3.5)，$d\mathbf z^\star$ 只通过 $\mathbf J_{\mathcal A}^\top$、$\mathbf J_{\mathcal A}\mathbf H^{-1}\mathbf J_{\mathcal A}^\top$ 与 $\mathbf w$ 依赖 $\mathcal A$，这些都是 $\operatorname{row}(\mathbf J_{\mathcal A})$ 的量；冗余约束（行线性相关）不改变 $\operatorname{row}$ 空间。乘子的具体分配只影响 $\mathbf D_a$ 的对角元，而 (3.5) 与 $\mathbf D_a$ 无关（相容情形），故不影响 $d\mathbf z^\star$。∎

> **注意**：Lemma 2.4 只保证 $d\mathbf z^\star$ 与乘子无关，**不**保证乘子本身可辨识。论文里的"四脚桌"例子应报告 $d\mathbf z$ 的精度，而不是乘子的值。

### 3.5 引理 2.5（弱活跃点的广义导数）

> **Lemma 2.5**　在弱活跃点（$g_j=\mu_j=0$）上，$a_j=b_j=1$，$\varphi_\varepsilon$ 的相对正则有 $\mathbf D_a=\mathbf I$，鞍点矩阵在 $O(1)$ 意义上良态；但此时 KKT 系统的 Jacobian 对 $\mu_j,g_j$ 的依赖退化（$d\mu_j\cdot g_j+\mu_j\cdot dg_j=0$ 两项系数均为 0），精确互补映射不可微。FB 在该点给出的是 Clarke 意义下的一个广义导数（正则化路径的极限）。**例 2.5.1 证明**：单约束线性化情形下该极限与 $\varepsilon$ 无关，且恰为 Clarke 中点 $1/2$；一般多约束/秩亏情形下是否仍为「自然的中点」未定（GAP 4 / Q2）。

**证明（要点）**：$\mu=g=0$ 时 $\partial\varphi_0/\partial\mu=1$、$\partial\varphi_0/\partial g=-1$（由 $\varphi_0=\mu-g-\sqrt{\mu^2+g^2}$ 在原点处的次微分：$\sqrt{\cdot}$ 在 $0$ 处次微分是单位球，取 $\mu,g\to0^+$ 的方向决定符号）。故 $\varphi_0$ 在原点不可微，其 Clarke 次微分是 $\{(1,0),(0,-1)\}$ 的凸包方向集。$\varepsilon>0$ 的正则化从中选出一个特定元素（$a=b=1$）。∎

#### 例 2.5.1（解析可算的 kink：FB 精确选出 Clarke 中点）

取 $\min_{z}\tfrac12(z-a)^2$ s.t. $z\ge0$，即 $g=-z\le0$、$\mu\ge0$。KKT：$(z-a)-\mu=0$ 与 $\mu z=\varepsilon^2/2$（由 Lemma 2.3(a) 与 $g=-z$）。消 $\mu$ 得
$$
z_\varepsilon(a)=\frac{a+\sqrt{a^2+2\varepsilon^2}}{2},\qquad \mu_\varepsilon(a)=z_\varepsilon(a)-a .
$$
真解映射是 $z^\star(a)=\max(a,0)$，在 $a=0$ 处不可微，Clarke 广义导数为 $[0,1]$。由隐函数微分，
$$
\frac{dz_\varepsilon}{da}\Big|_{a=0}=\frac12\Big(1+\frac{a}{\sqrt{a^2+2\varepsilon^2}}\Big)\Big|_{a=0}=\frac12,
$$
与 $\varepsilon$ 无关；且 $a\gg\varepsilon$ 时 $\to1$，$a\ll-\varepsilon$ 时 $\to0$。故 **FB 的极限恰好是 Clarke 广义导数的中点**，不是任意元素，也不是某一侧的单侧导数。

同一结果可由本文的 $6\times6$ 机器复现：在 $a=0$ 处 $r=\sqrt{2}\varepsilon$、$z=\mu=\varepsilon/\sqrt2$，故
$$
a_j=1-\frac{\mu}{r}=\frac12,\qquad b_j=1+\frac{g}{r}=1-\frac{z}{r}=\frac12,\qquad \mathbf D_a=\frac{a_j}{b_j}=1,
$$
$\mathbf J_{g}=-1$、$\mathbf H_{\rm eff}=1$、$\mathbf r=1$、$\mathbf w=0$，系统 $\begin{bmatrix}1&-1\\-1&-1\end{bmatrix}\begin{bmatrix}dz\\d\mu\end{bmatrix}=\begin{bmatrix}1\\0\end{bmatrix}$ 给出 $dz/da=1/2$ ✓。

> **数值验证**（`verify` F7）：$\varepsilon\in\{10^{-1},10^{-2},10^{-4},10^{-6}\}$ 下 $dz/da=0.500000$，与中心差分一致；$a=\pm10^{-2},\pm1$ 时分别给出 $0$ 与 $1$。17/17 项检查通过。
>
> **边界**：这一例说明 FB 在**单约束、线性化、kink 恰在 $a=0$** 时给出中点。**一般约束几何（多约束、非欧氏、$\mathbf J_{\mathcal A}$ 秩亏）下 FB 是否仍给 Clarke 广义导数的某个可刻画元素（例如"所有活跃约束的中点"），仍是 open（Q2）。** 论文只能写"在例 2.5.1 中精确等于中点；一般情形是一个可刻画的选定广义导数"，不能写"总是中点"。

> **GAP 4（P0）**：需要严格证明"$\varepsilon\to0$ 时正则化路径的极限是 Clarke 广义导数中的一个元素"，并刻画它选的是哪一个。已知文献：Bolte, Le, Pauwels, Silveti-Falls, *Nonsmooth Implicit Differentiation for Machine Learning and Optimization*, NeurIPS 2021（广义 Jacobian + 保守场）；Facchinei–Pang 的 NCP 专著对 FB 的 BD-正则性有完整讨论。**本文必须引用这些，并把"精确收敛"改成"收敛到一个可刻画的广义导数"**。

---

## 4. 与源文档的逐条差异

| 编号 | 源文档表述 | 本文修正 | 依据 | 严重度 |
|---|---|---|---|---|
| D1 | "无论 $p$ 多大，内层恒为一个标准的 4 变量**无约束** QPEP" | 内层是 $\mathbb S^3$ 上**带约束**的四次型（不是无约束，也不是 $q$ 上二次型）；"$10\times10$"仅指 $\mathbf v_2$ 提升形式 | 定理 2.1(c)、推论 2.1.1；`verify` C3c | 中（表述），高（若实现者当二次型处理会直接算错） |
| D2 | 增广拉格朗日展开 $+\text{const}$ | 常数项 $\kappa_{\mathcal A}=-\frac1{2\rho}\sum_{j\notin\mathcal A}\mu_j^2$，**只有非活跃约束**贡献 | 定理 2.1(a)；`verify` C2（错误常数下误差 0.557，正确下 $3.55\times10^{-15}$） | 高（实现级 bug） |
| D3 | "$\delta_{\rm gap}\approx0$ ⇒ 100% 认证全局最优" | 改为条件认证（定理 2.2）+ 14×14 Shor sound 下界（Lemma 2.2）；$\delta<0$ 一律视为数值误差 | 定理 2.2(d)；`verify` C5a | 高（学术诚信） |
| D4 | "$\varphi_\varepsilon=0$ 与互补条件等价（仅用于反向）" | $\varphi_\varepsilon=0\iff\mu g=-\varepsilon^2/2$，非等价；它是光滑化 | Lemma 2.3(a)；`verify` F3 | 高（数学错误） |
| D5 | "$a_j,b_j\in(0,1]$" | $a\in(0,1]$ 恒成立；$b\in[0,1]$ 仅当 $g\le0$；一般 $b\in(1,2)$ | Lemma 2.3(d)；`verify` F1 | 中 |
| D6 | "秩亏时鞍点矩阵仍严格可逆，且 $\varepsilon\to0$ 精确收敛到真方向导数" | 可逆成立；极限只在相容性条件下等于 $\mathbf H$-加权最小范数解；不相容时是选定广义导数、$d\mu$ 爆破 | 定理 2.3、Prop 2.1；`verify` F4/F5 | 高 |
| D7 | "弱活跃 $a=b=1\Rightarrow b/a=1$，良态平滑，绝不除零" | 正确，但弱活跃点 KKT Jacobian 本身退化，FB 只给广义导数 | Lemma 2.5 | 中 |
| D8 | 四脚桌"秩亏"作为招牌 | 需区分"刚体桌面参数"（相容，可微）与"独立腿高参数"（不相容，不可微） | Prop 2.1 注；`verify` F6/F5 | 高（审稿人必问） |
| D9 | "$O((10+p)^2)$ 升维灾难" | SDP 变量数是 $(10+p)^2/2$ 量级，内点法复杂度 $O((10+p)^{6})$；应写 $(10+p)^{6}$ 或 $O(p^6)$ | 标准内点法复杂度 | 中（数字写错） |
| D10 | "$\mu^*\ge0$ 自动成立" | 成立（ALM 更新 $\mu\leftarrow[\mu+\rho g]_+$），但需在算法里显式保证并检查 | 定理 2.2(d)；`verify` C6b | 低 |

---

## 5. GAP 汇总（按优先级）

| 编号 | GAP | 优先级 | 可做的事 |
|---|---|---|---|
| G1（=GAP 1） | 内层根数/消元模板是否仍为 40 根，Bézout 162 是否可达 | P0 | sympy 数商环维数 + 200 组随机实例统根数 |
| G2（=GAP 4） | 弱活跃点上 FB 极限与 Clarke 广义导数的关系 | P0 | 引用 Bolte 2021 + 构造 2D 反例，刻画选择规则 |
| G3（=GAP 2） | SDP 证书的严格数值验证（区间算术） | P1 | 显式对偶解 + 舍入误差界 |
| G4 | 非凸 ALM 只到 KKT：本文的"全局"完全来自 Lemma 2.2，需要 **ALM 的 KKT 点 + SDP 紧**才算认证；若 SDP 不紧则无结论 | P1 | 实验统计 SDP 紧的比例（YCB-V 全量） |
| G5 | $\mathbf H_{\rm tan}$ 是否含 $\sum\mu_j\nabla^2g_j$：含则精确但可能不定；不含则是 GN 近似 | P0 | 论文里必须二选一并声明；建议给两个版本 + 误差实验 |
| G6 | $\rho,\varepsilon,\delta_{\rm act}$ 的调度敏感性 | P1 | 网格实验 |
| G7 | $p$ 很大时 SDP 线性约束的规模 | P2 | $p=10^4$ 实测 |
| G8 | 多次全局解（$\Delta J=0$）时解映射的跳变与 FB 行为 | P1 | 与 Prototype 1/3 的共享问题，本文需声明范围外 |

> **G5 特别说明**：这是审稿人最容易抓的一条。若 $\mathbf H_{\rm tan}$ 取完整 Hessian（含 $\mu_j\nabla^2 g_j$），则定理 2.3 的"$\mathbf H_{\rm eff}\succ0$"不再自动成立，可逆性只能退回"$\mathbf H_{\rm eff}$ 在 $\ker\mathbf J_{\mathcal A}$ 上非奇异"；若取 GN 近似（丢掉 $\mu_j\nabla^2g_j$），可逆性有保证但算出的 $d\mathbf z$ 是 GN 近似而非精确导数。源文档的 $\mathbf H_{\rm eff}=\mathbf H_{\rm tan}+\sum(b_j/a_j)\mathbf J^\top\mathbf J$ 两边都想要，必须二选一。建议：主定理给 GN 版本（可证），精确版本作为 Remark 并给数值对比。

---

## 6. 附：为什么"ALM 只能到 KKT"不是致命伤（写给审稿人看）

非凸 ALM 的经典结论（Rockafellar–Wets 变分分析框架下的 Bertsekas 型结果）是：在 CQ 与有界乘子假设下，任意外层极限点是原始问题的 KKT 点/稳定点，**不是**全局最优。这一点本文完全承认，并且**不**声称 ALM 找全局。

本文的"全局"有两个独立来源，二者都不依赖 ALM：

1. **前向枚举**：内层子问题用 QPEP 型 Gröbner 求解器枚举全部驻点（A2），因此每个**内层**子问题是全局求解的。这只保证外层迭代的每一步是"精确的"，不保证外层极限是全局的。
2. **后验证书**：Lemma 2.2 的 14×14 Shor 下界 $\le J^\star$。若 $\mathrm{Shor}_{\rm ineq}\ge J(\hat z)-\tau$ 且 $\hat z$ 可行，则 $\hat z$ 是 $\tau$-全局最优。这是**唯一**的全局性来源，而且是后验的、可失败的。

因此正确的论文表述是：

> **Constrained-QPEPLayer 不是"全局求解器"，而是"带后验全局认证的位姿层"**：它能给出一个可行点，并在多数实例上用一个与 $p$ 无关的 14×14 SDP 证明该点全局最优；当证书不紧时，它报告未认证而不谎报。

这个定位比"100% 认证"更弱，但**可辩护**，并且正好对上了 SDPRLayers 的软肋（其"certifiable"依赖 tight 假设，而 tight 是建模工作、会失败）。本文的卖点是：
- 前向**不需要** tight（Gröbner 枚举不要求松弛紧）；
- 反向**不需要** LICQ（FB 正则化）；
- 认证**不需要** $p$ 维 PSD 块（14×14 恒定）。
