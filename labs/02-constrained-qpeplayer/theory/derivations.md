# Constrained-QPEPLayer：关键推导中间步骤

> 这份文件补齐 `paper/draft.md` 与 `theory/proofs.md` 之间被跳过的代数细节。目标是：一个熟悉线性代数与凸优化的读者，只看这份文件就能把整套公式重新推一遍，并知道每一步的符号与维度。
> 所有"实测"均指 `verify/` 两个脚本的输出（脚本与输出文件同目录）。

---

## 1. 为什么四类约束都对 $(\mathbf v_2,\mathbf t)$ 仿射

### 1.1 $\mathbf R(\mathbf q)\mathbf p$ 的显式二次表示

对单位四元数 $\mathbf q=(q_0,q_1,q_2,q_3)$（$\|\mathbf q\|=1$），

$$
\mathbf R(\mathbf q)=
\begin{bmatrix}
1-2(q_2^2+q_3^2) & 2(q_1q_2-q_0q_3) & 2(q_1q_3+q_0q_2)\\
2(q_1q_2+q_0q_3) & 1-2(q_1^2+q_3^2) & 2(q_2q_3-q_0q_1)\\
2(q_1q_3-q_0q_2) & 2(q_2q_3+q_0q_1) & 1-2(q_1^2+q_2^2)
\end{bmatrix}.
$$

记 $\mathbf p=(p_1,p_2,p_3)$。把对角线上的 $1$ 用 $1=q_0^2+q_1^2+q_2^2+q_3^2$ 替换（**只在 $\|\mathbf q\|=1$ 时成立**），逐行整理：

**第 1 行** $\mathbf e_1^\top\mathbf R\mathbf p$：
$$
\begin{aligned}
&(1-2q_2^2-2q_3^2)p_1 + 2(q_1q_2-q_0q_3)p_2 + 2(q_1q_3+q_0q_2)p_3\\
=\;&p_1(q_0^2+q_1^2-q_2^2-q_3^2) - 2p_2q_0q_3 + 2p_2q_1q_2 + 2p_3q_1q_3 + 2p_3q_0q_2 .
\end{aligned}
$$
按 $\mathbf v_2$ 的基 $[q_0^2,q_1^2,q_2^2,q_3^2,\sqrt2q_0q_1,\sqrt2q_0q_2,\sqrt2q_0q_3,\sqrt2q_1q_2,\sqrt2q_1q_3,\sqrt2q_2q_3]$ 取系数（注意除以 $\sqrt2$）：

$$
\mathbf m_1(\mathbf p)=\big[p_1,\,p_1,\,-p_1,\,-p_1,\;0,\;\sqrt2 p_3,\;-\sqrt2 p_2,\;\sqrt2 p_2,\;\sqrt2 p_3,\;0\big]^\top .
$$

**第 2 行**：
$$
\begin{aligned}
&2(q_1q_2+q_0q_3)p_1 + (1-2q_1^2-2q_3^2)p_2 + 2(q_2q_3-q_0q_1)p_3\\
=\;&2p_1q_1q_2+2p_1q_0q_3+p_2(q_0^2-q_1^2+q_2^2-q_3^2)+2p_3q_2q_3-2p_3q_0q_1,\\
\mathbf m_2(\mathbf p)=\;&\big[p_2,\,-p_2,\,p_2,\,-p_2,\;-\sqrt2p_3,\;0,\;\sqrt2p_1,\;\sqrt2p_1,\;0,\;\sqrt2p_3\big]^\top .
\end{aligned}
$$

**第 3 行**：
$$
\begin{aligned}
&2(q_1q_3-q_0q_2)p_1 + 2(q_2q_3+q_0q_1)p_2 + (1-2q_1^2-2q_2^2)p_3\\
=\;&2p_1q_1q_3-2p_1q_0q_2+2p_2q_2q_3+2p_2q_0q_1+p_3(q_0^2-q_1^2-q_2^2+q_3^2),\\
\mathbf m_3(\mathbf p)=\;&\big[p_3,\,-p_3,\,-p_3,\,p_3,\;\sqrt2p_2,\;-\sqrt2p_1,\;0,\;0,\;\sqrt2p_1,\;\sqrt2p_2\big]^\top .
\end{aligned}
$$

> **验证**：上式对 200 组随机 $(\mathbf p,\mathbf q\in\mathbb S^3)$ 的最大误差 $1.1\times10^{-15}$。
> **注意**：`verify/sanity_constraints_alm.py` 里的 $\mathbf m_a$ 是用 12 个随机单位四元数构造 Vandermonde 矩阵后 `lstsq` 解出来的，与本节的显式式独立；两者一致（C1a 误差 $4.3\times10^{-15}$），互为交叉验证。

### 1.2 四类约束的 $(\tilde{\mathbf a}_j,\mathbf b_j)$

用 $\mathbf X:=\mathbf R(\mathbf q)\mathbf p+\mathbf t$，$\tilde{\mathbf a}_j=\mathbf a_j+c_j\mathbf e_{\rm tr}$（$c_j$ 为常数项，靠 $\mathbf e_{\rm tr}^\top\mathbf v_2=\|\mathbf q\|^2=1$ 吸收）。

| 约束 | 原始形式 | $\mathbf a_j$ | $\mathbf b_j$ | $c_j$ |
|---|---|---|---|---|
| 正深度 | $z_{\min}-X_3\le0$ | $-\mathbf m_3(\mathbf p)$ | $(0,0,-1)$ | $z_{\min}$ |
| 平面非穿透 | $-\mathbf n^\top X-d\le0$ | $-\sum_a n_a\mathbf m_a(\mathbf v)$ | $-\mathbf n$ | $-d$ |
| FoV（$+$） | $X_1-\tau X_3\le0$ | $\mathbf m_1(\mathbf p)-\tau\mathbf m_3(\mathbf p)$ | $(1,0,-\tau)$ | $0$ |
| FoV（$-$） | $-X_1-\tau X_3\le0$ | $-\mathbf m_1(\mathbf p)-\tau\mathbf m_3(\mathbf p)$ | $(-1,0,-\tau)$ | $0$ |
| 倾角锥（在 $\mathbb S^3$ 上） | $q_1^2+q_2^2\le\sin^2(\alpha/2)$ | $\mathbf e_2+\mathbf e_3$ | $\mathbf 0$ | $-\sin^2(\alpha/2)$ |

**推导示例（FoV）.** 视锥约束为 $|X_1|\le\tau X_3$，$\tau=\tan(\text{FoV}_x/2)$，等价于两条线性不等式 $X_1-\tau X_3\le0$ 与 $-X_1-\tau X_3\le0$。展开 $X_1=\mathbf m_1(\mathbf p)^\top\mathbf v_2+t_1$、$X_3=\mathbf m_3(\mathbf p)^\top\mathbf v_2+t_3$，得表中系数。

**推导示例（倾角锥）.** $q_1^2+q_2^2\le\sin^2(\alpha/2)\|\mathbf q\|^2$ 在 $\mathbb S^3$ 上就是 $q_1^2+q_2^2\le\sin^2(\alpha/2)$。左边 $\mathbf v_2$ 的第 2、3 个分量（索引从 0 起为 1、2），右边常数经 $\mathbf e_{\rm tr}$ 吸收，故 $\mathbf a_j=\mathbf e_2+\mathbf e_3$，$c_j=-\sin^2(\alpha/2)$。

> **实测**：五条（含 FoV 两条）对 6 组随机 $(q,t,p)$ 的最大误差 $3.7\times10^{-15}$（`verify` C1b）。

### 1.3 类的边界（为什么距离约束不在范围内）

$g$ 是 $(\mathbf v_2,\mathbf t)$ 的**仿射**函数，等价于 $g$ 是 $\mathbf q$ 的二次型、$\mathbf t$ 的线性型、加交叉项 $\mathbf q^\top(\cdot)\mathbf t$ 的混合。检查常见约束：

- $\|\mathbf R\mathbf p+\mathbf t-\mathbf c\|^2\ge d^2$：左边是 $\mathbf q$ 的四次（因为 $\|\mathbf R\mathbf p\|^2$ 中 $\mathbf R$ 二次，平方后四次）加 $\mathbf t$ 的二次加三次交叉。**不是仿射**，且 ALM 的 $g^2$ 会变成 8 次 ⇒ 内层不再是 QPEP 类。
- 相对位置球约束 $\|\mathbf t_i-\mathbf t_j\|\le d$：$\mathbf t$ 的二次，不是仿射。
- 平面半空间、FoV、倾角锥、深度：仿射 ✓。
- 关节限位（若关节角是位姿的 $\arccos$ 函数）：非多项式，不在范围内。

**结论（可写进论文的判据）**：能用本文的是"$\mathbf q$ 的二次、$\mathbf t$ 的线性"这一层；**不能**用的是任何含 $\|\mathbf R\mathbf p+\mathbf t\|$ 的约束。工程上的规避办法是用切平面把凸障碍近似成若干半空间（每个半空间是仿射的），代价是近似误差——这属于实验设计，不属于理论范围。

---

## 2. 增广拉格朗日的完整配方

### 2.1 hinge 展开

$$
\frac{1}{2\rho}\Big(\big[\mu_j+\rho g_j\big]_+^2-\mu_j^2\Big)=
\begin{cases}
\mu_jg_j+\dfrac{\rho}{2}g_j^2, & \mu_j+\rho g_j>0\quad(\text{活跃}),\\[2mm]
-\dfrac{\mu_j^2}{2\rho}, & \mu_j+\rho g_j\le0\quad(\text{非活跃}).
\end{cases}
$$

第一个等号：$(\mu+\rho g)^2-\mu^2=2\mu\rho g+\rho^2g^2$，除以 $2\rho$ 得 $\mu g+\frac\rho2g^2$。**注意没有 $\mu^2$ 项**——这正是常数项只来自非活跃约束的原因。

### 2.2 三块系数：逐个来源

记 $\mathbf x=\mathbf v_2(\mathbf q)$，$g_j=\tilde{\mathbf a}_j^\top\mathbf x+\mathbf b_j^\top\mathbf t$。

**(i) $\mu_jg_j$ 的 $\mathbf x$-部分**：$\mu_j\tilde{\mathbf a}_j^\top\mathbf x$ 本身是线性的，不能写成 $\mathbf x^\top(\cdot)\mathbf x$。用 $\mathbf e_{\rm tr}^\top\mathbf x=1$：
$$
\mu_j\tilde{\mathbf a}_j^\top\mathbf x=\mu_j(\tilde{\mathbf a}_j^\top\mathbf x)(\mathbf e_{\rm tr}^\top\mathbf x)=\mathbf x^\top\Big[\tfrac{\mu_j}{2}\big(\tilde{\mathbf a}_j\mathbf e_{\rm tr}^\top+\mathbf e_{\rm tr}\tilde{\mathbf a}_j^\top\big)\Big]\mathbf x .
$$
对称化的 $\tfrac12$ 是必须的：$\mathbf x^\top\mathbf A\mathbf x$ 只依赖 $\mathbf A$ 的对称部分，若直接写 $\tilde{\mathbf a}_j\mathbf e_{\rm tr}^\top$ 会得到 $\tfrac12\mu_j\tilde{\mathbf a}_j^\top\mathbf x$，差一个因子 2。

**(ii) $\mu_jg_j$ 的 $\mathbf t$-部分**：$\mu_j\mathbf b_j^\top\mathbf t$ 线性于 $\mathbf t$，需要 $\mathbf e_{\rm tr}^\top\mathbf x=1$ 把它变成交叉项：
$$
\mu_j\mathbf b_j^\top\mathbf t=\mu_j(\mathbf b_j^\top\mathbf t)(\mathbf e_{\rm tr}^\top\mathbf x)=2\,\mathbf t^\top\Big[\tfrac{\mu_j}{2}\mathbf b_j\mathbf e_{\rm tr}^\top\Big]\mathbf x .
$$
因子 2 来自交叉项在二次型中的约定：若 $[\mathbf x;\mathbf t]^\top\mathbf Q[\mathbf x;\mathbf t]$，则 $\mathbf x$-$\mathbf t$ 交叉块 $\mathbf Q_{xt}$ 的贡献是 $2\mathbf t^\top\mathbf Q_{xt}^\top\mathbf x$。

**(iii) $\frac\rho2g_j^2$**：
$$
\tfrac\rho2 g_j^2=\tfrac\rho2(\tilde{\mathbf a}_j^\top\mathbf x)^2+\rho(\tilde{\mathbf a}_j^\top\mathbf x)(\mathbf b_j^\top\mathbf t)+\tfrac\rho2(\mathbf b_j^\top\mathbf t)^2 .
$$
三项分别进入 $\mathbf M$、$\mathbf D$（交叉，含因子 2 的约定）、$\mathbf G$。

合并：

$$
\boxed{
\begin{aligned}
\mathbf M_{\mathcal A}&=\mathbf M_0+\sum_{j\in\mathcal A}\Big[\tfrac{\mu_j}{2}\big(\mathbf e_{\rm tr}\tilde{\mathbf a}_j^\top+\tilde{\mathbf a}_j\mathbf e_{\rm tr}^\top\big)+\tfrac{\rho}{2}\tilde{\mathbf a}_j\tilde{\mathbf a}_j^\top\Big]\in\mathbb S^{10},\\
\mathbf D_{\mathcal A}&=\mathbf D_0+\sum_{j\in\mathcal A}\Big[\tfrac{\mu_j}{2}\mathbf b_j\mathbf e_{\rm tr}^\top+\tfrac{\rho}{2}\mathbf b_j\tilde{\mathbf a}_j^\top\Big]\in\mathbb R^{3\times10},\\
\mathbf G_{\mathcal A}&=\mathbf G_0+\tfrac{\rho}{2}\sum_{j\in\mathcal A}\mathbf b_j\mathbf b_j^\top\in\mathbb S^{3},\\
\kappa_{\mathcal A}&=-\frac{1}{2\rho}\sum_{j\notin\mathcal A}\mu_j^2 .
\end{aligned}}
$$

**符号自查**：$\mathbf D_{\mathcal A}$ 的定义使得 $\mathcal L\supset 2\mathbf t^\top\mathbf D_{\mathcal A}\mathbf x$。展开一项 $\frac\rho2\mathbf b_j\tilde{\mathbf a}_j^\top$ 的贡献：$2\mathbf t^\top(\frac\rho2\mathbf b_j\tilde{\mathbf a}_j^\top)\mathbf x=\rho(\mathbf b_j^\top\mathbf t)(\tilde{\mathbf a}_j^\top\mathbf x)$ ✓ 与 (iii) 一致。

**实测**：与直接计算 (1.1) 的差 $3.6\times10^{-15}$（`verify` C2，$|\mathcal A|=11$）。若误用 $\kappa=-\frac1{2\rho}\sum_{j=1}^p\mu_j^2$，误差 $5.6\times10^{-1}$。

### 2.3 消去 $\mathbf t$

$\partial_{\mathbf t}\mathcal L=2\mathbf G_{\mathcal A}\mathbf t+2\mathbf D_{\mathcal A}\mathbf x=0$ 给出
$$
\mathbf t^\star(\mathbf q)=-\mathbf G_{\mathcal A}^{-1}\mathbf D_{\mathcal A}\mathbf v_2(\mathbf q),\qquad \mathbf t^\star\text{ 二次于 }\mathbf q .
$$
代回：
$$
\begin{aligned}
\mathbf t^{\star\top}\mathbf G_{\mathcal A}\mathbf t^\star+2\mathbf t^{\star\top}\mathbf D_{\mathcal A}\mathbf x
&=\mathbf x^\top\mathbf D_{\mathcal A}^\top\mathbf G_{\mathcal A}^{-1}\mathbf G_{\mathcal A}\mathbf G_{\mathcal A}^{-1}\mathbf D_{\mathcal A}\mathbf x-2\mathbf x^\top\mathbf D_{\mathcal A}^\top\mathbf G_{\mathcal A}^{-1}\mathbf D_{\mathcal A}\mathbf x\\
&=-\mathbf x^\top\mathbf D_{\mathcal A}^\top\mathbf G_{\mathcal A}^{-1}\mathbf D_{\mathcal A}\mathbf x,
\end{aligned}
$$
故
$$
\min_{\mathbf t}\mathcal L_\rho\Big|_{\mathcal A}=\mathbf v_2(\mathbf q)^\top\underbrace{\big(\mathbf M_{\mathcal A}-\mathbf D_{\mathcal A}^\top\mathbf G_{\mathcal A}^{-1}\mathbf D_{\mathcal A}\big)}_{=:\mathbf M^{\rm red}_{\mathcal A}}\mathbf v_2(\mathbf q)+\kappa_{\mathcal A}.
$$

**白化技巧（数值稳定）**：$\mathbf G_{\mathcal A}=\mathbf L\mathbf L^\top$（Cholesky），令 $\tilde{\mathbf t}=\mathbf L^\top\mathbf t$，则 $\mathbf t^\top\mathbf G\mathbf t=\|\tilde{\mathbf t}\|^2$，$\mathbf t^\top\mathbf D\mathbf x=\tilde{\mathbf t}^\top\mathbf L^{-1}\mathbf D\mathbf x$。消去后 $\mathbf M^{\rm red}=\mathbf M_{\mathcal A}-(\mathbf L^{-1}\mathbf D_{\mathcal A})^\top(\mathbf L^{-1}\mathbf D_{\mathcal A})$，避免了显式求 $\mathbf G^{-1}$。这在 $\rho$ 很大时（$\mathbf G_{\mathcal A}\asymp\rho$）能显著改善条件数。

### 2.4 次数的账

$\mathbf v_2$ 的分量是 $\mathbf q$ 的二次单项式 ⇒ $\mathbf v_2^\top\mathbf M\mathbf v_2$ 是**四次**型。KKT：
$$
\nabla_{\mathbf q}\big[\mathbf v_2^\top\mathbf M\mathbf v_2\big]-2\lambda\mathbf q=0\quad(\text{4 个三次方程}),\qquad \|\mathbf q\|^2-1=0\quad(\text{1 个二次方程}).
$$
未知量 $(q_0,q_1,q_2,q_3,\lambda)$ 共 5 个，Bézout 上界 $3^4\cdot2=162$。QPEP 的参考实现返回至多 40 个根（`kb/sdpr-qpep/03.md` §2.3），说明其系统有额外结构。本文的内层 $\mathbf M^{\rm red}_{\mathcal A}$ 与 $\mathbf M_0$ 词支撑相同（全体四次单项式）但系数不同，**通用根数是否仍为 40 未证**（见 `math_questions.md` Q1）。

**一个容易犯的错**：把"$\mathbf v_2$ 上二次"当成"$\mathbf q$ 上二次"，于是以为可以直接用最小特征值/广义特征值求解。这是错的：$\mathbf q$ 上四次，KKT 非线性，必须用多项式系统或迭代。本仓库的 `verify` C3c 用"$\Phi(s\mathbf q)$ 对 $s$ 的拟合"专门检查了四次性（相对误差 $2.0\times10^{-16}$）。

---

## 3. 对偶函数 $d(\boldsymbol\mu)$：下界、符号、与"认证"的差距

### 3.1 定义与弱对偶

$$
d(\boldsymbol\mu)=\min_{\mathbf q\in\mathbb S^3,\ \mathbf t\in\mathbb R^3}\Big\{J_\theta(\mathbf q,\mathbf t)+\sum_{j=1}^p\mu_jg_j(\mathbf q,\mathbf t)\Big\}. \tag{3.1}
$$

**引理（弱对偶）**：若 $\boldsymbol\mu\ge0$，则 $d(\boldsymbol\mu)\le J^\star$。

**证明**：设 $\mathbf z$ 可行，即 $g_j(\mathbf z)\le0$。则 $\mu_jg_j(\mathbf z)\le0$（非正 × 非负），故 $J(\mathbf z)+\sum\mu_jg_j(\mathbf z)\le J(\mathbf z)$。对 $\mathbf z$ 取 $\min$ 的集合更大（全空间 vs 可行集），所以
$$
d(\boldsymbol\mu)=\min_{\text{全空间}}(\cdot)\le\min_{\text{可行}}(\cdot)\le\min_{\text{可行}}J=J^\star .\qquad\square
$$

**符号要点**：若某个 $\mu_j<0$，则 $\mu_jg_j\ge0$，$d(\boldsymbol\mu)$ 可能**大于** $J^\star$，下界失效。故 $\mu\ge0$ 是硬前提。本文的 ALM 更新 $\mu_j\leftarrow[\mu_j+\rho g_j]_+$ 保证 $\mu\ge0$（`verify` C6b 断言通过），但实现中若出现"乘子回代/热启动"就可能破坏它，必须显式检查。

### 3.2 三明治与"负 gap"

由 3.1 与可行性违反量 $\nu=\max_j[g_j(\hat{\mathbf z})]_+$：
$$
d(\hat{\boldsymbol\mu})\le J^\star\le J(\hat{\mathbf z})+L\nu .
$$
于是
$$
0\le J(\hat{\mathbf z})-J^\star\le\underbrace{J(\hat{\mathbf z})-d(\hat{\boldsymbol\mu})}_{=:\delta}+L\nu .
$$

**为什么 $\delta$ 会变成负数**：$d$ 必须由**全局**极小给出。若用数值方法（多起点/代数枚举）得到 $\hat d\ge d$，则 $\delta=J(\hat{\mathbf z})-\hat d$ 会被低估，甚至为负。本文实测 $\delta=-1.67\times10^{-11}$（`verify` C5），源文档另一实例为 $-1.40\times10^{-12}$：**同为数值误差的符号，不能作为证书**。

**要让它成为证书，需要**：(i) 一个**可验证的下界**代替 $\hat d$（本文用 14×14 Shor，见 §3.3）；(ii) $\mu\ge0$ 的显式检查；(iii) 可行性 $\nu$ 的显式检查。三者缺一不可。

### 3.3 为什么 14×14 Shor 是对的（逐步）

把 $\mathbf x=(\mathbf v_2,\mathbf t)\in\mathbb R^{13}$ 齐次化为 $\hat{\mathbf x}=(\mathbf x,1)\in\mathbb R^{14}$，$\mathbf X=\hat{\mathbf x}\hat{\mathbf x}^\top$。

1. **目标**：$J=\hat{\mathbf x}^\top\hat{\mathbf Q}\hat{\mathbf x}=\langle\hat{\mathbf Q},\mathbf X\rangle$，其中 $\hat{\mathbf Q}=\begin{bmatrix}\mathbf Q&0\\0&0\end{bmatrix}$，$\mathbf Q=\begin{bmatrix}\mathbf M_0&\mathbf D_0^\top\\\mathbf D_0&\mathbf G_0\end{bmatrix}$。
2. **齐次化约束**：$\mathbf x$ 的仿射函数 $\mathbf c^\top\mathbf x$ 在 $\mathbf X$ 上是线性的：$\mathbf c^\top\mathbf x=(\mathbf c^\top\mathbf x)(\mathbf e_{14}^\top\hat{\mathbf x})=\langle\operatorname{sym}(\hat{\mathbf c}\mathbf e_{14}^\top),\mathbf X\rangle$，$\hat{\mathbf c}=(\mathbf c,0)$。这就是为什么仿射不等式**不需要松弛变量**就能进 SDP。
3. **归一化**：$\mathbf e_{\rm tr}^\top\mathbf v_2=1$ 通过 $\langle\operatorname{sym}(\hat{\mathbf e}_{\rm tr}\mathbf e_{14}^\top),\mathbf X\rangle=1$ 加入；$\|\mathbf v_2\|^2=1$ 通过 $\operatorname{tr}(\mathbf X_{[10]})=1$ 加入。**后者不可省**：没有它，$\mathbf v_2$ 可以任意伸缩而 $\mathbf e_{\rm tr}^\top\mathbf v_2=1$ 仍成立，目标无下界（实测：不加 Veronese 时 SDP 值为 $-10^{-10}$，且 $\lambda_2/\lambda_1=0.33$，完全无信息）。
4. **Veronese 关系**：$\mathbf S=\mathbf q\mathbf q^\top$ 的 $2\times2$ 子式为零。用 $\mathbf v_2$ 表示 $\mathbf S$：$S_{ii}=v_i$，$S_{ij}=v_{ij}/\sqrt2$。于是 $2v_iv_j=v_k^2$（对角×对角=非对角²）与 $S_{ik}S_{jl}=S_{il}S_{jk}$（一般子式）都变成 $\mathbf X$ 的**线性**等式。
5. **soundness**：任何可行 $\mathbf z$ 的 $\mathbf X=\hat{\mathbf x}\hat{\mathbf x}^\top$ 满足全部约束且目标相等 ⇒ 松弛的可行集包含所有可行点 ⇒ 最优值 $\le J^\star$。

**实测紧性**（$p=12$ 合成实例）：

| SDP | 值 | $\lambda_2/\lambda_1$ |
|---|---|---|
| 无 Veronese | $-1.0\times10^{-10}$ | 0.33 |
| $+$Veronese | 0.0030334692 | $1.4\times10^{-9}$ |
| $+$Veronese$+$不等式 | **0.0680069355** | $2.2\times10^{-12}$ |
| ALM 输出 $J(\hat z)$ | 0.06800693548 | — |
| $\mathrm{Shor}$ of $d(\hat\mu)$ | 0.0034784073 | — |

第三行与第四行在 10 位有效数字上一致 ⇒ 本实例被认证。第五行说明：对**对偶函数**做 SDP 反而更松（因为 $d$ 的二次型在 $\mathbf v_2$ 上不定，Veronese 关系补不回来）。**结论：证书应对原问题做，不要对偶函数做。**

---

## 4. Fischer–Burmeister：定义、微分、符号、以及"为什么不等于互补"

### 4.1 定义与零点集

$$
\varphi_\varepsilon(\mu,g)=\mu-g-\sqrt{\mu^2+g^2+\varepsilon^2}. \tag{4.1}
$$

**求零点**：$\varphi_\varepsilon=0\Rightarrow\mu-g=\sqrt{\mu^2+g^2+\varepsilon^2}>0$，平方：
$$
\mu^2-2\mu g+g^2=\mu^2+g^2+\varepsilon^2\;\Longrightarrow\;-2\mu g=\varepsilon^2\;\Longrightarrow\;\mu g=-\frac{\varepsilon^2}{2}. \tag{4.2}
$$
所以 $\varepsilon>0$ 时零点集是**双曲线** $\mu g=-\varepsilon^2/2$，且要求 $\mu-g>0$。它完全不在互补集 $\{0\le\mu\perp(-g)\ge0\}=\{\mu\ge0,g\le0,\mu g=0\}$ 上（除极限）。实测：固定 $\mu\in\{0.5,1,2,5,10\}$，数值解出的 $g^\star$ 满足 $|\mu g^\star+\varepsilon^2/2|\le1.8\times10^{-15}$。

**互补点上的残差**：

| 类型 | $(\mu,g)$ | $\varphi_\varepsilon$ | 一阶近似 |
|---|---|---|---|
| 强活跃 | $(5,0)$ | $-1.0000\times10^{-9}$（$\varepsilon=10^{-4}$） | $-\varepsilon^2/(2\mu)=-10^{-9}$ |
| 弱活跃 | $(0,0)$ | $-1.0000\times10^{-4}$ | $-\varepsilon$ |
| 非活跃 | $(0,-3)$ | $-1.6667\times10^{-9}$ | $-\varepsilon^2/(2|g|)=-1.67\times10^{-9}$ |

（实测相对差 $8.3\times10^{-8}$ / $0$ / $8.3\times10^{-8}$。）

### 4.2 微分与 $a,b$ 的符号

$$
\partial_\mu\varphi_\varepsilon=1-\frac{\mu}{r}=:a,\qquad
\partial_g\varphi_\varepsilon=-\Big(1+\frac{g}{r}\Big)=:-b,\qquad
r=\sqrt{\mu^2+g^2+\varepsilon^2}.
$$

**范围（逐条证明）**：

- $a=1-\mu/r$。因 $r\ge\mu\ge0$，有 $\mu/r\in[0,1]$，故 $a\in[0,1]$；$r>\mu$（$\varepsilon>0$）时 $a>0$；$\mu=0$ 时 $a=1$。⇒ $a\in(0,1]$。
- $b=1+g/r$。$|g|<r$ 恒成立（因 $\varepsilon>0$ 或 $\mu>0$），故 $b\in(0,2)$。
  - 若 $g\le0$：$g/r\in[-1,0)$，故 $b\in[0,1)$。
  - 若 $g>0$：$g/r\in(0,1)$，故 $b\in(1,2)$。
- **注意**：$g>0$ 在 ALM 迭代中是**常态**（迭代点不可行），所以"$b\in(0,1]$"不能无条件写。只有输出点（可行）才满足 $b\le1$。

**极限**（$\varepsilon\to0$，固定 $\mu,g$ 的比值）：

| 类型 | $r\to$ | $a\to$ | $b\to$ | $b/a\to$ |
|---|---|---|---|---|
| 强活跃 $\mu>0,g=0$ | $\mu$ | $\varepsilon^2/(2\mu^2)$ | $1$ | $2\mu^2/\varepsilon^2\to\infty$ |
| 弱活跃 $\mu=g=0$ | $\varepsilon$ | $1$ | $1$ | $1$ |
| 非活跃 $\mu=0,g<0$ | $|g|$ | $1$ | $\varepsilon^2/(2g^2)$ | $\to0$ |

### 4.3 从 $\varphi_\varepsilon$ 到有效 Hessian（逐步）

全微分 $\varphi_\varepsilon=0$（**注意：这是对光滑化方程求微分，不是对原互补条件**）：
$$
a\,d\mu-b\,dg=0,\qquad dg=\mathbf J_{g,j}d\mathbf z+\partial_\theta g_j\,d\theta .
$$
故
$$
d\mu_j=\frac{b_j}{a_j}\big(\mathbf J_{g,j}d\mathbf z+\partial_\theta g_j\,d\theta\big). \tag{4.3}
$$

**对"其余"约束（$\mathcal R$）**：把 (4.3) 代入平稳性 $\mathbf H_{\rm tan}d\mathbf z+\sum_j\mathbf J_{g,j}^\top d\mu_j=\mathbf r$：
$$
\Big[\mathbf H_{\rm tan}+\sum_{j\in\mathcal R}\frac{b_j}{a_j}\mathbf J_{g,j}^\top\mathbf J_{g,j}\Big]d\mathbf z
=\mathbf r-\sum_{j\in\mathcal R}\frac{b_j}{a_j}\mathbf J_{g,j}^\top\partial_\theta g_j-\sum_{j\in\mathcal A}\mathbf J_{g,j}^\top d\mu_j .
$$
即得 $\mathbf H_{\rm eff}$ 的定义与 (9) 的左上块。

**对"强活跃"约束（$\mathcal A$）**：保留 $d\mu_j$ 作为未知量。把 (4.3) 写成 $a_jd\mu_j-b_j\mathbf J_{g,j}d\mathbf z=b_j\partial_\theta g_j$，两边除以 $b_j$：
$$
\frac{a_j}{b_j}d\mu_j-\mathbf J_{g,j}d\mathbf z=\partial_\theta g_j
\;\Longleftrightarrow\;
\mathbf J_{g,j}d\mathbf z-\frac{a_j}{b_j}d\mu_j=-\partial_\theta g_j .
$$
堆叠成矩阵即 (9) 的下行 $\mathbf J_{\mathcal A}d\mathbf z-\mathbf D_ad\boldsymbol\mu_{\mathcal A}=-\partial_\theta\mathbf g_{\mathcal A}$，$\mathbf D_a=\operatorname{diag}(a_j/b_j)$。

**权重观察**（与源文档的"$b/a\to0$ 自动忽略"一致）：
- 严格非活跃：$b/a\approx\varepsilon^2/(2g^2)\to0$，该约束对内层 Hessian 的贡献自动消失（不需要硬阈值）；
- 弱活跃：$a=b=1$，权重恰为 1，良态；
- 强活跃：$a$ 小，$b/a$ 大——**但这正是把它放进 $\mathcal A$（而不是 $\mathcal R$）的原因**，否则 $\mathbf H_{\rm eff}$ 会被 $1/\varepsilon^2$ 量级的项淹没。

**实测**：$a\in[0,1]$、可行 $b\le1$、不可行 $b<2$ 全部通过（`verify` F1）。

---

## 5. $6\times6$ 有效 Hessian 与 $(6+|\mathcal A|)$ 鞍点系统

### 5.1 $\mathbf H_{\rm tan}$ 的选择（本线已决策：Gauss–Newton）

真 Hessian：$\mathbf H^{\rm exact}_{\rm tan}=\nabla^2_{\mathbf z}J+\sum_{j\in\mathcal A}\mu_j\nabla^2_{\mathbf z}g_j$。
GN：$\mathbf H^{\rm GN}_{\rm tan}=\nabla^2_{\mathbf z}J$。

**为什么 $g_j$ 的二阶项不可忽略**：$g_j=\tilde{\mathbf a}_j^\top\mathbf v_2(\mathbf q)+\mathbf b_j^\top\mathbf t$，$\mathbf v_2$ 二次于 $\mathbf q$ ⇒ $\nabla^2_{\mathbf q}g_j$ 是一个**常量级**的张量（量级 $\|\tilde{\mathbf a}_j\|$），乘上接触力 $\mu_j$ 后与 $\nabla^2J$ 同阶。故 GN 不是"小扰动"，而是一个明确的建模选择。

**本文选择 GN（式 (9a)）**，因为：
1. $\mathbf H_{\rm eff}\succ0$ 自动成立（若 $\nabla^2J\succ0$），定理 2.3 / Prop 2.1 无需附加假设；
2. 反向的误差是 $O(\sum_j\mu_j\|\nabla^2g_j\|)$，可通过消融量化（§5.3 (i)）；
3. 精确 Hessian 版在接触构型下经常不定，需要 Levenberg 阻尼，工程上反而更脆。

精确版作为 Remark 4.1，并保留"$\mathbf S$ 可逆 iff $\mathbf H_{\rm eff}$ 在 $\ker\mathbf J_{\mathcal A}$ 上非奇异"的弱结论。

### 5.2 可逆性：Schur 补证明，逐块写清

$$
\mathbf S=\begin{bmatrix}\mathbf H_{\rm eff}&\mathbf J_{\mathcal A}^\top\\\mathbf J_{\mathcal A}&-\mathbf D_a\end{bmatrix},\qquad
\mathbf H_{\rm eff}\in\mathbb S^6,\ \mathbf J_{\mathcal A}\in\mathbb R^{|\mathcal A|\times6},\ \mathbf D_a=\operatorname{diag}(d_j)\succ0 .
$$

**第一步**：$\mathbf H_{\rm eff}\succ0$ ⇒ 可逆，Schur 补
$$
\mathbf S/\mathbf H_{\rm eff}=-\mathbf D_a-\mathbf J_{\mathcal A}\mathbf H_{\rm eff}^{-1}\mathbf J_{\mathcal A}^\top .
$$
**第二步**：$-\mathbf S/\mathbf H_{\rm eff}=\mathbf D_a+\mathbf J_{\mathcal A}\mathbf H_{\rm eff}^{-1}\mathbf J_{\mathcal A}^\top$。第一项 $\succ0$，第二项 $\succeq0$（$\mathbf H_{\rm eff}^{-1}\succ0$ 且形如 $\mathbf A^\top\mathbf H^{-1}\mathbf A$），和 $\succ0$。
**第三步**：$\mathbf S$ 可逆 $\iff$ Schur 补可逆 ⇒ 可逆。**不需要 $\mathbf J_{\mathcal A}$ 行满秩**。
**第四步（显式逆）**：
$$
\mathbf S^{-1}=\begin{bmatrix}
\mathbf H_{\rm eff}^{-1}-\mathbf H_{\rm eff}^{-1}\mathbf J_{\mathcal A}^\top\mathbf S_2^{-1}\mathbf J_{\mathcal A}\mathbf H_{\rm eff}^{-1} & \mathbf H_{\rm eff}^{-1}\mathbf J_{\mathcal A}^\top\mathbf S_2^{-1}\\
\mathbf S_2^{-1}\mathbf J_{\mathcal A}\mathbf H_{\rm eff}^{-1} & \mathbf S_2^{-1}
\end{bmatrix},\quad
\mathbf S_2=-\big(\mathbf D_a+\mathbf J_{\mathcal A}\mathbf H_{\rm eff}^{-1}\mathbf J_{\mathcal A}^\top\big).
$$
**第五步（条件数）**：$\sigma_{\max}(\mathbf S)\asymp\|\mathbf J_{\mathcal A}\|+\|\mathbf H_{\rm eff}\|$；$\sigma_{\min}$ 由 $\mathbf S_2$ 给出。若 $\mathbf J_{\mathcal A}$ 行满秩，$\mathbf J_{\mathcal A}\mathbf H_{\rm eff}^{-1}\mathbf J_{\mathcal A}^\top\succ0$ 与 $\varepsilon$ 无关 ⇒ $\sigma_{\min}\ge\lambda_{\min}(\mathbf J_{\mathcal A}\mathbf H^{-1}\mathbf J_{\mathcal A}^\top)$，$\kappa$ 有界。若秩亏，$\mathbf J_{\mathcal A}\mathbf H^{-1}\mathbf J_{\mathcal A}^\top$ 在 $\ker\mathbf J_{\mathcal A}^\top$ 上奇异，$\sigma_{\min}\asymp\lambda_{\min}(\mathbf D_a)\asymp\varepsilon^2/\max\mu^2$ ⇒ $\kappa\asymp1/\varepsilon^2$。
**实测**：四脚桌（$4\times3$，秩 3），$\varepsilon:10^{-1}\to10^{-6}$，$\kappa:5.16\times10^2\to5.12\times10^{12}$，$\log\kappa/\log\varepsilon=-2.00$。

### 5.3 $\varepsilon\to0$ 的极限：闭式与相容性

由 (3.4)/(9)，把第二行写成 $d\boldsymbol\mu=\mathbf D_a^{-1}(\mathbf J_{\mathcal A}d\mathbf z-\mathbf w)$ 并代入第一行：
$$
\big(\mathbf H_{\rm eff}+\mathbf J_{\mathcal A}^\top\mathbf D_a^{-1}\mathbf J_{\mathcal A}\big)d\mathbf z=\mathbf r+\mathbf J_{\mathcal A}^\top\mathbf D_a^{-1}\mathbf w . \tag{5.1}
$$

**分解**：令
$$
\mathbf P=\mathbf H_{\rm eff}^{-1}\mathbf J_{\mathcal A}^\top\big(\mathbf J_{\mathcal A}\mathbf H_{\rm eff}^{-1}\mathbf J_{\mathcal A}^\top\big)^{+}\mathbf J_{\mathcal A}
$$
（$\mathbf H_{\rm eff}$-正交投影到 $\operatorname{range}(\mathbf H_{\rm eff}^{-1}\mathbf J_{\mathcal A}^\top)$ 上；$\mathbf P^2=\mathbf P$，$\mathbf P^\top\mathbf H_{\rm eff}=\mathbf H_{\rm eff}\mathbf P$）。取 $d\mathbf z=\mathbf P d\mathbf z+(\mathbf I-\mathbf P)d\mathbf z$。

- 在 $\operatorname{range}(\mathbf P)$ 上：$\mathbf J_{\mathcal A}^\top\mathbf D_a^{-1}\mathbf J_{\mathcal A}$ 的作用 $\asymp1/\lambda_{\min}(\mathbf D_a)$，主导 $\mathbf H_{\rm eff}$；
- 在 $\operatorname{range}(\mathbf I-\mathbf P)$ 上：$\mathbf J_{\mathcal A}d\mathbf z=0$，该项消失，只剩 $\mathbf H_{\rm eff}d\mathbf z=\mathbf r$。

于是
$$
d\mathbf z\to\underbrace{\mathbf H_{\rm eff}^{-1}\mathbf J_{\mathcal A}^\top\big(\mathbf J_{\mathcal A}\mathbf H_{\rm eff}^{-1}\mathbf J_{\mathcal A}^\top\big)^{+}\mathbf w}_{=:\mathbf P\mathbf H_{\rm eff}^{-1}\mathbf r\text{ 的约束分量}}+\underbrace{(\mathbf I-\mathbf P)\mathbf H_{\rm eff}^{-1}\mathbf r}_{\text{零空间分量}} . \tag{5.2}
$$
若 $\mathbf r=0$（$\partial_\theta\nabla_{\mathbf z}\mathcal L=0$ 的一阶情形，本文的验证设置），即 (10) 式的第一项。

**相容性**：$\mathbf w\in\operatorname{range}(\mathbf J_{\mathcal A})$ 时，$(\mathbf J_{\mathcal A}\mathbf H^{-1}\mathbf J_{\mathcal A}^\top)^{+}\mathbf w$ 存在且唯一，$d\mathbf z$ 与 $\mathbf D_a$ 的分布无关（因伪逆不含 $\mathbf D_a$）。此时它就是原问题（可微时）的 KKT 导数。
**不相容**：$(\mathbf J_{\mathcal A}d\mathbf z^\star-\mathbf w)=-\boldsymbol\eta\neq0$，$d\boldsymbol\mu=\mathbf D_a^{-1}(\boldsymbol\eta+o(1))$ ⇒ $\|d\boldsymbol\mu\|\asymp\|\boldsymbol\eta\|/\lambda_{\min}(\mathbf D_a)\to\infty$。

**实测（闭式验证）**：$\mathbf H=\operatorname{diag}(1,1,10)$、$\mathbf J\in\mathbb R^{4\times3}$ 秩 2、$\mathbf w=\mathbf J\mathbf v$ 相容：
- 闭式 $\mathbf H$-加权解 $=(1.1667,2.1667,0.3333)$；
- 欧氏最小范数解 $=(0.1667,1.1667,1.3333)$，两者差 $1.732$；
- 数值 $d\mathbf z(\varepsilon)$ 在 $\mathbf D_a=10^{-6}$ 时与闭式差 $7.0\times10^{-7}$。

⇒ 极限是 **$H$-加权**最小范数解，不是欧氏最小范数解，更不是任意 $\mathbf v$。这一点如果写错，审稿人一算就发现。

### 5.4 四脚桌的相容性判定（可写进论文的结论）

线性化接触：$g_j=(1,a_j,b_j)^\top(z,\text{roll},\text{pitch})-h_j(\theta)$，$(a_j,b_j)$ 为四角坐标。$\mathbf J_{\mathcal A}$ 的第 1/2/3 列恰为 $(1,a_j,b_j)$ 的三个分量，即
$$
\mathbf J_{\mathcal A}=\big[\mathbf 1\ \ \mathbf a\ \ \mathbf b\big]\in\mathbb R^{4\times3},\qquad \operatorname{rank}=3 .
$$
- **刚体桌面参数** $\theta=(h,\text{roll}_T,\text{pitch}_T)$：$h_j(\theta)=h+a_j\text{roll}_T+b_j\text{pitch}_T$ ⇒ $-\partial_\theta\mathbf g_{\mathcal A}$ 是 $\mathbf J_{\mathcal A}$ 三列的组合 ⇒ **恒相容**，FB 给真导数；
- **四条腿独立**：$h_j$ 各自独立 ⇒ $-\partial_\theta\mathbf g_{\mathcal A}$ 是任意 $\mathbb R^4$ 向量，一般**不在** $\operatorname{range}(\mathbf J_{\mathcal A})$（该子空间是 3 维）⇒ **不相容**，解映射不可微，$d\mu$ 爆破（$1/\varepsilon^2$）。

**这就是"四脚桌例子"必须分成两半讲的原因。**

---

## 6. 一页公式速查（供实现对照）

| 量 | 公式 | 维度 | 实测 |
|---|---|---|---|
| $\mathbf v_2$ | $[q_0^2,q_1^2,q_2^2,q_3^2,\sqrt2q_0q_1,\dots]^\top$ | 10 | — |
| $g_j$ | $\tilde{\mathbf a}_j^\top\mathbf v_2+\mathbf b_j^\top\mathbf t$ | 1 | err $3.7\times10^{-15}$ |
| $\mathbf M_{\mathcal A}$ | $\mathbf M_0+\sum_{\mathcal A}[\frac{\mu}{2}(\mathbf e_{\rm tr}\tilde{\mathbf a}^\top+\tilde{\mathbf a}\mathbf e_{\rm tr}^\top)+\frac\rho2\tilde{\mathbf a}\tilde{\mathbf a}^\top]$ | $10\times10$ | err $3.6\times10^{-15}$ |
| $\mathbf D_{\mathcal A}$ | $\mathbf D_0+\sum_{\mathcal A}[\frac\mu2\mathbf b\mathbf e_{\rm tr}^\top+\frac\rho2\mathbf b\tilde{\mathbf a}^\top]$ | $3\times10$ | 同上 |
| $\mathbf G_{\mathcal A}$ | $\mathbf G_0+\frac\rho2\sum_{\mathcal A}\mathbf b\mathbf b^\top$ | $3\times3$ | $\lambda_{\min}=10.5$ |
| $\kappa_{\mathcal A}$ | $-\frac1{2\rho}\sum_{j\notin\mathcal A}\mu_j^2$ | 1 | （错版误差 0.557） |
| $\mathbf t^\star$ | $-\mathbf G_{\mathcal A}^{-1}\mathbf D_{\mathcal A}\mathbf v_2$ | 3 | — |
| $\mathbf M^{\rm red}_{\mathcal A}$ | $\mathbf M_{\mathcal A}-\mathbf D_{\mathcal A}^\top\mathbf G_{\mathcal A}^{-1}\mathbf D_{\mathcal A}$ | $10\times10$ | err $6.2\times10^{-15}$ |
| $\varphi_\varepsilon$ | $\mu-g-\sqrt{\mu^2+g^2+\varepsilon^2}$ | 1 | 零点 $\mu g=-\varepsilon^2/2$ |
| $a,b$ | $1-\mu/r$, $1+g/r$ | 1 | $a\in(0,1]$, $b\in(0,2)$ |
| $\mathbf H_{\rm eff}$ | $\nabla^2J+\sum_{\mathcal R}\frac{b}{a}\mathbf J^\top\mathbf J$ | $6\times6$ | — |
| $\mathbf D_a$ | $\operatorname{diag}_{\mathcal A}(a/b)$ | $|\mathcal A|$ | $\kappa\asymp1/\varepsilon^2$ |
| $\mathbf S$ | $\begin{bmatrix}\mathbf H_{\rm eff}&\mathbf J_{\mathcal A}^\top\\\mathbf J_{\mathcal A}&-\mathbf D_a\end{bmatrix}$ | $(6+|\mathcal A|)^2$ | 200/200 可逆 |
| $\mathrm{Shor}$ | 14×14 SDP + Veronese + 不等式 | $14\times14$ | 紧到 $2.2\times10^{-12}$ |
