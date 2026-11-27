# Robust-QPEPLayer 定理与证明

> 本文件是 Prototype 3 的数学主文档。所有定理写清**假设、结论、维度、符号**；能完整证的给完整证明，不能证的明确标 `GAP` 或 `证明草图`。与源文档（`/tmp/proto_doc.md` 第 328–429 行）的差异集中列在 §14。
>
> 约定：中文叙述，公式 LaTeX；定理名保留英文。数值证据在 `verify/`；实验数字不在本文件冒充。

---

## 0. 符号与空间

| 符号 | 含义 |
|---|---|
| $\mathcal M=\mathbb S^3\times\mathbb R^3$ | 位姿流形，$\dim\mathcal M=6$ |
| $z=(\mathbf q,\mathbf t)\in\mathcal M$ | 单位四元数旋转 + 平移；$\mathbf q^\top\mathbf q=1$ |
| $\theta\in\Theta\subset\mathbb R^p$ | 层参数（对应点/特征/权重参数） |
| $\mathbf p_i\in\mathbb R^3$ | 第 $i$ 个模型点（3D-3D）或物点（PnP） |
| $\mathbf y_i\in\mathbb R^3$ | 第 $i$ 个观测（3D-3D） |
| $W_i\in\mathbb S^3_{++}$ | 第 $i$ 个观测的矩阵权重 |
| $e_i(z,\theta)$ | 残差平方，$e_i=\|\mathbf r_i\|_{W_i}^2=\mathbf r_i^\top W_i\mathbf r_i\ge0$ |
| $\rho,\rho_\mu$ | 鲁棒核与 GNC 代理核 |
| $w_i$ | Black–Rangarajan 离群变量（outlier process） |
| $g_i,\ H_i$ | $e_i$ 的**黎曼**梯度/Hessian（$g_i\in T_z\mathcal M$，$H_i:T_z\mathcal M\to T_z\mathcal M$） |
| $g_i^E,\ H_i^E$ | $e_i$ 的**欧氏**梯度/Hessian（$\mathbb R^7$ 环境空间） |
| $\Phi(z,\theta)$ | 切空间一阶平衡映射，$\Phi=\sum_i\rho'(e_i)g_i\in T_z\mathcal M$ |
| $H_{\mathrm{rob}}$ | $\Phi$ 对 $z$ 的切空间导数，即 $F_{\mathrm{rob}}$ 的黎曼 Hessian |
| $\lambda_{\mathrm{rob}}$ | $\mathbb S^3$ 约束乘子，$\lambda_{\mathrm{rob}}=\mathbf q^\top G_q$ |
| $P=I_4-\mathbf q\mathbf q^\top$ | $\mathbb S^3$ 切空间投影 |
| $\Pi=\operatorname{diag}(P,I_3)$ | $\mathcal M$ 切空间投影（$7\times7$） |
| $B=\operatorname{diag}(B_q,I_3)$ | 切空间正交基（$7\times6$，$B_q\in\mathbb R^{4\times3}$） |

**3D-3D 残差**（本线主实验）：
$$
\mathbf r_i(z,\theta)=R(\mathbf q)\mathbf p_i+\mathbf t-\mathbf y_i,\qquad
e_i=\mathbf r_i^\top W_i\mathbf r_i,
$$
其中 $R(\mathbf q)$ 是四元数旋转矩阵，$R(\mathbf q)\mathbf p_i$ 对 $\mathbf q$ 是**二次**的（对非单位 $\mathbf q$ 也按齐次二次式定义）。记
$$
\mathbf u_i=R(\mathbf q)\mathbf p_i,\qquad J_i=\frac{\partial \mathbf u_i}{\partial \mathbf q}\in\mathbb R^{3\times4}.
$$
因为 $\mathbf u_i$ 对 $\mathbf q$ 是 2 次齐次，Euler 定理给出
$$
J_i\mathbf q=2\mathbf u_i. \tag{0.1}
$$

**核**：Geman–McClure（GM）
$$
\rho(e)=\frac{c^2e}{c^2+e},\qquad
\rho'(e)=\frac{c^4}{(c^2+e)^2},\qquad
\rho''(e)=-\frac{2c^4}{(c^2+e)^3}. \tag{0.2}
$$
GNC 代理核
$$
\rho_\mu(e)=\frac{\mu c^2e}{\mu c^2+e},\qquad \mu\ge1. \tag{0.3}
$$

---

## 1. 定理清单与状态

| 编号 | 名称 | 状态 | 备注 |
|---|---|---|---|
| Lemma 3.1 | Black–Rangarajan 对偶（GM） | **已证** | 含 $w^*$ 与对偶值；数值见 `verify/ift_1d_signflip.py` §[0] |
| Theorem 3.1 | GNC-QPEP 前向：精确对偶 + 常数维数 + 收敛到坐标稳定点 | **已证**（收敛部分条件性） | 全局性只属于内层加权 QPEP；外层无全局保证 |
| Lemma 3.2 | GNC 不动点 $\Leftrightarrow$ $F_{\mathrm{rob}}$ 稳定点（加加权全局最小条件） | **已证** | 反向需加权子问题全局最小 |
| Lemma 3.3 | 降权曲率修正 = 权重块的 Schur 补 | **已证** | 解释为何 $\Delta H_{\mathrm{redesc}}$ 不可丢 |
| Lemma 3.4 | $\mathbb S^3\times\mathbb R^3$ 上的精确黎曼 Hessian（含 $-\lambda_{\mathrm{rob}}$ 项） | **已证** | 修正源文档缺项；数值见 `verify/riemann_hessian_check.py` |
| Lemma 3.5 | $\lambda_i=\mathbf q^\top g_{i,q}=4\mathbf u_i^\top W_i\mathbf r_i\neq4e_i$ | **已证** | 反驳“$\lambda=4J$” |
| Theorem 3.2 | 精确不动点隐微分与二阶证书 | **已证**（IFT 标准） | 可微存在条件是 $H_{\mathrm{rob}}$ 可逆；$\lambda_{\min}>0$ 才是严格局部极小 |
| Theorem 3.3 | RHS 两项分解 + 观测敏感度 + 拐点变号（映射层面） | **已证** | 6D 的 $\partial z/\partial y$ 符号不由 $\lambda_\parallel$ 单独决定 |
| Proposition 3.4 | 一维精确变号命题 | **已证** | 源文档 cos$=-0.079$ 的理论硬版本 |
| Proposition 3.5 | 变号构型的存在性（两观测、充分大间隔） | **已证** | 说明命题 3.4 非空 |
| Proposition 3.6 | $O(\|\mathbf r_i\|^{-4})$ 逐观测梯度界 | **已证**（条件性） | 需要 $H_{\mathrm{rob}}^{-1}$ 一致有界；前向偏差是 $O(\|\mathbf r\|^{-3})$ |
| Proposition 3.7 | 相干多数外点下的模型崩溃（无鲁棒保证） | **已证** | 反“任意外点”声明；不是 GNC 算法问题，是模型/可辨识性问题 |
| Corollary 3.1 | 二阶外点翻转稳定性证书 | **已证** | $H_{\mathrm{rob}}\succ0$ 的判别含义与边界 |
| Remark 3.2 | 源文档过度声明清单 | — | 见 §14 |

---

## 2. 假设

**(A1) 光滑性**：$\Theta$ 是开集，$e_i\in C^2(\mathcal M\times\Theta)$；$g_i,H_i$ 对 $(z,\theta)$ 连续。$W_i$ 在紧集内正定，$\underline w I\preceq W_i\preceq\overline w I$。

**(A2) 内层全局解假设**：对任意固定 $w\in[0,1]^N$，加权 QPEP
$$
\min_{z\in\mathcal M}\ \sum_i w_i e_i(z,\theta)
$$
的全局最小解由 QPEP 型代数求解器返回（枚举全部 KKT 根并比较目标值；允许数值失败并报告）。这是“代数全局”假设，**不是凸证书**；它保证内层不受局部极小影响，但不保证外层 GNC 全局。

**(A3) 唯一性/可微性**：在讨论的不动点处，加权 QPEP 的全局最小解唯一；$\Phi$ 的切空间导数 $H_{\mathrm{rob}}$ 可逆（$\lambda_{\min}(H_{\mathrm{rob}})\neq0$）。若要“严格局部极小”，加强为 $H_{\mathrm{rob}}\succ0$。

**(A4) 参数进入方式**：$\theta$ 光滑地进入 $e_i$（例如 $\theta=\{\mathbf y_i\}$、特征、或 $W_i$ 的参数化）；$W_i$ 对 $\theta$ 的导数存在。本文推导默认 $W_i$ 固定；若 $W_i=W_i(\theta)$，需在 RHS 加 $\rho'\,\mathbf r_i^\top(\partial W_i/\partial\theta)\mathbf r_i$ 项（见 §10.3）。

**(A5) 非退化观测**：$N\ge3$ 且 $\sum_i w_iW_i\succ0$（平移可解）；3D-3D 的 $\{\mathbf p_i\}$ 不共线。PnP 需额外的深度/正深度条件；本线主推导以 3D-3D 为准。

---

## 3. Lemma 3.1（Black–Rangarajan 对偶，GM）

> **Lemma 3.1.** 固定 $\mu\ge1,c>0$。定义
> $$
> \psi_\mu(w)=\mu c^2(\sqrt w-1)^2,\qquad w\in[0,1]. \tag{3.1}
> $$
> 则对任意 $e\ge0$，
> $$
> \rho_\mu(e)=\min_{w\in[0,1]}\big\{we+\psi_\mu(w)\big\}, \tag{3.2}
> $$
> 且最小值点唯一，为
> $$
> w^*(e)=\rho_\mu'(e)=\left(\frac{\mu c^2}{\mu c^2+e}\right)^2. \tag{3.3}
> $$
> 进一步，对任意 $z$，
> $$
> \min_{w\in[0,1]^N}\sum_i\big\{w_ie_i(z)+\psi_\mu(w_i)\big\}
> =\sum_i\rho_\mu(e_i(z)). \tag{3.4}
> $$

**证明.** 令 $\varphi_e(w)=we+\mu c^2(\sqrt w-1)^2$。对 $w>0$，
$$
\varphi_e'(w)=e+\mu c^2\left(1-\frac{1}{\sqrt w}\right)
=e+\mu c^2-\frac{\mu c^2}{\sqrt w}.
$$
$\varphi_e''(w)=\frac{\mu c^2}{2}w^{-3/2}>0$，故 $\varphi_e$ 在 $(0,1]$ 上严格凸。一阶条件 $\varphi_e'(w)=0$ 给出
$$
\sqrt{w^*}=\frac{\mu c^2}{\mu c^2+e}
\quad\Longrightarrow\quad
w^*=\left(\frac{\mu c^2}{\mu c^2+e}\right)^2=\rho_\mu'(e).
$$
端点：$w=0$ 时 $\varphi_e(0)=\mu c^2$；$w=1$ 时 $\varphi_e(1)=e$。由于 $w^*\in(0,1]$ 且为内点极小（对 $e>0$）或 $w^*=1$（$e=0$），最小值在 $w^*$ 取得。代入：
$$
w^*e+\mu c^2(\sqrt{w^*}-1)^2
=w^*(e+\mu c^2)-2\mu c^2\sqrt{w^*}+\mu c^2.
$$
令 $a=\mu c^2$，$\sqrt{w^*}=a/(a+e)$，$w^*=a^2/(a+e)^2$：
$$
\frac{a^2}{(a+e)^2}(a+e)-\frac{2a^2}{a+e}+a
=\frac{a^2}{a+e}-\frac{2a^2}{a+e}+a
=a-\frac{a^2}{a+e}
=\frac{ae}{a+e}
=\rho_\mu(e).
$$
逐点最小可交换：$\min_{w_1,\dots,w_N}\sum_i\varphi_{e_i}(w_i)=\sum_i\min_{w_i}\varphi_{e_i}(w_i)$。$\square$

**备注 3.1**（与源文档核对）：源文档写 $w_i^{(t)}=(\mu c^2/(\mu c^2+e_i))^2=\rho_\mu'(e_i)$，对偶值为 $\rho_\mu(e_i)$，**正确**。注意 $\psi_\mu(w)$ 在 $w=0$ 处取值 $\mu c^2$，不是 $0$；这不影响内点最优，但在 $w=0$ 的边界解释（TLS 的硬阈值）要小心。

**备注 3.2**（其它核的统一写法）：设基核 $\rho(e;c)$ 满足 $\rho'(e;c)=w$ 可反解，则
$$
\rho(e)=\min_{w\in[0,1]}\{we+\psi(w)\},\qquad
\psi(w)=\rho(e(w))-w\,e(w),\quad e(w)=(\rho')^{-1}(w).
$$
常用核的对偶（$a=\mu c^2$）：

| 核 | $\rho_\mu(e)$ | $w^*=\rho_\mu'(e)$ | $\psi_\mu(w)$ | 备注 |
|---|---|---|---|---|
| Geman–McClure | $\dfrac{ae}{a+e}$ | $\left(\dfrac{a}{a+e}\right)^2$ | $a(\sqrt w-1)^2$ | 严格凸 $\psi$，光滑 |
| Cauchy | $a\log\!\left(1+\dfrac{e}{a}\right)$ | $\dfrac{a}{a+e}$ | $a(w-1-\log w)$ | 严格凸 $\psi$ |
| Welsch | $a\left(1-e^{-e/a}\right)$ | $e^{-e/a}$ | $a(1-w+w\log w)$ | 严格凸 $\psi$，$w\in(0,1]$ |
| TLS | $\min(e,a)$ | $1$ if $e<a$; $0$ if $e>a$ | $a(1-w)$ | $\psi$ 线性，$w$ 在 $e=a$ 不唯一；GNC-TLS 用二值权重 |

> **统一框架结论**：前向的内层加权 QPEP 与反向的 $6\times6$ 系统对上述核**完全同构**，差别只在 $w$-步的闭式与 $\mu$-调度；因此论文可以“一个框架 + 一个主核（GM）+ 其它核的闭式表”，而不是为每个核写一套算法。TLS 是非光滑极限，需单独处理（二值权重、无严格 $w^*$ 唯一性）。

---

## 4. Theorem 3.1（GNC-QPEP 前向）

> **Theorem 3.1.** 在 (A1)(A2)(A5) 下，固定 $\mu\ge1$：
>
> **(i) 精确对偶**：对每个 $z$，联合目标
> $$
> J_\mu(z,w)=\sum_i\big\{w_ie_i(z)+\psi_\mu(w_i)\big\} \tag{4.1}
> $$
> 在 $w$ 上的最小点由 (3.3) 给出，且 $J_\mu(z,w^*(z))=F_\mu(z):=\sum_i\rho_\mu(e_i(z))$。因此
> $$
> \min_{z\in\mathcal M,\,w\in[0,1]^N}J_\mu(z,w)=\min_{z\in\mathcal M}F_\mu(z). \tag{4.2}
> $$
>
> **(ii) 常数维数**：固定 $w$ 时，3D-3D 的加权子问题为
> $$
> \min_{\mathbf q\in\mathbb S^3,\ \mathbf t\in\mathbb R^3}\ \sum_i w_i\|R(\mathbf q)\mathbf p_i+\mathbf t-\mathbf y_i\|_{W_i}^2. \tag{4.3}
> $$
> 若 $G(w)=\sum_iw_iW_i\succ0$，则
> $$
> \mathbf t(w,\mathbf q)=G(w)^{-1}\sum_iw_iW_i(\mathbf y_i-R(\mathbf q)\mathbf p_i), \tag{4.4}
> $$
> 代回后得到 $\mathbf q$ 的**四次**多项式（等价地，$(\mathrm{vec}_2(\mathbf q),\mathbf t)$ 的二次型）。其 KKT 系统只有 $4+3$ 个变量，$N$ 仅通过 $O(N)$ 的累加进入 $3\times3$ 矩阵 $G(w)$ 与固定尺寸的系数矩阵；每轮内层求解的变量维数与 $N$ 无关。
>
> **(iii) 交替下降**：固定 $\mu$，交替执行
> $$
> w_i^{(k)}=\rho_\mu'(e_i(z^{(k)})),\qquad
> z^{(k+1)}=\arg\min_{z\in\mathcal M}\sum_iw_i^{(k)}e_i(z) \tag{4.5}
> $$
> （第二个 $\arg\min$ 由 (A2) 全局求解）。则 $F_\mu(z^{(k)})$ 单调不增；若每个加权子问题的全局最小解唯一，则 $z^{(k)}$ 收敛到不动点 $(z_\mu^*,w_\mu^*)$，满足
> $$
> w_{\mu,i}^*=\rho_\mu'(e_i(z_\mu^*)),\qquad
> z_\mu^*\in\arg\min_{z}\sum_iw_{\mu,i}^*e_i(z). \tag{4.6}
> $$
>
> **(iv) 退火极限**：令 $\mu_k\downarrow1$，每个 $\mu_k$ 内跑 (iii) 到收敛（或固定步数）。则任何聚点 $(z^*,w^*)$ 满足
> $$
> w_i^*=\rho'(e_i(z^*)),\qquad
> \Phi(z^*)=\sum_i\rho'(e_i(z^*))\,g_i(z^*)=0. \tag{4.7}
> $$
> 即 $z^*$ 是 $F_{\mathrm{rob}}$ 的**切空间稳定点**。
>
> **(v) 不保证**：$z^*$ 不一定是 $F_{\mathrm{rob}}$ 的全局最小，也不一定是局部最小（可能是鞍点）；GNC 轨迹可能落在一个坏的稳定点或坏的盆地。全局性只属于 **(ii) 的内层加权 QPEP**，不属于外层 GNC。

**证明.**

(i) 由 Lemma 3.1 逐点取最小。

(ii) 对 $\mathbf t$ 求偏导：
$$
\frac{\partial}{\partial\mathbf t}\sum_iw_i\mathbf r_i^\top W_i\mathbf r_i
=2\sum_iw_iW_i\mathbf r_i=0
\Longrightarrow
G(w)\mathbf t=\sum_iw_iW_i(\mathbf y_i-R(\mathbf q)\mathbf p_i).
$$
$G\succ0$ 时唯一。代回后，$R(\mathbf q)\mathbf p_i$ 的每个分量是 $\mathbf q$ 的二次型，故 $e_i$ 是 $\mathbf q$ 的四次多项式；KKT 为多项式系统，变量数固定为 7（或消去 $\mathbf t$ 后 4）。$N$ 只进入 $G$ 与系数累加，故每轮求解规模与 $N$ 无关。

(iii) 由 Lemma 3.1，$J_\mu(z,w^*(z))=F_\mu(z)$。$w$-步是精确最小化，$z$-步是精确（全局）最小化，故
$$
F_\mu(z^{(k+1)})
=J_\mu(z^{(k+1)},w^{(k+1)})
\le J_\mu(z^{(k+1)},w^{(k)})
\le J_\mu(z^{(k)},w^{(k)})
=F_\mu(z^{(k)}).
$$
$F_\mu\ge0$ 有下界，故收敛。若 $z$-步唯一，$z^{(k)}$ 收敛；极限处两步都不再下降，故 (4.6) 成立。

(iv) 对每个 $\mu_k$，由 (iii) 得 $w^{(k)}=\rho_{\mu_k}'(e(z^{(k)}))$ 与 $\sum_iw_i^{(k)}g_i(z^{(k)})=0$（加权子问题的一阶条件）。取子列收敛到 $(z^*,w^*)$；由 $\rho_{\mu}'$ 对 $\mu$ 连续、$\mu_k\to1$，得 $w^*=\rho'(e(z^*))$，且 $\Phi(z^*)=\sum_iw_i^*g_i(z^*)=0$。

(v) 反例/机制见 §16.2 与 Proposition 3.7。核心点：$z$-步的“全局”是**给定 $w$** 的全局；$w$ 随 $z$ 变，外层联合问题非凸，且退火路径可能有分岔。$\square$

**备注 4.1**（复杂度）：每轮 GNC 的 $w$-步 $O(N)$；加权子问题的系数累加 $O(N)$；QPEP 代数求解的规模只依赖固定维数（4 或 7）与多项式次数，不随 $N$ 增长。反向只解 $6\times6$ 线性系统，$H_{\mathrm{rob}}$ 与 RHS 的累加也是 $O(N)$。显存只存 $w^*\in\mathbb R^N$ 与 $z^*$（以及可选的 $H_{\mathrm{rob}}$），即 $O(N)$；不展开 $T$ 轮迭代，故对 $T$ 是 $O(1)$。这与“零升维”一致：**旋转维数不升、观测数不升**；但 $O(N)$ 的权重向量仍然要存，不能被说成 $O(1)$ 总显存。

**备注 4.2**（$\mu$-调度）：标准 GNC-GM 取
$$
\mu_0\ge\frac{3\max_i e_i(z^{(0)})}{c^2},\qquad
\mu_{k+1}=\max(1,\mu_k/\gamma),\quad \gamma\in[1.2,2.0], \tag{4.8}
$$
每个 $\mu_k$ 内跑若干次 (4.5) 直到 $\|z^{(k+1)}-z^{(k)}\|$ 或 $|F_\mu^{(k+1)}-F_\mu^{(k)}|$ 小于阈值。$\mu_0$ 的选择使每个残差的**径向**二阶系数 $2(\rho_\mu'+2e_i\rho_\mu'')$ 为正（$e_i<\mu c^2/3$），这是“先凸后非凸”的动机；注意这**不**意味着整个 $F_\mu$ 在 $z$ 上凸（残差对 $\mathbf q$ 非线性）。

---

## 5. Lemma 3.2（不动点与稳定点）

> **Lemma 3.2.** 设 $z^*\in\mathcal M$，$w_i^*=\rho'(e_i(z^*))$。
>
> (a) 若 $z^*$ 是加权问题 $\min_z\sum_iw_i^*e_i(z)$ 的全局最小，则 $(z^*,w^*)$ 是 $\mu=1$ 的 GNC 不动点，且 $\Phi(z^*)=0$。
>
> (b) 反之，若 $\Phi(z^*)=0$ 且 $z^*$ 是上述加权问题的全局最小，则 $(z^*,w^*)$ 是 GNC 不动点。
>
> (c) 若只假设 $\Phi(z^*)=0$（不加权全局最小），$z^*$ 只是 $F_{\mathrm{rob}}$ 的稳定点，**不一定**是 GNC 的极限点；GNC 的 $z$-步要求加权全局最小。

**证明.** (a) 加权问题的全局最小满足一阶条件 $\sum_iw_i^*g_i(z^*)=0$，即 $\Phi(z^*)=0$；$w$-步由 $w^*=\rho'(e(z^*))$ 固定，故为不动点。(b) 同理。(c) 一阶条件只保证稳定；GNC 的 $z$-步是全局最小，故其极限点还必须满足加权全局最小性。$\square$

**备注 5.1**：源文档 Theorem 3.2 说“GNC 收敛点 $(z^*,w^*)$ 若满足 $w_i=\rho'(e_i)$，则 $z^*$ 是 $F_{\mathrm{rob}}$ 的稳定点”——这个方向**正确**，但需要 $z^*$ 是加权子问题的全局最小（或至少是一阶稳定）才有 $\Phi=0$。如果只说“收敛点”，必须说明收敛的是哪个算法、在什么意义下收敛。

---

## 6. Lemma 3.3（降权曲率修正 = 权重块的 Schur 补）

> **Lemma 3.3.** 在 GNC 不动点 $(z^*,w^*)$（$\mu=1$）处，令
> $$
> H_{\mathrm{IRLS}}=\sum_i\rho'(e_i^*)H_i,\qquad
> \Delta H_{\mathrm{redesc}}=\sum_i\rho''(e_i^*)\,g_ig_i^\top\preceq0, \tag{6.1}
> $$
> $$
> D=[g_1,\dots,g_N]\in\mathbb R^{6\times N},\qquad
> \Psi''=\operatorname{diag}(\psi''(w_1^*),\dots,\psi''(w_N^*)), \tag{6.2}
> $$
> 其中 $\psi=\psi_{\mu=1}$。则联合目标 $J(z,w)=\sum_iw_ie_i+\sum_i\psi(w_i)$ 在切空间坐标下的 Hessian 为
> $$
> H_J=\begin{bmatrix}H_{\mathrm{IRLS}} & D\\ D^\top & \Psi''\end{bmatrix}, \tag{6.3}
> $$
> 且其关于 $w$ 块的 Schur 补恰为
> $$
> H_{\mathrm{rob}}=H_{\mathrm{IRLS}}-D\Psi''^{-1}D^\top
> =H_{\mathrm{IRLS}}+\Delta H_{\mathrm{redesc}}. \tag{6.4}
> $$
> 因此（在 $H_{\mathrm{IRLS}}\succeq0$、$\Psi''\succ0$ 下）
> $$
> H_J\succ0\iff H_{\mathrm{rob}}\succ0. \tag{6.5}
> $$

**证明.** $J$ 对 $z$ 的二阶导是 $\sum_iw_iH_i=H_{\mathrm{IRLS}}$；对 $w_i$ 的二阶导是 $\psi''(w_i)$；交叉项 $\partial^2J/\partial w_i\partial z=g_i$，故 $D$ 的定义成立。Schur 补：
$$
H_{\mathrm{IRLS}}-D\Psi''^{-1}D^\top
=H_{\mathrm{IRLS}}-\sum_i\frac{1}{\psi''(w_i^*)}g_ig_i^\top.
$$
在 $w_i^*=\rho'(e_i^*)$ 处，由 $\psi=(\rho)^*$ 的凹共轭关系，
$$
e=-\psi'(w),\qquad \frac{\mathrm de}{\mathrm dw}=-\psi''(w)=\frac{1}{\rho''(e)}
\quad\Longrightarrow\quad
\psi''(w_i^*)=-\frac{1}{\rho''(e_i^*)}.
$$
故 $1/\psi''(w_i^*)=-\rho''(e_i^*)$，于是
$$
H_{\mathrm{IRLS}}-D\Psi''^{-1}D^\top
=H_{\mathrm{IRLS}}+\sum_i\rho''(e_i^*)g_ig_i^\top
=H_{\mathrm{rob}}.
$$
正定性等价由分块矩阵的 Schur 补判据给出。$\square$

**解释**：源文档说“$\Delta H_{\mathrm{redesc}}\preceq0$ 有负定拉低效应”，这是对的；但更精确的说法是：**它是把 $w$ 消掉后的 Schur 补**。这解释了为什么“冻结权重”等于把 Schur 补里的 $-D\Psi''^{-1}D^\top$ 整块丢掉——不是一个小修正，而是消元的结构性项。

**备注 6.1**：$H_{\mathrm{IRLS}}\succeq0$ 来自 $z^*$ 是加权问题的全局最小（二阶必要条件）。若 $H_{\mathrm{rob}}$ 有负特征值，则 $H_J$ 也有负特征值，故 $(z^*,w^*)$ 是**联合** $J$ 的鞍点，不是局部极小；这比“$z^*$ 是 $F_{\mathrm{rob}}$ 的鞍点”更强。

---

## 7. Lemma 3.4（$\mathbb S^3\times\mathbb R^3$ 上的精确黎曼 Hessian）

> **Lemma 3.4.** 设 $f\in C^2(\mathcal M)$，$z=(\mathbf q,\mathbf t)$，$G=\nabla f\in\mathbb R^7$ 为欧氏梯度，$H_E=\nabla^2f\in\mathbb R^{7\times7}$ 为欧氏 Hessian。记 $G_q\in\mathbb R^4$ 为 $G$ 的 $\mathbf q$ 块，$P=I_4-\mathbf q\mathbf q^\top$，$\Pi=\operatorname{diag}(P,I_3)$，$D=\operatorname{diag}(I_4,0_3)$。则
> $$
> \operatorname{grad}f=\Pi G, \tag{7.1}
> $$
> $$
> \operatorname{Hess}f=\Pi\big(H_E-(\mathbf q^\top G_q)D\big)\Pi
> =\Pi H_E\Pi-(\mathbf q^\top G_q)\operatorname{diag}(P,0_3). \tag{7.2}
> $$
> 在切空间正交基 $B=\operatorname{diag}(B_q,I_3)$ 下，$6\times6$ 矩阵为
> $$
> H_f=B^\top H_E B-(\mathbf q^\top G_q)\operatorname{diag}(I_3,0_3). \tag{7.3}
> $$

**证明.** 先证 $\mathbb S^3$ 部分。对 $f$ 在 $\mathbb R^4$ 上的欧氏梯度 $g_q$ 与 Hessian $H_{qq}$，记 $\lambda=\mathbf q^\top g_q$（它是约束 $c(\mathbf q)=\frac12(\mathbf q^\top\mathbf q-1)=0$ 的法向系数；在稳定点处它就是 Lagrange 乘子；下面的 Hessian 公式对任意点都用这个 $\lambda$）。黎曼 Hessian 是 $\mathbb S^3$ 上的 Levi–Civita 联络：对切向量 $\xi$，
$$
\operatorname{Hess}f[\xi]
=P\big(D(\operatorname{grad}f)[\xi]\big)
=P\big(D(Pg_q)[\xi]\big).
$$
而 $D P[\xi]=-\xi\mathbf q^\top-\mathbf q\xi^\top$，故
$$
D(Pg_q)[\xi]=PH_{qq}\xi-(\xi\mathbf q^\top+\mathbf q\xi^\top)g_q
=PH_{qq}\xi-\xi(\mathbf q^\top g_q)-\mathbf q(\xi^\top g_q).
$$
$P$ 作用后最后一项消失（$P\mathbf q=0$），得
$$
\operatorname{Hess}f[\xi]=P\big(H_{qq}-(\mathbf q^\top g_q)I_4\big)\xi.
$$
即 $P(H_{qq}-\lambda I_4)P$（在切空间上 $P\xi=\xi$）。$\mathbb R^3$ 方向无约束，Hessian 为 $H_{tt}$；交叉项 $H_{qt}$ 经过 $\Pi$ 左右投影。合并得 (7.2)。在正交基下 $B^\top\Pi=B^\top$（因为 $B$ 的列在切空间），且 $B^\top\operatorname{diag}(P,0)B=\operatorname{diag}(I_3,0)$，得 (7.3)。$\square$

**推论 7.1（对 $F_{\mathrm{rob}}$ 的显式形式）**：取 $f=F_{\mathrm{rob}}=\sum_i\rho(e_i)$。则
$$
G=\sum_i\rho'(e_i)g_i^E,\qquad
H_E=\sum_i\Big(\rho'(e_i)H_i^E+\rho''(e_i)g_i^E(g_i^E)^\top\Big), \tag{7.4}
$$
$$
\boxed{\;
H_{\mathrm{rob}}
=B^\top\Big[\sum_i\big(\rho'(e_i)H_i^E+\rho''(e_i)g_i^E(g_i^E)^\top\big)\Big]B
-\lambda_{\mathrm{rob}}\operatorname{diag}(I_3,0_3),
\quad
\lambda_{\mathrm{rob}}=\mathbf q^\top G_q=\sum_i\rho'(e_i)\lambda_i,
\quad
\lambda_i=\mathbf q^\top g_{i,q}^E.
\;} \tag{7.5}
$$
等价地，若 $H_i$ 与 $g_i$ 指**黎曼** Hessian/梯度，则
$$
H_{\mathrm{rob}}=\sum_i\rho'(e_i)H_i+\sum_i\rho''(e_i)g_ig_i^\top. \tag{7.6}
$$

**证明.** 把 (7.4) 代入 (7.3)，并用 $\mathbf q^\top G_q=\sum_i\rho'\mathbf q^\top g_{i,q}^E=\sum_i\rho'\lambda_i$。等价形式 (7.6) 由
$$
\sum_i\rho'\Big(B^\top H_i^EB-\lambda_i\operatorname{diag}(I_3,0)\Big)
+\sum_i\rho''(B^\top g_i^E)(B^\top g_i^E)^\top
$$
与 (7.5) 逐项相同得到。$\square$

**反例 7.1（为什么 $-\lambda$ 不能丢）**：取 $f(\mathbf q)=\frac12\|\mathbf q\|^2$ 限制在 $\mathbb S^3$ 上。$f\equiv\frac12$ 是常数，黎曼 Hessian 应为 $0$。但 $G_q=\mathbf q$，$H_{qq}=I_4$，若丢掉 $-\lambda$ 项，得到 $B_q^\top I_4B_q=I_3\neq0$；保留 $-\lambda$ 项（$\lambda=\mathbf q^\top\mathbf q=1$）得 $I_3-I_3=0$。数值见 `verify/riemann_hessian_check.py` 的 `[constant test]`。

**与源文档的差异**：源文档写
$$
H_{\mathrm{rob}}=\sum_i\rho'(e_i)\nabla^2_{\mathbf z\mathbf z}e_i+\sum_i\rho''(e_i)\nabla_{\mathbf z}e_i(\nabla_{\mathbf z}e_i)^\top,
$$
并说“$\nabla_{\mathbf z}e_i$ 已包含 $\mathbb S^3$ 的切空间投影”。**若 $\nabla^2e_i$ 指欧氏 Hessian，则此式缺少 $-\lambda_{\mathrm{rob}}\operatorname{diag}(I_3,0)$ 项**；若指黎曼 Hessian，则此式正确，但必须明说。源文档没有区分，且其数值例中 $\lambda_{\mathrm{rob}}$ 未必小，故这是本线最关键的数学 GAP。`verify/riemann_hessian_check.py` 给出 $c=1.0$ 的 3D-3D 例子：源公式与有限差分相对误差 $2.59\times10^{-1}$，修正公式 $9.9\times10^{-8}$；$\lambda_{\mathrm{rob}}=1.033$。

**备注 7.2**：$H_{\mathrm{rob}}$ 的特征值是基无关的（作为 $T_z\mathcal M$ 上的自伴算子），因此 $\lambda_{\min}(H_{\mathrm{rob}})$ 可用于证书；但必须在**同一个**切空间内比较，不能把环境空间 $7\times7$ 矩阵的 $\lambda_{\min}$ 当作证书（环境空间多一个法向 $0$ 特征值，且投影会改变谱）。

---

## 8. Lemma 3.5（$\lambda_i$ 的显式形式）

> **Lemma 3.5.** 对 3D-3D 残差 $\mathbf r_i=R(\mathbf q)\mathbf p_i+\mathbf t-\mathbf y_i$，$W_i\in\mathbb S^3_{++}$，有
> $$
> g_{i,q}^E=2J_i^\top W_i\mathbf r_i,\qquad
> g_{i,t}^E=2W_i\mathbf r_i, \tag{8.1}
> $$
> $$
> \lambda_i:=\mathbf q^\top g_{i,q}^E=4\,\mathbf u_i^\top W_i\mathbf r_i
> =4\big(e_i-(\mathbf t-\mathbf y_i)^\top W_i\mathbf r_i\big), \tag{8.2}
> $$
> 其中 $\mathbf u_i=R(\mathbf q)\mathbf p_i$，$J_i=\partial\mathbf u_i/\partial\mathbf q$。特别地，$\lambda_i\neq4e_i$（除非 $(\mathbf t-\mathbf y_i)^\top W_i\mathbf r_i=0$），因此
> $$
> \lambda_{\mathrm{rob}}=\sum_i\rho'(e_i)\lambda_i
> =4\sum_i\rho'(e_i)\,\mathbf u_i^\top W_i\mathbf r_i
> \neq4\sum_i\rho'(e_i)e_i. \tag{8.3}
> $$

**证明.** $e_i=\mathbf r_i^\top W_i\mathbf r_i$，$\partial\mathbf r_i/\partial\mathbf q=J_i$，$\partial\mathbf r_i/\partial\mathbf t=I_3$，故 (8.1)。由 (0.1) $J_i\mathbf q=2\mathbf u_i$，故
$$
\lambda_i=\mathbf q^\top(2J_i^\top W_i\mathbf r_i)
=2(J_i\mathbf q)^\top W_i\mathbf r_i
=4\mathbf u_i^\top W_i\mathbf r_i.
$$
又 $e_i=\mathbf r_i^\top W_i\mathbf r_i=(\mathbf u_i+\mathbf t-\mathbf y_i)^\top W_i\mathbf r_i
=\mathbf u_i^\top W_i\mathbf r_i+(\mathbf t-\mathbf y_i)^\top W_i\mathbf r_i$，代入得 (8.2)。$\square$

**备注 8.1**：源文档说“$\lambda$ 不能直接用 $4J$”，这是对的；更准确地说，$\lambda_i$ 是 $4\mathbf u_i^\top W_i\mathbf r_i$，而不是 $4e_i$、也不是 $4\mathbf r_i^\top W_i\mathbf r_i$。数值核对：`verify/riemann_hessian_check.py` 中 $\max|\lambda_i-4\mathbf u_i^\top W_i\mathbf r_i|=7.1\times10^{-15}$，而 $\max|\lambda_i-4e_i|=51.0$。

**备注 8.2**（PnP/其它残差）：若残差对 $\mathbf t$ 仿射、对 $\mathbf q$ 的“点项”是 2 次齐次（例如代数 PnP 误差、点-to-plane），则 (0.1) 仍成立，$\lambda_i=4\mathbf u_i^\top W_i\mathbf r_i$ 的形式不变；若用重投影误差（非多项式），QPEP 需用代数误差/齐次化，公式需按具体残差重算。论文中应明确“我们的 $\lambda_i$ 公式对 3D-3D 与代数 PnP 成立；重投影误差不在本文范围”。

---

## 9. Theorem 3.2（精确不动点隐微分与二阶证书）

> **Theorem 3.2.** 在 (A1)–(A4) 下，设 $\theta_0\in\Theta$，$z^*\in\mathcal M$，$w_i^*=\rho'(e_i(z^*,\theta_0))$，且
> $$
> \Phi(z^*,\theta_0)=0,\qquad H_{\mathrm{rob}}(z^*,\theta_0)\ \text{可逆}. \tag{9.1}
> $$
> 则存在 $\theta_0$ 的邻域 $U$、$z^*$ 在 $\mathcal M$ 中的邻域 $V$，以及唯一的 $C^1$ 映射 $\bar z:U\to V$，使得
> $$
> \bar z(\theta_0)=z^*,\qquad
> \Phi(\bar z(\theta),\theta)=0\quad\forall\theta\in U, \tag{9.2}
> $$
> 且
> $$
> \boxed{\;
> \frac{\mathrm d\bar z}{\mathrm d\theta}
> =-H_{\mathrm{rob}}^{-1}\frac{\partial\Phi}{\partial\theta}
> \;} \tag{9.3}
> $$
> （在切空间正交坐标下；$\partial\Phi/\partial\theta$ 是固定 $z$ 的偏导）。
>
> 进一步：
>
> (a) 若 $H_{\mathrm{rob}}\succ0$，则 $\bar z(\theta)$ 在 $\theta_0$ 附近是 $F_{\mathrm{rob}}(\cdot,\theta)$ 的**严格局部极小**，且 $\lambda_{\min}(H_{\mathrm{rob}}(\bar z(\theta),\theta))>0$ 在小邻域内保持。
>
> (b) 若 $H_{\mathrm{rob}}$ 有负特征值，则 $z^*$ 是 $F_{\mathrm{rob}}$ 的鞍点（不是局部极小）；IFT 仍给出稳定点分支的导数（若可逆），但它不是极小点。
>
> (c) 若 $H_{\mathrm{rob}}$ 奇异，IFT 失效：稳定点分支可能分岔、导数可能爆炸；这是“外点翻转”的代数标志。
>
> (d) 即使 (a) 成立，$\bar z(\theta)$ **不必**是 $F_{\mathrm{rob}}$ 的全局最小，也**不必**是 GNC 算法在 $\theta$ 处的输出（算法可能切换到另一个分支）。式 (9.3) 是“所选稳定点分支”的导数，不是“全局 argmin”的导数。

**证明.** 取 $z^*$ 处的切空间正交基 $B$，用指数映射 $\operatorname{Exp}_{z^*}:T_{z^*}\mathcal M\to\mathcal M$ 作局部坐标：
$$
\widetilde\Phi(\xi,\theta):=\Phi(\operatorname{Exp}_{z^*}\xi,\theta),\qquad \xi\in\mathbb R^6.
$$
$\widetilde\Phi(0,\theta_0)=0$，且
$$
D_\xi\widetilde\Phi(0,\theta_0)=H_{\mathrm{rob}}(z^*,\theta_0),
$$
因为 $\Phi$ 是 $F_{\mathrm{rob}}$ 的黎曼梯度（在局部坐标下），其 Jacobian 就是黎曼 Hessian。由 (9.1) 与逆映射定理，存在 $U,V'$ 与唯一 $C^1$ 的 $\bar\xi:U\to V'$ 使 $\widetilde\Phi(\bar\xi(\theta),\theta)=0$。令 $\bar z(\theta)=\operatorname{Exp}_{z^*}\bar\xi(\theta)$。对 $\widetilde\Phi(\bar\xi(\theta),\theta)=0$ 求导：
$$
D_\xi\widetilde\Phi\,\frac{\mathrm d\bar\xi}{\mathrm d\theta}
+D_\theta\widetilde\Phi=0
\Longrightarrow
\frac{\mathrm d\bar z}{\mathrm d\theta}
=-H_{\mathrm{rob}}^{-1}\frac{\partial\Phi}{\partial\theta}.
$$
(a) $H_{\mathrm{rob}}\succ0$ 是 $F_{\mathrm{rob}}$ 在 $z^*$ 处的二阶充分条件；特征值对 $(z,\theta)$ 连续，故在小邻域内保持正定，$\bar z(\theta)$ 是严格局部极小。(b) 负特征值给出下降方向，故为鞍点。(c) 奇异时逆映射定理不适用；稳定点个数在分岔点两侧可改变。(d) 全局 argmin 的切换是解映射的**不连续**（在 tie 处），IFT 只描述局部分支；GNC 的选择由算法路径决定，不由 $\Phi=0$ 唯一决定。$\square$

**备注 9.1（可微存在条件的正确写法）**：源文档写“可微存在条件：$\lambda_{\min}(H_{\mathrm{rob}})>0$”。严格说：
- **IFT 可微**：$H_{\mathrm{rob}}$ **可逆**（$\lambda_{\min}\neq0$）；
- **严格局部极小 + 分支保持**：$H_{\mathrm{rob}}\succ0$（$\lambda_{\min}>0$）；
- **GNC 不动点良定义**：还需加权子问题全局最小唯一（A3）。
把三者混为一谈会掩盖鞍点/分岔情形。论文中应分三层写。

**备注 9.2（流形 IFT 的技术细节）**：上式在 $z^*$ 的固定切空间用指数映射作坐标，$\partial\Phi/\partial\theta$ 是在该坐标下的偏导。由于 $\Phi$ 是切向量场，严格写法需要用 $TM$ 的局部平凡化与联络；对一阶导数，结果 (9.3) 与平凡化选择无关（不同平凡化只差一个可逆的线性变换，$H_{\mathrm{rob}}$ 与 RHS 同时变换）。完整证明可参 Absil et al. (2008) 的黎曼隐函数定理；本文给出的是其坐标化版本，足够工程使用。

---

## 10. Theorem 3.3（RHS 分解、观测敏感度、映射层面的变号）

> **Theorem 3.3.** 在 (A1)–(A4) 下：
>
> **(a) 两项分解**：对参数 $\theta$，
> $$
> \frac{\partial\Phi}{\partial\theta}
> =\underbrace{\sum_i\rho'(e_i)\frac{\partial g_i}{\partial\theta}}_{\text{冻结权重项}}
> +\underbrace{\sum_i\rho''(e_i)\,g_i\left(\frac{\partial e_i}{\partial\theta}\right)^\top}_{\text{隐式权重敏感度项}}. \tag{10.1}
> $$
>
> **(b) 观测敏感度（3D-3D）**：若 $\theta=\mathbf y_i$，$W_i$ 固定，则
> $$
> \frac{\partial\Phi_t}{\partial\mathbf y_i}
> =-2\Big(\rho'(e_i)W_i+2\rho''(e_i)W_i\mathbf r_i\mathbf r_i^\top W_i\Big)\in\mathbb R^{3\times3}, \tag{10.2}
> $$
> $$
> \frac{\partial\Phi_q}{\partial\mathbf y_i}
> =-2J_i^\top W_i\Big(\rho'(e_i)I_3+2\rho''(e_i)\mathbf r_i\mathbf r_i^\top W_i\Big)\in\mathbb R^{4\times3}. \tag{10.3}
> $$
> 当 $W_i=I_3$ 时，$\partial\Phi_t/\partial\mathbf y_i$ 沿 $\hat{\mathbf r}_i=\mathbf r_i/\|\mathbf r_i\|$ 的特征值为 $-\lambda_\parallel(e_i)$，正交方向为 $-2\rho'(e_i)$，其中
> $$
> \lambda_\parallel(e)=2\big(\rho'(e)+2e\rho''(e)\big)
> =\frac{2c^4(c^2-3e)}{(c^2+e)^3}. \tag{10.4}
> $$
> 对一般 $W_i\succ0$，在 $W_i^{1/2}$ 度量下同样有广义特征值 $-\lambda_\parallel(e_i)$ 与 $-2\rho'(e_i)$。
>
> **(c) 冻结权重 vs 精确（映射层面）**：冻结权重（Naive IRLS-StopGrad）给出
> $$
> \frac{\partial\Phi_t^{\mathrm{naive}}}{\partial\mathbf y_i}=-2\rho'(e_i)W_i, \tag{10.5}
> $$
> 丢掉了 $-4\rho''(e_i)W_i\mathbf r_i\mathbf r_i^\top W_i$。沿 $\hat{\mathbf r}_i$，
> $$
> \frac{\partial\Phi_t}{\partial\mathbf y_i}\hat{\mathbf r}_i-\frac{\partial\Phi_t^{\mathrm{naive}}}{\partial\mathbf y_i}\hat{\mathbf r}_i
> =-4e_i\rho''(e_i)\,\hat{\mathbf r}_i
> =\frac{8c^4e_i}{(c^2+e_i)^3}\hat{\mathbf r}_i, \tag{10.6}
> $$
> 其系数严格为正。$\lambda_\parallel(e)$ 在 $e=c^2/3$ 变号：$e<c^2/3$ 时为正，$e>c^2/3$ 时为负；而 $-2\rho'(e)$ 恒为负。因此在 $e>c^2/3$ 时，两个映射在 $\hat{\mathbf r}$ 方向上的符号**严格相反**。
>
> **(d) 6D 位姿敏感度的符号不由 $\lambda_\parallel$ 单独决定**：
> $$
> \frac{\partial z^*}{\partial\mathbf y_i}
> =-H_{\mathrm{rob}}^{-1}\frac{\partial\Phi}{\partial\mathbf y_i}. \tag{10.7}
> $$
> $H_{\mathrm{rob}}^{-1}$ 会混合方向；$\lambda_\parallel<0$ 只说明 $\partial\Phi/\partial\mathbf y_i$ 在残差方向的特征值为正（映射“反向”），并不自动说明 $\partial z^*/\partial\mathbf y_i$ 在该方向为负。后者需要 $H_{\mathrm{rob}}$ 与 $\partial\Phi/\partial\mathbf y_i$ 的联合谱条件。源文档的 cos$=-0.079$ 是特定数值例，不能升格为定理。

**证明.** (a) 对 $\Phi=\sum_i\rho'(e_i)g_i$ 用乘积法则：
$$
\partial_\theta\Phi=\sum_i\big[\rho''(e_i)(\partial_\theta e_i)g_i+\rho'(e_i)\partial_\theta g_i\big].
$$
(b) 对 $\mathbf y_i$，只有第 $i$ 项依赖它；$\partial\mathbf r_i/\partial\mathbf y_i=-I_3$，$\partial e_i/\partial\mathbf y_i=-2\mathbf r_i^\top W_i$，$\partial g_{i,t}/\partial\mathbf y_i=-2W_i$，$\partial g_{i,q}/\partial\mathbf y_i=-2J_i^\top W_i$。代入 (10.1)：
$$
\partial_{\mathbf y_i}\Phi_t
=\rho''(2W_i\mathbf r_i)(-2\mathbf r_i^\top W_i)+\rho'(-2W_i)
=-2(\rho'W_i+2\rho''W_i\mathbf r_i\mathbf r_i^\top W_i).
$$
(10.3) 同理。$W_i=I$ 时，$\partial\Phi_t/\partial\mathbf y_i=-2(\rho'I+2\rho''\mathbf r\mathbf r^\top)$；沿 $\hat{\mathbf r}$ 的特征值为 $-2(\rho'+2e\rho'')=-\lambda_\parallel(e)$，正交方向 $-2\rho'$。代入 GM 的 $\rho',\rho''$ 得 (10.4)。各向异性时令 $\tilde{\mathbf r}=W_i^{1/2}\mathbf r_i$，$e=\|\tilde{\mathbf r}\|^2$，则 $W_i^{1/2}(\partial\Phi_t/\partial\mathbf y_i)W_i^{-1/2}=-2(\rho'I+2\rho''\tilde{\mathbf r}\tilde{\mathbf r}^\top)$，回到各向同性。

(c) 由 (b) 减去 (10.5)。(10.6) 的系数：$-4e\rho''=8c^4e/(c^2+e)^3>0$。$\lambda_\parallel$ 的符号由 $c^2-3e$ 决定。

(d) 由 Theorem 3.2 的 (9.3) 与链式法则。$\square$

**备注 10.1（$\lambda_\parallel$ 的极值）**：
$$
\lambda_\parallel(0)=2,\qquad
\lambda_\parallel(c^2/3)=0,\qquad
\lambda_\parallel(c^2)=-\frac12,\qquad
\lambda_\parallel(e)\sim-\frac{6c^4}{e^2}\quad(e\to\infty). \tag{10.8}
$$
因此 $\lambda_\parallel$ 的取值范围是 $(-1/2,2]$（在 $e\ge0$ 上），负值最“毒”的点在 $e=c^2$，而不是 $e\to\infty$。这解释了一个反直觉现象：**最大的梯度毒害来自“半外点”（$e\approx c^2$），不是最远的粗大外点**；粗大外点的梯度按 $O(\|\mathbf r\|^{-4})$ 衰减。

**备注 10.2（源文档的 cos$=-0.079$）**：该数字来自作者的预实验（待独立复现），本文件不把它当定理。严格可证的版本是 Proposition 3.4（一维）与 (10.6)（映射层面）；6D 的位姿层面需要额外的谱条件（见 §12 与 `math_questions.md` P1-3）。

---

## 11. Proposition 3.4（一维精确变号命题）

> **Proposition 3.4.** 令 $F(t)=\sum_{i=1}^N\rho((t-y_i)^2)$，$c>0$，$\rho$ 为 GM。设 $t^*$ 是 $F$ 的稳定点：
> $$
> \Phi(t^*)=\sum_i2\rho'(e_i)(t^*-y_i)=0,\qquad e_i=(t^*-y_i)^2. \tag{11.1}
> $$
> 若
> $$
> H_{\mathrm{rob}}=\sum_i\lambda_\parallel(e_i)>0, \tag{11.2}
> $$
> 则 $t^*$ 是严格局部极小，且对每个 $j$，
> $$
> \boxed{\;\frac{\mathrm dt^*}{\mathrm dy_j}
> =\frac{\lambda_\parallel(e_j)}{H_{\mathrm{rob}}}\;} \tag{11.3}
> $$
> 而冻结权重的 Naive IRLS-StopGrad 给出
> $$
> \left(\frac{\mathrm dt^*}{\mathrm dy_j}\right)_{\mathrm{naive}}
> =\frac{2\rho'(e_j)}{\sum_i2\rho'(e_i)}>0. \tag{11.4}
> $$
> 因此，若 $e_j>c^2/3$，则 $\lambda_\parallel(e_j)<0$，从而
> $$
> \frac{\mathrm dt^*}{\mathrm dy_j}<0
> \quad\text{而}\quad
> \left(\frac{\mathrm dt^*}{\mathrm dy_j}\right)_{\mathrm{naive}}>0. \tag{11.5}
> $$
> 即真实导数与 naive 导数**符号严格相反**。

**证明.** 对 $\Phi(t,y)=0$ 用隐函数定理。$H_{\mathrm{rob}}=\partial\Phi/\partial t$：
$$
\frac{\partial\Phi}{\partial t}
=\sum_i\big[2\rho'(e_i)+2\rho''(e_i)\cdot2(t-y_i)^2\big]
=\sum_i\big[2\rho'(e_i)+4e_i\rho''(e_i)\big]
=\sum_i\lambda_\parallel(e_i).
$$
对 $y_j$，只有第 $j$ 项依赖：
$$
\frac{\partial\Phi}{\partial y_j}
=2\rho''(e_j)\cdot(-2(t-y_j))(t-y_j)+2\rho'(e_j)\cdot(-1)
=-2\rho'(e_j)-4e_j\rho''(e_j)
=-\lambda_\parallel(e_j).
$$
故
$$
\frac{\mathrm dt^*}{\mathrm dy_j}
=-\frac{\partial\Phi/\partial y_j}{\partial\Phi/\partial t}
=\frac{\lambda_\parallel(e_j)}{H_{\mathrm{rob}}}.
$$
$H_{\mathrm{rob}}>0$ 保证 $t^*$ 是严格局部极小（一维二阶充分条件）。Naive 版本把 $w_i=\rho'(e_i^*)$ 冻结：
$$
\Phi_{\mathrm{naive}}(t)=\sum_i2w_i(t-y_i),\qquad
\frac{\partial\Phi_{\mathrm{naive}}}{\partial t}=\sum_i2w_i>0,\qquad
\frac{\partial\Phi_{\mathrm{naive}}}{\partial y_j}=-2w_j,
$$
故 (11.4)。若 $e_j>c^2/3$，由 (10.4) $\lambda_\parallel(e_j)<0$，结合 $H_{\mathrm{rob}}>0$ 得 (11.5)。$\square$

**边界情形**：
- $H_{\mathrm{rob}}=0$：$t^*$ 退化（可能是高阶平坦点），IFT 失效；不能谈“导数”。
- $H_{\mathrm{rob}}<0$：$t^*$ 不是局部极小（一维中为局部极大或高阶拐点）；(11.3) 仍可作为稳定点分支的导数，但“极小点变号”的前提不成立。
- $e_j=c^2/3$：$\lambda_\parallel=0$，精确导数为 $0$，naive 为正；这是“最弱点”，不是“反向点”。
- $H_{\mathrm{rob}}>0$ 且存在 $e_j>c^2/3$ 是可能的（例如一个内点 + 一个半外点，见 Proposition 3.5），故命题非空。

---

## 12. Proposition 3.5（变号构型的存在性）

> **Proposition 3.5.** 对任意 $c>0$，存在 $y_1<y_2$（具体地，$y_2-y_1$ 充分大），使得 $F(t)=\rho((t-y_1)^2)+\rho((t-y_2)^2)$ 有一个严格局部极小 $t^*\in(y_1,y_2)$，满足
> $$
> e_1=(t^*-y_1)^2<c^2/3<e_2=(t^*-y_2)^2,\qquad H_{\mathrm{rob}}>0, \tag{12.1}
> $$
> 从而
> $$
> \frac{\mathrm dt^*}{\mathrm dy_2}<0,\qquad
> \left(\frac{\mathrm dt^*}{\mathrm dy_2}\right)_{\mathrm{naive}}>0. \tag{12.2}
> $$

**证明.** 固定 $y_1=0$，$d=y_2>0$，$t=s\in(0,d)$。则
$$
\Phi(s)=2\rho'(s^2)s-2\rho'((d-s)^2)(d-s).
$$
在 $s=0$：$\Phi(0)=-2\rho'(d^2)d<0$。取 $s_d^\circ=Kc^4/d^3$（$K>4$ 足够大）。对 $d\gg c$：
$$
\rho'(s_d^{\circ2})\to1,\qquad
(d-s_d^\circ)^2=d^2(1+o(1)),\qquad
\rho'((d-s_d^\circ)^2)=\frac{c^4}{d^4}(1+o(1)),
$$
故
$$
\Phi(s_d^\circ)
=2s_d^\circ(1+o(1))-2\frac{c^4}{d^3}(1+o(1))
=\frac{2(K-1)c^4}{d^3}(1+o(1))>0.
$$
由介值定理，存在 $s_d\in(0,s_d^\circ)$ 使 $\Phi(s_d)=0$。因此 $s_d=O(c^4/d^3)\to0$，且
$$
e_1=s_d^2=O(d^{-6})<c^2/3,\qquad
e_2=(d-s_d)^2=d^2(1+o(1))>c^2/3
$$
对充分大 $d$ 成立。又
$$
H_{\mathrm{rob}}(s_d)=\lambda_\parallel(s_d^2)+\lambda_\parallel((d-s_d)^2)
\longrightarrow\lambda_\parallel(0)+0=2>0,
$$
故对充分大 $d$，$H_{\mathrm{rob}}>0$。由 Proposition 3.4 得 (12.2)。$\square$

**数值例**：$c=0.5$，$y_1=0$，$y_2=1.5$，$t^*=0.01543773$，$e_1=2.38\times10^{-4}$，$e_2=2.204$，$H_{\mathrm{rob}}=1.9348$，$\mathrm dt^*/\mathrm dy_2=-0.027815$（精确）vs $+0.010292$（naive）vs $-0.027815$（有限差分，相对误差 $1.2\times10^{-8}$）。见 `verify/ift_1d_signflip_output.txt`。

---

## 13. Proposition 3.6（$O(\|\mathbf r_i\|^{-4})$ 逐观测梯度界）

> **Proposition 3.6.** 在 (A1)(A5) 下，固定 $i$ 与其它观测，令 $\theta=\mathbf y_i$。假设：
>
> (B1) $W_i$ 满足 $\underline wI\preceq W_i\preceq\overline wI$；$\|\mathbf p_i\|\le\bar p$，故 $\|J_i\|_F\le\bar J$（$R(\mathbf q)$ 二次、$\mathbf q\in\mathbb S^3$）。
>
> (B2) 沿 $\|\mathbf y_i\|\to\infty$ 的一条路径，不动点 $z^*(\mathbf y_i)$ 停留在紧集 $K\subset\mathcal M$ 内，且 $H_{\mathrm{rob}}(z^*(\mathbf y_i),\mathbf y_i)$ 在 $K$ 上一致正定：$\lambda_{\min}(H_{\mathrm{rob}})\ge\mu_0>0$（或至少 $\|H_{\mathrm{rob}}^{-1}\|\le M$）。
>
> 则存在常数 $C=C(c,\underline w,\overline w,\bar J,\mu_0,K)$，使得对充分大的 $\|\mathbf r_i\|$，
> $$
> \left\|\frac{\partial\Phi}{\partial\mathbf y_i}\right\|_F
> \le C\,\frac{c^4}{(c^2+e_i)^2}
> \le C\,\frac{c^4}{\|\mathbf r_i\|^4}, \tag{13.1}
> $$
> $$
> \boxed{\;
> \left\|\frac{\partial z^*}{\partial\mathbf y_i}\right\|_F
> \le M\,C\,\frac{c^4}{\|\mathbf r_i\|^4}
> \;} \tag{13.2}
> $$
> 即逐观测梯度按 $\|\mathbf r_i\|^{-4}$ 衰减。相比之下，前向偏差（该观测对 $z^*$ 的拉力）按 $\|\mathbf r_i\|^{-3}$ 衰减：
> $$
> \left\|\rho'(e_i)g_i\right\|
> \le C'\,\frac{c^4}{\|\mathbf r_i\|^3}. \tag{13.3}
> $$

**证明.** 由 (10.2)–(10.3) 与 GM 的 $\rho',\rho''$：
$$
\rho'(e)=\frac{c^4}{(c^2+e)^2},\qquad
|\rho''(e)|=\frac{2c^4}{(c^2+e)^3}.
$$
对平移块：
$$
\left\|\frac{\partial\Phi_t}{\partial\mathbf y_i}\right\|
\le2\Big(\rho'\|W_i\|+2|\rho''|\,\|W_i\|^2\|\mathbf r_i\|^2\Big)
\le2\overline w\,\frac{c^4}{(c^2+e)^2}
+4\overline w^2\,\frac{2c^4e}{(c^2+e)^3}.
$$
用 $e/(c^2+e)\le1$，第二项 $\le8\overline w^2c^4/(c^2+e)^2$，故
$$
\left\|\frac{\partial\Phi_t}{\partial\mathbf y_i}\right\|
\le(2\overline w+8\overline w^2)\frac{c^4}{(c^2+e)^2}.
$$
对四元数块同理：
$$
\left\|\frac{\partial\Phi_q}{\partial\mathbf y_i}\right\|
\le2\bar J\Big(\overline w\rho'+2\overline w^2|\rho''|e\Big)
\le(2\bar J\overline w+4\bar J\overline w^2)\frac{c^4}{(c^2+e)^2}.
$$
合并得 (13.1)。由 (9.3) 与 (B2)，
$$
\left\|\frac{\partial z^*}{\partial\mathbf y_i}\right\|
\le\|H_{\mathrm{rob}}^{-1}\|\left\|\frac{\partial\Phi}{\partial\mathbf y_i}\right\|
\le MC\frac{c^4}{(c^2+e)^2}
\le MC\frac{c^4}{\|\mathbf r_i\|^4}.
$$
(13.3)：$\|\rho'g_i\|\le\rho'(2\overline w\|\mathbf r_i\|)=2\overline wc^4\|\mathbf r_i\|/(c^2+\|\mathbf r_i\|^2)^2\le2\overline wc^4/\|\mathbf r_i\|^3$。$\square$

**解释与边界**：
- (13.1) 是**逐观测**的界；$N$ 个外点的总 RHS 可能按 $N$ 累加，且相干外点可以同向叠加。因此“梯度免疫”不等于“前向鲁棒”（见 Proposition 3.7 与 §14）。
- 常数依赖 $\mu_0^{-1}$；若 $H_{\mathrm{rob}}$ 接近奇异（$\mu_0\to0$），界会爆炸。这正是“外点翻转”时梯度失稳的来源。
- 界对 $\|\mathbf r_i\|$ 的路径成立；若外点把不动点推出 $K$ 或使 $H_{\mathrm{rob}}$ 变号，(B2) 失效，界不成立。
- 前向偏差 $O(\|\mathbf r\|^{-3})$ 与梯度 $O(\|\mathbf r\|^{-4})$ 差一阶：**梯度免疫比前向免疫更强**，但不能反过来说“前向不受外点影响”。

---

## 14. Proposition 3.7（相干多数外点下的模型崩溃）

> **Proposition 3.7.** 对任意 $c>0$，存在观测集合使 $F_{\mathrm{rob}}$ 的**全局最小**是外点一致位姿，而不是内点位姿。具体地，一维定位模型 $F(t)=\rho(t^2)+2\rho((t-M)^2)$（1 个内点在 $0$，2 个相干外点在 $M$）满足：当 $M\to\infty$ 时，
> $$
> F(0)=\frac{2c^2M^2}{c^2+M^2}\to2c^2,\qquad
> F(M)=\frac{c^2M^2}{c^2+M^2}\to c^2. \tag{14.1}
> $$
> 更强地，对 $t\le M/2$，
> $$
> F(t)\ge 2\rho(M^2/4)=\frac{2c^2M^2/4}{c^2+M^2/4}\to2c^2, \tag{14.2}
> $$
> 而 $F(M)<c^2$；故对充分大 $M$，全局最小位于 $(M/2,M]$（外点簇），不是内点位姿 $t=0$。即使前向求解器是**精确全局**的，它也会返回外点位姿。

**证明.** 直接代入 GM：
$$
F(0)=\rho(0)+2\rho(M^2)=0+2\frac{c^2M^2}{c^2+M^2}\to2c^2,
$$
$$
F(M)=\rho(M^2)+2\rho(0)=\frac{c^2M^2}{c^2+M^2}+0<c^2.
$$
对 $t\le M/2$，$(t-M)^2\ge M^2/4$，故
$$
F(t)\ge 2\rho((t-M)^2)\ge2\rho(M^2/4)
=\frac{2c^2M^2/4}{c^2+M^2/4}\to2c^2.
$$
因此对充分大 $M$，$F(t)>F(M)$ 对所有 $t\le M/2$ 成立；又 $t>M$ 时 $t^2$ 与 $(t-M)^2$ 都递增，故 $F$ 在 $[M,\infty)$ 递增；全局最小在 $(M/2,M]$ 内，即外点簇附近。$\square$

**含义**：
1. 这不是 GNC 的局部极小失败，而是**模型/可辨识性**失败：当相干外点多于内点时，鲁棒代价的全局最小本身就是错的。任何基于该代价的方法（包括全局 SDP、GNC、我们）都无法区分。
2. 源文档“从任意严重外点污染中跳出局部极小”是过度声明。正确说法：**内层加权 QPEP 是全局的；外层 GNC 只保证稳定点；且即使全局求解 $F_{\mathrm{rob}}$，相干多数外点仍可使其全局最小错误。**
3. 实际含义：需要报告**崩溃点/外点率-召回曲线**，并在训练/部署中加一致性检查（inlier count、残差分布、$H_{\mathrm{rob}}$ 谱），不能宣称“无需 RANSAC 就任意鲁棒”。

**GNC 的额外失败模式（未证，标 GAP）**：$\mu$-同伦的全局最小路径在分岔点可能跳变；GNC 在每轮取全局最小，遇到分岔会跳到另一分支，最终可能落在 $F_{\mathrm{rob}}$ 的局部极小而非全局极小。构造显式一维反例是 `math_questions.md` P0-2。当前只能陈述机制，不能给定理。

---

## 15. Corollary 3.1（二阶外点翻转稳定性证书）

> **Corollary 3.1.** 在 GNC 不动点 $(z^*,w^*)$ 处：
>
> (a) 若 $H_{\mathrm{rob}}\succ0$，则 $(z^*,w^*)$ 是联合 BR 目标 $J$ 的严格局部极小，$z^*$ 是 $F_{\mathrm{rob}}$ 的严格局部极小；IFT 分支在 $\theta$ 小扰动下保持。
>
> (b) 若 $H_{\mathrm{rob}}$ 有 $k\ge1$ 个负特征值，则 $(z^*,w^*)$ 是 $J$ 的指标为 $k$ 的鞍点，$z^*$ 是 $F_{\mathrm{rob}}$ 的鞍点（至少存在下降方向）。
>
> (c) 若 $\lambda_{\min}(H_{\mathrm{rob}})$ 从正变负，稳定点分支发生分岔（鞍结/叉式）；在分岔点 $H_{\mathrm{rob}}$ 奇异，式 (9.3) 的梯度范数发散。数值上可用 $\lambda_{\min}(H_{\mathrm{rob}})$ 作为“翻转裕度”监测。
>
> (d) 该证书是**二阶局部**证书：它不证明 $z^*$ 是全局最小，也不证明前向对抗鲁棒。它只回答“这个不动点是不是一个稳定的局部极小，以及它离分岔有多远”。

**证明.** (a) 由 Lemma 3.3（$H_J\succ0$）与 $F_{\mathrm{rob}}$ 的 Schur 补关系。(b) 由惯性定理：$H_J$ 的负特征值数 = $H_{\mathrm{rob}}$ 的负特征值数（在 $H_{\mathrm{IRLS}}\succeq0,\Psi''\succ0$ 下）。(c) 特征值连续；过零时 IFT 失效，稳定点个数改变。(d) 由 Theorem 3.2 的局部性。$\square$

**备注 15.1**：不要把 $\lambda_{\min}(H_{\mathrm{rob}})>0$ 当作“鲁棒性证书”。一个错误的位姿（外点一致解）完全可以是 $F_{\mathrm{rob}}$ 的严格局部极小，$H_{\mathrm{rob}}\succ0$。证书只说明“局部稳定/可微”，不说明“正确”。

---

## 16. 边界情形与失败模式

### 16.1 $H_{\mathrm{rob}}$ 奇异/不定

- **奇异**：$\lambda_{\min}=0$。稳定点分支分岔；式 (9.3) 发散。数值上表现为 $H_{\mathrm{rob}}$ 条件数爆炸、梯度范数尖峰。
- **不定**：负特征值。$z^*$ 是鞍点；GNC 可能停在鞍点（因为 $z$-步只保证加权全局最小，不保证 $F_{\mathrm{rob}}$ 的二阶条件）。
- **近奇异**：$\lambda_{\min}$ 很小但非零。IFT 仍成立，但常数 $\mu_0^{-1}$ 很大，$O(\|\mathbf r\|^{-4})$ 的实际常数爆炸。

### 16.2 GNC 的全局性失败

1. **模型崩溃**（Prop 3.7）：相干多数外点使 $F_{\mathrm{rob}}$ 全局最小错误。这是根本限制。
2. **$\mu$-同伦分岔**：全局最小路径不连续，GNC 跳到另一分支；最终可能是局部极小。**未证反例，P0-2。**
3. **内层求解失败**：Gröbner 漏根、重根/退化、$G(w)$ 奇异、数值精度不足。QPEP 是“代数全局”而非“凸证书”，失败时无保证。
4. **$\mu_0$ 太小**：初始就处在非凸区域，$w$-步可能把内点也压掉；$\mu_0$ 应按 (4.8) 选。
5. **$c$ 误设**：$c$ 太小 → 内点被当外点；$c$ 太大 → 外点权重不够低。$c$ 应与残差尺度/噪声匹配；可用 MAD 或学习。

### 16.3 边界半外点最毒

由 (10.8)，$\lambda_\parallel$ 在 $e=c^2$ 取最小值 $-1/2$，在 $e\to\infty$ 反而趋于 $0$。因此：
- 粗大外点的梯度按 $O(\|\mathbf r\|^{-4})$ 衰减（Prop 3.6）；
- **半外点**（$e\approx c^2$）的梯度最大，是训练中最危险的样本；
- 这解释了源文档“单独微扰 #20 半外点 cos$=-0.079$”的数值现象；但 6D 的符号仍需 $H^{-1}$ 联合谱（Theorem 3.3(d)）。

### 16.4 各向异性 $W_i$ 与尺度

- $W_i$ 的各向异性会改变“残差方向”；$\lambda_\parallel$ 的公式在 $W_i^{1/2}$ 度量下成立，但 $W_i$ 的尺度与 $c$ 的可辨识性耦合：$W_i\to sW_i$ 与 $c^2\to sc^2$ 可能给出相同的权重，导致尺度不可辨识。实验中应固定 $W_i$ 的尺度（如 $\det W_i=1$）或学习 log-Cholesky。
- 若 $W_i$ 由网络学习，$\partial\Phi/\partial\theta$ 需加 $\rho'\mathbf r_i^\top(\partial W_i/\partial\theta)\mathbf r_i$ 项；本文公式默认 $W_i$ 固定。

### 16.5 多全局最优与 tie

- 对称物体/近平面场景可能有多个全局最小；此时全局 argmin 不连续，IFT 只描述所选分支。训练目标应避免在 tie 附近采样，或用连续松弛（EPro-PnP 式）处理。
- 若 GNC 的两个分支代价接近，梯度会很大（$H_{\mathrm{rob}}$ 近奇异），需要梯度裁剪/阻尼。

### 16.6 PnP/重投影误差

- 重投影误差不是多项式，QPEP 需用代数误差/齐次化；本文的 $\lambda_i=4\mathbf u_i^\top W_i\mathbf r_i$ 对“残差对 $\mathbf t$ 仿射、点项 2 次齐次”成立。重投影误差的 $\lambda_i$ 需按具体残差重算，不能直接套用。
- 正深度/手性约束不在本文范围（那是 Prototype 2 的线）。

### 16.7 数值实现

- $H_{\mathrm{rob}}$ 的组装：$H_i^E$ 显式（$7\times7$）或矩阵-向量乘；$O(N)$ 累加；反向解 $6\times6$ 线性系统。
- 有限差分校验：步长 $10^{-5}$–$10^{-4}$；$H_{\mathrm{rob}}$ 近奇异时 FD 不可靠。
- 批处理：每个 batch 元素独立 GNC；QPEP 求解器若为 CPU C++，需批量化/异步；这是工程风险。

---

## 17. 与源文档的差异（逐条）

| # | 源文档声明 | 本文件修正 | 严重度 |
|---|---|---|---|
| 1 | “GNC 从任意严重外点污染中跳出局部极小” | 内层加权 QPEP 全局；外层 GNC 只保证坐标稳定点；相干多数外点可使全局最小错误（Prop 3.7）；$\mu$-同伦可分岔（GAP） | **高**（过度声明） |
| 2 | “零展开精确鲁棒不动点隐微分” | 对**所选稳定点分支**精确；不是全局 argmin 的导数；需 $H_{\mathrm{rob}}$ 可逆；GNC 切换分支时失效 | 中高 |
| 3 | $H_{\mathrm{rob}}=\sum\rho'\nabla^2e+\sum\rho''\nabla e\nabla e^\top$ | 若 $\nabla^2e$ 为欧氏，缺 $-\lambda_{\mathrm{rob}}\operatorname{diag}(I_3,0)$；应使用黎曼 Hessian 或 (7.5) | **高**（关键 GAP） |
| 4 | “可微存在条件 $\lambda_{\min}(H_{\mathrm{rob}})>0$” | IFT 需**可逆**；$\lambda_{\min}>0$ 是严格局部极小/分支保持；GNC 不动点还需加权全局最小唯一 | 中 |
| 5 | “$O(\|\mathbf r\|^{-4})$ 免疫定律” | 逐观测界，需 $H_{\mathrm{rob}}^{-1}$ 一致有界；前向偏差 $O(\|\mathbf r\|^{-3})$；半外点最毒；不是鲁棒性保证 | 中高 |
| 6 | “Naive IRLS 方向完全相反（cos$=-0.079$）” | 一维严格命题（Prop 3.4）；6D 需 $H^{-1}$ 联合谱，cos 只是数值例 | 中 |
| 7 | “定理 3.1 严格等价” | BR 对偶逐点精确；GNC 算法不是全局算法；“等价”只对 $\min_{z,w}$ 的**值**成立 | 中 |
| 8 | “双向零升维” | 旋转维数不升、$N$ 不升；但 $w\in\mathbb R^N$ 仍要存，总显存 $O(N)$；反向 $6\times6$ | 低（措辞） |
| 9 | “$\lambda=4J$” | $\lambda_i=4\mathbf u_i^\top W_i\mathbf r_i\neq4e_i$；数值差 51.0 | 中 |
| 10 | “二阶外点翻转稳定性证书” | 只是局部二阶证书，不证明全局/正确/对抗鲁棒；错误位姿也可 $H_{\mathrm{rob}}\succ0$ | 中 |
| 11 | Black–Rangarajan 对偶被标为 “CVPR 2014” | 原始文献是 Black & Rangarajan, **IJCV 19(1):57–91, 1996**（DOI 10.1007/BF00131148）；CVPR 2014 不是原始出处 | 低（引用） |
| 12 | GNC 被标为 “Yang et al. CVPR 2020” | GNC 的期刊版是 **RA-L 5(2):1127–1134, 2020**；CVPR 2020 是 TEASER（同组，GNC-TLS 的应用）。两篇应分别引用 | 低（引用） |

---

## 18. 未证明的 GAP 清单

| 编号 | GAP | 影响 | 计划 |
|---|---|---|---|
| GAP-1 | 6D 位姿层面的变号命题：只证了映射层面 (10.6) 与一维 (Prop 3.4)；6D 的 $\partial z/\partial y$ 符号需要 $H^{-1}$ 与 $\partial\Phi/\partial y$ 的联合谱条件 | 论文若要写“6D 方向反转”需要额外定理 | P1-3：给出可证的特殊情形（如 $H$ 与 $\partial\Phi/\partial y$ 可交换、或标量平移子问题） |
| GAP-2 | GNC $\mu$-同伦分岔的显式反例（最终落局部极小而非全局） | “无全局保证”只能靠机制说明 | P0-2：一维/3D 数值搜索 + 区间算术 |
| GAP-3 | $O(\|\mathbf r\|^{-4})$ 的**一致** $H_{\mathrm{rob}}^{-1}$ 界：目前是条件 (B2) | 界的实际常数可能爆炸 | P1-4：在可观测性/非退化条件下证明 $\mu_0>0$ |
| GAP-4 | 重投影误差的 $\lambda_i$ 与 QPEP 多项式形式 | PnP 实验的公式适用性 | P1：用代数 PnP 误差，或对重投影做齐次化 |
| GAP-5 | TLS 的非光滑对偶（$w$ 不唯一、二值权重） | 统一框架的完整性 | P2：把 TLS 作为极限情形，单独给 GNC-TLS 调度 |
| GAP-6 | $\lambda_{\min}(H_{\mathrm{rob}})$ 与“外点翻转”概率的定量关系 | 证书的实用解释 | P1-2：Morse 理论 + 数值相图 |
| GAP-7 | 交替下降的收敛速率 | 只证单调收敛 | P2：在强凸/孤立条件下给线性速率 |
| GAP-8 | GPU 批量 QPEP 的数值完备性 | 工程可复现性 | P2：与 LibQPEP 对拍 + 失败率统计 |

---

## 19. 参考文献（本文件用到的）

- Black, M. J., & Rangarajan, A. (1996). On the unification of line processes, outlier rejection, and robust statistics with applications in early vision. *IJCV*, 19(1), 57–91. DOI [10.1007/BF00131148](https://doi.org/10.1007/BF00131148).
- Yang, H., Antonante, P., Tzoumas, V., & Carlone, L. (2020). Graduated non-convexity for robust spatial perception: From non-minimal solvers to global outlier rejection. *IEEE RA-L*, 5(2), 1127–1134. DOI [10.1109/LRA.2020.2969178](https://doi.org/10.1109/LRA.2020.2969178).
- Yang, H., Shi, J., & Carlone, L. (2020). TEASER: Fast and accurate 3D registration with extreme outlier rates. *CVPR 2020*.
- Wu, J., Zheng, Y., Gao, Z., Jiang, Y., Hu, X.-C., Zhu, Y.-L., Jiao, J.-H., & Liu, M. (2022). Quadratic pose estimation problems: Globally optimal solutions, solvability/observability analysis, and uncertainty description. *IEEE T-RO*, 38(5), 3314–3335. DOI [10.1109/TRO.2022.3155880](https://doi.org/10.1109/TRO.2022.3155880).
- Holmes, C., Dümbgen, F., & Barfoot, T. D. (2025). SDPRLayers: Certifiable backpropagation through polynomial optimization problems in robotics. *IEEE T-RO*, 41, 4120–4138. DOI [10.1109/TRO.2025.3578228](https://doi.org/10.1109/TRO.2025.3578228); arXiv:[2405.19309](https://arxiv.org/abs/2405.19309).
- Absil, P.-A., Mahony, R., & Sepulchre, R. (2008). *Optimization Algorithms on Matrix Manifolds*. Princeton University Press.
- Geman, S., & McClure, D. E. (1985). Bayesian image analysis: An application to single photon emission tomography. *Proc. Amer. Statist. Assoc.*, 12–18.
