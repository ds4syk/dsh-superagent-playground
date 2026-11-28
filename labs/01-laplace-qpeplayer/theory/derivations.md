# 关键推导中间步骤（Laplace-QPEPLayer）

> 配套 `theory/proofs.md`。这里只写“怎么算出来”的中间步骤，便于审稿人/合作者逐行核对。所有公式固定以下约定：
> - $q=[q_0,q_1,q_2,q_3]^\top$，$R(q)$ 为标准四元数旋转矩阵，$\mathrm{vec}$ 行主序；
> - $v_2(q)=[q_0^2,q_1^2,q_2^2,q_3^2,\sqrt2q_0q_1,\sqrt2q_0q_2,\sqrt2q_0q_3,\sqrt2q_1q_2,\sqrt2q_1q_3,\sqrt2q_2q_3]^\top$；
> - $x_R(q)=[\|q\|^2,\mathrm{vec}(R(q))^\top]^\top$（齐次提升）；
> - 数值验证脚本：`verify/verify_theorem0.py`、`verify/verify_theorem11.py`。

---

## 1. $U$ 的显式构造与 $x_R=2Uv_2$ 的验证

### 1.1 为什么 $U$ 一定存在

$x_R$ 的 10 个分量都是 $q$ 的二次型。二次型空间 $\mathrm{Sym}^2(\mathbb R^4)$ 的维数是 $\binom{4+1}{2}=10$，而
$$\{q_0^2,q_1^2,q_2^2,q_3^2,\;q_0q_1,q_0q_2,q_0q_3,q_1q_2,q_1q_3,q_2q_3\}$$
是一组基。$v_2$ 只是把 6 个交叉项乘了 $\sqrt2$，仍是基。因此存在唯一 $B\in\mathbb R^{10\times10}$ 使 $x_R=Bv_2$。

### 1.2 正交性

对任意 $q$，$R(q)$ 满足 $R(q)^\top R(q)=\|q\|^4I_3$，故 $\|R(q)\|_F^2=3\|q\|^4$。于是
$$\|x_R\|^2=\|q\|^4+3\|q\|^4=4\|q\|^4.$$
又 $\|v_2\|^2=\sum_iq_i^4+2\sum_{i<j}q_i^2q_j^2=(\sum_iq_i^2)^2=\|q\|^4$。所以
$$v_2^\top B^\top Bv_2=4\|v_2\|^2\quad\forall q.$$
$\{v_2(q)\}$ 张成 $\mathbb R^{10}$，故 $B^\top B=4I$，$U=B/2\in O(10)$。

### 1.3 显式块结构

把行按 $(x_0,R_{11},R_{22},R_{33})$ 重排，列按 $(q_0^2,q_1^2,q_2^2,q_3^2)$ 排，得到
$$\begin{bmatrix}x_0/2\\R_{11}/2\\R_{22}/2\\R_{33}/2\end{bmatrix}
=\frac12\underbrace{\begin{bmatrix}1&1&1&1\\1&1&-1&-1\\1&-1&1&-1\\1&-1&-1&1\end{bmatrix}}_{H_4}
\begin{bmatrix}q_0^2\\q_1^2\\q_2^2\\q_3^2\end{bmatrix}.$$
$H_4H_4^\top=4I_4$，故 $\tfrac12H_4$ 正交。

其余 6 行是 $R_{12},R_{13},R_{21},R_{23},R_{31},R_{32}$。例如
$$R_{12}=2(q_1q_2-q_0q_3)=\sqrt2\,v_{2,8}-\sqrt2\,v_{2,7},$$
所以第 $R_{12}$ 行在 $v_2$ 坐标下是 $(0,0,0,0,0,0,-\tfrac{\sqrt2}{2},\tfrac{\sqrt2}{2},0,0)$。类似地：
$$
\begin{aligned}
R_{13}&=2(q_1q_3+q_0q_2)=\sqrt2v_{2,9}+\sqrt2v_{2,6},\\
R_{21}&=2(q_1q_2+q_0q_3)=\sqrt2v_{2,8}+\sqrt2v_{2,7},\\
R_{23}&=2(q_2q_3-q_0q_1)=\sqrt2v_{2,10}-\sqrt2v_{2,5},\\
R_{31}&=2(q_1q_3-q_0q_2)=\sqrt2v_{2,9}-\sqrt2v_{2,6},\\
R_{32}&=2(q_2q_3+q_0q_1)=\sqrt2v_{2,10}+\sqrt2v_{2,5}.
\end{aligned}
$$
按 $(q_0q_3,q_1q_2)$、$(q_0q_2,q_1q_3)$、$(q_0q_1,q_2q_3)$ 三对排列，每对是一个
$$\frac1{\sqrt2}\begin{bmatrix}1&-1\\1&1\end{bmatrix}\quad\text{或}\quad\frac1{\sqrt2}\begin{bmatrix}1&1\\1&-1\end{bmatrix}$$
型的 $2\times2$ 正交块。故 $U$ 在适当行列置换下是 $\mathrm{diag}(\tfrac12H_4,\tfrac1{\sqrt2}H_2,\tfrac1{\sqrt2}H_2,\tfrac1{\sqrt2}H_2)$，正交。

### 1.4 数值/符号验证

`verify_theorem0.py` 用 sympy 对符号 $q$ 做精确多项式恒等检查：
- `U_orthogonal_exact = true`；
- `xR_equals_2Uv2_exact = true`；
- `xR_norm_identity = true`；
- $U$ 的条目只取 $\{0,\pm\tfrac12,\pm\tfrac{\sqrt2}{2}\}$（`U_entries`）。

**常见错误**：把 $x_R$ 的第一分量写成字面 1。此时 $2Uv_2$ 的第一分量是 $\|q\|^2$，只有在 $\|q\|=1$ 上才等于 1；离开球面恒等式不成立。论文中写“homogeneous lift”。

---

## 2. 21 条约束、秩 20 与二次理想

### 2.1 21 条约束的矩阵形式

令 $x=[x_0,r_{11},r_{12},\dots,r_{33}]$（行主序），$x_0=\|q\|^2$。每条约束写成 $x^\top A_ix=0$：

- **列约束**（$c_j=[r_{1j},r_{2j},r_{3j}]^\top$）：
  - 3 条范数：$\|c_j\|^2-x_0^2=0$，$A$ 在 $r_{1j},r_{2j},r_{3j}$ 对角为 $+1$、$x_0$ 对角为 $-1$；
  - 3 条点积：$c_a\cdot c_b=0$（$a<b$），$A$ 在对应 3 对 $(r_{ia},r_{ib})$ 上为 $\tfrac12$。
- **行约束**（$r_i^\top$ 是第 $i$ 行）：同理 3 条范数 + 3 条点积。
- **9 条叉积**：$c_a\times c_b-x_0c_c=0$。以 $(a,b,c)=(1,2,3)$ 的第一分量为例：
  $$r_{21}r_{32}-r_{31}r_{22}-x_0r_{13}=0,$$
  对应 $A$ 在 $(r_{21},r_{32})$、$(r_{31},r_{22})$ 上 $\pm\tfrac12$，在 $(x_0,r_{13})$ 上 $-\tfrac12$。其余 8 条循环排列。

### 2.2 秩与唯一线性相关

对 21 个 $55$ 维对称矩阵向量做精确有理消元，秩 $=20$。唯一的线性相关是
$$\sum_{j=1}^3(\|c_j\|^2-x_0^2)-\sum_{i=1}^3(\|r_i\|^2-x_0^2)=\|R\|_F^2-3x_0^2-(\|R\|_F^2-3x_0^2)=0,$$
即 $\operatorname{tr}(R^\top R)=\operatorname{tr}(RR^\top)$。注意：**只有 1 条**相关（秩 20 而不是 19）；这依赖“3 范数 + 3 点积”各 6 条都取齐。只取 6 个非对角点积会得到 15 条、秩 15。

### 2.3 核的维数

定义线性映射
$$\phi:\mathrm{Sym}^2(\mathbb R^{10})\to\mathbb R[q]_4,\qquad Q\mapsto Q(x_R(q)).$$
$\dim\mathrm{Sym}^2=\binom{11}{2}=55$，$\dim\mathbb R[q]_4=\binom{7}{4}=35$。$\phi$ 满射：$1\cdot R_{ij}$ 给出全部二次单项式，两个二次单项式之积张成全部四次单项式。故 $\operatorname{rank}\phi=35$，$\dim\ker\phi=20$。

### 2.4 相等，而不只是同维

- **包含**：每条约束在 $x_R(q)$ 上恒为 0（$R(q)\in SO(3)$），且代入后是四次齐次多项式、在球面上为 0，故恒等于 0；所以 $A_i\in\ker\phi$。
- **秩**：21 条张成 20 维。
- 20 维子空间 $\subseteq$ 20 维核 ⇒ **相等**。

在 $v_2$ 坐标下，$\ker\phi=\mathcal I_2(\nu_2(\mathbb R P^3))$，由 $4\times4$ 对称矩阵
$$W=\begin{bmatrix}v_1&v_5/\sqrt2&v_6/\sqrt2&v_7/\sqrt2\\ v_5/\sqrt2&v_2&v_8/\sqrt2&v_9/\sqrt2\\ v_6/\sqrt2&v_8/\sqrt2&v_3&v_{10}/\sqrt2\\ v_7/\sqrt2&v_9/\sqrt2&v_{10}/\sqrt2&v_4\end{bmatrix}$$
的 $2\times2$ 子式张成（$\nu_2(\mathbb P^3)$ 的经典生成元；Harris 1992）。36 个子式，但二次部分维数 20。

---

## 3. 切基 $E(q)$、旋转向量与因子 4

### 3.1 $E(q)$ 是右乘虚单位

把四元数写成 $q=q_0+q_1i+q_2j+q_3k$。直接计算
$$q\otimes i=(-q_1,q_0,q_3,-q_2),\quad q\otimes j=(-q_2,-q_3,q_0,q_1),\quad q\otimes k=(-q_3,q_2,-q_1,q_0),$$
正是 $E(q)$ 的三列。因此
$$E(q)=[\,q\otimes i\;\;q\otimes j\;\;q\otimes k\,].$$
四元数乘法保范 ⇒ $E^\top E=I_3$；$q\otimes i$ 与 $q$ 正交 ⇒ $q^\top E=0$。$E(-q)=-E(q)$。

### 3.2 $\delta q=E\delta\xi$ 对应的旋转向量是 $2\delta\xi$

设 $q(t)=q\otimes\exp(t\xi)$，$\xi\in\mathbb R^3$，$|\xi|=1$。则
$$\frac{d}{dt}q(t)\Big|_{t=0}=q\otimes(0,\xi)=E(q)\xi.$$
而四元数 $\exp(t\xi)=(\cos t,\sin t\,\hat\xi)$ 对应的旋转是绕 $\hat\xi$ 转 $2t$。故 $\delta q=E\delta\xi$ 对应的旋转向量是 $\omega=2\delta\xi$。这解释了：
- 若用 $E$ 基（$|\delta q|=|\delta\xi|$），黎曼 Hessian 是 $H_{\mathbb S^3}$；
- 若用旋转向量 $\omega$，$J\approx J_*+\tfrac12\delta\xi^\top H_{\mathbb S^3}\delta\xi=J_*+\tfrac12(\omega/2)^\top H_{\mathbb S^3}(\omega/2)=J_*+\tfrac12\omega^\top(\tfrac14H_{\mathbb S^3})\omega$，故 $H_\omega=\tfrac14H_{\mathbb S^3}$，$\det H_\omega=\det H_{\mathbb S^3}/64$。
- QPEP 论文的“观测度/协方差”用的是它自己的参数化；引用时必须说明“up to a constant factor 4”。

---

## 4. Euler 乘子 $\lambda^*=4J$

$v_2$ 对 $q$ 是二次齐次 ⇒ $J=v_2^\top Mv_2$ 是四次齐次。Euler 齐次函数定理：
$$q^\top\nabla J(q)=4J(q)\quad\forall q.$$
Lagrange 条件（约束 $\tfrac12(q^\top q-1)=0$）：
$$\nabla J(q^*)=\lambda^*q^*.$$
左乘 $q^{*\top}$：
$$\lambda^*=q^{*\top}\nabla J(q^*)=4J(q^*).$$
注意 $\lambda^*$ 不是独立变量；它由 $q^*$ 处的目标值闭式给出。这是 QPEP 的 $3\times3$ 反向能成立的关键。

---

## 5. 黎曼 Hessian 的完整推导

### 5.1 欧氏梯度与 Hessian

令 $J_v=\partial v_2/\partial q\in\mathbb R^{10\times4}$。则
$$\nabla J=2J_v^\top Mv_2,$$
$$\nabla^2J=2J_v^\top MJ_v+2\sum_{a=1}^{10}(Mv_2)_aH_{v,a},$$
其中 $H_{v,a}=\partial^2v_{2,a}/\partial q\partial q$ 是常数对称矩阵：
- 对 $a=1,\dots,4$（$q_{a-1}^2$）：$H_{v,a}=2e_{a-1}e_{a-1}^\top$；
- 对交叉项 $\sqrt2q_iq_j$：$H_{v,a}=\sqrt2(e_ie_j^\top+e_je_i^\top)$。

### 5.2 从 $F=E^\top\nabla J$ 到 $H_{\mathbb S^3}$

$F(q,\theta)=E(q)^\top\nabla J(q,\theta)=0$。沿 $\delta q=E\delta\xi$ 微分：
$$\delta F=(\delta E)^\top\nabla J+E^\top\nabla^2J\,\delta q.$$
在平稳点 $\nabla J=\lambda q$。由 $E(q)^\top q\equiv0$ 对 $q$ 微分：
$$(\delta E)^\top q+E^\top\delta q=0\;\Longrightarrow\;(\delta E)^\top q=-E^\top\delta q.$$
故
$$(\delta E)^\top(\lambda q)=-\lambda E^\top\delta q,$$
$$\delta F=-\lambda E^\top\delta q+E^\top\nabla^2J\delta q=E^\top(\nabla^2J-\lambda I)E\,\delta\xi.$$
代入 $\lambda=4J$：
$$\boxed{H_{\mathbb S^3}=E^\top(\nabla^2J-4J I_4)E.}$$
这是 $\mathbb S^3$ 上带诱导度量的黎曼 Hessian（在 $E$ 基下）。

### 5.3 隐式微分

$F(q^*(\theta),\theta)=0$ 全微分：
$$H_{\mathbb S^3}\,\delta\xi+E^\top\partial_\theta\nabla J=0.$$
$\partial_\theta\nabla J=2J_v^\top(\partial_\theta M)v_2$（$v_2$ 与 $\theta$ 无关）。故
$$\delta\xi=-H_{\mathbb S^3}^{-1}E^\top\big(2J_v^\top(\partial_\theta M)v_2\big),$$
$$\boxed{\frac{\partial q^*}{\partial\theta}=-E H_{\mathbb S^3}^{-1}E^\top\big(2J_v^\top(\partial_\theta M)v_2\big).}$$
平移 $t^*=-G^{-1}Dv_2(q^*)$：
$$\boxed{\frac{\partial t^*}{\partial\theta}=-G^{-1}\Big((\partial_\theta G)t^*+(\partial_\theta D)v_2+D J_v\frac{\partial q^*}{\partial\theta}\Big).}$$

### 5.4 符号检查

- $H_{\mathbb S^3}$ 的 $-4J$ 项来自球面约束的第二基本形式；若漏掉，有限差分 Hessian 会差一个 $-4J E^\top E=-4J I$。
- $\delta E^\top(\lambda q)=-\lambda E^\top\delta q$ 的负号来自 $E^\top q=0$ 的微分；写成正号会得到 $H=E^\top(\nabla^2J+4JI)E$，与数值不符。
- 数值验证：解析 $H$ vs 拉回有限差分相对误差 $9.6\times10^{-8}$。

---

## 6. 对跖双覆盖不变性

- $v_2(-q)=v_2(q)$；
- $J_v(-q)=-J_v(q)$（二次型导数是一次齐次）；
- $\nabla J(-q)=-\nabla J(q)$；
- $\nabla^2J(-q)=\nabla^2J(q)$（$J_v^\top MJ_v$ 是 $J_v$ 的二次型，$Mv_2$ 不变）；
- $E(-q)=-E(q)$。

于是
$$H_{\mathbb S^3}(-q)=(-E)^\top(\nabla^2J-4JI)(-E)=H_{\mathbb S^3}(q).$$
隐式解：$\delta\xi(-q)=-\delta\xi(q)$，故
$$\frac{\partial(-q^*)}{\partial\theta}=E(-q)\delta\xi(-q)=(-E)(-\delta\xi)=E\delta\xi=\frac{\partial q^*}{\partial\theta}.$$
等一下：$q^*$ 在 $-q^*$ 处的微分是 $d(-q^*)=-dq^*$；而 $E(-q)\delta\xi(-q)=(-E)(-\delta\xi)=E\delta\xi=dq^*$。所以“$q^*$ 的微分”在 $\pm$ 处反号，但 $R$ 的微分
$$\frac{\partial\,\mathrm{vec}R}{\partial\theta}=J_v(q)\frac{\partial q^*}{\partial\theta}$$
满足 $J_v(-q)(-dq^*)=J_v(q)dq^*$，严格不变。数值验证：$H$、$dq^*$ 关系、$dR$ 关系误差均为 0（机器精度）。

---

## 7. 平移精确高斯积分

$$J(q,t)=v^\top M_0v+2t^\top Dv+t^\top Gt,\qquad G\succ0.$$
对 $t$ 配方：
$$J=(t+G^{-1}Dv)^\top G(t+G^{-1}Dv)+v^\top(M_0-D^\top G^{-1}D)v.$$
令 $u=t+G^{-1}Dv$：
$$\int_{\mathbb R^3}e^{-J/\tau}dt=e^{-J_{\text{red}}/\tau}\int_{\mathbb R^3}e^{-u^\top Gu/\tau}du
=\frac{(\pi\tau)^{3/2}}{\sqrt{\det G}}e^{-J_{\text{red}}/\tau}.$$
这是**精确**的，无近似。注意 $G$ 必须正定；$G$ 奇异时积分发散，$t^*$ 不唯一。

---

## 8. 旋转 Laplace：常数、测度与 $O(\tau)$

### 8.1 局部坐标

在 $q^{(k)}$ 附近令
$$q(u)=\frac{q^{(k)}+E_ku}{\|q^{(k)}+E_ku\|},\qquad E_k=E(q^{(k)}).$$
则 $q(0)=q^{(k)}$，$dq/du|_0=E_k$，$dq=du\,(1+O(\|u\|^2))$，且
$$J(q(u))=J_k+\tfrac12u^\top H_ku+O(\|u\|^3).$$
标准 Laplace：
$$\int_{\mathbb R^3}e^{-J(q(u))/\tau}du=e^{-J_k/\tau}\frac{(2\pi\tau)^{3/2}}{\sqrt{\det H_k}}\big(1+O(\tau)\big).$$

### 8.2 从 $S^3$ 到 $SO(3)$

$S^3\to SO(3)$ 是二重覆盖。取 $SO(3)$ 的 Haar 测度 $\int_{SO(3)}dR=8\pi^2$；局部 $dR=8dq$（长度缩放为 2 ⇒ 体积缩放 $2^3=8$），二重覆盖给因子 $1/2$，故
$$\int_{SO(3)}f\,dR=\tfrac12\int_{S^3}8f\,dq=4\int_{S^3}f\,dq.$$
每个 $SO(3)$ 极小对应 $S^3$ 上的一对 $\pm q^{(k)}$，两片各贡献一份，合计
$$\int_{SO(3)}e^{-J/\tau}dR=8(2\pi\tau)^{3/2}\sum_k\frac{e^{-J_k/\tau}}{\sqrt{\det H_k}}\big(1+O(\tau)\big).$$
权重
$$w_k=\frac{e^{-J_k/\tau}/\sqrt{\det H_k}}{\sum_je^{-J_j/\tau}/\sqrt{\det H_j}}=\mathrm{softmax}_k\Big(-\frac{J_k}{\tau}-\frac12\log\det H_k\Big).$$
常数 $8(2\pi)^{3/2}$ 与 $k$ 无关，不影响 $w_k$。

### 8.3 误差阶的数值证据

`verify_theorem11.py` 在随机 $M$ 的一个非退化局部极小上做重要性采样，比较
$$I(\tau)=\int_{\mathbb R^3}e^{-J(q(u))/\tau}du\quad\text{与}\quad e^{-J_k/\tau}\frac{(2\pi\tau)^{3/2}}{\sqrt{\det H_k}}.$$
结果：

| $\tau$ | 相对误差 |
|-|-|
| 0.2 | $8.26\times10^{-2}$ |
| 0.1 | $3.81\times10^{-2}$ |
| 0.05 | $1.86\times10^{-2}$ |
| 0.025 | $9.25\times10^{-3}$ |

log–log 斜率 $1.05$，即误差 $O(\tau)$。这推翻“旋转方向精确积分”的说法；只有平移方向精确。

### 8.4 分岔处为什么失效

当两个极小 $q^{(1)},q^{(2)}$ 在参数 $\theta_c$ 处合并（鞍结分岔），
- $\Delta J\to0$，两个 Gaussian 的中心靠近；
- $H_k$ 的某个特征值 $\to0$，$1/\sqrt{\det H_k}\to\infty$；
- 三次项不能忽略，积分在合并方向变成 $\int e^{-(au^2+bu^3)/\tau}du$ 型，需要用 Airy 函数做一致渐近；
- 模式数在 $\theta_c$ 两侧不同（一个极小消失），Laplace 和本身不连续。
因此“在分支交叉处解析平滑可微”应归给**精确 Boltzmann 后验**（紧流形上 $e^{-J/\tau}$ 光滑，$Z$ 与期望由控制收敛光滑），而不是 Laplace 近似。Laplace 近似在分岔附近需要一致渐近或数值积分；这是 GAP-1.3。

---

## 9. KL 梯度与三阶导数

### 9.1 Envelope 定理

$J_k(\theta)=J_{\text{red}}(q^{(k)}(\theta);\theta)$。因为 $q^{(k)}$ 是 $J_{\text{red}}$ 的平稳点，
$$\frac{d}{d\theta}J_k=\frac{\partial J}{\partial\theta}\Big|_{q^{(k)}}+\underbrace{(\nabla_qJ)^\top\frac{\partial q^{(k)}}{\partial\theta}}_{=0}
=v_k^\top(\partial_\theta M)v_k.$$
所以一阶项不需要 $\partial q^{(k)}/\partial\theta$；但权重 $w_k$ 依赖 $\theta$，其导数需要 $\nabla_\theta\log\det H_k$。

### 9.2 $\log\det H_k$ 的导数

$$H_k=H(q^{(k)}(\theta),\theta),\qquad
\frac{dH_k}{d\theta}=\frac{\partial H}{\partial\theta}\Big|_{q^{(k)}}+\frac{\partial H}{\partial q}\Big|_{q^{(k)}}\frac{\partial q^{(k)}}{\partial\theta}.$$
$$\nabla_\theta\log\det H_k=\operatorname{tr}\Big(H_k^{-1}\frac{dH_k}{d\theta}\Big).$$
$\partial H/\partial q$ 含 $\partial_q\nabla^2J$，即 $J$ 的**三阶 $q$ 导数**；$\partial q^{(k)}/\partial\theta$ 由定理 1.1 的 $3\times3$ 公式给出。因此 Laplace KL 的反向不是“只解一个 $3\times3$”，而是“每个模式一个 $3\times3$ + 一个 $\operatorname{tr}(H^{-1}dH)$”。

### 9.3 权重依赖：协方差项

$$\bar\ell=\sum_kw_k\ell_k,\qquad
\nabla\bar\ell=\sum_kw_k\nabla\ell_k+\sum_k(\nabla w_k)\ell_k.$$
令 $s_k=-\frac1\tau J_k-\frac12\log\det H_k$，$w_k=e^{s_k}/\sum_je^{s_j}$。则
$$\nabla w_k=w_k\Big(\nabla s_k-\sum_jw_j\nabla s_j\Big),$$
$$\sum_k(\nabla w_k)\ell_k=\sum_kw_k\ell_k\nabla s_k-\Big(\sum_kw_k\ell_k\Big)\Big(\sum_jw_j\nabla s_j\Big)
=\mathrm{Cov}_w(\ell_k,\nabla s_k).$$
即
$$\nabla\bar\ell=\sum_kw_k\nabla\ell_k+\mathrm{Cov}_w\Big(\ell_k,\;-\frac1\tau\nabla J_k-\frac12\nabla\log\det H_k\Big).$$
这就是源文档公式的完整形式；注意协方差项里的 $\nabla J_k$ 用 Envelope 定理，$\nabla\log\det H_k$ 用 9.2。

### 9.4 $\tau\to0$

- 后验均值 $\hat q_\tau=\sum_kw_kq^{(k)}$：$\Delta J>0$ 时 $w_k\to\delta_{k^*}$，$\hat q_\tau\to q^{(k^*)}$，梯度趋于定理 1.1。
- $\mathcal L_{\text{KL}}=\frac1\tau J_{\text{gt}}+\log Z$：
  $$\nabla\mathcal L_{\text{KL}}=\frac1\tau\nabla J_{\text{gt}}-\frac12\nabla\log\det G-\sum_kw_k\Big(\frac1\tau\nabla J_k+\frac12\nabla\log\det H_k\Big).$$
  当 $\tau\to0$，第一项与 $-\frac1\tau\nabla J_{k^*}$ 合并为 $\frac1\tau(\nabla J_{\text{gt}}-\nabla J_{k^*})$，一般按 $1/\tau$ 发散。要恢复确定性梯度，用 $\tau\mathcal L_{\text{KL}}$ 或对损失做 $\tau$-缩放。

---

## 10. Choi–Lam 实例的矩阵

$$C(q)=q_0^2q_1^2+q_1^2q_2^2+q_2^2q_0^2+q_3^4-4q_0q_1q_2q_3.$$
在 $v_2$ 基下：
- $q_0^2q_1^2=\tfrac12(\sqrt2q_0q_1)^2=\tfrac12v_{2,5}^2$；
- $q_1^2q_2^2=\tfrac12v_{2,8}^2$；
- $q_2^2q_0^2=\tfrac12v_{2,6}^2$；
- $q_3^4=v_{2,4}^2$；
- $q_0q_1q_2q_3=\tfrac12(\sqrt2q_0q_1)(\sqrt2q_2q_3)=\tfrac12v_{2,5}v_{2,10}$，故 $-4q_0q_1q_2q_3=-2v_{2,5}v_{2,10}$。

所以
$$M_{CL}=\begin{pmatrix}
0&0&0&0&0&0&0&0&0&0\\
0&0&0&0&0&0&0&0&0&0\\
0&0&0&0&0&0&0&0&0&0\\
0&0&0&1&0&0&0&0&0&0\\
0&0&0&0&\tfrac12&0&0&0&0&-1\\
0&0&0&0&0&\tfrac12&0&0&0&0\\
0&0&0&0&0&0&0&0&0&0\\
0&0&0&0&0&0&0&\tfrac12&0&0\\
0&0&0&0&0&0&0&0&0&0\\
0&0&0&0&-1&0&0&0&0&0
\end{pmatrix},$$
（指标按 $v_2$ 顺序；$M_{CL,5,10}=M_{CL,10,5}=-1$ 对应 $-2v_5v_{10}$ 的对称矩阵 $\tfrac12$ 系数）。$M=M_{CL}+2I$ 的最小特征值 $1$。

**非负性**：$a=q_0q_1,b=q_1q_2,c=q_2q_0,d=q_3^2$，AM–GM：
$$a^2+b^2+c^2+d^2\ge4|abcd|^{1/2}=4|q_0q_1q_2q_3|\ge4q_0q_1q_2q_3.$$
**非 SOS**：Choi–Lam（1977）。$M_{CL}$ 不定（最小特征值 $-\tfrac12$），正是“非负但 Gram 矩阵非 PSD”的体现。

---

## 11. 尺度不变性与 $\tau$

若 $M\mapsto sM$（$s>0$），则 $J\mapsto sJ$，$H\mapsto sH$，$\det H\mapsto s^3\det H$。要后验不变：
$$\tau\mapsto s\tau,\qquad \log\det H_k\mapsto\log\det H_k+3\log s.$$
$3\log s$ 与 $k$ 无关，故 $w_k$ 不变；$Z$ 差一个 $s^{-3/2}$ 因子。实践建议：
- 用 $\tau=\tau_0\cdot\mathrm{scale}(M)$，$\mathrm{scale}(M)=\operatorname{tr}(M)/10$ 或 $\lambda_{\max}(M)$；
- $\tau_0$ 可学习，但必须固定尺度归一化，否则 $\tau$ 与 $M$ 的尺度不可辨识；
- $\tau\to\infty$：$w_k\to$ 均匀，梯度信号消失；$\tau\to0$：退化为硬 argmin（均值/argmin 意义），KL 梯度发散（9.4）。
