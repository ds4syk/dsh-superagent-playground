# 定理、假设与证明（Laplace-QPEPLayer）

> 本文件是本线全部数学主张的“单一事实来源”。每条定理都给出：精确陈述、假设、证明（完整或明确标 GAP）、与源文档的差异、边界情形。  
> 证据分级：**[证明]** 完整数学证明；**[符号验证]** 用 sympy 精确计算（脚本可复现）；**[数值验证]** 浮点数值证据；**[GAP]** 未证或仅有启发式论证；**[论文声称]** 源文档/原论文说法，未独立验证。

---

## 0. 作用域声明（必须先读）

源文档写“单帧 $SO(3)$ SDPRLayer 是 $10\times10$ PSD + 21 个约束”。这与 SDPRLayers 的**实际实现**不是同一个对象：

| 对象 | 变量 | 约束 | 出处 |
|-|-|-|-|
| SDPRLayers 实现（P6 矩阵加权位姿） | $x=[h;\mathrm{vec}(C);t]\in\mathbb R^{13}$ | 24 条（6 正交 + 9 handedness + 9 范数），发布版 `SDPPoseEstimator` 把后 18 条标为冗余 | 论文 + 官方仓库；仓库 KB `kb/sdpr-forward-net/01.md` §3.1 有代码级记录 |
| 本文件的“旋转子问题” | $x_R=[\|q\|^2;\mathrm{vec}(R(q))]\in\mathbb R^{10}$ | 21 条（6 列 + 6 行 + 9 叉积） | 源文档 + 本文件构造 |

因此：

- **定理 0.1–0.3 只对“消去 $t$ 后的旋转子问题 10×10 SDP”成立**，不能直接说成“SDPRLayers 实际求解的 13 维 SDP”。
- 本文件把 10×10 的对象明确标为**构造的旋转子问题 SDP**（constructed rotation-subproblem SDP）。它与 QPEP 的 $J_{\text{red}}(q)$ 精确对应；与 13×13 实现的关系是 **[GAP]**（见 Q2）。
- 预实验 0 若使用 10×10 SDP，必须标“构造的旋转子问题 SDP”，不能标成“SDPRLayers 原实现”。

另一条作用域：QPEP 的“全局”= **枚举全部 KKT 根后比较目标**，不是 SDPR 意义上的凸证书。$S^3$ 上“零松弛间隙”是因为**根本没有松弛**，不是因为有一个可验证证书。Laplace 求和依赖“枚举到全部实局部极小”，这是假设 A2，不是定理。

---

## 1. 符号与基本假设

- $q=[q_0,q_1,q_2,q_3]^\top\in\mathbb R^4$，$\|q\|=1$；$R(q)$ 为标准四元数旋转矩阵。
- 齐次提升 $x_R(q)=[\|q\|^2,\mathrm{vec}(R(q))^\top]^\top\in\mathbb R^{10}$，$\mathrm{vec}$ 为行主序（列主序只置换 $U$ 的行）。
- Veronese 向量
  $$v_2(q)=[q_0^2,q_1^2,q_2^2,q_3^2,\sqrt2 q_0q_1,\sqrt2 q_0q_2,\sqrt2 q_0q_3,\sqrt2 q_1q_2,\sqrt2 q_1q_3,\sqrt2 q_2q_3]^\top\in\mathbb R^{10}.$$
- 约化代价 $J_{\text{red}}(q;\theta)=v_2(q)^\top M(\theta)v_2(q)$，$M\in\mathbb S^{10}$ 由 $M_0-D^\top G^{-1}D$ 给出（$G\succ0$）。
- 切基 $E(q)\in\mathbb R^{4\times3}$（列是 $q\otimes i,q\otimes j,q\otimes k$）：
  $$E(q)=\begin{bmatrix}-q_1&-q_2&-q_3\\ q_0&-q_3&q_2\\ q_3&q_0&-q_1\\ -q_2&q_1&q_0\end{bmatrix}.$$
- $\theta$ 是进入 $M_0,D,G$ 的参数（测量、权重、特征），假设 $C^2$（定理 1.1）或 $C^3$（定理 1.3 的 $\log\det H$ 梯度）。

**假设 A0（光滑参数化）**：$M_0(\theta),D(\theta),G(\theta)$ 对 $\theta$ 局部 $C^2$（1.1）/$C^3$（1.3）。  
**假设 A1（平移可观测）**：$G(\theta)\succ0$，从而 $t^*(q)=-G^{-1}Dv_2(q)$ 唯一。  
**假设 A2（根枚举完整）**：代数求解器返回多项式系统在 $\mathbb S^3/\{\pm1\}$ 上的**全部实 KKT 根**（允许重根/漏根时定理 1.2/1.3 失效）。  
**假设 A3（非退化）**：所选局部极小满足 $H_{\mathbb S^3}(q^{(k)})\succ0$；不同极小值互异（$\Delta J>0$）除非显式讨论平局。

---

## 2. 定理 0.1（正交同构）

**定理 0.1 [证明 + 符号验证]**：存在常数 $U\in O(10)$，使得对任意 $q\in\mathbb R^4$
$$x_R(q)=2U\,v_2(q),\qquad \|x_R(q)\|^2=4\|v_2(q)\|^2=4\|q\|^4.$$
在 $\mathbb S^3$ 上 $x_R=[1,\mathrm{vec}(R(q))^\top]^\top$，且第一分量为字面常数 1 的写法**只在球面上**成立。

**证明**：

1. **线性表示**。$x_R$ 的每个分量都是 $q$ 的二次型；$\{q_i^2\}\cup\{q_iq_j\}_{i<j}$ 张成 $\mathrm{Sym}^2(\mathbb R^4)$（10 维），$v_2$ 是其一组基（交叉项带 $\sqrt2$ 归一化）。故存在唯一 $B\in\mathbb R^{10\times10}$ 使 $x_R=Bv_2$。
2. **范数恒等**。对任意 $q$，$R(q)$ 是正交矩阵（$R(q)^\top R(q)=\|q\|^4 I_3$），故 $\|\mathrm{vec}(R(q))\|^2=\|R(q)\|_F^2=3\|q\|^4$。于是
   $$\|x_R\|^2=\|q\|^4+3\|q\|^4=4\|q\|^4=4\|v_2\|^2.$$
   即 $v_2^\top B^\top Bv_2=4\|v_2\|^2$ 对一切 $q$ 成立。$\{v_2(q):q\in\mathbb R^4\}$ 张成 $\mathbb R^{10}$，故 $B^\top B=4I$。
3. **正交化**。令 $U=B/2$，则 $U^\top U=I$，即 $U\in O(10)$，且 $x_R=2Uv_2$。
4. **块结构（显式构造）**。按 $(x_0,R_{11},R_{22},R_{33})$ 与 $(q_0^2,q_1^2,q_2^2,q_3^2)$ 排成 $4\times4$ 块，该块为 $\tfrac12 H_4$，其中
   $$H_4=\begin{bmatrix}1&1&1&1\\1&1&-1&-1\\1&-1&1&-1\\1&-1&-1&1\end{bmatrix}$$
   是 Sylvester Hadamard 矩阵（$H_4H_4^\top=4I$），故 $\tfrac12H_4$ 正交。其余 6 个非对角 $R_{ij}$ 只用到 6 个交叉项 $q_kq_l$；按 $(q_0q_3,q_1q_2),(q_0q_2,q_1q_3),(q_0q_1,q_2q_3)$ 配对后，每对是一个 $\tfrac1{\sqrt2}\begin{bmatrix}1&-1\\1&1\end{bmatrix}$ 型的 $2\times2$ Hadamard 块。故 $U$ 在适当行列置换下是
   $$U\;\sim\;\mathrm{diag}\Big(\tfrac12H_4,\;\tfrac1{\sqrt2}H_2,\;\tfrac1{\sqrt2}H_2,\;\tfrac1{\sqrt2}H_2\Big),$$
   从而正交。$\square$

**与源文档差异**：源文档写“$x_R(q)=[1,\mathrm{vec}(R(q))]^\top$ 对任意 $q\in\mathbb R^4$ 恒有 $x_R=2Uv_2$”。这是**错误的**：$v_2$ 是二次齐次，$2Uv_2$ 的第一分量是 $\|q\|^2$，不是 1；只有把第一分量写成 $\|q\|^2$ 才是恒等式。球面上二者相同。论文/推导中必须写齐次提升。

**边界情形**：$q=0$ 时 $x_R=0$，$R(0)$ 无定义但齐次极限为 0；约束分析只在 $\|q\|=1$ 上使用。$\mathrm{vec}$ 的坐标顺序只置换 $U$ 的行，不改变正交性。

**数值验证**：`verify/verify_theorem0.py` 用 sympy 对符号 $q$ 精确检查 `U_orthogonal_exact=true`、`xR_equals_2Uv2_exact=true`。独立地，父 agent 用 $B$ 的奇异值全为 2、$BB^\top=4I$（误差 $8.9\times10^{-16}$）复核。

---

## 3. 定理 0.2（21 约束张成二次 Veronese 理想）

**定义 0.2**（21 条约束）。在 $x=[x_0,\mathrm{vec}(R)]\in\mathbb R^{10}$ 上定义齐次二次型：
- 6 条列约束：$c_a\cdot c_b=0$（$a<b$），$\|c_a\|^2-x_0^2=0$（$a=1,2,3$）；
- 6 条行约束：$r_a\cdot r_b=0$（$a<b$），$\|r_a\|^2-x_0^2=0$（$a=1,2,3$）；
- 9 条右手叉积约束：$c_a\times c_b-x_0c_c=0$（$(a,b,c)$ 取 $(1,2,3),(2,3,1),(3,1,2)$ 的 3 个分量）。

**定理 0.2 [证明 + 符号验证]**：设 $A_i\in\mathbb S^{10}$ 是上述 21 条约束的矩阵（$x^\top A_ix=0$）。则
$$\dim\operatorname{span}\{A_i\}=20,$$
且在 $x=2Uv_2$ 的坐标变换下，该 20 维子空间**恰好等于** $\nu_2(\mathbb R P^3)\subset\mathbb P^9$ 的二次消没理想 $\mathcal I_2(\nu_2(\mathbb R P^3))$（即 $\mathrm{Sym}^2(\mathbb R^{10})\to\mathbb R[q]_4$ 的核）。

**证明**：

1. **包含**：对任意 $q\in\mathbb S^3$，$R(q)\in SO(3)$，故每条约束在 $x=x_R(q)$ 处为 0；把 $x_R(q)$ 代入是一条四次齐次多项式，它在整个球面上为 0，故恒等于 0（齐次多项式在球面上消失 ⇒ 多项式为 0）。因此每个 $A_i$ 属于核
   $$\ker\phi=\Big\{Q\in\mathrm{Sym}^2(\mathbb R^{10}):Q(x_R(q))\equiv0\Big\}.$$
2. **核的维数**：$\phi:\mathrm{Sym}^2(\mathbb R^{10})\to\mathbb R[q]_4$，$Q\mapsto Q(x_R(q))$。$\dim\mathrm{Sym}^2(\mathbb R^{10})=\binom{11}{2}=55$，$\dim\mathbb R[q]_4=\binom{7}{4}=35$。$\phi$ 是满射：$1\cdot R_{ij}$ 给出全部二次单项式 $q_iq_j$，而两个二次单项式之积张成全部四次单项式；故 $\operatorname{rank}\phi=35$，$\dim\ker\phi=55-35=20$。（sympy 精确秩：`rank_veronese_degree4_map=35`。）
3. **21 条的秩**：对 21 个 $55$ 维系数向量做精确有理消元，得 $\operatorname{rank}=20$（`rank_21_constraints_v_coords_exact=20`）。唯一线性相关是
   $$\sum_{a=1}^3\big(\|c_a\|^2-x_0^2\big)-\sum_{a=1}^3\big(\|r_a\|^2-x_0^2\big)=\operatorname{tr}(R^\top R)-\operatorname{tr}(RR^\top)\equiv0,$$
   即列范数与行范数之和相同；没有其它相关（秩恰为 20，不是 19）。
4. **相等**：20 维子空间包含于 20 维核 ⇒ 相等。$\square$

**推论 0.2.1**：在 $v_2$ 坐标下，$\mathcal I_2$ 由 $4\times4$ 对称矩阵
$$W=\begin{bmatrix}v_1&v_5/\sqrt2&v_6/\sqrt2&v_7/\sqrt2\\ v_5/\sqrt2&v_2&v_8/\sqrt2&v_9/\sqrt2\\ v_6/\sqrt2&v_8/\sqrt2&v_3&v_{10}/\sqrt2\\ v_7/\sqrt2&v_9/\sqrt2&v_{10}/\sqrt2&v_4\end{bmatrix}$$
的 $2\times2$ 子式张成（经典：$\nu_2(\mathbb P^3)$ 的 ideal 由 $2\times2$ 子式生成；见 Harris 1992）。总共有 $\binom42^2=36$ 个 $2\times2$ 子式，但二次部分的维数是 20，不是 36。

**与源文档差异**：源文档只用维数 $55-\binom74=20$ 就断言“严格等于”，这是**必要但不充分**。本文件补了 (a) 包含（每条约束在 $x_R(q)$ 上恒为 0）与 (b) 秩恰为 20，二者合起来才是相等。另一个差异：源文档写“6 列正交 + 6 行正交 + 9 右手叉积”其实应是“3 列范数 + 3 列点积 + 3 行范数 + 3 行点积 + 9 叉积”；只取 6 个非对角点积会得到 15 条、秩 15，凑不出 20。

**边界情形 / 不覆盖**：
- 该定理只对**旋转子问题**的 10×10 约束成立。SDPRLayers 实际 13 维实现的 24 条约束（含 $t$ 与 9 条范数）不在本定理范围内。
- 21 条约束在 $SO(3)$ 上**冗余**（秩 20），因此它们的约束 Jacobian 不可能行满秩；这与 QPEP 只有 1 条约束、处处 LICQ 是两件事。不要把“21 条冗余”与“SDP 的 ACQ/SOSC”混为一谈。

---

## 4. 定理 0.3（Veronese/Shor 松弛等价）与 Choi–Lam 认证间隙

### 4.1 定理 0.3

**定理 0.3 [证明，作用域受限]**：考虑旋转子问题
$$\min_{q\in\mathbb S^3}\;v_2(q)^\top Mv_2(q).$$
设 $\{B_k\}_{k=1}^{20}$ 是 $\mathcal I_2$ 的一组基（$\langle B_k,Y\rangle=0$ 即“$Y$ 与二次理想正交”）。则 SDP
$$\min_{Y\succeq0}\;\langle M,Y\rangle\quad\text{s.t.}\quad \langle B_k,Y\rangle=0\;(k=1,\dots,20),\quad \operatorname{tr}Y=1$$
是上述四次问题在 Veronese 提升 $v=v_2(q)$ 下的 **Shor（二次/Veronese）松弛**。等价地，在 $x$ 坐标下
$$\min_{X\succeq0}\;\langle Q,X\rangle\quad\text{s.t.}\quad X_{00}=1,\;\langle A_i,X\rangle=0\;(i=1,\dots,21),\qquad Q=\tfrac14UMU^\top$$
与上式同值，双射为 $X=4UYU^\top$。

**证明**：

1. **双射**：$Y\succeq0\iff X=4UYU^\top\succeq0$（$U$ 正交，合同变换保 PSD）。
2. **归一化**：令 $e=(1,1,1,1,0,\dots,0)^\top$。由 $x_R=2Uv_2$ 与 $x_R[0]=e^\top v_2$ 得 $U^\top e_0=\tfrac12 e$。于是
   $$X_{00}=e_0^\top Xe_0=4\,(U^\top e_0)^\top Y(U^\top e_0)=\langle ee^\top,Y\rangle.$$
   在 $(\mathcal I_2)^\perp$ 上恒有 $\operatorname{tr}Y=\langle ee^\top,Y\rangle$：两边都是 $Y$ 的线性函数，只需在张成 $(\mathcal I_2)^\perp$ 的秩一生成元 $v_2(q)v_2(q)^\top$ 上验证：
   $$\operatorname{tr}(v_2v_2^\top)=\|v_2\|^2=\|q\|^4,\qquad \langle ee^\top,v_2v_2^\top\rangle=(e^\top v_2)^2=\|q\|^4.$$
   故 $X_{00}=1\iff\operatorname{tr}Y=1$。
3. **目标**：$\langle Q,X\rangle=\tfrac14\langle M,U^\top XU\rangle=\tfrac14\langle M,4Y\rangle=\langle M,Y\rangle$。
4. **松弛解释**：原问题等价于提升 QCQP $\min_v v^\top Mv$ s.t. $v\in\nu_2(\mathbb R P^3),\|v\|=1$。用 $Y=vv^\top$ 提升、把 $v\in\nu_2$ 的二次定义方程与 $\|v\|^2=1$ 线性化，即得上述 SDP；其对偶为
   $$\max_{\mu,\lambda}\;\mu\quad\text{s.t.}\quad M-\mu I-\sum_k\lambda_kB_k\succeq0,$$
   即“$M-\mu I$ 在 Veronese 理想模下是 SOS”的证书。$\square$

**必须写清的 GAP**：

- **GAP-0.3-a（13 维实现）**：定理只对 10×10 旋转子问题成立。SDPRLayers 实际解 13×13（含 $t$）SDP；把 $t$ 从 SDP 层（而不是从非凸代价层）消去会改变松弛值，除非另证“聚合等式”。本文件不声称 13×13 与 10×10 同值。
- **GAP-0.3-b（“二阶 Lasserre”措辞）**：上述 SDP 是 Veronese/Shor 松弛；它与教科书“二阶 Lasserre 矩松弛 + 球面 localizing matrix”是否同值，需要额外证明（球面约束的 localizing matrix 是否冗余）。在证明之前，论文只写 “degree-2 Veronese/Shor relaxation”，不写 “second-order Lasserre”。
- **GAP-0.3-c（Hilbert ⇒ 正测度）**：Hilbert 定理只保证存在非 SOS 的非负四次型；它**不直接**保证对**可实现的** $M$ 存在正测度参数区域使 SDP 不紧。需要：(i) 构造的 $M$ 在真实 PnP/矩阵加权配对的 Gram 锥内（或内点）；(ii) 间隙在 $M$ 的小扰动下保持（开性）；(iii) 从“值间隙”推出 $\operatorname{rank}(X^*)>1$（严格互补）。本文件只给出 (ii) 的数值证据与 (iii) 的实例，见 4.2 与 Q1/Q9。

### 4.2 命题 0.4（Choi–Lam 构造实例的认证间隙）

**命题 0.4 [数值验证 + 对偶证书]**：取
$$C(q)=q_0^2q_1^2+q_1^2q_2^2+q_2^2q_0^2+q_3^4-4q_0q_1q_2q_3,$$
$$M=M_{CL}+2I_{10},\qquad M_{CL}\text{ 为 }C\text{ 在 }v_2\text{ 基下的对称矩阵}.$$
则：

1. $C(q)\ge0$ 对一切 $q$ 成立，且 $C(e_0)=0$；$M_{CL}$ 的特征值为 $\{-\tfrac12,0,\tfrac12,1,\tfrac32\}$ 型（最小 $-\tfrac12$），$M\succ0$（$\lambda_{\min}=1$）。
2. 四次问题真最小值 $f^*=2$（解析：$C\ge0$，$+\|v_2\|^2=+1$）。
3. 10×10 旋转子问题 SDP 值 $f^*_{\text{SDP}}=1.9658119658$；对偶证书 $Z=Q-\mu E_{00}-\sum_i\lambda_iA_i\succeq0$ 的 $\lambda_{\min}(Z)=-1.8\times10^{-11}$，$\mu=1.9658119658$。故**认证间隙** $f^*-f^*_{\text{SDP}}\ge0.034188$。
4. $\operatorname{rank}(X^*)=6$（阈值 $10^{-6}\lambda_{\max}$），$\lambda_1/\lambda_2=1.451$。

**$C\ge0$ 的证明**：令 $a=q_0q_1,b=q_1q_2,c=q_2q_0,d=q_3^2$。则 $C=a^2+b^2+c^2+d^2-4q_0q_1q_2q_3$。由 $abcd=(q_0q_1q_2q_3)^2\ge0$ 与 AM–GM，
$$a^2+b^2+c^2+d^2\ge4|abcd|^{1/2}=4|q_0q_1q_2q_3|\ge4q_0q_1q_2q_3.$$
**非 SOS**：这是 Choi–Lam 的经典非负非 SOS 四次型（Choi & Lam 1977；Choi–Lam–Reznick 1980）。

**与源文档差异**：源文档给 $f^*_{\text{QPEP}}=1.98$、$f^*_{\text{SDP}}=1.96226843$、gap $0.01773$、rank 6、$\lambda_1/\lambda_2=1.561$，并称“作者预实验”。本文件**没有复现**那组数字，因为其对称破缺扰动未给出；本文件用**自己的构造** $M=M_{CL}+2I$ 独立复现了**定性现象**：严格正间隙、rank 6、比值约 1.45、对偶证书。两者不能混为一谈。源文档的数字只能写“作者预实验（待独立复现）”。

**边界情形**：
- 该实例的全局极小**不唯一**（$C$ 在 $e_0,e_1,e_2$ 等多个点取 0），所以它只证明“值间隙 + 非紧”，不证明“唯一全局最优下仍失效”。要唯一化需加对称破缺扰动并重新认证；这是 Q1 的一部分。
- $M$ 是否是**真实 PnP/矩阵加权配对的 Gram 矩阵**未知；这是 Q1/Q9 的 P0 问题。若不可实现，则命题 0.4 只是“构造的旋转子问题反例”，不能直接说“SDPRLayers 在真实 PnP 上失效”。

---

## 5. 定理 1.1（Euler 乘子与 $3\times3$ 黎曼隐微分）

**定理 1.1 [证明 + 数值验证]**：设 $J(q)=v_2(q)^\top Mv_2(q)$，$q^*\in\mathbb S^3$ 是一阶平稳点，$G\succ0$。

1. **闭式乘子**：$\nabla J(q^*)=\lambda^*q^*$，且
   $$\lambda^*=q^{*\top}\nabla J(q^*)=4J(q^*).$$
2. **切基**：$E(q)$ 满足 $q^\top E(q)=0$、$E(q)^\top E(q)=I_3$、$E(-q)=-E(q)$。
3. **黎曼 Hessian**：
   $$H_{\mathbb S^3}(q^*)=E(q^*)^\top\Big(\nabla^2J(q^*)-4J(q^*)\,I_4\Big)E(q^*)\in\mathbb S^3,$$
   其中
   $$\nabla^2J=2J_v^\top MJ_v+2\sum_{a=1}^{10}(Mv_2)_aH_{v,a},$$
   $J_v=\partial v_2/\partial q\in\mathbb R^{10\times4}$，$H_{v,a}=\partial^2v_{2,a}/\partial q\partial q$。
4. **隐微分**：若 $H_{\mathbb S^3}(q^*)\succ0$，则在 $q^*$ 附近存在唯一 $C^1$（实为 $C^\infty$，若 $\theta$ 光滑）局部极小 $q^*(\theta)$，且
   $$\frac{\partial q^*}{\partial\theta}=-E(q^*)H_{\mathbb S^3}(q^*)^{-1}E(q^*)^\top\Big(2J_v(q^*)^\top\frac{\partial M}{\partial\theta}v_2(q^*)\Big).$$
   平移
   $$\frac{\partial t^*}{\partial\theta}=-G^{-1}\Big(\frac{\partial G}{\partial\theta}t^*+\frac{\partial D}{\partial\theta}v_2(q^*)+DJ_v(q^*)\frac{\partial q^*}{\partial\theta}\Big).$$
5. **对跖不变性**：$H_{\mathbb S^3}(-q^*)=H_{\mathbb S^3}(q^*)$，$\partial(-q^*)/\partial\theta=-\partial q^*/\partial\theta$，且
   $$\frac{\partial\,\mathrm{vec}(R(q^*))}{\partial\theta}=J_v(q^*)\frac{\partial q^*}{\partial\theta}$$
   对 $\pm q^*$ 严格相同。

**证明**：

1. **Euler**：$J$ 是 $q$ 的四次齐次多项式（$v_2$ 二次），故 $q^\top\nabla J=4J$。Lagrange 条件 $\nabla J=\lambda q$（约束 $\tfrac12(q^\top q-1)=0$ 的乘子记法），左乘 $q^\top$ 得 $\lambda=q^\top\nabla J=4J$。
2. **切基**：直接计算 $q^\top[q\otimes i,q\otimes j,q\otimes k]=0$；四元数乘法保范，故 $E^\top E=I$；$(-q)\otimes i=-(q\otimes i)$ 给出对跖。注意 $E$ 是“右乘虚单位”的基，$\delta q=E\delta\xi$ 对应的旋转向量是 $2\delta\xi$（见 `derivations.md` §3 的因子 4 讨论）。
3. **Hessian**：对 $F(q,\theta)=E(q)^\top\nabla J(q,\theta)=0$ 沿 $\delta q=E\delta\xi$ 微分：
   $$\delta F=(\delta E)^\top\nabla J+E^\top\nabla^2J\,\delta q.$$
   在平稳点 $\nabla J=\lambda q$，且由 $E^\top q\equiv0$ 微分得 $(\delta E)^\top q=-E^\top\delta q$，故
   $$(\delta E)^\top(\lambda q)=-\lambda E^\top\delta q.$$
   于是
   $$\delta F=E^\top(\nabla^2J-\lambda I)\delta q=E^\top(\nabla^2J-4J I)E\,\delta\xi.$$
   $\nabla^2J$ 的公式由 $J=v^\top Mv$、$\nabla J=2J_v^\top Mv$、对 $q$ 再求导得到（$M$ 与 $q$ 无关）。
4. **隐微分**：对 $F(q^*(\theta),\theta)=0$ 全微分，$H\delta\xi+E^\top\partial_\theta\nabla J=0$，其中 $\partial_\theta\nabla J=2J_v^\top(\partial_\theta M)v_2$（$v_2$ 与 $\theta$ 无关）。解出 $\delta\xi$ 后 $dq^*=E\delta\xi$。平移公式由 $t^*=-G^{-1}Dv_2(q^*)$ 直接求导。
5. **对跖**：$v_2(-q)=v_2(q)$，$J_v(-q)=-J_v(q)$，$\nabla J(-q)=-\nabla J(q)$，$\nabla^2J(-q)=\nabla^2J(q)$，$E(-q)=-E(q)$，故 $H(-q)=H(q)$。隐式解 $\delta\xi(-q)=-\delta\xi(q)$，$dq^*(-q)=E(-q)\delta\xi(-q)=E(q)\delta\xi(q)=dq^*(q)$？等一下：$dq^*$ 是“$q^*$ 的微分”，在 $-q^*$ 处 $d(-q^*)=-dq^*$；而 $R$ 的微分 $J_v(-q)d(-q^*)=(-J_v)(-dq^*)=J_vdq^*$ 不变。$\square$

**与源文档差异**：
- 源文档写“$\delta E^\top(\lambda q)=-\lambda E^\top\delta q$”，符号正确；但要注意这里的 $\delta E$ 是沿 $\delta q$ 的方向导数，且 $\lambda=4J$ 只在平稳点成立。
- 源文档把 $H_{\mathbb S^3}$ 直接叫“QPEP 旋转观测度/信息矩阵”。这**只差一个常数因子**：若用旋转向量 $\omega=2\delta\xi$ 的参数化，$H_\omega=\tfrac14H_{\mathbb S^3}$；QPEP 的协方差还依赖其自身参数化与噪声模型。论文中应写“up to the parametrization factor 4”。
- 源文档说“反向仅需 $3\times3$ 求逆”。对**确定性**层成立；对定理 1.3 的 Laplace KL，还要 $\nabla_\theta\log\det H_k$（含三阶 $q$ 导数）与权重协方差项，见定理 1.3。

**数值验证**（`verify/verify_theorem11.py`，随机 $M$）：
- $E^\top E-I$ 误差 $1.1\times10^{-16}$，$q^\top E$ 误差 $4.4\times10^{-18}$；
- 投影梯度 $2.1\times10^{-10}$；Euler 恒等误差 $5.3\times10^{-15}$；
- 解析 $H$ vs 拉回有限差分 Hessian 相对误差 $9.6\times10^{-8}$；
- 隐式 $dq^*/d\theta$ vs 中心差分相对误差 $1.5\times10^{-7}$；$dt^*/d\theta$ 误差 $4.6\times10^{-8}$；
- 对跖：$H$、$dq^*$、$dR$ 关系误差均为 0（机器精度）。

---

## 6. 定理 1.2（可微性证书）——只成立为“充分条件”

**定理 1.2（充分条件）[证明]**：设 $q^{(1)}$ 是全局极小，且
$$\text{(i) }\Delta J=J(q^{(2)})-J(q^{(1)})>0,\qquad
\text{(ii) }G\succ0,\quad H_{\mathbb S^3}(q^{(1)})\succ0,$$
其中 $q^{(2)}$ 是次优的**不同 SO(3) 类**局部极小。则存在 $\theta$ 的邻域，使全局最优位姿映射 $\theta\mapsto(R^*( \theta),t^*(\theta))$ 在该邻域内 $C^\infty$，且由定理 1.1 的公式给出导数。

**证明**：由 $H\succ0$ 与隐函数定理，$q^{(1)}(\theta)$ 局部唯一且光滑；由 $\Delta J>0$ 与 $J$ 对 $(\theta,q)$ 连续，$\Delta J(\theta)$ 在小邻域内保持 $>0$，故 $q^{(1)}(\theta)$ 仍是全局极小；$t^*$ 由 $G\succ0$ 唯一。$\square$

**命题 1.2′（源文档的“精确充要”不成立）[反例]**：

1. **$\Delta J=0$ 但存在光滑选择**：一维参数化优化 $\min\{J_1(\theta),J_2(\theta)\}$，$J_1\equiv0$，$J_2(\theta)=(\theta-\theta_0)^2$。在 $\theta_0$，$\Delta J=0$（两个不同极小同值）；但 $\theta\mapsto$ 分支 1 是常数、$C^\infty$，且对所有 $\theta$ 都是全局极小（$\theta_0$ 处并列）。所以“$\Delta J>0$”不是“存在光滑全局选择”的必要条件。若把“全局最优映射”定义为**集合值映射**，则在 $\theta_0$ 不是单值，谈不上“可微”；源文档的“精确充要”缺少“单值”这一前提。
2. **$H\succ0$ 不是必要的**：一维 $J(x;\theta)=x^4+\theta^2x^2$。在 $\theta=0$，唯一全局极小 $x^*=0$，$H=0$（二阶退化）；但对所有 $\theta$，$x^*(\theta)\equiv0$ 是 $C^\infty$。故 $H\succ0$ 只是充分条件。
3. **真正的必要条件**：约化平稳方程 $F(q,\theta)=E(q)^\top\nabla J=0$ 存在**可微的单值选择**，且该选择在每点都是全局极小。$H\succ0$ 是该选择存在的标准充分条件，但不是必要；$\Delta J>0$ 等价于“全局极小在 SO(3) 类中唯一”（在有限个非退化临界点下）。

**修正后的写法**（论文采用）：
> **Theorem 1.2 (sufficient certificate).** If the global minimiser is unique up to $q\leftrightarrow-q$, the second-order condition $H_{\mathbb S^3}\succ0$ holds, and $G\succ0$, then the global pose map is locally $C^\infty$ and its derivative is given by (…). Uniqueness of the global minimiser is equivalent to $\Delta J>0$ under A2–A3; $H\succ0$ is sufficient but not necessary for smoothness.

**与 QPEP 根数有关的边界**：
- 源文档写 $K\le27$。仓库 KB 的代码级记录是 LibQPEP `sols` 为 $4\times40$，即**至多 40 个候选复根**（`kb/sdpr-qpep/03.md` §2.3）。27 无出处，且 27 是奇数，与对跖对称 $(q,\lambda)\mapsto(-q,\lambda)$（无不动点，有限解集必为偶数）不相容；27 可能来自某个特定消元模板/坐标卡/含无穷远解或重数的计数。论文统一写“$K$ 个候选根，官方实现 $K\le40$；其中实局部极小 $K_{\min}\le K$”，并把“枚举完整”列为假设 A2。
- $K$ 个根必须按 **SO(3) 类**去重（$q$ 与 $-q$ 同一个旋转），否则 $\Delta J$ 恒为 0。

---

## 7. 定理 1.3（平移精确积分 + 旋转多峰 Laplace）

### 7.1 平移的精确高斯积分

**命题 1.3a [证明]**：设 $J(q,t)=v^\top M_0v+2t^\top Dv+t^\top Gt$，$G\succ0$，$v=v_2(q)$。则
$$\int_{\mathbb R^3}\exp\!\Big(-\frac{J(q,t)}{\tau}\Big)dt=\frac{(\pi\tau)^{3/2}}{\sqrt{\det G}}\exp\!\Big(-\frac{J_{\text{red}}(q)}{\tau}\Big),$$
其中 $J_{\text{red}}=v^\top Mv$，$M=M_0-D^\top G^{-1}D$，$t^*(q)=-G^{-1}Dv$。

**证明**：配方 $J=(t-t^*)^\top G(t-t^*)+v^\top Mv$；用 $\int_{\mathbb R^3}e^{-u^\top Gu/\tau}du=(\pi\tau)^{3/2}/\sqrt{\det G}$。$\square$

### 7.2 旋转方向的 Laplace 近似

**命题 1.3b [推导 + 数值验证；非精确]**：设 $q^{(k)}$（$k=1,\dots,K$）是 $J_{\text{red}}$ 在 $\mathbb S^3/\{\pm1\}$ 上的全部非退化局部极小，$H_k=H_{\mathbb S^3}(q^{(k)})\succ0$。则当 $\tau\to0^+$ 且各极小分离时，
$$\int_{SO(3)}e^{-J_{\text{red}}(q)/\tau}dq
=8\,(2\pi\tau)^{3/2}\sum_{k=1}^K\frac{e^{-J_k/\tau}}{\sqrt{\det H_k}}\big(1+O(\tau)\big),$$
其中 $dq$ 取 $\int_{SO(3)}dq=8\pi^2$ 的 Haar 测度，$H_k$ 用 $\mathbb S^3$ 切基 $E$（$|\delta q|=|\delta\xi|$）表示。

**推导要点**（完整见 `derivations.md` §7–§8）：
- 在 $q^{(k)}$ 附近用 $q=\mathrm{normalize}(q^{(k)}+E_k u)$，$dq=du\,(1+O(\|u\|^2))$，$J=J_k+\tfrac12u^\top H_ku+O(\|u\|^3)$；
- 高斯积分给 $(2\pi\tau)^{3/2}/\sqrt{\det H_k}$；
- $S^3\to SO(3)$ 是二重覆盖；局部 $dR=8dq$（长度缩放 2 ⇒ 体积缩放 $2^3=8$），二重覆盖给因子 $1/2$，故 $\int_{SO(3)}=\tfrac12\int_{S^3}8(\cdot)dq=4\int_{S^3}$。每对 $\pm q^{(k)}$ 两片各贡献 $I$，合起来 $4\cdot2I=8(2\pi\tau)^{3/2}/\sqrt{\det H_k}$。
- 模式权重
  $$w_k=\mathrm{softmax}_k\Big(-\frac{J_k}{\tau}-\frac12\log\det H_k\Big)$$
  与上述常数无关。

**误差阶**：标准 Laplace 展开给相对误差 $O(\tau)$（下一阶是 $O(\tau^2)$ 的奇数阶项一般不为零，故通常就是 $O(\tau)$）。本文件数值验证：$\tau=0.2,0.1,0.05,0.025$ 时相对误差 $8.26\times10^{-2},3.81\times10^{-2},1.86\times10^{-2},9.25\times10^{-3}$，log–log 斜率 $1.05$，确认 $O(\tau)$。

**与源文档差异**：
- 源文档写 $Z_{\text{Lap}}=(\pi\tau)^3/\sqrt{\det G}\sum_k e^{-J_k/\tau}/\sqrt{\det H_k}$。按上述严格常数，平移因子 $(\pi\tau)^{3/2}$、旋转因子 $8(2\pi\tau)^{3/2}$，乘积是 $8\cdot2^{3/2}\pi^3\tau^3$ 量级，不是 $(\pi\tau)^3$。**常数错不影响 $w_k$，但影响 $Z$ 与 KL 损失的绝对数值**；若 $\tau$ 固定，常数与 $\theta$ 无关，对梯度无影响。论文中统一写成 $Z=C(\tau,G)\sum_k e^{-J_k/\tau}/\sqrt{\det H_k}$，不写死常数。
- 源文档称“基于全部代数极小值的黎曼多峰 Laplace 积分”并暗示“精确”。**旋转方向不是精确积分**，只是 $O(\tau)$ 近似；只有平移方向是精确的。

### 7.3 平滑损失、KL 与梯度

**命题 1.3c [推导；含 GAP 标注]**：设 $\bar\ell(\theta)=\sum_kw_k(\theta)\ell_k(\theta)$，$w_k=\mathrm{softmax}_k(-J_k/\tau-\tfrac12\log\det H_k)$。则
$$\nabla_\theta\bar\ell=\sum_kw_k\nabla_\theta\ell_k+\mathrm{Cov}_w\!\Big(\ell_k,\;-\frac1\tau\nabla_\theta J_k-\frac12\nabla_\theta\log\det H_k\Big),$$
其中 $\nabla_\theta J_k=v_k^\top(\partial_\theta M)v_k$（Envelope 定理：$q^{(k)}$ 是 $J$ 的平稳点，$\partial_\theta J$ 在固定 $q$ 处求导即可），
$$\nabla_\theta\log\det H_k=\operatorname{tr}\!\Big(H_k^{-1}\frac{dH_k}{d\theta}\Big),$$
$$\frac{dH_k}{d\theta}=\frac{\partial H}{\partial\theta}\Big|_{q^{(k)}}+\frac{\partial H}{\partial q}\Big|_{q^{(k)}}\frac{\partial q^{(k)}}{\partial\theta}.$$
第二项含 $J$ 的**三阶 $q$ 导数**（因为 $\partial_qH$ 含 $\partial_q\nabla^2J$），这是源文档说“需要三阶导数”的正确含义；第一项只含二阶。

**KL 损失**：
$$\mathcal L_{\text{KL}}(\theta)=\frac1\tau J_\theta(q_{\text{gt}},t_{\text{gt}})+\log Z_{\text{Lap}}(\theta),$$
$$\nabla_\theta\mathcal L_{\text{KL}}=\frac1\tau\nabla_\theta J_{\text{gt}}-\frac12\nabla_\theta\log\det G-\sum_kw_k\Big(\frac1\tau\nabla_\theta J_k+\frac12\nabla_\theta\log\det H_k\Big).$$

**$\tau\to0$ 的正确表述 [重要修正]**：
- 对**后验均值/软 argmin** $\hat q_\tau=\sum_kw_kq^{(k)}$，当 $\Delta J>0$ 且 $\tau\to0$ 时 $w_k\to\delta_{k^*}$，$\hat q_\tau\to q^{(k^*)}$，其梯度趋于定理 1.1 的确定性隐微分。**这是“退化”的正确含义。**
- 对 $\mathcal L_{\text{KL}}$ 本身，$\nabla\mathcal L_{\text{KL}}$ 含 $\tfrac1\tau(\nabla J_{\text{gt}}-\nabla J_{k^*})$，当 $\tau\to0$ 时**按 $1/\tau$ 发散**（除非 $\nabla J_{\text{gt}}=\nabla J_{k^*}$）。要恢复确定性梯度，应使用 $\tau\mathcal L_{\text{KL}}$ 或对损失做 $\tau$ 缩放，而不是直接令 $\tau\to0$。源文档“它严格退化为精确的全局 QPEP 隐微分”只对均值/argmin 成立，对 KL 损失不成立。

**GAP-1.3（一致渐近）**：当 $\Delta J\to0$ 或 $H_k$ 退化（两个模式合并/鞍结分岔）时，上述“分离模式 + 独立高斯”的 Laplace 展开**不一致**：两个 Gaussians 重叠、模式数在分岔处改变、$1/\sqrt{\det H_k}$ 发散。此时需要：
- 一致 Laplace/Airy 展开（把合并方向的三次项保留，得到 Airy 函数）；
- 或对合并后的“槽”做一维数值积分；
- 或改用精确的 Boltzmann 积分（但那就回到采样/数值积分）。
**注意区分**：**精确**的 Boltzmann 平滑后验 $p_\theta\propto e^{-J/\tau}$ 在 $\tau>0$ 时对 $\theta$ 是 $C^\infty$ 的（紧流形 + 控制收敛），在分支交叉处也光滑；**不光滑的是 Laplace 近似**，不是精确后验。源文档把“解析平滑可微”归给 Laplace 近似，是过度声明。正确写法：精确平滑层光滑；Laplace 近似在远离分岔时 $O(\tau)$，在分岔附近需要一致渐近。

### 7.4 测度与归一化

- $S^3$ 的体积 $2\pi^2$；$SO(3)$ 取 $\int dR=8\pi^2$ 的 Haar 测度；局部 $dR=8dq$（每片），二重覆盖给因子 $1/2$，故 $\int_{SO(3)}=4\int_{S^3}$。
- 若用旋转向量 $\omega=2\delta\xi$ 的参数化，$H_\omega=\tfrac14H_{\mathbb S^3}$，$\det H_\omega=\det H_{\mathbb S^3}/64$；Laplace 常数相应变化，但 $w_k$ 与 $\nabla\log\det H_k$ 的 $\theta$-依赖不变（差常数）。论文必须固定一种约定。

---

## 8. 与源文档的差异汇总

| # | 源文档说法 | 本文件修正 | 状态 |
|-|-|-|-|
| 1 | $x_R=[1,\mathrm{vec}R]$ 对任意 $q$ 满足 $x_R=2Uv_2$ | 必须用齐次提升 $[\|q\|^2,\mathrm{vec}R]$；球面上才等于 $[1,\mathrm{vec}R]$ | 已修正 |
| 2 | 21 约束秩 20 ⇒ 等于二次 Veronese 理想 | 维数只是必要；补“包含 + 秩 20”才是证明 | 已证明 |
| 3 | SDPRLayer 等价于 $S^3$ 上四次型的二阶 Lasserre/SOS | 只对 10×10 旋转子问题；叫 Veronese/Shor；“二阶 Lasserre”需另证；13 维实现不覆盖 | 降级 + GAP |
| 4 | 正定物理代价下正测度参数区域必然非紧 | 构造实例有认证间隙；可实现性与正测度未证 | 降级为构造反例 + GAP |
| 5 | 预实验 0：$f^*=1.98$, gap $=0.01773$, rank 6 | 标“作者预实验（待独立复现）”；本文件独立构造得 gap $0.034188$, rank 6, 比值 1.451 | 未复现原数字 |
| 6 | 定理 1.2 精确充要 | 充分条件；$\Delta J=0$ 但光滑选择存在、$H=0$ 但 argmin 光滑的反例 | 已修正 |
| 7 | 旋转 Laplace 是精确积分 | 平移精确；旋转 $O(\tau)$，分岔处需一致渐近 | 已修正 |
| 8 | $\tau\to0$ 严格退化为 QPEP 隐微分 | 对后验均值/argmin 成立；KL 梯度按 $1/\tau$ 发散 | 已修正 |
| 9 | $K\le27$ | 官方 LibQPEP 至多 40 候选复根；27 无出处且与 $\pm$ 对称不相容 | 已修正（Q3） |
| 10 | $3\times3$ 反向即全部 | 确定性层是；Laplace KL 还要三阶导数的 $\nabla\log\det H_k$ 与权重协方差 | 已补充 |
| 11 | “零采样” | 无 Monte Carlo 正确；但旋转积分是 Laplace 近似，不是精确 | 措辞修正 |
| 12 | 21 约束“处处 LICQ” | QPEP 只有 1 条约束才处处 LICQ；21 条在 SO(3) 上冗余、Jacobian 必行相关 | 已修正 |

---

## 9. 边界情形清单

1. **$G$ 奇异/近奇异**：平移不可观测；命题 1.3a 的 Gaussian 积分发散，$t^*$ 不唯一。需加先验或降阶。
2. **$H_k$ 奇异/近奇异**：Laplace 权重 $1/\sqrt{\det H_k}$ 发散；模式在该方向“平坦”，需一致渐近或数值积分。
3. **$\Delta J=0$（对称物体、Necker 翻转）**：硬 argmin 不唯一；精确 Boltzmann 平滑仍光滑；Laplace 近似在合并处失效。
4. **多个全局极小**：$\Delta J=0$；同上。
5. **根枚举失败/漏根/重根**：A2 失效；定理 1.2/1.3 的“全局”与“全部模式”都不成立。需要残差校验、Sturm 序列或同伦延拓作为数值证书。
6. **$M$ 不定**：$J$ 在球面上仍有最小值（紧集），但 $M\not\succeq0$ 时 SDP 可能无界或无强对偶；QPEP 不要求 $M\succeq0$，只要求枚举完整。
7. **尺度**：$M\mapsto sM$ 时 $J\mapsto sJ$；要保持后验不变需 $\tau\mapsto s\tau$，$\log\det H\mapsto\log\det H+3\log s$（模式间常数，$w_k$ 不变）。
8. **$q\leftrightarrow-q$**：所有 $S^3$ 上的量必须按 SO(3) 类去重；$R$ 的导数不变。
9. **参数化因子**：$E$ 基下 $\delta q=E\delta\xi$ 对应旋转向量 $2\delta\xi$；$H_\omega=H/4$。论文固定约定。
10. **13 维实现**：本文件所有 10×10 结论不自动适用于 13×13 SDP；见 Q2。
