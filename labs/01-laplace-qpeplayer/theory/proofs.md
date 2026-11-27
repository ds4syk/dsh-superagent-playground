# 定理、假设与证明（Laplace-QPEPLayer）

> 本文件是本线全部数学主张的“单一事实来源”。每条定理都给出：精确陈述、假设、证明（完整或明确标 GAP）、与源文档的差异、边界情形。  
> 证据分级：**[证明]** 完整数学证明；**[符号验证]** 用 sympy 精确计算（脚本可复现）；**[数值验证]** 浮点数值证据，对应 `independent reproduction (this repo)`（附脚本/结果文件）；**[GAP]** 未证或仅有启发式论证；**[论文声称]** 源文档/原论文/用户补充证明的说法，未在本仓库独立验证，对应 `paper claim`。

---

## 0. 作用域声明（必须先读）

源文档写“单帧 $SO(3)$ SDPRLayer 是 $10\times10$ PSD + 21 个约束”。这与 SDPRLayers 的**实际实现**不是同一个对象：

| 对象 | 变量 | 约束 | 出处 |
|-|-|-|-|
| SDPRLayers 实现（P6 矩阵加权位姿，上游 `sdprlayer@62d8ac6`） | $x=[h;\mathrm{vec}(C);t]\in\mathbb R^{13}$ | 24 条（6 正交 + 9 handedness + 9 范数）；在 $\mathrm{Sym}^{10}$ 上**只张成 17 维**，缺 3 条行正交 | 论文 + 官方仓库；`sdp/results/sdp_check.md` 精确复刻 |
| 补齐的 13×13（满 20 维理想） | $x\in\mathbb R^{13}$ | 24 + 3 条行正交 = 27 条，秩 20，$t$ 块零耦合 | 定理 P1-2（§4.3） |
| 本文件的“旋转子问题” | $x_R=[\|q\|^2;\mathrm{vec}(R(q))]\in\mathbb R^{10}$ | 21 条（6 列 + 6 行 + 9 叉积），秩 20 | 源文档 + 本文件构造 |

因此：

- **定理 0.1–0.3 只对“消去 $t$ 后的旋转子问题 10×10 SDP”成立**，不能直接说成“SDPRLayers 实际求解的 13 维 SDP”。
- 本文件把 10×10 的对象明确标为**构造的旋转子问题 SDP**（constructed rotation-subproblem SDP）。它与 QPEP 的 $J_{\text{red}}(q)$ 精确对应；与 13×13 实现的关系已由**定理 P1-2（§4.3）**结案：**满 20 维下同值同秩；上游 24 条/17 维严格更松**。这既不是未决的 [GAP]，也不是“等价性被推翻”。
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

**作用域 GAP 清单（0.3-a 已结案，0.3-b/c 仍开放）**：

- **GAP-0.3-a（13 维实现）—— 已结案（定理 P1-2，§4.3）**：定理只对 10×10 旋转子问题成立，这一点仍是事实；但 13×13 与 10×10 的关系不再是空白：在满 20 维理想 $\mathcal I_2(SE(3))=\mathcal I_2(SO(3))\oplus\mathbf 0_{3\times3}$ 下二者严格同值同秩（Schur 补双射 $\mathbf X^*_{13}=\mathbf L\mathbf X^*_{10}\mathbf L^\top$）；上游 24 条约束只张成 17 维（$\mathcal V_{17}\subsetneq\mathcal I_2(SO(3))$），是严格更松的松弛，恒有 $f^*_{13,\text{up}}\le f^*_{13,\text{full}}=f^*_{10,\text{full}}\le f^*_{\text{QPEP}}$。故正确写法是“上游 13×13 不等价于 10×10”，既不是“关系未知”，也不是“等价性被推翻”。
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

### 4.3 定理 P1-2（修订版）：13×13 与 10×10 的完整分层等价定理与 6 点物理反例

> **状态（2026-11-27）：P1-Q2/Q13 不是“重新打开”，而是分层结案。** 正确口径是：**满 20 维：等价（Q13 关闭）；上游 17 维：严格更松，且有物理反例**。此前“13×13 等值被数值反例推翻”的表述，是把**上游 24 条约束的 17 维实现**误当成了**满 20 维理想**；二者不是同一个对象，因此并不矛盾。

**来源与证据**：定理陈述与证明来自用户 2026-11-27 补充证明（Feishu `RhzDwI1RKinQsekHDhUcjOVenHL` §一（外部文档，本仓库未提交副本）），标 **[论文声称]（`paper claim`）**。本节数值由本仓库独立测量（`experiments/sdp/results/six_point.md`，SCS `eps=1e-11`），标 **[数值验证]（`independent reproduction (this repo)`）**；§4.2 与本节的 $10^{-2}$ 级扰动证据在 `experiments/sdp/results/sdp_check.md`，同样是本仓库独立复现。

**设定.** 记 $\mathbf x_{13}=[\mathbf x_R^\top,\mathbf t^\top]^\top\in\mathbb R^{13}$，$\mathbf x_R=[h,\operatorname{vec}(\mathbf R)^\top]^\top$，$\mathbf t\in\mathbb R^3$ 无约束；代价矩阵
$$\mathbf Q_{13}=\begin{bmatrix}\mathbf Q_{RR}&\mathbf Q_{Rt}\\ \mathbf Q_{Rt}^\top&\mathbf G\end{bmatrix},\qquad \mathbf G\succ0,\qquad \mathbf Q_{10}\triangleq\mathbf Q_{RR}-\mathbf Q_{Rt}\mathbf G^{-1}\mathbf Q_{Rt}^\top .$$
记：

- $f^*_{10,\text{full}}$：施加满 20 维理想 $\mathcal I_2(SO(3))$ 的 10×10 SDP 最优值；
- $f^*_{13,\text{full}}$：13×13 上施加满 20 维理想 $\mathcal I_2(SE(3))=\mathcal I_2(SO(3))\oplus\mathbf 0_{3\times3}$（上游 24 条 + 补齐 3 条行正交）的最优值；
- $f^*_{13,\text{up}}$：上游 `sdprlayer@62d8ac6` 实际 24 条约束（17 维子空间 $\mathcal V_{17}\subsetneq\mathcal I_2(SO(3))$）的最优值；
- $f^*_{\text{QPEP}}$：非凸问题在 $\mathbb S^3$（$\mathbf t$ 由 $\mathbf G\succ0$ 唯一消去）上的全局最优值。

**定理 P1-2（修订版）[论文声称]：**

1. **（满 20 维理想下的精确等价）** $SE(3)$ 在 $\mathbb S^{13}$ 中的极大二次消没理想精确为 $\mathcal I_2(SE(3))=\mathcal I_2(SO(3))\oplus\mathbf 0_{3\times3}$（维数 20，与 $\mathbf t$ 零耦合）。令 $\mathbf L\triangleq\begin{bmatrix}\mathbf I_{10}\\-\mathbf G^{-1}\mathbf Q_{Rt}^\top\end{bmatrix}\in\mathbb R^{13\times10}$，则块 Schur 补给出双射
   $$\mathbf X^*_{13,\text{full}}=\mathbf L\,\mathbf X^*_{10,\text{full}}\,\mathbf L^\top,\qquad f^*_{13,\text{full}}\equiv f^*_{10,\text{full}},\qquad \operatorname{rank}(\mathbf X^*_{13,\text{full}})\equiv\operatorname{rank}(\mathbf X^*_{10,\text{full}}).$$
2. **（上游 17 维实现严格更松）** 上游 24 条约束在 $\operatorname{Sym}^{10}$ 上只张成 17 维，缺的恰是三条行正交 $\mathbf r_1^\top\mathbf r_2=\mathbf r_1^\top\mathbf r_3=\mathbf r_2^\top\mathbf r_3=0$；故 $\mathcal V_{17}\subsetneq\mathcal I_2(SO(3))$、可行域 $\mathcal F_{13,\text{up}}\supsetneq\mathcal F_{13,\text{full}}$，对**任意**实例恒有
   $$f^*_{13,\text{up}}\;\le\;f^*_{13,\text{full}}=f^*_{10,\text{full}}\;\le\;f^*_{\text{QPEP}} .$$
等号成立当且仅当 17 维松弛的最优解恰好落在三条行正交面上。**推论**：定理 P1-1 的 68 点真实 3D-3D 实例及其正测度开邻域满足 $f^*_{13,\text{up}}\le f^*_{10,\text{full}}<f^*_{\text{QPEP}}$，故上游 13×13 继承 10×10 的正测度非紧性（自动强化 P1-Q1/Q9）。
3. **（6 点物理 3D-3D 分离实例）** 取 $N=6$ 个 3D 点 $\mathbf p_{1,\dots,6}=\pm\mathbf e_1,\pm\mathbf e_2,\pm\mathbf e_3$，目标 $\mathbf y_i=\varepsilon\mathbf p_i$（$\varepsilon=0.05$，真旋转 $\mathbf R^*=\mathbf I_3$ 严格唯一），共享权重
   $$\mathbf W=\begin{bmatrix}1&-0.6&0\\-0.6&1&0\\0&0&1\end{bmatrix}\in\mathbb S^3_{++},\qquad \operatorname{eig}(\mathbf W)=\{0.4,\,1,\,1.6\}.$$
则 $\mathbf D=\mathbf 0$、$\mathbf G=6\mathbf W\succ0$，且
   $$f^*_{13,\text{up}}=3.5943\ldots\;<\;5.415=f^*_{10,\text{full}}=f^*_{13,\text{full}}=f^*_{\text{QPEP}},$$
$\operatorname{rank}(\mathbf X^*_{10})=1$ 而 $\operatorname{rank}(\mathbf X^*_{13,\text{up}})=4$。$\mathbf D=\mathbf 0$ 不是本质的：两个最优值对数据连续，间隙 $1.82$ 在数据的非空开邻域上保持，故小扰动（包括打破 $\mathbf D=\mathbf 0$）后严格不等式仍成立；秩在扰动下不保证不变。

**证明要点.**

(i) 二次型在 $\mathbf x_{13}=[\mathbf x_R;\mathbf t]$ 上对 $\mathbf t$ 配平方：
$$\mathbf x_{13}^\top\mathbf Q_{13}\mathbf x_{13}=\mathbf x_R^\top\mathbf Q_{10}\mathbf x_R+\big(\mathbf t+\mathbf G^{-1}\mathbf Q_{Rt}^\top\mathbf x_R\big)^\top\mathbf G\big(\mathbf t+\mathbf G^{-1}\mathbf Q_{Rt}^\top\mathbf x_R\big),$$
最小值在 $\mathbf t=-\mathbf G^{-1}\mathbf Q_{Rt}^\top\mathbf x_R$ 取到，即 $\mathbf x_{13}=\mathbf L\mathbf x_R$。直接计算 $\mathbf L^\top\mathbf Q_{13}\mathbf L=\mathbf Q_{10}$（因 $\mathbf Q_{13}\mathbf L=[\mathbf Q_{10};\,\mathbf 0]$）。满 20 维理想的约束全部支撑在 $\mathbf x_R$ 块上，故 $\mathbf X_{13}=\mathbf L\mathbf X_{10}\mathbf L^\top$ 与 $\mathbf X_{10}$ 的可行性互相唯一对应，目标值相同；$\mathbf L$ 列满秩（顶部为 $\mathbf I_{10}$），故 $\operatorname{rank}(\mathbf L\mathbf X_{10}\mathbf L^\top)=\operatorname{rank}(\mathbf X_{10})$。

(ii) 24 条→17 维与缺失的 3 条行正交，已由本仓库 sympy 精确复现（`sdp_check.md` §2，`[exact]`：$\operatorname{span}(24)\subseteq\operatorname{span}(21)$，并集秩 20，逐条投影只有三条 `row_orth` 残差非零）。约束空间嵌套 ⇒ 可行域反向嵌套 ⇒ 值不等式。定理 P1-1 的实例与开邻域部分是 `paper claim`；本仓库另有独立证据（构造代价矩阵，非物理 3D-3D）：Choi–Lam 8 个 $10^{-2}$ 对称破缺扰动 **8/8** 严格 $f^*_{13}<f^*_{10}$（差 $2.5\times10^{-3}$–$1.4\times10^{-2}$，$\operatorname{rank}(\mathbf X^*_{13})\in\{4,5\}$ 对 $\operatorname{rank}(\mathbf X^*_{10})=6$；`sdp_check.md` §1b）。

(iii) 由于 $\pm\mathbf p_i$ 成对抵消（$\sum_i\mathbf p_i=\mathbf 0$、$\sum_i\mathbf y_i=\mathbf 0$），交叉项 $\mathbf t^\top\mathbf W\sum_i(\mathbf y_i-\mathbf R\mathbf p_i)=0$，即 $\mathbf D=\mathbf 0$；同时 $\sum_i\mathbf p_i\mathbf p_i^\top=2\mathbf I_3$，且
$$J(\mathbf R,\mathbf t)=\sum_i\|\mathbf y_i-\mathbf R\mathbf p_i-\mathbf t\|_{\mathbf W}^2=2\operatorname{tr}(\mathbf W\mathbf R\mathbf R^\top)-4\varepsilon\operatorname{tr}(\mathbf W\mathbf R)+2\varepsilon^2\operatorname{tr}(\mathbf W)+6\,\mathbf t^\top\mathbf W\mathbf t .$$
关键项
$$2\operatorname{tr}(\mathbf W\mathbf R\mathbf R^\top)=2\big(\|\mathbf r_1\|^2+\|\mathbf r_2\|^2+\|\mathbf r_3\|^2-1.2\,\mathbf r_1^\top\mathbf r_2\big).$$
在满 20 维可行面上，行范数约束与 $X_{00}=1$ 给出 $\|\mathbf r_i\|^2=1$，且 $\mathbf r_1^\top\mathbf r_2=0$，故该项恒等于 $6$；于是目标只剩线性项 $-4\varepsilon\operatorname{tr}(\mathbf W\mathbf R)+2\varepsilon^2\operatorname{tr}\mathbf W$，它在 rank-1 可行点 $\mathbf R=\mathbf I_3$ 取最大 $\operatorname{tr}(\mathbf W\mathbf R)=\operatorname{tr}\mathbf W=3$，给出 $f^*=6(1-\varepsilon)^2=5.415$。本仓库测得该 SDP 的最优解恰为 rank 1（$\operatorname{rank}(\mathbf X^*_{10})=1$），故线性松弛在此实例上紧。上游 17 维没有 $\mathbf r_1^\top\mathbf r_2=0$，松弛最优解可取 $\langle\mathbf r_1^\top\mathbf r_2\rangle>0$ 把 $2(3-1.2\,\mathbf r_1^\top\mathbf r_2)$ 压到 $6$ 以下，得 $f^*_{13,\text{up}}=3.5943\ldots$、$\operatorname{rank}=4$。$\square$

**数值表（本仓库独立测量，SCS `eps=1e-11`；`six_point.md`）：**

| 量 | 本仓库实测 | 用户证明 |
|-|-|-|
| $f^*_{10}$（21 条，满 20D） | 5.41500000 | 5.41500000 |
| $f^*_{13,\text{full}}$（24+3 条，满 20D） | 5.41500000 | 5.41500000 |
| $f^*_{13,\text{up}}$（24 条，17D） | 3.59431373 | 3.59430886 |
| $f^*_{10}-f^*_{13,\text{up}}$ | 1.82068627 | 1.82069114 |
| $\operatorname{rank}(\mathbf X^*_{10})$ / $\operatorname{rank}(\mathbf X^*_{13,\text{up}})$ | 1 / 4 | 1 / 4 |
| $f^*_{\text{QPEP}}$（解析：$\max_{\mathbf R\in SO(3)}\operatorname{tr}(\mathbf W\mathbf R)=3$ 在 $\mathbf R=\mathbf I$） | 5.415 | 5.415 |

实测与证明引用的 $3.59430886$ 差 $4.9\times10^{-6}$。该残差**不是**求解器容差：SCS（$\varepsilon=10^{-11}$）与 CLARABEL 内点法分别给出 $3.59431373$ 与 $3.59431371$，二者一致到 $<10^{-9}$；因此最后 5 位未能由本仓库的同一 SDP 构造复现，但严格分离（$1.82$ 间隙、rank $1$ 对 $>1$）完全确认。$\operatorname{rank}(X^\star_{13,\rm up})$ 在 $10^{-5}$ 阈值下为 4、$10^{-6}$ 下为 6（4 个 $O(1)$ 特征值后接 $\sim10^{-5}$ 小特征值），定理只需要 rank $>1$。$f^*_{13,\text{full}}=f^*_{10}=5.415$ 且 rank 均为 1，也在数值上支持 (i)；但 (i) 的成立性依据是上面的 Schur 恒等式，不是这个实例。

**边界与不声称**：

- 需要 $\mathbf G\succ0$（假设 A1）。$\mathbf G$ 奇异时 $\mathbf t$ 不可观测，Schur 双射需换成广义 Schur 补/约束消元，本节不声称。
- 满 20 维的等价**不**说明上游 24 条实现与 10×10 同值；相反 (ii) 说明它在正测度物理域上严格更松。
- 6 点实例是真矩阵加权 3D-3D 问题（$N=6$，$\mathbf W\succ0$），但 $\mathbf D=\mathbf 0$ 是方便选择的对称点；本节只声称该点及其开邻域上的严格分离，不声称“所有物理实例都非紧”（那是定理 P1-1 的正测度结论）。
- 扰动下只声称值与严格不等式保持，不声称秩保持。

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

**状态（2026-11-27 修订）**：前两条已被**定理 P1-Q5c（§7.5）**在“40 根复共轭对 + 半线 Airy/Scorer”框架内统一：它给出跨 Fold 严格 $C^\infty$ 的零跳变极限，并在 $\zeta\gg1$ 时精确退化为 3D Laplace；6D 实现待单独一轮。同时**命题 P1-Q5b（§7.5）**证明：任何纯实根代理（包括 $\tau^{1/3}$ 正则化）都不可能消除该值跳变，故“只用实根 + 正则化”的路线在此处有硬上限。

**注意区分**：**精确**的 Boltzmann 平滑后验 $p_\theta\propto e^{-J/\tau}$ 在 $\tau>0$ 时对 $\theta$ 是 $C^\infty$ 的（紧流形 + 控制收敛），在分支交叉处也光滑；**不光滑的是 Laplace 近似**，不是精确后验。源文档把“解析平滑可微”归给 Laplace 近似，是过度声明。正确写法：精确平滑层光滑；Laplace 近似在远离分岔时 $O(\tau)$，在分岔附近需要一致渐近。

### 7.4 测度与归一化

- $S^3$ 的体积 $2\pi^2$；$SO(3)$ 取 $\int dR=8\pi^2$ 的 Haar 测度；局部 $dR=8dq$（每片），二重覆盖给因子 $1/2$，故 $\int_{SO(3)}=4\int_{S^3}$。
- 若用旋转向量 $\omega=2\delta\xi$ 的参数化，$H_\omega=\tfrac14H_{\mathbb S^3}$，$\det H_\omega=\det H_{\mathbb S^3}/64$；Laplace 常数相应变化，但 $w_k$ 与 $\nabla\log\det H_k$ 的 $\theta$-依赖不变（差常数）。论文必须固定一种约定。

### 7.5 Fold 处的必然跳变与 Scorer 一致延拓（P1-Q5，修订）

> 2026-11-27 补充证明（Feishu `RhzDwI1RKinQsekHDhUcjOVenHL` §二（外部文档，本仓库未提交副本））把 Q5 分成两层：**命题 P1-Q5b**（任何纯实根 Laplace 代理，包括 $\tau^{1/3}$ 正则化，在 Fold 处必有非零值跳变）与**定理 P1-Q5c**（复共轭根对 + 半线 Airy/Scorer 一致延拓，给出跨 Fold 严格 $C^\infty$ 零跳变）。两者均标 **[论文声称]（`paper claim`）**；6D 实现见 §7.5 末段（已完成）。

**7.5.1 命题 P1-Q5b（纯实根截断的必然值跳变）**

**设定.** 单参数族 $J_\theta(\mathbf q)$ 在 $\theta=\theta_f$ 处经历标准余秩-1 Fold（$A_2$）：$\theta>\theta_f$ 时存在次级实极小 $\mathbf q_+(\theta)$ 与邻近一阶鞍点 $\mathbf q_-(\theta)$，$\theta\to\theta_f^+$ 时合并到拐点 $\mathbf q_0$（$\lambda_1(\mathbf q_+)\to0^+$，横截二阶阵 $\mathbf H_\perp(\mathbf q_0)\succ0$），且存在更深的分离极小使 $J_0=J_{\theta_f}(\mathbf q_0)>J_{\text{glob}}$；$\theta<\theta_f$ 时该实极小消失（实极小数 $K\to K-1$）。

**命题 P1-Q5b [论文声称].** 对任意只在实局部极小集合上求和、且对合并模式赋予正正则化下界权重
$$\tilde w_+^{\text{reg}}(\theta_f^+)\triangleq\frac{(2\pi\tau)^{3/2}}{\sqrt{\det(\mathbf H_++\eta(\tau)\mathbf I_3)}}\,e^{-J_0/\tau}>0$$
的实根代理 $\bar\ell_{\text{reg}}(\theta)$，只要 $\ell(\mathbf q_0)\neq\bar\ell_{\text{rest}}(\theta_f)$，则在 $\theta_f$ 处必有非零值跳变
$$\lim_{\theta\to\theta_f^+}\bar\ell_{\text{reg}}(\theta)-\lim_{\theta\to\theta_f^-}\bar\ell_{\text{reg}}(\theta)=\frac{\tilde w_+^{\text{reg}}(\theta_f^+)}{Z_{\text{rest}}(\theta_f)+\tilde w_+^{\text{reg}}(\theta_f^+)}\big(\ell(\mathbf q_0)-\bar\ell_{\text{rest}}(\theta_f)\big)\neq0 .$$
特别地，取标度匹配地板 $\eta(\tau)=c_0\tau^{1/3}$ 时，跳变幅度为 $\Theta\big(\tau^{1/3}e^{-(J_0-J_{\text{glob}})/\tau}\big)$：仅当 $J_0-J_{\text{glob}}\gg\tau$ 时指数小，而 $J_0-J_{\text{glob}}=O(\tau)$ 时保持 $O(1)$ 有限跳变（该 $\tau$ 标度按补充证明的权重约定，本仓库未独立拟合其指数。**内部一致性待核**：复核者指出把文中 $\tilde w_+^{\rm reg}\propto\tau^{3/2}\det(H+\eta I)^{-1/2}$ 与 $\eta=c_0\tau^{1/3}$ 代入上面的跳变分数，按其权重约定可能得到 $\tau^{4/3}$ 而非 $\tau^{1/3}$；本仓库未独立拟合，故此处保留证明原文的 $\tau^{1/3}$ 并标注该点需证明作者复核，不影响“跳变非零”的定性结论）。

**与实测的关系 [数值验证].** 本仓库的 6D Fold 测量（`experiments/results/fold_6d.md`，`independent reproduction (this repo)`）与该预测完全一致：标准 Laplace 的跨 Fold 位姿跳变为 $3.089\times10^{-1}$，用 $\det(\mathbf H+\tau^{1/3}\mathbf I)$ 正则化后只降到 $1.070\times10^{-1}$；$K$ 仍 $4\to3$，`jump/step` 仍约 $5\times10^3$（真阶跃的特征），说明正则化没有、也不可能消除值跳变。它改善的是导数奇点：近 Fold 梯度误差 $2.90\times10^{-1}\to1.19\times10^{-2}$（24×），$|\mathrm{dpose}|$ 从 $3529$ 降到 $443$。这正是 P1-Q5b 的预测：$\tau^{1/3}$ 正则化把值跳变从 $0.309$ 降到 $0.107$，但跳变仍非零。

**7.5.2 定理 P1-Q5c（复共轭根对 + Scorer 半线 Airy 延拓）**

**规范形.** 由 Gromoll–Meyer 分裂引理，在 $\theta_f$ 邻域存在光滑局域微分同胚 $(u,\mathbf v)\in\mathbb R\times\mathbb R^2$，把 $J_\theta$ 化为标准 Fold 范式
$$J_\theta(u,\mathbf v)=J_0(\theta)+\tfrac12\mathbf v^\top\mathbf H_\perp(\theta)\mathbf v+\frac{c_3(\theta)}{6}u^3-\alpha(\theta)u,\qquad c_3(\theta)>0,\quad\alpha(\theta_f)=0,\quad\alpha'(\theta_f)>0 .$$
40 根代数求解器（Action Matrix 特征分解）在 Fold 两侧都精确追踪合并根对 $(\mathbf q_+(\theta),\mathbf q_-(\theta))\in(\mathbb{CP}^3)^2$：

- $\alpha>0$：实极小-鞍点对 $u_\pm=\pm\sqrt{2\alpha/c_3}$，$\lambda_1(\mathbf q_\pm)=\pm\sqrt{2c_3\alpha}$，$J(\mathbf q_\pm)=J_0\mp\frac{2\sqrt2}{3}\alpha^{3/2}c_3^{-1/2}$；
- $\alpha<0$：复共轭根对 $u_\pm=\pm i\sqrt{2|\alpha|/c_3}$，复黎曼 Hessian 特征值 $\lambda_1^{\mathbb C}(\mathbf q_\pm)=\pm i\sqrt{2c_3|\alpha|}$，$J(\mathbf q_\pm)=J_0\mp i\frac{2\sqrt2}{3}|\alpha|^{3/2}c_3^{-1/2}$。

**四个代数不变量.** 仅由该根对决定、在 $\alpha>0$ 与 $\alpha<0$ 两侧形式完全一致：
$$J_0(\theta)\triangleq\frac{J(\mathbf q_+)+J(\mathbf q_-)}{2}\in\mathbb R,$$
$$\zeta(\theta,\tau)\triangleq\operatorname{sgn}(\alpha)\left(\frac{3\,|J(\mathbf q_-)-J(\mathbf q_+)|}{4\tau}\right)^{2/3}=\alpha(\theta)\left(\frac{2}{c_3(\theta)}\right)^{1/3}\tau^{-2/3}\in\mathbb R,$$
$$\left(\frac{2}{c_3(\theta)}\right)^{1/3}=\left(\frac{6\,|J(\mathbf q_-)-J(\mathbf q_+)|}{|\lambda_1(\mathbf q_+)|^3}\right)^{1/6}>0,\qquad \det\mathbf H_\perp(\theta)=\operatorname{Re}\big(\lambda_2(\mathbf q_+)\lambda_3(\mathbf q_+)\big)>0 .$$

**Scorer 函数与统一权重.**
$$\mathcal S(\zeta)\triangleq\int_0^{+\infty}\exp\!\left(-\frac{s^3}{3}+\zeta s\right)ds,\qquad \tilde w_{\text{fold}}(\theta,\tau)\triangleq\frac{2\pi\tau}{\sqrt{\det\mathbf H_\perp(\theta)}}\cdot\tau^{1/3}\left(\frac{2}{c_3(\theta)}\right)^{1/3}e^{-J_0(\theta)/\tau}\,\mathcal S\big(\zeta(\theta,\tau)\big),$$
配跨 Fold 实光滑代表位姿 $\mathbf q_{\text{rep}}(\theta)\triangleq\frac{\mathbf q_++\mathbf q_-}{\|\mathbf q_++\mathbf q_-\|}$（$\alpha<0$ 时即 $\operatorname{Re}(\mathbf q_+)/\|\operatorname{Re}(\mathbf q_+)\|$；$\alpha>0$ 远离 Fold 时用与 $\zeta$ 平滑混合的 $\mathbf q_+$）。$\mathcal S$ 对一切 $\zeta\in\mathbb R$ 绝对收敛且为整函数，$\mathcal S(0)=\Gamma(1/3)/3^{2/3}\approx1.2879$，$\mathcal S'(\zeta)=\int_0^\infty s\,e^{-s^3/3+\zeta s}ds>0$。

**定理 P1-Q5c [论文声称]：**

1. **（跨 Fold 严格 $C^\infty$ 值与梯度连续 — 零跳变）** $(J_0,\alpha,c_3,\mathbf H_\perp)$ 都是特征多项式的对称多项式函数，跨 $\alpha=0$ 全部保持 $C^\infty$（$\operatorname{sgn}(\alpha)|\alpha|^{(3/2)\cdot(2/3)}=\alpha$ 恰好抵消 $|\alpha|^{3/2}$ 在 $0$ 处的非光滑），且 $\mathcal S$ 为整函数，故 $\tilde w_{\text{fold}}(\theta,\tau)$ 与期望损失 $\bar\ell_{\text{Scorer}}(\theta)$ 在 $\theta=\theta_f$ 处**严格 $C^\infty$ 连续，值跳变恒等于 $0$，导数有限且光滑**。
2. **（$\zeta\gg1$ 精确退化为 3D 标准 Laplace）** 当 $\alpha>0$ 且 $\Delta J_{\text{bar}}=|J(\mathbf q_-)-J(\mathbf q_+)|\gg\tau$（$\zeta\to+\infty$）时，被积函数在唯一内驻点 $s_+=\sqrt\zeta>0$ 处满足二阶展开 $-\frac{s^3}{3}+\zeta s=\frac23\zeta^{3/2}-\sqrt\zeta\,(s-\sqrt\zeta)^2-\frac13(s-\sqrt\zeta)^3$，由 Laplace 渐近法
   $$\mathcal S(\zeta)=\sqrt{\frac{\pi}{\sqrt\zeta}}\,e^{\frac23\zeta^{3/2}}\big(1+O(\zeta^{-3/2})\big).$$
代入 $\frac23\zeta^{3/2}=\frac{J_0-J(\mathbf q_+)}{\tau}$ 与 $\tau^{1/3}(2/c_3)^{1/3}\sqrt\pi/\zeta^{1/4}=\sqrt{2\pi\tau/\lambda_1}$，得
   $$\tilde w_{\text{fold}}(\theta,\tau)=\frac{(2\pi\tau)^{3/2}}{\sqrt{\lambda_1\det\mathbf H_\perp}}\,e^{-J(\mathbf q_+)/\tau}\big(1+O(\tau/\Delta J_{\text{bar}})\big)\equiv\tilde w_{\text{Lap}}(\mathbf q_+)\big(1+O(\tau/\Delta J_{\text{bar}})\big).$$
3. **（$\zeta\ll-1$ 平滑衰减消失）** 当 $\alpha<0$ 离开 Fold（$\zeta\to-\infty$）时，分部积分得 $\mathcal S(\zeta)=\frac1{|\zeta|}\big(1-2|\zeta|^{-3}+\dots\big)$，故 $\tilde w_{\text{fold}}=O(\tau/|\alpha|)\,e^{-J_0/\tau}$ 光滑衰减为零。$\blacksquare$

**实现状态（2026-11-27 晚更新：6D 已实现）.** 定理本身仍是 `paper claim`；**6D 复根追踪与 Scorer 延拓已在本仓库实现并测量**（`experiments/theory/run_scorer_fold.py`，结果 `experiments/theory/results/scorer_fold.{json,md,png}`，`independent reproduction (this repo)`）。**(a) 1D 三路验证**：模型 $J_\alpha=x^3/3-\alpha x$（$c_3=2$）；跨 $\alpha=+10^{-5}\to-10^{-5}$ 的 fold-mode 后验分数：标准实根 Laplace $0.86703\to0$、$\tau^{1/3}$ 正则化 $0.49595\to0$、Scorer 延拓 $0.33838\to0.33834$，与证明引用的 $0.867/0.497/0.339113/0.339069$ 差 $\le1.6\times10^{-4}$。**(b) 6D 应用**：fold 位于 $\theta_f=1.3185485\times10^{-3}$，$c_3=2.55981$，$\det H_\perp\approx7.49$；实侧极小/鞍点对与复侧复共轭根对均被 Newton 追踪成功（$\alpha'$ 两侧独立测量相对差 $\sim2\times10^{-3}$；$J_0$ 相对残差 $1.7\times10^{-12}$，$c_3$ $7\times10^{-3}$），四个不变量与 $\mathcal S(\zeta)$ 跨 $\theta_f$ 连续。**(c) 位姿跳变**：$\theta_f\pm10^{-4}$、步长 $2\times10^{-5}$ 的网格上最大相邻位姿跳变：标准 $3.041\times10^{-1}$（jump/step $1.52\times10^4$）、$\det(H+\tau^{1/3}I)$ 正则化 $1.068\times10^{-1}$（$5.34\times10^3$）、**Scorer 延拓 $1.206\times10^{-2}$（$6.03\times10^2$，与平滑映射的 $|\mathrm d\,\mathrm{pose}|$ 同量级）**，K 恒为 4；即真阶跃变为导数有限的光滑过渡。**(d) 限制**：该 6D fold 很浅（$\Delta J\sim10^{-6}\ll\tau=0.01$、$\zeta\sim10^{-3}$），fold 处 Scorer 权重与标准 Laplace 权重相差很大，定理只承诺 $\zeta\gg1$（$\Delta J\gg\tau$）时退化为 3D Laplace；$w_{\rm fold}$ 在给定四个不变量时精确，但四者与规范形的对应是渐近的（四阶项 $\varphi''''\approx8.3$）。工程主表仍未启用复根追踪；$\tau^{1/3}$ 部分修复的纯实根对照数字（`fold_6d.md`：$|\mathrm{dpose}|$ 3529→443、值跳变 0.309→0.107）保留。

**论文口径.** 若工程主表不启用复根追踪，援引命题 P1-Q5b 明确写“纯实根 Laplace（含 $\tau^{1/3}$ 正则化）在 Fold 处必有 $\Theta(\tau^{1/3}e^{-(J_0-J_{\text{glob}})/\tau})$ 值跳变，故定理主范围排除 Fold 邻域”；同时把定理 P1-Q5c 作为 Theory/Discussion 里的严格延拓定理，说明 Fold 连续性在代数 40 根框架内**理论完全可解**，实现是工程问题。

---

## 8. 与源文档的差异汇总

| # | 源文档说法 | 本文件修正 | 状态 |
|-|-|-|-|
| 1 | $x_R=[1,\mathrm{vec}R]$ 对任意 $q$ 满足 $x_R=2Uv_2$ | 必须用齐次提升 $[\|q\|^2,\mathrm{vec}R]$；球面上才等于 $[1,\mathrm{vec}R]$ | 已修正 |
| 2 | 21 约束秩 20 ⇒ 等于二次 Veronese 理想 | 维数只是必要；补“包含 + 秩 20”才是证明 | 已证明 |
| 3 | SDPRLayer 等价于 $S^3$ 上四次型的二阶 Lasserre/SOS | 10×10 叫 Veronese/Shor（“二阶 Lasserre”仍需另证）；满 20D 的 13×13 与 10×10 同值同秩（定理 P1-2(i)），上游 24 条/17 维严格更松（P1-2(ii)） | 分层结案（Q2/Q13） |
| 4 | 正定物理代价下正测度参数区域必然非紧 | 构造实例有认证间隙；可实现性与正测度未证 | 降级为构造反例 + GAP |
| 5 | 预实验 0：$f^*=1.98$, gap $=0.01773$, rank 6 | 标“作者预实验（待独立复现）”；本文件独立构造得 gap $0.034188$, rank 6, 比值 1.451 | 未复现原数字 |
| 6 | 定理 1.2 精确充要 | 充分条件；$\Delta J=0$ 但光滑选择存在、$H=0$ 但 argmin 光滑的反例 | 已修正 |
| 7 | 旋转 Laplace 是精确积分 | 平移精确；旋转 $O(\tau)$，分岔处需一致渐近 | 已修正 |
| 8 | $\tau\to0$ 严格退化为 QPEP 隐微分 | 对后验均值/argmin 成立；KL 梯度按 $1/\tau$ 发散 | 已修正 |
| 9 | $K\le27$ | 官方 LibQPEP 至多 40 候选复根；27 无出处且与 $\pm$ 对称不相容 | 已修正（Q3） |
| 10 | $3\times3$ 反向即全部 | 确定性层是；Laplace KL 还要三阶导数的 $\nabla\log\det H_k$ 与权重协方差 | 已补充 |
| 11 | “零采样” | 无 Monte Carlo 正确；但旋转积分是 Laplace 近似，不是精确 | 措辞修正 |
| 12 | 21 约束“处处 LICQ” | QPEP 只有 1 条约束才处处 LICQ；21 条在 SO(3) 上冗余、Jacobian 必行相关 | 已修正 |
| 13 | （2026-11-27 补充证明旧稿）13×13 与 10×10 同值同秩，无需限定 | 只在满 20 维理想下成立；上游 `sdprlayer@62d8ac6` 的 24 条只张成 17 维，严格更松；6 点物理反例 $f^*_{13,\text{up}}=3.5943<5.415=f^*_{\text{QPEP}}$（rank 4 vs 1） | 已修正（定理 P1-2） |
| 14 | 纯实根 Laplace 在 Fold 处可用 $\det(H+\tau^{1/3}I)$ 正则化消除值跳变 | 不可能：任意实根代理必有 $\Theta(\tau^{1/3}e^{-(J_0-J_{\text{glob}})/\tau})$ 值跳变；正则化只把 $0.309$ 降到 $0.107$ | 命题 P1-Q5b |
| 15 | Fold 连续性只能靠数值/一致渐近，无严格延拓 | 40 根复共轭对 + Scorer 半线 Airy 延拓给出跨 Fold 严格 $C^\infty$ 零跳变（`paper claim`）；6D 实现待做 | 定理 P1-Q5c |

---

## 9. 边界情形清单

1. **$G$ 奇异/近奇异**：平移不可观测；命题 1.3a 的 Gaussian 积分发散，$t^*$ 不唯一。需加先验或降阶。
2. **$H_k$ 奇异/近奇异**：Laplace 权重 $1/\sqrt{\det H_k}$ 发散；模式在该方向“平坦”，需一致渐近或数值积分。$\tau^{1/3}$ 正则化只是缓解导数奇点；跨 Fold 的严格处理见 §7.5（P1-Q5b/c）。
3. **$\Delta J=0$（对称物体、Necker 翻转）**：硬 argmin 不唯一；精确 Boltzmann 平滑仍光滑；Laplace 近似在合并处失效。失效的必然性与复根 Scorer 延拓见 §7.5。
4. **多个全局极小**：$\Delta J=0$；同上。
5. **根枚举失败/漏根/重根**：A2 失效；定理 1.2/1.3 的“全局”与“全部模式”都不成立。需要残差校验、Sturm 序列或同伦延拓作为数值证书。
6. **$M$ 不定**：$J$ 在球面上仍有最小值（紧集），但 $M\not\succeq0$ 时 SDP 可能无界或无强对偶；QPEP 不要求 $M\succeq0$，只要求枚举完整。
7. **尺度**：$M\mapsto sM$ 时 $J\mapsto sJ$；要保持后验不变需 $\tau\mapsto s\tau$，$\log\det H\mapsto\log\det H+3\log s$（模式间常数，$w_k$ 不变）。
8. **$q\leftrightarrow-q$**：所有 $S^3$ 上的量必须按 SO(3) 类去重；$R$ 的导数不变。
9. **参数化因子**：$E$ 基下 $\delta q=E\delta\xi$ 对应旋转向量 $2\delta\xi$；$H_\omega=H/4$。论文固定约定。
10. **13 维实现**：本文件所有 10×10 结论不自动适用于上游 13×13 SDP；满 20 维补齐版与 10×10 同值同秩，上游 17 维版严格更松（定理 P1-2，§4.3）。

---

## 10. P1 缺口状态（Q2/Q13 与 Q5，2026-11-27 修订）

| 缺口 | 旧口径 | 新状态 | 依据 |
|-|-|-|-|
| **P1-Q2/Q13** | “重新打开”：13×13 等值被 SDP 数值反例推翻 | **分层结案**：满 20D 下 13×13 $\equiv$ 10×10（同值同秩，Q13 关闭）；上游 17D（24 条，缺 3 条行正交）严格更松，$f^*_{13,up}\le f^*_{10,full}\le f^*_{QPEP}$；6 点物理反例 $f^*_{13,up}=3.5943<5.415=f^*_{QPEP}$（rank 4 vs 1），且在数据的非空开邻域上保持 | 定理 P1-2（§4.3，`paper claim`）；`sdp/results/six_point.md`（`independent reproduction (this repo)`）；`sdp/results/sdp_check.md` 的 8/8 扰动（本仓库独立复现） |
| **P1-Q5（Fold 连续性）** | 分岔处只能靠数值/一致渐近，是 P0 缺口 | **分层结案（理论）**：任意纯实根代理（含 $\tau^{1/3}$ 正则化）必有 $\Theta(\tau^{1/3}e^{-(J_0-J_{\text{glob}})/\tau})$ 值跳变（P1-Q5b）；复共轭根 Scorer 延拓给出跨 Fold 严格 $C^\infty$ 零跳变（P1-Q5c，`paper claim`）。工程侧仍是 $\tau^{1/3}$ 部分修复，6D Scorer 实现在单独一轮 | 命题 P1-Q5b、定理 P1-Q5c（§7.5）；`experiments/results/fold_6d.md`（`independent reproduction (this repo)`） |

注：

- Q2/Q13 的旧“重新打开”口径对应的证据（17D vs 20D、8/8 严格 $f^*_{13}<f^*_{10}$、构造目标差 2.0）**仍然正确**；它们否定的是“上游 24 条实现 = 10×10”，不是“满 20 维 13×13 = 10×10”。新定理把这两件事分开，因此旧证据不需要撤回。
- Q5 的“梯度余项引理”（Laplace 梯度相对误差 $O(\tau)$ 的严格余项界）是另一个工程证明项，**不因 P1-Q5b/c 自动关闭**；本节只结案 Fold 连续性/一致渐近这一层。
