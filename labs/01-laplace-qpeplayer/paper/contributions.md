# 贡献清单、逐维对比与审稿人质疑

> 配套 `paper/draft.md` 与 `theory/proofs.md`。本文件是“主张边界”的清单：哪些是已证明的、哪些是数值验证的、哪些是 GAP、哪些**不**声称。

## 0. 一句话定位

`Laplace-QPEPLayer` = QPEP 的**零升维四元数全局枚举** + $3\times3$ **黎曼隐微分** + **平移精确边际化 / 旋转多峰 Laplace** 的可微位姿层；目标不是“再做一个 SDPR”，而是回答：**当全局解由代数枚举给出、而不是由凸松弛给出时，可微性与多模态平滑能做到什么程度？**

## 1. 贡献清单

| # | 贡献 | 证据等级 | 对应文件 |
|-|-|-|-|
| C1 | **正交同构**：$x_R=2Uv_2$，$U\in O(10)$，显式 $H_4$ + 三个 $H_2$ 块结构；齐次提升的第一分量必须是 $\|q\|^2$，不能是字面 1 | **已证明**（符号恒等 + 秩） | proofs 定理 0.1；derivations §1；verify_theorem0 |
| C2 | **21 约束 = 二次 Veronese 理想**：6 列 + 6 行 + 9 叉积，秩 20；包含 + 秩 20 ⇒ 等于 $\mathcal I_2(\nu_2(\mathbb{RP}^3))$ | **已证明**（精确有理秩 + 恒等消去） | proofs 定理 0.2；derivations §2；verify_theorem0 |
| C3 | **Veronese/Shor 松弛等价**：10×10 旋转子问题 SDP 与 $\min\langle M,Y\rangle$ s.t. $Y\succeq0,\langle B_k,Y\rangle=0,\operatorname{tr}Y=1$ 同值；对偶是 SOS-modulo-ideal 证书 | **已证明（作用域受限）** | proofs 定理 0.3；derivations §2 |
| C4 | **Choi–Lam 认证间隙实例**：$M=M_{CL}+2I$，真值 2，SDP 1.9658119658，对偶 $Z\succeq0$（$\lambda_{\min}=-1.8\times10^{-11}$），认证 gap $0.034188$，rank 6，$\lambda_1/\lambda_2=1.451$ | **数值验证 + 对偶证书**（构造实例） | proofs 命题 0.4；verify_theorem0 |
| C5 | **Euler 乘子 $\lambda^\star=4J$** 与 **$3\times3$ 黎曼隐微分**：$H_{\mathbb S^3}=E^\top(\nabla^2J-4JI)E$，$dq^\star/d\theta$、$dt^\star/d\theta$ 闭式；对跖严格不变 | **已证明 + 数值验证**（dq 1.5e-7，dt 4.6e-8，H 9.6e-8） | proofs 定理 1.1；derivations §3–§5；verify_theorem11 |
| C6 | **可微性证书（充分）**：$\Delta J>0$、$H\succ0$、$G\succ0$ ⇒ 局部 $C^\infty$；源文档“精确充要”不成立（反例） | **已证明充分性；必要性反例** | proofs 定理 1.2 + 命题 1.2′ |
| C7 | **平移精确 + 旋转 Laplace**：$t$ 高斯积分精确；旋转 $O(\tau)$（数值斜率 1.05）；正确常数与 Haar 测度因子；分岔处需一致渐近 | **推导 + 数值验证；一致渐近为 GAP** | proofs 定理 1.3；derivations §7–§8；verify_theorem11 |
| C8 | **Laplace KL 的完整梯度**：Envelope 定理 + 权重协方差项 + $\nabla\log\det H_k$（含三阶 $q$ 导数）；$\tau\to0$ 只对均值/argmin 成立，KL 按 $1/\tau$ 发散 | **推导完整；实现待做** | proofs 1.3c；derivations §9 |
| C9 | **诚实的失败分析**：根数 27 vs 40、$\pm$ 对称与奇数矛盾、13 维实现不覆盖、Choi–Lam 可实现性、Laplace 非一致 | **审计结论** | proofs §8；math_questions |

## 2. 与 SDPRLayers / EPro-PnP 的逐维对比

| 维度 | SDPRLayers [1] | EPro-PnP [2] | **Laplace-QPEPLayer（本线）** |
|-|-|-|-|
| 非凸变量 | $x=[h;\mathrm{vec}(C);t]\in\mathbb R^{13}$ | $T\in SE(3)$（6 DoF） | $q\in\mathbb S^3$（4 维，1 约束）+ 闭式 $t$ |
| 约束 | 24 条二次等式（6 正交 + 9 handedness + 9 范数；后 18 条发布版标冗余） | 无硬约束（参数化） | 仅 $q^\top q=1$，处处 LICQ |
| 全局机制 | Shor SDP 松弛 + 对偶证书；tight 时 rank-1 | 依赖采样覆盖；无全局证书 | 枚举全部 KKT 根 + 比较目标；**无松弛、无证书** |
| 松弛/紧性 | 必须 tight；非 tight 无位姿 | 无松弛 | **零松弛间隙**（因为没有松弛）；但依赖枚举完整（A2） |
| 前向维度 | $O(n^2)$ PSD 矩阵；CPU 内点法（Mosek/SCS） | 采样 $N_s$ 个 6D 粒子 | 固定 $10\times10$ $M$ + $K$ 个根（$K\le40$ 候选） |
| 反向 | KKT 隐微分 + ACQ/SOSC/行约简；$16\times16$ 级 KKT | 采样加权重；MC 方差 | $3\times3$ 黎曼 Hessian / 模式；对跖不变 |
| 歧义/分支交叉 | 无定义（非 tight 或 corank>1 时失效） | 用后验质量覆盖；高方差 | 多峰 Laplace 权重；精确后验光滑，Laplace 在分岔处需一致渐近 |
| 采样 | 无 | AMIS 蒙特卡洛 | **无 Monte Carlo**（枚举 + 解析 Laplace） |
| 不等式约束 | 不支持 | 不支持（软惩罚可扩展） | **不支持**（companion track 2） |
| 外点鲁棒 | 靠 RANSAC | 无 | **不声称**（companion track 3） |
| 不确定性 | 对偶证书，不建模协方差 | 后验协方差（采样） | QPEP 观测度 $H_k$ + Laplace 体积权重 |
| 典型失效 | 非 tight、corank>1、冗余约束识别错 | 高方差、漏模、6D 采样 | 漏根/重根、Laplace 分岔、$G$ 奇异、可实现性 GAP |
| 计算复杂度 | 前向 SDP $O(n^6)$ 级；反向线性系统 | $O(N_s)$ 次残差/采样 | 前向 $O(N)$ 建 $M$ + 根枚举；反向 $O(K)$ 个 $3\times3$ |

### 2.1 与 SDPRLayers 的具体差异（写论文时按这 5 条展开）

1. **变量与约束**：我们用 4 维四元数 + 1 条约束，SDPRLayers 用 13 维 + 24 条（有效 6 条）。不是“同一问题的两种解法”，而是“在消去 $t$ 后的旋转子问题上，QPEP 的 4 维参数化 + 代数枚举”对“联合 $R,t$ 的 SDP 提升 + 松弛”。
2. **全局性来源**：SDPRLayers 的全局性是**凸松弛 + 对偶证书**；QPEP 的是**穷举 KKT 根 + 比较目标**。前者可独立验证（$H\succeq0$），后者不能；后者不要求 tight，前者要求。两者不是互相替代，而是互补。
3. **反向**：SDPRLayers 对原 QCQP 的 KKT 做隐微分，处理冗余约束用行约简 + 隐式选择定理；我们对约化后的 4 维平稳方程做 $3\times3$ 黎曼隐微分，没有冗余约束，但需要 QPEP 的根。
4. **歧义**：SDPRLayers 在非 tight 时没有位姿；EPro-PnP 用采样；我们用枚举 + Laplace。三者的失效模式不同。
5. **Scope**：我们的定理 0 只覆盖 10×10 旋转子问题；不声称等价于 SDPRLayers 的 13×13 实现。这是审稿人最可能抓的点，必须主动写清。

### 2.2 与 EPro-PnP 的具体差异

- **分布支撑**：EPro-PnP 在 $SE(3)$ 上定义 Boltzmann；我们在 $\mathbb S^3/\{\pm1\}$ 上定义，$t$ 精确边际化。我们的“6D 采样”被“4D 枚举 + 3D 高斯积分”替代。
- **积分方式**：EPro-PnP 用 AMIS 数值积分；我们用 Laplace 解析近似。前者渐近精确但高方差；后者确定性但有 $O(\tau)$ 偏差，分岔处不一致。
- **训练损失**：EPro-PnP 用 KL/负对数似然；我们给出 Laplace KL 的解析梯度（含 $\nabla\log\det H_k$ 与权重协方差项）。两者都可以端到端，但我们的梯度没有采样方差。
- **公平对比**：必须把“方差”和“偏差”分开。合成 benchmark 上可用稠密网格/求积算精确后验，分别测 EPro-PnP 的 MC 方差与我们的 Laplace 偏差；真实数据上报告多种子方差与 runtime。

## 3. 可能的审稿人质疑与回应

**Q1. “这不就是 QPEP + 普通 IFT 吗？增量太小。”**  
A：普通 IFT 需要：(i) 在正确的根上做；(ii) 处理分支交叉；(iii) 对 $q$ 的球面约束做黎曼化。源文档的“陷阱 A”正是 naive QPEP+IFT：它在 $\Delta J\to0$ 时给出错误/零梯度。我们的增量是：Euler 闭式乘子 + $H_{\mathbb S^3}$ 的 $-4J$ 项 + 对跖不变性（C5），以及用 Laplace 后验替代硬 argmin（C7–C8）。这些都不是 QPEP 论文的内容；QPEP 不提供对参数的梯度。

**Q2. “27 个根的说法从哪来？为什么和官方 40 不一致？”**  
A：27 无出处，且 27 为奇数、与 $(q,\lambda)\mapsto(-q,\lambda)$ 的对跖对称不相容；官方 LibQPEP 返回至多 40 个候选复根。我们统一写 $K\le40$ 候选、$K$ 个实局部极小，并把“枚举完整”列为假设 A2。这是**修正**，不是掩盖；见 proofs §6、math_questions Q3。

**Q3. “Laplace 近似不是精确积分，凭什么说平滑可微？”**  
A：**精确** Boltzmann 后验在紧流形上对 $\theta$ 是 $C^\infty$ 的（控制收敛），在分支交叉处也光滑；**Laplace 近似**只是它的 $O(\tau)$ 近似，在分岔处不一致。我们把“平滑”归给精确后验，把“$O(\tau)$”归给实现，并明确分岔处需要一致 Airy 渐近或一维数值积分。源文档“精确积分/解析平滑”是过度声明，我们在 proofs §7.3 修正。

**Q4. “$\tau\to0$ 真的退化为 QPEP 隐微分吗？”**  
A：对后验均值/软 argmin 成立；对 KL 损失不成立，因为 $\nabla\mathcal L_{\mathrm{KL}}$ 含 $\tau^{-1}(\nabla J_{\mathrm{gt}}-\nabla J_{k^\star})$，按 $1/\tau$ 发散。要恢复确定性梯度，用 $\tau\mathcal L_{\mathrm{KL}}$ 或对损失做 $\tau$-缩放。见 proofs §7.3、derivations §9.4。

**Q5. “21 约束 = Veronese 理想，维数 20 就够了，为什么还要秩？”**  
A：维数相等只是必要。若 21 条只张成 20 维的**另一个**子空间（例如只取 6 个非对角点积会得到 15 维），维数论证不成立。我们补了：(i) 每条约束在 $x_R(q)$ 上恒为 0（包含）；(ii) 21 条秩恰为 20。二者合起来才是相等。见 proofs 定理 0.2。

**Q6. “Choi–Lam 的 $M$ 是真实 PnP 的 Gram 矩阵吗？”**  
A：**未知，这是 P0 GAP。** 我们的实例是“构造的旋转子问题 SDP”的合法对称代价矩阵，不声称它来自真实 2D–3D 对应。要升级为“SDPRLayers 在实际 PnP 上失效”，需要证明可实现性 + 可实现锥内的开性/正测度。见 math_questions Q1/Q2/Q9。论文中必须写成“constructed instance”，不能写成“SDPRLayers fails on PnP”。

**Q7. “10×10 和 SDPRLayers 的 13×13 不是同一个 SDP，凭什么对比？”**  
A：我们不声称它们同值。定理 0.3 只对“消去 $t$ 后的旋转子问题”成立；13×13 的联合 SDP 是另一个松弛，二者关系是 GAP。论文的对比是**问题层面**的（零升维 vs 升维、枚举 vs 松弛、3×3 vs KKT），不是“同一个 SDP 的两种算法”。见 proofs §0、math_questions Q2。

**Q8. “没有实验，论文不完整。”**  
A：本 draft 明确标“Results TODO”，只交付可审的技术主张与可复现的理论/数值检查。实验协议已冻结（`experiments/protocol.md`），下一步按合成 → LINEMOD → In-The-Dark 推进；所有数字会标“论文声称/独立复现/未验证”。

**Q9. “和 EPro-PnP 比方差不公平：你们没有采样，当然方差低。”**  
A：所以我们不把“低方差”当唯一卖点。公平对比要分开：(i) 在合成 benchmark 上用稠密求积算精确后验，测 EPro-PnP 的 MC 方差与我们的 Laplace 偏差；(ii) 真实数据上报多种子方差、runtime、NLL 校准。我们的优势是**确定性 + 全局模式覆盖**，代价是**Laplace 偏差 + 分岔失效**。

**Q10. “$\tau$ 怎么选？可学习吗？尺度怎么固定？”**  
A：$\tau$ 有成本量纲；$M\mapsto sM$ 时需 $\tau\mapsto s\tau$ 才保持后验不变。建议 $\tau=\tau_0\cdot\operatorname{tr}(M)/10$（或 $\lambda_{\max}$），$\tau_0$ 可学习但必须固定尺度归一化，否则不可辨识。$\tau\to\infty$ 退化为均匀权重（梯度消失），$\tau\to0$ 退化为硬 argmin（KL 梯度发散）。见 derivations §11、math_questions Q8。

**Q11. “根枚举在 GPU 批量训练中可行吗？”**  
A：这是工程风险，不是理论主张。$40\times40$ 复特征分解可按 batch 做，但 GPU 上复数特征分解较慢；备选是 CPU 批量 + 多初值黎曼 Newton + 代数残差校验。我们在 protocol 中把它列为 P1 风险，并给出“先 CPU worker pool、后 GPU action matrix”的路线。

**Q12. “这与另外两条线（Constrained/Robust）的边界？”**  
A：本线不做不等式约束、不做外点鲁棒核。$M(\theta)$ 可以是任意加权 Gram 矩阵，但定理不依赖另外两条线的未发表结果；三条线共享 QPEP 前向与 $3\times3$ 反向作为基础设施，但贡献互不依赖。

## 4. 我们不声称的内容（写进论文的 Limitations）

1. 不声称 10×10 旋转子问题 SDP 与 SDPRLayers 的 13×13 实现同值。
2. 不声称 Choi–Lam 实例可由真实 PnP/矩阵加权对应实现。
3. 不声称“正测度物理参数区域必然非紧”（只有构造实例 + 开性数值证据）。
4. 不声称旋转 Laplace 是精确积分；只说 $O(\tau)$，并给出分岔失效。
5. 不声称“零采样”等于“精确后验”；无 Monte Carlo 正确，Laplace 有偏。
6. 不声称 $K\le27$；官方实现至多 40 候选根。
7. 不声称 $\tau\to0$ 对 KL 损失退化为确定性梯度；只对均值/argmin 成立。
8. 不声称支持不等式约束或粗大外点；那是 companion tracks。
