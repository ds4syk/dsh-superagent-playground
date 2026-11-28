# 相关工作：逐篇差异点与引用核实状态

> 说明：本文线的正式引用以"论文原名 + arXiv/DOI"为准。**内部精读记录**（本仓库 `kb/sdpr-qpep/01–04.md`、`kb/sdpr-forward-net/01.md`）只作为证据来源标注在此文件，不进论文正文。
> 状态标记：`[verified]` 已核对 arXiv 摘要/官方页面；`[verify]` 需最终核对卷期页；`[kb]` 仅有本仓库内部精读。

---

## 1. 核心对标：SDPRLayers `[verified]`

- **C. Holmes, F. Dümbgen, T. D. Barfoot.** *SDPRLayers: Certifiable Backpropagation Through Polynomial Optimization Problems in Robotics.* IEEE T-RO, vol. 41, 2025. arXiv:2405.19309v3. <https://arxiv.org/abs/2405.19309>
- 代码：`utiasASRL/sdprlayer`（PyTorch）。

**它做什么**：把机器人里的多项式优化问题（POP）写成齐次二次等式 QCQP，解其 Shor 半定松弛；松弛 tight（$\operatorname{rank}\mathbf X^\star=1$，$\operatorname{corank}\mathbf H^\star=1$）时读出全局解与对偶证书；反向**不**微分 SDP，而是回到原 QCQP 的 KKT 系统，用"隐式选择"（implicit selection）处理冗余约束（Theorem 4），得到 $J=-PM_r^\dagger N$ 且与乘子选择无关。

**与本文的差异（逐条）**：
1. **约束类**：它严格限于**二次等式**（$x^\top A_ix=0$）。不等式要靠松弛变量（PSD 块 $10\to10+p$）或 Putinar 局部化矩阵，代价是 $(10+p)^6$ 与严格互补失效。本文原生支持仿射不等式。
2. **SDP 的位置**：它的 SDP 在**正向**，每个样本每个训练步都解，CPU-only。本文正向不解 SDP，证书在后验、可异步。
3. **全局性的来源**：它靠"正向松弛 tight"；本文靠"后验 $14\times14$ 松弛 tight"。它的失败模式是"梯度无意义"；本文的失败模式是"无证书但前向仍是可行 KKT 点"。
4. **反向的数学**：它用序列二次/隐式选择处理**等式**冗余；本文用 FB 正则化处理**不等式**的秩亏与弱活跃。两者的"秩亏"不是同一件事（它的是等式约束线性相关，本文的是活跃不等式梯度线性相关）。
5. **不做的事**：它不建模不确定性（QPEP 有协方差 SDP）；本文也不做（不在本线范围）。

**内部证据**：`kb/sdpr-qpep/02.md` 对 Theorem 2 的证明链、三种反向模式（SDPR-SDP / SDPR-IS / CIFT）、紧性失效条件有逐条精读；`kb/sdpr-qpep/00.md` 记录本仓库复现其标定实验时发现"SCS 3.3.1 下 SDPR-IS 有 28% 求解失败，自适应容差重试降到 2%"，这条对本文的 baseline 复现风险（R4）直接相关。

---

## 2. 数学内核：QPEP `[verified]`

- **J. Wu et al.** *Quadratic Pose Estimation Problems: Globally Optimal Solutions, Solvability/Observability Analysis, and Uncertainty Description.* IEEE T-RO, 38(5):3314–3335, 2022. DOI 10.1109/TRO.2022.3155880.
- 代码：`LibQPEP`（C++/MATLAB）。

**它做什么**：把 PnP、PnL、PnPL、手眼标定、点到面配准、conic 位姿、测距相对位姿、Stewart 平台正运动学、多 GNSS 姿态统一成"四元数二次型 + $\|\mathbf q\|=1$"；消去 $\mathbf t$ 后用 Gröbner 基/作用矩阵枚举**全部** KKT 根（参考实现至多 40 个），按目标排序取最小，再 LM 精化；协方差用另一个小 SDP。

**与本文的差异**：
1. **无不等式、无梯度、无层**。本文保留其消元结构与枚举式全局性，加约束、加 ALM、加 FB 反向、加证书。
2. **"全局"的含义不同**：QPEP 的 globally optimal 是"枚举全部驻点后取最优"（在消元完备的前提下），**不是** SDPR 意义上的凸证书。本文因此**从不**把 QPEP 的输出单独当证书（`theory/proofs.md` §2、`paper/draft.md` §4.3）。
3. **根数**：QPEP 的 40 根来自其系统结构；本文内层的 Bézout 上界是 162，是否仍为 40 根是 open（`math_questions.md` Q1）。

**内部证据**：`kb/sdpr-qpep/03.md` 对"消 $t$ → 四次多项式 → Gröbner → 排序取最小 → LM"的代码级精读，以及"QPEP 无对偶证书，只有枚举完备性"的判断，是本文 §4.3 与 related work 的直接依据。

---

## 3. 可微凸 QP 层：dQP `[verified]`

- **C. W. Magoon, F. Yang, N. Aigerman, S. Z. Kovalsky.** *Differentiation Through Black-Box Quadratic Programming Solvers.* NeurIPS 2025. arXiv:2410.06324v4. <https://arxiv.org/abs/2410.06324>

**它做什么**：观察到"活跃集确定后，QP 的解与其导数共享同一个简化线性系统"，从而把求解器与微分解耦，支持 15+ 种黑盒 QP 求解器。

**与本文的差异**：
1. **凸性**：它的正确性依赖凸 ⇒ KKT 是充要条件。本文的非凸 $\mathbb S^3$ 上 KKT 只是必要，全局性必须另找证书。
2. **LICQ**：它假设活跃集可识别且 LICQ 成立。本文的四脚桌正是 LICQ 失效。
3. **借用与超越**：本文借用"小线性系统"的精神（$6+|\mathcal A|$），但用 FB 正则化替代硬活跃集判定，从而在秩亏/弱活跃处仍有定义。

---

## 4. 局部可微流形优化：Theseus `[verify]`

- **L. Pineda et al.** *Theseus: A Library for Differentiable Nonlinear Optimization.* NeurIPS 2022. HAL: hal-03739485. `[verify]`（arXiv 号待核）

**它做什么**：流形上的非线性最小二乘（GN/LM），支持展开与隐式两种反向；支持任意代价（含鲁棒核与惩罚项）。

**与本文的差异**：
1. **软惩罚 vs 硬约束**：惩罚的穿透随权重变化，永远非零；本文的约束把穿透压到机器精度。这条必须用 Pareto 曲线呈现才公平。
2. **局部 vs 全局**：它不声称全局；本文有后验证书（可失败）。
3. **互补**：Theseus 更适合"没有硬约束但需要复杂鲁棒代价"的情形（这正是 Prototype 3 的地盘）。

---

## 5. 可微凸锥层：OptNet / CVXPYLayers `[verified]`

- **B. Amos, J. Z. Kolter.** *OptNet: Differentiable Optimization as a Layer in Neural Networks.* ICML 2017. arXiv:1703.00443.
- **A. Agrawal, B. Amos, S. Barratt, S. Boyd, S. Diamond, J. Z. Kolter.** *Differentiable Convex Optimization Layers.* NeurIPS 2019. arXiv:1910.12430.

**与本文的差异**：二者都要求凸（QP / 锥规划），且反向依赖 KKT 可微（隐含 LICQ + 严格互补）。本文的非凸性来自旋转流形，不能靠 canonicalization 解决；本文的秩亏鲁棒性也是它们没有的。

---

## 6. 可微 PnP：BPnP / EPro-PnP `[verified]`

- **B. Chen et al.** *End-to-End Learnable Geometric Vision by Backpropagating PnP Optimization.* CVPR 2020. arXiv:1909.06043.
- **H. Chen et al.** *EPro-PnP: Generalized End-to-End Probabilistic Perspective-n-Points for Monocular Object Pose Estimation.* CVPR 2022. arXiv:2203.13254. `[verify]`

**差异**：BPnP 是局部 IFT（依赖初值、歧义位姿处不可微）；EPro-PnP 建模位姿后验（MC 采样，方差大）。两者都没有硬约束。本文与它们的可比点是"同骨干、只换层"的 T1 实验。

---

## 7. 硬约束学习：DC3 `[verified]`

- **P. Donti, D. Rolnick, J. Z. Kolter.** *DC3: A Learning Method for Optimization with Hard Constraints.* ICLR 2021. arXiv:2104.12225.

**差异**：DC3 在推理时做"完成+修正"以满足约束，但**不**对修正过程求导（或用近似）。本文把约束做进层，梯度经过约束。

---

## 8. 互补与半光滑：Fischer–Burmeister 一系 `[verified]/[verify]`

- **A. Fischer.** *A Special Newton-Type Optimization Method.* Optimization, 24(3–4):269–284, 1992. `[verified]`（经典条目）
- **C. Kanzow.** *Some Noninterior Continuation Methods for Linear Complementarity Problems.* SIAM J. Matrix Anal. Appl., 17(4), 1996. `[verify pages]`
- **F. Facchinei, J.-S. Pang.** *Finite-Dimensional Variational Inequalities and Complementarity Problems.* Springer, 2003. `[verified]`（标准专著）

**用法**：本文的 $\varphi_\varepsilon$ 是 FB 的光滑化版本；$a=1-\mu/r$、$b=1+g/r$ 是其一阶导。FB 的 BD-正则性与半光滑 Newton 的局部收敛是该文献的标准结论，本文不重复证明，只引用。**注意**：本文**不**声称 $\varphi_\varepsilon=0$ 等价于互补（见 `theory/proofs.md` Lemma 2.3）；这一点在 NCP 文献里也是常识（光滑化改变零点集）。

---

## 9. 增广拉格朗日与非凸收敛 `[verified]`

- **D. P. Bertsekas.** *Constrained Optimization and Lagrange Multiplier Methods.* Academic Press, 1982.
- **R. T. Rockafellar, R. J.-B. Wets.** *Variational Analysis.* Springer, 1998.
- **R. Fletcher.** *Practical Methods of Optimization* (2nd ed.). Wiley, 1987.（精确罚）

**用法**：非凸 ALM 收敛到 KKT 点、乘子有界性、精确罚的 exactness 条件。本文用这些作为**假设**（A3/A4），不重证。

---

## 10. 非光滑隐微分 `[verify]`

- **J. Bolte, T. Le, E. Pauwels, A. Silveti-Falls.** *Nonsmooth Implicit Differentiation for Machine Learning and Optimization.* NeurIPS 2021. `[verify arXiv id]`

**用法**：为"解映射不可微时，层给出的是什么"提供框架（保守场 / Clarke 广义 Jacobian）。本文的 Prop 2.1 与 Lemma 2.5 用它来**限定**声明，而不是用来声称可微。

---

## 11. 数值求解器 `[verified]`

- **B. O'Donoghue, E. Chu, N. Parikh, S. Boyd.** *Conic Optimization via Operator Splitting and Homogeneous Self-Dual Embedding.* J. Optim. Theory Appl., 169:1042–1068, 2016.（SCS；本文 sanity check 的 SDP 求解器）
- 另有 MOSEK（商用内点法）作为大规模 SDP 的备选，本文不依赖。

---

## 12. 对比总表

| 工作 | venue | 约束 | 非凸流形 | 全局性来源 | 反向鲁棒性 | 证书 | 与本文的关系 |
|---|---|---|---|---|---|---|---|
| SDPRLayers | T-RO'25 | 等式 QCQP | ✓ 提升 | 正向 SDP tight | 等式冗余（隐式选择） | ✓ 正向 | 最近对标；本文补不等式 + 后验证书 |
| QPEP | T-RO'22 | $\|\mathbf q\|=1$ | ✓ | 枚举全根 | 无反向 | ✗ | 数学内核；本文加约束/梯度/证书 |
| dQP | NeurIPS'25 | 线性（凸 QP） | ✗ | 凸 | LICQ 假设 | n/a | 小线性系统精神的来源 |
| Theseus | NeurIPS'22 | 软惩罚 | ✓ 局部 | 无 | 局部 IFT | ✗ | 局部/惩罚 baseline |
| OptNet | ICML'17 | 线性（凸 QP） | ✗ | 凸 | LICQ + 严格互补 | n/a | 凸层代表 |
| CVXPYLayers | NeurIPS'19 | 锥 | ✗ | 凸 | KKT 可微 | n/a | 凸层代表 |
| BPnP | CVPR'20 | 无 | ✓ 局部 | 无 | 局部 IFT | ✗ | 局部 PnP baseline |
| EPro-PnP | CVPR'22 | 无 | ✓ 概率 | 无 | 后验采样 | ✗ | 概率 PnP baseline |
| DC3 | ICLR'21 | 硬（推理时修正） | ✗ | n/a | 不可微/近似 | ✗ | 对照：约束不进梯度 |
| **Ours** | — | **仿射不等式** | ✓ | **枚举 + 后验 SDP** | **FB 正则化（秩亏/弱活跃）** | **✓ 后验 sound** | — |

---

## 13. 引用核实待办

- [ ] Theseus 的 arXiv 编号与页码（目前只核到 HAL: hal-03739485）。
- [ ] EPro-PnP 的 arXiv:2203.13254 与 TPAMI 2024 扩展版的关系。
- [ ] Kanzow 1996 的卷期页。
- [ ] Bolte et al. 2021 的 arXiv 编号。
- [ ] SDPRLayers 的 T-RO 卷期页（ADS 记录为 vol. 41, p. 4120 起，需核末页）。
- [ ] dQP 的 NeurIPS 2025 正式页码（目前只有 arXiv v4）。
- [ ] QPEP 的作者全名列表（目前只写 "J. Wu et al."，需在投稿前补全）。

**规则**：以上任一条未核实前，论文里不得出现具体页码；宁可只写 venue + 年份 + arXiv/DOI。
