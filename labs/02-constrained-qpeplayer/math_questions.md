# Constrained-QPEPLayer：未解决数学问题清单

> 用法：每条含 **编号 / 精确问题陈述 / 为什么重要 / 已尝试思路 / 需要用户提供什么 / 优先级**。
> 优先级定义：**P0** = 不解决就不能提交（审稿人会直接拒或本文的核心声明失效）；**P1** = 影响可信度或实验结论，需在 4 周内解决或明确降级声明；**P2** = 可留作 future work。
> 状态标记：`[open]` 未动、`[partial]` 有部分结果/数值证据、`[narrowed]` 已把问题缩小到可判定的子问题。

---

## Q1（P0）内层 KKT 系统的通用根数与 QPEP 消元模板的可复用性 `[narrowed]`

**精确陈述.** 设内层子问题为
$$
\min_{\mathbf q\in\mathbb S^3}\ \Phi(\mathbf q)=\mathbf v_2(\mathbf q)^\top\mathbf M\,\mathbf v_2(\mathbf q),\qquad
\mathbf M=\mathbf M_0+\sum_{j\in\mathcal A}\Big[\tfrac{\mu_j}{2}(\mathbf e_{\rm tr}\tilde{\mathbf a}_j^\top+\tilde{\mathbf a}_j\mathbf e_{\rm tr}^\top)+\tfrac{\rho}{2}\tilde{\mathbf a}_j\tilde{\mathbf a}_j^\top-\mathbf D_{\mathcal A}^\top\mathbf G_{\mathcal A}^{-1}\mathbf D_{\mathcal A}\text{-type terms}\Big].
$$
记 $\mathcal I=\langle \partial_{q_i}\Phi-2\lambda q_i\ (i=0..3),\ \|\mathbf q\|^2-1\rangle\subset\mathbb C[\mathbf q,\lambda]$。问：对一般的 $(\mathbf M_0,\{\tilde{\mathbf a}_j,\mathbf b_j\},\mu,\rho)$，$\dim_{\mathbb C}\mathbb C[\mathbf q,\lambda]/\mathcal I$ 等于多少？特别地，是否等于 QPEP 原问题（$\mathcal A=\varnothing$）的 **40**？若不等，上界是否为 Bezout 数 $3^4\cdot2=162$？

**为什么重要.**
- 论文的"前向复杂度与 $p$ 无关"只依赖 **维数锁定**（已证），但"复用 QPEP 现成求解器/249×249 模板"依赖根数不变。若根数从 40 涨到上百，前向常数变大 2–4 倍，且现成 `solver_WQ` 需要重写模板。
- 若某些 $(\mu,\rho)$ 使系统出现**正维数解集**（例如 $\mathbf M$ 退化到使 KKT 有连续解），Gröbner 求解器会直接失败，ALM 会卡住。需要知道这种参数集的余维数。

**已尝试思路.**
1. 理论：$(\mathbf q,\lambda)$ 共 5 个未知量，4 个三次 + 1 个二次，Bezout 162。QPEP 的 40 是**该系统族的实际通用根数**（其代码 `solution_size=40`，见 `kb/sdpr-qpep/03.md` §2.3），远小于 Bezout，说明有结构性的线性相关（来自 $\mathbf v_2$ 的二次结构）。
2. 关键观察：$\Phi$ 是 $\mathbf v_2$ 的二次型 + $\mathbf v_2$ 上的线性项（经 $\mathbf e_{\rm tr}$），其梯度结构 $\nabla_q\Phi=4\,\mathbf J_{v_2}(\mathbf q)^\top\mathbf M\mathbf v_2$ 与 QPEP 的 $\mathbf M_0$ 情形**词支撑完全相同**（都是全体三次单项式），只是系数不同。这提示通用根数应与 40 相同——**但这是猜测，未证明**。
3. 可行的判定路径（未执行）：
   - (a) 用 `sympy` 取符号 $\mathbf M$（10×10 对称 + 3 个约束参数），对 $\mathcal I$ 做 Gröbner 基，数标准单项式个数 ⇒ 得商环维数；
   - (b) 数值：随机 200 组 $(\mathbf M_0,\mathcal A,\mu,\rho)$，用同伦延拓（`PHCpack`/`Bertini`）或 Macaulay2 数复根，统计上界与重根比例；
   - (c) 退化检测：用最小奇异值/条件数监测 KKT Jacobian，找正维数解集的参数区域。

**需要用户提供什么.**
- 是否有 Macaulay2 / Bertini / PHCpack 可用（本机只有 sympy + numpy；sympy 的 Gröbner 在 5 变量三次系统上可能很慢）。
- 是否可以接受"重新生成模板"（约 1 周工程）作为 Plan B；若接受，Q1 降级为 P1。
- QPEP 官方 `solver_WQ` 的模板生成脚本是否可得（决定能否直接改）。

**当前处置.** 论文里写成"同一代数族 + 维数锁定已证 + 模板复用待验"（draft §4.2 remark (b)、G1）。`[narrowed]` 到"数商环维数 + 随机根数统计"两件可执行的事。

---

## Q2（P0）弱活跃点上 FB 正则化选出的到底是哪一个广义导数 `[open]`

**精确陈述.** 设 $\varphi_0(\mu,g)=\mu-g-\sqrt{\mu^2+g^2}$，在 $(\mu,g)=(0,0)$ 处不可微。设 $(\mu_\varepsilon,g_\varepsilon)$ 是正则化系统 (proofs 式 3.4) 的解路径。问：
1. 当 $\varepsilon\downarrow0$ 时 $d\mathbf z_\varepsilon$ 的极限 $d\mathbf z^\star$ 是否总存在（本文已在 $\mathbf J_{\mathcal A}$ 固定、$\mathbf D_a\to0$ 的线性模型下证存在，见 Prop 2.1）；
2. 该极限对应 $\varphi_0$ 的 Clarke 广义 Jacobian 中的哪一个元素？能否用"$\sqrt{\cdot}$ 在原点的次微分取 $\mu,g\to0$ 的方向极限"来刻画（本文 Lemma 2.5 给的是 $a=b=1$，即取 $\mu,g$ 同阶趋于 0 的方向）？
3. 若 $\varepsilon$ 与 $\mu,g$ 的消失速率不同（例如 $\varepsilon=\varepsilon(\mu)$），选择规则如何变化？极限是否仍唯一？

**为什么重要.**
- 这是本文相对源文档最实质的修正。源文档写"$\varepsilon\to0$ 精确收敛到真方向导数"，在弱活跃点**不成立**：真导数可能不存在，FB 只能给一个选择。
- 审稿人（尤其做 nonsmooth optimization 的）一定会问这一条。答不上来，定理 2.3 就只能降级为"一个正则化算法"，失去理论贡献。
- 这也决定了实验里"弱活跃点梯度与有限差分一致"的结论能否写。目前 `verify` 只验证了强活跃/秩亏与相容情形，**未**验证弱活跃。

**已尝试思路.**
1. 文献映射：Facchinei–Pang 的 NCP 专著对 FB 的 BD-正则性与半光滑 Newton 有完整结论；Bolte–Le–Pauwels–Silveti-Falls (NeurIPS 2021) 给了非光滑隐微分的保守场框架，可用于定义"层在不可微点的导数"。两者都需要与本文的 $\mathbf D_a$ 参数化对齐。
2. **2D 反例已做，结论明确**：$\min\tfrac12(z-a)^2$ s.t. $z\ge0$ 在 $a=0$ 处真解 $z^\star=\max(a,0)$ 不可微、Clarke 广义导数 $[0,1]$；FB 正则化给出 $z_\varepsilon(a)=\tfrac12(a+\sqrt{a^2+2\varepsilon^2})$，$dz_\varepsilon/da|_{a=0}=1/2$ **与 $\varepsilon$ 无关**，即精确等于 Clarke 中点；$|a|\gg\varepsilon$ 时给出 $0$ 或 $1$。数值见 `verify` F7（17/17 通过），解析见 `theory/proofs.md` 例 2.5.1。**因此至少单约束线性化情形下 FB 选出的是中点**；一般多约束/秩亏几何是否仍是"中点"是仍未解决的部分。
3. 待做：把例 2.5.1 推广到 $m$ 个同时弱活跃约束，检验 FB 给出的是"每个约束 $1/2$ 权重"还是某个与 $\mathbf J$ 几何相关的加权。这决定论文能否写"FB 在弱活跃点给出自然的中点选择"这一强表述。
3. 方向性论证：$a=1-\mu/r$、$b=1+g/r$，在 $(\mu,g)=(0,0)$、$r=\varepsilon$ 时 $a=b=1$ ⇒ $\mathbf D_a=\mathbf I$ ⇒ 该约束在反向系统里表现得像"半活跃、权重 1"。这等价于把弱活跃约束**当作 $50\%$ 活跃**处理。是否与 Clarke 中点一致，需证明。

**需要用户提供什么.**
- 是否接受把定理 2.3 的结论改为"收敛到一个可刻画的广义导数（Clarke 中点）"？这会削弱 novelty 表述但提高可信度。
- 是否有熟悉 nonsmooth analysis 的合作者可审这一段（这是本线唯一需要外部数学审稿的地方）。

**当前处置.** `[open]`，记为 G2/§4.6，并在 draft §4.5、§6 明确降级声明。**建议立刻做 Q2 的 2D 反例数值**（成本 < 30 min，收益高）。

---

## Q3（P0）$\mathbf H_{\rm tan}$ 到底含不含 $\sum_j\mu_j\nabla^2 g_j$：GN 还是精确 Hessian `[narrowed]`

**精确陈述.** 式 (9) 的 $\mathbf H_{\rm tan}$ 有两种取法：
- (E) 精确：$\mathbf H_{\rm tan}=\nabla^2J+\sum_{j\in\mathcal A}\mu_j\nabla^2 g_j$。此时 (9) 是真 KKT 微分，但 $\mathbf H_{\rm eff}\succ0$ **不自动成立**，定理 2.3 的可逆性只能退到"$\mathbf H_{\rm eff}$ 在 $\ker\mathbf J_{\mathcal A}$ 上非奇异"；
- (G) Gauss–Newton：$\mathbf H_{\rm tan}=\nabla^2J$（或 $\nabla^2J$ 的 PSD 部分）。此时 $\mathbf H_{\rm eff}\succ0$ 自动（若 $\nabla^2J\succ0$），定理 2.3 干净，但算出的 $d\mathbf z$ **不是**精确导数。

问：在本文的约束类（$g_j$ 关于 $\mathbf v_2$ 仿射，但关于 $\mathbf q$ 二次）下，$\|\mathbf H_{\rm tan}^{(E)}-\mathbf H_{\rm tan}^{(G)}\|$ 的量级是多少？是否存在一个**加权**GN（保留 $\mu_j\nabla^2g_j$ 的 PSD 部分）既保持可逆性又保持一阶精度？

**为什么重要.**
- 这是审稿人最容易抓的一条：源文档同时写了 $\mathbf H_{\rm eff}=\mathbf H_{\rm tan}+\sum(b_j/a_j)\mathbf J^\top\mathbf J\in\mathbb S^6_{++}$（要求 PSD）和"精确收敛到真方向导数"（要求精确 Hessian）。二者不可兼得。
- 若只能给 GN，则实验里"与有限差分误差 $4\times10^{-11}$"的预实验数字必须在**精确 Hessian** 下才成立；GN 下会有一个与 $\mu_j\|\nabla^2 g_j\|$ 成正比的系统误差。**这直接影响源文档预实验 2B 的可信度。**

**已尝试思路.**
1. 量级估计：$g_j$ 关于 $\mathbf q$ 是二次型，$\nabla^2_{\mathbf q}g_j$ 的量级 $\sim\|\tilde{\mathbf a}_j\|$；$\mu_j$ 在强活跃点 $\sim$ 接触力。故 GN 误差 $\sim\mu_j\|\tilde{\mathbf a}_j\|$，不是小量。
2. 折中方案（未验证）：$\mathbf H_{\rm tan}^{(W)}=\nabla^2J+\sum_{j\in\mathcal A}\mu_j\big[\nabla^2g_j\big]_{\rm PSD}$（取正部）。这保持 $\mathbf H_{\rm eff}\succeq\nabla^2J\succ0$（若 $\nabla^2J\succ0$），同时保留部分二阶信息。代价是引入一个非光滑的谱投影。
3. 实现路径：在 `torch.autograd.Function` 里分别实现 E 与 G，对同一 batch 比较 $d\mathbf z$ 与中心差分；用实验数据决定论文主推哪一个。

**需要用户提供什么.**
- 决策：论文主定理用 (G) 干净版 + (E) 作为 Remark/实验，还是花时间做加权 GN？**建议前者**（RSS/ICRA 审稿更看重假设干净）。
- 是否接受"精确 Hessian 版本在部分样本上不可逆，需要 Levenberg 阻尼"作为工程 fallback。

**当前处置.** `[narrowed]`：定理 2.3 用 (G)，(E) 写成 Remark + 消融项（draft §4.5、§5.3 ablation (i)）。

---

## Q4（P1）14×14 Shor 松弛的紧性是否有结构性的充分条件 `[partial]`

**精确陈述.** 设 $\mathrm{Shor}$ 为 (6) 的最优值，$J^\star$ 为 (C-QPEP) 全局最优。已知 $\mathrm{Shor}\le J^\star$。问：存在什么可检验的条件（关于 $\mathbf M_0,\mathbf G_0,\{\tilde{\mathbf a}_j,\mathbf b_j\}$，或关于对偶变量），使得 $\mathrm{Shor}=J^\star$？特别地：
1. 是否存在类似"SOSC + 严格互补 + 乘子唯一"的充分条件？
2. 加更多 Veronese 关系（3 次、4 次关系？）能否把紧性从"经验"变成"定理"？
3. 松弛不紧时，$\mathrm{Shor}$ 与 $J^\star$ 的差距是否有可计算的界（例如用 $\lambda_2/\lambda_1$ 估计）？

**为什么重要.**
- 这是本文"certifiable"一词的全部依据。若只能给经验紧性，论文的定位必须从 "certifiable layer" 降为 "layer with a sound a posteriori bound"，直接影响投稿档次与 story。
- 目前只有一个 $p=12$ 的合成实例显示紧（$\lambda_2/\lambda_1=2.2\times10^{-12}$）。审稿人会问"这是不是特例"。

**已尝试思路.**
1. 已知：对**无约束**情形（$p=0$），加上 Veronese 关系后 $\mathrm{Shor}$ 恰等于 QPEP 的数值全局值（0.0030334692 vs 0.0030334724）。这不是巧合：$\|\mathbf v_2\|^2=1$ + Veronese 关系把松弛限制在 Veronese 簇的凸包里，而对四次型在球面上的最小化，"Veronese + 凸包"很可能就是精确的（**这是猜想**：$\operatorname{conv}(\nu_2(\mathbb{S}^3))$ 上的线性优化在特定目标下取到顶点）。
2. 文献线索：Lasserre 层级在**球面**上的 tightness 结果（degree-2 Lasserre on $\mathbb S^{n}$ 对二次型是精确的，即"$\min_{\mathbb S^n}$ 二次型 = 最小特征值"）。本文的四次型 = $\mathbf v_2$ 上的二次型，而 $\mathbf v_2$ 落在 Veronese 簇上；若把 Veronese 关系全部加入（不只 2×2 子式，而是整个二次关系理想），则松弛等价于"在 $\operatorname{conv}(\nu_2(\mathbb S^3))$ 上最小化二次型"。问：二次型在 $\operatorname{conv}$ 上的最小是否总在顶点取到？**一般不是**（凸包上的线性函数才总在顶点取到；这里目标二次、变量是 $\mathbf X$，不等价）。
3. 可做的实验：在 YCB-V 上统计紧性率，按 $p$、按接触构型、按 $\lambda_2/\lambda_1$ 分层。这是 §5.4 的 headline。

**需要用户提供什么.**
- 是否接受"经验紧性 + 失败率报告"作为论文的核心证据（我建议接受，并把叙事改成"fail-safe"）。
- 若要有定理，需要 2–3 周的纯理论工作（Lasserre on Veronese / convex hull of Veronese），且不保证成功。**建议不投这个时间**，改写 story。

**当前处置.** `[partial]`：soundness 已证（Lemma 2.2），紧性只有数值；论文定位为 fail-safe（draft Abstract、§6）。

---

## Q5（P1）ALM 的乘子与 $\mathbf D_a$ 的关系，以及 $\varepsilon$ 的自适应调度 `[open]`

**精确陈述.** 本文用 $\mathbf D_a=\operatorname{diag}(a_j/b_j)$，强活跃时 $a_j/b_j\approx\varepsilon^2/(2\mu_j^2)$。问：
1. 给定目标精度 $\tau_{\rm grad}$（希望 $\|d\mathbf z_\varepsilon-d\mathbf z^\star\|\le\tau_{\rm grad}$），$\varepsilon$ 应如何取？由 Prop 2.1 的证明，误差 $\asymp\lambda_{\min}(\mathbf D_a)\asymp\varepsilon^2/\max_j\mu_j^2$，故 $\varepsilon\lesssim\sqrt{\tau_{\rm grad}}\cdot\max_j\mu_j$。但这与条件数 $\kappa\asymp1/\varepsilon^2$ 冲突：$\tau_{\rm grad}=10^{-8}$ 要求 $\varepsilon\lesssim10^{-4}\mu$，此时 $\kappa\approx10^{8}\mu^2/\varepsilon^2$ 已接近双精度极限。**存在一个最优 $\varepsilon^\star$ 使"截断误差 + 舍入误差"最小**，需要精确的权衡公式。
2. $\varepsilon$ 是否应随 $\mu_j$ 逐约束不同（$\varepsilon_j\propto\mu_j$），以平衡各约束的 $a_j$？
3. $\rho$ 的调度（本文用 violation 不降则 $\times5$）是否会影响 $\mathbf D_a$ 的尺度，从而影响 1？

**为什么重要.**
- 这是唯一一个"理论给了公式但工程上必须调参"的地方，也是复现风险最高的地方。
- 若 $\varepsilon$ 选不好，实验里会出现"梯度与 FD 差 $10^{-2}$"或"$\kappa>10^{12}$ 导致 NaN"，审稿人会归因为方法不稳。

**已尝试思路.**
1. 数值：`verify` F4b 实测 $\kappa$ 斜率 $-2.00$（$\kappa\asymp1/\varepsilon^2$）；F5a 实测 $d\mathbf z$ 误差随 $\mathbf D_a$ 线性下降（$1.3\times10^{-5}\to1.3\times10^{-11}$）。故"误差 $\asymp\mathbf D_a$"成立。
2. 权衡：截断误差 $\asymp\mathbf D_a$，舍入误差 $\asymp\kappa\cdot\epsilon_{\rm mach}\asymp\epsilon_{\rm mach}/\mathbf D_a$。最小化 $\mathbf D_a+\epsilon_{\rm mach}/\mathbf D_a$ 得 $\mathbf D_a^\star=\sqrt{\epsilon_{\rm mach}}\approx10^{-8}$，即 $\varepsilon^\star\approx\sqrt{2}\mu\cdot10^{-4}$。**这是一个可写进论文的实用性结论**，但目前只是量纲分析，未做严格误差界。
3. 实现：$\varepsilon_j=\max(\varepsilon_{\min},\ c\,\mu_j)$，$c=10^{-4}$，$\varepsilon_{\min}=10^{-8}$；对非活跃约束 $\varepsilon$ 无关紧要（$b/a\approx\varepsilon^2/(2g^2)$ 已足够小）。

**需要用户提供什么.**
- 是否要为此写一个引理（"$\varepsilon$ 的最优尺度是 $\sqrt{\epsilon_{\rm mach}}\max\mu$"）？需要一段严格的浮点误差分析，约 1 周。
- 实验上是否允许把 $\varepsilon$ 作为超参在验证集上选（我建议允许，并在论文里报告敏感性曲线）。

**当前处置.** `[open]`，但已有可用的启发式（$c=10^{-4}$）与敏感性实验计划（§5.3 ablation (ii)）。

---

## Q6（P1）非刚体四脚桌（4 条腿独立）的解映射：是否存在单侧方向导数 `[open]`

**精确陈述.** 设 $g_j=z+a_j\,\text{roll}+b_j\,\text{pitch}-h_j(\theta)$，$j=1..4$，四个 $h_j$ 独立参数化。此时 $\mathbf w=-\partial_\theta\mathbf g_{\mathcal A}\in\mathbb R^4$ 一般 $\notin\operatorname{range}(\mathbf J_{\mathcal A})$（$\mathbf J_{\mathcal A}\in\mathbb R^{4\times3}$ 秩 3）。问：
1. 约束问题的解映射 $\theta\mapsto\mathbf z^\star(\theta)$ 在 $\theta=0$ 处是否存在**单侧**方向导数？若有，是哪一个方向（$\theta$ 增大侧）？
2. 该单侧导数是否等于"活跃集从 4 降到 3（丢掉乘子最小的那个）"后的 KKT 导数？即 FB 极限与 active-set QP 的解是否一致？
3. 若一致，能否把 FB 的 $\mathbf D_a$ 权重（$\propto1/\mu_j^2$）解释为"丢约束的概率/顺序"？

**为什么重要.**
- 决定了"四脚桌"这个招牌例子能否用于**非刚体**场景（例如四条腿各自有柔性、或四个接触点来自不同物体）。
- 与 Q2 相关：这是"解映射不可微"的最具体实例。
- 若 (2) 成立，本文可以给一个非常漂亮的结论：**FB 在不可微点自动选出了"活跃集即将退出"的那个约束**，这正是 active-set 方法手工要做的事。

**已尝试思路.**
1. 数值（未做）：对一个 4 约束/3 变量的线性化问题，取 $h_j(\theta)=h_{j,0}+\theta\,\delta_j$ 使 $\delta\notin\operatorname{range}(\mathbf J_{\mathcal A})$，用 `scipy.optimize` 解参数化 QP 得 $\mathbf z^\star(\theta)$，做单侧差分；与 FB 极限（Prop 2.1 的 (10) 式，$\mathbf D_a\to0$）比较。
2. 理论上：4 个约束、3 个变量，$\theta>0$ 时必有一个约束的乘子变为负（互补松弛不允许），故活跃集降为 3；解沿一个 facet 滑动，单侧导数存在。**这个论证几乎就是证明**，只需处理"哪个约束退出"的并列情形。
3. 风险：若两个乘子同时到 0（退化），单侧导数也可能不存在，需要二阶展开。

**需要用户提供什么.**
- 是否值得为这条写一个 Proposition（约 3–5 天）？我判断**值得**：它把"四脚桌"从"我们能处理秩亏"升级为"我们知道不可微时会发生什么"，是审稿人眼里的加分项。
- 若不做，论文必须明确写"非刚体参数不在范围内"（draft §6 已写）。

**当前处置.** `[open]`；draft §4.5 已区分刚体/非刚体两种情形并声明范围。

---

## Q7（P1）约束"仿射于 $(\mathbf v_2,\mathbf t)$"这一类的边界在哪里 `[narrowed]`

**精确陈述.** 哪些机器人几何约束可以写成 $\tilde{\mathbf a}^\top\mathbf v_2(\mathbf q)+\mathbf b^\top\mathbf t\le0$？已知：正深度、平面非穿透、FoV（两条）、倾角锥。问：
1. **距离/避障** $\|\mathbf R\mathbf p+\mathbf t-\mathbf c\|\ge d$：平方后是四次（$\|\cdot\|^2$ 是 $(\mathbf v_2,\mathbf t)$ 的二次型，$\ge d^2$ 是关于 $(\mathbf v_2,\mathbf t)$ 的二次不等式）。**注意：它是 $(\mathbf v_2,\mathbf t)$ 的二次不等式，不是仿射！** 但本文的 ALM 能处理吗？——不能直接：$g_j$ 的平方会产生四次×四次=八次项。**结论：不在范围内**，但需在论文里给出理由。
2. 相对位姿/多机一致性约束 $\|\mathbf t_i-\mathbf t_j\|\le d$（凸，但是 $\mathbf t$ 的二次）。
3. 关节限位、速度限位（对 $\mathbf q$ 线性？倾角锥是；关节角不是）。
4. 平面接触的**摩擦锥** $\|f_t\|\le\mu f_n$：力不在位姿变量里，不在范围内。
5. **半空间交集**的一般形式：任意"点在某平面一侧"都是仿射 ✓。

**为什么重要.**
- 审稿人会问"你们的约束类比 SDPRLayers 的等式类大吗？"。需要一个清楚的、可判定的刻画。
- 避障是最常见的机器人约束；若不能处理，必须明确说，否则会被当作 overclaim。

**已尝试思路.**
1. 判据：$g$ 必须是 $(\mathbf v_2,\mathbf t)$ 的**仿射**函数。因 $\mathbf v_2$ 的分量是 $\mathbf q$ 的二次单项式，等价于 $g$ 是 $\mathbf q$ 的二次型 + $\mathbf t$ 的线性 + 交叉项 $\mathbf q^\top(\cdot)\mathbf t$ 的形式。据此：
   - 深度、非穿透、FoV、倾角锥、平面半空间、$\mathbf q$ 的二次型不等式（如 $\mathbf q^\top\mathbf A\mathbf q\le c$）✓；
   - 距离下界、球约束、任何含 $\|\mathbf R\mathbf p+\mathbf t\|$ 的约束 ✗（除非平方后仍保持仿射，一般不）。
2. 一个有用的补充：**二次型不等式** $(\mathbf v_2^\top\mathbf A\mathbf v_2+\dots)\le0$ 可以通过"把 $\mathbf v_2$ 的二次型当新变量"处理吗？不行，会引入 $\mathbf v_4$。**结论：本文的类 = 仿射于 $(\mathbf v_2,\mathbf t)$。**
3. 可能的扩展（future work）：若 $g$ 是 $(\mathbf v_2,\mathbf t)$ 的**凸**二次，ALM 的内层变成"四次型 + 四次惩罚"，无法再用 QPEP 的 5 变量系统（会升到 9 变量、次数 8）。这是一条清晰的边界。

**需要用户提供什么.**
- 目标应用里到底需不需要避障约束？若需要（例如 VLA 桌面操作要避开其他物体），本文的 scope 会显著受限，需要考虑"用多个半空间逼近凸障碍"的工程替代（把球/柱障碍用切平面近似成仿射约束），这可以在实验里做，但要在论文里说清是近似。
- 是否接受"半空间逼近"作为规避方案（我建议接受并在 T1 里实测近似误差）。

**当前处置.** `[narrowed]`：draft §3.4 明确 out of scope；roadmap 里列为可能的扩展实验（切平面逼近）。

---

## Q8（P2）$\rho\to\infty$ 时内层问题的退化与 Gröbner 求解器的数值稳定性 `[open]`

**精确陈述.** $\mathbf M_{\mathcal A}$ 含 $(\rho/2)\tilde{\mathbf a}_j\tilde{\mathbf a}_j^\top$ 项，$\mathbf G_{\mathcal A}=\mathbf G_0+(\rho/2)\sum\mathbf b_j\mathbf b_j^\top$。$\rho\to\infty$ 时 $\mathbf M_{\mathcal A}$ 的条件数 $\asymp\rho$，$\mathbf G_{\mathcal A}$ 亦 $\asymp\rho$，$\mathbf M^{\rm red}_{\mathcal A}$ 可能出现 $\rho$ 量级的抵消（两个 $\asymp\rho$ 的量相减）。问：
1. $\mathbf M^{\rm red}_{\mathcal A}$ 的条件数随 $\rho$ 的增长速率？
2. 这会不会使 Gröbner/作用矩阵的数值秩判定失效（漏根/多根）？
3. 是否应改用 $\mathbf M^{\rm red}$ 的稳定形式（例如用 $\mathbf G_{\mathcal A}$ 的 Cholesky 因子做白化）？

**为什么重要.** ALM 通常要把 $\rho$ 增大 2–3 个数量级才能把可行性压到 $10^{-10}$；若这破坏了内层求解器，前向就崩了。这是**实现风险**，不是理论风险。

**已尝试思路.**
1. 白化：令 $\mathbf G_{\mathcal A}=\mathbf L\mathbf L^\top$，$\tilde{\mathbf t}=\mathbf L^\top\mathbf t$，则 $\mathbf t^\top\mathbf G\mathbf t=\|\tilde{\mathbf t}\|^2$，$\mathbf D_{\mathcal A}\to\mathbf D_{\mathcal A}\mathbf L^{-\top}$。这样 $\mathbf G$ 消失，只剩 $\mathbf M_{\mathcal A}$ 的 $\rho$ 增长。
2. $\mathbf M^{\rm red}=\mathbf M_{\mathcal A}-\mathbf D^\top\mathbf G^{-1}\mathbf D$ 的抵消：注意 $\mathbf M_{\mathcal A}$ 与 $\mathbf D^\top\mathbf G^{-1}\mathbf D$ 的 $\rho$ 主导项分别是 $(\rho/2)\tilde{\mathbf a}\tilde{\mathbf a}^\top$ 与 $(\rho/2)\mathbf b\tilde{\mathbf a}^\top(\cdot)^{-1}(\rho/2)\tilde{\mathbf a}\mathbf b^\top$ 的合成，**未必同阶抵消**，需具体算。`verify` C3 实测 $\lambda_{\min}(\mathbf G_{\mathcal A})=10.5$（$\rho=3$），未测大 $\rho$。
3. 实验计划：$\rho\in\{1,10,10^2,10^3,10^4\}$，测根数、残差、条件数、最终可行性。这是 §5.4 的"failure taxonomy"之一。

**需要用户提供什么.** 无（纯实验）。若时间紧，可只做 $\rho\le10^3$ 并声明上限。

**当前处置.** `[open]`，P2。

---

## Q9（P2）与 SDPRLayers 松弛变量版的严格等价性/优势量化 `[narrowed]`

**精确陈述.** 设 SDPRLayers 处理同一 (C-QPEP) 时引入 $p$ 个松弛变量，得到 $(10+p)\times(10+p)$ 的 Shor 松弛及其对偶证书。本文的 $14\times14$ 松弛 (6) 与它的关系是什么？
1. 是否存在一个线性映射把 $(10+p)$ 松弛的最优值映到本文松弛的某个下界（或反之）？
2. 在什么条件下两者给出**相同**的下界（即本文没有损失紧性）？
3. 若不 tight，两者的 gap 谁更小？

**为什么重要.** 论文的核心对比是"$p$-无关 vs $p$-线性"。若审稿人指出"$(10+p)$ 松弛在 $p$ 小时更紧、且 $p\le10$ 时成本也可接受"，本文的优势就只在 $p$ 大时成立。需要一个**量化**的对比，而不是"我们更省"。

**已尝试思路.**
1. 结构差异：SDPRLayers 的松弛把 $\mathbf X=\mathbf x\mathbf x^\top$ 的秩-1 约束丢掉，但保留 $\mathbf x$ 的完整二次结构（即 $\mathbf X$ 的每个元素自由）；本文的松弛同样丢掉秩-1，但**额外加了 Veronese 关系**。因此本文的松弛是"$(10+p)$ 松弛在 $\mathbf v_2$ 子空间上的投影 + 额外线性约束"，**不一定是子集也不一定是超集**。
2. $p=0$ 时：本文松弛 = Veronese + $\|\mathbf v_2\|^2=1$；SDPRLayers 松弛 = 一般 Shor。两者在 $p=0$ 的对比是干净的实验（同一个 QPEP 实例），可以直接跑。
3. 计划：在 T1 上取 $p\in\{8,16,64\}$，同时跑本文 $14\times14$ 与 SDPRLayers 风格 $(10+p)$ 松弛，比较 tightness 率与耗时。**这是 §5.3 baseline 1 的核心内容。**

**需要用户提供什么.**
- SDPRLayers 官方代码能否跑通（`utiasASRL/sdprlayer`，本仓库知识库记录过 28% 失败率与容差重试 patch，见 `kb/sdpr-qpep/00.md`）。若能，直接对比；若不能，需自实现 $(10+p)$ 松弛，工作量 +3 天。

**当前处置.** `[narrowed]` 到实验计划；理论等价性留 P2。

---

## Q10（P2）多个全局最优解时层的行为（与另两条线的交界） `[open]`

**精确陈述.** 当 (C-QPEP) 有两个全局最优 $(\mathbf q_1^\star,\mathbf t_1^\star)$、$(\mathbf q_2^\star,\mathbf t_2^\star)$ 且 $J_1^\star=J_2^\star$ 时：
1. ALM 会收敛到哪一个？是否取决于初始化/$\rho$ 调度？
2. 本文的 $14\times14$ 松弛此时是否仍 tight（$\mathbf X^\star$ 可能是两个秩-1 点的凸组合，秩 2）？
3. 此时解映射的导数不存在（跳变），本文的 FB 反向会输出什么？

**为什么重要.** 这是与 Prototype 1（Laplace-QPEPLayer，处理多峰）和 Prototype 3（Robust-QPEPLayer，处理外点）的**明确分界**。本文必须在论文里写清"多峰不在范围内"，否则会被要求与那两条线比较。

**已尝试思路.**
1. 猜想：多全局解时 $\mathrm{Shor}$ 会 tight 到 $\mathbf X^\star=\tfrac12(\mathbf x_1\mathbf x_1^\top+\mathbf x_2\mathbf x_2^\top)$，秩 2，$\lambda_2/\lambda_1=1$。此时 Lemma 2.2 的"秩-1 则认证"不触发，但 $\mathrm{Shor}=J^\star$ 仍成立（**这正是凸松弛的一个漂亮性质：多解时值仍对，只是解不唯一**）。这个猜想值得数值验证——若成立，本文可以声称"即使多峰，值也是对的，只是解不可辨识"，这是一个**强于**"不在范围内"的正面结果。
2. 实验：构造对称物体的两个等价位姿（例如立方体 90° 对称），跑本文层，看 $\lambda_2/\lambda_1$ 与 $\mathrm{Shor}$ 是否等于 $J^\star$。

**需要用户提供什么.**
- 是否要把"多峰时值仍正确"作为一个卖点写进去（我建议做，成本低、效果好，且天然与另两条线区分：它们解多峰，我们解约束，二者正交）。

**当前处置.** `[open]`，P2；draft §6 暂时写"不在范围内"。

---

## 优先级汇总

| 编号 | 一句话 | 优先级 | 状态 | 预计成本 |
|---|---|---|---|---|
| Q1 | 内层根数是否仍 40、模板能否复用 | **P0** | narrowed | 3–5 天（若需重生成模板 +1 周） |
| Q2 | 弱活跃点 FB 选的是哪个广义导数 | **P0** | partial（单约束已证为中点） | 多约束推广 1 周 |
| Q3 | GN 还是精确 Hessian | **P0** | narrowed | 决策 + 消融 3 天 |
| Q4 | Shor 紧性的充分条件 | P1 | partial | 实验 1 周；定理 2–3 周（不建议） |
| Q5 | $\varepsilon$ 的最优调度与误差界 | P1 | open | 启发式已有；严格界 1 周 |
| Q6 | 非刚体四脚桌的单侧导数 | P1 | open | 3–5 天 |
| Q7 | 约束类的边界（距离/避障） | P1 | narrowed | 写作 0.5 天 + 切平面近似实验 2 天 |
| Q8 | $\rho$ 增大时的数值退化 | P2 | open | 2 天纯实验 |
| Q9 | 与 SDPRLayers 松弛变量版的等价性 | P2 | narrowed | 实验 3 天 |
| Q10 | 多全局解时的行为 | P2 | open | 2 天（低成本高收益） |

## 给主线的三句话

1. **必须决策的两件事**：(a) 定理 2.3 是否接受"广义导数"的降级表述（Q2）；(b) 论文主定理用 GN 还是精确 Hessian（Q3）。这两条决定论文的理论定位。
2. **最高性价比的一步**：立刻做 Q2 的 2D 反例数值（`min ½(z-a)² s.t. z≥0` 在 $a=0$ 处），它能在 30 分钟内把 Lemma 2.5 从"论点"变成"例证"，并且是审稿人最可能要求的证据。
3. **最可能被拒的理由**：Q1（模板复用被证伪 ⇒ 前向常数变大，削弱"$p$-无关"的说服力）与 Q4（Shor 在真实数据上不紧 ⇒ "certifiable" 失效）。二者都已有 Plan B（重新生成模板 / 改成 fail-safe 叙事），不需要现在阻塞。
