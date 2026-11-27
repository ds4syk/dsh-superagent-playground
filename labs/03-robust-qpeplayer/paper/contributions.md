# 贡献清单、对比表与审稿人质疑

> 本文件服务 `paper/draft.md`。目标：把“我们声称什么、证据在哪、边界在哪”写成可审稿的形式。数字一律不编；预实验数字标“作者预实验（待独立复现）”。

---

## 1. 贡献清单（Claim → Evidence）

| 编号 | 贡献 | 证据 | 状态 |
|---|---|---|---|
| **C1** | **精确 BR 对偶 + GNC-QPEP 前向**：GM 核的 $w^*=\rho_\mu'(e)$、对偶值 $=\rho_\mu(e)$；内层加权 QPEP 常数维数（4+3），$N$ 只进 $O(N)$ 累加 | Lemma 3.1、Theorem 3.1；`verify/ift_1d_signflip.py` §[0]（对偶值误差 $10^{-11}$） | **已证**；收敛部分条件性 |
| **C2** | **精确不动点隐微分 + 降权曲率修正**：GNC 不动点 $\Rightarrow$ 稳定点；$\Delta H_{\mathrm{redesc}}$ = 权重块 Schur 补；$6\times6$ 反向，$O(1)$ 对 GNC 迭代数 | Lemma 3.2、3.3、Theorem 3.2；`verify/riemann_hessian_check.py`（$H_{\mathrm{alt}}=H_{\mathrm{corr}}$ 到 $2.3\times10^{-15}$） | **已证** |
| **C3** | **S³ 约束乘子修正**：精确黎曼 Hessian 含 $-\lambda_{\mathrm{rob}}\operatorname{diag}(I_3,0)$；$\lambda_i=4\mathbf u_i^\top W_i\mathbf r_i\neq4e_i$ | Lemma 3.4、3.5；数值：源公式 vs FD 误差 $2.97\times10^{-1}$，修正 $1.46\times10^{-7}$；常数测试 $H_{\mathrm{doc}}=I_3$ vs $H_{\mathrm{corr}}=0$ | **已证 + 数值** |
| **C4** | **一维精确变号定理 + 存在性**：$\mathrm dt^*/\mathrm dy_j=\lambda_\parallel(e_j)/H_{\mathrm{rob}}$；$e_j>c^2/3$ 时真实导数负、naive 正 | Proposition 3.4、3.5；数值：精确 vs FD $1.2\times10^{-8}$，naive 误差 $137\%$，符号相反 | **已证 + 数值** |
| **C5** | **$O(\|\mathbf r_i\|^{-4})$ 逐观测梯度界**：常数显式；前向偏差 $O(\|\mathbf r_i\|^{-3})$；区分“梯度免疫”与“前向鲁棒” | Proposition 3.6；数值 log-log 斜率 $-3.9979$ | **已证（条件性）** |
| **C6** | **统一核框架**：GM/Cauchy/Welsch/TLS 的 BR 对偶表；同一前后向结构 | Lemma 3.1 备注、Appendix B | **已证**（TLS 非光滑边界另注） |
| **C7** | **实验协议与可复现计划**：In-The-Dark / 3DMatch / KITTI，SDPR+RANSAC / Theseus / BPnP / Pyatov / POLI 对比 | `experiments/protocol.md` | **设计完成，结果 TODO** |

**不声称**：全局鲁棒最优；可验证鲁棒性；6D 变号定理；GNC 输出 = 全局 argmin；任意外点下替代 RANSAC。

---

## 2. 与 SDPRLayers / Theseus / BPnP 的逐维对比

| 维度 | **Robust-QPEPLayer（本线）** | SDPRLayers [1] | Theseus [6] | BPnP [7] | Pyatov et al. [9] | POLI [10] |
|---|---|---|---|---|---|---|
| 前向问题 | 鲁棒 M-估计 $\sum\rho(e_i)$，GNC + 加权 QPEP | 齐次 QCQP → Shor SDP；tight 时全局 | 非线性最小二乘（局部 GN/LM） | PnP 局部求解 | 鲁棒两视图几何（隐层） | 点云局部几何（Gaussian/ellipsoid） |
| 全局性 | **内层加权 QPEP 代数全局**；外层 GNC 局部 | **凸证书全局**（tight 时） | 局部 | 局部 | 局部/隐式 | 前向几何估计，非位姿层 |
| 外点处理 | **BR 对偶 + GNC，无 RANSAC** | 无（真实数据靠 RANSAC） | 软核/局部核 | 无 | 鲁棒损失 + 学习权重 | 提供各向异性 $W_i$ |
| 变量维数 | $4+3$（切空间 6） | $O(n^2)$ SDP，$n$ 随问题 | 李群局部 | PnP 变量 | 两视图 | 每点 Gaussian |
| $N$ 依赖 | 每轮 $O(N)$ 累加；维数不随 $N$ | SDP 规模随变量数；鲁棒 SDP 升到 $(10+9N)$ | $O(N)$ 稀疏 | $O(N)$ | $O(N)$ | $O(N)$ |
| 反向 | **精确不动点 IFT，$6\times6$，$O(1)$ 对 GNC 迭代** | 隐式选择 IFT，KKT 系统；需 tight/ACQ/SOSC | 隐式/展开 | IFT | 隐式 | 非本文 |
| 二阶项 | **显式降权曲率修正 + 约束乘子** | 证书矩阵 $H\succeq0$ | 局部 GN Hessian | 局部 | 鲁棒损失隐式 | — |
| 可微性条件 | $H_{\mathrm{rob}}$ 可逆；$\lambda_{\min}>0$ 为严格局部极小 | tight + corank=1 + ACQ + SOSC | 局部非奇异 | 局部非奇异 | 局部非奇异 | — |
| 证书 | 二阶局部稳定性（非鲁棒证书） | 全局最优证书 | 无 | 无 | 无 | 无 |
| 失败模式 | 相干多数外点、$\mu$ 分岔、$H_{\mathrm{rob}}$ 奇异、QPEP 漏根 | 非 tight、冗余约束标错、CPU 慢 | 局部极小污染梯度 | 局部/歧义 | 两视图局限 | 几何估计误差 |
| 计算 | 前向 CPU/GPU 取决于 QPEP；反向 $6\times6$ | CPU SDP，慢 | GPU 友好 | 轻 | 轻 | 前端 |
| 代码/复现 | 本仓库 `labs/03-robust-qpeplayer/` | 官方 PyTorch + Mosek/SCS | 官方 PyTorch | 官方 | 官方 | 官方 |

**一句话差异**：
- 对 SDPRLayers：**不追求全局证书，追求鲁棒 + 常数维数 + 精确鲁棒梯度**；SDPR 的 RANSAC 是我们要去掉的非可微前端。
- 对 Theseus：**同样局部，但我们把 redescending 核的曲率修正显式写进反向**；Theseus 的接口把核作为代价项，反向是对该代价的隐式微分；它不把 $w$ 作为独立变量消去，因此不会显式出现我们的 Schur 补/约束乘子分解。
- 对 BPnP：**同样 IFT，但 BPnP 无外点机制**；我们处理 $w_i$ 的隐式敏感度。
- 对 Pyatov：**他们做两视图/基础矩阵，我们做 3D-3D 位姿 + S³ 约束 + 二阶理论**。
- 对 POLI：**互补**；POLI 的 per-point Gaussian 可作为我们的 $W_i$ 输入。

---

## 3. 审稿人可能的质疑与回应

### Q1. “GNC 会陷局部极小，你们凭什么叫鲁棒？”

**回应**：
1. 我们**不**声称 GNC 全局。Theorem 3.1(v) 明确：内层加权 QPEP 全局；外层 GNC 只保证坐标稳定点。
2. 我们给出两类失败模式：(a) 相干多数外点使 $F_{\mathrm{rob}}$ 的**全局最小**本身错误（Prop 3.7）；(b) $\mu$-同伦分岔（GAP-2）。
3. 实验必须报告外点率-成功率曲线和崩溃点，而不是只报一个 50% 外点的漂亮数字。
4. 与 SDPR+RANSAC 的公平对比：RANSAC 也有失败概率；我们报告两者的成功率/耗时/可微性，不宣称“全面替代”。

**如果审稿人要求全局鲁棒**：我们会指出“全局鲁棒”需要鲁棒 SDP/分支定界，代价是 $(10+9N)$ 升维；本文的定位是“常数维数 + 精确可微 + 明确的局部保证”，两者是不同 trade-off。

### Q2. “梯度免疫是否等于鲁棒？”

**回应**：不。Prop 3.6 是**逐观测**的 $O(\|\mathbf r_i\|^{-4})$ 界，条件是 $H_{\mathrm{rob}}^{-1}$ 一致有界。它不防止：
- $N$ 个外点同向累加；
- 相干外点使全局最小错误（Prop 3.7）；
- 边界半外点（$e\approx c^2$）的梯度最大（$\lambda_\parallel(c^2)=-1/2$）；
- $H_{\mathrm{rob}}$ 近奇异时常数爆炸。
前向偏差是 $O(\|\mathbf r\|^{-3})$，比梯度弱一阶；所以“梯度免疫”比“前向免疫”更强，但不能反推鲁棒性。论文中会把这句话写成 limitation，而不是卖点。

### Q3. “$H_{\mathrm{rob}}$ 的 $-\lambda_{\mathrm{rob}}$ 项是不是你们编的？”

**回应**：不是。它是 $\mathbb S^3$ 约束的拉格朗日乘子。给出三个证据：
1. 解析反例：$f=\frac12\|\mathbf q\|^2$ 在 $\mathbb S^3$ 上常数，正确 Hessian 为 0；不含 $-\lambda$ 的公式给 $I_3$。
2. 数值反例：$c=1.0$ 的 3D-3D，源公式 vs 测地线 FD 误差 29.7%，修正公式 1.46e-7；$\lambda_{\mathrm{rob}}=1.056$。
3. 文献：Absil et al. (2008) 的球面 Hessian 公式；这是标准结果。
同时说明：如果源文档的 $\nabla^2e_i$ 本来就指黎曼 Hessian，则公式正确，但必须明确符号；我们给出两种等价写法 (7.5)/(7.6)。

### Q4. “6D 的 cos$=-0.079$ 能推广吗？”

**回应**：不能直接推广。我们把它降级为数值例，并给出：
- 一维严格命题（Prop 3.4）+ 存在性（Prop 3.5）；
- 映射层面的严格变号（Theorem 3.3(c)）：$\partial\Phi/\partial y$ 的平行特征值在 $e=c^2/3$ 变号；
- 6D 位姿层面需要 $H_{\mathrm{rob}}^{-1}$ 与 $\partial\Phi/\partial y$ 的联合谱条件，标为 GAP-1/P1-3。
如果审稿人要求 6D 定理，我们会在 revision 中给可证的特殊情形（如 $H_{\mathrm{rob}}$ 与 $\partial\Phi/\partial y$ 共享特征基、或平移-only 子问题）。

### Q5. “为什么不直接用鲁棒 SDP / TEASER？”

**回应**：
- 鲁棒 SDP 的变量数随 $N$ 升到 $(10+9N)\times(10+9N)$（源文档/文献），无法放进训练循环；
- TEASER 主要是 rotation-only / 两阶段，且不是可微层；
- 我们的目标是“可微层”，不是“最强鲁棒求解器”。
- 如果审稿人只关心前向精度，我们承认专用鲁棒求解器可能更强；本文的贡献是**反向**与**常数维数**。

### Q6. “和 Pyatov et al. / JFB 有什么区别？”

**回应**：
- Pyatov：两视图基础矩阵，隐式鲁棒损失，学习匹配权重；没有 S³ 约束乘子、没有 3D-3D QPEP、没有 Schur 补/一维变号定理。
- JFB：用廉价近似替代逆 Jacobian；我们的“naive IRLS-StopGrad”正是一种 phantom gradient。我们证明它在 1D 可以**符号错误**，比“近似误差”更强；精确 IFT 是 $6\times6$，代价可接受，不需要 JFB 的近似。
- 若审稿人认为“精确 IFT 太贵”，我们报告 $6\times6$ 的实际耗时（TODO）。

### Q7. “二阶证书 ($\lambda_{\min}>0$) 有什么用？”

**回应**：
- 它是**局部稳定性/可微性**证书：$\lambda_{\min}>0$ 保证严格局部极小、IFT 分支保持；$\lambda_{\min}<0$ 说明是鞍点；$\lambda_{\min}\to0$ 说明分岔/梯度爆炸。
- 它**不是**鲁棒性证书：错误的相干外点位姿也可以是 $\lambda_{\min}>0$ 的严格局部极小。
- 实验把它作为 diagnostic（训练中监控），并报告它与梯度范数/失败率的相关性。

### Q8. “为什么用 GM，不用 TLS/Welsch/Cauchy？”

**回应**：
- GM 是主核：光滑、BR 对偶严格凸、$w^*$ 闭式、$\lambda_\parallel$ 闭式；适合做理论与可微性。
- 统一框架表给出 Cauchy/Welsch/TLS 的对偶；实验会做核消融。
- TLS 非光滑（$w$ 在 $e=a$ 不唯一），需要 GNC-TLS 调度；我们把它标为边界情形，不声称同一套光滑理论。

### Q9. “$\mu$ 调度/超参是不是调出来的？”

**回应**：
- $\mu_0\ge3\max e_i/c^2$ 有明确的“径向凸性”动机；
- $\gamma\in[1.2,2.0]$ 是 GNC 文献常用范围；
- 实验会报告 $\gamma,\mu_0,K$ 的敏感性，并与 GNC-TLS 的默认调度对比。
- 若审稿人要求自适应调度，这是 P2 工程项。

### Q10. “为什么不用 QPEP 的协方差做二阶信息？”

**回应**：
- QPEP 的协方差是**前向**不确定性（观测噪声），不是**反向**的 $\partial z/\partial\theta$；
- 我们的 $H_{\mathrm{rob}}$ 是鲁棒 M-估计的 Hessian，含 $\rho''$ 项；QPEP 的协方差对应 $\rho'=1$（最小二乘）情形；
- 两者可以互补：协方差给前向置信度，$H_{\mathrm{rob}}$ 给反向条件数。

### Q11. “计算复杂度/实时性？”

**回应**：
- 前向每轮 $O(N)$ 累加 + 固定维数代数求解；GNC 轮数 $O(\log(\mu_0)/\log\gamma)$；
- 反向 $O(N)$ 组装 + $O(6^3)$ 解，不展开 GNC；
- 真正的瓶颈是 QPEP 代数求解器的批处理/GPU；这是工程风险（protocol §5.7），不是理论复杂度。
- 与 SDPR 的公平计时：同硬件、同 batch、warm-up、同步、分开 forward/backward。

### Q12. “如果 $H_{\mathrm{rob}}$ 在训练中经常不定怎么办？”

**回应**：
- 这是预期现象：GNC 可能停在鞍点，或 $\mu$ 退火不够慢；
- 处理：监控 $\lambda_{\min}$，若 $<0$ 则加大 $\mu$ 的退火步数/减小学习率/用阻尼；若 $\approx0$ 则用伪逆或回退 FD；
- 实验报告“不定 Hessian 比例”，不隐藏。
- 这也是一个 research question：能否用 $H_{\mathrm{rob}}$ 的负特征值做外点/翻转检测（P1-2）。

---

## 4. 声明-证据-边界总表

| 声明 | 证据等级 | 边界 |
|---|---|---|
| BR 对偶精确 | 解析证明 + 数值 $10^{-11}$ | GM；其它核见 Table |
| GNC 前向收敛到坐标稳定点 | 解析证明（单调 + 紧性） | 需加权子问题唯一；不保证全局 |
| 内层加权 QPEP 全局 | 假设 A2（QPEP 代数全局） | 数值漏根/退化时失效 |
| GNC 不动点 $\Rightarrow$ $F_{\mathrm{rob}}$ 稳定 | 解析证明 | 需加权全局最小；否则只是加权稳定 |
| $H_{\mathrm{rob}}$ 公式含 $-\lambda$ | 解析证明 + FD 数值 | 需切空间基；环境空间谱不可直接用 |
| 一维变号 | 解析证明 + FD | 仅 1D；6D 需联合谱 |
| 变号构型存在 | 解析证明 | 两观测、大间隔 |
| $O(r^{-4})$ | 解析证明（条件 B2） | 需 $H^{-1}$ 一致有界；逐观测 |
| 相干多数外点崩溃 | 解析例子 | 1D；3D 需几何构造 |
| 端到端优于 SDPR+RANSAC | **未验证** | 实验 TODO |

---

## 5. 什么结果会证伪我们的核心声明？

1. **梯度**：在 95% 的随机 3D-3D 实例上，精确 IFT 与中心 FD 的余弦 $<0.99$，或修正后的 $H_{\mathrm{rob}}$ 与 FD 的相对误差 $>10^{-4}$ → 公式/实现有错。
2. **曲率修正**：去掉 $\Delta H_{\mathrm{redesc}}$ 后，端到端训练指标不变（在统计显著意义上）→ 修正的实际价值弱。
3. **约束项**：去掉 $-\lambda_{\mathrm{rob}}$ 后，$H_{\mathrm{rob}}$ 的谱和训练都不变 → 该项在实际配置中可忽略（但解析反例仍成立；需报告 $\lambda_{\mathrm{rob}}$ 的分布）。
4. **前向**：在 50% 随机外点下，GNC-QPEP 的成功率不高于 Theseus-GM/多起点 → 前向卖点弱。
5. **系统**：在 $N=1000$ 时，前向耗时比 SDPR+RANSAC 更差且不可批处理 → 系统卖点弱。
6. **端到端**：在 In-The-Dark 上，无 RANSAC 的鲁棒层显著差于 RANSAC+clean 训练 → “去掉 RANSAC”的实际价值弱。
7. **理论**：找到 6D 实例，精确 $\partial z/\partial y$ 与 naive 同号但 $\lambda_\parallel<0$（说明 6D 变号不成立）→ 我们只保留 1D/映射层面声明。

---

## 6. 与另外两条线的边界

- **Prototype 1（Laplace-QPEPLayer）**：用 $O(10)$ Veronese 同构、Hilbert 非紧性、Laplace 多峰；关注**全局/多峰/协方差**。我们不做同构、不做 Laplace、不声称全局。
- **Prototype 2（Constrained-QPEPLayer）**：不等式/互补/物理约束；关注**约束可行域**。我们不做不等式、不做 ALM/FB。
- **共享**：QPEP 作为内层求解器；可以组合（例如 Prototype 2 的约束 QPEP 作为我们的内层，或 Prototype 1 的多峰 Laplace 作为我们的初始化）。论文必须独立成篇，不依赖它们未发表的结果。
