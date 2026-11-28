# 三篇论文的数学问题汇总（2026-11-27）

> 把源文档（Feishu wiki，未随仓库分发）的三个 QPEP 可微位姿层 prototype 推进为三篇论文时，遇到并整理出的数学问题。每条给出：**问题、影响、已尝试、需要作者做什么、优先级**。细节与推导在各线的 `math_questions.md` / `theory/proofs.md`。
>
> 优先级：**P0 = 不解决就不能声称对应定理/主结果**；**P1 = 影响实验解释、审稿说服力或论文范围**；**P2 = 可后补/工程性**。

## 先看这 5 条（最需要作者判断）

| # | 问题 | 卡住什么 | 需要你 |
|-|-|-|-|
| **CC-1 [P0]** | 训练规模下有没有可用的**批量全局 QPEP 前向**？ | 三条线的端到端实验都依赖它 | 判断是否接受"推理全局、训练局部/多起点"的折中；是否有现成 GPU 实现 |
| **P1-Q1 [P0]** | Choi–Lam 反例能否由**真实测量**构造（10×10 Q 是否是合法 QPEP Gram 矩阵）？ | 定理 0 的"物理问题必然非紧"强度 | 给一个可实现的 PnP/配准实例，或同意把定理限定在"存在 PSD 代价矩阵"层面 |
| **P1-Q5 [P0]** | $\Delta J\to0$ 时多峰 Laplace 的**一致渐近**怎么处理？ | 定理 1.3 在分支交叉附近的有效性 | 指定可接受的近似（Airy/uniform Laplace、混合高斯、或只做 $\Delta J>\delta$ 的局部声明） |
| **P2-Q2 [P0]** | 弱活跃点上 FB 的 $\varepsilon\to0$ 极限是否是**可刻画的广义导数**？ | 定理 2.3 的"精确"措辞 | 判断用 Clarke/conservative-field 重新表述，还是限定在相容/非退化子类 |
| **P2-Q3 [P0]** | $\mathbf H_{\rm tan}$ 用精确 Hessian 还是 Gauss–Newton？ | 可逆性与"精确导数"只能二选一 | 确认主定理给 GN 版（可证）+ 精确版 Remark 的写法 |

## 0. 跨三条线的共同问题

### CC-1 [P0] 训练规模下的批量全局 QPEP 前向

**问题**：QPEP 的"全局"来自 Gröbner 基/作用矩阵一次枚举全部 KKT 根（LibQPEP 返回至多 40 个候选复根），目前是 CPU、单样本、毫秒级。三条线每个训练 step 都需要每样本一次全局求解：P2 的 ALM 外层 10–20 次、P3 的 GNC 10–20 次、P1 还要保留全部极小。以 10 万样本、每 epoch 10 次内层求解计，约 $10^6$ 次/epoch，CPU 串行不可行。

**候选路线**：
1. 把 40×40 作用矩阵的特征分解搬到 GPU 批量（每个样本一个小矩阵，天然可批）；
2. 同伦连续/多初值黎曼牛顿 + 代数/区间验证（不漏根）；
3. 训练用局部求解 + 热启动，推理用全局求解并检查 tightness/最优性；
4. 用 amortized 网络预测初值，再用 QPEP 做精化与验证。

**需要作者**：是否有现成的批量 QPEP/LibQPEP-GPU 实现？可接受的精度/时延/显存目标？如果只能做到"推理全局、训练局部"，三篇论文的实验 claim 要相应缩小。

### CC-2 [P1] 全局 argmin 的分支切换处可微性

P2/P3 的隐微分都在"当前全局解"处线性化。当 $\theta$ 穿过两个全局极小等值点（$\Delta J=0$）时，argmin 映射跳变，经典导数是 Dirac/不存在；P1 用 Laplace 平滑正面处理，P2/P3 没有。对 P2（对称物体、四点接触）和 P3（外点导致模式切换），这不是罕见事件。

**需要作者**：P2/P3 是把 claim 限定在"固定分支的邻域 + 报告切换频率"，还是也加一层平滑/混合？审稿人可能把"在切换点梯度错误"视为致命缺陷。

### CC-3 [P1] 平移可观测性 $\mathbf{G}(\mathbf{w})\succ0$ 的失效

QPEP 消去 $t$ 的前提是 $3\times3$ 信息矩阵 $\mathbf{G}\succ0$。P3 的权重 $w_i$ 可趋于 0（外点/退化），若某些方向没有有效观测，$\mathbf{G}(\mathbf{w})$ 奇异，$t^*$ 无界/不唯一；P2 的 $\mathbf{G}_{\mathcal A}=\mathbf{G}_0+\frac{\rho}{2}\sum b_jb_j^\top$ 在 $\mathbf{G}_0$ 本身奇异（纯旋转、共线点）时也失效。

**需要作者**：接受"加 ridge $\lambda I$ + 可观测性检查（QPEP 论文的 solvability/observability 判据）"作为工程折中，还是要求严格的退化处理与误差界？

### CC-4 [P1] 三篇论文"全局"的语义必须统一但不同

| 方法 | "全局"的来源 | 证书 | 失效模式 |
|-|-|-|-|
| SDPRLayers | Shor SDP 松弛 + 对偶证书 | $H\succeq0$、corank=1 | 松弛不紧 |
| QPEP | 枚举全部 KKT 根 + 比较 | **无**凸证书 | 漏根、重根、数值消元失败 |
| P2 | 内层 QPEP 全局 + 14×14 Shor 下界 | SDP 下界 + 可行性 | ALM 只到 KKT；下界不紧 |
| P3 | 每轮加权 QPEP 全局；GNC 轨迹 | 无（局部稳定点 + 二阶证书） | GNC 落局部极小 |

**需要作者**：确认三篇各自可接受的 claim 强度（"certified"只能给 P2 的 SDP 证书；P1 的"全局覆盖"依赖不漏根；P3 只能声明局部稳定点）。

### CC-5 [P2] baseline 计时与硬件公平性

SDPRLayers 实际是 13 维/24 约束、CPU Mosek；QPEP 是 4 变量但需全根求解。若声称"10×–50× 加速"，必须固定：单样本/批量、CPU/GPU、warm start、前向/反向、是否含数据搬运。否则审稿人会质疑比较不公平。

**需要作者**：指定一个可接受的计时协议（建议：单样本 CPU、批量 GPU 两组都报，并报 p50/p95）。

## 1. P1 `Laplace-QPEPLayer`

> 完整 15 条（Q1–Q15）见 `01-laplace-qpeplayer/math_questions.md`；下面按同一编号列 P0/P1 的前 11 条，Q12–Q15 见分线文档。

### P1-Q1 [P0] Choi–Lam 反例能否由真实测量实现

$M=M_{CL}+2I$ 是 PSD 的，但一般不是某个 QPEP 实例的缩减矩阵 $M=M_0-D^\top G^{-1}D$（需要 $M_0\succeq0$、$G\succ0$ 与 Schur 补结构）。若不可实现，定理 0.3 只能声称"存在 PSD 代价矩阵的旋转子问题反例"，不能直接说 SDPRLayers 在真实 PnP 上失效。已做：符号+秩证明 21 约束 = Veronese 二次理想；Choi–Lam 实例 $f^*=2$、SDP $=1.9658$、gap $1.71\%$、rank 6、对偶证书。需要你给可实现测量集，或同意限定 scope。**P0**。

### P1-Q2 [P0] 13×13 SDP 与 10×10 旋转子问题 SDP 的关系

SDPRLayers 实际解 $x=[h;\mathrm{vec}(C);t]\in\mathbb R^{13}$、24 条约束；定理 0.1–0.3 只对消去 $t$ 的 10×10 旋转子问题成立。$f^\star_{13}\overset{?}{=}f^\star_{10}$ 未证；若不等，论文只能把 10×10 当构造子问题/动机，不能声称"等价于实际实现"。需要：反例或聚合映射证明；或确认对比口径。**P0**。

### P1-Q3 [P0] QPEP 根数：27 还是 40

源文档写 $K\le27$，LibQPEP 返回至多 40 个候选复根；27 为奇数、与对跖 $\pm q$ 成对不相容。定理 1.2/1.3 依赖 A2（不漏根）。需要：27 的出处，或统一写"$K\le40$ 候选 + $K_{\min}$ 个实局部极小"，并做漏根/重根统计。**P0**。

### P1-Q4 [P1] 定理 1.2 的精确必要条件

源文档"$\Delta J>0$ + $H\succ0$ 是精确充要"不成立：$\Delta J=0$ 时仍可能存在光滑全局选择；$H$ 退化时也可能光滑。已给出反例并改为充分条件。需要确认论文写法（"sufficient certificate + uniqueness under A2/A3"）。**P1**。

### P1-Q5 [P0] $\Delta J\to0$ 时多峰 Laplace 的一致渐近

分离模式 + 独立高斯展开在 $\Delta J\sim\tau$、模式合并或 $H_k$ 退化时不一致（Gaussian 重叠、模式数改变、$1/\sqrt{\det H_k}$ 发散）。数值上远离交叉时误差 $O(\tau)$（斜率 1.05），交叉附近未验证。P1 的卖点正是交叉处平滑，需要指定：一致 Laplace/Airy、一维数值积分、或只声明 $\Delta J>\delta(\tau)$ + 精确 Boltzmann 后验光滑。**P0**。

### P1-Q6 [P1] $\nabla_\theta\log\det H_k$ 的完整公式

$dH_k/d\theta$ 含 $\partial_qH\cdot\partial q/\partial\theta$，即 $J$ 的三阶 $q$ 导数；实现与显存开销要算清。需要确认是否在论文中给显式公式/自动微分替代。**P1**。

### P1-Q7 [P1] Haar 测度与 Laplace 常数

$S^3$ 体积 $2\pi^2$、$SO(3)$ Haar $8\pi^2$、局部 $dR=8dq$（长度缩放 2 ⇒ 体积 8）、二重覆盖 $1/2$：$\int_{SO(3)}=4\int_{S^3}$，每对 $\pm q$ 的模式因子是 $8(2\pi\tau)^{3/2}$，不是源文档的 $(\pi\tau)^3$。常数不影响 $w_k$，但影响 $Z$/KL 的绝对值。需要统一约定。**P1**。

### P1-Q8 [P1] $\tau$ 的选取、可学习性与尺度不变性

$\tau$ 固定时 KL 梯度的 $1/\tau$ 因子只是缩放；$\tau$ 可学习时会影响模式权重与数值稳定。需要确认 $\tau$ 固定/学习/按数据尺度归一，并给出默认值。**P1**。

### P1-Q9 [P0] Hilbert 定理 ⇒ 正测度非紧：还缺哪一步

Hilbert 只保证存在非 SOS 的非负四次型；要推出"对可实现的 $M$ 存在正测度区域使 SDP 不紧"，还需要：(i) $M$ 在真实 Gram 锥内（或内点）；(ii) 间隙在扰动下保持（开性）；(iii) 值间隙 ⇒ $\operatorname{rank}(X^\star)>1$（严格互补）。目前只有 (ii) 的数值证据与 (iii) 的实例。**P0**。

### P1-Q10 [P1] GPU 批量根枚举的可行性

与 CC-1 同源；P1 还需要保留全部极小/全部 $\det H_k$，比只取 argmin 更贵。需要批量作用矩阵特征分解/同伦延拓的精度与失败率。**P1**。

### P1-Q11 [P1] 重根/近重根时 action matrix 特征分解的梯度

在 $\Delta J\to0$ 或 $H_k$ 近奇异时，特征值/特征向量导数不稳定；Laplace 权重与梯度会爆。需要重根处理（Jordan 块/伪逆）或退化为数值积分。**P1**。

> 另有 Q12（与 EPro-PnP 的方差/偏差公平对比，P2）、Q13（13 维联合 SDP 的理想类比，P1）、Q14（"零松弛间隙"措辞，P2）、Q15（对称物体群轨道平均，P1）见分线文档。

## 2. P2 `Constrained-QPEPLayer`

> 完整 10 条（Q1–Q10）见 `02-constrained-qpeplayer/math_questions.md`；这里列 P0/P1。

### P2-Q1 [P0] 内层 KKT 根数与 QPEP 消元模板复用

内层 $\mathbf M^{\rm red}_{\mathcal A}\neq\mathbf M_0$，KKT 是 4 个三次 + 1 个二次，Bézout 上界 162；QPEP 原模板是 40 根。通用根数是否仍为 40？若不是，249×249 消元模板不能复用，前向常数会变（但 $p$-独立性不变）。需要：sympy 数商环维数 + 200 组随机根数统计。**P0（实现）**。

### P2-Q2 [P0] 弱活跃点上 FB 选出的广义导数

$\varphi_0$ 在 $(\mu,g)=(0,0)$ 不可微；Clarke 次微分是集合，$\varepsilon>0$ 的正则化路径选出 $a=b=1$。1D 例子 $\min\tfrac12(z-a)^2$ s.t. $z\ge0$ 在 $a=0$ 给出 $\mathrm dz/\mathrm da\to1/2$（Clarke 中点），可作为示例；一般约束几何下是否仍是中点、$\varepsilon$ 与 $\mu,g$ 消失速率不同时如何选择，仍 open。需要：与 Bolte et al. 2021 的 conservative field 对齐，或接受"a selected generalized derivative"的降级表述。**P0**。

### P2-Q3 [P0] $\mathbf H_{\rm tan}$：精确 Hessian 还是 Gauss–Newton

精确版含 $\sum_j\mu_j\nabla^2g_j$，可能不定、$\mathbf H_{\rm eff}\succ0$ 不自动成立；GN 版丢掉它，可逆性有保证但算的是 GN 近似导数。源文档两者都想要。需要二选一；建议主定理 GN 版（可证）+ 精确版 Remark + 数值对比。**P0**。

### P2-Q4 [P1] 14×14 Shor 松弛紧性的结构条件

实例上 tight 到 $2.3\times10^{-11}$，但什么条件下必然 tight？需要结构性的充分条件或至少真实数据上的紧性统计；不紧时只能 fail-safe（报 uncertified）。**P1**。

### P2-Q5 [P1] $\varepsilon$ 的最优调度与误差界

$\kappa\asymp1/\varepsilon^2$；$\varepsilon$ 与 $\rho$、活跃集阈值 $\delta_{\rm act}$ 的联合调度缺严格误差界。启发式已有，需实验网格。**P1**。

### P2-Q6 [P1] 非刚体四脚桌的单侧方向导数

桌面位姿为参数时四脚桌相容（可微）；4 条腿高独立参数时 $\mathbf w\notin\operatorname{range}(\mathbf J_{\mathcal A})$，解映射不可微。需要给出单侧方向导数或明确排除。**P1**。

### P2-Q7 [P1] 约束类的边界

约束必须仿射于 $(v_2,t)$：正深度、平面非穿透、FoV、倾角锥可以；$\|Rp+t-c\|\ge d$ 这类距离/避障约束不行。论文必须写清边界，避免审稿人拿避障场景质疑。**P1**。

---

## 3. P3 `Robust-QPEPLayer`

> 完整 12 条（MQ-01–MQ-12）见 `03-robust-qpeplayer/math_questions.md`；这里列 P0/P1。

### P3-Q1 [P0] 6D 位姿层面的变号定理（MQ-01）

一维 Prop 3.4 已严格证明；6D 的 $\partial z/\partial y=-H_{\rm rob}^{-1}S_j$ 符号需 $H^{-1}$ 与 $S_j$ 的联合谱条件，不能由 $\lambda_\parallel<0$ 直接推出。需要：可证的特殊情形（$H$ 与 $S_j$ 可交换、标量平移子问题、$W=I$ 且残差对齐）或 6D 反例；否则论文写"1D 严格 + 6D 数值/GAP"。**P0**。

### P3-Q2 [P0] GNC 分岔的显式反例（MQ-02）

源文档"从任意严重外点污染中跳出局部极小"；实际只证了"内层全局、外层坐标稳定点"。需要 1D/3D 反例：GNC 收敛到 $F_{\rm rob}$ 的严格局部极小，而全局极小更低（$\mu$-同伦 fold）。**P0**。

### P3-Q3 [P0] $O(\|r\|^{-4})$ 的一致 $H_{\rm rob}^{-1}$ 界（MQ-03）

逐观测界依赖条件 (B2)；需要去掉第 $i$ 个观测后剩余问题二阶充分且 $\lambda_{\min}\ge\mu_0$ 的可观测性/非退化条件，否则常数可能爆炸。**P0**。

### P3-Q4 [P0] 相干外点下的 3D 崩溃点（MQ-04）

随机外点下 GNC 很强；但**相干**外点（单一刚体变换生成）可以让鲁棒全局最优本身错误。需要 3D 崩溃点/相图，明确"无 RANSAC"的适用边界。**P0**。

### P3-Q5 [P1] $\lambda_{\min}(H_{\rm rob})$ 与翻转概率（MQ-05）

二阶证书只说明局部稳定，不说明位姿正确；错误位姿也可 $H_{\rm rob}\succ0$。需要 Morse 理论/数值相图。**P1**。

### P3-Q6 [P1] PnP/重投影误差的 $\lambda_i$（MQ-06）

3D-3D 的 $\lambda_i=4u_i^\top W_ir_i$；PnP 重投影经齐次化后次数结构不同，需要重推或改用代数 PnP 误差。**P1**。

### P3-Q7 [P1] $W_i$ 的可辨识性与尺度（MQ-07）

学习各向异性 $W_i$ 时，尺度/规范自由度与鲁棒核 $c$ 耦合，可能不可辨识；需要参数化/归一化约定。**P1**。

## 4. 已解决 / 降级的问题（不必再问）

| 编号 | 已解决/已修正的事 | 证据 |
|-|-|-|
| R1 | 定理 0.1/0.2：$x_R=2Uv_2$、$U^\top U=I$、21 条约束秩 20 且**等于** $\nu_2(\mathbb{RP}^3)$ 的二次消没理想 | P1 `verify_theorem0.py`：符号精确 + 秩 20 + 维数 20/20 |
| R2 | 定理 1.1：Euler 乘子 $\lambda^*=4J$、$H_{\mathbb S^3}=E^\top(\nabla^2J-4JI)E$、对跖不变性 | P1 `verify_theorem11.py`：$H$ 对 FD 误差 $1.5\times10^{-7}$，$dq$ 对 FD $1.5\times10^{-7}$ |
| R3 | Laplace 误差是 $O(\tau)$ 而非精确；平移积分精确；KL 梯度按 $1/\tau$ 发散 | P1 数值：$\tau=0.2\to0.025$ 误差 $8.3\times10^{-2}\to9.3\times10^{-3}$，斜率 1.05 |
| R4 | P2 增广拉格朗日活跃集配方与常数项：只有**非活跃**约束贡献 $-\mu_j^2/(2\rho)$ | P2 `sanity_constraints_alm.py` C2：正确 $3.6\times10^{-15}$，错误常数 $5.6\times10^{-1}$ |
| R5 | P2 FB 极限：$\varepsilon>0$ 恒可逆；极限 = $H$-加权最小范数解；相容时=真导数；四脚桌（桌面位姿为参数）自动相容 | P2 `sanity_fb_saddle.py` F4/F5/F6 |
| R6 | P2 证书：14×14 Shor + Veronese 在 $p=12$ 实例 tight 到 $2.3\times10^{-11}$；$d(\mu^*)$ 的负 gap 不可用 | P2 `sanity_constraints_alm.py` C5 |
| R7 | P3 源文档 $H_{\rm rob}$ 缺 $\mathbb S^3$ 约束乘子项 $-\lambda_{\rm rob}\operatorname{diag}(I_3,0)$；补上后对 FD 误差从 25.9% 降到 $9.9\times10^{-8}$ | P3 `riemann_hessian_check.py` |
| R8 | P3 一维 sign-flip 严格成立；$O(\|r\|^{-4})$ 斜率实测 $-3.998$；$\lambda_i=4u_i^\top W_ir_i\neq4e_i$ | P3 `ift_1d_signflip.py` + `proofs.md` Prop 3.4 |
| R9 | QPEP "27 根"无出处；官方实现返回至多 40 个候选复根，且 27 为奇数与 $\pm q$ 成对不相容 | P1 README/Q4；`kb/sdpr-qpeep/03.md` §2.3 |



## 5. 建议的求助顺序

1. 先定 **CC-1**（实验可行性）与 **CC-4**（claim 语义），这两条决定三篇论文能写到什么程度；
2. 再定 **P1-Q1/P1-Q2** 与 **P2-Q1/P2-Q2/P3-Q1**（定理强度与适用范围）；
3. 其余 P2 级问题可在实现阶段边做边补。
