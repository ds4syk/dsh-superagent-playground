# Laplace-QPEPLayer — 零升维、多峰 Laplace 平滑的可微位姿层

> 一句话定位：把 QPEP 的“四元数二次型 + Gröbner 基枚举全部 KKT 根”接上 **3×3 黎曼隐微分**与**平移精确高斯边际化 + 旋转多峰 Laplace**，做成一个**不引入 SDP 升维、不做 Monte Carlo 采样、在分支交叉处用平滑后验代替硬 argmin** 的可微位姿层。

## 核心卖点（3 条）

1. **零升维的代数全局层**：优化变量只有单位四元数 $q\in\mathbb S^3$（4 维）+ 闭式平移 $t^*(q)$；约束只有 $q^\top q=1$，处处 LICQ。前向用 QPEP 枚举全部实 KKT 根，**不求解 SDP、不依赖松弛紧性**。代价矩阵无论多少对应点都被压成固定 $10\times10$ 的 $M(\theta)$。
2. **$3\times3$ 黎曼反向**：Euler 齐次性给出闭式乘子 $\lambda^*=4J$；切基 $E(q)$ 上黎曼 Hessian 为 $H_{\mathbb S^3}=E^\top(\nabla^2J-4J I)E$；局部灵敏度只解一个 $3\times3$ 正定系统，且对跖双覆盖 $q\leftrightarrow -q$ 严格不变。对比 SDPR-IS 的 KKT 行约简与 SDPR-SDP 的锥系统，反向维度从 $O(10^2)$ 降到常数 3。
3. **多峰 Laplace 平滑（非采样）**：平移方向**精确**高斯积分；旋转方向对 QPEP 枚举出的全部局部极小做二阶 Laplace，得到解析模式权重 $w_k\propto e^{-J_k/\tau}/\sqrt{\det H_k}$ 与解析梯度（含 $\nabla_\theta\log\det H_k$ 与权重协方差项）。$\tau\to0$ 时退化为确定性全局隐微分；$\Delta J\to0$ 时用体积权重混合竞争模式，而不是让硬 argmin 产生阶跃。

## 当前完成度

| 交付 | 状态 |
|-|-|
| `paper/draft.md` 英文论文草稿 | 结构完整（Title/Abstract/Intro/RW/Formulation/Method/Experiments/Discussion/Conclusion/References），实验节已填真实测量（12 张表，合成 6D + 1D + SDP + 真实 LINEMOD 双峰），无编造数字。**Q2/Q13 以修订版定理 P1-2 为准**（满 20D 13×13 ≡ 10×10 同值同秩；上游 17D 严格更松，6 点物理反例 $f^\star_{13,\text{up}}=3.59431373<5.41500000=f^\star_{10}=f^\star_{\text{QPEP}}$，rank 4 vs 1）。论文 §5.3.4 与 Discussion 已同步为修订版三段式（2026-11-27 论文轮次），不再有 “Q2 open / proof revision pending” 表述 |
| `theory/proofs.md` 定理 0.1/0.2/0.3、1.1、1.2、1.3 | 已证明 / 有明确 GAP / 数值验证；2026-11-27 P2 证明结案 Q5/Q4/Q8/Q11/Q15；Q1/Q9/Q3/Q6/Q7 已结案。**P1-Q2/Q13 已于 2026-11-27 以修订版定理 P1-2 结案（CLOSED）**：上游 `sdprlayer@62d8ac6` 的 24 条约束只张成 17 维（缺 3 条行正交），补齐后**满 20D** 13×13 与 10×10 同值同秩；上游 17D 恒有 $f^\star_{13,\text{up}}\le f^\star_{10,\text{full}}$ 且严格更松，**6 点物理 3D-3D 反例**给出 $f^\star_{13,\text{up}}=3.59431373 < f^\star_{10,\text{full}}=f^\star_{\text{QPEP}}=5.41500000$（rank 4 vs 1）。见 `experiments/sdp/results/six_point.md` 与 `experiments/sdp/results/sdp_check.md` |
| `theory/derivations.md` | U 构造、21 约束与 20 维理想、$E(q)$、$\lambda^*=4J$、$H_{\mathbb S^3}$、对跖不变性、Laplace 常数与误差阶、KL 梯度 |
| `verify/` 两个最小数值脚本 | 已运行并保存 JSON；定理 0 的“21 约束 = Veronese 二次理想”为符号+秩证明；Choi–Lam 构造实例有对偶证书 |
| `experiments/protocol.md` | 数据集、baseline、指标、消融、算力、复现风险 |
| `related_work.md` / `roadmap.md` / `math_questions.md` | 已完成 |

**证据分级**：本目录严格区分「论文声称」「独立推导/符号验证」「数值验证」「未验证」。源文档中的预实验数字（$f^*=1.98$、gap $=0.01773$、rank $=6$、$w_A=0.4467$ 等）一律标为「作者预实验（待独立复现）」，不当作本线结果。

## 文件导航

- [`paper/draft.md`](paper/draft.md)：英文论文草稿（可直接给合作者读技术主张）
- [`paper/contributions.md`](paper/contributions.md)：贡献清单、与 SDPRLayers/EPro-PnP 逐维对比、审稿人质疑与回应
- [`theory/proofs.md`](theory/proofs.md)：逐条定理、假设、证明/证明草图、与源文档差异、边界情形
- [`theory/derivations.md`](theory/derivations.md)：关键推导中间步骤
- [`experiments/protocol.md`](experiments/protocol.md)：实验设计（结果 TODO）
- [`related_work.md`](related_work.md)：真实文献对比表与每篇差异点
- [`roadmap.md`](roadmap.md)：4/8/12 周里程碑、风险、投稿目标
- [`math_questions.md`](math_questions.md)：未解决数学问题（含 P0/P1/P2）
- [`verify/`](verify/)：`verify_theorem0.py`、`verify_theorem11.py` + JSON 输出 + 一句话结论

## 最近更新（2026-11-27）

- 完成定理 0 的**作用域修正**：源文档的“单帧 $SO(3)$ 10×10 + 21 约束”与 SDPRLayers 实现（13 维 $[h;\mathrm{vec}(C);t]$ + 24 约束，发布版把后 18 条标为冗余）**不是同一个对象**；本线只对“消去 $t$ 后的旋转子问题 10×10 SDP”证明等价性，并把它标为**构造的旋转子问题 SDP**。
- 用 sympy 精确证明：$x_R=2Uv_2$、$U^\top U=I$；21 条约束秩 20；21 条张成空间**等于** $\nu_2(\mathbb R P^3)$ 的二次消没理想（秩+包含，不只用维数数数）。
- 用 cvxpy/SCS 独立构造 Choi–Lam + $2I$ 实例：$q$ 的真最小值 2（解析），10×10 SDP 值 1.9658119658，**对偶证书** $Z\succeq0$（$\lambda_{\min}=-1.8\times10^{-11}$），认证间隙 $0.034188$，$\mathrm{rank}(X^*)=6$，$\lambda_1/\lambda_2=1.451$。**这是本线独立构造，不是源文档那组 $0.01773$/rank 6/$1.561$ 的复现。**
- 修正定理 1.3 的常数与措辞：平移积分精确；旋转 Laplace 误差 $O(\tau)$（数值斜率 1.05），不是精确积分；$\tau\to0$ 只在后验均值/argmin 意义上退化为确定性隐微分，KL 损失本身按 $1/\tau$ 发散。
- 修正定理 1.2：唯一性间隙 $\Delta J>0$ + $H\succ0$ + $G\succ0$ 是**充分条件**，源文档的“精确充要”不成立（给出反例）。
- 修正 QPEP 根数：官方 LibQPEP 返回**至多 40 个候选复根**（KB 代码级记录），源文档“27”无出处且 27 为奇数、与对跖 $\pm$ 对称不相容；本文统一写 $K\le 40$ 候选、$K$ 个实局部极小，并把“枚举完整”列为假设。

## 下一步 3 条

> **2026-11-27 晚（phase-3）状态**：Q2/Q13 分层结案（满 20D 13×13 ≡ 10×10；上游 17D 严格更松；6 点物理反例 + 精确有理数对偶下界 3.594313732834，证明引用的 3.59430886 低于该下界 4.87e-6，36 变体不复现）；Q5b/Q5c 分层结案（纯实根代理必有非零跳变，幂次：权重 τ^{4/3}、跳变分数 τ^{−1/6}、ζ≪−1 权重 τ²/|α|；Scorer 6D 已在不变变量层面实现，位姿跳变 1.21e-2 vs 3.04e-1/1.07e-1，网格细化下 Scorer 跳变 ∝ d 而标准/正则化保持常数）。未做：真实图像精度、In-The-Dark、外部基线、40 根本地认证。

1. **P0 数学**：2026-11-27 补充证明已给出真实矩阵加权 3D-3D 的 Carathéodory 构造（N≤72，显式 68 点实例）与正测度开性；Q2/Q3/Q6/Q7 也已结案。**唯一剩余 P0 是 Q5 的梯度余项引理**（期望值部分已证，梯度仍是草图），1D 数值斜率 1.07 / relG 3.4e-3 支持。
2. **P0 实现**：把 LibQPEP 的 40 候选根枚举做成 GPU 批量（或 CPU 批量 + 代数校验），并测量 $K$ 个实极小、$\det H_k$、$w_k$ 的数值稳定性；先在 UTIAS In-The-Dark 的矩阵加权双目子问题上打通前向。
3. **P1 实验**：合成分支交叉主表 + 1D 精确后验（定理 P1-5 交叉/分离/Fold）+ 6D 消融（τ、logdet、top-K）已完成；LINEMOD 对称物体为负结果（未产生双峰），In-The-Dark 未做。SDPR-IS/EPro-PnP/BPnP/Theseus 对比仍未接。

## 与另外两条线的边界

- 本线**不处理不等式约束**（手性/非穿透/视锥）——那是 `02-constrained-qpeplayer` 的 Fischer–Burmeister/ALM 路线；本线的定理与实验都不依赖它。
- 本线**不处理外点鲁棒核的二阶修正**——那是 `03-robust-qpeplayer` 的 Black–Rangarajan/GNC 路线；本线的 $M(\theta)$ 可以是任意加权 Gram 矩阵，但不声称对粗大外点免疫。
- 本线只声称：**零升维 + 代数全局枚举 + $3\times3$ 隐微分 + 多峰 Laplace 平滑**这一组合，以及与 SDPRLayers/EPro-PnP 的差异化对比。
