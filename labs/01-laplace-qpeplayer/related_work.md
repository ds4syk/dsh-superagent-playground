# Related Work：对比表与逐篇差异

> 正式引用以原论文为准；仓库内 KB 只作为“代码级/精读证据”的旁证。所有链接在 2026-11-27 核对。未核实完整条目的邻近工作单列在 §4，不进入论文 References。

## 1. 总表

| 工作 | 发表 | 核心机制 | 全局性来源 | 升维？ | 采样？ | 可微？ | 与 Laplace-QPEPLayer 的关键差异 |
|-|-|-|-|-|-|-|-|
| **SDPRLayers** [1] | IEEE T-RO 2025；arXiv:2405.19309 | 齐次 QCQP → Shor SDP → rank-1 读解 + 对偶证书；KKT 隐微分 + 隐式选择（SDPR-IS） | 凸松弛 + 证书（tight 时） | 是（$13\times13$ PSD） | 否 | 是 | 我们零升维、无松弛、枚举全部 KKT 根；反向只解 $3\times3$；不依赖 tight。**注意**：我们的 10×10 定理只覆盖消去 $t$ 后的旋转子问题，不声称等价于其 13×13 实现 |
| **QPEP** [2] | IEEE T-RO 2022；DOI:10.1109/TRO.2022.3155880 | 四元数二次型 + 消去 $t$ + Gröbner 基枚举全部 KKT 根 + 协方差 SDP | 穷举 + 比较目标（无可验证证书） | 否（$q\in\mathbb S^3$） | 否 | **否** | QPEP 提供解与协方差，**不提供对参数的梯度**；我们补 $3\times3$ 隐微分、对跖不变性、多峰 Laplace 平滑。QPEP 的“全局”≠ SDPR 的“可验证全局” |
| **EPro-PnP** [3] | CVPR 2022（Best Student）；TPAMI 2024 | $SE(3)$ Boltzmann + AMIS 蒙特卡洛 + KL 损失 | 采样覆盖（无证书） | 否 | **是**（6D MC） | 是 | 我们在 $\mathbb S^3/\{\pm1\}$ 上定义后验，$t$ 精确边际化，旋转用枚举 + Laplace；无 MC 方差，但有 $O(\tau)$ 偏差/分岔失效 |
| **BPnP** [4] | CVPR 2020 | 局部 PnP 求解 + 欧氏 IFT | 局部（依赖初值） | 否 | 否 | 是 | 我们枚举全部根并用 Laplace 处理歧义；BPnP 在分支交叉处给错误分支的梯度 |
| **Theseus** [5] | NeurIPS 2022 | 李群 GN/LM + 隐式/直接损失最小化 | 局部 | 否 | 否 | 是 | Theseus 是通用局部 DNLS 库；我们针对四元数四次型做全局枚举 + 解析 Laplace |
| **sQPEP** [6] | 2025（待核） | 尺度化 QPEP 的全局解 | 穷举 | 否 | 否 | 否 | 邻接工作；本线未独立核对其完整条目/代码，只在 related work 提及，不作为对比 baseline |
| **dQP** / **POLI** / **Pyatov et al.** | 源文档提及 | 凸 QP 活跃集微分 / SDPR 点云协方差 / 鲁棒两视图 | 各自 | 部分 | 否 | 是 | **未独立核实**；不进入正式引用（见 §4） |

## 2. 逐篇差异点

### 2.1 SDPRLayers（Holmes, Dümbgen, Barfoot, T-RO 2025）

- **链接**：[arXiv:2405.19309](https://arxiv.org/abs/2405.19309)；DOI [10.1109/TRO.2025.3578228](https://doi.org/10.1109/TRO.2025.3578228)；代码 [`utiasASRL/sdprlayer`](https://github.com/utiasASRL/sdprlayer)。
- **它做什么**：把非凸 POP 写成齐次 QCQP，Shor 松弛成 SDP；tight 时 rank-1 解即全局解，并给对偶证书；反向对原 QCQP 的 KKT 做隐微分，用 ACQ/SOSC/隐式选择处理冗余约束，保证梯度是全局最优解的 Jacobian。
- **它的实际维度（代码级）**：$x=[h;\mathrm{vec}(C);t]\in\mathbb R^{13}$，24 条约束（6 正交 + 9 handedness + 9 范数）；发布版 `SDPPoseEstimator` 把后 18 条标为冗余。仓库 KB `kb/sdpr-forward-net/01.md` §3.1 有逐行记录。
- **差异 1（变量/约束）**：我们用 $q\in\mathbb S^3$（4 维，1 约束）+ 闭式 $t$；它用 13 维联合变量 + 24 条约束。
- **差异 2（全局性）**：它靠凸松弛 + 证书；我们靠枚举全部 KKT 根 + 比较目标。前者可独立验证但要求 tight；后者不要求 tight，但没有证书、依赖枚举完整。
- **差异 3（反向）**：它对 KKT 系统做行约简 + 隐式选择；我们对约化后的 4 维平稳方程做 $3\times3$ 黎曼隐微分（无冗余约束）。
- **差异 4（歧义）**：它在非 tight 时没有位姿；我们用 Laplace 后验混合模式。
- **差异 5（scope）**：我们的定理 0 只对 10×10 旋转子问题成立；**不声称**与它的 13×13 实现同值。这是审稿人最可能问的点，论文必须主动写清。
- **内部证据**：`kb/sdpr-qpep/01.md`（五个问题、紧性、复现）、`kb/sdpr-qpep/02.md`（Thm.2 证明链、三种反向）、`kb/sdpr-forward-net/01.md`（真实前向管线/数据）。这些是“读论文+读代码”的证据，不是独立复现。

### 2.2 QPEP（Wu et al., T-RO 2022）

- **链接**：DOI [10.1109/TRO.2022.3155880](https://doi.org/10.1109/TRO.2022.3155880)；HKUST 记录 [hdl.handle.net/1783.1/117007](https://hdl.handle.net/1783.1/117007)。作者：Jin Wu, Yu Zheng, Zhi Gao, Yi Jiang, Xiangcheng Hu, Yilong Zhu, Jianhao Jiao, Ming Liu；T-RO 38(5):3314–3335。
- **它做什么**：把 PnP/PnL/手眼/点到面/Stewart 正运动学等统一成“四元数二次型 + 单位约束”；消去 $t$；用 Gröbner 基枚举全部 KKT 根；按目标排序取最小；协方差用另一个凸 SDP（CSDP/SeDuMi）。
- **它的“全局”**：穷举 + 比较目标；**没有** SDPR 意义上的对偶证书；失效模式是漏根/重根/数值消元失败/$t(q)$ 奇异。
- **差异 1（梯度）**：QPEP 不提供对参数的梯度；我们的 $3\times3$ 隐微分是新增贡献。
- **差异 2（概率）**：QPEP 只给解 + 协方差；我们给 Boltzmann 后验、模式权重、Laplace KL 梯度。
- **差异 3（根数）**：官方 LibQPEP 返回至多 40 个候选复根（`sols` 4×40，仓库 KB `kb/sdpr-qpep/03.md` §2.3）；源文档“27”无出处，且 27 为奇数、与对跖对称不相容。论文统一写 $K\le40$ 候选。
- **差异 4（零间隙措辞）**：QPEP 不引入松弛，故“gap=0”是平凡的；我们写“no relaxation is introduced; global optimality conditional on exhaustive enumeration (A2)”，不写“certified”。
- **内部证据**：`kb/sdpr-qpep/03.md`（消去、Gröbner、可观测性、协方差、与 SDPR 关系、局限）。

### 2.3 EPro-PnP（Chen et al., CVPR 2022 / TPAMI 2024）

- **链接**：[arXiv:2203.13254](https://arxiv.org/abs/2203.13254)（CVPR 2022，Best Student Paper）；扩展版 [arXiv:2303.12787](https://arxiv.org/abs/2303.12787)（TPAMI 2024）；代码 [`tjiiv-cprg/EPro-PnP-v2`](https://github.com/tjiiv-cprg/EPro-PnP-v2)。作者：Hansheng Chen, Pichao Wang, Fan Wang, Wei Tian, Lu Xiong, Hao Li（CVPR 版顺序）。
- **它做什么**：把 PnP 残差解释为 $SE(3)$ 上的 Boltzmann 分布；用 AMIS 采样近似配分函数/期望；用 KL 损失端到端训练。
- **差异 1（支撑/积分）**：它在 $SE(3)$（6D）上采样；我们在 $\mathbb S^3/\{\pm1\}$ 上枚举 + 对 $t$ 精确高斯积分。
- **差异 2（方差/偏差）**：它渐近精确但高方差；我们确定但有 $O(\tau)$ Laplace 偏差，分岔处不一致。公平对比必须分离二者（见 `experiments/protocol.md` §7）。
- **差异 3（损失）**：我们给出 Laplace KL 的解析梯度，含 $\nabla\log\det H_k$（三阶 $q$ 导数）与权重协方差项；它用采样估计。
- **差异 4（失效）**：它可能漏掉远端模式；我们在分岔/重根处失效。两者互补。

### 2.4 BPnP（Chen et al., CVPR 2020）

- **链接**：[CVF Open Access](https://openaccess.thecvf.com/content_CVPR_2020/html/Chen_End-to-End_Learnable_Geometric_Vision_by_Backpropagating_PnP_Optimization_CVPR_2020_paper.html)；[arXiv:1909.06043](https://arxiv.org/abs/1909.06043)；代码 [`BoChenYS/BPnP`](https://github.com/BoChenYS/BPnP)。作者：Bo Chen, Alvaro Parra, Jiewei Cao, Nan Li, Tat-Jun Chin。
- **它做什么**：把 PnP 求解器当黑盒，用隐函数定理反传；可学习相机内参/外参/3D 结构。
- **差异**：BPnP 的 IFT 在**局部解**上做；分支交叉时全局 argmin 换支，局部解的梯度是错的。我们枚举全部根 + Laplace 平滑；确定性层仍用全局根的 $3\times3$ 隐微分。
- **差异（冗余约束）**：BPnP 不处理冗余约束；我们的约化问题只有 1 条约束，天然无冗余。

### 2.5 Theseus（Pineda et al., NeurIPS 2022）

- **链接**：[arXiv:2207.09442](https://arxiv.org/abs/2207.09442)；代码 [`facebookresearch/theseus`](https://github.com/facebookresearch/theseus)。
- **它做什么**：通用可微非线性最小二乘库；李群、稀疏求解、批处理、GPU、隐式微分/直接损失最小化。
- **差异**：Theseus 是**局部** DNLS（依赖初始化），不枚举全局模式、不给证书；我们针对四元数四次型做全局枚举 + 解析 Laplace。两者可以组合：Theseus 做局部精化，QPEP 做全局枚举。

## 3. 数学/优化背景文献（用于定理 0、1.3）

| 主题 | 文献 | 用途 |
|-|-|-|
| 非负 ≠ SOS | Choi & Lam 1977；Choi–Lam–Reznick 1980；Hilbert 1888；Blekherman 2006 | Choi–Lam 反例；Hilbert 定理的背景 |
| 矩/SOS 层级 | Lasserre 2001；Laurent 2009 | Veronese/Shor 松弛的定位；**不**把我们的 SDP 直接等同于二阶 Lasserre |
| Veronese 理想 | Harris 1992 | $\nu_2(\mathbb P^3)$ 的 $2\times2$ 子式生成元 |
| Laplace/一致渐近 | Bleistein–Handelsman 1986；Wong 1989 | $O(\tau)$ 误差；分岔处的 Airy 一致渐近 |
| 隐函数/优化 | Dontchev–Rockafellar 2009；Nocedal–Wright 2006 | 隐式选择、IFT、KKT |
| Haar 测度/双覆盖 | Chirikjian 2012 | $S^3\to SO(3)$ 的测度因子 |

## 4. 未核实/不进入正式引用的邻近工作

源文档的对比表提到以下条目；本线**没有**独立核实其完整 bibliographic 信息，故不写进 `paper/draft.md` 的 References，只在内部文档标注：

- **dQP**（Magoon et al., NeurIPS 2025?）：黑盒凸 QP 活跃集/软惩罚微分。与我们的非凸四元数问题不同；若要做 baseline，需先核实论文/代码。
- **POLI**（arXiv 2026?）：SDPRLayer 点云协方差学习。与 Robust 线更相关，不是本线主对比。
- **Pyatov et al.**（IROS 2024）：鲁棒两视图隐微分。与 Robust 线更相关。
- **sQPEP**（Xue, Zhu, Liu et al., 2025）：HKUST 记录显示 “SQPEP: Global Optimal Solutions to Scaled Quadratic Pose Estimation Problems”，2025；本线未读全文/未跑代码，故只在 §1 表里标“待核”，不作为 baseline。

> 处理原则：宁缺毋滥。正式引用只放已核对作者/ venue/ DOI 或 arXiv 的条目；不确定的写“待核”，不编造。

## 5. 内部证据路径（非正式引用）

- `kb/sdpr-qpep/01.md`：SDPR/QPEP 的五个问题、紧性、复现、对比。
- `kb/sdpr-qpep/02.md`：SDPR 精读（Thm.2、三种反向、紧性失效）。
- `kb/sdpr-qpep/03.md`：QPEP 精读（消去 $t$、Gröbner 40 根、可观测性、协方差、局限）。
- `kb/sdpr-forward-net/01.md`：SDPR 真实前向网络（VGG16 + 双 UNet）、13 维 QCQP、24 约束、In-The-Dark 数据。
- 证据分级沿用 KB 的“论文声称 / 代码核对 / 未验证”；不要把 KB 的精读说成“独立复现”。
