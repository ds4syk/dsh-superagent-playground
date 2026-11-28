# 相关工作对比（Robust-QPEPLayer）

> 口径：`论文声称` / `独立复现` / `未验证`。本页只把“读过的原文/官方代码/仓库 KB”写成事实；不确定处标 `TODO`。正式引用用原论文，仓库 KB 只作为内部证据来源。

---

## 1. 总表

| 工作 | 出处（核实） | 核心机制 | 与我们的关键差异 | 证据等级 |
|---|---|---|---|---|
| **SDPRLayers** | Holmes, Dümbgen, Barfoot, IEEE T-RO 41:4120–4138, 2025；DOI [10.1109/TRO.2025.3578228](https://doi.org/10.1109/TRO.2025.3578228)；arXiv:[2405.19309](https://arxiv.org/abs/2405.19309) | 齐次 QCQP → Shor SDP；tight 时全局 + 对偶证书；隐式选择 IFT 反向 | 我们不做凸松弛/证书；我们做 BR 对偶 + GNC + 常数维数 QPEP + 精确鲁棒反向。SDPR 真实数据靠 RANSAC + clean samples（KB 已核对论文 §VI-C 与上游 `pipeline.py`），我们去掉 RANSAC | 论文声称 + 部分独立复现（KB `kb/sdpr-qpep/01.md` §5、`02.md` §6.3） |
| **QPEP** | Wu, Zheng, Gao, Jiang, Hu, Zhu, Jiao, Liu, IEEE T-RO 38(5):3314–3335, 2022；DOI [10.1109/TRO.2022.3155880](https://doi.org/10.1109/TRO.2022.3155880) | 四元数二次型 + 消去 $t$ + Gröbner 基枚举全部 KKT 根 + LM 精化；代数全局，无凸证书 | 我们用 QPEP 作为**内层加权求解器**；QPEP 本身无梯度/无鲁棒外层。我们把 GNC 的 $w$ 与反向 $6\times6$ 加上 | 论文声称（正文付费墙；KB 依据摘要+补充材料+LibQPEP 代码）；数值未独立复现（KB `03.md`） |
| **Theseus** | Pineda et al., NeurIPS 2022；arXiv:[2207.09442](https://arxiv.org/abs/2207.09442) | 李群 DNLS；GN/LM；隐式微分/直接损失最小化；软鲁棒核 | Theseus 是局部优化器，核通常作为软惩罚；不暴露 redescending 曲率修正/Schur 补。我们同样局部，但**显式推导并消去 $w$**，给出 $6\times6$ 精确反向 | 论文声称 |
| **BPnP** | Chen, Parra, Cao, Li, Chin, CVPR 2020；[CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Chen_End-to-End_Learnable_Geometric_Vision_by_Backpropagating_PnP_Optimization_CVPR_2020_paper.html) | PnP 局部求解 + 欧氏 IFT | BPnP 无外点机制；我们处理 $w_i$ 的隐式敏感度与 S³ 约束乘子 | 论文声称 |
| **EPro-PnP** | Chen et al., CVPR 2022；[CVF](https://www.openaccess.thecvf.com/content/CVPR2022/html/Chen_EPro-PnP_Generalized_End-to-End_Probabilistic_Perspective-N-Points_for_Monocular_Object_Pose_Estimation_CVPR_2022_paper.html) | $SE(3)$ Boltzmann 分布 + AMIS 采样 | 概率/采样路线，处理歧义；我们做 redescending M-估计 + 精确 IFT；两者可互补 | 论文声称 |
| **Pyatov et al.** | Pyatov, Koshelev, Lefkimmiatis, IROS 2024；arXiv:[2410.17983](https://arxiv.org/abs/2410.17983) | 鲁棒两视图几何作为隐层；学习匹配置信度权重；避免 BPTT | 他们做 fundamental matrix/两视图；我们做 3D-3D 位姿、S³ 约束、Schur 补/一维变号/$O(r^{-4})$。我们的“naive IRLS-StopGrad”与他们的隐式鲁棒损失有概念联系，但我们的贡献是二阶结构与符号定理 | 论文声称 |
| **POLI** | Lee, Kim, Shin, Kim, Oh, arXiv:[2605.10456](https://arxiv.org/abs/2605.10456), 2026 | 点云局部几何建模为统计流形（per-point Gaussian/ellipsoid），自监督估计 | **互补**：POLI 提供各向异性 $W_i$，我们提供鲁棒位姿层。不是竞争者；实验可把 POLI 的 ellipsoid 作为 $W_i$ 输入 | 论文声称（2026 arXiv；未复现） |
| **GNC** | Yang, Antonante, Tzoumas, Carlone, IEEE RA-L 5(2):1127–1134, 2020；DOI [10.1109/LRA.2020.2969178](https://doi.org/10.1109/LRA.2020.2969178) | 渐进非凸；GNC-GM/GNC-TLS；非最小求解器 + 外点拒绝 | 我们用 GNC 作为前向骨架；贡献是**可微层**与二阶理论，不是 GNC 本身 | 论文声称 |
| **TEASER** | Yang, Shi, Carlone, CVPR 2020；[CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Yang_TEASER_Fast_and_Accurate_3D_Registration_with_Extreme_Outlier_Rates_CVPR_2020_paper.html) | GNC-TLS + 旋转 SDP 证书 + 平移估计；极端外点率 | TEASER 是配准求解器，不是可微层；它处理 rotation-only/两阶段，我们做 6-DoF 位姿层 + 反向 | 论文声称 |
| **Black–Rangarajan** | Black & Rangarajan, IJCV 19(1):57–91, 1996；DOI [10.1007/BF00131148](https://doi.org/10.1007/BF00131148) | 鲁棒统计与 outlier process 的统一；对偶/共轭 | 我们用其 GM 对偶做前向桥；贡献是把它与 QPEP、GNC、隐微分结合，并给出 Schur 补解释 | 论文声称（经典） |
| **JFB** | Fung, Heaton, Li, Wu, He, Zhang, AAAI 2022；[AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/20619) | 用廉价近似替代隐层逆 Jacobian | 我们的 Naive IRLS-StopGrad 是同类 phantom gradient；我们证明 1D 可符号错误，精确 IFT 只需 $6\times6$ | 论文声称 |
| **RANSAC** | Fischler & Bolles, CACM 24(6):381–395, 1981 | 随机采样一致性 | 我们的目标是去掉不可微 RANSAC 前端；公平对比必须调 RANSAC 预算 | 经典 |

---

## 2. 逐篇差异点

### 2.1 与 SDPRLayers

**相同**：都把位姿估计作为可微层；都用隐函数定理反向；都关心全局/局部与梯度污染。

**不同**：
1. **前向**：SDPR 解 Shor SDP，变量 $O(n^2)$，tight 时全局；我们解 GNC 加权 QPEP，变量固定 $4+3$，内层代数全局但不 tight/无证书。
2. **外点**：SDPR 本身不处理外点；论文 §VI-C 在 In-The-Dark 上实际用 RANSAC + clean samples 训练（KB `kb/sdpr-forward-net/01.md` 第 235 行、`kb/sdpr-qpep/02.md` §6.3(d)）。我们的层直接吃带外点的匹配，无 RANSAC。
3. **反向**：SDPR 的隐式选择 IFT 处理冗余约束；我们的 IFT 处理 **redescending 曲率** + S³ 约束乘子。两者数学结构不同：SDPR 的困难是乘子不唯一，我们的困难是 $w$ 随 $z$ 变。
4. **保证**：SDPR 有全局最优证书（tight 时）；我们只有局部二阶证书。我们明确不声称全局。
5. **计算**：SDPR CPU SDP；我们前向取决于 QPEP（可 C++/GPU 批处理），反向 $6\times6$。

**可组合性**：可以把我们的 GNC 外层与 SDPR 的内层结合（每轮解一个鲁棒 SDP），得到“有证书的鲁棒层”，但代价是 $(10+9N)$ 升维；这是 future work，不是本文。

### 2.2 与 Theseus

**相同**：局部非线性最小二乘；李群/流形；隐式微分。

**不同**：
1. Theseus 的鲁棒核通常作为**软惩罚/标量权重**；它不把 $w_i$ 作为独立变量消去，也不显式给 $\rho''$ 的 Schur 补。
2. Theseus 的 Hessian 是 GN/LM 近似（PSD），不区分 $H_{\mathrm{IRLS}}$ 与 $H_{\mathrm{rob}}$；我们的 $H_{\mathrm{rob}}$ 可以不定，且我们给出 $-\lambda_{\mathrm{rob}}$ 项。
3. Theseus 支持 GPU 稀疏求解；我们的反向是稠密 $6\times6$，更适合小位姿层；前向 QPEP 的 GPU 化是工程风险。
4. Theseus 不处理“外点权重冻结”的梯度偏差；我们的 Prop 3.4 给出严格反例。

### 2.3 与 BPnP

**相同**：IFT 反向；把几何优化嵌入网络。

**不同**：
1. BPnP 是 PnP 专用、无鲁棒核；我们是 3D-3D（可扩展 PnP）+ 鲁棒 M-估计。
2. BPnP 的 KKT 是局部 PnP 的；我们的 $\Phi$ 含 $w=\rho'(e)$ 的隐式依赖，RHS 多一项。
3. BPnP 不讨论 S³ 约束乘子；我们的 $-\lambda_{\mathrm{rob}}$ 是核心修正。

### 2.4 与 Pyatov et al.

**相同**：鲁棒损失 + 隐式层；避免 BPTT；学习权重。

**不同**：
1. 问题：两视图基础矩阵 vs 3D-3D 位姿。
2. 数学：他们的隐式层是鲁棒损失拟合；我们显式 BR 对偶 + GNC + QPEP，给 $H_{\mathrm{rob}}$ 的闭式与 Schur 补。
3. 理论：我们给 1D 符号定理、$O(r^{-4})$ 界、S³ 乘子；他们没有这些。
4. 实验：他们户外/室内相机位姿；我们 In-The-Dark/3DMatch/KITTI。

### 2.5 与 POLI

**互补，不竞争**：
- POLI 学习 per-point Gaussian/ellipsoid，可作为 $W_i$；
- 我们把这个 $W_i$ 放进鲁棒位姿层；
- 实验可以设“POLI $W_i$ + 我们的层” vs “POLI $W_i$ + SDPRLayer”，检验各向异性权重与鲁棒层的交互。
- 注意：POLI 是 2026 arXiv，未复现；引用标“论文声称”。

### 2.6 与 QPEP

**相同**：用四元数 + 消去 $t$ + 代数全局求解。

**不同**：
1. QPEP 是**求解器**，无外层鲁棒、无梯度；我们是**层**，有 GNC 外层 + 精确反向。
2. QPEP 的全局性是“枚举全部 KKT 根”；我们的内层继承这一点，但外层 GNC 不继承全局性。
3. QPEP 的协方差是前向不确定性；我们的 $H_{\mathrm{rob}}$ 是反向鲁棒 Hessian。两者可互补。

### 2.7 与 BR / GNC / TEASER

**BR**：我们用它做“核 → 加权最小二乘”的精确桥；贡献是把 $w$ 消去后的 Schur 补与隐微分结合，而不是 BR 本身。
**GNC**：我们用它的退火骨架；贡献是可微性与二阶理论，不是 GNC 调度。
**TEASER**：它证明 GNC-TLS + 证书在极端外点下强；我们做 6-DoF 可微层，不做 rotation SDP 证书。若审稿人问“为什么不用 TEASER”，回答：TEASER 不是可微层，且其 rotation-only 结构与我们的 6-DoF QPEP 不同。

---

## 3. 引用核实与勘误

| 源文档写法 | 核实结果 | 处理 |
|---|---|---|
| “Black–Rangarajan duality (CVPR 2014)” | 原始文献是 **IJCV 19(1):57–91, 1996**，DOI 10.1007/BF00131148 | 正式引用用 IJCV 1996；如需 CVPR 2014 的二次文献，另找（TODO），不把 1996 标成 2014 |
| “GNC (Yang et al. CVPR 2020)” | GNC 期刊版是 **RA-L 5(2):1127–1134, 2020**；CVPR 2020 是 **TEASER**（同组，GNC-TLS 的应用） | 分别引用 [4] 与 [5] |
| “SDPRLayers T-RO 2025, arXiv:2405.19309v3” | 核实：T-RO 41:4120–4138, 2025；arXiv v3 2025-01-08 | 正确 |
| “QPEP T-RO 2022” | 核实：T-RO 38(5):3314–3335, 2022；DOI 10.1109/TRO.2022.3155880；作者 Jin Wu, Yu Zheng, Zhi Gao, Yi Jiang, Xiang-Cheng Hu, Yi-Long Zhu, Jian-Hao Jiao, Ming Liu | 正确 |
| “Pyatov et al. IROS 2024” | 核实：arXiv:2410.17983，IROS 2024 accepted | 正确 |
| “POLI 2026” | 核实：arXiv:2605.10456，2026-05-11，Point-to-Ellipsoid (POLI) | 正确；标“论文声称/未复现” |
| “Theseus NeurIPS 2022” | 核实：arXiv:2207.09442，NeurIPS 2022 | 正确 |
| “BPnP CVPR 2020” | 核实：CVPR 2020，End-to-End Learnable Geometric Vision by Backpropagating PnP Optimization | 正确 |
| “EPro-PnP CVPR 2022” | 核实：CVPR 2022 | 正确 |
| “JFB AAAI 2022” | 核实：AAAI 2022 | 正确 |

---

## 4. 仓库内部证据来源

| 主题 | 仓库路径 | 用途 |
|---|---|---|
| SDPRLayers 五问、SDPR 流程、复现 | `kb/sdpr-qpep/01.md` | Related work / baseline 细节 |
| SDPR 精读、Thm.2、紧性、§VI-C | `kb/sdpr-qpep/02.md` | SDPR 边界与 RANSAC 证据 |
| QPEP 精读、消去 $t$、Gröbner、协方差 | `kb/sdpr-qpep/03.md` | QPEP 作为内层求解器的依据 |
| In-The-Dark 前向网络/数据 | `kb/sdpr-forward-net/01.md` | 数据集与端到端设置 |
| 复现代码/结果 | `repro/cleanroom/`、`repro/results_clean/` | 复现风险与计时参考 |

> 注意：KB 的证据分级是“论文声称/独立复现/未验证”；本页沿用。不要把 KB 的复现数字当成我们的实验数字；我们的实验还没跑。

---

## 5. 结论

- 最直接的竞争者是 **Pyatov et al.**（鲁棒隐式层）与 **SDPRLayers**（可微全局层）；我们的差异是 3D-3D/S³、BR-GNC-QPEP、精确曲率修正、一维符号定理、$O(r^{-4})$ 界。
- **Theseus/BPnP** 是局部可微基线，不是鲁棒层；我们与它们的差异是 redescending 核的精确反向。
- **QPEP** 是内层求解器，不是竞争者；**POLI** 是前端互补。
- 引用勘误两处（BR 1996、GNC RA-L 2020），必须在论文中改，否则审稿人会抓。
