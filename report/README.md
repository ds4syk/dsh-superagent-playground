# SDPR 与 QPEP 研读：五个问题的回答

本目录是两篇 IEEE T-RO 论文的研读与复现材料：

1. **SDPRLayers**：C. Holmes, F. Dümbgen, T. D. Barfoot,
   *SDPRLayers: Certifiable Backpropagation Through Polynomial Optimization Problems in Robotics*,
   IEEE T-RO 41 (2025), DOI [10.1109/TRO.2025.3578228](https://doi.org/10.1109/TRO.2025.3578228),
   arXiv:2405.19309。
2. **QPEP**：J. Wu, Y. Zheng, Z. Gao, Y. Jiang, X. Hu, Y. Zhu, J. Jiao, M. Liu,
   *Quadratic Pose Estimation Problems: Globally Optimal Solutions,
   Solvability/Observability Analysis, and Uncertainty Description*,
   IEEE T-RO 38(5):3314–3335, 2022, DOI
   [10.1109/TRO.2022.3155880](https://doi.org/10.1109/TRO.2022.3155880)。

## 从哪读起

| 文档 | 内容 |
|---|---|
| [**answers.md**](answers.md) | **主文档**：五个问题的直接回答（Q1–Q5），含 QPEP 对照与复现摘要 |
| [sdprlayers_deep_dive.md](sdprlayers_deep_dive.md) | SDPRLayers 详细精读：公式、Thm.2 证明、22 篇延伸文献表格、复现细节 |
| [qpep_reading_notes.md](qpep_reading_notes.md) | QPEP 精读：四元数 QCQP、Gröbner 基全局解、协方差 SDP、与 SDPR 的互补关系 |
| [reproduction.md](reproduction.md) | 相机 baseline 标定仿真的复现报告（设置、双路线、Table V、失败修复、局限） |
| [../references/INDEX.md](../references/INDEX.md) | 22 篇延伸文献索引（标题/作者/期刊/DOI/用途） |
| [DOCX 知识库](../deliverables/README.md) | 00 总览 + 01–04 子页的 Word 版本（默认交付格式） |
| [知识库模板](../docs/KB_CONVENTION.md) | 00 总览 + NN 子页的结构、证据分级、Feishu 实现步骤 |

## 五个问题（速览）

1. **什么是可验证最优**：给解 + 可在多项式时间内独立检验的最优性证书。对 QCQP，证书是对偶
   矩阵 $H(\lambda)\succeq0$ 且 $Hx=0$。QPEP 是“枚举全部 KKT 根的代数全局解”，没有这种凸证书，
   严格说与 SDPR 的 certifiable 不同。
2. **SDPR 的目的/手段/效果/代价**：让可微优化层回传全局最优解的梯度（而不是伪局部极小解的）；
   齐次 QCQP → Shor SDP → rank-1 读解 + 对偶证书 → 复用证书隐式微分；论文标定误差 3.45e-4 m
   （随机初始化局部解 1.13e-1 m），真实定位误差降 3–4 倍；代价是每步解 SDP、CPU-only、
   必须 tight、常需冗余约束、依赖 Mosek/SCS。
3. **怎么微分/存在性**：对 KKT 条件全微分；有冗余约束时 $M$ 奇异，删去相关行得 $M_r$，
   用 SOSC 证明行满秩，用 Dontchev–Rockafellar 隐式选择定理得 $J=-PM_r^\dagger N$；
   再证任意局部选择的 $x$ 微分相同；$H\succeq0$ 说明这是全局最优解的 Jacobian。
4. **能解什么/紧性**：矩阵元素参数化的齐次二次等式 QCQP；对应位姿图/同步、配准、矩阵加权
   位姿与标定、手眼、PnP/PnL、Stewart 平台正运动学、GNSS 姿态、IK/接触规划/CBF/分割等。
   紧性不是普遍性质：估计类多有 tight 证据（常需冗余约束），非估计类多为不等式/离散/层级，
   不能直接进 SDPRLayer。QPEP 的问题族不需要 tight，但没有梯度。
5. **复现**：SDPRLayers §VI-B4 的 stereo baseline 标定已复现（50 runs × 20 poses）。
   核心结论一致：SDPR 与 GT 初始化局部解几乎重合，随机初始化局部解失败；10.8 万次 SDP 解
   100% rank-1、$H\succeq0$、corank=1（99.97%）；局部极小比例 59% vs 论文约 60%。
   绝对数字差 20–45%，已逐条归因；官方代码路径 28% 求解失败已用自适应重试降到 2%。

## 证据等级

| 等级 | 含义 | 例子 |
|---|---|---|
| **论文声称** | 论文正文/图表给出的数字或结论 | Table V 的 3.454e-4 m；Table III/IV 的 Jacobian |
| **独立复现** | 本仓库代码跑出、可追溯 | cleanroom Table V；tightness；Jacobian FD；局部极小 59% |
| **未验证** | 只读摘要/补充材料/代码，未独立跑数值 | QPEP 的数值实验、可解性/可观测性定理；SDPRLayers §VI-A/§VI-C |

正文中凡三者混用之处均已标注；不把“论文声称”冒充“独立复现”。

## 复现资产

- 代码：[`repro/cleanroom/`](../repro/cleanroom/)（独立实现，已验证）、
  [`repro/upstream/`](../repro/upstream/)（官方代码修复版，见 patches）。
- 结果：[`repro/results_clean/`](../repro/results_clean/)、
  [`repro/results_upstream/final/`](../repro/results_upstream/final/)。
- 环境：[`repro/ENV.md`](../repro/ENV.md)、[`repro/cleanroom/ENV_clean.md`](../repro/cleanroom/ENV_clean.md)。
- 一键自检：`python repro/cleanroom/smoke_test.py`（16/16）。
- 复现总报告：[reproduction.md](reproduction.md)。

## 说明

- 论文 PDF 与提取文本因版权不随本仓库分发；索引给 DOI/arXiv。
- QPEP 的 T-RO 正文在付费墙后，QPEP 笔记依据摘要 + 作者补充材料 + 官方 LibQPEP 代码，
  未伪造正文定理编号。
- 本仓库只包含研读/复现所需的最小资产；原始 113 MB 的逐 run 日志与 pkl 留在原实验机器。
