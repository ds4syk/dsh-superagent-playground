# references/ 文献索引（lit owner 维护）

> **分发说明**：本仓库不随附论文 PDF（版权原因）；下表文件名仅作索引，PDF 可通过
> DOI / arXiv 获取。原始实验机器上的 `references/` 目录含全部 PDF。


> 命名规则：`FirstAuthorYear_Keyword.pdf`。年份取正式发表年（在线优先见备注）。
> “用途”列对应最终报告的问题编号（Q1–Q5）；Q4 = 本文（SDPRLayer）能解什么形式的问题、对应机器人问题、紧性如何。
> 每篇均已用 `pdfinfo` 验证为有效 PDF、页数 > 4，并用 `pdftotext` 核对首页标题/作者确为目标论文。

## 一、本文

| 文件 | 标题 | 作者 | 年份 | 期刊/会议（卷期页/DOI） | arXiv | 主题标签 | 用途 |
|---|---|---|---|---|---|---|---|
| `Holmes2025_SDPRLayers.pdf` | SDPRLayers: Certifiable Backpropagation Through Polynomial Optimization Problems in Robotics | C. Holmes, F. Dümbgen, T. D. Barfoot | 2025 | IEEE T-RO, DOI 10.1109/TRO.2025.3578228 | 2405.19309 | 可微优化、SDP 松弛、隐式微分 | 全文主线 Q1–Q5 |

## 二、Q4 核心文献（TRO / IJRR 为主）

| 文件 | 标题 | 作者 | 年份 | 期刊/会议（卷期页/DOI） | arXiv | 主题标签 | 用途 |
|---|---|---|---|---|---|---|---|
| `Rosen2019_SESync.pdf` | SE-Sync: A Certifiably Correct Algorithm for Synchronization over the Special Euclidean Group | D. M. Rosen, L. Carlone, A. S. Bandeira, J. J. Leonard | 2019 | IJRR 38(2–3):95–125, DOI 10.1177/0278364918784361 | 1612.07386（扩展版） | SE(d) 同步、位姿图 SLAM、Shor 松弛、可认证正确 | Q1 定义、Q4 位姿图/SLAM 紧性、Q2 对照 |
| `Yang2021_TEASER.pdf` | TEASER: Fast and Certifiable Point Cloud Registration | H. Yang, J. Shi, L. Carlone | 2021 | IEEE T-RO 37(2):314–333, DOI 10.1109/TRO.2020.3033695 | 2001.07715 | TLS 鲁棒配准、旋转子问题 SDP、a posteriori 证书 | Q4 配准/outlier-robust 紧性与冗余约束 |
| `Yang2023_OutlierRobust.pdf` | Certifiably Optimal Outlier-Robust Geometric Perception: Semidefinite Relaxations and Scalable Global Optimization | H. Yang, L. Carlone | 2023 | IEEE TPAMI 45(3):2816–2834, DOI 10.1109/TPAMI.2022.3179463 | 2109.03349 | POP、Lasserre/稀疏 SDP、六类几何感知、STRIDE | Q4 非估计类（配准/物体位姿）与层次化松弛；补充（TPAMI） |
| `Dumbgen2024_AutoTighten.pdf` | Toward Globally Optimal State Estimation Using Automatically Tightened Semidefinite Relaxations | F. Dümbgen, C. Holmes, B. Agro, T. D. Barfoot | 2024 | IEEE T-RO 40:4338–4358, DOI 10.1109/TRO.2024.3454570 | 2308.05783 | 自动寻找冗余约束、RDG/rank tightness、range/stereo | Q4 紧性失效与冗余约束核心 |
| `Holmes2024_MatrixWeighted.pdf` | On Semidefinite Relaxations for Matrix-Weighted State-Estimation Problems in Robotics | C. Holmes, F. Dümbgen, T. D. Barfoot | 2024 | IEEE T-RO 40:4805–4824, DOI 10.1109/TRO.2024.3475220 | 2308.07275 | 矩阵加权代价、双目位姿、冗余约束、证书与 FIM | Q4 与 SDPR 实验 P6/P7 同型问题 |
| `Papalia2024_RangeAidedSLAM.pdf` | Certifiably Correct Range-Aided SLAM | A. Papalia, A. Fishberg, B. W. O’Neill, J. P. How, D. M. Rosen, J. J. Leonard | 2024 | IEEE T-RO 40:4265–4283, DOI 10.1109/TRO.2024.3454430 | 2302.11614 | RA-SLAM、QCQP→SDP、Riemannian Staircase、紧性参数研究 | Q4 紧性“经验非严格 tight”的典型案例 |
| `Talak2023_ObjectPose.pdf` | Certifiable Object Pose Estimation: Foundations, Learning Models, and Self-Training | R. Talak, L. Peng, L. Carlone | 2023 | IEEE T-RO 39(4):2805–2824, DOI 10.1109/TRO.2023.3271568 | 2206.11215 | ζ-correctness、可观测正确性/非退化证书、可微修正层 | Q1 定义谱系、Q4 证书型（非 tight SDP）对照 |
| `Barfoot2024_Cayley.pdf` | Certifiably optimal rotation and pose estimation based on the Cayley map | T. D. Barfoot, C. Holmes, F. Dümbgen | 2025（在线 2024） | IJRR 44:366–387, DOI 10.1177/02783649241269337 | —（HAL hal-04901876） | Cayley 噪声模型、QCQP→SDP、旋转/位姿/轨迹、冗余约束 | Q4 近年 IJRR、Shor 松弛默认不 tight 的证据 |

## 三、Q4 补充文献（RA-L / ICRA / CVPR / ECCV / CDC / RSS / 预印本；已标注）

| 文件 | 标题 | 作者 | 年份 | 期刊/会议（卷期页/DOI） | arXiv | 主题标签 | 用途 |
|---|---|---|---|---|---|---|---|
| `Dellaert2020_Shonan.pdf` | Shonan Rotation Averaging: Global Optimality by Surfing SO(p)^n | F. Dellaert, D. M. Rosen, J. Wu, R. Mahony, L. Carlone | 2020 | ECCV 2020, LNCS 12351:292–308, DOI 10.1007/978-3-030-58539-6_18 | 2008.02737 | 旋转平均、connection Laplacian、SDP 精确性定理 | Q4 旋转同步紧性（补充：ECCV） |
| `Dumbgen2023_RangeOnly.pdf` | Safe and Smooth: Certified Continuous-Time Range-Only Localization | F. Dümbgen, C. Holmes, T. D. Barfoot | 2023 | IEEE RA-L 8(2):1117–1124, DOI 10.1109/LRA.2022.3233232 | 2209.04266 | 连续时间 range-only、对偶证书、无需解 SDP 的认证 | Q4 “只认证不求解”的形态（补充：RA-L） |
| `Giamou2022_CertifiedIK.pdf` | Convex Iteration for Distance-Geometric Inverse Kinematics | M. Giamou, F. Marić, D. M. Rosen, V. Peretroukhin, N. Roy, I. Petrović, J. Kelly | 2022 | IEEE RA-L 7(2):1952–1959, DOI 10.1109/LRA.2022.3141763 | 2109.03374 | 距离几何 IK、QCQP 可行性、秩最小化/凸迭代、障碍约束 | Q4 非估计（IK）核心（补充：RA-L） |
| `Maric2020_IK_SOS.pdf` | Inverse Kinematics for Serial Kinematic Chains via Sum of Squares Optimization | F. Marić, M. Giamou, S. Khoubyarian, I. Petrović, J. Kelly | 2020 | ICRA 2020, pp. 7101–7107, DOI 10.1109/ICRA40945.2020.9196704 | 1909.09318 | 最近点 IK QCQP、Sparse-BSOS、强对偶的局部条件、不可行证书 | Q4 IK/控制（补：ICRA） |
| `Hu2019_MRF_Inference.pdf` | Accelerated Inference in Markov Random Fields via Smooth Riemannian Optimization | S. Hu, L. Carlone | 2019 | IEEE RA-L 4(2):1295–1302, DOI 10.1109/LRA.2019.2895124 | 1810.11689 | MRF MAP、SDP 松弛、图像分割、per-instance 次优界 | Q4 非估计（分割）核心（补充：RA-L） |
| `Iglesias2020_PointSetReg.pdf` | Global Optimality for Point Set Registration Using Semidefinite Programming | J. P. Iglesias, C. Olsson, F. Kahl | 2020 | CVPR 2020, pp. 8284–8292, DOI 10.1109/CVPR42600.2020.00831（CVF 开放） | — | 点集配准、SO(3)×R^3、对偶证书 M_Λ⪰0、噪声阈值定理 | Q4 配准紧性与证书（补充：CVPR） |
| `Goudar2024_RangeOnlyInit.pdf` | Optimal Initialization Strategies for Range-Only Trajectory Estimation | A. Goudar, F. Dümbgen, T. D. Barfoot, A. P. Schoellig | 2024 | IEEE RA-L 9(3):2160–2167（据本文 [34]；DOI 未核实） | 2309.09011 | range-only 轨迹估计初始化、SDP 候选解 | Q4 range-only 初始化与紧性（补充：RA-L） |
| `Wise2020_HandEye.pdf` | Certifiably Optimal Monocular Hand-Eye Calibration | E. Wise, M. Giamou, S. Khoubyarian, A. Grover, J. Kelly | 2020 | IEEE MFI 2020, pp. 271–278, DOI 10.1109/MFI49285.2020.9235219 | 2005.08298 | 手眼标定、齐次 QCQP→对偶 SDP、强对偶稳定性 | Q4 标定类（补充：MFI） |
| `Wu2025_TraceConstrained.pdf` | Certifiably Optimal Estimation and Calibration in Robotics via Trace-Constrained Semi-Definite Programming | L. Wu, R. Tron | 2025 | arXiv 预印本（投稿 ACC 2026，未正式发表） | 2509.23656 | trace-constrained SDP、低秩通道/rounding、PnP/手眼/双机器人标定 | Q4 标定与 rounding（补充：预印本，标注未核实） |
| `Kang2024_RobustCBF.pdf` | Verification and Synthesis of Robust Control Barrier Functions: Multilevel Polynomial Optimization and Semidefinite Relaxation | S. Kang, Y. Chen, H. Yang, M. Pavone | 2023 | IEEE CDC 2023, pp. 8215–8222, DOI 10.1109/CDC49753.2023.10383434 | 2303.10081 | 多层 POP、Lasserre/SOS、CBF 验证与综合、渐近全局收敛 | Q4 非估计（控制）核心（补充：CDC） |
| `Yang2025_ContactPlanning.pdf` | Global Contact-Rich Planning with Sparsity-Rich Semidefinite Relaxations | S. Kang, G. Liu, H. Yang | 2025 | RSS 2025, DOI 10.15607/RSS.2025.XXI.046 | 2502.02829 | contact-rich 规划、稀疏 Moment-SOS、可认证次优界 | Q4 非估计（接触/摩擦）核心（补充：RSS） |
| `Marcucci2023_GCS.pdf` | Motion Planning around Obstacles with Convex Optimization | T. Marcucci, M. Petersen, D. von Wrangel, R. Tedrake | 2023 | Science Robotics 8(84):eadf7843, DOI 10.1126/scirobotics.adf7843 | 2205.04422 | 图凸集（GCS）、凸优化运动规划、MICP 全局最优 | Q4 Science Robotics 的“凸松弛规划”对照（非 SDP 可认证） |
| `Rosen2021_Review.pdf` | Advances in Inference and Representation for Simultaneous Localization and Mapping | D. M. Rosen, K. J. Doherty, A. Terán Espinoza, J. J. Leonard | 2021 | Annual Review of Control, Robotics, and Autonomous Systems 4:215–242, DOI 10.1146/annurev-control-072720-082553 | 2103.05041 | SLAM 综述、“certifiably correct”定义、SE-Sync 谱系 | Q1 定义、Q4 综述背景（补充：综述） |

## 四、备注

- `paper/sdpr.txt` 是本文 `pdftotext -layout` 文本（Lead 提供，行号可引用）；本索引中的页码一律指 **PDF 页（1-based）**。
- 所有结论在 `notes/literature_Q4.md` 中给出 `[作者年份, 期刊, §/Thm/p]`，并指向本目录文件。
- 刻意**未**收录“只有搜索摘要、未读原文”的条目；不确定处标“未核实”。
- 未收录付费墙后无法获得开放版的 TRO/IJRR 文章；Q4 需要的 TRO/IJRR 覆盖已由上面 8 篇一手 TRO/IJRR + 1 篇 IJRR 扩展版满足。
