# 实验协议（Laplace-QPEPLayer）

> 状态：**设计冻结，尚未运行**。本文件中的一切数字都是“待测”，不是结果。每个结果表必须标注证据等级：`论文声称` / `独立复现` / `未验证`。不把源文档的预实验数字（$f^\star=1.98$、gap $=0.01773$、rank $=6$、$w_A=0.4467$ 等）当作本线结果；它们只能写成“作者预实验（待独立复现）”。

## 1. 要回答的实验问题

| # | 问题 | 对应主张 | 主要指标 |
|-|-|-|-|
| E1 | 零升维 + 代数枚举是否在真实位姿任务上达到/超过 SDPR/EPro-PnP 的精度？ | C1/C3/C5 | ADD(-S), ADD-S AUC, 旋转/平移误差 |
| E2 | $3\times3$ 黎曼反向是否与有限差分一致，且比 KKT/autograd 更快？ | C5 | 梯度相对误差、cos 相似度、反向 wall-clock |
| E3 | 多峰 Laplace 是否在分支交叉/对称物体上比硬 argmin 更稳？ | C7/C8 | 收敛速度、梯度方差、失败率 |
| E4 | Laplace 的偏差有多大？在 $\Delta J\to0$ 时如何失效？ | C7 | 与稠密求积的 NLL/权重误差、$\tau$ 扫描 |
| E5 | 根枚举在 GPU/CPU 批量下的吞吐与失败率？ | C9 | 前向/反向吞吐、漏根率、$K$ 分布 |
| E6 | 与 EPro-PnP 的方差/偏差是否被公平分离？ | C8 | MSE 分解、种子方差、runtime |

## 2. 数据集

### 2.1 LINEMOD / LINEMOD-Occlusion

- **用途**：对称/歧义 6-DoF 物体位姿；重点 `eggbox`、`glue`（对称），以及近平面帧。
- **输入**：RGB（+ 可选深度）；2D–3D 对应由前端/标注给出；BOP 格式。
- **划分**：按 BOP 官方 train/test；对称物体用 ADD-S，非对称用 ADD(-S)。
- **风险**：LINEMOD 的对称物体真值本身有歧义；评价时必须用 ADD-S 并报告“多解比例”。不要用单帧挑好的结果。

### 2.2 UTIAS In-The-Dark

- **用途**：矩阵加权双目视觉定位端到端特征学习；直接复现 SDPRLayers 的管线。
- **管线**：VGG16（ImageNet 预训练，截到 `conv5_3`）+ 两个 UNet 式解码器 → 关键点/描述子/得分 → dense matching → 双目三角化 → 矩阵权重 $W_k$ → 13 维 QCQP（baseline）或 QPEP 的 $M$（本线）。
- **数据规模（论文声称，未复现）**：100k 训练 / 20k 验证；测试 runs 2/11/16/17/23/28/35；39 条 runs、约 30 小时、260 m 闭环。
- **风险**：数据需从 ASRL 网站下载；训练/验证 id 从 `deep_learned_visual_features` 的 Google Drive 获取；`pose_sdp_block.py` 在 HEAD 缺失，baseline 需 pin 到 `ba66e408` 或用 `SDPPoseEstimator`。

### 2.3 合成 branch-crossing benchmark（本线自建）

- **构造**：两模式实例，参数 $\theta\in[-\epsilon,\epsilon]$ 穿过 $\theta_c=0$；已知两个分支的真值位姿、$J_1(\theta),J_2(\theta)$、$\Delta J(\theta)$、$H_k(\theta)$。
- **用途**：唯一能算**精确后验**（稠密网格/求积）的场景；用于分离 EPro-PnP 的 MC 方差与 Laplace 偏差。
- **指标**：$\partial q^\star/\partial\theta$ 的解析 vs 中心差分；$w_k$ vs 精确权重；NLL 校准；$\Delta J\to0$ 时的误差曲线。

## 3. Baselines

| Baseline | 版本/配置 | 说明 |
|-|-|-|
| SDPRLayers SDPR-IS | `utiasASRL/sdprlayer`，`diff_qcqp=True`，Mosek 或 SCS | 主 baseline；报告 tight/non-tight、corank、前后向时间 |
| SDPRLayers SDPR-SDP | `diff_qcqp=False` | 对照反向模式；非 tight 时的行为 |
| EPro-PnP | `tjiiv-cprg/EPro-PnP-v2`（CVPR 2022 / TPAMI 2024） | 概率 baseline；固定采样预算/种子，报告方差 |
| BPnP | `BoChenYS/BPnP` | 局部 IFT baseline；初始化用网络预测 |
| Theseus | `facebookresearch/theseus`，GN/LM | 局部李群优化 baseline；报告初始化敏感性 |
| SVD/EPnP | 经典闭式 | 传统非可微 baseline（仅前向精度/速度） |

**公平性约定**：
- 所有可微层使用**同一个前端/同一组权重/同一损失**；只替换位姿层。
- SDPR 必须显式检查 `rank(X)`/证书；非 tight 样本单独报告，不偷偷换成局部解。
- EPro-PnP 固定采样数与随机种子；报告 5 个种子的均值/方差。
- 所有方法报告前向+反向总时间；CPU/GPU 分开；Mosek license 与 SCS 分开。

## 4. 指标

### 4.1 精度

- **ADD(-S)**：非对称物体用 ADD；对称物体用 ADD-S。
- **ADD-S AUC**：阈值 0–0.1 m（BOP 口径）。
- **旋转误差**（deg）、**平移误差**（m）。
- **成功率**：误差 < 阈值（如 5°/5 cm）的样本比例。

### 4.2 全局性/紧性

- SDPR 的 `rank(X)`、$\lambda_1/\lambda_2$、对偶证书 $\lambda_{\min}(H)$、corank。
- 本线的 $K$（候选根）、$K_{\min}$（实局部极小）、漏根率（残差 > 阈值）、$H_k$ 最小特征值分布。
- “全局正确”比例：返回位姿与真值在容差内；对多解物体用“任一对称解正确”。

### 4.3 可微性

- 解析梯度 vs 中心差分：相对误差、cos 相似度；在 $\Delta J=0.1,0.01,0.001$ 三档分别报告。
- 梯度方差：EPro-PnP 跨种子；本线跨“根枚举扰动/数值初值”（应接近 0）。
- 训练曲线：达到目标精度所需 epoch/样本数；损失是否出现 spike（分支交叉）。

### 4.4 效率

- 前向/反向 wall-clock（ms/sample）、batch throughput（samples/s）、峰值显存。
- 根枚举占比；$3\times3$ 求解占比；Laplace 三阶导数占比。
- CPU vs GPU；LibQPEP vs 多初值 Newton。

## 5. 消融

| # | 消融 | 目的 |
|-|-|-|
| A1 | 硬 argmin vs Laplace（固定 $\tau$） vs 学习 $\tau$ | 平滑是否帮助收敛/歧义 |
| A2 | 均匀权重 vs $1/\sqrt{\det H_k}$ 权重 | 体积权重是否有用 |
| A3 | top-1 模式 vs 全部模式 | 多峰是否必要 |
| A4 | 解析 $3\times3$ vs autograd through solver | 验证“陷阱 B”的成本/不稳定 |
| A5 | $H_{\mathbb S^3}$ vs 欧氏 $\nabla^2J$（漏 $-4J$） | 验证黎曼项 |
| A6 | $E$ 基 vs 旋转向量基（因子 4） | 验证参数化约定 |
| A7 | LibQPEP vs 多初值 Newton | 根枚举的鲁棒性/速度 |
| A8 | SDPR 有/无冗余约束 | 复现 SDPR 的 tight 行为 |
| A9 | 对称物体：softmax vs 群轨道平均 | Q15 |

## 6. 算力预算（规划假设，不是承诺）

- **合成 benchmark**：CPU 即可；1–2 天。
- **LINEMOD 子集（eggbox/glue + 5 个非对称物体）**：1× RTX 4090/A100，1–2 周（含前端训练）。
- **In-The-Dark 全量**：1–2× A100，3–4 周；dense matching 是瓶颈（768×196,608）。
- **SDPR baseline**：CPU + Mosek；10.8 万次 SDP 求解在论文复现中约数小时–数天（取决于 solver）。
- **总计**：4–6 周 wall-clock（与 roadmap 的 8–12 周有重叠）。

## 7. 复现风险与缓解

| 风险 | 影响 | 缓解 |
|-|-|-|
| LibQPEP 构建（Eigen/LAPACK/CMake） | 前向不可用 | 先用 Python 多初值 Newton + 残差校验；LibQPEP 作为对照 |
| 根枚举漏根/重根 | “全局”与 Laplace 权重错 | 残差检查、$K$ 统计、对偶/Sturm 校验（Q3） |
| Mosek license | SDPR baseline 不可跑 | SCS 替代；报告 solver 差异 |
| In-The-Dark 下载/划分 | 无法复现论文数字 | 严格按官方 `download_datasets.sh` 与 test runs；记录 commit |
| `pose_sdp_block.py` 缺失 | baseline import 失败 | pin `ba66e408` 或用 `SDPPoseEstimator` |
| EPro-PnP 采样方差 | 对比不公平 | 固定种子/采样数；报告 5 seed 方差 |
| 我们的 Laplace 偏差 | 被误读为“更准” | 合成 benchmark 分离 bias/variance；报告 $\tau$ 扫描 |
| GPU 根枚举 | 训练慢 | CPU worker pool 先行；GPU action matrix 作为 P1 |
| 对称物体真值歧义 | 指标失真 | ADD-S + 多解比例 + 轨道平均（Q15） |

## 8. 结果表模板（待填）

> 所有表格先留空；填表时每格标注证据等级。禁止把源文档的预实验数字直接填入。

**表 1：LINEMOD 精度（均值 ± 标准差，5 seeds）**

| 方法 | ADD(-S) ↓ | ADD-S AUC ↑ | 旋转误差 ↓ | 平移误差 ↓ | 成功率 ↑ |
|-|-|-|-|-|-|
| SDPR-IS | TODO | TODO | TODO | TODO | TODO |
| SDPR-SDP | TODO | TODO | TODO | TODO | TODO |
| EPro-PnP | TODO | TODO | TODO | TODO | TODO |
| BPnP | TODO | TODO | TODO | TODO | TODO |
| Theseus | TODO | TODO | TODO | TODO | TODO |
| **Ours (det.)** | TODO | TODO | TODO | TODO | TODO |
| **Ours (Laplace)** | TODO | TODO | TODO | TODO | TODO |

**表 2：In-The-Dark 定位误差（论文口径：纵向/横向 m）**

| 方法 | 普通测试集 | 更难测试集 | 推理时间 | 备注 |
|-|-|-|-|-|
| SDPR-IS | TODO | TODO | TODO | tight 比例 |
| SDPR-SDP | TODO | TODO | TODO | |
| SVD | TODO | TODO | TODO | |
| **Ours** | TODO | TODO | TODO | |

**表 3：梯度正确性与效率（合成）**

| 方法 | 梯度相对误差 | cos | 反向 ms/sample | 方差（跨种子） |
|-|-|-|-|-|
| 解析 $3\times3$ | TODO | TODO | TODO | TODO |
| autograd | TODO | TODO | TODO | TODO |
| EPro-PnP | TODO | TODO | TODO | TODO |

**表 4：Laplace 偏差 vs $\tau$（合成，精确后验可算）**

| $\tau$ | $w_1$ 精确 | $w_1$ Laplace | NLL 误差 | 备注 |
|-|-|-|-|-|
| 0.2 | TODO | TODO | TODO | |
| 0.1 | TODO | TODO | TODO | |
| 0.05 | TODO | TODO | TODO | |
| 0.025 | TODO | TODO | TODO | |

## 9. 证据分级与报告约定

- **论文声称**：原论文/源文档的数字，未经本线复现。引用时写明“论文声称”。
- **独立复现**：本线跑出的数字，附脚本/配置/commit/种子。
- **未验证**：只有设计/推导，没有数值。
- 每个表/图必须有“证据等级”列或脚注；没有等级的表格视为未完成。
- 负结果也记录（例如 Laplace 在 $\Delta J\to0$ 时不如硬 argmin），不删。
