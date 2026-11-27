# labs：三条论文线（QPEP 可微位姿层）

> 2026-11-27 起，把源文档（Feishu wiki，未随仓库分发）里的三个 prototype 分别推进为**独立论文**。三条线并行推进、共享同一个 QPEP 代数内核与"SDPRLayers 紧性/升维"动机，但贡献声明、实验与目标会议互不依赖。

## 总览

| 论文线 | 目录 | 核心问题 | 一句话方法 | 主要对照 | 目标 venue |
|-|-|-|-|-|-|
| **P1 `Laplace-QPEPLayer`** | [`01-laplace-qpeplayer/`](01-laplace-qpeplayer/) | 位姿歧义/分支交叉处确定性 argmin 不可微；SDPR 升维且可能非紧 | 在 $\mathbb{S}^3$ 上枚举全部代数极小，用平移精确积分 + 多峰 Laplace 做无采样平滑 | SDPRLayers、EPro-PnP、QPEP | T-RO / CVPR / RSS |
| **P2 `Constrained-QPEPLayer`** | [`02-constrained-qpeplayer/`](02-constrained-qpeplayer/) | 物理不等式约束：SDP 侧升维、反向互补非光滑、冗余活跃导致 LICQ 失效 | 增广拉格朗日把 $p$ 个约束压回常数维 QPEP；Fischer–Burmeister 约化 $6\times6$ 隐微分 | SDPRLayers（松弛变量）、dQP、Theseus | RSS / ICRA / RA-L / NeurIPS |
| **P3 `Robust-QPEPLayer`** | [`03-robust-qpeplayer/`](03-robust-qpeplayer/) | 外点：鲁棒 SDP 随 $N$ 升维；IRLS 冻结权重丢二阶项、梯度方向出错 | Black–Rangarajan 对偶 + GNC 保持 $N$ 无关的 QPEP；不动点精确隐微分 | SDPRLayers+RANSAC、Theseus、BPnP、POLY/POLI | T-RO / ICRA / CVPR / IROS |

三条线共享的数学起点：

1. 任意 QPEP 型问题（PnP、PnL、矩阵加权双目、点到面配准、手眼标定及其加权和）都可以写成
   $J_\theta(\mathbf{q},\mathbf{t})=\mathbf{v}_2(\mathbf{q})^\top\mathbf{M}_0\mathbf{v}_2(\mathbf{q})+2\mathbf{t}^\top\mathbf{D}\mathbf{v}_2(\mathbf{q})+\mathbf{t}^\top\mathbf{G}\mathbf{t}$，其中 $\mathbf{v}_2(\mathbf{q})=\operatorname{svec}(\mathbf{q}\mathbf{q}^\top)\in\mathbb{R}^{10}$。
2. 消去平移后只剩 $\mathbb{S}^3$ 上的 4 变量问题，代价被压缩为固定的 $10\times10$ 矩阵 $\mathbf{M}=\mathbf{M}_0-\mathbf{D}^\top\mathbf{G}^{-1}\mathbf{D}$——**与观测数 $N$ 无关**。
3. 与 SDPRLayers 的桥：旋转齐次向量 $\mathbf{x}_R(\mathbf{q})=[1,\operatorname{vec}(\mathbf{R}(\mathbf{q}))^\top]^\top$ 与 $\mathbf{v}_2(\mathbf{q})$ 只差一个常数正交变换；21 个 $SO(3)$ 约束张成 20 维子空间，对应四元数四次型的二阶 Lasserre/SOS 松弛。Hilbert (1888) 的四元四次非负多项式 $\supsetneq$ SOS 给出**结构性非紧**风险，而 QPEP 在 $\mathbb{S}^3$ 上零松弛。详见 P1 的 `theory/proofs.md` 与 P2/P3 的动机章节。

## 当前状态（2026-11-27 初版）

| 线 | 英文论文 draft | 定理/证明 | 数值 sanity check | P0 问题 |
|-|-|-|-|-|
| P1 | `01-laplace-qpeplayer/paper/draft.md` | `theory/proofs.md` + `derivations.md` | 2 个脚本（定理 0、定理 1.1） | 5 |
| P2 | `02-constrained-qpeplayer/paper/draft.md` | `theory/proofs.md` | 2 个脚本（ALM/SDP、FB/鞍点） | 3 |
| P3 | `03-robust-qpeplayer/paper/draft.md` | `theory/proofs.md` + `derivations.md` | 2 个脚本（1D 变号、黎曼 Hessian） | 4 |
| 汇总 | — | — | — | **13**（含跨线 CC-1） |

**已完成**：三条线的英文论文初稿（结构完整、实验标 TODO）、定理/证明/GAP、实验协议、相关工作、路线图、本线数学问题、可复现的最小数值检查。  
**未完成**：真实数据实验、端到端训练、GPU 批量 QPEP 工程；源文档的预实验数字一律标"作者预实验（待独立复现）"，不当作本仓库结果。  
**证据分级**见下文；**数学问题**见 [`MATH_QUESTIONS.md`](MATH_QUESTIONS.md)（P0 13 / P1 16，另有各线 P2）。

## 三条线怎么区分（避免自我撞车）

| 维度 | P1 | P2 | P3 |
|-|-|-|-|
| 微分对象 | 多峰后验/配分函数（平滑） | 不等式 KKT / 互补系统（非光滑） | 鲁棒 M-估计不动点（重降权） |
| 反向核心 | $3\times3$ 黎曼 Hessian + Laplace 权重 | $(6+|\mathcal A|)$ FB 约化系统 | $6\times6$ 曲率修正 Hessian |
| "全局"来源 | 代数枚举全部极小（局部平滑，不宣称单点全局） | 内层 QPEP 全局 + 对偶下界证书 | 每轮加权 QPEP 全局；GNC 轨迹只保证稳定点 |
| 关键失败模式 | 模式重叠时 Laplace 失效 | 非凸 ALM 只到 KKT；$\epsilon\to0$ 条件数 | 固定点可能是鞍点；$\mathbf{H}_{\text{rob}}$ 不定 |
| 数据 | LINEMOD/Occlusion、UTIAS In-The-Dark | YCB-Video/BOP、KITTI/TUM/UTIAS | UTIAS In-The-Dark、3DMatch/KITTI |

## 目录约定

```
labs/
├── README.md                 # 本文件：总览、共享骨架、证据分级、Git 约定
├── MATH_QUESTIONS.md         # 三线数学问题汇总（P0/P1/P2），求助入口
├── 01-laplace-qpeplayer/
│   ├── README.md             # 项目卡：定位、完成度、导航、下一步
│   ├── paper/                # draft.md（英文正文）+ contributions.md
│   ├── theory/               # proofs.md（定理/证明/GAP）+ derivations.md
│   ├── experiments/          # protocol.md（数据/baseline/指标/预算）
│   ├── related_work.md
│   ├── roadmap.md
│   ├── math_questions.md     # 本线问题（汇总文件的来源）
│   └── verify/               # 可复现的最小数值 sanity check
├── 02-constrained-qpeplayer/ # 同上
└── 03-robust-qpeplayer/      # 同上
```

论文正文用英文（目标 venue 要求），项目文档、数学问题、路线图用中文。公式统一 LaTeX。

## 证据分级

沿用仓库既有知识库（`kb/sdpr-qpep/00.md`）的口径：

| 等级 | 含义 | 本工作区示例 |
|-|-|-|
| **已证明** | 本仓库给出完整推导/证明，可逐步核对 | P1 的 $3\times3$ 隐微分（待补全符号约定后） |
| **数值验证** | 本仓库 `verify/` 脚本可复现 | P3 的 $\lambda_\parallel$ 变号点、$O(\|r\|^{-4})$ 斜率 |
| **作者预实验** | 源文档记录、尚未在本仓库独立复现 | 三个 prototype 的预实验数字 |
| **待验证/假设** | 只有论证思路，缺证明或实验 | GNC 全局性、Laplace 模式重叠误差 |

**不把"作者预实验"写成"已复现"，不把"数值现象"写成"定理"。**

## Git 与复现约定

- 分支：`agent/three-paper-tracks-95244014fb3c`，一个 PR 汇总三条线。
- 每条线独立目录、独立提交（`docs(labs/p1): ...` 等），避免三条线的文件互相覆盖。
- `verify/` 脚本必须能从仓库根目录直接运行，并固定随机种子；输出小文件随仓库提交，大文件/权重不进 git。
- 不提交源文档原文、Feishu 私有链接或聊天记录；论文草稿是二次写作与推导，不是原文搬运。
- 复现环境：本机 `/home/syk/miniconda3/bin/python`（numpy/scipy/sympy/matplotlib 可用）。

## 数学问题与求助

- 每条线把未解决问题写进 `math_questions.md`：编号、精确陈述、影响哪条定理/实验、已尝试思路、需要用户提供什么、优先级。
- 汇总版 [`MATH_QUESTIONS.md`](MATH_QUESTIONS.md) 按"最影响论文成立性"排序；**P0 = 不解决就不能声称该定理/主结果**，P1 = 影响实验解释或审稿说服力，P2 = 可后补。
- 求助时优先请用户做三类判断：给证明思路/反例、判断某个声明是否值得保留、指定可接受的近似/工程折中。

## 阅读顺序

1. 本文件 → 2. 目标论文线的 `README.md` → 3. `paper/draft.md`（看故事与贡献）→ 4. `theory/proofs.md`（看假设与 GAP）→ 5. `math_questions.md` → 6. `experiments/protocol.md` 与 `roadmap.md`。

## 相关既有材料

- `kb/sdpr-qpep/01.md`–`03.md`：SDPRLayers / QPEP 精读（可验证最优、Thm.2 证明链、QPEP 消元与协方差）。
- `kb/sdpr-forward-net/01.md`：SDPRLayers 真实前向网络（VGG16 + 双 UNet、13 维 QCQP、In-The-Dark 数据划分）。
- `docs/visuals/`：本工作区的一页总览图与历史可视化。
