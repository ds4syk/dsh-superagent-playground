# 实验协议（Robust-QPEPLayer）

> 状态：**设计完成，结果 TODO**。本文件不包含任何实测数字；源文档的预实验数字只以“作者预实验（待独立复现）”出现。所有表格的 `TODO` 必须在真实运行后填写。

---

## 0. 目标与假设

| 编号 | 假设 | 对应实验 |
|---|---|---|
| H1 | 在 30–70% 随机外点下，GNC-QPEP 的前向成功率高于 SDPRLayers（无 RANSAC）、Theseus-GM、BPnP | E1/E2 |
| H2 | 精确 IFT 与中心 FD 的梯度余弦 $\ge0.99$；Naive IRLS-StopGrad 在边界外点上显著更差（余弦可负） | E3 |
| H3 | 去掉 $\Delta H_{\mathrm{redesc}}$ 或 $-\lambda_{\mathrm{rob}}$ 会降低梯度精度/端到端指标 | E3/E5 |
| H4 | 前向/反向耗时随 $N$ 近似线性；反向不随 GNC 迭代数增长 | E6 |
| H5 | 在 In-The-Dark 上，无 RANSAC 的鲁棒层可接近或超过 SDPR+RANSAC+clean 的定位精度 | E7 |
| H6 | 相干多数外点会使所有方法（包括我们的全局内层）崩溃；崩溃点可测 | E4 |

**可证伪**：若 H2 在 95% 随机实例上失败，或 H1 在 50% 外点下不成立，或 H5 显著差于 RANSAC 基线，则论文的核心声明需要降级。

---

## 1. 数据集

### 1.1 UTIAS In-The-Dark（真实双目定位）

| 项 | 设置 |
|---|---|
| 来源 | UTIAS Long-Term Localization and Mapping Dataset，In the Dark 子集：http://asrl.utias.utoronto.ca/datasets/2020-vtr-dataset/ |
| 规模 | 39 runs，约 30 小时，Grizzly + Bumblebee XB3（基线 0.24 m，512×384） |
| 训练/验证 | 100k / 20k 样本（与 SDPRLayers 相同；样本 id 来自 `deep_learned_visual_features` 的 `download_datasets.sh`） |
| 测试 | runs 2, 11, 16, 17, 23, 28, 35；另加“hard”测试集（源-目标位移 >0.5 m 或旋转 >4°） |
| 前端 | VGG16 + 双 UNet + 关键点/匹配/权重块（与 SDPRLayers 对齐）；**训练时去掉 RANSAC**，直接喂原始匹配 |
| 标签 | VT&R Multi-Experience Localization 相对位姿 |
| 预处理 | 双目校正、归一化、关键点采样；与 SDPR 仓库的 `pipeline.py` 对齐（KB `kb/sdpr-forward-net/01.md`） |

**注意**：In-The-Dark 的“外点”来自误匹配/遮挡/光照；不是人工注入。为了控制变量，另做“注入外点”的版本：在原始匹配中按比例替换为随机 3D 点，报告 0/10/30/50/70%。

### 1.2 3DMatch / KITTI Odometry（3D-3D 配准）

| 项 | 设置 |
|---|---|
| 3DMatch | 50 个室内场景；标准 train/test split；用 FCGF/ Predator 描述子 + 最近邻匹配，阈值 0.05 m |
| KITTI Odometry | sequences 00–10；LiDAR 点云配准；用 KISS-ICP/点云描述子生成对应 |
| 外点协议 | (a) 随机外点：对应替换为随机点；(b) 相干外点：外点由同一刚体变换生成；(c) 边界外点：残差控制在 $e\in[c^2/3,3c^2]$；(d) 混合 |
| 规模 | 每场景采样 200–1000 帧；每帧 $N\in\{50,200,1000\}$ |

### 1.3 合成 3D-3D（梯度 oracle）

- 随机 $\mathbf p_i\sim\mathcal N(0,I)$，$\mathbf y_i=R(\mathbf q_{\mathrm{gt}})\mathbf p_i+\mathbf t_{\mathrm{gt}}+\epsilon_i$；
- 外点按比例替换；$W_i$ 从 Wishart/对角 SPD 采样；
- 用于：FD 梯度、$H_{\mathrm{rob}}$ 校验、崩溃点曲线、$N$ 扫描。
- 固定种子；每个配置 ≥50 trials。

---

## 2. Baselines 与设置

| Baseline | 实现/来源 | 关键设置 |
|---|---|---|
| SDPRLayers | 官方 `utiasASRL/sdprlayer`（KB `repro/upstream/`） | 默认 SDPR-IS；Mosek/SCS；tightness 检查；CPU |
| SDPRLayers + RANSAC | 官方 pipeline 的 RANSAC 前端 | RANSAC 阈值/迭代/置信度在验证集上调；报告预算 |
| Theseus | 官方 `theseus`；GN/LM + GM/Cauchy | 真值初始化（GT）与随机初始化（RND）两种；迭代到收敛 |
| BPnP | 官方实现（CVPR 2020） | PnP 设置；无鲁棒核 |
| Pyatov et al. | 官方 `VladPyatov/ihls`（arXiv:2410.17983） | 两视图；在可比任务上报告 |
| POLI + SDPRLayer | POLI 官方（arXiv:2605.10456）+ SDPR | POLI 提供 $W_i$；SDPR 解位姿 |
| QPEP + GNC（无精确 backward） | LibQPEP + 我们的 GNC 外层 | 前向 only；用于隔离反向贡献 |
| Naive IRLS-StopGrad | 我们的实现 | 冻结 $w^*$，只回传 $\rho'\partial g/\partial\theta$ |
| Finite differences | 中心差分 | 小 $N$；多步长；作为梯度 oracle |

**公平性**：
- 所有方法同硬件、同 batch、同数据、同预处理；
- RANSAC 调参用验证集，不用测试集；
- Theseus 给 GT 初始化是“强 baseline”，给 RND 是“弱 baseline”，两者都报；
- SDPR 的 SCS 失败用自适应重试（KB 记录的 28% 失败率），并报告失败率；
- 计时：warm-up 10 次，同步，取 median/IQR；分开 forward/backward/端到端；包含 host-device 拷贝。

---

## 3. 指标

### 3.1 前向精度

- 旋转误差 $E_R=\arccos((\operatorname{tr}(R_{\mathrm{est}}^\top R_{\mathrm{gt}})-1)/2)$（度）；
- 平移误差 $E_t=\|\mathbf t_{\mathrm{est}}-\mathbf t_{\mathrm{gt}}\|_2$（米）；
- 成功率：$E_R<5^\circ$ 且 $E_t<0.1$ m（3D-3D）；In-The-Dark 用纵向/横向误差；
- AUC：阈值扫描；Recall@outlier rate。

### 3.2 外点鲁棒性

- 外点率-成功率曲线（0–90%）；
- 崩溃点：成功率首次 <90% 的外点率（线性插值）；
- 相干外点 vs 随机外点的对比；
- 边界外点比例 vs 成功率。

### 3.3 梯度正确性

- 相对误差：$\|\delta z_{\mathrm{IFT}}-\delta z_{\mathrm{FD}}\|/\|\delta z_{\mathrm{FD}}\|$；
- 余弦相似度：$\cos(\delta z_{\mathrm{IFT}},\delta z_{\mathrm{FD}})$；
- 分参数/分观测：$\mathbf q$、$\mathbf t$、每个 $\mathbf y_i$；
- 分箱：内点（$e<c^2/3$）、边界（$c^2/3\le e\le3c^2$）、粗大外点（$e>3c^2$）；
- FD 设置：中心差分，步长 $\{10^{-6},10^{-5},10^{-4},10^{-3}\}$；报告条件数与最优步长。

### 3.4 系统

- Forward：单帧/批 $B\in\{1,8,32\}$，$N\in\{50,200,1000\}$；CPU/GPU；QPEP 求解时间 vs 累加时间；
- Backward：$6\times6$ 解 + 组装；是否随 GNC 迭代数增长（应为 O(1)）；
- 峰值显存；端到端吞吐（samples/s）；
- SDPR 的 SDP 时间单独报告；RANSAC 时间单独报告。

### 3.5 二阶诊断

- $\lambda_{\min}(H_{\mathrm{rob}})$、$\kappa(H_{\mathrm{rob}})$、负特征值比例；
- $\lambda_{\mathrm{rob}}$ 的分布；
- 梯度范数 vs $\lambda_{\min}$ 的相关性；
- GNC 迭代数/内层次数与成功率的 relation。

---

## 4. 实验矩阵

| 编号 | 内容 | 数据 | 指标 | 状态 |
|---|---|---|---|---|
| E1 | 前向鲁棒性（随机外点） | 合成 + 3DMatch/KITTI | 成功率、RRE/RTE、AUC | TODO |
| E2 | 前向鲁棒性（相干/边界外点） | 合成 + 3DMatch | 成功率、崩溃点 | TODO |
| E3 | 梯度正确性（IFT vs FD vs naive） | 合成（小 $N$） | 相对误差、余弦、分箱 | TODO |
| E4 | 相干多数外点崩溃 | 合成 1D/3D | 全局最小切换点 | TODO |
| E5 | 消融：曲率项/约束项/核/调度 | 合成 + 3DMatch | 梯度 + 成功率 | TODO |
| E6 | 系统：$N$/batch/CPU-GPU | 合成 + 3DMatch | 时间、显存、吞吐 | TODO |
| E7 | 端到端 In-The-Dark（无 RANSAC） | In-The-Dark | 纵向/横向误差 | TODO |
| E8 | 与 SDPR+RANSAC / Theseus / BPnP / Pyatov / POLI 对比 | 全部 | 全部 | TODO |

---

## 5. 消融

| 消融 | 变量 | 预期（待验证） |
|---|---|---|
| 曲率修正 | $\Delta H_{\mathrm{redesc}}$ on/off | off 时边界外点梯度偏差大，端到端变差 |
| 约束项 | $-\lambda_{\mathrm{rob}}\operatorname{diag}(I_3,0)$ on/off | off 时 $H_{\mathrm{rob}}$ 谱偏移，$\lambda_{\min}$ 可能虚假为正 |
| 核 | GM/Cauchy/Welsch/TLS | GM 光滑；TLS 在极端外点更强但不可微边界 |
| $\mu$ 调度 | $\gamma\in\{1.2,1.4,1.7,2.0\}$，$\mu_0$ 倍数，$K$ | 太快的退火易落坏盆地；太慢耗时 |
| $W_i$ | $I$ / 对角 / 各向异性 / POLI | 各向异性提升精度，但尺度需固定 |
| $N$ | 20/50/200/1000 | 前向/反向近线性；QPEP 常数时间 |
| RANSAC | 训练/测试 on/off | 去掉 RANSAC 的端到端收益（核心卖点） |
| 初始化 | identity / previous / multi-start | GNC 对初始化敏感；multi-start 是强 baseline |

---

## 6. 算力预算（估算，待实测修正）

| 项 | 估计 | 说明 |
|---|---|---|
| In-The-Dark 端到端 1 run | 3–7 GPU-days | 30 epochs × 10k samples；VGG16+UNet；取决于 QPEP 是否 CPU |
| In-The-Dark 3 seeds | 2–3 GPU-weeks | 含 SDPR+RANSAC baseline |
| 3DMatch/KITTI 推理 | 1–2 GPU-days | 不含训练前端 |
| 合成 + 消融 | 2–4 GPU-days | 小 $N$，CPU 也可 |
| 总计 | ~4–6 GPU-weeks | 保守；先做 E3/E5 的小规模计时，再修正 |

**硬件假设**：单卡 ≥24 GB（RTX 4090/A100 级）；QPEP 若 CPU-only，需多核 + 异步流水线。Slurm 提交（实验室规范），日志落 `~/logs/`，不占登录节点。

---

## 7. 复现风险与缓解

| 风险 | 影响 | 缓解 |
|---|---|---|
| LibQPEP C++/Gröbner 构建（Eigen/LAPACK/CMake） | 前向无法跑 | 预先构建 Docker/conda 环境；或纯 Python 小规模实现 + C++ 生产 |
| QPEP 漏根/重根/数值失败 | 前向“全局”不成立 | 记录失败率；KKT 残差/多初值回退；报告失败样本 |
| SDPR 无 Mosek / SCS 失败 | baseline 不完整 | 自适应重试（KB 方案）；报告 strict/relaxed 两种 |
| Theseus `sksparse`/SuiteSparse | 环境装不上 | 用官方 wheel/conda；或改用 Python 局部 GN 实现并注明 |
| In-The-Dark 下载/对齐 | 端到端不可比 | 用官方 sample id + 校验位姿；固定预处理版本 |
| RANSAC 调参不公平 | 审稿人质疑 | 验证集调参；报告预算；开源配置 |
| FD 在 $e=c^2/3$ 非光滑 | 梯度 oracle 不可靠 | 多步长；避开切换点；报告条件数 |
| GNC 随机性 | 结果方差大 | 多 seed；报告成功率/IQR；固定初始化协议 |
| GPU 批量 QPEP 缺失 | 系统卖点弱 | 先报 CPU 时间 + 批处理流水线；GPU 求解器作为工程项 |
| 相干外点导致全方法崩溃 | “鲁棒”声明弱 | 明确报告崩溃点；不宣称任意鲁棒 |

---

## 8. 预实验数字（仅作假设，不是我们的结果）

源文档报告的作者预实验（**待独立复现**）：

| 项 | 源文档数字 | 我们的处理 |
|---|---|---|
| 50% 外点 3D-3D，GNC-QPEP | 0.69° / 0.0170 m | 作为 H1 的假设；E1 复现 |
| Vanilla QPEP / SDPR（无 RANSAC） | 72.05° / 0.6672 m | 作为失败 baseline 的假设；E1 复现 |
| 精确 IFT vs FD | 余弦 +1.00000000 | E3 复现 |
| Naive IRLS vs FD | 15.41% / 38.16% 相对误差 | E3 复现 |
| 边界外点 naive 余弦 | −0.079344 | E3 复现（6D 数值例） |
| $O(r^{-4})$ log-log 斜率 | −4.00 | E3/Prop 3.6 复现 |
| $\lambda_{\min}(H_{\mathrm{IRLS}})\to\lambda_{\min}(H_{\mathrm{rob}})$ | 24.66 → 17.96 | E5 复现；注意源公式缺 $-\lambda$ 项，数字可能变 |

> **禁止**：把这些数字写进论文的结果表或摘要，除非 E1/E3 独立复现并标注“独立复现”。

---

## 9. 成功/失败判据

**成功**：
- E3：精确 IFT 与 FD 的余弦中位数 $\ge0.99$，90% 分位 $\ge0.95$；naive 在边界外点显著更差（$p<0.01$）。
- E1：50% 随机外点下成功率 $\ge90\%$，且不低于最强 baseline 的 5 个百分点。
- E6：$N$ 从 50 到 1000，前向时间近似线性（$R^2>0.95$）；反向不随 GNC 迭代数增长。
- E7：In-The-Dark 纵向/横向误差不差于 SDPR+RANSAC 的 10%。

**失败/降级**：
- E3 失败 → 检查 $H_{\mathrm{rob}}$ 公式/实现；若仍失败，理论有错。
- E1 在 50% 外点下 <70% → 前向卖点降级为“中等外点”。
- E2 相干外点全崩 → 明确写“model breakdown”，不宣称鲁棒。
- E7 显著差 → 论文定位改为“可微性/理论”，不主打“替代 RANSAC”。

---

## 10. 产物与目录

```
labs/03-robust-qpeplayer/
  experiments/
    protocol.md          # 本文件
    results/             # TODO: 运行后生成
      e1_forward.csv
      e2_coherent.csv
      e3_gradient.csv
      e4_breakdown.csv
      e5_ablation.csv
      e6_systems.csv
      e7_endtoend.csv
    figures/             # TODO: 论文图
    configs/             # TODO: 超参/种子
```

**记录规范**：每个 run 保存 config、seed、git hash（由主线统一提交，本线不执行 git）、环境、耗时、失败原因；不覆盖旧结果；原始日志留在数据盘，仓库只放汇总 CSV/图。
