# Roadmap：Laplace-QPEPLayer

> 假设：1 名主要作者 + 1 名合作者，每周 3–4 个有效工作日；GPU 资源为组内 1–2 张（RTX 4090/A100 级）；Mosek license 可用（否则用 SCS，时间/紧性结论需重测）。所有 deadline 是**假设**，投稿前需按目标 venue 官网核对。

## 1. 目标 venue 与理由

**首选：IEEE Transactions on Robotics (T-RO)，rolling submission，目标 2027-02/03 投出。**

理由：
1. **贡献类型匹配**：本线的核心是“代数几何 + 可微优化层 + 机器人定位实验”，不是纯视觉 benchmark。T-RO 接受“理论 + 系统 + 真实机器人数据”的长文；SDPRLayers（T-RO 2025）与 QPEP（T-RO 2022）都在同一 venue，审稿人熟悉这套语言，对比直接。
2. **篇幅**：定理 0.1–0.3、1.1–1.3 的完整证明、13×13 vs 10×10 的 scope 讨论、Laplace 误差分析需要 12–14 页；T-RO 容得下，RSS/ICRA 的 6–8 页会迫使我们把关键 GAP 塞进附录。
3. **实验规模**：In-The-Dark 端到端训练 + LINEMOD 对称物体 + 合成 benchmark，T-RO 的审稿标准接受“多数据集 + 消融”，CVPR 更偏好单一大 benchmark 的 SOTA。

**备选**：
- **RSS 2028**（假设 deadline 2028-01 左右）：如果 T-RO 审稿周期过长，或我们想把“零升维 + Laplace”作为 system/algorithm 亮点快速发表。RSS 的 8 页需要砍掉部分证明，把完整版放 arXiv。
- **ICRA 2028**（假设 deadline 2027-09 左右）：更工程化，适合以 In-The-Dark/LINEMOD 实验为主体的版本。
- **NeurIPS 2027**（假设 deadline 2027-05 左右）：如果理论部分（Veronese 理想 + Laplace 一致渐近）做得足够深，可以投 learning/optimization track；但机器人实验不是该 venue 的强项。
- **CVPR 2027 已来不及**（假设 deadline 2026-11 已过）；CVPR 2028 作为远期备选。

**不推荐**：ICLR（优化层理论可以，但机器人定位实验不匹配）；IJRR（周期太长，除非有期刊级实验矩阵）。

## 2. 4 周里程碑（2026-11-27 → 2026-12-25）

| 周 | 目标 | 交付/判据 |
|-|-|-|
| W1 | 冻结数学 scope；把 verify 脚本扩成可复现的“定理 0 + 1.1 + Laplace 常数”三件套 | `verify/` 三个脚本全绿；JSON 有对偶证书；proofs 的 GAP 列表定稿 |
| W1–W2 | 前向：Python 多初值黎曼 Newton + 残差校验；对随机 $M$ 枚举全部实极小 | 与 LibQPEP 的 $K$ 对比；漏根率 < 1%；单样本 < 5 ms |
| W2–W3 | 反向：$3\times3$ 隐微分 + Laplace KL 的 $\nabla\log\det H_k$；`gradcheck` | 解析 vs FD < $10^{-6}$；Laplace 权重的梯度 vs FD < $10^{-5}$ |
| W3–W4 | 合成 branch-crossing benchmark；精确后验（稠密网格） | 偏差/方差分解图；$\Delta J=0.1/0.01/0.001$ 三档；确定“Laplace 何时失效”的边界 |
| W4 | 论文 draft v1（本仓库 draft + 合成实验） | 合作者可读；所有数字有证据等级；GAP 标注完整 |

**4 周判据**：如果 W2 的根枚举漏根率 > 5% 或反向 > $10^{-4}$，暂停实验、回到数学/实现（Q3/Q6）。

## 3. 8 周里程碑（→ 2027-01-22）

| 周 | 目标 | 交付/判据 |
|-|-|-|
| W5–W6 | LINEMOD 子集（eggbox/glue + 5 个非对称物体）：接前端、跑 SDPR-IS/EPro-PnP/BPnP/Theseus/本线 | 表 1 初版；ADD(-S)/ADD-S AUC；训练曲线；SDPR tight 比例 |
| W6–W7 | UTIAS In-The-Dark：复现 SDPR 管线，换成 QPEP 层 | 表 2 初版；普通/更难测试集；推理时间；夜间误差 |
| W7–W8 | 消融 A1–A9；$\tau$ 扫描；群轨道平均（Q15） | 消融表；Laplace 偏差 vs $\tau$；对称物体分析 |
| W8 | 论文 draft v2 + 内部 review | 所有 claim 有证据；GAP 降级/解决记录；参考文献核对 |

**8 周判据**：如果 In-The-Dark 的 dense matching 吞吐导致训练不可行，降级为“固定前端特征 + 只训练权重/浅层”，并在论文中明确“端到端”范围。

## 4. 12 周里程碑（→ 2027-02-19）

| 周 | 目标 | 交付/判据 |
|-|-|-|
| W9–W10 | 补实验：多种子、失败案例、non-tight 分析、GPU 批量根枚举 benchmark | 方差表；失败案例图；$K$ 分布；吞吐表 |
| W10–W11 | 理论收尾：Q2（13×13 vs 10×10）至少给反例或 scope 证明；Q5 一致渐近的适用范围 | proofs 更新；GAP 列表缩短；审稿人 Q&A 更新 |
| W11–W12 | 论文 v3 + 投稿包（cover letter、reproducibility appendix、代码/脚本） | T-RO 投稿；arXiv 预印本；`verify/` 可一键复现 |
| 缓冲 | 审稿意见预演（内部 mock review） | 至少 3 条“会被拒”的理由 + 回应 |

## 5. 风险登记

| 风险 | 概率 | 影响 | 缓解 | 触发条件 |
|-|-|-|-|-|
| Q2（13×13 vs 10×10）无法证 | 高 | 定理 0 的 scope 被审稿人攻击 | 论文明确写“rotation subproblem”；把 13×13 作为 future work；用 10×10 的构造实例作为“结构性风险”证据 | 审稿人问“这和 SDPRLayers 什么关系” |
| Q1（Choi–Lam 不可实现） | 中高 | “SDPRLayers 失效”主张降级 | 只写“constructed instance”；把可实现性列为 open problem；不写“SDPRLayers fails on PnP” | 审稿人问“这是真实 PnP 吗” |
| 根枚举漏根/重根 | 中 | “全局”与 Laplace 权重错 | 残差校验 + LibQPEP 对照 + 漏根率统计；A2 明写 | 漏根率 > 1% |
| Laplace 分岔失效 | 高 | “平滑可微”卖点被削弱 | 把“平滑”归给精确后验；Laplace 标 $O(\tau)$；合成实验量化失效边界；Q5 一致渐近作为 future work | $\Delta J<0.01$ 时梯度误差 > 10% |
| GPU 根枚举慢 | 中 | 训练时间超预算 | CPU worker pool 先行；GPU action matrix 作为 P1；必要时蒸馏/近似 | 单 batch > 1 s |
| In-The-Dark 复现失败 | 中 | 少一个真实数据集 | 先 LINEMOD + 合成；In-The-Dark 作为“尽力复现”，不阻塞投稿 | 数据/权重/划分无法对齐 |
| EPro-PnP 对比不公平 | 中 | 审稿人质疑 | 固定采样/种子；bias/variance 分离；报告 runtime | 审稿人问“方差低不是必然吗” |
| 时间不足 | 高 | 错过 T-RO 目标 | 4 周先出“理论 + 合成”版本；8 周出“LINEMOD”；12 周出“In-The-Dark + 投稿” | W8 未完成表 1/2 |

## 6. 投稿目标与 deadline 假设

| Venue | 假设 deadline | 适配度 | 备注 |
|-|-|-|-|
| **T-RO** | rolling；目标 2027-02/03 | ★★★★★ | 首选；12–14 页；理论 + 实验 |
| RSS 2028 | 假设 2028-01 | ★★★★ | 8 页；砍证明；arXiv 放完整版 |
| ICRA 2028 | 假设 2027-09 | ★★★ | 工程/实验版 |
| NeurIPS 2027 | 假设 2027-05 | ★★★ | 理论版；机器人实验弱 |
| CVPR 2028 | 假设 2027-11 | ★★ | 需要更强视觉 SOTA；本线不优先 |

> **deadline 是假设**，投稿前必须核对官网。若 T-RO 审稿周期 > 6 个月，考虑 RSS/ICRA 的“快速版 + T-RO 扩展版”策略（注意双投/扩展政策）。

## 7. 每周检查清单

- [ ] 本周新增/修改的 claim 是否有证据等级？
- [ ] 是否有新的 GAP 被写成“已证明”？
- [ ] `verify/` 是否仍可一键复现？
- [ ] 根枚举漏根率/Laplace 误差是否在阈值内？
- [ ] 实验表是否填了“证据等级”列？
- [ ] 参考文献是否只含已核实条目？
- [ ] 与 companion tracks 的边界是否仍清晰（不依赖未发表结果）？
