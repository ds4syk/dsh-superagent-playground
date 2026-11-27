# 路线图（4 / 8 / 12 周）

> 起始日期假设：2026-11-27。所有 deadline 是**假设**，投稿前必须到官网核对。目标：把 Prototype 3 从“数学草稿 + 最小数值 sanity check”推进到“可投稿的完整论文 + 代码 + 实验”。

---

## 0. 一句话策略

**先锁数学，再做小实验，最后打真实数据。** 不要在 QPEP 工程/GPU 批处理上过早消耗时间；先用 1D/小 $N$ 把定理和反向跑通，再用 3DMatch/KITTI 证明鲁棒，最后用 In-The-Dark 打 SDPR+RANSAC。

---

## 1. 目标 venue 与理由

| 目标 | 类型 | 理由 | 风险 |
|---|---|---|---|
| **IEEE RA-L**（首选，rolling） | 期刊（~6–8 页） | 机器人位姿估计 + 可微层 + 理论 + 系统；RA-L 对“方法+实验”友好；rolling 不押 deadline；审稿周期相对可控 | 需要完整真实数据实验；理论深度要求中等 |
| **IROS 2027**（次选） | 会议 | 机器人系统/方法；与 RA-L 主题契合；会议 deadline 明确，适合集中冲刺 | deadline 假设 ~2027-03；需查官网 |
| **IEEE T-RO**（stretch） | 期刊 | 理论 + 真实数据 + 长文；如果 6D 定理/崩溃点分析做出来，T-RO 更匹配 | 审稿周期长（6–12 月）；要求更强的理论/实验完整性 |
| **CVPR 2027**（不推荐） | 会议 | 若强调“学习特征 + 3D-3D 配准”可投；但本文核心是机器人几何/可微优化，不是视觉学习 | 需要 vision-centric 故事；deadline 假设 ~2026-11（可能已过） |
| **ICRA 2027** | 会议 | 机器人；但 deadline 假设 ~2026-09（已过） | 时间不匹配 |

**建议**：以 **RA-L** 为主，**IROS 2027** 为备选；如果 8 周时 6D 定理 + 真实数据都强，再考虑 T-RO。不要同时押多个 deadline。

**Deadline 假设（待官网核对）**：
- IROS 2027：~2027-03-01（论文）+ 2027-03-08（视频/附件）。
- RA-L：rolling，目标 2027-01 提交，争取 2027-05 前接收。
- T-RO：rolling，目标 2027-06 提交（如果走 stretch）。

---

## 2. 4 周：数学冻结 + 最小可运行层

### 里程碑 M1（第 1–2 周）

- [ ] **数学冻结**：确定 Theorem 3.2/3.3 的最终表述；在 `theory/proofs.md` 中把 GAP-1（6D 变号）、GAP-2（GNC 分岔）、GAP-3（一致 $H^{-1}$）标为“已证/条件/开放”，不再改口径。
- [ ] **修正 $H_{\mathrm{rob}}$**：把 $-\lambda_{\mathrm{rob}}$ 项写进实现与论文；用 `verify/riemann_hessian_check.py` 的 FD 校验作为单元测试。
- [ ] **1D/小 $N$ 反向**：PyTorch `autograd.Function` 实现 forward（GNC + 简化 QPEP/闭式 3D-3D）+ backward（$6\times6$）；`torch.autograd.gradcheck` 通过。
- [ ] **梯度三对照**：精确 IFT / naive IRLS-StopGrad / 中心 FD；在合成 3D-3D 上跑 50 trials，记录余弦/相对误差。

### 里程碑 M2（第 3–4 周）

- [ ] **QPEP 内层接口**：LibQPEP C++ 构建（Eigen/LAPACK/CMake）或纯 Python 小规模实现；跑通 $N=50/200$ 的加权 3D-3D；记录失败率/耗时。
- [ ] **GNC 调度**：实现 $\mu_0$/$\gamma$/$K$；在合成数据上扫 3 组参数；报告成功率/耗时。
- [ ] **小规模前向实验**：合成 3D-3D，30/50/70% 随机外点；对比 QPEP+GNC vs Theseus-GM vs 多起点 GN；出第一版曲线。
- [ ] **论文 v1**：把 `paper/draft.md` 的 Method/Theory 定稿；Experiments 只写设计。

**4 周交付**：可运行的 `RobustQPEPLayer`（小 $N$）、梯度三对照结果、M1/M2 的数值证据；论文 theory 部分可给合作者读。

---

## 3. 8 周：真实数据 + 消融

### 里程碑 M3（第 5–6 周）

- [ ] **3DMatch/KITTI**：数据加载、对应生成、外点协议（随机/相干/边界）；跑 E1/E2。
- [ ] **Baselines**：Theseus-GM、BPnP、SDPR（无 RANSAC）；SDPR+RANSAC 的调参与计时。
- [ ] **消融 E5**：曲率项 on/off、$-\lambda$ on/off、核、$W_i$、$N$。
- [ ] **崩溃点 E4**：1D + 3D 相干外点；报告 $\alpha^*$。

### 里程碑 M4（第 7–8 周）

- [ ] **In-The-Dark baseline 复现**：先复现 SDPR+RANSAC 的定位误差（与 KB 的复现口径对齐）；确认数据/前端一致。
- [ ] **无 RANSAC 训练**：我们的层直接吃原始匹配；跑 1–2 seeds，观察是否收敛。
- [ ] **系统 E6**：$N$/batch/CPU-GPU 时间；反向 $O(1)$ 验证。
- [ ] **论文 v2**：Experiments 填真实数字；Discussion 写失败模式；Related Work 定稿。

**8 周交付**：3DMatch/KITTI 的完整曲线 + 消融；In-The-Dark 的 1–2 seed 初步结果；论文 v2 可内部评审。

---

## 4. 12 周：端到端 + 投稿

### 里程碑 M5（第 9–10 周）

- [ ] **In-The-Dark 3 seeds**：完整训练；与 SDPR+RANSAC、Theseus、BPnP 对比；报告纵向/横向误差、耗时、显存。
- [ ] **公平计时**：同硬件、warm-up、同步、分开 forward/backward；RANSAC 预算公开。
- [ ] **失败分析**：哪些帧/外点类型失败；$\lambda_{\min}$ 与失败的相关性；相干外点的崩溃点。
- [ ] **图**： teaser（方法框图）、$H_{\mathrm{rob}}$ 谱、梯度余弦分箱、外点率-成功率、计时。

### 里程碑 M6（第 11–12 周）

- [ ] **论文定稿**：Abstract/Intro/Conclusion 重写；Limitations 不回避；引用核对（BR 1996、GNC RA-L 2020、SDPR T-RO 2025、QPEP T-RO 2022、Pyatov IROS 2024、POLI 2026）。
- [ ] **开源**：代码/配置/种子/结果 CSV；QPEP 构建说明；复现脚本。
- [ ] **内部 review**：请 1–2 位不熟悉本线的人按 `paper/contributions.md` 的 Q&A 挑刺；修 6D/GNC 表述。
- [ ] **投稿**：RA-L 或 IROS 2027；准备 cover letter、摘要图、补充材料。

**12 周交付**：可投稿的 PDF + 代码 + 实验结果；如果 RA-L，目标 2027-01 提交。

---

## 5. 关键风险与应对

| 风险 | 概率 | 影响 | 应对 |
|---|---|---|---|
| QPEP 无法 GPU 批处理/构建困难 | 高 | 前向慢，系统卖点弱 | 先 CPU 多核 + 异步；纯 Python 小规模；把 GPU 作为工程项，不阻塞论文 |
| GNC 在真实数据上不稳/崩溃 | 中 | 前向卖点弱 | 多 seed/多起点；调 $\mu$ 调度；报告崩溃点；必要时退回 Theseus-GM 作为前向 |
| 6D 变号定理证不出 | 高 | 理论强度下降 | 保留 1D 严格 + 映射层面 + 6D 数值；在 Limitations 明确 |
| 一致 $H^{-1}$ 界证不出 | 中 | $O(r^{-4})$ 条件性 | 条件性定理 + 实验 $\lambda_{\min}$ 分布；不宣称无条件 |
| SDPR baseline 复现失败/太慢 | 中 | 对比不完整 | 用 KB 的 patch/自适应重试；报告失败率；必要时只对比 SDPR+RANSAC |
| In-The-Dark 端到端不如 RANSAC | 中 | 核心卖点弱 | 论文定位改为“可微性/理论/系统”；诚实报告 |
| 相干外点全方法崩溃 | 高 | “鲁棒”声明弱 | 明确 model breakdown；报告崩溃点；不宣称任意鲁棒 |
| 时间不够 | 中 | 延期 | 先 RA-L（rolling）；IROS 作为备选；宁可少一个 seed，不编数字 |

---

## 6. 每周检查清单

- [ ] 本周的定理/公式是否有假设、维度、边界？
- [ ] 新增数字是否有脚本/日志/种子？是否标“独立复现”？
- [ ] 是否又出现了“全局/零展开/免疫/任意”这类过度措辞？
- [ ] Baselines 是否公平（同数据/同硬件/调参预算）？
- [ ] 失败案例是否记录（不是只报成功）？
- [ ] 论文/README/contributions 的口径是否一致？

---

## 7. 需要用户/主线决策的点

1. **venue**：RA-L 首选是否同意？IROS 2027 作为备选？T-RO 是否值得 stretch？
2. **6D 定理**：是否投入 1–2 周补 6D 变号，还是接受“1D 严格 + 6D 数值”？
3. **实验范围**：是否必须加 PnP？只做 3D-3D 是否够？
4. **算力**：4–6 GPU-weeks 是否可接受；In-The-Dark 3 seeds 是否必须？
5. **开源**：QPEP C++ 依赖是否允许不开源、只给构建说明？

---

## 8. 12 周后的可能延伸（不作为承诺）

- **T-RO 扩展**：6D 变号定理 + 3D 崩溃点 + 鲁棒 SDP 证书（MQ-12）。
- **Prototype 2 组合**：把不等式/物理约束加进 GNC 内层（约束 QPEP），做“鲁棒 + 物理一致”的桌面操作。
- **Prototype 1 组合**：用 Laplace-QPEP 的多峰初始化 GNC，处理对称物体。
- **POLI 组合**：学习 $W_i$ + 鲁棒层，做点云配准的统计流形。
