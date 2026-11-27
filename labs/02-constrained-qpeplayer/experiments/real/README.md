# P2 real-data: cross-dataset constrained PnP (BOP LINEMOD / LM-O / YCB-Video)

> **一句话结论**：在三个真实 BOP 数据集上，无约束 QPEP 的可行率分别只有 **39% / 0% / 67%**；ALM-QPEP 全部 **100% 硬可行、0 mm 穿透**。真实数据上约束买的是“硬可行”，不是位姿精度（四法位姿接近）；14×14 Shor 证书在真实数据上不总紧，认证率 22%–83%，按 fail-safe 报告，不能写 100%。

## 设置

- 数据：`/mnt/storage1/syk/data/bop/{lm,lmo,ycbv}`（BOP 格式；`test/` 帧、`models/` 网格、`scene_gt.json`、`scene_camera.json`、`depth/`、`mask/`）。
- 每帧：从背景深度 RANSAC 拟合桌面平面（内点 5 mm），法向朝向物体；平面 offset 校准到 GT 最低点接触（`d -= min(n^T x_gt + d)`），保证 GT 可行。
- 观测：30 个随机模型顶点，按 GT 位姿 + `cam_K` 投影，加 1 px 噪声 + 10/20/30 px 向下像素偏置；bearing 由 `K^{-1}` 反投影。
- 约束：沿桌面法向最低的 16 个模型顶点，非穿透 `n^T(Rv+t)+d >= 0`；`p=16`。
- 方法：Unconstrained QPEP / Soft penalty λ=1e3 / Post-hoc projection / ALM-QPEP。
- 指标：穿透（mm）、rot/trans、可行（最大违反 ≤1e-6）、14×14 Shor 认证。
- 脚本：`real/linemod/run_linemod.py`（LM 原始表）；`real/bop/run_bop.py <dataset>`（LM-O/YCB；`BOP_MAX_FRAMES`/`BOP_MAX_IMG` 控制规模）。
- 加载校验：LINEMOD 模型投影 bbox 与 `scene_gt_info.bbox_obj` 对齐（`[221.95,295.66]-[263.94,341.08]` vs `[222,296,41,44]`）；BOP 帧/内参/GT 单位（m）核对。

## 跨数据集主表

| 数据集 | trials | Unconstrained 穿透/可行 | Soft λ=1e3 可行/认证 | Projection 可行/认证 | **ALM 穿透/可行/认证** |
|-|-|-|-|-|-|
| LINEMOD | 18 | 4.76 mm / 39% | 94% / 50% | 100% / 33% | **0 mm / 100% / 50%** |
| LM-O（遮挡） | 18 | 23.06 mm / **0%** | 89% / 17% | 100% / 6% | **0 mm / 100% / 22%** |
| YCB-Video | 12 | 0 mm / 67% | 100% / 67% | 100% / 75% | **0 mm / 100% / 83%** |

- 位姿误差四法接近（LM rot 中位 2.0–2.2°、trans 34–39 mm；LM-O 1.9–2.1°、29–63 mm；YCB 1.1–1.5°、16–20 mm），约束主要消除穿透。
- 认证率随数据集变化（50%/22%/83%）：真实数据上 SDP 不一定紧；论文写“条件认证 / fail-safe”，不写“100%”。
- LM-O 的 unconstrained 0% 可行、ALM 100% 是跨数据集最强对比；YCB 的 unconstrained 67% 可行，ALM 补到 100%。

## 限制

- 对应点由 GT 位姿投影 + 像素噪声生成，不是学习到的检测器/匹配器；这是**层实验**，不是端到端感知。
- 桌面平面是深度 RANSAC + GT 接触校准的“虚拟支撑面”，不是完整接触物理。
- YCB 只跑 2 帧 × 1 图 × 3 物体 × 3 偏置 = 12 trials；LM-O 1 帧 × 2 图 × 3 物体 × 3 偏置 = 18；规模小于合成表。
- 认证率对 SDP 求解器容差敏感（SCS 有 “solution may be inaccurate” 警告）。
