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

## 跨数据集主表（2026-11-27 扩样，n=90/72/84）

> **重要修正**：早期 n=18/12 的 50%/22%/83% 是**单侧** `J-Shor<=1e-6` 计数，会把 `Shor>J`（负 gap 到 −4e-4）也算作认证，只是**上界**。下表同时给单侧（upper bound）与两侧 `|J-Shor|<=1e-6`（necessary condition）；两者都不是严格证书（SCS 非对偶可行）。合成 16 个噪声实例上 `repair_dual` 的严格证书 16/16、gap 2.28e-8；真实数据严格重跑仍 TODO。

| 数据集 | trials | Unconstrained 穿透/可行 | Soft 可行 | Projection 可行 | **ALM 穿透/可行** | ALM 单侧（上界） | ALM 两侧（必要） | ALM+RLT 单侧 | ALM+RLT 两侧 |
|-|-|-|-|-|-|-|-|-|-|
| LINEMOD | 90 | 11.26 mm / 13.3% | 88.9% | 100% | **0 mm / 100%** | 33.3% | 18.9% | 31.1% | 14.4% |
| LM-O | 72 | 14.93 mm / 6.9% | 91.7% | 100% | **0 mm / 100%** | 30.6% | 12.5% | 33.3% | 4.2% |
| YCB-Video | 84 | 14.09 mm / 6.0% | 90.5% | 100% | **0 mm / 100%** | 33.3% | 9.5% | 23.8% | 6.0% |

- 真实 ALM **硬可行率 100%**（软惩罚 88.9%/91.7%/90.5%）；无约束可行 13.3%/6.9%/6.0%。约束买“硬可行”，不是精度。
- **单侧认证 31–33% 是上界；两侧只有 9–19%**。旧 50/22/83 不再使用。
- RLT 的配对增益不显著且方向不一致：LM 33.3→31.1（McNemar p=0.85）、LM-O 30.6→33.3（p=0.86）、YCB 33.3→23.8（p=0.15，变差）；两侧看 RLT 处处更差。**结论：真实 RLT 无可复现增益，按负结果写，不作为改进 claim。**
- Wilson CI 与逐 trial gap 在 `real/*/results/main_table.md` / `raw.json`（n=90/72/84）。
- 6 个 LM-O（plane-only）实例 `repair_dual` 仍无法认证（gap 1e-4–5e-4）；需 tilt/depth 约束或专门一轮。

## 限制

- 对应点由 GT 位姿投影 + 像素噪声生成，不是学习到的检测器/匹配器；这是**层实验**，不是端到端感知。
- 桌面平面是深度 RANSAC + GT 接触校准的“虚拟支撑面”，不是完整接触物理。
- YCB 只跑 2 帧 × 1 图 × 3 物体 × 3 偏置 = 12 trials；LM-O 1 帧 × 2 图 × 3 物体 × 3 偏置 = 18；规模小于合成表。
- 认证率对 SDP 求解器容差敏感（SCS 有 “solution may be inaccurate” 警告）。
