# P2 real-data: constrained PnP on BOP LINEMOD

> **一句话结论**：在 BOP LINEMOD 的真实物体网格、真实相机内参与**从深度图 RANSAC 拟合的真实桌面平面**上，无约束 QPEP 有 4.76 mm 中位穿透、仅 39% 可行；ALM-QPEP 全部 18 trials 可行（0 mm 穿透），其中 50% 通过 14×14 Shor 证书；软惩罚 94% 可行、投影 100% 可行，但认证率 50%/33%。位姿误差四种方法接近（真实数据上约束主要买“硬可行 + 可认证”，不是精度）。

## 设置

- 数据：BOP `lm_test_bop19`（下载 `lm_base.zip` / `lm_models.zip` / `lm_test_bop19.zip` 到 `/mnt/storage1/syk/data/bop/lm/`）。
- 帧：`000001, 000002, 000003, 000005, 000006, 000008, 000010, 000013, 000015`，每帧前 2 张图；对象 1/5/10（ape / can / eggbox）。
- 加载校验：把模型顶点用 GT 位姿 `cam_R_m2c,cam_t_m2c` + `cam_K` 投影，投影 bbox 与 `scene_gt_info.bbox_obj` 对齐（例：`[221.95,295.66]-[263.94,341.08]` vs `[222,296,41,44]`）。
- 真实桌面：对背景深度（mask=0）在 bbox 外扩 40 px 内 RANSAC 拟合平面（内点阈值 5 mm），法向朝向物体；GT 物体最低点贴桌。
- 观测：30 个随机模型顶点，按 GT 投影 + 1 px 像素噪声 + 向下像素偏置（10/20/30 px）；bearing 由 `K^{-1}` 反投影。约束：沿桌面法向最低的 16 个模型顶点，非穿透 `n^T(Rv+t)+d>=0`。
- 方法：Unconstrained QPEP / Soft penalty λ=1e3 / Post-hoc projection / ALM-QPEP；`p=16`。指标：穿透 mm、rot/trans、可行（违反≤1e-6）、14×14 Shor 认证。
- 运行：`srun --partition=cpu --time=00:30:00 --ntasks=1 /home/syk/miniconda3/envs/WM/bin/python run_linemod.py`；结果 `results/main_table.md`、`raw.json`。

## 主表（18 trials）

| method | 穿透中位 (mm) | 可行率 | 认证率 | rot 中位 (deg) | trans 中位 (mm) |
|-|-|-|-|-|-|
| Unconstrained | 4.76 | 39% | 33% | 2.16 | 36.8 |
| Soft penalty λ=1e3 | 0.00 | 94% | 50% | 2.16 | 38.6 |
| Projection | 0.00 | 100% | 33% | 2.03 | 33.7 |
| **ALM-QPEP** | **0.00** | **100%** | **50%** | 2.11 | 37.7 |

- ALM 的硬可行 100%、软惩罚 94%（仍有 6% 违反），投影 100% 但认证率低（解非最优）。
- 认证率只有 50%：真实数据上 14×14 Shor 不总是紧（合成实例是 100%）；论文必须写“fail-safe / 条件认证”，不能写“100%”。
- 位姿误差四种方法接近：真实 LM 的对应点由 GT 投影 + 像素噪声生成，约束主要消除穿透，不改善旋转/平移精度（与合成结论一致）。

## 代码/数据校验

- 投影 bbox 与 GT bbox 对齐（加载正确）；桌面法向朝向物体；GT 最低点贴桌（必要时把平面下移 ≤0.1 mm 保证 GT 可行）。
- 无约束在 10–30 px 偏置下确实穿透（4.76 mm 中位，39% 可行），说明约束不是 inactive。
- `cert` 由 `certificate.shor_bound` 计算；SDP 求解器 SCS 有“solution may be inaccurate”警告，认证率按当前数值容差报告。

## 限制

- 对应点由 GT 位姿投影 + 像素噪声生成，不是学习到的检测器/匹配器；这是**层实验**，不是端到端感知。
- 桌面平面从深度 RANSAC 拟合；未做真实遮挡/多物体接触的复杂约束。
- 只做 LINEMOD 三个对象/九帧；YCB-Video/KITTI 未做。
