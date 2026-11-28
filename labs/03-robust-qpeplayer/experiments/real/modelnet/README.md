# P3 real-geometry 3D-3D: ModelNet40

> **一句话结论**：在 ModelNet40 真实物体点云（N=200，50 trials/cell，0–70% 均匀外点）上，GNC-GM 0–70% 全部 100% 成功、精度与 RANSAC 持平、对固定 1000 迭代 RANSAC **约 30× 加速**（1.7 ms vs 54 ms）；IRLS 在 70% 外点降到 98%，Kabsch 在 50% 已 0%。真实几何复现了合成主表的结论。

## 设置

- 数据：`/mnt/storage1/syk/data/modelnet40/modelnet40_ply_hdf5_2048/`（test split，2468 个物体，每物体 2048 点）。
- 每个 trial：随机物体 → 随机 200 点 → 去中心/归一化到单位球 → 随机 $R_{gt},t_{gt}$ → 加 $\sigma=0.01$ 高斯噪声 → 按 $p$ 替换为 $[-1,1]^3$ 均匀外点。
- 方法/指标与合成表一致：Kabsch / RANSAC-1000（3 点，阈值 $c=0.05$）/ IRLS-GM / GNC-GM；成功 = 旋转 $<5^\circ$ 且平移 $<0.1$。
- 运行：`srun --partition=cpu --time=00:20:00 --ntasks=1 /home/syk/miniconda3/envs/WM/bin/python run_modelnet.py`，固定种子。

## 主表（成功率）

| method | 0% | 10% | 20% | 30% | 40% | 50% | 60% | 70% |
|-|-|-|-|-|-|-|-|-|
| Kabsch | 100 | 88 | 48 | 22 | 4 | 0 | 2 | 0 |
| RANSAC+Kabsch | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| IRLS-GM | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 98 |
| **GNC-GM** | **100** | **100** | **100** | **100** | **100** | **100** | **100** | **100** |

中位旋转误差（deg，节选）：p=30% 时 RANSAC 0.20 / GNC 0.19；p=70% 时 0.25 / 0.26（两位小数持平）。runtime p50：GNC 1.69–1.82 ms，RANSAC-1000 53.5–56.1 ms（约 30×）。

## 代码校验

- 与合成实验共用 `kabsch`/`solve_gnc`/`solve_irls`/`solve_ransac`；合成实验的已知真值/闭式/FD 校验仍适用。
- 真实数据额外检查：0% 外点时 GNC≈Kabsch（同一解），RANSAC 全内点≈Kabsch；点数/尺度归一化后 $c=0.05$ 与前表一致。
- 未做：真实 3DMatch/KITTI 的 fragment/overlap 管线；ModelNet40 是单物体点云 + 随机刚体变换，不是 SLAM 式场景配准。

## 限制

- 点云是单物体、随机刚体变换 + 均匀外点；不是 3DMatch 的真实重叠片段，也不是 KITTI 的室外场景。
- 真实感知前端（描述子/匹配）未接；这里隔离的是鲁棒外层。
