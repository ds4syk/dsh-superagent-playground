# P3 real-geometry 3D-3D: ModelNet10（第二真实数据集）

> **一句话结论**：在 ModelNet10 test split 的真实物体网格（908 个物体 / 10 类，按面积均匀采样 2048 点）上，协议与 ModelNet40 完全一致（N=200，σ=0.01，c=0.05，50 trials/cell，0–70% 均匀外点）。GNC-GM 0–70% 全部 100% 成功；RANSAC-1000 同样 100%；IRLS-GM 100%（ModelNet40 为 98%@70%）；Kabsch 在 50% 归零。与 ModelNet40 相比，鲁棒方法最大成功率偏差 **2 pp**、最大中位旋转偏差 **0.038°**；对固定 1000 迭代 RANSAC 的加速为 **31.7–32.5×**（GNC 1.9 ms vs RANSAC 59–61 ms）。**ModelNet40 的结论在第二个真实数据集上成立。**

## 数据集差异（诚实说明）

| 项 | ModelNet40（已有表） | ModelNet10（本表） |
|---|---|---|
| 来源 | 官方预处理 `ply_data_test*.h5` | 官方 `ModelNet10.zip` 的 OFF 网格 |
| 划分 | test split，2468 个物体 | test split，**908** 个物体 |
| 类别 | 40 类 | **10 类**：bathtub 50 / bed 100 / chair 100 / desk 86 / dresser 86 / monitor 100 / night_stand 86 / sofa 100 / table 100 / toilet 100 |
| 每物体点数 | 官方 2048 点 | **本实验自行采样** 2048 点（面积加权均匀三角面采样 + 均匀重心坐标，`prep_modelnet10.py`，seed 20261127） |
| 预处理 | 无（直接读 h5） | OFF 解析 + 三角化 + 表面重采样 → `modelnet10_test_2048.npz` |

差异只影响“点云怎么来”；配准协议（采样 200 点 → 去中心/单位球归一化 → 随机刚体变换 → σ 高斯噪声 → 按 p 替换为 U[-1,1]³ 外点 → 同一套 solver/指标/成功阈值）逐行沿用 `../modelnet/run_modelnet.py`。`../modelnet/run_modelnet.py` 未做任何修改。

## 设置与命令

- 环境：`/home/syk/miniconda3/envs/WM/bin/python`（numpy/h5py/matplotlib；未装 trimesh，OFF 解析与采样为自写脚本）。
- 数据盘：`/mnt/storage1/syk/data/modelnet10/`（模型与点云缓存均不放仓库）：

```bash
mkdir -p /mnt/storage1/syk/data/modelnet10 && cd /mnt/storage1/syk/data/modelnet10
curl -sSL -x http://127.0.0.1:7899 -o ModelNet10.zip \
  https://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip   # 473,402,300 B
unzip -q ModelNet10.zip                                                    # 2.2 GB, 908 test + 3991 train .off

# 预处理（test split -> 2048 点/物体，约 16 s）
srun --partition=cpu --time=00:10:00 --ntasks=1 --cpus-per-task=2 \
  /home/syk/miniconda3/envs/WM/bin/python prep_modelnet10.py \
  --root /mnt/storage1/syk/data/modelnet10/ModelNet10 --split test \
  --out  /mnt/storage1/syk/data/modelnet10/modelnet10_test_2048.npz

# 主实验（50 trials × 8 p × 4 方法）
srun --partition=cpu --time=00:20:00 --ntasks=1 --cpus-per-task=2 \
  /home/syk/miniconda3/envs/WM/bin/python run_modelnet10.py 2>&1 | tee logs/run_modelnet10.log

# 与 ModelNet40 对比（只读 ../modelnet/results/main_table.json）
/home/syk/miniconda3/envs/WM/bin/python compare_modelnet40.py
```

产物：`results/main_table.{json,md,png}`、`results/meta.json`、`results/trials.jsonl`（每 trial 一行，1600 行 = 8 p × 50 trials × 4 方法）、`results/comparison.md`。

## 主表（成功率 %）

| method | 0% | 10% | 20% | 30% | 40% | 50% | 60% | 70% |
|-|-|-|-|-|-|-|-|-|
| Kabsch | 100 | 100 | 60 | 24 | 2 | 0 | 0 | 0 |
| RANSAC+Kabsch | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| IRLS-GM | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| **GNC-GM (ours)** | **100** | **100** | **100** | **100** | **100** | **100** | **100** | **100** |

中位旋转误差（deg，节选）：p=30% RANSAC 0.16 / GNC 0.16；p=70% RANSAC 0.28 / GNC 0.29。runtime p50：GNC 1.85–1.93 ms vs RANSAC-1000 58.9–61.5 ms（31.7–32.5×）。

## 与 ModelNet40 对比（详见 `results/comparison.md`）

同协议下两表逐格并排（`ModelNet40 / ModelNet10`）：

| method | 0% | 20% | 40% | 60% | 70% |
|-|-|-|-|-|-|
| Kabsch | 100/100 | 48/60 | 4/2 | 2/0 | 0/0 |
| RANSAC+Kabsch | 100/100 | 100/100 | 100/100 | 100/100 | 100/100 |
| IRLS-GM | 100/100 | 100/100 | 100/100 | 100/100 | 98/100 |
| GNC-GM (ours) | 100/100 | 100/100 | 100/100 | 100/100 | 100/100 |

- 鲁棒方法（RANSAC/IRLS/GNC）最大成功率偏差 **2 pp**（IRLS 70% 由 98→100），最大中位旋转偏差 **0.038°**。
- 非鲁棒 Kabsch 在两表都于 50% 首次归零；其 10–20% 处 ±12 pp 波动来自单个 trial 的随机残差（50 trials 分辨率 2 pp），不影响结论。
- GNC/RANSAC 加速比：ModelNet40 30.1–31.7×，ModelNet10 31.7–32.5×（同一量级）。
- **结论保持**：GNC-GM 在 0–70% 外点全胜、精度与 RANSAC/IRLS 持平、对固定 1000 迭代 RANSAC 约 30× 加速；真实几何（两套数据）与合成主表一致。

## P3 注册表建议行（供父 agent 合并，本 PR 未改 `labs/EXPERIMENTS.md`）

`labs/EXPERIMENTS.md` 第 15 行 P3 摘要与第 112 行“真实几何（ModelNet40）”段可各补一句：

> 第二真实数据集 ModelNet10（test split 908 物体/10 类，协议同前）复现同一结论：GNC 0–70% 全 100%，鲁棒方法相对 ModelNet40 成功率偏差 ≤2 pp、中位旋转偏差 0.038°，RANSAC 加速 31.7–32.5×；见 `03-robust-qpeplayer/experiments/real/modelnet10/`。

## 代码校验与限制

- solver 与合成/ModelNet40 表完全共用 `../synthetic/run_registration.py`（`kabsch`/`solve_ransac`/`solve_irls`/`solve_gnc`），未改一行；0% 外点时 GNC≈Kabsch（同一解）、RANSAC 全内点≈Kabsch，与 ModelNet40 行为一致。
- ModelNet10 的点云是**本实验从 OFF 网格重采样**的，不是官方 2048 点发布版；采样种子固定，`prep_modelnet10.py` 可重放。
- 仍是“单物体点云 + 随机刚体变换 + 均匀外点”，不是 3DMatch 重叠片段 / KITTI 场景，也不含真实描述子/匹配前端；外点类型只覆盖均匀随机，未覆盖相干/边界外点（那些在合成 E1/E2/E4）。
- 只使用 test split；train split（3991 物体）未参与，未做跨类别泛化实验。
