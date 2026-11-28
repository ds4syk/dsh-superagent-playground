# P3 real LiDAR 3D-3D: KITTI raw drive 0001

> **一句话结论**：在 KITTI raw 2011_09_26 drive 0001 的真实 LiDAR 连续帧上（N=300，10 对相邻帧，20 trials/cell，0–70% 注入外点），GNC-GM 在随机/相干/边界三类协议下 0–70% 成功率为 100%（相干 50% 例外：35%，与 RANSAC 的 0% 同属多数外点不可辨识）；随机外点下 GNC p50 2.7–2.8 ms vs 固定 1000 迭代 RANSAC 61–64 ms（约 22×）。这是室外稀疏 LiDAR 的独立复现，不是 3DMatch 室内片段的重复。

## 数据与运动真值

- 数据：`/mnt/storage1/syk/data/kitti_raw/2011_09_26/2011_09_26_drive_0001_sync/`（438 MB zip，108 帧 Velodyne + OXTS + 图像，2011-09-26）。
- 位姿：KITTI raw 的 sync drive 不直接给逐帧真值；从 OXTS 的 lat/lon/alt + roll/pitch/yaw 按 KITTI devkit 的 Mercator 公式构造世界位姿，再取相邻帧相对运动：
  `T_rel_velo = T_velo_imu @ inv(T_w_j) @ T_w_i @ inv(T_velo_imu)`，其中 `T_velo_imu = calib_imu_to_velo.txt`。
- 方向校验：`verify_kitti.py` 对 6 对相邻帧比较 `T` 与 `inv(T)` 的 LiDAR 最近邻中位距离；正确方向 0.100–0.123 m，反向 0.143–0.183 m。相邻帧 |t|≈1.6–2.2 m，旋转 0.11–0.25°。
- 对应：源帧采样点 → GT 变换 → 目标帧最近邻；`tau=0.45 m` 保留真实重叠点。基础集 1000 对/帧对，trial 抽 300；LiDAR 点距约 0.1 m，故 `c=0.25 m`（3DMatch 的 0.05 m 会丢弃大多数正确匹配）。
- 外点：random（替换为目标帧随机 LiDAR 点）、coherent（单一竞争刚体变换，30–60°/0.6–1.5 m）、boundary（`e/c²∈[1/3,3]`）。
- 成功：旋转 <5° 且平移 <0.1 m（绝对位姿误差，不是相对运动误差）。

## 命令

```bash
cd labs/03-robust-qpeplayer/experiments/real/kitti
PY=/home/syk/miniconda3/envs/WM/bin/python

# 1) 方向/外参校验（~10 s；首次会缓存 0.15 m 体素点云）
srun --partition=cpu --time=00:15:00 --ntasks=1 --cpus-per-task=2 $PY verify_kitti.py

# 2) 主表（10 帧对 × 3 外点类型 × 8 p × 20 trials × 5 方法，~36 s）
srun --partition=cpu --time=00:35:00 --ntasks=1 --cpus-per-task=2 \
  $PY run_kitti.py 2>&1 | tee results/logs/kitti_$(date +%Y%m%d_%H%M%S).log
```

产物：`results/kitti.{json,md,png}`、`results/kitti_raw.jsonl`、`results/verify_kitti.json`、`results/logs/`。

## 主表（成功率 %）

| method | 0% | 10% | 20% | 30% | 40% | 50% | 60% | 70% |
|-|-|-|-|-|-|-|-|-|
| Kabsch | 100 | 10 | 0 | 5 | 0 | 0 | 0 | 0 |
| RANSAC-1000 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| RANSAC-adaptive | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| IRLS-GM | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| **GNC-GM** | **100** | **100** | **100** | **100** | **100** | **100** | **100** | **100** |

中位旋转误差（deg，节选）：p=30% RANSAC 0.04 / GNC 0.04；p=70% RANSAC 0.06 / GNC 0.06。runtime p50：GNC 2.7–2.8 ms，RANSAC-1000 61–64 ms，RANSAC-adaptive 0.9–25.8 ms（随 p 上升）。相干 60–70% 所有方法都恢复外点变换（0% 成功率）；边界 0–70% 全部 100%，GNC 70% 中位旋转 0.07° vs RANSAC 0.14°。

## 与 3DMatch/ModelNet 的关系

- 3DMatch 是室内 RGB-D TSDF 片段（密集、有真实遮挡/扫描噪声）；KITTI 是室外稀疏 LiDAR（点距 ~0.1 m、动态物体、运动真值来自 OXTS）。两套真实数据都只测试鲁棒外层，基础对应由 GT 生成（oracle NN），不是学习描述子。
- 真实描述子前端只在 3DMatch 上做了 FPFH 审计（`../threedmatch/run_descriptor.py`）；KITTI 未接描述子，这是明确的缺口。
- KITTI 的 c=0.25 m 与 3DMatch/ModelNet 的 c=0.05 m 不同，原因是 LiDAR 稀疏；跨数据集比较只看成功率/相对加速，不把绝对旋转误差直接横比。

## 限制

- OXTS 位姿不是毫米级真值；clean Kabsch 的 0.02–0.03° / 0.01–0.03 m 是当前噪声地板。绝对平移误差阈值 0.1 m 对 2 m 帧间运动仍然有意义，但不是 SLAM 级评测。
- 只用了 drive 0001 的 10 对相邻帧，未覆盖夜间/雨天/高速/城市峡谷。
- 外点注入是受控协议，不是真实 LiDAR 描述子误匹配；真实前端缺口未关闭。
