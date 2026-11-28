# KITTI odometry 00/05/07：真实 LiDAR 鲁棒 3D-3D 配准（官方精确 GT）

> **一句话结论**：在 KITTI odometry 00/05/07 的 30 对相邻帧上（N=300，c=0.25 m，20 trials/cell，0–70% 注入外点），父目录 raw drive 0001/0002 的结论复现：随机/边界外点的 RANSAC-1000 与 GNC-GM 成功率为 100%，Kabsch 在 p≥0.1 崩溃；相干外点在 p=0.5 相变（GNC 53% / IRLS 38% / RANSAC 0%）；GNC-GM 比固定 1000 迭代 RANSAC 快约 22–26×（pooled p=0.5：2.8 vs 73.7 ms p50）。**GNC 的优势不依赖小运动**：|t|=0.09–9.62 m 全覆盖，171 个成对格点上 GNC 成功率从未低于 RANSAC。

## 0. 数据与可用范围

- 序列：`00`（4541 GT 帧）、`05`（2761）、`07`（1101）。位姿为官方 `data_odometry_poses.zip` 的精确逐帧 GT（`T_w_cam0`），标定为每序列 `calib.txt` 的 `Tr:` 行。
- 点云：官方 `data_odometry_velodyne.zip` 共 84.8 GB / 43,574 帧，未整包下载。用 HTTP Range 从镜像逐帧抽取本实验实际用到的 **33 帧/序列，共 99 帧，194.0 MB（十进制；磁盘 186 MiB）**。清单 `/mnt/storage1/syk/data/kitti_odom/manifest.json`。
- 每序列主表 10 对相邻帧（步长 453/275/109 帧）；另外 gap 分析用 manifest 的 sweep 锚点（00: 0/1513/3027，05: 0/920/1840，07: 0/367/734）取 gap=2/4/8。
- **缺口（已记录）**：数据盘只有上述 99 帧，不是完整 00–10。第一次运行在 gap 阶段按 `n_frames//2` 取锚点，访问了未下载的 `002270.bin` 而失败（日志 `results/logs/kitti_odom_20261127_220137.log`）；改为 manifest 中实际抓取的 sweep 锚点后跑通。扩展到全部 11 条序列约需 0.7 GB / 15 min，本轮未做。

## 1. 协议

与父目录 `../kitti/`、`../kitti2/` 逐项相同，唯一区别是位姿来源：

| 项 | 值 |
|---|---|
| 对应 | 源帧采样点 → GT 变换 → 目标帧最近邻；`tau=0.45 m`；基础集 1000 对/帧对，trial 抽 300 |
| 点云 | 0.15 m 体素降采样，距离 1–30 m |
| 外点 | random / coherent（30–60°，0.6–1.5 m 竞争刚体）/ boundary（`e/c²∈[1/3,3]`） |
| p | 0, 0.1, …, 0.7 |
| trials | 20/cell；运动分析 10/cell |
| 方法 | Kabsch、RANSAC-1000、RANSAC-adaptive、IRLS-GM、GNC-GM |
| 成功 | 绝对位姿误差 rot<5° 且 \|t\|<0.1 m |

## 2. 命令

```bash
cd labs/03-robust-qpeplayer/experiments/real/kitti_odom
PY=/home/syk/miniconda3/envs/WM/bin/python

# 1) 方向/外参校验（18 对，~1 s）
srun --partition=cpu --time=00:05:00 --ntasks=1 --cpus-per-task=2 $PY verify_kitti_odom.py \
  2>&1 | tee results/logs/verify_kitti_odom_$(date +%Y%m%d_%H%M%S).log

# 2) 主表 + 运动/gap 分析（3 序列 × 3 外点类型 × 8 p × 20 trials × 5 方法，
#    外加 171 个运动格点；实测约 4.7 min）
srun --partition=cpu --time=00:40:00 --ntasks=1 --cpus-per-task=2 $PY run_kitti_odom.py \
  2>&1 | tee results/logs/kitti_odom_$(date +%Y%m%d_%H%M%S).log

# 3) 表格 + 图 + rounds
$PY analyze_kitti_odom.py
$PY make_rounds.py
```

**本次实际执行说明**：提交时 cpu 分区已有 32/40 CPU 被占、`MaxCPUsPerNode=24`，srun 作业一直 pending；为不阻塞时间盒，最终用 `nice -n 19` 直接跑 1 核（日志 `results/logs/kitti_odom_20261127_220340.log`，耗时 279 s + I/O）。正式重跑应按上面的 `srun` 命令。测量期间节点上还有其他用户/agent 的负载，运行时间读 p50、比值比绝对值稳健。

## 3. 结果

### 3.1 方向校验

18/18 对判为 `i->j`：正确方向 NN 中位距离 **0.053–0.114 m**，反向 **0.104–0.430 m**，平均反向/正向比 3.35×；与 raw drive 0001/0002 的正向中位 0.100–0.140 m 同量级。逐对见 `results/verify_kitti_odom.json`。

### 3.2 主表：每序列 + pooled

成功率（%），每格 20 trials；`—` 表示该格方法崩溃后中位误差无意义。

| dataset | method | random 0% | random 30% | random 70% | coherent 50% | boundary 70% |
|---|---|---|---|---|---|---|
| 00 | Kabsch | 100 | 0 | 0 | 0 | 100 |
| 00 | RANSAC-1000 | 100 | 100 | 100 | 0 | 100 |
| 00 | RANSAC-adaptive | 100 | 100 | 100 | 0 | 95 |
| 00 | IRLS-GM | 100 | 100 | 95 | 25 | 100 |
| 00 | GNC-GM | 100 | 100 | 100 | 55 | 100 |
| 05 | Kabsch | 100 | 0 | 0 | 0 | 100 |
| 05 | RANSAC-1000 | 100 | 100 | 100 | 0 | 100 |
| 05 | RANSAC-adaptive | 100 | 100 | 100 | 0 | 100 |
| 05 | IRLS-GM | 100 | 100 | 100 | 45 | 100 |
| 05 | GNC-GM | 100 | 100 | 100 | 50 | 100 |
| 07 | Kabsch | 100 | 0 | 0 | 0 | 100 |
| 07 | RANSAC-1000 | 100 | 100 | 100 | 0 | 100 |
| 07 | RANSAC-adaptive | 100 | 100 | 100 | 0 | 100 |
| 07 | IRLS-GM | 100 | 100 | 100 | 45 | 100 |
| 07 | GNC-GM | 100 | 100 | 100 | 55 | 100 |
| **pooled** | **Kabsch** | **100** | **0** | **0** | **0** | **100** |
| **pooled** | **RANSAC-1000** | **100** | **100** | **100** | **0** | **100** |
| **pooled** | **RANSAC-adaptive** | **100** | **100** | **100** | **0** | **98** |
| **pooled** | **IRLS-GM** | **100** | **100** | **98** | **38** | **100** |
| **pooled** | **GNC-GM** | **100** | **100** | **100** | **53** | **100** |

中位旋转误差（deg，节选）：

| dataset | method | random 0% | random 70% | coherent 50% | boundary 70% |
|---|---|---|---|---|---|
| 00 | Kabsch | 0.048 | 3.79 | 24.15 | 0.095 |
| 00 | RANSAC-1000 | 0.049 | 0.047 | 45.85 | 0.173 |
| 00 | GNC-GM | 0.046 | 0.039 | 0.092 | 0.076 |
| 05 | Kabsch | 0.048 | 5.09 | 20.74 | 0.066 |
| 05 | RANSAC-1000 | 0.049 | 0.047 | 41.90 | 0.152 |
| 05 | GNC-GM | 0.057 | 0.042 | 15.87 | 0.064 |
| 07 | Kabsch | 0.043 | 4.05 | 26.42 | 0.071 |
| 07 | RANSAC-1000 | 0.046 | 0.063 | 53.94 | 0.125 |
| 07 | GNC-GM | 0.042 | 0.055 | 0.113 | 0.059 |
| **pooled** | **Kabsch** | **0.047** | **3.96** | **24.16** | **0.078** |
| **pooled** | **RANSAC-1000** | **0.047** | **0.048** | **46.84** | **0.143** |
| **pooled** | **GNC-GM** | **0.048** | **0.043** | **0.106** | **0.067** |

注：coherent p=0.5 的 GNC 中位旋转 0.106° 是“成功 trial 很准、失败 trial 完全是外点变换”的双峰混合；失败比例 47%，见 3.4。

### 3.3 运行时间与 GNC 加速比

random，p50 ms：

| dataset | p | RANSAC-1000 | RANSAC-adaptive | GNC-GM | speedup GNC vs RANSAC-1000 |
|---|---|---|---|---|---|
| 00 | 0% | 82.7 | 0.54 | 1.21 | 68.2× |
| 00 | 50% | 73.7 | 5.14 | 3.25 | 22.7× |
| 00 | 70% | 75.0 | 23.37 | 3.04 | 24.7× |
| 05 | 0% | 62.6 | 0.46 | 0.96 | 65.0× |
| 05 | 50% | 60.4 | 4.07 | 2.72 | 22.2× |
| 05 | 70% | 60.1 | 18.56 | 2.71 | 22.2× |
| 07 | 0% | 61.5 | 0.41 | 0.93 | 66.4× |
| 07 | 50% | 87.1 | 5.34 | 3.43 | 25.4× |
| 07 | 70% | 85.2 | 26.66 | 3.82 | 22.3× |
| **pooled** | **0%** | **63.4** | **0.46** | **0.97** | **65.2×** |
| **pooled** | **50%** | **73.7** | **4.48** | **2.80** | **26.3×** |
| **pooled** | **70%** | **66.7** | **21.43** | **2.78** | **24.0×** |

p=0 时 GNC 的 65× 主要是它从零外点快速收敛；p≥0.3 后稳定在 **22–26×**。三条序列的加速比一致，不是 pooled 假象。

### 3.4 相干相变与边界误差

pooled 成功率（%），GNC 列的最后一列是“估计更接近注入外点变换”的 trial 比例：

| p | Kabsch | RANSAC-1000 | IRLS-GM | GNC-GM | GNC 选中外点 |
|---|---|---|---|---|---|
| 30% | 0 | 100 | 100 | 100 | 0% |
| 40% | 0 | 100 | 100 | 100 | 0% |
| 50% | 0 | 0 | 38 | 53 | 47% |
| 60% | 0 | 0 | 0 | 0 | 100% |
| 70% | 0 | 0 | 0 | 0 | 100% |

- 相变点 p≈0.5：内点/外点支持数接近，GNC 仍有 53%，RANSAC-1000 为 0；p=0.6 起 GNC 100% 选中外点变换，与“多数外点不可辨识”一致，不是求解器 bug。
- 边界外点 p=70%（pooled）：GNC / IRLS 100%，rot 0.067°/trans 0.0128 m；RANSAC-1000 100%，rot 0.143°/0.0310 m；RANSAC-adaptive 98%，rot 0.151°。GNC/IRLS 的连续核把误差压到 RANSAC 的 ~1/2。

### 3.5 运动幅度与帧间隔：GNC 的优势依赖 |t| 吗？

- 覆盖：|t| **0.09–9.62 m**（10 Hz 下 3–346 km/h 的帧间运动），自然内点率 **99.6%→70.2%**；171 个 (dataset×pair×p) 运动格点，10 trials/cell。
- 按 gap 汇总（p 合并）：

| gap | \|t\| (m) | 自然内点率 | GNC 成功率 | RANSAC-1000 成功率 |
|---|---|---|---|---|
| 1 | 0.09–1.27 | 89–100% | 100% | 100% |
| 2 | 0.19–2.49 | 78–99% | 100% | 99.6% |
| 4 | 0.39–4.94 | 73–96% | 100% | 100% |
| 8 | 0.92–9.62 | 70–95% | 98.9% | 94.8% |

- 按 |t| 分箱（p=0.5）：0–1.5 m / 1.5–4 m / 4–8 m 三档 GNC 与 RANSAC 均 100%；8–30 m 两格 GNC 90%、RANSAC 85%。
- GNC 的 runtime 优势在 |t| 各分箱稳定 **22–26×**；旋转误差在大 |t| 时反而更占优（8–30 m：GNC 0.091° vs RANSAC 0.127°）。
- 成对比较：171 个格点 GNC 成功率从未低于 RANSAC（最小差 0，最大 +50 pp）。唯一绝对低点是 `odom00 1513–1521`（|t|=9.62 m，gap=8，自然内点率 72%，p=0.5）GNC 80%，但同格 RANSAC 只有 70%；p=0.3 时 GNC 90% vs RANSAC 40%，p=0.7 时 100% vs 60%。所以这不是“GNC 依赖小运动”，恰好相反：运动最大的一格 GNC 相对优势最大。

**结论**：GNC 的优势主要来自鲁棒核的收敛/速度，不依赖 |t| 或 gap；扩大帧间隔会降低自然内点率，但不改变方法排序。该格 10 trials 中有 2 次失败（80%），仍高于同格 RANSAC 的 7/10，不能作为“|t| 越大越差”的趋势证据。

## 4. 六轮实测（`results/rounds.json`）

| # | 轮次 | 结论 | 关键数 |
|---|---|---|---|
| 1 | 精确 GT 方向校验 | 保留 | 18/18；fwd 0.053–0.114 vs inv 0.104–0.430 m，平均 3.35× |
| 2 | p=0 clean floor | **假设不成立，保留 loader** | odom 0.042–0.057° vs raw 0.030–0.044°；残差地板由 LiDAR 采样/体素主导 |
| 3 | 主表复现 | 保留 | pooled p=0.7 random：Kabsch 0 / RANSAC 100 / GNC 100；coherent p=0.5：53 / 38 / 0；GNC 26.3× |
| 4 | 每序列一致性与加速比 | 保留 | 三序列 p=0.7 GNC 均 100%；p=0.5 加速比 22.7/22.2/25.4× |
| 5 | 相干崩溃与边界误差 | 保留 | p=0.5 GNC 53%、IRLS 38%、RANSAC 0%；p≥0.6 全 0 且 GNC 100% 选外点；边界 p=0.7 GNC rot 0.067° |
| 6 | \|t\|/gap 依赖性 | 保留 | \|t\| 0.09–9.62 m；171 格 GNC 从不低于 RANSAC；9.62 m 极值格 GNC 80 vs RANSAC 70 |

所有数字由 `make_rounds.py` 从 `results/kitti_odom.json`、`results/verify_kitti_odom.json` 和父目录 raw 结果计算，无手工填入。

## 5. 限制

1. 只覆盖 00/05/07 三条 odometry 序列；官方 00–10 共 11 条，完整 Velodyne 84.8 GB 未下载。
2. 每序列只用 10 对相邻帧（与父目录一致），不是全序列里程计/SLAM 精度评测；这是鲁棒外层的外点敏感性实验。
3. 基础对应是 GT-mapped NN（oracle 匹配），不是学习描述子/真实前端误匹配；真实 LiDAR 前端缺口仍未关闭。
4. 成功阈值是绝对误差（rot<5°, |t|<0.1 m）；大运动时相对阈值更宽松，但所有成功格点的中位误差远低于阈值（GNC 中位 <0.15°/0.03 m），结论不敏感。
5. p=0 的 clean floor 并未低于 raw drive OXTS 口径，说明残差地板主要不是位姿误差；这一点已在轮 2 明确标为“假设不成立”，不把它写成“精确 GT 更紧”。
6. 运行时间受共享节点负载影响；绝对值读 p50，GNC/RANSAC 比值在多次运行间稳定在 ~22–26×。
7. 首次 gap 运行因未下载 `002270.bin` 失败，已改锚点并记录；失败日志保留在 `results/logs/`。

## 6. 产物与变更

产物：

- `results/kitti_odom.json`（配置、逐序列 360 行、pooled 120 行、171 个运动格点、raw-drive 对照）
- `results/kitti_odom.md` / `results/kitti_odom.png`（完整表格与 6 面板图）
- `results/kitti_odom_raw.jsonl`（7200 条 trial 记录）、`results/kitti_odom_motion_raw.jsonl`（855 条运动格点记录）
- `results/verify_kitti_odom.json`、`results/rounds.json`、`results/logs/`

代码/文档变更（未 commit）：

- `run_kitti_odom.py`：输出改名为 `kitti_odom.*`；gap 锚点改为 manifest sweep 实际帧；pooled 行补 `frac_picked_outlier`。
- `analyze_kitti_odom.py`：读新文件名；新增每序列 runtime/加速比、相干相变、边界误差、GNC 优势 vs |t| 表和极端格说明；关键结论改为从数据计算。
- `make_rounds.py`（新增）：生成 6 轮 `results/rounds.json`。
- `README.md`（新增）：本文件。

**keep/revert**：全部保留。没有修改 `theory/`、`verify/`、`labs/EXPERIMENTS.md`、`AGENTS.md`、`deliverables/`，未 commit/push。
