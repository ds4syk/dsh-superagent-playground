# P3 室外真实 LiDAR 跨 drive 复核：KITTI raw drive 0002

> **一句话结论**：在第二条独立 drive（2011_09_26 drive 0002，77 帧，306 MB 解压）上，用与父目录 drive 0001 完全相同的 loader/协议（OXTS Mercator + `calib_imu_to_velo` + T_rel_velo，voxel 0.15 m、tau 0.45 m、c 0.25 m、N=300、10 对相邻帧、20 trials/cell）复现主表：随机/边界外点下 4 个鲁棒方法 0–70% 成功率 100%，GNC-GM 比固定 1000 迭代 RANSAC 快 20.8–22.9×，相干外点在 p≈0.5–0.6 发生相变。**drive 0001 的结论全部成立**（10 项检查 10/10 PASS），无需回退。

本目录是父目录 `../kitti/`（drive 0001）的**跨 drive 复核副本**，不修改父目录。代码改动仅三处：drive 路径、缓存命名空间、模块名（`load_kitti2.py` / `run_kitti2.py` / `verify_kitti2.py`）；几何与算法代码逐行相同。帧对从 0001 的前 38 帧改到覆盖 0002 全部 77 帧的 10 对相邻帧。

## 数据与下载

- 下载（代理 `http://127.0.0.1:7899`，实测 20 s，15–20 MB/s）：

```bash
cd /mnt/storage1/syk/data/kitti_raw
curl -L -x http://127.0.0.1:7899 -O \
  https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0002/2011_09_26_drive_0002_sync.zip
# calib 已在父任务下载：2011_09_26_calib.zip（4 KB，3 个 calib txt）
unzip -qo 2011_09_26_drive_0002_sync.zip -d .
```

- `2011_09_26_drive_0002_sync.zip` = **319,776,816 B（305 MiB / 320 MB 十进制）**，`unzip -t` 无错误。
- 解压后 306 MB：77 帧 Velodyne + OXTS（drive 0001 为 108 帧 439 MB）。
- 数据落在数据盘 `/mnt/storage1/syk/data/kitti_raw/`，未进仓库、未进 `/home` 或 `/tmp`。
- 0.15 m 体素缓存单独放 `_cache/2011_09_26_drive_0002_sync/`，避免与 drive 0001 的缓存同名冲突（父代码缓存键只含体素/量程）。

## 方向校验（`verify_kitti2.py`，6 对相邻帧）

正确方向 NN 中位距离 **0.108–0.140 m**，反向 **0.136–0.269 m**，6/6 判为 `i->j`；帧间 |t|=1.32–1.81 m，旋转 0.14–0.61°。与 drive 0001（fwd 0.100–0.123、inv 0.143–0.183）同一量级。完整数值见 `results/verify_kitti2.json`。

| pair | \|t\| (m) | rot (deg) | fwd NN (m) | inv NN (m) | 判定 |
|---|---|---|---|---|---|
| 0→1 | 1.317 | 0.37 | 0.121 | 0.269 | i->j |
| 7→8 | 1.493 | 0.16 | 0.140 | 0.268 | i->j |
| 14→15 | 1.528 | 0.61 | 0.136 | 0.228 | i->j |
| 21→22 | 1.567 | 0.47 | 0.122 | 0.192 | i->j |
| 35→36 | 1.557 | 0.35 | 0.108 | 0.136 | i->j |
| 63→64 | 1.808 | 0.14 | 0.122 | 0.140 | i->j |

## 协议（与 drive 0001 完全一致）

- 帧对（10 对连续帧，覆盖 0–64 帧）：(0,1) (7,8) (14,15) (21,22) (28,29) (35,36) (42,43) (49,50) (56,57) (63,64)。
- 基础对应：源帧采样 → GT 变换 → 目标帧最近邻，`tau=0.45 m`；每帧对 1000 对，trial 抽 300（抽后自然内点率 78–93%）。10 个帧对基础集的自然内点率（`d<c=0.25 m`）为 82–89%。
- 外点：random（替换为目标帧随机 LiDAR 点）、coherent（单一竞争刚体变换 30–60°/0.6–1.5 m）、boundary（`e/c²∈[1/3,3]`）。
- 方法：Kabsch、RANSAC-1000、RANSAC-adaptive、IRLS-GM、GNC-GM；20 trials/cell，p=0–0.7。
- 成功：旋转 <5° 且平移 <0.1 m（绝对位姿误差）。

## 主结果（drive 0002）

成功率（%），drive 0002 / drive 0001 并排见 `results/kitti2_compare.md`：

| outlier | Kabsch | RANSAC-1000 | RANSAC-adaptive | IRLS-GM | GNC-GM |
|---|---|---|---|---|---|
| random 0% | 100 | 100 | 100 | 100 | 100 |
| random 10–70% | 10→0 | 100 | 100 | 100 | 100 |
| coherent ≤40% | 100 | 100 | 100 | 100 | 100 |
| coherent 50% | 0 | 0 | 0 | 30 | **40**（0001: 35） |
| coherent 60/70% | 0 | 0 | 0 | 5/0 | 0/0 |
| boundary 0–70% | 100 | 100 | 100 | 100 | 100 |

- 中位旋转误差：random 下 GNC/RANSAC/IRLS 全程 0.04–0.07°；boundary p=70% GNC **0.092°** vs RANSAC 0.160°。Kabsch 在 random 下随 p 退化到 5.13° 且成功率 0%（已崩溃）；boundary 下外点只偏离 GT 点 0.14–0.43 m，Kabsch 中位旋转仍 0.04–0.08°、成功率 100%，说明边界协议本身对最小二乘不构成杀伤。
- 运行时间 p50（两次独立 run 逐 trial 取 min，滤掉共享节点上 ~1.8× 的同租户抖动；`kitti2.md` 里的 runtime 表是单次 run 原值，有抖动，以 `kitti2_compare.md` 的 min-of-2 为准）：GNC 0.95 / 2.66 / 2.71 / 4.14 / 4.69 / 2.77 / 2.81 / 2.81 ms；RANSAC-1000 61.6 / 61.0 / 61.3 / 86.0 / 104.1 / 62.6 / 62.3 / 62.6 ms；RANSAC-adaptive 0.67→26.1 ms。
- 加速比 GNC vs RANSAC-1000：p≥0.1 为 **20.8–22.9×**（p=0 干净场景 65×，GNC 一步收敛）。与 README 的 ~22× 一致。
- 平移中位误差（random，GNC）：0.014–0.026 m，高于 drive 0001 的 0.008–0.012 m；旋转中位 0.04–0.07°（drive 0001 0.03–0.06°）。都远低于 0.1 m / 5° 成功阈值，只是本 drive 的噪声地板略高。

## 与 drive 0001 的结论对照（`results/kitti2_compare.md`，10 项检查）

| 检查 | drive 0002 | 结论 |
|---|---|---|
| random：4 个鲁棒方法 p=0.1–0.7 成功率 100% | 100% | PASS |
| random：Kabsch 在 p≥0.1 崩溃（≤15%） | 10%→0 | PASS |
| random：GNC 约 20× RANSAC-1000 | 20.8–22.9× | PASS |
| random：鲁棒方法中位旋转 <0.1° | 最大 0.073° | PASS |
| coherent：p≥0.6 失去 GT 位姿（≤10%） | IRLS 5%（p=0.6），其余 0 | PASS |
| coherent p=0.5：GNC/IRLS 少数成功、RANSAC 全失败 | GNC 40 / IRLS 30 / RANSAC 0 | PASS |
| boundary：全方法 0–70% 全 100% | 100% | PASS |
| boundary p≥0.4：GNC 旋转误差低于 RANSAC | 0.092 vs 0.160°（p=0.7） | PASS |
| 跨 drive：random+boundary 80 个格点成功率差 ≤10 点 | 最大 5 点 | PASS |
| 跨 drive：coherent p=0.5 GNC 差 ≤10 点 | 40 vs 35 | PASS |

确定性：同一代码/种子的两次 run，全部非时间字段逐 trial 完全一致（0 处差异），只有计时抖动。

## 命令

```bash
PY=/home/syk/miniconda3/envs/WM/bin/python
cd labs/03-robust-qpeplayer/experiments/real/kitti2

# 1) 方向校验（~15 s，首次含 0.15 m 体素缓存）
srun --partition=cpu --time=00:15:00 --ntasks=1 --cpus-per-task=2 $PY verify_kitti2.py

# 2) 主表（~43 s）
srun --partition=cpu --time=00:35:00 --ntasks=1 --cpus-per-task=2 \
  $PY run_kitti2.py 2>&1 | tee results/logs/kitti2_$(date +%Y%m%d_%H%M%S).log

# 3) 跨 drive 对照 + 图（~5 s）
srun --partition=cpu --time=00:10:00 --ntasks=1 --cpus-per-task=2 $PY compare_kitti2.py
```

## 产物

| 文件 | 内容 |
|---|---|
| `results/verify_kitti2.json` | 6 对相邻帧的 GT 变换、\|t\|、旋转、fwd/inv NN 中位、方向判定 |
| `results/kitti2.json` | 全部 120 个 cell 的汇总（success/rot/trans/time） |
| `results/kitti2_raw.jsonl` | 2400 条 per-trial 原始记录 |
| `results/kitti2_run1.json`、`results/kitti2_raw_run1.jsonl` | 计时复现 run（非时间字段与 run2 完全一致） |
| `results/kitti2.md` / `kitti2.png` | run_kitti2 生成的表与 6 联图 |
| `results/kitti2_compare.{json,md,png}` | 与 drive 0001 的并排表、10 项检查、2×2 对照图 |
| `results/logs/` | verify / run×2 / compare 的日志 |

## 限制与缺口

- 仍未接真实描述子前端：基础对应是 GT-mapped NN（oracle），只测鲁棒外层；与 drive 0001、3DMatch、ModelNet 一样，描述子误匹配的缺口未关闭。
- OXTS 位姿不是毫米级真值；clean GNC 的 0.04°/0.014–0.026 m 是本 drive 的噪声地板（略高于 drive 0001）。
- 只覆盖 2011_09_26 同一天、白天、城市道路；未覆盖夜间/雨天/高速/城市峡谷，也未跨日期（跨天气需要 2011_09_28 等日期）。
- coherent 相变点 p≈0.5–0.6 附近两种子有 ±5 点抖动（20 trials），不宜把 40% vs 35% 当精确差异。

## 结论：保留

跨 drive 复核通过，drive 0001 的结论（鲁棒方法 100% 成功、GNC 约 22× 加速、coherent 相变、boundary 全成功且 GNC 旋转误差更低）在独立 drive 0002 上成立。建议保留本目录作为室外跨 drive 证据；不需要回退，也不需要修改父目录。
