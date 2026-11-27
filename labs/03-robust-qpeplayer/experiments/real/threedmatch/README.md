# P3 real 3D-3D: 3DMatch scene fragments

> **一句话结论**：在两套官方 3DMatch 室内场景片段（redkitchen、sun3d-home_at，各 8 个高重叠帧对，N=300，20 trials/cell，0–70% 注入外点）上，GNC-GM 与 RANSAC-1000 在随机外点 0–70% 都 100% 成功，GNC 的 p50 为 2.2–2.5 ms，比固定 1000 迭代 RANSAC（61–71 ms）快 **27–30×**；IRLS 在 redkitchen 70% 降到 95%。相干外点 60–70% 时所有方法都恢复外点变换（模型不可辨识，不是求解器 bug）；边界外点全部 100%，但 GNC 的 70% 中位旋转 0.25° 优于 RANSAC 的 0.74°。另有真实 FPFH 前端审计：自然内点率 67.8–80.0% 时 GNC 100% 成功，RANSAC-1000 97–100%。

## 数据与真值方向

- 数据：`/mnt/storage1/syk/data/3dmatch_fragments/`，官方 3DMatch geometric-registration benchmark 的 `7-scenes-redkitchen`（39 MB）与 `sun3d-home_at-home_at_scan1_2013_jan_1`（42 MB）片段 + evaluation zip。
- 每个 zip 含 60 个 TSDF-fused 片段（`cloud_bin_*.ply`，25 万点级）；evaluation 含 `gt.log`（`id1 id2 n` + 4×4 变换）。
- **方向**：`verify_gt_direction.py` 在 12 对/场景上比较 `T` 与 `inv(T)` 的最近邻中位距离：redkitchen 正向 0.496 m vs 反向 0.053 m；sun3d-home_at 0.374 m vs 0.070 m。结论：`gt.log` 的变换把 **id2 映射到 id1**，runner 必须取逆。
- 体素下采样 0.03 m；每场景按 GT-NN 重叠率选 8 个高重叠帧对（top-8 采样重叠率≈100%，中位残差 12.0–18.8 mm）。

## 基础对应与协议

- `gt_nn`：源片段采样点 → GT（取逆）→ 目标片段最近邻，`tau=0.06 m` 保留；4× 过采样后取 N=300。基础匹配是 **oracle GT-NN**（真实扫描/遮挡/采样噪声，但不是描述子），用于隔离鲁棒外层。
- 外点：random（替换为目标场景随机点）、coherent（单一竞争刚体变换 30–60°/0.5–1.5 m）、boundary（`e/c²∈[1/3,3]`）。
- 方法：Kabsch / RANSAC-1000（固定 1000） / RANSAC-adaptive（0.999 置信度） / IRLS-GM / GNC-GM；`c=0.05 m`。
- 成功：旋转 <5° 且平移 <0.1 m；runtime 为单次调用 wall-clock p50/p95。

## 命令

```bash
cd labs/03-robust-qpeplayer/experiments/real/threedmatch
PY=/home/syk/miniconda3/envs/WM/bin/python

# 0) 下载/解压（代理；数据盘）
#    7-scenes-redkitchen.zip, 7-scenes-redkitchen-evaluation.zip,
#    sun3d-home_at-home_at_scan1_2013_jan_1.zip + -evaluation.zip
#    来自 http://3dvision.princeton.edu/projects/2016/3DMatch/downloads/scene-fragments/

# 1) 方向校验（~10 s）
srun --partition=cpu --time=00:15:00 --ntasks=1 --cpus-per-task=4 $PY verify_gt_direction.py

# 2) 主表（2 场景 × 3 外点类型 × 8 p × 20 trials × 5 方法，~80 s）
srun --partition=cpu --time=00:45:00 --ntasks=1 --cpus-per-task=4 \
  $PY run_threedmatch.py 2>&1 | tee results/logs/threedmatch_$(date +%Y%m%d_%H%M%S).log

# 3) 可选：真实 FPFH 前端审计（Open3D venv，~10 s + 特征缓存）
#    /mnt/storage1/syk/venvs/open3d/bin/python run_descriptor.py
```

产物：`results/threedmatch.{json,md,png}`、`threedmatch_raw.jsonl`、`pairs_*.json`、`verify_gt_direction.json`、`descriptor.{json,md}`、`logs/`。

## 主表（成功率 %，两场景分别）

redkitchen：

| method | 0% | 10% | 20% | 30% | 40% | 50% | 60% | 70% |
|-|-|-|-|-|-|-|-|-|
| Kabsch | 100 | 100 | 65 | 45 | 30 | 20 | 5 | 0 |
| RANSAC-1000 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| RANSAC-adaptive | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| IRLS-GM | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 95 |
| **GNC-GM** | **100** | **100** | **100** | **100** | **100** | **100** | **100** | **100** |

sun3d-home_at：Kabsch 100/100/50/15/30/5/0/0；RANSAC/IRLS/GNC 全部 100%（含 IRLS 70%）。

中位旋转误差（deg，节选）：redkitchen p=70% RANSAC 0.29 / GNC 0.24；sun3d-home_at p=70% 0.23 / 0.21。runtime p50：GNC 2.2–2.5 ms，RANSAC-1000 61–71 ms，RANSAC-adaptive 0.6–18.7 ms。

相干外点：p≤40% 全部 100%；p=50% GNC 65%/35%（redkitchen/sun3d-home_at），RANSAC 0%；p=60–70% 所有方法 0%，且 100% 恢复外点变换。这与 Prop 3.7 的不可辨识性一致。边界外点：0–70% 所有方法 100%；redkitchen 70% GNC 0.25° vs RANSAC 0.74°。

## 真实 FPFH 前端审计（`run_descriptor.py`）

- Open3D FPFH（radius 0.20 m）互最近邻 + 成对距离一致性过滤；**匹配阶段不用 GT**，GT 只做评测/选帧对。
- 自然内点率：redkitchen 80.0%，sun3d-home_at 67.8%。
- GNC 成功 100%/100%，RANSAC-1000 100%/97%，Kabsch 100%/83%；GNC p50 1.22/1.46 ms vs RANSAC-1000 71.7/64.7 ms（44–59×）。
- 这是手工 FPFH，不是 FCGF/Predator；它证明真实前端 + 几何一致性过滤后鲁棒层仍可用，但不替代学习描述子实验。

## 限制

- GT-NN 主表是 oracle 基础匹配，不是描述子误匹配；真实描述子结果只覆盖一个手工 FPFH 前端。
- 只选高重叠帧对（top-8），没有覆盖低重叠/长基线；3DMatch 的 50 场景 train/test 全量未做。
- 外点注入是受控协议；真实感知外点的分布可能不同。
- `gt.log` 方向已实测为 id2→id1；若复用其它 3DMatch 发布版，必须先重跑 `verify_gt_direction.py`。
