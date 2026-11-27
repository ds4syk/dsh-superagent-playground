# P3 fallback: KITTI raw stereo front-end → 3D-3D robust layer

> **一句话结论**：在 KITTI raw drive 0001 的真实校正双目上，用 SIFT/ORB + 亚像素 + SGBM 稠密视差（或稀疏左右三角化）构造真实 3D-3D 对应，自然外点率 17.8%（c=0.5 m，30 对帧）；GNC-GM/IRLS/RANSAC 把旋转压到 **0.05°**，但前向平移系统性低估约 30%（中位误差 **0.70 m** / GT 1.92 m），**所有方法在 rot<5°、trans<0.1 m 判据下成功率 0%**。GT-inlier-only Kabsch（floor）中位误差 0.24 m（c=0.5）或 0.13 m（c=0.3），说明 0.1 m 判据的瓶颈是前端的平移偏置，而不是鲁棒外层。**这是 fallback，数据集不是 In-The-Dark；前端是手写 stereo 特征前端，不是 SDPR 的 VGG/UNet。**

## 1. 为什么是 fallback

In-The-Dark 官方所有 run ≥ 590 MB，直连 FTP 实测 14 KB/s（8 路 34 KB/s），单 run 需 7–17 h，超出预算；证据与格式文档见 `../inthedark/README.md`。KITTI raw drive 0001 已在数据盘，且具备同样的关键要素：真实校正双目、真实标定、真实自车运动 GT、真实手写视觉前端（SIFT/ORB 匹配错误是真实外点，不是注入的）。

## 2. 数据与 GT

- 图像：`/mnt/storage1/syk/data/kitti_raw/2011_09_26/2011_09_26_drive_0001_sync/image_02`（左）与 `image_03`（右），108 帧，**已校正**（rectified），1242×375，10 Hz。
- 标定：`calib_cam_to_cam.txt` 的 `P_rect_02`/`P_rect_03`（fx=fy=721.5377，cx=609.5593，cy=172.854）。由 P 的第四列差得到校正后基线 **b = 0.5327 m**（与 `T_03` 的 -0.5372 m 一致，差值来自校正缩放）。
- GT 位姿：OXTS 的 lat/lon/alt + roll/pitch/yaw → 世界位姿（KITTI devkit Mercator），链式 `T_rect02_imu = R_rect_02 @ T_cam0_velo @ T_velo_imu`，相对运动 `T_{j<-i} = inv(T_w_j) @ T_w_i`，即 `y = T x`。方向校验：把同一对 3D 对应用 GT 与其逆变换比较，中位残差比 **4.1–4.9×**（正确方向残差更小），与 `../kitti/verify_kitti.py` 的 LiDAR 方向校验一致。
- 视差→深度闭式：`Z = fx·b/d`，`X=(u-cx)Z/fx`，`Y=(v-cy)Z/fy`。
- **深度校验**：同帧 LiDAR 投影到左图，与 SGBM 视差深度逐点比较，中位比值 3–10 m:1.000，10–20 m:1.013，20–30 m:1.050，30–45 m:1.047，45–60 m:1.061 → 深度无 30% 级偏置，GT 组合正确。

## 3. 前端协议（无 GT 泄漏）

1. 检测：SIFT（nfeatures=4000，contrastThreshold=0.01）或 ORB（3000，nlevels=1 保证亚像素有意义），左图灰度；`cv2.cornerSubPix` 双侧亚像素（win=5）。
2. 3D：`dense` = StereoSGBM（numDisparities=128，block=5，uniqueness=10，speckle=100/2）在左图关键点处采样视差，要求 d>1 px、3 m<Z<60 m；`sparse` = 左右 SIFT 互检+比值 0.8 匹配后解析三角化（行差<2 px）。
3. 帧间链接：左右图 SIFT/ORB 描述子比值 0.8 + 互检；只保留两端都有有效 3D 的匹配 → (x_i, y_j)。
4. 标注：`e = ||T_{j<-i} x_i − y_j||`，`e < c_label`（c=0.5 m 主用）为内点。**GT 只用于标注与评估，不参与匹配、三角化、鲁棒求解。**
5. 配对：20 对相邻帧（gap=1，Δt≈0.103 s，|t|≈1.9 m）+ 10 对 gap=3（|t|≈6 m），共 30 对；每对只有一个确定性实例，**无注入外点、无 trials**。

## 4. 主结果（30 对帧，c=0.5 m，SIFT+SGBM）

| method | success@0.1 m | success@0.25 m | med rot (°) | med trans (m) | p50 (ms) | p95 (ms) | max (ms) |
|-|-|-|-|-|-|-|-|
| Kabsch | 0% | 0% | 0.224 | 0.663 | 0.19 | 0.25 | 0.27 |
| RANSAC-1000 | 0% | 0% | 0.068 | 0.710 | 81.28 | 117.96 | 124.86 |
| RANSAC-adaptive | 0% | 0% | 0.072 | 0.700 | 2.66 | 14.12 | 19.54 |
| IRLS-GM | 0% | 0% | 0.052 | 0.704 | 4.85 | 7.14 | 7.58 |
| **GNC-GM** | 0% | 0% | **0.052** | 0.704 | **4.21** | **6.46** | **6.62** |
| floor（GT 内点 Kabsch） | 0% | 54% | 0.071 | **0.239** | – | – | – |

- 自然内点率中位 **17.8%**（最小 0.9%），对应数中位 890（最小 278），内点中位深度 27.3 m。
- GT 运动：中位旋转 0.197°，中位平移 1.92 m。
- 旋转：所有鲁棒方法 ≈0.05–0.07°，比 floor 的 0.07–0.14° 更准（外点抑制有效）。
- 平移：所有方法 0.66–0.71 m，**floor 0.24 m**，鲁棒外层比 floor 差 3×。

### 阈值/前端/检测器消融

| 配置 | n (中位) | 内点率 | floor trans (m) | GNC rot (°) | GNC trans (m) | GNC p50 (ms) |
|-|-|-|-|-|-|-|
| sparse + SIFT, c=0.5 | 229 | 17.7% | 0.233 | 0.047 | 0.694 | 2.12 |
| dense + SIFT, c=0.5 | 890 | 17.8% | 0.239 | 0.052 | 0.704 | 4.21 |
| dense + ORB, c=0.5 | 600 | 20.7% | 0.202 | 0.084 | 0.739 | 3.39 |
| dense + SIFT, c=0.3 | 890 | 5.9% | **0.129** | 0.050 | 0.700 | 4.64 |
| dense + SIFT, c=1.0 | 890 | 69.7% | 0.473 | 0.068 | 0.706 | 3.81 |

两种前端的平移误差几乎相同（0.69–0.70 m），说明瓶颈不是对应数量；c 越小 floor 越准（0.13 m @ c=0.3），但鲁棒层对 c 不敏感（0.70 m），因为它的共识集由**相干的深度/匹配偏置**主导，而不是由 GT 标签主导。这与 P3 在合成 coherent 外点上的结论一致（相干外点占多数时不可辨识），本数据里相干偏置来自同一场景结构在不同帧的深度误差相关。

## 5. 失败案例（诚实记录）

- 全部 30 对都“失败”（trans>0.1 m），误差集中在**前向轴**：例 0→1 GT t=(0.008,-0.006,-2.198)，GNC t=(0.006,0.000,-1.380)；24→25 GT -1.775 → GNC -1.297；gap3 0→3 GT -6.172 → GNC -4.321。比值 0.63–0.73，系统性低估。
- GT 内点集上的平均残差向量约为 (+0.11, -0.02, -0.19) m（0→1），即第 j 帧观测点比 GT 预测更远，与“前向平移被低估”自洽。
- 逐点深度与 LiDAR 一致（比值 ≤1.06），所以不是视差尺度错误；最可能是**帧间匹配的相干深度偏置**（远场点匹配到相邻结构）在残差阈值内自洽，把平移拉偏。
- gap=3 的配对更差（GNC trans 1.60–1.88 m / GT 5.9–6.5 m），因为 0.3 s 内重叠减少、匹配错误率上升（内点率 7–17%）。

## 6. 命令

```bash
cd labs/03-robust-qpeplayer/experiments/real/kitti_stereo
PY=/mnt/storage1/syk/venvs/opencv/bin/python   # cv2 5.1.0; 注：该 venv 的 python 有
                                               # importlib.metadata 兼容问题，用
                                               # PYTHONPATH=/mnt/storage1/syk/venvs/opencv/lib/python3.10/site-packages \
                                               # /home/syk/miniconda3/envs/WM/bin/python 亦可
# 完整实验（30 对 × 5 配置，约 4 分钟；本轮 Slurm cpu 分区被占，直接在登录节点跑单核）
$PY run_kitti_stereo.py 2>&1 | tee results/logs/kitti_stereo_$(date +%Y%m%d_%H%M%S).log
```

产物：`results/kitti_stereo_main.{json,md}`（pair 级 + method 级 + summary，646 KB）、`results/logs/`。

## 7. 限制（必须一起读）

1. **不是 In-The-Dark**：没有夜间/头灯/眩光数据；本实验不能替代 In-The-Dark 结论。
2. **不是 SDPR 前端**：手写 SIFT/ORB + SGBM，不是 VGG/UNet 学习前端；论文不能说“SDPR 前端在真实数据上”。
3. 单序列（drive 0001）、30 对帧、单实例无 trials：数字是确定性的，但没有跨序列方差。
4. `c_label` 用 GT 选择，因此“自然内点率”依赖 c；已给 c∈{0.3,0.5,1.0} 的敏感性。
5. 0.1 m 判据未达成；0.25 m 判据只有 floor 达成（54%），鲁棒方法 0%。不要把它写成成功。
6. 前端深度在远场（>30 m）有 +5% 的系统偏差（LiDAR 对比），对前向平移有贡献，但不足以解释 30% 的低估。
