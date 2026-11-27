# UTIAS In-The-Dark: dataset card and why no run was measured

> **一句话结论**：In-The-Dark 的数据格式本身完全适合本实验（双目 512×384、f=387.777 px、基线 0.239965 m、逐帧相对位姿），但**所有官方 run 压缩包 ≥ 590 MB，直连 FTP 实测 ~14 KB/s（8 路并行 ~34 KB/s）**，单个 run 需 7–17 小时，超出本轮预算。因此本轮没有 In-The-Dark 实测量；真实前端的端到端/测试时实验改用 KITTI raw 双目完成（`experiments/real/kitti_stereo/`），并在此文档中如实标注为 fallback。

## 1. 实测：下载不可能在预算内完成

| 项目 | 实测值 | 证据 |
|-|-|-|
| run_000002 大小 | 874,634,312 B (837 MB) | `curl -I ftp://asrl3.utias.utoronto.ca/2020-vtr-dataset/UTIAS-In-The-Dark/run_000002.zip` → `Content-Length: 874634312` |
| 单连接持续速率 | **14 KB/s**（30 s 内 434,176 B） | `run_000002.zip` 增长 434,176 B / 30 s |
| 8 路并行 range 请求 | **~34 KB/s 合计**（20 s 内 673,320 B） | `curl -r` ×8 到 16 MB 区间 |
| 单 run ETA | 单连接 ~17.4 h；8 路 ~7 h | 上两行外推 |
| 最小的 In-The-Dark run | run_000012 = 590 MB（ETA 单连接 ~11.7 h） | 官方列表 / FTP `LIST` |
| 代理 | HTTP 代理对 FTP 返回 502（任务说明已给出） | 未再尝试，直连可用 |
| 已落地文件 | `run_000002.zip` 仅 1.6 MB 的部分下载，**不是有效压缩包** | `/mnt/storage1/syk/data/utias_in_the_dark/` |
| 本地镜像 | 无；全盘 `find` 无 In-The-Dark 副本 | `find /mnt/storage1/syk -iname "*inthedark*"` 只命中本 worktree 与数据目录 |

带宽是**唯一**阻塞项：目录树、文件命名、标定、变换文件格式都有官方文档与小体积文本文件可核实，见 §2。恢复条件很明确——在能跑到 ≥2 MB/s 的链路上重跑 §4 的命令即可（837 MB ≈ 7 分钟）。

## 2. 有一手证据的部分（官方文档 + 已下载的小文件）

从 <https://asrl.utias.utoronto.ca/datasets/2020-vtr-dataset/> 的 Hardware / File Structure / Interpreting Text Files 节，以及同站 `text_files/` 下两个已下载并核对的小文件：

**相机与基线**（`/mnt/storage1/syk/data/utias_in_the_dark/camera_parameters.txt`，原样）：

```
width: 512
height: 384
focal length: 387.777
cx: 257.446
cy: 197.718
baseline: 0.239965
```

无畸变系数被发布 ⇒ 只能把图像当作已校正的针孔模型；**这一点无法在未下载数据时验证**，必须在真实 run 上先做一次重投影/行对齐检查（见 §4 的 `verify_inthedark.py` 占位）。

**相机-车辆外参**（`transform_camera_vehicle.txt`，T_camera_vehicle，行主序 4×4）：

```
 0.000796327  -1.0         0.0        0.119873
-0.330472     -0.000263164 -0.943816   1.49473
 0.943815      0.000751586 -0.330472   0.354804
 0.0           0.0          0.0        1.0
```

**run_000002 元数据**（官方列表）：1092 帧，July 19 09:46 EDT，morning / sunny / small shadows，GPS = Yes。整段 260 m 闭环，关键帧约每 0.2 m 一张。

**官方给出的目录结构**（每个 run 一个 zip）：

```
run_000002/
  images/left/000000.png ... 000000.png 对应顶点 (2,0)
  images/right/000000.png ...
  timestamps_images.txt     # 每行: 顶点号, Unix 纳秒时间戳
  transforms_spatial.txt    # repeat run；teach run 是 transforms_temporal.txt
  gps.txt, timestamps_gps.txt  # 仅 23/39 个 run 有
```

`transforms_spatial.txt` 每行：`idA_run, idA_vertex, idB_run, idB_vertex, 16 个行主序 T_ba`，约定是 **T_to_from**（`p_b = T_ba p_a`），且“拓扑邻近顶点的变换才可信”。这正是本实验需要的 GT 相对位姿，无需 GPS。

> 以上结构来自官方文档与工具库 `utiasASRL/vtr-dataset-tools`，**未能在本轮用真实压缩包内的文件逐项核对**（无数据）。README 中把它标为“文档声明”，不是“已验证”。

## 3. 明确“没有”的东西

- 没有逐帧 IMU/GPS 绝对位姿做外部校验（只有 GT 相对位姿与部分 run 的低精度 GPS）；
- 没有发布畸变模型/校正参数（所以 stereo rectification 只能假设已校正，需实测验证）；
- 没有 LiDAR、没有深度图 ⇒ 3D 只能来自双目三角化；
- 没有本轮的实测量：任何 In-The-Dark 的 inlier rate、成功率、运行时都**不存在**，本文档不提供，也不会用 KITTI 数字冒充。

## 4. 一旦拿到数据就能跑的命令（脚本已就绪，未经真实数据验证）

`load_inthedark.py` 按 §2 的文档格式实现：读 `images/left|right`、`transforms_spatial.txt`、`timestamps_images.txt`，组装成 `stereo_vo` 需要的 `{left, right, P_L, P_R}`（P_L/P_R 由 `f, cx, cy, b` 生成，`P_R[0,3] = P_L[0,3] - f*b`），复用与 KITTI 完全相同的 `run_kitti_stereo.py` 前端口与鲁棒层。缺数据时它不会静默跑出数字：没有 `transforms_spatial.txt` 直接 `FileNotFoundError`。

```bash
cd labs/03-robust-qpeplayer/experiments/real/inthedark
PY=/mnt/storage1/syk/venvs/opencv/bin/python

# 0) 下载（需 ≥2 MB/s 链路；本机 ~14 KB/s 不可行）
mkdir -p /mnt/storage1/syk/data/utias_in_the_dark && cd $_
curl -C - -o run_000002.zip ftp://asrl3.utias.utoronto.ca/2020-vtr-dataset/UTIAS-In-The-Dark/run_000002.zip
unzip -q run_000002.zip -d /mnt/storage1/syk/data/utias_in_the_dark/

# 1) 数据集自检：图像可读、左右行对齐、GT 方向、帧间距
$PY verify_inthedark.py --run /mnt/storage1/syk/data/utias_in_the_dark/run_000002

# 2) 真实前端口 + 鲁棒层（30 对帧，SIFT/SGBM 或稀疏三角化，GT 标注）
$PY run_inthedark.py --run /mnt/storage1/syk/data/utias_in_the_dark/run_000002 --pairs 30
```

## 5. 本轮实际做了什么（fallback）

`experiments/real/kitti_stereo/`：KITTI raw drive 0001 的真实校正双目（image_02/03，基线 0.5327 m），SIFT/ORB + 亚像素 + SGBM 稠密视差（或稀疏左右匹配三角化）→ 相邻帧 2D-2D 链接 → 3D-3D 对应 → OXTS GT 相对位姿标注 → 五个鲁棒外层方法。**这是 fallback，数据集不是 In-The-Dark**，但满足“真实手写前端 + 真实外点 + 真实 GT”的测试时要求，且不使用 SDPR 的 VGG/UNet 前端。详见该目录 README。
