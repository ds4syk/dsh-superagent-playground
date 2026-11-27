# UTIAS In-The-Dark: GT protocol recovered, real measurements obtained

> **一句话结论**：In-The-Dark 的 per-pair GT **不需要下载 905 MB 的 teach run**——用 FTP range 请求从 zip 中央目录里只取 `transforms_temporal.txt`（315,463 B，CRC 校验通过）即可；repeat run 的 `transforms_spatial.txt` × teach 的 `transforms_temporal.txt` 复合，再用官方 `transform_camera_vehicle.txt` 共轭到相机系，得到 **0.25–1.32 px 重投影误差**（逆变换 12–22 px、恒等 6–11 px、不共轭 14–27 px）的逐对 GT。在 run_000002 / run_000012 上各跑 15 对真实相邻帧：自然内点率 **79.4% / 88.5%**，GNC-GM **0.078° / 0.020 m**，成功@0.1 m **100%**，15 对 3.6–4.0 s。**局限：这两个 run 是白天（官方元数据 morning/sunny，图像均值亮度 133/131），不是夜间测试。**

## 1. GT 协议（用真实压缩包核实，不是文档推断）

官方每个 zip 的成员（从 run_000000.zip 中央目录实读，2,236 项）：

| 成员 | 压缩后 | 解压后 | 说明 |
|-|-|-|-|
| `images/left|right/*.png` | 2,232 张 | – | 512×384 双目 |
| `transforms_temporal.txt` | 128,820 B | 315,463 B | **teach run 才有**，1,116 条 `(0,j,0,j+1)` 相邻链接 |
| `transforms_spatial.txt` | – | – | **repeat run 才有**，每个 repeat 顶点一条到 teach 顶点的匹配 |
| `timestamps_images.txt` | 8,656 B | 27,906 B | `顶点, Unix ns` |
| `gps.txt`, `timestamps_gps.txt` | 8,374 / 10,027 B | 39,066 / 27,906 B | 部分 run 有 |

行格式：`idA_run,idA_vertex,idB_run,idB_vertex,T_ba(16 个行主序)`，`p_b = T_ba p_a`
（a = 前两个 id，b = 后两个 id）。**已用几何验证，不是照抄文档**。

### 1.1 复合公式（repeat run 逐对位姿）

repeat 顶点 i 匹配 teach 顶点 j_i，spatial 给 `M_i = T_{teach_{j_i} <- rep_i}`：

```
T_{rep_{i+1} <- rep_i} = inv(M_{i+1}) · T_{teach_{j_{i+1}} <- teach_{j_i}} · M_i
```

teach 腿由 temporal 的 gap-1 链接链式相乘（run_000002 的 teach 跳变分布：+1 890、+2 103、0 90、+3 3）。

**teach delta = 0 也可解**：两帧注册进**同一个** teach 帧 j，因此
`p_rep_{i+1} = inv(M_{i+1}) M_i p_rep_i` 精确成立，实测 0.75–1.15 px、97–100% 匹配 < 2 px。
所以 run_000002 的 1,086 对相邻帧**全部**有 GT（不是丢掉 90 对）。

### 1.2 必须共轭到相机系

发布的变换是**车体系**之间的。双目前端在**相机系**工作，官方 `transform_camera_vehicle.txt`
（E，det=1，把车体系点映射到相机系）给出：

```
T_cam = E · T_veh · inv(E)
```

| 候选 | 中位重投影（12 对标定） | 解读 |
|-|-|-|
| **cam_ba（采用）** | **0.56 px** | 正确读法 + 正确共轭 |
| cam_ab（转置读法） | 4.38 px | 行约定读反 |
| identity | 8.64 px | 无运动 |
| cam_ba_inv | 16.60 px | 方向反 |
| cam_ab_inv | 16.88 px | 双向都错 |
| veh_ba（不共轭） | 19.92 px | 车体系直接投到相机 |

约定在**标定集（前 12 对）**上选定，报数用的是**另一批**（200–215、313–316），不重叠。
两项独立证据同时成立：行约定 `ba` 与共轭 `E T E⁻¹`。

## 2. 怎么在预算内拿到 teach 元数据

`fetch_zip_members.py`：先取 zip 末尾 128 KB 解析 EOCD + 中央目录（2,236 项），
再按 member 的 local-header offset 做区间请求，解 raw deflate 并校验 size/CRC32。

```bash
curl -s -r 904477094-904605593 -o tail.bin 'ftp://.../run_000000.zip'      # 128 KB
curl -s -r 904452427-904605593 -o cd.bin   'ftp://.../run_000000.zip'      # 153 KB 中央目录
python3 fetch_zip_members.py --url 'ftp://asrl3.utias.utoronto.ca/2020-vtr-dataset/UTIAS-In-The-Dark/run_000000.zip' \
  --size 904605594 --member transforms_temporal.txt --member timestamps_images.txt \
  --out-dir /mnt/storage1/syk/data/utias_in_the_dark/teach_meta
# 51.6 s，共传输 ~282 KB（对比整包 904,605,594 B，约 3,200×）
```

结果落在 `/mnt/storage1/syk/data/utias_in_the_dark/teach_meta/`。全量 `run_000000.zip`
**在本轮结论里不需要**（他人若已下完，可直接 unzip 对照，结论不会变）。

## 3. 实测量（run_000002 / run_000012，各 15 对相邻帧）

前端标签：**手写 SIFT + StereoSGBM 稠密视差（或稀疏左右三角化）**，即 `../kitti_stereo/stereo_vo.py`，
与 KITTI fallback 同一份代码；**不是 SDPR 的 VGG/UNet 前端**。

| config | run | 内点率中位 [min,max] | 对应数 | 深度中位 | GT \|t\| | GNC rot | GNC trans | 成功@0.1 m | 用时 |
|-|-|-|-|-|-|-|-|-|-|
| dense SGBM | run_000002 | 79.4% [76.1, 84.2] | 445 | 7.2 m | 0.123 m | 0.078° | 0.020 m | 100% | 3.6 s |
| sparse 三角化 | run_000002 | 66.7% [61.4, 70.1] | 265 | 5.9 m | 0.123 m | 0.103° | 0.011 m | 100% | 3.7 s |
| dense SGBM | run_000012 | 88.5% [77.3, 90.7] | 637 | 7.0 m | 0.283 m | 0.092° | 0.010 m | 100% | 4.0 s |

鲁棒层（dense, run_000002）：Kabsch 0.171°/0.143 m、成功 47%；RANSAC-adaptive 0.091°/0.019 m（0.8 ms）；
IRLS-GM 0.078°/0.020 m（3.0 ms）；GNC-GM 0.078°/0.020 m（2.3 ms）；floor（GT 内点 Kabsch）0.109°/0.019 m。
**朴素 Kabsch 只有 47% 成功，是鲁棒层把它抬到 100%。** 完整表见 `results/inthedark_main.md`。

## 4. 自检（`verify_inthedark.py`，两 run 全绿）

| run | GT 重投影 | 逆变换 | 恒等 | 不共轭 | 左右行差 |
|-|-|-|-|-|-|
| run_000002 | 0.65 px | 18.07 px | 9.53 px | 21.47 px | 0.58 px |
| run_000012 | 0.52 px | 27.89 px | 14.72 px | 31.34 px | 0.56 px |

检查项：图像 512×384 可解码、左右行对齐（< 2 px，验证“已校正”假设）、GT 优于逆变换/恒等/不共轭、
GT 中位重投影 < 2 px。全部通过（`results/verify_run_*.json` 的 `checks` 块）。

## 5. 复现

```bash
cd labs/03-robust-qpeplayer/experiments/real/inthedark
PY=/mnt/storage1/syk/venvs/opencv/bin/python
$PY verify_inthedark.py --run /mnt/storage1/syk/data/utias_in_the_dark/run_000002 --pairs 8 \
    --out results/verify_run_000002.json
$PY calibrate_convention.py --run /mnt/storage1/syk/data/utias_in_the_dark/run_000002 \
    --teach /mnt/storage1/syk/data/utias_in_the_dark/teach_meta --pairs 12 \
    --out results/convention_calib_r2.json
$PY run_inthedark.py --run /mnt/storage1/syk/data/utias_in_the_dark/run_000002 \
    --start 200 --pairs 15 --fe dense --out results/inthedark_dense.json
$PY summarize_inthedark.py
```

## 6. 局限（不要外推）

1. **两个 run 都是白天**：左图均值亮度中位 133 / 131，官方元数据 run_000002 为 morning / sunny / small shadows。本轮验证的是**压缩包格式、标定、GT 协议、前端管线**在真实 In-The-Dark 数据上可用，**不是夜间鲁棒性**。
2. GT 本身来自官方配准（place recognition + ICP），逐对噪声约 0.3–1.4 px 等效；0.01–0.02 m 的误差应读作“到达 GT 噪声底”，不是绝对精度。
3. 每配置 15 对、每对单一确定性实例，无注入外点、无 trials。
4. run_000012 有一条 spatial 匹配向回跳（teach 顶点 −1106），这类非相邻跳变被 `max_teach_jump=4` 跳过。
5. 本轮**不覆盖**夜间 run（未下载），也不覆盖 SDPR 前端的对比。

## 7. 历史记录（已解决）

早前结论是“所有官方 run ≥ 590 MB、直连 FTP ~14 KB/s，因此 In-The-Dark 不可测，只能用 KITTI raw 双目 fallback”（`../kitti_stereo/`）。
该结论的**下载前提已不成立**：元数据可以 range 提取；repeat run 的 874 MB / 617 MB 压缩包已在数据盘并解压出图像。
带宽仍然限制“整包下载”，但不再限制“拿到 GT 并做实验”。
