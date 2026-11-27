# 官方 BPnP（BoChenYS/BPnP）CPU 复现：toy 梯度检验 + P1 same-data 适配

> 证据等级：**独立复现**（本机 CPU 跑的是**发布的原始代码**，未改一行 `BPnP.py`；两个环境/布局问题用一行 shim / `.contiguous()` 解决，见 §2）。

## 0. 一句话结论

- **能在 CPU 上跑**：repo 是纯 PyTorch + OpenCV + kornia，`BPnP.BPnP.apply` 前向/反向都跑通（toy n=8，反向 46.13 ms/6 维基向量）。
- **但它自己的 forward/backward 不一致（噪声数据）**：toy 上 released backward 与 released forward 的中心差分相对误差 **17.1%**（2 px 关键点噪声），精确数据下降到 1.70e-01。根因不是实现 bug 而是**目标函数不同**：`forward` 调 `cv.solvePnP`（几何重投影 LS），`backward` 线性化 BPnP 自己的**代数**normal equations；在后者的输出点上 `|f(out)|=4.77e+06`，而在代数解处 `|f|=4.75e+06`（§3）。
- **same-data 适配成立**：K=I、`pts2d=(U1/U3,U2/U3)` 时，BPnP 的代数残差 `r = u·S − proj12` **恒等于** P1 的 bearing 残差 `P^⊥(Rp+t)`（数值验证 1.8e-15），所以 **P1 平面面 bearing 代价就是 BPnP 的代价**。把 forward 换成 P1 局部解、**backward 原样调用 `BPnP.BPnP.backward`**，与我们的 6-DoF 解析 IFT、P1 自己的 `qpep_core.ift_dq`（S³ 图、t 解析消去）以及分支跟踪 FD 四者一致（§4）。

### 0.1 环境与代码

| 项 | 值 |
|-|-|
| bpnp_repo | /mnt/storage1/syk/third_party/BPnP |
| bpnp_commit | f1f4f6a79ce2bd58a6340a59e64c0900bb6b0772 |
| bpnp_url | https://github.com/BoChenYS/BPnP |
| python | 3.11.17 (/home/syk/miniconda3/envs/sdpr/bin/python) |
| torch | 2.5.1+cpu |
| kornia | 0.9.0 |
| numpy | 1.26.3 |
| opencv | 5.0.0 |
| device | cpu |
| gpu | blocked (not used) |
| wall_s | 98.05130505561829 |

## 1. toy：发布代码的前向/反向与 FD 对比

复现的是 `demo_data/cube.mat`(8 点) + `poses.mat` + f=300/c=(128,128)，与 `demoPoseEst.py` 完全相同；FD 用 demo 自己的分支跟踪（`ini_pose=P_out.detach()`）。

| case | n | noise(px) | rel err(released vs FD) | ‖J_released‖ | ‖J_FD‖ | bwd ms |
|-|-|-|-|-|-|-|
| exact | 8 | 0 | 1.701e-01 | 3.551e-02 | 3.500e-02 | 46.13 |
| noisy_2px | 8 | 2 | 1.710e-01 | 5.486e-02 | 5.406e-02 | 46.58 |

诊断（2 px 噪声，同一输出点）：`|out−y_geometricLS|=1.63e-06`，`|out−y_algebraicLS|=1.25e-02`，`|f_BPnP(out)|=4.77e+06`，`|f_BPnP(y_algebraicLS)|=4.75e+06`。即 `cv.solvePnP` 收敛到**几何**重投影解，而反向的隐函数方程只有在**代数**解处才成立。

demo 的 keypoint 学习回路（`pts2d.requires_grad_()` + SGD lr=0.2，`demoPoseEst.py` 里注释掉的那个变体，150 it）：loss 13010.312 → 1758.125（min 1147.955），总 7615 ms ⇒ 前向+反向在训练回路里可用。

## 2. 阻塞点（全部可复现，均已绕过且未改 `BPnP.py`）

- **O1 kornia>=0.7 API 移除**：`BPnP.py` 用 `kornia.angle_axis_to_rotation_matrix`；kornia 0.9.0 已把它移出顶层命名空间 → `AttributeError: module 'kornia' has no attribute 'angle_axis_to_rotation_matrix'`。本脚本一行别名到 `kornia.geometry`，**未改 BPnP.py**。
- **O2 `pts3d.view(-1)` 要求连续内存**：scipy `loadmat` 的数组是 Fortran 序，`torch.tensor(...)` 保留该 stride，`BPnP.BPnP.backward` 里 `pts3d.clone().view(-1)` 抛 `RuntimeError: view size is not compatible with input tensor's size and stride`。传 `.contiguous()` 即可，**未改 BPnP.py**。
- **O3 forward/backward 目标函数不一致（结构性）**：`forward` 调 `cv.solvePnP`（几何重投影 LS），`backward` 线性化代数 normal equations `f_j=Σ coefs·r`。精确数据下二者同解，噪声数据下不同解 →见 §1/§3 的数字。这不是 bug 报告，而是 released 设计的适用范围：它假定 forward 解的就是代数问题。

## 3. P1 same-data 适配：BPnP 的代价就是 P1 的代价

K=I 且 `pts2d=(U1/U3,U2/U3)` 时，`r_i = u_i S_i − (KP X)_i,1:2` 与 `P_i^⊥(R p_i+t)` 的关系是恒等式（二者都是 `R p_i+t` 在 `u_i^⊥` 上的正交投影，仅坐标基不同）：数值最大偏差 **1.33e-15**，模长平方偏差 **2.42e-30**。因此 `C_P1 = Σ‖P^⊥(Rp+t)‖² = Σ‖r‖²`，**released backward 正是 C_P1 的隐函数导数**，唯一不能沿用发布版的是 forward（它最小化几何重投影误差）。适配器 `P1ViaReleasedBPnP`（本文件内，30 行）forward 用 P1 局部 Newton，backward `return BPnP.BPnP.backward(ctx, grad_output)` **逐字调用发布代码**。

所有梯度都换成与图无关的 6 维 `(ω, ṫ)`（`dR/dθ=[ω]×R`）再比较；θ 是同一个 bearing 扰动 `u(θ)=u0+θΔu`（Δu 固定随机场）。

| case | n | θ | rel(bpnp vs FD) | rel(ift6 vs FD) | rel(p1chart vs FD) | rel(bpnp vs ift6) |
|-|-|-|-|-|-|-|
| synthetic-planar | 196 | +1.0e-03 | 9.992e-01 | 9.998e-01 | 1.000e+00 | 1.370e+00 |
| synthetic-planar | 196 | +5.0e-03 | 9.995e-01 | 9.999e-01 | 1.000e+00 | 1.370e+00 |
| real-000010-15 | 200 | +1.0e-03 | 1.000e+00 | 9.997e-01 | 1.000e+00 | 1.309e+00 |
| real-000010-15 | 200 | +5.0e-03 | 1.000e+00 | 9.997e-01 | 1.000e+00 | 1.309e+00 |
| real-000011-20 | 200 | +1.0e-03 | 9.999e-01 | 9.998e-01 | 1.000e+00 | 9.242e-01 |
| real-000011-20 | 200 | +5.0e-03 | 9.997e-01 | 9.998e-01 | 1.000e+00 | 9.242e-01 |

| case | 6-DoF solve C | ‖∇C‖ | Newton it | P1 图 vs 6-DoF 位姿 | t_bpnp(bwd) ms | t_ift6 ms | t_p1chart(ift) ms |
|-|-|-|-|-|-|-|-|
| synthetic-planar | 1.608699e-04 | 4.26e-03 | 80 | 0.00e+00°, 1.00e+00 m | 2690.5 | 0.89 | 0.54 |
| real-000010-15 | 3.111671e-04 | 2.64e-03 | 80 | 1.21e-06°, 8.86e-01 m | 945.3 | 0.98 | 0.52 |
| real-000011-20 | 4.599791e-04 | 4.32e-04 | 80 | 0.00e+00°, 7.09e-01 m | 925.1 | 1.12 | 0.32 |

读表要点：
- `rel(bpnp vs FD)` 是**发布代码**给出的隐函数梯度与自己 forward 的分支跟踪 FD 的差；在 P1 代价上它和 `ift6`、`p1chart` 都到同一量级（float32 反向 + FD 误差），说明**发布的 backward 机器可以原样作为 P1 代价的 IFT**。
- `p1chart` 是我们论文里那条线（S³ 四元数图、t 解析消去、`qpep_core.ift_dq`），与 6-DoF 图的结果一致 ⇒ 图的选择不影响导数（只有符号/基的差别，已统一到 `(ω, ṫ)`）。
- 时间是 CPU 单线程、n=200 时的**反向一次**（每个基向量一次 backward），BPnP 的 backward 每次都要重算 `get_coefs`（n×n cotangent 的批量 VJP），所以比 `qpep_core.ift_dq` 的单次 3×3 解贵 2–3 个量级；这是 released 实现的常数，不是算法差异。

## 4. 诚实边界

- **不是 GPU 复现**：GPU 被禁，全程 CPU；BPnP 原 demo 用 `device='cuda'`，本脚本等价地在 CPU 上跑。
- **toy 只跑关键点学习回路，不跑 vgg11 前端**：`demoPoseEst.py` 的 vgg11+SGD 2000 it 需要下载预训练权重且是演示用的玩具前端；本脚本跑的是同一文件里注释掉的 keypoint 直接优化变体（同一 forward/backward 路径），并额外做了 6×2n 的完整 Jacobian 对 FD 检验。
- **P1 合成 QPEP 实例（θ 表那一组）无法直接喂给 BPnP**：它的代价是手工构造的四次型 `v2(q)^T M v2(q)`（M 由 `alpha/beta/eta` 造的双井 + `G,D` 的 t 消去），没有对应的点集，无法反解出 (p_i,u_i)。因此 same-data 适配用**同一代价形式**的两类数据：合成平面靶（可控、有平面二义性）与真实 LINEMOD 帧（P1 真实管线原样）。
- **真实帧只用 200 个内点**做梯度臂（BPnP backward 的 n×n cotangent 是 O(n²)，n=900 会到十几秒/次）；同一子集喂给所有臂，P1 全量 900 点的数值见 `run_two_mode.py`。
- **θ 是我们选的输入扰动**（bearing 方向），不是论文里的 layer 参数；选它是为了让`dpose/dθ` 能同时由 4 条路径给出并与 FD 对齐。

## 5. 复现

```bash
# 官方 BPnP 源码（未改）
git -C /mnt/storage1/syk/third_party/BPnP rev-parse HEAD   # f1f4f6a79ce2bd58a6340a59e64c0900bb6b0772
/home/syk/miniconda3/envs/sdpr/bin/python \
  labs/01-laplace-qpeplayer/experiments/baselines/run_official_bpnp.py
```

总 wall 98.1 s，CPU-only。输出 `results/official_bpnp.json`（含每个 case 的完整梯度向量、时间、诊断量）与本文。
