# 官方 BPnP（BoChenYS/BPnP）CPU 复现：toy 梯度检验 + P1 same-data 适配

> 证据等级：**独立复现**（本机 CPU 跑的是**发布的原始代码**，未改一行 `BPnP.py`；两个环境/布局问题用一行 shim / `.contiguous()` 解决，见 §2）。
>
> **2026-11-28 更正**：本文 §3 的梯度表与"四者一致"的结论**已撤回**，由 §4「Gradient re-adjudication (2026-11-28)」取代。§0 的三条结论按 §4 的实测重写；旧表保留在 §3 仅作已发布记录，**不要引用**。

## 0. 一句话结论（按 2026-11-28 重判更新）

- **能在 CPU 上跑**：repo 是纯 PyTorch + OpenCV + kornia，`BPnP.BPnP.apply` 前向/反向都跑通（toy n=8，反向 40–46 ms/6 维基向量）。
- **它自己的 forward/backward 不一致，而且不是噪声问题**：toy 上 released backward 与 released forward 的中心差分相对误差 **17.01%**（**精确**关键点）、**17.10%**（2 px 噪声）——精确数据并没有让误差消失。根因不是实现 bug，而是**两个不同的目标函数**：`forward` 调 `cv.solvePnP`（几何重投影 LS），`backward` 线性化的是 `f_j = Σ coefs·r`，即**深度加权**的归一化重投影代价 `Σ_i S_i‖e_i‖²` 的驻点方程（§4.1/§4.3）。
- **发布的 backward 不是 P1 代价的 IFT**：把前向换成代数 LS 解（`min Σ‖r‖²`）后，released backward 与该前向的中心差分仍差 **16.77%**（demo cube，8 点，精确数据，§4.3）——"四者一致"撤回。它对自己那组方程 `f=0` 是精确的（对 f-根前向的 FD：**1.6e-07**；对 f 的精确 IFT：**1.6e-07**）。
- **P1 same-data 表的 1.37/1.31/0.92 主要是基错误**：`g_bpnp` 是 `d(rvec,t)/dθ`（参数基），发布的 `g_ift6` 是切空间 `ω`；同一向量的基因子本身就是 **1.37/1.26/0.76**。发布解的 `‖∇C‖` 还有 4e-3/3e-3/4e-4（未收敛），FD 参考比两条梯度臂大 ~1000×（失效）。**同基 + 收敛解**重测：released vs 我们的 6-DoF IFT = **1.5% / 9.4% / 30%**（cos 0.9999/0.9956/0.9661），我们的 IFT vs 分支跟踪 FD = **5e-6 / 2e-5 / 3e-4**（§4.4）。
- **"代价恒等式 1.8e-15" 是把 0 和 0 相比**：在生成 `u` 的那个 pose 上 `P^⊥X` 与 `r` 都恒为 0。正确关系是逐点 `‖r‖² = (1+‖u‖²)‖P^⊥X‖² − ‖u×X_12‖²`（数值 5e-15 / 1e-13），代价只在残差一阶意义下相同，梯度并不相同（§4.5）。

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

诊断（2 px 噪声，同一输出点）：`|out−y_geometricLS|=1.63e-06`，`|out−y_algebraicLS|=1.25e-02`，`|f_BPnP(out)|=4.77e+06`，`|f_BPnP(y_algebraicLS)|=4.75e+06`。即 `cv.solvePnP` 收敛到**几何**重投影解，而反向的隐函数方程 `f=0` 的解不在这个点上；上面两个 `|f|` 都是 1e6 量级，**不能**用来说明"到代数解就一致"——精确/收敛的判据见 §4.3（`|f|` 在它自己的根上是 1e-10 量级）。

demo 的 keypoint 学习回路（`pts2d.requires_grad_()` + SGD lr=0.2，`demoPoseEst.py` 里注释掉的那个变体，150 it）：loss 13010.312 → 1758.125（min 1147.955），总 7615 ms ⇒ 前向+反向在训练回路里可用。

## 2. 阻塞点（全部可复现，均已绕过且未改 `BPnP.py`）

- **O1 kornia>=0.7 API 移除**：`BPnP.py` 用 `kornia.angle_axis_to_rotation_matrix`；kornia 0.9.0 已把它移出顶层命名空间 → `AttributeError: module 'kornia' has no attribute 'angle_axis_to_rotation_matrix'`。本脚本一行别名到 `kornia.geometry`，**未改 BPnP.py**。
- **O2 `pts3d.view(-1)` 要求连续内存**：scipy `loadmat` 的数组是 Fortran 序，`torch.tensor(...)` 保留该 stride，`BPnP.BPnP.backward` 里 `pts3d.clone().view(-1)` 抛 `RuntimeError: view size is not compatible with input tensor's size and stride`。传 `.contiguous()` 即可，**未改 BPnP.py**。
- **O3 forward/backward 目标函数不一致（结构性）**：`forward` 调 `cv.solvePnP`（几何重投影 LS），`backward` 线性化 `f_j=Σ coefs·r`，即**深度加权**的归一化重投影驻点方程 `Σ_i S_i∇‖e_i‖²`。精确数据下两者**解点相同**（`r=0`），但**解映射不同**，所以 17% 的误差在精确数据上照样存在（§4.3 用 FD 直接测到：对代数 LS 前向 16.77%，对它自己的 f-根前向 1.6e-07）。这不是 bug 报告，而是 released 设计的适用范围：它假定 forward 解的就是它自己那组方程。

## 3. P1 same-data 适配（**已撤回，见 §4**）

> **本节作废（2026-11-28）**。下面这段文字与两张表是**已发布的原始记录**，其中
> "released backward 正是 C_P1 的隐函数导数"、"四者一致"两处结论是错的：
> (a) 成本恒等式只在残差一阶意义下成立、且发布时的 1.8e-15 是 0 比 0（§4.5）；
> (b) 发布的 `g_ift6` 是切空间 `ω`，`g_bpnp` 是参数速率 `ṙ`，两者不在同一基（§4.1）；
> (c) 表里的解并未收敛（`‖∇C‖` 仍有 4e-3/3e-3/4e-4），FD 参考比两条梯度臂大 ~1000×，
> 且整表**不能**由仓库里的脚本重算（§4.2）。修正后的数字见 §4.4，**引用请用 §4**。

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

## 4. Gradient re-adjudication (2026-11-28)

> 新增脚本 `baselines/recheck_official_bpnp_gradient.py`（CPU ≈11 s，未改任何已有 `.py`）。
> 原始记录 `results/official_bpnp_recheck.json`；同一份内容写进 `results/official_bpnp.json` 的
> `gradient_recheck` 块。证据等级：**独立复现 + 数值判定**。

### 4.0 结论

1. 发布的 backward 是它**自己那组方程** `f=0` 的精确 IFT：在 demo cube 上对 f-根前向的 FD 差 **1.55e-07**，对 f 的精确 IFT 差 **1.65e-07**（float32 量级）。
2. 但它**不是** P1 代价 `C=Σ‖r‖²` 的 IFT，也不是几何前向的 IFT：把前向换成**代数 LS 解**（`min Σ‖r‖²`）后，released backward 与该前向的 FD 仍差 **16.77%**（cube，8 点，精确数据）。**"四者一致"撤回**。
3. 发布表的 `rel(bpnp vs ift6) = 1.37/1.31/0.92` **主要是基错误**：`g_bpnp` 是参数速率 `ṙ = d(rvec,t)/dθ`，发布的 `g_ift6` 是切空间 `ω`（两者差一个左 Jacobian `J_l(w)`，`|w|≈2.7 rad`）。同一向量的基因子本身就 **1.37/1.26/0.76**。
4. 同基 + 收敛解重测：released vs 我们的 6-DoF IFT = **1.5% / 9.4% / 30%**（cos 0.9999/0.9956/0.9661）；我们的 IFT vs 分支跟踪 FD = **5e-6 / 2e-5 / 3e-4**。残余差距可分解为"深度加权的目标差"+ "发布实现的 float32 精度"，随 case 变，**不是**同一个 6-DoF IFT。
5. 成本恒等式不能按发布的方式成立：1.8e-15 是 0 比 0（§4.5）。

### 4.1 `g_bpnp` 是什么（参数化 / 基审计）

`BPnP.BPnP.forward` 返回 `P_6d = [rvec(3); t(3)]`，`BPnP.BPnP.backward` 返回 `dL/dpts2d = grad_output · d(rvec,t)/dpts2d`。所以

- `g_bpnp = d(rvec, t)/dθ` —— **参数基**（旋转用 Rodrigues 向量本身求导）；
- 我们论文里的 6-DoF 梯度是**切空间** `(ω, ṫ)`，`dR/dθ = [ω]×R`；两者关系 `ω = J_l(w)·ṙ`，`J_l` 是 SO(3) 左 Jacobian。

数值验证（`A_basis`）：

| 检查 | 值 |
|-|-|
| `dR/dw_a` vs `[J_l(w)e_a]×R`（正确） | **1.99e-09** |
| `dR/dw_a` vs `[e_a]×R`（切向导数，‖w‖=2.7） | 9.95e-01 |
| 基线 `p1_cost_derivs` 的 J vs 参数基 J·blkdiag(J_l⁻¹,I) | **1.40e-09**（即它确实是切空间 J） |

| case | ‖w‖ (rad) | max&#124;J_l−I&#124; | cond J_l |
|-|-|-|-|
| synthetic-planar | 2.713 | 0.806 | 1.39 |
| real-000010-15 | 2.796 | 0.822 | 1.42 |
| real-000011-20 | 2.662 | 0.827 | 1.37 |

⇒ 发布的 `g_ift6` 列是**切空间**量，`g_bpnp` 是**参数**量；发布文里"所有梯度都换成与图无关的 6 维 (ω,ṫ)"对 bpnp 这一臂不成立。

### 4.2 复现（哪些能对上、哪些对不上）

| 项 | 结果 |
|-|-|
| toy exact | 1.70128991e-01，与 JSON 完全一致（差值 0.0） |
| toy noisy_2px | 1.70969274e-01，差值 0.0 |
| 发布表的 `g_ift6` 列（3 个 case） | 用 JSON 里存的 pose 重算，rel = **0.0**（逐位一致） |
| 整表能否由仓库脚本重算 | **不能**：`P1ViaReleasedBPnP.forward` 给 `p1_solve` 传了不存在的 `iters/tol` → `TypeError: p1_solve() got an unexpected keyword argument 'iters'` |
| 当前 `p1_solve` 的解 | C = **6.63e-02**、`‖∇C‖`(切向) = **7.10**，而 JSON 记的是 C = 1.61e-04、`‖∇C‖` = 4.26e-03 |
| `p1_solve` 给 `least_squares` 的 Jacobian | **错排**：`J.transpose(0,2,1).reshape(-1,6)`（应为 `J.reshape(-1,6)`） |

另外，发布表自己的 `solve.grad_norm` = 4.26e-03 / 2.64e-03 / 4.32e-04 已经说明那三个"解"都不在驻点（正确解 `‖∇C‖ ≤ 1e-13`，见 §4.4）。

### 4.3 受控实验：前向 = 代数 LS 解（demo cube，8 点，精确数据）

问题：把 forward 换成 `min Σ‖r‖²` 的解（`r = u·S − (K X)_12`，K=f·I）之后，released backward 与这个前向的 FD 一致吗？

| 量 | 值 |
|-|-|
| 代数 LS 解 | C = 8.9e-09、`‖∇C‖` = 2.2e-10、‖y−y_gt‖ = 8.2e-08 |
| FD 步长研究 `‖J_FD‖`（h = 1e-3/1e-4/1e-5 px） | 2.12584e-02 / 2.12584e-02 / 2.12584e-02（平台） |
| 精确 IFT(`∇C=0`) vs FD（FD 自检） | **6.9e-10** |
| **released vs FD(代数 LS 前向)** | **1.677e-01** |
| released vs 精确 IFT(`∇C=0`) | 1.677e-01 |
| f-根前向（解 `f=0`）的解残差 &#124;f&#124; | 8.3e-11 |
| **released vs FD(f-根前向)** | **1.551e-07** |
| released vs 精确 IFT(f) | 1.645e-07 |
| 精确 IFT(f) vs FD(f-根前向) | 3.4e-08 |
| 深度 spread `(S_max−S_min)/S_mean` | 0.611 |

读法：
- **"设成代数解就一致"不成立（16.77%）**。原因不是实现错误，而是"同根 ≠ 同映射"：精确数据上 `f=0` 与 `∇C_alg=0` 都在真解处成立，但它们是**两个不同的方程**，解映射不同。`f` 线性化的是**深度加权**的归一化重投影驻点 `Σ_i S_i‖e_i‖²`（`coefs = −2∂p̂/∂y`，而 `r = S(u−p̂)`），代数代价是 `Σ‖r‖² = Σ S_i²‖e_i‖²`；cube 的深度从 3.25 跨到 6.11，所以 17%。
- **对"它自己那组方程"，released backward 是精确的**（1.6e-07）：这才是"它自己的假设成立"时的受控检验，也应该作为对发布实现的正面结论。

### 4.4 P1 same-data 重判（同基 = 参数基，收敛解）

每个 case 先解到 `‖∇C‖ ≤ 1e-13`（LM + 精确 Hessian 抛光），再在**同一 6 维参数基**里比较三条臂；θ=1（线性缩放）。

| case | n | 深度 spread | C | ‖∇C‖ | rel(released, FD) | rel(ours, FD) | rel(released, ours) | cos(released, FD) |
|-|-|-|-|-|-|-|-|-|
| synthetic-planar | 196 | 0.167 | 1.448e-04 | 7.9e-14 | 1.537e-02 | 5.07e-06 | 1.537e-02 | 0.99988 |
| real-000010-15 | 200 | 0.079 | 3.104e-04 | 3.7e-14 | 9.363e-02 | 2.48e-05 | 9.364e-02 | 0.99562 |
| real-000011-20 | 200 | 0.107 | 4.600e-04 | 1.7e-15 | 2.985e-01 | 3.23e-04 | 2.988e-01 | 0.96609 |

把 30%/9%/1.5% 的差距拆开（`released vs FD` ≈ `IFT(f) vs FD` ⊕ `released vs IFT(f)`）：

| case | 目标差 `rel(IFT(f), FD)` | 实现精度差 `rel(released, IFT(f))` | cond(f_y) |
|-|-|-|-|
| synthetic-planar | 1.542e-02 | 5.53e-04 | 1.39e+03 |
| real-000010-15 | 8.895e-02 | 7.34e-03 | 1.51e+04 |
| real-000011-20 | 2.610e-01 | 1.127e-01 | 9.11e+04 |

- 第一项 = **深度加权目标差**（released 微分 `Σ S_i‖e_i‖²`，我们微分 `Σ S_i²‖e_i‖²`），随几何变，不是常数；
- 第二项 = **float32 实现的精度损失**：`J_fy` 的条件数从 1.4e3 涨到 9.1e4，发布的 float32 autograd + `torch.inverse` 就跟不上了（float64 复算同一 IFT 只差 1e-5 量级）。

发布值 vs 修正值：

| case | 发布 rel(bpnp, ift6) | 同 pose 基因子 rel(g, J_l g) | 发布 ‖∇C‖ | 发布 rel(bpnp, FD) | 修正 rel(bpnp, FD) (同基) |
|-|-|-|-|-|-|
| synthetic-planar | 1.3704 | 1.3726 | 4.26e-03 | 0.9992 | 1.537e-02 |
| real-000010-15 | 1.3093 | 1.2625 | 2.64e-03 | 1.0000 | 9.363e-02 |
| real-000011-20 | 0.9242 | 0.7625 | 4.32e-04 | 0.9999 | 2.985e-01 |

⇒ 发布的 1.37/1.31/0.92 与"纯基因子"1.37/1.26/0.76 同量级；其中 real-000011-20 的 0.92 与 0.76 差得较多，来自未收敛解。发布表里 `rel(*, FD) ≈ 0.999` 是因为 FD 用了一个**未收敛的求解器**（`‖∇C‖` 4e-3 量级、h=5e-5 太小），其方向随 θ 翻转（θ=1e-3 与 5e-3 的 `cos(ift6,FD)` 分别为 0.34 与 0.03），量级比两条梯度臂大 ~1000×。

基线 helper 在收敛点的对照（说明"映射后是同一个量"）：

| case | rel(helper, J_l·ours) | rel(helper 映射回参数基, ours) |
|-|-|-|
| synthetic-planar | 3.87e-03 | 2.93e-03 |
| real-000010-15 | 4.19e-02 | 3.90e-02 |
| real-000011-20 | 1.29e-01 | 1.17e-01 |

（残余的 0.4–13% 出在 helper 自己的 δ 参数化 Hessian/二阶项上，不是基的问题——它的 J 已单独核对为切空间精确，1.40e-09。）

**判定**：P1 适配的梯度是**可修正、不是"四者一致"**——同基、收敛后 released 与我们的 IFT 在 synthetic 上差 1.5%，在真实帧上差 9%/30%；发布表的三条臂之所以互不一致，是基错误 + 未收敛解 + 失效 FD 三者叠加。

### 4.5 成本恒等式（与梯度比较分开记账）

- 发布写法（`main()` 里的 `max_abs`/`max_abs_sq` = 1.33e-15 / 2.42e-30）在**生成 u 的那个 pose** 上计算：那里 `X ∥ û`，所以 `P^⊥X ≡ 0`、`r ≡ 0`（实测 max 4.4e-16 / 2.2e-16）——**0 比 0**。
- 正确关系（逐点，`u` 为归一化像坐标、`a=‖u‖²`、`c=‖X_12‖²`、`b=u·X_12`）：

  `‖r‖² = (1+a)·‖P^⊥X‖² − (a·c − b²)`，即 `‖r‖² = (1+‖u‖²)‖P^⊥X‖² − ‖u×X_12‖²`

  数值残差：synthetic **4.9e-15**，real-000010-15 **1.0e-13**。
- 逐点比值 `‖r‖²/‖P^⊥X‖²`：synthetic 1.0000–1.0056（`1+‖u‖²≈1.0014`），real-000010-15 1.0000–1.0737（`1+‖u‖²≈1.0508`）。
- 所以"P1 的 bearing 代价 = BPnP 的代数代价"只在**残差一阶**意义下成立；发布把它当成恒等式、再推成"梯度相同"，是两处跳跃。

### 4.6 复现

```bash
/home/syk/miniconda3/envs/sdpr/bin/python \
  labs/01-laplace-qpeplayer/experiments/baselines/recheck_official_bpnp_gradient.py
# ≈11 s，CPU-only；--quick 只跑 toy+cube，--no-real 跳过 LINEMOD
```

输出：`results/official_bpnp_recheck.json`（完整原始记录，含每个 case 的 pose、三条梯度向量、cond 数、provenance sha256）与 `results/official_bpnp.json::gradient_recheck`。

### 4.7 证据边界

- 只重判**梯度**；toy 的 vgg11 学习回路没有重跑（§1 的 150 it 循环仍是原记录）。
- 只覆盖发布用的 3 个 case（synthetic-planar + 2 帧 LINEMOD，n=196/200），没有扩大样本。
- `f = −2Σ_i A_i^T r_i` 的"深度加权"解释是从 `get_coefs`/`backward` 源码读出的（`coefs=−2∂p̂/∂y`、`r=S(u−p̂)` ⇒ `f/2 = −Σ S_i ∇D_i`），并用 f-根前向的 FD 验证（1.6e-07）；没有做符号层面的形式化证明。
- cube 的 `u` 来自 BPnP 自己的 float32 `batch_project`，所以代数最小残差是 8.9e-09 而不是 0；这不影响 1e-3 量级的结论。
- 未测：`p1chart`（`qpep_core.ift_dq`）这一臂没有重跑，§3 表里它的列一并不再引用。

## 5. 诚实边界

- **不是 GPU 复现**：GPU 被禁，全程 CPU；BPnP 原 demo 用 `device='cuda'`，本脚本等价地在 CPU 上跑。
- **toy 只跑关键点学习回路，不跑 vgg11 前端**：`demoPoseEst.py` 的 vgg11+SGD 2000 it 需要下载预训练权重且是演示用的玩具前端；本脚本跑的是同一文件里注释掉的 keypoint 直接优化变体（同一 forward/backward 路径），并额外做了 6×2n 的完整 Jacobian 对 FD 检验。
- **P1 合成 QPEP 实例（θ 表那一组）无法直接喂给 BPnP**：它的代价是手工构造的四次型 `v2(q)^T M v2(q)`（M 由 `alpha/beta/eta` 造的双井 + `G,D` 的 t 消去），没有对应的点集，无法反解出 (p_i,u_i)。因此 same-data 适配用**同一代价形式**的两类数据：合成平面靶（可控、有平面二义性）与真实 LINEMOD 帧（P1 真实管线原样）。
- **真实帧只用 200 个内点**做梯度臂（BPnP backward 的 n×n cotangent 是 O(n²)，n=900 会到十几秒/次）；同一子集喂给所有臂，P1 全量 900 点的数值见 `run_two_mode.py`。
- **θ 是我们选的输入扰动**（bearing 方向），不是论文里的 layer 参数；选它是为了让 `dpose/dθ` 能由多条路径给出并与 FD 对齐。注意：能这样比的前提是**先把基统一**——发布版 backward 给的是参数速率 `ṙ`，我们的 6-DoF IFT 给的是切空间 `ω`（§4.1）。

## 6. 复现

```bash
# 官方 BPnP 源码（未改）
git -C /mnt/storage1/syk/third_party/BPnP rev-parse HEAD   # f1f4f6a79ce2bd58a6340a59e64c0900bb6b0772

# 原复现（toy + 旧 P1 适配；产出 §1/§3 的记录，§3 的梯度表已作废）
/home/syk/miniconda3/envs/sdpr/bin/python \
  labs/01-laplace-qpeplayer/experiments/baselines/run_official_bpnp.py

# 梯度重判（§4，≈11 s；产出 official_bpnp_recheck.json 与 official_bpnp.json::gradient_recheck）
/home/syk/miniconda3/envs/sdpr/bin/python \
  labs/01-laplace-qpeplayer/experiments/baselines/recheck_official_bpnp_gradient.py
```

原脚本总 wall 98.1 s，重判脚本 ≈11 s，均 CPU-only。输出 `results/official_bpnp.json`（含每个 case 的完整梯度向量、时间、诊断量，以及 `gradient_recheck` 块）、`results/official_bpnp_recheck.json` 与本文。
