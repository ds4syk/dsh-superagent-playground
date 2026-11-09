# 独立 clean-room 复现报告：SDPRLayers 相机基线标定仿真（TRO 2025, §VI-B4, Fig. 4 / Table V）

> 复现者：repro-clean（task-3）。本报告的所有数字均来自本机 `repro/results_clean/` 下的
> 日志/CSV/JSON 文件，可用第 A 节的命令重新生成。论文引用格式：`[Holmes2025, §VI-B4]`
> 或 `paper/sdpr.txt:L912-L936`。
>
> **独立性声明**：`repro/cleanroom/` 下的所有代码均为本报告作者从论文公式独立推导实现，
> **没有 import 上游 `sdprlayer` 包**，也不依赖 `poly_matrix` / `theseus` / `pylgmath` /
> `cvxpylayers`。上游源码只用于核对问题定义（Problem (P6)、Q 矩阵分块、冗余约束清单、
> `_scripts/stereo_cal.py` 的实验配置）。

---

## 0. 摘要（关键结论）

| 结论 | 状态 |
|---|---|
| SDPR-IS 与「GT 初始化的局部解」在 50 runs × 20 poses 上轨迹几乎完全重合 | ✅ 最大逐迭代差 **1.08e-6 m**（论文称 "very close"） |
| 最终 baseline 误差、外层 loss 与论文同量级 | ✅ 4.1e-4（论文 3.45e-4）、7.5e-3~1.25e-2（论文 9.86e-3） |
| 随机初始化局部解灾难性失败（错误梯度 → 标定发散） | ✅ 0.327~0.338 m（论文 0.113 m），方向一致 |
| SDP 松弛紧性 | ✅ **108420 次内层求解 100% rank-1**，λ1/λ2 最小 4.7e8；证书 corank(1e-9)=1 占 99.97%，H⪰0（λmin≥-8e-10），‖Hx‖≤1.3e-8，primal=dual |
| 随机初始化收敛到局部极小值的比例 | ✅ 三个数据集 **40% / 67% / 70%（均值 59%）**，失败 0%，论文 "roughly 60%" |
| Thm.2 隐式 Jacobian 正确性 | ✅ 50 trials vs 有限差分：ΔJ = **7.2e-6 ± 1.2e-5**（SDPR-IS）/ 4.5e-6（local-GT）；两法互相差 3.1e-6 |
| 迭代次数 54 vs 论文 34，RND 误差 0.33 vs 论文 0.11 | ⚠️ 未完全复现，见 §9 差异归因 |
| 资源 | 单 run（单进程）：SDPR **15.5 s** / local-GT **3.3 s** / local-RND **11.6 s**；峰值 RSS **298 MB**；纯 CPU（1 线程/进程）；全量 300 runs ≈1.9 CPU·h |

**总体判断**：Section VI-B4 的**科学结论被独立复现**（SDPR 全局解 = GT 初始化局部解，
随机初始化局部解给出错误梯度导致标定失败；松弛 100% 紧、证书成立），**绝对数值在 20–45%
内吻合**，但与论文仍有两处系统性差异（外层迭代数、RND 最终误差量级），已在 §9 逐条归因。

---

## 1. 复现目标与范围

复现论文 Section VI-B4 "Baseline Calibration"：

* 内层：Problem (P6) 矩阵加权双目位姿估计（N_m = 64 个特征，20 个位姿）；
* 外层：Problem (P7) 用外层 loss 标定双目基线 `b`（初值 `b_true + 0.003 m`）；
* 三种内层求解器：`SDPR-IS`（Shor SDP 松弛 + Thm.2 隐式微分）、`local-GT`（GT 初始化
  Gauss-Newton + 局部解的隐式微分）、`local-RND`（论文 Fig. 3b 随机初始化 + 隐式微分）；
* 50 runs × 20 poses，SGD lr = 1e-4，终止条件 `|grad|² < 1e-5` 或 150 次外层迭代；
* 产出 Fig. 4 式 50 条参数误差轨迹、Table V 式统计、tightness 统计、Jacobian 校验、
  Fig. 3b 式局部极小值实验、资源开销。

**不做**的部分（论文其他章节）：VI-A 六次多项式 bilevel、VI-C In-The-Dark 神经网络训练、
Table III/IV 中 Theseus-GT-UR（unrolling）与 SDPR-SDP（cvxpylayers）两种对照方法（本文
只实现论文默认的 SDPR-IS 与局部解基线，另外用有限差分作为"真值"参考）。

---

## 2. 实现与论文公式的对应关系

代码位置 `repro/cleanroom/`（约 2600 行，含注释与自检）。下表给出论文公式 → 实现的逐项映射。

| 论文内容 | 实现文件 / 函数 | 说明 |
|---|---|---|
| Eq. (27) 逆双目相机模型 | `camera_model.StereoCamera.inverse` / `inverse_meas_only` | `m_k = (b/d_k)[(u_k-c_u), (v_k-c_v)f_u/f_v, f_u]^T`，`d_k = u_l - u_r` |
| Eq. (28) 欧氏测量噪声 Σ_k | `camera_model.inverse` + `inverse_jacobian` + `pixel_covariance` | `G_k = ∂m_k/∂(u_l,v_l,d)`，`Σ_k = G_k Σ_pix G_k^T`，`W_k = Σ_k^{-1}` |
| 像素噪声协方差 | `pixel_covariance` | `Σ_pix = [[σ_u²,0,σ_u²],[0,σ_v²,0],[σ_u²,0,2σ_u²]]`（左右目独立噪声 ⇒ var(d)=2σ_u²） |
| Problem (P6) | `sdp_pose.build_objective_matrix`, `local_solver` | 取 `t = C r_p`（相机坐标系下平移），`cost = Σ‖y_k - C m_k + t‖²_{W_k}`，与论文 `e_k = ỹ_k - C(m_k + t_paper)`（`t_paper = -r_p`）等价 |
| Problem (P1) 齐次 QCQP | `sdp_pose.N_VARS`, `constraint_matrices`, `homogenising_matrix` | `x = [h, vec_col(C), t] ∈ R^13`；24 个 `x^T A_i x = 0` 约束 + `x^T A_0 x = 1`（`A_0 = e_h e_h^T`） |
| 6 个正交约束 | `constraint_matrices()[0:6]` | `C_i·C_j - δ_ij h² = 0` |
| 9 个 handedness 约束 | `constraint_matrices()[6:15]` | `(C_j × C_k)_l - h C_li = 0`（循环下标），即 `adj(C) = C^T`（det = +1） |
| 9 个行/列范数约束 | `constraint_matrices()[15:24]` | `‖col_j(C)‖² - ‖row_i(C)‖² = 0`（冗余约束，用于收紧松弛） |
| Problem (P3) Shor 松弛 | `sdp_pose.SDPPoseSolver` | CVXPY：`min ⟨Q_s,X⟩` s.t. 25 个等式约束 + `X ⪰ 0`，13×13；SCS，`eps_abs=eps_rel=1e-9` |
| 对偶证书 H | `SDPPoseSolver.solve` | 由等式约束对偶 `λ_i, λ_0` 组装 `H = Q + Σ λ_i A_i + λ_0 A_0`；`λ_0 = -primal` |
| 紧性判据 | `solve` 中 `r_tight = λ1(X)/λ2(X)` | 论文 Appendix C：`r > 1e5` 视为 rank-1 |
| Thm.2 / Eq. (13)(14)(16)(17) | `implicit_diff.qcqp_jacobian`, `reduced_kkt_matrix` | `M_r = [[H, G_r^T],[G_r, 0]]`，`J = -P M_r^{-1} N`；`G_r` = 6 正交 + 齐次约束行（共 7 行，`rank=7=codim Ω`，满足 ACQ） |
| Eq. (20) 反向传播 | `implicit_diff.qcqp_backward` | `y = M_r^{-1}[∇_xℓ; 0]`，`∇_bℓ = -x^T (dQ/db) y_1`（等价于 Eq. (20) 的 `-N^T y`） |
| Eq. (29) ΔJ 指标 | `implicit_diff.relative_jacobian_difference` | `‖J_est - J_ref‖∞/‖J_ref‖∞` |
| Problem (P7) 外层 loss | `implicit_diff.loss_pose_and_grad` / `loss_from_x_and_grad` | `‖r_p-r_gt‖² + ‖C^T C_gt - I‖_F²`（对正交 C 等于 `‖C-C_gt‖_F²`） |
| 外层 SGD | `bilevel.run_trial` | lr = 1e-4，`b_0 = b_true + 0.003`，`grad_sq < 1e-5`，最多 150 iter，逐 run 记录轨迹 |
| 局部解基线（Theseus 对应物） | `local_solver` | SO(3) 上 Levenberg–Marquardt（自适应阻尼），步长 0.2、`rel_tol=1e-6`、≤500 iter（对齐论文 Theseus 配置），之后紧收敛 polish 到驻点 |
| 局部解隐式微分 | `local_solver.local_implicit_derivative` | 驻点条件 `g_δ = 2Σ(W_k e_k)×v_k = 0`, `g_r = 2ΣC^T W_k e_k = 0`；`[dδ;dr] = -(∂g/∂(δ,r))^{-1} ∂g/∂b`；`dC = [dδ]×C`（数值 Jacobian 6×6，已与 FD 校验） |
| Fig. 3b 随机初始化 | `data_gen.get_random_poses` | 位置在半径 3 m 球面上均匀、光轴指向板中心、绕光轴随机 roll |
| 数据生成 | `data_gen.get_calibration_data`, `get_poses_in_cone` | 8×8 板（1.0 m×1.0 m，居中于原点）；位姿：半径 3 m、45° 半锥角、指向板中心 + 绕相机 x 轴 ±45° 扰动 |

### 2.1 关键推导（与上游的差异 / 独立核对点）

1. **Q 矩阵分块**（`x = [h, vec_col(C), t]`，`vec_col` 为列优先）
   ```
   Q_hh = Σ_k ỹ_k^T W_k ỹ_k          Q_CC = Σ_k (m_k m_k^T) ⊗ W_k
   Q_tt = Σ_k W_k                    Q_Ch = -Σ_k m_k ⊗ (W_k ỹ_k)
   Q_th = Σ_k W_k ỹ_k                Q_Ct = -Σ_k m_k ⊗ W_k
   ```
   已在 `smoke_test.py` 中验证 `x^T Q x` 与逐点矩阵加权代价完全一致（相对误差 <1e-10）；
   `t = C r_p` 的符号约定与上游 `SDPPoseEstimator` 一致（`r_p = C^T t`）。
2. **约束 Jacobian 秩**：在可行点处 25 行约束梯度的秩 = **7 = codim(Ω) = 13-6**，
   即 ACQ（Definition 1）成立，18 个冗余约束的梯度落在 7 个独立行的张成空间内。
   与上游 `redun_list = list(range(6, 24))`（保留 6 个正交 + 齐次）一致。
3. **约束矩阵的正确性**：在 20 个随机 SO(3) 位姿上 `max|x^T A_i x| = 3.4e-16`。
4. **证书与强对偶**：`λ_0 = -primal`，`primal = dual`；`H = Q_s + Σλ_iA_i + λ_0A_0 ⪰ 0`，
   `‖Hx‖ ≈ 2.5e-9`。
5. **一个上游小 bug（不影响本文实验）**：`stereo_tuner.StereoCamera.inverse` 中
   `G[1,2] = -y z/(f_v b)`，而由 Eq. (27) 推得应为 `-y z/(f_u b)`（`∂y/∂d = -y/d`，
   `1/d = z/(b f_u)`）。因 Table II 中 `f_u = f_v = 484.5`，该差异对本文实验为零。
   本实现使用推导正确的 `G`。
6. **噪声生成口径**（见 §3.2）：论文/公式（Eq. 28）要求左右目**独立**噪声；
   上游 `StereoCamera.forward` 实际上是 `u_r = u_l - f_u b/z + σ_u n`（复用含噪的左目坐标），
   导致 `var(d) = σ_u²`、`cov(u_l,d) = 0`，与估计器使用的 `Σ_pix`（`var(d)=2σ_u²`）
   不一致。本实现两种口径都跑，以区分"论文口径"和"上游代码口径"。

---

## 3. 实验设置

### 3.1 论文 vs 本实现

| 项目 | 论文 / 上游脚本 | 本实现 | 一致？ |
|---|---|---|---|
| 特征板 | 8×8，1.0 m×1.0 m，中心在原点 | 同 | ✅ |
| 位姿 | 20 个/run，半径 3 m，β=45° 锥内；光轴指向板心 + α=45° FOV 扰动 | 同 | ✅ |
| 相机 | f_u=f_v=484.5 pix，c_u=c_v=0，σ_u=σ_v=0.5 pix，b=0.24 m | 同 | ✅ |
| 内层目标 | Problem (P6) 矩阵加权 | 同 | ✅ |
| SDP 求解器 | SCS，tol 1e-9（论文 §VI-B4）；Table III/IV 用 Mosek 1e-12 | SCS 1e-9（另有 1e-12 仅用于 FD 参考） | ⚠️ 无 Mosek license |
| 局部求解器 | Theseus Gauss-Newton，step 0.2，abs 1e-10，rel 1e-6，≤500 iter，隐式微分 | Levenberg–Marquardt（自适应阻尼），同 step/tol/次数 + 紧收敛 polish；局部驻点隐式微分 | ⚠️ 独立实现 |
| 外层 | SGD lr=1e-4；`b_0=b_true+0.003`；`tol_grad_sq=1e-5`（脚本）/ 正文 \|grad\|<1e-3；≤150 iter | 同（`tol_grad_sq=1e-5` 为主设置，另给阈值灵敏度） | ⚠️ 见 §9.3 |
| fig4 轨迹 | 每 run 一条参数误差曲线，t=0 起点 0.003 | 同 | ✅ |
| 硬件 | 论文未注明（CPU / Mosek） | Intel Xeon w7-3445，40 核；纯 CPU，OMP/MKL=1；机器另有他人负载 | ⚠️ 不可直接比时间 |

### 3.2 两种噪声模型（本报告的关键对照）

| 名称 | 说明 | var(d) | cov(u_l,d) | 与估计器 Σ_pix 一致？ |
|---|---|---|---|---|
| `independent`（**论文口径**） | 左/右目各自独立加 σ_u/σ_v 噪声 | 2σ_u² | σ_u² | ✅ 一致 |
| `upstream`（**上游代码口径**） | 复制上游 `StereoCamera.forward`：`u_r = u_l - f_u b/z + σ_u n` | σ_u² | 0 | ❌ 估计器过信深度 √2 倍 |

两者共用相同的位姿/特征/随机初始化随机流，仅像素噪声生成不同。下文 Table V 两种都给出。

---

## 4. 数值验证

### 4.1 `smoke_test.py`：16/16 通过

| 检查 | 结果 |
|---|---|
| 24 个齐次约束在 SO(3) 上为零 | `max|x^T A_i x| = 3.4e-16` |
| 约束 Jacobian 秩 = codim(Ω) = 7（ACQ） | 秩 7，奇异值 [2.81, 2.45, 2.41, 1.88, 1.27, 1.23, 0.47, 0, 0] |
| `x^T Q x` = 矩阵加权代价 | 相对误差 <1e-10 |
| 逆相机模型解析 Jacobian vs FD | 7.7e-9 |
| Σ_pix PSD / W 对称正定 | ✅ |
| SDP tight | λ1/λ2 = 1.1e9 |
| 证书 H ⪰ 0 | eigmin = -7e-11 |
| 证书 Hx = 0 | ‖Hx‖ = 8.6e-10 |
| 强对偶 | primal = dual = -2.477981e-2 |
| 解出 C 的正交性 | 1.1e-8 |
| **Thm.2 Jacobian vs FD（b）** | **3.9e-6**（SCS 1e-9）/ **8.2e-8**（SCS 1e-12 参考） |
| 反向规则 == Jacobian 缩并 | 一致到 1e-8 |
| 外层 loss 梯度 vs FD | 7.4e-4（受 FD 参考精度限制） |
| loss 对 x 的梯度 vs FD | 2.4e-10 |
| 计时 | Q 构建 0.17 ms，SDP 单次 2.6 ms（warm）/ 113 ms（首次含 canonicalisation），backward 0.087 ms |

### 4.2 50 trials 的 Jacobian 校验（`validate_jacobians.py`）

*参数 θ = 基线 b*：参考值用"SCS 1e-12 求解的 SDP 解的有限差分"（`h = 1e-3 b_true`，
中心差分）。50 个独立问题（每题 1 个随机 GT 位姿，Fig. 3a 分布），Eq. (29) 指标：

| 方法 | ΔJ（mean） | ΔJ（std） | ΔJ（median） |
|---|---|---|---|
| **SDPR-IS（Thm.2）vs FD** | **7.21e-06** | 1.24e-05 | 2.08e-06 |
| **local-GT（局部隐式微分）vs FD** | **4.53e-06** | 7.05e-06 | 1.45e-06 |
| SDPR-IS vs local-GT | 3.10e-06 | 1.03e-05 | 5.22e-07 |

*参数 θ = 特征点位置 m_k*（论文 Table III/IV 的口径）：SDPR-IS 与 local-GT 的
`12 × 192` Jacobian（9 个 dC 分量 + 3 个 dr_p 分量 × 64 点 × 3 轴）对比：

| 指标 | mean | std | median |
|---|---|---|---|
| ΔJ（矩阵 ∞-范数意义下的 Eq.29） | **2.25e-06** | 6.87e-06 | 2.87e-07 |
| 逐元素平均绝对差 | 2.57e-07 | 8.95e-07 | 1.81e-08 |
| FD 抽检（10 trials × 6 个随机参数，相对误差）SDPR-IS | 3.92e-04 | 2.38e-04 | — |
| FD 抽检 local-GT | 3.92e-04 | 2.38e-04 | — |

其它观测：
* 两种方法解出的位姿差 RMSE：平移 1.23e-06 m、旋转 5.75e-07（二者都收敛到全局最优）；
* corank(H)：**50 个 trial 全部为 1**（1e-9 相对容差），1e-6 容差下 median=1（mean 1.16）；
* λ1/λ2 中位数 1.03e10；
* 单 pose 反向耗时：SDPR-IS 0.60 ms（含 `dQ/db` 复步与 KKT 解），local-GT 1.83 ms。

对照论文 Table IV（矩阵加权）：SDPR-IS 的 ΔJ 为 3.95e-05 ± 8.31e-05（参考量是
Theseus-GT-UR 的 unrolling Jacobian）；本实现的 ΔJ 更小（2.25e-06），但**参考量不同**
（我们以"两种独立方法互相印证 + FD"为参考），因此只能比较量级：两者都在 1e-5~1e-6 量级，
即"不同实现给出的 Jacobian 在数值精度内一致"。图：`figures/fig_jacobian_checks.png`。

### 4.3 Fig. 3b 式局部极小值实验（`local_minima.py`）

每个数据集 1 个 GT 位姿、100 个随机初始化（论文 Fig. 3b 分布），局部求解器先按论文
Theseus 配置（step 0.2, rel 1e-6, ≤500）求解，再紧收敛 polish 到驻点：

| 数据集 | 收敛到局部极小 | 收敛到全局最优 | 失败（非驻点） |
|---|---|---|---|
| dataset 0 | 40% | 60% | 0% |
| dataset 1 | 67% | 33% | 0% |
| dataset 2 | 70% | 30% | 0% |
| **合计（300 inits）** | **59%** | **41%** | **0%** |

论文："roughly 60% the initializations converge to poor local minima" → **59% 高度吻合**。
两种噪声模型的分类结果相同（随机初始化序列相同）。局部极小值的 inner cost 在 1e5–1e6，
全局约 200。图：`figures/fig3b_local_minima.png`、`figures/fig3b_cost_histogram.png`。

---

## 5. 主结果：Table V / Fig. 4

### 5.1 Table V 对照

论文 Table V（`paper/sdpr.txt:L912-L921`）：

| Method | Final Baseline Error (avg) | Final Baseline Error (std) | Number of Outer Iterations (avg) | Outer Loss (avg) |
|---|---|---|---|---|
| Theseus-RND | 1.132e-01 | 3.191e-02 | 8.034e+01 | 1.384e+02 |
| Theseus-GT | 3.463e-04 | 4.901e-04 | 3.540e+01 | 9.854e-03 |
| SDPR-IS | 3.454e-04 | 4.907e-04 | 3.422e+01 | 9.857e-03 |

本实现（50 runs × 20 poses，均值 ± 标准差，**误差为有符号的 b - b_true**，与上游
`get_statistics` 的口径一致）：

| Method | Final Baseline Error (avg) | (std) | Iterations (avg) | Outer Loss (avg) |
|---|---|---|---|---|
| **independent（论文噪声口径）** | | | | |
| local-RND（≈Theseus-RND） | 3.375e-01 | 1.144e-01 | 143.28 | 1.554e+02 |
| local-GT（≈Theseus-GT） | 4.112e-04 | 4.237e-04 | 54.62 | 7.496e-03 |
| **SDPR-IS** | **4.105e-04** | **4.237e-04** | **54.50** | **7.496e-03** |
| **upstream（上游代码噪声口径）** | | | | |
| local-RND | 3.271e-01 | 6.589e-02 | 142.88 | 1.569e+02 |
| local-GT | 4.992e-04 | 5.902e-04 | 54.10 | 1.251e-02 |
| **SDPR-IS** | **4.985e-04** | **5.902e-04** | **53.92** | **1.251e-02** |

**逐项对比与差异**：

1. **SDPR-IS vs local-GT**：论文 3.454e-4 vs 3.463e-4（差 0.3%）；本实现 4.105e-4 vs
   4.112e-4（差 0.17%），且 50 条轨迹逐迭代最大差 **1.08e-6 m**、最终 outer loss 相同到
   1e-10 —— 与论文 "the individual trajectories of these two approaches are very close to
   each other" 完全一致。这是本复现**最强的一致性证据**。
2. **Final baseline error**：本实现 4.1e-4（independent）/5.0e-4（upstream），论文 3.45e-4；
   大 19%/44%。该量是"样本意义下外层 loss 的极小点"位置，对噪声实现和数据随机性敏感
   （50 次 run 的 std 就有 4.2e-4–4.9e-4，比均值本身还大）。量级一致，符号的分布一致。
3. **Outer loss**：论文 9.86e-3 落在本实现两种噪声口径之间（7.50e-3 / 1.25e-2）。这与
   §3.2 的分析自洽：上游代码的像素噪声与估计器 Σ_pix 不一致（深度过信 √2 倍），
   会抬高位姿误差与外层 loss 下限；"论文口径"则给出更低 loss。
4. **Iterations**：本实现 53.9–54.6，论文 34.2（SDPR）/35.4（GT）。**SDPR 与 GT 的相对关系
   完全一致**（几乎相等），绝对数量偏大 ~58%。见 §9.3 的阈值灵敏度与归因。
5. **RND**：本实现 0.327–0.338，论文 0.113（本实现大 ~3 倍）；但**失败方向/性质一致**
   （错误梯度把 b 推向完全错误的值；外层 loss 高两个数量级 155 vs 9.86e-3；
   Fig. 4 红色轨迹形态一致）。见 §9.4 归因。50 个 RND run 的终止状态：
   independent 模式 **7 收敛（但收敛到错误点）/ 42 到 150 次上限仍未达梯度阈值 / 1 发散**
   （发散定义：|b-b_true|>1）；upstream 模式 8 / 42 / 0。

### 5.2 Fig. 4 式轨迹

* `figures/fig4_baseline_trajectories_independent.png` / `_upstream.png`：
  左图全量 50 条轨迹（绿=SDPR-IS、蓝=local-GT、红=local-RND，t=0 起点 0.003），
  右图放大到论文 Fig. 4 的坐标范围（x∈[0,70]，y∈[-2e-3,4e-3]）。
* `figures/fig4_zoom_*.png`：仅放大面板。
* `figures/fig_tableV_bars_*.png`：三方法 final error / iterations / outer loss 柱状图（含 std）。

### 5.3 迭代次数阈值灵敏度（无需重跑，从记录轨迹重算）

`tables/threshold_sensitivity_independent.csv`（`independent`，均值）：

| tol_grad_sq | 1e-2 | 3e-3 | 1e-3 | 1e-4 | 1e-5（本实现主设置） | 1e-6 |
|---|---|---|---|---|---|---|
| SDPR-IS | 28.1 | 32.7 | **37.0** | 46.2 | 54.5 | >150 |
| local-GT | 28.1 | 32.7 | 36.9 | 45.8 | 54.6 | >150 |
| local-RND | 129.3 | 134.6 | 138.1 | 142.9 | 144.9 | >150 |

**论文的 34.22 落在本实现 `tol_grad_sq ≈ 1e-3` 的行**（37.0）。这提示：论文正文的
"|grad| < 1e-3"与脚本里的 `tol_grad_sq = 1e-5` 并不等价（相差 3 倍），而论文报告的
34.22 更接近"较松阈值"或"外层 loss 曲率更大"的情形，见 §9.3。

---

## 6. 紧性与证书统计（每一次内层 SDP 求解都记录）

`SDPR-IS` 在两种噪声模式下共 **108,420 次 SDP 求解**（independent 54,500 +
upstream 53,920；统计来自 `raw/histories_*.pkl` 中每个 solve 的诊断）：

| 统计量 | independent | upstream |
|---|---|---|
| λ1(X)/λ2(X)：min / 中位数 | 6.46e8 / 1.32e10 | 4.68e8 / 1.37e10 |
| rank-1 比例（r > 1e5，论文 Appendix C） | **100%** | **100%** |
| corank(H) @1e-6 相对容差：=1 / =2 / =3 | **99.13%** / 0.73% / 0.14% | **99.17%** / 0.71% / 0.12% |
| corank(H) @1e-9 相对容差：=1 / 其它 | **99.97%** / 0.03% | **99.96%** / 0.04% |
| H 最小特征值：mean / 最负 | +4.45e-11 / -8.03e-10 | +4.81e-11 / -8.03e-10 |
| ‖Hx‖：mean / max | 2.51e-9 / 1.13e-8 | 2.50e-9 / 1.34e-8 |
| C 正交性误差（mean） | 4.01e-9 | 3.98e-9 |
| primal − dual：median / max（`certificate_sample.json`，300 次求解） | 1.9e-11 / 1.0e-9 | — |
| H 第二小特征值 / λ1(H)：median / max | 7.6e-5 / 5.7e-4 | — |
| SDP 求解失败 / 重试次数 | 0 / 0 | 0 / 0 |
| 证书回退（lstsq 重算乘子）次数 | 0 | 0 |

解读：
* **紧性 100%**：每次求解 X* 的 λ1/λ2 都 >1e8（阈值 1e5），与论文"所有实验均 rank-1、
  corank(H)=1、满足 Thm.2 前提"一致。
* corank(H) 在 1e-6 容差下有 ~0.9% 的求解 >1，是 SCS 对偶解的非唯一性 + 数值噪声
  （低秩 SDP 的对偶通常不唯一，论文 footnote 13 也指出这一点）；把容差收到 1e-9 后
  **99.97% 为 1**。紧性（λ1/λ2）与梯度结果不受影响：50 个 Jacobian trial 中
  corank 全为 1，且用 SDP 对偶证书 vs 用 lstsq 重算乘子得到的 Jacobian 差 1e-8 量级
  （smoke test）。检查 `‖Hx‖ ≤ 1.3e-8`、`H ⪰ 0`（最负 -8e-10）保证 KKT 一阶条件成立。
* 独立抽样复核（`certificate_sample.py`，5 个数据集 × 20 位姿 × 3 个 b 值 = **300 次求解**，
  `repro/results_clean/certificate_sample.json`）：λ1/λ2 min = 1.21e9、median = 1.01e10；
  corank(1e-6)=1 占 96.0%（max 3）、corank(1e-9)=1 占 99.3%（max 2）；
  H 第二小特征值 / λ1 ∈ [9.6e-11, 5.7e-4]（median 7.6e-5，与 corank=1 一致，第三大特征值
  起就跳到 O(0.1–1)）；λmin(H) ∈ [-6.3e-10, 5.2e-10]；‖Hx‖ ≤ 9.3e-9；
  |primal-dual| ≤ 1.0e-9。全量与抽样两个口径一致。
* 图：`figures/fig_tightness_independent.png` / `_upstream.png`（λ1/λ2 分布、corank、
  H 的第二小特征值、λmin(H)、单次 solve 与 backward 时间直方图）。

## 7. 资源开销

数据来源：`repro/results_clean/benchmark.csv` / `benchmark.json`（单进程、`OMP_NUM_THREADS=1`，
每种方法 3 个 run × 20 poses；峰值 RSS 由子进程 `resource.getrusage(RUSAGE_SELF).ru_maxrss`
自报）与 `timing_*.json`（12 进程并行实验）。

### 7.1 单进程（无自身并行竞争）

| Method | wall / run | 外层 iter 时间 | 前向（内层求解） | 反向（隐式微分） | 峰值 RSS |
|---|---|---|---|---|---|
| SDPR-IS | 15.52 s（63.0 iter） | 243 ms | 10.9 ms/pose（SDP solve + rank-1 recovery） | 0.174 ms/pose | **298 MB** |
| local-GT | 3.27 s（63.3 iter） | 52 ms | 1.25 ms/pose（LM + polish） | 1.31 ms/pose | **298 MB** |
| local-RND | 11.59 s（~143 iter 量级） | 94 ms | 3.32 ms/pose | 1.37 ms/pose | **298 MB** |

（benchmark 使用独立随机种子（seed0=9000），其迭代数为主实验（54.5 / 54.6 / 143）之外的
另一批样本：SDPR/local-GT 63.0 / 63.3，local-RND 126.3；local-RND 的 wall 偏高既因为
迭代数更多，也因为局部解更差时 LM polish 更贵。）

* SDP 单次求解（warm、同一 Q 反复解）：**2.66 ms**；**实际 trial 中**每个 pose 有自己的
  求解器、warm start 来自上一次外层迭代：**10.9 ms/pose**；若 warm start 来自另一个 pose
  （Q 差异大）则 ~78 ms；进程内**首次**求解含 CVXPY canonicalisation：**0.59 s**
  （每个 pose 一次 ⇒ 单 run 前 20 次求解约 2–3 s，是 SDPR 单 run 前向的重要固定开销）。
* 其它单项耗时：Q 矩阵构建 0.049 ms，`dQ/db`（复步法）0.187 ms，Thm.2 backward 0.070 ms
  （benchmark 单点测量）/ 0.174 ms（含全部 Python 开销，trial 内平均）。
* SDP 规模：**变量 x ∈ R^13**（n=13），**24 个二次约束 + 1 个齐次约束**（等式 25 个），
  `X ∈ S^13`（91 个自由变量），PSD 约束 1 个；Q/约束均为稠密 13×13。
* 线程设置：实验期每个 worker `OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=1`
  （12 workers，≤12 线程）；benchmark 单进程同样 1 线程。SCS 本身单线程。
* GPU：**未使用**（`torch.cuda.is_available()` 未启用；实验期间 nvidia-smi 显示 GPU0/1
  被其他用户占用 24–29 GB，与本实验无关）。

### 7.2 并行全量实验（12 workers）

| noise_mode | method | 50 runs 累计 trial 时间 | 并行 wall | 失败 run |
|---|---|---|---|---|
| independent | SDPR-IS | 1852.9 s | 160.7 s | 0 |
| independent | local-GT | 390.0 s | 37.1 s | 0 |
| independent | local-RND | 1652.8 s | 148.4 s | 0（46/50 未达梯度阈值，见 §9.4） |
| upstream | SDPR-IS | 1049.9 s | 95.6 s | 0 |
| upstream | local-GT | 281.5 s | 25.4 s | 0 |
| upstream | local-RND | 1461.2 s | 130.5 s | 0（42/50 未达梯度阈值） |

* 全量总计：**300 次 run（6000 个内层位姿问题 × 平均 ~50 外层迭代，共 10.8 万次 SDP 求解）**，
  CPU 时间约 **1.9 CPU·h**，并行 wall 约 **10 min**（期间机器 load average ≈ 50/40 核，
  另有他人任务占用约 22 核，故表中 wall 时间是**上界**）。
* 软件版本（与论文不同，需注意）：Python 3.11.17、numpy **2.4.6**、scipy 1.17.1、
  cvxpy **1.9.3**、SCS **3.3.1**、torch 2.14.1+cpu；论文使用当年的 SCS/cvxpy 版本与
  Mosek（有 license）。求解器版本差异会影响 SDP 精度与计时，但不改变算法结论。
* 与论文 Table III/IV 的 backprop 时间对比：论文 SDPR-IS 0.208–0.231 s、Theseus-GT
  0.142–0.186 s（Mosek/1e-12、小规模 SDP，按 trial 计）；本实现单 pose 反向 0.15 ms（SDPR）
  vs 2.1 ms（local）。**硬件、求解器、tolerance 与计时口径均不同，不可直接比较**；
  可比的结论是：在本实现中，SDPR 的**前向**（SDP 求解）比局部求解器贵 ~3–5 倍，
  而**反向**比局部求解器便宜（一次 13×13/20×20 线性解 vs 6×6 系统 + 8 次数值 Jacobian），
  与论文"复用乘子/证书使 SDPR-IS 反向更快"的结论方向一致。

---

## 8. 与上游复现（repro-up）的交叉对比

写作时 `repro/results_upstream/` 只有 smoke 阶段产物（1 次 run），且该 smoke run
失败（`NameError: name 'SDPRLayer' is not defined`），**尚无可用数字**。因此本节给出
可直接对齐的口径与预期，待 repro-up 出全量结果后补表（lead 会统一对齐）。

已确认双方**设置相同**（来自 `repro/results_upstream/smoke/prob_data_meta.json`）：
`n_batch=20, radius=3.0, board 1.0×1.0, N_squares=[8,8]`（64 特征），与 clean-room 一致；
b_init = 0.243、b_gt = 0.24、term_crit `max_iter=150, tol_grad_sq=1e-5` 也一致。

| 项目 | clean-room（本报告） | repro-up（待填） | 对齐说明 |
|---|---|---|---|
| SDPR-IS final err / iter / loss | 4.105e-4 / 54.5 / 7.50e-3（independent）<br>4.985e-4 / 53.9 / 1.25e-2（upstream） | | **upstream 口径**与上游代码的噪声生成器一致，最应接近 |
| local-GT | 与 SDPR 逐迭代最大差 1.08e-6 | | 应同样重合（若 repro-up 用 Theseus） |
| local-RND | 0.327–0.338 / ~143 iter | | 方向一致；平台值可能不同 |
| tightness | 100% rank-1，corank(1e-9)=1 达 99.97% | | 应同为 100% |
| Fig.3b 局部极小比例 | 59%（40/67/70%） | | 应 ~60% |

**对齐时的三个注意事项**：(a) 噪声生成口径（本报告 `upstream` = 上游代码口径）、
(b) 终止阈值 / 外层迭代计数的定义、(c) 随机种子（论文未公开，只能比较 50 runs 的分布）。
若 repro-up 使用的是 `sdpr-dq`（diff_qcqp=True，即 SDPR-CIFT）而不是 `sdpr`（SDPR-IS），
两者梯度应几乎相同（论文 Table IV：3.11e-5 vs 3.95e-5），迭代/误差统计也应一致。

## 9. 差异归因与局限

### 9.1 SDPR-IS ≡ local-GT（最强证据，非差异）

两者 50 条轨迹逐迭代最大差 1.08e-6 m，最终 loss 一致到 1e-10。说明：
* 本实现的两条**完全独立的**求解路径（SDP 松弛 + Thm.2 证书微分 vs SO(3) 局部 LM + 驻点隐式微分）
  收敛到同一解，并且给出相同的梯度；
* 这也与论文"二者轨迹非常接近"的结论一致。

### 9.2 最终 baseline error / outer loss 的 20–45% 差异

* 外层 loss 是**噪声实现相关的样本量**：50 runs 的 std（4.2e-4–5.9e-4）与均值同量级，
  论文的 3.45e-4 完全落在我们的分布内（我们 50 个 run 的中位数 3.9e-4）。
* 论文未公开数据种子、Mosek 版本与 tolerance，无法逐 run 对齐。
* **噪声口径**解释了 outer loss 下限的系统差：上游代码生成器与估计器 Σ_pix 不一致
  （§3.2），其 loss 下限更高（1.25e-2），论文的 9.86e-3 介于两种口径之间。
  → 说明论文的数字很可能来自与上游代码相同的噪声生成器，而我们的 `upstream` 模式复现了它，
  但数据随机性导致具体值仍差 27%。

### 9.3 外层迭代数 54 vs 34（未完全复现）

已排除/考虑的候选：

1. **Q 归一化（lead 提出的候选）→ 已数值排除**。上游在 `get_obj_matrix_vec` 里把
   `Q ← Q/‖Q‖_F`（并去掉常数 offset），且该 scale 依赖 b。但**对目标函数做正数因子缩放
   不改变解映射**；我们做了直接数值检验：用"冻结 scale"（本实现）与"每次前向按 ‖Q(b)‖_F
   归一化"（上游口径）分别计算外层梯度，50 个位姿上相对差 **1e-6–5e-6**，
   总梯度差 1.2e-6。→ 该候选**不是**迭代数差异的原因（本实现与上游在数学上等价）。
2. **终止阈值**（最可能）：论文正文 `|grad|<1e-3` 与脚本 `tol_grad_sq=1e-5`（`|grad|<3.2e-3`）
   不一致。本实现的阈值灵敏度表（§5.3）显示：`tol_grad_sq=1e-3` 时迭代数 **37.0**，
   `3e-3` 时 **32.7** —— 论文的 34.22 正好落在这一区间。
3. **外层 loss 曲率 H = d²L/db²**：迭代数 ∝ log(tol)/log(1-lr·H)。由本实现的梯度衰减率
   反推 H，比论文数字隐含的 H 小约 1.6 倍；H 依赖位姿半径/噪声实现/权重，属数据差异。
4. **数据随机性**：论文未给种子；本实现 50 runs 的迭代数 std 为 7.1（54.5 ± 7.1），
   不同种子批次的均值可以漂移数个百分点。

**结论**：相对关系（SDPR ≈ GT，二者都远快于 RND）与论文一致；绝对迭代数受终止阈值与
数据实现的强烈影响，未逐位复现，**存疑但不影响论文结论**。

### 9.4 RND 误差 0.33 vs 论文 0.11（未完全复现）

* **性质一致**：两者都是"错误的局部极小 → 错误梯度 → b 被推向完全错误的值"，外层 loss
  高两个数量级；红色轨迹的形态（快速上升 + kink）一致。
* **量级差异的候选原因**：
  1. **局部求解器实现不同**：论文 Theseus Gauss-Newton（step 0.2、`rel_err_tolerance=1e-6`、
     `max_iterations=500`、`track_best_solution`）；本实现 LM（自适应阻尼）+ 紧收敛 polish。
     LM 收敛更深、找到更"深"的局部极小（inner cost 1e5–1e6 vs 全局 ~200），
     其残差结构不同 ⇒ 梯度不同。我们验证了"GN 近似 vs 精确驻点 Jacobian"在单个局部极小上
     相差 ~10–80%，但不足以单独解释 3 倍差异。
  2. **发散/收敛判据**：论文 RND 平均 80.3 次迭代（达到梯度阈值），本实现 143 次
     （86% 的 run 到 150 次上限仍未达 `tol_grad_sq=1e-5`；但 b 已进入平台区，
     未真正发散：所有 |b-b_true| < 1.0，无 NaN）。若用 `tol_grad_sq=1e-3`，RND 平均
     仍为 ~135（梯度噪声大，无法降到该阈值）。
  3. **随机初始化的具体抽样**：论文未给种子；20 个位姿 × 每 run 随机初始化的组合决定
     最终平台值。本实现平台值分布 0.20–1.00 m（均值 0.34），论文 0.113±0.032 更集中，
     说明论文的随机初始化批次恰好落在较"温和"的局部极小组合。
  4. **图 3b 的局部极小比例（59%）与论文 60% 吻合**，说明局部极小结构本身是可复现的；
     差异主要在"外层 SGD 在错误梯度下走到哪个平台"。
* **终止状态统计**（见 §5.1）：100 个 RND run 中仅 15 个达到 `tol_grad_sq=1e-5`
  （且都停在错误点），84 个跑到 150 次上限但 b 已进入平台区，仅 1 个真正发散到
  |b-b_true|>1。→ **"收敛到局部极小 / 到上限仍未收敛 / 发散"三分法是清楚的：
  0% 收敛到正确解，~15% 收敛到错误的驻点，~85% 在错误梯度下持续漂移，
  <1% 数值发散**。
* **结论**：RND 的失败机制被复现；平台值未逐位复现，**存疑**，已给出 4 条归因。

### 9.5 其他局限

* 未复现 SDPR-SDP（cvxpylayers 对 SDP KKT 微分）与 SDPR-CIFT（重算乘子）两种对照方法，
  也未复现 Theseus-GT-UR（unrolling）；Table III/IV 的 ΔJ 只能以 FD 与 local-GT 为参考。
* 无 Mosek license：SDP 用 SCS 1e-9（论文 §VI-B4 同样用 SCS 1e-9，故此实验口径一致；
  但 Table III/IV 论文用 Mosek 1e-12）。
* 局部求解器为本实现（LM），非 Theseus；两者数值行为在 GT 初始化时一致到 3e-5
  （外层梯度），在随机初始化/远局部极小处可能收敛到不同点。
* 本报告的所有计时都在共享机器上取得（另一用户长期占用约 22 核），
  wall time 是上界；per-solve 微基准（2.6 ms）是较干净的测量。

---

## 10. 结论

1. **论文 Section VI-B4 的核心结论被独立复现**：紧 Shor 松弛 + 证书隐式微分给出的梯度
   与"全局最优的局部解"的梯度在 1e-6 量级一致，50 runs 统计几乎相同；
   随机初始化的局部解给出错误梯度，把基线标定推向完全错误的值（0.33 m vs 4e-4 m）。
2. **数值细节**：final baseline error 4.1e-4（论文 3.45e-4），outer loss 7.5e-3~1.25e-2
   （论文 9.86e-3，落在两种噪声口径之间）；迭代数 54（论文 34，阈值灵敏度显示论文数字
   对应更松的有效阈值）。
3. **紧性**：50000+ 次 SDP 求解 **100% rank-1**（λ1/λ2 > 4.7e8），证书 H ⪰ 0、corank 1、
   Hx ≈ 0、primal = dual，与论文假设一致。
4. **局部极小**：59%（40/67/70%）随机初始化落入局部极小，与论文"约 60%"一致。
5. **代价**：SDPR 前向比局部解慢 3–5 倍（10 ms vs 2–3.5 ms/pose，SCS 1e-9，纯 CPU），
   但反向更快；全量实验（300 runs）约 1.9 CPU·h。

---

## 附录 A. 文件清单与复现命令

```
repro/cleanroom/
  camera_model.py        Eq.(27)/Σ_k/W_k，支持复步微分
  data_gen.py            数据生成（Fig.3a / Fig.3b）
  sdp_pose.py            (P6) 齐次 QCQP + (P3) Shor SDP + 对偶证书
  implicit_diff.py       Thm.2 Jacobian / Eq.(20) 反向 / Eq.(29) 指标 / FD
  local_solver.py        SO(3) LM + 局部解隐式微分
  bilevel.py             (P7) 外层 SGD，三种内层方法
  run_experiment.py      全量 50 runs × 20 poses × 3 方法 × 2 噪声口径
  benchmark.py           资源/时延/峰值内存/Q 规模/tightness 汇总
  validate_jacobians.py  Jacobian vs FD + Table III/IV 口径的 m-Jacobian 对比
  local_minima.py        Fig.3b 式局部极小值统计
  sample_certificate.py  证书谱/紧性的独立抽样复核
  make_figures.py        Fig.4 / Table V / tightness / Fig.3b 图
  smoke_test.py          16 项自检
repro/results_clean/
  raw/histories_*.pkl           每 run 完整轨迹 + 每迭代诊断（~25 MB/个，仅本机保留，不进 git）
  raw/histories_*_compact.pkl   每 run 轨迹 + 诊断的 run 级均值（1.8 MB，进 git）
  tables/tableV_*.{csv,json}    Table V
  trajectories_*.csv            Fig.4 长表（method,run,iteration,b,err,loss,grad_sq）
  run_summary_*.csv             每 run 一行
  timing_*.json                 汇总计时/失败/tightness
  benchmark.{csv,json}          资源开销
  jacobians.{csv,json}          Jacobian 校验
  certificate_sample.json       300 次独立采样的证书谱/紧性统计
  local_minima_*.{json,csv,npz} Fig.3b
  figures/*.png                 全部图
  logs/*.log                    主实验/benchmark 日志
```

复现命令（环境 `sdpr-clean`，见 `repro/cleanroom/ENV_clean.md`）：

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python repro/cleanroom/smoke_test.py                          # 16 项自检
python repro/cleanroom/run_experiment.py --runs 50 --workers 12   # 主实验（~10 min）
python repro/cleanroom/benchmark.py --runs 3 --batch 20        # 资源开销
python repro/cleanroom/validate_jacobians.py --trials 50       # Jacobian 校验
python repro/cleanroom/local_minima.py --datasets 3 --inits 100    # Fig.3b
python repro/cleanroom/sample_certificate.py --datasets 5     # 证书谱抽样复核
python repro/cleanroom/make_figures.py                        # 全部图
```

## 附录 B. 逐项对照汇总

| 论文数字 | 本实现 | 相对差 | 判断 |
|---|---|---|---|
| SDPR-IS err 3.454e-4 | 4.105e-4（indep）/4.985e-4（upstream） | +19% / +44% | 同量级，统计涨落内 |
| SDPR-IS err std 4.907e-4 | 4.237e-4 / 5.902e-4 | -14% / +20% | 同量级 |
| GT err 3.463e-4 | 4.112e-4 / 4.992e-4 | +19% / +44% | 与 SDPR 关系一致 |
| SDPR iters 34.22 | 54.50（=37.0 @ tol 1e-3） | +59% | 阈值/曲率差异，见 §9.3 |
| GT iters 35.40 | 54.62（=36.9 @ tol 1e-3） | +54% | 同上 |
| SDPR outer loss 9.857e-3 | 7.496e-3 / 1.251e-2 | -24% / +27% | 论文值落在两口径之间 |
| RND err 1.132e-1 | 3.375e-1 / 3.271e-1 | +198% / +189% | 机制一致，平台值差异，§9.4 |
| RND iters 80.34 | 143.3 / 142.9 | +78% | 收敛判据差异 |
| RND outer loss 1.384e2 | 1.554e2 / 1.570e2 | +12% | ✅ 接近 |
| 局部极小 ~60% | 59%（40/67/70%） | -1 pt | ✅ |
| corank(H)=1 | 100% rank-1；corank≈1 | — | ✅ |
```
