# 外部基线可得性核查（2026-11-27）

> 结论先行：**Theseus 与 SDPRLayers 都能实测**（Theseus 用 dense Cholesky
> 绕过 SuiteSparse；SDPR 用 SCS 无 Mosek）。**BPnP / Pyatov(ihls) / POLI
> 不能给出与 3D-3D/GM 表同口径的数字**：前两者任务域不同（PnP / 两视图），
> POLI 是逐点几何估计器且未找到官方代码/权重。没有把任何跨任务数字塞进
> 3D-3D 表。

核查环境：`ls` 单节点，CPU；所有仓库 clone 到
`/mnt/storage1/syk/third_party/`，**没有** vendor 进 worktree。

---

## 0. 总表

| 基线 | 官方来源 | 本次可得性 | 能否进 3D-3D 表 | 关键阻断 |
|---|---|---|---|---|
| Theseus | PyPI `theseus-ai==0.2.3` | ✅ 实测（dense solver） | ✅ | `scikit-sparse` 编译失败（缺 `cholmod.h`） |
| SDPRLayers | `utiasASRL/sdprlayer@cb3aed2` | ✅ 实测（SCS 前向） | ✅（仅前向） | 无 Mosek license；submodule 未初始化 |
| BPnP | `BoChenYS/BPnP@f1f4f6a` | ✅ 代码可得，未跑 | ❌ | 只有 PnP（3D-2D），无 3D-3D 路径 |
| Pyatov et al. (ihls) | `VladPyatov/ihls@8580bbf` | ✅ 代码可得，未跑 | ❌ | 两视图（2D-2D）；权重在 Google Drive |
| POLI | arXiv:2605.10456 | ❌ 未找到官方代码/权重 | ❌（本身不是 pose solver） | 需先训练/获取逐点 Gaussian，再喂 SDPR |

---

## 1. Theseus

### 1.1 安装结果

- 隔离 venv：`/mnt/storage1/syk/venvs/theseus`
- Python 3.10.22；`torch 2.11.0+cu128`；`theseus-ai 0.2.3`
  （cp310 manylinux wheel，7.3 MB）。
- 安装日志：`labs/03-robust-qpeplayer/experiments/baselines/logs/theseus_install.log`、
  `theseus_install2.log`。

**精确阻断**：`theseus-ai` 的依赖 `scikit-sparse>=0.4.11` 只能源码编译，
而本机没有 SuiteSparse 头文件、无 sudo：

```text
src/sksparse/cholmod.c:1310:10: fatal error: cholmod.h: 没有那个文件或目录
ERROR: Failed building wheel for scikit-sparse
theseus-ai 0.2.3 requires scikit-sparse>=0.4.11, which is not installed.
```

**绕过**：先装 wheel，再单独装其余依赖（`torchlie`、`torchkin`、
`semantic-version`、`pybind11`、`pytest` 等），不装 `scikit-sparse`。
`import theseus` 成功，仅打印：

```text
UserWarning: Couldn't import skparse.cholmod. Cholmod solver won't work.
```

`theseus.CholmodSparseSolver` 不可用；本次实验用
`th.CholeskyDenseSolver`。N=100 / 13×13 规模下数值有效；**不能**代表
Theseus 的稀疏大 N 性能。

### 1.2 API 差异（已修）

Theseus 0.2.3 的 `info.status` 是 `numpy.ndarray(dtype=object)`，元素为
`NonlinearOptimizerStatus` 枚举，不是 tensor：

```text
TypeError: int() argument must be a string, a bytes-like object or a real
number, not 'NonlinearOptimizerStatus'
```

已在 `run_theseus.py` 改为 `int(info.status.flatten()[0].value)`。
另外 `LevenbergMarquardt(..., vectorize=False)` 会提示建议用
`TheseusLayer`，属正常 warning。

### 1.3 目标一致性

`theseus_cost.self_check`（20 点，c=0.05）：

| 项 | 值 |
|---|---:|
| 解析 Jacobian vs 中心差分相对误差 | `1.09e-10` |
| Theseus GM 目标值 vs NumPy 参考相对误差 | `0.0` |
| 目标值（Theseus / NumPy） | `0.04991106427965817` / 同值 |

结论：跑的是与 `common.solve_gnc` **同一个** per-point GM 目标
（`rho(e)=c^2 e/(c^2+e)`），不是近似。

### 1.4 计时口径

`run_theseus.py` 的 `solve_theseus` 在 `t0` 之前构建目标，因此报告的是
**solver-only** 时间；每 trial 的目标构建约 6 ms 未计入。GNC/RANSAC 的
时间含完整调用。主表已注明。

---

## 2. SDPRLayers

### 2.1 代码与环境

- 官方 clone：`/mnt/storage1/syk/third_party/sdprlayer`
  （`utiasASRL/sdprlayer@cb3aed2`，shallow）。
- 应用的 patch（来自 worktree `repro/patches/`）：
  - `0001-Make-optional-backends-mosek-theseus-sparseqr-option.patch`
    — 让 mosek/theseus/sparseqr 变可选，`SDPPoseEstimator` 支持
    `solver="SCS"`。
  - `0003-Make-SDPR-IS-backward-robust-when-SCS-multipliers-ar.patch`
    — **必需**：0001 的 SCS 分支使用 `SDPRLayer`，但当前上游
    `pose_est.py` 只 import 了 `SDPRLayerMosek`；只打 0001 会在构造时报
    `NameError: name 'SDPRLayer' is not defined`。0003 补上
    `from sdprlayers.layers.sdprlayer import SDPRLayer, SDPRLayerMosek`。
- **submodule 未初始化**：`.gitmodules` 用 SSH URL
  （`git@github.com:utiasASRL/...`），本次没有 `git submodule update`。
  运行所需的 `poly_matrix`、`cert_tools`、`diffcp`、`cvxpylayers` 由既有
  `sdpr` conda 环境的 editable install 提供（指向
  `/home/syk/labs/SDPR/code/sdprlayer/src/submodules/*`）。这是
  **provenance caveat**：SDPR 主代码是 fresh clone，四个子模块依赖来自
  既有复现环境。
- 运行 env：`/home/syk/miniconda3/envs/sdpr`，Python 3.11.17，
  numpy 1.26.3，torch 2.5.1+cpu，scs 3.3.1，cvxpy 1.5.4。
  用 `PYTHONPATH=/mnt/storage1/syk/third_party/sdprlayer/src` 覆盖
  site-packages 里的旧 `sdprlayers`。
- Mosek：无 license，未安装、未尝试；全程 SCS。

### 2.2 前向结果

脚本：`labs/03-robust-qpeplayer/experiments/baselines/run_sdpr_forward.py`。
设置：`SDPPoseEstimator(T_s_v=I, diff_qcqp=False, solver="SCS",
scs_eps=1e-9)`，`weights=1`，`inv_cov_weights=None`，`N∈{20,50,100}`，
20/12/20 trials per cell。

| N | p | 成功率 | rot 中位 (°) | p50 (ms) | SCS 异常 |
|---:|---:|---:|---:|---:|---:|
| 20 | 0.0 | 1.00 | 0.23 | 11.5 | 0/20 |
| 20 | 0.3 | 0.00 | 27.50 | 11.3 | 0/20 |
| 20 | 0.5 | 0.00 | 37.88 | 12.8 | 0/20 |
| 20 | 0.7 | 0.00 | 55.38 | 12.5 | 0/20 |
| 50 | 0.0 | 1.00 | 0.16 | 12.0 | 0/12 |
| 50 | 0.3 | 0.00 | 12.96 | 10.9 | 0/12 |
| 50 | 0.5 | 0.00 | 24.70 | 11.1 | 0/12 |
| 50 | 0.7 | 0.00 | 48.69 | 13.4 | 0/12 |
| 100 | 0.0 | 1.00 | 0.09 | 12.5 | 0/20 |
| 100 | 0.3 | 0.00 | 9.51 | 12.8 | 0/20 |
| 100 | 0.5 | 0.00 | 21.63 | 13.3 | 0/20 |
| 100 | 0.7 | 0.00 | 31.38 | 13.0 | 0/20 |

**SCS 前向失败率：0/208（0%）**。这里“失败”指 forward 抛异常。
上游 `pose_est.py` 在解非 tight（`xs=None`）时直接执行 `xs[:, 10:]`，
会抛 `TypeError: 'NoneType' object is not subscriptable`
（既有复现 `repro/upstream/REPORT.md` §3 问题 11 记录过；该复现在
eps=1e-10 时偶发，eps=1e-9 规避）。本次 eps=1e-9 下 **0 次**；
p≥0.3 的失败是**解出来的位姿错**，不是 SCS 异常。

### 2.3 关键限制：W_i=I 时 SDPR 等价于 Kabsch

SDPR 的目标是加权最小二乘
`sum_i w_i ||R x_i + t - y_i||^2`。当 `W_i=I`、`w_i=1` 时，它的全局最优
与 Kabsch 闭式解**是同一个**。逐 trial 配对 80 组（N=100）：

| 量 | 最大绝对差 | 均值 |
|---|---:|---:|
| 旋转误差 | `1.48e-5°` | `4.6e-7°` |
| 平移误差 | `1.27e-9 m` | `1.3e-10 m` |

证据：`experiments/baselines/results/sdpr_kabsch_equiv.json`
（脚本 `check_sdpr_kabsch_equivalence.py`，80 对）。
所以 SDPR 这一行是**前向正确性/可得性证据**，不是独立的鲁棒方法。
要让它成为真正的 SDPR 基线，需要：
1. 非单位 `W_i`（POLI 的逐点 Gaussian），或
2. SDPR + RANSAC 前端（协议里的 SDPRLayers+RANSAC）。

### 2.4 与既有复现“28% 失败率”的关系

`repro/upstream/REPORT.md` §8.2 的 **28.6%（4/14）** 是 SDPR-IS
**反向/标定**在 strict `kkt_tol=1e-5` 下的 run 失败率（SCS 3.3.1 对偶
精度导致部分算例非 tight / 乘子不可解），不是本次 forward SDP 的求解
失败率。两者不能混用；本文件只报告 forward 的 0/208。

---

## 3. BPnP（CVPR 2020）

- 官方仓库：`https://github.com/BoChenYS/BPnP`，commit `f1f4f6a`，
  clone 在 `/mnt/storage1/syk/third_party/BPnP`。
- 任务域：**Perspective-n-Point（3D-2D）**。`BPnP.py` 的
  `forward(ctx, pts2d, pts3d, K, ini_pose)` 用
  `cv2.solvePnP(..., cameraMatrix=K, ...)`；demo 用 cube/cat mesh +
  相机内参 `K` 做 2D 重投影。
- 无 3D-3D 代码路径；没有需要下载的预训练权重（synthetic PnP 可直接跑）。
- **为什么没跑**：即使跑一个小 PnP 实验，它的误差是 2D 重投影误差、
  变量是相机内参/位姿，和本表的 3D-3D + GM 核 + 成功判据
  （rot<5°, trans<0.1 m）不同口径。把它放进 3D-3D 表会是**跨任务拼表**，
  所以只记录 scope，不产出数字。
- 如果后续要做，最小实验是：合成 3D 点 + 已知 `K` + 2D 投影 + 外点，
  跑 `BPnP.apply` 前向，报 PnP 的 rot/trans；需单独标注为 PnP 任务。

---

## 4. Pyatov et al. / ihls（IROS 2024）

- 论文：*Robust Two-View Geometry Estimation with Implicit
  Differentiation*，arXiv:2410.17983。
- 官方仓库：`https://github.com/VladPyatov/ihls`，commit `8580bbf`，
  clone 在 `/mnt/storage1/syk/third_party/ihls`。
- 任务域：**两视图几何**（Fundamental / Essential matrix），输入是
  2D-2D 匹配；README 的复现需要 ScanNet / MegaDepth 测试子集 +
  Google Drive 权重
  （`https://drive.google.com/drive/folders/1JI2fLXG3vfUX9Wi2ijgQdu_wEE6F7DzL`）。
- **为什么没跑**：没有 3D-3D 点对应，也没有位姿的 3D 平移；权重是
  indoor/outdoor 两视图模型，不是 3D-3D 配准权重。构建 LoFTR 数据 +
  Drive 权重下载是跨任务工程，且结果不能与本表同列。
- 阻断性质：**任务域不匹配 + 权重需外部下载**，不是代码不可得。

---

## 5. POLI（Point-to-Ellipsoid）

- 论文：*Learning Point Cloud Geometry as a Statistical Manifold:
  Theory and Practice*，arXiv:2605.10456（2026-05-11）。
- POLI 是什么：**逐点 Gaussian 几何估计器**（Point-to-Ellipsoid），
  自监督预测每个点的局部 Gaussian（均值 + 协方差），不是 pose solver。
  在协议里的角色是给 SDPR 提供 `W_i`（逆协方差），再由 SDPR 解位姿。
- 代码/权重核查（2026-11-27）：
  - arXiv abs 页面没有 code link；
  - GitHub API 搜索 `Point-to-Ellipsoid` 返回 9 个仓库，均非官方
    （最近的是无关的 ellipsoid fitting / 可视化项目）；
  - 未找到作者项目页或权重。
- 结论：**没有官方代码/权重，POLI+SDPR 无法在本 session 复现**。
  这是“未找到”的负结果，不排除后续发布。
- 若后续可得，最小路径：在点云上训练/加载 POLI → 每点
  `Sigma_i` → `W_i = Sigma_i^{-1}` → `SDPPoseEstimator` 的
  `inv_cov_weights`。这仍不是独立 pose baseline，而是 SDPR 的输入前端。

---

## 6. 环境级注意事项

1. **GitHub TLS**：本机 `git`/`curl` 对 github.com 报
   `server certificate verification failed. CAfile: none CRLfile: none`
   （curl: `self-signed certificate in certificate chain`）。本次 clone 用
   `git -c http.sslVerify=false clone ...`。这是环境事实，不是仓库配置；
   长期应修 CA bundle，而不是默认关校验。
2. **不要在 worktree vendor 仓库**：三个 clone 都在
   `/mnt/storage1/syk/third_party/`；worktree 只保留脚本、结果、日志。
3. **没有 Mosek**：SDPR 的 Mosek 路径未验证；SCS 是论文 §VI-B4 的
   baseline 设置，可作为 licence-free 替代，但精度/失败率与 Mosek 不同。
4. **没有 SuiteSparse 头文件**：Theseus 稀疏求解器不可用；`scikit-sparse`
   与 `sparseqr` 都受影响。

---

## 7. 若要把缺口补上，需要什么

| 缺口 | 需要 | 影响 |
|---|---|---|
| Theseus 稀疏/大 N | 管理员装 SuiteSparse（CHOLMOD）或 conda 包 | 才能报 Theseus 的稀疏性能 |
| SDPR Mosek 对照 | Mosek license | 才能报论文 Table III/IV 的 Mosek 数字 |
| SDPR 非单位 W_i | POLI 代码/权重，或离线训练的 W_i | 才能把 SDPR 做成独立鲁棒基线 |
| Pyatov 对比 | ScanNet/MegaDepth + Drive 权重 + 两视图协议 | 只能做 2D-2D/相对位姿表，不能进 3D-3D 表 |
| BPnP 对比 | 单独的 PnP 协议 + 已知 K | 只能做 PnP 表 |
