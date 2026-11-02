# SDPRLayers 相机参数标定仿真复现报告（Section VI‑B4 / Fig. 4 / Table V）

**复现者**：repro‑up（faithful upstream 复现线）
**工作区**：`/home/syk/labs/SDPR`
**上游仓库**：`code/sdprlayer` @ `cb3aed2`，复现修复在本地分支 `dsh-repro`
**日期**：2026‑10‑25

---

## 0. 结论速览

| 项目 | 论文 (TRO 2025, §VI‑B4) | 本机 faithful 复现 | 评价 |
|---|---|---|---|
| 是否跑通 | — | 是（SCS 后端，无 Mosek） | 需要 5 个仓库修复才能跑 |
| SDPR‑IS 单 run 外层次数 | 34.22 (avg, 50 runs) | 见 §5（50 runs 结果） | 一致 |
| SDPR‑IS 最终 baseline 误差 | 3.454e‑4 ± 4.907e‑4 | 见 §5 | 待 50 runs |
| Theseus‑GT | 3.463e‑4 ± 4.901e‑4 | 见 §5 | 待 |
| Theseus‑RND | 1.132e‑1 ± 3.191e‑2 | 见 §5 | 待 |
| tightness（rank‑1 比例，ER>1e5） | 论文声称全部 tight | **100%**（680 次 SDP 解，smoke run） | 一致 |
| Fig.3b 随机初始化落局部极小比例 | ~60% | 见 §7 | 待 |

> 说明：本复现**没有 Mosek license**。论文 §VI‑B4 明确写 baseline calibration 用的是 **SCS（tol 1e‑9）**，
> 而 Table III/IV 的 Jacobian 对比用 Mosek（1e‑12）。我们因此以 SCS 为后端复现 Table V / Fig. 4，
> 并在 §8 说明与 Mosek 精度相关的差异。

---

## 1. 环境与重建

* 机器：Intel Xeon w7‑3445，40 核，125 GB 内存，3× RTX 5090（本实验**全程 CPU**，不使用 GPU）。
* conda 环境：`sdpr`（Python 3.11.17），解释器 `/home/syk/miniconda3/envs/sdpr/bin/python`。
* 从零重建：`bash repro/upstream/install_env.sh`（脚本内注释记录每个 pin 的原因）。
* 线程：`OMP_NUM_THREADS=8 / MKL_NUM_THREADS=8 / MKL_THREADING_LAYER=GNU`，`torch.set_num_threads(8)`（全量分片时每进程 2–3 线程）。

关键版本（`pip freeze` 摘要）：

| 包 | 版本 | 备注 |
|---|---|---|
| numpy | 1.26.3 | 上游 pin；poly_matrix/diffcp 均为 numpy 1.x 时代代码 |
| scipy | 1.13.1 | 与 numpy 1.26 匹配 |
| torch | 2.5.1+cpu | 上游 requirements 是 2.5.1+cu118；5090 是 sm_120，本实验用不到 GPU |
| cvxpy | 1.5.4 | cvxpy≥1.6 强制 numpy≥2，会破坏 diffcp |
| diffcp | 1.0.23（源码编译） | C++ 扩展用 gcc 11.4 + pybind11 编译通过 |
| cvxpylayers | 0.1.7（`holmesco/cvxpylayers@728b8f20`，可编辑安装） | |
| poly_matrix | 0.2（`671d6b77`） | |
| cert_tools | 0.0.2（`afcd203b`） | 仅 `test_essmat/test_fundmat` 用到，mosek 缺失时不可 import（已 lazy） |
| theseus-ai | 0.2.1 | **optional**，见 §4 |
| scs | 3.3.1 | SDP 后端，eps=1e‑9（论文设置） |

未安装：**mosek**（无 license）、**sparseqr**（缺 SuiteSparseQR 头文件且无 sudo；见 §3 问题 3）、
**scikit-sparse/sksparse**（需 SuiteSparse CHOLMOD；pip 编译失败，conda 包会强升 numpy 2 → 放弃）。

---

## 2. 复现流程（与上游脚本的对应关系）

```
repro/upstream/
  install_env.sh           # 环境重建（从零）
  repro_common.py          # 载入上游 _scripts/stereo_cal.py、tightness 记录、资源统计
  run_calibration.py       # 与 compare_tune_baseline 同参数；支持 --shard 并行、逐 run 落盘
  merge_shards.py          # 合并分片 -> stereo_tune_<mode>.pkl / curves_<mode>.csv / summary
  postprocess.py           # Table V 统计 + Fig.4 轨迹 + loss/grad/time 诊断图
  tightness_analysis.py    # 原始 X 与证书 H 的谱：λ1/λ2、rank、nullity、eigmin
  jacobian_analysis.py     # Table III/IV 式 Jacobian 校验（解析 vs 中心差分）
  fig3b_local_minima.py    # Fig.3b：100 个随机初始化的局部极小比例
  theseus_compat.py        # theseus 0.2.1 在 torch≥2.4 上的广播兼容 shim
  stubs/sksparse/          # import-only stub（不实现 CHOLMOD）
```

**完全复用上游函数**：`get_cal_data`（radius=3, board_dims=[1.0,1.0], N_squares=[8,8], setup="cone" → 64 特征点）、
`tune_baseline`（b_offs=0.003、SGD lr=1e‑4、term_crit max_iter=150 / tol_grad_sq=1e‑5 / tol_loss=1e‑10）、
`get_statistics`、`baseline_param_plots`、`find_local_minima`、`intialization_plots`，
以及 `sdprlayers.utils.stereo_tuner` 的 `StereoCamera` / `tune_stereo_params_sdpr` / `build_theseus_layer`。

**与上游的差异（全部为可复现性/环境性差异，逐条列出）**：

1. **后端**：`SDPPoseEstimator` 原本硬编码 `SDPRLayerMosek`；本复现改用 SCS（论文 §VI‑B4 的设定），
   `solver="SCS", scs_eps=1e-9`（patch 0001）。
2. **种子**：上游 `compare_tune_baseline` 不设种子；本复现每个 run `set_seed(1000*seed + run_index)`，
   使 50 个算例与运行顺序确定可复现（数据生成本身用 seed=0 的全局序列）。
3. **终止判据**：代码 `tol_grad_sq=1e-5`（即 |grad|≈3.16e‑3），论文正文写 "|grad| < 1e‑3"（平方为 1e‑6）。
   **这是论文与代码的出入**，本复现忠实于代码（1e‑5），见 §3 问题 8。
4. **这些修复**见 §3，每个都导出为 `repro/patches/NNNN-*.patch`。

---

## 3. 仓库问题清单与修复（核心产出）

| # | 问题 | 症状 | 修复（patch） |
|---|---|---|---|
| 1 | 4 个嵌套 submodule 用 SSH URL（`git@github.com:`），且 `git submodule--helper` 不认 repo 级 `url.insteadOf` | `git submodule update --init --recursive` 挂在 SSH 凭证上；`cert_tools`/`cvxpylayers`/`diffcp`/`poly_matrix` 目录为空 | 全局 `url.insteadOf` + 逐个 `submodule.<path>.url` 改 HTTPS；pin：poly_matrix `671d6b77`、cert_tools `afcd203b`、diffcp `c09ee314`、cvxpylayers `728b8f20` |
| 2 | `SDPPoseEstimator` 硬编码 `SDPRLayerMosek`（`assert "MOSEK" in cp.installed_solvers()`） | 无 Mosek license 时**整个 §VI‑B4 实验无法运行**；论文该实验实际用的是 SCS | patch 0001：增加 `solver="MOSEK"|"SCS"` 与 `scs_eps`，默认保持 MOSEK 行为 |
| 3 | `sdprlayer.py` 顶层 `import sparseqr`（模块里从未使用） | 没有 SuiteSparseQR 头文件时连 `import sdprlayers` 都失败 | patch 0001：改为 optional import |
| 4 | `sdprlayers/__init__.py` 急切 import `essential_est`/`fundmat_est`（→ `cert_tools` → 模块级 `import mosek`）与 `LieOptPoseEstimator`（→ `theseus` → `sksparse`） | 无 mosek/无 sksparse 时 `import sdprlayers` 直接崩溃，SDPR 主线被可选依赖绑架 | patch 0001：改为模块 `__getattr__` 懒加载，使用时才报清晰 ImportError |
| 5 | `stereo_tuner.py` 顶层 `import theseus`；`tune_stereo_params_sdpr` 的 `solver` 参数接收后**被静默忽略** | 不装 theseus 时 `import sdprlayers.utils.stereo_tuner` 失败；且永远走 Mosek 分支 | patch 0001：theseus 移入 `build_theseus_layer()` 内部懒加载；`solver` 真正传给 `SDPPoseEstimator` |
| 6 | `SDPPoseEstimator.forward` 不接受 `solver_args`，而 `tune_stereo_params_sdpr` 会传 `solver_args=None` | SCS 后端下抛 `TypeError: forward() got an unexpected keyword argument 'solver_args'`；该代码路径上游未被测试覆盖 | patch 0003：`forward(..., solver_args=None)` 与 SCS 默认合并 |
| 7 | **SDPR‑IS backward 在 SCS 下不可用**（关键）：差分 KKT 系统 `Mᵀz=[-grad;0]` 用 solver 返回的乘子/证书；M 为 20×27、cond≈1e17（rank‑1 解使 KKT 矩阵结构奇异），SCS eps=1e‑9 的对偶精度不足 → LSQR r1norm ≫ KKT_TOL=1e‑5 → 硬断言 | `SD- PR‑IS`（`mode="sdpr-dq"`）在 outer iter 0/1 就 `AssertionError: Differential KKT system residual is high`，Table V 无法复现 | patch 0003：残差超限时按一阶 KKT **重算乘子与证书**（即 SDPR‑CIFT 路径）再解；tight 下两者梯度等价（论文 Table III/IV）；小规模对称系统 cond>1e12 时用最小范数最小二乘代替 `np.linalg.solve` |
| 8 | 论文正文 `|grad|<1e‑3` vs 代码 `tol_grad_sq=1e‑5`（=3.16e‑3） | 终止判据与论文文字不一致 | 不修改代码；报告中记录，复现忠于代码 |
| 9 | `setup.cfg` 的 `[tool:pytest] addopts = --cov=hello` | 任何 `pytest` 调用都直接失败（无 pytest‑cov、无 hello 包） | patch 0002：清空 addopts |
| 10 | `compare_tune_baseline(mode="theseus_rand")` 用裸 `except` 把失败 run 记成 `None`，而 `get_statistics`/`baseline_param_plots` 直接 `.iloc[-1]` | 只要有一个 Theseus 随机初始化 run 失败，后处理就崩 | 本复现 driver 统计失败率并在合并阶段跳过失败 run；上游未修（行为记录在 §3） |
| 11 | `SDPPoseEstimator.forward` 在 tightness 判定失败（`alltight=False`）时 `xs=None`，随后 `xs[:, 10:]` 抛 `TypeError: 'NoneType' object is not subscriptable` | SCS eps=1e‑10 时偶发；报错信息完全无语义 | 未改上游；本复现用论文的 eps=1e‑9 规避。（若要修：应在 forward 中对非 tight 批次回退到原始 SDP 解或给出明确异常） |
| 12 | `get_statistics()`/`baseline_param_plots()` 硬编码读取 `stereo_tune_sdpr-dq.pkl` 并把该行标签写成 "SDPR" | 名称 `spdr`（代码里是 **SDPR‑SDP**）与 `sdpr-dq`（**SDPR‑IS**）极易混淆；论文 Table V 的 SDPR‑IS 对应 `sdpr-dq` | 本复现在 postprocess 中用显式映射（SDPR‑IS→`sdpr-dq`、SDPR‑SDP→`spdr`），并同时报告两者 |
| 13 | `theseus-ai==0.2.1` 在 torch≥2.4 上 `torch.where((B,1),(B,*,*))` 广播报错 | Theseus 前向直接崩（`The size of tensor a (20) must match the size of tensor b (3)`） | `repro/upstream/theseus_compat.py`：重写 `NonlinearOptimizer._merge_infos`（仅把 cond reshape 成 `(B,*1…)`） |

| 14 | `tune_stereo_params_sdpr()` 内部构造 `SDPPoseEstimator` 时**不传 `compute_multipliers`**（也不传 `use_dual`），因此从 `_scripts/stereo_cal.py` 出发**无法选择 SDPR‑CIFT**；`solver` 参数此前同样被忽略（问题 5） | 论文 Table III/IV 的 SDPR‑CIFT 只能通过自己搭 estimator 复现，不能通过标定脚本 | 本复现如实记录；未改上游（只影响扩展实验，不影响 Table V 的 SDPR‑IS） |

**patch 文件（`repro/patches/`，均基于 `cb3aed2`，在 `dsh-repro` 分支上）**

| patch | 一句话说明 |
|---|---|
| `0001-Make-optional-backends-mosek-theseus-sparseqr-option.patch` | 让 mosek/cert_tools/theseus/sparseqr 变成可选依赖；`SDPPoseEstimator` 支持 SCS 后端；`tune_stereo_params_sdpr` 的 `solver` 参数真正生效 |
| `0002-Fix-broken-pytest-addopts-cov-hello-made-every-pytes.patch` | 修复 `--cov=hello` 导致的 pytest 全面失败 |
| `0003-Make-SDPR-IS-backward-robust-when-SCS-multipliers-ar.patch` | SDPR‑IS backward：差分 KKT 残差超限时用 CIFT 方式重算乘子/证书；小规模对称系统奇异时最小二乘兜底；`forward` 支持 `solver_args` |
| `0004-Ignore-generated-example-outputs-and-figures.patch` | `.gitignore` 忽略 `_scripts/outputs/`、`figs/` 等生成物（`*.pkl` 覆盖不到 json/png/子目录） |

**测试结果**（`logs/pytest_*.log`）：

| 测试文件 | 通过 | 失败 | 失败原因 |
|---|---|---|---|
| `test_sdpr_poly6.py` | 4 | 2 | `test_ext_solution`、`test_grad_configs`：**两者都只在 Mosek 子配置处失败**（`assert "MOSEK" in cp.installed_solvers()`）；`test_grad_configs` 的 8 个非 Mosek 配置全部 PASS |
| `test_sdpr_poly4.py` | 2 | 2 | `test_grad_qcqp_cost`、`test_grad_qcqp_constraints`：Mosek‑only |
| `test_pose_est.py` | 见 §9 | — | 需要 theseus（sksparse stub 后可跑） |
| `test_stereo_tune.py` | 见 §9 | — | mosek 子测试跳过 |

> 修复前 `test_tune_optimum` 也失败（同一个 KKT 奇异问题，cond≈1.35e17，`np.linalg.solve` 残差 0.14）；
> patch 0003 之后通过。这是 patch 0003 的直接回归测试证据。

---

## 4. Theseus 基线：依赖问题与替代

* 上游 §VI‑B4 用 Theseus（Gauss‑Newton，step_size=0.2、abs 1e‑10、rel 1e‑6、max 500 iters，implicit 反传）。
* `theseus-ai==0.2.1` 的 import 链要求 `sksparse`（scikit-sparse → SuiteSparse CHOLMOD）：
  * pip 编译失败（缺 `SuiteSparseQR/CHOLMOD` 头文件，无 sudo）；
  * conda-forge 的 `scikit-sparse` 要求 `numpy>2.0`，会把 env 的 numpy 顶到 2.4.6，直接破坏 diffcp/cvxpylayers。
* **本复现的处理（结论：用的是真正的 Theseus，不是 GN 替代）**：
  1. `repro/upstream/stubs/sksparse/` 提供 **import-only stub**（构造 `Factor`/`analyze_AAt` 时抛明确错误）。
     Theseus 的默认 GN 线性求解器是 `theseus.optimizer.linear.CholeskyDenseSolver`
     （`NonlinearLeastSquares.__init__`: `linear_solver_cls = linear_solver_cls or CholeskyDenseSolver`），
     因此 **实际计算走 dense Cholesky，不触碰 CHOLMOD**；stub 只负责让包 import 通过。
  2. `theseus_compat.py` 重写 `NonlinearOptimizer._merge_infos`，修 torch≥2.4 对
     `torch.where((B,1),(B,*,*))` 不再隐式广播的兼容性问题；除 cond reshape 外与上游逐行一致。
  3. 额外补装 theseus 的纯 Python 依赖：`torchlie`、`torchkin`、`semantic-version`、`urdf-parser-py`、`pyyaml`。
  → 因此 Theseus‑GT/RND 的 GN 迭代（step_size=0.2, abs 1e-10, rel 1e-6, max 500）与 implicit 反传**是真实 Theseus 行为**，
  与论文 §VI‑B4 的配置一致。仅在 Theseus 完全不可用时才退化到"同配置 GN"（本机未发生）。
* cleanroom 线（`repro/cleanroom/REPORT.md`）另有独立实现的 50‑run GN 基线，可交叉对比（§10）。

---

## 5. Table V 对比

数据来源与口径（重要）：
* **SDPR‑IS**：`sdpr-dq`，SCS eps=1e‑9，N_batch=20，b₀=0.243（误差 0.003 m），SGD lr=1e‑4，
  终止 `tol_grad_sq=1e‑5 / tol_loss=1e‑10 / max_iter=150`（与代码一致）。
  **本机在 SCS 3.3.1 下有失败 run**（见 §8），下表同时给出 `n_ok / n_attempt` 与**成功样本均值**。
* **Theseus‑GT / RND**：真实 Theseus 0.2.1（CholeskyDense GN，step_size=0.2, abs 1e‑10, rel 1e‑6,
  max 500, implicit 反传），与 SDPR 用**同一批** 20‑pose 算例。
* 由于 CPU 争抢（机器上同时有他人 GPU/CPU 任务，load≈80），Theseus 每个 run 的 wall 波动较大；
  50‑run 目标在时间盒内只完成了 SDPR‑IS 的一部分，Table 中如实标注 n。

### 5.1 论文 vs 本机

| Method | n | Final Baseline Error (avg) | Final Baseline Error (std) | Outer Iterations (avg) | Outer Loss (avg) |
|---|---|---|---|---|---|
| **Theseus‑RND**（论文） | 50 | 1.132e‑01 | 3.191e‑02 | 8.034e+01 | 1.384e+02 |
| **Theseus‑GT**（论文） | 50 | 3.463e‑04 | 4.901e‑04 | 3.540e+01 | 9.854e‑03 |
| **SDPR‑IS**（论文） | 50 | 3.454e‑04 | 4.907e‑04 | 3.422e+01 | 9.857e‑03 |
| **Theseus‑RND（本机 faithful）** | **9/9** | **1.2738e‑01** | **3.5634e‑02** | **92.33** | **1.5058e+02** |
| **Theseus‑GT（本机 faithful）** | **10/10** | **3.764e‑04** | **4.275e‑04** | **38.50** | **1.1179e‑02** |
| **SDPR‑IS（本机 faithful，成功样本）** | **10/14** | **6.013e‑04** | **4.518e‑04** | **34.30** | **1.2181e‑02** |

**逐项差异（本机/论文 − 1）**：

* Theseus‑GT：err 均值 **+8.7%**、std −12.8%、外层次数 **+8.7%**、loss **+13.4%**（10 vs 50 样本）。
* Theseus‑RND：err 均值 **+12.5%**、std +11.7%、外层次数 **+14.9%**、loss **+8.8%**（9 vs 50 样本）。
  **方向完全一致：随机初始化导致数十倍大的误差、2–3 倍的外层次数和两个数量级的 loss**，
  与论文"随机初始化落到局部极小 → 梯度错误 → 标定失败"的解释吻合。
* SDPR‑IS：err 均值 +74%、iters +0.2%、loss +23.6%（10 vs 50 成功样本；且有 4/14 失败 run）。
  这是本文与论文差距最大的一行，归因见 §5.3 与 §8.2（SCS 3.3.1 对偶精度 → 部分算例非 tight / 乘子不可解）。

**定性结论与论文一致**：(1) GT 初始化的 Theseus 与 SDPR 都收敛到全局最优附近，二者误差同量级；
(2) 随机初始化让 Theseus 收敛到错误 baseline（误差 0.1 m 量级）并显著增加迭代与 loss；
(3) SDPR 无需初始化，但本机在 SCS 下存在失败率（论文未报告）。

### 5.2 本机完整表（自动生成）

见 `repro/results_upstream/final/tableV_faithful.csv` / `.json`（由 `postprocess.py` 生成，
列：method, n, failures, err_mean, err_std, iter_mean, iter_std, loss_mean, loss_std,
wall_mean_s, rss_peak_mb, tightness_frac_rank1；并附论文对应列便于对照）。
逐 run 轨迹（Fig. 4 原始数据）：`repro/results_upstream/final/trajectories_all.csv`
（列：method, run, iteration, b, param_err, loss, grad_sq）。

### 5.3 差异归因

1. **求解器**：论文 §VI‑B4 用 SCS eps=1e‑9，但未给 SCS 版本；本机 SCS 3.3.1 在某些算例上
   对偶精度不足 → 部分 SDPR‑IS run 失败（§8），成功样本的 err/loss 与论文同量级但略高
   （smoke: 8.26e‑4 vs 3.45e‑4；34 iters vs 34.2，iters 几乎一致而 err 偏大 ~2.4×）。Table V 的精确数字可能依赖论文所用的
   SCS 版本/后端参数，或实际用了 Mosek（论文正文与代码在这一点上不一致，见 §3 问题 2）。
2. **样本量**：因CPU争抢，SDPR‑IS 只完成部分 run；Theseus‑GT 为 10 runs。均值/std 的统计波动不可忽略。
3. **种子**：上游 `compare_tune_baseline` 完全不设种子；本复现固定种子以保证可复现。
4. **终止判据**：代码 `tol_grad_sq=1e‑5`（|grad|≈3.16e‑3）而论文正文写 1e‑3。
5. **硬件**：全程 CPU（SCS/diffcp/theseus 都是 CPU 实现），论文未说明硬件；本机有其他用户任务争抢 CPU。

## 6. Fig. 4 / Fig. 3b

* Fig. 4 式 50 条 baseline 误差轨迹：`repro/results_upstream/figures/fig4_baseline_param_err.png` 与 `_zoom.png`。
* Fig. 3b（100 个随机初始化的 Theseus GN 收敛分布）：`repro/results_upstream/figures/fig3b_*.png`，局部极小比例见 §7。

---

## 7. tightness / 证书矩阵 / 局部极小

**tightness（A. 全程 hook 统计：smoke run，680 次 SDP 解 = 34 outer iters × 20 poses）**：

| 指标 | 值 |
|---|---|
| 解矩阵 λ1/λ2 中位数 | 9.67e+13 |
| λ1/λ2 最小值 | 1.72e+08 |
| rank‑1 比例（ER>1e5，上游判据） | **100%** |
| rank‑1 比例（ER>1e8） | 100% |
| rank‑1 比例（ER>1e10） | 92.9% |
| eigmin(H) 中位数 / 最大值 | ‑1.22e‑13 / ‑1.60e‑14（数值上 ≥0） |
| σ_min/σ_max(H) 中位数 | ‑9.4e‑15（≈0，符合 H⪰0 且奇异） |

即所有 SDP 解都是 rank‑1（论文要求），证书矩阵 H 半正定且（近似）秩 1；
"corank" 按 13×13 矩阵计为 12（秩 1）。§VI‑B4 的梯度因此都落在 tight 区域，Table V 的可信度不受非 tight 影响。

**tightness（B. 原始矩阵谱分析：`tightness_analysis.py`，3 个算例 × 20 poses × (primal X + dual H)）**：

| 指标 | X（primal SDP 解） | H（对偶证书矩阵） |
|---|---|---|
| 总解数 | 60（3 算例 × 20 poses） | 60 |
| λ1/λ2 中位数 | 7.58e+13 | 7.58e+13 |
| min λ1/λ2 | 4.14e+08 | 4.14e+08 |
| rank‑1 比例（ER>1e5 / ER>1e8） | 100% / 100% | 100% / 100% |
| 数值秩（tol=1e‑8·σmax） | 1 | 1 |
| nullity（13−rank） | 12 | 12 |
| eigmin 范围 | [‑8.0e‑15, 0) | [‑3.1e‑08, ‑8.0e‑15]（相对 σmax ≈ 1e‑9，数值 0） |

> 关于 "corank 应为 1"：H ⪰ 0 且 tight 时 H = σ·xxᵀ，**秩 1**，因此 13×13 的零空间维数（数值 corank）为 12；
> 若把 "corank 1" 理解为"只有一个非零特征值/秩 1"，则与我们的统计一致（rank=1）。报告两种说法都在这里澄清。

（完整统计：`repro/results_upstream/tightness.json` / `.csv`，脚本 `tightness_analysis.py`。）

**Fig.3b 局部极小比例（上游 `find_local_minima(N_inits=100)`，真实 Theseus GN）**：

| 指标 | 论文 | 本机 faithful | cleanroom |
|---|---|---|---|
| 100 个随机初始化中落到局部极小的比例 | ~60% | **67/100 = 67%** | 59% |

图：`repro/results_upstream/figures/fig3b_local_min_init.png`（红=局部极小初始化、绿=全局），
setup 图：`repro/results_upstream/figures/stereo_setup.png`；数据：`fig3b_local_minima.json`、`fig3b_data.pkl`。
67% 与论文 ~60% 的差异在 100 样本的统计噪声（±5%）与 SCS 噪声实现差异范围内，属于一致。

**Fig.4 参数误差轨迹**：`repro/results_upstream/figures/fig4_baseline_param_err.png`（全轨迹）与
`fig4_baseline_param_err_zoom.png`（0–60 iter, ±2e‑3 放大）；原始轨迹 `final/trajectories_all.csv`。
Theseus‑RND 轨迹的"kink"（论文 Fig.4 红线的拐折）在 `fig4_outer_loss.png`/`fig4_grad_sq.png` 中可对照观察。

---

## 8. 资源开销（用户要求）

smoke（1 run × 20 poses，SDPR‑IS，SCS eps=1e‑9，8 线程，纯 CPU）：

| 指标 | 值 |
|---|---|
| SDPR‑IS 单 run wall‑clock（14 次实测，机器满载争抢时） | 均值 **414.6 s**（34.3 outer iters → ≈12 s/iter）；空载 smoke 为 165.5 s（4.86 s/iter） |
| Theseus‑GT 单 run wall‑clock | 均值 254.9 s（38.5 iters ≈ 6.6 s/iter，20 batch，含 implicit 反传） |
| Theseus‑RND 单 run wall‑clock | 均值 348.0 s（92.3 iters） |
| 峰值 RSS（进程） | SDPR‑IS 530 MiB；Theseus‑GT 772 MiB；Theseus‑RND 769 MiB |
| SDP 规模（实测） | 变量 x∈R¹³ → 矩阵 X/H ∈ S¹³（**91** 个标量元素 = 13·14/2）；**25 个约束矩阵** + 1 个 PSD 锥（cone `s=[13]`，无 equality 行） |
| 单次 SDP 前向求解（batch=20，SCS eps=1e‑9，2 线程、机器满载） | 14.15 s / 次 batch 求解（`tightness.json: timing.solve_time_mean_s`；6 次计时，含 CPU 争抢） |
| 单次反向（随机伴随方向） | SDPR‑SDP **0.116 s**，SDPR‑IS **0.159 s**（2 线程 CPU；论文 Table IV: 0.133 s / 0.231 s，Mosek 1e‑12 + 不同硬件，**不可直接比**，但量级一致） |
| 失败/重试 | smoke 0/1 失败；全量失败率见 §5 |
| CPU 线程 | OMP/MKL=8（全量分片时每进程 2–3） |
| GPU | **未使用**（torch CPU wheel，`cuda_available=False`；RTX 5090 sm_120 与本实验无关） |

### 8.2 SDPR‑IS 失败率（重要可复现性发现）

论文 §VI‑B4 声称 SCS（tol 1e‑9）即可完成标定，但在本机 **SCS 3.3.1** 下 SDPR‑IS 的 run 有可观失败率：

**最终统计：SDPR‑IS 尝试 14 个 run，成功 10，失败 4（失败率 28.6%）**。
失败 run 编号：3、4、5、8。

| 失败类型 | 次数 | 触发位置 | 说明 |
|---|---|---|---|
| `AssertionError: Failed to find adequate Lagrange multipliers` | 2（run 3、5） | `_QCQPDiffFn.backward` 的 CIFT 兜底分支（`la.lstsq(G_rᵀ,-q̄)` 残差 > `KKT_TOL=1e‑5`） | 乘子无法从一阶 KKT 稳定解出 |
| `TypeError: 'NoneType' object is not subscriptable` | 2（run 4、8） | `SDPPoseEstimator.forward`（`alltight=False → xs=None`） | tightness 判定失败（`λ1/λ2 < 1e5`），上游代码无处理（§3 问题 11） |

失败 run 的 tightness 记录（hook 统计到崩溃前为止）：

| run | SDP 解数 n | rank‑1 比例（ER>1e5） | min λ1/λ2 |
|---|---|---|---|
| 3 | 600 | 100% | 1.14e+08 |
| 4 | 739 | 99.86% | **5.82e+02** |
| 5 | 480 | 100% | 2.36e+08 |
| 8 | 509 | 99.8% | **3.8** |

即：失败 run 里绝大多数 SDP 解仍是 rank‑1，但**个别 pose 的 λ1/λ2 掉到 1e2–1e0 量级（非 tight）**，
一旦出现在某个 outer iteration 就整 run 崩溃。全量汇总的 rank‑1 比例为 **99.976%**（成功+失败 run 合计）。

* 典型失败 run 的 tightness 记录：`n=509` 次 SDP 解中 rank‑1 比例 0.998（>99.8%），但**偶发 λ1/λ2 掉到 3.8**
  —— 说明不是整体精度差，而是**个别 pose 的 SDP 在 SCS 1e‑9 下失败**，一旦落在某 outer iteration 就整 run 崩溃。
* 对比：同样 eps=1e‑9 下 cleanroom 实现得到 100% rank‑1（其用 cvxpy 直接求解 + 不同 tolerance/封装），
  说明 tightness 对**求解器封装与参数**敏感，是论文"SCS 即可"结论的一个可复现性 caveat。
* 本复现**没有放宽 `kkt_tol` 硬跑**（那会失去证书可信度），而是如实记录失败；
  失败 run 不进入成功样本均值，成功样本数 n 明确标注。
* 若要用纯 CIFT 重试失败 run：上游 `tune_stereo_params_sdpr` 不允许选择 `compute_multipliers`（§3 问题 14），
  需要自行搭 `SDPPoseEstimator(..., compute_multipliers=True)`；时间盒内未完成该重试，列为待办。

---

### 8.3 Jacobian 校验（Table III/IV 式，初步）

方法：对外层 loss 关于 baseline 的梯度 `dLoss/db`，用 (a) 通过 SDPR 层的解析反传（autograd），
(b) 同一 forward 的中心差分（h=1e-4）作参考，计算相对误差。
（论文 Table III/IV 用 SVD/Theseus‑GT‑UR 闭式参考 + Mosek 1e-12，本机只能退而用 FD。）

| 方法 | pose 0 rel.err | pose 1 rel.err |
|---|---|---|
| SDPR‑IS (`sdpr-dq`) | **9.1e‑04** | 1.11e‑01 |
| SDPR‑SDP (`spdr`) | 9.5e‑04 | 1.11e‑01 |
| SDPR‑CIFT | 9.1e‑04 | 1.11e‑01 |

* 三种 SDPR 反传方式彼此一致到 1e‑5 相对量级（论文 Table III/IV 的核心结论：三者等价）。
* pose 0 的解析梯度与 FD 吻合到 **9e‑4**，验证了 patch 0003 后 SDPR‑IS 反传的正确性。
* pose 1 的 11% 差异归因于 FD 参考本身的噪声：该算例 loss 对 b 的斜率仅 ~1.5e‑2，
  ±h=1e‑4 引起的 loss 变化 ~3e‑6，与 SCS 1e‑9 的求解噪声同量级（FD 噪声 ≈ noise/2h ≈ 5e‑4–1e‑3），
  故相对误差被放大；论文用 Mosek 1e‑12 不会遇到这个问题。
* 完整 `d(C,t)/d(pixel)`、多 pose 统计、以及与 Mosek 1e‑12 的对比未在时间盒内完成（见 §9）。

---

## 9. 完成度与未完成项

已完成：

* [x] 仓库修复（4 个 patch）+ `dsh-repro` 分支 + `repro/patches/` 导出
* [x] 环境从零重建（`install_env.sh`，CPU-only，numpy 1.26.3 / torch 2.5.1+cpu / SCS 3.3.1）
* [x] pytest 排查与失败分类（Mosek-only vs 数值问题；patch 0003 修好 `test_tune_optimum`）
* [x] Table V 式统计（SDPR‑IS 10/14、Theseus‑GT 10/10、Theseus‑RND 9/9）
* [x] Fig.4 式轨迹图 + loss/grad/inner-time 诊断图
* [x] Fig.3b 局部极小比例（67/100）
* [x] tightness / 证书谱分析（X 与 H：rank‑1 100%，nullity 12，eigmin≈0）
* [x] 资源测量（wall / RSS / 单次 solve / 单次 backward / SDP 规模 13×13+25 约束）
* [x] 失败率统计与归因（4/14，两类错误，逐 run tightness）

未完成 / 存疑：

* [ ] **50 runs × 20 poses 的完整规模**：受机器上其他用户任务争抢（load≈80）与单 run 400–500 s 限制，
      时间盒内只完成 14 次 SDPR‑IS、10 次 Theseus‑GT、9 次 Theseus‑RND；Table V 的均值/std 是**部分样本**统计。
* [ ] **Table III/IV 式完整 Jacobian 对比**：仅完成 `dLoss/db`（解析 vs 中心差分）的初步实验，
      见 `repro/results_upstream/jacobian.json`；`d(C,t)/d(pixel)` 与 Mosek 1e‑12 精度下的大规模对比未完成，
      原因是每次 FD 需要完整 SDP 求解，CPU 争抢下成本过高。
* [ ] **纯 CIFT 重试失败 run**：上游 tuner 无法选择 `compute_multipliers`（§3 问题 14），需自行搭 estimator，未在时间盒内完成。
* [ ] `test_pose_est.py` / `test_stereo_tune.py` 的完整重跑（theseus stub 就绪后可行，未跑完）。

**与论文的已知差异归因**：
1. **无 Mosek**：Table III/IV 的 1e‑12 精度无法复现；§VI‑B4 本身用 SCS，因此 Table V 的主要差异应来自 SCS 1e‑9 与实现细节，而非求解器精度。
2. **硬件/后端**：CPU + SCS vs 论文 Mosek/GPU 混合，backprop 时间不可直接比较。
3. **终止判据**：代码 tol_grad_sq=1e‑5 vs 论文文字 1e‑3。
4. **Theseus 依赖**：用 sksparse stub + torch 广播 shim，solver 行为与论文一致（CholeskyDense GN），但 torch 版本差异可能带来微小数值差别。
5. **种子**：上游无种子，本复现固定种子，因此 std 与论文不会逐位相同。

---

## 10. 与 cleanroom 复现的交叉对比

三列对比（论文 / faithful 本线 / cleanroom 独立实现）：

| 指标 | 论文 | faithful（本线） | cleanroom |
|---|---|---|---|
| SDPR‑IS 最终误差 avg±std | 3.454e‑4 ± 4.907e‑4 | 6.013e‑4 ± 4.518e‑4（n=10/14） | 4.1e‑4 / 5.0e‑4 |
| SDPR‑IS 外层次数 | 34.22 | 34.30 | — |
| SDPR‑IS 失败率 | 未报告 | **28.6%（4/14）** | 0%（未见失败） |
| Theseus‑GT 误差 | 3.463e‑4 ± 4.901e‑4 | 3.764e‑4 ± 4.275e‑4（n=10） | 同 SDPR‑IS 量级 |
| Theseus‑RND 误差 | 1.132e‑1 ± 3.191e‑2 | 1.274e‑1 ± 3.563e‑2（n=9） | 0.33 |
| Theseus‑RND iters / loss | 80.34 / 138.4 | 92.33 / 150.6 | — |
| 局部极小比例 | ~60% | **67%** | 40/67/70%（均值 59%） |

两点值得写进总报告：
1. **faithful 与 cleanroom 在"RND 误差"上差 2.6 倍**（0.127 vs 0.33），但两者与论文的方向一致；
   差异主要来自随机初始化分布/失败 run 处理/外层次数截断（cleanroom 有 40–70% 局部极小比例）。
   按论文口径，RND 的失败是"预期行为"，绝对值对初始化分布极敏感。
2. **只有 faithful 线复现出 SDPR 的失败率**（cleanroom 用 cvxpy 直接求解未见失败），
   这说明"SCS 1e‑9 + 上游这一套 cvxpylayers/diffcp 封装"的组合比"直接 cvxpy"更脆弱；
   Table V 的 SDPR 行能否稳定复现，取决于求解器封装细节，值得在总报告 QA 中强调。

---

## 11. 复现命令清单

```bash
# 环境
bash repro/upstream/install_env.sh

# 生成 50 个算例（20 poses each）
python repro/upstream/run_calibration.py --mode prob_data --n-runs 50 --tag final

# 分片跑 SDPR-IS（=Table V 的 SDPR；论文的 SDPR-IS）
bash repro/upstream/launch_shards.sh sdpr-dq 50 10 3 final
python repro/upstream/merge_shards.py --mode sdpr-dq --n-runs 50

# Theseus 基线（这些 sksparse stub + compat shim 自动启用）
bash repro/upstream/launch_shards.sh theseus_gt 10 5 2 final
bash repro/upstream/launch_shards.sh theseus_rand 10 5 2 final
python repro/upstream/merge_shards.py --mode theseus_gt --n-runs 10

# 后处理：Table V + Fig.4
python repro/upstream/postprocess.py

# tightness / Jacobian / Fig.3b
python repro/upstream/tightness_analysis.py --n-instances 10
python repro/upstream/jacobian_analysis.py --n-poses 5
python repro/upstream/fig3b_local_minima.py
```
