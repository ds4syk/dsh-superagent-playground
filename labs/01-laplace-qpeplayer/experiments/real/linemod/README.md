# P1 real-data experiments on BOP LINEMOD

本目录记录 P1 线在真实 BOP 数据上的两轮尝试。**结论先行**：

1. 旧的「对称标注混合」构造（`run_symmetry.py`）是**负结果**，且已定位到两个具体原因（mm/m 单位错误 + 共享平移消去把双峰拉成折中解）。这个负结果保留，不删除。
2. 新构造（`run_two_mode.py`）在 **15/15 帧**真实 LINEMOD eggbox/glue 帧上得到**真实双峰代价**（K=2 个物理极小），并在同一真实几何上得到一个**横截的受控分支交叉**，Laplace 解析梯度对中心差分相对误差 **1e-9–1e-10**，而 mode-level MC 为 **1e-2–1e-1**。
3. 同时发现并修复了 `qpep_core.dE` 的真实 bug（见 §4），它让闭式 `dlogdetH_dtheta` 一直是错的（未被主表使用，因此此前没暴露）。

证据等级：全部为**本仓库独立复现/测量**；不涉及真实图像前端，也不对发表 pipeline 的精度作任何 claim。

---

## 1. 旧构造的负结果（保留）

`run_symmetry.py` 用同一组 bearing 分别与模型点 $P$ 和对称变换后的 $SP$ 配对，得到两个代价矩阵后按 $\theta$ 混合，并在某个 $\theta$ 求「双峰等值」。运行结果：9 帧候选里 `find_modes` 只返回 1 个可用极小（`results/symmetry.json` 为 `[]`）。

**注意：空表不能当作「数据不支持多峰」的负证据**——复现审计（`../REPRO.md` B1）发现它的交叉搜索本身有 bug：`gap_at()` 先 `np.sort(J)` 再取 `J[1]-J[0]`，返回值恒 $\ge0$，变号判据永远不成立；另外 `FRAMES` 漏了 `000011`、多数帧的对象不是 obj 10/11 而被静默跳过。脚本头已标注 deprecated。

2026-11-27 诊断出以下原因：

- **(B1) 单位错误**：`sym_of()` 把 `models_info.json` 的 4×4 对称变换平移（单位 **mm**）直接当 **m** 用，第二组点云整体偏移约 0.69 m。修正后 `J_B` 在其自身位姿处为 $4\times10^{-14}$（正确极小）。
- **(B2) 共享平移消去**：把两个完整 bearing 代价的 `M0,D0,G0` 相加后再消去**同一个** $t$，两个假设会争夺同一个平移；极小被拉到折中位姿（$t\approx-0.1$，而真值 $t_z\approx0.887$），代价于是单峰。即使修正 (B1) 也无法在交叉点得到两个分离的盆地。
- **(B3) 交叉搜索单调**：`gap_at` 对 $J$ 排序后取差，恒非负，见上。
- **(B4) 帧覆盖**：`FRAMES` 只列了 `000010`，且按 `obj_id` 过滤后 9 个名字里只有 2 个是 eggbox；glue（`000011`）从未被测试。

结论：**「混合完整代价 + 共享 t 消去」这一构造在数学上就得不到双峰**；这是构造问题，不是对象对称性问题。

## 2. 新构造 A：真实平面片的双峰代价（`run_two_mode.py` Part A）

- 对应点是真实的：每个 2D bearing 来自 BOP `mask_visib` 内的真实深度像素（真实内参反投影）；3D 模型点是**最近的真实 mesh 顶点**（KD-tree），因此真值位姿有真实的 2–7 mm 模型/深度失配残差，而不是「把 GT 反解回去」的零残差。
- 用确定性 RANSAC 选物体主平面片（2.5 mm 内点带）；平面目标有经典的两解位姿歧义，代价因此有第二个真极小。
- 过滤掉变换后点云在相机后方的模式（$z\le0$，非物理）。

结果（`results/two_mode.md` 表 A）：15/15 帧 K=2；$\Delta J\in[1.25\times10^{-4},2.56\times10^{-3}]$；翻转模式与真值夹角 3.8°–149.6°；Laplace 对错误模式的权重 $w_{\rm wrong}\approx0.45$–$0.61$。

**诚实的结论**：真实近简并时，真值位姿在 14/15 帧是全局极小（hard argmin 中位误差 2.06°），但 Laplace 后验均值被错误模式强烈拉偏（中位 58°）。所以这里 **Laplace 不是精度优势**，而是「用偏差换平滑」；1/15 帧（`000010/72`）hard argmin 真的选错（44°），说明 hard argmin 在真实噪声下也并非总对。

## 3. 新构造 B：真实几何上的受控横截交叉（Part B）

$J_\theta=(1-\theta)J_{\rm true\ labelling}+\theta J_{\rm sym\ labelling}$，每个假设**各自**消去自己的平移（避免 B2）。真实帧 `000010/15` 上：

- 实测零点在 $\theta^*\approx0.475$（不是精确 0.5，因为重采样后两个标注代价不再严格对称），$\Delta J$ 在 $[0.46,0.479]$ 之间**变号**，是横截交叉；
- 两个极小始终在其各自标注位姿的约 2° 以内；
- Laplace 解析梯度 vs 层输出中心差分：相对误差 **1.9e-10–1.8e-9**；
- mode-level MC（每次 400 样本）：相对误差 **9e-5–1.7e-1**。

这张表验证的是「闭式 $3\times3$ 反向 + Laplace 权重导数」这条链路在真实数据代价上也成立，不是精度 claim。

## 4. 顺带修复的代码 bug（`qpep_core.dE`）

`core.dlogdetH_dtheta` 是论文「$\nabla_\theta\log\det H_k$ 闭式」的实现，但此前**从未被主表使用**（`run_main.py` 用 `dlogdet_numeric`，`run_ablation.py`/`verify_q6.py` 用独立的 `dlogdet_exact`，后者用方向导数 `E(dq)` 从而绕开了 `dE`）。本次用 `dlogdet_exact` + 分支 FD 双重对照，发现 `core.dE` 的 `d1/d2/d3` 在第 1–2 行有错符号/错项，导致 `dlogdetH_dtheta` 偏差 O(1)。修复后：

- `dlogdetH_dtheta` vs `dlogdet_exact`：最大绝对误差 **8.9e-16**；
- vs 分支 FD：**1.8e-9**；
- 回归测试 `code/test_derivatives.py` 通过（`dE` 2.9e-11、`dH` 1.6e-8、`dq` 2.5e-10）。

## 5. 复现

```bash
cd labs/01-laplace-qpeplayer/experiments/real/linemod
/home/syk/miniconda3/envs/WM/bin/python run_two_mode.py            # ~1-2 min, 15 frames + crossing
/home/syk/miniconda3/envs/WM/bin/python run_two_mode.py --quick    # smoke test
/home/syk/miniconda3/envs/WM/bin/python ../../code/test_derivatives.py
# 旧负结果（保留）：
/home/syk/miniconda3/envs/WM/bin/python run_symmetry.py
```

数据：`/mnt/storage1/syk/data/bop/lm`（BOP LINEMOD；`test/000010` eggbox、`test/000011` glue）。

## 6. 仍未做

- 真实图像前端（2D 检测/描述子）与 ADD(-S)/ADD-S AUC 精度表：未接，不能声称。
- In-The-Dark 双目矩阵加权子问题：未做（P1-EXP-03）。
- 对称群轨道平均（Q15）作为 loss 的消融：未做；本目录的两个构造都不依赖 Q15 的结论。

## 7. Theseus 外部基线（2026-11-27 补）

P1 的「双模 + Laplace」需要一个**不枚举模式**的局部优化参照：`run_theseus_baseline.py` 在**同一个** `C(R,t)=Σ‖P_i^⊥(R p_i+t)‖²` 上跑 Theseus LM（复用 P3 `baselines/theseus_cost.py` + `run_theseus.py` 的模式：解析 Jacobian、LM+稠密 Cholesky、GT/随机初始化、20 trials），细节见 [`THESEUS_BASELINE.md`](THESEUS_BASELINE.md)、数据见 [`results/theseus_baseline.md`](results/theseus_baseline.md)。

```bash
/mnt/storage1/syk/venvs/theseus/bin/python run_theseus_baseline.py   # ~9 s, 370 次 LM
```

一句话结论：GT 初始化能逐帧回到 P1 的 hard-argmin 模式（局部收敛），但冷启动/多启动大多进入 flip 或面朝后的伪极小（多启动 11/40 命中 GT），且无 z>0 约束时这些伪极小的代价可与物理模式近简并、在 2/15 帧更低 —— 局部优化替代不了模式枚举，这个基线是**诚实的负结果**而不是精度 claim。

## 8. LMO 全序列扩展（177 帧）与 SQPNP 同对应点基线（2026-11-28）

**一句话**：把 LMO 样本从 37 帧扩到 **177 帧**（96 eggbox / 81 glue；200 次尝试，K≥2 通过率 88.5%），hard 中位 **3.33° / 19.8 mm / ADD-S 9.4 mm**、Laplace **42.5° / 32.2 mm / 18.7 mm**，与 37 帧结果**全部落在彼此 95% CI 内**；分层检验支持「小可见平面片驱动 hard 退化」（`n_plane < 350` 组 8.97° vs `≥350` 组 2.57°，置换 p=4e-4，obj 10 对象内同样成立），且**最近的物理模式本身**也变差（6.52° vs 2.57°）→ 是代价面被推离而非模式排序；SQPNP 在**同一平面内点**上与 hard 逐帧等价（中位 |Δrot| 0.03°、最大 0.75°）→ 两模层没有留下可回收的精度，瓶颈是对应点/深度噪声。

| 指标（n=177，中位 [95% CI]） | hard | Laplace | MC |
|-|-|-|-|
| 旋转 (deg) | 3.33 [2.77, 4.15] | 42.49 [40.20, 44.10] | 42.27 [40.33, 44.50] |
| 平移 (mm) | 19.8 [15.6, 23.1] | 32.2 [27.7, 37.3] | 32.2 [27.6, 37.5] |
| ADD-S (mm) | 9.4 [7.4, 10.8] | 18.7 [17.5, 20.9] | 18.4 [17.4, 21.0] |
| ADD-S AUC(0–0.1 m) | 0.906（均值 0.857） | 0.813（均值 0.776） | 0.816（均值 0.775） |

分层（描述性，边界在跑估计器前由候选池确定）：`n_plane ≥ 350` n=122 → hard 2.57° / ADD-S 6.9 mm；`< 350` n=55 → 8.97° / 13.8 mm（closest-mode 2.57° vs 6.52°）。可见性三分位（0.73 / 0.93）hard 中位 5.50° / 2.35° / 2.61°。**对象内重复**：obj 10 内部 `n_plane ≥350`（45 帧）1.70° vs `<350`（51 帧）8.97°（p=4e-4）；obj 11 只有 4 个小片帧，无法对象内检验。

**SQPNP 对比（17 帧，同一批 3-D/2-D 对应点；cv2 5.0.0，`sqpnp_lmo_helper.py` 在 sdpr 环境跑）**：`plane`（与估计器相同的 RANSAC 内点）旋转 3.12° / ADD-S 7.0 mm，与 hard 的 3.16° / 7.1 mm 配对差 **-0.05° [-0.16, 0.02]**、ADD-S -0.1 mm → **逐帧等价**（3/17 帧平面 PnP 返回多解，仍落在同一支）。`all`（全部 900 点）旋转中位 3.33° 未改善，平移/ADD-S 反而更差（平移 +55 mm、ADD-S +40 mm，CI 不含 0），且逐帧双峰：6 帧改善 >5°、5 帧恶化 >5° —— 非平面部分误差较大的 nearest-vertex 对应点会把无外点剔除的最小二乘 PnP 拖坏。

失败 23 帧（14 帧 K_phys<2、9 帧近空深度图，`mask_visib` 内实际有效深度 6–857 像素），逐条在 [`results/lmo_full.md`](results/lmo_full.md) §6；不替换、不重抽。完整逐帧表（177 行）在 `lmo_full.md` §2.1，机器可读在 `results/lmo_full.json`。

```bash
cd labs/01-laplace-qpeplayer/experiments/real/linemod
# 全部计算走 Slurm；实测 142.9 s（200 帧，Slurm job 3202）
srun --partition=cpu -n1 --time=00:40:00 \
  /home/syk/miniconda3/envs/WM/bin/python run_two_mode_lmo_full.py
# 只重画图表/报告（不重跑估计器）
/home/syk/miniconda3/envs/WM/bin/python run_two_mode_lmo_full.py --from-json results/lmo_full.json
```

产物：`run_two_mode_lmo_full.py`（复用 `run_two_mode_lmo.build_frame`，未改任何已有 `.py`）、`sqpnp_lmo_helper.py`、`results/lmo_full.{json,md,png}`。**Caveat**：仍是 GT 位姿给对应点的受控估计器实验，不是检测器；`px_count_visib ≥ 1200` 使极重遮挡实例系统性缺席（visib_fract 0.25–1.00），因此本表对 LMO 最难遮挡是乐观的；AUC 的 n=177 仍带台阶，配对差比绝对值更可信。

## 9. 检测器噪声鲁棒性（模拟前端，2026-11-28 补）

GPU 被占，训练好的检测器前端不可行；`run_detector_noise.py` 用**显式模拟的检测器噪声**替代前端，回答「层的相对退化 + hard/Laplace/MC 在噪声下的排序」。完整报告与图见 [`results/detector_noise.md`](results/detector_noise.md)。

**做法**：固定 `run_two_mode.py` Part A 的同一 15 帧（8 eggbox + 7 glue，与 `two_mode.json`/`pose_error.json` 逐帧核对相等）和**干净数据的 RANSAC 平面内点子集**；噪声下**不重跑 RANSAC**，所有条件共用同一批 3-D 点与同一批索引。扰动只作用于 2-D bearing：高斯像素噪声 σ∈{0,0.5,1,2,5} px（`U=normalize(K^{-1}[u+du,v+dv,1])`）；或把 f∈{0.05,0.1,0.2} 的 bearing 替换为图像内随机像素（粗差；另加 f=0.005/0.01 仅用于定位 breakdown）。每条件 10 个噪声种子。层与主实验相同：physical modes → hard / Laplace 投影弦均值 / mode-level MC（400 样本 × 10 seeds）；指标为 K_phys、ΔJ、w_other、旋转误差、ADD-S（1000 顶点 vs 完整 GT mesh）、>10° 失败率、GT-like 仍是 hard argmin 的帧比例，按帧 bootstrap 给 95% CI。

**高斯噪声（中位数；σ 行的 CI 见报告）**：

| σ (px) | K_phys | hard rot | Laplace rot | MC rot | hard ADD-S | Laplace ADD-S | hard fail | GT-like |
|-|-|-|-|-|-|-|-|-|
| 0 | 2 | 2.06 | 44.24 | 44.29 | 3.7 mm | 14.2 mm | 7% | 93% |
| 0.5 | 2 | 2.03 | 43.64 | 43.50 | 2.9 mm | 14.4 mm | 7% | 93% |
| 1 | 2 | 1.98 | 43.66 | 43.11 | 2.7 mm | 12.8 mm | 5% | 93% |
| 2 | 2 | 2.74 | 42.51 | 42.34 | 14.1 mm | 21.2 mm | 23% | 93% |
| 5 | 2 | 20.49 | 41.83 | 42.19 | 164.2 mm | 168.4 mm | 66% | 100% |

**粗差（f=0.005/0.01 为诊断行，不属于要求的网格）**：

| f | K_phys≥1 | fallback | hard rot | hard ADD-S | hard fail |
|-|-|-|-|-|-|
| 0 | 100% | 0% | 2.06° | 3.7 mm | 7% |
| 0.005 | 93% | 7% | 70.7° | 405 mm | 87% |
| 0.01 | 80% | 20% | 77.4° | 598 mm | 98% |
| 0.05 | 5% | 95% | 151.7° | 889 mm | 100% |
| 0.1 | 0% | 100% | 150.4° | 887 mm | 100% |
| 0.2 | 0% | 100% | 145.5° | 872 mm | 100% |

**结论**：

1. **1 px 以内无可测退化**（hard 2.0°、ADD-S 2.7–2.9 mm，甚至略低于 clean 3.7 mm——噪声平均掉了一部分 clean 的最近顶点偏差）；**2 px 起开始坏**（2.7°、14 mm、23% 失败）；**5 px 时 hard 20.5°、164 mm、66% 失败**。退化是超线性的：最优解沿双模之间的平坦谷滑动（平移误差 ~220 mm），而不是切换到 flip——GT-like 仍为 100%，但那个「GT-like」本身已偏 ~20°，所以这不是模式切换而是谷内滑动。
2. **Laplace/MC 对像素噪声几乎不敏感**（44.2°→41.8°），因为它本来就被近简并偏置（w_other≈0.48，flip 中位 88°）；hard 在旋转上始终更好（clean 2.1 vs 44.2；5 px 20.5 vs 41.8），但优势随噪声缩小。这是「偏差换平滑」的代价，不是精度优势。
3. **粗差是 breakdown，不是渐变**：f=0.005（558 个内点里的 2–3 个随机方向）时 hard 已 71°、405 mm、87% 失败；f=0.05 时 95% 的 draw 找不到 physical mode（K_phys=0），回退 all-minima 后 hard 152°、889 mm、100% 失败。机制是 L2 bearing 代价没有鲁棒核：一个残差 0.05–0.2 m 的粗差贡献 2.5e-3–4e-2 m²，而全部 558 个内点只有 ~1e-3 m²。**这是层的 breakdown，不是检测器的失败——真实前端必须有 outlier rejection / robust kernel，这也正是模拟噪声不能替代训练前端的原因。**

**诚实边界**：(i) 模拟噪声，不是训练好的检测器，没有 appearance/descriptor/训练；(ii) 对应点经 GT 位姿关联，clean 的绝对误差偏乐观，只能用相对退化与 hard/Laplace 排序下结论；(iii) 层只在主平面片上评估，平面片的深度-倾角病态使 5 px 的绝对退化在这一点上偏悲观；两种偏差方向相反，报告里都写明；(iv) 15 帧的 CI 很宽，不做 eggbox/glue 分组结论；(v) 粗差条件在 all-minima fallback 上评估并逐 draw 标记（`phys_ok=False`），那些数字只是 breakdown 指示，不是层在「轻微污染」数据上的精度。

运行 461 s（10 条件 × 15 帧 × 10 seeds，登录节点 CPU），未削减种子；确定性种子写死在脚本里：

```bash
/home/syk/miniconda3/envs/WM/bin/python run_detector_noise.py
```

## 10. 结构化噪声：边界偏置 / 空间相关 / 异方差（模拟前端，2026-11-28 补）

§9 的 i.i.d. 结论只对「方差」负责，不对「独立性」负责。`run_structured_noise.py` 在同一 15 帧、同一固定干净 RANSAC 子集、同一层上跑三种**结构化**前端误差模型，每种都配一个**等方差 i.i.d. 孪生**（同一 `(frame, seed)` 键配对抽取），问题从「噪声更大？」变成「同样方差、换个排列方式，会不会更坏？」。这是**完整跑**（15 帧 × 6 noise seeds × 16 条件 = 1440 draws，每帧每噪声种子 8 个 MC seeds，284 s 登录节点 CPU，非 `--quick`），clean 条件复现 `detector_noise.json` 的 σ=0（最大绝对差 2.9e-12°）。完整报告、图与逐帧数据见 [`results/structured_noise.md`](results/structured_noise.md) / `.json` / `.png`。

**协议**：(a) **边界偏置粗差**——把 f 比例的 bearing 换成 `mask_visib` 的 3×3 形态学轮廓上真实像素的 bearing（中位 309 px/帧，96.2% 轮廓像素有有效深度），替换光线是真实「擦边」光线，不是随机方向；对照为同 f 的均匀随机像素粗差。(b) **空间相关噪声**——每帧一个常数像素偏移 + 每点抖动，1+0.5 px（孪生 σ_eq=1.12 px）与 2+1 px（σ_eq=2.24 px）。(c) **异方差噪声**——σ_i 由可见掩膜距离变换给出，0.5→4 px/λ=6（实测 RMS 1.61 px）与 1→5 px/λ=10（RMS 2.69 px），孪生为**同帧 RMS** 的 i.i.d.。

**结论**（配对差 = 结构化 − 等方差/同 f 对照，95% 帧 bootstrap CI）：

1. **相关性不比等方差 i.i.d. 更坏**：hard 旋转配对差 −0.29 [−0.69, +0.01]°（1.12 px，CI 含 0）与 −3.49 [−5.48, −1.61]°（2.24 px）；2.24 px 时相关臂反而更好（2.04° vs 3.11°），因为公共偏移被位姿吸收，而等方差 i.i.d. 的独立抖动落进平面病态方向。代价落在平滑估计量：Laplace +0.44 [+0.14, +0.79]° / +0.82 [+0.33, +1.38]°，MC +0.59 / +0.99°（CI 不含 0）。**§9 的「σ≤1 px 无害」作为方差结论成立**。
2. **异方差更坏，但只在尾部**：hard 旋转配对均值 +2.76 [+0.07, +6.93]°（H1）/ +10.90 [+1.30, +23.70]°（H2），但配对中位数都只有 +0.15°，ADD-S 中位不可分（7.3 vs 7.6；27.5 vs 30.1 mm），只有失败率动（38% vs 26%，H2）。少数平面子集贴近轮廓的帧被推过近简并：**方差定量级、排列通过近简并的杠杆起作用**，不是典型帧变差。
3. **边界粗差仍是 breakdown，但坏得「安静」**：f=0.10 时 hard 71.28 [43.64, 88.23]° / 447.6 mm vs 均匀粗差 165.07° / 890.6 mm（配对 −82.22 [−94.87, −69.42]°），每个粗差代价更小（3.8°/62 mm vs 21.1°/326 mm，内点 2.2 mm）；但物理模式还在（K_phys=2、fallback 2%），一圈擦边光线把代价自身的最小点整体拉偏——100% draws >10°、GT-like 仅 60%（均匀臂 27%），即**「看起来健康但自信地错」**，比均匀粗差的 all-minima fallback（97–100% 回退）更危险。f=0.05 时边界臂 100% draws 有物理模式（均匀臂 3%），且已 93% 失败。
4. **hard < Laplace/MC 排序在所有结构化臂下保持**（Laplace clean 44.2°、相关 43.7°、边界 f=0.10 时 51.0°）。三种结构里**边界偏置粗差危害最大**，异方差次之（仅尾部），相关性在 1–2 px 对 hard 无害。

**诚实边界**：(i) 仍是模拟 + GT 关联，绝对值偏乐观，只有等方差/同 f 的配对比较承载结论；(ii) RANSAC 平面子集固定，所以「偏移大到能换平面」以及 >2 px 的相关偏移未测；(iii) 15 帧 × 6 noise seeds 的 CI 很宽，只有配对 CI 功效足够，eggbox/glue 不分开下结论；(iv) 1440 draws 中有 3 个默认 384 multi-start 找不到局部极小（约化矩阵近奇异，cond(M)~1e16），用 1024–16384 starts 重跑，未丢弃任何 draw。

```bash
cd labs/01-laplace-qpeplayer/experiments/real/linemod
/home/syk/miniconda3/envs/sdpr/bin/python run_structured_noise.py   # 需要 cv2；284 s CPU
```

- 真实图像前端（2D 检测/描述子）与**检测器级** ADD(-S)/ADD-S 精度表：仍未接，不能声称。**估计器级** ADD-S AUC(0–0.1 m) 已用同一 15 帧的逐帧 ADD-S 补算，见 §10——但它是描述性统计，且在这批数据上恒等于 $1-\overline{\mathrm{ADD\text{-}S}}/0.1$，是 mean ADD-S 的等价改写而非独立测量。

## 11. ADD-S AUC / recall 曲线（15 帧，2026-11-28 补）

论文 Table 5.20 的 LINEMOD AUC 三个格子此前是 `not measured`（LMO 行则有 ADD-S AUC(0–0.1 m)）。本节用 `results/pose_error.json` 里**已经存好的逐帧 ADD-S**（15 帧 × hard/Laplace/MC）补齐这三个格子：不读网格、不重拟合、不插值，每帧只贡献一个数，所以曲线最多 15 个台阶。脚本 `run_auc15.py`，报告 [`results/auc15.md`](results/auc15.md)，数据 [`results/auc15.json`](results/auc15.json)，图 [`results/auc15.png`](results/auc15.png)。

**口径**：$\mathrm{recall}(t)=\frac1n\#\{i:\mathrm{ADD\text{-}S}_i\le t\}$，

$$\mathrm{AUC}(0\text{-}T)=\frac1T\int_0^T\mathrm{recall}(t)\,dt=\frac1n\sum_i\Big(1-\frac{\min(\max(\mathrm{ADD\text{-}S}_i,0),T)}{T}\Big),\quad T=0.1\ \mathrm{m}$$

即 BOP 归一化；第二式是台阶函数的精确积分（也是实际算的），与 20 万点数值积分一致到 2.4e-7。CI 为帧 bootstrap 百分位区间（1000 次，与 `run_pose_error.py` 同一批种子）。

| 估计器 | AUC eggbox (n=8) | AUC glue (n=7) | AUC pooled (n=15) | AUC pooled（等对象权重） |
|-|-|-|-|-|
| hard argmin | 0.952 [0.927, 0.968] | 0.926 [0.865, 0.969] | **0.940 [0.909, 0.964]** | 0.939 [0.908, 0.964] |
| Laplace | 0.895 [0.882, 0.909] | 0.789 [0.761, 0.814] | **0.845 [0.814, 0.875]** | 0.842 [0.827, 0.856] |
| MC | 0.895 [0.880, 0.907] | 0.789 [0.758, 0.813] | **0.845 [0.816, 0.875]** | 0.842 [0.826, 0.856] |

Table 5.20 的 cell 口径是 `median (mean)`，LINEMOD 行对应 hard `0.963 (0.940 [0.909, 0.964])`、Laplace `0.861 (0.845 [0.814, 0.875])`、MC `0.859 (0.845 [0.816, 0.875])`。配对差（同一批重采样帧）：Laplace − hard $-0.094\,[-0.126,-0.064]$，MC − hard $-0.094\,[-0.125,-0.064]$；Laplace 与 MC 只差 1e-4。

**recall（pooled，k/15）**：hard `0.733/0.800/0.933/1.000`、Laplace/MC `0.000/0.267/0.733/1.000`（@5/@10/@20/@50 mm）。@50 mm 三者都是 15/15 饱和（全样本最大 ADD-S 28.2 mm），该列没有信息量；@5 mm 对 Laplace/MC 结构性为 0（其最小 ADD-S 8.4 mm，后验均值落在两模之间）。

**四条必须一起读的话**：

- **这不是一次新测量。** 15 帧的最大 ADD-S 只有 28.2 mm，没有一帧触到 0.1 m 截断，于是 $\mathrm{AUC}=1-\overline{\mathrm{ADD\text{-}S}}/0.1$ **精确成立**（数值差 1.1e-16）：hard 均值 6.0 mm → 0.940、Laplace 15.5 mm → 0.845。用同一批种子做 bootstrap，其 CI 与 `pose_error.json` 里 mean/median ADD-S 的 CI 经单调映射后**逐位相等**（最大差 2.2e-16）。所以填这一格只是让格式与 LMO 行一致；若不愿让人以为多了一个指标，用 mean ADD-S 一列承载同样的信息。
- **n=15，描述性。** 曲线最多 15 个台阶；pooled AUC 的 CI 半宽约 0.027（hard）/0.030（Laplace）。按对象分层（8/7 各自内重采样）会把 Laplace/MC 区间收窄约一半（对象身份解释了相当一部分帧间离散），但正文其它 CI 都是普通帧 bootstrap，故分层只作敏感性、不作主口径。
- **1/15 hard 失败帧 = `000010/72`**（eggbox，44.46°，ADD-S 12.5 mm；Laplace/MC 落在两模之间反而更近，8.4 mm）。它是唯一逐帧排序反转的帧，但不足以让曲线相交：在 $[0,0.1]$ m 上 100 001 点扫描里 hard 的 recall 从不下穿 Laplace/MC。它**也不是**决定 recall@20 mm 的那一帧（那是 `000011/49`，22.4 mm）。
- **单阈值数很脆**：Laplace 的 recall@10 mm（最近帧距阈值 0.25 mm，其自身顶点子采样极差 0.36 mm）、Laplace@20 mm（0.20 vs 0.27 mm）、MC 的 @10/@20 mm 都可能因换一次顶点/种子抽样而 ±1 帧（±0.067）；hard 的 @10/@20 mm 不在这个区间（0.67/2.39 mm vs 0.22/1.32 mm）。这是选 AUC 而不是 recall@t 进表的具体理由，与上面「等价改写」的保留意见是两回事。

```bash
cd labs/01-laplace-qpeplayer/experiments/real/linemod
/home/syk/miniconda3/envs/WM/bin/python run_auc15.py     # ~0.5 s，只读 pose_error.json（+ lmo_two_mode.json 做交叉校验）
```

一句话结论：LINEMOD 的 ADD-S AUC 可以补齐 —— hard **0.940 [0.909, 0.964]** vs Laplace/MC **0.845 [0.814, 0.875]** —— 但在这批数据上它**恒等于 $1-\overline{\mathrm{ADD\text{-}S}}/0.1$**，是已报 mean ADD-S 的等价改写；15 帧只支持描述性结论，不是 benchmark，也不改变「hard 更准、Laplace 用偏差换平滑」的既有判断。
