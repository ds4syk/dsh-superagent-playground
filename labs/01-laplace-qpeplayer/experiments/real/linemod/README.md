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

## 8. LMO 扩展（2026-11-28）

同一套「真实平面片双模代价 + hard/Laplace/MC」在 **BOP LMO** 上的复测，用来检验 LINEMOD 结论是否只对无遮挡数据成立。脚本 `run_two_mode_lmo.py` 完全复用 `run_two_mode.py`/`run_pose_error.py` 的 helper（未改动那些文件），只把模块级 `ROOT` 指到 LMO；指标、阈值、RANSAC、900 点、`n_starts=384`、`tau=0.01`、MC 设置都与 LINEMOD 表一致。完整报告 [`results/lmo_two_mode.md`](results/lmo_two_mode.md)，数据 [`results/lmo_two_mode.json`](results/lmo_two_mode.json)，图 [`results/lmo_two_mode.png`](results/lmo_two_mode.png)。

- **数据**：LMO 只有一条 test 序列 `000002`（1214 张图）；obj 10/11 的 GT 实例 1175/903 个，可见像素 ≥1200 的候选 1031/726 个。按数据可用性等距各抽 20 帧，**37/40 帧通过 K≥2 物理模式过滤**（eggbox 20/20，glue 17/20；失败 2 帧有效深度 <900、1 帧 K=1）。**K_phys=2 在 37/37 帧**，中位 ΔJ=1.37e-3，错误模 Laplace 权重中位 0.474。
- **hard argmin 变差但主因不是模式排序**：旋转中位 2.95°（LINEMOD 2.06°）、平移中位 19.5 mm（6.2 mm）、ADD-S 中位 10.2 mm（3.7 mm）。hard 在 34/37 帧选中的就是离 GT 最近的物理模式，但该最近模式本身在 15/37 帧 >5°；约化代价在 GT 处的值中位比最小值高 17%。2× starts 复算 37/37 帧 K 与 hard 完全不变（最大差 8.6e-13°），排除枚举漏解。机制是 LMO 可见平面片小（中位 61 mm）+ 深度-模型残差尾部重（p50 1.4–9.3 mm）；`n_plane≥350` 子集 hard 中位回到 2.29°/7.1 mm。
- **Laplace/MC 仍偏向翻转模**：旋转中位 43.7°/43.5°（LINEMOD 44.2°），ADD-S 中位 19.0 mm（13.9 mm）；对 hard 的配对偏差 +31.2° [+22.9, +38.9]、+9.7 mm ADD-S。方向与 LINEMOD 一致，没有精度优势。
- **ADD-S AUC(0–0.1 m)**：hard 0.862 [0.812, 0.903]，Laplace/MC 0.766；n=37 时曲线只有 ≤37 个台阶，只作描述性统计。
- **口径与 caveat**：主 ADD-S 用 `models/`（sha256 与 LINEMOD 网格一致），`models_eval` 作为次要列；AUC 不是 BOP 排行榜口径；对应点仍由 GT 位姿构造，不是检测器 claim；最重遮挡（无法提供 900 个有效深度像素）不在样本内。

```bash
cd labs/01-laplace-qpeplayer/experiments/real/linemod
/home/syk/miniconda3/envs/WM/bin/python run_two_mode_lmo.py            # 全量 ~36 s
/home/syk/miniconda3/envs/WM/bin/python run_two_mode_lmo.py --quick    # smoke
```

一句话结论：LMO 上「双模结构 + Laplace 被翻转模拉偏」的方向不变，但 **hard argmin 的精度在遮挡/小平面片/深度残差下明显退化**；这是该构造在真实遮挡数据上的一个诚实限制，而不是新的精度 claim。

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
