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
