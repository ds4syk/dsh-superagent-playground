# Laplace-QPEPLayer 合成实验（2026-11-27）

> **一句话结论**：在可控双峰位姿实例上，硬 argmin 在分支交叉处没有可用的有限差分梯度；多峰 Laplace 的解析梯度与有限差分一致到约 $4\times10^{-4}$（交叉处）/ $4\times10^{-6}$（远离交叉），而 mode-level Monte Carlo 的梯度误差约 $8.5\times10^{-2}$、跨种子方差 $10^{-1}$ 量级。这验证了“平滑层在交叉处仍可微、且比采样方差低”的方向性结论。

## 0. 范围与限制

- 合成、已知真值的双峰 QPEP 实例（2 个全局极小 + 2 个局部极小）；没有真实数据集（LINEMOD/UTIAS 未下载）。
- 这是**定理/机制验证**，不是论文主表；真实数据/端到端训练是下一阶段。
- 所有数字由 `code/run_main.py` 生成，原始 per-theta 记录在 `results/raw_grid.json`；固定种子；Slurm CPU 运行。

## 1. 设置

### 1.1 实例（`code/instance.py`）

- $q\in\mathbb S^3$，$v_2(q)\in\mathbb R^{10}$，$J(q)=v_2(q)^\top M(\theta)v_2(q)$。
- 两个对称位姿 $q_A=e_0$、$q_B=e_3$（绕 z 轴 180°）；通过 180° 对称的 10×10 正交矩阵 $P$ 构造 $P$-不变的对称双井，使 $J_A(0)=J_B(0)$，即 $\theta=0$ 是分支交叉。
- $M(\theta)=M_{\rm sym}+\theta M_{\rm break}$，$M_{\rm break}$ 破坏对称性，使 $d(J_B-J_A)/d\theta|_0=2.0$。
- 平移块固定 $G\succ0,D$，$t^*(q)=-G^{-1}Dv_2(q)$。
- $\theta$ 网格 $-0.2..0.2$（81 点）；$\tau=0.01$；FD 步长 $10^{-4}$；MC 每点 200 样本 × 20 种子。

### 1.2 方法

| 方法 | 说明 |
|---|---|
| Hard | 枚举全部极小，取 $J$ 最小者；在选中极小处用 3×3 黎曼 IFT |
| Laplace | $w_k=\mathrm{softmax}(-J_k/\tau-\tfrac12\log\det H_k)$；平滑位姿与解析梯度（含 $\nabla\log\det H_k$ 与权重协方差项） |
| MC | 按 $w_k$ 采样 mode，score-function 估计 $\mathrm d\mathbb E[\phi]/\mathrm d\theta$；20 种子估方差 |

## 2. 代码校验（先验后跑）

- `ift_dq` 对分支跟随 FD：误差 $5\times10^{-9}$（修复了缺少因子 2 的 bug）。
- `dpose_dq` 对 FD：误差 $2\times10^{-9}$（修复了 einsum 下标错误）。
- **闭式 $\mathrm d\log\det H/\mathrm d\theta$（2026-11-27 修复）**：`core.dlogdetH_dtheta` 经 `dHsphere_dq -> dE` 收缩。审计发现 `qpep_core.dE` 的 `d1/d2/d3` 在第 1–2 行有错项，导致该路径偏差 $O(1)$（$8\times10^{-2}$–$2.8\times10^{-1}$）；修正后与独立闭式 `dlogdet_exact` 一致到 $8.9\times10^{-16}$、与分支 FD 一致到 $1.8\times10^{-9}$。主表已切换到该闭式路径，数值列与旧 numeric fallback **逐位相同**。回归测试 `code/test_derivatives.py`。
- `dlogdet_numeric`：显式 $\theta$ 部分 + 切空间 $q$ 部分，保留作交叉校验（对分支 FD $3\times10^{-8}$）。
- **FD 参考步长（2026-11-27）**：交叉附近 Laplace 映射的变化尺度是 $\tau/\text{slope}\approx5\times10^{-3}$，主表原用 $h=10^{-4}$ 在 $\theta=0$ 已进入截断区；`results/fd_step.md` 显示 $4.34\times10^{-4}$ 严格按 $h^2$ 缩放（$h=10^{-5}\to4.34\times10^{-6}$，$h=10^{-6}\to4.34\times10^{-8}$）。所以主表 `at` 行 $4\times10^{-4}$ 应读作 **FD 截断底**，不是解析公式误差。
- Laplace 权重导数：必须用 $\dot g=-\dot J/\tau-\tfrac12\dot{\log\det H}$，不能用 $g$ 本身；修复后远/近/交叉处分别 $4\times10^{-6}/10^{-4}/4\times10^{-4}$。
- 硬 argmin 在 $\pm h$ 内分支切换时 FD 标为 N/A，不用错误跳变填表。

## 3. 命令（Slurm）

```bash
cd labs/01-laplace-qpeplayer/experiments/code
srun --partition=cpu --time=00:40:00 --ntasks=1 \
  /home/syk/miniconda3/bin/python run_main.py
```

输出：`../results/main_table.md/.json/.png`、`raw_grid.json`。`fit` 默认关闭（`FIT_SEEDS=0`）；设为 >0 可生成 `fit_table` 诊断。

## 4. 主表

| regime | n | Laplace grad rel | MC grad rel | Hard grad rel | MC pose std | t Laplace (ms) |
|-|-|-|-|-|-|-|
| far ($\lvert\theta\rvert>0.05$) | 61 | **4.7e-7** | 4.7e-7 | 4.7e-7 | 2.8e-16 | 13.1 |
| near ($0.005<\lvert\theta\rvert<0.05$) | 19 | **3.5e-5** | 9.4e-5 | 2.2e-5 | 2.9e-16 | 18.5 |
| at ($\lvert\theta\rvert<0.005$) | 1 | **4.3e-4**（FD 截断；$h=10^{-6}$ 时 $4.3\times10^{-8}$） | 3.3e-2 | **N/A** | **1.1e-1** | 51.4 |

> 计时列是空载 CPU 测量；2026-11-27 晚在 5 个并行子任务负载下复跑同一脚本得到 17.7/34.6/89.5 ms，数值列逐位不变。引用计时请用 `results/robustness.md` 的重复测量中位数，并注明机器负载。

- 远离交叉：三种方法都等价于硬 argmin（全局模式权重≈1），梯度与 FD 一致到 $10^{-7}$–$10^{-5}$；MC 方差≈0（所有样本落在同一模式）。
- 交叉处：硬 argmin 在 $\pm h$ 内切换分支，FD 无定义（N/A）；Laplace 的解析梯度仍与平滑后验的 FD 一致到 **$4.3\times10^{-4}$**；MC 的误差 **$3.3\times10^{-2}$**、跨种子位姿标准差 **$1.1\times10^{-1}$**。这正是“Laplace 比采样方差低、且交叉处仍可微”的方向性证据。
- 交叉处 Laplace 的 $4\times10^{-4}$ 残差是 **FD 参考的截断**（严格 $h^2$，见 `results/fd_step.md`），不是解析公式误差：解析梯度对 $h=10^{-6}$ 的 FD 只差 $4.3\times10^{-8}$。
- **Fold（2026-11-27 实测）**：多峰 Laplace 映射在 $\lvert\theta\rvert\approx1.315\times10^{-3}$ 处有真实 fold；次级极小在该邻域外消失（$K:4\to3$），接近 fold 时 $\det H\to0$、体积权重 $1/\sqrt{\det H}$ 发散、$|\mathrm d\,\mathrm{pose}|$ 从 548 跳到 8362 再落到 412。不能靠更小 $\tau$ 或更多初值修复；"交叉处光滑"只能限定在排除该邻域的区间。
- **Fold 的部分修复（2026-11-27）**：把 $\det H$ 换成 $\det(H+\tau^{1/3}I)$（导数仍闭式 $\mathrm{tr}((H+\epsilon I)^{-1}\dot H)$）后，$|\mathrm d\,\mathrm{pose}|$ 尖峰 3529→443、近 fold 梯度误差 2.90e-1→1.19e-2，远离 fold 与标准版一致；但模式生灭造成的**值**跳变只从 0.309 降到 0.107，$K$ 仍 4→3。即正则化修导数奇点、不修值不连续；后者需要 Airy 一致展开或对合并模式做局部数值积分。见 `results/fold_6d.md`。
- **P1-Q5b/Q5c（2026-11-27 证明，理论结果）**：**命题 P1-Q5b** 证明任何“只对实局部极小求和 + 对合并模式加正则化下界”的实根代理，在 fold 处必然有 $\Theta(\tau^{1/3}e^{-(J_0-J_{\text{glob}})/\tau})$ 量级的**值跳变**（与实测 0.309→0.107 一致，不是实现 bug）；**定理 P1-Q5c** 用 40 根 Action Matrix 在 fold 另一侧自然生成的复共轭根对做半线 Airy/Scorer 一致延拓，证明跨 fold 严格 $C^\infty$、值跳变恒为 0，且 $\zeta\gg1$ 时回到 3D Laplace。**2026-11-27 晚更新：6D Scorer 延拓已实现**（`theory/run_scorer_fold.py`，结果 `theory/results/scorer_fold.md`）：1D 三路复现（标准 0.867→0、$\tau^{1/3}$ 0.496→0、Scorer 0.33838→0.33834）；6D fold $\theta_f=1.3185\times10^{-3}$ 的复共轭根对追踪成功，四个不变量跨 fold 连续（$J_0$ 相对残差 $1.7\times10^{-12}$），位姿最大跳变 **1.21e-2**（标准 3.04e-1、$\tau^{1/3}$ 1.07e-1），jump/step 6.0e2 与真实 $|\mathrm d\,\mathrm{pose}|$ 同量级。限制：6D fold 很浅（$\Delta J\ll\tau$）、规范形是首阶近似（$\varphi''''\approx8.3$）。工程主表仍未启用复根追踪。

![主表](results/main_table.png)


## 5. 真实数据（BOP LINEMOD，`real/linemod/`）

旧「对称标注混合」构造是负结果（mm/m 单位 + 共享 t 消去两个原因，见 `real/linemod/README.md`）。替代构造 `run_two_mode.py` 在 **15/15 帧** eggbox/glue 上得到真实双峰代价（真实深度 bearing + 最近 mesh 顶点 + 主平面片 RANSAC + $z>0$ 过滤）：$K=2$、$\Delta J\in[1.25\times10^{-4},2.56\times10^{-3}]$、错误模式 Laplace 权重 0.45–0.61。

诚实结论：**真实近简并处 Laplace 不是精度优势**——hard argmin 中位旋转误差 2.06°（1/15 帧失败），Laplace 后验均值中位 44.24°（投影 chordal mean；旧的 58.28° 是未投影 9-vector 值，不是 geodesic 误差）（被错误模式拉偏）。同一真实几何上的受控横截交叉（两个标注各自消去平移）给出 Laplace 解析梯度 rel **1.9e-10–1.8e-9**，MC **9e-5–1.7e-1**。详见 `real/linemod/results/two_mode.md`。

## 6. Q2/Q13 修订：满 20D 等价，上游 17D 严格更松（2026-11-27）

- **修订版定理 P1-2**：上游 `sdprlayer@62d8ac6` 的 24 条约束在 $\mathbb{S}^{10}$ 上只张成 17 维（缺 3 条行正交 $r_1^\top r_2=r_1^\top r_3=r_2^\top r_3=0$）。补齐这 3 条后的**满 20 维** 13×13 与 10×10 **同值同秩**；上游 17 维 13×13 对任意实例恒满足 $f^\star_{13,\text{up}}\le f^\star_{10,\text{full}}\le f^\star_{\text{QPEP}}$，且可严格更松。Q2/Q13 **结案**。
- **6 点物理 3D-3D 反例**（`sdp/run_6point_check.py`）：$p_i=\pm e_1,\pm e_2,\pm e_3$，$y_i=\varepsilon p_i$，$\varepsilon=0.05$，共享严格正定权重 $W$（eigs $0.4/1/1.6$），$D=0$，$G=6W$。实测 $f^\star_{10}=f^\star_{13,\text{full}}=5.41500000$（rank 1），$f^\star_{13,\text{up}}=3.59431373$（rank 4），差 $1.82068627$；10×10/full-20D 在该实例上严格 rank-1 紧，所以上游 17D 的间隙**不能**归因于“数据退化”。与证明值 $3.59430886$ 差约 $5\times10^{-6}$；这**不是** SCS 容差：精确有理数对偶证书给 $f^\star_{13,\text{up}}\ge3.594313732834$，证明值比下界低 $4.87\times10^{-6}$，36 个自然变体无一命中（`sdp/results/f13up_forensics.md`）；建议改引 $3.5943137\ldots$ / $1.8206863\ldots$。数值区间证书：`sdp/results/six_point_cert.md`；图：`sdp/results/q2_summary.png`。表见 [`sdp/results/six_point.md`](sdp/results/six_point.md)。
- 扰动：`sdp/run_six_point_perturb.py` 的 10/10 个 $W/y$ 小扰动保持间隙 $[0.106,1.821]$ 严格正。复核：`./run_all.sh --with-sdp` 会跑 6 点检查 + 证书 + Q2 图（约 35 s），并在 `--with-scorer` 下追加 Scorer 延拓；输出归档到 `results/logs/<ts>/outputs/`；确定性见 `REPRO.md` §4.1。

## 7. 限制

- 合成双峰实例；$\theta=0$ 处两个全局极小严格简并，Laplace 的 FD 参考在简并处受模式集变化影响（$4\times10^{-4}$ 已确认是 FD 截断，见 `results/fd_step.md`）。
- 真实数据实验只到「几何代价 + 层可微性」层，没有图像前端，不能给 ADD(-S)/AUC 精度 claim。
- MC 是 mode-level 采样，不是 $SE(3)$ 上的连续 MCMC；它验证的是“离散模式权重采样方差”，不是 EPro-PnP 的完整采样器。
- 优化拟合是诊断（目标为两分支平均位姿），不是训练一个真实前端。

---

# 2026-11-27 补充证明后的主表 / 消融

作者提供了 P1 补充证明（Feishu wiki `AlOnwYsYdi7jJ8kiHd7c9UvRnfe`，2026-11-27）。本目录新增两层实验：

## A. 定理 P1-5 的精确后验验证（1D normal form）

模型 $J=(x^2-1)^2(1+cx)+gx$，$c=0.25$：$g=0$ 时两个极小恰在 $x=\pm1$、能量相等（$\Delta J=0$），曲率分别为 $5$ 和 $3$；$g=0.2$ 为分离盆地。精确后验用 2M 点梯形积分，Laplace 用两个局部极小。

| regime | $\tau$ | $\Delta J$ | $E_{\rm exact}$ | $E_{\rm Lap}$ | $\|E_{\rm Lap}-E_{\rm exact}\|$ | 梯度相对误差 |
|-|-|-|-|-|-|-|
| crossing | 0.1 | 0 | -0.122930 | -0.127017 | 4.09e-3 | 3.73e-2 |
| crossing | 0.01 | 0 | -0.126649 | -0.127017 | 3.67e-4 | **3.44e-3** |
| crossing | 0.005 | 0 | -0.126834 | -0.127017 | 1.83e-4 | 1.71e-3 |
| separated | 0.01 | 0.401 | -1.030838 | -1.032296 | 1.46e-3 | 6.27e-3 |

- 期望值误差在 $\Delta J=0$ 处仍是 $O(\tau)$：log–log 斜率 **1.066**（crossing）/ **1.036**（separated）。
- 梯度相对误差随 $\tau$ 下降（3.7e-2 → 3.4e-3 → 1.7e-3），支持定理的 $O(\tau)$ 相对界；**严格证明仍需补 Q5 的梯度余项引理**（见下）。
- 结果文件：`theory/results/uniform_unequal_table.md`、`uniform_unequal.json`；脚本 `theory/run_unequal.py`。

## B. Fold（模式物理合并）与 $\tau^{1/3}$ 正则化

模型 $J=x^4/4-a x^2/2$；$a\to0$ 时两极小与鞍点合并，$H=2a\to0$。标准 Laplace 的 $1/\sqrt{H}$ 发散；正则化用 $H\to H+\tau^{1/3}$。

| $\tau$ | $a$ | $H$ | 标准 Laplace 相对误差 | 正则化相对误差 |
|-|-|-|-|-|
| 0.01 | 0.1 | 0.2 | 0.166 | 0.191 |
| 0.01 | 0.02 | 0.04 | 1.911 | **0.152** |
| 0.01 | 0.01 | 0.02 | 3.235 | **0.234** |
| 0.005 | 0.01 | 0.02 | 2.518 | **0.138** |

- 标准 Laplace 在 $a\to0$ 时误差爆炸（3.2× / 2.5×）；$\tau^{1/3}$ 正则化把误差压到 0.14–0.23，验证 regime II 的 Airy 标度思路。
- 结果文件：`theory/results/fold_table.md`、`fold.json`；脚本 `theory/run_fold.py`。

## C. 6D QPEP 消融

| 消融 | 结果 |
|-|-|
| $\tau$ sweep（交叉处） | $\tau=0.05/0.02/0.01/0.005$：梯度误差 1.20e-3 / 5.84e-4 / **4.34e-4** / 1.51e-3；远处约 $8.5\times10^{-7}$ |
| logdet：精确闭式 vs 切空间 FD | 两者一致到 $10^{-8}$ 量级；$\theta=0$ 处梯度差 2.2e-8 |
| logdet：去掉体积因子 | 交叉处位姿差 **0.0647**、梯度差 **53.4**；near 处位姿差 7.8e-4、梯度差 0.44 |
| top-K 模式（$\theta=0$） | K=1 位姿差 **1.44**、梯度差 **499**；K=2 位姿差 0.105、梯度差 68；K=4 精确 |
| P1-Q6 闭式公式 | 5 个随机实例对分支 FD 的相对误差 **1.7e-9–8.8e-9** |

文件：`results/ablation.md`、`ablation.json`、`theory/results/q6_verify.md`；脚本 `code/run_ablation.py`、`theory/verify_q6.py`。

## D. 补充证明核验结论

- **完备（证明层面）**：P1-Q1/Q9（Carathéodory 物理实现 + 正测度）、P1-Q2/Q13、P1-Q3（40 根）、P1-Q6（闭式 logdet）、P1-Q7（Haar 常数）、P1-Q5 的**期望值**部分。
- **Q2/Q13 修订结论（2026-11-27）**：`sdp/run_sdp_check.py` 逐行复刻 SDPRLayers 的 24 条约束后发现，13×13 的约束在 $\mathrm{Sym}^{10}$ 上比 10×10 的 21 条**少 3 条行正交** $r_1\cdot r_2=r_1\cdot r_3=r_2\cdot r_3=0$，所以**上游 17 维实现**对任意实例恒有 $f^\star_{13,\text{up}}\le f^\star_{10,\text{full}}$，且一般**严格更松**：构造反例 $Q=-r_1\cdot r_2+I/4$ 给出 $f^\star_{10}=+1.0$、$f^\star_{13}=-1.0$（差 2.0）；Choi–Lam 实例两者相等（1.9658119658，rank 6）只是对称性巧合，8 个 $10^{-2}$ 破缺扰动后 **8/8** 严格更松。**补齐 3 条行正交后的满 20 维 13×13 则与 10×10 同值同秩**（修订版定理 P1-2），并用 6 点真实物理反例（$f^\star_{10}=f^\star_{13,\text{full}}=5.41500000$ rank 1 vs 上游 $f^\star_{13,\text{up}}=3.59431373$ rank 4）证明上游 17 维的严格松弛在物理实例上也会发生。详见 `sdp/results/sdp_check.md` 与 `sdp/results/six_point.md`。
- **fold 核验（2026-11-27）**：多峰 Laplace 映射在 $|\theta|\approx1.31\times10^{-3}$ 处有模式生灭（fold），$\det H\to0$、权重尖峰、$|\mathrm d\,\mathrm{pose}|$ 峰值 3529（常规 379），且映射本身在该点不连续；这不是漏根，也不能靠更小 $\tau$/更多初值修复。详见 `results/robustness.md`。
- **可复现**：`run_all.sh`（任意 cwd）默认 6 脚本 **35.1 s**，`--with-sdp` **39.0 s**（6 点检查 +1.5 s），`--with-linemod --with-sdp` **43.0 s**；非计时字段与已提交结果逐位一致，`six_point.json` 两次运行字节相同。`REPRO.md` 记录 clean-cwd、确定性与 B1–B11 脆弱点。
- **仍需补**：**P1-Q5 的跨交叉梯度一致界**仍是证明草图（注意：2026-11-27 的 **Q5b/Q5c 只关闭 Fold 连续性**，不是这件事；以并行的 `theory/proofs.md` 更新为准）：定理要求 $\|\nabla E-\nabla\bar\ell\|=O(1)$ 且相对误差 $O(\tau)$，但需要 (i) 局部余项 $\partial_\theta r_k$ 的界，(ii) 交叉的横截性/真梯度 $\Theta(1/\tau)$ 的下界。1D 数值算例（slope 1.07，relG 3.4e-3 @ $\tau=0.01$）支持该结论，但还不是严格证明。
- P1 级（非 P0）仍 open：Q4（精确充要条件）、Q8（$\tau$ 学习/尺度）、Q11（重根梯度）、Q12（EPro 公平对比）、Q14（措辞）、Q15（群轨道平均）。

![P1 补充证明后的主表/消融](../../../docs/visuals/2026-11-27-p1-theory.png)

## 8. 真实数据位姿误差 / ADD-S 与外部基线（phase-4, 2026-11-27）

- **位姿误差 / ADD-S（15 帧）**：`real/linemod/run_pose_error.py`，结果 `real/linemod/results/pose_error.{md,json,png}`。hard argmin 旋转中位 **2.06° [1.47,3.49]**、ADD-S 中位 **3.7 mm [3.0,4.3]**；Laplace（**投影 chordal mean**）中位 **44.24° [33.90,46.87]**、ADD-S **13.9 mm [10.3,20.2]**，配对偏差 +36.9° [+27.2,+44.9]；MC 同 estimand（seed SD 1.011°）。**旧的 58.28° 是未投影 9-vector mean，不是 geodesic 误差**；`two_mode.md` 顶部已加更正块。ADD-S 用 1000 mesh 查询点 vs **完整 GT mesh** KD-tree（与 full-vs-full 差 ≤0.51 mm）。
- **经典基线**：`real/linemod/run_classical_baselines.py`（sdpr env 的 cv2 5.0）。同 15 帧、同一 RANSAC 平面对应；SQPNP 中位 2.04°/ADD-S 4.6 mm（15/15 真模式），EPnP 2.05°（12/15），IPPE 2.21°（15/15），LM(identity) 106°（3/15）、LM(GT) 2.06°。**前向求解器，不是可微层；LM 不是 BPnP**。见 `results/classical_baselines.md`。
- **Theseus 基线**：`real/linemod/run_theseus_baseline.py`（theseus venv，复用 P3 `theseus_cost/run_theseus` 模式）。目标对账 1.1e-13；GT init 2.06°/15-15；identity 178.82°/1-15；random 100/300 GT；8-init multistart 11/40。**局部优化器无枚举/物理过滤，无 z>0 时存在面朝后伪极小（2/15 帧代价更低）**；见 `results/theseus_baseline.md`、`THESEUS_BASELINE.md`。
- **40 根计数**：`roots/run_rootcount_fast.py`（total-degree homotopy，4 chart 取并）：**4/4 实例 40 个复射影根**（80 仿射根），实射影 26/12/12/20、极小 5/2/2/4，残差 ≤1.2e-13，24 s；**数值完整性，不是符号商环上界**。见 `results/rootcount_fast.md`。
- 一页图：`real/linemod/results/real_summary.png`（P1 vs 经典/局部基线，旋转误差与 ADD-S 中位+CI）。
