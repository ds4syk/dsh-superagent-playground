# P1-Q2：10×10 Veronese/Shor 松弛 vs 13×13 SDPRLayers SDP（旋转子问题）

> **一句话结论：不等价。** 13×13 的 24 条 SDPRLayers 约束在 $\mathrm{Sym}^{10}$ 中只张成 17 维，比 10×10 的 21 条（20 维）少了 3 维，少的恰好是三条**行正交**约束 $r_a\cdot r_b=0$；所以 13×13 是 10×10 的一个严格更松的松弛，恒有 $f^\star_{13}\le f^\star_{10}\le f_{\rm nonconvex}$。项目自己的 Choi–Lam 松实例因对称性使 13×13 最优解恰好落在行正交面上，两者数值相等（1.9658119658 / 1.9658119658）；但做 8 个 $10^{-2}$ 量级的对称破缺扰动后，**8/8** 都严格出现 $f^\star_{13}<f^\star_{10}$。

## 0. 口径与证据标签

- `[measured-here]`：本目录 `run_sdp_check.py` 本次运行测得的数值。
- `[quoted-verify]`：引自 `../../verify/verify_theorem0_output.json`（未重跑 verify 脚本）。
- `[exact]`：本脚本内 sympy 有理数精确秩/恒等式。
- 求解器：SCS（cvxpy 1.7.5），`eps=1e-09`，`max_iters=200000`；两次 SDP 都是 10/13 维小问题。
- 13×13 的 24 条约束**逐行复刻** `sdprlayer@62d8ac6` `src/sdprlayers/layers/pose_est.py` L341–L417 的三个生成器（不靠论文文字猜），并与上游 `PolyMatrix` 实现逐元素对拍（§4）。
- 13×13 的 $Q$ 由 $x=[h;\mathrm{vec}_{\rm col}(C);t]$ 的 P6/QPEP 代价组装：$Q_{xx}=\tfrac14 U M_0U^\top$，$Q_{xt}=\tfrac12 U D^\top$，$Q_{tt}=G$；10×10 用 $M=M_0-D^\top G^{-1}D$（对**非凸代价**的精确 $t$ 消去）。
- **没有构造/没有声称**：不声称这些 $M$ 是真实矩阵加权 PnP 的 Gram 矩阵（那是 Q1）；不涉及非凸 P6 的求解；不声称 13×13 等于「二阶 Lasserre」。

## 1. 主表

| instance | f_nonconvex | f_10 | f_13 | gap_10 | gap_13 | rank(X_10*) | rank(X_13*) | solver status | evidence |
|-|-|-|-|-|-|-|-|-|-|
| CL_exact | +2.000000 | +1.965812 | +1.965812 | +0.034188 | +0.034188 | 6 | 6 | 10:optimal / 13:optimal | f_10 quoted-verify; rest measured-here |
| CL_coupled | +2.000000 | +1.965812 | +1.965812 | +0.034188 | +0.034188 | 6 | 6 | 10:optimal / 13:optimal | measured-here |
| CL_perm0 | +1.953444 | +1.952843 | +1.943098 | +0.000601 | +0.010345 | 6 | 4 | 10:optimal / 13:optimal | measured-here |
| CL_perm1 | +1.966813 | +1.960360 | +1.943487 | +0.006453 | +0.023325 | 6 | 4 | 10:optimal / 13:optimal | measured-here |
| CL_perm2 | +1.972141 | +1.959463 | +1.947922 | +0.012678 | +0.024219 | 6 | 4 | 10:optimal / 13:optimal | measured-here |
| rand_psd | +1.102284 | +1.102284 | +1.102284 | +0.000000 | -0.000000 | 1 | 1 | 10:optimal / 13:optimal | measured-here |
| rand_indef | +1.401368 | +1.401368 | +1.401368 | -0.000000 | +0.000000 | 1 | 1 | 10:optimal / 13:optimal | measured-here |
| p6_data | +0.252326 | +0.252326 | +0.252326 | +0.000000 | -0.000000 | 1 | 1 | 10:optimal / 13:optimal | measured-here |
| rowdot | +1.000000 | +1.000000 | -1.000000 | +0.000000 | +2.000000 | 10 | 7 | 10:optimal / 13:optimal | measured-here |
| rowdot_gen | -0.060192 | -0.060192 | -2.029230 | +0.000000 | +1.969037 | 1 | 4 | 10:optimal / 13:optimal | measured-here |

- `gap_10 = f_nonconvex - f_10`，`gap_13 = f_nonconvex - f_13`（$\ge0$ 表示松弛下界；$=0$ 表示紧）。
- `CL_exact` 的 `f_10` 实测 1.965811965845，与 `[quoted-verify]` 的 1.9658119658452384 相差 0.0e+00；两者是同一构造，验证了 10×10 部分被原样复用。
- `rowdot` / `rowdot_gen` 是**构造的反例**（见 §3），不是 PnP 实例；其余行是随机或 Choi–Lam 实例。

### 1b. Choi–Lam 的 8 个对称破缺扰动（同一 $M$ 族的「松」实例）

| instance | f_nonconvex | f_10 | f_13 | f_10 - f_13 | rank(X_10*) | rank(X_13*) | max&#124;rowdot&#124; @ X_13* |
|-|-|-|-|-|-|-|-|
| CL_pert0 | +1.980669 | +1.962860 | +1.960078 | +0.003 | 6 | 5 | +0.784 |
| CL_pert1 | +1.972055 | +1.963143 | +1.956564 | +0.007 | 6 | 4 | +0.588 |
| CL_pert2 | +1.969200 | +1.961223 | +1.953436 | +0.008 | 6 | 4 | +0.702 |
| CL_pert3 | +1.974848 | +1.958632 | +1.956179 | +0.002 | 6 | 5 | +0.878 |
| CL_pert4 | +1.975934 | +1.968825 | +1.957548 | +0.011 | 6 | 5 | +0.717 |
| CL_pert5 | +1.976729 | +1.964571 | +1.950275 | +0.014 | 6 | 4 | +0.848 |
| CL_pert6 | +1.976337 | +1.965149 | +1.961762 | +0.003 | 6 | 5 | +0.721 |
| CL_pert7 | +1.985283 | +1.966638 | +1.960776 | +0.006 | 6 | 4 | +0.808 |

- `max|rowdot| @ X_13*` = 13×13 最优解 $X_r^\star$ 上三条 $r_a\cdot r_b$ 残差的最大绝对值；它就是 13×13 与 10×10 唯一的行为差异。

## 2. 为什么不等价（精确结构，不是数值巧合）

- 10×10 的 21 条约束张成 **20** 维（`[exact]`；`[quoted-verify]` 也是 20）。
- 13×13 的 24 条约束只含 $(h,\mathrm{vec}\,C)$（$t$ 块恒为 0），投影回 $\mathrm{Sym}^{10}$ 后张成 **17** 维（`[exact]`）。
- $\mathrm{span}(24)\subseteq\mathrm{span}(21)$（并集秩 20）。把 21 条逐条投影到 $\mathrm{span}(24)$：只有 **`row_orth_r1r2`、`row_orth_r1r3`、`row_orth_r2r3`** 残差非零（1.63–1.73），其余 18 条残差 ≤ 6.2e-16。
- 即 $C_{10}=C_{13}\cap\{r_1\cdot r_2=r_1\cdot r_3=r_2\cdot r_3=0\}\subsetneq C_{13}$；行**范数** $\|r_i\|^2=h^2$ 已由 9 条 row/col-norm + 列正交线性组合得到，缺的恰好只有 3 条行**点积**。（这与 Q13 猜的「10 维理想 + $t$ 块」不同：13 维的约束秩是 17，不是 20。）
- 因为 24 条约束都不含 $t$ 且 $G\succ0$，对 $t$ 块做 Schur 补可**精确**消去：$f^\star_{13}=\min_{X_r\in C_{13}}\langle Q_{10},X_r\rangle$。每行都数值核对 `schur_err = |f_13 - <Q_10,X_r^*>|`，最大 4.2e-09；`Q_xx - Q_cost - Q_xt G^{-1}Q_xt^T` 的最大元素为 9.9e-15（`[measured-here]`）。
- 因此 $f^\star_{13}\le f^\star_{10}$ 对一切实例成立；等号 $\iff$ 13×13 最优解满足行正交（等价地：13×13 的最优解同时是 10×10 的可行点）。

## 3. 数值结论（哪些支持、哪些否定「同值 + 同秩」）

- **CL_exact（支持，但只是巧合）**：$f^\star_{10}=1.9658119658$，$f^\star_{13}=1.9658119658$，差 3.7e-11，rank 均 6；13×13 最优解的 rowdot 残差 ≤ 8.8e-15。它是对称性使 13×13 最优解恰好落在 $C_{10}$ 内。
- **8 个对称破缺扰动（否定）**：8/8 严格 $f^\star_{13}<f^\star_{10}$，差 2.5e-03–1.4e-02；$\mathrm{rank}(X_{13}^\star)\in\{4,5\}$ 而 $\mathrm{rank}(X_{10}^\star)=6$；rowdot 残差最大 0.88（= 13×13 最优解明显不在 $C_{10}$ 内）。
- **构造反例 `rowdot`（否定，且差距最大）**：$Q=-r_1\cdot r_2+\tfrac14 I$（在 $S^3$ 上恒等于 1 的四次型），$f_{\rm nonconvex}=f^\star_{10}=1.000000$（10×10 紧，rank 10），而 $f^\star_{13}=-1.000000$，差 **2.00**，$\mathrm{rank}(X_{13}^\star)=7$。`rowdot_gen` 加 $10^{-2}$ 随机扰动后差距仍有 1.97，说明这不是退化特例的数值假象。
- **紧实例（平凡支持）**：`rand_psd`、`rand_indef`、`p6_data` 上 $f^\star_{10}=f^\star_{13}=f_{\rm nonconvex}$（两者都 rank 1）。这解释了「大量随机/真实 PnP 实验看起来等价」：松弛一旦紧，两个可行集给出同一个值。
- **`CL_coupled`**：与 `CL_exact` 同一 reduced $M$、但 $D\ne0,G\ne I$；两行的 $f^\star_{10}$、$f^\star_{13}$ 分别只差 1.6e-15 / 1.5e-13，验证 $(M_0,D,G)\to Q$ 组装与 $t$ 消去实现正确。
- **消融（13×13 只用 6 orth + 9 handedness，15 条）**：每个实例上 $f^\star_{13,15}\le f^\star_{13}$，最大下降 1.00e+00，说明那 9 条 row/col-norm 确实在收紧——但补上它们**仍缺** 3 条行点积，所以 13×13 永远达不到 10×10 的 20 维约束空间。
- **`p6_data`**：$Q$ 从随机矩阵加权 P6 对应点组装，再 $Q\to(M_0,D,G)\to Q'$ 往返，相对误差 1.9e-16，确认 13×13 的 $Q$ 就是 SDPRLayers 的 P6 代价、10×10 的 $M$ 就是它的精确 $t$ 消去。

## 4. 构造与可复现性

- 10×10：`build_10x10()` 逐行复用 `verify/verify_theorem0.py`（sympy 精确 $U$、行主序 `vec`、21 条约束），回归值 = 1.9658119658 vs `[quoted-verify]` 1.9658119658452384。
- 13×13：`build_13x13()` 逐行复刻上游；`--check-upstream /home/syk/labs/SDPR/code/sdprlayer` 用 AST 抽取上游三个生成器的**原始源码**执行，与本脚本复刻逐元素对拍：**24/24 完全一致，max |diff| = 0.0**；在随机旋转上 $\max|x^\top A_ix|=1.4e-16$。
- 目标组装：`build_Qfull()`（$Q_{xx},Q_{xt},Q_{tt}$），并用 Schur 分解恒等式与 P6 往返各查一次。
- 非凸值：160 次多起点 SLSQP + 黎曼 Newton 精修，外加 200000 个稠密球面采样点取 min；`CL_exact` 得 2.000000（解析真值 2.0，因为 $C(q)\ge0$ 且 $C(e_0)=0$）。
- 全部随机数种子固定（`seed=12345`；实例构造用 `default_rng(0)`、优化用 `12345`），结果可逐位复现。
- 原始数据：`results/sdp_check.json`（含每行的特征值、对偶界、rowdot 残差、耗时）。复现：

```bash
cd labs/01-laplace-qpeplayer/experiments/sdp
/home/syk/miniconda3/envs/WM/bin/python run_sdp_check.py \
    --check-upstream /home/syk/labs/SDPR/code/sdprlayer
```

## 5. 结论（一段话）

P1-Q2 的「13×13 与 10×10 同最优值、同秩相关解」**不成立**：13×13 的 24 条 SDPRLayers 约束（6 正交 + 9 handedness + 9 row/col-norm）在 $\mathrm{Sym}^{10}$ 中只张成 17 维，比 10×10 的 21 条（3 列范数 + 3 列点积 + 3 行范数 + 3 行点积 + 9 handedness，20 维）少 3 维，少的恰好是三条行正交约束 $r_a\cdot r_b=0$；因此 13×13 是 10×10 的一个**严格更松**的松弛，恒有 $f^\star_{13}\le f^\star_{10}\le f_{\rm nonconvex}$。项目自己的 Choi–Lam 松实例因为对称性使 13×13 最优解恰好满足行正交，两者数值一致（1.9658119658，rank 均 6），这很可能就是「看起来等价」的来源；但把该实例做 8 个 $10^{-2}$ 量级的对称破缺扰动后，8/8 都严格出现 $f^\star_{13}<f^\star_{10}$（差 2.5e-03–1.4e-02，$\mathrm{rank}(X_{13}^\star)$ 从 6 降到 4,5），而构造目标 $-r_1\cdot r_2+I/4$ 给出 $f^\star_{10}=1.00$、$f^\star_{13}=-1.00$ 的 2.0 差距。**可用话术**：定理 0.3 的 10×10 Veronese/Shor 结论只覆盖「消去 $t$、且带 3 条行正交约束」的旋转子问题；把它写成「SDPRLayers 的 13×13 实现」是错的，除非额外证明 13×13 的最优解落在行正交面上（即这 3 条缺失约束恰好无效）。按当前证据，Q2 应改写为「一般情形已证否不等价（$f^\star_{13}\le f^\star_{10}$，构造实例差 2.0）+ 对称实例上数值巧合相等」。

