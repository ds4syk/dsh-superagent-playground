# f*_{13,up} 取证：3.59430886 能否由 6 点实例的任何自然变体重现？

> **结论：不能。** 精确有理数对偶证书证明该 24 约束 13×13 SDP 的**真最优值下界为 $3.594313732834$**，比证明引用的 $3.59430886$ **高 4.87e-06**。因此 3.59430886 不可能是该 SDP 的最优值：它只能是（i）某个更松/不同 SDP 的值、（ii）非最优但对偶可行的数值（欠收敛的对偶解 / 安全余量），或（iii）转写错误。本报告测试了全部要求的自然变体：**36 个自然变体里 0 个**在 $10^{-8}$ 内命中 3.59430886。

复现：`/home/syk/miniconda3/envs/WM/bin/python sdp/run_f13up_forensics.py`（新文件；不修改既有脚本/结果）。运行 147.2 s。

## 0. 口径与证据标签

- `[measured-here]`：本脚本本次运行；`[exact]`：sympy 有理数精确构造/精确 LDL^T；`[mpmath-60]`：60 位十进制特征值。
- 实例与 `run_6point_check.py` 完全一致：`import run_6point_check.build_Q10` 取 float 目标；另用 sympy 有理数重建同一目标，`max|Q_exact-Q_float| = 0.0e+00`（`[exact]`）。
- 约束：10×10 21 条（满 20D，精确秩 20）；13×13 上游 24 条（秩 17）；13×13 full 27 条。
- SCS `eps=1e-11/1e-12, max_iters=2e6`；CLARABEL 默认/`tol_*=1e-12`；所有 SDP 都保留 $X\succeq0,\ X_{00}=1$。

## 1. 高精度基线与对偶证书（核心）

| 量 | SCS 1e-11 | SCS 1e-12 | CLARABEL default | CLARABEL 1e-12 |
|-|-|-|-|-|
| $f^\star_{10}$ (10×10, 21) | 5.4150000000 | 5.4150000000 | 5.4150000025 | 5.4150000003 |
| $f^\star_{13,\mathrm{full}}$ (27) | 5.4150000000 | 5.4150000000 | 5.4149999989 | 5.4150000000 |
| $f^\star_{13,\mathrm{up}}$ (24) | 3.5943137329 | 3.5943137338 | 3.5943137133 | 3.5943137329 |

对偶证书（显式解 $\max\mu$ s.t. $Q-\mu E_{00}-\sum_i\lambda_iA_i-\eta I\succeq0$，再用**精确有理数 LDL^T** 验证 $Q-\mu E_{00}-\sum\lambda_iA_i\succ0$）：

| SDP | 证书下界 $\mu_{\rm low}$ | 余量 $\eta$ | 精确 PD | 最小主元 | mpmath 60 位 $\lambda_{\min}$ @ $\mu_{\rm low}$ |
|-|-|-|-|-|-|
| f10 | 5.414999999996236 | 1e-12 | True | 3.764e-12 | 9.410116531436419419488953e-13 |
| f13_full | 5.414999999995948 | 1e-12 | True | 4.051e-12 | 1.012797754546683290220774e-12 |
| **f13_up** | 3.594313732834349 | 1e-11 | True | 1.003e-11 | 8.031321016887189783294163e-12 |

- **$f^\star_{13,up}$ 的严格区间**：$[\mu_{\rm low}, V_{\rm prim}] = [3.594313732834,\ 3.594313732875]$（SCS 原始值，等式残差 $8.9e-16$；宽度 $\sim10^{-10}$）。CLARABEL 默认原始值 $3.5943137133$ 比证书下界还低 $2.0e-08$，因为它的 $X$ 有 $\sim10^{-9}$ 不可行度（不是更精确的值）。
- 对 $f^\star_{13,up}$ 还有一个**精确有理上界**：$X=xx^\top$，$x=[1,\mathrm{vec}(I),0]$ 精确可行，目标 $=6(1-\varepsilon)^2=1083/200=5.415$。因此 $f^\star_{13,up}\in[3.594313732834,5.415]$；结合 SCS 原始值可把它夹到 $10^{-10}$ 内。
- mpmath 复算：在 $\mu_{\rm low}$ 处对偶松弛的最小特征值为 8.031321016887189783294163e-12（60 位十进制）；未加余量（$\eta=0$）时该值仅 -9.787103159293687876404777e-12，即浮点层面的 0。

**关键数字**：证书下界比证明值高 **4.873e-06**（相对 1.36e-06）。任何可行点目标值都不可能低于该下界，所以 3.59430886 不是该 SDP 的最优值；它比最优值低 4.873e-06。

## 2. 变体扫描（值 / 秩）

| 变体 | 说明 | 值 | rank(1e-6) | rank(1e-5) | 与基线差 | 命中 3.59430886? |
|-|-|-|-|-|-|-|
| `base_f13up_scs1e-11` | 13x13, 24 upstream constraints, SCS 1e-11 | 3.5943137329 | 4 | 4 | +0.000e+00 | no |
| `base_f13up_scs1e-12` | same, SCS 1e-12 | 3.5943137338 | 4 | 4 | +9.631e-10 | no |
| `base_f13up_clarabel` | same, CLARABEL default | 3.5943137133 | 6 | 4 | -1.958e-08 | no |
| `base_f10` | 10x10, 21 constraints (full 20D) | 5.4150000000 | 1 | 1 | +1.821e+00 | no |
| `base_f13full` | 13x13, 24 + 3 row-orth = 27 constraints | 5.4150000000 | 1 | 1 | +1.821e+00 | no |
| `base_rot_only_10x10` | 10x10 with the 24 upstream constraints mapped to the rotation block (t-eliminated) | 3.5943137329 | 4 | 4 | -2.705e-13 | no |
| `i_Xtt_zero` | 13x13 + explicit X_tt=0 (t-block eliminated) | 3.5943137329 | 4 | 4 | +4.610e-13 | no |
| `i_Xxt_zero` | 13x13 + explicit X_xt=0 | 3.5943137329 | 4 | 4 | -7.994e-15 | no |
| `i_Xtt_I` | 13x13 + t-block normalised to X_tt=I (adds tr G = 18) | 21.5943137329 | 7 | 7 | +1.800e+01 | no |
| `i_traceX_eq4` | replace X00=1 by trace(X)=4 (rank-1 trace normalisation) | 3.5943137356 | 4 | 4 | +2.679e-09 | no |
| `i_X00_ge1` | relax X00=1 to X00>=1 | 3.5943137329 | 4 | 4 | +3.935e-13 | no |
| `ii_no_perm` | objective Q10 inserted without the row/col-major permutation P | 5.4150000000 | 1 | 1 | +1.821e+00 | no |
| `ii_inverse_perm` | objective with P^T Q10 P (P is an involution, so = correct) | 3.5943137329 | 4 | 4 | +0.000e+00 | no |
| `ii_transposed_C` | objective built for C^T (J(R^T)), mapped with P | 5.4150000000 | 1 | 1 | +1.821e+00 | no |
| `ii_constraints_rowmajor` | col-major objective Q13 against row-major-interpreted 24 constraints | 5.4150000000 | 1 | 1 | +1.821e+00 | no |
| `iii_h2_to_1_equiv` | diag orthogonality C_i.C_i=1 (h^2 term moved to RHS); algebraically identical with X00=1 | 3.5943137329 | 4 | 4 | -2.388e-12 | no |
| `iii_drop_h2` | drop h^2 term and set C_i.C_i=0 (not equivalent) | 0.0150000000 | 1 | 1 | -3.579e+00 | no |
| `iv_roworth_s_0` | 24 upstream constraints + s * (3 row-orthogonality); s=0 | 3.5943137329 | 4 | 4 | +3.610e-12 | no |
| `iv_roworth_s_0.5` | 24 upstream constraints + s * (3 row-orthogonality); s=0.5 | 5.4150000000 | 1 | 1 | +1.821e+00 | no |
| `iv_roworth_s_0.9` | 24 upstream constraints + s * (3 row-orthogonality); s=0.9 | 5.4150000000 | 1 | 1 | +1.821e+00 | no |
| `iv_roworth_s_0.99` | 24 upstream constraints + s * (3 row-orthogonality); s=0.99 | 5.4150000000 | 1 | 1 | +1.821e+00 | no |
| `iv_roworth_s_1` | 24 upstream constraints + s * (3 row-orthogonality); s=1 | 5.4150000000 | 1 | 1 | +1.821e+00 | no |
| `iv_roworth_first_1` | 24 upstream constraints + first 1 of the 3 row-orthogonality constraints | 5.4150000000 | 1 | 1 | +1.821e+00 | no |
| `iv_roworth_first_2` | 24 upstream constraints + first 2 of the 3 row-orthogonality constraints | 5.4150000000 | 1 | 1 | +1.821e+00 | no |
| `iv_roworth_first_3` | 24 upstream constraints + first 3 of the 3 row-orthogonality constraints | 5.4150000000 | 1 | 1 | +1.821e+00 | no |
| `iv_roworth_relaxed_0.01` | 24 upstream + |<rowdot_i,X>| <= 0.01 (genuine relaxation) | 5.4049780598 | 4 | 4 | +1.811e+00 | no |
| `iv_roworth_relaxed_0.001` | 24 upstream + |<rowdot_i,X>| <= 0.001 (genuine relaxation) | 5.4139976958 | 4 | 4 | +1.820e+00 | no |
| `iv_roworth_relaxed_0.0001` | 24 upstream + |<rowdot_i,X>| <= 0.0001 (genuine relaxation) | 5.4148997685 | 4 | 2 | +1.821e+00 | no |
| `v_scs_eps_1e-05` | 13x13 upstream, SCS eps=1e-05 | 3.5943131358 | 6 | 4 | -5.970e-07 | no |
| `v_scs_eps_1e-06` | 13x13 upstream, SCS eps=1e-06 | 3.5943138399 | 4 | 4 | +1.070e-07 | no |
| `v_scs_eps_1e-09` | 13x13 upstream, SCS eps=1e-09 | 3.5943137329 | 4 | 4 | +0.000e+00 | no |
| `v_scs_eps_1e-11` | 13x13 upstream, SCS eps=1e-11 | 3.5943137329 | 4 | 4 | +0.000e+00 | no |
| `v_scs_eps_1e-12` | 13x13 upstream, SCS eps=1e-12 | 3.5943137338 | 4 | 4 | +9.631e-10 | no |
| `v_scs_default_1e-4` | 13x13 upstream, SCS default-ish eps=1e-4 | 3.5946157830 | 10 | 6 | +3.021e-04 | no |
| `v_clarabel_tight` | 13x13 upstream, CLARABEL tol 1e-10 | 3.5943137324 | 4 | 4 | -4.849e-10 | no |
| `v_clarabel_loose` | 13x13 upstream, CLARABEL loose (gap 1e-4, feas 1e-6) | 3.5943129803 | 7 | 7 | -7.525e-07 | no |

- **(iv) 行正交尺度 $s$**：$s=0$ 给基线；$s>0$ 的值全部跳到 $\approx5.415$，$s\in\{0.5,0.9,0.99,1.0\}$ 之间没有连续插值——因为约束是齐次的，$s\cdot\langle A_i,X\rangle=0$ 与 $\langle A_i,X\rangle=0$ 同解。真正的离散插值是“只加前 $k$ 条”：$k=0$（基线）是 $3.59431373$，而 $k=1,2,3$ 全部跳到 $5.415$——只要补上 $r_1\cdot r_2$ 这一条，松弛就塌到满 20D 的紧值；$r_1\cdot r_2$ 是真正起作用的那一条。
- **(ii) 布局/转置**：不做 $P$ 或换 $P^\top$、用 $J(R^\top)$ 都给出 $\approx5.415$（或与基线完全相同，因为 $P$ 是对合）；没有任何布局变体给出 3.5943 附近的 $\sim10^{-6}$ 偏移。
- **(iii) $h$ vs $1$**：把对角正交约束的 $-h^2$ 项移到右端（$C_i\cdot C_i=1$）在 $X_{00}=1$ 下**代数恒等**，值与基线逐位一致；真正删掉 $h^2$ 项（$C_i\cdot C_i=0$）给出 $0.015000$，差 O(1)。
- **(i) t-块**：显式 $X_{tt}=0$ / $X_{xt}=0$ / $X_{00}\ge1$ 都不改变值（$G\succ0$ 强制 $t^\star=0$）；$X_{tt}=I$ 或 $\mathrm{tr}(X)=4$ 改变归一化，值显著不同。
- **(v) 求解器**：SCS 从 $\varepsilon=10^{-4}$ 到 $10^{-12}$、CLARABEL 松/紧，值都在 $3.59431$ 附近，**全部高于** 3.59430886；松弛的 SCS 反而偏高（$3.5946$），CLARABEL 最松的原始值也只到 $3.59431298$，仍比证明值高 $4.1\times10^{-6}$。
- **约束消融**：前 6/15/18 条分别 2.316000 / 2.402057 / 2.607504；前 20 条起就已是 3.59431373（9 条 row/col-norm 里后 4 条在原 24 条下冗余）。逐条删除 24 条中的任意一条：离 target 最近的值仍差 4.87e-06（即基线本身），有些删除会显著降低到 3.398，但从未产生 3.59430886。

## 3. `rowdot` 反例 10 位一致性检查

| 量 | SCS 1e-11 | CLARABEL | 对偶证书下界 | rank(1e-6) | 证明 claim |
|-|-|-|-|-|-|
| f10 | 1.0000000000 | 1.0000000000 | 1.0000000000 | 10 | +1.0 |
| f13_up | -1.0000000000 | -1.0000000000 | -1.0000000000 | 7 | -1.0 |
| f13_full | 1.0000000000 | 1.0000000000 | 1.0000000000 | 10 | n/a |

- $f^\star_{10}=+1.0000000000$、$f^\star_{13,\mathrm{up}}=-1.0000000000$、gap $=2.0000000000$（10 位）；对偶证书分别给出 $\ge 0.999999999996$ 与 $\ge -1.00000000004$。与 claim $+1.0/-1.0$ 一致。
- 13×13 最优解上的 rowdot 期望（= $2r_a\cdot r_b$）：2.000, 0.000, 0.000，即上游松弛取 $r_1\cdot r_2\approx1$，把目标压到 $-1$。

## 4. 数据敏感性（排除“邻近实例”）

- 把 $W$ 的非对角元 $\rho$ 从 $0.6$ 改成 $0.599/0.601$：$f^\star_{13,up}$ 分别 3.5982856241 / 3.5903417214，斜率 $df/d\rho=-3.972$。要把 3.59431373 变到 3.59430886 需要 $\Delta\rho\approx1.23e-06$——即 $\rho=0.6000012$ 这种非自然数据微扰；证明写的是精确的 $-0.6$，所以“换了实例”也解释不了这个数。

## 5. 判定

**(a) 稍有不同的 SDP 的值？** 不是（在我们测试的自然变体族里）。布局/转置、$h$ vs $1$、$t$-块归一化、行正交尺度 $s$、约束消融、SCS/CLARABEL 容差、甚至 $W$ 的 $10^{-6}$ 数据扰动，都给不出 3.59430886；要么同值（$3.59431373$），要么差 $O(0.1$–$1.8)$。

**(b) 求解器 artefact？** **机制上最可能**：3.59430886 比证书下界低 $4.87\times10^{-6}$，而它恰好可以作为“对偶可行但非最优”的 $\mu$（对偶值单调且合法地给出下界）；证明原文也写“对偶 SDP 精确数值验证”。欠收敛/带 PSD 安全余量的对偶解会系统性地偏低。但用本机 SCS/CLARABEL 的任何容差都**没有**复现出这 5 位：SCS 松容差偏高（$3.5946$），CLARABEL 最松也只到 $3.59431298$。

**(c) 转写错误？** 不能排除，但**不是**两位数字的单纯笔误：$3.59431373\to3.59430886$ 在有效数字第 6 位整体错位，且 $1.82069114=5.415-3.59430886$ 与它自洽（同一个错值派生，不是两个独立笔误）。

**建议的表述**：把 $f^\star_{13,up}$ 写成 $3.5943137\ldots$（或 $\approx3.59431$），并用证书区间 $[3.5943137328,\ 3.5943137329]$；把 $f^\star_{10}-f^\star_{13,up}$ 改成 $1.8206863\ldots$。证明的定性结论（gap $>1.8$、rank $1$ vs $>1$、严格更松）不受影响。

## 6. 复现与输出

```bash
cd labs/01-laplace-qpeplayer/experiments/sdp
/home/syk/miniconda3/envs/WM/bin/python run_f13up_forensics.py
```

- 原始数据：`results/f13up_forensics.json`（含所有变体、逐条删除、证书明细、mpmath 值）。
- 本脚本只**新增**文件；`run_6point_check.py`、`run_sdp_check.py`、既有 results 未改动。
