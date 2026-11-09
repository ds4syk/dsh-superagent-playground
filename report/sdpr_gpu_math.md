# SDPR 的 GPU 并行实现：数学解析

> **结论先行**：整个实现可以拆成三层数学：① 把 (P6) 的齐次 QCQP 松弛成 SDP，并取它的对偶；② 用对偶 log-det barrier 沿中心路径求 `X(μ)=μH⁻¹`，对偶间隙恰好是 `nμ`，所以 `μ→0` 同时给出全局解和证书；③ 对 QCQP 的 KKT 条件做隐式微分，得到 `∂L/∂Q`。GPU 的“并行”只是把这三层里的 13×13 eigh、18×18 Newton 系统和 20×20 KKT 系统按 batch 维一次做完。

**如果只记三件事**：

1. **SDP 对偶**：`min <Q,X> s.t. <A_i,X>=b_i, X⪰0` 的对偶是 `max -bᵀy s.t. H(y)=Q+Σy_iA_i⪰0`；`H` 就是证书。
2. **Barrier**：`min_y bᵀy - μ log det H(y)` 的最优点满足 `X(μ)=μH⁻¹` 可行、`<A_i,X>=b_i`，且对偶间隙 `=nμ`。
3. **隐式微分**：对 `Hx=0, xᵀA_i x=0, xᵀA_0x=1` 全微分，解 `M=[H G_rᵀ; G_r 0]`，反向一次线性求解即得 `∂L/∂Q`。

## 0. 记号与问题

- `x ∈ R^n`，`Q, A_i ∈ S^n`（对称矩阵），`b_i ∈ R`；`X ∈ S^n_+`（半正定）。
- (P6)：`x = [h, vec_col(C), t] ∈ R^13`，`n=13`；约束为 24 个 `xᵀA_i x=0` 加 1 个齐次约束 `xᵀA_0x=1`（`A_0=e_he_hᵀ`）。
- 25 个约束矩阵的秩是 18（9 个范数约束的秩只有 2）；实现里保留 18 个线性无关约束，丢弃 7 个范数约束的线性组合。
- 目标矩阵 `Q(θ)` 由测量 `(m_k, y_k, W_k)` 组装；`θ` 是待标定的 baseline（外层参数）。

## 1. QCQP → Shor SDP

原问题（齐次 QCQP）：

$$
\min_{x}\ x^{\top} Q x
\quad\text{s.t.}\quad
x^{\top} A_i x = 0\ (i=1,\dots,m),\qquad
x^{\top} A_0 x = 1 .
$$

令 `X = x xᵀ`，把 `xᵀ A x` 写成 `<A, X> = tr(A X)`，并丢掉 `rank(X)=1`：

$$
\min_{X}\ \langle Q, X\rangle
\quad\text{s.t.}\quad
\langle A_i, X\rangle = b_i,\quad
X \succeq 0,
$$

其中 `b_i=0`（`i≥1`），`b_0=1`。这是 Shor 松弛；`X` 的可行域是 PSD 锥与仿射子空间的交。

## 2. 对偶：证书矩阵 H

对等式约束引入乘子 `y_i`，对 `X⪰0` 引入 `Z⪰0`：

$$
\mathcal{L}
= \langle Q,X\rangle
+ \sum_i y_i\bigl(b_i-\langle A_i,X\rangle\bigr)
- \langle Z,X\rangle
= \Bigl\langle Q-\sum_i y_i A_i - Z,\ X\Bigr\rangle
+ b^{\top}y .
$$

对 `X` 取极小，驻点条件给出

$$
Z = Q-\sum_i y_i A_i \succeq 0 .
$$

因此对偶是

$$
\max_{y}\ b^{\top}y
\quad\text{s.t.}\quad
Q-\sum_i y_i A_i \succeq 0 .
$$

**代码里的符号**：令 `y_code = -y`，并定义

$$
H(y) = Q+\sum_i y_i A_i ,
$$

对偶变成

$$
\max_{y}\ -b^{\top}y
\quad\text{s.t.}\quad
H(y)\succeq 0 .
$$

互补松弛：在最优点 `H X = 0`。`H` 就是 SDPR 里的**证书矩阵**：`H⪰0` 且 `Hx=0`；当 `X=xxᵀ` 时，`x` 是 `H` 的零空间向量。

> 直观：`H` 同时是“原问题最优性的证明”（对偶可行 + 互补松弛）和“梯度的载体”（下一节的 KKT 矩阵用它）。

## 3. 对偶 barrier 与中心路径

把 `H(y)⪰0` 换成 `H(y)≻0` 上的对数行列式 barrier：

$$
\min_{y}\ \varphi_\mu(y)
= b^{\top}y - \mu \log\det H(y),
\qquad H(y)\succ 0 .
$$

`μ>0` 是 barrier 参数。梯度与 Hessian：

$$
\nabla \varphi_\mu(y)_i
= b_i - \mu\,\langle A_i, H^{-1}\rangle,
$$

$$
\bigl[\nabla^2 \varphi_\mu(y)\bigr]_{ij}
= \mu\,\langle A_i,\, H^{-1} A_j H^{-1}\rangle
= \mu\,\langle B_i, B_j\rangle,
\quad B_i = L^{\top} A_i L,\ \ H^{-1}=LL^{\top}.
$$

若 `A_i` 线性无关，Hessian 正定；实现里保留的 18 个约束正是线性无关的。

**驻点条件**：

$$
b_i = \mu\,\langle A_i, H^{-1}\rangle
\quad\Longleftrightarrow\quad
\langle A_i,\, X(\mu)\rangle = b_i,
\qquad
X(\mu) := \mu H^{-1} \succeq 0 .
$$

所以 `X(μ)` 天然是原问题的**可行点**（满足全部等式约束且 PSD），不需要额外投影。

**对偶间隙**：记原目标 `p = <Q,X>`，对偶目标 `d = -bᵀy`。利用 `H = Q + Σ y_i A_i` 与 `<A_i,X>=b_i`：

$$
\langle H, X\rangle
= \langle Q,X\rangle + \sum_i y_i \langle A_i,X\rangle
= p + b^{\top}y
= p - d .
$$

而 `X=μH⁻¹`，所以

$$
\langle H,X\rangle = \mu\,\mathrm{tr}(H^{-1}H) = \mu\,\mathrm{tr}(I_n) = n\mu .
$$

于是

$$
\boxed{\ p - d = n\mu\ } .
$$

这就是“精度”的数学来源：`μ=1e-8`、`n=13` 时 gap `=1.3e-7`。`μ→0` 时 `X(μ)` 收敛到原 SDP 最优解，`H(y)` 收敛到对偶最优证书。

**Newton 迭代**：每个 `μ` 阶段解

$$
\nabla^2\varphi_\mu(y)\, \Delta y = -\nabla\varphi_\mu(y),
$$

沿 `Δy` 做线搜索，要求 `H(y+αΔy)≻0` 且 `φ` 下降；`μ` 按 `0.1` 衰减。实现里 `μ: 1e-2 → 1e-8`。

**从 H 恢复 x**：在最优处 `H X=0`、`H⪰0`、`X⪰0`。若松弛是 tight 的，`rank(X)=1`，`X=xxᵀ`，于是 `Hx=0`；`x` 就是 `H` 的最小特征值对应的特征向量（代码按 `x_0=1` 归一化）。若非 tight，`H` 的零空间维数 `>1`，`x` 不唯一，此时 SDPR 的全局最优保证不成立；实现用 `λ₁/λ₂` 和 `corank(H)` 检查 tightness。

## 4. 为什么需要“严格可行起点”

Barrier 要求存在 `H(y)≻0`（对偶 Slater 条件）。P6 的约束张成空间包含

$$
P = I_C + A_0 = A_{00}+A_{11}+A_{22}+4A_0
$$

（`I_C` 是 `C` 的 9 个元素的单位阵）。`P` 在 `(h,C)` 上是单位阵、在 `t` 上为零；而 `Q` 的 `t` 块是 `Σ_k W_k ≻ 0`，所以对足够大的 `s`，

$$
H_0 = Q + sP \succ 0 .
$$

代码用倍增找 `s`。这一步是 P6 专用的；一般 QCQP 需要 Phase-I（例如 `max t s.t. H(y)-tI⪰0`）。

## 5. 隐式微分：Theorem 2 的矩阵形式

在 tight 情形，原 QCQP 的 KKT 条件是

$$
H x = 0,\qquad
x^{\top} A_i x = 0\ (i=1,\dots,m),\qquad
x^{\top} A_0 x = 1,
$$

其中 `H = Q + Σ λ_i A_i`。把 `z=(x,λ)` 对参数 `θ` 全微分。只有 `Q` 依赖 `θ`（P6 实验里 `A_i` 固定），令 `G_r` 为**线性无关**约束的梯度行：

$$
G_r = \bigl[\,A_i x\,\bigr]_{i\in\mathcal{I}},\qquad
\mathcal{I}=\{\text{orthogonality}\}\cup\{A_0\},
$$

则

$$
\underbrace{
\begin{bmatrix}
H & G_r^{\top}\\
G_r & 0
\end{bmatrix}
}_{M}
\begin{bmatrix}
dx\\
d\lambda
\end{bmatrix}
=
-\begin{bmatrix}
dQ\,x\\
0
\end{bmatrix}.
$$

因此解映射的 Jacobian 是

$$
J = -P\,M^{-1}
\begin{bmatrix}
dQ\,x\\
0
\end{bmatrix},
\qquad P=[I_n\ \ 0].
$$

**反向模式**：给定 `∂L/∂x = g`，解一次（对称）系统

$$
M\,y = \begin{bmatrix} g\\ 0\end{bmatrix},
\qquad y=(y_1,y_2),
$$

则

$$
\boxed{\ \frac{\partial L}{\partial Q}
= -\frac12\bigl(x\,y_1^{\top} + y_1\,x^{\top}\bigr)\ }.
$$

推导：`dL = <g,dx> = -<y,[dQx;0]> = -<y_1,dQx> = -tr(y_1ᵀ dQ x) = -<y_1xᵀ,dQ>`；对对称 `dQ` 取对称化即得上式。实现里 `y` 是一次批量 `torch.linalg.solve`，`n+r=20`。

**为什么 `H` 不唯一也没关系**：有冗余约束时，乘子 `λ` 可以沿约束梯度的零空间移动，`H` 也随之改变；但 Theorem 2 证明 `x` 的 Jacobian 唯一（在 ACQ + SOSC + `H⪰0` 下）。实现里 `G_r` 只取 7 个独立约束，去掉了冗余方向；数值上 GPU 的 `H` 与 SCS 的 `H` 可以逐元素不同（单实例诊断中相对差可达 ~2.8e-2），但两者都满足 `H⪰0`、`Hx=0`，且算出的 `∂L/∂Q` 一致。

## 6. 为什么不用原始 ADMM（数学上）

原始 ADMM 在 `X` 与 `Z` 之间分裂，`Z`-步做 PSD 投影 `Π_+(·)`。当最优 `X` 是 rank-1（PSD 锥边界）时，投影把负特征值截断为 0；`X`-步的等式约束与 `Z`-步的投影在边界上形成 dead-zone，primal residual 停在 `O(1e-4)`，`H` 出现负特征值，证书不成立。

对偶 barrier 不投影，而是让 `H` 始终 `≻0`，用 `μ→0` 逼近边界；`X(μ)=μH⁻¹` 始终可行，gap 可控。这是“边界解”与“内点”的差别。

## 7. 批量并行的复杂度

每个 Newton 迭代、batch 大小 `B`：

| 操作 | 复杂度 | 说明 |
|---|---|---|
| `eigh(H)` | `O(B n³)` | 主要开销；`n=13` |
| `B_i = LᵀA_iL` | `O(B m n³)` | `m=18`，批量 matmul |
| Hessian `G` | `O(B m² n²)` | Gram 矩阵 |
| 解 `G Δy = -∇φ` | `O(B m³)` | `m=18`，批量 solve |
| 线搜索 Cholesky | `O(B n³)`/试探 | 通常 1–2 次 |

`B` 个 SDP 之间没有数据依赖，所以 batch 维是“天然并行”的；GPU 的 eigh/Cholesky 一次处理 `B` 个 13×13 矩阵。实测 `B=20→200` 前向时间几乎不变（0.072→0.078 s），而单进程 CPU SCS 线性增长（0.95→11.7 s）。

## 8. 精度与条件数

- 中心路径 gap `=nμ`；`μ=1e-8` → `1.3e-7`。
- `X(μ)` 与最优 `X*` 的误差 `O(μ)`；实测 `x` 相对误差 ~`7e-6`。
- `H` 的条件数 `~1/μ`；`μ` 越小，Newton 系统越病态。`μ=1e-9` 时 `cond(H)~1e9`，线搜索在双精度下失败，所以当前实现停在 `1e-8`。
- 梯度误差还受 KKT 矩阵 `M` 的条件数影响；实测 batch 梯度的相对误差 ~`1.6e-5`（单 pose 的绝对误差 ~`6.6e-5`）。

## 9. 一句话总结

$$
\boxed{
\text{SDPR-GPU}
=
\underbrace{\text{Shor SDP dual}}_{H(y)\succeq0}
+
\underbrace{\text{log-det barrier}}_{\mu\to0,\ \text{gap}=n\mu}
+
\underbrace{\text{Theorem 2 implicit diff}}_{M^{-1}\text{ one linear solve}}
}
$$

GPU 版本没有改变这三层的数学，只是把每层的 `B` 个 13×13 / 18×18 / 20×20 线性代数从“逐 pose 的 Python/CPU 循环”变成“一次批量 GPU 调用”。

## 参考

1. C. Holmes, F. Dümbgen, T. D. Barfoot. *SDPRLayers: Certifiable Backpropagation Through Polynomial Optimization Problems in Robotics.* IEEE T-RO 41 (2025); arXiv:2405.19309.
2. S. Boyd, L. Vandenberghe. *Convex Optimization.* Cambridge University Press, 2004（对偶、log-det barrier、中心路径）。
3. 本仓库实现：`src/gpu_sdpr/barrier.py`（barrier + KKT）、`src/gpu_sdpr/layer.py`（autograd）、`repro/cleanroom/implicit_diff.py`（CPU 参考）。
