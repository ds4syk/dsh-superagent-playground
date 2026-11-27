# Robust-QPEPLayer 实验（2026-11-27）

> **一句话结论**：在合成 3D-3D 上，GNC-GM 与 RANSAC 在 0–70% 随机外点下成功率同为 100%、精度在两位小数内持平；对 v0 的固定 1000 次迭代 RANSAC 基线，GNC 在 p≥10% 时快 **31–33×**（p50：1.48–1.63 ms vs 48–52 ms），且可微。换成自适应 RANSAC 后，低外点率下 GNC 不再有速度优势（p=10% 时慢 3×），p≥60% 时重新领先（p=70% 约 8×）。相干多数外点在 60–70% 让**所有方法**返回外点位姿（Prop 3.7 的可辨识性失败，不是求解器 bug）；边界外点主要毒害梯度而不是前向成功率。6D 梯度表：精确 IFT vs FD 的相对误差中位数 **4.9e-8**（内点）/ **3.1e-8**（边界），cosine **1.000000**；naive 在边界 bin 相对误差 **1.07**、cosine **0.234**，在 e>c²/3 的观测上映射层与位姿平移层的径向符号均 **100% 相反**。

---

## 0. 状态与范围

| 项 | 状态 |
|---|---|
| 合成 3D-3D 主表（随机/相干/边界外点 + runtime p50/p95 + RANSAC 预算 + N scaling） | **完成**，`synthetic/results/main_table_extended.*` |
| 6D 梯度表（exact IFT vs Naive IRLS-StopGrad vs FD；含 `-λ` 消融） | **完成**，`synthetic/results/gradient_table.*` |
| End-to-end 边界坐标精化（exact vs naive 梯度） | **完成**（GT 监督的 test-time 诊断，不是学习层），`synthetic/results/e2e_table.*` |
| 已知真值/闭式/FD 校验 | **完成**，`synthetic/results/verify_smoke.txt` |
| 真实数据：ModelNet40/10、3DMatch 8 个官方 test 场景（4 场景子集 + 4 新场景，含 FPFH）、KITTI raw 2 drives、KITTI odometry 00–10（每序列 33 协议帧） | **完成**（子集），见 §12 |
| KITTI raw 立体前端回退 | **完成（负结果）**：旋转 0.052°，平移 0.70 m，0% @0.1 m；见 §16 |
| UTIAS In-The-Dark | **完成（2 白天 + 2 夜间前缀）**：repeat spatial × teach temporal × 相机系共轭；day 0.078–0.092°/0.010–0.020 m；night17/18 dense 0.135/0.150°，trans 0.011 m；sparse 0.197°；均 100% @0.1 m；见 §16 与 `real/inthedark/` |
| 3DMatch 8 场景 / low–mid–high overlap 扩展 | **完成**：8 场景 pooled random 0–70% GNC/RANSAC 100%，70% 27.7×；low/mid/high 重叠 bins 均 100%@p=0.7；`real/threedmatch_full/` |
| KITTI odometry 00–10 | **完成（每序列 33 帧子集）**：11/11 序列有覆盖；pooled 99/99/100/100% at p=0/.3/.5/.7，22.4×；seq08 p=0 有 80% 真实 bad-basin；`real/kitti_odom/` + `real/kitti_more/` |
| QPEP 40 根多项式求解器 | **未实现**（当前主表用标量权重 4×4 特化）；范围见 `qpep_gpu/qpep_scope.md` |
| GPU / CUDA-graph probe | **见 §13 与 `qpep_gpu/results/qpep_gpu.md`**（执行状态以该文件为准） |

本目录包含合成、真实数据、系统与消融实验；不修改 `theory/`、`paper/`、`verify/`。所有数字由本目录脚本生成；原始 per-trial 记录按实验放在各自 `results/*_raw.jsonl`。

---

## 1. 设置

### 1.1 数据与指标

* $N=100$（梯度/精化实验 $N=30$ 或 $50$），$x_i\sim U(-1,1)^3$，随机旋转 $R_{gt}$，$t_{gt}\sim U(-0.5,0.5)^3$，噪声 $\sigma=0.01$。
* 残差 $r_i=R(q)x_i+t-y_i$，$e_i=\|r_i\|^2$；GM 核 $\rho(e)=c^2e/(c^2+e)$，$c=0.05$。
* 成功率：旋转误差 $<5^\circ$ 且平移误差 $<0.1$ m；旋转误差用 trace/arccos（近零有 ~1e-6° 的数值地板）。
* runtime 为单次调用的 wall-clock，报告 p50/p95；100 trials/cell（scaling 50 trials/cell）。

### 1.2 外点类型

| 类型 | 生成方式 | 目的 |
|---|---|---|
| random | 被替换的 $y_i\sim U(-2,2)^3$ | v0 协议，主表基线 |
| coherent | 被替换的 $y_i=R_{out}x_i+t_{out}$，$R_{out}=R_\Delta R_{gt}$，$R_\Delta$ 为 30–60°、平移 0.6–1.5 m | 单一竞争刚体变换；检验多数外点崩溃 |
| boundary | $y_i=R_{gt}x_i+t_{gt}+r_i$，$e_i/c^2\sim U(1/3,3)$ | 半外点/最毒梯度带；前向不一定失败 |

### 1.3 方法

| 方法 | 说明 |
|---|---|
| Kabsch | 非鲁棒加权 Kabsch（全 1 权重） |
| RANSAC-1000 | 3 点采样，固定 1000 次迭代，阈值 $\|r\|<c$，内点重拟合；v0 基线 |
| RANSAC-adaptive | 同上，但用标准置信界 0.999 提前停止（上限 1000） |
| IRLS-GM | GM 核 IRLS，40 次，不升温 |
| GNC-GM (ours) | GNC 升温 + 加权 Kabsch 内层，末 5 步 μ=1 IRLS |

> 前向是 3D-3D 的闭式加权 Kabsch（QPEP 的 3D-3D 特化），没有调用 QPEP 多项式求解器；因此主表隔离的是**鲁棒外层**，不是 QPEP 代数。

### 1.4 种子与预算

* 数据种子：`SeedSequence([20261127, exp_id, type_id, round(100p), trial])`；p=0 时三种类型共用同一批干净实例。
* RANSAC 采样种子独立；固定/自适应 RANSAC 共享同一采样流，便于对比。
* 主表 100 trials/cell；scaling 50 trials/cell；梯度 20 个实例（10 boundary 40% + 10 random 20%）；e2e 10 seeds × 2 lr × 2 mode × 2 method。

---

## 2. 命令（Slurm）

```bash
cd labs/03-robust-qpeplayer/experiments/synthetic
PY=/home/syk/miniconda3/envs/WM/bin/python

# 0) 已知真值 / 闭式 / FD 校验（~12 s）
srun --partition=cpu --time=00:10:00 --ntasks=1 $PY verify_smoke.py

# 1) 扩展主表（~2m43；含 random/coherent/boundary、budget、scaling）
srun --partition=cpu --time=00:30:00 --ntasks=1 --cpus-per-task=1 \
  $PY run_main_extended.py 2>&1 | tee results/logs/main_extended_$(date +%Y%m%d_%H%M%S).log

# 2) 6D 梯度表（~14 s）
srun --partition=cpu --time=00:30:00 --ntasks=1 --cpus-per-task=1 \
  $PY run_gradient.py 2>&1 | tee results/logs/gradient_$(date +%Y%m%d_%H%M%S).log

# 3) End-to-end 边界坐标精化（~42 s）
srun --partition=cpu --time=00:30:00 --ntasks=1 --cpus-per-task=1 \
  $PY run_e2e.py 2>&1 | tee results/logs/e2e_$(date +%Y%m%d_%H%M%S).log

# 4) v0 主表的 10-trial 复现（写入 results/verify_repro/）
srun --partition=cpu --time=00:10:00 --ntasks=1 $PY -c "
import sys; from pathlib import Path; sys.path.insert(0,'.')
import run_registration as rr
rr.N_TRIALS=10; rr.OUT=Path('results/verify_repro'); rr.OUT.mkdir(parents=True,exist_ok=True); rr.run()"
```

> 所有脚本固定种子、无交互；主表/图/JSON/日志均由脚本生成。`run_registration.py` 保留为 v0，不改动。

---

## 3. 主表关键结果

完整表见 [`main_table_extended.md`](synthetic/results/main_table_extended.md)，图见
[`main_table_extended.png`](synthetic/results/main_table_extended.png)。

### 3.1 随机外点：同精度 + 约 30× 加速（对固定预算 RANSAC）

| p | Kabsch | RANSAC-1000 | RANSAC-adaptive | IRLS-GM | GNC-GM |
|---|---|---|---|---|---|
| 0% | 100% | 100% | 100% | 100% | 100% |
| 10% | 52% | 100% | 100% | 100% | 100% |
| 30% | 0% | 100% | 100% | 100% | 100% |
| 50% | 0% | 100% | 100% | 100% | 100% |
| 70% | 0% | 100% | 100% | 98% | 100% |

中位旋转误差（deg，节选）：p=30% 时 RANSAC-1000 0.13 / GNC 0.14；p=70% 时 RANSAC-1000 0.21 / GNC 0.21。**在两位小数内持平**。

runtime p50（ms）：

| p | RANSAC-1000 | RANSAC-adaptive | GNC | GNC vs 1000 | GNC vs adaptive |
|---|---|---|---|---|---|
| 10% | 51.47 | 0.41 | 1.59 | **32.3×** | 0.3×（GNC 慢） |
| 30% | 51.85 | 1.03 | 1.63 | **31.9×** | 0.6× |
| 50% | 50.92 | 2.90 | 1.61 | **31.6×** | 1.8× |
| 70% | 47.69 | 12.36 | 1.50 | **31.9×** | **8.3×** |

> 结论要分层说：对 v0 的固定 1000 次迭代 RANSAC，GNC 在 p≥10% 快 **31.6–32.5×**；对自适应 RANSAC，低外点率下 GNC 没有速度优势（p=10% 慢 3×），p=40% 持平，p≥60% 领先，p=70% 约 **8.3×**，p≥80% 约 **25–30×**（见预算表）。RANSAC 无论如何不可微；GNC 的 backward 是 6×6 线性系统。

### 3.2 相干外点：多数外点时所有方法都崩（如实写）

| p | RANSAC-1000 | RANSAC-adaptive | IRLS-GM | GNC-GM | 说明 |
|---|---|---|---|---|---|
| ≤40% | 100% | 100% | 100% | 100% | 内点多数，恢复 GT |
| 50% | 53% | 53% | 47% | 49% | 两个假设并列，近似掷硬币 |
| 60% | 0% | 0% | 0% | 0% | 100% 恢复**外点**位姿（误差 43.0°） |
| 70% | 0% | 0% | 0% | 0% | 100% 恢复**外点**位姿（误差 45.3°） |

> 上表是 100-trial 主表值；更细的 200-trial 扫描（`ablations/breakdown/`）给出 p=50% GNC 51.5%/RANSAC 50.0%、p=55% GNC 2.5%/RANSAC 0%，与上表在二项噪声内一致；论文 §5.2 用后者。

`frac_recovered_outlier` 在 60%/70% 对所有鲁棒方法均为 **100%**：它们准确恢复了竞争刚体变换。这验证 Prop 3.7：当相干外点多于内点时，鲁棒代价的**全局最小**本身就是错的；不是 GNC 局部极小失败。50% 时 GNC/RANSAC 的差异在二项噪声内。

### 3.3 边界外点：前向不失败，梯度才是考点

所有方法在 0–70% 边界外点下成功率均 **100%**；中位旋转误差：

| p | Kabsch | RANSAC-1000 | RANSAC-adaptive | IRLS-GM | GNC-GM |
|---|---|---|---|---|---|
| 40% | 0.28 | 0.26 | 0.24 | 0.15 | 0.15 |
| 70% | 0.33 | 0.62 | 0.60 | 0.27 | 0.27 |

> 边界外点是**零均值**的中等残差，不是错误对应；非鲁棒 Kabsch 也不一定崩（p=70% 仍 100% 成功、0.33°）。因此**不能用前向成功率来证明边界外点上的鲁棒性**；它的价值在梯度表（§4）：naive 在 $e>c^2/3$ 符号相反。GNC/IRLS 在边界档的精度优于 Kabsch/RANSAC（0.27° vs 0.33–0.62°），但这是精度差异，不是成功率差异。

### 3.4 RANSAC 预算与 N scaling

RANSAC 预算（随机外点，3 点采样；`ransac_budget.png`）：

| p | 50 iters | 1000 iters | adaptive | 理论（无放回） |
|---|---|---|---|---|
| 50% | 100% | 100% | 100% | 99.8% |
| 70% | 74% | 100% | 100% | 72.0% |
| 80% | 26% | 100% | 100% | 29.8% |
| 90% | 5% | 46% | 46% | 3.6% |

* 50% 外点 + 50 iters **不是**失败档：干净 3 点样本概率 $\binom{50}{3}/\binom{100}{3}\approx0.121$，50 次成功概率 ~99.9%。
* 失败档在 p≥70%：p=70%/50 iters 成功率 74%，p=80% 26%，p=90% 5%；p=90% 即使 1000 iters 也只有 46%（理论 52.4%，无放回）。
* 自适应 RANSAC 在 p≥80% 基本退化为 1000 iters（mean 864），runtime ~43–52 ms，与固定预算相当；在 p=50–70% 则只要 52–253 iters（2.6–13.0 ms）。

N scaling（p=30% 随机；`scaling.png`）：GNC 1.32→1.99 ms（N=20→200），RANSAC-1000 49.7→55.4 ms，RANSAC-adaptive 1.01→1.10 ms，IRLS 1.57→2.21 ms。GNC/IRLS 随 N 近似线性；RANSAC-1000 的时间由迭代数主导。

---

## 4. 6D 梯度表

完整表见 [`gradient_table.md`](synthetic/results/gradient_table.md)，图见
[`gradient_table.png`](synthetic/results/gradient_table.png)、
[`gradient_sign.png`](synthetic/results/gradient_sign.png)。

设置：N=30，10 个 boundary 40% + 10 个 random 20% 实例，共 600 个观测；每个实例 GNC + Newton polish 到 $\|\Phi\|\sim10^{-13}$；FD 为 polished fixed-point branch 的中心差分（h=1e-5）。

| bin | method | n | relerr p50 | relerr p90 | cos p50 | cos min |
|---|---|---|---|---|---|---|
| inlier (<c²/3) | exact | 414 | **4.9e-8** | 5.6e-8 | **1.000000** | 1.000000 |
| inlier | naive | 414 | 2.12e-1 | 4.30e-1 | 0.985 | 0.791 |
| inlier | no-λ | 414 | 8.96e-4 | 2.16e-3 | 1.000000 | 0.99999 |
| boundary (c²/3..3c²) | exact | 122 | **3.1e-8** | 5.3e-8 | **1.000000** | 1.000000 |
| boundary | naive | 122 | **1.07** | 1.19 | **0.234** | **-0.358** |
| boundary | no-λ | 122 | 9.80e-4 | 2.39e-3 | 1.000000 | 0.99999 |
| gross (>3c²) | exact | 64 | 1.25e-4* | 5.98e-4 | 1.000000 | 0.9986 |
| gross | naive | 64 | 1.16 | 1.20 | **-0.118** | **-0.397** |
| gross | no-λ | 64 | 9.44e-4 | 3.26e-3 | 1.000000 | 0.9986 |

\* gross 的相对误差受 FD 相减抵消限制（$\|dz^*/dy_i\|\sim10^{-8}$），绝对误差仍很小；cosine 仍为 1。

**符号翻转**（e>c²/3 的观测）：

| bin | n | 映射层径向 $\hat r^\top(\partial\Phi_t/\partial y_i)\hat r$ | 位姿平移层 $\hat r^\top(\partial t^*/\partial y_i)\hat r$ |
|---|---|---|---|
| inlier | 414 | 0% | 0% |
| boundary | 122 | **100%** | **100%** |
| gross | 64 | **100%** | **100%** |

映射层是定理：$a_{\rm exact}=-\lambda_\parallel(e)$，$a_{\rm naive}=-2\rho'(e)$，在 $e>c^2/3$ 严格异号（数值上 $\max|a_{\rm exact}+\lambda_\parallel|=8.9\times10^{-16}$）。位姿层不保证（Theorem 3.3(d)），但在这个良态 6D 样本上 100% 翻转；代表实例中 $e/c^2=1.06$ 时 $b_{\rm exact}=-0.0149$、$b_{\rm naive}=+0.0111$，naive 的 cosine 降到 0.36。

**Hessian / λ 校验**（代表实例）：

* $\|H_{\rm rob}-H_{\rm fd}\|_F/\|H_{\rm fd}\|_F=3.98\times10^{-7}$（h=1e-5；Richardson 6.8e-9）
* $\|H_{\rm doc}-H_{\rm fd}\|/\|H_{\rm fd}\|=1.08\times10^{-3}$（缺 $-\lambda$；本例 $\lambda_{\rm rob}=-0.096$，数据项主导，所以影响小；把 `verify/` 的 N=6 实例用修正后的 `Rhom` 重跑（`verify_smoke.txt` [6b]），$\lambda_{\rm rob}=1.033$，$H_{\rm doc}$ vs FD 为 **25.9%**，$H_{\rm rob}$ vs FD 为 1.5e-6）
* $\|H_{\rm naive}-H_{\rm fd}\|/\|H_{\rm fd}\|=0.280$（冻结权重）
* $\max|\lambda_i-4u_i^\top W_ir_i|=2.8\times10^{-17}$；$\max|\lambda_i-4e_i|=0.189$
* 1D 对照（已有 `verify/ift_1d_signflip.py`）：exact $-0.027815$，naive $+0.010292$，FD $-0.027815$；相对误差 $1.2\times10^{-8}$，符号相反。

---

## 5. End-to-end 边界坐标精化

完整表见 [`e2e_table.md`](synthetic/results/e2e_table.md)，曲线见
[`e2e_curves.png`](synthetic/results/e2e_curves.png)。

**协议**：N=50，boundary 50%/70%，10 个固定种子；GNC + Newton 得到初始位姿；只更新注入的 boundary 观测（`boundary-only`）或全部观测（`all-points` 对照），20 步 Adam，损失 $L=\|R-R_{gt}\|_F^2+\|t-t_{gt}\|^2$；每步重新解 fixed point。**这是 GT 监督的 test-time 诊断，不是学习层。**

| p | mode | lr | exact final loss | naive final loss | exact rot | naive rot | 配对 exact 胜 |
|---|---|---|---|---|---|---|---|
| 0.5 | boundary-only | 1e-3 | **1.90e-6** | 1.66e-5 | **0.029°** | 0.087° | **10/10** |
| 0.5 | boundary-only | 5e-3 | **6.36e-7** | 1.56e-6 | **0.022°** | 0.042° | 8/10 |
| 0.7 | boundary-only | 1e-3 | **2.14e-6** | 2.26e-5 | **0.030°** | 0.117° | **9/10** |
| 0.7 | boundary-only | 5e-3 | **1.70e-6** | 3.66e-6 | **0.032°** | 0.046° | **10/10** |
| 0.5 | all-points | 1e-3 | 1.20e-6 | 1.87e-6 | 0.016° | 0.019° | 6/10 |
| 0.7 | all-points | 5e-3 | 4.57e-6 | 4.72e-6 | 0.055° | 0.041° | 6/10 |

* boundary-only + lr=1e-3：exact 的最终 loss 是 naive 的 **0.059–0.071**（低 14–17×），旋转误差低 **3–4×**，配对 10/10 或 9/10。
* all-points 对照：4–9/10，loss ratio 0.64–1.04，**不可区分**；因为内点可以补偿 naive 在边界点上的错误方向。这正说明要把诊断限制在边界集。
* End-to-end 梯度校验（一个边界观测的 3 个坐标）：exact 相对误差 1.6e-7 / 3.9e-8 / 9.3e-8；naive 1.42 / 0.60 / 0.38。

---

## 6. 数值校验记录

完整输出见 [`verify_smoke.txt`](synthetic/results/verify_smoke.txt)。

| 检查 | 结果 | 判据 |
|---|---|---|
| Kabsch 无噪声 | $\|R-R_{gt}\|_F=2.6\times10^{-15}$，平移 $2.4\times10^{-16}$ | 通过 |
| GNC 0% vs Kabsch | 位姿差 p50 0.023°（max 0.046°），$\max(F_{\rm GNC}-F_{\rm Kabsch})=-3.4\times10^{-6}\le0$ | 通过 |
| RANSAC 全内点 vs Kabsch | $\|R-R_{\rm Kabsch}\|_F=0$，$|t-t_{\rm Kabsch}|=0$ | 通过 |
| GNC μ 调度 | p=0%：μ0 1.13–1.88，n_μ 2–3；p=30/70%：μ0 1.0–1.8e4，n_μ 29–31；$\|\Phi\|$ p50 ~1e-7 | 通过 |
| 相干多数（70%） | RANSAC 到 GT 43.2°，到外点变换 0.16°；20/20 恢复外点 | 通过（预期失败） |
| $H_{\rm rob}$ vs FD | 4.75e-7（h=1e-5），Richardson 6.8e-9；$\lambda_{\min}=24.58>0$ | 通过 |
| 原 verify 实例（修正版 [6b]） | $H_{\rm rob}$ vs FD 1.5e-6；$H_{\rm doc}$ vs FD 25.9%；$\lambda_{\rm rob}=1.033$；$\max|\lambda_i-4e_i|=51.05$ | 通过 |
| λ 恒等式 | $\max|\lambda_i-4u_i^\top W_ir_i|=1.1\times10^{-16}$；$\max|\lambda_i-4e_i|=0.386$ | 通过 |
| 6D Jacobian vs FD | 内点/边界 exact relerr 4.9e-8/4.1e-8，cos 1.000000；naive 0.21/1.06，cos 0.98/0.31 | 通过 |
| 自适应 RANSAC vs 理论 | p=0.5/0.7/0.8 成功率 100/100/98%，mean iters 52/253/864 | 通过（50 trials，二项噪声内） |
| v0 10-trial 复现 | 10-trial 与 100-trial 的成功率/中位误差在噪声内一致；runtime 因 trial 数少略高（如 RANSAC p=10% 69 vs 50 ms） | 通过 |

---

## 7. 发现并修正的代码/口径问题

### 7.1 `verify/riemann_hessian_check.py` 的 `build_A` 用了非齐次旋转公式（重要）

现有脚本用
`A[k,i,j] = (Rmat(E[i]+E[j]) @ p)[k] - ...` 做极化，但 `Rmat` 是**单位四元数**公式（含常数项 1），不是 2 次齐次式。因此由 `A` 重构的 `u(q)=[q^T A_k q]` **不等于** `R(q)p`（在单位 q 上也不等；随机测试差 0.09–0.33）。该脚本的 FD 与解析式共用同一个 `u`，所以 H 公式的**自洽校验**（1.46e-7）仍然成立，但它验证的是一个二次型代理，不是 3D-3D 旋转残差；输出里的 $\lambda_i$ 数值（如 52.4）也不是实际 3D-3D 的 λ。

修正：用齐次式
`Rhom(q)`（`w²+x²-y²-z²` 等）做极化，见 [`common.py`](synthetic/common.py) 的 `Rhom/build_A`。修正后本目录的校验给出 $H_{\rm rob}$ vs FD 4.75e-7、$\lambda_i-4u_i^\top W_ir_i=1.1e-16$、$\lambda_i-4e_i=0.386$。用 `verify/` 的同一 N=6 实例（c=1）重跑修正版（`verify_smoke.txt` [6b]）：$H_{\rm rob}$ vs FD **1.5e-6**、$H_{\rm doc}$ vs FD **25.9%**、$\lambda_{\rm rob}=1.033$、$\max|\lambda_i-4e_i|=51.05$、$\lambda_i-4u_i^\top W_ir_i=7.1e-15$。因此**定性结论不变**（缺 $-\lambda$ 约 26% 误差，修正后 1e-6），但原输出里的 29.7% / 1.46e-7 / 52.4 / 1.056 这些具体数字应替换。**本轮已修**：把 `verify/riemann_hessian_check.py` 的 `Rmat` 改成齐次式（`w²+x²-y²-z²` 等）后重跑；`verify/riemann_hessian_check_output.txt` 现在给出 $H_{\rm doc}$ vs FD **25.87%**、$H_{\rm corr}$ vs FD **$9.88\times10^{-8}$**、$\lambda_{\rm rob}=1.0329$、$\max|\lambda_i-4e_i|=51.05$。theory/paper/README 中的 29.7% / 1.46e-7 / 52.4 / 1.056 已同步替换为修正值。

### 7.2 RANSAC 基线的公平性：固定 1000 vs 自适应

v0 的“~30×”是对**固定 1000 次迭代**的 RANSAC 说的；标准自适应 RANSAC 在低外点率下只需 5–50 次迭代，p=10% 时 0.41 ms，比 GNC 1.59 ms 还快。因此主表同时报告两种 RANSAC，并把“30×”限定为“对固定预算基线”；自适应基线下 GNC 的优势在 p≥60% 才明显（p=70% 8.3×，p≥80% 25–30×）。这是本轮最重要的口径修正，避免 strawman。

### 7.3 RANSAC 成功率的理论公式应用无放回超几何

3 点采样是无放回；`1-(1-(1-p)^3)^k` 在 p=90% 给出 63.2%，无放回精确值
$p_{\rm clean}=\binom{n_{in}}{3}/\binom{N}{3}=0.000742$ 给出 52.4%，与实测 46%（100 trials）在二项噪声内。预算表已用精确公式；`verify_smoke` 的 [9] 仍用近似（p≤0.8 差异可忽略），已在表中注明。

### 7.4 arccos 近零数值地板

`rot_err_deg` 在误差 ~1e-8 rad 时受 `arccos(1-ε)` 精度限制，地板 ~1e-6°；Kabsch 无噪声的 1.7e-6° 不是算法误差。校验 [1] 改用 $\|R-R_{gt}\|_F$（2.6e-15）作为严格判据；主表 0.1–100° 的比较不受影响。

### 7.5 GNC 末步精度

`solve_gnc` 末 5 步 μ=1 IRLS 后 $\|\Phi\|\sim10^{-7}$，不是机器精度；这足以做主表，但不足以做 IFT/FD。梯度/精化实验统一用 `newton_polish` 把 fixed point 推到 ~1e-13，再算 Jacobian。FD 是**polished branch** 的差分，不是有限 GNC unroll。

### 7.6 边界外点不能用前向成功率证明鲁棒性

边界外点是零均值中等残差；Kabsch 在 p=70% 仍 100% 成功（0.33°），GNC 0.27°。前向成功率不是区分器；边界档的价值在梯度表。这一条要写进论文的“指标选择”而不是“鲁棒性胜利”。

---

## 8. 限制

1. **合成协议**：外点是人工注入（random/coherent/boundary），不是真实前端误差；真实数据见 §12（8 个官方 3DMatch test 场景、2 条 KITTI raw drive、KITTI odometry 00–10（每序列 33 协议帧）、2 白天 + 2 夜间前缀 In-The-Dark、KITTI stereo 回退）。3DMatch 的 8 个场景就是官方 geometric-registration test benchmark；54 个训练场景（62 场景语料）未用。In-The-Dark 夜间只测了 7 对前缀，整夜/眩光/端到端训练未做。
2. **前向是加权 Kabsch**，不是 QPEP 多项式求解器；因此不能验证 QPEP 的漏根/数值完备性，也不能声称“QPEP 全局内层”。标量权重下它与 4×4 Horn 等价，范围见 `qpep_gpu/qpep_scope.md`。
3. **$W_i=I$**：未测试各向异性权重、尺度可辨识性、学习 $W_i$。
4. **RANSAC 主基线是 3 点 + 重拟合**：主表报告固定/自适应两种预算；LO-RANSAC/自适应/固定三种预算的对比见 §14 与 `ablations/ransac_lo/`（LO 精度持平但比自适应慢 1.03–1.52×）。
5. **runtime 是共享节点 wall-clock**：p95 有系统噪声；结论“30×”对固定预算在多次运行中稳定在 30–33×，但绝对 ms 会随负载漂移。
6. **coherent 只有一种构造**（30–60° / 0.6–1.5 m）；p=50% 的“掷硬币”依赖几何，未做相图。
7. **e2e 是 GT 监督的 test-time 诊断**：不是学习层，不能证明“训练一个前端会更好”；boundary 集由注入协议已知，真实系统需要先识别边界点。
8. **未做**：6D 位姿层的符号翻转定理（只有数值 100%）、GNC μ-同伦分岔反例、$H_{\rm rob}$ 近奇异时的梯度行为。批量计时已完成（§13）；GPU 计时只跑到 CPU 校验：2026-11-28 提交的作业 3184 在 `gpu` 分区 PENDING（`Reason=Resources`，`StartTime=2026-11-28T21:23:07`，约 21 h 后），等待 5 min 后取消（`JobState=CANCELLED`），三张 RTX 5090 被 3151/3180/3181 占用；逐字状态见 `qpep_gpu/results/logs/slurm_gpu_3184_state.txt`，汇总见 `qpep_gpu/results/qpep_gpu.md`。
9. **v0 与扩展表的随机流不同**：v0 `main_table.*` 用单个顺序 RNG（RANSAC 采样会推进数据 RNG），扩展表用 per-cell `SeedSequence`（RANSAC 采样不扰动数据 RNG）。两者都是 100 trials 的独立样本；Kabsch 在 10% 外点的成功率 42%（v0）vs 52%（扩展）属于二项抽样差异（n=100, sd≈5%），不影响鲁棒方法的结论。v0 的 10-trial 复现在 `synthetic/results/verify_repro/`。

---

## 9. 文件清单

```
experiments/
├── README.md                          # 本文件
├── protocol.md                        # 原始实验协议（设计）
└── synthetic/
    ├── common.py                      # 数据/方法/度量/精确导数（含 Rhom 修正）
    ├── run_registration.py            # v0 主表（保留，不改）
    ├── run_main_extended.py           # 扩展主表（random/coherent/boundary/budget/scaling）
    ├── run_gradient.py                # 6D 梯度表（exact/naive/no-λ/FD）
    ├── run_e2e.py                     # 边界坐标精化（exact vs naive）
    ├── verify_smoke.py                # 已知真值/闭式/FD 校验
    └── results/
        ├── main_table_extended.{json,md,png}
        ├── main_table_extended_raw.jsonl
        ├── ransac_budget.png
        ├── scaling.png
        ├── gradient_table.{json,md,png}
        ├── gradient_sign.png
        ├── e2e_table.{json,md}
        ├── e2e_curves.png
        ├── verify_smoke.txt
        ├── main_table.{json,md,png}   # v0 原表（100 trials）
        ├── verify_repro/              # v0 10-trial 复现
        └── logs/                      # Slurm 运行日志
```

---

## 10. 判断：主表是否站得住？

**站得住的部分**：
* “在 0–70% 随机外点下，GNC 与 RANSAC 同成功率、同精度（两位小数），对固定预算 RANSAC 快 31–33×，且可微”——有 100 trials × 8 档、p50/p95、闭式校验和 10-trial 复现支撑。
* “相干多数外点让所有方法返回外点位姿”——60/70% 两档、所有方法 100% 恢复外点，和 Prop 3.7 一致；这是诚实的失败边界。
* “精确 IFT 与 FD 一致到 5e-8，naive 在边界/gross 上 cosine 0.23/-0.12，e>c²/3 符号 100% 相反”——600 个观测、分 bin、1D+6D 双证据，梯度表是全文最硬的一张表。
* “end-to-end 边界精化 exact 优于 naive”——10 seeds × 2 lr，boundary-only 14–17× lower loss；all-points 对照说明为什么必须限制在边界集。

**站不住/需要降级的部分**：
* **不能**说“GNC 比 RANSAC 精度更高”：随机外点下两者持平；边界档 GNC 略优但 Kabsch 也不失败；相干档所有方法一起崩。
* **不能**无条件说“30× 加速”：只对固定 1000 次迭代 RANSAC 成立；对自适应 RANSAC，p=10% 时 GNC 慢 3×，p=70% 才 8×。
* **不能**说“边界外点导致前向失败”：前向 100% 成功；边界档是梯度毒害，不是前向鲁棒性胜利。
* **不能**说“e2e 训练赢了”：这是 GT 监督的 test-time 诊断，不是学习层；真实端到端（In-The-Dark）已用 FTP range 请求解阻，run_000002/12 各 15 对已测（GNC 0.078°/0.092°，100% @0.1 m），night17/18 只测 7 对前缀（0.440°/0.137°），前端是手写 SIFT+SGBM、无训练，所以不是整夜或 SDPR 级证据（见 `real/inthedark/README.md`）；KITTI stereo 回退旋转成立、平移 0.70 m / 0% @0.1 m，不能替代。

**总体**：主表可以作为“同精度 + 对固定预算大幅加速 + 可微 + 明确失败边界”的合成证据；不足以单独支撑“精度胜出”或“任意鲁棒”。论文里应把 30× 的基线、边界档的定位、相干档的失败写清楚，梯度表单独作为核心贡献证据。

---

## 11. 各向异性权重 $W_i$ 消融（2026-11-27，`ablations/anisotropic/`）

主表只用标量权重。真实前端给出的可能是逐观测 SPD 权重 $W_i$；此时正确的 GM 目标
$\Phi=\sum_i\rho(r_i^\top W_i r_i)$ 的内层 $\min_{R,t}\sum_i u_i r_i^\top W_i r_i$
是 6-DoF **非线性**最小二乘，标量加权 Kabsch 只在 $W_i\propto I$ 时精确。

`ablations/anisotropic/run_anisotropic.py` 用 Cholesky 白化残差 + 6-DoF Gauss-Newton
（每步重锚定旋转图卡）解内层，GNC-GM/IRLS 调度与 `common.solve_gnc` 一致，并用
`scipy.optimize.least_squares` 独立校验。结论：

* 正确性：$W_i=I$ 时与闭式 Kabsch 差 $5\times10^{-13}$（Frobenius），与
  `common.solve_gnc` 差 $5\times10^{-12}$；scipy 相对代价差 $<10^{-9}$。
* 标量 Kabsch 近似对 $W_i=I$ 的目标比 1.000000，对 diag/full 分别为
  1.002–1.007 / 1.035–1.088（单 trial 最大 3.8）——**标量内层不是各向异性目标的最优解**。
* 本协议噪声各向同性，$W_i$ 是错配模型：full-W 的 GNC-W 旋转误差
  0.19–0.35° vs $W=I$ 0.11–0.19°，配对改善比例仅 8–32%，成功率 92–100%；
  **各向异性权重没有提升精度**，但正确解在所有格子取得更低 $\Phi$。
  p=0.7 的 2/25 次 full-W 失败是 159°/175° 的粗大局部极小（同实例 $W=I$
  成功，且其 $\Phi=0.245$ 劣于 identity 的 0.182）。
* 精确内层比标量近似慢 4.9–23×（p≥0.3 时约 20–48 ms vs 1.9–2.1 ms；计时固定 BLAS 单线程，
  p50 仍有 run-to-run 波动，比较时看同 run 内的比值）。

细节：`ablations/anisotropic/README.md`、`results/anisotropic.md`、
`results/anisotropic.json`。

---

## 12. 真实数据扩展（2026-11-27）

| 数据 | 目录 | 关键数字 | 限制 |
|---|---|---|---|
| ModelNet10（第二物体数据集） | `real/modelnet10/` | GNC 0–70% 全 100%；鲁棒方法相对 ModelNet40 成功偏差 ≤2 pp、中位旋转偏差 ≤0.038°；31.7–32.5× vs RANSAC-1000 | OFF 网格自采样 2048 点；仅 test split |
| 3DMatch（室内片段，4 场景子集） | `real/threedmatch/` | 随机 0–70% GNC/RANSAC 100%；GNC 2.2–2.5 ms vs RANSAC 61–71 ms（27–30×）；相干 60–70% 全 0%；边界全 100%，70% GNC 0.25° vs RANSAC 0.74° | GT-NN oracle 基础匹配；2 主 + 2 unseen 场景；仅高重叠帧对 |
| 3DMatch FPFH 前端 | `real/threedmatch/run_descriptor.py` | 自然内点 80.0%/67.8%；GNC 100%/100%，RANSAC-1000 100%/97%，Kabsch 100%/83% | 手工 FPFH，非 FCGF/Predator |
| KITTI raw 0001 + 0002（室外 LiDAR） | `real/kitti/`, `real/kitti2/` | 两 drive 随机 0–70% GNC/RANSAC 100%；GNC 2.7–2.8 ms vs 61–64 ms（~22×）；相干 60–70% 全 0%；边界全 100%，70% GNC 0.07–0.09° vs RANSAC 0.14–0.16° | OXTS 位姿；**两条 raw drive**；无真实描述子 |
| KITTI raw 立体前端（回退） | `real/kitti_stereo/` | 自然内点 17.8%；GNC 旋转 0.052°、平移 0.70 m（floor 0.239 m）；所有方法 0% @0.1/0.25 m；GNC 19.3× vs RANSAC-1000 | **不是 In-The-Dark、不是 SDPR 前端**；见 §16 |
| UTIAS In-The-Dark（真实） | `real/inthedark/` | day run_000002/12：15 对，内点 79.4%/88.5%，GNC 0.078°/0.020 m、0.092°/0.010 m；night 17/18：10 对窗口，内点 96.0%/97.1%，GNC 0.135°/0.011 m、0.150°/0.011 m；稀疏 0.197°；均 100% @0.1 m | 夜间只是 10 对窗口；手写 SIFT+SGBM，不是 SDPR VGG/UNet；无训练 |
| 3DMatch 8 场景 / overlap bins / true low overlap | `real/threedmatch_full/` + `real/threedmatch_low_overlap/` | 8/8 方向；8 场景 pooled random 0–70% GNC/RANSAC 100%，70% 27.7×；low/mid/high bins 均 100%@p=.7；真 0.20–0.50 几何重叠 band（32 low+14 high pairs）GNC/RANSAC 100%@p=.7，IRLS 92%，Kabsch 1% | GT-NN oracle 基础匹配；真 low overlap 仍 base inlier 0.88–0.99，不代表真实描述子低重叠 |
| KITTI odometry 00–10 | `real/kitti_odom/` + `real/kitti_more/` | 11/11 序列有覆盖（每序列 33 帧）；新 8 序列 pooled 99/99/100/100% at p=0/.3/.5/.7，22.4×；seq08 p=0 80%（真实 bad basin）；00/05/07 相干 50% GNC 53%/RANSAC 0%，边界 70% GNC 0.067° | 不是完整 00–10 LiDAR 流；只是协议触到的 33 帧/序列 |

3DMatch 的 `gt.log` 实测方向是 **id2→id1**，runner 取逆；换发布版必须重跑 `verify_gt_direction.py`。KITTI 的阈值 `c=0.25 m` 来自 ~0.10 m 点距（见 `real/kitti/results/c_sensitivity.md`），不能直接套 3DMatch 的 0.05 m。

## 13. 内层求解器与系统

- `inner/`：Horn/Davenport 4×4 与加权 Kabsch 数值等价（max 旋转差 1.07e-14）；p50 仅 N=500 有 1.27× 收益，N=100 中性、N=20 慢 7%。**不是** 40 根 QPEP；标量权重 3D-3D 的 QPEP 退化为该 4×4 特征问题。
- `systems/`：numpy 批量 GNC 在 B=32,N=100 时 56.2→8.5 ms（6.60×），B=1 反而慢；GPU float64 eager 在 B≤128 被 ~40 ms launch 地板限制，只在 B=128 有 5–10×。生产建议 B≥8 用批量，B=1 用循环。
- `qpep_gpu/`：eager-dynamic / static-shape / CUDA-graph 的 4×4 eigen GNC 对比（B={1,8,32,128,512}×N={100,500}，float64/float32）；结果与**执行状态**见 `qpep_gpu/results/qpep_gpu.{json,md,png}`，数学范围与非声明见 `qpep_gpu/qpep_scope.md`。该文件不覆盖 40 根 QPEP。
- 2026-11-27 修复：`gpu_solver.eigh4_jacobi` 的 Givens 旋转符号错误（原实现 static-jacobi 与 numpy 参考 maxdR≈1.4、n_mu 仅 50% 匹配）；修正后 CPU 校验 maxdR≈1e-16、n_mu 100% 匹配。

## 14. 消融与调度

- `ablations/kernels/`：GM/Cauchy/Welsch 在随机/边界 0–70% 全 100%；GM 在边界 50–70% 中位旋转最低（0.199/0.229/0.340°）。默认 GM 有依据。
- `ablations/gnc_schedule/`：gamma/mu0/polish 扫描 24 组全部 100%，中位旋转相同；gamma=2.0 快 1.04–1.79×，gamma=1.2 慢 1.47–2.49×。默认保守。
- `ablations/anisotropic/`：correct full-W 内层在 W=I 与 Kabsch 差 5.3e-13；但各向同性噪声下 W 错配，full-W 精度反而下降（配对 Δrot +0.055..+0.124°），且慢 4.9–23×。保留为正确性/负结果。

## 15. 仍未关闭的缺口

1. 外部基线：Theseus-GM 与 SDPRLayers+SCS 已在**合成 3D-3D** 上实测（`results/baselines.*`）；SDPR 只跑前向（无 Mosek），Theseus 用 dense Cholesky；BPnP/Pyatov/POLI 任务不同或代码/权重不可得，已标 out-of-scope，未做假表。
2. UTIAS In-The-Dark：已用 range 提取的 teach temporal + repeat spatial + 相机系共轭得到真实 GT，run_000002/12 各 15 对跑通（GNC 0.078°/0.020 m，100% @0.1 m）；night17/18 各 10 对窗口（0.135°/0.150°，100% @0.1 m；稀疏 0.197°）；手写前端、无训练，整夜/眩光/端到端训练仍未做。
3. 学习描述子（FCGF/Predator）未接；3DMatch 真实前端只有手工 FPFH。
4. 3DMatch 8 场景 + 真 low overlap 0.20–0.50 已完成；KITTI odom **00–10 全部 11 序列**已跑（每序列只 33 个协议帧，不是完整流）；54 个训练场景（62 场景语料）/ 真实描述子低重叠 / 更多 raw drive 仍未做。
5. 完整 40 根 QPEP 多项式求解器、根完备性/失败率未实现；GPU 批处理只覆盖 4×4 eigen 内层（`qpep_gpu/`），GPU 节点被占满，只有 CPU 校验；作业 3184 的 PENDING→CANCELLED 逐字状态见 `qpep_gpu/results/logs/slurm_gpu_3184_state.txt`，范围见 `qpep_gpu/qpep_scope.md`。

## 16. 真实立体前端回退（2026-11-27，`real/kitti_stereo/`）

UTIAS In-The-Dark 已解阻：FTP 支持 range 请求，只需取 teach run 的 `transforms_temporal.txt`（315 KB）与 repeat run 的 `transforms_spatial.txt`，按 repeat spatial × teach temporal × 相机系共轭得到逐对 GT；重投影误差 0.52–0.65 px（逆 18–28 px、恒等 9.5–14.7 px、不共轭 21–31 px）。run_000002/12 各 15 对、手写 SIFT+SGBM 前端：自然内点 79.4%/88.5%，GNC 0.078°/0.020 m、0.092°/0.010 m，100% @0.1 m。两个白天 + 两个夜间**窗口**；夜间只 10 对/run，不是整夜证据；也不是 SDPR VGG/UNet/训练。KITTI raw stereo 仍是独立负结果（旋转 0.052°、平移 0.70 m、0% @0.1 m）：

- 自然内点率 17.8% @ c=0.5 m，中位 890 对/帧对。
- 旋转：GNC/IRLS 中位 **0.052°**（GT-inlier floor 0.071°）——鲁棒层能解旋转。
- 平移：GNC 中位 **0.70 m** vs 帧间运动 1.92 m，GT-inlier floor 0.239 m；**所有方法 0% @0.1 m，0% @0.25 m**。立体 vs LiDAR 深度比 1.000–1.061（3–60 m），排除尺度 bug；误差是前向平移系统偏差（0.63–0.73），瓶颈在前端/遮挡/动态，不在优化器。
- 运行时：GNC 4.21 ms vs RANSAC-1000 81.3 ms（19.3×）。
- 结论：这是“旋转可解、平移被前端限制”的负结果；不能替代 In-The-Dark 端到端训练。

## 17. Phase-3 轮次与一致性记录（2026-11-27）

- `results/phase3_rounds.json`：本轮 4–6 个实测轮次（假设 / 改动 / 命令 / before→after / keep-revert / artifact），与 `rounds_2026-11-27.md` 的 42 轮互不覆盖。
- `phase3_consistency.md`：本轮论文 / README / repro audit 的一致性修复清单与仍未关闭的矛盾。
- `qpep_gpu/qpep_scope.md`：标量权重 3D-3D 的 QPEP 退化为 4×4 Horn/Kabsch、40 根多项式求解器的作用与未实现范围、精确非声明。


## 18. 外部基线与 QPEP/GPU 范围（2026-11-27）

- `results/baselines.*`、`experiments/baselines/`：Theseus-GM 0.2.3（dense Cholesky）随机初始化 p=0/0.3/0.5/0.7 成功 100/80/55/25%，GNC 100%；SDPRLayers+SCS 前向 0/208 异常、8.5–13.4 ms，但 p≥0.3 全失败，且 W=I 时与 Kabsch 数值等价（max 1.5e-5° / 1.3e-9 m），不是独立鲁棒基线；BPnP/Pyatov/POLI out-of-scope。
- `qpep_gpu/`：4×4 eigen 内层 scope 文档 + CPU 校验网格；GPU 节点 3 张卡被 cyr 的作业占满，CUDA-graph 未捕获、无 GPU 数字；不是 40 根 QPEP 实现。
