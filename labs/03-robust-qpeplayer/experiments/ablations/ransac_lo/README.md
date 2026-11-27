# P3 消融：LO-RANSAC / MSAC vs 固定/自适应 RANSAC vs GNC-GM

> **一句话结论**：把基线换成更强的 LO-RANSAC（MSAC 打分 + 5 轮局部优化 + 自适应停止）之后，结论不变且更不利于"RANSAC 慢"的叙事——LO-RANSAC 在 p≤70% 随机外点上成功率仍为 100%，精度与 GNC-GM 打平（中位旋转误差差 ≤0.006°），但它比主表里已有的自适应 RANSAC **慢 1.03–1.52×**，且在高外点率下远慢于 GNC（p=70% 时 LO-RANSAC 是 GNC 的 **8.2×**）。固定 1000 次迭代的 RANSAC 确实是被 ~32× 加速的稻草人；公平基线是自适应 RANSAC，它只在低外点率（p≤30%）快于 GNC，p=40% 持平，p≥50% 起 GNC 更快。LO-RANSAC 没有改变这个分层，也没有让任何方法在相干多数外点下免于可辨识性失败。

本目录是 `labs/03-robust-qpeplayer/experiments/synthetic/` 主表的**补强基线消融**，不改动 `common.py`、`theory/`、`verify/`、主表结果或任何交付文档。

---

## 1. 为什么做这个消融

主表（`synthetic/results/main_table_extended.md`）对 GNC 的速度叙事是"比固定 1000 次迭代 RANSAC 快 ~31–33×"。审稿人最容易攻击的点是：固定预算 RANSAC 是稻草人，标准做法是自适应停止；更强的做法是 LO-RANSAC（Chum et al., 2003）加 MSAC/截断二次打分。本消融把这两项都加上，在**完全相同的数据、种子、阈值和最终重拟合**下重跑，检验"成功/精度/运行时差距"是否被填平。

## 2. 方法（精确定义）

所有方法共用：N=100，σ=0.01，距离阈值 c=0.05，成功率判据 rot<5° 且 trans<0.1 m，最终都在最优模型的内点集上做一次无权 Kabsch 重拟合。

| 方法 | 定义 |
|---|---|
| `RANSAC-1000` | 直接调用 `common.solve_ransac(..., iters=1000, adaptive=False)`：3 点采样、内点数打分、`||r||<c` 为内点、固定 1000 次（唯一提前退出是某假设覆盖全部 N 点）。 |
| `RANSAC-adaptive` | 直接调用 `common.solve_ransac(..., iters=1000, adaptive=True)`：同上，但用置信度 0.999 的停止界 `need = ceil(log(1-0.999)/log(1-w³))`，上限 1000。 |
| `LO-RANSAC` | 见下方伪代码：3 点假设，**MSAC 打分** `s = Σ min(||r_i||², c²)`；仅当假设改进当前最优时做局部优化（≤5 轮"内点重拟合 + 重新打分"，内点集不变或分数不再下降即停）；每次最优更新后用内点率 w 更新自适应停止界，上限 1000。 |
| `GNC-GM` | 参考方法，直接调用 `common.solve_gnc`（未改动）：GNC 升温 + 加权 Kabsch + 末 5 步 μ=1 IRLS。 |

LO-RANSAC 伪代码：

```text
K=1000, conf=0.999, R=5, c2=c*c
s_best=+inf, best=None, k=0, target=K
while k < target:
    idx ~ Uniform(3-subset)
    (R,t) = Kabsch(x[idx], y[idx])              # 最小 3 点解
    e2 = ||R x_i + t - y_i||^2
    s  = Σ_i min(e2_i, c2)                      # MSAC 分数
    if s < s_best:
        inl = (e2 < c2); s_cur = s
        for r in range(R):                      # 局部优化，≤5 轮
            if |inl| < 3: break
            (R',t') = Kabsch(x[inl], y[inl])    # 无权重拟合
            e2' = ||R' x_i + t' - y_i||^2
            s' = Σ_i min(e2'_i, c2); inl' = (e2' < c2)
            if s' >= s_cur: break               # 不再改进
            same = (inl' == inl)
            R,t,e2,inl,s_cur = R',t',e2',inl',s'
            if same: break                      # 内点集不动点
        if s_cur <= s and s_cur < s_best:
            s_best = s_cur; best = (R,t,inl)
            if |inl| == n: k += 1; break        # 全内点，提前退出
            w = |inl|/n
            if 0 < w^3 < 1:
                target = max(1, min(K, ceil(log(1-conf)/log(1-w^3))))
    k += 1
(R,t) = Kabsch(x[best.inl], y[best.inl])        # 与两个基线相同的最终重拟合
```

实现细节：`_lo_refine` 只接受严格降低 MSAC 分数的重拟合，因此不会把输入模型变差；当某轮重拟合得到相同内点集时下一轮必然复现同样结果，故立即停止。MSAC 只用于**假设选择/局部优化**，最终估计仍是内点集上的无权 Kabsch，和两个 RANSAC 基线完全一致——这样比较隔离的是"选假设 + 局部优化"，不是换最终估计器。

## 3. 种子与公平性

* 数据种子与主表完全一致：`SeedSequence([20261127, 10, type_id, round(100p), trial])`，其中 random→type_id=1、coherent→2、p=0 共用 type_id=0。因此本目录的 `RANSAC-1000` / `RANSAC-adaptive` / `GNC-GM` 是主表的**逐 trial 复现**（见 §5 校验）。
* 三个 RANSAC 变体各用独立 RNG，但种子相同（`..., 999`），即共享同一串 3 点采样——按假设配对的比较。
* 没有在测试 cell 上调参：c、置信度、1000 上限、LO 5 轮都是运行前固定的默认值。random 100 trials/cell，coherent 50 trials/cell。
* 计时：单次调用 wall-clock，Slurm 节点上 BLAS/OpenMP 固定 1 线程；报告 p50/p95，并额外报告**同一 trial 内**的方法/GNC 比值中位数（对共享节点负载漂移更稳健）。
* 1000 次上限足够公平：LO-RANSAC 平均迭代数最高 253（random p=70%），单次最大 264；p=0 时全内点提前退出只需 1 次。若需要 >1000 次才能公平，应把预算写进结论——本消融中不需要。

## 4. 运行

```bash
cd labs/03-robust-qpeplayer/experiments/ablations/ransac_lo
PY=/home/syk/miniconda3/envs/WM/bin/python

# 完整运行（~76 s，Slurm）
srun --partition=cpu --time=00:30:00 --ntasks=1 --cpus-per-task=2 \
  $PY run_ransac_lo.py 2>&1 | tee results/logs/ransac_lo_$(date +%Y%m%d_%H%M%S).log

# 冒烟（2 trials/cell，写到临时目录，不覆盖正式结果）
$PY run_ransac_lo.py --trials-random 2 --trials-coherent 2 \
  --out-dir /tmp/ransac_lo_smoke --no-repro
```

脚本每次运行都会重新生成 `results/ransac_lo.{json,md,png}` 和 `results/ransac_lo_raw.jsonl`，并做一次与主表 raw 记录的逐 trial 复现校验；不会覆盖主表结果。

## 5. 关键结果（本次运行，不可手改）

完整表见 [`results/ransac_lo.md`](results/ransac_lo.md)，图见 [`results/ransac_lo.png`](results/ransac_lo.png)，逐 trial 数据在 `results/ransac_lo_raw.jsonl`。

### 5.1 随机外点：成功率与精度打平

| p | RANSAC-1000 | RANSAC-adaptive | LO-RANSAC | GNC-GM |
|---|---|---|---|---|
| 0% | 100% | 100% | 100% | 100% |
| 10% | 100% | 100% | 100% | 100% |
| 30% | 100% | 100% | 100% | 100% |
| 50% | 100% | 100% | 100% | 100% |
| 70% | 100% | 100% | 100% | 100% |

中位旋转误差（deg）：p=30% 时 RANSAC-1000 0.135 / adaptive 0.136 / LO 0.135 / GNC 0.140；p=70% 时 0.211 / 0.211 / 0.211 / 0.211。LO 与 GNC 的中位旋转误差在全 p 上最多差 **0.006°**（平移最多差 **1e-4 m**）。

### 5.2 随机外点：运行时（p50，ms；Slurm，1 线程 BLAS）

| p | RANSAC-1000 | RANSAC-adaptive | LO-RANSAC | GNC-GM | LO/GNC（配对中位） |
|---|---|---|---|---|---|
| 10% | 56.45 | 0.45 | 0.62 | 1.73 | 0.36× |
| 30% | 55.72 | 1.08 | 1.23 | 1.73 | 0.71× |
| 40% | 56.65 | 1.78 | 1.94 | 1.76 | 1.08× |
| 50% | 59.77 | 3.10 | 3.34 | 1.81 | 1.82× |
| 70% | 58.02 | 14.76 | 15.18 | 1.83 | 8.24× |

* GNC 相对固定 1000 次 RANSAC 的加速在本运行中为 **~32×**（p50 比值 31.7–33.0×），与主表一致。
* LO-RANSAC 比自适应 RANSAC **慢 1.03–1.52×**（p=0 时 1.52×，p≥40% 后差距收敛到 1.03–1.09×）：LO 的重拟合是纯开销，因为自适应 RANSAC 已经在内点集上做了最终重拟合。
* GNC 的耗时几乎不随 p 变化（1.7–1.8 ms）；RANSAC 的耗时由内点率决定。分界点在本数据上：p≤30% RANSAC 更快，p=40% 持平，p≥50% GNC 更快。

### 5.3 相干外点：没有任何方法被 LO 拯救

| p | RANSAC-1000 | RANSAC-adaptive | LO-RANSAC | GNC-GM |
|---|---|---|---|---|
| 40% | 100% | 100% | 100% | 100% |
| 50% | 56% | 56% | 48% | 52% |
| 60% | 0% | 0% | 0% | 0% |
| 70% | 0% | 0% | 0% | 0% |

* p=50% 时真值变换与竞争变换的期望共识数相同，胜负接近抛硬币；LO 的 MSAC tie-break 选到外点盆地的比例 52%，两个普通 RANSAC 为 44%，GNC 为 48%（50 trials 中 2–4 个 trial 的差别，二项标准误 ~7%），不应解读为系统性退化。
* p=60–70% 时所有方法 100% 返回竞争变换：这是 Prop 3.7 的可辨识性失败，局部优化无法修复。

### 5.4 复现校验

与 `synthetic/results/main_table_extended_raw.jsonl` 逐 trial 对比 **3600** 条记录：max |Δrot| = 0、max |Δtrans| = 0、success 不一致 0、迭代数不一致 0，**PASS**。即本目录的 RANSAC-1000 / adaptive / GNC 与主表逐位一致，差异只可能来自新加的 LO-RANSAC。

## 6. 结论与对论文的建议

1. **LO-RANSAC 在精度上没有击败 GNC**：两者在 p≤70% 随机外点上都是 100% 成功，中位误差差 ≤0.006°。原因是所有方法的最终估计都是同一内点集上的无权 Kabsch；LO 只改变选中的内点集，而在 N=100、c=0.05 下最优内点集基本已被共识数/ MSAC 选中。
2. **LO-RANSAC 在运行时上也没有击败 GNC**：它只在 p≤30% 比 GNC 快（0.4–0.7×），p=40% 持平，p≥50% 明显更慢（p=70% 8.2×）。而且它始终比更便宜的自适应 RANSAC 慢，因此"用 LO 把 RANSAC 做强"对本文的对比没有帮助。
3. **固定 1000 次 RANSAC 是稻草人，这点必须承认**：公平基线是自适应 RANSAC。按本数据，GNC 的运行时优势只在 p≥50%（随机外点）成立；低外点率下自适应 RANSAC 快 2–4×。论文/摘要若继续只写"比 RANSAC 快 30×"，应明确限定为"固定预算 RANSAC"，并同时给出自适应 RANSAC 的数字（主表 README 已有）。
4. **GNC 的独立卖点是可微性与高外点率下的平坦耗时**，不是"全面快于 RANSAC"。LO 消融支持这个更窄、更诚实的定位。
5. **相干多数外点**：LO 不改变"所有方法都返回竞争变换"的结论；不要宣称任意鲁棒。

## 7. 文件

| 文件 | 内容 |
|---|---|
| `run_ransac_lo.py` | 全部方法（两个基线直接 `import common`）、协议、聚合、复现校验、作图、Markdown 生成。 |
| `results/ransac_lo.json` | 配置 + 每 cell 聚合（success、rot/trans 中位、p50/p95、平均迭代、配对运行时比、LO 调用/轮数）+ 复现校验。 |
| `results/ransac_lo_raw.jsonl` | 4800 条逐 trial 记录（每 cell × trial × method），含 `n_lo_calls` / `n_lo_rounds` / `msac_score` 等诊断字段。 |
| `results/ransac_lo.md` | 自动生成的完整表与结论。 |
| `results/ransac_lo.png` | 成功率的 random/coherent、旋转误差、p50 运行时、相对 GNC 的运行时比。 |
| `results/logs/` | Slurm 运行日志。 |

## 8. 边界

* 只测了 N=100、σ=0.01、c=0.05、p≤0.7；更高 p、更大 N、不同 c 下 RANSAC/LO 的相对位置可能变化，本消融不外推。
* 只测了 3D-3D 的闭式 Kabsch 前向，不涉及 QPEP 多项式求解器或真实数据（3DMatch/KITTI/In-The-Dark）。
* 计时是 CPU 单线程、共享 Slurm 节点；绝对值有 ±几 % 的负载噪声，因此结论依据同一 trial 内的配对比值，而不是跨 cell 的绝对 p50。
* `common.py`、主表结果、`theory/`、`verify/`、`deliverables/` 均未修改。
