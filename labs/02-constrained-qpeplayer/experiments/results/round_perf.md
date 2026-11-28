# 第 1 轮（P2 性能 + B4 SCP baseline）

- 日期：2026-11-27（会话 20:50–21:10 Asia/Shanghai）
- worktree：`/mnt/storage1/syk/.superagent-work/wt-p2-perf`，branch `opt/p2-perf`
- 环境：`/home/syk/miniconda3/envs/sdpr/bin/python`（numpy 1.26 / scipy 1.13 / cvxpy 1.5.4）；
  计时单线程（`OMP/OPENBLAS/MKL/NUMEXPR_NUM_THREADS=1`）
- 未提交、未 push；未改动 `qpep_core.py` / `certificate.py` / `run_main.py` / theory / verify / deliverables

| 项 | 内容 |
|---|---|
| **假设** | H1：ALM 前向的 p 标度是「常数维求解 + O(p) 装配」，装配不是瓶颈、SDP 证书不主导，且证书可按 p 缓存。H2：dQP+SCP（协议 B4）是强凸层 baseline，可行、精度同档，但**单起点**认证率低于 ALM（尤其初值落进 cheirality 镜像盆地时）。 |
| **改动** | 仅新增两个实验脚本（只读 import `qpep_core` / `certificate` / `run_main`）：`experiments/code/run_perf.py`（ALM 墙钟拆解 + p 标度 + 证书缓存）、`experiments/code/run_scp.py`（信任域 SCP baseline，同一实例/热启动/指标/证书）。 |
| **命令** | `srun --partition=cpu --time=00:30:00 --ntasks=1 bash -c 'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1; /home/syk/miniconda3/envs/sdpr/bin/python run_perf.py 2>&1 \| tee ../results/run_perf.log'`（在 `experiments/code/` 下）<br>`srun ... '... /home/syk/miniconda3/envs/sdpr/bin/python run_scp.py 2>&1 \| tee ../results/run_scp.log'` |
| **指标 before → after** | **H1**：主表只有 ALM 总时长 260–288 ms、SDP 51–84 ms 的黑箱 → 拆解为 ALM 231/240/225/219 ms（p=8/16/32/64），其中内层 BFGS 占 97.7–98.8%、装配 0.6–1.3%、SDP base 42–60 ms（占 ALM 21.5–28.4%）；参数化缓存后 SDP 1.4–1.8 ms（冷解 26–33×）。交叉检查同 seed ALM 的 `J` 与 `raw_main.json` **max\|ΔJ\|=0.0**。**H2**：主表无 SCP 行 → 新增 SCP（p=8/16/32，n=50）：可行 100%，认证 98%/98%/98%（1 个镜像初值 trial 失败），runtime 130/135/142 ms vs ALM 218/239/232 ms；压力档（p=16, σ=0.02/bias=0.12, n=20）：SCP 认证 85%（3 个失败全为镜像初值）vs ALM 100%，SCP 147 ms vs ALM 226 ms。 |
| **结论** | **keep 两个脚本**（H1 得到支持；H2 部分被推翻：SCP 在普通档认证率只差 2pp 且更快，是必须报告的强 baseline；差距集中在镜像初值/高噪声档）。另：p=64 时 ALM 有 30% trial 触发 active-set 一致性 fallback（主表 raw 为 26%，一致），fallback 本身只 ~1.7 ms。 |
| **产物** | `experiments/results/perf_breakdown.json` / `perf_breakdown.md` / `run_perf.log`；`experiments/results/scp_baseline.json` / `scp_baseline.md` / `run_scp.log`；本文件。 |

## 1. ALM 墙钟分解（`run_perf.py`，每 p 20 trials，median [IQR] 见 md/JSON）

| p | ALM 总 ms | 装配 ms（占 ALM） | 内层 ms（占 ALM） | other ms | SDP base ms | SDP base+RLT ms | pipeline ms | SDP/ALM |
|---|---|---|---|---|---|---|---|---|
| 8 | 231.4 | 1.47（0.6%） | 228.8（98.8%） | 1.3 | 42.4 | 63.6 | 407.1 | 21.5% |
| 16 | 240.4 | 1.90（0.7%） | 236.1（98.8%） | 1.1 | 43.0 | 155.6 | 458.4 | 22.5% |
| 32 | 224.6 | 2.09（0.8%） | 221.2（98.6%） | 1.2 | 48.9 | 401.3 | 410.5 | 23.0% |
| 64 | 219.2 | 2.96（1.3%） | 214.6（97.7%） | 1.1 | 60.4 | 1556.9 | 415.8 | 28.4% |

p 标度（log–log，4 点）：ALM 总 **-0.033**（R²=0.56，p 效应 < trial 噪声）、装配 **+0.316**（R²=0.96，接近 O(p)）、
内层 **-0.037**（R²=0.65，常数）、SDP base **+0.172**（R²=0.88）、SDP base+RLT **+1.520**（R²=0.99）。

证书缓存（同 p 参数化 Shor SDP，编译一次后逐实例更新 `Qh`/约束系数）：cold 36.9–60.8 ms →
param-first 7.3–15.0 ms（含编译）→ 无 warm start 1.5–2.6 ms → SCS warm start **1.4–1.8 ms**；
warm 值与冷解 max\|Δ\|=7.2e-9，tightness max 9.2e-10，仍满足 1e-6 证书阈值。

## 2. B4 SCP vs ALM（`run_scp.py`；同实例、同 unc 热启动、同指标、同 `shor_bound` 证书）

主比较（noise=0.01, bias_z=0.08, n=50/格）：

| method | p=8 认证/可行/时间 | p=16 认证/可行/时间 | p=32 认证/可行/时间 |
|---|---|---|---|
| ALM-QPEP | 100% / 100% / 218 ms | 100% / 100% / 239 ms | 100% / 100% / 232 ms |
| SCP 单起点 | 98% / 100% / 130 ms | 98% / 100% / 135 ms | 98% / 100% / 142 ms |
| SCP + 2 重启（诊断） | 98% / 100% / 130 ms | 98% / 100% / 134 ms | 98% / 100% / 143 ms |

- 所有方法的 pen 中位 = 0.00 mm、rot 4.17°、trans 10.63 mm（p=16）；SCP 的可行解与 ALM 落在同一全局最优点（J=Shor）。
- SCP 每格 1 个失败 trial，全部是 seed=4（无约束初值为 cheirality 镜像解：depth_min<0、rot err≈180°），
  SCP 停在 J≈0.53 的可行但非最优姿态（J−Shor≈0.52），100 次迭代未收敛。
- 压力档（p=16, noise=0.02, bias_z=0.12, n=20，unc 镜像率 15%）：SCP 认证 **85%**（3 个失败，全镜像初值）
  vs ALM 100%；SCP 可行 100%、rot 5.43° vs ALM 5.25°、trans 13.07 vs 12.65 mm、147 vs 226 ms。
- 扰动初值重启（+2 起点）**没有救回**任何失败 trial；ALM 的内层多起点是在 ALM 增广二次型上重解，机制不同。
- SCP 模型自检（有限差分）：generic 点梯度相对误差 ≤4.6e-10、约束 Jacobian ≤1.8e-10、
  Hessian 方向 ≤9.1e-10、交叉项 ≤3.2e-7。

## 3. 判定与影响

1. **SDP 不主导**（ALM 的 21.5–28.4%），且可按 p 摊销到 1.4–1.8 ms/实例；主表把 SDP 成本与 ALM 前向分开报是合理的。
   RLT 的 p 标度 ~1.52（p=64 时 1.56 s、是 base 的 25.8×），应保持可选、只在难题档做紧性消融。
2. **ALM 的瓶颈是内层多起点 BFGS（~98%）**；优化方向是减少外层/内层重解次数，而不是 10×10 装配。
3. **SCP 是强 baseline，不能省略**：普通档认证率只差 2pp、中位快 ~1.7×；论文必须同时报 SCP 的收敛率/迭代数与
   certified 率。ALM 的差异化在：镜像初值鲁棒性（高噪声档 100% vs 85%）、p 无关的常数维先验、
   以及 FB/rank-deficiency 的理论链；不能对本批合成数据声称「认证率普遍高于 SCP」。
4. **附带发现**：p=64 时 ALM 的 active-set 一致性检查在 30%（主表 26%）trial 上失败并走 full-L fallback；
   fallback 便宜（~1.7 ms）但说明 LICQ 退化在高 p 下常见，论文应报告 fallback 率。

## 4. Caveats

- 单线程墙钟；测量时机器 load ~20–35（40 核共享），绝对 ms 会漂移，且与主表（默认多线程）不可逐位比较；
  可靠结论是同进程内的占比/p 斜率与配对比较。主表交叉检查（ALM `J` 逐位一致）排除了插桩改结果的可能。
- 内层是 6 起点 BFGS（QPEP Gröbner 枚举的数值替代）；换真实枚举后内层绝对占比会变。
- SCP 的「更快」只对本实现（cvxpy/CLARABEL vs scipy BFGS）成立；SCP 的失败是初值敏感性，
  不是凸子问题不可行。
- 合成实例底面为平面，p≥16 后精度饱和；只测前向 + 证书，不含反向（FB）与真实数据。

## 5. 复现

```bash
cd labs/02-constrained-qpeplayer/experiments/code
srun --partition=cpu --time=00:30:00 --ntasks=1 bash -c \
  'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1; \
   /home/syk/miniconda3/envs/sdpr/bin/python run_perf.py 2>&1 | tee ../results/run_perf.log'
srun --partition=cpu --time=00:30:00 --ntasks=1 bash -c \
  'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1; \
   /home/syk/miniconda3/envs/sdpr/bin/python run_scp.py 2>&1 | tee ../results/run_scp.log'
```
