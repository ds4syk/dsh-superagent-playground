# Systems / 批量 GNC-GM（3D-3D）

> 一句话结论：把 `synthetic/common.py::solve_gnc` 的逐实例循环向量化到 (B,N,3) 批量后，
> 同一调度下 CPU 中位加速 **6.60×**（B=32,N=100：loop 56.2 ms → batch 8.5 ms），
> 正确性（旋转 Frobenius / 平移）与循环差 **~1e-15 / ~1e-15**，`n_mu` 逐实例一致；
> GPU（RTX 5090, float64, eager torch）在 B≤128 时被 kernel launch 开销主导（~40 ms 地板），
> 只在 B=128 时相对 CPU 循环有 5–10× 优势，不建议替代 CPU 批量实现。

本目录只做**工程吞吐**验证，不修改 `theory/`、`verify/` 或既有实验结论。

## 文件

| 文件 | 内容 |
|---|---|
| `batch_gnc.py` | numpy 批量 GNC-GM：批量加权 Kabsch（`eigh` Horn/Davenport 4×4 或 `svd` 3×3 后端）、批量残差、逐实例 μ 调度 |
| `batch_gnc_torch.py` | 同算法的 torch 镜像（CPU/CUDA），供 GPU 对照 |
| `run_batch.py` | benchmark 主脚本：B∈{1,8,32,128} × N∈{100,500} × 100 trials，生成 JSON/MD/PNG |
| `verify_batch.py` | 固定种子正确性检查（eigh/svd、torch CPU/CUDA、`n_mu` 一致性） |
| `results/batch_gnc.{json,md,png}` | 最终结果（CPU + GPU） |
| `results/batch_gnc_cpu.json` / `batch_gnc_gpu.json` | 分节原始结果，供 `--mode merge` 重建 |
| `results/batch_gnc_raw*.jsonl` | 每 trial 的计时与误差原始记录 |
| `results/logs/` | Slurm 运行日志 |

## 设计要点

1. **调度逐位复刻**：`mu0 = max(3 max e / c², 1)`、`gamma=1.4`、≤40 次 μ 迭代、
   `mu <- max(1, mu/gamma)`、μ=1 那次迭代后退出、末尾 5 次 exact-kernel IRLS。
   每个实例按自己的 μ0 走自己的迭代数；结束的实例从 active 集合移除，**不重算**，
   因此 `n_mu` 与循环逐实例相等（表中最后一列为 0 mismatch）。
2. **线性代数批量**：内部 (B,3,N) 布局，每次迭代只有若干个 BLAS `matmul` +
   一次批量 4×4 `eigh`（或 3×3 SVD）；残差用一次 `matmul` + `einsum` 求平方和。
3. **eigh vs svd**：Horn/Davenport 四元数法与 Kabsch SVD 给出同一最优旋转；
   `verify_batch.py` 逐实例对比两者差 ≤2e-15（本报告固定实例），另在随机/近奇异 H 上
   压测 worst 1.6e-13。要求与循环走同一 LAPACK 路径时用 `method="svd"`。

## 正确性（判据）

任务判据：旋转 Frobenius < 1e-8、平移 < 1e-10。实测（固定种子，B=8，p∈{0,0.3,0.7}，
N∈{100,500}）：旋转 ≤2.1e-15，平移 ≤1.2e-15，`n_mu` mismatch = 0，torch CUDA float64
同样通过（≤1e-14）。`verify_batch.py` 失败时退出码非 0。

## 运行命令（Slurm，单节点 `ls`）

```bash
cd labs/03-robust-qpeplayer/experiments/systems
PY=/home/syk/miniconda3/envs/WM/bin/python

# 0) 正确性（~6 s，含 torch CPU/CUDA）
srun --partition=cpu --time=00:10:00 --ntasks=1 $PY verify_batch.py

# 1) CPU benchmark（~3 min；loop + numpy batched，100 trials/cell）
srun --partition=cpu --time=00:30:00 --ntasks=1 --cpus-per-task=4 \
  $PY run_batch.py --mode cpu 2>&1 | tee results/logs/batch_cpu_$(date +%Y%m%d_%H%M%S).log

# 2) GPU 对照（~3 min；torch float64，复用 CPU loop 作为加速比参照，同一节点）
srun --partition=gpu --gres=gpu:1 --time=00:30:00 --ntasks=1 --cpus-per-task=4 \
  $PY run_batch.py --mode gpu 2>&1 | tee results/logs/batch_gpu_$(date +%Y%m%d_%H%M%S).log

# 3) 可选：只重建合并报告
$PY run_batch.py --mode merge
```

阈值/网格在 `run_batch.py` 顶部（`B_LIST`, `N_LIST`, `TRIALS`, `P_OUT`, `METHOD`）。
种子固定为 `SeedSequence([20261127, 11, B, N, round(100p), trial])`，CPU 与 GPU 模式
使用同一批实例，因此 GPU 正确性可直接与 CPU 循环逐实例对比。

## 计时口径

* `time.perf_counter` 包住**整批调用**（B 个实例）；每 cell 先 warm-up 一次不计入；
  实例生成在计时之外。
* 循环 = `for i in range(B): common.solve_gnc(x[i], y[i])`；批量 = 一次
  `batch_solve_gnc(x, y)`。二者同一批输入。
* GPU：`torch.cuda.synchronize` 包住计算；H2D 传输单独计时（p50 见报告），不计入 batch ms。
* 共享节点（Slurm `cpu` 分区，同一台 `ls`），绝对 ms 有负载漂移，以 p50 为主、p95 参考。

## 结果摘要

见 [`results/batch_gnc.md`](results/batch_gnc.md)。要点：

| B | N=100 loop→batch p50 | 加速 | N=500 loop→batch p50 | 加速 |
|---|---|---|---|---|
| 1 | 1.82 → 3.35 ms | 0.54× | 3.17 → 3.82 ms | 0.83× |
| 8 | 14.01 → 4.75 ms | 2.95× | 24.76 → 7.58 ms | 3.27× |
| 32 | 56.22 → 8.52 ms | **6.60×** | 101.68 → 53.54 ms | 1.90× |
| 128 | 222.22 → 49.10 ms | 4.53× | 410.50 → 228.16 ms | 1.80× |

* B=1 时批量调用的 numpy/Python 固定开销占主导，比循环慢；批量收益在 B≥8 出现。
* N=500 时每轮迭代的残差/中心化偏内存受限，加速比低于 N=100。
* GPU（float64，eager）：B=128,N=500 时 410.5 → 49.8 ms（~8×），但 B≤32 时被 ~40 ms
  launch 地板拖累（B=1 比 CPU 循环慢 ~20×）。详见报告 §5/§6。

## 边界

* 只改 `labs/03-robust-qpeplayer/experiments/systems/`，未动 `theory/`、`verify/`、
  `labs/EXPERIMENTS.md`、`AGENTS.md`、`deliverables/`；无 commit/push。
* GPU 只是对照：未做 CUDA graph / torch.compile 融合、未测 float32、未扫更大 B。
* 未做 end-to-end 学习层吞吐（只测前向 GNC-GM）。
