# P2 内层枚举 quartic_roots 性能：`_track_batch` 优化（S32）

> 数字全部由 `results/perf_quartic.json` 汇总；基准脚本为本次会话的 `/tmp/bench_s32.py`，base = `git show 3fa99be:.../quartic_roots.py`，opt = 当前 worktree。**login node、单线程、共享负载（load ~20–23/40）**，绝对秒数会漂移，同进程配对比较是主结论。

## 结论

- 优化：`Jqq = 2·einsum('nip,ij,njq->npq', Jv, M, Jv)` 改为先 `MJ = M @ Jv` 再 `Jvᵀ @ MJ`（两次 batched matmul / BLAS），数学上完全等价；单次枚举 **0.488 s → 0.168 s（2.92x）**，路径吞吐 **356 → 1077 paths/s**。
- 等价性：15 个 case（10 个 p=16 主表实例 + 3 个 generic 随机 M + p=8/64 各 1 个）根数/有限射影根/实根/发散/丢根计数**全部逐项相同**，max |Δf| = **3.3e-16**（阈值 1e-12）。
- 批评估次数不变（median 470），说明省掉的是每次 `kkt_eval_batch` 内部的 3 操作数 einsum，不是少算/跳步；`t_polish` 仍可忽略（<0.01 s）。

## 设置

- 实例：`qpep_core.make_instance(trial, 16, noise=0.01, bias_z=0.08)`，`Mred=obj_min_t(M0,D0,G0)`，trials 0–9；generic 为固定种子（7）的随机对称 10×10；另测 p=8/64 的 trial 3。
- 每次 `critical_points(M, n_gamma=1, return_stats=True)`；每版本每实例 3 次，取 wall 中位数。
- numpy 1.26.3，节点 ls，OMP/OPENBLAS/MKL/NUMEXPR=1。

## 表 1. 逐实例（10 个 p=16 主表实例，每格 3 次中位）

| case | 根数 base/opt | 有限射影 | 丢根 | max|Δf| | base s | opt s | 加速 | paths/s base→opt |
|---|---|---|---|---|---|---|---|---|
| p16_t0 | 16/16 | 40/40 | 0/0 | 6.9e-17 | 0.543 | 0.176 | 3.09x | 311→965 |
| p16_t1 | 16/16 | 40/40 | 0/0 | 5.6e-17 | 0.543 | 0.173 | 3.14x | 310→1083 |
| p16_t2 | 16/16 | 40/40 | 0/0 | 5.6e-17 | 0.487 | 0.165 | 2.95x | 347→1146 |
| p16_t3 | 16/16 | 40/40 | 0/0 | 4.2e-17 | 0.432 | 0.152 | 2.84x | 393→964 |
| p16_t4 | 16/16 | 40/40 | 0/0 | 4.2e-17 | 0.463 | 0.168 | 2.76x | 365→1096 |
| p16_t5 | 16/16 | 40/40 | 0/0 | 4.2e-17 | 0.489 | 0.160 | 3.05x | 349→1156 |
| p16_t6 | 16/16 | 40/40 | 0/0 | 6.9e-17 | 0.494 | 0.169 | 2.92x | 364→1052 |
| p16_t7 | 16/16 | 40/40 | 0/0 | 2.8e-17 | 0.501 | 0.172 | 2.91x | 322→1051 |
| p16_t8 | 16/16 | 40/40 | 0/0 | 4.2e-17 | 0.468 | 0.160 | 2.93x | 363→1159 |
| p16_t9 | 16/16 | 40/40 | 0/0 | 2.8e-17 | 0.453 | 0.168 | 2.70x | 376→1071 |

汇总（n=10）：median **0.488 s → 0.168 s（2.92x）**；min/max 加速 2.70/3.14；paths/s median 356 → 1077；median wall 分解 t_track 0.438 → 0.148 s，t_polish <0.01 s。

## 表 2. 附加等价性 case（不参与计时汇总）

| case | 根数 base/opt | 有限射影 | max|Δf| | base s | opt s | 加速 |
|---|---|---|---|---|---|---|
| generic_0 | 12/12 | 40/40 | 3.3e-16 | 0.437 | 0.148 | 2.94x |
| generic_1 | 28/28 | 40/40 | 1.1e-16 | 0.415 | 0.154 | 2.69x |
| generic_2 | 16/16 | 40/40 | 2.2e-16 | 0.452 | 0.163 | 2.78x |
| p8_t3 | 16/16 | 40/40 | 2.8e-17 | 0.431 | 0.144 | 2.98x |
| p64_t3 | 16/16 | 40/40 | 4.2e-17 | 0.432 | 0.147 | 2.94x |

## 为什么是这一项（cProfile，优化前）

单次枚举 0.579 s 中：`c_einsum` 0.372 s（64%）、`kkt_eval_batch` 自身 0.072 s、`np.linalg.solve` 0.031 s；479 次批评估 / 162 条路径 ≈ 3 次/步。热点的 3 操作数 einsum 把固定的 M 夹在两个 (n,10,4) 张量中间，numpy 不做收缩顺序优化；先算 `MJ = M @ Jv`（每路径 10×10×4）再 `Jvᵀ @ MJ`（每路径 4×10×4），乘加量从 n·10·10·4·4 降到 n·(10·10·4 + 4·10·4)，且走 BLAS。

## 试过但否掉的优化：提前剔除明显发散路径

在 6 个 p=16 实例上把 `critical_points(div_tol=...)` 从 1e4 降到 1e3：中位 0.469 s → 0.476 s（无收益）；降到 1e2 时有 1/6 实例的有限射影根从 40 掉到 39，降到 30 时 3/6 实例掉根（38–40）。**不满足 exact parity，revert**。根因：发散路径的范数增长与有限根的端game 重叠，早剔除会误杀最终收敛到有限根的路径；`div_tol=1e4` 只能作为“最终分类”阈值，不能作为早期 cull 阈值。

## 复现

```bash
cd labs/02-constrained-qpeplayer/experiments/code
# base：
git -C /mnt/storage1/syk/.superagent-work/wt-p2b-eng show 3fa99be:labs/02-constrained-qpeplayer/experiments/code/quartic_roots.py > /tmp/quartic_base.py
# 然后按 /tmp/bench_s32.py 的 15 case 配对协议跑 3 rep；或用 run_quartic.py 复核根数/值：
/home/syk/miniconda3/envs/sdpr/bin/python run_quartic.py
```
