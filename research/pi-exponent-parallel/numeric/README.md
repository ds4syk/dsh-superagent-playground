# numeric/ — π 连分数的高精度数值实验（μ(π) ≤ 2 的数值证据）

**主报告**：[report.md](report.md)（结论、数据、反例、猜想、复现清单）
**研究日志**：[log.md](log.md)（每轮的动机、错误与修正）

一句话：用 1e6 位 π（两条独立算法逐位一致）认证了 **970,563 个连分数部分商**，
把 μ(π) ≤ 2 化为 `T_K := sup_{k≥K} log q_{k+1}/log q_k → 1`，数据给出
`T_K−1 = 1.02e-5 @ K=9e5`（与 Gauss–Kuzmin 随机模型一致），并证明（数值上）
所有经典构造族都无法绕过 Legendre 定理。**这不是证明。**

```
scripts/   全部脚本（纯 Python；pi_agm, pi_chudnovsky, certified_cf, cf_analyze,
           verify_convergents, gk_montecarlo, compare_constants, constructions,
           best_rationals, counterexamples, plots）
data/      数字、商表、统计 JSON/CSV、图（data/plots/）
logs/      运行日志
```

快速核对：`python3 -c` 见 report.md §11；核心数字在 `data/cf_d1e6_stats.json`。
