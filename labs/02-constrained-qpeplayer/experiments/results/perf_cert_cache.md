# 证书缓存：参数化 14×14 Shor SDP（编译一次 / p，warm start）

> 由 `experiments/code/cert_cache.py --bench` 自动生成；逐 trial 原始数据在 `perf_cert_cache.json`。**不修改 certificate.py**：等式/Veronese 约束取自 `certificate._base_eq_specs` / `_veronese_specs`，返回字段与 `shor_bound` 兼容。

设置：p=16，noise=0.010，bias_z=0.080，n=20 个实例（seed=trial，与主表同场景）；SCS eps=5e-11、max_iters=80000（比主表冷解默认 eps=1e-9 更紧）。`cache warm` 按生产顺序热启动（实例 i 从实例 i-1 的解出发），第 1 个解含 cvxpy 编译 (45.3 ms)，其余为稳态。

DPP 检查：`prob.is_dpp() = True`（参数是 Q 与 Gp；编译结果可跨实例复用）。

## 表 1. 冷解 vs 缓存（ms，median [IQR]）

| 模式 | 时间 ms | Δvalue abs vs 同精度冷解 | Δvalue abs vs 主表默认冷解 | tightness λ₂/λ₁ max |
|---|---|---|---|---|
| 冷解（主表默认 eps=1e-9） | 35.0 [33.8, 38.9] | — | — | 1.0e-09 |
| 冷解（同精度 eps=5e-11） | 36.1 [34.2, 38.7] | — | — | 2.3e-11 |
| 缓存 warm（含第 1 次编译） | 8.6 [7.2, 9.5] | 8.1e-11（max 3.6e-10） | 3.8e-10（max 3.4e-09） | 4.3e-11 |
| 缓存 warm（稳态，n−1） | 8.4 [7.2, 9.4] | — | — | — |
| 缓存 nowarm（仅复用编译，诊断） | 12.6 [11.2, 13.8] | max 4.1e-10 | — | — |

**判定**：稳态 warm 比同精度冷解快 **4.2×**，`|Δvalue|` max = **3.6e-10**（≤1e-9），tightness max = 4.3e-11（≤1e-6，与冷解同档）；回退冷解 0 次，状态 optimal。

- 对主表默认冷解（eps=1e-9）的差最大 3.4e-09：这是冷解自身的求解误差量级，仍比证书阈值 1e-6 小 292×。

## 复现

```bash
cd labs/02-constrained-qpeplayer/experiments/code
/home/syk/miniconda3/envs/sdpr/bin/python cert_cache.py --bench --n 20 --p 16
QPE_CERT_CACHE=1 /home/syk/miniconda3/envs/sdpr/bin/python run_main.py   # 主表走缓存
```
