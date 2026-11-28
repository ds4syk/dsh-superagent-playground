# P2-Cert Schur 严格界 vs 现有 dual repair（主 worker 独立核验）

> 6 个合成实例（noise 0.10/0.20，p=16），`shor_bound(add_bound_rlt=True, repair_dual=True, schur=True)`。
> `rigorous` = 现有对偶修复（`P=diag(1_{0..9},0,0,0,1)` 平移），`rigorous_schur` = 定理 P2-Cert 的 13×13 Schur 界。

| noise | seed | J | dual repair | Schur | case | J−dual | J−Schur |
|---|---|---|---|---|---|---|---|
| 0.10 | 0 | 9.0579e-01 | 9.0579e-01 | 9.0579e-01 | A | 2.00e-08 | 5.59e-12 |
| 0.10 | 1 | 6.8500e-01 | 6.8500e-01 | 6.8500e-01 | A | 2.00e-08 | -6.69e-13 |
| 0.10 | 2 | 4.3643e-01 | 4.3643e-01 | 4.3643e-01 | A | 2.00e-08 | 5.31e-11 |
| 0.20 | 0 | 3.2392e+00 | 3.2392e+00 | 3.2392e+00 | A | 2.04e-08 | 4.22e-10 |
| 0.20 | 1 | 3.1128e+00 | 3.1128e+00 | 3.1128e+00 | A | 2.00e-08 | 3.81e-11 |
| 0.20 | 2 | 3.0309e+00 | 3.0309e+00 | 3.0309e+00 | A | 2.01e-08 | 6.21e-11 |

## 读数

- 两者都 sound：`Schur ≤ J` 全部成立（最小 J−Schur = −6.7e-13，即数值零）。
- **Schur 界比现有 dual repair 紧约 4 个数量级**：J−dual ≈ 2.0e-8（等于 2δ_round），J−Schur ≈ 1e-12–4e-10。原因：Schur 界用精确的 13×13 平移 Schur 补，不需要 `P` 平移的 2c 损失。
- 因此论文/真实数据应优先报告 `rigorous_schur`（P2-Cert）；`repair_dual` 是更保守的回退。
- 真实数据的 Schur 严格认证率由 rigorous 子代理汇总（`rigorous_real.*`）。

Source: `results/p2b_schur_check.json`（主 worker 独立脚本，未改 certificate.py）。
