# 外部基线实测（合成 3D-3D，N=100，20 trials）

> 结论先行：在**同一批实例、同一 GT/随机初始化协议**下，Theseus-GM 随机初始化在 p=0.3/0.5/0.7 的成功率为 **80%/55%/25%**，而 GNC-GM 为 **100%**；Theseus-GT 初始化全成功（但这是局部盆地检查，不是全局恢复）。SDPRLayers+SCS（无鲁棒核）在 p≥0.3 全部失败，但 SCS 前向本身 **0/208 异常**。

## 0. 口径

- 任务：合成 3D-3D，N=100，20 trials/cell；外点=随机均匀 `[-2,2]^3`；噪声 σ=0.01；GM c=0.05。
- 成功判据：旋转误差 < 5° **且** 平移误差 < 0.1 m。
- 实例流：所有方法共用 `SeedSequence([20261127,103,0,100p,trial])`（N=100 逐位一致）。
- 计时：单进程 CPU；Theseus 计时**不含**目标函数构建（约 6 ms/trial），GNC/RANSAC 含完整调用。
- 证据分级：**独立复现**（本次实测）。原始 JSON 见 `experiments/baselines/results/`。

## 1. 主表（N=100）

| p | 方法 | 成功率 | rot 中位 (°) | trans 中位 (m) | 时间 p50 (ms) |
|---:|---|---:|---:|---:|---:|
| 0.0 | Kabsch | 1.00 | 0.09 | 0.001 | 0.0 |
| 0.0 | RANSAC-1000 | 1.00 | 0.09 | 0.001 | 0.2 |
| 0.0 | RANSAC-adaptive | 1.00 | 0.09 | 0.001 | 0.2 |
| 0.0 | IRLS-GM | 1.00 | 0.10 | 0.001 | 1.9 |
| 0.0 | GNC-GM | 1.00 | 0.10 | 0.001 | 0.4 |
| 0.0 | Theseus-GM[gt] | 1.00 | 0.10 | 0.001 | 127.2 |
| 0.0 | Theseus-GM[rand] | 1.00 | 0.10 | 0.001 | 358.4 |
| 0.0 | SDPR-SCS | 1.00 | 0.09 | 0.001 | 12.5 |
| 0.3 | Kabsch | 0.00 | 9.51 | 0.158 | 0.1 |
| 0.3 | RANSAC-1000 | 1.00 | 0.12 | 0.002 | 56.2 |
| 0.3 | RANSAC-adaptive | 1.00 | 0.12 | 0.002 | 1.1 |
| 0.3 | IRLS-GM | 1.00 | 0.12 | 0.002 | 1.9 |
| 0.3 | GNC-GM | 1.00 | 0.12 | 0.002 | 1.7 |
| 0.3 | Theseus-GM[gt] | 1.00 | 0.12 | 0.002 | 118.2 |
| 0.3 | Theseus-GM[rand] | 0.80 | 0.12 | 0.002 | 401.8 |
| 0.3 | SDPR-SCS | 0.00 | 9.51 | 0.158 | 12.8 |
| 0.5 | Kabsch | 0.00 | 21.63 | 0.258 | 0.1 |
| 0.5 | RANSAC-1000 | 1.00 | 0.14 | 0.001 | 54.9 |
| 0.5 | RANSAC-adaptive | 1.00 | 0.14 | 0.001 | 3.0 |
| 0.5 | IRLS-GM | 1.00 | 0.14 | 0.002 | 1.9 |
| 0.5 | GNC-GM | 1.00 | 0.14 | 0.002 | 1.7 |
| 0.5 | Theseus-GM[gt] | 1.00 | 0.14 | 0.002 | 121.9 |
| 0.5 | Theseus-GM[rand] | 0.55 | 0.24 | 0.003 | 665.2 |
| 0.5 | SDPR-SCS | 0.00 | 21.63 | 0.258 | 13.3 |
| 0.7 | Kabsch | 0.00 | 31.38 | 0.345 | 0.1 |
| 0.7 | RANSAC-1000 | 1.00 | 0.21 | 0.003 | 54.9 |
| 0.7 | RANSAC-adaptive | 1.00 | 0.21 | 0.003 | 14.1 |
| 0.7 | IRLS-GM | 1.00 | 0.21 | 0.003 | 1.9 |
| 0.7 | GNC-GM | 1.00 | 0.21 | 0.003 | 1.7 |
| 0.7 | Theseus-GM[gt] | 1.00 | 0.21 | 0.003 | 121.0 |
| 0.7 | Theseus-GM[rand] | 0.25 | 101.12 | 0.637 | 669.5 |
| 0.7 | SDPR-SCS | 0.00 | 31.38 | 0.345 | 13.0 |

## 2. 解读

1. **Theseus 随机初始化是弱基线**：p=0.3/0.5/0.7 成功率 0.80/0.55/0.25，中位旋转误差在 p=0.7 达 101.1°；典型局部极小，不是实现错误。
2. **Theseus GT 初始化不是全局性证据**：起点就是 GT，p=0.7 仍 100% 成功，只能说明 GM 目标在 GT 附近没有被外点拉走；不能据此说 Theseus 能全局恢复。
3. **GNC-GM 与 RANSAC-adaptive 全 p 成功**；GNC-GM 时间 p50 ≈ 1.7 ms，RANSAC-adaptive 在 p=0.7 退化到 14.1 ms。
4. **SDPR-SCS 在 W_i=I 时等价于 Kabsch**：逐 trial 配对 80 组，旋转最大差 **1.5e-5°**、平移最大差 **1.3e-9 m**；p=0 成功、p≥0.3 全部失败（中位旋转误差 9.5–31.4°）。因此这一行是 SDPR 前向正确性/可得性证据，**不是**独立的鲁棒方法；要有区分度必须给非单位 W_i（POLI）或 RANSAC 前端。208 次前向没有一次 SCS 异常。
5. 因此“Theseus/SDPR 在 30–70% 随机外点下失败、GNC 成功”在本协议下是**可测的**，不是假设；但 Theseus 的强/弱初始化必须分开报，否则会误导。

## 3. SDPR 的 N 依赖（前向，SCS eps=1e-9）

| N | p | 成功率 | rot 中位 (°) | trans 中位 (m) | p50 (ms) | SCS 异常率 |
|---:|---:|---:|---:|---:|---:|---:|
| 20 | 0.0 | 1.00 | 0.23 | 0.003 | 11.5 | 0.00 |
| 20 | 0.3 | 0.00 | 27.50 | 0.290 | 11.3 | 0.00 |
| 20 | 0.5 | 0.00 | 37.88 | 0.471 | 12.8 | 0.00 |
| 20 | 0.7 | 0.00 | 55.38 | 0.424 | 12.5 | 0.00 |
| 50 | 0.0 | 1.00 | 0.16 | 0.002 | 12.0 | 0.00 |
| 50 | 0.3 | 0.00 | 12.96 | 0.207 | 10.9 | 0.00 |
| 50 | 0.5 | 0.00 | 24.70 | 0.365 | 11.1 | 0.00 |
| 50 | 0.7 | 0.00 | 48.69 | 0.481 | 13.4 | 0.00 |
| 100 | 0.0 | 1.00 | 0.09 | 0.001 | 12.5 | 0.00 |
| 100 | 0.3 | 0.00 | 9.51 | 0.158 | 12.8 | 0.00 |
| 100 | 0.5 | 0.00 | 21.63 | 0.258 | 13.3 | 0.00 |
| 100 | 0.7 | 0.00 | 31.38 | 0.345 | 13.0 | 0.00 |

## 4. 范围与缺口

- **Theseus 稀疏求解器**：`scikit-sparse` 因缺 SuiteSparse `cholmod.h` 装不上，本次用 `CholeskyDenseSolver`；结果数值有效，但不能代表 Theseus 的稀疏大 N 性能。
- **SDPR 只跑了前向**：SCS 下 SDPR-IS 反向的 KKT 乘子精度问题见 `repro/upstream/REPORT.md` §5.6；本表不给反向数字。且 W_i=I 时前向目标与 Kabsch 相同，不能当作鲁棒性证据。
- **BPnP/Pyatov/POLI 不进入本表**：任务/几何不同或代码权重不可得，逐条理由见 `availability.md`。
- **SDPR≡Kabsch 配对校验**：`check_sdpr_kabsch_equivalence.py` → `results/sdpr_kabsch_equiv.json`（80 对，max |Δrot|=1.5e-5°，max |Δt|=1.3e-9 m）。
- **统计噪声**：20 trials/cell，Theseus-GM[rand] 的 Wilson 95% CI 在 p=0.3/0.5/0.7 为 [0.58,0.92]/[0.34,0.74]/[0.11,0.47]；GNC-GM 为 20/20，CI [0.84,1.00]。p=0.3 的区间重叠，**不能**声称 16/20 vs 20/20 显著；p=0.5/0.7 的区间分离。
- **未做**：相干外点、边界外点、真实数据；RANSAC 预算未调参。

## 5. 复现命令

```bash
cd labs/03-robust-qpeplayer/experiments/baselines
# GNC/RANSAC/IRLS/Kabsch, 20 trials, same instances
/home/syk/miniconda3/envs/WM/bin/python run_reference.py \
  --trials 20 --out results/reference_n20.json
# Theseus-GM, GT + random init
THESEUS_TRIALS=20 /mnt/storage1/syk/venvs/theseus/bin/python run_theseus.py
# SDPRLayers + SCS forward, N=20/50/100
PYTHONPATH=/mnt/storage1/syk/third_party/sdprlayer/src \
  /home/syk/miniconda3/envs/sdpr/bin/python run_sdpr_forward.py
# table
/home/syk/miniconda3/envs/WM/bin/python make_baselines_table.py
```
