# P1-Q3 稳健性：11 个实例的 40 根计数（2026-11-28）

**结论：11/11 个实例恰有 40 个复射影根**（10 个随机对称 M + `Instance.M(0)`），最大 chart 残差 7.1e-15，单实例 7–9 s。

方法同 `roots/run_rootcount_fast.py`（total-degree homotopy，4 个仿射 chart 取并）；本表只列计数与残差，根坐标在 `results/rootcount_many.json`。

| instance | 复射影根 | 实射影 | 复射影(非实) | 局部极小 |
|-|-|-|-|-|
| random_seed1 | 40 | 12 | 28 | 2 |
| random_seed2 | 40 | 18 | 22 | 3 |
| random_seed3 | 40 | 14 | 26 | 2 |
| random_seed4 | 40 | 12 | 28 | 3 |
| random_seed5 | 40 | 14 | 26 | 2 |
| random_seed6 | 40 | 22 | 18 | 4 |
| random_seed7 | 40 | 12 | 28 | 3 |
| random_seed8 | 40 | 14 | 26 | 3 |
| random_seed9 | 40 | 22 | 18 | 4 |
| random_seed10 | 40 | 20 | 20 | 3 |
| instance_M_theta0 | 40 | 20 | 20 | 4 |

读法：40 个复射影根 = 80 个仿射根（± 双覆盖）；实射影根数随 M 变化（12–22），局部极小 2–4。
这支持 P1-Q3 的 generic 40 根；完整性仍是数值层面（4 卡并饱和），不是符号商环上界。

复现：`/home/syk/miniconda3/envs/WM/bin/python roots/run_rootcount_many.py`（约 80 s）。
