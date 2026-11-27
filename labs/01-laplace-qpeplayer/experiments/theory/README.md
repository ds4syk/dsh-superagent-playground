# P1 补充证明的数值验证

对应 Feishu 补充证明（2026-11-27；wiki `AlOnwYsYdi7jJ8kiHd7c9UvRnfe`）。

| 脚本 | 验证内容 | 结果 |
|-|-|-|
| `run_unequal.py` | 定理 P1-5 regime I：$\Delta J=0$ 且曲率不等的分离盆地交叉；精确后验 vs Laplace | 误差斜率 **1.066**；$\tau=0.01$ 时相对梯度误差 **3.4e-3** |
| `run_uniform.py` | 同一模型的 $\tau$ sweep 与 $\Delta J$ sweep（辅助） | 见 `results/uniform_table.md` / `deltaJ_sweep.md` |
| `run_fold.py` | 定理 P1-5 regime II：Fold 合并；标准 Laplace vs $\tau^{1/3}$ 正则化 | $a=0.01,\tau=0.01$：标准 3.24 vs 正则 0.23 |
| `verify_q6.py` | P1-Q6 闭式 $d\log\det H$ vs 分支 FD | 5 个实例相对误差 1.7e-9–8.8e-9 |

命令（WM 环境，Slurm 或本地短任务）：

```bash
cd labs/01-laplace-qpeplayer/experiments/theory
/home/syk/miniconda3/envs/WM/bin/python run_unequal.py
/home/syk/miniconda3/envs/WM/bin/python run_fold.py
/home/syk/miniconda3/envs/WM/bin/python verify_q6.py
```

限制：1D normal form 是定理验证，不是 6D QPEP 主表；6D 主表/消融在 `../results/main_table.md` 与 `../results/ablation.md`。P1-Q5 的**梯度余项引理**仍未严格证明，只有证明草图 + 上述数值；这是当前唯一 P0 级缺口。
