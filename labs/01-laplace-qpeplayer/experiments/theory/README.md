# P1 补充证明的数值验证

对应 Feishu 补充证明（2026-11-27；wiki `AlOnwYsYdi7jJ8kiHd7c9UvRnfe`）。

| 脚本 | 验证内容 | 结果 |
|-|-|-|
| `run_unequal.py` | 定理 P1-5 regime I：$\Delta J=0$ 且曲率不等的分离盆地交叉；精确后验 vs Laplace | 误差斜率 **1.066**；$\tau=0.01$ 时相对梯度误差 **3.4e-3** |
| `run_uniform.py` | 同一模型的 $\tau$ sweep 与 $\Delta J$ sweep（辅助） | 见 `results/uniform_table.md` / `deltaJ_sweep.md` |
| `run_fold.py` | 定理 P1-5 regime II：Fold 合并；标准 Laplace vs $\tau^{1/3}$ 正则化 | $a=0.01,\tau=0.01$：标准 3.24 vs 正则 0.23 |
| `verify_q6.py` | P1-Q6 闭式 $d\log\det H$ vs 分支 FD | 5 个实例相对误差 1.7e-9–8.8e-9 |
| `run_scorer_fold.py` | P1-Q5c Scorer/半线 Airy 跨 Fold 延拓 | 1D 三路复现 0.867/0.496/0.3384→0.3383；6D fold $\theta_f=1.3185\times10^{-3}$，复共轭根追踪成功，位姿跳变 1.21e-2（标准 3.04e-1、正则 1.07e-1） |

命令（WM 环境，Slurm 或本地短任务）：

```bash
cd labs/01-laplace-qpeplayer/experiments/theory
/home/syk/miniconda3/envs/WM/bin/python run_unequal.py
/home/syk/miniconda3/envs/WM/bin/python run_fold.py
/home/syk/miniconda3/envs/WM/bin/python verify_q6.py
```

限制：1D normal form 是定理验证，不是 6D QPEP 主表；6D 主表/消融在 `../results/main_table.md` 与 `../results/ablation.md`。**2026-11-27 修订**：P1-Q5 已由新证明分层结案——任意纯实根 Laplace 代理（含 $\tau^{1/3}$ 正则化）在 Fold 处必有非零值跳变（命题 P1-Q5b，解释了 `../results/fold_6d.md` 的 0.309→0.107），复共轭根 + Scorer 半线 Airy 延拓给出跨 Fold 严格 $C^\infty$ 零跳变（定理 P1-Q5c，`paper claim`；6D 实现待单独一轮）。详见 `../../theory/proofs.md` §7.5。
