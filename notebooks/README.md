# Notebooks：SDPR 与 Dreamer 的训练全过程

把两份训练代码整理成可执行的 notebook，每一步都有解释、真实输出和曲线：

| Notebook | 内容 | 代码来源 | 运行环境 |
|-|-|-|-|
| [`sdpr/sdpr_training_walkthrough.ipynb`](sdpr/sdpr_training_walkthrough.ipynb) | QCQP → Shor SDP → 隐式微分 → 有限差分校验 → 论文 §VI-B4 相机基线标定训练（真跑 20 步） | [utiasASRL/sdprlayer](https://github.com/utiasASRL/sdprlayer)（MIT） | `sdpr`，CPU |
| [`dreamer/dreamer_training_walkthrough.ipynb`](dreamer/dreamer_training_walkthrough.ipynb) | 配置 → 环境/回放 → 世界模型损失 → 想象 → actor-critic → 真跑 3000 步 → 开放回路预测 | [danijar/dreamerv3](https://github.com/danijar/dreamerv3)（MIT） | `dreamer`，单张 GPU |

`assets/` 里是两张训练管线示意图（notebook 正文引用）。一页总览图在
[`../docs/visuals/2026-11-09-sdpr-dreamer-notebooks.png`](../docs/visuals/2026-11-09-sdpr-dreamer-notebooks.png)。

## 本次执行结果（2026-11-09，本机实测）

| | 结果 |
|-|-|
| SDPR | 20 个位姿全部 rank-1（min λ1/λ2 = 1.4e9）；位姿旋转误差 0.33°、平移 1.8 mm；梯度 autograd vs 有限差分 = 4.065 vs 4.083（0.44%）；标定 b: 0.243 → 0.24088 m（真值 0.240），外层 loss 1.75e-2 → 1.31e-2 |
| Dreamer | 10,494,158 参数；3000 环境步 / 约 1900 次更新，训练循环约 100 s；`loss/image` 约 960 → 43，`loss/rew` 5.5 → 0.4，`loss/con` 0.58 → 0.02，`loss/dyn` 11.3 → 5.5 |
| Dreamer 完整 run（对照） | 本机已有约 30 万步的 `dmc_walker_walk`：episode 得分 57.7 → 956（最高 990） |

## 怎么跑

两个 notebook 都依赖本机（实验室 server-2026-2）上的代码快照和 conda 环境：

```bash
# SDPR：CPU 约 4 分钟
conda activate sdpr
export SDPR_ROOT=~/labs/SDPR
python -m nbconvert --to notebook --execute --inplace \
  notebooks/sdpr/sdpr_training_walkthrough.ipynb

# Dreamer：单卡 GPU 约 6 分钟（含 JIT 编译），需要 MuJoCo/EGL
conda activate dreamer
export DREAMER_REPO=~/labs/Dreamer/repos/dreamerv3
export MUJOCO_GL=egl
python -m nbconvert --to notebook --execute --inplace \
  notebooks/dreamer/dreamer_training_walkthrough.ipynb
```

在本机按实验室规范用 Slurm 提交：

```bash
# Dreamer（GPU）
sbatch --partition=gpu --gres=gpu:1 --cpus-per-task=8 --mem=48G --time=01:00:00 \
  ~/labs/Dreamer/tools/slurm/run_gpu.sbatch bash -c '
    source ~/labs/Dreamer/tools/env.sh
    cd <repo> && python -m nbconvert --to notebook --execute --inplace \
      --ExecutePreprocessor.kernel_name=dreamer-jax \
      notebooks/dreamer/dreamer_training_walkthrough.ipynb'

# SDPR（CPU）
sbatch --partition=cpu --cpus-per-task=4 --mem=8G --time=00:40:00 --wrap '
  export PATH=~/miniconda3/envs/sdpr/bin:$PATH
  cd <repo> && python -m nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.kernel_name=sdpr-torch \
    notebooks/sdpr/sdpr_training_walkthrough.ipynb'
```

> `sdprlayer` 官方默认用 MOSEK；本机快照加了 `solver="SCS"` 选项，notebook 用 SCS 跑通。
> 换成 MOSEK 时梯度更干净（论文的报告精度在 `1e-6` 量级），把 `solver="SCS"` 改掉即可。

## 两本 notebook 的阅读顺序

1. **Dreamer**：先看「一次训练迭代」的完整数据流，再看世界模型损失和想象损失逐项怎么算，
   最后看真跑出来的曲线和开放回路预测。
2. **SDPR**：先看一个 13×13 QCQP 怎么被解出来、怎么对解求导，再看梯度校验和标定训练；
   最后一节说明论文 §VI-C 的 VGG16+UNet 视觉管线，以及「训练一个网络」和「训练一个标量」
   为什么是同一个循环。

每本 notebook 都是自包含的：路径、环境、公式解释、代码位置、结果解读和「局限」都写在里面。
