# MC-PPO：在真实 Minecraft 上用 PPO 训练（DreamerV3 同款环境 + 手写稠密奖励）

这是一个可运行的 PPO × Minecraft 训练系统：真实 Minecraft 1.11.2（MineRL/Malmo 驱动），
沿用 DreamerV3 官方 Minecraft Diamond 任务的 64×64 观测与 25 复合动作，使用**手写稠密奖励**
（物品获取量/效率、合成里程碑、生命值、探索/深度）训练一个循环 PPO 策略，目标是在
30 分钟 episode 内从零走到钻石。

> 交付口径：本仓库交付**训练系统 + 奖励函数 + 真实训练曲线/数据**。
> 受单节点 CPU 环境吞吐限制，本次实验训练到 **约 100 万策略步**量级（见 `kb/04`），
> **未**在本会话内达成钻石；到钻石的算力估算与可行路线见 `kb/05`。

## 目录

```
mcppo/                  # 训练系统
  envs/minecraft.py     # 真实 Minecraft env（MineRL EnvSpec + Dreamer 动作抽象）
  envs/vec.py           # 8 路子进程并行 + 自动 reset
  rewards/items.py      # 物品价值表（MC Wiki 科技树）
  rewards/reward.py     # 手写奖励引擎（势函数 + 里程碑 + 生存 + 探索）
  ppo/model.py          # CNN + inventory MLP + LSTM actor-critic
  ppo/ppo.py            # rollout/GAE/PPO 更新（分块 BPTT）
  train.py              # 训练循环、JSONL/TensorBoard 日志、checkpoint
  evaluate.py           # 单 episode 评测 + 录像
  plot_curves.py        # 曲线 PNG/PDF
configs/                # 训练/冒烟配置
scripts/                # Slurm 作业 + Xvfb/MineRL 环境脚本
tests/                  # 奖励/模型/GAE 单元测试
kb/                     # 中文知识库（设计、奖励、实验、可行性）
```

## 快速开始

环境（已在实验室节点安装好，可直接用）：

```bash
MC=/mnt/storage1/syk/dreamer/envs/mcppo
bash scripts/prepare_mc.sh        # 一次性：把 0.7 GB 游戏树落到 NVMe（/tmp/mcppo）
source scripts/mc_env.sh          # Java 8 / MineRL / Xvfb 依赖 + TMPDIR 指向 NVMe
DISP=:90; $XP/root/usr/bin/Xvfb $DISP -screen 0 1024x768x24 +extension GLX +render &
export DISPLAY=$DISP
```

> 节点上的 `/mnt/storage1` 是繁忙 HDD，8 个实例各自复制 0.7 GB 游戏树会互相拖慢；
> `prepare_mc.sh` 先把游戏树复制到 NVMe，训练时每个实例从 NVMe 复制到 `TMPDIR`。

冒烟（2 envs，128 步，约 3 分钟）：

```bash
python -m mcppo.train --config configs/smoke.json --out-dir runs/smoke --steps 128
```

正式训练（6 envs，120 万样本，Slurm）：

```bash
sbatch scripts/train_diamond.sbatch
# 任意时刻画曲线：
python -m mcppo.plot_curves --run /mnt/storage1/syk/dreamer/runs/mcppo/diamond_s0
```

评测/录像：

```bash
python -m mcppo.evaluate --run <run_dir> --episodes 1 --record <run_dir>/eval.mp4
```

## 奖励函数（摘要）

```
r = β(γΦ(s')-Φ(s)) + Σ 里程碑 + 生存 + 探索/深度 + 时间惩罚
Φ = Σ v_i log(1+max_inv_i) + 0.005·min(400, 新格子数) + 0.02·(64-min_y)
```

完整公式、物品价值表（含 MC Wiki 依据）和反 reward-hacking 分析见
[`kb/03 手写奖励函数.md`](kb/03%20手写奖励函数.md)。

## 训练数据与曲线

每次运行在 `run_dir` 下保存：

| 文件 | 内容 |
|-|-|
| `config.json` | 完整配置快照 |
| `metrics.jsonl` | 每个 update 的 loss/熵/KL/clip/吞吐 |
| `episodes.jsonl` | 每个 episode 的 return/长度/score/奖励分量 |
| `tb/` | TensorBoard events |
| `ckpt/latest.pt`, `ckpt/step_*.pt` | 模型 + 优化器 + 步数（可 `--resume`） |
| `curves/curves.png`, `curves.pdf` | 汇总曲线 |

## 来源与致谢

- 环境与 25 动作抽象改编自 **DreamerV3** 官方实现
  ([danijar/dreamerv3](https://github.com/danijar/dreamerv3)，MIT)，`embodied/envs/minecraft_flat.py`。
- 运行时：MineRL 0.4.4（含 Malmo + Minecraft 1.11.2）+ Java 8 + Xvfb + Mesa llvmpipe。
- PPO：Schulman et al., 2017；势函数塑形：Ng, Harada & Russell, 1999。
- 物品/合成/生命机制参考 [Minecraft Wiki](https://minecraft.wiki/)。
