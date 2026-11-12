# SuperMimic 复现（2026-11）

> **一句话结论**：官方实现单卡 1.5 h 得到 LAFAN1 40 条动作的 Appendix-F 归一化回报 **0.859**（最低 0.815，失败 1/40），复现了论文 LAFAN1 对照的 0.82；同预算 PPO-Sim 只有 **0.210**（失败 38/40）。量化结果见 [results/](results/) 与 [report/复现报告.md](report/复现报告.md)。

本目录是论文 **SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World Model**（CoRL 2026，[arXiv:2610.28480](https://arxiv.org/abs/2610.28480)）的复现记录，放在知识库的 `vla/` 方向下。

## 目录

| 路径 | 内容 |
|-|-|
| `repro/` | 数据转换、Slurm 训练/评测脚本、日志解析 |
| `results/` | 训练曲线数据（CSV/JSON）、评测结果与图 |
| `report/` | 复现报告（方法、环境、实验协议、结果、差距） |

## 论文与官方实现

- 论文：arXiv:2610.28480（CoRL 2026），项目页 <https://supermimic.github.io>
- 官方代码：<https://github.com/HybridRobotics/supermimic>（Apache-2.0），配套 <https://github.com/HybridRobotics/mjlab_research>
- 官方实现依赖：Python 3.13、`uv`、mjlab（MuJoCo Warp）、单张 NVIDIA GPU

## 复现范围

1. **完整训练链路**：官方 SuperMimic 世界模型 + 策略联合训练（算法 A.1），在 LAFAN1 动作库上跑通。
2. **受控基线对比**：同一动作、同一观测/奖励/终止/域随机化下，mjlab 的 rsl_rl PPO-Sim 作为基线；比同墙钟预算下的训练回报与检查点评测。
3. **世界模型性质**：8 步开环预测误差、想象 rollout 吞吐、以及 BPTT 相对 PPO-WM 的样本效率（可选）。
4. **明确不覆盖**：BONES-SEED 全量 12.9 万条动作、SONIC 官方检查点、真机 G1 部署、多卡扩展。原因见报告「与论文的差距」。

## 快速开始

```bash
# 1) 数据：下载 LAFAN1 G1 retarget CSV -> npz
python repro/csv_to_npz_local.py --input-glob '.../lafan1_csv/g1/*.csv' --output-dir '.../lafan1_npz'

# 2) 训练（Slurm 单卡）
sbatch --time=01:30:00 --export=ALL,ITERS=100000,NUM_ENVS=4096,HISTORY_LENGTH=10,ACTOR_OBS=proprio repro/train_sm.sbatch
# PPO-Sim 需要已存在的动作路径（打包目录），见报告 §3
sbatch --time=01:30:00 --export=ALL,ITERS=20000,NUM_ENVS=4096,MOTION_GLOB=/mnt/storage1/syk/VLA/SuperMimic/motions/lafan1_packed repro/train_ppo_sim.sbatch

# 3) 评测（两个检查点）
sbatch repro/eval_both.sbatch
```

环境准备、数据来源、完整命令见 [report/复现报告.md](report/复现报告.md)。
