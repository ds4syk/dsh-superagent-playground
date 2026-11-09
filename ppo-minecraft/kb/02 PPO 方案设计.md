# 02 PPO 方案设计

> **结论先行**：本方案不是"把 Dreamer 的 world model 换成 PPO"那么简单，而是围绕 PPO 的弱点做了四个工程决定：**沿用 Dreamer 的 25 动作抽象**、**用 4 帧堆叠 + LSTM 处理 POMDP**、**用手写稠密奖励替代 Dreamer 的 +1 稀疏里程碑**、**用 6 路并行环境 + 分块 BPTT 把 on-policy 采样做满**。目标是在本机 CPU 上每小时采集约 14.5 万样本，使 100 万样本级实验可在数小时内跑完。

**如果只记三件事**：

1. 网络：CNN（64×64×12）→ 256 + 背包/生命 MLP → 128，拼接后进 LSTM(256)，输出 25 动作 logits 与 value。
2. PPO：γ=0.999、GAE λ=0.95、clip=0.2、4 epoch、minibatch=64 条 32 步序列；奖励整体除以 5 做尺度归一。
3. 采样：8 个独立 Minecraft 进程；父进程做帧堆叠和 LSTM 隐状态管理，rollout 内每 32 步记录一次隐状态，更新时按 chunk 重放。

## 0. 口径

| 项目 | 出处/设置 |
|-|-|
| PPO 算法 | Schulman et al., *Proximal Policy Optimization Algorithms*, arXiv:1707.06347 |
| 循环 PPO | 参考 MineRL 竞赛常用做法（LSTM + truncated BPTT），非官方实现 |
| 环境 | 见 01 页（DreamerV3 Minecraft Diamond） |
| 超参 | 本仓库 `mcppo/config.py`、`configs/diamond_ppo.json` |

**证据分级**：算法公式是**论文声称**；网络与超参是**本仓库实现**；实际吞吐与学习曲线是 04 页的**独立复现**。

## 1. 为什么 PPO 在 MC 上难

| 难点 | 具体表现 | 本方案对策 |
|-|-|-|
| 部分可观测 | 单帧看不到背后的树/矿洞；导航需要记忆 | 4 帧堆叠 + LSTM(256) |
| 稀疏奖励 | Dreamer 原版只在 12 个里程碑各给 +1，随机策略几乎学不到 | 手写稠密奖励（03 页），把"拿到第一个原木"拆成形状奖励 + 里程碑 + 探索 |
| 动作语义 | 原始 5.7M 组合动作无法直接 PPO | 沿用 Dreamer 的 25 复合动作 |
| on-policy 样本贵 | 每个样本只用一次；Minecraft 环境每秒只有 ~10 tick | 6 进程并行 + `repeat=1` + 大 rollout |
| 长 episode | 30 分钟游戏时间，回报折扣要覆盖数千步 | γ=0.999、GAE、reward scale |

## 2. 网络结构

```
image (4×3=12, 64,64) ──► Conv8x8/4 (32) ──► Conv4x4/2 (64) ──► Conv3x3/1 (64)
                              ──► flatten 1024 ──► FC 256 ─┐
inventory (391) ┐                                          │
health/hunger/breath (3) ┼─► FC 128 ─► FC 128 ─────────────┼─► concat 384
equipped (~380) ┘                                          │
                                                           ▼
                                                    LSTM(384→256)
                                                      │        │
                                                  FC 24      FC 1
                                                  (logits)   (value)
```

- 参数量约 **1.0M**（含 LSTM），在 CPU 上单次更新（2048 条 32 步序列 × 4 epoch）约 1–2 s，相对于采集时间可忽略。
- 背包向量先做 `log1p(clip(x,0,64))/log(65)` 归一化；生命/饥饿/氧气分别除以 20/20/300。
- 动作头是 24 类 `Categorical`；没有动作掩码——非法合成在环境里只是无操作，不产生奖励。

## 3. PPO 目标与超参

标准 clipped surrogate：

```
L_pi = -E[ min( r_t A_t, clip(r_t, 1-ε, 1+ε) A_t ) ]
L_v  = 0.5 * E[ (V(s_t) - R_t)^2 ]
L    = L_pi + c_v L_v - c_e H[pi]
```

| 超参 | 值 | 说明 |
|-|-|-|
| γ | 0.999 | 30 分钟 episode，等效 horizon ~1000 步 |
| GAE λ | 0.95 | |
| clip ε | 0.2 | |
| epochs | 4 | |
| rollout | 1024 步 × 6 envs | 每次更新 6,144 个样本 |
| chunk | 32 | LSTM 截断 BPTT 长度 |
| minibatch | 64 chunks | 即 2048 步 |
| lr | 3e-4 → 1e-4（线性退火） | |
| entropy | 0.01 → 0.003 | 先探索后收敛 |
| value coef | 0.5 | |
| grad clip | 0.5 | |
| advantage | 全局标准化 | |
| reward scale | 5.0 | 把里程碑（~1）与钻石（~40）压到 O(1)–O(10) |

**循环策略的更新细节**：rollout 时按 `t % 32 == 0` 保存 LSTM 的 `(h,c)`；更新时每条 32 步序列从保存的隐状态开始重放，梯度只穿过这 32 步。这样既有记忆，又不会把 1024 步的 BPTT 图撑爆内存。

## 4. 采样系统

| 组件 | 实现 |
|-|-|
| 并行 | `mcppo/envs/vec.py`：`multiprocessing` spawn 8 个 worker，每 worker 一个 Malmo/Minecraft 实例 |
| 通信 | Pipe 传 dict（12 KB 图像 + 1.5 KB 背包 + 标量），父进程转 tensor |
| 帧堆叠 | 父进程每 env 一个 `deque(maxlen=4)`；episode 开始时复制首帧 4 次 |
| 隐状态 | 父进程维护 `(h,c)`；env done 时该 env 的隐状态清零 |
| 动作重复 | 每策略步发 1 个 tick（与 Dreamer 相同） |
| Episode | 18,000 策略步后截断；环境内部 done 时 worker 自动 reset 并把 episode 统计带回 |
| 日志 | `metrics.jsonl`（每 update）+ `episodes.jsonl`（每 episode）+ TensorBoard + `curves.png/pdf` |

## 5. 与 Dreamer 的对照

| 维度 | DreamerV3 | 本方案 |
|-|-|-|
| 学习信号 | RSSM 世界模型 + imagination actor-critic | 纯 on-policy PPO |
| 观测 | 64×64×3 + inventory | 相同（外加 4 帧堆叠） |
| 动作 | 25 复合动作 | 相同 |
| 奖励 | 12 个 +1 里程碑 + 0.01×health | 手写稠密：形状 + 里程碑 + 速度 + 生存 + 探索 |
| 样本效率 | 71M 步中位到钻石 | 未验证；预期显著更差（见 05 页） |
| 计算 | 单 A100 约 9 天 / 1 亿步 | CPU 6 envs，约 14–16 万样本/小时 |

## 6. 已知风险

1. **PPO 可能只学会"到处走"**：形状奖励的探索项（上限 +4）与时间惩罚会让策略偏好移动；里程碑奖励需要"看到树 → 对准 → 攻击"的完整链条。缓解：里程碑权重（0.5–1.5）大于探索项单步收益；训练曲线按奖励分量拆分监控。
2. **长 episode 的信用分配**：30 分钟、18,000 步，钻石奖励在末端。缓解：稠密里程碑把长链拆成 12 段；γ=0.999 保证 ~1000 步的有效 horizon。
3. **动作抽象限制**：25 个动作里没有"连续视角"，转身粒度 15°，瞄准树/矿可能不够精细。Dreamer 用同样动作到达钻石，说明可行但需要更多步。
4. **无示范**：MineRL 竞赛的强基线普遍用人类示范做 BC 预训练。本仓库保留了这个接口（`evaluate.py` 可从任意 checkpoint 继续），但本版实验未使用示范。
