# 01 环境与基准：Dreamer 的 Minecraft 设置

> **结论先行**：本工作使用 **DreamerV3 官方仓库里的 Minecraft Diamond 环境**（真实 Minecraft 1.11.2，经 Malmo/MineRL 驱动），而不是 MineRL 竞赛原始的 5,761,566 维组合动作版本的 `ObtainDiamond`。两者是同一个"采钻石"任务，但 Dreamer 版本把动作压缩成 25 个复合动作、把背包展平成向量观测，并设置 `break_speed=100`，因此同一策略比较才有意义。

**如果只记三件事**：

1. 观测：64×64 RGB 画面 + 391 维背包 + 装备 one-hot + 生命/饥饿/氧气 + 玩家坐标；时间上限 36,000 tick（30 分钟游戏内时间）。
2. 动作：25 个离散复合动作（移动/视角/攻击/放置 + 13 个合成/熔炼/装备步骤）。
3. DreamerV3 官方 20 seeds 的钻石中位步数是 **71.0M**，14/20 成功；到铁镐中位 13.0M。

## 0. 口径

| 内容 | 出处 | 本页用途 |
|-|-|-|
| 环境实现 | `dreamerv3/embodied/envs/minecraft_flat.py`（MIT，Danijar Hafner） | 本仓库 `mcppo/envs/minecraft.py` 的直接来源 |
| DreamerV3 论文与代码 | <https://github.com/danijar/dreamerv3>；Nature 640:647–653 (2025) | 任务设置、步数、动作/观测 |
| 官方 20 seeds 曲线 | `dreamerv3/scores/minecraft_diamond-dreamerv3.json.gz` | 里程碑中位步数（表 1） |
| MineRL 0.4.4 运行时 | 修补版 wheel（含 Malmo + MC 1.11.2 + assets） | 无外网、无 sudo 的本机运行 |

**证据分级**：环境与动作设置是**代码事实**（已在本机创建并 step 验证）；70M 级别的步数是**论文/官方曲线声称**；本仓库自己的 PPO 实验数据在 04 页标为**独立复现**。

## 1. 任务定义

| 项目 | 设置 |
|-|-|
| 游戏 | Minecraft Java Edition **1.11.2**（Malmo 内置服务端） |
| 世界 | 默认世界生成器，`force_reset=True`（每个 episode 重新生成） |
| 观测 | `pov` 64×64×3 uint8；`inventory/*` 391 维；`equipped_items` one-hot（~380 类）；`life/food/air`；`xpos/ypos/zpos` |
| 动作 | 25 个复合动作（见 §2） |
| 单步 | 1 tick = 1/20 s；本工作 PPO 用 `repeat=1`（与 Dreamer 相同），每个 tick 做一次决策 |
| Episode | 本仓库训练截断在 18,000 tick（15 分钟游戏内时间）；Dreamer 原设定为 36,000 tick（30 分钟） |
| 破坏速度 | `break_speed=100`（挖方块几乎瞬间完成，Dreamer 为了降低动作序列长度而设） |
| 奖励 | Dreamer 原版：每种里程碑物品首次获得 +1，生命变化 ×0.01（极稀疏） |

## 2. 25 个复合动作

| 类别 | 动作 |
|-|-|
| 移动 | `forward` `back` `left` `right` `jump`（同时前进） `place_dirt` |
| 视角 | `turn_up/down/left/right`（每步 15°） |
| 交互 | `attack` `noop` |
| 合成 | `craft_planks` `craft_stick` `craft_crafting_table` `craft_wooden_pickaxe` `craft_stone_pickaxe` `craft_iron_pickaxe` `craft_furnace` |
| 放置 | `place_crafting_table` `place_furnace` |
| 装备 | `equip_wooden_pickaxe` `equip_stone_pickaxe` `equip_iron_pickaxe` |
| 熔炼 | `smelt_iron_ingot` |

动作是"复合"的：例如 `craft_planks` = `{craft: "planks"}`，`turn_left` = `{camera: (0,-15)}`。相机是相对角度且被限制在俯仰 ±60°；移动键在 `repeat` 窗口内保持，相机/合成只作用一次（与 DreamerV3 完全一致）。

> 为什么不用 MineRL 原始 `ObtainDiamond-v0`：原始动作空间是 `Dict` 组合（相机连续 + 12 个二值键 + 5 个 300+ 类别的 Enum），组合数 5,761,566。Dreamer 把"合成哪件物品"提升成原子动作，把动作空间压到 24，这是 Dreamer 能在 71M 步内学会钻石的关键工程前提，也是 PPO 可比较的前提。

## 3. 官方步数基准（20 seeds）

| 里程碑 | 中位步数 | 最小步数 |
|-|-|-|
| log（原木） | 220,192 | 26,112 |
| planks（木板） | 226,544 | 26,112 |
| stick（木棍） | 264,624 | 118,560 |
| crafting_table（工作台） | 637,456 | 268,576 |
| wooden_pickaxe（木镐） | 775,968 | 442,688 |
| cobblestone（圆石） | 1,083,536 | 442,688 |
| stone_pickaxe（石镐） | 1,328,512 | 442,688 |
| iron_ore（铁矿石） | 2,912,752 | 1,236,800 |
| furnace（熔炉） | 3,366,672 | 1,236,800 |
| iron_ingot（铁锭） | 6,397,024 | 1,236,800 |
| iron_pickaxe（铁镐） | 13,015,168 | 4,547,936 |
| **diamond（钻石）** | **71,002,192** | **44,655,744** |

14/20 seeds 在 1 亿步内拿到钻石（最晚 81.9M），6/20 未达到 —— 单 seed 失败风险约 30%。这组数字是**论文/官方曲线声称**，本仓库未重跑 Dreamer。

## 4. 本机运行条件（实测）

| 项目 | 值 |
|-|-|
| 运行方式 | 单节点 Slurm；`cpu` 分区单作业 ≤24 CPU、≤2 天；`gpu` 分区单作业 ≤3 小时 |
| 本机 GPU | 3×RTX 5090（32 GB）；PPO 不需要 GPU（CPU 训练足够） |
| 环境 | conda `mcppo`（克隆自 Dreamer 的 `mc` 环境：Python 3.11 + MineRL 0.4.4 + Java 8 + JAX（未用） + PyTorch CPU） |
| 渲染 | Xvfb + Mesa llvmpipe（无 GPU 渲染依赖） |
| 单 env 启动 | 首次 reset 约 43 s（世界生成）；之后正常 step |
| 环境吞吐 | 1 env ≈ 9–11 tick/s；8 env（约 12–16 CPU）≈ 90–110 tick/s（详见 04 页） |

## 5. 与 PPO 的关系

Dreamer 用世界模型在潜空间里"想象"训练，因此能用 71M 真实交互步达到钻石；PPO 是纯 on-policy、无模型方法，样本效率显著更低。本仓库因此做了三件事：

1. **保留 Dreamer 的观测/动作抽象**（去掉 PPO 不需要从零学的动作维度）；
2. **手写稠密奖励**（03 页），把"物品获取量/效率、合成、生命值"变成每步可学的信号；
3. **循环策略 + 6 路并行 + 长 rollout**，把 PPO 的样本效率压到本机可承受的范围。

**未验证**：本仓库没有在 71M 步量级复现 Dreamer；因此"PPO 到钻石需要多少步"只能给外推区间（05 页），不能给实测数字。
