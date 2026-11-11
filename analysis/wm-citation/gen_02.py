import json, os
ROOT=os.path.dirname(os.path.abspath(__file__))
REPO=os.path.abspath(os.path.join(ROOT,'../..'))
table=open(os.path.join(ROOT,'data/table.md')).read()
app=open(os.path.join(ROOT,'data/table_appendix.md')).read()
head='''# 02 主表：46 篇高被引论文的动机 / 贡献 / 方法 / 硬件

> **结论先行**：主表按 6 种方法类型分组，每组内按 OpenAlex 引用数降序。最集中的两组是“视频生成式世界模型”（20 篇）与“机器人视频-动作/VLA”（12 篇），它们合计贡献了主表约 74% 的引用量。

**如果只记三件事**

1. **视频生成式世界模型是最大公约数**：从 Cosmos 的基础模型，到 GameNGen 的实时游戏引擎，再到 Matrix-Game / RELIC / LingBot-World 的“实时 + 长时记忆”，方法差异主要在“怎么把生成做快、做长、做可控”。
2. **机器人侧的关键词是“视频预训练 + 动作条件”**：DreamZero、Cosmos Policy、GR-2、DreamDojo、mimic-video 都在回答同一个问题——如何把互联网视频里的物理先验接到低层动作上。
3. **纯潜空间想象训练的新工作最少，但被引用得最“深”**：Dreamer 4 是 Dreamer 系唯一的主表新作；TD-MPC2 放在附录，但几乎所有连续控制论文都拿它当基线。

## 0. 口径与读表说明

- **引用数**：OpenAlex `cited_by_count`，抓取日 2026-11-11；仅用于排序，跨库不可精确比较（见 01）。
- **谱系缩写**：`D`=Dreamer（V1）、`D2`=DreamerV2、`D3`=DreamerV3、`J`=I-JEPA、`JV`=V-JEPA、`JV2`=V-JEPA 2、`R`=World Models / Recurrent World Models、`P`=PlaNet。该列表示“这篇论文引用了哪些种子”，来自 OpenAlex 的施引记录，不等价于“主要受谁启发”。
- **方法类型**是人工判读，判别标准见 03；**创新点**标签见 04。
- **硬件**只写论文明确报告的内容；“未报告”表示论文没有给出训练硬件，不代表没有用 GPU。
- 表格较宽，Word/PDF 阅读时建议横屏或缩小字号；完整可筛选版本见 `deliverables/世界模型引用调研_2026-11.xlsx`（CSV 同目录）。

---

## 1. 主表（按方法类型分组）
'''
mid='''

---

## 2. 附录 A：2023 年上半年奠基工作（窗口外）

这 4 篇发表于 2023-01 至 2023-10，早于严格三年窗口，但它们是理解近三年世界模型演化的必要背景：TD-MPC2 是连续控制世界模型的强基线，UniSim 是“生成式模拟器”路线的早期代表，UniPi 把视频生成当规划器，Structured World Models 则把人类视频变成结构化世界模型。
'''
out=head+table+mid+app+'''

---

## 3. 读表后的五个观察

1. **“世界模型”的重心从“学动力学”转向“生成未来”**：主表 46 篇中，34 篇的核心是生成视频/图像未来（视频生成式 20 + 机器人视频-动作 12 + 3D 占用 1 + 部分 JEPA 工作），纯潜空间动力学只剩 Dreamer 4。
2. **“可控性”成为高频创新点**：15 篇明确以动作/轨迹/文本条件控制为核心创新，驾驶与机器人是主要落点。
3. **“长时程 + 实时”是一对矛盾**：7 篇做长时记忆、9 篇做实时推理，能同时做到的只有 RELIC、Matrix-Game 3.0、LingBot-World 等少数工作。
4. **硬件报告极不均衡**：11 篇未报告训练硬件；报告了的工作从单卡数小时到万卡数月，跨四个数量级。
5. **JEPA 的影响被低估**：主表只有 6 篇直接以 JEPA 为方法类型，但 DINO-WM、VLA-JEPA、FLARE、LeWM 被大量后续工作当作“表征/世界模型骨架”引用；JEPA 更像“方法论基础设施”，而非一个独立应用方向。
'''
open(os.path.join(REPO,'kb/wm-citation/02.md'),'w').write(out)
print("wrote 02.md", len(out))
