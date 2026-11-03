# 00 总览：JEV 作为世界模型与动作来源（2026-11）

> 本知识库回答一个很具体的问题：**能不能把 JEV（TypeSafe 的决策模型）当世界模型，让它产出「最合理的动作」去训练一个更便宜的学习器？** 结论是「可以，但只在窄条件下，而且动作生成与蒸馏要分开算账」。本页给索引、一句话结论与证据分级；细节在 01–07。

## 范围

- **模型**：JEV（托管、闭源）及其开源代理 Tacit / AnyJev / Kev；不包含 Jamba（用户原消息写错，附录 A 澄清）。
- **任务**：文本 agent、可枚举的离散状态与动作（文本网格世界）；不覆盖像素世界模型、连续控制、机器人。
- **实验**：两个预实验（Exp1 世界模型精度与成本；Exp2 动作生成与蒸馏），在本机 RTX 5090 + Slurm 上跑，代码与原始结果可复现。
- **时间**：2026-11-03；文献截止同日。

## 资料索引

| 页面 | 内容 |
|-|-|
| [01 JEV 是什么](https://yide4vukqxv.feishu.cn/docx/ItzIdNP79osZ0mxUuTccgvxtn3F) | 决策模型 vs 生成式 LLM、接口、价格、能力边界 |
| [02 相关研究](https://yide4vukqxv.feishu.cn/docx/HsHtdQk07oPAHDxUAGucKUChngb) | KIIS 2026、REFLEX、JEV-as-Q、Good Judgment、JevVLA、经典世界模型 |
| [03 机制与边界](https://yide4vukqxv.feishu.cn/docx/O69sdBm6poLKljxMIWIcYFT2n0e) | 因子化转移、三种动作生成、成本公式、误差来源、选型矩阵 |
| 04 预实验一 | JEV 世界模型：一步/四步精度、动作类型、延迟与成本（待发布） |
| 05 预实验二 | JEV 产动作 + 蒸馏小学习器：闭环成功率、成本对照（待发布） |
| [06 结论与建议](https://yide4vukqxv.feishu.cn/docx/TdWtd5MTbopiVBxMwaocTg2xn4b) | 能不能用、怎么用、下一步 |
| [07 复现指南](https://yide4vukqxv.feishu.cn/docx/OKubd3eLeoPNyExsdq8cXkoYn0b) | 代码、数据、Slurm、结果文件 |
| [08 附录](https://yide4vukqxv.feishu.cn/docx/E9L6dUbegoK5gJx3BfEcFOlDnue) | A. Jamba vs JEV；B. 限制与未验证 |
| [09 参考文献](https://yide4vukqxv.feishu.cn/docx/L6jidPl3moqkuTx9kD0cD4mDn7e) | 42 条一手来源与链接 |

## 一句话结论

1. **JEV 可以当「文本、离散、可枚举」状态的世界模型**：把下一状态拆成若干选择题，冻结模型即可做一步/多步预测；KIIS 2026 在 TextWorld 上报告 93.4% 闭环成功率、4 步精确匹配 77.9%。
2. **但它不是生成式模拟器**：不能预测像素/连续量/新物体；选项必须预先枚举；需要推导的判断（尤其「命令是否执行」）是主要误差源。
3. **JEV 可以直接输出动作，但要外接选择逻辑**：直接 choice 最便宜；逐动作打分更稳；世界模型前瞻最贵、最像「用世界模型选动作」。
4. **蒸馏是成本最低的落地方式**：用 JEV 给一批状态打动作标签，训练一个小策略；教师只贵一次，学生常驻。前提是标签覆盖学生实际会访问的状态（否则会学到窄策略）。
5. **不要把它当 Dreamer/Genie 的替代品**：像素世界模型、连续控制、长时程组合推理仍应使用 Dreamer/TD-MPC2/LeWAM/V-JEPA 等。
6. **成本要按「问题数 × 动作数」算**：世界模型前瞻每步 50–100 次决策，$0.01/回合量级；直接 choice 便宜两个数量级。JEV 的「便宜」不是无条件的。

> 上述 1、3、4 的数字来自本报告实验（独立复现，见 04/05）；KIIS 的数字是论文声称（单任务、单 LLM、单 seed，尚未独立复现）。两者在正文中分开标注。

## 证据分级

| 等级 | 含义 | 本知识库中的例子 |
|-|-|-|
| **官方声明** | 厂商博客/文档 | JEV 价格、延迟、选项上限 |
| **论文声称** | 作者自报，未必复现 | KIIS 的 93.4%、REFLEX 的 72.7% |
| **独立复现** | 本仓库代码跑出、可追溯 | Exp1/Exp2 的数字（04/05） |
| **未验证** | 只读摘要/代码/二手材料 | JEV 的 MoE 主干、Genie 3 规格 |

## 维护约定

- 更新实验数字时，先改 `jev-wm/results/*.json`，再重跑 `jev-wm/scripts/analyze.py` 生成图表，最后同步 04/05 与 00 的结论。
- 论文声称与独立复现不得混写；新增文献先进 `research/lit_notes.md`，再进 02。
- 本知识库的阅读副本在 Feishu；Word 在 `deliverables/`；代码与原始日志在仓库 `jev-wm/` 与数据盘 `/mnt/storage1/syk/jev-wm/`。
