This is by an automated agent, not a human.

## 背景与更正

用户原消息写「Jamba（不是 ZEV）」，随后更正为 **JEV**（TypeSafe AI 的 System One 决策模型）。本 PR 按 JEV 执行；`kb/jev-wm/08.md` 附录 A 说明若确实指 Jamba，结论会怎样不同。

JEV 本体闭源、无 API key，因此实验使用开放 JEV 风格代理：Tacit-4B（AnyJev 自蒸馏）与 Qwen3-4B-Instruct-2507 typed readout（AnyJev `Decider`）。

## 做了什么

- **文献**：JEV 架构/价格/边界；KIIS 2026（JEV 当世界模型，TextWorld 93.4%）；REFLEX（JEV 选择性控制）；JEV-as-Q（RL 参考策略）；Good Judgment（负结果）；JevVLA（8 万参数动作评价头）；AnyJev/Kev；经典世界模型与动作蒸馏。42 条参考、44 个一手链接：`research/lit_notes.md`。
- **Exp1（世界模型）**：自建确定性文本网格世界（两房间/门/箱子/钥匙/食物，9 动作，7 状态变量 + executes）。24 个未见词表测试世界、500 条测试转移；把下一状态拆成 8 个 typed MCQ，测一步/四步精度、按动作类型拆分、延迟与成本。
- **Exp2（动作生成 + 蒸馏）**：三种教师（直接 choice、逐动作 score、世界模型一步前瞻）；48 个测试世界、6 步闭环；把教师标签蒸进约 1M 参数 transformer 学生；对照 teacher/student/oracle/random 与成本。
- **交付**：DOCX 知识库（00–10，含图表）+ Feishu 阅读副本 + 可复现代码/Slurm/结果。

## 主要结果（本实验独立复现，非论文数字）

### Exp1：JEV 风格模型的世界模型精度（500 条转移）

| 模型 | 一步精确匹配 | 执行判断 | observe | simple | derive | 4 步精确匹配 |
|-|-|-|-|-|-|-|
| persistence | 0.745 | 0.620 | – | – | – | 0.200 |
| Tacit-4B | 0.754 | 0.696 | 0.981 | 0.781 | 0.709 | 0.180 |
| Qwen3-4B typed | **0.790** | 0.703 | 1.000 | 0.671 | 0.791 | 0.150 |
| Qwen3-4B generative | 0.540 | 0.740 | 0.459 | 0.620 | 0.526 | 0.167 |

读状态强、推后果弱、4 步开环降到 0.15–0.18。Tacit-4B 只比 persistence 高 0.009，因为 318/500 是失败动作；在真正执行的 182 条上 Tacit 0.566（persistence 0）。

### Exp2：动作生成 + 蒸馏（48 世界，6 步）

| policy | success | JEV decisions/episode | student params |
|-|-|-|-|
| random | 0.625 | 0 | – |
| oracle | 1.000 | 0 | – |
| teacher: choice | 0.833 | 4.5 | – |
| teacher: score | **1.000** | 9.8 | – |
| teacher: WM lookahead | 0.833 | 55.8 | – |
| student from score labels | **0.958** | 0 | 1M |
| student from oracle labels | 1.000 | 0 | 1M |
| student from WM labels | 0.833 | 0 | 1M |

逐动作打分教师 100%，蒸馏学生 95.8%、部署 0 次 JEV 调用；世界模型前瞻贵 5.7 倍却没有更准。**在这个任务上，JEV 更像一个便宜的动作打分器/教师，而不是可靠的完整模拟器。**

## 验证

```bash
cd jev-wm
PYTHONPATH=. pytest -q tests          # 6 passed
PYTHONPATH=. python scripts/exp1.py --backend persistence --limit 200
PYTHONPATH=. python scripts/analyze.py
```

Slurm 作业与日志在 `/mnt/storage1/syk/jev-wm/logs/`；结果 JSON 在 `jev-wm/results/`。

## 交付

- Word：`deliverables/JEV世界模型知识库_2026-11.docx`（40 页，含 5 张图）
- 飞书知识库阅读副本链接在交付消息中（不写入仓库）。
- 详细限制：`kb/jev-wm/08.md`

## 限制

- 开放代理不是托管 JEV；单 seed、单世界族；确定性/完全可观测；Exp2 用同词表 `testsame`。
- 动作集 9 类且 mask 到可执行集合；没有像素/连续控制。
- 成本是 API 等价估算，不含重试/网络/工程。
