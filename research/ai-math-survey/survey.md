# AI 做数学证明：主流方案、工具需求与模型差距

> 调研日期：2026-11-21。范围：以“从题面到可检验证明”为主线，覆盖通用推理模型、Lean 形式化、
> 智能体搜索、自动形式化、工具增强与人机协作；并以 OpenAI 2026-10 发布的 722 篇数学手稿及
> 其中 10 个公开推理摘要作为案例，做一次 DeepSeek / Kimi 的信息隔离复现实验。
>
> 证据分级：**[论文/官方声称]**、**[独立复现]**、**[未验证]**。本文明确区分这三类。

## 结论先行

1. **主流方案不是单一模型，而是“长思维链模型 + 工具 + 验证”的组合**：通用模型负责提出证明
   思路和写自然语言证明；Lean/Mathlib（必要时 Comparator）负责把可形式化的部分变成机器可检验
   的对象；CAS/数值/SAT/图搜索负责有限验证与反例搜索；智能体框架负责长程规划、子目标分解和
   上下文管理。
2. **“需不需要客制化工具/Skill”取决于目标**：如果只要“探索性证明草稿”，通用 API + Python 就够；
   如果要“可信交付”，必须做 Lean 形式化/证明修复/独立复核，这是明显的客制化工具链。OpenAI 这次
   10 个公开案例 **全部** 有 Lean 文档与 Comparator challenge，但形式化范围普遍小于论文声称
   （例如 017 不含 Flint–Hills、159 不含定量上界、197 不含非 sofic、271 不含 Bloch 律）。
3. **模型差距很大，且“ benchmark 分数”与“研究级证明”不是一回事**：FrontierMath 上头部闭源模型
   约 52%，开放模型 Kimi K2.6 约 39%、DeepSeek-V3.2 约 22%（2026-11 快照）；IMO 2025 上公开
   可测模型最高约 31%，不到铜牌，而厂商未公开模型声称金牌。OpenAI 的内部模型在 722 篇上展现了
   更强的长程研究能力，但官方也承认“并非所有结果都经过 Lean 验证，部分未形式化结果可能有问题”。
4. **本票的盲复现实验**（10 案例 × 2 模型 × 4 轮，题面隔离）用于回答“公开模型能否独立复现”；
   结果与失败模式见 `repro/openai-10/` 与知识库的“复现”页。**不要把厂商 benchmark 数字直接
   当作研究级证明能力**。

## 0. 口径

| 术语 | 本文含义 |
|---|---|
| 自然语言证明 | 用英文/中文写的、可被人读的证明；正确性靠同行评审 |
| 形式化证明 | Lean/Coq/Isabelle 等证明助手接受的证明；正确性靠内核检查 |
| 自动形式化 | 把自然语言题面/证明翻译成形式语言 |
| 证明搜索 | 在形式系统里搜索证明项（可能用 LLM、树搜索、RL） |
| 智能体 | 能规划、调用工具、分解子目标、维护记忆的 LLM 系统 |
| 研究级问题 | 未解决或仅有部分结果的开放问题；不是竞赛题或教材习题 |

**证据分级**：论文/官方声称 = 作者或厂商给出的数字/结论；独立复现 = 本仓库或第三方可追溯地
跑出/检查过；未验证 = 只读了摘要、文档或代码，没有独立跑数值/证明。

## 1. OpenAI 722 篇：事实核查

### 1.1 数量与结构

- 2026-10-06/07，OpenAI 在 GitHub 发布 `openai/math`：初始目录 **722 篇手稿 / 372 个 result
  family**，覆盖数论、几何、理论计算机、数学物理等 17 个领域。[新闻](https://finance.sina.cn/2026-10-07/detail-iniuknsf8239213.d.html)
- 截至 2026-11-21 的 `main` 快照：README 写 **721 篇 / 373 个 family**；`overview.tex` 有
  373 个 `\resultentry`。差异来自撤回、修订与新增。[README](https://github.com/openai/math)
- **10 个 family 公开了“abridged reasoning summary”**（007, 017, 087, 102, 159, 197, 221,
  271, 287, 362）。这是“10 个公开案例”的准确口径；其余手稿 PDF 也公开，但没有公开推理过程。
- 约 **42%** 的 top-line 结果有 Lean 形式化；`history.md` 在 2026-10-07 记的是
  `300 / 719 = ~42%`。[history](https://github.com/openai/math/blob/main/history.md)
- 2026-10-07 已撤回 3 篇（Weil classes 的符号错误），并修订 14 篇；10-26 又新增 2 篇
  （local smoothing、ETH is false）。这说明“722 篇”不是 722 个已独立验证的定理，而是一批
  不同验证阶段的研究手稿。

### 1.2 生产方式

- 官方：绝大多数结果用同一个**未发布内部模型**，平均每个结果约 **3 小时 ChatGPT Pro thinking
  compute**；评估过程中模型被提出约 **4000 个问题**，聚合成 373 个 family。
- 例外：Riemann zeta 的零-free 区域与 CM abelian varieties 的 Hodge 猜想用了不同流程；
  Re(s)>11/12 的写法人做了可读性编辑。
- 官方原话：这些手稿“最好被理解为 raw research data，而不是 polished academic papers”；
  “not all have accompanying Lean formalizations”；“some of the unformalized results could have
  issues”。

### 1.3 10 个公开案例与 Lean 覆盖

| family | 主题 | Lean 文档 | 形式化范围（要点） |
|---|---|---|---|
| 007 | 两点相关 / 修正 Elliott | `lean/docs/007.md` | 定理本身；不含所有定量变体 |
| 017 | \(\mu(\pi)=2\) | `lean/docs/017.md` | 定理本身；**不含** Flint–Hills 推论 |
| 087 | Mahler 猜想 | `lean/docs/087.md` | 对称/一般 Mahler 与 Gromov 宽；**不含**泛函不等式 |
| 102 | Unique Games / BasicSDP | `lean/docs/102.md` | UGC 与若干直接归约；BasicSDP 阈值是推论 |
| 159 | Erdős 倒数和 | `lean/docs/159.md` | 主猜想；**不含**定量 Szemerédi 上界 |
| 197 | Kaplansky 直接有限性 | `lean/docs/197.md` | 特征 2/奇特征反例、行列式反例；**不含**非 sofic |
| 221 | Mézard–Parisi | `lean/docs/221.md` | 主公式 |
| 271 | Heisenberg 自发磁化 | `lean/docs/271.md` | 自发磁化；**不含** Bloch 律与格点修正 |
| 287 | 自由群因子 | `lean/docs/287.md` | 可插值因子同构；**不含** fundamental group 结论 |
| 362 | 相对论 Vlasov–Maxwell | `lean/docs/362.md` | 全局光滑解 |

**读法**：10/10 都有 Lean 文档，但形式化的是“选定的主定理”，不是论文的全部结论。这既说明
OpenAI 对形式化很重视，也说明“论文声称”与“机器验证”之间仍有距离。

## 2. 主流方案地图

### 2.1 通用长思维链模型（natural-language proof）

**做法**：用 o1/o3、GPT-5、DeepSeek-R1/V3/V4、Kimi K1.5/K2/K3、Gemini Deep Think、Claude
等模型，在长上下文里做“读题 → 找思路 → 写证明 → 自审”。Tool use 可选（Python、检索、CAS）。

**优点**：覆盖所有数学领域；能读文献、写自然语言；对竞赛题和“有标准套路”的问题很强；成本
相对形式化低。

**缺点**：正确性没有机器保证；长证明容易在中段出现“看似合理但错误”的步骤；对研究级问题，
模型可能把猜想当定理、循环论证、或引用不存在的定理；需要专家评审。

**代表证据**：
- IMO 2025：MathArena 用统一协议测公开模型，最高 Gemini 2.5 Pro 31.55%，不到铜牌；
  OpenAI/DeepMind 的未公开模型声称金牌（分别由前 IMO 选手/IMO 官方验证）。
  [MathArena](https://matharena.ai/imo/)
- FrontierMath：2026-11 快照，GPT-5.5 Pro 52.4%、Kimi K2.6 38.97%、DeepSeek-V3.2 22.1%。
  [evals.report](https://evals.report/benchmarks/frontiermath)
- 这些数字说明：**公开模型在“研究级、非标准”问题上仍有很大差距**；竞赛题饱和不等于研究能力。

### 2.2 形式化证明（Lean/Coq/Isabelle + RL/LLM）

**做法**：
- 用 Lean 4 + Mathlib 表示定理与证明；模型生成 tactic/term；编译器给反馈；树搜索/RL 探索。
- 代表系统：AlphaProof（Nature 2025，IMO 2024 银牌级，测试时 RL）、DeepSeek-Prover-V2
  （671B，MiniF2F 88.9%，PutnamBench 49/658）、Kimina-Prover、Goedel-Prover、LeanDojo、
  COPRA、Aristotle 等。
- 验证层：Lean 内核、`#print axioms`、`sorry`/`native_decide` 扫描、Comparator（比较
  “题面”与“解答”的独立声明）。

**优点**：正确性可机器检查；可复用 Mathlib；适合“可信交付”。

**缺点**：形式化成本高；Mathlib 覆盖有限；自动形式化仍是瓶颈；研究级定理的形式化可能比
自然语言证明长一个数量级；RL 系统通常绑定特定版本/工具链。

**代表证据**：
- AlphaProof：Nature 2025，AlphaZero 风格 RL + 百万级自动形式化题目；IMO 2024 解出 3/5 非几何题。
  [Nature](https://www.nature.com/articles/s41586-025-09833-y)
- DeepSeek-Prover-V2：MiniF2F-test 88.9%，PutnamBench 49/658；子目标分解 + RL。
  [arXiv](https://arxiv.org/abs/2504.21801)
- PutnamBench：1692 个形式化 / 640 道 Putnam 题；当时神经/符号证明器只能解“handful”。
  [arXiv](https://arxiv.org/abs/2407.11214)
- LeanDojo：检索增强的 Lean 证明器；证明搜索需要 premise retrieval。
  [arXiv](https://arxiv.org/abs/2306.15626)

### 2.3 智能体与搜索（agentic proof search）

**做法**：把“证明”拆成子目标；用规划器/多智能体分工；调用 Lean、Python、CAS、检索；
用 verifier 反复修；维护工作记忆。代表：AlphaGeometry（几何专用）、GPT-f、COPRA、
LeanCopilot、Aristotle、以及各家“deep research”式长程智能体。

**优点**：能处理长程、多步、需要工具的问题；比单次生成更稳；可把“提出—验证—修复”闭环。

**缺点**：编排复杂；错误会级联；成本高；评估难（不同 scaffold 不可比）；容易“看起来在工作”
但没有实质进展。

### 2.4 自动形式化与证明修复

**做法**：自然语言题面 → Lean 命题（autoformalization）；证明草图 → Lean tactic；编译错误 →
修复；`sorry` 定位 → 子目标。代表：Wu et al. 2022、LeanDojo/LeanCopilot、以及 2025-2026 的
agentic autoformalization 工作。

**优点**：把人类可读的数学转成机器可检验对象；是“自然语言 + Lean”的桥梁。

**缺点**：题面翻译错误比证明错误更隐蔽；需要领域专家校对；对非标准记号/定义不友好。

### 2.5 工具增强（CAS/数值/SAT/图搜索）

**做法**：
- 数值实验：找反例、猜公式、验证渐近、检查有限情形（Python/NumPy/SymPy/Sage/Mathematica）。
- 符号计算：恒等式、级数、积分、代数簇。
- SAT/SMT/ILP：有限组合问题、存在性证书。
- 图搜索/枚举：反例、极端构型、覆盖。
- 区间算术/浮点严格化：数值证书。

**优点**：把“有限可验证”的部分从证明里剥离；能快速否证错误猜想；能给证明提供可复现证据。

**缺点**：不能替代一般性证明；数值证据不是证明；工具链/版本/精度会引入新错误；需要把计算
转成可审计的证书。

### 2.6 人机协作与验证流水线

**做法**：AI 出草稿/候选证明 → 专家评审 → Lean 形式化关键部分 → 独立复现/审计 → 版本化。
OpenAI 的 722 篇、DeepMind 的 AlphaProof、以及数学界的“AI 辅助研究”都在往这个方向走。

**关键点**：**“AI 生成”不等于“已验证”**。可信交付需要：明确假设、可追溯证据、独立复核、
以及“哪些部分机器验证/哪些部分只是论文声称”的标注。

## 3. 需不需要客制化工具 / Skill？

### 3.1 结论

- **探索阶段**：不需要很重的客制化。通用模型 + 一个能跑 Python 的沙箱 + 好的题面/评审 rubric
  就能做“能不能想出思路”的实验。
- **可信交付阶段**：需要。最小工具链是：
  1. **Lean 4 + Mathlib + 固定版本/缓存**：把主定理形式化；
  2. **Comparator/独立声明**：避免“证明了一个不同的命题”；
  3. **证明修复/自动形式化**：把自然语言草稿变成 Lean；
  4. **领域工具**：CAS、数值、SAT/SMT、图搜索；
  5. **编排与隔离**：多模型独立窗口、信息隔离、预算/日志/评分；
  6. **审计**：`sorry`/公理/`native_decide` 扫描、`#print axioms`、反 vacuity 检查。

### 3.2 为什么不是“一个万能 Skill”就够

- 数学子领域差异极大：数论/组合/PDE/算子代数需要不同的引理库、计算工具和评审标准。
- Lean 形式化的瓶颈在“题面/定义”而不只是“证明”；同一句话的自然语言歧义会让形式化偏离原意。
- 研究级问题往往需要“提出新定义/新不变量”，这不是检索或模板能覆盖的。
- 因此更合理的形态是：**通用编排 Skill + 领域工具包 + 验证流水线**，而不是一个巨大的 prompt。

### 3.3 本仓库可复用的资产

- `repro/openai-10/`：信息隔离的 API 会话 harness、题面、transcripts、评分 rubric。
- 之前的四色定理 Lean 4 复现/审计（`fourcolor-lean/`，另一个分支）展示了“固定 commit + 构建 +
  公理扫描 + 反 vacuity”的流程，可作为 Lean 验证 Skill 的模板。
- 之前的 \(\pi\) 无理性指数独立窗口（`research/pi-exponent-parallel/`，另一个分支）展示了
  “独立窗口 + 文献隔离 + no-go 论证”的流程，可作为“盲探索”的模板。

## 4. 模型差距

### 4.1 公开 benchmark 的差距

| benchmark | 头部闭源 | 开放/公开模型 | 读法 |
|---|---|---|---|
| FrontierMath（2026-11 快照） | GPT-5.5 Pro 52.4% | Kimi K2.6 38.97%；DeepSeek-V3.2 22.1% | 研究级数学仍有 10–30 个百分点差距 |
| IMO 2025（MathArena 统一协议） | 未公开模型声称金牌 | 公开可测最高 Gemini 2.5 Pro 31.55% | 公开模型不到铜牌；厂商声明需独立验证 |
| PutnamBench | — | DeepSeek-Prover-V2 49/658 | 形式化本科竞赛题仍很难 |
| MiniF2F | — | DeepSeek-Prover-V2 88.9% | 已接近饱和，区分度下降 |

注意：这些数字来自不同时间、不同 scaffold、不同评测协议，不能直接横比；它们只说明“量级”。

### 4.2 研究级证明的差距

- OpenAI 的内部模型在 722 篇上展示了“批量产出研究级手稿”的能力；但官方承认验证不完整，
  且有撤回/修订。公开模型在同样的题面上能否复现，是本票实验要回答的问题。
- **差距不只在“知识”**，更在：长程规划、提出有用引理、识别死路、 sustained self-audit、
  以及“知道何时停止/承认失败”。这些能力很难用单次 benchmark 衡量。
- **工具会放大差距**：会调用 Lean/CAS/检索的模型，比只会写自然语言的模型更容易产生可检验结果；
  但工具不能替代想法。

## 5. 本票实验

见 `repro/openai-10/README.md` 与知识库的“复现”页。简要设计：

- 10 个公开案例的题面（来自 `reasoning_traces/*` 的 `Original prompt`，去掉解法提示）。
- DeepSeek-V4-Pro 与 Kimi-K3，各自独立会话，4 轮（独立求解 → 敌意审稿 → 工具 → 终稿）。
- 沙箱 Python（bubblewrap、无网络、60 s、numpy/sympy）；Slurm 管理。
- 评分：官方结论 + 候选终稿，双评分模型 + 人工复核；0–5 rubric。
- 结果：见 `repro/openai-10/results/` 与 `kb/ai-math-survey/`。

## 6. 建议

1. **把“AI 数学”拆成两条流水线**：探索流水线（通用模型 + 工具 + 隔离）与验证流水线
   （Lean/Comparator + 专家 + 审计）。不要用一条流水线同时追求“快”和“可信”。
2. **优先做领域 Skill**：数论（L-函数/解析数论）、组合（极值/概率方法）、PDE（能量估计）、
   算子代数（von Neumann 代数）。每个 Skill 包含：常用定义/引理、已知反例、数值检查脚本、
   Lean 模板、评审 rubric。
3. **建立“题面—草稿—形式化—审计”的证据链**：每个结论标注证据级别；不把论文声称当独立复现。
4. **对研究级问题使用多窗口隔离**：不同模型/不同提示独立探索，最后交叉评审；避免互相污染。
5. **把计算放进 Slurm/沙箱**：可追踪、可停止、可复现；不要让模型直接跑无约束代码。
6. **评估要报告失败模式**：不仅是“解出/没解出”，还要记录“错在哪、是方向错还是技术缺口、
   是否把猜想当定理”。这对改进模型/工具比单一分数更有用。

## 参考资料

- OpenAI, `openai/math` repository: <https://github.com/openai/math>（README、overview.tex、CONTENTS.md、history.md、lean/docs/）
- AlphaProof, *Olympiad-level formal mathematical reasoning with reinforcement learning*, Nature 2025: <https://www.nature.com/articles/s41586-025-09833-y>
- DeepSeek-Prover-V2: <https://arxiv.org/abs/2504.21801>
- PutnamBench: <https://arxiv.org/abs/2407.11214>
- LeanDojo: <https://arxiv.org/abs/2306.15626>
- Autoformalization with Large Language Models: <https://arxiv.org/abs/2205.12615>
- FrontierMath: <https://arxiv.org/abs/2411.04872>；leaderboard snapshot: <https://evals.report/benchmarks/frontiermath>
- MathArena IMO 2025: <https://matharena.ai/imo/>
- Kimi K2 Thinking model card: <https://huggingface.co/moonshotai/Kimi-K2-Thinking>
- LLMs in Automated Theorem Proving survey: <https://zenodo.org/records/19186905>
- 中文报道：<https://finance.sina.cn/2026-10-07/detail-iniuknsf8239213.d.html>；<https://eu.36kr.com/en/p/4017976429907840>
