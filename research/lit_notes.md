# JEV 作为世界模型与动作生成器：文献笔记

> 核验日期 2026-11-03。只记录能从一手来源核实的内容；不能核实的明确标注。

## 0. 口径与证据分级

- **JEV**：TypeSafe AI 的托管闭源决策模型，无权重、无微调。**Kev** 是开放权重复刻（0.8B/4B/9B/27B）；**AnyJev** 把任意开放 LLM 变成 Jev 式单次前向决策器；**Tacit-1.7B** 用 AnyJev 自蒸馏。三者均非 JEV 本体。
- 证据分四档：**官方声明**、**论文声称**、**独立测量/复现**、**未验证**。
- 纠正一处常见引用错误：IRIS 的 arXiv 号不是 `2209.03936`（那是一篇房价论文），而是 **`2209.00588`**。

## 1. JEV 是什么

### 1.1 发布与定位

2026-09-15，TypeSafe AI（创始人 Diogo Almeida，官方文档称其为 RLHF 共同发明人）发布首个「System One Model」JEV[1][5]。定位是「frontier-intelligence function call：unstructured state in, typed probabilistic decisions out」，明确放弃字符串生成，宣称因此不会产生类型错误/幻觉、比 LLM 快且便宜两个数量级。名字取自 William Stanley Jevons[1]。

### 1.2 架构

**官方可确认**：JEV 一次读入 state，所有 question 并行对同一 state 求值，输出并行概率而非逐 token 生成；同一请求可混用三类问题，问题间无依赖。上下文每请求 64k、每问题 32k，开长头后均 262k[1][2][6]。

**独立逆向**（最系统的公开重构）：Archer Hume 2026-09-17 用约 1 万次 API 探测（192 次延迟、445 次 tokenizer 指纹、311 次 token 计数、181 次选项位置）重构[3]：

1. **因果 Transformer（推断）**：MMLU-Pro 84.6% 要求前沿规模预训练，该规模模型几乎都是 causal decoder；黑盒无法区分 causal 与 bidirectional。
2. **共享 state + 隔离分支**：token 计数严格可加（1 个 Noul 268、2 个 276），符合「公共前缀 + 后缀」；秘密放兄弟问题时另一问题读到概率 0.00，放 state 则 0.90–0.92；≤100 问题延迟几乎不变、之后缓升，符合「state 算一次、问题批量打包」；最自然的实现是 prefix KV cache + 各自 causal suffix。
3. **直接概率读出**：末端隐向量经线性头 + softmax，而非生成「0.91」；`output_tokens` 只是计费字段（Noul = 4 + 15×答案数 + 问题名长度），不反映解码步数；200 选项与 2 选项延迟基本相同。255 上限与固定 256 槽位头一致，但由请求校验强制。
4. **选项整体打分**：加一个无关选项会改变已有选项 odds（log-odds +0.38→+0.11，10 区组全降），排除「独立打分后统一 softmax」；final-position head 与 pointer scorer 未被区分。
5. **稀疏 MoE（最不确定）**：30k tokens ~160ms；稠密 70B 在 8×H100 约 1s，~10B 激活的 MoE 正好匹配，且 prefill-only 下 MoE 服务代价大多消失。作者明说这是推断。

> **「共享 state 编码 + 并行问题分支 + 直接概率读出」有官方文档与独立探测双重支持；「因果 Transformer」「稀疏 MoE」是证据强度递弱的推断，MoE 未被证实。**

### 1.3 训练：RLCD

官方称训练方法为 **RLCD（Reinforcement Learning for Calibrated Decisions）**，是继 RLHF、RLVR 之后的第三条 post-training 路径，优化「认识论上诚实的概率」而非人类偏好或可验证奖励[1][5]。校准是**分组层面**性质（预测 0.8 的那批里约 80% 为真），不保证单条正确；JEV 不生成解释。损失、数据、更新范围未公开；独立分析只能指出 log loss/Brier 是自然选择，不能证明实际使用[3]。JEV 不针对客户微调，所有账户共用同一套权重，领域适配只靠 state 与 instructions/criteria[6]。

### 1.4 API

`POST https://api.typesafe.ai/v1/systemone`，Bearer 鉴权。请求体为 `model` + `state`（string/JSON object/array）+ `questions`（自定义 key，值为 `{type: noul|choice|score|box, instructions, criteria?, options?}`）；响应为同 key 的 `answers`（概率、选项分布或期望分）+ `usage`[6]。`confidence` 是派生字段（Choice 为 `(p_max−1/K)/(1−1/K)`），不是另一个学习出的正确率估计[3]。state 与问题分离，便于「同一份证据、多个判断」[4]。

### 1.5 成本与延迟（厂商口径）

- 输入 **$0.042/MTok**（$42/B），**输出免费**；对比 LLM 输入 $0.20–$10/MTok、输出约 5 倍[1]。端到端 **70–500ms**，官方称快 **40×–200×**；首页宣传 **193.6× 更快、444.6× 更便宜**，官方注明偏高[1]。
- 可验证性：官方承认「无法证明没有补贴」[1]。第三方用 346,009 次请求跑 37 个数据集，总花费 **不到 10 美元**[12]。
- 独立延迟：360 tokens 中位 57.5ms，30k tokens 218ms，1,500 个问题 610ms（服务端头，含共享开销，非硬件基准）[3]。

### 1.6 限制

1. **闭源托管**：无权重、不能本地部署或微调；`jev-latest` 别名会漂移，复现需固定版本[6]。
2. **选项必须预枚举**：输出空间在请求里写死；JevSpawn 指出这限制自主任务求解；上限 255 个选项[1][16]。
3. **无自由生成/推理链/解释**：不能写代码、不能说明理由[1][2]；黑箱性也使偏差难以审计[42]。
4. **英文为主**：CJK 可用但准确率较低，官方要求自行测试[6]。
5. **能力锯齿**：字面理解、数学/计数、日期、多跳、无关细节、对抗内容、矛盾指令、选项顺序均会致错；官方称「不是计算器」[7]。
6. **校准是分组性质**：单条概率不等于单条正确率；agent 控制任务中已观察到明显失准（§2.3）。

## 2. 直接相关工作

### 2.1 KIIS 2026 Fall：把 JEV 当 LLM agent 的世界模型（核心）

**出处**：Yong Hwi Song、Keon Myung Lee（忠北大学），《A World Model for LLM Agents: Comparing a Decision Model with General-Purpose LLMs》，韩国智能系统学会（KIIS）2026 秋季大会，oral，论文 00027，2026-11-26–28 金浦大学；两页学会模板，camera-ready 2026-10-23；代码/日志/稿件在 GitHub `thddydgnl/jev-world-model`[8][9]。截至 2026-11-03 会议未开，未见 proceedings/DOI，**非期刊论文，无独立复现**。

**方法**：TextWorld 双房间（开箱取钥匙→开锁→取食物→进远房，最短 6 步）。策略固定 Qwen3-4B，每步提出至多 16 条长度 2 的候选；规划器用世界模型想象前缀终态，按 `J = conj + 0.25·progress − 0.1·E[#invalid] − 0.01·h + 0.05·prior` 打分，执行最优前缀首动作后重规划（receding horizon，beam 4）。**世界模型只答「(state, 单命令) → 下一 state」**：决策式每动作问 7 个选择题（是否执行 + 6 个状态变量），取最大概率选项拼状态；「是否执行」单独请求（混问会从 0.285 漂到 0.576）。JEV 用 `jev-1.13.0` **冻结、未训练**，只走概率接口；对照为零样本 LLM（决策式 B0/生成式 D0）、LoRA 微调（B/D，6,000 条引擎转移）、policy-only C、失败记忆 C_fm、oracle。24 世界 × 4 起始 × 3 seed = 288 episode/agent，world-clustered bootstrap 95% CI[8][9]。

**数字（作者稿件表 1、图 2）**：

| Agent | 成功率 % [95% CI] | 无效动作 % | 世界模型成本/episode |
|---|---|---|---|
| C（仅策略） | 17.0 [7.3, 29.2] | 87.8 | – |
| C_fm（+失败记忆） | 79.5 [70.5, 87.5] | 35.0 | – |
| **A（JEV，冻结）** | **93.4 [85.1, 98.6]** | 15.4 | 129 次请求，$0.010 |
| B0（LLM，决策式） | 83.0 [73.3, 91.3] | 48.6 | 49 GPU-s |
| D0（LLM，生成式） | 59.0 [48.6, 69.1] | 36.3 | 115 GPU-s |
| B（LLM+LoRA，决策式） | 92.7 [85.8, 97.9] | 13.2 | 42 GPU-s |
| D（LLM+LoRA，生成式） | 92.7 [85.8, 97.9] | 13.2 | 73 GPU-s |
| Oracle（引擎真值） | 92.7 [85.8, 97.9] | 12.8 | – |

关键配对差：A − C_fm **+13.9** [2.1, 24.3]；B0 − D0 **+24.0** [11.5, 35.8]；A − oracle **+0.7** [−1.7, 3.1]（不可区分）；A − B0 **+10.4** [−1.7, 21.5]（不可区分）。4 步 rollout 精确状态匹配（800 条）：JEV 77.9%，B 98.5%，D 96.8%，D0 6.2%，persistence 15.6%；新结构（侧房、诱饵钥匙、第二容器）上 JEV 85.5%，B 71.5%，D 66.5%，JEV 降幅比二者分别小 **25.6/26.7 个百分点**。JEV 的错误几乎全在「命令是否执行」（单步 7.0%），该判断正确时状态错误 0.0%。D0 主因是格式：26.0% 生成状态无法解析，被规则判为「执行失败」，导致 agent 回避取食物。总成本：JEV $2.77（闭环）+ $2.42（rollout）；GPU 约 53 小时（双卡 A5000）[8][9]。

**作者自列 caveat**：① 单任务结构、单 LLM、单 seed、单解析规则；② 评测世界仅「名字新」，结构同训练，69.5% 评测转移在训练集中出现过；新结构只是 rollout，非闭环；③ parsed 答案上 D0 不比 B0 差，决策式优势是**格式稳健性**，大小依赖解析规则；④ B/D 在 288 episode 上与 oracle 完全一致，天花板效应下 JEV 与微调 LLM 不可区分；⑤ oracle 是诊断参照，不是上限[8]。

**与本课题的关系**：首次把决策模型放进 agent 世界模型位置，给出成功率、k 步预测、成本三元组；但没把 JEV 当动作生成器，也没蒸馏到更小学习器——策略/规划器都是 Qwen3-4B，JEV 只在推理时调用。

### 2.2 REFLEX：JEV 作为选择性控制/动作决策层

arXiv 2609.26532（2026-09-22，Tiantong Wu）。用 JEV 做快速 typed decision layer，置信度低或需生成时才调强 LLM。冻结 100 任务基准上成功率 95%，比 strong-only agent 少 **72.7%** 强模型调用，三种 fallback 家族下均成立；但 BFCL 与 τ-style 外部评测上相对「便宜的生成式级联」优势有限；可靠性取决于动作集大小与授权边界附近的近有效替代项[13]。这是「JEV 选动作」最直接的证据之一，但它是控制器，不是世界模型，也未训练新策略。

### 2.3 Good Judgment, Bad Advice：负结果

arXiv 2610.24649（2026-10-17，Naman Vats）。Terminal-Bench 2.1 固定 Sonnet 5.5 agent loop，只换控制器（无/规则/小 LLM/JEV/安慰剂）；21 个 held-out 任务、315 trials。JEV 把每次尝试成本抬高 **19%**（几何均值比 1.19，95% CI 1.02–1.36），通过率不变（73.0% vs 73.0%）；比小 LLM 便宜（0.81），但比安慰剂贵 22%。离线 JEV 排序更好（AUROC 0.78 vs 规则 0.62、小 LLM 0.60），但校准很差：它给出「已完成」均概率 0.50 时实际 87% 已完成，于是让 agent 复核本来正确的活；没有控制器选择放弃任务[14]。**有信息量 ≠ 能变成有用干预**。

### 2.4 Can Jev be Your Q or Policy in RL？

arXiv 2610.11692（2026-10-08，Yi Ma）。JEV 除价值函数外可占 RL 系统多数「位置」；作者把它用作 **reference policy、exploration judge、replay rater**，在 9 个 MiniGrid + 3 个 Atari 上，用 JEV 辅助训练优于标准 RL learner，包括 learner 单独学不动的任务；JEV 始终冻结[17]。这是「JEV 作为动作/训练信号来源」最接近目标的一篇，但优化的是样本效率，不是蒸馏。

### 2.5 JevSpawn：绕开预枚举

arXiv 2610.00437（2026-09-30，Haoyang Su）。把自然语言任务规范接到有限概率探索：并行 spawn 动作分支、反馈驱动分支选择与表示修订、从保留候选恢复；共享动作结构与模型前缀，**无需额外训练**。8 基准、7 个 agent 基线 + 一个 TypeSafe JEV 变体，任务表现与导航速度有改进[16]。直接针对 JEV 的预枚举瓶颈。

### 2.6 JevVLA：受 JEV 启发的动作偏好函数

arXiv 2610.35010（2026-10-26，Wei Wang）。不用 JEV 本体，而借其「结构化概率决策接口」：冻结 VLA 基座，学习约 **8 万参数**的可微 energy，用执行反馈/专家纠正/偏好训练；energy 值比较离散候选，梯度修正连续动作。MetaWorld MT50、VIMA、LanguageTable、LIBERO Plus、CLIPort 上分别比匹配基座高 **+1.28/+2.31/+1.60/+1.52/+1.28** 个百分点[18]。说明 JEV 式「给候选动作打概率」的接口可压成极小动作评价头。

### 2.7 AnyJev：把任意开放 LLM 变成 Jev 式决策器（Nokia）

arXiv 2610.00831（2026-09-30），Nokia Applied Research 联合 Tencent Hunyuan、香港理工大学，Jiamu Zhang 等；代码 `nokia-applied-research/AnyJev`[10][11]。一次 prefill，在答案位置把 next-token 分布限制到选项 token 上读出 typed decision，无需训练；缺陷是 label prior 与 position bias。修正：除以无标签估计的 label prior；对选项列表 K 个循环旋转取 log-prob 平均。两个 20 选项任务上 order-flip 率 0.33→0.14/0.18，11/11 模型提升；代价是 K 次 prefill，无标签停止规则把 18 次降到 10.6 次，部分设置给出 0.008 偏差界[10]。它不是 JEV 复刻，而是无权重时最便宜的研究代理，保留「预枚举 + 单次读出」契约。

### 2.8 Kev：开放权重 Jev 式模型族

`jaredpalmer/kev`，Apache-2.0，基于 Qwen3.5/Qwen3.8，0.8B/4B/9B/27B 四档；0.8B/4B/9B 是冻结基座上的 adapter，27B 全参微调（51GB）。API 与 TypeSafe System One 兼容，官方 Python SDK 可直接指向本地服务；每个 checkpoint 自带拟合 temperature；「问题共享同一段文本，但彼此不可读」[19]。社区 Decision Index：Kev-27B 52.3 vs JEV 54.0；新来源准确率 0.851 vs 0.857，Brier 0.225 vs 0.211；作者明说不知道 JEV 训练数据，**不是受控架构比较**[19]。它是可本地训练/部署的 Jev 类基线（另有 AnyJev/Tacit）。

### 2.9 JEV-as-a-Judge

arXiv 2609.26550（2026-09-22，Yubo Li）。用 JEV 做只返回标签概率的 judge，置信高则采纳、低则升级给 reasoning judge。对 16 个 judge、盲法人工裁决：verdict 可直接从文本读出时，JEV 与 GPT-6 差距在 3 个百分点内，费用 **0.36%**、延迟中位 **0.15s**；需推导（数学、代码、逻辑）时落后，而其置信度正好标出边界。冻结阈值后，1,610 对 held-out 上「采纳+升级」比 GPT-6 高 0.9 个百分点、费用 41%；两个新工作负载的预注册 live test 与 GPT-6 完全一致；风格对抗与无参考散文上路由变弱[15]。

### 2.10 LeWAM：端到端 JEPA 世界+动作模型

arXiv 2610.12818（2026-10-08，Minghao Fu）。JEPA 式联合嵌入世界模型，联合训练「下一隐状态预测」与「动作 flow matching」，隐状态加各向同性高斯约束，使同一模型既能执行策略、又能用隐动力学采样评估动作做规划。接触密集任务成功率 28.6%→**89.7%**，长时程 10.9%→**46.2%**，规划快 **32.3×**；仿真与真实机器人行为克隆上与成熟方法相当[20]。与 JEV 无关，但是「世界模型直接输出/评估动作」的强基线。

### 2.11 其他直接证据

- **Flide**（arXiv 2610.42148，2026-10-30）：针对 Jev 类负载的精确推理引擎，ForkAttention 在单条扁平 token 流中表达分支树，保证每分支与独立前向一致；比 vLLM/SGLang 快至多 **9.5×**[21]。从系统侧印证「共享前缀 + 分支」负载形态。
- **Jev in the Wild**（2609.30216，2026-09-24）统计 2,170 个公开项目：属性判断/打分最常用，动作选择尚非主流[22]。
- **Evaluating and Benchmarking the System One Model Jev**（arXiv 2609.37647，2026-09-29）：37 数据集、346,009 请求、<$10；JEV 在 27/37 上胜过 Qwen3.8-27B，校准良好、支持 selective prediction；三者都在低资源语言、噪声标签、rubric 评分上退化[12]。目前最扎实的**独立**评测。

## 3. 相邻文献

### 3.1 LLM 作为文本世界模型

- **RAP**（arXiv 2305.14992，EMNLP 2023）：LLM 同时当世界模型与推理 agent，用 MCTS 在推理空间规划[23]。
- **Xie et al.**（arXiv 2409.12278，COLING 2025）：分别微调 precondition 与 effect 两个 LLM，合成数据扩增，人类被试研究验证其世界动力学知识与人的理解一致，并分析能否支持动作链[24]。
- **Wang et al.**（arXiv 2406.06485，ACL 2024）：ByteSized32-State-Prediction 首次直接量化 LLM 作为文本世界模拟器的状态转移；GPT-4 仍不可靠[25]。

三篇共同确立「LLM 世界模型的瓶颈在状态转移的精确性与格式」，也解释了 KIIS 为何强调 JEV 的优势是**格式稳健性**而非内容更准。

### 3.2 经典学习型世界模型

- **Ha & Schmidhuber**（1803.10122）：VAE + MDN-RNN + 小控制器，「在梦中学策略」[26]。
- **DreamerV3**（2301.04104）：RSSM + imagination，单一配置在 150+ 任务超过专用方法，首次从零在 Minecraft 挖到钻石[27]。
- **TD-MPC2**（2310.16828）：decoder-free 隐世界模型 + 局部轨迹优化，单一超参在 104 个在线 RL 任务稳定，317M 单 agent 跨 80 任务/具身/动作空间[28]。
- **IRIS**（**2209.00588**）：离散 token 世界模型 + 想象训练，Atari 100k 样本效率强[29]。
- **DIAMOND**（2405.12399）：扩散世界模型，Atari/CS:GO 上证明视觉细节重要[30]。
- **Genie**（2402.15391）：从无标注视频学习生成式交互环境，11B 参数，含 latent action model[31]。

### 3.3 动作生成、策略蒸馏与数据生成

- **Decision Transformer**（2106.01345）：return-conditioned 序列建模，直接生成动作[32]。
- **DAgger**（1011.0686）：专家在 learner 访问到的状态上打标并聚合数据集，给出 no-regret 模仿学习界——「用强动作选择器教小策略」最标准的理论工具[33]。
- **DreamGen**（2505.12705）：视频世界模型生成 neural trajectories，再用 latent action model 或 IDM 恢复伪动作；仅用单环境单任务遥操作数据，让人形机器人学会 **22** 种新行为并泛化到新环境[34]。
- **UniSim**（2310.06114）：从视频学习可交互真实世界模拟器，用于训练具身/自动驾驶策略[35]。
- **VLA 动作头**：OpenVLA（2406.09246）与 π0（2410.24164）[36][37]。
- **WorldVLA**（2506.21539）：自回归 action world model，统一 VLA 与世界模型，用未来图像预测改进动作生成，并指出自回归动作序列的误差传播[38]。
- **V-JEPA 2**（2506.09985）：1M+ 小时视频预训练的动作无关 JEPA，加少量机器人轨迹后零样本机器人规划[39]。
- **Genie 3**（DeepMind 博客，2025-08-05，Parker-Holder & Fruchter）：仅博客，**无论文**；正文抓取被截断，常见的 720p/24fps/实时等规格**本次未能核验**，引用须标注未核实[40]。

## 4. 证据表

| # | 主张 | 来源 | 证据级别 | Caveat |
|---|---|---|---|---|
| 1 | JEV 2026-09-15 发布，System One 决策模型，state+typed questions→概率 | TypeSafe[1] | 官方声明 | 厂商新闻稿，非同行评审 |
| 2 | $0.042/MTok 输入、输出免费 | TypeSafe[1][6] | 官方声明 | 官方承认可能补贴；价格会变 |
| 3 | 70–500ms、比 LLM 快 40–200× | TypeSafe[1] | 官方声明 | 首页 193.6× 为偏高值；依赖负载/区域 |
| 4 | 346,009 次请求总花费 <$10 | Deußer[12] | 独立测量 | 单一第三方；未含重试/工程成本 |
| 5 | 一次读入 state、问题并行且互不依赖 | TypeSafe[2][6] | 官方声明 | 「并行」是调度语义，不是一题一卡 |
| 6 | 因果 Transformer + 共享前缀 + 隔离分支 + 直接 softmax 读出 | Archer Hume[3] | 独立逆向（行为证据） | 黑盒探测；causal 属推断；无法排除其他实现 |
| 7 | 稀疏 MoE 主干 | Archer Hume[3] | **未验证**（作者自称最不确定） | 纯推断；换成稠密模型不影响其余结论 |
| 8 | RLCD 训练、输出校准概率 | TypeSafe[1][5] | 官方声明 | 损失、数据、更新范围未公开 |
| 9 | 选项需预枚举、上限 255、无自由生成 | 官方[1][2]；JevSpawn[16] | 官方声明 + 论文引述 | 255 是请求校验，不是模型结构证据 |
| 10 | 英文最强，CJK 较弱 | TypeSafe[6] | 官方声明 | 无公开分语言指标 |
| 11 | KIIS：JEV 冻结当世界模型，93.4% vs oracle 92.7%；k=4 为 77.9%，新结构 85.5%；$0.010/ep | Song & Lee[8][9] | 论文声称（作者自跑） | 2 页会议稿、单任务/单 LLM/单 seed；会议未开、无独立复现；新结构仅 rollout；只算 API 费 |
| 12 | REFLEX：95% 成功率、少 72.7% 强模型调用 | Wu[13] | 论文声称 | 冻结 100 任务；对便宜级联优势有限 |
| 13 | JEV 控制器 +19% 成本、通过率不变；校准差 | Vats[14] | 论文声称（预注册） | 单基准、21 任务/315 trials；作者即设计者 |
| 14 | JEV 可作 reference policy/judge/rater 改善 RL | Ma[17] | 论文声称 | 9 MiniGrid+3 Atari；代码/复现状态未核 |
| 15 | AnyJev 单 prefill 读出、旋转去偏 | Zhang et al.[10] | 论文声称 + 开源代码 | Early report；小规模任务；K 倍 prefill 成本 |
| 16 | Kev 27B 接近 JEV（0.851 vs 0.857） | Kev README[19] | 厂商/社区自报 | 非受控比较；社区基准 |
| 17 | 世界模型/动作生成经典结果 | 各原文[26]–[39] | 论文声称（多数有后续复现） | 与 JEV 无直接关系，仅作方法坐标 |
| 18 | Genie 3 的交互世界生成能力 | DeepMind 博客[40] | 官方博客，**无论文** | 规格未核验；不可作可复现基线 |

## 5. 仍然缺失（我们的实验可补的口子）

1. **JEV 直接做动作选择器的证据很薄**：KIIS 把 JEV 放世界模型位置，动作仍由 Qwen3-4B 提出；REFLEX 是控制器；Jev-as-Q 是 RL 辅助信号。**没有一篇在相同候选集上做「JEV 选动作 vs 策略模型选动作」的闭环对照**，也没报告 JEV 选择与引擎最优动作的一致率。
2. **蒸馏到小学习器几乎空白**：Kev/Tacit 蒸「决策行为」，TypeSafe cookbook 蒸「特征」（CatBoost 读 38 个问题，held-out RMSE 1.77）[41]；**没有工作把 JEV 的动作选择/世界模型预测蒸馏进小策略**，也无 DAgger 式聚合与误差界。JevVLA 的 8 万参数 energy head 说明可行，但蒸的是 VLA 基座。
3. **成本核算不完整**：只有「JEV API 费/episode」（KIIS $0.010）与单价；缺 ① 规划/候选生成/重试的端到端 TCO，② 问题数增长带来的输入 token 非线性，③ 与小模型自托管的摊销对比，④ 补贴退出后的敏感性。
4. **长时程误差未系统测量**：KIIS 只到 k=4，环境确定、全观测；错误集中在「是否执行」二值判断。缺 ① k=8/16/32 误差曲线，② 随机/部分可观测环境，③ replan 后的错误恢复，④ 与 persistence/ensemble 对照。
5. **校准与动作价值耦合**：Good Judgment 表明「排序好」≠「干预有用」；缺在决策论框架（阈值、代价矩阵、弃权/升级）下评估概率的实际收益与漂移下阈值稳定性。
6. **独立复现与同 harness 对比**：KIIS 无独立复现；Kev/AnyJev/Tacit 与 JEV 没有同任务、同候选集、同预算的对比。应固定 harness、只换「决策源」，公开日志与成本。
7. **预枚举瓶颈**：JevSpawn 与 Flide 从动作空间与推理系统两侧缓解，但未回答「动作空间多大时 JEV 仍划算」。

## 6. 参考文献

[1] TypeSafe AI, "Introducing System One Models & Jev," 2026-09-15. https://typesafe.ai/blog/introducing-system-one-models-and-jev
[2] TypeSafe Docs, "System One." https://docs.typesafe.ai/concepts/system-one
[3] Archer Hume, "Jev's Architecture Unmasked," 2026-09-17. https://archerhume.com/posts/jevs-architecture-unmasked
[4] TypeSafe Docs, "State." https://docs.typesafe.ai/concepts/state
[5] TypeSafe Docs, "AI primer." https://docs.typesafe.ai/introduction/machine-learning-primer
[6] TypeSafe Docs, "POST System One" / "Models." https://docs.typesafe.ai/api/system-one ；https://docs.typesafe.ai/models
[7] TypeSafe Docs, "Jev 1.13 jaggedness," reviewed 2026-10-02. https://docs.typesafe.ai/model-jaggedness/jev-1.13
[8] Y. H. Song, K. M. Lee, "A World Model for LLM Agents: Comparing a Decision Model with General-Purpose LLMs," KIIS 2026 Fall, oral, paper 00027. https://github.com/thddydgnl/jev-world-model
[9] 同上仓库，`README.md` 与 `kiis2026f/paper/원고.md`（Table 1、Fig. 2、slots.json）. https://github.com/thddydgnl/jev-world-model/blob/main/kiis2026f/paper/%EC%9B%90%EA%B3%A0.md
[10] J. Zhang et al., "AnyJev Technical Report," arXiv:2610.00831, 2026-09-30. https://arxiv.org/abs/2610.00831
[11] Nokia Applied Research, AnyJev code. https://github.com/nokia-applied-research/AnyJev
[12] T. Deußer, "Evaluating and Benchmarking the System One Model Jev," arXiv:2609.37647, 2026-09-29. https://arxiv.org/abs/2609.37647
[13] T. Wu, "REFLEX with Jev for Efficient Selective Control in LLM Agents," arXiv:2609.26532, 2026-09-22. https://arxiv.org/abs/2609.26532
[14] N. Vats, "Good Judgment, Bad Advice: Decision-Model Controllers Raise Agent Cost Without Improving Success," arXiv:2610.24649, 2026-10-17. https://arxiv.org/abs/2610.24649
[15] Y. Li, "JEV-as-a-Judge: Accept When Confident, Escalate When Unsure," arXiv:2609.26550, 2026-09-22. https://arxiv.org/abs/2609.26550
[16] H. Su, "JevSpawn: Adaptive Agentic Inference through Compositional Action Spaces," arXiv:2610.00437, 2026-09-30. https://arxiv.org/abs/2610.00437
[17] Y. Ma, "Can Jev be Your Q or Policy in Reinforcement Learning?," arXiv:2610.11692, 2026-10-08. https://arxiv.org/abs/2610.11692
[18] W. Wang, "JevVLA: Jev-Inspired Energy-Based Adaptation for Vision-Language-Action Policies," arXiv:2610.35010, 2026-10-26. https://arxiv.org/abs/2610.35010
[19] J. Palmer, "Kev: Small Jev-like decision models you can train and run yourself." https://github.com/jaredpalmer/kev
[20] M. Fu, "LeWAM: End-to-End World and Action Modeling with JEPAs," arXiv:2610.12818, 2026-10-08. https://arxiv.org/abs/2610.12818
[21] B. Li, "Flide: Accelerating Decision Inference with Exact State Forking," arXiv:2610.42148, 2026-10-30. https://arxiv.org/abs/2610.42148
[22] G. Ling, "Jev in the Wild: A Data-Driven Analysis of the Jev Model's Functionality, Applications and Ecosystem," arXiv:2609.30216, 2026-09-24. https://arxiv.org/abs/2609.30216
[23] S. Hao et al., "Reasoning with Language Model is Planning with World Model," arXiv:2305.14992, EMNLP 2023. https://arxiv.org/abs/2305.14992
[24] K. Xie et al., "Making Large Language Models into World Models with Precondition and Effect Knowledge," arXiv:2409.12278, COLING 2025. https://arxiv.org/abs/2409.12278
[25] R. Wang et al., "Can Language Models Serve as Text-Based World Simulators?," arXiv:2406.06485, ACL 2024. https://arxiv.org/abs/2406.06485
[26] D. Ha, J. Schmidhuber, "World Models," arXiv:1803.10122, 2018. https://arxiv.org/abs/1803.10122
[27] D. Hafner et al., "Mastering Diverse Domains through World Models" (DreamerV3), arXiv:2301.04104, 2023. https://arxiv.org/abs/2301.04104
[28] N. Hansen et al., "TD-MPC2: Scalable, Robust World Models for Continuous Control," arXiv:2310.16828, 2023. https://arxiv.org/abs/2310.16828
[29] V. Micheli, E. Alonso, F. Fleuret, "Transformers are Sample-Efficient World Models" (IRIS), arXiv:2209.00588, ICLR 2023. https://arxiv.org/abs/2209.00588
[30] E. Alonso et al., "Diffusion for World Modeling: Visual Details Matter in Atari" (DIAMOND), arXiv:2405.12399, 2024. https://arxiv.org/abs/2405.12399
[31] J. Bruce et al., "Genie: Generative Interactive Environments," arXiv:2402.15391, 2024. https://arxiv.org/abs/2402.15391
[32] L. Chen et al., "Decision Transformer: Reinforcement Learning via Sequence Modeling," arXiv:2106.01345, 2021. https://arxiv.org/abs/2106.01345
[33] S. Ross, G. Gordon, D. Bagnell, "A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning" (DAgger), arXiv:1011.0686, AISTATS 2011. https://arxiv.org/abs/1011.0686
[34] J. Jang et al., "DreamGen: Unlocking Generalization in Robot Learning through Video World Models," arXiv:2505.12705, 2025. https://arxiv.org/abs/2505.12705
[35] S. Yang et al., "Learning Interactive Real-World Simulators" (UniSim), arXiv:2310.06114, ICLR 2024. https://arxiv.org/abs/2310.06114
[36] M. J. Kim et al., "OpenVLA: An Open-Source Vision-Language-Action Model," arXiv:2406.09246, 2024. https://arxiv.org/abs/2406.09246
[37] K. Black et al., "π0: A Vision-Language-Action Flow Model for General Robot Control," arXiv:2410.24164, 2024. https://arxiv.org/abs/2410.24164
[38] J. Cen et al., "WorldVLA: Towards Autoregressive Action World Model," arXiv:2506.21539, 2025. https://arxiv.org/abs/2506.21539
[39] M. Assran et al., "V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning," arXiv:2506.09985, 2025. https://arxiv.org/abs/2506.09985
[40] J. Parker-Holder, S. Fruchter, "Genie 3: A new frontier for world models," Google DeepMind blog, 2025-08-05. https://deepmind.google/blog/genie-3-a-new-frontier-for-world-models/
[41] TypeSafe Docs, "Autoresearch feature discovery." https://docs.typesafe.ai/cookbooks/autoresearch_feature_discovery
[42] S. Willison, "Jev introduces a new shape of LLM—System One, aka Decision Models," 2026-09-21（二手背景，交叉核对发布叙事）. https://simonwillison.net/2026/Sep/21/jev/
