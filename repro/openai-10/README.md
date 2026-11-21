# 盲复现：OpenAI 公开的 10 个数学案例

本目录是一次**信息隔离**的复现实验：把 OpenAI 在 `openai/math` 中公开的 10 个案例的
**题面**分别交给 DeepSeek-V4-Pro 与 Kimi-K3，让它们在不知道官方结论、论文、推理摘要或
Lean 文件的情况下独立探索；随后按预先写好的 rubric 评分。

## 1. 十个案例是什么

`openai/math` 在 2026-10-06 发布时目录为 **722 篇手稿 / 372 个 result family**；截至
2026-11-21 的 `main` 快照为 **721 篇 / 373 个 family**（期间有撤回、修订与新增）。
其中只有 **10 个 family** 公开了“abridged reasoning summary”，即公开的推理摘要：

| # | family | 主题 | 公开推理摘要 |
|---|---|---|---|
| 1 | 007 | 乘性函数的两点相关 / 修正 Elliott 猜想 | `ordinary-two-point-correlations.pdf` |
| 2 | 017 | \(\pi\) 的无理性指数 | `irrationality-exponent-of-pi.pdf` |
| 3 | 087 | 对称与一般 Mahler 猜想 | `symmetric-and-general-mahler-conjectures.pdf` |
| 4 | 102 | 基本 SDP 阈值 / Unique Games | `basic-semidefinite-threshold-np-hardness.pdf` |
| 5 | 159 | Erdős 倒数和猜想 / 拟多项式 Szemerédi | `quasipolynomial-arithmetic-progressions.pdf` |
| 6 | 197 | Kaplansky 直接有限性（正特征） | `kaplansky-direct-finiteness-characteristic-two.pdf` |
| 7 | 221 | 稀释自旋玻璃的 Mézard–Parisi 公式 | `mezard-parisi-formula.pdf` |
| 8 | 271 | 量子 Heisenberg 铁磁体的自发磁化 | `spontaneous-magnetization-quantum-heisenberg-ferromagnet.pdf` |
| 9 | 287 | 自由群因子同构 | `free-group-factor-isomorphism.pdf` |
| 10 | 362 | 三维相对论 Vlasov–Maxwell | `relativistic-vlasov-maxwell.pdf` |

“10 个案例公开”在这里指**公开了推理摘要的 10 个 family**；其余手稿的 PDF 也是公开的，
但没有公开推理过程。这个口径与多家报道一致（“10 abridged inference abstracts”）。

## 2. 题面从哪里来

十个 `prompts/*.md` 来自上述推理摘要中标注为 `Original prompt (excerpts)` 的题面。
处理规则：

1. 保留定义、量词、目标陈述、以及“什么算完整解决”的说明；
2. 删除官方推理摘要的正文、标题、结论；
3. 删除指向解法的提示（例如 102 的 “intended improvement”、197 的 “research direction”
   与 references）；
4. 017 的摘要没有附原题面，按论文摘要与标准定义重写为中性题面（“determine \(\mu(\pi)\)”，
   并给出已知上界 7.103… 作为背景，不声称答案）；
5. 不把 OpenAI / 722 / family 编号写进题面，避免模型通过元信息检索。

完整题面在 `prompts/`。评分用的官方结论在 `research/ai-math-survey/official_key.md`，
它**不会**进入任何求解上下文。

## 3. 模型与预算

| 模型 | API id | 上下文 | 单次最大输出 | 轮次 |
|---|---|---|---|---|
| DeepSeek | `deepseek-v4-pro` | 1,048,576 | 32,768 | 4（2 轮独立 + 2 轮工具/终稿） |
| Kimi | `kimi-k3` | 1,048,576 | 32,768 | 4 |

- 每个 (模型, 案例) 是**独立会话**；不共享上下文，不交叉投喂。
- 第 1 轮：独立求解（约 24k 输出 token）；若 reasoning 占满、没有终稿，用
  `thinking: disabled` 把 reasoning 摘要合并成终稿（`s1_force_final`）。
- 第 2 轮：敌意审稿 + 终稿（约 16k）；同样在无终稿时强制合并（`s2_force_final`）。
- **本版未启用自动 Python 工具条件**：`run_python` 已实现（bubblewrap、uid 65534、无网络、
  60 s CPU、4 GB 内存），保留给后续“工具条件”实验；本票的 10 案例结果均来自纯推理 + 强制终稿。
- 全部 API 调用与代码执行都在 Slurm 作业内完成（`submit.sh`），日志在 `logs/`。

预算说明：OpenAI 声称平均每个结果约 3 小时 ChatGPT Pro thinking；本实验每个案例只有
4 次、每次最多 32k 输出 token 的预算，**不是算力对齐的比较**，而是“公开模型在有限预算下
能否独立复现”的下界测试。

## 4. 隔离与污染

- 求解上下文只含：系统提示、题面、模型自己的前文、工具输出。
- 不包含：官方论文/摘要/推理、评分 key、其他模型输出、本仓库其它案例。
- API 模型可能通过预训练见过 2026-10 的新闻；这一点无法用 prompt 完全排除。实验另做
  **污染探针**（`contamination/`），在独立会话里问模型“截至你的知识截止，该问题状态如何”，
  用来标注哪些案例可能被记忆污染。
- 评分者可以看到官方结论；评分者与求解者分离。

## 5. 运行

```bash
cd repro/openai-10
sbatch submit.sh          # 全量：10 案例 × 2 模型 × 2 阶段
# 或本地小规模：
python3 run_sessions.py --models deepseek --problems 287_free_group_factors --stage1 1 --stage2 0
```

结果：`transcripts/<model>/<problem>.json`；汇总：`results/run_summary.json`、
`results/grades_summary.json`、`RESULTS.md`。
评分：`python3 grade.py --graders deepseek,kimi --force`；汇总 `make_summary.py`；
知识库表格 `update_kb.py`。

## 6. 评分

`grade.py` 用独立评分会话，把“官方结论 + 候选终稿”交给评分模型，输出结构化 JSON：
方向是否正确、主定理是否证出、最大缺口、0–5 分、证据引文。每个候选由两个评分模型
（DeepSeek/Kimi）各评一次，分歧由人工复核。评分只看“是否复现官方声称的方向与主定理”，
不把官方声称当作已独立验证的真理。

## 7. 已知限制

- 10 个案例是 OpenAI 选出的公开样例，存在选择偏差（很可能偏向“成功且可讲清楚”的结果）。
- 单次采样、无置信区间；模型温度、effort 默认设置可能影响结果。
- 评分研究级证明本身很难；rubric 只能区分“完整/部分/错误”，不能替代领域专家。
- 题面经过删减；删减可能改变难度（通常使题面更中性、更难）。
- 官方结论中有相当比例尚未 Lean 形式化；本实验不复现 Lean 证明，只复现“从题面到证明”的探索。
