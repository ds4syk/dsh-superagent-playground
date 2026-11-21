# 实验设计与实现细节

本文档解释 `run_sessions.py` 的关键决定，方便以后复用/审计。

## 1. 会话结构

每个 `(model, problem)` 是独立的 API 会话，消息序列只包含：

1. `system`：研究数学家的角色、禁止检索、要求诚实标注缺口；
2. `user`：题面（`prompts/<id>.md`）+ 第一轮指令；
3. `assistant`：模型输出（content；reasoning 单独保存）；
4. `user`：第二轮“敌意审稿 + 终稿”指令；
5. 必要时：`user`：把上一轮的 reasoning 摘要 + “停止推理，写终稿”。

不包含：官方论文/摘要/推理、评分 key、其它模型输出、其它案例。

## 2. 为什么用 streaming + cap

DeepSeek-V4-Pro / Kimi-K3 是 thinking 模型：`max_tokens` 同时限制 reasoning + content。
在硬题上，模型可能把全部预算花在 reasoning 上，返回 `content=""`。非 streaming 调用还可能
长时间无输出，难以诊断。

因此：

- 每次调用用 `stream:true`，累计 `reasoning_content` 与 `content`；
- 按 `~3.5 chars/token` 估算，达到 cap 就主动关闭连接（服务端停止生成）；
- 记录 `truncated_by_harness=true`，不假装模型自然结束；
- 如果 content 为空，下一轮用 `thinking: {"type":"disabled"}` + `reasoning_effort:"none"`
  把 reasoning 摘要交给模型，要求它“停止推理，写终稿”。这保证有可评分的文本，同时明确标注
  这是预算截断后的合并。

## 3. 预算

| 阶段 | cap（tokens） | 说明 |
|---|---:|---|
| s1_solve | 24576 | 独立求解；通常 6–7 分钟 |
| s1_force_final | 12288 | 仅在 s1 content 为空时；thinking disabled，通常 <1 分钟 |
| s2_final | 16384 | 敌意审稿 + 终稿；通常 4–5 分钟 |
| s2_force_final | 12288 | 仅在 s2 content 为空时 |

这不是算力对齐：OpenAI 官方平均 3 小时 ChatGPT Pro thinking/结果；本实验每个案例只有
~40k 输出 token 的预算。结论应表述为“在有限预算下的下界”，不是“模型上限”。

## 4. 沙箱（可选）

`run_python()` 用 bubblewrap：

- `--unshare-user --uid 65534 --gid 65534`：以 nobody 运行；
- `--unshare-net`：无网络；
- 只读绑定 `/usr`、`/lib`、`/lib64`、`/etc` 与 conda 前缀；
- `--tmpfs /tmp`、临时工作目录；
- `RLIMIT_CPU=60s`、`RLIMIT_AS=4GB`、`RLIMIT_FSIZE=16MB`。

本版主实验**没有**自动执行模型代码；`run_python` 保留给后续“工具条件”。任何真正的计算任务应通过
Slurm 提交，日志落 `logs/`。

## 5. Slurm

`submit.sh` 在 `cpu` 分区申请 1 节点 8 CPU、12 小时；`--chdir` 固定到本目录（`sbatch` 会把
脚本复制到 spool 目录，不能靠 `dirname $0`）。API 调用在计算节点进行（已确认有外网）。

## 6. 评分

`grade.py` 用两个评分模型，各自看“官方结论 + 候选终稿”，输出 0–5 分与缺口。评分者与求解者
分离；官方 key 只进入评分会话，不进入求解会话。人工复核高分/分歧项。

## 7. 污染

`contamination/probe.py` 在独立会话里问模型“截至知识截止，该问题状态如何”。若模型能复述
2026-10 的官方结论，则该案例可能被预训练污染；结果表会标注。探针输出不回流到求解会话。

## 8. 已知限制

- 单次采样；温度/effort 默认；无置信区间。
- 题面经删减，可能改变难度。
- 评分研究级证明很难；双评分 + 人工复核仍可能误判。
- 没有复现 Lean 证明；形式化验证是下一步。
- 10 个案例是 OpenAI 选出的公开样例，存在选择偏差。
