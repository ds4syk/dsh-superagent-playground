# 成本审计：Kimi API 来源与 token/费用

> 数据快照：2026-11-21 22:15（Asia/Shanghai）。金额按**列表价**估算，不是平台账单；
> 实际账单可能因缓存、峰谷、赠送额度、汇率而不同。

## 1. Kimi API 从哪来

- 不是本轮申请的新 key。Kimi 的 key 在本机 DSH 凭据库
  `~/.dsh/.credentials.yaml` 中以 `MOONSHOTAI_CN_API_KEY` 引用存在；文件
  `mtime = 2026-11-02 17:57:26`，说明在本票（2026-11-21）之前就已配置。
- DSH 的 web profile（`~/.dsh/profiles/web/cordis.yml`）里有 provider
  `moonshotai-cn`，配置 `apiKeyEnv: MOONSHOTAI_CN_API_KEY`；本实验直接按该引用调用
  `https://api.moonshot.cn/v1/chat/completions`，模型 `kimi-k3`。
- key 前缀 `sk-DhK...`（长度 51），**不打印完整 key**。如果这不是你预期的 Moonshot
  账号，需要确认是否更换；本轮没有创建/轮换任何 key。

## 2. 实测 token（可追溯）

来源：`transcripts/` 中每个 API turn 的 `usage`，加 `results/grades/` 中每次评分的
`_usage`。这些是**实际返回的 usage**，不是估算。

| 项目 | 调用数 | prompt | completion | 合计 | 其中 cache hit |
|---|---:|---:|---:|---:|---:|
| DeepSeek-V4-Pro 实验（10 案例） | 33 | 743,400 | 351,638 | 1,095,038 | 192,000 |
| DeepSeek-V4-Pro 最终评分（20 次） | 20 | 90,407 | 8,145 | 98,552 | 40,704 |
| Kimi-K3 缩减 direct 实验（10 案例） | 10 | 9,634 | 57,731 | 67,365 | 0 |
| Kimi-K3 最终评分（20 次） | 20 | 93,085 | 8,253 | 101,338 | 0 |
| **小计** | **83** | **936,526** | **425,767** | **1,362,293** | **232,704** |

**注意**：上表只统计“最终保留的 transcripts + 最终评分”。它还**不包括**：
- 被取消的 Kimi thinking 预跑（见下）；
- 早期评分试跑、校准调用、污染探针、失败重试。

## 3. 未记录的调用（估算）

| 项目 | 估算依据 | 估算 tokens |
|---|---|---:|
| Kimi thinking 预跑（两次，被取消） | 15 个 S1 调用 × 约 22k 输出 + 约 1.5k 输入/次 | ~342k |
| DeepSeek 被取消的 362 S2 | 16k 输出 + 输入 | ~17k |
| 早期评分试跑（14+1+20 次） | 约 4k prompt + 300 输出/次 | ~150k |
| 校准（6k 调用 + 32k 调用） | 6k + 32k + prompt | ~39k |
| 污染探针（20 次，fast） | ~250 tokens/次 | ~5k |
| 其他测试/重试 | 小样本 | ~10k |
| **小计** | | **~563k** |

合并后**总 token 约 1.93M**（1.36M 实测 + 0.56M 估算），其中约 0.8M 是输出 token。

## 4. 费用估算（列表价）

价格来源（2026-11-21）：
- DeepSeek：<https://api-docs.deepseek.com/quick_start/pricing>。`deepseek-v4-pro`
  高峰价：cache hit $0.044 / cache miss $1.32 / output $3.96 每 1M tokens；低谷半价。
  本实验在 21:00–21:35（高峰）运行，按高峰价估算。
- Kimi K3：<https://www.kimi.com/resources/kimi-k3-pricing>。cache hit $0.30 /
  cache miss $3.00 / output $15.00 每 1M tokens。

| 项目 | 费用（USD） | 说明 |
|---|---:|---|
| DeepSeek 实验 | $2.13 | 743k prompt（192k hit）+ 352k output |
| DeepSeek 评分 | $0.10 | 含 40.7k cache hit |
| Kimi direct 实验 | $0.89 | 67k tokens，输出为主 |
| Kimi 评分 | $0.40 | 101k tokens |
| **实测小计** | **$3.53** | 1.36M tokens |
| Kimi thinking 预跑（估算） | ~$5.1 | 342k 输出 × $15/M |
| DeepSeek 取消/重试/校准/探针（估算） | ~$0.7 | 约 220k tokens |
| **总计（估算）** | **~$9.3** | 约 1.93M tokens |

按 1 USD ≈ 7.1 CNY，约 **¥66**；合理区间 **$8.5–10.5（¥60–75）**。

## 5. 结论

- **Kimi key 不是本轮新建**：来自本机 DSH 凭据库，2026-11-02 就已存在。
- **实测可追溯部分**：1.36M tokens，约 **$3.5**（¥25）。
- **含被取消的 Kimi thinking 预跑**：约 **1.9M tokens，$9–10（¥65–72）**。
- 费用大头是 **Kimi K3 的输出 token（$15/M）**，尤其是被取消的 thinking 预跑；
  DeepSeek-V4-Pro 反而便宜（$3.96/M 输出）。
- 若重跑：Kimi 直接用 `thinking: disabled` + 16k 上限（本轮最终采用的缩减协议），
  或先做小样本；DeepSeek 可保留 2 阶段 + 强制终稿。

## 6. 复算

```bash
cd repro/openai-10
python3 - <<'EOF'
import json,glob
def s(model):
    t={'p':0,'c':0,'n':0}
    for f in glob.glob(f'transcripts/{model}/*.json'):
        for u in json.load(open(f)).get('usage',[]) or []:
            t['p']+=u.get('prompt_tokens',0); t['c']+=u.get('completion_tokens',0); t['n']+=1
    return t
print(s('deepseek'), s('kimi'))
EOF
```
