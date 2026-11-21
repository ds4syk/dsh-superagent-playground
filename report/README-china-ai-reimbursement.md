# 高校报销与境外 AI 订阅：调研索引（2026-11）

> 回答 Feishu 问题：国内高校/课题组怎么报销？境外 AI（ChatGPT/Claude）开不出中国发票，弄成“技术服务费”能不能报？

## 一句话结论

**不建议把境外 AI 订阅“弄成技术服务费”。** 如果实际交易是境外软件订阅，却让境内公司开“技术服务费”发票，属于票货不符/变名开票，可能触发虚开、套取科研经费的审计与税务风险。合规路径是：先确认预算科目与票据口径 → 优先境内模型/云服务或学校统一采购 → 横向课题按真实服务合同采购 → 学校对公境外采购（门槛高）→ 个人垫付（不建议）。

## 交付物

| 文件 | 内容 |
|---|---|
| [`kb/reimbursement-ai/00.md`](../kb/reimbursement-ai/00.md) | 总览：六条结论、资料索引、证据分级 |
| [`kb/reimbursement-ai/01.md`](../kb/reimbursement-ai/01.md) | 高校/课题组报销体系：经费类型、四个硬条件、科目地图、票据与流程 |
| [`kb/reimbursement-ai/02.md`](../kb/reimbursement-ai/02.md) | 境外 AI 订阅：为什么没有中国发票、增值税/所得税、外汇与备案 |
| [`kb/reimbursement-ai/03.md`](../kb/reimbursement-ai/03.md) | “技术服务费”能不能报：三种情形、发票规则、变名开票案例 |
| [`kb/reimbursement-ai/04.md`](../kb/reimbursement-ai/04.md) | 各校口径样本：新疆大学、山西工商、北信科、东南、中科大、中国劳动关系学院 |
| [`kb/reimbursement-ai/05.md`](../kb/reimbursement-ai/05.md) | 合规路径与行动清单：决策表、五条路径、问财务的 6 个问题 |
| [`kb/reimbursement-ai/06.md`](../kb/reimbursement-ai/06.md) | 风险清单：虚开、套取、税务、外汇、数据、采购、账号、学术诚信 |
| [`docs/visuals/2026-11-21-ai-reimbursement.png`](../docs/visuals/2026-11-21-ai-reimbursement.png) | 一页决策图 |
| [`deliverables/高校报销与境外AI订阅_2026-11.docx`](../deliverables/高校报销与境外AI订阅_2026-11.docx) | DOCX 知识库（34 页，内嵌决策图） |
| [`src/build_docx_reimbursement.py`](../src/build_docx_reimbursement.py) | DOCX 生成脚本 |
| [`src/make_visual_reimbursement.py`](../src/make_visual_reimbursement.py) | 决策图生成脚本 |

## 核心证据

- 国家自然科学基金资金管理办法（财教〔2021〕177号）：直接费用分为设备费/业务费/劳务费；业务费含“出版/文献/信息传播/知识产权事务”和“其他相关支出”；第三十二条允许特定科研活动无发票报销。
- 国家重点研发计划资金管理办法（财教〔2025〕2号）：同上口径。
- 《增值税法》（2026-01-01 施行）：境外单位在境内发生应税交易，购买方为扣缴义务人；服务税率 6%。
- 《企业所得税法》及实施条例：非居民源泉扣缴；特许权使用费减按 10%。
- 国家税务总局公告 2017 年第 16 号：发票内容必须与实际销售一致，不得变名开具。
- 高校样本：新疆大学 2026 年明确纵向经费不报“各类软件服务费/会员年费”；山西工商学院 2025 年允许“数据库使用权限费”但不报会员费；北信科把“信息传播费”限定在出版/文献/检索；东南大学、中科大允许境外 Invoice/Receipt + 付款记录。
- 2026-11 农工党社情民意：指出现行制度没有“大模型订阅/API 调用”科目，建议设“人工智能与科研计算服务费”。

## 限制

- 没有做全国问卷；高校样本有限，结论是“制度逻辑 + 可核对样本”，不是全国统一口径。
- 各校财务细则差异大；具体报销以本单位财务处、科研院和主管税务机关书面答复为准。
- 本报告不构成法律、税务或财务意见。
- 本次执行环境的 Feishu bot 未申请 `docx:document` / `drive:drive` 权限，无法创建 Feishu 知识库页面；交付通过 GitHub PR 的 Markdown、PNG 和 DOCX 提供。

## 复现

```bash
~/miniconda3/envs/sdpr-clean/bin/python src/make_visual_reimbursement.py
~/miniconda3/envs/sdpr-clean/bin/python src/build_docx_reimbursement.py
```
