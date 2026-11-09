This is by an automated agent, not a human.

## 结论

基于 SDPR 写新方法，主表不应只有一张精度表，而是**四件套**：主比较（精度）、尾部与失败（鲁棒性）、证书与效率（可认证性的代价）、消融（每个设计点）。数据集由创新点类型决定：延续 spacecraft 主线用 **SPEED+**（HIL 双域，官方 `ET`/`ER`/`Epose`）；改进 SDPR 层本身则用 PnP、位姿图/旋转同步、标定与 BAL/Replica。Baseline 必须三层同时在场：经典几何、可微/可认证几何、领域 SOTA。

## 交付物

- **知识库 DOCX（21 页，A4）**：[`deliverables/SDPR新方法主表实验设计知识库_2026-11.docx`](deliverables/SDPR新方法主表实验设计知识库_2026-11.docx)
- **Markdown 源**：[`kb/sdpr-main-table/`](kb/sdpr-main-table/00.md)（00 总览、01 数据集、02 baseline、03 主表模板、04 文献范式、05 风险与清单）
- **一页可视化**：[`docs/visuals/2026-11-09-sdpr-main-table-design.png`](docs/visuals/2026-11-09-sdpr-main-table-design.png)
- 生成脚本：[`scripts/make_visual.py`](scripts/make_visual.py)、[`scripts/build_sdpr_main_table_docx.py`](scripts/build_sdpr_main_table_docx.py)

![一页速览：主表设计决策图](https://raw.githubusercontent.com/ds4syk/dsh-superagent-playground/agent/sdpr-main-table-design-895c860a6064/docs/visuals/2026-11-09-sdpr-main-table-design.png)

## 主表建议（摘要）

| 项 | 建议 |
|-|-|
| 主数据集 | SPEED+：synthetic train + lightbox/sunlamp HIL；官方 `Epose`，同时给绝对 `ET`(m) |
| 第二数据集 | 按问题选一个：SPEED（泛化）、prisma25（真实飞行）、SPADES（事件/光照）、BAL/Replica（层改进） |
| 必做 baseline | EPnP+LM、SQPnP、SDPRLayers、PnP\*、EPro-PnP、Theseus、SPNv2-H、SPNv2-E |
| 领域 SOTA | KRN、UDA、StructGCN、Wang 2022/2023（至少再加 2 个） |
| 六张表 | T1 主比较、T2 尾部/失败率、T3 证书/效率、T4 消融、T5 鲁棒 sweep、T6 跨数据集 |
| 当前草稿缺口 | 只有 3 个 baseline；只报相对平移；无 `Epose`、runtime、证书率、失败率、跨数据集 |

检索结论：截至 2026-11-09，Semantic Scholar 显示 SDPRLayers 仅 13 篇引用，公开检索未见将 SDPR 用于 spacecraft pose 的论文；最接近的是事件卫星的可认证测试时自监督（arXiv:2409.06240）与可认证 PnP 求解器 PnP\*（arXiv:2610.31797）。

## 验证

- `pypandoc` 从 6 页 markdown 构建 DOCX；`python-docx` 后处理为 A4、中文字体、表格固定列宽与跨页表头。
- LibreOffice 转 PDF：21 页，检查了目录、总览、数据集表、主表模板、文献范式与末页，无表格行断裂。
- 一页图单独渲染并检查（16:9）。
- 未跑任何模型实验；本文是调研与实验设计，不含新数值结果。

## 局限与需要拍板

- 检索来源为 arXiv/OpenAlex/Semantic Scholar/DOI，未穷尽 IEEE Xplore/Scopus；投稿前建议复查。
- 未逐一复现所有 baseline；“开源状态”为公开检索结果，clone 前需再确认。
- 需要拍板：新方法定位（A 层改进 / B 新任务 / C 表征桥接）、是否用 HIL 标签、是否上官方 leaderboard、目标 venue。
