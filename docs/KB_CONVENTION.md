# 交付约定：默认 DOCX 知识库

> 用户 2026-11-02 设定：面向用户的研究/研读/调研/方案类交付，**默认发 DOCX 文档，整理成知识库形式**。
> 样板是「实验」群的世界模型知识库：一个 `00 总览` + `01/02/03` 子页 + 内嵌图片 + 维护约定。
> 本文件是可版本化的详细模板；全局 `~/.dsh/AGENTS.md` 只保留简短规则。

## 1. 结构

```
<主题>（YYYY-MM）/
  00 总览：<主题>（YYYY-MM）
  01 <子主题>
  02 <子主题>
  ...
  assets/                 # 图片、图集（本地源文件）
```

- **00 总览**：说明本次知识库范围；`资料索引` 表（页面/内容/链接）；`一句话结论` 3–6 条；
  `证据分级`；`维护约定`。
- **NN 子页**：标题 → `> 结论先行` → `如果只记三件事` → `## 0. 口径`（论文/数据/证据分级）→
  主体章节（表格、公式、图）→ 文献/复现/行动清单。
- 命名 `NN 标题`；一个主题一个文件夹；总览索引同步更新。
- 图片内嵌在对应页面；代码、原始日志、大文件留在仓库/数据盘，知识库只放可读结论。

## 2. 页面模板

```markdown
# NN <标题>

> **结论先行**：1–3 句，直接给结论。

**如果只记三件事**：

1. ...
2. ...
3. ...

## 0. 口径

| 论文/数据 | 出处 | 本页用途 |
|-|-|-|
| ... | ... | ... |

**证据分级**：论文声称 / 独立复现 / 未验证；正文分别标注。

---

## 1. ...
```

## 3. 证据分级

| 等级 | 含义 |
|-|-|
| **论文声称** | 论文正文/图表给出的数字或结论 |
| **独立复现** | 本仓库代码跑出、可追溯 |
| **未验证** | 只读摘要/补充材料/代码，未独立跑数值 |

不把“论文声称”冒充“独立复现”；复现类页面必须写清设置、路线、结果、差异归因、未复现范围、资源。

## 4. Feishu 实现（lark-cli）

```bash
# 1) 建文件夹（bot 身份；随后给 ticket owner 授权）
lark-cli drive +create-folder --as bot --name '<主题>（YYYY-MM）'

# 2) 建子页（Markdown 中的 $...$ 会转成 <latex> 公式）
cd <repo>
lark-cli docs +create --as bot --parent-token <folder_token> \
  --doc-format markdown --content @./kb/01.md

# 3) 追加图片（先跑完文字页，再按顺序追加；图片进入文末“图集”）
lark-cli docs +media-insert --as bot --doc <doc_url> \
  --file ./kb/assets/fig1.png --type image --caption '图 1：...' --width 900

# 4) 授权用户（子文档通常继承文件夹权限，仍建议抽查）
lark-cli drive +member-add --as bot --token <folder_token> --type folder \
  --member-type openid --member-id <owner_open_id> --perm full_access --yes

# 5) 导出 .docx（默认 Word 交付）
lark-cli drive +export --as bot --url <doc_url> --file-extension docx \
  --output-dir ./deliverables --overwrite
```

注意：
- `docs +create` 的 `--content @file` 只接受 CWD 内相对路径；在仓库根目录执行。
- 本地图片不要写进 Markdown 的 `![](...)`（导入时可能报 path unsafe）；用 `+media-insert` 在文末追加。
- `+media-insert` 每次追加到文末；页面里先留 `## 图集` 小节。
- 网络/API 偶发 EOF，命令要带重试；创建后用 `drive files list` 核对名称与 token。
- 权限变更/公开分享是高危操作；只给 ticket owner 授权，不公开分享。

## 5. 示例

本仓库的 SDPR/QPEP 知识库即按此模板生成：

- `00 总览`：索引、一句话结论、证据分级、维护约定
- `01 五个问题的回答`
- `02 SDPR 精读`
- `03 QPEP 精读`
- `04 相机标定仿真复现`

本地源文件：`kb/00.md`–`kb/04.md`、`kb/assets/`；可读副本：Feishu docx 链接 + `deliverables/*.docx`。
