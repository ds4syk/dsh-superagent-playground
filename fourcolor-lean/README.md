# 四色定理的 Lean 4 形式化：复现与审计

本目录是论文《四色定理的 Lean 4 形式化：一次可复现的机器验证》的配套工程。
上游的形式化证明在 [`corun1024/4ct`](https://github.com/corun1024/4ct)，
本仓库**不复制**它的 59 万行源码，而是按固定 commit 克隆、构建、检查，
并额外运行本目录的独立审计。

## 结论

- 在 Lean `v4.34.0-rc2` 与 Mathlib `141f6b6` 上，从干净源码构建
  `corun1024/4ct@3db71e0` 成功；
- 最终定理 `FourColor.fourColorTheorem : FourColorTheorem` 通过内核检查；
- `#print axioms` 只有 `propext`、`Classical.choice`、`Quot.sound`；
- 没有 `sorry`、没有 `native_decide`、没有自定义公理；
- 反 vacuity 负对照与非空性（K4 地图）证据通过。

详细的构建日志、公理输出与源码扫描见 [`evidence/`](evidence/)。

## 本次实测（2026-11-17）

- 机器：40 线程，125.5 GiB 内存，Linux x86_64；
- 构建：821 个模块，16 jobs，80 GB 预算，计划 7.9 core-hour；最慢的证书模块 `Mask/Cfg561` 约 720 秒、`Mask/Cfg612` 约 671 秒；
- 构建墙钟：约 39 分钟（编译 2287 秒；821 个模块，0 失败）；
- 结果：`check.sh` 通过，`#print axioms` = `[propext, Classical.choice, Quot.sound]`；
- 独立审计：`AUDIT OK`（详细日志见 `evidence/`）。

## 目录

| 路径 | 内容 |
|---|---|
| `upstream.lock` | 上游仓库、commit、Lean 与 Mathlib 版本、许可证 |
| `audit/Audit.lean` | 独立审计：最终定理的类型、定义展开、公理闭包、`sorryAx` 扫描、声明计数 |
| `audit/run.sh` | 一键复现：克隆、构建、审计、保存证据 |
| `evidence/` | 本次复现的原始日志与摘要 |

## 复现

依赖：Linux x86_64、`elan`、`git`、`curl`、`gcc`、`python3`（含 `numpy`）、`zstd`，
约 25 GB 磁盘，建议 64 GB 以上内存（单模块峰值可达 20 GB）。

```shell
cd fourcolor-lean/audit
./run.sh
```

或分步：

```shell
git clone https://github.com/corun1024/4ct
cd 4ct
git checkout 3db71e0a786a1f19be04c5266fe3d06fed216885
lake exe cache get
JOBS=16 MEMORY=80 bash build.sh
lake env lean /path/to/fourcolor-lean/audit/Audit.lean
```

网络受限时，先设置 `http_proxy` / `https_proxy`；`Audit.lean` 的路径换成
本仓库中的实际路径。

## 本次任务的 token 与成本（估算）

截至 2026-11-17 19:11（本条回复本身尚未计入），DSH 会话 `tokenUsage` 记录：

| 项 | 数值 |
|---|---:|
| 未缓存输入 | 467,323 tokens |
| 缓存读取 | 512,394,880 tokens |
| 输出 | 299,036 tokens |
| 合计 | 513,161,239 tokens |
| 缓存命中率 | 99.909% |

按 DeepSeek 官方 CNY 列表价（`deepseek-flash`：缓存命中 ¥0.04 / 未命中 ¥2 / 输出 ¥8 每百万 token；非高峰半价），本次任务在 18:06–19:11 的非高峰时段运行，估算 **约 ¥11.91**；若按高峰价约 **¥23.82**。这是列表价估算，不是平台账单。

## 参考来源与原创边界

- **主体形式化**：`corun1024/4ct`（Chris Emery，MIT，commit `3db71e0`）。它是 Gonthier–Werner 的 Coq 证明 [`math-comp/fourcolor`](https://github.com/math-comp/fourcolor) 的 Lean 4 移植，跟随 Robertson–Sanders–Seymour–Thomas 的 633 构型证明。
- **交叉比较**：`RBarish-UTokyo/FourColorTheorem-Lean4`（Robert D. Barish，Apache-2.0，commit `20fa345`）。本仓库只做源码/文档层面的比较，未完整构建。
- **工具与库**：Lean 4 `v4.34.0-rc2`、Mathlib `141f6b6`。
- **本仓库新增**：`audit/Audit.lean`、`audit/run.sh`、论文、图表、DOCX/PDF 构建脚本与证据日志；**没有**从零重写 59 万行证明。

## 许可证与归属

- 上游 `corun1024/4ct`：MIT License，Copyright (c) 2026 Chris Emery；
- 上游证明跟随 Georges Gonthier 与 Benjamin Werner 的 Coq 证明
  [`math-comp/fourcolor`](https://github.com/math-comp/fourcolor)（CeCILL-B）；
- 本目录的审计脚本、论文与图表由本仓库维护，按仓库约定发布；
  引用上游代码或数据时请保留其许可证与归属。

本仓库的 `Audit.lean` 只检查上游构建产物，不修改上游源码。
