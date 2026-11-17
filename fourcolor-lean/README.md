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

## 目录

| 路径 | 内容 |
|---|---|
| `upstream.lock` | 上游仓库、commit、Lean 与 Mathlib 版本、许可证 |
| `audit/Audit.lean` | 独立审计：最终定理的类型、定义展开、公理闭包、`sorryAx` 扫描、声明计数 |
| `audit/run.sh` | 一键复现：克隆、构建、审计、保存证据 |
| `evidence/` | 本次复现的原始日志与摘要 |

## 复现

依赖：Linux x86_64、`elan`、`git`、`curl`、`gcc`、`python3`、`zstd`，
约 25 GB 磁盘，建议 16 GB 以上内存（单模块峰值可达 20 GB，建议 64 GB）。

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
JOBS=24 MEMORY=80 bash build.sh
lake env lean /path/to/fourcolor-lean/audit/Audit.lean
```

网络受限时，先设置 `http_proxy` / `https_proxy`；`Audit.lean` 的路径换成
本仓库中的实际路径。

## 许可证与归属

- 上游 `corun1024/4ct`：MIT License，Copyright (c) 2026 Chris Emery；
- 上游证明跟随 Georges Gonthier 与 Benjamin Werner 的 Coq 证明
  [`math-comp/fourcolor`](https://github.com/math-comp/fourcolor)（CeCILL-B）；
- 本目录的审计脚本、论文与图表由本仓库维护，按仓库约定发布；
  引用上游代码或数据时请保留其许可证与归属。

本仓库的 `Audit.lean` 只检查上游构建产物，不修改上游源码。
