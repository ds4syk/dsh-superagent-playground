# 复现与审计证据

本目录保存 2026-11-17 在实验室服务器上独立复现上游 `corun1024/4ct`
（commit `3db71e0`）时产生的原始文本。所有文件都是命令的直接输出，
没有手工改写。

| 文件 | 内容 |
|---|---|
| `environment.txt` | 主机、CPU、内存、Lean/Lake/git/gcc/Python 版本，上游 commit |
| `cache-get.log` | `lake exe cache get` 的完整输出 |
| `build.log` | `build.sh` 的完整输出（含证书生成、编译、`check.sh`） |
| `audit.txt` | 本仓库 `audit/Audit.lean` 的完整输出 |
| `summary.txt` | 公理、源码扫描、模块与行数统计 |
| `leanchecker.txt` | （可选）Lean 独立检查器对最终模块的重放结果 |

上游源码不复制进本仓库；`audit/run.sh` 会按 `upstream.lock` 克隆并检出。
