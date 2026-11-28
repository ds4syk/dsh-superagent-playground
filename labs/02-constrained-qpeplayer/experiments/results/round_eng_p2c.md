# P2 phase-3 工程轮（性能 / 复现）— 2026-11-27

- worktree：`/mnt/storage1/syk/.superagent-work/wt-p2c-eng`，branch `opt/p2c-eng`，base HEAD `17d7ffc`
- 环境：`/home/syk/miniconda3/envs/sdpr/bin/python`（numpy 1.26.3 / scipy 1.13 / cvxpy 1.5.4 / matplotlib 3.8.4）；
  全部单线程（`OMP/OPENBLAS/MKL/NUMEXPR=1`）、login node `nice -n 10`（Slurm `cpu` 分区被他人任务占满，`sinfo` = `mix`，未排队）
- 未 commit / 未 push；未改动他人拥有的 `theory/`、`verify/`、`deliverables/`、`paper/`、`certificate.py`、`isotropic*.py`、`run_rigorous_real.py`
- 本轮只做“测量 + 可复现性 + 结论固化”；除 `run_scp.py` 的结论文本与 `run_all.sh` 外，不改算法代码。所有数字来自本轮实测或明确标注的已提交产物。

## 5 轮记录

| round | hypothesis | change | command | metric before→after | keep/revert | artifact |
|---|---|---|---|---|---|---|
| S1 | 换成优化后的 quartic 枚举器后，exact 内层的端到端倍数应 ≈ 旧倍数/2.9；回退率是否随之下降？ | 用 `quartic_roots` batched-matmul 版重测 exact-vs-multistart：S20（p=16, 4 seeds）+ S21（p=8/64, 各 2 seeds）+ S22/S23/S24；并重测 unpinned 对照 | `nice -n 10 .../python run_exact_alm.py --phases s20 s21 s22 s23 s24`<br>`... --phases s20 s23 s24 --tag unpinned --no-pin` | 单次 162-path 枚举 **652 → 221 ms**（2.95×）；S20 总时比 **37.3×–64.2× → 14.6×–23.3×**（median 49.6→19.3，2.6×）；S20+S21 **13.6×–23.3×**；回退 **25/114 = 21.9%**（S20 13/63 = 20.6%）与优化前逐 call 相同；Δ 平价 max‖Δq‖ 7.9e-10、max|ΔJ| 1.8e-12 | **keep**：纯性能优化，根数/回退/解均不变；exact 仍只作离线证书 | `results/exact_alm.json` / `.md`、`results/round_exactalm.md`、`results/exact_alm_unpinned.json` / `.md` |
| S2 | 参数化证书缓存能否在数值/证书决定不变的前提下省 SDP 时间；是否可默认开？ | 复测 `cert_cache --bench`（p=16, n=20），并在 `run_main` 里做同 seed 4-trial 的 cache vs 默认冷解对照 | `nice -n 10 .../python cert_cache.py --bench --n 20 --p 16`<br>`QPE_CERT_CACHE=1 QPE_TRIALS=4 QPE_MAIN_P=16 QPE_NOISE=0 QPE_PARETO=0 .../python run_main.py`（默认路径同参数再跑一次） | 冷解 43.3 ms → warm 稳态 **8.9 ms（4.9×）**；旧测 4.2×；\|Δvalue\| max 3.6e-10；主表同 seed 对照 \|ΔShor\| max **3.43e-9**、certified 标志 **4/4 一致**、四个方法 J 逐位相同；SDP 39.8 → 24.9 ms（含首 call 编译） | **keep 代码，flag 默认保持 OFF**：数值非逐位平价（~1e-9），按规则不默认开 | `results/perf_cert_cache.json` / `.md` |
| S3 | 补全被中断的 SCP 压力档：SCP 在高噪声/偏差档的 cert/可行/时间、失败模式、重启能否救回？ | 重跑完整 `run_scp.py`（主比较 p=8/16/32 ×50 + 3 档 stress ×20 + 2/3 重启变体）；把生成结论里“重启能补部分差距”的硬编码改为数据驱动 | `nice -n 10 .../python run_scp.py` | 主比较 SCP **98/98/98%** vs ALM 100%，127/149/158 vs 208/229/244 ms；stress **85/80/55%** vs ALM 100/100/100%，139/192/306 vs 212/294/233 ms（n050 反而更慢）；失败 **19** 个（不可行 1、未收敛 18，全部撞 max_iter；镜像初值 18）；**+2/+3 重启救回 0** | **keep 表**：普通档 SCP 是强 baseline；压力档诚实报告其失败；重启无效的结论已写死为数据驱动文本 | `results/scp_baseline.json` / `.md`、`results/run_scp.log`；`code/run_scp.py`（结论文本） |
| S4 | 从干净目录（/tmp 拷贝）能否复现 40/40、主表/图，且不写回源码树？`run_all.sh` 是否列全当前脚本？ | 在 `/tmp` 拷贝里跑 validate / make_results / run_main；重写 `run_all.sh`（核心 4 步 + `QPE_FULL=1` 扩展 7 步、`QPE_NO_SLURM=1`、cwd 无关） | `cp -a experiments /tmp/p2c_repro/experiments`<br>`cd /tmp/p2c_repro/experiments && $PY code/validate_code.py`<br>`$PY code/make_results.py`<br>`QPE_TRIALS=2 QPE_MAIN_P=16 $PY code/run_main.py`<br>`QPE_NO_SLURM=1 bash code/run_all.sh`（小参数 smoke） | validate **40/40**；make_results 重建 **10/10 逐字节相同**（3 表 + 7 PNG）；run_main 5.2 s，`raw_main.json`（46,643 B）/`raw_pareto.json`（1,851 B）落在副本；未发现 cwd bug（脚本均用 `__file__`）；`run_all.sh` 补上 7 个缺失步骤 | **keep**：只改 `run_all.sh`；cwd/path 无 bug 可修 | `code/run_all.sh`；`/tmp/p2c_repro` 复现日志 |
| S5 | 把本轮结论收成一页并让数字可追溯 | 新增本文件 + README §F（perf/repro 段） | — | 5 行轮次表 + F.1–F.5：exact 13.6×–23.3×/21.9%、cert 4.9×/3.4e-9、SCP 55–98%、RLT 1.5×–25.8×、clean-cwd 命令 | **keep** | `results/round_eng_p2c.md`；`README.md` §F |

## 1. S1 — exact 内层重测（优化枚举器）

命令与产物同上。核心数字：

| 指标 | 旧枚举器 | 新枚举器（batched matmul） | 变化 |
|---|---|---|---|
| S20 每个 trial 的 exact 总时（mean） | 10.278 s | **3.492 s** | 2.94× |
| 单次 162-path 枚举（mean / median） | 652 / 630 ms | **221 / 184 ms** | 2.95× / 3.4× |
| S20 总时比（exact / multistart） | 37.3×–64.2×（median 49.6） | **14.6×–23.3×**（median 19.3） | 2.6× |
| S20+S21 总时比 | 37.3×–64.2× | **13.6×–23.3×**（median 16.3） | — |
| S20 枚举 calls / fallbacks | 63 / 13（20.6%） | **63 / 13（20.6%）** | 不变 |
| S20+S21 枚举 calls / fallbacks | 114 / 25（21.9%） | **114 / 25（21.9%）** | 不变 |
| 平价 | max‖Δq‖ 7.9e-10、max|ΔJ| 1.8e-12 | 同左 | 不变 |
| 可行率一致 | 8/8 trials | 8/8 trials | 不变 |

要点：

- 加速只改时间，不改 homotopy：回退 trial 与优化前**逐个相同**（S20 seeds 0–3：0/4/4/5 次）。回退仍是“只找到 37–39 个有限根”的路径丢失，不是秩亏（114 个活跃集矩阵满秩，min eig 8.2e-04；KKT Jacobian 无奇异）。
- 每个枚举的独立 64-start 探针在 114 次中 **0** 次找到更低点；枚举也没有输给 64-start（0 次）。
- S22 成本杠杆重测：full exact 3.03 s/trial；hybrid `first` 1.12 s（比 full 便宜 2.56–2.98×，但仍比纯 multistart 慢 4.4–5.1×）；`stable2` 2.87 s（0.87–1.23×）；`every3` 1.23 s（2.34–2.65×）。`n_gamma=2` 把 seed1 的 4 次回退降到 0，代价 1.8–2.1×。
- unpinned 对照：12.8×–23.1×，exact 3.235 s/trial vs pinned 3.492 s → 枚举成本与线程 pinning 基本无关；绝对秒数随共享负载 ±25%。

## 2. S2 — 证书缓存（参数化 14×14 Shor SDP）

| 模式 | 时间 ms（median [IQR]） | \|Δvalue\| vs 同精度冷解 | \|Δvalue\| vs 主表默认冷解 | tightness λ₂/λ₁ max |
|---|---|---|---|---|
| 冷解（主表默认 eps=1e-9） | 42.9 [34.8, 46.3] | — | — | 1.0e-09 |
| 冷解（同精度 eps=5e-11） | 43.3 [39.4, 47.6] | — | — | 2.3e-11 |
| 缓存 warm（稳态，n−1） | **8.9 [7.6, 9.4]** | max 3.6e-10 | max 3.4e-9 | 4.3e-11 |
| 缓存 nowarm（诊断） | 13.0 [11.8, 15.1] | max 4.1e-10 | — | — |

- 稳态 **4.9×**（旧测 4.2×；冷解从 36.1→43.3 ms 是共享负载所致，warm 基本不变）；回退冷解 0 次，`is_dpp()=True`。
- 主表同 seed 4-trial 对照（p=16, noise=0.01, bias=0.08）：`|ΔShor|` max **3.43e-9**、max Δ(λ₂/λ₁) **2.21e-10**；`certified` 标志 0 处不同（4/4 sound）；unc/soft/proj/alm 的 J 与默认路径**逐位相同**。
- 耗时：SDP 39.8 → 24.9 ms（4-trial 均值；第 1 个 call 含 76 ms 编译；稳态 ~8.9 ms，50-trial 格摊薄后约 3.9×）。
- **默认 OFF**（`run_main.py` 中 `QPE_CERT_CACHE` 默认 `"0"`，本轮未改）：多 trial 的经验 warm 值并非逐位平价，按“除非逐位一致否则默认关”的规则保持关闭；开时 JSON 里标 `cached=True`。

## 3. S3 — SCP baseline + 压力档（完整）

| 格 | n | ALM 认证 | SCP 认证 | ALM 可行 | SCP 可行 | ALM ms | SCP ms | unc 镜像率 |
|---|---|---|---|---|---|---|---|---|
| p=8 | 50 | 100% | **98%** | 100% | 100% | 208.2 | 127.2 | 2% |
| p=16 | 50 | 100% | **98%** | 100% | 100% | 229.4 | 148.6 | 2% |
| p=32 | 50 | 100% | **98%** | 100% | 100% | 244.3 | 158.1 | 2% |
| n020_b120 | 20 | 100% | **85%** | 100% | 100% | 211.8 | 139.2 | 15% |
| n030_b180 | 20 | 100% | **80%** | 100% | 100% | 293.5 | 191.6 | 20% |
| n050_b250 | 20 | 100% | **55%** | 100% | **95%** | 232.7 | 306.3 | 40% |

- 普通档 SCP 可行、精度同档（p=16：双方 pen 0.00 mm / rot 4.17° / trans 10.63 mm），时间快 **1.5–1.6×**；每个 p 格只有 1 个失败，且都是同一个 seed=4（镜像初值、撞 max_iter=100、J−Shor≈0.52–0.53）。
- 压力档：σ/bias 越大，unc 镜像率 15%→40%，SCP 认证 85%→55%；n050 有 1 个 trial 迭代到不可行（viol=2.7e-1），SCP 时间反超 ALM（306 vs 233 ms）。
- 失败模式合计 **19**：不可行 1、未收敛 18、收敛但次优 0；初值镜像 18、输出镜像 0、撞 max_iter 19。
- 重启：`+2`/`+3` 扰动起点救回 **0** 个（19 个失败全部仍在），平均起点仅 1.04/1.06（普通档）、1.90/2.35（n050）。结论：瓶颈是局部收敛/迭代上限，不是起点数量。生成 md 的结论 5 已从硬编码“能补部分差距”改为数据驱动文本。
- 备注：p32 SCP 的 CLARABEL 在少数子问题打印 “Solution may be inaccurate”；最终解仍 100% 可行，证书按最终解计算，未影响认证结论。

## 4. S4 — clean-cwd 复现与 run_all

在 `/tmp/p2c_repro/experiments`（从 worktree 复制的 experiments 树）中：

| 步骤 | 结果 |
|---|---|
| `validate_code.py` | **40/40** 项通过；`results/validation.json` 写在副本 |
| `make_results.py`（读已提交 raw JSON） | `main_table.md/.csv/.json` + 7 张 PNG **10/10 与提交版逐字节相同** |
| `run_main.py`（`QPE_TRIALS=2 QPE_MAIN_P=16`） | 5.2 s；`raw_main.json`（46,643 B，2 trials × p16 + 4 噪声档）、`raw_pareto.json`（1,851 B，10 行）落在副本；配置 `n_trials=2, main_p=[16]` |
| 写回源码树 | 无（源码树 `main_table.*`/`validation.json`/`raw_main.json` mtime 未变） |
| cwd/path bug | 未发现：validate/run_main/make_results 全部用 `__file__` 定位 `results/` |

`run_all.sh` 重写：

- 核心 4 步保持：validation → main（50 trials/格）→ fb_table → make_results。
- `QPE_FULL=1` 追加 7 步：`run_quartic.py` + `make_quartic_tables.py`、`bench_inner.py`、`cert_cache.py --bench --n 20 --p 16`、`run_perf.py`、`run_scp.py`、`run_exact_alm.py --phases s20 s21 s22 s23 s24`、`results/make_p2_overview.py`（共 11 步）。
- `QPE_NO_SLURM=1` 时用 `nice -n 10` 直跑，不在被占满的 `cpu` 分区排队；默认仍 `srun`。
- 脚本自身 `cd` 到 `experiments/`（在任意目录调用均可），并固定 `OMP/OPENBLAS/MKL/NUMEXPR=1`。
- smoke test：小参数（`QPE_TRIALS=2 QPE_MAIN_P=8,16 QPE_NOISE=0 QPE_PARETO=0 QPE_NO_SLURM=1`）下第 1–3 步通过；第 4 步在“截断数据”下会被 `make_results.py` 的空噪声/Pareto 分支拒绝（`ValueError: x and y must have same first dimension`），这是 `make_results.py` 不支持截断 raw，不是 run_all 的问题；全量 `QPE_NOISE=1 QPE_PARETO=1` 路径已用提交数据单独验证 10/10 逐字节一致。

## 5. 结论、caveats、改动文件

**结论**

1. 枚举器优化是纯性能收益：exact 内层端到端从 37×–64× 降到 **13.6×–23.3×**，回退率/根数/解**逐项不变**——exact 仍只能作为离线正确性证书，不能当生产内层。
2. 证书缓存稳态 **4.9×**，数值差 ~1e-9（比阈值小 2 个数量级以上）、证书决定不变，但**默认仍关**；需要速度的批量/离线证书场景再开。
3. SCP 是普通档的强 baseline（98% 认证、快 1.5–1.6×），但在高噪声/偏差档掉到 55–85%，且重启救不回（0/19）——ALM 的差异化是镜像初值鲁棒性与 100% 认证。
4. RLT 成本随 p 二次增长（1.5×→25.8×），**只作可选/难题档**；默认证书用 base。
5. clean-cwd 复现通过（40/40、10/10 逐字节、raw 落副本）；`run_all.sh` 已列全当前脚本并支持 Slurm 被占时的直跑。

**Caveats**

- 单线程、共享 login node（load 6–13）；绝对毫秒 ±25%，同进程/同 run 的相对量是主结论。SDP/cert 的 warm 时间受冷解基线漂移影响（4.2×→4.9×）。
- 本轮不重跑主表 50-trial（raw_main.json 未动）；主表数字来自已提交 raw。make_results 的逐字节复现只证明“从 raw 重建”确定，不证明“重新采样主表”逐位相同（runtime 列必然漂移）。
- `exact_alm.md` / `round_exactalm.md` 已按新枚举器重生成；`run_exact_alm.py`（非本 subagent 所有）的 verdict 段落仍有旧硬编码（hybrid 4.7–6.8×/1.1–1.4×、“slowest observed trial 15.9 s”），以本文件 §1 为准。
- SCP 表含 `scp_restart`/`scp_restart3` 两个诊断变体；它们不改变主结论（救回 0）。
- `__pycache__/*.pyc` 因 import 被刷新，与本轮逻辑无关（已还原）。

**改动文件**

| 文件 | 说明 |
|---|---|
| `experiments/code/run_all.sh` | 重写：核心 4 步 + `QPE_FULL=1` 扩展 7 步；`QPE_NO_SLURM=1`；cwd 无关；单线程 |
| `experiments/code/run_scp.py` | 只改自动生成结论文本：重启救回改为数据驱动（本次 0/19）、压力档不可行计数不再写死 |
| `experiments/results/perf_cert_cache.json` / `.md` | 本轮复测（4.9×，|Δvalue| 3.6e-10） |
| `experiments/results/exact_alm.json` / `.md` / `round_exactalm.md` | 优化枚举器重测（S20–S24） |
| `experiments/results/exact_alm_unpinned.json` / `.md` | unpinned 对照重测 |
| `experiments/results/scp_baseline.json` / `.md` / `run_scp.log` | 完整 SCP（p32 + 3 档 stress）重跑 |
| `experiments/results/round_eng_p2c.md` | 本文件（新增） |
| `experiments/README.md` | 新增 §F 性能/复现（F.1–F.5） |

未改：`run_main.py`（`QPE_CERT_CACHE` 默认关，符合“非逐位平价不开”）、`cert_cache.py`、`quartic_roots.py`（优化已在 base）、`make_results.py`、`validate_code.py`。

**复现命令**

```bash
cd labs/02-constrained-qpeplayer/experiments/code
PY=/home/syk/miniconda3/envs/sdpr/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
nice -n 10 $PY run_exact_alm.py --phases s20 s21 s22 s23 s24
nice -n 10 $PY cert_cache.py --bench --n 20 --p 16
QPE_CERT_CACHE=1 QPE_TRIALS=4 QPE_MAIN_P=16 QPE_NOISE=0 QPE_PARETO=0 nice -n 10 $PY run_main.py
nice -n 10 $PY run_scp.py
```
