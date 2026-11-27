# P1 实验套件复现报告（REPRO）

> 日期：2026-11-27 · 工作区：`wt-p1-repro`（branch `opt/p1-repro`，HEAD `c4cca04`）
> 范围：`labs/01-laplace-qpeplayer/experiments/` 下全部 7 个实验脚本的 clean-cwd 可执行性、输出落盘位置、墙钟时间、确定性与脆弱点；新增单命令 harness `run_all.sh`。
> 本轮**没有修改任何现有 `.py` 文件、没有 commit、没有 push**；发现的问题只记录在本文件并给出最小修复建议。

## 0. 一分钟结论

| 检查项 | 结论 |
|---|---|
| 6 个快速脚本 clean-cwd（空目录）执行 | **全部通过**，rc=0，无 cwd 残留文件 |
| 7 个脚本总墙钟 | 快速套件 **34.2–38.0 s**（随同机负载）；再加 LINEMOD **42.1 s**（LINEMOD 6.5 s，远低于 3 min，无需 srun） |
| 输出落盘 | 全部写到脚本自身位置对应的 `results/`，与 cwd 无关 |
| 确定性（`run_main.py` 跑两次 + 已提交结果对比） | **通过**：非计时字段逐位相同（max abs diff = 0）；其余脚本输出与已提交版本 **md5 完全一致** |
| 已提交 `results/` 是否被覆盖 | **否**：harness 在临时副本里跑；已提交结果 hash 前后不变 |
| 是否发现真 bug | **是，1 个实质 bug**：`real/linemod/run_symmetry.py` 的“交叉搜索”变量恒非负，永远找不到交叉，所以空表是**代码 bug**，不能当作“数据不支持”的负面证据（B1） |
| 其他脆弱点 | B2 `verify_q6.py` 不建 `results/` 目录；B3 `run_main.py` 的 fit 分支打印不存在文件且默认参数下会崩；B4 `main_table.json` 写入非标准 `NaN`；B5 `.gitignore` 反选导致 `code/__pycache__` 变 untracked；B6 LINEMOD 数据路径硬编码；B7 `run_main.py` 无显式 `sys.path` insert |

## 1. 一键复现（新增 `run_all.sh`）

### 1.1 命令（任意 cwd）

```bash
# 从任意目录执行都可以；默认用 WM python
cd /tmp/p1_clean
/home/syk/miniconda3/envs/WM/bin/python -V   # Python 3.10.22

EXP=/mnt/storage1/syk/.superagent-work/wt-p1-repro/labs/01-laplace-qpeplayer/experiments
"$EXP/run_all.sh"                    # 快速合成 + theory 套件（默认）
"$EXP/run_all.sh" --with-linemod     # 追加 real/linemod（需要 BOP LINEMOD）
"$EXP/run_all.sh" --no-compare       # 跳过与已提交结果的数值比对
PYTHON=/path/to/python P1_STEP_TIMEOUT=1200 "$EXP/run_all.sh"   # 换解释器/超时
```

实测（本轮）：

```text
run_all.sh (默认)          exit 0，total wall 34.2 s（低负载） / 38.0 s（末次，负载较高）
run_all.sh --with-linemod  exit 0，total wall 42.1 s
```

失败路径已实测：`P1_STEP_TIMEOUT=0.05 run_all.sh` 得到每个脚本 `rc=124`、`missing: ...`，harness **exit 1**，且不写已提交结果。

### 1.2 输出与退出码

- 日志目录：`experiments/results/logs/<YYYYmmdd-HHMMSS>/`
  - `run.log`（汇总表 + 每步 rc/墙钟 + 失败脚本 stderr 尾部）
  - `env.txt`（python/numpy/scipy/matplotlib/cvxpy 版本、host、git head/branch）
  - `<script>.stdout` / `<script>.stderr`（逐脚本原始输出）
  - `outputs/`（本次 fresh 输出：`results/`、`theory_results/`，加 LINEMOD 时还有 `linemod_results/`）
  - `outputs.sha256`、`results_before.sha256`、`results_after.sha256`
- 退出码：`0` = 全部 rc=0 + 输出齐全非空 + clean-cwd 未被写 + 非计时数值与已提交结果一致 + 已提交结果 hash 未变；否则 `1`。
- 默认（不加 `--with-linemod`）执行：`code/run_main.py`、`code/run_ablation.py`、`theory/run_unequal.py`、`theory/run_uniform.py`、`theory/run_fold.py`、`theory/verify_q6.py`。

### 1.3 设计决定：临时副本，而不是 in-place + 备份/恢复

`run_all.sh` 把整个 `experiments/` 树 `tar` 复制到 `mktemp -d` 目录，在副本里跑；脚本的输出因此落在副本的 `results/`、`theory/results/` 下，跑完再复制到 `results/logs/<ts>/outputs/` 归档。

理由：

1. **零风险**：已提交的 `results/main_table.json`、`raw_grid.json`、PNG 等不会被覆盖；harness 结束前用 sha256 manifest 前后比对，被改动会直接 FAIL。
2. **避免“备份/恢复”失败模式**：如果脚本崩溃或 harness 被 Ctrl-C，备份可能没恢复，in-place 方案会留下半新半旧的结果。
3. **clean-cwd 检查更自然**：每个脚本都在一个空 cwd 下执行，能真实暴露相对路径/`results/` 目录缺失问题（B2 就是这么发现的）。
4. 想更新已提交结果时，仍按 §1.4 的逐脚本命令手动 in-place 跑，然后 review diff——这是显式的、可控的操作，而不是 harness 的副作用。

harness 还做了两个额外检查：

- 每个脚本运行前先删掉副本里它自己的旧输出，防止“脚本崩溃、旧数字还在”被误判为通过；
- fresh 数值与已提交结果比对时忽略计时字段（`t_*` / `*_ms`），rtol 1e-9；本轮默认套件 8/8、含 LINEMOD 9/9 个 JSON 全部 `OK`。

### 1.4 逐脚本手动复现（clean cwd，本轮实测命令）

```bash
EXP=/mnt/storage1/syk/.superagent-work/wt-p1-repro/labs/01-laplace-qpeplayer/experiments
PY=/home/syk/miniconda3/envs/WM/bin/python
mkdir -p /tmp/p1_clean && cd /tmp/p1_clean

"$PY" "$EXP/code/run_main.py"          # -> $EXP/results/{main_table.json,md,png,raw_grid.json}
"$PY" "$EXP/code/run_ablation.py"      # -> $EXP/results/{ablation.json,md}
"$PY" "$EXP/theory/run_unequal.py"     # -> $EXP/theory/results/{uniform_unequal.json,uniform_unequal_table.md}
"$PY" "$EXP/theory/run_uniform.py"     # -> $EXP/theory/results/{uniform.json,uniform_table.md,deltaJ_sweep.json,deltaJ_sweep.md}
"$PY" "$EXP/theory/run_fold.py"        # -> $EXP/theory/results/{fold.json,fold_table.md}
"$PY" "$EXP/theory/verify_q6.py"       # -> $EXP/theory/results/{q6_verify.json,q6_verify.md}

# 真实数据（BOP LINEMOD，3.3 GB，位于 /mnt/storage1/syk/data/bop/lm）
"$PY" "$EXP/real/linemod/run_symmetry.py"   # -> $EXP/real/linemod/results/{symmetry.json,md}（当前为空表，见 B1）
```

## 2. 环境

本轮所有测量均在实验服务器 `ls`（40 核，Linux 6.8.0-107-generic）上完成：

| 项 | 值 |
|---|---|
| python | 3.10.22（`/home/syk/miniconda3/envs/WM/bin/python`） |
| numpy | 2.2.6（脚本使用 `np.trapezoid`，**要求 numpy ≥ 2.0**；1.x 无此函数） |
| scipy | 1.15.3 |
| matplotlib | 3.10.9（`run_main.py` 显式 `matplotlib.use("Agg")`，headless 正常） |
| cvxpy | 1.7.5（本套件 7 个脚本**均未使用** cvxpy，仅列入环境） |
| Pillow / plyfile | 12.3.0 / 未声明版本（仅 LINEMOD 脚本用） |
| git | `opt/p1-repro` @ `c4cca045e8e0df9d061994eefbf58d796aff71a3` |

证据：`experiments/results/logs/20261127-205351/env.txt`。

## 3. 逐脚本 clean-cwd 实测

在 `/tmp/p1_clean`（空目录）下用绝对路径执行，测量 wall time、rc、输出位置：

| 脚本 | wall（多次实测，s） | rc | 输出目录（相对 `experiments/`） | clean-cwd |
|---|---|---|---|---|
| `code/run_main.py` | 15.7 / 15.9 / 15.9 / 16.2 / 16.3 / 19.1 | 0 | `results/` | ✅ |
| `code/run_ablation.py` | 4.2 / 4.3 / 4.6 / 4.8 / 7.1 | 0 | `results/` | ✅ |
| `theory/run_unequal.py` | 1.3 / 1.3 / 1.5 / 1.6 / 1.9 | 0 | `theory/results/` | ✅ |
| `theory/run_uniform.py` | 9.2 / 9.2 / 9.2 / 9.3 / 11.0 | 0 | `theory/results/` | ✅ |
| `theory/run_fold.py` | 3.2 / 3.2 / 3.2 / 3.4 | 0 | `theory/results/` | ✅ |
| `theory/verify_q6.py` | 0.4 / 0.4 / 0.5 / 0.5 | 0 | `theory/results/` | ✅（前提：`theory/results/` 已存在；单独在干净副本里跑会失败，见 B2） |
| `real/linemod/run_symmetry.py` | 6.5 / 6.6 / 6.7 | 0 | `real/linemod/results/` | ✅（但输出空表，见 B1） |

墙钟波动主要来自同机负载，单次偏差约 ±10%–20%，不要当 benchmark 用。快速套件由 harness 串行跑完为 **34.2 s**（低负载）到 **38.0 s**（末次，负载较高）；加 LINEMOD 为 **42.1 s**。LINEMOD 虽有 3.3 GB 数据和多次 `find_modes`，实测 6.5 s，**不需要 srun**。

### cwd 独立性结论

- 7 个脚本都把输出锚定到 `Path(__file__).resolve().parent`（`run_symmetry.py` 用 `HERE.parents[1]/"code"`），因此从任何 cwd 执行都写到正确位置；运行后 `/tmp/p1_clean` 为空（harness 的 cwd-pollution 检查也为空）。
- 没有任何脚本依赖 cwd 下的相对路径；也没有脚本写 cwd（无临时文件、无 `__pycache__` 落在 cwd）。
- `run_main.py` 没有显式 `sys.path.insert`，它 `import instance, qpep_core` 依赖 Python 把**脚本所在目录**放进 `sys.path[0]`。按“脚本路径”调用时任何 cwd 都正常；但如果把它当模块 `import`（如从别的目录 `python -c "import run_main"`），会 `ModuleNotFoundError`，需要先把 `experiments/code` 加进 `PYTHONPATH`（实测：加 `sys.path.insert` 后 import 成功）。
- `run_ablation.py` 自己 `sys.path.insert(0, HERE)` 后再 import `qpep_core/instance/run_main`，import 顺序安全；import `run_main` 时有副作用会在 import 阶段创建 `results/`（无害）。
- `verify_q6.py`、`run_symmetry.py` 也显式 insert 了 code 目录，import 顺序安全。
- matplotlib：`run_main.py` 在 `import pyplot` 之前 `matplotlib.use("Agg")`，本机无 DISPLAY 也正常；未发现 backend 问题。

## 4. 确定性

方法：

1. 在 `/tmp/p1_clean` 连续跑两次 `run_main.py`，对 `results/main_table.json`、`raw_grid.json` 做结构化比较，**剔除所有计时字段**（键名以 `t_` 开头或以 `_ms` 结尾），其余浮点要求逐位相等；
2. 同一比较再对“已提交版本”（备份）做一次；
3. 其余脚本的输出与已提交版本做 md5 对比。

结果：

| 对比 | 非计时字段差异 | 最大绝对差 |
|---|---|---|
| `main_table.json` run1 vs run2 | 0 | 0.0 |
| `raw_grid.json` run1 vs run2 | 0 | 0.0 |
| `main_table.json` fresh vs 已提交 | 0 | 0.0 |
| `raw_grid.json` fresh vs 已提交 | 0 | 0.0 |
| `ablation.json/.md`、theory 5 个 JSON/MD、`symmetry.json/.md` | md5 与已提交**完全一致** | — |

- **只变的字段**：`raw_grid.json` 每行的 `t_laplace_ms / t_hard_ms / t_mc_ms`，以及 `main_table.json` 同名的中位数（far ≈13–18 ms、near ≈22–23 ms、at ≈58–62 ms，run-to-run 波动 ±10%–20%）。`main_table.md` 的差异也只在这三列。
- 原因：计时用 `time.perf_counter()`，属于固有非确定性；数值部分没有不受控随机源。
- 种子是固定的：`Instance(seed=20261127)`；MC 用 `default_rng(20261127)` 与 `default_rng(1000+s)`；`verify_q6.py` 用 `default_rng(0)`；`run_symmetry.py` 用 `default_rng(20261127)`。注意 `instance.find_modes()` 每次调用都重新 `default_rng(self.seed+7)`，所以**每次调用的初始点云完全相同**，这也是模式枚举可重复的原因。
- `main_table.png` 的字节内容与已提交版本不同（79,348 B vs 87,700 B；应为 matplotlib 版本/渲染元数据差异），但它画的数据与 JSON 一致；**不要用 PNG 字节做复现判据**。

**判定：确定性通过（非计时字段逐位可复现，跨两次运行与跨时间均一致）。**

## 5. 发现的问题（未修改任何现有 `.py`，只给最小修复建议）

### B1（实质 bug）`real/linemod/run_symmetry.py` 的交叉搜索变量恒非负，永远找不到交叉

- 现象：9 帧里 8 帧因 `obj_id` 不在 `{10,11}` 被静默跳过，剩下的 `000010`（2 张图）也永远找不到 bracket，`symmetry.json` 写成 `[]`，`symmetry.md` 只有表头，**rc=0、静默成功**。README 把它记为“negative/inconclusive attempt（可能数据不支持）”。
- 根因（代码级）：`gap_at()` 里

  ```python
  Js = np.sort(np.asarray(mm["J"]))
  return float(Js[1] - Js[0])
  ```

  排序后的两个最小能量之差**恒 ≥ 0**；随后 `if vals[i] * vals[i+1] <= 0` 的变号判据几乎永远不成立（只有严格相等即 0 才成立），所以 `bracket` 永远 `None`，循环 `continue`，最后输出空表。
- 反证（用“模式连续性贴标签”的有符号 gap 实测，帧 `000010`，obj 10）：
  - img `000015`：θ=0.1 → +1.256e-2，θ=0.9 → −2.660e-3，**在 θ≈0.69 穿过 0**（0.6: +1.43e-3 → 0.7: −2.5e-4）；
  - img `000018`：θ=0.1 → +6.1e-4，θ=0.9 → −9.2e-3，**在 θ≈0.21 穿过 0**（0.2: +9e-5 → 0.3: −5.6e-4）。
  说明两个模式确实会交叉、数据可用；空表**主要是代码 bug，而不是“数据没有双峰”**。
- 最小修复（方向性，约 6 行，未落地）：在 `build_objective` 之后一次性求两个分支参考极小
  `qA = modes(M0a) 的 argmin`、`qB = modes(M0b) 的 argmin`；`gap_at(th)` 内按 `argmax |q·qA|` / `argmax |q·qB|` 给两个模式贴标签，返回 `J[iB] - J[iA]`（有符号）。同时把 `FRAMES` 改成真正的目标物体序列 `["000010", "000011"]`。
- 附带问题：
  - `FRAMES` 里的 `000001/000002/000003/000005/000006/000008/000013/000015` 只含 obj 1/2/3/5/6/8/13/15，全部不在 `OBJ={10,11}`，被 `continue` **静默跳过**；glue（obj 11）对应的 `000011` 根本不在列表里，即当前代码**从未测试 glue**。
  - 每个 LINEMOD test frame 的每张图只有该帧的单一物体，`for img, gt in gt_all.items(): if n_img >= 2: break` 的“每帧取前 2 张图”与实验目标无关，纯属任意选择。
  - 第 184 行 `dld = np.array([...])  # placeholder` 紧接着被覆盖，是死代码。
- 影响：`real/linemod` 的负结论目前**不可引用**；README 的“可能原因”三条猜测应替换为上述确定性原因。

### B2 `theory/verify_q6.py` 不创建输出目录

- 现象：在 `theory/results/` 不存在的新副本里单独运行 → `FileNotFoundError: .../theory/results/q6_verify.json`（已实测，rc=1）；其他脚本都会 `mkdir(parents=True, exist_ok=True)`。
- 当前仓库能跑通只是因为 `theory/results/` 已被提交（有文件所以目录存在）。一旦清理 results 目录，harness/CI 会失败。
- 最小修复：在 `HERE = Path(__file__).resolve().parent` 之后加一行
  `(HERE / "results").mkdir(parents=True, exist_ok=True)`。

### B3 `run_main.py` 的 fit 分支：打印不存在的文件，默认参数下会崩

- `FIT_SEEDS = 0`、`FIT_STEPS = 0` 时，主流程结尾仍打印 `wrote ... fit_table.md`，但 `fit_table.md` 从未生成（日志/输出有误导性）。
- 若把 `FIT_SEEDS` 改为 >0 而 `FIT_STEPS` 仍为 0：`for step in range(0)` 不执行，`losses.append(loss)` 中 `loss` 未定义 → `UnboundLocalError`（静态阅读结论，未执行）。
- 影响：只影响被关闭的诊断路径；但文档/日志应避免声称生成了不存在的文件。

### B4 `results/main_table.json` 含非标准 JSON 的 `NaN`

- `med()` 对空/全 NaN 列返回 `float("nan")`，`json.dump` 默认写出裸 `NaN`（`main_table.json` 的 `at` 行 `hard_grad_rel`）。
- Python `json` 能读；标准 JSON 解析器会拒绝（实测：`json.loads(..., parse_constant=raise)` 抛 `non-standard constant: NaN`）。下游用 jq/JS/其它语言读表会炸。
- 建议：NaN 写成 `null`（`allow_nan=False` + 自定义编码，或聚合时用 `None`）。

### B5 `.gitignore` 反选把 `code/__pycache__/` 变成 untracked

- `.gitignore` 第 1 行忽略 `__pycache__/`，但第 13–14 行 `!labs/01-laplace-qpeplayer/experiments/code/` 与 `!.../code/**` 又把这些目录下的所有文件反选回来。跑完脚本后 `git status` 出现 `?? labs/01-laplace-qpeplayer/experiments/code/__pycache__/`（`real/linemod/__pycache__` 则被正常忽略）。
- 影响：仅卫生问题，但未来 `git add experiments/code` 可能把 `.pyc` 带进提交。
- 建议：在该反选块后补 `__pycache__/`，或把 `**` 改成 `*.py`/`!*.py` 形式。
- 本轮已删除运行产生的两个 `__pycache__/`。

### B6 `real/linemod/run_symmetry.py` 数据路径硬编码

- `ROOT = Path("/mnt/storage1/syk/data/bop/lm")`，无环境变量覆盖；换机器/换数据盘直接不可跑（该目录不存在时会在 `load_model` 以 `FileNotFoundError` 失败）。
- 另需 `PIL`、`plyfile`，这两个依赖没有写进 `real/linemod/README.md`。
- 建议：`ROOT = Path(os.environ.get("BOP_LM_ROOT", "/mnt/storage1/syk/data/bop/lm"))`。

### B7 `run_main.py` 依赖隐式 `sys.path`

- 见 §3：作为脚本执行没问题；作为模块被 import 时（无 `PYTHONPATH`）`import instance` 失败。
- 建议：与 `run_ablation.py` 一致，在 import 前加 `sys.path.insert(0, str(HERE))`。

### B8 计时字段的固有非确定性（非 bug，但需在引用数字时说明）

- `t_*_ms` 每次运行都变（±10%–20%）；`main_table.md` 的计时列也随之变。任何“Laplace 比 hard 快/慢”的论断不能基于这些单次数字，需要多次重复或固定 benchmark。

## 6. 复现边界与负面结果（诚实声明）

- **LINEMOD 没有真实数据证据**：`symmetry.json` 为 `[]`；本轮进一步定位到这是 B1 的代码 bug，但**在修复并重跑之前，不要引用任何真实数据结论**，也不要把该负结果写成“数据不支持多峰”。修复方向已给出，是否落地由后续轮次决定。
- 本套件**没有用到 cvxpy/GPU**；“零升维 + 枚举 + Laplace”主表仍是合成双峰实例（`experiments/README.md` 的口径不变）。
- 墙钟数字只代表本机本次负载下的量级，不是性能对比。
- `run_main.py` 的曲线图 PNG 跨环境字节不可比；JSON/MD 中的数值可比。
- 本轮为审计曾 in-place 重跑过各脚本，结束后已用运行前备份恢复 `results/` 与 `theory/results/`（sha256 与备份完全一致）；最终 worktree 中只新增 `run_all.sh`、本文件、`results/logs/`。

## 7. 证据文件

| 文件 | 内容 |
|---|---|
| `experiments/run_all.sh` | 本轮新增的单命令 harness |
| `experiments/results/logs/20261127-205551/` | 默认快速套件 PASS 日志（total 38.0 s；低负载时 34.2 s） |
| `experiments/results/logs/20261127-205351/` | `--with-linemod` PASS 日志（total 42.1 s，含空表告警） |
| `experiments/results/logs/*/outputs.sha256` | fresh 输出 hash 清单 |
| `/tmp/p1_audit_20261127-204657/` | 审计用备份与逐次运行的 stdout/stderr、`run_main` 两次输出快照（临时目录，非交付物） |
