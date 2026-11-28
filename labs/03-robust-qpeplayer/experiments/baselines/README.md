# 外部基线（Theseus / SDPRLayers / BPnP / Pyatov / POLI）

本目录只放**脚本、原始结果、日志**；仓库 clone 在
`/mnt/storage1/syk/third_party/`，不 vendor 进 worktree。
汇总表在 lab 级 `../../results/`：

- `../../results/baselines.md` — 主表 + 解读（人读）
- `../../results/baselines.json` — 机器读，所有数字直接来自原始 JSON
- `../../results/availability.md` — 各基线可得性、精确阻断、任务域
- `../../results/rounds.json` — 6 个 measured/blocked 轮次

canonical 是 lab 级 `../../results/`；同内容副本也放在本目录 `results/`，
方便就近查看（以 canonical 为准）。

## 1. 文件

| 文件 | 作用 |
|---|---|
| `run_reference.py` | Kabsch / RANSAC-1000 / RANSAC-adaptive / IRLS-GM / GNC-GM；`--trials/--out` |
| `theseus_cost.py` | Theseus per-point GM 目标 + 解析 Jacobian + 自检 |
| `run_theseus.py` | Theseus LM（GT / 随机初始化），20 trials |
| `run_sdpr_forward.py` | SDPRLayers + SCS 前向，N=20/50/100 |
| `make_baselines_table.py` | 汇总 → `../../results/baselines.{json,md}`、`rounds.json` |
| `check_sdpr_kabsch_equivalence.py` | 配对校验 SDPR(W=I) vs Kabsch → `results/sdpr_kabsch_equiv.json` |
| `results/reference_n20.json` | 5 个参考方法，20 trials，N=100 |
| `results/theseus.json` | Theseus GT/rand，20 trials，N=100 |
| `results/sdpr_forward.json` | SDPR 前向，N=20/50/100 |
| `results/sdpr_kabsch_equiv.json` | SDPR vs Kabsch 配对差（80 对） |
| `results/*_raw.jsonl` | 每 trial 一条记录 |
| `logs/` | 安装 / 运行日志 |

## 2. 环境

| 用途 | 解释器 / 环境 |
|---|---|
| 参考方法、汇总 | `/home/syk/miniconda3/envs/WM/bin/python` |
| Theseus | `/mnt/storage1/syk/venvs/theseus/bin/python`（theseus 0.2.3, torch 2.11, CPU） |
| SDPR | `/home/syk/miniconda3/envs/sdpr/bin/python`（numpy 1.26.3, torch 2.5.1+cpu, SCS 3.3.1） |

Theseus 的 `scikit-sparse` 因缺 `cholmod.h` 装不上，本次用
`th.CholeskyDenseSolver`；`import theseus` 可用。SDPR 无 Mosek license，
用 SCS（论文 §VI-B4 设置），只跑前向。

## 3. 复现

```bash
cd labs/03-robust-qpeplayer/experiments/baselines

# (1) 参考方法，20 trials，与 Theseus 同实例
/home/syk/miniconda3/envs/WM/bin/python run_reference.py \
  --trials 20 --out results/reference_n20.json

# (2) Theseus-GM，GT + 随机初始化
THESEUS_TRIALS=20 /mnt/storage1/syk/venvs/theseus/bin/python run_theseus.py

# (3) SDPR + SCS 前向（先 patch fresh clone；见 availability.md §2.1）
PYTHONPATH=/mnt/storage1/syk/third_party/sdprlayer/src \
  /home/syk/miniconda3/envs/sdpr/bin/python run_sdpr_forward.py

# (4) 汇总表
/home/syk/miniconda3/envs/WM/bin/python make_baselines_table.py
```

SDPR 的 fresh clone 需要 `repro/patches/0001` + `0003`；只打 0001 会在
构造 `SDPPoseEstimator` 时报 `NameError: name 'SDPRLayer' is not defined`。

## 4. 结果速览（N=100，20 trials）

| p | GNC-GM | RANSAC-adaptive | Theseus(rand) | Theseus(GT) | SDPR-SCS |
|---:|---:|---:|---:|---:|---:|
| 0.0 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 0.3 | 1.00 | 1.00 | 0.80 | 1.00 | 0.00 |
| 0.5 | 1.00 | 1.00 | 0.55 | 1.00 | 0.00 |
| 0.7 | 1.00 | 1.00 | 0.25 | 1.00 | 0.00 |

- Theseus 随机初始化在 p≥0.3 落局部极小；GT 初始化是**局部盆地检查**，
  不是全局恢复。
- SDPR 在 `W_i=I, w=1` 时目标与 Kabsch 相同（配对最大差 1.5e-5°），
  p≥0.3 的失败是非鲁棒 LS 的预期行为；SCS 前向 0/208 异常。
- 时间口径：Theseus 为 solver-only（目标构建约 6 ms/trial 未计），
  GNC/RANSAC/SDPR 含完整调用。

## 5. 未进入本表的基线

- **BPnP**：只有 PnP（3D-2D），无 3D-3D 路径。
- **Pyatov / ihls**：两视图（2D-2D），权重在 Google Drive，需
  ScanNet/MegaDepth。
- **POLI**：逐点 Gaussian 几何估计器（给 SDPR 提供 `W_i`），未找到官方
  代码/权重。

逐条证据与精确报错见 `../../results/availability.md`。
