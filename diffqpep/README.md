# diffqpep — 原子位姿层 / Face-Certified Differentiation

论文 *Atomic Pose Layers: Selection-Aware Differentiation for Quadratic Pose Estimation* 与中文研究知识库的复现代码。

## 核心结论

- **可微性属于最优面，不属于最优值**。QPEP 的 12×12 R-form 在一般 PnP 上值紧且解紧（rank 1、`‖RᵀR−I‖≈1e−11`、`det=±1`、`x*=±x_gt`）；13×13 齐次化的 last-row 得到 R=0 是恢复伪影，不是 tightness 失败。
- **planar PnP 有精确 flip 双覆盖**：两支代价相同（~1e−15），没有 image-only 梯度能选；需要 prior/cue。
- **q-form 无噪声 gap 是 vacuous**：`dim ker W` 6–7（planar 10）才是可辨识性指标；不是 rank-1 证书。
- **FCD / Atomic Pose Layers**：CERTIFIED / AMBIGUOUS / VALUE-ONLY 三态；只有单个物理原子 + `m>0,ν>0` 才发 pose/KKT 导数；多原子返回分支集、margin、SO(3) 混合协方差；ASAD 的 selection term 是硬损失跳变的 mollified jump。

## 环境

`/home/syk/miniconda3/envs/sdpr/bin/python`（numpy 1.26、scipy、cvxpy+SCS、torch CPU、matplotlib）。

## 目录

| 路径 | 内容 |
|---|---|
| `diffqpep/poly.py` | 四元数多项式、代数 PnP 残差、消去 t、Gram 矩阵 |
| `diffqpep/moment.py` | Lasserre 矩松弛、面维数 |
| `diffqpep/local.py` | 球面 SLSQP 局部求解、多起点 |
| `diffqpep/implicit.py` | KKT 隐式微分、SO(3) 角速度 |
| `diffqpep/rform.py` | 旋转矩阵提升（注意：`recover()` 的 last-row 仅适用于 rank-1；新实验用 `exp_correct`/`exp_double_cover` 的 12×12/原子提取） |
| `diffqpep/torch_layer.py` | 分支 starts/求解 + selection-aware autograd 原型（未用于论文主实验） |
| `experiments/exp_correct.py` | 12×12 R-form 正确恢复 + q-form 面诊断 |
| `experiments/exp_face.py` | 13×13 面原子/cheirality |
| `experiments/exp_double_cover.py` | planar PnP 精确双覆盖 |
| `experiments/exp_tie3d.py` | 3D 平面分支/sign 检查 |
| `experiments/exp_fold.py` | 1D fold / ν 发散 |
| `experiments/exp1d.py` | 1D tie、fixed-hypothesis selection、Laplace vs 混合 |
| `experiments/exp3d_gradient.py` | KKT vs 同分支 FD |

旧版 `exp3d.py`/`test_rform.py`/`test_rank.py` 已删除：它们用了 last-row section，结果不可信。

## 运行（Slurm，遵守实验室规范）

```bash
cd diffqpep
P=/home/syk/miniconda3/envs/sdpr/bin/python
PYTHONPATH=$PWD srun -p cpu -c 4 --mem=12G -t 20:00 $P experiments/exp_correct.py
PYTHONPATH=$PWD srun -p cpu -c 4 --mem=12G -t 15:00 $P experiments/exp_double_cover.py
PYTHONPATH=$PWD srun -p cpu -c 4 --mem=12G -t 15:00 $P experiments/exp_fold.py
PYTHONPATH=$PWD srun -p cpu -c 4 --mem=12G -t 15:00 $P experiments/exp1d.py
PYTHONPATH=$PWD srun -p cpu -c 4 --mem=12G -t 20:00 $P experiments/exp3d_gradient.py
```

结果在 `results/`，图在 `figures/`，日志在 `logs/`（不提交）。

## 论文/报告

- `paper/main.tex` / `paper/main.pdf`：6 页 IROS 格式（Atomic Pose Layers）。
- `report/diff_qpep_research.md`：中文研究知识库（含 12 个问题、修订记录、未来实验）。
- `deliverables/可微QPEP_研究知识库_2026-11.docx`：Word 版。
- `docs/visuals/2026-11-24-diff-qpep.png`：一页可视化。

## 限制

无论文主实验的 3D end-to-end、verified dual、PSD+trace 协方差层；forward 是多起点，不是 QPEP 全根；协方差是 1D synthetic；face/rank 是数值 SDP。旧版“R-form not solution-tight”等结论已废弃。
