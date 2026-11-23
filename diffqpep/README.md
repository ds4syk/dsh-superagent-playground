# diffqpep — 可微全局 QPEP 研究代码

本目录是论文 *Certifiably Differentiable Quadratic Pose Estimation* 与中文研究知识库的复现代码。

## 安装 / 环境

使用实验室 `sdpr` conda 环境（Python 3.11，numpy 1.26、scipy、cvxpy+SCS、torch CPU、matplotlib）：

```bash
P=/home/syk/miniconda3/envs/sdpr/bin/python
```

## 目录

| 路径 | 内容 |
|---|---|
| `diffqpep/poly.py` | 四元数多项式、代数 PnP 残差、消去 t、Gram 矩阵 |
| `diffqpep/moment.py` | Lasserre 矩松弛（order 2/3）、有效秩、q 恢复 |
| `diffqpep/local.py` | 球面 SLSQP 局部求解、多起点 |
| `diffqpep/implicit.py` | KKT 隐式微分、SO(3) 角速度 |
| `diffqpep/rform.py` | 旋转矩阵 Shor 提升（负结果：值紧、解不紧） |
| `diffqpep/instances.py` | PnP / 平面 PnP / 对称 PnP 实例 |
| `experiments/exp3d.py` | 3D 证书 + 矩 gap |
| `experiments/exp3d_gradient.py` | KKT vs 同分支有限差分 |
| `experiments/exp1d.py` | 1D 切换、局部污染、Laplace vs 混合、风险动作 |
| `experiments/test_pollution_instances.py` | 平面 PnP 的局部 vs 全局分支 |

## 运行（Slurm，遵守实验室规范）

```bash
cd diffqpep
PYTHONPATH=$PWD srun -p cpu -c 4 --mem=16G -t 30:00 $P experiments/exp3d.py
PYTHONPATH=$PWD srun -p cpu -c 4 --mem=12G -t 20:00 $P experiments/exp3d_gradient.py
PYTHONPATH=$PWD srun -p cpu -c 4 --mem=12G -t 15:00 $P experiments/exp1d.py
```

结果写入 `results/*.json`，图写入 `figures/*.png`；日志在 `logs/`（不提交）。

## 论文与报告

- `paper/main.tex` / `paper/main.pdf`：IROS 格式论文（6 页，tectonic 编译）。
- `report/diff_qpep_research.md`：中文研究知识库（含 12 个问题的逐条回答）。
- `deliverables/可微QPEP_研究知识库_2026-11.docx`：Word 版。
- `docs/visuals/2026-11-24-diff-qpep.png`：一页可视化。

## 关键限制

- 矩松弛只给**价值** gap，不是 rank-1 全局解证书；R-form SDP 值紧但解无效。
- 前向仍是多起点，不是全根枚举/分支定界；时序/不等式/真实机器人实验是下一步。
