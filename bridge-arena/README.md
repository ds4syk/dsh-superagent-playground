# 造桥大赛 · Transformer / Diffusion / CNN 三模型对比

用官方开源引擎（[chen-006/meow-ai-arena](https://github.com/chen-006/meow-ai-arena)，
对应 [arena.meowllm.top/ep9](https://arena.meowllm.top/ep9/) 造桥大赛）做两件事：

1. **自己试玩**：在关卡 1 从光板桥开始，手工迭代到 61.47 分；在关卡 2 验证
   “把 15 米桁架直接放大到 30 米”会失败，并载入官方参考答案 / 选手设计做对照。
2. **训练三个生成模型**：Transformer、Diffusion、CNN-VAE，在同一设计空间里生成
   关卡 1 的桥；每个模型采样 512 座，用官方 C++ 引擎跑完全部 3 个公开场景，
   按比赛公式计分。

## 结果（详见 PDF）

关卡 1，每个对象 512 个样本，官方 C++ 引擎跑完 3 个公开场景：

| 对象 | 通关率 | 平均分 | 最高分 | 平均造价 | 唯一率 | 多样性 |
|---|---:|---:|---:|---:|---:|---:|
| 随机基线 | 82% | 48.58 | 61.47 | 5892 | 70% | 0.081 |
| Transformer | 100% | 57.59 | 61.47 | 6075 | 96% | 0.122 |
| Diffusion | 100% | 60.56 | 61.47 | 5799 | 30% | 0.042 |
| CNN-VAE | 100% | 61.47 | 61.47 | 5710 | 0% | 0.000 |
| 专家数据 | 100% | 59.87 | 61.47 | 5862 | 100% | 0.095 |

CNN-VAE 的 512 个先验样本全部解码成同一座种子桥（后验塌缩）：分数最高，但没有多样性。
Transformer 保留了 96% 的唯一设计，质量与多样性最均衡。

## 快速开始

需要 Linux/macOS、Python 3.9+、g++（编译官方引擎）。本仓库不提交第三方引擎。

```bash
./setup.sh                                  # 克隆官方 repo + 编译 bridge_sim
python scripts/play.py                      # 手工试玩 + 出图
python -m arena_bridge.dataset --mode dataset --level 1 --n 30000 --workers 16
python scripts/train.py --data results/dataset_l1.npz --out results/checkpoints --epochs 400
python scripts/evaluate.py --n 512 --workers 16
python scripts/make_figures.py
python scripts/build_report.py              # 生成 report/bridge_arena_report.pdf
```

GPU 训练通过 Slurm 提交：`sbatch sbatch/train.sbatch`、`sbatch sbatch/eval.sbatch`。

## 设计空间

关卡 1（15 m 跨）被离散成：

- 固定桥面：y=0，A0→A2，每 1.5 m 一根路面；
- 桥面以下 1 层（y=-2，接底部锚点 A1/A3）、以上 2 层（y=1.5/3.0）；
- 121 根候选杆件，每根取 0=空 / 1=木 / 2=钢 / 3=钢索。

模型只生成上部/下部结构，桥面和锚点固定。这一约束保证输出天然是合法 JSON，
也让 3 个模型可以在同一表示上公平比较；官方引擎本身支持任意节点、杆件和机关，
手工试玩不受这个约束。

## 目录

```
arena_bridge/     设计空间、官方引擎封装、数据生成、2D 渲染
models/           Transformer / Diffusion / CNN-VAE
scripts/          play / train / evaluate / make_figures / build_report
sbatch/           Slurm 提交脚本
report/           PDF 与图表
third_party/      官方引擎（setup.sh 拉取，不入库）
```

## 口径

- 关卡分 = 100 × 参考答案造价 ÷ 你的造价（过关才有分），与比赛一致。
- 评测不做拒绝采样、不做模拟器引导；三个模型各 512 个样本，同一随机种子。
- 训练数据是 30,000 个“围绕通过设计做 1~4 处变异”的候选，其中 20,661 个通过
  全部 3 个场景；训练集取分数最高的 6,000 个。
