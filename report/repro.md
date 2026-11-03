# 复现说明：SDPR × V-JEPA 2 桥接输出层实验

本目录记录实验的环境、资产、命令与产物。所有数字来自 `artifacts/results/` 的
`metrics_*.json` / `pred_*.npz`；图表由 `src/analyze.py` 重建。

## 1. 环境

| 项 | 值 |
|---|---|
| 机器 | server-2026-2（单节点 Slurm，40 CPU / 124 GB / 3× RTX 5090 32GB） |
| GPU 作业 | `--partition=gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G` |
| CPU 作业 | `--partition=cpu --cpus-per-task=8 --mem=32G` |
| Python | `~/miniconda3/envs/jepa`：Python 3.10，PyTorch 2.11.0+cu128，CVXPY 1.7.5，SCS 3.3.1 |
| DOCX | `~/miniconda3/envs/sdpr-clean`（python-docx） |
| SDPR | 自行实现的旋转同步 QCQP/SDP/隐式微分（`src/rotsync.py`）；CVXPY + SCS，CPU |

V-JEPA 2 代码使用官方仓库 `/home/syk/labs/JEPA/repos/vjepa2`；本实验只用了
`src/models/vision_transformer.py`（ViT-L/16）与官方权重。

## 2. 资产

| 资产 | 路径/来源 | 说明 |
|---|---|---|
| V-JEPA 2 ViT-L | `https://dl.fbaipublicfiles.com/vjepa2/vitl.pt`（5,127,726,842 B） | 本地 `artifacts/models/vitl.pt`；target encoder，303.9M 参数 |
| 全景图 | Lorem Picsum seeds `dsh1..dsh4` → `artifacts/assets/pano_{1,2,3,7}.jpg` | 2:1 自然图像；不随仓库分发 |
| UCF101（可选） | `/mnt/storage1/syk/jepa/datasets/ucf101_alt/UCF-101` | `panorama.build_panorama` 的本地 fallback |
| 数据/特征 | `artifacts/features/vjepa2_vitl.npz`（约 30 MB） | 810 条序列的 8×1024 特征 + GT 旋转 |
| 大文件 | `artifacts/` | `.gitignore` 忽略；仓库只保留报告/图/小指标 |

全景构建：`panorama.build_panorama(image_path=...)` 把 2:1 图像当作等距柱状环境贴图；
相机原地旋转，用球面采样渲染 16 帧 256×256。纯旋转下视图与贴图采样精确对应，GT 无渲染误差。

## 3. 命令

以下命令均在仓库根目录（`/mnt/storage1/syk/.superagent-work/sdpr-jepa-bridge-c46ae10a1a2f`）
执行。Slurm 包装器来自 `/home/syk/labs/JEPA/tools/slurm/`。

### 3.1 特征提取（GPU）

```bash
cd /home/syk/labs/JEPA
sbatch --job-name=sdpr-jepa-feats --cpus-per-task=8 --mem=32G --time=01:00:00 \
  tools/slurm/run_gpu.sbatch \
  /home/syk/miniconda3/envs/jepa/bin/python \
  /mnt/storage1/syk/.superagent-work/sdpr-jepa-bridge-c46ae10a1a2f/src/extract_features.py \
  --device cuda \
  --out /mnt/storage1/syk/.superagent-work/sdpr-jepa-bridge-c46ae10a1a2f/artifacts/features/vjepa2_vitl.npz
```

规模：train 480、val 90、test_id 120、test_ood 120；每序列 16 帧。
输出 `feats [N,8,1024]`、`R [N,8,3,3]`、`split`。

### 3.2 两阶段（相对损失 + 冻结 SDPR）

```bash
sbatch --job-name=sdpr-jepa-2stage --cpus-per-task=8 --mem=32G --time=01:00:00 \
  tools/slurm/run_cpu.sbatch \
  /home/syk/miniconda3/envs/jepa/bin/python \
  .../src/train_eval.py --features .../artifacts/features/vjepa2_vitl.npz \
  --method twostage --epochs 40 --batch-size 64 --tag v1
```

### 3.3 端到端（全局损失穿 SDPR 微调）

```bash
sbatch --job-name=sdpr-jepa-e2e --cpus-per-task=8 --mem=32G --time=02:00:00 \
  tools/slurm/run_cpu.sbatch \
  /home/syk/miniconda3/envs/jepa/bin/python \
  .../src/train_eval.py --features .../artifacts/features/vjepa2_vitl.npz \
  --method e2e --init .../artifacts/results/edgehead_twostage_v1.pt \
  --epochs 10 --batch-size 8 --lr 1e-5 --tag v1
```

### 3.4 对照

```bash
# 直接头（无 SDPR）
... src/train_eval.py --method direct --epochs 40 --batch-size 64 --tag v1
# oracle：GT 测量 + SDPR
... src/train_eval.py --method oracle --tag v1
```

`twostage`/`e2e` 的评测会同时跑 chordal（谱）同步基线，写入 `*_chordal` 字段。

### 3.5 梯度回传检查（GPU 或 CPU）

```bash
... src/bridge_check.py \
  --features .../artifacts/features/vjepa2_vitl.npz \
  --head .../artifacts/results/edgehead_e2e_v1.pt \
  --samples .../artifacts/data/samples.npz \
  --device cuda
```

### 3.6 图表与 DOCX

```bash
~/miniconda3/envs/jepa/bin/python src/analyze.py --tag v1
~/miniconda3/envs/jepa/bin/python src/make_figures.py --which arch,samples
~/miniconda3/envs/sdpr-clean/bin/python src/build_docx.py
```

## 4. 产物

| 产物 | 位置 |
|---|---|
| 特征 | `artifacts/features/vjepa2_vitl.npz` |
| 指标 | `artifacts/results/metrics_{method}_v1.json` |
| 每序列预测 | `artifacts/results/pred_{method}_{split}_v1.npz` |
| 模型 | `artifacts/results/edgehead_*.pt`（约 6 MB） |
| 图 | `report/figures/*.png` |
| DOCX | `deliverables/SDPR_VJEPA2_桥接输出层_2026-11.docx` |

## 5. 已知限制

- 全景图来自外部图片站点（Picsum），不随仓库分发；复现需按 seed 重新下载，或用 UCF101
  fallback（视觉不同，数字会变）。
- SDPR 在 CPU（SCS）上每样本约 0.2–0.4 s（N=8，13 条边）；大 N 未测。
- 证书是数值检查（SCS 容差），不是形式化证明。
- 本实验是合成受控基准；真实视频/真实机器人上的结论需要另做。

## 6. 本次运行记录（2026-11-03）

- Slurm job `2560`（`sdpr-jepa-pipeline`，GPU 0，8 CPU/32G），22:56:42 起，23:48:33 结束，
  `exit=0`；同时完成特征提取、训练、bridge check、图表与结果拷贝。
- 特征：810 条序列 × 8 节点 × 1024 维，提取 75.6 s；`target_encoder` 292 个 tensor，
  missing=0。
- 训练：two-stage 40 epochs（相对损失，验证 0.94°）；e2e 10 epochs，lr=1e-5（验证最优
  1.11°）；direct 40 epochs；oracle 无训练。
- 主要数字：`report/results/tables.md`；证书/梯度：`report/results/bridge_check.json`。
- 复现环境中的 `pano_*.jpg` 来自 Picsum seeds；仓库不分发原图，DOCX 内嵌了样本与结果图。
