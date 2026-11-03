# `sdpr-clean` 环境记录（repro-clean / task-3）

为与上游复现（`sdpr` 环境）**完全隔离**而单独创建的 conda 环境。**未修改** `sdpr` / `WM`
环境，未使用 GPU。

## 创建命令

```bash
/home/syk/miniconda3/bin/conda create -n sdpr-clean -y python=3.11
/home/syk/miniconda3/envs/sdpr-clean/bin/python -m pip install \
    --index-url https://download.pytorch.org/whl/cpu torch
/home/syk/miniconda3/envs/sdpr-clean/bin/python -m pip install \
    numpy scipy cvxpy matplotlib tqdm pandas
```

（torch 仅用于 `loss_from_x_and_grad` 之外的零星张量工具/兼容性，核心算法全部 numpy；
GPU 版 torch 未安装。）

## 版本（2026-10-25 实测）

| 包 | 版本 |
|---|---|
| Python | 3.11.17 |
| numpy | 2.4.6 |
| scipy | 1.17.1 |
| cvxpy | 1.9.3 |
| SCS | 3.3.1 |
| Clarabel | 0.11.1（备用，未使用） |
| torch | 2.14.1+cpu（CUDA 不可用） |
| matplotlib | 3.11.2 |
| pandas | 3.1.0 |

**与论文的差异**：论文（2025）使用的 cvxpy/SCS 版本更早，且 Table III/IV 用 Mosek
（1e-12，需 license，本机没有）。本文 §VI-B4 的实验论文本身就用 SCS 1e-9，因此主实验
口径一致；求解器版本差异只影响数值精度与计时，不影响算法与结论。

## 运行环境变量

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1   # 每 worker 1 线程
# 12 workers（主实验），nice -n 10
```

机器：Intel(R) Xeon(R) w7-3445，40 核，125 GB；NVIDIA RTX 5090 ×3（他人占用，本实验未用）。
