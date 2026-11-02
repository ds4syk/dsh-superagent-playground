# repro：SDPRLayers §VI-B4 相机 baseline 标定复现

对象：SDPRLayers (T-RO 2025) §VI-B4 / Fig.4 / Table V。
详细报告：[`../report/reproduction.md`](../report/reproduction.md)。

## 目录

| 路径 | 内容 |
|---|---|
| `cleanroom/` | 独立实现（不看上游），含相机模型、SDP、证书、隐式微分、局部求解器、bilevel 外层、自检 |
| `results_clean/` | cleanroom 的 tables / figures / run_summary / Jacobian / tightness（逐迭代轨迹 CSV 可由 `run_experiment.py` 重新生成，未入库） |
| `upstream/` | 官方代码修复版的驱动、环境脚本、报告（上游 `sdprlayer` 仓库本身不内嵌） |
| `patches/` | 对上游 `cb3aed2` 的修复 patch（0001–0009） |
| `results_upstream/final/` | 官方代码 strict 配置的 Table V 与曲线（36/50） |
| `results_upstream/final_adaptive/` | 自适应重试后的最终 Table V 与 summary（49/50；逐迭代曲线可重新生成，未入库） |
| `ENV.md` | 原始实验环境记录 |

## 最快验证：cleanroom 自检

```bash
conda create -n sdpr-clean python=3.11 -y
conda activate sdpr-clean
pip install numpy scipy cvxpy matplotlib tqdm pandas
cd repro/cleanroom
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python smoke_test.py
```

期望：`16/16 checks passed`（约束、目标、逆模型 Jacobian、SDP 紧性、证书 PSD、强对偶、
隐式 Jacobian、外层梯度、反向计时）。
本次交付前已在 `sdpr-clean` 环境复跑通过；一次 2-run 端到端复跑给出
SDPR $2.66\times10^{-4}$ m、local-GT $2.67\times10^{-4}$ m、local-RND $0.392$ m。

## 全量复现（cleanroom）

```bash
cd repro/cleanroom
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python run_experiment.py --runs 50 --workers 12
```

默认写回 `repro/results_clean/`；用 `--out-dir /tmp/...` 可避免覆盖已有结果。
两种噪声口径（independent / upstream）都会跑；论文口径是 independent。

## 官方代码修复版（faithful）

上游仓库不内嵌（避免把第三方代码与许可证带入本仓库）。复现步骤：

```bash
git clone https://github.com/utiasASRL/sdprlayer.git code/sdprlayer
cd code/sdprlayer && git checkout cb3aed2
for p in ../../repro/patches/*.patch; do git apply "$p"; done
```

然后按 [`upstream/REPORT.md`](upstream/REPORT.md) 的 §2 构建环境、跑 `run_calibration.py`；
结果已放在 `results_upstream/final/`。

## 结果摘要

| 方法 | 论文 | cleanroom (independent) | faithful |
|---|---|---|---|
| SDPR-IS final err | 3.454e-4 | 4.105e-4 | 4.699e-4（修复后 4.615e-4，49/50） |
| Theseus-GT | 3.463e-4 | 4.112e-4 | 4.620e-4 |
| Theseus-RND | 1.132e-1 | 3.375e-1 | 1.224e-1 |

核心结论：SDPR 与 GT 初始化局部解几乎重合；随机初始化局部解失败；SDP 解 100% rank-1、
$H\succeq0$、corank=1（99.97%）；局部极小比例 59% vs 论文约 60%。绝对数字差 20–45%，
主因是终止阈值口径与局部求解器实现，详见报告。

## 资源

纯 CPU；cleanroom 单 run SDPR 约 15.5 s（单进程），单次 SDP solve 约 2.66 ms，
反向 0.15–0.17 ms，峰值 RSS 约 298 MB。全量 300 runs 约 1.9 CPU·h。
