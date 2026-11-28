# `sdp/` — P1-Q2/Q13：10×10 全 20D 松弛 vs 上游 13×13（17D）

本目录是 P1-Q2/Q13 的数值实验与图表入口。核心结论：**满 20D 的 13×13（24 条上游约束 + 3 条行正交）与 10×10 等价；上游 `sdprlayer@62d8ac6` 的 13×13 只有 24 条约束（秩 17），严格更松。** 6 点物理 3D-3D 反例上，$f^\star_{10}=f^\star_{13,\rm full}=5.415$（rank 1）而 $f^\star_{13,\rm up}=3.59431373$（rank 4），间隙 1.82068627。

## 1. 文件索引

```text
sdp/
├── run_sdp_check.py           # 10 实例 + 8 个对称破缺扰动 + 21/24 约束的精确结构分析
├── run_6point_check.py        # 6 点物理反例：f10 / f13_full / f13_up + 非凸值
├── run_six_point_perturb.py   # 6 点反例鲁棒性：base + 9 个 W/y 扰动
├── run_six_point_cert.py      # 6 点实例的对偶数值证书（区间 + PSD 裕度）
├── make_q2_summary.py         # Q2 两栏图（左：逐实例；右：6 点实例柱状图）
├── README.md                  # 本文件
└── results/
    ├── sdp_check.{json,md}          # 主检查的原始数据 / 中文报告
    ├── six_point.{json,md}          # 6 点反例的原始数据 / 英文短报告
    ├── six_point_perturb.{json,md}  # 扰动鲁棒性
    ├── six_point_cert.{json,md}     # 数值证书（区间、对偶松弛、裕度）
    └── q2_summary.{png,json}        # Q2 图 + 图元数据（含证书区间）
```

| 脚本 | 作用 | 产物 | 本机耗时（2026-11-27） |
|-|-|-|-|
| `run_sdp_check.py` | 10 个实例上对比 10×10 / 13×13；精确比较 21 条与 24 条约束的秩；8 个扰动 | `results/sdp_check.json`、`results/sdp_check.md` | ≈39 s（不含 `--check-upstream`） |
| `run_6point_check.py` | 构造 6 点物理反例并求解三个 SDP；CLARABEL 独立复核 | `results/six_point.json`、`results/six_point.md` | ≈1.5 s |
| `run_six_point_perturb.py` | 同一反例在 W/y 小扰动（含 $D\neq0$）下保持严格间隙 | `results/six_point_perturb.json`、`results/six_point_perturb.md` | 数秒 |
| `run_six_point_cert.py` | 直接解对偶，给出 $[lower,upper]$ 与 PSD/Cholesky 核验 | `results/six_point_cert.json`、`results/six_point_cert.md` | ≈0.6 s |
| `make_q2_summary.py` | 生成 Q2 两栏图；只读 `sdp_check.json`、`six_point.json`（有证书时并入区间） | `results/q2_summary.png`、`results/q2_summary.json` | ≈0.8 s |

## 2. 复现

环境固定为 `/home/syk/miniconda3/envs/WM/bin/python`（Python 3.10、numpy 2.2.6、scipy 1.15.3、cvxpy 1.7.5、CLARABEL 0.11.1、SCS 3.3.1、sympy、matplotlib）。

```bash
cd labs/01-laplace-qpeplayer/experiments/sdp
PY=/home/syk/miniconda3/envs/WM/bin/python

# 最小路径：6 点反例 → 证书 → Q2 图
$PY run_6point_check.py
$PY run_six_point_cert.py
$PY make_q2_summary.py

# 全量检查（10 实例 + 8 扰动 + 结构分析）
$PY run_sdp_check.py
# 可选：直接对拍上游生成器源码（本机路径存在时）
$PY run_sdp_check.py --check-upstream /home/syk/labs/SDPR/code/sdprlayer
$PY run_six_point_perturb.py
```

说明：

- `make_q2_summary.py` 依赖 `results/sdp_check.json` 与 `results/six_point.json`；先跑 `run_sdp_check.py`、`run_6point_check.py`，或直接使用仓库中已提交的结果。
- `--check-upstream` 是**可选**项：它用 AST 抽取上游 `pose_est.py` 的三个生成器原始函数体，与 `build_13x13()` 逐元素对拍。2026-11-27 在本机核对：`24/24` 完全一致，`max|diff|=0.0`，随机旋转上 `max|xᵀAᵢx|=1.4e-16`（上游 commit `62d8ac6`）。仓库里不 vendored 上游代码；路径不存在时省略该参数即可。
- 所有随机种子固定（`--seed 12345`），结果可复现；单个 SDP 规模 ≤13×13，全部实验 <10 min。

## 3. 已确认数字（2026-11-27 复跑）

6 点物理反例（$p_i=\pm e_1,\pm e_2,\pm e_3$，$y_i=0.05p_i$，$W$ 特征值 {0.4, 1, 1.6}）：

| 量 | 值 | rank |
|-|-|-|
| $f^\star_{10}$（21 约束，满 20D） | 5.41500000 | 1 |
| $f^\star_{13,\rm full}$（24 + 3 行正交 = 27 约束，满 20D） | 5.41500000 | 1 |
| $f^\star_{13,\rm up}$（24 约束，17D） | 3.59431373 | 4 |
| $f^\star_{10}-f^\star_{13,\rm up}$ | 1.82068627 | — |

数值证书（`results/six_point_cert.md`，标签 **numerical certificate (float), not exact rational**；PSD 裕度 $10^{-8}$，数值安全裕度 $10^{-7}$；下界 CLARABEL 对偶、上界 SCS 原始值/精确可行点）：

| 量 | 数值区间 |
|-|-|
| $f^\star_{10}$ | [5.41499990, 5.41500000] |
| $f^\star_{13,\rm full}$ | [5.41499990, 5.41500000] |
| $f^\star_{13,\rm up}$ | [3.59431363, 3.59431383] |
| $f^\star_{10}-f^\star_{13,\rm up}$ | [1.82068607, 1.82068637] |

其他复核：

- 10 个主实例：8/8 个对称破缺扰动严格 $f^\star_{13,\rm up}<f^\star_{10}$；`rowdot` 构造反例的间隙为 2.0。
- 6 点反例扰动：10/10 个案例（base + 9 个 W/y 扰动）保持严格间隙（0.106–1.821）。
- 上游 13×13 的 24 条约束在 $\mathrm{Sym}^{10}$ 中秩 17，10×10 的 21 条秩 20；缺的正是 $r_1\cdot r_2=r_1\cdot r_3=r_2\cdot r_3=0$ 三条行正交。

## 4. 口径与 caveat

- **满 20D ≠ 上游 13×13。** 满 20D 的 13×13（27 条约束）与 10×10 由 Schur 补双射等价，数值上 $f^\star_{13,\rm full}=f^\star_{10}$；上游实现只有 24 条约束（17D），一般严格更松。图与表都区分这两层。
- **数值而非精确有理。** 所有 SDP 值来自 SCS / CLARABEL 浮点求解；证书区间带显式安全裕度，不能当作有理数精确证明。
- **证明引用值与实测值的 4.9e-6 差异。** 修订证明引用 $f^\star_{13,\rm up}=3.59430886$；SCS 原始值与 CLARABEL 对偶值均为 3.59431373（CLARABEL 原始校验 3.59431371，差 2e-8）。该引用值落在证书区间之外（差 4.9e-6，高于求解器容差）。定性的 O(1) 分离（间隙 ≈1.82）不受影响，但引用最后几位时应改用实测值或说明该差异。
- **rank 阈值。** $X^\star_{13,\rm up}$ 有 4 个 O(1) 特征值，后面是 ~5e-6 的数值尘；按 1e-4 相对阈值 rank 4，按 1e-6 则 rank 6。定理只需 rank > 1。
- **范围。** 只处理旋转子问题；13×13 是 SDPRLayers 的提升 SDP 复刻，不声称采样矩阵是真实 PnP 的 Gram 矩阵（那是 Q1）。

## 5. 关键数据出处

- `results/sdp_check.json` 的 `config` 记录完整环境与上游 commit；`structural` 是精确秩/残差。
- `results/six_point.json` 是 6 点反例的三个 SDP 值与特征值；`results/six_point_cert.json` 是对偶证书的完整浮点数据（对偶乘子未落盘，但 slack 的 $\lambda_{\min}$、Cholesky 结果、区间端点均在）。
- Q2 图元数据 `results/q2_summary.json` 同时记录逐实例坐标、6 点反例柱值、证书区间与 caveat。
