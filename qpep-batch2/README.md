# QPEP 第二批：构建与预实验（2026-11）

> 源文档：Feishu wiki《20261128-deepresearch-第二批》（未随仓库分发）。
> 本目录是**第二批**三套原型的独立工作区：可复现代码、预实验、数学缺口。

## 这次做了什么

1. **新建独立工作目录** `qpep-batch2/`，与第一批 `labs/`（三条论文线）互不依赖。
2. **实现三套原型的最小可复现内核**：
   - `p1_riem/`：RiemQPEP-Layer，$S^3$ 全局极小 + 3×3 黎曼切空间隐式 VJP + 平移链式；
   - `p2_barrier/`：BarrierQPEP-Layer，倾角/深度不等式 → $\bm m_2$ 线性约束，同伦对数障碍 + 3×3 VJP；
   - `p3_spec/`：SpecQPEP-Layer，代数极小谱 + Boltzmann/Laplace 配分函数 + 包络梯度。
3. **四项预实验 E0–E3**（全部可一键重跑，JSON 可追溯）。
4. **数学缺口独立整理**：[`math/MATHEMATICAL_GAPS.md`](math/MATHEMATICAL_GAPS.md)，并导出 DOCX。
5. 一页总览图：[`../docs/visuals/2026-11-28-qpep-batch2.png`](../docs/visuals/2026-11-28-qpep-batch2.png)。

## 结论先行（最重要的五条）

1. **定理 0 的“Hilbert 紧性间隙”不适用于真实 QPEP**。$\det\bm T=-8192$、21 条约束秩 20 都复核成立；Choi–Lam 的 0.034188 间隙也精确复现，但它的 $\bm Q$ **不定**（特征值含 $-2,-0.5$），不是合法 QPEP Gram 矩阵。真实 QPEP 的 $\bm Q_m=\bm T^\top\bm Q_x\bm T$ 是 PSD（Schur 补的合同变换），12 个真实实例 + 40 个随机 PSD 实例的 10×10 SDP 间隙都 $\le 4.8\times10^{-8}$。
2. **RiemQPEP 的 3×3 反向成立且很准**：旋转 VJP 与中心差分最大相对误差 $8.3\times10^{-8}$，平移链式 $2.4\times10^{-8}$，双覆盖差 **0.0**；单样本 CPU 反向约 $220\,\mu$s（本实现）。
3. **BarrierQPEP 的 $O(\tau)$ 积极集收敛成立**：$\tau=10^{-1}\to10^{-6}$，梯度误差 $8.6\times10^{-4}\to9.1\times10^{-9}$，log-log 斜率 **0.997**；平方松弛的弱活跃 KKT 奇异也确认。
4. **SpecQPEP 的零阶 $Z_0$ 有偏**：$Z_0$ 包络梯度与 FD 一致到 $1.2\times10^{-6}$、零采样方差，但相对真实 $S^3$ 积分的梯度偏差达 **4.59**（$\beta=0.05$）；Laplace 曲率修正把它降到 **0.45**，丢掉曲率项的“朴素”版本偏差 3.10。tie 处 SDP 原解秩 2（$\lambda_1/\lambda_2=1.0$）。
5. **三套原型的故事需要改写**：Riem 的卖点是“去升维 + 3×3 反向”，不是“逃离 SDP 间隙”；Barrier 的卖点是“不等式可微”，但需要 LICQ/严格互补/SOSC；Spec 的卖点是“零采样方差 + 曲率修正”，不是“零偏差”。

## 目录

```
qpep-batch2/
├── core/                 # QPEP 代数、问题构造、临界点枚举、SDP
│   ├── algebra.py        #   m2/J/E/T、21 条约束、Hessian
│   ├── problems.py       #   3D-3D Schur 归约、随机实例、双峰实例
│   ├── modes.py          #   多起点黎曼牛顿（数值枚举，非全根证书）
│   └── sdp.py            #   10×10 SDP 原/对偶（Clarabel）
├── p1_riem/layer.py      # 原型 1
├── p2_barrier/           # 原型 2（constraints.py + layer.py）
├── p3_spec/layer.py      # 原型 3
├── experiments/          # e0–e3 + make_figures.py
├── results/              # 全部 JSON（数字可追溯）
├── figures/              # 四张实验图
├── math/MATHEMATICAL_GAPS.md
└── README.md
```

## 快速开始

```bash
cd qpep-batch2
P=/home/syk/miniconda3/envs/sdpr/bin/python

$P experiments/e0_theorem0.py   # 定理 0 审计
$P experiments/e1_riem.py       # RiemQPEP
$P experiments/e2_barrier.py    # BarrierQPEP
$P experiments/e3_spec.py       # SpecQPEP
$P experiments/make_figures.py  # 四张图 + 一页总览

# 可选：重建 DOCX
$P scripts/build_docx.py
```

依赖：`numpy scipy cvxpy matplotlib python-docx`（本机 `sdpr` 环境已装；
SDP 用 Clarabel，**不需要 Mosek**）。

## 实验一览

| 实验 | 内容 | 关键结果 |
|-|-|-|
| **E0** | 定理 0：$\det\bm T$、21 约束秩、Choi–Lam、真实 QPEP 的 SDP 紧性 | det $=-8192$；秩 20；Choi–Lam 间隙 0.034188（Q 不定）；真实 QPEP 间隙 $\le4.8\times10^{-8}$；随机 PSD 40 例最坏 $8.2\times10^{-9}$ |
| **E1** | RiemQPEP：前向、3×3 VJP、双覆盖、平移、计时 | VJP $8.3\times10^{-8}$；双覆盖 0；反向 $220\,\mu$s；局部牛顿 $605\,\mu$s；多起点 $328$ ms |
| **E2** | BarrierQPEP：平方松弛奇异、同伦、$O(\tau)$、VJP | 斜率 0.997；VJP $1.3\times10^{-10}$；弱活跃 KKT 精确奇异；直接小 $\tau$ 会停鞍点 |
| **E3** | SpecQPEP：$Z_0$/$Z_{\rm ML}$、Maxwell 交叉、MC 方差、SDP 秩 | $Z_0$ 偏差 4.59；$Z_{\rm ML}$ 0.45；朴素 ML 差 3.10；tie 秩 2；MC std 7–10 |

## 证据分级

沿用仓库既有口径：

| 等级 | 含义 | 本目录示例 |
|-|-|-|
| **已证明** | 可逐步核对的推导 | $\det\bm T=-8192$；21 约束秩 20；Schur 补 PSD；平方松弛 KKT 奇异 |
| **数值验证** | `experiments/` 可复现 | 3×3 VJP；$O(\tau)$；SDP 紧性；秩 2 坍缩 |
| **源文档声称** | 第二批 wiki 的数字 | 114.72 µs；5.198e-7；0.03418806 |
| **待验证/假设** | 缺证明或实验 | “PSD ⇒ 10×10 SDP 恒紧”；Laplace 一致渐近；13×13 真实实例紧性 |

**不把“源文档声称”写成“已复现”，不把“数值现象”写成“定理”。**

## 与第一批的关系

- 第一批（PR #39，`labs/`）把另一份源文档的三个 prototype 推进为三篇论文，已有真实数据实验与证明。
- 本批是**新的一份源文档**（2026-11-28）的三套原型（Riem/Barrier/Spec），与第一批的
  Laplace/Constrained/Robust 有概念重叠但贡献声明不同。本批**不依赖**第一批代码；
  只在数学缺口 G1 中引用第一批的 13×13 反例（标为“内部前序结果”）。
- 两批的共同结论：SDPRLayer 的实际 13×13/24 约束与 10×10/21 旋转子问题不是同一个松弛；
  QPEP 的“全局”需要根枚举证书（27 vs 40）；真实数据/端到端训练仍是下一阶段。

## 明确未覆盖

- 真实数据集（LINEMOD / YCB-Video / KITTI / UTIAS In-The-Dark / 3DMatch）未跑；
- LibQPEP / GPU 批量绑定未接；前向用多起点数值枚举，不是 27×27 全根证书；
- 13×13 SDPRLayer 未在本批独立复跑（引用第一批 PR #39）；
- Fold 分岔、弱活跃、多约束 LICQ 失效、端到端训练未测；
- 源文档两张内嵌 sheet（baseline 选型表）因 bot 缺 `sheets:spreadsheet:read` 权限未读取。

这些边界不影响本批的数学结论，但决定了它只是**预实验/可行性验证**，不是论文级主表。
