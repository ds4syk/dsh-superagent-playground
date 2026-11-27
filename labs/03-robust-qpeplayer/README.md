# Robust-QPEPLayer（Prototype 3）

> **一句话定位**：用 Black–Rangarajan 对偶 + GNC + 常数维数 QPEP，做一个**无 RANSAC、反向精确到鲁棒不动点曲率**的可微位姿层；不声称全局，只给局部二阶保证与明确的失败边界。

---

## 核心卖点

1. **双向零升维（旋转 / 观测数）**：内层加权 QPEP 只有 $4+3$ 个变量；$N$ 只进 $O(N)$ 累加；反向只解 $6\times6$ 线性系统，不展开 GNC 迭代。
2. **精确鲁棒隐微分**：显式保留降权曲率项 $\Delta H_{\mathrm{redesc}}=\sum_i\rho''(e_i)g_ig_i^\top$；证明它等于 BR 权重块的 **Schur 补**，不是“小修正”。
3. **S³ 约束乘子修正**：源文档的 $H_{\mathrm{rob}}$ 公式若用欧氏 Hessian，缺少 $-\lambda_{\mathrm{rob}}\operatorname{diag}(I_3,0)$；我们给出精确黎曼 Hessian，并用有限差分验证（源公式误差 25.9%，修正 9.9e-8）。
4. **硬定理而非数值传说**：一维精确变号命题（真实导数负、naive 正）+ 存在性构造；6D 的 cos$=-0.079$ 降级为数值例，不冒充定理。
5. **$O(\|\mathbf r\|^{-4})$ 逐观测梯度界**：显式常数与条件；同时指出前向偏差是 $O(\|\mathbf r\|^{-3})$，且“梯度免疫 ≠ 鲁棒”。
6. **失败模式写清楚**：GNC 不全局；相干多数外点会让全局鲁棒解本身错误；边界半外点（$e\approx c^2$）梯度最毒；$H_{\mathrm{rob}}$ 可不定/奇异。

---

## 当前完成度

| 模块 | 状态 |
|---|---|
| 数学（BR 对偶、GNC、Schur 补、黎曼 Hessian、IFT、一维变号、$O(r^{-4})$） | **已写 + 已证/条件性证明**，见 `theory/proofs.md`、`theory/derivations.md` |
| 最小数值 sanity check（2 个脚本） | **通过**，见 `verify/` |
| 论文草稿（英文，结构完整） | **v0.1**，实验数字全是 TODO，见 `paper/draft.md` |
| 贡献/对比/审稿 Q&A | **完成**，见 `paper/contributions.md` |
| 相关工作/引用核实 | **完成**，见 `related_work.md`（含 2 处引用勘误） |
| 实验协议 | **设计完成；合成 E1–E4 已跑**，真实数据 TODO，见 `experiments/protocol.md` |
| 合成实验（扩展主表 + 6D 梯度表 + 边界精化 + 校验） | **完成**（2026-11-27），见 `experiments/README.md`、`experiments/synthetic/results/` |
| 路线图 | **完成**，见 `roadmap.md`（RA-L 首选 / IROS 2027 备选 / T-RO stretch） |
| 未解决数学问题 | **12 条，P0×4**，见 `math_questions.md` |

**没有做的**：真实数据实验、真实端到端训练（只有 GT 监督的 test-time 精化诊断）、QPEP C++/GPU 工程、6D 变号定理、GNC 分岔反例。**没有编造**任何实验数字；源文档的预实验数字在 `experiments/protocol.md` §8 标为“待独立复现”。**待修**：`verify/riemann_hessian_check.py` 的 `build_A` 用非齐次 `Rmat` 做极化（重构的 $u(q)\neq R(q)p$），应换成 `Rhom` 后重跑；修正版与修正后的数值见 `experiments/README.md` §7.1。

---

## 文件导航

```
labs/03-robust-qpeplayer/
├── README.md                     # 本文件
├── paper/
│   ├── draft.md                  # 英文论文草稿（Title/Abstract/1–7/References）
│   └── contributions.md          # 贡献清单 + 逐维对比 + 审稿 Q&A
├── theory/
│   ├── proofs.md                 # 定理/引理/命题：假设、结论、证明、GAP
│   └── derivations.md            # BR 对偶、GNC、黎曼 Hessian、RHS、λ∥、O(r^-4)
├── experiments/
│   └── protocol.md               # 数据集/baseline/指标/消融/算力/风险/预实验标注
├── related_work.md               # 对比表 + 逐篇差异 + 引用核实/勘误
├── roadmap.md                    # 4/8/12 周里程碑、venue、风险
├── math_questions.md             # 12 条未解决数学问题（P0/P1/P2）
└── verify/
    ├── README.md                 # 一句话结论 + 环境说明
    ├── ift_1d_signflip.py        # 脚本 1：BR 对偶 + 一维变号 + FD + λ∥ + 斜率 -4
    ├── ift_1d_signflip_output.txt
    ├── ift_1d_signflip.png
    ├── riemann_hessian_check.py  # 脚本 2：S³×R³ 黎曼 Hessian vs FD；λ_i 校验
    ├── riemann_hessian_check_output.txt
    └── riemann_hessian_check.png
```

---

## 最近更新（2026-11-27）

- 完成 `theory/proofs.md`（19 节，含 5 个 Lemma、3 个 Theorem、4 个 Proposition、1 个 Corollary、GAP 清单、源文档差异表）。
- 完成 `theory/derivations.md`（9 节，含 BR 对偶全配方、Schur 补、$H_i^E$ 的 $A_k$ 技巧、$\lambda_\parallel$ 的完整求导、$O(r^{-4})$ 常数）。
- 完成 `paper/draft.md`（英文，451 行；实验表格全 TODO）。
- 两个数值脚本通过：
  - BR 对偶值误差 $10^{-11}$；$\lambda_\parallel$ 闭式误差 $6.7\times10^{-16}$；一维精确 vs FD $1.2\times10^{-8}$，naive 误差 $137\%$，符号相反；$\log$-$\log$ 斜率 $-3.9979$。
  - 黎曼 Hessian：源公式 vs FD $2.59\times10^{-1}$；修正公式 $9.9\times10^{-8}$；$\lambda_i-4\mathbf u_i^\top W_i\mathbf r_i$ 误差 $7.1\times10^{-15}$，$\lambda_i-4e_i$ 差 $51.0$。
- 修正源文档三处过度声明：GNC 全局、$H_{\mathrm{rob}}$ 缺 $-\lambda$、6D 变号。
- 引用勘误：Black–Rangarajan 是 IJCV 1996（不是 CVPR 2014）；GNC 是 RA-L 2020（CVPR 2020 是 TEASER）。

---

## 下一步 3 条

1. **锁 6D 变号口径**（P0）：决定是补 6D 定理（加 $H_{\mathrm{rob}}$ 与 $\partial\Phi/\partial\mathbf y$ 的联合谱条件），还是保留“1D 严格 + 映射层面 + 6D 数值/GAP”。这直接决定 Theorem 3.3 的最终写法与投稿档次。
2. **实现最小可运行层**：PyTorch `autograd.Function` + 简化 QPEP/3D-3D 闭式；`gradcheck` + 50 trials 的 IFT/naive/FD 三对照；把 `verify/` 的结论升级成 CI 单测。
3. **跑 3DMatch/KITTI 的 E1/E2/E5**（随机/相干/边界外点；曲率项、$-\lambda$、核、$W_i$ 消融），先出“外点率-成功率”和“梯度余弦分箱”两张图；再决定是否值得投入 In-The-Dark 端到端。

---

## 与另外两条线的边界

- **Prototype 1（Laplace-QPEPLayer）**：$O(10)$ Veronese 同构、Hilbert 非紧性、Laplace 多峰/协方差；我们不做同构、不做 Laplace、不声称全局。
- **Prototype 2（Constrained-QPEPLayer）**：不等式/互补/物理约束；我们不做不等式、不做 ALM/FB。
- **共享**：QPEP 作为内层求解器。可以组合，但本文独立成篇，不依赖它们未发表的结果。

---

## 快速自检

```bash
cd labs/03-robust-qpeplayer
/home/syk/miniconda3/bin/python verify/ift_1d_signflip.py
/home/syk/miniconda3/bin/python verify/riemann_hessian_check.py
```

> 环境：`/home/syk/miniconda3/bin/python` 有 numpy/sympy/matplotlib，但**没有 scipy**（与任务说明不符）；两个脚本只用 numpy + matplotlib，已避开 scipy。
