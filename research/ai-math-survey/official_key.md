# 十个公开案例的官方结论（仅用于评分，不进入盲测上下文）

来源：`openai/math` 仓库（2026-10-06 发布，当前 `main` 为 2026-11-21 的快照），
README、`overview.tex`、`CONTENTS.md`、`reasoning_traces/` 与 `lean/formalization.yaml`。
本文件只给评分者与报告作者看；`repro/openai-10/prompts/` 中的题面不含这些结论。

| 家族 | 题目 | 官方结论（论文声称） | 关键出处 |
|---|---|---|---|
| 007 | 乘性函数的两点相关 / 修正 Elliott 猜想 | 证明 ordinary two-point Chowla；对 Liouville 沿固定非比例仿射形式给 \(O(X/(\log X)^c)\)；更一般地证明复乘性函数的二元修正 Elliott 猜想（在均匀非 pretend 条件下） | `overview.tex` 第 007 条；`preprints/Ordinary-two-point-correlations-of-multiplicative-functions-September-24-2026/final.pdf` |
| 017 | \(\pi\) 的无理性指数 | 证明 \(\mu(\pi)=2\)：对每个 \(\nu>2\)，所有充分大分母 \(q\) 与任意整数 \(p\) 有 \(\lvert\pi-p/q\rvert\ge q^{-\nu}\)；推论 Flint–Hills 级数收敛 | `overview.tex` 第 017 条；`preprints/The-irrationality-exponent-of-pi-is-2-September-24-2026/paper.pdf`；`lean/docs/017.md` |
| 087 | 几何 Mahler 猜想 | 证明对称与非对称 Mahler 猜想（所有维数）；极小体分别为 Hanner 多面体与单形；分类等号；相应泛函不等式；对称极积的 Gromov 宽为 4 | `overview.tex` 第 087 条；三篇 preprint；`lean/docs/087.md` 与 `ComparatorChallenges/MahlerConjecture.json` |
| 102 | 基本 SDP 阈值 / Unique Games | 证明 Khot 的 Unique Games Conjecture；并由直接归约得到 Max-Cut、Vertex Cover、Min-UnCut、directed FVS 的最优不可近似性。因此题面所述 BasicSDP 阈值断言为真（经 UGC + Raghavendra 或直接归约） | `overview.tex` 第 102 条；`preprints/The-Unique-Games-Theorem-September-23-2026/paper.pdf` 等 |
| 159 | Erdős 倒数和猜想 | 证明：正整数集合若倒数和发散，则含任意长度的等差数列；并给定量拟多项式 Szemerédi 上界 \(\lvert A\rvert\le C_kN\exp(-c_k(\log N)^{\varepsilon_k})\) | `overview.tex` 第 159 条；`preprints/Quasipolynomial-Bounds-for-Arithmetic-Progressions-September-23-2026/paper.pdf` |
| 197 | Kaplansky 直接有限性（正特征） | 反例：存在有限表示无挠群 \(G\)，使 \(\mathbb F_2[G]\) 不是直接有限的（\(ab=1\not\Rightarrow ba=1\)）；并构造非 sofic 群、否定 Gottschalk 与行列式猜想等 | `overview.tex` 第 197 条；`preprints/A-Counterexample-to-Kaplanskys-Direct-Finiteness-Conjecture-in-Characteristic-Two-September-23-2026/paper.pdf`；`lean/docs/197.md` |
| 221 | 稀释自旋玻璃的 Mézard–Parisi 公式 | 在 Panchenko–Talagrand 分解与正性假设、仅一阶矩可积下，证明热力学极限存在且等于有限 RSB 层级变分泛函的下确界；覆盖 Viana–Bray、对称稀释偶自旋、加权 soft even-\(K\) SAT | `overview.tex` 第 221 条；`preprints/The-Mezard-Parisi-formula-for-diluted-spin-glasses-September-23-2026/paper.pdf` |
| 271 | 量子 Heisenberg 铁磁体的自发磁化 | 证明 \(d\ge3\)、任意自旋 \(S\) 的最近邻铁磁体在足够低正温度存在平移不变 \(\beta\)-KMS 态且自发磁化非零；并证明 Bloch \(T^{3/2}\) 律及三维格点修正 | `overview.tex` 第 271 条；`preprints/Spontaneous-magnetization-in-the-quantum-Heisenberg-ferromagnet-September-24-2026/paper.pdf` 等 |
| 287 | 自由群因子的同构 | 证明 \(L(\mathbb F_2)\cong L(\mathbb F_3)\)，从而所有可插值自由群因子（含 \(L(\mathbb F_\infty)\)）同构；公共因子的 fundamental group 为 \(\mathbb R_{>0}\) | `overview.tex` 第 287 条；`preprints/An-isomorphism-of-the-free-group-factors-September-23-2026/...pdf` |
| 362 | 三维相对论 Vlasov–Maxwell | 证明大初值全局光滑解：对任意光滑 admissible 初值（紧支集、有限能量、所有阶导数有界），\(T_{\max}=\infty\)，解在任意有限时间光滑 | `overview.tex` 第 362 条；`preprints/Global-classical-solutions-of-the-three-dimensional-relativistic-Vlasov-Maxwell-system-September-23-2026/paper.pdf`；`lean/docs/362.md` |

## 评分口径

- **5 分**：完整证明/反例，方向与官方一致，关键步骤可核对（反例最好有计算证书）。
- **4 分**：方向与官方一致，主定理基本证出，但有一处可补的次要缺口。
- **3 分**：实质性部分结果：正确的关键引理、特殊情形、或清晰的路线 + 已证中间步骤。
- **2 分**：正确复述背景与已知结果，给出一些正确但非核心的观察。
- **1 分**：方向或论证有致命错误，或把猜想当定理。
- **0 分**：无有效数学内容 / 答非所问。

注意：官方结论本身是“论文声称”，其中部分有 Lean 形式化（007? 017, 087, 197, 271?, 362?），
部分没有；`~42%` 的 top-line 结果有形式化。评分只判断“是否复现了官方声称的方向与主定理”，
不把官方声称当作已独立验证的真理。
