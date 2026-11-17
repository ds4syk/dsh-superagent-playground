# 研究日志（periods / μ(π) ≤ 2）

方向：π 作为周期、Picard–Fuchs/模形式、定量代数独立性/超越性度量。目标 μ(π) ≤ 2。
工作目录：`research/pi-exponent-parallel/periods/`。共 8 轮实质推进 + 若干次验证。

## 轮 1：界定与文献

- 确认目标就是「μ(π) ≤ 2」，即对所有 $q$、$p$：$|\pi-p/q|\ge q^{-2-\varepsilon}$。
- 确认已知：Dirichlet $\mu\ge2$；上界链 Mahler 42 → Mignotte → Chudnovsky → Hata 8.0161 (1993) → Salikhov 7.606308 (2008/2010) → Zeilberger–Zudilin **7.103205334137…** (2020)。
- 关键事实：**连 $\mu(\pi)>2$ 都未知**（等价于 π 的部分商无界）。
- 文献边界：只读经典数学（Apéry/Beukers/Hata/Salikhov/ZZ/Baker/Nesterenko/Legendre/Jarník）。未读任何 Family 017 材料、主窗口 `research/pi-exponent/`、PR #30、§9。检索时看到一条 Zenodo 记录（"structural terminal criterion for μ(π)=2"），按边界判断属于禁区周边，**未打开**。

## 轮 2：连分数

- `scripts/cf_pi.py`：200–1200 项，400 位。得到 $\mu=1+\limsup\log q_{n+1}/\log q_n$ 的数值行为，$\tau_n=\log q_{n+1}/\log q_n\ge1$。
- `scripts/cf_pi2.py`：2000 项、2600 位。$\max a_n=20776$（$n\le2000$，$a_{431}$）；$\tau_n\in[1.0000,2.43]$；
- 证明定理 1.2：$\mu=1+\limsup\log q_{n+1}/\log q_n=2+\limsup\log a_{n+1}/\log q_n$。目标 $\iff a_{n+1}=q_n^{o(1)}$。由 ZZ 界得 $a_{n+1}\le q_n^{5.10321+o(1)}$。
- 精度坑：4000 项需要 >4600 位才能算 $\varepsilon_n$；降到 2000 项/2600 位后自洽（$\tau_n$ 不再出现 inf）。

## 轮 3：主障碍（多次修正）

这是最花时间的一轮，经历了三个错误版本：

1. **朴素单边间隙**：若有 $L=A\pi+B$，$q|L|<1/2$，则非例外有理数满足 $|\pi-p/q|\ge1/(2Hq)\ge1/(2q^2)$ ⟹ 似乎 μ≤2。
   **错**：例外情形 $p/q=-B/A$ 无法排除，而它恰是构造本身；对 $\tau<1$ 连 $q|L|<1/2$ 都做不到。这就是文献里该论证看似循环的原因。
2. **单边+密度**：得到 $\mu\le1+\delta$ 之类，自相矛盾（对 ζ(3) 会给出 <2）。
   **错**：漏了下界。
3. **双边+密度**（最终版，定理 2.1）：$cH^{-\tau}\le|L|\le CH^{-\tau}$，高度对数稠密 ⟹ $\mu\le1+1/\tau$（$\tau\le1$）。
   证明中还要处理例外情形的尺度窗口：$\theta\in(1/\tau,\min(M-1,(M+1)/(1+\tau)))$，非空 $\iff M>1+1/\tau$。见 `notes/lemma-type-barrier.md`。
- **发现**：$\tau=1$（唯一能给出 2 的值）的假设 $\iff$ π 坏逼近（部分商有界），严格强于目标。这就是本报告的结论 2。
- 反推：现有构造 $\tau\le0.1638$，故 $1+1/\tau\ge7.10$；要 2 需 $\tau=1$（6.1 倍）。

## 轮 4：周期/PF

- `scripts/legendre_pf.py`（第一版，收敛慢/公式错）→ `legendre_pf2.py`（正确）。
- 验证：PF 方程残差 $10^{-42}$；指数 $(0,0)$；幂单；单值 $\begin{pmatrix}1&2\pi i\\0&1\end{pmatrix}$；第二解 $F\log\lambda+G$，$\mathrm{PF}(G)=F-2(1-\lambda)F'$。
- 关键恒等式：$\int_0^\lambda dx/\sqrt{x(x-1)(x-\lambda)}=2K(\lambda)\to\pi$（数值 5 个 $\lambda$）。
- 节点留数：$\pi=2\pi i\cdot\frac1{2i}$。
- CM 点：$K(-1)=\Gamma(1/4)^2/(4\sqrt{2\pi})$（20 位）；双纽线常数。
- 周期代数：$\zeta(2n)\in\mathbb Q\pi^{2n}$（数值核对 $1/6,1/90,1/945,1/9450$）⟹ $\mathbb Q[\pi]$。
- 教训：PF 方程齐次有理系数，**不能生成 π**；π 只作为初始条件/留数。

## 轮 5：Nesterenko 退化与因子分离

- 命题 4.1（$\kappa_2\le1\Rightarrow\mu\le2$，用 $P=x(qx-p)$）与命题 4.3（「无有理根」限制使因子分离失明）。
- `scripts/barriers2.py`：周期连分延拓构造 $P_n$，$|P_n(\pi)|H(P_n)\in[3.4,8]$ 有界 ⟹ $\kappa_2\ge1$；$-\log|P|/\log H\to0.995$。
- 数值坑：第一版用 $p_n$ 而非 $p_n-a_0q_n$，导致 $|\pi-r_n|=3.0$；修正后 $|\pi-r_n|\approx1.7/q_n^2$。
- Legendre 壁垒数值：$q\le10^5$ 违例 0（约分后）。发现「$<1/q^2\Rightarrow$ 收敛子或半收敛子」为假（$19/6$ 反例），必须用 $1/(2q^2)$。

## 轮 6：ZZ 鞍点


- `scripts/saddles2.py`：相位 $\phi(x)=2\sum\log(x-z_j)-3\log x-3\log(x-10)$，6 个鞍点：2 个实根（${\rm Re}\phi=+9.99$）+ 4 个复根 $4.6460\pm1.4301i$、$5.3540\pm1.4301i$（${\rm Re}\phi=-3.5247729$）。
- `scripts/salikhov_saddle.py`：200 位精度复积分，局部斜率 $\Delta\log|I_n|/\Delta n$ 在 $n=24$ 处 $=-3.524768$，与鞍点 $-3.5247729$ 六位吻合 ✓（$n=28$ 因四鞍点干涉偏到 $-3.566$）。
- $\rho=0.0294584959$；数值接近但**不等于** $(\sqrt2-1)^4=0.0294372515$。
- 复原：$\tau=3.5248/21.51=0.1638\Rightarrow7.1032$；$\tau=0.1026\Rightarrow10.7477$（ZZ 的 crude bound）。$\lambda=21.51$ 由文献 $\tau$ 反推（未独立重推算术记账）。

## 轮 7：测度论与「缺的定理」

- 例外集 $\{\mu\ge2+\varepsilon\}$ 的 Hausdorff 维数 $2/(2+\varepsilon)$（Jarník–Besicovitch），测度 0、补集余贫。软论证无效。
- 正规性不蕴含连分数信息。
- 整理 (T1)–(T4)（主报告 §6）与「必须做什么」（§7）。

## 轮 8：撰写与自检

- 写完 `report.md`、`notes/lemma-type-barrier.md`、`notes/period-structure.md`。
- 自检发现并修正的错误：
  - 表 2.4 中 Beukers ζ(3) 的 $\log(1/\rho)$ 应为整系数型的 0.526（而非积分本身的 3.526），否则 $\tau=0.54$ 与 13.4 矛盾。
  - 推论 2.2 中 $\tau>1$ 的表述（不是「假设不可能」，而是「假设与目标不相容」）。
  - 命题 4.2 的 $|\pi-r_n|\asymp q_n^{-2}\Rightarrow|P|H^{-1/2}$ 拆分错误，改为直接报告 $|P|H$ 有界。
  - `notes` 中构造侧 $\varepsilon H$ 的符号（应为 $\to\infty$，即线性型大于临界尺寸）。
- 全部脚本重跑通过（见 `scripts/verify_all.sh`）。

## 结论

**未证** μ(π) ≤ 2。**已证**：类型壁垒（定理 2.1 + 推论 2.2/2.3）——整类 Apéry/周期方法不可能达到目标；二次路线需要边缘值 $\kappa_2=1$；因子分离失明；周期代数坍缩为 $\mathbb Q[\pi]$。**精确障碍**：$a_{n+1}=q_n^{o(1)}$，或等价的（更强）$\tau=1$ 双边 Dirichlet 临界型 = π 坏逼近。

## 若继续，最值得做的三件事

1. 把定理 2.1 的常数与 (H1)–(H3) 打磨成「无 (H1)、允许多项式因子」的最终形式，并检查能否把 $\max(1+1/\tau,1+\tau)$ 在 $\tau>1$ 一侧改进（当前是平凡等式）。
2. 计算 Salikhov (3,5) 族与更一般的 $(A,B)$ 族的 $\log(1/\rho)$ 与算术 $\lambda$，画出 $\tau(A,B)$ 的上包络，确认 $\sup\tau\approx0.166$（本文只算了 (2,3)）。
3. 检验「$\kappa_2=1$」是否与「π 坏逼近」有形式蕴含（若有，则二次路线与主路线汇合，报告可加强）。

## 附：文件与清理

- 保留：`report.md`、`notes/lemma-type-barrier.md`、`notes/period-structure.md`、`notes/log.md`、`scripts/{cf_pi,cf_pi2,legendre_pf2,barriers2,salikhov_saddle,saddles2}.py`、`scripts/verify_all.sh`。
- 删除：`scripts/barriers.py`、`scripts/legendre_pf.py`（早期版本，前者对未约分有理数误判 Legendre 违例、后者 PF 级数截断不够且 CM 公式写错，均由 `*2.py` 取代）。
- 复跑：`bash scripts/verify_all.sh`（约 2–5 分钟；`cf_pi2.py` 最慢）。
