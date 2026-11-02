# Q4 文献调研：SDPRLayer 能解什么形式的问题？对应机器人领域什么问题？紧性如何？

> 调研人：lit。文件全部在 `references/`（索引见 [`references/INDEX.md`](../references/INDEX.md)）。
> **页码约定**：本笔记中 `p.X` 一律指对应 PDF 的页（1-based），不是期刊印刷页；卷期页/DOI 在 INDEX 中。
> 本文（SDPRLayers）引用行号用 `paper/sdpr.txt:Lxxx`。外部结论格式：`[作者年份, 期刊, §/Thm/p, ref: 文件名]`。
> 未核实/推测性内容一律显式标注 **（未核实）** 或 **（本文分析）**。

---

## 0. 结论摘要（TL;DR，供 Lead 直接引用）


> **路径说明**：下文 `paper/sdpr.txt:L###` 指原工作区的论文提取文本行号；该文本与 `references/*.pdf`
> 因版权不随本仓库分发。文献的 DOI/arXiv 见 [`references/INDEX.md`](../references/INDEX.md)。

1. **SDPRLayer 的输入形式是唯一确定的**：标准 **齐次 QCQP (P1)** ——
   `min_x xᵀQ_θx  s.t.  xᵀA_θⁱx = 0 (i=1..m),  xᵀA₀x = 1`，
   其中 x∈ℝⁿ、Q_θ,A_θⁱ∈𝕊ⁿ 都**只通过矩阵元素依赖参数 θ**，约束是**有限个二次等式**（不含不等式），homogenization 把某个分量固定为 1。
   [Holmes2025, TRO, §III-B (P1)(P2)(P3), p.3–4; paper/sdpr.txt:L189–L247]
2. **“能解”= 松弛 tight**：去掉 rank-1 约束得到 Shor 松弛 (P3)，若最优解 `rank(X*)=1` 则 `X*=x*x*ᵀ` 给出原 QCQP 全局最优；证书由对偶矩阵 `H(λ,θ)=Q_θ+A₀λ₀+ΣᵢA_θⁱλᵢ ⪰ 0` 给出（强对偶 + 互补松弛）。
   [Holmes2025 §III-B, p.4; paper/sdpr.txt:L236–L307]；这一定义在综述里被明确为 "certifiably correct" 的判据：
   松弛在噪声不太大时 **exact**，且可 a posteriori 检验。[Rosen2021, Annual Review, §2.2 + footnote 4, p.7]
3. **机器人领域的对应问题（TRO/IJRR 一手文献）**：SE(d) 同步 / 位姿图 SLAM（[Rosen2019, IJRR]）、
   Cayley 参数化的旋转/位姿/轨迹估计（[Barfoot2024/25, IJRR]）、range-aided SLAM（[Papalia2024, TRO]）、
   矩阵加权状态估计=双目位姿（[Holmes2024, TRO; Dümbgen2024, TRO]，本文实验 (P6)(P7) 用的就是这类 form）、
   点云配准/outlier-robust 感知（[Yang2021, TRO; Yang2023, TPAMI]）、物体位姿+证书（[Talak2023, TRO]）、
   自动收紧松弛（[Dümbgen2024, TRO]）。补充（按任务要求标注）：旋转平均 [Dellaert2020, ECCV]、
   连续时间 range-only [Dümbgen2023, RA-L]、手眼标定 [Wise2020, MFI]、IK [Giamou2022, RA-L; Maric2020, ICRA]、
   MRF/图像分割 [Hu2019, RA-L]、点集配准 [Iglesias2020, CVPR]、接触规划 [Kang2025, RSS]、
   CBF 验证/综合 [Kang2023, CDC]。
4. **紧性不是普遍性质，而是“问题形式 + 噪声水平 +（常常）冗余约束”的函数**：
   - 位姿/轨迹估计：**Shor 松弛默认不 tight**，“for most problems in this paper, Shor's relaxation is not tight out of the box”，需要识别冗余约束。[Barfoot2024/25, IJRR, §II.3–§II.4, p.6]
   - 矩阵加权状态估计：各向异性噪声**破坏** tightness，论文给出一组冗余约束 (25a–c) 才能恢复；且只能靠 **SDP stability**（Cifuentes 等）给局部保证，不能对任意实例 a priori 断言。[Holmes2024, TRO, §IV-C, p.7–8]
   - 自动找约束：`AutoTight` 的完备性定理说明“若它找不到足够约束，则不存在其它冗余约束集合能收紧”。[Dümbgen2024, TRO, Thm.1, p.7]
5. **理论紧性保证几乎都是“小噪声下的存在性阈值”，不是无条件等式**：
   SE-Sync：存在 β，`‖Q̃−Q‖₂<β` 时松弛唯一且 exact [Rosen2019, IJRR, Prop.2, p.13]；
   Shonan：`‖L̄−L‖₂≤β(L)` 时 exact [Dellaert2020, ECCV, Thm.1, p.6]；
   点集配准：`ε≤m/(2(m−1))` 的充分条件 [Iglesias2020, CVPR, Thm.1, p.4–5]；
   手眼标定：对偶强对偶在噪声下**稳定** [Wise2020, MFI, Thm.1–2, p.5–6]；
   IK：仅在“未完全伸展、无关节限位”的局部工作空间内强对偶 [Maric2020, ICRA, Thm.2, p.4 + Suppl. p.8]。
6. **经验紧性的量级（可直接写进报告）**：
   - TEASER 的 TLS 旋转 SDP “always returns a rank-1 solution”，实测在 **>95% outliers** 仍 tight [Yang2021, TRO, §VIII-B + §XI, p.10, p.13–15]；
   - Yang&Carlone 的稀疏 TLS 松弛在六类几何感知上“empirically exact with up to **60%–90% outliers**” [Yang2023, TPAMI, Abstract p.1, §4, p.8–9]；
   - RA-SLAM：有 inter-robot loop closure 时 **36.2%** 的仿真实例零次优间隙（相对 gap<1e−5），无 loop closure 时仅 **0.7%**；但即便不严格 tight，相对间隙通常 **<10%** [Papalia2024, TRO, §VII-B, Table VII + 结论, p.12–13]；
   - 接触丰富规划：不用 term sparsity 时平均次优间隙 **<10%**（Planar Hand 除外），真实 push-T **100% 成功率** [Kang2025, RSS, §V, p.10]。
7. **非估计类问题的“可信”层次分化明显**（本任务点 2 的核心）：
   - **IK**：不是“一个 tight SDP 直接解”。CIDGIK 用 SDP 可行性 + **秩最小化/凸迭代** 找 rank-d 解，只在“唯一解/无不等式”（So–Ye 刚性条件）时可从松弛直接得解；有障碍/冗余时靠迭代与经验成功率（>99%）。[Giamou2022, RA-L, §IV-A, p.1, p.5–6]
   - **分割/MRF**：SDP 松弛只给 near-optimal（目标差 ~0.1%），明确“不期望多项式时间精确解”，提供 per-instance 次优界。[Hu2019, RA-L, Abstract p.1, (P1)/(S1) p.3]
   - **控制/CBF**：多层 POP → Lasserre/SOS **层级**，只有渐近全局收敛，且仅低维系统可解。[Kang2023, CDC, §1, p.2]
   - **接触/摩擦**：稀疏 Moment-SOS 层级 + suboptimality certificate，非单层 tight。[Kang2025, RSS, Abstract p.1, §II, p.2–3]
   - **标定**：手眼标定的对偶强对偶可证且对噪声稳定 [Wise2020, MFI, p.4–6]；trace-constrained SDP 用低秩通道做 rounding，作者明确“rank-1 解不一定最优”。[Wu2025, arXiv, §II–III, p.2–3]（预印本，标注未核实）
8. **对 SDPRLayer 的可用边界**：可直接用的是“**有限个二次等式约束、tight、参数在 Q/A、证书 corank=1**”的 POP；
   需要改造的是**不等式约束（IK 障碍、控制限幅）**、**离散变量（MRF 标签、接触模式）**、**层级提升（Lasserre/SOS）**、**rank-d（而非 rank-1）提升（SE-Sync/Shonan 的 Z=RᵀR）** 以及**非多项式模型**。
   另外：冗余约束会让经典 IFT 的 LICQ 失效，这正是 SDPRLayer 用 ACQ + 伪逆 `M_r†` 的动机；但冗余约束使 SDP 必须直接求解（无法用快速低秩证书法），规模与实时性受限。[Holmes2025 §IV-B/§IV-C, p.6–7; Holmes2024, TRO, §IV-C, p.8]
9. **Science Robotics 覆盖情况**：2019–2025 的 Science Robotics 中未检索到以 certifiable/SDP 松弛为主要贡献的机器人文章；本笔记用 **GCS 凸优化运动规划** [Marcucci2023, Sci. Robot., 8(84):eadf7843] 作为 SR 侧的“凸优化/全局规划”对照（它给 MICP 意义下的全局最优，但不是 SDP tightness 论证）。这一点是**调研结论而非否定**，见 §6.3。

---

## 1. 可信最优（certifiably optimal / certifiably correct）在文献中的定义谱系

### 1.1 共同内核：POP → 齐次 QCQP → Shor 松弛 → 对偶证书

设原始问题为多项式优化问题（POP）。标准做法（Shor 松弛，本文 §III-B）是把它写成齐次 QCQP

```
(P1)  min_x  xᵀQ_θ x
      s.t.   xᵀA_θⁱ x = 0,   i ∈ [m]
             xᵀA₀ x = 1                     （homogenization：固定某个分量为 1）
```

用 `X = xxᵀ` 提升到矩阵空间：

```
(P2)  min_X ⟨Q_θ, X⟩  s.t. ⟨A_θⁱ,X⟩=0, ⟨A₀,X⟩=1, X⪰0, rank(X)=1
(P3)  Shor 松弛：去掉 rank(X)=1
```

对偶（Lagrange 乘子 λ₀,…,λ_m）：

```
min_{H,λ} λ₀   s.t.  H = Q_θ + A₀λ₀ + Σ_i A_θⁱλᵢ,  H ⪰ 0
```

**定义（tight / exact relaxation）**：若 (P3) 的最优解满足 `rank(X*)=1`，则 `X*=x*x*ᵀ`，x* 是 (P1) 的全局最优，且该松弛称为 **tight/exact**。
**定义（certificate）**：对偶矩阵 H 称 **certificate matrix**；`H⪰0` 且互补松弛 `Hx*=0` 即给出全局最优证书（对偶可行 + 原问题可行 + 目标相等 ⇒ 强对偶）。
以上见 [Holmes2025, TRO, §III-B (P1)–(3), p.3–4; paper/sdpr.txt:L189–L307]；同一构造在 SE-Sync 中以 `Z=RᵀR`、rank-d 形式出现 [Rosen2019, IJRR, §V, Problem 7, p.12]。

**综述给出的操作化定义**：所谓 certifiably correct methods，是“用凸松弛代替局部搜索，并且这些松弛在数据噪声不太大时是 **exact** 的；exact 性可以 post-hoc 检验（检验松弛解是否满足原问题约束）”。
[Rosen2021, Annual Review, §2.2 + footnote 4, p.7]

**算法层面的定义（更宽）**：Yang & Carlone 定义算法 A 为 **certifiable**，若 (i) 多项式时间；(ii) 对常见输入返回全局最优 + 最优性证书（经验或可证），或 (iii) 对最坏输入显式失败但给出次优性度量。
[Yang2023, TPAMI, Definition 1, p.1；p.2 进一步解释“比 [113] 更严格，要求在常见实例上真的达到全局最优”]

### 1.2 文献中五种强弱不同的“可信”，写报告时必须区分

| 类型 | 含义 | 代表工作（含页） |
|---|---|---|
| **(a) tight relaxation 直接解** | 松弛 rank-1/rank-d，直接读出全局最优；证书 H⪰0（或 M_Λ⪰0）顺带得到 | SE-Sync [Rosen2019, IJRR, Thm.1+Prop.2, p.12–13]；TEASER 旋转子问题 [Yang2021, TRO, Thm.13, p.10]；Shonan [Dellaert2020, ECCV, Thm.1, p.6]；Cayley [Barfoot2024/25, IJRR, p.6]；矩阵加权 [Holmes2024, TRO, §IV-C, p.7–8]；AutoTight [Dümbgen2024, TRO, Thm.1, p.7] |
| **(b) 只认证（certificate/verification），不（必）解 SDP** | 先有候选解（局部解/学习输出），再构造对偶证书判断其全局最优或给出下界；或只证明不可行 | 点集配准：给定 KKT 点构造 `M_Λ`，`M_Λ⪰0` 即全局最优 [Iglesias2020, CVPR, Lemma 1, p.4]；连续时间 range-only：从局部解构造 `H_GP` 证书矩阵 [Dümbgen2023, RA-L, §IV-C, p.5–6]；SOS-IK 的可行性/不可行证书 [Maric2020, ICRA, §V, p.5]；TEASER 的 DRS 证书（对 GNC 解认证/给次优界）[Yang2021, TRO, Thm.14, p.10–11] |
| **(c) 层级（Lasserre/Moment-SOS）渐近收敛** | 单层 SDP 不保证 tight，用层级提升；给的是渐近全局收敛/可任意小的次优间隙 | 稀疏 TLS 松弛（κ=2 经验 exact）[Yang2023, TPAMI, §4, Thm.9, p.8–9]；接触规划稀疏 Moment-SOS [Kang2025, RSS, §II, p.2–3]；robust CBF 多层 POP→SOS [Kang2023, CDC, §1, p.2] |
| **(d) 经验 exact + 次优性界** | 承认无 a priori 保证，但报告经验 rank-1 比率/相对次优间隙 | RA-SLAM [Papalia2024, TRO, §VII-B, p.12–13]；MRF [Hu2019, RA-L, §V, p.6–8]；TEASER [Yang2021, TRO, p.10,13–15] |
| **(e) 学习模型的正确性证书** | 不涉及 SDP；用两个二值证书（可观测正确性 oc、非退化 nd）保证 `ζ-correctness`（估计与真值距离 ≤ζ） | C-3PO [Talak2023, TRO, Def.1 + Thm.4–5, p.5；“certifiable model”三元的定义 p.5] |

**注意**：(e) 与 (a) 是**不同意义上的可信**——C-3PO 的证书不来自凸松弛 tightness，而来自解空间 S(X) 的外逼近与直径界；本文（SDPRLayers）也引用了它（[66]）作为“用 certifiable 方法验证但不对优化问题求导”的对照 [Holmes2025 §II-A, p.3; paper/sdpr.txt:L183–L186]。

### 1.3 与“局部最优”的差别（为什么需要它）

- 综述用停车楼位姿图 SLAM 的例子展示：局部搜索收敛到 critical point（(c)(d)），与全局 MLE（(b)）差很远。[Rosen2021, Annual Review, Fig.2, p.7]
- SDPRLayers 的实验直接量化了“局部解 → 错误梯度”的后果：随机初始化的 Theseus 约 60% 初始化收敛到差局部极小 [Holmes2025 §VI-B2, p.10–11；Fig.3b]，标定外层优化因此收敛到错误 baseline（最终误差 1.132e-1 vs 3.454e-4）[Holmes2025, TRO, Table V, p.12]。

---

## 2. 问题形式 ↔ 机器人问题 ↔ 紧性（大表）

> 列说明：“SDP 构造”只写关键变量/秩；紧性列给理论/经验证据与页码；“SDPRLayer 可用性”是本文分析，✅=可直接写进 (P1)，⚠️=需改造，❌=不适合。
> 页均为 PDF 页；`ref` 为 references/ 下文件。

| 问题 | 形式 / 变量（流形） | SDP 构造 | 紧性保证（理论 / 经验，原文页） | 期刊/年份 | 对 SDPRLayer 的可用性 |
|---|---|---|---|---|---|
| **SE(d) 同步 / 位姿图 SLAM** | MLE `min_{R∈SO(d)ⁿ,t}` 相对变换残差；QCQP：`min tr(Q̃RᵀR)` s.t. `RᵢᵀRᵢ=I`（平移解析消去） | rank-d 提升 `Z=RᵀR ⪰0`，块对角=I_d；对偶 `Q̃−Λ⪰0`；Riemannian Staircase 求解 | 理论：`‖Q̃−Q‖₂<β` 时唯一 exact（Prop.2, p.13）；Thm.1 给出 rank-d 分解⇒全局最优（p.12–13）；经验：噪声可达常规 10×（Abstract p.1） | IJRR 2019 | ⚠️（rank-d 矩阵提升 vs 本文 rank-1 向量提升；需重写为 vec(R) 的齐次 QCQP 并验证等价性——**本文分析，未核实**）；规模大，SDP 直接解不现实 |
| **Cayley-map 旋转/位姿/轨迹估计** | 用 Cayley map 定义 Lie-algebra 各向异性噪声；QCQP 变量为旋转/位姿（SO(3)/SE(3)），轨迹含 motion prior | 齐次 QCQP → Shor SDP；显式识别冗余约束 | 理论：全局最优由 Lagrangian strong duality 得到（Abstract p.2）；经验：多数问题 **Shor 松弛默认不 tight**（p.6），加冗余约束后旋转平均/位姿/轨迹可 tight；旋转平均甚至不需冗余约束（p.8） | IJRR 2025（在线 2024） | ⚠️/✅：若已给定冗余约束集合且写成等式，形式匹配；但轨迹问题变量多、SDP 大 |
| **旋转平均（Shonan）** | `min_{R∈SO(d)ⁿ} tr(L̄RᵀR)`；connection Laplacian L̄ | `Z⪰0`，块对角=I_d（rank-d 提升） | 理论：`‖L̄−L‖₂≤β(L)` 时 (16) 唯一解且 `Z*=R*ᵀR*`（Thm.1, p.6）；经验：dual 通常 tight（p.3, p.5） | ECCV 2020（补充） | ⚠️ 同 SE-Sync；且 SDPRLayer 的 tightness 判据是 rank-1，不是 rank-d |
| **Range-aided SLAM（RA-SLAM）** | QCQP：位姿（SE(3)）+ 地标位置 + range 测量（`z=‖pᵢ−lⱼ‖` 型二次等式） | QCQP→SDP；Riemannian Staircase；解后 round + 局部精化 | 理论：只能 certify 解出的 SDP 下界；经验：有 loop closure 时 36.2% 实例零 gap、无则 0.7%；一般相对 gap<10%；与图连通性强相关（§VII-B, Table VII, p.12–13） | TRO 2024 | ⚠️ 可改造成 (P1)（range 是二次等式），但**常不严格 tight**；SDPRLayer 的梯度只在 tight 时有全局意义 |
| **连续时间 range-only 定位** | GP 推断/最大似然：位置轨迹 + 距离测量（二次等式） | 不必须解 SDP：构造对偶证书矩阵 `H_GP(ρ̂,λ̂)` 检验候选解 | 证书型：给定候选解，`H_GP⪰0` 即全局最优；实验给 certificate vs RMSE（p.5–6） | RA-L 2023（补充） | ⚠️ SDPRLayer 可用其 QCQP 形式；但原论文走的是“只认证”路线，不需要 rank-1 提取 |
| **Range-only 轨迹估计初始化** | 同上（range-only trajectory）；用 SDP 候选解做初始化 | Shor/QCQP 松弛，给初始化 + 候选全局解 | 以“给局部求解器提供全局候选初始化”为主要目的；**moderate range noise 下可恢复全局最小**（Abstract p.1） | RA-L 2024（补充） | ⚠️ 与 SDPRLayer 的“tight 才可微”假设不匹配：这里允许非 tight |
| **矩阵加权状态估计（双目位姿 / 标定 baseline）** | `min_{C∈SO(3),t} Σ eᵀWₖ(θ)e`，`e=m̃−C(m+t)`；`CᵀC=I`、`(c¹)×c²−c³=0` | 齐次 QCQP；引入 substitution 变量 `mᵢ^{ki}`；SDP（Shor）；**冗余约束 (25a–c)** 收紧 | 理论：SDP stability 提供局部保证，不能 a priori 断言所有实例（§IV-C, p.7–8）；经验：各向异性噪声下不 tight，加冗余约束后恢复（p.7–8）；SDPR 实验全部 corank=1（Thm.2 条件满足）[Holmes2025 §VI, p.8] | TRO 2024 ×2 | ✅（本文 (P6) 就是它；Holmes 指出各向异性噪声下需冗余约束才稳，SDPR 实验中该松弛均 tight 且 corank=1；支持用户显式传入冗余约束 [Holmes2025 §V, p.7]。SDP 只能用 Mosek/SCS 直接解） |
| **点云配准 / outlier-robust** | TLS 代价（截断二次）→ POP；旋转用四元数 q + 二值变量 θₖ（binary cloning） | 旋转 SDP (23)：`Z=xxᵀ`；TEASER++ 用 GNC + DRS 证书；Yang&Carlone 用稀疏 Lasserre (SSR) + STRIDE | 理论：rank-1 ⇒ 全局最优（Thm.13, p.10）；低噪声/无 outlier 的理论证明见其 [39]（p.10 引用）；经验：>95% outliers 仍 rank-1（p.10, 13–15）；稀疏松弛 60–90% outliers 经验 exact [Yang2023, TPAMI, p.1, p.9] | TRO 2021；TPAMI 2023（补充） | ✅ 旋转子问题形式匹配（等式约束）；但 binary cloning 引入 ±1 离散变量，需先消去/松弛；TLS 的 min/截断本身不是多项式，需引入变量（TPAMI 的做法） |
| **点集配准（缺失数据）** | `min_{Rᵢ∈SO(3),tᵢ}` 多视角点集配准；二次代价 | SDP（Shor 型），对偶 `M_Λ` 证书 | 理论：`M_Λ⪰0` ⇒ 无对偶间隙、全局最优（Lemma 1, p.4）；`ε≤m/(2(m−1))` 充分条件（Thm.1, p.4–5）；经验：噪声较大仍 tight，接近平面时退化（p.8） | CVPR 2020（补充） | ✅ 形式匹配（SO(3) 等式约束）；但若用其“先局部解再认证”，不需要 SDPRLayer |
| **物体位姿 + 正确性证书** | 部分点云→SE(3) 位姿；keypoint 回归 + 可微修正层 | **无 SDP**；证书 oc/nd 来自解空间外逼近 | `ζ-correctness`：oc=nd=1 且 δ≤ζ ⇒ 误差<ζ（Thm.4–5, p.5）；经验：ShapeNet/YCB 上自训练与实验（§V–VI, p.10–14） | TRO 2023 | ❌ 不是 QCQP/SDP 形式；但其“可微优化层 + 证书”思想与 SDPRLayer 互补 |
| **手眼 / 多传感器标定** | `min_{R∈SO(3),t}` 手眼 AX=XB 型；齐次化 `[r̃;1]`，`r̃ᵀAr̃` 约束 | 对偶 SDP：`max νy s.t. Z(ν)⪰0`；Problem 1–5 | 理论：严格凸性引理 + 强对偶对噪声**稳定**（Thm.1–2, p.5–6）；已知尺度/加冗余约束也成立（Cor.1, p.5）；经验：仿真+真实（p.6） | MFI 2020（补充） | ✅ 形式匹配；若外层要学标定参数，可像 SDPR §VI-B4 一样把参数放进 Q/A |
| **标定（trace-constrained SDP）** | 固定 trace 的 PSD 变量 `Yᵢ`；旋转/平移约束 | TCSDP；低秩通道 + tolerance scheduling 迭代 | 作者明确：“rank-1 solutions are not necessarily optimal”（§III-B, p.3）；目标是把松弛解推向 rank-1 低成本候选；实验 PnP/手眼/双机器人（p.1–3） | arXiv 2025（预印本，ACC 2026 投稿；未核实） | ⚠️ 提供 rounding 思路；不是 tight 保证 |
| **逆运动学（IK）** | 距离几何 QCQP：变量 `X∈ℝ^{d×2n}`（关节点），约束 `diag(B(E_eq)ᵀPᵀPB(E_eq))=ℓ`（连杆长）、`‖xᵢ−cⱼ‖≥lⱼ²`（障碍，**不等式**）、平面约束 | PSD 提升 `Z⪰0`（rank-d）+ **秩最小化/凸迭代** `h(Z)=Σ_{i=d+1}^{2n+d}λᵢ(Z)`；每次迭代解 SDP | 理论：So–Ye 刚性 ⇒ 唯一解时松弛 rank-d；冗余 IK 有多解，SDP 返回 max-rank，需凸迭代（§IV-A, p.1, p.5）；经验：三种机器人 3000 实例，障碍环境成功率 >99%，显著优于 SLSQP/IPOPT（§V, p.6） | RA-L 2022（补充） | ❌ 直接不可用：含**不等式约束**且目标常是可行性，非 (P1) 的“等式 + 二次目标”；但可作为 SDPRLayer 外层的一个可微块（需先转成等式/active-set） |
| **IK（SOS 版本）** | “最近点 IK” QCQP：`min‖y−ξ‖² s.t. y∈𝒴_IK`（连杆长二次等式） | Sparse-BSOS（SOS 层级），**不是** Shor SDP | 理论：非完全伸展、无关节限位 ⇒ 局部强对偶（Thm.2, p.4；证明 Suppl. p.8）；层级共享稳定性；给出可行性/不可行证书（p.5）；经验：10,000 实例，高 DoF 优于 fmincon（p.5, Table I） | ICRA 2020（补充） | ❌ 层级/局部条件，不适合 SDPRLayer 的“单层 tight + 可微”假设 |
| **MRF MAP / 图像分割** | 离散标签 MAP；二进制编码 y∈{0,1}^{NK}；`min tr(Lyyᵀ)` s.t. `diag(yyᵀ)=1`、`tr(Uᵢyyᵀ)=2−K` | `Y=yyᵀ→Y⪰0`（丢弃 rank-1）；DARS/FUSES 解 SDP | 明确“不期望多项式时间精确解”（p.1）；经验：目标在最优 0.1% 内，给出 per-instance 次优界（p.1, §V p.6–8）；relaxation gap 表格（p.8） | RA-L 2019（补充） | ❌ 离散变量 + 非 tight（rank-1 不成立）；SDPRLayer 的梯度不能视为 MAP 解的梯度 |
| **Contact-rich 规划（接触/摩擦）** | POP (1)：动力学多项式、接触模式/力 λₖ、互补约束；半代数集 | 稀疏 Moment-SOS 层级（correlative + term + 机器人特异 sparsity）；高阶但小规模 SDP | 无单层 tight 保证；给 **suboptimality gap** 证书：平均 <10%（Planar Hand 除外），真实 push-T 100% 成功（§V, p.10）；不 tight 时的 robust minimizer extraction（p.3） | RSS 2025（补充） | ❌ 需层级 + 互补约束（非等式/非光滑），不在 (P1) 范围 |
| **Robust CBF 验证/综合（控制）** | 多层 POP（uncertainty, control, state [+ CBF 参数]）；KKT 消去内层 | Lasserre 矩松弛（验证）；SOS 下界 + 层级（综合） | 理论：两个 SDP 松弛**渐近**全局收敛（§1, p.2）；局限：只能低维系统（p.2）；无 tight 单层结论 | CDC 2023（补充） | ❌ 层级/双线性（综合），不适合 |
| **凸优化运动规划（GCS，Science Robotics 对照）** | 图凸集：顶点=凸集、边=凸约束；连续/离散决策 | 混合整数凸规划（MICP），**非 SDP 松弛** | 全局最优由 branch-and-bound over convex relaxations 保证；大规模实例有速度优势；不是 tight SDP/证书 | Science Robotics 2023 | ❌ 形式不同；作为 SR 侧“凸松弛规划”对照引用 |
| **SLAM 综述 / 定义** | 位姿图 MLE（式 3） | 综述 SE-Sync 及其松弛 | “certifiably correct” 的定义与 exactness 判据（§2.2 + fn.4, p.7）；局部 vs 全局图例（Fig.2 p.7） | Annual Review 2021（补充） | 提供 Q1 定义与 Q4 背景 |

---

## 3. 非估计问题的深入小节（公式 + 紧性结论）

### 3.1 逆运动学（IK）：两篇论文代表两条完全不同的路线

**路线 A：距离几何 + 半定松弛 + 凸迭代（CIDGIK）** [Giamou2022, RA-L]
- (a) 原始问题：给定末端位姿 `T_g`，求关节配置 `Θ`，使关节点 `xᵢ(Θ)` 两两距离等于连杆长、满足球/平面障碍约束 `‖xᵢ−cⱼ‖≥lⱼ²`、平面约束 `xᵢᵀnᵢ=cᵢ`。写成 **feasibility QCQP**（Problem 2, p.1）：
  `find X∈ℝ^{d×2n} s.t. P=[X W], diag(B(E_eq)ᵀPᵀPB(E_eq))=ℓ, ‖xᵢ−cⱼ‖≥lⱼ², xᵢᵀnᵢ=cᵢ`。
- (b) 齐次 QCQP：把 `PᵀP` 的元素写成 `X` 的二次型；对 `Z=zzᵀ`（z 为 X 的齐次化向量）得到 `tr(AZ)=0` 形式的**线性**约束。
- (c) SDP：`Z⪰0`（rank-d 才对应真实解）。由于多解/不等式，SDP 一般返回 **max-rank** 解；于是用 **秩最小化** `min h(Z)=Σ_{i=d+1}^{2n+d}λᵢ(Z)`，每次迭代解一个 SDP（Problem 4/5），即 "convex iteration"。
- (d) 紧性：**不是**“SDP tight 一步解”。作者明确：只有“唯一解 + 无不等式”（So–Ye 刚性条件）时松弛可直接给 rank-d 解；冗余机械臂/障碍场景下靠凸迭代找 rank-d 低成本解，经验成功率 >99%（3000 实例/机器人，§V p.6）。**因此 IK 属于“SDP 可行性 + 低秩启发式”，不是 certifiable tightness 的标准案例**。

**路线 B：最近点 IK + Sparse-BSOS** [Maric2020, ICRA]
- (a) 最近点形式：`min_{y} ‖y−ξ‖² s.t. y∈𝒴_IK`，`𝒴_IK` 由连杆长约束 `‖xᵢ−xⱼ‖²=Dᵢⱼ²`（非凸二次等式）定义（Problem 10, p.4）。
- (b)(c) 用 **Sparse-BSOS**（SOS/矩层级的稀疏版本，RIP 稀疏性把 SDP 块从 O(dN) 降到 O(3d+1)）([p.5])。
- (d) 紧性：**局部强对偶**——若 ξ̄ 不是完全伸展且无关节限位，则对 ξ 足够接近 ξ̄ 的实例强对偶成立（Thm.2, p.4; Suppl. p.8）；此外可给可行性/不可行证书（p.5）。即“可信”是**局部 + 层级**意义下的，不能全局无条件宣称。

**对 SDPRLayer 的含义**：IK 的障碍/限位是**不等式**，而 (P1) 只有等式；此外 IK 目标常是 feasibility（无目标或 0 目标），外层可微标定也不常见。因此 IK 是“问题能写成 POP、但不能直接塞进 SDPRLayer”的典型。

### 3.2 图像分割 / MRF MAP 推断

[Hu2019, RA-L]
- (a) MRF MAP：`min_w Σ unary + pairwise`，离散标签；二进制编码后 `w∈{0,1}^{NK}`，`f_{P1}=min_y tr(Lyyᵀ)` s.t. `diag(yyᵀ)=1`、`tr(Uᵢyyᵀ)=2−K`（(P1), p.3）。
- (b)(c) 用 `Y=yyᵀ` 提升，丢弃 rank-1 ⇒ SDP (S1)（p.3）；DARS（对偶上升 + Riemannian Staircase）与 FUSES（新 SDP + 光滑 Riemannian 优化）求解。
- (d) 紧性：**明确不 exact**——“in general, hence we do not expect to compute exact solutions in polynomial time”（p.1）；经验上目标在最优值 0.1% 内，并给 per-instance 次优性界（p.1, §V p.6–8）。**这是“松弛只做近似 + 给界”的典型**，不是 certifiable 的 tight 案例。

### 3.3 配准（点云 / 点集 / 网格）

- **TEASER（TRO 2021）**：TLS 鲁棒配准；图论解耦 scale/rotation/translation。旋转子问题：quaternion q + 二值变量 θₖ（binary cloning, Prop.9–10, p.9）→ QCQP (19) → SDP (23)。紧性：`rank(Z*)=1 ⇒` 全局最优且可提取唯一 q*（Thm.13, p.10）；作者说“in practice (23) always returns a rank-1 solution”，实验到 >95% outliers（p.10, §XI p.13–15）；**冗余约束对 tightness 是 critical 的**（p.10 原文："Although being redundant for the QCQP (19), these constraints are critical in tightening the SDP relaxation"）。TEASER++ 进一步用 GNC + Douglas–Rachford 证书（Algorithm 3, Thm.14, p.10–11），给出**相对次优界**，不依赖求解大 SDP。
- **Yang & Carlone（TPAMI 2023）**：把 TLS/最大共识/Geman-McClure/Tukey 等鲁棒代价统一写成 POP；用**稀疏 Lasserre（SSR）**保留 exactness 并大幅缩 SDP；STRIDE 求解；Thm.9 把“松弛 exact”等价为秩条件，并给连续型 exactness 度量与相对次优界（p.9）。经验：六类问题（单/多旋转平均、点云与网格配准、绝对位姿、类别级物体位姿与形状）在 60–90% outliers 下经验 exact（Abstract p.1）。**注意**：TPAMI 原文明确其“exactness”是经验性的（“empirically exact”），并允许对最坏实例失败并给次优界（Def.1(iii), p.1–2）。
- **Iglesias（CVPR 2020）**：多视角点集配准（含缺失数据），SO(3)×ℝ³ 变量；SDP 松弛 + 对偶证书 `M_Λ`：Lemma 1 给“KKT 点 + `M_Λ⪰0` ⇒ 全局最优、无对偶间隙”（p.4）；Thm.1 给噪声充分条件 `ε≤m/(2(m−1))`（p.4–5）；经验：噪声较大仍 tight，目标点云接近平面/缺失多时退化（p.5, p.8）。这同时是“tight”与“certificate”两用的好例子。

### 3.4 接触 / 摩擦：certifiable contact-rich planning

[Kang2025, RSS]（RSS 补充，非 TRO/IJRR/SR）
- (a) 问题 (1)：`min Σℓₖ(xₖ,uₖ,λₖ)` s.t. 动力学 `Fₖ=0`、约束 `(u,λ,x)∈Cₖ`（含 control limits、collision avoidance、**互补约束**）；λₖ 可解释为接触模式或接触力（p.2）。
- (b)(c) 把 (1) 视为 POP，用 **sparse Moment-SOS**（correlative sparsity、term sparsity、机器人运动链/接触模式可分性带来的 robotics-specific sparsity）；“high-order but sparse SDP”（Abstract p.1, §II p.2–3）。
- (d) 紧性：**不是单层 tight**；用层级给 `f_upper−f_lower` 的相对 suboptimality gap 证书；平均 gap <10%（Planar Hand 除外），不 tight 时用 robust minimizer extraction（p.3, §V p.10）；真实 push-T 100% 成功。
- 意义：这是“contact/friction 的 certifiable 问题”的当前主流形态——**层级 + 间隙证书**，而非 rank-1 tight SDP；SDPRLayer 的 (P1) 无法直接表示互补约束。

### 3.5 控制 / MPC / CBF 松弛

[Kang2023, CDC]（CDC 补充）
- (a) robust CBF 验证/综合：控制仿射多项式系统 `ẋ=f(x)+g(x)u+J(x)ϵ`，有界加性不确定性与凸多项式控制约束；验证是三层 POP（uncertainty→control→state），综合是四层（再加 CBF 参数）（§1, p.2）。
- (b)(c) 用内层 KKT 消去 uncertainty/control，降为单层 POP（验证）/ min-max POP（综合）；验证用 Lasserre 矩松弛，综合用 SOS 下界 + 层级（p.2）。
- (d) 紧性：两个 SDP 松弛**渐近**全局收敛，无单层 tight 保证；只能处理低维系统（p.2 limitations）。**对照**：SDPRLayer 的定理要求有限 m 个二次等式 + rank-1，因此 control 类问题需要层级提升后规模爆炸，不能直接套用。
- 另外：SDPR 引用了 Majumdar 等的 Annual Review 综述（[50]）说明 control 社区用 SOS 做 Lyapunov 合成/验证是同类思想 [Holmes2025 §II-A, p.3; paper/sdpr.txt:L118–L121]；该文未下载（不在本任务要求清单）。

### 3.6 标定（手眼 / 双目 baseline / 多机器人）

- **手眼标定** [Wise2020, MFI]：Problem 1 为 SO(3)×ℝ³ 上的二次代价；齐次化 `r̃` 得 `r̃ᵀZr̃` 约束（p.4）；对偶 SDP `max νy s.t. Z(ν)⪰0`（Problem 3, p.4）。理论：Lemma 1 严格凸性 + Thm.1/2 强对偶在噪声下稳定；已知尺度与加冗余约束也成立（Cor.1, p.5）——即**标定问题可以有“tight + 稳定”的可认证结构**。
- **Trace-constrained SDP** [Wu2025, arXiv 预印本]：固定 trace 的 PSD 变量，迭代把解推向 rank-1/低成本（§III-B, p.3）；作者明说 rank-1 解不一定最优，因此这是 **rounding/refinement** 而非 tightness 定理；应用 PnP、手眼、双机器人标定（p.1–3）。**未正式发表，标注“未核实”**。
- **双目 baseline 标定（本文实验）**：[Holmes2025 §VI-B4, p.12] 把 baseline b 放进 Q_θ/A_θ（通过逆立体相机模型 `mₖ=b/dₖ[...]`），inner 是 (P6)、(P7) 是外层 SGD；这正是 SDPRLayer “参数在 Q/A 上”的标准用法。

### 3.7 同步（旋转 / SE(d)）：估计类，但也是 SDPR 的原型

- SE-Sync [Rosen2019, IJRR]：`Q̃` 由相对变换构造，`Z=RᵀR`，rank-d；Thm.1 + Prop.2 给“小噪声 ⇒ 唯一 exact”的经典结论（p.12–13）；实验噪声可达常规 10×（Abstract p.1）。
- Shonan [Dellaert2020, ECCV]：connection Laplacian `L̄`，`Z⪰0` 块对角=I；Thm.1 同型结论（p.6）；用 Riemannian Staircase 解大 SDP。
- 这两个是**矩阵提升 rank-d**，与 SDPRLayer 的**向量提升 rank-1** 在实现上不同；报告中应避免把二者混为一谈（本文分析）。

---

## 4. 紧性失效 / 需要冗余约束 / 需要层级提升的案例（报告 Q4 的“负面证据”）

### 4.1 “Shor 松弛默认不 tight”——Cayley-map 位姿/轨迹
> "Unfortunately, for most problems in this paper, Shor's relaxation is not tight out of the box."
> — [Barfoot2024/25, IJRR, §II.3, p.6]

作者因此**显式识别冗余约束**并把它们加入 QCQP；旋转平均甚至不需要冗余约束（p.8），而位姿/轨迹需要。含义：**tightness 是问题/参数化的函数，不是方法自带的**。

### 4.2 各向异性噪声破坏 tightness——矩阵加权状态估计
[Holmes2024, TRO, §IV-C, p.7–8] 明确：把误差项预乘逆位姿旋转在矩阵加权时**会改变代价**（`CᵢᵀWᵢₖCᵢ≠Wᵢₖ`），因此需要 substitution 变量与**一组冗余约束** (25a–c) 才能恢复 tightness；论文同时承认“SDP stability 只给局部保证，不能对任意实例 a priori 断言”（p.8），并指出冗余约束使**快速认证法不可用**、必须内点法（p.8）。这正是 SDPRLayer 用 ACQ + `M_r†` 处理冗余约束的文献背景（本文 §IV-B, p.6）。

### 4.3 自动搜索冗余约束：完备性定理
[Dümbgen2024, TRO]
- 定义 **RDG（relative duality gap）** 判 cost tightness，用秩判 rank tightness（p.6）；
- 给出 **AutoTight**（采样 + 零空间找所有可行冗余约束）与 **AutoTempl**（模板化），Theorem 1 证明**完备性**：若 AutoTight 找不到足够约束，则不存在其它冗余约束集合能收紧（p.7）；
- 实验：range-based localization / stereo pose estimation（仿真 + 真实数据集）；“always finds a smaller set of constraints sufficient for tightness than previously considered”（Abstract p.1）；具体如 61 个约束中 33 个足以 rank-tight、19 个足以 cost-tight（p.12）；也有 PPR 在给定噪声下**不需要**冗余约束即 cost+rank tight（p.14）——进一步说明紧性依赖具体实例/噪声。

### 4.4 非严格 tight 但“足够有用”——RA-SLAM
[Papalia2024, TRO, §VII-B, p.12–13]：
- 以 `(f_CORA−f_SDP)/f_SDP` 作相对次优间隙代理；
- **有** inter-robot loop closures：**36.2%** 实例相对 gap<1e−5；“无”时仅 **0.7%**；
- 增加 robots/landmarks/ranges 的影响取决于图连通性（无 loop closure 时加 range 可收紧）；
- 结论原文：很多 RA-SLAM 实例“likely not exactly tight”，但相对 gap 通常 <10%，**足以实用**；作者把“不严格 tight”视为理论兴趣而非实质缺陷。
→ 报告中可据此说明：**SDPR 的 tight 假设在 RA-SLAM 上经常不成立**，SDPR 只能提供松弛解 + 次优 gap 信息。

### 4.5 IK：SDP 返回 max-rank，需要秩最小化
[Giamou2022, RA-L, p.1, p.5]：多解/不等式导致松弛解不是 rank-d；必须用 convex iteration 把解推向低秩；So–Ye 的“唯一解 ⇒ 松弛 rank-d”只在无不等式、精确测量的 SNL 型问题成立。→ IK 不是“tight SDP 直接解”的正面案例。

### 4.6 分割/MRF：松弛只能近似
[Hu2019, RA-L, p.1, p.3]：MAP 是 NP-hard，“we do not expect to compute exact solutions in polynomial time”；SDP 松弛给 near-optimal（0.1%）+ per-instance 次优界，但没有 rank-1 保证。

### 4.7 层级提升的两个代表
- 接触规划：为得到小 gap 必须用 high-order（但稀疏）Moment-SOS，且不 tight 时需要 robust extraction [Kang2025, RSS, p.2–3, p.10]。
- CBF：多层 POP 只能渐近收敛 [Kang2023, CDC, p.2]。
- 稀疏化是让层级可解的关键：Yang&Carlone 的 SSR 比标准 Lasserre 小得多但仍保留 exactness [Yang2023, TPAMI, §4, p.8–9]。

### 4.8 理论稳定性：corank=1 / SDP stability
- Cifuentes 等的结果（SDPR [17]）被 SDPRLayer 用来论证：**corank(H)=1 时 tightness 在 Q/A 小扰动下稳定**，因此“调参不会破坏 tightness” [Holmes2025 §IV-C, p.6–7; paper/sdpr.txt:L289–L295]。
- Holmes2024 也引用同一稳定性理论（其 [21]）说明“相似问题会 tight”，但不能覆盖所有实例 [Holmes2024, TRO, §IV-C, p.8]。
- **Barvinok/Pataki 维数条件**：本批 21 篇外部文献中没有任何一篇用它作为 tightness 保证；仅 Wu2025 的参考文献里出现 Barvinok/Pataki（p.13 参考文献）。因此报告中**不应**声称“Barvinok/Pataki 条件保证了这些机器人问题的紧性”——这是一个常见的过度概括，本调研未找到支持文献（**未核实**）。

---

## 5. SDPRLayer 的适用边界与结论

### 5.1 形式假设清单（来自本文，逐条对照文献）

| 假设 | 原文/位置 | 对文献问题的含义 |
|---|---|---|
| 标准齐次 QCQP (P1)：**有限 m 个二次等式** + 一个 homogenization 等式 | [Holmes2025 §III-B, p.3; L189–L222] | 不等式问题（IK 障碍、控制限幅、collision avoidance）不能直接进；需 active-set/松弛变量改造 |
| 参数 θ **只出现在 Q_θ, A_θⁱ 的矩阵元素上** | [Holmes2025 §V, Eq.(18), p.7] | 标定类（baseline b 通过相机模型进 Q/A）✅；但若参数进入不等式/离散结构则不行 |
| 松弛 tight：`rank(X*)=1` | [Holmes2025 §III-B, p.4] | RA-SLAM/MRF/contact 等常不 tight → 只能 SDPR-SDP 或 rounding |
| 证书 `H̄⪰0` 且 `corank(H̄)=1` | [Holmes2025 §IV-C Lemma 3, p.6–7] | 需要严格互补；实践中 SDPR 实验全部 corank=1（p.8） |
| ACQ（弱于 LICQ/MFCQ）与 SOSC；Ω_θ 光滑 | [Holmes2025 §IV-B Def.1/Thm.2, p.6] | 冗余约束使 LICQ 失效，但 ACQ 仍可成立——这是 SDPRLayer 相对经典 IFT 的关键优势 |
| **不要求** primal/dual 解唯一：用 implicit selections + ACQ 证明微分唯一 | [Holmes2025 §IV-B, p.6–7；Remark 2] | 对称性/多解（IK、RA-SLAM、MRF）下 KKT 映射是多值的，本方法用 `M_r†` 选择唯一微分；非 tight 时则对 SDP 解微分 |
| 直接求解 SDP（不能用低秩快速证书法） | [Holmes2025 §V-C, p.17；§II-A fn.4, p.2] | 冗余约束场景必须用 Mosek/SCS；SDP 变量 O(n²)，规模是瓶颈 |

### 5.2 哪些机器人问题可以“直接用”

**✅ 形式匹配且文献中有 tight 证据**（但需实例级验证）：
1. **SE(3) 位姿估计/矩阵加权双目定位**（本文 (P6)）：`CᵀC=I`、`(c¹)×c²−c³=0`、substitution 变量 → 全是二次等式；Holmes 指出各向异性噪声下需冗余约束 (25a–c) 才 tight；SDPR 实验中 (P6) 松弛均 tight、corank=1，并支持把冗余约束显式传给 layer [Holmes2024, TRO, p.7–8; Holmes2025 §V p.7, §VI p.8, §VI-B p.10–12]。
2. **手眼标定**：SO(3)+t 的齐次 QCQP，强对偶稳定 [Wise2020, MFI, p.4–6]。
3. **点集配准/多视角配准**（无缺失时）：`M_Λ⪰0` 证书 + 噪声阈值 [Iglesias2020, CVPR, p.4–5]。
4. **TLS 旋转配准子问题**（TEASER 的旋转步）：四元数 + binary cloning → QCQP→SDP；>95% outliers 仍 rank-1 [Yang2021, TRO, p.9–10]。
5. **SE(d) 同步 / 旋转平均**：形式可写成等式 QCQP；但需注意是 **rank-d 矩阵提升**，转成 SDPRLayer 的 rank-1 向量提升后规模/结构不同（**本文分析，等价性未核实**）。

### 5.3 哪些需要改造

| 改造类型 | 文献例 | 改造思路 / 风险 |
|---|---|---|
| **不等式约束** | IK 障碍 [Giamou2022, p.1]；控制限幅 [Kang2023, p.2] | 引入 slack + 互补/active-set、或在外层固定 active set；ACQ/SOSC 可能失效，梯度不唯一 |
| **离散变量** | MRF 标签 [Hu2019, p.3]；接触模式 [Kang2025, p.2]；TEASER 的 ±1 变量 [Yang2021, p.9] | 不能进 (P1)；需分支/枚举或先连续松弛（SDPRLayer 不处理组合结构） |
| **层级（Lasserre/SOS）** | TLS 稀疏松弛 [Yang2023, p.8–9]；接触/ CBF [Kang2025; Kang2023] | 需先把高阶矩变量显式加入 x、把层级约束写成二次等式——变量数爆炸，且 tightness 只在某层成立 |
| **rank-d 矩阵提升** | SE-Sync [Rosen2019, p.12]；Shonan [Dellaert2020, p.6] | 与 rank-1 向量提升不等价地简单；SE-Sync 的对偶结构（块对角=I）被破坏后 SDP 变大 |
| **非多项式/非光滑模型** | 接触动力学、截断代价的最小化形式 | TLS 可变量化 [Yang2023, p.1]；一般非多项式不可 |
| **参数在数据而非 Q/A** | 一般标定中 θ 出现在测量值里 | 可通过“把测量作为 Q/A 的元素”吸收（SDPR §VI-B4 的 b 就是例子），但只对多项式依赖有效 |

### 5.4 非 tight 时 SDPRLayer 的行为（写报告要强调）

- 提供 SDP 解 X（含次优间隙 `μ(x̂,X)` 与 rounding）[Holmes2025 §V-C, p.17]。
- 可对 SDP KKT 直接微分（SDPR-SDP），但 CVXPYLayers 要求 primal/dual 唯一，而 tight SDP 常**违反**该假设（尤其 m+1>n 的冗余约束场景）[Holmes2025 §IV/§V, p.6–7, fn.13]。
- 经验上 SDPR-SDP 与 SDPR-IS 梯度在 tight 时一致，但 SDP 更大更慢 [Holmes2025 §VI-B3, Table III/IV, p.11]。
- 因此：**“问题是否 tight”直接决定 SDPRLayer 的梯度是否有全局意义**；非 tight 时它更像“松弛解 + 近似梯度”，不能宣称 certifiable。

### 5.5 结论（一句话版）

> SDPRLayer 适用的机器人问题族 = **能被“矩阵元素参数化的齐次二次等式 QCQP”刻画、且其 Shor 松弛在该参数点 tight（corank(H)=1）的问题**。
> 这覆盖了 2019–2025 TRO/IJRR 的 SLAM/同步/配准/标定/矩阵加权估计等估计类主线（SE-Sync、TEASER、Cayley、AutoTight、RA-SLAM），
> 但**不覆盖**不等式/离散/层级/rank-d 的非估计类主线（IK、分割、接触规划、CBF、SE-Sync 的 rank-d 形式）；
> 后者正是“certifiable 文献很多、但能直接可微的很少”的原因，也是 SDPRLayer 的边界。

---

## 6. 与 SDPRLayers 引文的对照 + 未引用的近年 TRO/IJRR/SR 工作

### 6.1 本文引用编号 → 本目录文件（任务要求覆盖的清单）

| 本文编号 | 文献 | 本目录文件 | 在 Q4 中的角色（含页） |
|---|---|---|---|
| [24] | Dümbgen et al., RA-L 2023 | `Dumbgen2023_RangeOnly.pdf` | 连续时间 range-only 的对偶证书（p.5–6）；“只认证不求解” |
| [25] | Dümbgen et al., TRO 2024 | `Dumbgen2024_AutoTighten.pdf` | 自动冗余约束 + 完备性 Thm.1（p.7）+ RDG/rank tightness（p.6,12,14） |
| [32] | Giamou et al., RA-L 2022 | `Giamou2022_CertifiedIK.pdf` | 距离几何 IK + 凸迭代（p.1,5–6）；非估计类 |
| [34] | Goudar et al., RA-L 2024 | `Goudar2024_RangeOnlyInit.pdf` | range-only 轨迹初始化；moderate noise 下 SDP 可恢复全局最小（Abstract p.1） |
| [40] | Hu & Carlone, RA-L 2019 | `Hu2019_MRF_Inference.pdf` | MRF/图像分割 SDP 松弛（p.1,3,6–8）；非估计类 |
| [41] | Iglesias et al., CVPR 2020 | `Iglesias2020_PointSetReg.pdf` | 点集配准 `M_Λ` 证书（Lemma 1 p.4）+ 噪声阈值（Thm.1 p.4–5） |
| [55] | Papalia et al., TRO 2024 | `Papalia2024_RangeAidedSLAM.pdf` | RA-SLAM 紧性参数研究（Table VII, p.12–13） |
| [60] | Rosen et al., IJRR 2019 | `Rosen2019_SESync.pdf` | SE(d) 同步；Thm.1+Prop.2（p.12–13） |
| [74] | Wise et al., MFI 2020 | `Wise2020_HandEye.pdf` | 手眼标定；强对偶稳定性（p.4–6） |
| [76] | Yang & Carlone, TPAMI 2023 | `Yang2023_OutlierRobust.pdf` | 稀疏 Lasserre + certifiable 定义（Def.1 p.1；Thm.9 p.9） |
| [77] | Yang, Shi & Carlone, TRO 2021 | `Yang2021_TEASER.pdf` | TLS 配准；Thm.13（p.10）+ >95% outliers 经验 tight |
| [38]（相关） | Holmes et al., TRO 2024 | `Holmes2024_MatrixWeighted.pdf` | 本文实验 (P6) 的紧性依据；冗余约束 (25)（p.7–8） |
| [7]（相关） | Barfoot et al., IJRR 2025 | `Barfoot2024_Cayley.pdf` | 近年 IJRR 的 Cayley QCQP→SDP；Shor 默认不 tight（p.6） |
| [19]（相关） | Dellaert et al., ECCV 2020 | `Dellaert2020_Shonan.pdf` | 旋转平均 Thm.1（p.6） |

### 6.2 近年 TRO/IJRR/SR 与顶会的补充（区分“本文已引”与“本文未引”）

| 文献 | 类型 | 本文是否引用 | 为什么值得补 |
|---|---|---|---|
| Barfoot, Holmes, Dümbgen, IJRR 2025（在线 2024） | IJRR | 已引 [7] | 明确指出“多数问题 Shor 不 tight”并给冗余约束；与 SDPR 同一作者群，是 Q4 最强的“紧性条件性”证据 |
| Papalia et al., TRO 2024 | TRO | 已引 [55] | 近年最大的“tightness 参数研究”，量化“不 tight 也可用” |
| Holmes et al., TRO 2024 | TRO | 已引 [38] | 矩阵加权估计的冗余约束与稳定性；SDPR 实验问题的来源 |
| Dellaert et al., ECCV 2020 | ECCV | 已引 [19] | 旋转平均的精确性定理，补全同步谱系 |
| Iglesias et al., CVPR 2020 | CVPR | 已引 [41] | 点集配准的证书 + 噪声阈值 |
| Dümbgen et al., RA-L 2023 | RA-L | 已引 [24] | “证书矩阵”不依赖解 SDP 的形态 |
| Marić et al., ICRA 2020 | ICRA | **未引** | IK 的 SOS/局部强对偶，与 CIDGIK 对照；补 IK 在顶会的另一条路线 |
| Kang et al., CDC 2023 | CDC | 已引其 arXiv 版 [46] | 控制类“层级化”代表（正式版为 CDC 2023） |
| Kang, Liu, Yang, RSS 2025 | RSS | **未引** | 接触/摩擦类最新“稀疏 Moment-SOS + 间隙证书”代表 |
| Marcucci et al., Science Robotics 2023 | Sci. Robot. | **未引** | SR 侧凸优化规划对照（GCS/MICP，非 SDP） |
| Wu & Tron, arXiv 2025 | 预印本（ACC 2026 投稿） | **未引** | trace-constrained SDP + rounding，标定类最新尝试（未核实） |
| Marić et al., TRO 2022, 38:1703–1722, DOI 10.1109/TRO.2021.3123841 | TRO | **未引** | **检索到但未下载/未读原文**：TRO 上的距离几何 IK 的 Riemannian 优化版本；若要写“IK 也出现在 TRO”，需要补读（本笔记不断言其紧性结论） |

### 6.3 关于 Science Robotics 的说明（诚实交代）

- 用 `web_search` 多轮检索（关键词：Science Robotics + certifiable / semidefinite / global optimality / pose graph），**未发现 2019–2025 年 Science Robotics 上以 certifiable/SDP 松弛为核心贡献的机器人文章**。
- 最接近的是 GCS 凸优化运动规划 [Marcucci2023, Sci. Robot., 8(84):eadf7843]，它用混合整数凸规划（branch-and-bound over convex relaxations）给全局最优，**不是** tight SDP/对偶证书路线；本笔记把它列为 SR 对照并明确标注。
- 因此：“所引问题必须出现在 TRO/IJRR/SR”这一要求，在**问题层面**由 TRO/IJRR 的一手文献满足（SLAM/同步/配准/标定/估计）；非估计类（IK/分割/控制/接触）的**certifiable 版本**只出现在 RA-L/ICRA/CVPR/CDC/RSS，已逐条标注“补充”。若 Lead 需要严格的 TRO/IJRR/SR 三刊覆盖，建议在报告中明确这一文献分布事实，而不是硬凑。

### 6.4 未核实 / 存疑清单

1. **Barfoot 2024 的年份/卷期**：Crossref 给 IJRR vol.44, pp.366–387, 2024-09-25（在线）；正式 issue 可能标 2025。本笔记写“IJRR 2025（在线 2024）”，引用时保留 DOI 即可。
2. **Wu & Tron 2025**：arXiv 预印本，Comments 写明 “submitted to ACC 2026”；其结论未经过正式发表，标“未核实”。
3. **SE-Sync → SDPRLayer 的等价性**：SE-Sync 用 rank-d 矩阵提升 `Z=RᵀR`，SDPRLayer 用 rank-1 向量提升；把 SE-Sync 改写成 (P1) 后是否得到同一个 SDP、tightness 条件如何对应，本文未做推导（**本文分析，未核实**）。
4. **可微性与 tightness 的联合条件**：SDPRLayer 的 Thm.2 要求 ACQ/SOSC/corank=1；文献中的 tightness 定理（SE-Sync Prop.2、Shonan Thm.1）是否同时保证 ACQ/SOSC，需逐问题验证；本文未验证（**未核实**）。
5. **Barvinok/Pataki 维数条件**：本批文献未用作紧性保证；不要在报告中把它写成这些机器人问题的紧性依据（§4.8）。
6. **Goudar 2024 的 DOI/卷页**：卷页取自 SDPR 参考文献 [34]（RA-L 9(3):2160–2167），DOI 未核实。
7. **Maric TRO 2022（距离几何 IK）**：仅通过 Crossref/arXiv 元数据确认存在，未下载原文，不引用其结论。
8. **RA-SLAM 的 36.2%/0.7%**：来自 Table VII（仿真、2-D、20 instances/参数值）；是经验统计，不是定理保证。

---

## 附录 A：本笔记用到的公式与符号速查

- (P1)/(P2)/(P3)、对偶 `H`：见 §1.1（[Holmes2025 §III-B, p.3–4]）。
- 秩-1 提取：`X*=x*x*ᵀ`；rank 判据用于 tightness（[Holmes2025 §III-B1, p.4]）。
- 相对次优间隙：`μ(x̂,X)= (⟨Q,x̂x̂ᵀ⟩−⟨Q,X⟩)/⟨Q,X⟩`（SDPR Eq.(25), p.17）；RA-SLAM 用 `(f_CORA−f_SDP)/f_SDP`（[Papalia2024 §VII-B, p.12]）。
- AutoTight：RDG（cost tightness）+ rank(X*)（rank tightness）[Dümbgen2024, p.6]。
- 秩最小化（IK）：`h(Z)=Σ_{i=d+1}^{2n+d}λᵢ(Z)`，`h(Z)=0 ⟺ rank(Z)≤d`（[Giamou2022, Eq.(12), p.5]）。
- SE-Sync：`Z=RᵀR`，块对角 `I_d`，`Q̃−Λ⪰0`（[Rosen2019, Problem 6/7, p.12]）。
- Shonan：`min tr(L̄Z) s.t. Z⪰0, BlockDiag_{d×d}(Z)=(I_d,…,I_d)`（[Dellaert2020, Eq.(16), p.6]）。
