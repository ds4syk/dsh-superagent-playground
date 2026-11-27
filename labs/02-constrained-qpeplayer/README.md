# Constrained-QPEPLayer（第 2 条工作线）

> **一句话定位**：给 QPEP 的代数全局四元数位姿内核加上**物理不等式硬约束**与**秩亏鲁棒的隐微分**，并用一个**与约束数无关的 14×14 半定下界**做后验全局认证——一个 "fail-safe certifiable" 的 6-DoF 位姿层。

## 核心卖点（三条，按重要性）

1. **加性常数维归约**：无论 $p=8$ 还是 $p=256$ 个非穿透/正深度/FoV/倾角锥约束，内层子问题恒为 $(\mathbf v_2,\mathbf t)$ 上的固定尺寸二次型（$10\times10$ / $3\times10$ / $3\times3$），消去平移后是 $\mathbb S^3$ 上的四次型。装配 $O(p)$、内存 $O(1)$、正向**不解 SDP**（对比 SDPRLayers 的 $(10+p)\times(10+p)$ 与 $O((10+p)^6)$）。
2. **$p$-无关的 sound 全局证书**：对原问题做 **14×14 Veronese 增强 Shor 松弛**，PSD 块恒为 14，不等式以线性约束进入。$\mathrm{Shor}\le J^\star$；秩-1 即认证全局最优，不紧则老实报告"未认证"。合成实例上紧到 $2.2\times10^{-12}$，而"对偶 gap"路线给出**负值**（$-1.67\times10^{-11}$）不可用。
3. **Fischer–Burmeister 约化的隐微分**：$p$ 维互补块解析消元到 $(6+|\mathcal A|)\times(6+|\mathcal A|)$ 鞍点系统，在 $\mathbf J_{\mathcal A}$ **任意秩**下可逆（四脚桌 4 接触秩 3 也行）；$\varepsilon\to0$ 极限被精确刻画为 $H$-加权最小范数解，相容时等于真导数、不相容时乘子 $\asymp1/\varepsilon^2$ 爆破且 $dz$ 只是选定广义导数；二维 kink 例中 FB **精确给出 Clarke 中点 $1/2$**。

## 当前完成度

| 模块 | 状态 | 说明 |
|---|---|---|
| 定理 2.1（常数维归约） | ✅ 证明完整 | 修正了源文档的常数项（只有非活跃约束贡献 $-\mu^2/2\rho$）与"$10\times10$ QPEP"的表述（实为提升形式） |
| 定理 2.2（条件全局最优） | ✅ 证明完整 | 三明治 + 负 gap 的正确解读；"100% 认证"改为条件认证 |
| 定理 2.3 + Prop 2.1（FB 隐微分） | ✅ 证明完整 | 可逆性、极限闭式、相容性条件、条件数 $\asymp1/\varepsilon^2$ |
| 新增 Lemma 2.2（14×14 Shor 证书） | ✅ 证明 + 数值 | 本线相对源文档最重要的新增结果 |
| 新增 Lemma 2.4/2.5 + 例 2.5.1 | ✅ | 冗余乘子无关性；弱活跃点的广义导数 + 解析 kink 例 |
| 数值 sanity check | ✅ 2 脚本 / 31 项检查全通过 | `verify/`，合成实例，**不是论文结果** |
| 论文草稿 | 🟡 v0.1 | 结构完整；§5 实验结果为 TODO |
| 实验 | 🟢 合成 + LINEMOD 真实数据 | 合成：4 方法 × p=8/16/32/64 × 50 trials + 4 噪声/偏差档 + 软惩罚 Pareto；代码校验 40/40；证书 400/400 紧。真实 LINEMOD（18 trials）：ALM 0 mm/100% 可行/50% 认证 vs 无约束 4.76 mm/39%。QPEP Gröbner 内层/端到端反向未做 |
| 与 QPEP 求解器的对接 | ❌ 未开始 | 依赖 Q1（根数/模板）结论 |
| PyTorch 层实现 | ❌ 未开始 | 先 NumPy 原型（roadmap W2–W4） |

**一句话**：数学与协议已就绪；合成 + LINEMOD 真实数据主表已完成（`experiments/`），QPEP Gröbner 内层与端到端反向待接。

## 文件导航

```
labs/02-constrained-qpeplayer/
├── README.md                  ← 本文件（项目卡）
├── roadmap.md                 ← 4/8/12 周里程碑、venue 决策、风险
├── math_questions.md          ← Q1–Q10 未解数学问题（P0/P1/P2）
├── related_work.md            ← 逐篇差异点 + 引用核实状态
├── paper/
│   ├── draft.md               ← 英文论文草稿（Title→References，§5 实验 TODO）
│   └── contributions.md       ← 贡献清单 + 15 维对比表 + 9 条审稿人质疑预案
├── theory/
│   ├── proofs.md              ← 定理 2.1/2.2/2.3 + 新增引理，假设/证明/GAP/边界
│   └── derivations.md         ← 关键推导中间步骤（仿射性/ALM 配方/对偶/FB/Hessian）
├── experiments/
│   └── protocol.md            ← 数据集、baseline、指标、消融、算力、复现风险
└── verify/                    ← 两个数值 sanity check 脚本 + 输出 + 结论
    ├── sanity_constraints_alm.py / _out.txt / _out.json
    ├── sanity_fb_saddle.py     / _out.txt / _out.json
    └── README.md
```

## 最近更新（2026-11-27）

- 建立本目录，完成 9 份编排文档（+ 本目录的 verify/README）+ 2 个验证脚本（31 项检查全通过）。
- **修正源文档三处过度声明**：①"内层是标准 4 变量 QPEP"→实为 $\mathbb S^3$ 上四次型（提升形式）；②"$\delta_{\rm gap}\approx0$ 即 100% 认证"→改为条件认证，负 gap 是数值误差，并给出 14×14 Shor sound 证书；③"$\varphi_\varepsilon$ 等价于互补、$\varepsilon\to0$ 精确收敛到真导数"→改为光滑化 + 极限理论（相容性条件）。
- 发现并验证两个实现级坑：ALM 常数项只有**非活跃**约束贡献（写错误差 0.557）；$\mathbf G_{\mathcal A}$ 的 $\rho$ 项必须用白化避免条件数爆炸。
- 支持主线的两处反馈已合入：C2 常数项修正、C5c 检查修正；并采纳"轻量级"建议把 dQP/Theseus 的定位写得更具体。

## 下一步 3 条

1. **Q1（P0）**：统计内层 KKT 系统的通用根数（sympy 商环维数 + 200 组随机实例），确定能否复用 QPEP 的消元模板；这决定前向常数与"用现成 QPEP"的叙事。
2. **Q2（P0）**：把例 2.5.1 从单约束推广到 $m$ 个同时弱活跃约束，确认 FB 是否给出"自然的中点权重"；决定定理 2.3 能否写成"给出 Clarke 广义导数的可刻画元素"。
3. **前向 + 反向 NumPy 原型**（roadmap W2–W4）：ALM 内层接多起点、FB 约化反向、gradcheck；通过后再接 QPEP 求解器与 PyTorch 封装。

## 与另两条线的区别（一句话）

- 不碰多峰/Laplace（Prototype 1），不碰外点/鲁棒核（Prototype 3）；本线只做**约束**，且只做**仿射于 $(\mathbf v_2,\mathbf t)$** 的那一类物理不等式。
