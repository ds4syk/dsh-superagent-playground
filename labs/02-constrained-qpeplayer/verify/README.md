# verify/：两个最小数值 sanity check

> **结论一句话**：31 项检查全部通过；验证了"四类约束确实对 $(\mathbf v_2,\mathbf t)$ 仿射""ALM 活跃集配方与源文档公式一致（常数项只来自非活跃约束）""消 $t$ 后内层是四次型""14×14 Shor 在本实例紧到 $2.2\times10^{-12}$""FB 的 $a,b$ 范围与零点集是双曲线""鞍点在秩亏下可逆、$\kappa\asymp1/\varepsilon^2$""$\varepsilon\to0$ 极限是 $H$-加权最小范数解""四脚桌 Jacobian 秩 3"——并顺带**否证**了"$d(\mu^*)$ 的 gap 可作严格证书"（实测 $-1.67\times10^{-11}<0$）。
>
> **证据级别：prototype sanity check，不是论文结果。** 合成实例（$p=12$）、内层用多起点 BFGS 代替 QPEP 多项式求解器，因此耗时无代表性。

## 环境

基座解释器 `/home/syk/miniconda3/bin/python`（numpy 2.5.4 / sympy 1.14.0）**缺 scipy 与 cvxpy**，故改用同机 conda 环境：

```
/home/syk/miniconda3/envs/sdpr/bin/python     # numpy 1.26.3, scipy 1.13.1, cvxpy 1.5.4, SCS
```

复跑：

```bash
cd labs/02-constrained-qpeplayer/verify
/home/syk/miniconda3/envs/sdpr/bin/python sanity_constraints_alm.py | tee sanity_constraints_alm_out.txt
/home/syk/miniconda3/envs/sdpr/bin/python sanity_fb_saddle.py       | tee sanity_fb_saddle_out.txt
```

两个脚本都是确定性的（固定 seed），单次运行 < 20 s。

## 文件

| 文件 | 内容 | 检查数 |
|---|---|---|
| `sanity_constraints_alm.py` | C1 仿射性；C2 ALM 展开；C3 Schur 补+四次性；C4 维数锁定；C5 弱对偶与 14×14 Shor 证书；C6 ALM 正向 | 14 项（C0–C6） |
| `sanity_constraints_alm_out.txt/.json` | 输出与结构化结果 | — |
| `sanity_fb_saddle.py` | F1 $a,b$ 范围；F2 互补点残差；F3 零点集；F4 鞍点可逆/条件数；F5 极限与相容性；F6 四脚桌秩；F7 二维 kink | 17 项 |
| `sanity_fb_saddle_out.txt/.json` | 输出与结构化结果 | — |

## 关键数字（复跑可复现）

| 检查 | 结果 |
|---|---|
| C1 $\mathbf R(\mathbf q)\mathbf p$ 与四类约束的仿射表示 | 误差 $4.3\times10^{-15}$ / $3.7\times10^{-15}$ |
| C2 增广拉格朗日展开 | 误差 $3.6\times10^{-15}$（错误常数项时 $5.6\times10^{-1}$） |
| C3 Schur 补 / 四次性 | $6.2\times10^{-15}$ / 相对 $2.0\times10^{-16}$ |
| C4 $p=256$ 装配 | 尺寸恒 $(10,10)/(3,10)/(3,3)$；4.4 ms（单核 numpy） |
| C5 $d(\mu^*)$ gap | $-1.67\times10^{-11}$（**负，不能当证书**） |
| C5 14×14 Shor（+Veronese+不等式） | $0.0680069355$ vs ALM $0.06800693548$；$\lambda_2/\lambda_1=2.2\times10^{-12}$（**紧**） |
| C6 ALM | 30 次外层，可行性 $2.4\times10^{-11}$；穿透 $18.94\text{ cm}\to2.4\times10^{-9}\text{ cm}$ |
| F1 $a,b$ | $a\in(0,1]$；可行 $b\in[0,1]$；不可行 $b\in(1,2)$ |
| F2/F3 | 强活跃 $\varphi=-\varepsilon^2/(2\mu)$（相对差 $8\times10^{-8}$）；零点集 $\mu g=-\varepsilon^2/2$ 到 $1.8\times10^{-15}$ |
| F4 | 200/200 随机秩亏可逆；$\log\kappa/\log\varepsilon=-2.00$ |
| F5 | 相容：误差 $\to1.3\times10^{-11}$，乘子有界且与分配无关（差 $2.3\times10^{-9}$）；不相容：$\|d\mu\|=10^{-1}\to10^{6}$，$\|dz\|$ 恒 $0.707$；闭式 $H$-加权解误差 $7.0\times10^{-7}$ |
| F6 | 四脚桌 $4\times3$ 秩 $3<4$；$\mathbf D_a=0$ 时鞍点秩 $6/7$ |
| F7 | 二维 kink：$dz/da=0.5$ **对所有 $\varepsilon$**（Clarke 中点）；$\lvert a\rvert\gg\varepsilon$ 时给出 $0/1$ |

## 两个诚实的负面观察

1. **约束没有让合成实例的位姿更准**：无约束 $1.38^\circ/9.44$ cm，约束后 $2.24^\circ/9.96$ cm。约束把穿透从 18.94 cm 压到机器精度，但残差被注入的 10 cm bearing 系统偏差主导，桌面约束修不了它。**不要在论文里假设"约束 ⇒ 更准"**，要按数据集测（见 `experiments/protocol.md` Q-I 预案）。
2. **对偶 gap 是负的**：$-1.67\times10^{-11}$。这直接否证了"$\delta\approx0$ 即 100% 认证"的写法，也是改用 14×14 Shor 下界的直接动机。

## 局限（不要过度解读）

- 内层用多起点 BFGS 代替 QPEP 的 Gröbner 求解，**没有**验证枚举完备性（那是 A2 假设，见 `math_questions.md` Q1）。
- 只有一个合成实例；$14\times14$ Shor 的紧性是**经验**结果（Q4）。
- SDP 用 SCS 默认容差，证书"sound up to $\tau_{\rm sdp}\approx2\times10^{-11}$"；严格证书需区间算术（GAP 3）。
- 未测：真实数据、批处理、GPU、$p>256$、弱活跃多约束（Q2 的推广）。
