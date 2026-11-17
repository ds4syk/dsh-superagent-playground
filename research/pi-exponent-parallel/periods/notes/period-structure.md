# 周期、Picard–Fuchs、单值与 CM：计算细节

对应主报告第 3 节。所有数字由 `scripts/legendre_pf2.py`、`scripts/salikhov_saddle.py`、`scripts/saddles2.py` 产生（mpmath，工作精度 40–50 位）。

## 1. $2\pi i$ 是 $Z(1)$ 的周期

$$\oint_{|z|=1}\frac{dz}{z}=2\pi i .$$
$H^1(\mathbb P^1\setminus\{0,\infty\})$ 是秩 1 的常数局部系统（Tate 动机 $Z(1)$）。Gauss–Manin 联络平凡，Picard–Fuchs 方程为 $u'=0$。**没有模参数 ⟹ 没有 PF 族 ⟹ 标准「族+PF+单值+分母」机器无输入。**

## 2. Legendre 族与 PF 方程

$$K(\lambda)=\int_0^{\pi/2}\frac{d\theta}{\sqrt{1-\lambda\sin^2\theta}}=\frac{\pi}{2}\,{}_2F_1\Bigl(\frac12,\frac12;1;\lambda\Bigr).$$

**PF 方程**（由超几何方程 $a=b=1/2,c=1$）：
$$\lambda(1-\lambda)F''+(1-2\lambda)F'-\frac14F=0 .$$
数值验证：用 `mpmath.diff(hyper([1/2,1/2],[1],·), ·, 2)` 在 $\lambda=0.13,\,0.6,\,-0.4,\,0.2+0.1i$ 处残差分别为 $0,\ 5.7\cdot10^{-42},\ -2.9\cdot10^{-42},\ -3.6\cdot10^{-43}i$。

**指标方程**：$\lambda=0$ 处 $r(r-1)+r=r^2=0$，指数 $(0,0)$。故 $\lambda=0$ 是**幂单**奇点，第二解为
$$F_2=F(\lambda)\log\lambda+G(\lambda),\qquad \mathrm{PF}(G)=F-2(1-\lambda)F' .$$
（推导：$\mathrm{PF}(F\log\lambda)=\log\lambda\,\mathrm{PF}(F)+\lambda(1-\lambda)(2F'/\lambda-F/\lambda^2)+(1-2\lambda)F/\lambda$，而 $\lambda(1-\lambda)(-F/\lambda^2)+(1-2\lambda)F/\lambda=-F$，故非对数部分为 $2(1-\lambda)F'-F$，取负号得右端。）$G$ 的幂级数逐项可解，$g_0$ 自由。单值（基 $(F,F_2)$）：
$$M_0=\begin{pmatrix}1&2\pi i\\0&1\end{pmatrix},\qquad F\mapsto F,\quad F_2\mapsto F_2+2\pi i\,F .$$

**精确周期恒等式**：对 $0<\lambda<1$，代换 $x=\lambda\sin^2\varphi$：
$$x(x-1)(x-\lambda)=\lambda^2\sin^2\varphi\cos^2\varphi\,(1-\lambda\sin^2\varphi),\qquad \frac{dx}{\sqrt{x(x-1)(x-\lambda)}}=\frac{2\,d\varphi}{\sqrt{1-\lambda\sin^2\varphi}},$$
故
$$\int_0^\lambda\frac{dx}{\sqrt{x(x-1)(x-\lambda)}}=2K(\lambda)=\pi\,{}_2F_1\Bigl(\frac12,\frac12;1;\lambda\Bigr)\ \xrightarrow[\lambda\to0]{}\ \pi .$$
数值（40 位精度）：

| $\lambda$ | 积分值 |
|---|---|
| $10^{-2}$ | 3.1494911230347119053 |
| $10^{-4}$ | 3.1416711978243044721 |
| $10^{-6}$ | 3.1415934389883984227 |
| $10^{-8}$ | 3.1415926614437749166 |
| $10^{-10}$ | 3.1415926536683330548 |
| $\to0$ | $\pi=3.1415926535897932385$ |

## 3. 节点退化与留数：$\pi=2\pi i\cdot\frac1{2i}$

$\lambda\to0$ 时 $y^2=x(x-1)(x-\lambda)\to x^2(x-1)$，在 $x=0$ 有节点。两支 $y\sim\pm ix$，
$$\frac{dx}{y}\sim\frac{dx}{\pm ix},\qquad \text{留数}=\frac1{\pm i}=\mp i .$$
消失圈的周期极限是有限非零（圈长 $\sim\sqrt\lambda$，被积函数 $\sim1/\sqrt{\lambda x}$，乘积有限），值为 $\pi=2\pi i\cdot\frac{1}{2i}$。**π 是 $2\pi i$ 乘一个 Gauss 有理数**——$Z(1)$ 的指纹。这也解释了为什么 π 的定量算术只能来自「$\log(-1)$ 的线性型」。

## 4. CM 点 $\lambda=-1$：$\Gamma(1/4)$

$\lambda=-1$ 对应 $\tau=i$（方形格，$j=1728$）。Chowla–Selberg 给出
$$K(-1)=\int_0^{\pi/2}\frac{d\theta}{\sqrt{1+\sin^2\theta}}=\frac{\Gamma(1/4)^2}{4\sqrt{2\pi}}=1.3110287771460599052\ldots$$
数值验证 20 位吻合；双纽线常数 $\varpi=2K(-1)=\Gamma(1/4)^2/(2\sqrt{2\pi})=2.6220575542921198105\ldots$。

**结构对比**：
- 尖点 $\lambda=0$：周期 $=\pi$，局部单值幂单，定量工具 = 线性型 in $\log(-1)$（Baker）。
- CM 点 $\lambda=-1$：周期 $=\Gamma(1/4)$ 值，定量工具 = Nesterenko 型定量代数独立性（有效指数巨大，远超 Baker）。

**结论**：模形式/CM 方向把 π 的逼近问题转化为 $\Gamma(1/4)$ 的逼近问题，丢番图信息严格**亏损**。π 已经坐在这个族里「最有利」的位置（尖点），而那里也只有指数 7。

## 5. 周期代数 $=\mathbb Q[\pi]$

$$\zeta(2n)=(-1)^{n+1}\frac{B_{2n}(2\pi)^{2n}}{2\,(2n)!}\in\mathbb Q\cdot\pi^{2n}.$$
数值：$\zeta(2)/\pi^2=1/6$，$\zeta(4)/\pi^4=1/90$，$\zeta(6)/\pi^6=1/945$，$\zeta(8)/\pi^8=1/9450$。

故 π 的经典周期（$2\pi i$，$\zeta(2n)$，尖点周期）生成的 $\mathbb Q$-代数 $=\mathbb Q[\pi]$。**推论**：任何「周期线性型」$=\sum q_i\omega_i$ 都化为 $P(\pi)$，$P\in\mathbb Q[x]$。于是「线性型方法」自动坍缩为「辅助多项式/Nesterenko 方法」（主报告 §4）：$d=1$ 即目标，$d\ge2$ 已知度量极弱。

## 6. PF 方程为什么不能生成 π

PF 方程是齐次线性 ODE、系数有理。它只给出周期**之间**的线性关系（例如 $K(\lambda)$ 与 $iK'(\lambda)$ 的关系、Legendre 关系）。π 的出现总是通过**初始条件/留数**：
$$K(\lambda)=\frac{\pi}{2}F(\lambda),\qquad K(0)=\int_0^{\pi/2}d\theta=\frac{\pi}{2}.$$
即：**PF 方程对 π 的数值一无所知**。把它用来做丢番图逼近，等价于直接用 $\log(-1)=i\pi$ 的 Baker 理论。任何「用 PF 结构推出更好指数」的方案都必须先解释 π 的**值**从何而来，而 PF 方程本身不提供。

## 7. ZZ/Salikhov 积分作为同一族的周期（计算）

ZZ「test bunny」：
$$I_n=5i\int_{4-2i}^{4+2i}\frac{(x-5)^{2n}(x-4+2i)^{2n}(x-4-2i)^{2n}(x-6+2i)^{2n}(x-6-2i)^{2n}}{x^{3n+1}(x-10)^{3n+1}}\,dx .$$
（被积函数是有理函数，奇点只在 $x=0,10$；积分是开路径上的周期。等价地，它是该有理函数族的 PF 方程——一个 3 阶递推——的解，[ZZ20] 用 Almkvist–Zeilberger 算法得到。）

相位 $\phi(x)=2\sum_j\log(x-z_j)-3\log x-3\log(x-10)$，$z_j\in\{4\pm2i,5,6\pm2i\}$。$\phi'(x)=0$ 的 6 个根：
- 实根 $x=-3.1449065,\ 13.1449065$，${\rm Re}\,\phi=+9.99203$（增长，与路径不同伦，不贡献）；
- 复根 $x=4.6459909\pm1.4300606i$ 与 $5.3540091\pm1.4300606i$，${\rm Re}\,\phi=-3.5247729$。

故
$$\rho=e^{-3.524772923965284}=0.0294584959281\ldots$$
直接高精度复数值积分（200 位）给局部斜率 $\Delta\log|I_n|/\Delta n$：$n=20$ 时 $-3.4743$，$n=24$ 时 $-3.524768$，与鞍点 $-3.5247729$ 在 6 位有效数字上一致（$n=28$ 的 $-3.566$ 偏大，源于四个简并鞍点的相位干涉）。$\rho\approx(\sqrt2-1)^4=0.0294372515$ 但不等（第 4 位小数分开）。

**指数**：$\tau=\log(1/\rho)/\lambda$，$\lambda=\log H_n/n$ 是整系数化后的高度增长。文献的 $\tau=0.163848$（由 $7.103205334137=1+1/\tau$ 反推）对应 $\lambda=3.5247729/0.163848=21.512$。该 $\lambda$ 由 [ZZ20] 的算术记账（Lemma 1：$2^{-\lfloor(5n+3j)/2\rfloor+1}5^{-j}A_j\in\mathbb Z$；极点阶 $3n+1$；$\mathrm{lcm}$；额外除子 $F(n)$）提供。**我们独立验证的是 $\log(1/\rho)$（鞍点 + 直接积分），未独立重推 $\lambda$。**

## 8. 一个观察（不改结论）

$\rho\approx(\sqrt2-1)^4$ 与 Beukers ζ(3) 积分的衰减常数数值接近，这解释了为什么 ZZ 的 test bunny 与 ζ(3) 的 Beukers 积分给出同量级的 $\log(1/\rho)\approx3.52$；两者的差别全在 $\lambda$（6.53 vs 21.5），也就是全在**算术记账**上。这从另一个角度说明：$\tau$ 的两个因子中，衰减率由**几何**（奇点构型/容量）决定，高度增长由**算术**（分母/$\mathrm{lcm}$）决定；把 $\tau$ 推到 1 需要两者精确相等，这是几何与算术之间的一个等式约束，而非单方面的优化。
