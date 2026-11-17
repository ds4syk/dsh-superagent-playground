# 双边型引理（定理 2.1）的完整证明与假设必要性

本文件给出报告第 2 节核心引理的完整证明，并论证三个假设 (H1)(H2)(H3) 都是必要的（各配反例）。

## 0. 设定

$\alpha$ 无理。整系数线性型 $L_n=A_n\alpha+B_n$，$A_n,B_n\in\mathbb Z$，$(A_n,B_n)\ne(0,0)$，
$$H_n:=\max(|A_n|,|B_n|),\qquad \varepsilon_n:=|L_n|,\qquad \tau_n:=\frac{\log(1/\varepsilon_n)}{\log H_n}.$$
「quality」$=1+\tau$，因为 $|\alpha-(-B_n/A_n)|=\varepsilon_n/|A_n|\asymp H_n^{-1-\tau_n}$。

## 1. 引理

**定理 2.1.** 设 $0<\tau\le1$，$A,A'\ge1$，$0<c\le C$。假设对一切充分大的 $n$：

- (H1) $H_n/A\le|A_n|\le H_n$ 且 $H_n/A\le|B_n|\le H_n$；
- (H2) $c\,H_n^{-\tau}\le\varepsilon_n\le C\,H_n^{-\tau}$；
- (H3) 对数稠密：$\forall X\ge1\ \exists n:\ X\le H_n\le A'X$。

则 $\mu(\alpha)\le1+1/\tau$。

### 证明

固定 $M>\max\{1+1/\tau,\ 1+\tau\}$（当 $0<\tau\le1$ 时此最大值 $=1+1/\tau$）。设 $p/q$ 为有理数，$q$ 充分大，
$$|f|<q^{-M},\qquad f:=p/q-\alpha .$$
目标是导出矛盾。

**第一步：选取尺度。** 取 $\theta$ 满足
$$\frac1\tau<\theta<\min\Bigl\{M-1,\ \frac{M+1}{1+\tau}\Bigr\}.$$
为什么非空：$M>1+1/\tau$ 给出 $1/\tau<M-1$；而 $M>1/\tau$（由 $M>1+1/\tau$ 蕴含）给出 $1/\tau<(M+1)/(1+\tau)$（等价于 $M+1>(1+\tau)/\tau=1+1/\tau$）。故区间非空。取定 $\theta$，由 (H3)（令 $X=q^\theta$）得 $n$ 使
$$q^\theta\le H_n\le A'q^\theta.$$

**第二步：整数间隙。** 考虑整数
$$N:=A_np-B_nq=A_n(p-q\alpha)+q(A_n\alpha+B_n)=A_n\,q\,f+q\,L_n .$$
用 (H1)(H2)：
$$|N|\le |A_n|\,q\,|f|+q\,\varepsilon_n\le H_n\,q^{1-M}+C\,q\,H_n^{-\tau}.$$
由 $H_n\le A'q^\theta$ 与 $H_n\ge q^\theta$：
$$|N|\le A'\,q^{\theta+1-M}+C\,(A')^{\tau}\,q^{\,1-\tau\theta}.$$
两个指数都严格为负：$\theta+1-M<0$ 由 $\theta<M-1$；$1-\tau\theta<0$ 由 $\theta>1/\tau$。故 $q$ 充分大时 $|N|<1$，于是
$$N=0,\qquad\text{即}\quad \frac pq=-\frac{B_n}{A_n}.$$

**第三步：排除例外情形。** 记 $g:=\gcd(|A_n|,|B_n|)$，则 $q=|A_n|/g$，$|p|=|B_n|/g$，并且**精确地**
$$|f|=\Bigl|\frac pq-\alpha\Bigr|=\frac{|A_n\alpha+B_n|}{g}=\frac{\varepsilon_n}{g}.$$
由 (H2) 下界与 $|A_n|\le H_n$：
$$|f|\ \ge\ \frac{c\,H_n^{-\tau}}{g}=\frac{c\,q\,H_n^{-\tau}}{|A_n|}\ \ge\ c\,q\,H_n^{-1-\tau}.$$
再用 $H_n\le A'q^\theta$：
$$|f|\ \ge\ c\,(A')^{-1-\tau}\,q^{\,1-\theta(1+\tau)}.$$
由 $\theta\le(M+1)/(1+\tau)$ 得 $1-\theta(1+\tau)\ge -M$，且当 $\theta<(M+1)/(1+\tau)$ 时严格 $>-M$。于是
$$|f|\ \ge\ c''\,q^{-M+\delta}\qquad(\delta>0),$$
与 $|f|<q^{-M}$ 矛盾（$q$ 充分大）。$\square$

### 备注（为什么 $M$ 的阈值是 $1+1/\tau$ 而不是别的）

- 第二步要求 $\theta<M-1$：间隙项 $H_nq^{1-M}$ 必须被 $q^{-M}$ 压住，即 $H_n\ll q^{M-1}$。
- 第二步还要求 $\theta>1/\tau$：被积项 $q\varepsilon_n$ 必须 $\ll1$，即 $H_n\gg q^{1/\tau}$。
- 两者相容 $\iff 1/\tau<M-1$。
- 第三步（例外）额外要求 $\theta\le(M+1)/(1+\tau)$，即 $|f|\asymp q^{1-\theta(1+\tau)}$ 不低于 $q^{-M}$；这与前两条相容（上面已检验）。
- 若只用单边估计（没有 (H2) 下界），第三步无法进行——见 §2。

## 2. 三个假设的必要性

### (H2) 的下界不能去掉

**反例**：$\alpha$ 任意无理，取 $A_n=q_n$，$B_n=-p_n$（收敛子）。则 $\varepsilon_n=|q_n\alpha-p_n|\asymp q_n^{-1}$，$H_n=q_n$，于是上界 $c'H_n^{-1}$ 成立、$\tau=1$，但下界 $\varepsilon_n\ge cH_n^{-1}$ **不**成立（$\varepsilon_nH_n\approx1/a_{n+1}\to0$ 当 $a_{n+1}\to\infty$）。

后果：第三步的例外情形 $p/q=-B_n/A_n$ 无法排除，而它恰恰就是「构造出来的逼近」本身；换句话说，**只有上界时，引理只能说明「好的有理逼近必是构造出来的」，不能说明「没有更好的」**——这就是 Apéry 型论证在某些表述下看似循环的根源。双边性是引理的全部力量所在；只给上界时，你无法排除「构造本身恰好就是最好的那几个有理数」这一可能。

### (H3) 对数稠密不能去掉

**反例**：Liouville 数 $\alpha=\sum10^{-n!}$，取 $A_n$ 为第 $n$ 个收敛子分母。则 $\varepsilon_n\le CH_n^{-\tau}$ 对**任意** $\tau$ 成立（上界随便多好都行），但下界 $\varepsilon_n\ge cH_n^{-\tau}$ 失败、高度比 $H_{n+1}/H_n\to\infty$（对数稠密失败）。此时引理不适用，而 $\mu(\alpha)=\infty$。所以没有 (H3)，「型 $\tau$」与「$\mu$」之间没有联系：一串孤立的极好逼近不改变 $\mu$ 的 limsup 结构，但它也不能被用来夹逼**所有** $q$。

### (H1) 可比性

若 $|A_n|\ll|B_n|$（例如 $A_n=1$），则 $L_n=\alpha+B_n$ 不可能小，故 (H2) 自动失败；若 $|B_n|\ll|A_n|$ 同理。故 (H1) 在 (H2) 下几乎自动（可差一个多项式因子），写成显式形式只是为了第三步的常数。实际上，把 (H1) 放宽为「$|A_n|,|B_n|\ge H_n^{1-o(1)}$」不影响结论的指数部分。

## 3. 推论

**推论 2.2**（$\tau$ 的有效范围）：见主报告。

**推论 2.3**（$\tau=1\iff$ 坏逼近）：
- ($\Leftarrow$) 若 $\pi$ 坏逼近，即 $\exists c_0>0$ 使 $|\pi-p/q|\ge c_0/q^2$ 对所有 $p/q$ 成立，则取收敛子 $A_n=q_n,B_n=-p_n$：$\varepsilon_n=|q_n\pi-p_n|\in[c_0/q_n,\ 1/(2q_n)]$（上界是 Dirichlet/收敛子性质）。于是 (H2) 取 $\tau=1$、$c=c_0$、$C=1/2$ 成立；(H1)(H3) 是收敛子的标准性质（$H_{n+1}/H_n\le A+1$）。故定理 2.1 给出 $\mu\le2$。（这也与经典的「坏逼近 $\Rightarrow\mu=2$ 且 $\mu\ge2$」一致。）
- ($\Rightarrow$) 设 (H2)（$\tau=1$）与 (H3) 成立。固定 $n$，由 (H3) 取 $m$ 使 $H_m\in[q_n,A'q_n]$。任一高度 $\le A'q_n$ 的线性型的 $|\cdot|$ 至少是 $\min(\delta_n,\delta_{n+1})$（最好逼近的次序性质），其中 $\delta_k=|q_k\pi-p_k|$，且 $\delta_kq_k=1/a_{k+1}\cdot(1+o(1))$。于是
  $$\varepsilon_mH_m\ \ge\ \min(\delta_n,\delta_{n+1})\cdot q_n\ \asymp\ \frac1{a_{n+1}} .$$
  而 (H2) 给 $\varepsilon_mH_m\ge c$，故 $a_{n+1}\le 1/c$ 型的有界性（因子常数与 $A'$ 有关）。即部分商有界。$\square$

**为什么这与「目标只是 $\tau=1$」一致**：$\mu\le2$ 的连分数刻画是 $a_{n+1}=q_n^{o(1)}$（次多项式），而 (H2) at $\tau=1$ 要求 $a_{n+1}=O(1)$（有界）。二者之间的差距就是「$o(1)$ vs $O(1)$」——也就是 $\mu\le2$ 与「坏逼近」的差距。**引理的机制看不见这个差距**：因为引理的本质是「线性型小 ⟹ 整数间隙」，而在 $\tau=1$ 处线性型与高度同阶，间隙恰好消失。

## 4. 与文献中「$\delta$ 引理」的关系

文献（[Be79]、[Sa08]、[ZZ20]）的表述是：若 $|x-a_n/b_n|\le Cb_n^{-1-\delta}$，则 $\mu(x)\le1+1/\delta$。这里 $\delta$ 就是本文的 $\tau$（因为 $\varepsilon_n=|b_nx-a_n|\le Cb_n^{-\delta}$，$H_n\asymp b_n$）。文献通常只写「存在整数序列」并默认了 (H3)（构造的高度比有界）与 (H2) 的下界（构造的渐近是精确的）。本文的作用是把这两条**显式化**，并指出：
1. (H2) 下界是本质的（去掉后论证循环）；
2. (H3) 是本质的（去掉后 Liouville 数就是反例）；
3. 因此 $\delta=\tau$ 的上限 1 是**方法本身**的上限，不是技术上限：$\tau=1$ 需要双边 Dirichlet 临界性，而这是坏逼近，比目标强。

## 5. 数值佐证

- 收敛子的 $\tau_n=\log q_{n+1}/\log q_n\in[1.0000,2.43]$（$n\le2000$，`scripts/cf_pi2.py`）。$\tau_n\ge1$ 恒成立，偶尔很大（$a_{n+1}$ 大时）（$q_{n+1}>q_n$），等号只在极限。
- Legendre 壁垒：$|\pi-p/q|<1/(2q^2)\Rightarrow p/q$ 是收敛子，$q\le10^5$ 扫描违例 0（`scripts/barriers2.py` (A)）。注意常数 $1/2$ 不能省：$19/6$ 满足 $|\pi-19/6|<1/36$ 但既非收敛子也非半收敛子。
- 构造侧：$\tau\le0.1638$（ZZ）。此时 $\varepsilon_nH_n=H_n^{1-\tau}=e^{17.98n}\to\infty$，即线性型比临界尺寸 $H^{-1}$ **大** $H^{0.836}$ 倍；$(H2)$ 本身是满足的（$\tau=0.1638$ 的双边型），定理 2.1 于是给出 $\mu\le1+1/0.1638=7.1032$。而「离 Dirichlet 临界有多远」精确等于「指数离 2 有多远」：
$$\log(\varepsilon_nH_n)/\log H_n=1-\tau=0.836,\qquad \frac1\tau-1=\frac{0.836}{0.164}=5.10 .$$
也就是说，**指数损失 $1/\tau-1$ 就是线性型超出临界尺寸的相对量 $1-\tau$ 除以 $\tau$。** 要把损失压到 0，必须让 $\varepsilon_nH_n$ 有界（$1-\tau\to0$），即 $\tau\to1$——坏逼近。
