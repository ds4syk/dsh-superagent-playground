# 机器与 no-go 定理：紧凑版

（与 `report.md` §3–§4 一致；供快速核对。所有 [证明] 均可在 report 附录 A 找到完整推导。）

## 机器 II 的数据

- 整数 $n\ge1$（变量/插值点数），$d\ge1$（每变量次数），$1\le m\le d$（重数）。
- 有理点 $x_i=p_i/q_i$（互不相同，$(p_i,q_i)=1$，$q_i\ge1$），满足
  $$|\pi-x_i|\le q_i^{-\kappa},\qquad q=\max_i q_i.$$
- $N=(d+1)^n$，$M=n\binom{m+n-1}{n}$，$r=M/N$（要求 $r<1$）。
- 整数矩阵 $B$（$M\times N$）：对每点 $i$ 与 $|a|<m$，行
  $$\sum_{e\ge a} c_e\,\frac{e!}{(e-a)!}\,p_i^{|e-a|}\,q_i^{\,nd-|e-a|}=0.$$
- 核格 $L=\ker_{\mathbb Z}B$，秩 $N-M>0$。

## 三个引理（[证明]）

**(S) Siegel.** 存在非零 $P\in L$：
$$\log H(P)\le\frac{r}{1-r}\bigl(nd\log q+O_{n,d}(1)\bigr).$$

**(D) 可除性.** 对任意 $P\in L$，$Q=P(X,\dots,X)$ 满足
$$Q=G\cdot S,\quad G=\prod_i(q_iX-p_i)^m,\quad S\in\mathbb Z[X].$$
若 $Q\not\equiv0$，则 $H(Q)\ge\prod_i q_i^m\ge q^m$。

**(T) Taylor.** 若 $Q\not\equiv0$，
$$|Q(\pi)|\le C_{n,d,\pi}\,H(Q)\,q^{-m\kappa}.$$

## 平衡

若算术输入 $|Q(\pi)|\ge cH(Q)^{-A}$ 对机器中一切非零 $Q$ 成立，则
$$cH(Q)^{-A}\le C H(Q)q^{-m\kappa}
\Longrightarrow
q^{m\kappa}\le C'H(Q)^{A+1}.$$
因 $H(Q)\ge q^m$，矛盾条件为
$$\kappa>(A+1)\frac{\log H(Q)}{m\log q}\ge A+1.$$
故机器能证明的**最好**上界是 $\mu(\pi)\le A+1$（当 $\log H\approx m\log q$）；一般更差。

## 显式元素（no-go 的核心）

$$P_0=\prod_i(q_iX_i-p_i)^m\in L,\qquad Q_0=P_0(X,\dots,X)=G.$$
$$H(Q_0)=\Theta\Big(\prod_i q_i^m\Big),\qquad
|Q_0(\pi)|=\prod_i|q_i\pi-p_i|^m\le C\Big(\prod_i q_i^m\Big)^{1-\kappa}=C'H(Q_0)^{1-\kappa}.$$
故对任何 $A<\kappa-1$：
$$\frac{|Q_0(\pi)|}{H(Q_0)^{-A}}\le C''H(Q_0)^{A+1-\kappa}\longrightarrow0,$$
即 $|Q_0(\pi)|\ge cH(Q_0)^{-A}$ 不可能。

## 结论

- 要证 $\mu(\pi)\le2$，机器需要 $A\le1$。
- $\kappa>2$ 时 $\kappa-1>1\ge A$，所以 $A\le1<\kappa-1$，$Q_0$ 否定该输入。
- $n=1$ 时 $Q_0=(qX-p)^m$，$A_{\rm eff}=\kappa-1$；方法退化为线性形式，$A=1$ 与目标等价。

**因此：架构 II 不能证明 μ(π)≤2；它只能把线性形式的指数当作输入。** 唯一未被本定理覆盖的是"对特定 Siegel 解的特殊结构下界"——那正是缺失引理，本报告没有发现其证明。

## 常见陷阱（自检清单）

1. **间隙原理方向**：$q_{\rm large}\ge q_{\rm small}^{\kappa-1}/2$ 是"大 ≥ 小"的下界，不能推出 $\prod q_i\ge q^{n}$。
2. **Siegel 高度是上界**：不能把 $H(Q)$ 直接替换成 $q^{m(A+1)}$；只能用 $H(Q)\ge q^m$ 得矛盾条件的下界。
3. **$Q\equiv0$**：Siegel 小高度解可能全在对角理想里（数值上很常见），此时没有值可评估。
4. **$Q_0$ 必须在机器里**：$P_0$ 的每变量次数是 $m\le d$，且在每点 $m$ 阶消失；这是合法的辅助多项式，不是"外部反例"。
5. **量词**：固定 $n,d,m,A$，$q\to\infty$；常数可依赖这些量，不可依赖 $q$。
