# 数值实验笔记

所有脚本在 `scripts/`，输出在 `data/`。运行环境：`~/miniconda3/bin/python`
（sympy 1.14、mpmath 1.3、matplotlib 3.11，numpy 2.5）。

## 1. 高精度 π 与连分数

`roth_machine.py` 的 `pi_cf`：mpmath 140 位精度，Machin/连分数标准算法，
返回部分商与收敛子 $(p_i,q_i,a_i)$。`approx_exponent(p,q)` 给出
$\kappa\approx-\log|\pi-p/q|/\log q$。

前 22 个部分商：
`3, 7, 15, 1, 292, 1, 1, 1, 2, 1, 3, 1, 14, 2, 1, 1, 2, 2, 2, 2, ...`
（注意 $a_4=292$、$a_{21}=84$ 等较大值。）

## 2. 插值机器（`exp_machine.py`）

对每个 $(n,d,m)$ 和收敛子索引：

1. 构造整数插值矩阵 $B$（$M\times N$）：
   行 $(i,a)$，$|a|<m$，列 $e\in\{0,\dots,d\}^n$，
   元素 $\frac{e!}{(e-a)!}p_i^{|e-a|}q_i^{nd-|e-a|}$（$e\ge a$，否则 0）。
2. `saturated_kernel(B)`：sympy `nullspace` 得 $\mathbb Q$-基，清分母，
   再用 Smith 标准形取饱和（`smith_normal_decomp` 返回 $(S,U,V)$，$S=UKV$，
   饱和基 = $U^{-1}$ 的前 $R$ 列）。**注意**：$\ker_{\mathbb Z}B$ 总是饱和的
   （若 $kx\in\ker B$ 则 $x\in\ker B$），所以饱和是必要且充分的。
3. `lll_reduce`：自实现精确有理 LLL（$\delta=3/4$），把基约化到小高度。
4. 对每个约化基向量 $v$，计算对角限制 $Q(X)=\sum_k(\sum_{|e|=k}v_e)X^k$；
   若 $Q\not\equiv0$，计算 $H(Q)$、mpmath 求值 $|Q(\pi)|$、
   $A_{\rm eff}=-\log|Q(\pi)|/\log H(Q)$。
5. 输出：核维数、$Q\equiv0$ 的个数、最小 $Q\not\equiv0$ 的 $H(Q)$、$A_{\rm eff}$、
   平衡阈值 $(A_{\rm eff}+1)\log H(Q)/(m\log q_{\max})$、
   "要证 $\kappa\le2$ 所需 $A$" $=2m\log q_{\max}/\log H(Q)-1$。

### 结果（`data/machine.csv`）

| n,d,m | q | ker | Q≡0 | Q≠0 | H(Q) | A_eff | κ 阈值 | 需要 A |
|---|---|---|---|---|---|---|---|---|
| 1,1,1 | 33215 | 1 | 0 | 1 | 1.04e5 | 0.988 | 2.21 | 0.80 |
| 2,1,1 | 106,113 | 2 | 1 | 1 | 1.18e5 | 1.296 | 5.67 | −0.19 |
| 2,2,1 | 106,113 | 7 | 4 | 3 | 1.18e5 | 1.198 | 5.43 | −0.19 |
| 2,2,2 | 106,113 | 3 | 2 | 1 | 1.78e10 | 1.283 | 5.70 | −0.20 |
| 3,1,1 | 106,113,33102 | 5 | 4 | 1 | 1.23e10 | 1.119 | 4.73 | −0.10 |

**解释：**
- 小高度解经常全在 $Q\equiv0$（对角理想），说明"非零值"本身是稀缺的。
- 最小 $Q\not\equiv0$ 全是线性因子乘积：
  - n=2,d=1：$Q=(106X-333)(113X-355)$；
  - n=2,d=2,m=1：$Q=X\cdot$上式（额外的 $X$ 因子把值放大、把 $A_{\rm eff}$ 压低）；
  - n=2,d=2,m=2：$Q=G^2$；
  - n=3,d=1：$Q=\prod_{i=1}^3(q_iX-p_i)$。
- $A_{\rm eff}\approx1.1$–$1.3$，**不随 $n,d,m$ 改善**。
- "需要 A" 为负是因为这些具体参数的 $H(Q)\gg q^m$（多因子 + $\pi$ 因子）；
  渐近上定理 4.1 的结论只需 $A\le1<\kappa-1$。

## 3. 连分数乘积（`exp_products.py`）

对 $k=1,2,3,4$ 个连续收敛子，构造 $Q=\prod_j(q_jX-p_j)$，计算
$A_{\rm eff}=-\log|Q(\pi)|/\log H(Q)$。

结果：所有 $k$ 都在 $0.95$–$1.10$，**不随 $k$ 增长**。
原因：$|q_i\pi-p_i|=q_i^{1-\kappa_i}$，故
$$|Q(\pi)|=\prod_i q_i^{1-\kappa_i}=H^{1-\kappa_{\rm eff}},\quad
A_{\rm eff}\approx\kappa_{\rm eff}-1,$$
与因子个数无关。数据见 `data/products.csv`。

**含义：** 用多个线性因子堆高次多项式是"伪高次"；它的有效算术指数仍是线性的 $\kappa-1$。

## 4. 鸽笼极小值（`exp_pigeonhole.py`）

对 $D=1,2,3$ 与若干 $H$，枚举系数 $\in[-H,H]$ 的全部多项式，
记录 $\min|Q(\pi)|$ 与最优多项式的实际高度 $h$。数据 `data/pigeonhole.csv`。

在 $\log_{10} h$ vs $\log_{10}|Q|$ 坐标下，斜率随 $D$ 增大（有限 $H$ 下不稳定；趋势与 $A\ge D$ 一致）。
这说明一般多项式的值下界指数随次数退化（引理 5.1）。

## 5. 图（`make_fig.py` → `data/fig_no_go.png`）

- 左：连分数乘积，所有 $k$ 落在斜率 $-1$ 的参考线上；
- 右：鸽笼极小值，斜率随 $D$ 变陡（虚线是 $-D$ 参考）。

## 6. 复现

```bash
cd research/pi-exponent-parallel/interpolation/scripts
~/miniconda3/bin/python exp_pigeonhole.py
~/miniconda3/bin/python exp_products.py
~/miniconda3/bin/python exp_machine.py
~/miniconda3/bin/python make_fig.py
```

全部脚本约数秒到一分钟；无外部网络依赖。数据为确定性输出（无随机种子）。
