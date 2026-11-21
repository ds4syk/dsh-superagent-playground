# Problem: the geometric Mahler conjectures

Let \(n\ge2\). A convex body \(K\subset\mathbb R^n\) is a compact convex set with nonempty
interior \(\operatorname{int}K\). Write \(|A|\) for the \(n\)-dimensional Lebesgue volume of a
measurable set \(A\subset\mathbb R^n\) and \(\langle x,y\rangle=\sum_{j=1}^n x_jy_j\) for the
Euclidean inner product.

For \(z\in\operatorname{int}K\), define
\[
(K-z)^\circ=\{y\in\mathbb R^n:\langle y,x-z\rangle\le1\text{ for every }x\in K\},
\qquad
P(K)=\inf_{z\in\operatorname{int}K}|K|\,|(K-z)^\circ|.
\]
The infimum is attained at a unique interior point, the Santaló point. A simplex is the convex
hull of \(n+1\) points \(v_0,\ldots,v_n\in\mathbb R^n\) for which \(v_1-v_0,\ldots,v_n-v_0\) are
linearly independent.

**Part A (general Mahler conjecture).** Resolve the conjecture that, for every integer \(n\ge2\)
and every convex body \(K\subset\mathbb R^n\),
\[
P(K)\ge\frac{(n+1)^{n+1}}{(n!)^2},
\]
with equality if and only if \(K\) is a simplex. The quantifier includes arbitrary convex bodies;
no central symmetry, smoothness, or polytope assumption is imposed.

**Part B (symmetric Mahler conjecture).** For an integer \(n\ge4\), let \(K\subset\mathbb R^n\) be a
compact convex set with nonempty interior, with \(K=-K\) and \(0\) in its interior. Define its
polar by
\[
K^\circ=\{y\in\mathbb R^n:\langle x,y\rangle\le1\text{ for every }x\in K\}.
\]
Prove or disprove that
\[
\operatorname{vol}_n(K)\operatorname{vol}_n(K^\circ)\ge\frac{4^n}{n!}
\]
for every \(n\ge4\) and every such \(K\), and determine the equality cases.

For each part, a complete resolution must either prove the inequality with the stated equality
characterization or give a rigorous counterexample.
