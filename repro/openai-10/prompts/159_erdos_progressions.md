# Problem: Erdős's reciprocal-sum conjecture

Let \(\mathbb N_0=\{0,1,2,\ldots\}\), and let \(A\subseteq\mathbb N_0\) be a set such that
\[
\sum_{n\in A,\ n\ge1}\frac1n
\]
diverges to infinity.

**Task.** Prove or disprove that \(A\) contains arithmetic progressions of every finite length:
for every integer \(k\ge1\), there are integers \(a\ge0\) and \(d\ge1\) such that
\[
a,\ a+d,\ \ldots,\ a+(k-1)d\in A.
\]
The requirement \(d\ge1\) makes the progression nontrivial for \(k\ge2\).

A disproof must provide a set with divergent reciprocal sum and some fixed length \(k\ge3\) for
which no such progression exists. A proof must cover every \(k\) and every such \(A\).

**Optional quantitative target.** If you can, also prove a quantitative density bound: for each
fixed \(k\ge3\), every subset of \(\{1,\ldots,N\}\) with no nonconstant \(k\)-term arithmetic
progression has size at most
\[
C_kN\exp\left(-c_k(\log N)^{\varepsilon_k}\right)
\]
for positive constants \(C_k,c_k,\varepsilon_k\) depending only on \(k\). This is not required for
a resolution of the main conjecture.
