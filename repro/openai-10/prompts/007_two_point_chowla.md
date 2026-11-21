# Problem: ordinary two-point correlations of multiplicative functions

Let \(\mathbb N=\{1,2,\ldots\}\) and \(\mathbb D=\{z\in\mathbb C:|z|\le 1\}\). A function
\(f:\mathbb N\to\mathbb D\) is multiplicative if
\[
f(mn)=f(m)f(n)\qquad ((m,n)=1).
\]
For a Dirichlet character \(\chi\), \(t\in\mathbb R\), and \(N\ge 2\), write
\[
D(f,\chi n^{it};N)^2
=
\sum_{p\le N}
\frac{1-\operatorname{Re}\bigl(f(p)\overline{\chi(p)p^{it}}\bigr)}{p},
\]
where the sum is over primes \(p\), \(p^{it}=e^{it\log p}\), and \(D\) denotes the nonnegative
square root.

Say that \(f\) is uniformly non-pretentious if, for every fixed Dirichlet character \(\chi\),
\[
\inf_{|t|\le N}D(f,\chi n^{it};N)\to\infty
\qquad (N\to\infty).
\]

**Conjecture / open problem (binary corrected Elliott conjecture for ordinary averages).**
Let \(f_1,f_2:\mathbb N\to\mathbb D\) be multiplicative functions. Suppose that at least one of
\(f_1,f_2\) is uniformly non-pretentious. Then for every fixed pair of distinct shifts
\(h_1,h_2\in\mathbb Z_{\ge 0}\),
\[
\frac1N\sum_{n=1}^N f_1(n+h_1)f_2(n+h_2)=o(1)
\qquad (N\to\infty).
\]

**Task.** Prove or disprove this conjecture. A complete affirmative resolution must prove the
displayed asymptotic for all admissible \(f_1,f_2,h_1,h_2\) with explicit uniformity in the
\(o(1)\); a complete negative resolution must give a counterexample satisfying all hypotheses.
