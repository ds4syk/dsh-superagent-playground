# Problem: the irrationality exponent of \(\pi\)

For an irrational real number \(x\), its irrationality exponent (irrationality measure) is
\[
\mu(x)=\sup\left\{\nu>0:\ 0<\left|x-\frac pq\right|<q^{-\nu}
\text{ for infinitely many coprime }p,q\in\mathbb Z,\ q\ge2\right\}.
\]
The pigeonhole principle gives \(\mu(x)\ge2\) for every irrational \(x\). For \(\pi\), known
upper bounds include
\[
\mu(\pi)\le 7.103205334137\ldots
\]
(Zeilberger--Zudilin, 2020). No finite upper bound follows from irrationality alone.

**Task.** Determine \(\mu(\pi)\). In particular, prove or disprove that \(\mu(\pi)=2\), i.e. that
for every real \(\nu>2\) there is an integer \(Q(\nu)\) such that
\[
\left|\pi-\frac pq\right|\ge q^{-\nu}
\qquad (p\in\mathbb Z,\ q\in\mathbb Z,\ q\ge Q(\nu)).
\]
Also determine whether the Flint--Hills series
\[
\sum_{n\ge1}\frac{1}{n^3\sin^2 n}
\]
converges, with angles in radians, and prove the implication you use.

A complete resolution must either prove the stated bound for all sufficiently large denominators
or exhibit infinitely many rational approximations violating it. State clearly which known
theorems you invoke and verify that their hypotheses are met.
