# Problem: NP-hardness at the basic-SDP approximation threshold

Prove or refute the following ordinary NP-hardness assertion for the basic-SDP approximation
threshold of every fixed finite Max-CSP. The deterministic inapproximability consequence, and the
interpretation of an algorithmic counterexample, use the assumption \(\mathrm P\ne\mathrm{NP}\).

**Fixed language and finite inputs.** Fix a nonempty finite alphabet \(D\) and a finite nonempty
collection \(\Gamma\) of predicates \(f:D^{k_f}\to\{0,1\}\), with positive integer arities \(k_f\).
The alphabet, all predicate truth tables, and \(k=\max_f k_f\) are fixed independently of the input
size; they are not growing-alphabet or growing-arity parameters.

An instance \(I\) has finitely many variables \(x_1,\ldots,x_n\) in \(D\) and a finite nonempty list
of constraint occurrences \(e=1,\ldots,m\). Discard unused variables and relabel the variables that
occur as \(x_1,\ldots,x_n\); thus \(n\) is at most the total number of argument positions in the
listed constraints. Occurrence \(e\) specifies \(f_e\in\Gamma\), an ordered tuple
\((v_{e,1},\ldots,v_{e,k_{f_e}})\) of variable indices, and a positive rational weight \(w_e\), with
\(\sum_e w_e=1\). Variable indices may repeat within a tuple, and different occurrences may have the
same tuple or the same underlying scope. Rational numerators and denominators and variable indices
are encoded in binary. This weighted input convention is part of the problem; zero-weight
occurrences can be discarded before normalization. Unweighted instances are the special case
\(w_e=1/m\).

For an assignment \(a:\{1,\ldots,n\}\to D\) define
\[
\operatorname{val}_I(a)=\sum_e w_e f_e(a(v_{e,1}),\ldots,a(v_{e,k_{f_e}})),
\qquad
\operatorname{OPT}(I)=\max_a \operatorname{val}_I(a).
\]
Thus the objective is the normalized weight of satisfied predicates. There is no promise that all
constraints can be simultaneously satisfied.

**The basic SDP.** Use the following specific relaxation, in the vector/local-distribution form of
Raghavendra's \(\mathrm{SDP}(I)\) and its standard BasicSDP formulation. Let \(S_e\) be the set of
distinct variable indices appearing in occurrence \(e\). Introduce real vectors \(u_0\) and
\(u_{v,d}\), for \(v\in\{1,\ldots,n\}\) and \(d\in D\). Introduce a separate probability
distribution \(\mu_e\) on \(D^{S_e}\) for every constraint occurrence \(e\). Here \(\beta\in D^{S_e}\)
means a map \(\beta:S_e\to D\). Maximize
\[
\sum_e w_e\sum_{\beta\in D^{S_e}}\mu_e(\beta)
f_e(\beta(v_{e,1}),\ldots,\beta(v_{e,k_{f_e}}))
\]
subject to all of the following:
\[
\langle u_0,u_0\rangle=1;
\]
\[
\sum_{d\in D}u_{v,d}=u_0\quad\text{for each variable }v;
\]
\[
\langle u_{v,d},u_{v,d'}\rangle=0\quad\text{for }d\ne d';
\]
\[
\mu_e(\beta)\ge0,\qquad \sum_\beta\mu_e(\beta)=1\quad\text{for every }e;
\]
\[
\langle u_{v,d},u_{w,d'}\rangle
=\sum_{\beta:\,\beta(v)=d,\,\beta(w)=d'}\mu_e(\beta)
\quad\text{for every }e,\ v,w\in S_e,\ d,d'\in D,\ \text{including }v=w;
\]
\[
\langle u_{v,d},u_0\rangle
=\sum_{\beta:\,\beta(v)=d}\mu_e(\beta)
\quad\text{for every }e,\ v\in S_e,\ d\in D.
\]
Here \(\langle\cdot,\cdot\rangle\) denotes the real inner product. The vectors have a common
finite-dimensional real inner-product space; equivalently their Gram matrix is positive semidefinite.
Dimension \(n|D|+1\) suffices to express all such Gram matrices. The value \(\mathrm{BasicSDP}(I)\)
is the maximum of this relaxation. No higher-round consistency or other strengthening is added.

Repeated positions refer to the same \(\beta(v)\), not independently sampled copies. Even when
\(S_e=S_{e'}\), the distributions \(\mu_e\) and \(\mu_{e'}\) are separate: they must agree with the
shared vectors on the specified singleton and pairwise marginals, but are not required to agree on
their entire joint distributions. This convention fixes the relaxation even for duplicate scopes.

**Ratio threshold.** Use the reward-ratio convention, in which larger ratios are better, and define
the real constant
\[
\alpha_\Gamma=\inf\{\operatorname{OPT}(I)/\mathrm{BasicSDP}(I):\ I\text{ is an instance over }
\Gamma\text{ and }\mathrm{BasicSDP}(I)>0\}.
\]
Set \(\alpha_\Gamma=1\) if this set is empty, in particular when all allowed predicates are
identically false. Zero-SDP instances are excluded from the quotient; their objective does not
create a division-by-zero convention. This constant lies in \([0,1]\). It is a threshold: the
problem does not require exact attainment by a single rounding algorithm or a uniform algorithm for
computing \(\alpha_\Gamma\).

**Assertion to establish.** For every fixed \(D,\Gamma\) as above and every fixed rational number
\(r\) with
\[
\alpha_\Gamma<r<1,
\]
there exist rational constants \(s,c\) with
\[
0\le s<c<1,\qquad s<rc,
\]
such that the following promise problem is NP-hard under deterministic polynomial-time many-one
reductions from Boolean satisfiability:
\[
\text{YES: }\operatorname{OPT}(I)\ge c;
\qquad
\text{NO: }\operatorname{OPT}(I)\le s.
\]
A reduction must map every satisfiable Boolean formula into the YES set and every unsatisfiable
formula into the NO set, using only instances of this same fixed \(\Gamma\) and the stated weight
encoding. Behavior outside the two promised sets is unrestricted. The reduction, its polynomial
bound, and \(c,s\) may depend on \(D,\Gamma,r\); they are fixed before the input formula size grows.

This is a single worst-case multiplicative-ratio assertion. It does not require hardness at every
point of an SDP approximation curve, for every SDP-gap witness, or with perfect completeness \(c=1\).
The strict slack in \(r\) and the existence of some separated constants \(c,s\) are intentional. For
\(\alpha_\Gamma=1\) there is no \(r\) in the stated interval and no hardness claim is required. If
\(\alpha_\Gamma<1\), an exact polynomial-time optimizer would also achieve every \(r<1\), so a
separate endpoint assertion for \(r=1\) is unnecessary.

**Complexity interpretation.** Such a gap reduction excludes, under \(\mathrm P\ne\mathrm{NP}\), a
deterministic polynomial-time algorithm that returns an assignment of value at least
\(r\operatorname{OPT}(I)\) on every instance: YES instances would yield a verifiable value above
\(s\) and NO instances could not. No claim excluding bounded-error randomized approximation
algorithms follows here from \(\mathrm P\ne\mathrm{NP}\) alone, and no stronger randomized-complexity
hypothesis is silently imposed.

A complete affirmative resolution establishes the displayed NP-hardness statement for every fixed
language and every indicated \(r\). A complete negative resolution rigorously refutes that universal
hardness assertion under the stated \(\mathrm P\ne\mathrm{NP}\) interpretation. For example, a fixed
language together with a deterministic polynomial-time algorithm and a proved guarantee
\(\operatorname{val}_I(a)\ge\beta\operatorname{OPT}(I)\) for all its instances, for some constant
\(\beta>\alpha_\Gamma\), would suffice under \(\mathrm P\ne\mathrm{NP}\). A resolution is not
required to prove the assumed separation \(\mathrm P\ne\mathrm{NP}\) itself. Failure of one rounding
rule or one attempted reduction is not a negative resolution.
