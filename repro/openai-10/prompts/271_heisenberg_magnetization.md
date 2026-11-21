# Problem: spontaneous magnetization in the quantum Heisenberg ferromagnet

For every integer \(d\ge3\) and spin \(S\in\{\tfrac12,1,\tfrac32,\ldots\}\), consider the
nearest-neighbor isotropic quantum Heisenberg ferromagnet on \(\mathbb Z^d\) at coupling \(J=1\)
and zero external field. At a site use \(\mathbb C^{2S+1}\) with orthonormal basis
\(e_m\), \(m=-S,-S+1,\ldots,S\), and define \(S^z e_m=m e_m\),
\[
S^+ e_m=\sqrt{S(S+1)-m(m+1)}\,e_{m+1},\qquad S^+ e_S=0,
\]
and put \(S^-=(S^+)^*\), \(S^x=(S^++S^-)/2\), \(S^y=(S^+-S^-)/(2i)\). The operator \(S_x^a\)
acts on site \(x\).

For a finite subset \(\Lambda\) of \(\mathbb Z^d\), the free-boundary Hamiltonian is
\[
H_\Lambda=-\sum_{\{x,y\}\subset\Lambda:\ |x-y|_1=1}\ \sum_{a=x,y,z} S_x^a S_y^a,
\]
where each unordered edge is counted once. Let \(\mathcal A\) be the norm completion of the union
of all finite-site matrix algebras, and let \(\tau_t\) be the infinite-volume time evolution
obtained as the norm limit on local observables of
\(\exp(itH_\Lambda)A\exp(-itH_\Lambda)\) along increasing boxes. This finite-range interaction
defines the limit independently of the exhausting boxes.

A state \(\omega\) is a positive linear functional on \(\mathcal A\) with \(\omega(1)=1\). For
finite \(\beta>0\), call it an equilibrium Gibbs state if it satisfies the \(\beta\)-KMS condition
for \(\tau\): for all \(A,B\in\mathcal A\) there is a bounded function \(F_{A,B}\), continuous on
\(0\le\operatorname{Im}z\le\beta\) and analytic in its interior, with
\[
F_{A,B}(t)=\omega(A\tau_t(B)),\qquad F_{A,B}(t+i\beta)=\omega(\tau_t(B)A)
\]
for real \(t\). Translation invariance means \(\omega(T_x(A))=\omega(A)\) for every lattice
translation \(T_x\).

**Task.** Prove or refute that for every \(d\) and \(S\) specified above there is a finite
\(\beta_0(d,S)>0\) such that for every finite \(\beta\ge\beta_0\) there exists a
translation-invariant \(\beta\)-KMS state \(\omega\) with
\[
(\omega(S_0^x),\omega(S_0^y),\omega(S_0^z))\ne(0,0,0).
\]
The state may depend on \(\beta,d,S\). The target is a zero-field state with spontaneous
magnetization at every sufficiently low positive temperature. It imposes no magnetization
requirement on the rotation-invariant mixture and asserts no pointwise-correlation equivalence. A
negative resolution must give some fixed \(d,S\) and arbitrarily large finite \(\beta\) for which
every translation-invariant equilibrium state has zero magnetization.
