# Problem: Kaplansky's direct-finiteness conjecture in positive characteristic

Resolve the following unrestricted positive-characteristic form of Kaplansky's direct-finiteness
conjecture, by a complete proof or a rigorous counterexample.

**Target and definitions.** For every prime \(p\), every field \(K\) of characteristic \(p\), every
group \(G\), and every \(a,b\) in the group algebra \(K[G]\), prove
\[
ab=1\implies ba=1.
\]
Here \(K[G]\) consists of finite formal sums \(a=\sum_g a_g g\) with \(a_g\in K\), with convolution
multiplication induced by the multiplication in \(G\); its identity is the basis element corresponding
to the group identity. Groups with torsion are included. Neither \(K\) nor \(G\) is assumed finite.
It is enough to establish the assertion for every finitely generated \(G\) and then explain the
reduction: the union of the finite supports of \(a\) and \(b\) generates a subgroup \(H\), and
\(K[H]\) embeds in \(K[G]\).

The target is direct finiteness of group algebras, not the assertion that every group has no zero
divisors or only trivial units. In particular, proving a zero-divisor theorem only for torsion-free
groups does not resolve this task. Stable finiteness asks the analogous identity \(AB=I\implies BA=I\)
for every finite square matrix size. That stronger conclusion is welcome if proved, but distinguish
it from direct finiteness and justify every claimed equivalence; direct finiteness does not imply
stable finiteness for arbitrary rings.

Check explicitly where finite support, the field characteristic, torsion, and any approximation
assumption enter. Verify the argument recovers the elementary finite-group case, where \(K[G]\) is a
finite-dimensional \(K\)-algebra even when its characteristic divides the group order.

For a counterexample, specify a prime, a field with an exact representation, a well-defined group,
and two finite sums. Certify the multiplication \(ab=1\) and exhibit a nonzero coefficient or other
rigorous certificate for \(ba-1\). A finitely presented group together with unproved word equalities
or inequalities is not a certificate. One-sided shifts in an unrestricted infinite-dimensional
endomorphism ring or an example in a nongroup monoid algebra do not qualify.
