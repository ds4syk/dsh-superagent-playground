# Problem: large-data global classical well-posedness for 3+3-dimensional relativistic Vlasov–Maxwell

Work in the one-species normalized convention. For \(v\in\mathbb R^3\), set
\[
v_0=\sqrt{1+|v|^2},\qquad \hat v=\frac{v}{v_0}.
\]
For a phase-space density \(f=f(t,x,v)\), define
\[
\rho_f(t,x)=\int_{\mathbb R^3} f(t,x,v)\,dv,\qquad
j_f(t,x)=\int_{\mathbb R^3}\hat v\, f(t,x,v)\,dv.
\]

A smooth admissible datum is a triple \((f_0,E_0,B_0)\) such that
\[
f_0\in C_c^\infty(\mathbb R^3_x\times\mathbb R^3_v),\qquad f_0\ge0,
\]
\[
E_0,B_0\in C_b^\infty(\mathbb R^3;\mathbb R^3)\cap L^2(\mathbb R^3;\mathbb R^3),
\]
and
\[
\nabla_x\cdot E_0=\int_{\mathbb R^3} f_0(x,v)\,dv,\qquad
\nabla_x\cdot B_0=0.
\]
Here \(C_b^\infty\) denotes smooth functions with all spatial derivatives bounded.

For such data, consider the Cauchy problem
\[
\partial_t f+\hat v\cdot\nabla_x f+(E+\hat v\times B)\cdot\nabla_v f=0,
\]
\[
\partial_t E-\nabla_x\times B=-j_f,\qquad \partial_t B+\nabla_x\times E=0,
\]
\[
\nabla_x\cdot E=\rho_f,\qquad \nabla_x\cdot B=0,
\]
with
\[
(f,E,B)|_{t=0}=(f_0,E_0,B_0).
\]

Let \([0,T_{\max})\) denote the maximal time interval of the unique local classical solution, where
classical means \(f,E,B\) are \(C^1\), solve the system pointwise, \(E,B\in C([0,T);L^2_x)\), and
\(f\) has compact \((x,v)\)-support on every compact subinterval of \([0,T)\).

**Task.** Determine whether the following assertion is true:

For every smooth admissible datum \((f_0,E_0,B_0)\), one has
\[
T_{\max}=\infty.
\]
Equivalently, every such datum generates a unique global solution such that, for every
\(T<\infty\),
\[
f\in C^\infty([0,T]\times\mathbb R^3_x\times\mathbb R^3_v),\qquad
E,B\in C^\infty([0,T]\times\mathbb R^3_x;\mathbb R^3),
\]
with the above finite-energy and compact-support properties on \([0,T)\).

A negative resolution would be a smooth admissible datum for which \(T_{\max}<\infty\), i.e. the
corresponding local classical solution admits no \(C^1\) continuation as a classical solution beyond
\(T_{\max}\).
