# Laplace-QPEPLayer: Zero-Lift Differentiable Pose Estimation via Quaternion Algebra and Multi-Modal Laplace Smoothing

> **Draft status (2026-11-27, v4).** This draft contains the theory and a results section: a synthetic 6D branch-crossing benchmark (with an FD-step study and a measured fold plus its fold theory), 1D exact-posterior checks, an SDP certificate and the three-part $10\times10$/$13\times13$ relation, a real-data LINEMOD two-mode study with a pose-error/ADD-S proxy and bootstrap CIs, classical forward-solver and Theseus comparisons on the same real correspondences, and a total-degree-homotopy confirmation of the generic $40$-root count. There is still no real-image detector-level accuracy claim: no learned front-end, no detector-level ADD(-S) or ADD-S AUC benchmark, and UTIAS In-The-Dark has not been run. The earlier LINEMOD symmetric-object attempt is retained as a diagnosed negative construction result. Numerical claims are tagged `paper claim`, `independent reproduction (this repo)`, or `unverified`; preliminary numbers from an earlier internal formulation are not reported as results. References are real; entries we could not verify are omitted rather than guessed.

---

## Abstract

Differentiable pose estimation layers sit between learned front-ends and geometric back-ends. Deterministic layers built on semidefinite programming (SDP) relaxations provide global optimality when the relaxation is tight, but they lift the problem to a matrix variable, rely on solver infrastructure, and break down when two global branches cross. Probabilistic layers built on Boltzmann distributions avoid hard branch selection, but typically rely on Monte Carlo sampling in a six-dimensional pose space. We show that the quaternion formulation of quadratic pose estimation problems (QPEP) supplies the missing algebraic structure for a third option. First, for the rotation subproblem obtained after exact elimination of translation, we prove that the standard $10$-dimensional rotation lift is *exactly* $2Uv_2(q)$ for an orthogonal $U$, that the 21 $SO(3)$ constraints span precisely the degree-two vanishing ideal of the Veronese variety $\nu_2(\mathbb{RP}^3)$, and that the resulting $10\times10$ SDP is the Veronese (Shor) relaxation of a quartic on $\mathbb{S}^3$. A constructed Choi–Lam instance exhibits a certified relaxation gap of $0.0342$ with $\mathrm{rank}(X^\star)=6$. Second, we derive a closed-form Euler multiplier $\lambda^\star=4J$ and a $3\times3$ Riemannian implicit differentiation on $\mathbb{S}^3$, including exact antipodal invariance. Third, we replace the hard argmin by a multi-modal Laplace posterior: translation is integrated out exactly, and the rotation integral is approximated analytically over *all* QPEP minima with weights $w_k\propto e^{-J_k/\tau}/\sqrt{\det H_k}$. The resulting layer has no Monte Carlo variance, no SDP lift at inference, and a deterministic backward pass that reduces to one $3\times3$ solve per active mode (plus a trace term for the smoothed KL). We analyze the $O(\tau)$ error of the Laplace approximation, its breakdown at branch merging, and the $1/\tau$ scaling of the KL gradient. We report controlled synthetic branch-crossing experiments, 1D exact-posterior checks, the certified gap instance and the three-part $10\times10$/$13\times13$ relation (full-20D equivalence, upstream strict looseness, and a 6-point physical separation), and a real-data LINEMOD two-mode study with a pose-error/ADD-S proxy: at real near-ties the hard argmin is accurate (median $2.06^\circ$ rotation, $3.7$ mm ADD-S) while the Laplace posterior mean — the projected chordal mean of the per-mode rotations, not the unprojected 9-vector average, which is not a rotation — is biased toward the flip (median $44.2^\circ$, ADD-S $13.9$ mm, paired rotation bias $+36.9^\circ$), so the layer trades accuracy for smoothness there. Classical forward solvers (EPnP/SQPNP/IPPE) reach a $2.0$–$2.2^\circ$ median on the same correspondences, and a total-degree homotopy confirms the generic $40$ complex projective root count on $4/4$ instances. Detector-level ADD(-S) and ADD-S AUC on LINEMOD/LINEMOD-Occlusion, and UTIAS In-The-Dark, are not established.

---

## 1. Introduction

End-to-end learning for geometry requires differentiating through a pose estimator. The estimator is a nonconvex optimization problem with multiple local minima, and the derivative of its *global* solution is a discontinuous object whenever the identity of the global minimum changes. Two families of methods currently dominate.

**Deterministic global layers.** SDPRLayers [1] lifts a homogeneous quadratically constrained quadratic program (QCQP) to a matrix variable $X\succeq0$, solves the Shor relaxation, and, when the relaxation is tight, reads off a rank-one global solution together with a dual certificate. The backward pass differentiates the KKT system of the original QCQP, not the SDP, using an implicit-selection argument that handles redundant constraints. The approach is certifiable and has been demonstrated on matrix-weighted stereo localization. Its price is structural: the forward pass solves an SDP on the CPU, the problem dimension grows with the number of variables, and the guarantees are conditional on tightness and on constraint qualifications. A symmetric object or a near-planar configuration can produce a non-tight relaxation; the layer then has no pose to return.

**Probabilistic layers.** EPro-PnP [2] interprets the PnP residual as a Boltzmann distribution over $SE(3)$ and trains with a KL loss, using adaptive multiple importance sampling. This removes hard branch selection: ambiguous poses receive probability mass, and the gradient averages over hypotheses. The price is variance: Monte Carlo integration in six dimensions can miss distant modes, and the variance of the gradient must be controlled by sample count.

**The gap.** A layer that is (i) globally exhaustive without a convex lift, (ii) differentiable with a constant-size backward pass, and (iii) smooth at branch crossings without sampling, is missing. The QPEP line of work [3] provides exactly the algebraic ingredients: a quaternion parameterization with a single unit-norm constraint, exact elimination of translation, a Gröbner-basis solver that enumerates all KKT roots, and a closed-form observability/covariance analysis. What QPEP does not provide is a derivative with respect to the data and a probabilistic treatment of ambiguity.

This paper builds `Laplace-QPEPLayer` on those ingredients. The optimization variable is $q\in\mathbb{S}^3$; translation is eliminated in closed form; the forward pass enumerates all real KKT roots; the backward pass uses a $3\times3$ Riemannian implicit differentiation; and ambiguity is handled by a Laplace approximation of the Boltzmann posterior over the enumerated minima. The design is deliberately narrow: we do not handle inequality constraints (companion track `Constrained-QPEPLayer`) or robust outlier kernels (companion track `Robust-QPEPLayer`); those are orthogonal extensions.

**Contributions.**

1. **An exact algebraic bridge for the rotation subproblem.** We prove that the homogeneous rotation lift satisfies $x_R(q)=2Uv_2(q)$ with $U\in O(10)$ (Theorem 0.1), that the 21 standard $SO(3)$ constraints span exactly the degree-two vanishing ideal of $\nu_2(\mathbb{RP}^3)$ (Theorem 0.2), and that the associated $10\times10$ SDP is the Veronese (Shor) relaxation of the quartic on $\mathbb{S}^3$ (Theorem 0.3). The scope is explicit: this is the *rotation subproblem after exact translation elimination*. It covers the $13\times13$ SDP with the full 20-dimensional ideal, which is proved equivalent to the $10\times10$ one (Section 5.3.4), but not the released $24$-constraint implementation, whose constraint span is 17-dimensional and which is strictly looser in general.
2. **A constant-size Riemannian backward pass.** We derive $\lambda^\star=4J$ from Euler homogeneity, the Riemannian Hessian $H_{\mathbb{S}^3}=E^\top(\nabla^2J-4JI)E$, and the implicit Jacobians of $q^\star$ and $t^\star$ (Theorem 1.1), with exact antipodal invariance. The backward pass solves a $3\times3$ positive-definite system per active mode, instead of a KKT system with redundant constraints.
3. **A multi-modal, sampling-free Laplace layer.** Translation is integrated out exactly; rotation is handled by a Laplace approximation over all enumerated minima, giving analytic weights and gradients (Theorem 1.3). We give the correct measure/normalization constants, prove that the rotation error is $O(\tau)$ away from bifurcations, and identify where the approximation fails.
4. **A differentiability certificate and an honest failure analysis.** We give sufficient conditions for a $C^\infty$ global pose map (Theorem 1.2), show that a natural "exact iff" claim is false with counterexamples, and isolate the branch-merging regime where uniform asymptotics are required. A constructed Choi–Lam instance with a certified gap and a reproducible verification suite accompany the theory.

---

## 2. Related Work

**SDPRLayers [1].** Holmes, Dümbgen, and Barfoot lift a homogeneous QCQP to $X\succeq0$, solve the Shor relaxation, and differentiate the original KKT system with an implicit-selection theorem that tolerates redundant constraints. The released pose estimator uses a $13$-dimensional vector $x=[h;\mathrm{vec}(C);t]$ and 24 quadratic constraints (6 orthogonality, 9 handedness, 9 norm), with the last 18 marked redundant. Our work differs in four ways. (i) *Variable:* we optimize over $q\in\mathbb{S}^3$ (four variables, one constraint) and eliminate $t$ exactly; the dimension of the nonconvex problem is constant and independent of the number of correspondences. (ii) *Global mechanism:* QPEP enumerates all KKT roots and compares objectives; there is no convex relaxation and hence no tightness assumption. (iii) *Backward:* we differentiate the reduced stationarity system with a $3\times3$ Riemannian Hessian; SDPRLayers differentiates a KKT system with redundant rows after row reduction. (iv) *Ambiguity:* we return a smoothed posterior over modes; SDPRLayers returns a single certifiable solution and has no defined behavior when the relaxation is not tight. We do *not* claim that our $10\times10$ analysis applies to the released 24-constraint implementation, whose constraint span is 17-dimensional; the full-20D $13\times13$ relaxation, to which it does apply, is proved equivalent to the $10\times10$ one (Section 5.3.4).

**EPro-PnP [2].** Chen et al. define a Boltzmann distribution on $SE(3)$ and train with a KL loss, using adaptive multiple importance sampling. Our posterior is also Boltzmann, but the support is $\mathbb{S}^3/\{\pm1\}$ rather than $SE(3)$, translation is marginalized exactly, and the rotation integral is approximated analytically over an exhaustive list of modes rather than sampled. The trade is bias for variance: EPro-PnP is asymptotically exact but high-variance; we are deterministic but have an $O(\tau)$ Laplace bias away from bifurcations and a non-uniform error near them. A fair comparison must separate these two error sources (Section 5).

**QPEP [3].** Wu et al. unify a large class of pose problems as quadratic problems in a unit quaternion, eliminate translation, solve the resulting polynomial system with a Gröbner basis, and analyze solvability, observability, and covariance. Their solver returns at most 40 candidate complex roots in the released implementation (the generic 40-root count is confirmed independently by a total-degree homotopy in Section 5.3.5); the global solution is obtained by comparing objectives, not by a convex certificate. QPEP provides no derivative with respect to the data. Our contribution is precisely the missing differentiable layer: the closed-form Euler multiplier, the $3\times3$ implicit differentiation, the antipodal invariance, and the Laplace posterior over the enumerated roots. We rely on QPEP for the forward enumeration and treat completeness of the enumeration as an explicit assumption.

**BPnP [4].** Chen et al. backpropagate through a PnP solver by implicit differentiation of its optimality conditions. The solver is local and initialized from a network prediction; the gradient is the gradient of the local solution, not of the global one. At a branch crossing the local solution may be the wrong one, and the derivative of the global argmin does not exist as a classical function. Our layer differs by enumerating all roots and by smoothing the selection.

**Theseus [5].** Pineda et al. provide a differentiable nonlinear least-squares library with Lie-group support, sparse solvers, batching, and implicit/direct loss minimization. It is a general-purpose local optimizer; it does not enumerate global modes and does not provide a certificate. Our layer is narrower (quartic costs in a quaternion) but exhaustive and has a closed-form backward pass.

**Scope relative to companion tracks.** `Constrained-QPEPLayer` adds inequality constraints (handedness, non-penetration, frustum) via an augmented Lagrangian and Fischer–Burmeister reduction; `Robust-QPEPLayer` adds outlier-robust kernels and the curvature correction of the IRLS weights. The present layer assumes neither; its $M(\theta)$ may be any weighted Gram matrix, and its theory does not depend on unpublished results from the other tracks.

---

## 3. Problem Formulation and Notation

Let $q=[q_0,q_1,q_2,q_3]^\top\in\mathbb{R}^4$ be a unit quaternion and $R(q)\in SO(3)$ the associated rotation. We use the homogeneous lift
$$x_R(q)=\big[\|q\|^2,\ \mathrm{vec}(R(q))^\top\big]^\top\in\mathbb{R}^{10},$$
and the Veronese vector
$$v_2(q)=\big[q_0^2,q_1^2,q_2^2,q_3^2,\sqrt2q_0q_1,\sqrt2q_0q_2,\sqrt2q_0q_3,\sqrt2q_1q_2,\sqrt2q_1q_3,\sqrt2q_2q_3\big]^\top\in\mathbb{R}^{10}.$$
The QPEP family of costs is
$$J_\theta(q,t)=v_2(q)^\top M_0(\theta)v_2(q)+2t^\top D(\theta)v_2(q)+t^\top G(\theta)t,\qquad q^\top q=1,$$
with $G(\theta)\succ0$. Eliminating $t$ gives $t^\star(q)=-G^{-1}Dv_2(q)$ and
$$J_{\mathrm{red}}(q;\theta)=v_2(q)^\top M(\theta)v_2(q),\qquad M=M_0-D^\top G^{-1}D.$$
The map $\theta\mapsto(M_0,D,G)$ is assumed $C^2$ (Theorem 1.1) or $C^3$ (Theorem 1.3). The optimization variable is four-dimensional; the data enter only through the fixed-size $10\times10$ matrix $M$.

At a stationary point $q^\star$, $\nabla J=\lambda q$ and $\lambda=4J$ (Section 4.4). The tangent basis is
$$E(q)=\begin{bmatrix}-q_1&-q_2&-q_3\\ q_0&-q_3&q_2\\ q_3&q_0&-q_1\\ -q_2&q_1&q_0\end{bmatrix},\qquad q^\top E=0,\quad E^\top E=I_3.$$
The layer returns either the deterministic global pose, or the smoothed posterior over the enumerated minima. We assume throughout that the algebraic solver returns all real KKT roots (assumption A2); this is a property of the forward solver, not of the theory.

---

## 4. Method and Theory

### 4.1 Exact reduction and zero lift

The only nonlinear constraint is $q^\top q=1$. Translation is eliminated exactly (not relaxed), so the nonconvex problem is a quartic on $\mathbb{S}^3$. The "zero lift" refers to the optimization variable: no matrix variable is introduced at inference. The $10$-dimensional objects $v_2$ and $x_R$ are *analysis* devices; they are not optimization variables.

**Lemma 4.1 (exact translation elimination).** For $G\succ0$, $\min_t J_\theta(q,t)=J_{\mathrm{red}}(q;\theta)$, attained at $t^\star(q)=-G^{-1}Dv_2(q)$, and
$$\int_{\mathbb{R}^3}e^{-J_\theta(q,t)/\tau}dt=\frac{(\pi\tau)^{3/2}}{\sqrt{\det G}}e^{-J_{\mathrm{red}}(q;\theta)/\tau}.$$
*Proof.* Completing the square gives $J=(t-t^\star)^\top G(t-t^\star)+J_{\mathrm{red}}$; the Gaussian integral is standard. $\square$

### 4.2 The rotation lift is an orthogonal image of the Veronese map

**Theorem 4.2 (Theorem 0.1).** There exists $U\in O(10)$ such that for all $q\in\mathbb{R}^4$,
$$x_R(q)=2Uv_2(q),\qquad \|x_R(q)\|^2=4\|v_2(q)\|^2=4\|q\|^4.$$
*Proof sketch.* Every entry of $x_R$ is a quadratic form in $q$, and $v_2$ is a basis of $\mathrm{Sym}^2(\mathbb{R}^4)$; hence $x_R=Bv_2$ for a unique $B$. Since $R(q)$ is orthogonal, $\|x_R\|^2=\|q\|^4+3\|q\|^4=4\|q\|^4=4\|v_2\|^2$; as $\{v_2(q)\}$ spans $\mathbb{R}^{10}$, $B^\top B=4I$. Set $U=B/2$. The block structure is explicit: after permuting rows and columns, $U=\mathrm{diag}(\tfrac12H_4,\tfrac1{\sqrt2}H_2,\tfrac1{\sqrt2}H_2,\tfrac1{\sqrt2}H_2)$ with $H_4$ the Sylvester Hadamard matrix. $\square$

A subtlety that matters for reproducibility: $x_R$ must be the *homogeneous* lift with first entry $\|q\|^2$. With a literal constant $1$, the identity $x_R=2Uv_2$ holds only on $\mathbb{S}^3$; a literal constant $1$ would make the identity hold only on $\mathbb{S}^3$; the homogeneous lift is required.

**Theorem 4.3 (Theorem 0.2).** Let $A_i\in\mathbb{S}^{10}$ be the matrices of the 21 standard $SO(3)$ constraints: 3 column norms, 3 column inner products, 3 row norms, 3 row inner products, and 9 homogeneous cross-product equations $c_a\times c_b-x_0c_c=0$. Then
$$\dim\operatorname{span}\{A_i\}=20,$$
and this span is exactly the degree-two vanishing ideal $\mathcal{I}_2(\nu_2(\mathbb{RP}^3))$ after the change of coordinates $x=2Uv_2$.

*Proof sketch.* Each $A_i$ vanishes on $x_R(\mathbb{S}^3)$; the pullback is a homogeneous quartic vanishing on the sphere, hence identically zero, so $A_i$ lies in the kernel of $\phi:\mathbb{S}^{10}\to\mathbb{R}[q]_4$, $Q\mapsto Q(x_R(q))$. The map $\phi$ is surjective (the entries $1\cdot R_{ij}$ span all quadratics, and products of quadratics span all quartics), so $\dim\ker\phi=55-35=20$. An exact rational rank computation gives $\operatorname{rank}\{A_i\}=20$; the only linear dependency is $\operatorname{tr}(R^\top R)=\operatorname{tr}(RR^\top)$. A 20-dimensional subspace of a 20-dimensional kernel is the kernel. $\square$

A dimension count alone, $55-\binom74=20$, is necessary but not sufficient; the rank computation and the containment argument are both needed. The count is also sensitive to the constraint set: six off-diagonal inner products alone give rank 15, not 20.

### 4.3 The rotation SDP is a Veronese relaxation

**Theorem 4.4 (Theorem 0.3).** Let $\{B_k\}_{k=1}^{20}$ be a basis of $\mathcal{I}_2$, and let $M\in\mathbb{S}^{10}$. The SDP
$$\min_{Y\succeq0}\ \langle M,Y\rangle\quad\text{s.t.}\quad \langle B_k,Y\rangle=0\ (k=1,\dots,20),\quad \operatorname{tr}Y=1$$
is the Shor (Veronese) relaxation of $\min_{q\in\mathbb{S}^3}v_2(q)^\top Mv_2(q)$. In $x$-coordinates, with $Q=\tfrac14UMU^\top$, it is equivalent to
$$\min_{X\succeq0}\ \langle Q,X\rangle\quad\text{s.t.}\quad X_{00}=1,\quad \langle A_i,X\rangle=0,$$
under $X=4UYU^\top$.

*Proof sketch.* The two PSD cones are related by an orthogonal congruence. On $(\mathcal{I}_2)^\perp$, $\operatorname{tr}Y=\langle ee^\top,Y\rangle$ with $e=(1,1,1,1,0,\dots,0)^\top$, because both sides agree on the rank-one generators $v_2(q)v_2(q)^\top$; since $U^\top e_0=\tfrac12e$, $X_{00}=\langle ee^\top,Y\rangle$. The objectives agree by orthogonality. The dual is $\max\mu$ s.t. $M-\mu I-\sum_k\lambda_kB_k\succeq0$, an SOS-modulo-ideal certificate. $\square$

**Scope.** This is the rotation subproblem after exact translation elimination. The released SDPRLayers implementation solves a $13\times13$ SDP in $[h;\mathrm{vec}(C);t]$ with 24 constraints, whose span is only 17-dimensional; the two relaxations are therefore not the same, and the released one is strictly looser in general (Section 5.3.4). With the three missing row-orthogonality constraints restored, the $13\times13$ relaxation is *exactly* the $10\times10$ one by a rank-preserving block Schur bijection, so the $10\times10$ results of this section apply to the full-20D $13\times13$ relaxation, and the gap to the released implementation is entirely attributable to the three missing constraints. We also do not identify our SDP with the textbook second-order Lasserre moment relaxation with a sphere localizing matrix; we call it the degree-two Veronese/Shor relaxation.

**Example 4.5 (certified gap, constructed instance).** Let
$$C(q)=q_0^2q_1^2+q_1^2q_2^2+q_2^2q_0^2+q_3^4-4q_0q_1q_2q_3\ge0,$$
the Choi–Lam nonnegative non-SOS quartic, and let $M=M_{CL}+2I_{10}$, where $M_{CL}$ is the symmetric matrix of $C$ in the $v_2$ basis. The true minimum is $f^\star=2$ (since $C\ge0$ and $\|v_2\|^2=1$). Solving the $10\times10$ rotation SDP gives $f^\star_{\mathrm{SDP}}=1.9658119658$, with a dual certificate $Z=Q-\mu E_{00}-\sum_i\lambda_iA_i$ satisfying $\lambda_{\min}(Z)=-1.8\times10^{-11}$ and $\mu=1.9658119658$. The certified gap is therefore at least $0.034188$; the primal solution has $\operatorname{rank}(X^\star)=6$ and $\lambda_1/\lambda_2=1.451$. This is our own construction; a different perturbation can change the numerical gap and eigen-ratios, so we report only the certified instance above. Whether $M$ is realizable as the Gram matrix of an actual PnP or matrix-weighted correspondence set is open (Section 6, Q1).

### 4.4 The $3\times3$ Riemannian implicit differentiation

**Theorem 4.6 (Theorem 1.1).** Let $J(q)=v_2(q)^\top Mv_2(q)$ and let $q^\star$ be a stationary point on $\mathbb{S}^3$. Then:

1. $\nabla J(q^\star)=\lambda^\star q^\star$ with $\lambda^\star=q^{\star\top}\nabla J(q^\star)=4J(q^\star)$.
2. The Riemannian Hessian in the basis $E$ is
$$H_{\mathbb{S}^3}(q^\star)=E(q^\star)^\top\big(\nabla^2J(q^\star)-4J(q^\star)I_4\big)E(q^\star),$$
where $\nabla^2J=2J_v^\top MJ_v+2\sum_{a=1}^{10}(Mv_2)_aH_{v,a}$.
3. If $H_{\mathbb{S}^3}(q^\star)\succ0$, then the local minimizer is locally $C^\infty$ in $\theta$ and
$$\frac{\partial q^\star}{\partial\theta}=-E H_{\mathbb{S}^3}^{-1}E^\top\Big(2J_v^\top\frac{\partial M}{\partial\theta}v_2\Big),$$
$$\frac{\partial t^\star}{\partial\theta}=-G^{-1}\Big(\frac{\partial G}{\partial\theta}t^\star+\frac{\partial D}{\partial\theta}v_2+DJ_v\frac{\partial q^\star}{\partial\theta}\Big).$$
4. The map is antipodally invariant: $H_{\mathbb{S}^3}(-q^\star)=H_{\mathbb{S}^3}(q^\star)$, $\partial(-q^\star)/\partial\theta=-\partial q^\star/\partial\theta$, and $\partial\,\mathrm{vec}(R(q^\star))/\partial\theta$ is identical for $\pm q^\star$.

*Proof sketch.* Euler homogeneity gives the multiplier. Differentiating $F=E^\top\nabla J=0$ along $\delta q=E\delta\xi$ and using $(\delta E)^\top q=-E^\top\delta q$ gives the Hessian. The implicit Jacobian follows by differentiating $F(q^\star(\theta),\theta)=0$. Antipodal invariance follows from $v_2(-q)=v_2(q)$, $J_v(-q)=-J_v(q)$, and $E(-q)=-E(q)$. $\square$

The factor conventions are worth stating once: $\delta q=E\delta\xi$ corresponds to a rotation vector $2\delta\xi$, so the Hessian in rotation-vector coordinates is $H_\omega=\tfrac14H_{\mathbb{S}^3}$. QPEP's observability/covariance is defined in its own parameterization; comparisons should be made up to this constant.

### 4.5 A differentiability certificate, and what is not true

**Theorem 4.7 (Theorem 1.2, sufficient).** Suppose the global minimizer is unique up to $q\leftrightarrow-q$, $G\succ0$, $H_{\mathbb{S}^3}(q^{(1)})\succ0$, and $\Delta J=J(q^{(2)})-J(q^{(1)})>0$ for the next-best distinct $SO(3)$ class. Then there is a neighborhood of $\theta$ on which the global pose map is $C^\infty$ and its derivative is given by Theorem 4.6.

*Proof.* The implicit function theorem gives a smooth local minimizer; continuity of $\Delta J$ keeps it globally optimal; $G\succ0$ makes $t^\star$ unique. $\square$

A natural "exact iff" claim is false as written. (i) $\Delta J=0$ does not preclude a smooth global *selection*: with $J_1\equiv0$ and $J_2(\theta)=(\theta-\theta_0)^2$, branch 1 is a smooth global minimizer for all $\theta$, although the two branches tie at $\theta_0$. The correct statement concerns uniqueness of the global minimizer, not differentiability of every selection. (ii) $H_{\mathbb{S}^3}\succ0$ is not necessary: for $J(x;\theta)=x^4+\theta^2x^2$, the unique minimizer $x^\star(\theta)\equiv0$ is smooth at $\theta=0$ although the Hessian vanishes. The sharp necessary condition is that the reduced stationarity system admits a differentiable single-valued selection that is globally optimal at every point; $H\succ0$ is the standard sufficient condition.

A QPEP-specific counting issue: a naive count gives $K\le27$ real minima. The released LibQPEP solver returns up to 40 candidate complex roots; 27 has no cited source, and an odd count is incompatible with the free antipodal symmetry $(q,\lambda)\mapsto(-q,\lambda)$ for a finite root set. We therefore write "$K$ enumerated real minima, at most 40 candidates in the released solver", and we treat enumeration completeness as an assumption (A2), verified numerically by residuals rather than assumed. The generic complex count of 40 is confirmed independently on four instances by a total-degree homotopy (E18, Section 5.3.5): $40$ complex projective roots ($80$ affine roots with the $\pm$ double cover) on $4/4$ instances, with the real projective count varying by instance.

### 4.6 Multi-modal Laplace smoothing

**Theorem 4.8 (Theorem 1.3).** Let $q^{(k)}$, $k=1,\dots,K$, be the enumerated nondegenerate local minima of $J_{\mathrm{red}}$ on $\mathbb{S}^3/\{\pm1\}$, with $H_k=H_{\mathbb{S}^3}(q^{(k)})\succ0$. For the Boltzmann posterior $p_\theta(q,t)\propto e^{-J_\theta(q,t)/\tau}$:

1. **Translation is exact.** $\int_{\mathbb{R}^3}e^{-J_\theta(q,t)/\tau}dt=(\pi\tau)^{3/2}(\det G)^{-1/2}e^{-J_{\mathrm{red}}(q;\theta)/\tau}$.
2. **Rotation is Laplace-approximated.** With the Haar measure $\int_{SO(3)}dR=8\pi^2$,
$$\int_{SO(3)}e^{-J_{\mathrm{red}}/\tau}dR=8(2\pi\tau)^{3/2}\sum_{k=1}^K\frac{e^{-J_k/\tau}}{\sqrt{\det H_k}}\big(1+O(\tau)\big),$$
and the mode weights are
$$w_k=\operatorname{softmax}_k\Big(-\frac{J_k}{\tau}-\frac12\log\det H_k\Big).$$
3. **Gradients.** For a downstream loss $\ell$,
$$\nabla_\theta\bar\ell=\sum_kw_k\nabla_\theta\ell_k+\operatorname{Cov}_w\Big(\ell_k,\ -\frac1\tau\nabla_\theta J_k-\frac12\nabla_\theta\log\det H_k\Big),$$
with $\nabla_\theta J_k=v_k^\top(\partial_\theta M)v_k$ and
$$\nabla_\theta\log\det H_k=\operatorname{tr}\Big(H_k^{-1}\frac{dH_k}{d\theta}\Big),\qquad
\frac{dH_k}{d\theta}=\frac{\partial H}{\partial\theta}\Big|_{q^{(k)}}+\frac{\partial H}{\partial q}\Big|_{q^{(k)}}\frac{\partial q^{(k)}}{\partial\theta}.$$
The second term contains third $q$-derivatives of $J$.
4. **Temperature limit.** As $\tau\to0$ with $\Delta J>0$, the posterior mean and the soft argmin converge to the deterministic global minimizer and their gradients converge to Theorem 4.6. The KL loss itself, $\mathcal{L}_{\mathrm{KL}}=\tau^{-1}J_{\mathrm{gt}}+\log Z_{\mathrm{Laplace}}$, has a gradient that scales as $1/\tau$ and does not converge without a $\tau$-rescaling.

*Proof sketch.* The translation integral is Lemma 4.1. For rotation, expand $J$ to second order around each $q^{(k)}$ in the normal chart $q(u)=\mathrm{normalize}(q^{(k)}+E_ku)$; the Gaussian integral gives $(2\pi\tau)^{3/2}(\det H_k)^{-1/2}$. The local Jacobian is $dR=8dq$ per sheet; the two sheets of $\pm q$ and the double-cover factor $1/2$ combine to $8$ per $SO(3)$ mode. The standard Laplace expansion gives relative error $O(\tau)$. The gradient formula is the chain rule with the envelope theorem for $J_k$ and the softmax Jacobian for $w_k$. $\square$

**What "smooth at branch crossings" means.** The *exact* Boltzmann posterior is $C^\infty$ in $\theta$ for every $\tau>0$, because $\mathbb{S}^3$ (and $SO(3)$) is compact and $J_\theta$ is smooth; this follows from dominated convergence and does not require $\Delta J>0$. The *Laplace approximation* is not uniformly accurate at a branch crossing: as $\Delta J\to0$ and an eigenvalue of $H_k$ tends to zero, the two Gaussians overlap and the $1/\sqrt{\det H_k}$ weights diverge. A uniform (Airy) expansion or a one-dimensional numerical integral in the merging direction is required. We therefore claim smoothness for the exact smoothed layer, and $O(\tau)$ accuracy for the Laplace implementation away from bifurcations; we do not claim that the Laplace approximation is exact at the crossing.

**Normalization.** The constants matter for $Z$ but cancel in $w_k$. A commonly used normalization $Z_{\mathrm{Lap}}=(\pi\tau)^3(\det G)^{-1/2}\sum_k e^{-J_k/\tau}/\sqrt{\det H_k}$ is missing the $2^{3/2}$-type factor from the rotation integral; with fixed $\tau$ this is a $\theta$-independent constant and does not affect gradients, but it should be written correctly. If rotation-vector coordinates $\omega=2\delta\xi$ are used instead of $E$, then $\det H_\omega=\det H_{\mathbb{S}^3}/64$; the convention must be fixed once.

### 4.7 Algorithm and complexity

```
Forward (per batch element):
  1. Build M0, D, G from theta; M = M0 - D^T G^{-1} D.
  2. Enumerate all real KKT roots q^(k) of grad J = 4J q, q^T q = 1
     (Groebner/action matrix or multi-start Riemannian Newton + algebraic check).
  3. Keep local minima with H_k > 0; sort by J_k; compute w_k (softmax).
  4. Return (q^(1), t*(q^(1))) and/or the posterior {(q^(k), w_k)}.
Backward:
  - For each active mode: H_k, dH_k/dtheta, one 3x3 solve for dq^(k)/dtheta.
  - Assemble the deterministic Jacobian or the Laplace-smoothed gradient.
```

The nonconvex problem has 4 variables and 1 constraint. The dense linear algebra per mode is a $4\times10$ Jacobian, a $4\times4$ Euclidean Hessian, and a $3\times3$ solve. The data-dependent cost is building $M$ (a sum over correspondences) and enumerating roots; both are independent of the SDP dimension. In the deterministic regime, the backward pass is $O(1)$ per sample. In the smoothed regime, it is $O(K)$ with $K$ the number of active modes, plus the third-derivative contraction for $\nabla\log\det H_k$.

---

## 5. Experiments

> **Section status (2026-11-27, draft v4).** This section reports the synthetic 6D branch-crossing benchmark (with an FD-step study and a measured fold), the 1D exact-posterior checks, an SDP certificate and the three-part $10\times10$/$13\times13$ relation, and a real-data LINEMOD two-mode study with a pose-error/ADD-S proxy (hard argmin, projected Laplace mean, mode-level MC), classical forward-solver and Theseus comparisons on the same real correspondences, and a total-degree-homotopy root count. The LINEMOD study is a controlled ambiguity, correctness and bias/smoothness measurement, not a detector accuracy benchmark: there is no learned front-end, no detector-level ADD(-S)/ADD-S AUC number, and UTIAS In-The-Dark has not been run.

> **Evidence levels.** `paper claim` denotes a statement asserted in Sections 1–4 and not re-measured here. `independent reproduction (this repo)` denotes a number produced by a script in this repository, with its grid, seed, and tolerance recorded in the result file or the accompanying script; it means re-runnable in this repository, not a reproduction of an external paper's number. `unverified` denotes a design target or expectation with no measurement. Each table is labelled once; no `paper claim` is presented as an independent reproduction.

### 5.1 Tasks and datasets: what was run

Table 5.1 lists the experiment groups that have been executed and the ones that remain unrun. The synthetic 6D benchmark provides all local minima together with a branch-following finite-difference (FD) reference over a grid; the 1D normal forms are the only setting in which the posterior integral can be computed by dense quadrature.

**Table 5.1. Executed experiment groups and unrun items.**

| ID | Setting | Instance / data | Scale and settings | Status | Result file |
|-|-|-|-|-|-|
| E1 | 6D QPEP branch crossing (main) | synthetic $10\times10$ QPEP cost; $q_A=e_0$, $q_B=e_3$ (180° about $z$); branch crossing at $\theta=0$; 2 global and 2 local minima at $\theta=0$ | 81-point $\theta$ grid on $[-0.2,0.2]$; $\tau=0.01$; FD $h=10^{-4}$; mode-level MC $n=200$ with 20 seeds; 128 multi-starts | run | `results/main_table.json`, `results/raw_grid.json` |
| E2 | 1D exact posterior, unequal curvatures | $J=(x^2-1)^2(1+cx)+gx$, $c=0.25$; minima at $x=\pm1$ with curvatures 5 and 3 | 2 regimes ($g=0$ crossing, $g=0.2$ separated) $\times$ 6 values of $\tau\in[0.005,0.2]$ | run | `theory/results/uniform_unequal_table.md` |
| E3 | Fold merging and $\tau^{1/3}$ regularization | $J=x^4/4-ax^2/2$ | 3 values of $\tau$ $\times$ 6 values of $a$ | run | `theory/results/fold_table.md` |
| E4 | Closed-form $\partial_\theta\log\det H$ | random branch-following quadratic instances | 5 trials | run | `theory/results/q6_verify.md` |
| E5 | Rotation-SDP certificate | constructed Choi–Lam $+2I$ instance | $10\times10$ SDP solved with SCS | run | `verify/verify_theorem0_output.json` |
| E6 | Backward pass and Laplace constant | random branch-following quadratic instance | 4 values of $\tau$ | run | `verify/verify_theorem11_output.json` |
| E7 | LINEMOD symmetric objects, old construction | BOP `eggbox` (obj 10) and `glue` (obj 11) | 9 candidate frames, up to 2 images each, $N=30$ model points, $\tau=0.01$, FD $h=10^{-3}$ | attempted; negative, diagnosed (B1/B2) | `real/linemod/results/symmetry.json` |
| E8 | LINEMOD real two-mode cost (Part A) | BOP depth pixels + mesh vertices, dominant planar face | 15 frames (8 `eggbox`, 7 `glue`), $n_{\rm points}=900$, RANSAC band 2.5 mm, $n_{\rm starts}=384$, $\tau=0.01$ | run | `real/linemod/results/two_mode.md` (A) |
| E9 | LINEMOD controlled transverse crossing (Part B) | same real geometry, two labelled hypotheses with separate $t$ | 11 values of $\theta$ on $[0.35,0.65]$, FD $h=10^{-4}$, MC $n=400$ | run | `real/linemod/results/two_mode.md` (B) |
| E10 | FD-step study | synthetic 6D crossing | $h\in[10^{-6},10^{-3}]$ at 6 values of $\theta$ | run | `results/fd_step.md` |
| E11 | Derivative regression test | 6 random quadratic instances | `code/test_derivatives.py` | run | `code/test_derivatives.py` |
| E12 | $10\times10$ vs $13\times13$ SDP check | Choi–Lam, random and constructed instances, plus the 6-point physical 3D–3D instance ($13\times13$ constraints replicated from upstream code) | 10 instances + 8 perturbations + 1 physical instance | run | `sdp/results/sdp_check.md`, `sdp/results/six_point.md` |
| E13 | Robustness, fold and timing study | synthetic 6D crossing | `n_starts` sweep, 20 timing repetitions, batched forward | run | `results/robustness.md` |
| E14 | Scorer/Airy fold continuation (P1-Q5c) | 1D $A_2$ normal form + synthetic 6D `instance.py` fold | 1D crossing $\alpha=\pm10^{-5}$; 14 two-sided 6D points around $\theta_f=1.3185\times10^{-3}$; 21-point pose grid, step $2\times10^{-5}$ | run | `theory/results/scorer_fold.md` |
| E15 | Real pose error and ADD-S proxy | same 15 real `eggbox`/`glue` frames as E8 | $\tau=0.01$; MC $n=400$, 20 seeds; ADD-S: 1000 query vertices vs the full GT-mesh KD-tree $\times$ 3 subsamples; 1000-resample frame bootstrap; same `n_starts=384` | run | `real/linemod/results/pose_error.md` |
| E16 | Classical forward baselines | same 15 real correspondences, bitwise | cv2 5.0.0 EPnP/ITERATIVE/SQPNP/IPPE + local GN/LM (identity/GT/flip inits); 2000-resample frame bootstrap | run | `real/linemod/results/classical_baselines.md` |
| E17 | Theseus LM baseline | same 15 frames and bearing cost | Theseus 0.2.3 LM float64 CPU; GT, identity, 20 random inits per frame (300 trials); 8-init multistart on 5 frames (40 trials) | run | `real/linemod/results/theseus_baseline.md` |
| E18 | Fast root count (P1-Q3) | 3 random symmetric $M$ + the E1 instance $M(0)$ | total-degree homotopy on the projective stationarity system, all 4 affine charts; residual $\le7.1\times10^{-15}$; 5.3–8.0 s per instance | run | `roots/results/rootcount_fast.md` |
| N1 | LINEMOD / LINEMOD-Occlusion accuracy | full benchmark with SDPR-IS, SDPR-SDP, EPro-PnP, BPnP (Theseus only as the local reference of E17) | — | **not run** | TODO |
| N2 | UTIAS In-The-Dark | full SDPRLayers pipeline | — | **not run** | TODO |
| N3 | End-to-end training and GPU batching | — | — | **not run** | TODO |

*Evidence level: E1–E18 are `independent reproduction (this repo)`; N1–N3 are `unverified` (design only). E16–E17 are forward/local references on the same real correspondences, not differentiable-layer comparisons.*

The E1 instance is constructed, not fitted (`experiments/code/instance.py`): $M(\theta)=M_{\rm sym}+\theta M_{\rm break}$ with a $P$-invariant double well, $M_{\rm break}$ chosen so that $d(J_B-J_A)/d\theta|_{\theta=0}=2.0$, and a fixed translation block with $G\succ0$; the instance seed is 20261127. Mode enumeration is a batched damped Newton on $\mathbb{S}^3$ with 128 random starts, followed by an $SO(3)$-distance deduplication (threshold $10^{-5}$ rad) and a stationarity filter ($\|\nabla J\|\le10^{-8}$, $\lambda_{\min}(H_{\mathbb S^3})>10^{-8}$). It is not the released LibQPEP solver, so the completeness assumption (A2) is not independently certified in these runs. The generic $40$-root complex count for that algebraic system is now confirmed by an independent total-degree homotopy (E18, Section 5.3.5); that certifies the root count of the system, not the multi-start enumerator used here.

Not run, and not estimated: every **learned or differentiable-layer** baseline comparison (SDPR-IS, SDPR-SDP, EPro-PnP, BPnP), the LINEMOD/LINEMOD-Occlusion accuracy matrix, the UTIAS In-The-Dark pipeline, end-to-end training, and GPU batched root enumeration. Theseus (E17) and the cv2 classical solvers (E16) are run only as forward/local references on the controlled real cost; they are not differentiable-layer comparisons and no gradient claim is attached to them. No detector-level ADD(-S), ADD-S AUC, detector-based rotation error, translation error, success rate, training curve, end-to-end training throughput, or peak-memory number exists in the repository; these are marked TODO rather than filled with guesses. The LINEMOD entries E8–E9 and E15–E17 are controlled ambiguity, correctness, bias and forward-reference measurements on real geometry, not a detector accuracy benchmark: no learned front-end, and the ADD-S in E15 is a proxy on the constructed two-mode cost, not an AUC.

### 5.2 Metrics

The executed experiments use the following quantities; all are computed from the result files cited in Table 5.1.

- **Gradient relative error.** $\|\hat d\phi-d\phi_{\mathrm{FD}}\|_2/\|d\phi_{\mathrm{FD}}\|_2$ with $\phi=[\mathrm{vec}\,R(q);t^\star(q)]\in\mathbb{R}^{12}$. The reference is a branch-following central FD with $h=10^{-4}$ (E1, E8, E9) or $h=10^{-3}$ (old E7); E13 uses a Richardson-extrapolated reference built from $h=10^{-4}$ and $5\times10^{-5}$. For the Laplace layer the FD differentiates the same smoothed map; for the hard argmin at a global tie the reference branch switches inside $\pm h$, and the entry is reported as N/A rather than filled with a jump. Because the FD itself carries a truncation error, the $h$-scaling is measured separately in E10.
- **MC pose spread.** Euclidean norm of the standard deviation of the 12-vector $\phi$ over 20 independent mode-level MC means, each from $n=200$ mode samples (E1; synthetic units; the translation block is fixed at scale $D_{\text{scale}}=0.30$, so this is a relative spread, not metres). In E9 the MC baseline is a single mode-level estimate with $n=400$ samples per $\theta$; E13 uses $n=200$ with 3 seeds and also reports the 20-seed mean.
- **Real-data pose error (E8, E15).** Rotation error is the geodesic angle $\arccos((\operatorname{tr}(R_{\rm est}^\top R_{\rm gt})-1)/2)$ in degrees, computed on a proper rotation; translation error is $\|t_{\rm est}-t_{\rm gt}\|$ in metres (BOP `cam_t_m2c` is in millimetres). The Laplace estimate is the **projected chordal mean** $R_{\rm Lap}=\mathrm{Proj}_{SO(3)}(\sum_kw_kR_k)$ (SVD polar projection) with $t_{\rm Lap}=\sum_kw_kt_k$. The raw 9-vector average $\sum_kw_kR_k$ is not orthogonal, so applying the geodesic formula to it is not a rotation error: that unprojected quantity equals $\arccos(\sum_kw_k\cos e_k)$ and gives a median of $58.28^\circ$ on these 15 frames, and it is not reported as a pose error anywhere in this paper. On the projected mean the median is $44.24^\circ$. MC applies the same projection to $n=400$ modes sampled i.i.d. from $w$, for 20 seeds.
- **ADD-S and ADD (E15).** A fixed random subsample $V$ of 1000 mesh vertices is transformed by the estimated pose, and each point is matched to the nearest vertex of the GT-transformed **full** mesh $V_{\rm gt}$ (KD-tree over all 18,473 eggbox / 7,479 glue vertices; three independent query subsamples, their mean reported): $\mathrm{ADD\text{-}S}=\frac{1}{|V|}\sum_{x\in V}\min_{y\in V_{\rm gt}}\|R_{\rm est}x+t_{\rm est}-(R_{\rm gt}y+t_{\rm gt})\|$. Querying against the dense GT mesh matters: with a 1000-vertex GT set as well, the same-subsample variant carries a $+1.1$ mm discretisation floor on average (hard pose), while the 1000-query/dense-GT estimator agrees with full-mesh-vs-full-mesh ADD-S to $0.51$ mm. ADD is the same mean with the same-index distance $\|(R_{\rm est}-R_{\rm gt})x+(t_{\rm est}-t_{\rm gt})\|$ and is reported only as a reference. ADD-S is the appropriate object-space metric here because `eggbox` (obj 10) and `glue` (obj 11) each have a discrete $\sim180^\circ$ symmetry in `models_info.json`: a symmetry-equivalent pose is the same physical pose, and ADD would charge it an error of order the object diameter ($0.165$ m / $0.176$ m). ADD-S is invariant to that relabelling.
- **Bootstrap CI and paired bias (E15, E16).** Percentile bootstrap over the 15 frames (1000 resamples in E15, 2000 in E16), reported as mean/median with a 95% CI. The paired bias is the per-frame difference (Laplace or MC minus the hard argmin), summarised by its mean and a CI over frames. With 15 frames the CIs describe this fixed frame set, not a population and not a detector's error bars. For MC, a separate seed-level bootstrap over the 20 per-seed aggregates isolates seed noise (between-seed SD of the aggregate rotation error $1.011^\circ$).
- **Wall-clock.** CPU milliseconds per layer call. E13 reports 20-repetition medians and p95; the synthetic per-point medians of the main-table run are indicative only because they move with machine load. It is not a memory benchmark.
- **1D posterior error.** $|E_{\mathrm{Lap}}-E_{\mathrm{exact}}|$, its ratio to $\tau$, the log-log slope of $|E_{\mathrm{Lap}}-E_{\mathrm{exact}}|$ versus $\tau$, and the relative error of $\hat E=dE/dg$ against the exact quadrature value.
- **Partition-function error (fold).** $|Z_{\mathrm{Lap}}-Z_{\mathrm{exact}}|/Z_{\mathrm{exact}}$, with the same ratio for the $\tau^{1/3}$-regularized volume factor.
- **SDP certificate quantities.** Primal value, dual $\mu$, $\lambda_{\min}(Z)$ of the dual certificate, $\mathrm{rank}(X^\star)$ at tolerance $10^{-6}$, $\lambda_1/\lambda_2$, and the gap $f^\star-\mu$. For the $10\times10$/$13\times13$ comparison we also report the exact constraint-span ranks ($21$ constraints $\to$ 20; upstream 24 $\to$ 17; 24 + 3 row-orthogonality $\to$ 20) and the optimal values and ranks of the three relaxations on the same instance.
- **Derivative checks.** Relative error of the analytic Riemannian Hessian, implicit $q^\star$ and $t^\star$ Jacobians, and $\partial_\theta\log\det H$ against central FD.
- **Not measured.** Detector-level ADD(-S) and ADD-S AUC, rotation error (deg) on a detector, translation error (m) on a detector, success rate, NLL calibration, training curves, end-to-end training throughput, peak memory, and the missed-root rate of the enumerator against an independent certified solver. An ADD-S proxy on the controlled two-mode cost is measured in E15; it has no AUC and is not a detector benchmark.

*Evidence level: `independent reproduction (this repo)`.*

### 5.3 Results and ablations

#### 5.3.1 Synthetic 6D branch crossing (main result)

**Table 5.2. Hard argmin vs multi-modal Laplace vs mode-level MC on the 81-point crossing grid** ($\tau=0.01$, FD $h=10^{-4}$, mode-level MC $n=200$ with 20 seeds, CPU). Each regime entry is the median over its grid points; per-point numerical values are in `raw_grid.json`. The hard entry at $\theta=0$ is N/A because the global branch switches within $\pm h$. Timing is reported separately in Section 5.4.

| regime | n | Laplace grad rel | MC grad rel | Hard grad rel | MC pose std |
|-|-|-|-|-|-|
| far | 61 | 4.74e-07 | 4.74e-07 | 4.74e-07 | 2.76e-16 |
| near | 19 | 3.46e-05 | 9.40e-05 | 2.23e-05 | 2.89e-16 |
| at | 1 | 4.34e-04 | 3.26e-02 | N/A | 1.07e-01 |

*Evidence level: `independent reproduction (this repo)`; sources `results/main_table.md` and `results/raw_grid.json`. Regimes: far (61 points), near (19 points), at (the single point $\theta=0$), with nominal boundaries $|\theta|=0.05$ and $|\theta|=0.005$. The FD-limited reading of the `at` and `near` entries is in the text below.*

Three observations, stated without extrapolation beyond the measured grid.

1. **Away from the crossing the three methods coincide.** In the far regime all three median relative errors are $4.74\times10^{-7}$, and the median MC pose spread is $2.76\times10^{-16}$: the global mode carries essentially all of the Laplace mass and the sampler sees a single mode.
2. **The Laplace advantage is specific to the crossing.** In the near regime the hard argmin is still defined and has the smallest median error ($2.23\times10^{-5}$), ahead of Laplace ($3.46\times10^{-5}$) and MC ($9.40\times10^{-5}$); the median MC spread is $2.89\times10^{-16}$, although the per-point maximum is $3.41\times10^{-2}$ (`raw_grid.json`), so a minority of near points does show cross-mode sampling variance. At $\theta=0$ the hard entry is N/A, the Laplace-vs-FD entry is $4.34\times10^{-4}$ (FD-limited; see below), and the MC error is $3.26\times10^{-2}$ with a pose spread of $1.07\times10^{-1}$.
3. **The crossing is the only grid point with four enumerated minima.** This is also where the mode count is largest; timing is reported separately in Section 5.4.

**The $4.34\times10^{-4}$ entry is FD truncation, not method error.** Against a Richardson-extrapolated FD it is $1.8\times10^{-7}$ at $\theta=0$, while against the plain central FD at $h=10^{-4}$ it is $4.3\times10^{-4}$; the sequence $4.34\times10^{-8}$ ($h=10^{-6}$), $4.34\times10^{-6}$ ($h=10^{-5}$), $4.34\times10^{-4}$ ($h=10^{-4}$) makes the same point (`results/fd_step.md`, `results/robustness.md`). The near row is FD-limited for the same reason: at $\theta=0.01$ the study gives $3.46\times10^{-9}$ at $h=10^{-6}$ versus $3.46\times10^{-5}$ at $h=10^{-4}$. For $|\theta|\ge0.05$ the $h=10^{-5}$ error is at most $7.0\times10^{-8}$. The cleanest statement is therefore that the analytic gradient agrees with a Richardson FD to $1.8\times10^{-7}$ at the crossing, not that it has a $4\times10^{-4}$ error.

**Hard argmin is a step map; the Laplace map is not.** A direct diagnostic separates the two: the hard pose jump across $\theta=0$ stays $2.86$ as $\delta\to0$ ($\delta=10^{-3},10^{-4},10^{-5}$), while the Laplace jump decays linearly ($9.73\times10^{-1}$, $1.00\times10^{-1}$, $1.00\times10^{-2}$) with FD slope $\approx502$, i.e. the true derivative. The hard map is discontinuous at the crossing; the Laplace map is smooth there. This is the empirical content of the "smooth at the crossing" claim, and it is measured, not assumed. Where its central FD is defined, the hard map is as self-consistent as Laplace (median $1.2\times10^{-4}$ over the non-switching cases), but it switches branch inside $\pm h$ in half of the exact-crossing cases.

**The MC baseline is unbiased but noisy at a fixed budget.** On the same smoothed target, a fixed budget of $n=200\times3$ seeds gives a median per-seed gradient relative error of $0.05$ (20-seed mean $7.7\times10^{-3}$) and a median pose standard deviation of $0.09$ across the robustness conditions, versus $9.3\times10^{-8}$ for the analytic Laplace gradient against the Richardson reference. This is the measured variance cost of mode-level sampling; it is not an EPro-PnP comparison (Section 5.5).

**Mode count and fold.** The mode count over the grid is $K\in\{1,2,3,4\}$ ($K=1$ at 30 points, $K=2$ at 38, $K=3$ at 12, $K=4$ at the crossing point); these are the local minima returned by multi-start Newton with the filter of Section 5.1, not a certified root count. The smoothness statement is restricted to $\theta=0$ and $|\theta|\lesssim1.3\times10^{-3}$: a secondary minimum exists only for $|\theta|\lesssim1.3\times10^{-3}$ and disappears at the fold, where the Laplace map is discontinuous (Section 5.5).

#### 5.3.2 1D exact posterior: the Laplace error is $O(\tau)$

**Table 5.3. Exact posterior vs two-mode Laplace in the 1D normal form with unequal curvatures** ($c=0.25$; at $g=0$ both minima sit at $x=\pm1$ with curvatures 5 and 3). $E$ is the posterior mean of $x$; errors are against a $2{,}000{,}001$-point trapezoid on $[-3,3]$. Energies are dimensionless, and $\Delta J=6.16\times10^{-32}$ is numerical zero.

| regime | $\tau$ | $\Delta J$ | $E_{\mathrm{exact}}$ | $E_{\mathrm{Lap}}$ | $\vert{\rm err}\vert$ | $\vert{\rm err}\vert/\tau$ | rel grad err |
|-|-|-|-|-|-|-|-|
| crossing | 0.05 | 6.16e-32 | -0.125102 | -0.127017 | 1.92e-03 | 3.83e-02 | 1.77e-02 |
| crossing | 0.02 | 6.16e-32 | -0.126275 | -0.127017 | 7.42e-04 | 3.71e-02 | 6.93e-03 |
| crossing | 0.01 | 6.16e-32 | -0.126649 | -0.127017 | 3.67e-04 | 3.67e-02 | 3.44e-03 |
| crossing | 0.005 | 6.16e-32 | -0.126834 | -0.127017 | 1.83e-04 | 3.66e-02 | 1.71e-03 |
| separated | 0.05 | 4.01e-01 | -1.024203 | -1.031750 | 7.55e-03 | 1.51e-01 | 2.77e-02 |
| separated | 0.02 | 4.01e-01 | -1.029356 | -1.032296 | 2.94e-03 | 1.47e-01 | 1.27e-02 |
| separated | 0.01 | 4.01e-01 | -1.030838 | -1.032296 | 1.46e-03 | 1.46e-01 | 6.27e-03 |
| separated | 0.005 | 4.01e-01 | -1.031570 | -1.032296 | 7.26e-04 | 1.45e-01 | 3.12e-03 |

*Evidence level: `independent reproduction (this repo)`; source `theory/results/uniform_unequal_table.md`. The full table also contains $\tau=0.2$ and $\tau=0.1$.*

The expected-value error scales as $O(\tau)$: the log-log slope is 1.066 at the crossing and 1.036 in the separated basin, and $|{\rm err}|/\tau$ stays in $3.66$–$3.83\times10^{-2}$ at the crossing and $1.45$–$1.51\times10^{-1}$ when separated. At $\tau=0.01$ the relative gradient error is $3.44\times10^{-3}$ at the crossing and $6.27\times10^{-3}$ when separated. Over the printed rows the relative gradient error falls monotonically with $\tau$ (crossing: $1.77\times10^{-2}$ at $\tau=0.05$ to $1.71\times10^{-3}$ at $\tau=0.005$), but no slope is fitted for it and the gradient remainder is still a proof gap, so this is a numerical check rather than a proof of the gradient claim.

#### 5.3.3 Fold merging and the $\tau^{1/3}$ regularizer

**Table 5.4. Standard Laplace vs $\tau^{1/3}$-regularized volume factor at $\tau=0.01$**, for $J=x^4/4-ax^2/2$; exact partition by a $2{,}000{,}001$-point trapezoid on $[-3.5,3.5]$. $H=2a$ is the curvature at the minimum and the barrier is $a^2/4$. The full file also contains $\tau=0.02$ and $\tau=0.005$.

| $a$ | $H$ | barrier | $Z_{\mathrm{exact}}$ | $Z_{\mathrm{Lap}}$ | rel err Lap | $Z_{\mathrm{reg}}$ | rel err reg |
|-|-|-|-|-|-|-|-|
| 0.50 | 1.00e+00 | 6.25e-02 | 2.6976e+02 | 2.5969e+02 | 3.73e-02 | 2.3556e+02 | 1.27e-01 |
| 0.20 | 4.00e-01 | 1.00e-02 | 2.3994e+00 | 2.1547e+00 | 1.02e-01 | 1.7371e+00 | 2.76e-01 |
| 0.10 | 2.00e-01 | 2.50e-03 | 1.2349e+00 | 1.4394e+00 | 1.66e-01 | 9.9871e-01 | 1.91e-01 |
| 0.05 | 1.00e-01 | 6.25e-04 | 9.7810e-01 | 1.6876e+00 | 7.25e-01 | 9.5017e-01 | 2.86e-02 |
| 0.02 | 4.00e-02 | 1.00e-04 | 8.6986e-01 | 2.5318e+00 | 1.91e+00 | 1.0019e+00 | 1.52e-01 |
| 0.01 | 2.00e-02 | 2.50e-05 | 8.3916e-01 | 3.5538e+00 | 3.24e+00 | 1.0358e+00 | 2.34e-01 |

*Evidence level: `independent reproduction (this repo)`; source `theory/results/fold_table.md`. Relative errors are dimensionless ratios, not percentages.*

As $a\to0$ the two minima and the saddle merge, $H\to0$, and the standard Laplace volume factor $1/\sqrt{H}$ diverges: the relative error grows from $3.73\times10^{-2}$ at $a=0.5$ to $1.91$ at $a=0.02$ and $3.24$ at $a=0.01$. Replacing $H$ by $H+\tau^{1/3}$ reduces the error to $1.52\times10^{-1}$ and $2.34\times10^{-1}$ at the two smallest $a$, but it is not uniformly better: at $a=0.5$ it is worse ($1.27\times10^{-1}$ versus $3.73\times10^{-2}$). The regularized form is therefore evidence for the $\tau^{1/3}$ scaling in the merging regime, not a drop-in replacement everywhere.

#### 5.3.4 Rotation-SDP certificate and the three-part $10\times10$/$13\times13$ relation

**Table 5.5. Numerical certificate for the constructed $10\times10$ rotation SDP** (Choi–Lam quartic plus $2I$; SCS). The gap is $f^\star-\mu$, where $f^\star=2$ is the analytic minimum of the quartic on $\mathbb{S}^3$ and $\mu$ is the certified dual value.

| quantity | value | evidence |
|-|-|-|
| $f^\star$ (analytic, $C\ge0$, $C(e_0)=0$) | 2.0 | `paper claim` (Section 4.3) |
| SDP primal value | 1.9658119658 | `independent reproduction (this repo)` |
| dual $\mu$ | 1.9658119658 | `independent reproduction (this repo)` |
| $\lambda_{\min}(Z)$ of the dual certificate | $-1.79\times10^{-11}$ | `independent reproduction (this repo)` |
| certified gap $f^\star-\mu$ | 0.034188 | `independent reproduction (this repo)` |
| $\mathrm{rank}(X^\star)$ at tolerance $10^{-6}$ | 6 | `independent reproduction (this repo)` |
| $\lambda_1/\lambda_2$ | 1.451 | `independent reproduction (this repo)` |
| multi-start QPEP minimum (independent check) | 2.00186 | `independent reproduction (this repo)` |
| random perturbations with positive gap | 8 of 8 (gaps 0.0453–0.0782) | `independent reproduction (this repo)` |
| rank of the 21 $SO(3)$ constraints | 20 | `independent reproduction (this repo)` |
| dimension of the degree-two vanishing ideal | 20 | `independent reproduction (this repo)` |

*Evidence level: `independent reproduction (this repo)`; source `verify/verify_theorem0_output.json`. "Certified" here means the dual matrix is PSD at SCS tolerance, not a symbolic proof.*

The primal value, dual value, and gap of the $10\times10$ instance are mutually consistent to the printed digits; the gap is $0.034188034\ldots$ in the JSON. This is our own construction (Section 4.3), not the source document's instance.

**Theorem P1-2 (revised).** The relation between the $10\times10$ rotation SDP and the released $13\times13$ SDP is not one equality or one inequality: it has three layers — an exact equivalence, an inequality hierarchy, and a strict physical separation. Let $x_{13}=[x_R;t]$ with $t\in\mathbb{R}^3$ unconstrained, $Q_{13}=\begin{bmatrix}Q_{RR}&Q_{Rt}\\Q_{Rt}^\top&G\end{bmatrix}$ with $G\succ0$, and $Q_{10}=Q_{RR}-Q_{Rt}G^{-1}Q_{Rt}^\top$. Write $f^\star_{10,\mathrm{full}}$, $f^\star_{13,\mathrm{full}}$ and $f^\star_{13,\mathrm{up}}$ for (a) the $10\times10$ value under the full 20-dimensional ideal, (b) the $13\times13$ value under the same full ideal, i.e. the upstream 24 constraints plus the three row-orthogonality equations (27 constraints, 20 dimensions), and (c) the $13\times13$ value under the released 24 constraints, whose span is 17-dimensional. Then:

1. **Full 20D equivalence (proved).** The degree-two vanishing ideal of $SE(3)$ in $\mathbb{S}^{13}$ is $\mathcal{I}_2(SE(3))=\mathcal{I}_2(SO(3))\oplus\mathbf{0}_{3\times3}$: dimension 20, with no coupling to $t$. Hence the block Schur-complement bijection $X^\star_{13,\mathrm{full}}=LX^\star_{10,\mathrm{full}}L^\top$, with $L=[I_{10};\,-G^{-1}Q_{Rt}^\top]$ of full column rank, gives
$$f^\star_{13,\mathrm{full}}\equiv f^\star_{10,\mathrm{full}},\qquad\operatorname{rank}\big(X^\star_{13,\mathrm{full}}\big)\equiv\operatorname{rank}\big(X^\star_{10,\mathrm{full}}\big).$$
This is an algebraic identity, not a numerical coincidence, and it is the correct content of the old "same value, same rank" claim.
2. **The upstream 17D implementation is looser, strictly so in general (proved).** The released 24 constraints span only 17 of the 20 dimensions, so $\mathcal{V}_{17}\subsetneq\mathcal{I}_2(SO(3))$ and the upstream feasible set strictly contains the full-20D one. Therefore, for **every** instance,
$$f^\star_{13,\mathrm{up}}\;\le\;f^\star_{13,\mathrm{full}}=f^\star_{10,\mathrm{full}}\;\le\;f^\star_{\mathrm{QPEP}},$$
the last inequality being the relaxation bound of Theorem 4.4. Because the argument is instance-independent, every full-20D instance that is not tight is automatically a non-tight instance of the released implementation; this is the transfer the revised proof uses, and we do not re-measure that transfer here.
3. **Physical 6-point separation (proved and independently reproduced).** Layer (2) is strict on a genuine matrix-weighted 3D–3D cost with positive weights and a unique ground-truth rotation, even where the full-20D SDP is exactly tight at rank one (Table 5.7). By continuity the strict inequality persists on an open neighbourhood of that instance, so the separation is not a measure-zero pathology.

**Exact constraint comparison.** The 21 $SO(3)$ constraints of the $10\times10$ SDP span 20 dimensions (Section 4.3). The 24 upstream constraints of the $13\times13$ SDP span only 17, and projecting the 21 constraints one at a time onto that 17-dimensional span leaves exactly three nonzero residuals — the row orthogonality equations $r_1\cdot r_2=r_1\cdot r_3=r_2\cdot r_3=0$ (residuals 1.63–1.73; the other 18 are $\le6.2\times10^{-16}$). Row norms $\|r_i\|^2=h^2$ are already implied by the nine norm constraints together with column orthogonality, so the row dot products are the whole deficit (`sdp/results/sdp_check.md` §2). The $13\times13$ constraint generators were replicated row-by-row from the released SDPRLayers `pose_est.py` (commit `62d8ac6`, lines 341–417) and compared element-wise against the AST-extracted upstream generator bodies: all 24 constraints match (24/24) with max $|\Delta|=0$, and $\max_i|x^\top A_i x|=1.4\times10^{-16}$ on random rotations (`sdp_check.json`, `upstream.ran = true`). The $10\times10$ side reuses `verify/verify_theorem0.py` and reproduces the regression value above.

**Table 5.6. The upstream 24-constraint $13\times13$ relaxation is looser than the full-20D $10\times10$ relaxation, and strictly so in general.** $f^\star_{10}$ is the $10\times10$ value and $f^\star_{13,\mathrm{up}}$ the value of the released 24-constraint implementation (17-dimensional span); `CL_exact` is the constructed Choi–Lam instance and `rowdot` is the constructed counterexample. The difference column is computed from the printed $f$-values; for `CL_exact` and `p6_data` the true differences are $3.7\times10^{-11}$ and $-2.9\times10^{-10}$, below the six-digit columns. Representative rows; the full 10-instance and 8-perturbation tables are in `sdp/results/sdp_check.md`.

| instance | $f^\star_{10}$ | $f^\star_{13,\mathrm{up}}$ | $f^\star_{10}-f^\star_{13,\mathrm{up}}$ | rank $X_{10}^\star$ | rank $X_{13,\mathrm{up}}^\star$ |
|-|-|-|-|-|-|
| `CL_exact` | 1.965812 | 1.965812 | 3.7e-11 | 6 | 6 |
| `CL_pert0` | 1.962860 | 1.960078 | 2.8e-03 | 6 | 5 |
| `CL_pert5` | 1.964571 | 1.950275 | 1.4e-02 | 6 | 4 |
| `rowdot` | 1.000000 | -1.000000 | 2.00 | 10 | 7 |
| `rowdot_gen` | -0.060192 | -2.029230 | 1.97 | 1 | 4 |
| `p6_data` | 0.252326 | 0.252326 | 0.0 | 1 | 1 |

*Evidence level: `independent reproduction (this repo)`; source `sdp/results/sdp_check.md` and `sdp/run_sdp_check.py`.*

**Interpretation of Table 5.6.** The Choi–Lam equality is a coincidence of symmetry: its upstream optimum happens to lie on the row-orthogonality face, so $f^\star_{13,\mathrm{up}}=f^\star_{10}=1.9658119658$ with rank 6 on both sides, and the row-dot residual at the upstream optimum is at most $8.8\times10^{-15}$. After eight symmetry-breaking perturbations of order $10^{-2}$, $8/8$ instances have $f^\star_{13,\mathrm{up}}<f^\star_{10}$ by $2.5\times10^{-3}$–$1.4\times10^{-2}$, with $\mathrm{rank}(X^\star_{13,\mathrm{up}})\in\{4,5\}$ against $\mathrm{rank}(X^\star_{10})=6$, and $\max|r_a\cdot r_b|$ at the upstream optimum is $0.59$–$0.88$: the upstream optimum is far outside the row-orthogonality face. The constructed objective $Q=-r_1\cdot r_2+\tfrac14 I$ isolates the mechanism in objective form — it is the constant 1 on the full-20D feasible set ($f^\star_{10}=1.000000$), while upstream it reaches $-1.000000$ (gap $2.00$), and still $1.97$ after a $10^{-2}$ perturbation. These measurements are layer (2) of Theorem P1-2; they are not evidence against layer (1), because the full-20D relaxation is not what they measure.

**Table 5.7. 6-point physical 3D–3D counterexample separating the upstream $13\times13$ relaxation from the full-20D one.** Construction: $p_i=\pm e_1,\pm e_2,\pm e_3$ and $y_i=\varepsilon p_i$ with $\varepsilon=0.05$, shared matrix weights $W=\begin{bmatrix}1&-0.6&0\\-0.6&1&0\\0&0&1\end{bmatrix}$ (eigenvalues $0.4,1.0,1.6>0$), $N=6$, so $D=0$ and $G=6W\succ0$; the ground-truth rotation $R^\star=I_3$ is unique. "Measured here" is `sdp/results/six_point.md`; "revised proof" is the quoted claim. The middle rank follows from layer (1); the proof states ranks 1 and 4 explicitly.

| quantity | measured here | revised proof |
|-|-|-|
| $f^\star_{10}$ (21 constraints, exact rank 20) | 5.41500000 | 5.41500000 |
| $f^\star_{13,\mathrm{full}}$ (24 + 3 row-orthogonality = 27 constraints, full 20D) | 5.41500000 | 5.41500000 |
| $f^\star_{13,\mathrm{up}}$ (24 constraints, 17D span) | 3.59431373 | 3.59430886 |
| $f^\star_{10}-f^\star_{13,\mathrm{up}}$ | 1.82068627 | 1.82069114 |
| $\operatorname{rank}(X^\star_{10})/\operatorname{rank}(X^\star_{13,\mathrm{full}})/\operatorname{rank}(X^\star_{13,\mathrm{up}})$ | 1 / 1 / 4 | 1 / 1 / 4 |
| $f_{\rm nonconv}-f^\star_{10}$ | $-8.38\times10^{-13}$ | 0 |
| $f_{\rm nonconv}-f^\star_{13,\mathrm{full}}$ | $-3.38\times10^{-13}$ | 0 |
| $f_{\rm nonconv}-f^\star_{13,\mathrm{up}}$ | 1.820686 | 1.820691 |

*Evidence level: the "measured here" column is `independent reproduction (this repo)` (source `sdp/results/six_point.md`, script `sdp/run_6point_check.py`); the "revised proof" column is the analytic claim of Theorem P1-2. $f_{\rm nonconv}=6(1-\varepsilon)^2=5.415$ is exact, and the two negative entries are SCS feasibility error at the zero gap. The proof's $f^\star_{13,\mathrm{up}}$ entry is shown as quoted, but it lies $4.87\times10^{-6}$ below the exact-rational dual lower bound for the same SDP and is therefore not a feasible optimum (Section 5.3.4, `sdp/results/f13up_forensics.md`).*

**Why this instance separates the two.** With $\pm p_i$ cancelling pairwise, $D=\mathbf{0}$, and the cost is
$$J(R,t)=2\,\mathrm{tr}(WRR^\top)-4\varepsilon\,\mathrm{tr}(WR)+2\varepsilon^2\mathrm{tr}(W)+6t^\top Wt.$$
The only term that sees row orthogonality is $2\,\mathrm{tr}(WRR^\top)=2(\|r_1\|^2+\|r_2\|^2+\|r_3\|^2-1.2\,r_1^\top r_2)$. Under the full-20D constraints $r_1^\top r_2=0$ makes it the constant $6$, and the remaining term $-4\varepsilon\,\mathrm{tr}(WR)$ is minimized at $R=I$ with value $6(1-\varepsilon)^2=5.415$, so $f^\star_{10}=f^\star_{13,\mathrm{full}}=f^\star_{\mathrm{QPEP}}=5.41500000$ with rank 1 and gap zero at SCS level. Upstream, the missing $r_1^\top r_2=0$ lets the relaxation take $\langle r_1^\top r_2\rangle>0$ and drop to $f^\star_{13,\mathrm{up}}=3.59431373$ with rank 4. The measured values agree with the proof's $3.59430886$ and $1.82069114$ to $4.9\times10^{-6}$. That residual is **not** solver tolerance and not a property of our construction: SCS ($\varepsilon=10^{-11}$) gives $3.59431373$ and CLARABEL gives $3.59431371$ (agreement $2\times10^{-8}$), and an exact-rational dual certificate (explicit dual solution, exact rational LDU with $\eta=10^{-11}$; minimum pivot $1.0\times10^{-11}$, mpmath 60-digit $\lambda_{\min}=+8.0\times10^{-12}$) yields the rigorous lower bound $f^\star_{13,\mathrm{up}}\ge3.594313732834$. The proof's quoted $3.59430886$ lies $4.87\times10^{-6}$ **below this bound**, so it cannot be the optimum of the stated 24-constraint SDP; 36 natural variants (row-orthogonality scaling, subsets, layout/transpose, $h$ normalization, $t$ block, constraint ablation, solver tolerances) do not reproduce it within $10^{-8}$, the closest being $4.1\times10^{-6}$ away (`sdp/results/f13up_forensics.md`). The only same-SDP mechanism is an under-converged dual value (a valid but non-optimal lower bound); we recommend quoting $f^\star_{13,\mathrm{up}}=3.5943137\ldots$ and gap $1.8206863\ldots$. The qualitative separation (gap $1.82$, rank $1$ versus $>1$) is unaffected; $\operatorname{rank}(X^\star_{13,\mathrm{up}})$ is $4$ at a $10^{-5}$ threshold and $6$ at $10^{-6}$ (four $O(1)$ eigenvalues followed by $\sim10^{-5}$ ones). Because this is a genuine matrix-weighted 3D–3D cost with positive weights and a unique ground-truth rotation, the Choi–Lam realizability question (Q1) does not apply to it: the gap is a property of the released constraint set, not of a non-physical objective.

**Status of Q2/Q13.** This item is no longer open, and it should not be described as "reopened". The correct status is: **full-20D equivalence proven; the upstream implementation strictly looser in general and separated by a physical counterexample.** The earlier numerical reading — $f^\star_{13}<f^\star_{10}$ in general — is correct for the released 24-constraint implementation, and the three missing row-orthogonality constraints are the entire cause; it is not a statement about the $13\times13$ relaxation as such. The supplement's older equality claim is right for the full-20D pair and wrong for the released implementation; that mismatch, not the $10\times10$ theory, is what the revision corrects.

#### 5.3.5 Derivative formulas and the Laplace constant

**Table 5.8. Numerical checks of the backward pass (E6) and the closed-form $\partial_\theta\log\det H$ (E4).** All entries are relative errors unless marked absolute.

| check | value |
|-|-|
| $E(q)$ orthonormal error | 1.14e-16 |
| $E(q)^\top q$ error | 4.41e-18 |
| Euler identity absolute error | 5.33e-15 |
| Riemannian Hessian, analytic vs FD | 9.63e-08 |
| implicit $\partial q^\star/\partial\theta$ vs FD | 1.51e-07 |
| implicit $\partial t^\star/\partial\theta$ vs FD | 4.58e-08 |
| antipodal errors ($E$, $H$, $\partial q^\star$, $\partial R$) | 0 (exact) |
| Laplace integral relative error (chart on $S^3$) at $\tau=0.2/0.1/0.05/0.025$ | 8.26e-02 / 3.81e-02 / 1.86e-02 / 9.25e-03 |
| log-log slope of Laplace error vs $\tau$ | 1.05 |
| closed-form $\partial_\theta\log\det H$ vs FD, 5 trials | 1.65e-09 – 8.84e-09 (max abs err 1.59e-09) |
| closed-form `dlogdetH_dtheta` vs `dlogdet_exact`, after fix (absolute) | 8.9e-16 |
| closed-form `dlogdetH_dtheta` vs branch FD, after fix (absolute) | 1.8e-9 |
| regression `test_derivatives.py`: `dE` / `dH` / `dq` (absolute) | 2.9e-11 / 1.6e-8 / 2.5e-10 |

*Evidence level: `independent reproduction (this repo)`; sources `verify/verify_theorem11_output.json`, `theory/results/q6_verify.md`, `code/test_derivatives.py`, and `experiments/rounds_2026-11-27.md`.*

The backward-pass checks confirm the formulas of Theorem 4.6 to FD accuracy; they do not certify the root set. The Laplace-constant check gives a slope of 1.05, consistent with the $O(\tau)$ claim of Theorem 4.8 and inconsistent with an exact rotation integral. The closed-form $\log\det$ derivative (Q6) matches FD to $9\times10^{-9}$ or better on all five trials. A real bug in the closed-form path was found on 2026-11-27: `qpep_core.dE` had wrong entries in rows 1–2 of the $d_1/d_2/d_3$ partials, so `dlogdetH_dtheta` disagreed with the independent `dlogdet_exact` and branch FD by $O(1)$ (the round log records $8.4\times10^{-2}$–$2.8\times10^{-1}$). The main table and the Q6 check were unaffected because they used the numeric fallback and `dlogdet_exact` (which uses the directional $E(dq)$). After the fix the closed form matches `dlogdet_exact` to $8.9\times10^{-16}$ and branch FD to $1.8\times10^{-9}$; `code/test_derivatives.py` pins `dE`, `dH` and `dq` and passes. The closed form is now the path used for the regenerated main table (`results/main_table.*`), whose numerical columns are unchanged.

**Root count (P1-Q3, E18).** The generic algebraic count is now measured rather than assumed. A total-degree homotopy on the projective stationarity system, run in all four affine charts with a batched complex predictor–corrector and residual-verified union, finds **40 complex projective roots on 4/4 instances** — three random symmetric $M$ and the 6D instance $M(0)$ — with maximum chart residual $7.1\times10^{-15}$ and 5.3–8.0 s per instance. Each projective root corresponds to 80 affine roots with the $\pm$ double cover. The real picture varies by instance: real projective roots $26/12/12/20$, real local minima $5/2/2/4$. The complex count is generic (the QPEP action-matrix bound of 40); the real count is not, so a real-root count is an instance property and must not be quoted as a constant. This confirms the "at most 40 candidate complex roots" statement of Section 4.5 and closes the earlier blocked Q3 local count, but it certifies the algebraic system, not the multi-start Newton enumerator used in E1/E8; the missed-root rate of that enumerator against the homotopy is still unmeasured. Evidence: `independent reproduction (this repo)`; source `roots/results/rootcount_fast.md` (raw data in `results/rootcount_fast.json`; script `roots/run_rootcount_fast.py`, ~30 s for the four instances).

#### 5.3.6 Ablations

**A. Temperature.** Table 5.9 reports the gradient relative error of the exact-logdet Laplace layer for $\tau\in\{0.05,0.02,0.01,0.005\}$ at the crossing, near the crossing, and far from it.

**Table 5.9. $\tau$ sweep (gradient relative error).**

| $\theta$ | $\tau=0.05$ | 0.02 | 0.01 | 0.005 |
|-|-|-|-|-|
| +0.000 | 1.20e-03 | 5.84e-04 | 4.34e-04 | 1.51e-03 |
| +0.010 | 1.29e-05 | 1.07e-04 | 3.46e-05 | 1.64e-05 |
| -0.010 | 1.29e-05 | 1.06e-04 | 3.46e-05 | 1.63e-05 |
| +0.100 | 8.47e-07 | 8.46e-07 | 8.46e-07 | 8.46e-07 |
| -0.100 | 8.43e-07 | 8.42e-07 | 8.42e-07 | 8.42e-07 |

*Evidence level: `independent reproduction (this repo)`; source `results/ablation.md`, section A.*

At the crossing the error is smallest at $\tau=0.01$ ($4.34\times10^{-4}$) and rises again at $\tau=0.005$ ($1.51\times10^{-3}$), which is consistent with the FD reference becoming unreliable when the two modes are nearly degenerate and $\tau$ is small; it is not evidence for an optimal $\tau$. Far from the crossing the error is $\approx8.5\times10^{-7}$ and independent of $\tau$ at the printed precision. No learned-$\tau$ experiment was run.

**B. Volume factor.** Table 5.10 compares the closed-form $\log\det$ derivative, a tangent-space FD, and dropping the volume factor from the weights.

**Table 5.10. Volume-factor variants.** `exact` is the closed form of Section 4.6, `numeric` is a tangent-space FD, `none` drops $\tfrac12\log\det H_k$ from the weights; `pose diff` and `grad diff` are Euclidean norms of the differences in the 12-vector against `exact`.

| $\theta$ | mode | grad rel | pose diff | grad diff |
|-|-|-|-|-|
| +0.000 | exact | 4.34e-04 | 0.00e+00 | 0.00e+00 |
| +0.000 | numeric | 4.34e-04 | 0.00e+00 | 2.22e-08 |
| +0.000 | none | 3.75e-04 | 6.47e-02 | 5.34e+01 |
| +0.010 | exact | 3.46e-05 | 0.00e+00 | 0.00e+00 |
| +0.010 | numeric | 3.46e-05 | 0.00e+00 | 1.22e-11 |
| +0.010 | none | 3.86e-05 | 7.77e-04 | 4.38e-01 |
| -0.010 | exact | 3.46e-05 | 0.00e+00 | 0.00e+00 |
| -0.010 | numeric | 3.46e-05 | 0.00e+00 | 1.89e-11 |
| -0.010 | none | 3.86e-05 | 7.79e-04 | 4.39e-01 |

*Evidence level: `independent reproduction (this repo)`; source `results/ablation.md`, section B. At $|\theta|=0.1$ all three variants coincide (pose and gradient differences $0.00\times10^{0}$).*

Dropping the volume factor leaves the gradient relative error almost unchanged at the crossing ($3.75\times10^{-4}$ versus $4.34\times10^{-4}$) but changes the smoothed pose by $6.47\times10^{-2}$ and the gradient by $5.34\times10^{1}$ in absolute norm; near the crossing the gradient changes by $0.44$. The volume factor is therefore not cosmetic for the smoothed map even when the FD-relative error is similar. The closed form and the tangent-space FD agree to $2.22\times10^{-8}$ or better.

**C. Top-$K$ mode truncation.** Table 5.11 keeps only the $K$ highest-weight modes and compares the result with the sum over all modes.

**Table 5.11. Top-$K$ mode truncation (reference: all modes).**

| $\theta$ | $K$ | pose diff | grad diff |
|-|-|-|-|
| +0.000 | 1 | 1.44e+00 | 4.99e+02 |
| +0.000 | 2 | 1.05e-01 | 6.83e+01 |
| +0.000 | 3 | 7.95e-02 | 4.93e+01 |
| +0.000 | 4 | 0.00e+00 | 0.00e+00 |
| +0.010 | 1 | 2.72e-03 | 1.98e+00 |
| +0.010 | 2 | 3.02e-05 | 2.65e-02 |
| +0.010 | 3 | 0.00e+00 | 0.00e+00 |
| +0.010 | 4 | 0.00e+00 | 0.00e+00 |

*Evidence level: `independent reproduction (this repo)`; source `results/ablation.md`, section C. At $\theta=0.1$ all $K$ give zero difference.*

At $\theta=0$, keeping only the top mode changes the pose by $1.44$ and the gradient by $499$ relative to the all-mode sum; $K=2$ cuts these to $0.105$ and $68.3$, and $K=4$ (all modes) is exact by construction. At $\theta=0.01$, $K=3$ is already exact at the printed precision. This is the measured justification for summing over all enumerated minima rather than top-1; it does not establish how many modes a real cost would have.

The following protocol ablations were **not run**: analytic $3\times3$ vs autograd through the root solver; $H_{\mathbb S^3}$ vs Euclidean $\nabla^2J$; $E$-basis vs rotation-vector scaling; LibQPEP vs multi-start Newton; dropping modes with $w_k<10^{-3}$; SDPR with and without redundant constraints; learned $\tau$; group-orbit averaging. TODO.

### 5.4 Compute actually used

- **CPU only.** All reported runs are CPU-only single-process jobs: the 6D main run is a Slurm `cpu` job, and the ablation, LINEMOD, 1D, `verify/`, SDP-check (including the 6-point instance), robustness, FD-step and regression scripts are short CPU runs (the LINEMOD timings are login-node CPU). No GPU, no Mosek, and no multi-node run was used. The SDP certificate uses cvxpy 1.9.3 with SCS ($\varepsilon=10^{-9}$, `max_iters=200000`); the SDP comparison uses SCS with the same tolerances under cvxpy 1.7.5, and the 6-point check uses SCS with $\varepsilon=10^{-11}$ and `max_iters=10^6`. The phase-4 real-data and root-count runs are short login-node CPU jobs: pose error/ADD-S $\approx10$ s (E15), classical baselines $3.0$ s (E16), Theseus $9.5$ s (E17), root count $\approx30$ s for four instances (E18).
- **Timing (E13).** From 20 repetitions, a full synthetic layer call at $\theta=0$ takes 61.9 ms median (p95 62.1): mode enumeration is 88%, the $K$ implicit $3\times3$ solves 1.6%, and the analytic backward pass (weight gradient and mixing) 0.1%. At $\theta=-0.1$ the whole call is 17.7 ms. The per-point medians in an earlier main-table run were measured under machine load and are not used as the timing claim; they are indicative only. The LINEMOD runs report 24.2–265.9 ms per frame forward (E8) and 33.1–42.6 ms per $\theta$ including FD (E9). Total wall-clock, end-to-end training throughput, and peak memory were not recorded, so we do not quote them.
- **Batching (E13).** A true batched multi-start Newton is only 1.30× faster than the loop at batch size 64 (59.9 versus 46.2 samples/s), 0.78× at 8, and 0.46× at 1: convergence heterogeneity and the per-instance classify loop dominate, so a GPU or custom kernel is needed for a real throughput win. Mode enumeration is the bottleneck, not the $3\times3$ solves.
- **Not run.** GPU batched root enumeration, end-to-end training, and the full LINEMOD/In-The-Dark accuracy matrix. The earlier planning assumption of 1–2 GPUs over 4–6 weeks is a plan, not a measurement (`unverified`).

### 5.5 Reproduction risks and negative results

- **Root enumeration is not certified for the reported runs.** The 6D runs use the batched multi-start Newton of `instance.py`, not LibQPEP; completeness (A2) is assumed for E1/E8 and the missed-root rate of that enumerator against an independent algebraic solver is not measured. Against a 1024-start reference, $n_{\rm starts}\ge64$ reproduces the mode set on the sweep and $32$ matches the median $K$, while $n_{\rm starts}=16$ misses the fragile mode in 10% of independent draws (below); the mode counts in Section 5.3 should therefore be read as lower bounds on the true count, not certified counts. The counting question is now separately measured: a total-degree homotopy finds 40 complex projective roots on $4/4$ instances (E18, Section 5.3.5), with real counts $26/12/12/20$ and $5/2/2/4$ local minima. That certifies the generic complex count of the algebraic system, not the enumerator used in the experiments.
- **"Certified" gap is numerical.** The SDP certificate is PSD only within SCS tolerance ($\lambda_{\min}(Z)=-1.79\times10^{-11}$); it is not a symbolic certificate. The $13\times13$ comparison of Section 5.3.4 is mixed: the full-20D equivalence is an algebraic identity, the upstream 17-dimensional inequality is proved, and measured strict on the perturbed Choi–Lam set and on the 6-point instance (Table 5.7), whose values are numerical.
- **No learned or differentiable-layer baseline has been run.** SDPR-IS, SDPR-SDP, EPro-PnP, and BPnP are absent from every table; no accuracy, variance, or timing comparison against them exists. Theseus (E17) and the cv2 classical solvers (E16) are present only as forward/local references on the same controlled real correspondences — Theseus is a local optimiser with no posterior and no enumeration, and EPnP/SQPNP/IPPE are classical forward solvers — so they are not differentiable-layer comparisons and no gradient claim is attached to them. The missing learned-layer comparison remains the largest gap in the experimental section.
- **The MC baseline is mode-level.** It samples the discrete mode weights ($n=200$, 20 seeds) and does not implement the $SE(3)$ AMIS sampler of EPro-PnP; it measures discrete-mode sampling variance only, and its numbers must not be read as an EPro-PnP comparison.
- **Laplace bias near bifurcations.** The 1D slope ($1.07$), the 1D fold blow-up (relative error $3.24$ at $a=0.01$), the genuine 6D fold at $|\theta|\approx1.3\times10^{-3}$ (Section 5.5), and the real-data near-tie bias (Laplace median $44.24^\circ$ versus hard $2.06^\circ$, Section 5.5; the earlier $58.28^\circ$ was the unprojected 9-vector mean and is not a rotation error) bound the regime where the Laplace sum is reliable. The Airy/Scorer continuation is derived (Section 6, item 4) and has since been implemented at the level of the measured normal-form invariants on this instance (`theory/results/scorer_fold.md`): the 1D three-way comparison is reproduced (standard $0.8670\to0$, $\tau^{1/3}$ $0.4959\to0$, Scorer $0.33838\to0.33834$), the 6D complex root pair is tracked through the fold at $\theta_f=1.3185\times10^{-3}$, and the resulting pose jump is $1.21\times10^{-2}$ versus $3.04\times10^{-1}$ (standard) and $1.07\times10^{-1}$ (regularized), at smooth-map derivative scale. The main table still reports the real-root-only layer, and the implementation is asymptotic in the normal form (fourth-order term $\approx8.3$), so we claim continuity of the *measured* Scorer weight, not an exact 6D zero-jump proof.
- **In-The-Dark not attempted.** Its download, run split, and the `pose_sdp_block.py`-at-HEAD issue are untested; no claim is made about the pipeline.

**Fold at $|\theta|\approx1.3\times10^{-3}$.** The 6D synthetic map has a genuine fold just off the crossing. A secondary minimum exists only for $|\theta|\lesssim1.3\times10^{-3}$ (the fine fold scan puts the mode death at $\theta_f=1.31\times10^{-3}$; the coarser robustness scan brackets it at $\approx1.4\times10^{-3}$); as it approaches the fold its $3\times3$ Hessian determinant falls through $3.35\times10^{-1}$ while its Laplace weight rises from $0.09$ to $0.187$ ($0.122$ at that determinant), and then the mode disappears ($K$ falls from 4 to 3). Across the fold the pose map steps by $0.31$ over one $2\times10^{-5}$ step (constant under grid refinement, i.e. a genuine discontinuity), and $\Vert d\phi\Vert$ peaks at $3529$ versus a median of $379$ away from the fold; the effective FD slope reaches $1.5\times10^4$. Neither a smaller $\tau$ nor more starts removes it: $n_{\rm starts}\ge32$ matches the 1024-start median $K$ ($64$ starts reproduce the full mode set on the sweep), while $n_{\rm starts}=16$ misses the fragile mode in 10% of independent draws and a miss moves the pose by up to 28% and the gradient by up to 97%. The smoothness and $O(\tau)$ accuracy statements are therefore restricted to $\theta=0$ and $|\theta|\lesssim1.3\times10^{-3}$; the multi-modal Laplace map is discontinuous where a minimum is born or dies. Evidence: `independent reproduction (this repo)`; sources `results/robustness.md` (section a2), `results/fold_6d.md`, and the robustness round logged in `experiments/rounds_2026-11-27.md`.

**A cheap partial fix for the fold.** Replacing $\det H$ by $\det(H+\epsilon I)$ with $\epsilon=\tau^{1/3}$ (the 1D rule of Section 5.3.3, whose derivative is still closed form, $\mathrm{tr}((H+\epsilon I)^{-1}\dot H)$) largely removes the *derivative* singularity: at $\theta=1.30\times10^{-3}$ the standard $\Vert d\phi\Vert$ is $3529$ while the regularized one is $443$, and the near-fold gradient error falls from $2.90\times10^{-1}$ to $1.19\times10^{-2}$ (a factor 24); away from the fold ($\theta=0.1,0.02$) the two variants agree to the printed precision. It does **not** remove the *value* discontinuity caused by a minimum dying: the pose step across the fold only falls from $0.309$ to $0.107$ and $K$ still changes from 4 to 3. Hessian regularization is therefore a defensible cheap mitigation for the fold neighbourhood, but a continuity claim across the fold still needs the Scorer continuation (Section 6, item 4) or a local numerical integral over the merging pair. Evidence: `independent reproduction (this repo)`; source `results/fold_6d.md` and `code/run_fold_6d.py`.

**The value step across the fold is structural.** The measured $0.309\to0.107$ is what the fold theory predicts. By P1-Q5b (Section 6, item 4), any proxy that sums only over real local minima and gives the merging mode a positive regularized floor weight must jump at a standard codimension-1 fold by an amount proportional to that weight — $\Theta(\tau^{1/3}e^{-(J_0-J_{\rm glob})/\tau})$ for the $\eta=c_0\tau^{1/3}$ floor — so the $\tau^{1/3}$ regularizer can cure the derivative and cannot cure the value. P1-Q5c removes the jump by tracking the merged pair as a complex-conjugate pair and replacing the two real-root weights by a semi-line Airy/Scorer weight, which is $C^\infty$ across the fold and recovers the 3D Laplace weight for $\zeta\gg1$. That continuation is a theoretical result; its 6D implementation is a separate round in progress, so the measured numbers in this section are the real-root-only ones.

**Real-data two-mode result (LINEMOD).** A construction that does produce a genuine two-mode cost on real data is implemented in `real/linemod/run_two_mode.py`. Part A uses real depth pixels inside the BOP `mask_visib`, back-projects them with the real intrinsics to bearings, assigns each a nearest real mesh vertex (KD-tree) as the 3-D model point, and uses deterministic RANSAC (2.5 mm inlier band) to select the dominant planar face. A planar target has the classical two-fold pose ambiguity, so the bearing cost has two genuine minima; modes whose transformed points fall behind the camera ($z\le0$) are rejected. The settings are 15 frames (8 `eggbox`, 7 `glue`), $n_{\rm points}=900$ per observation, $n_{\rm starts}=384$, $\tau=0.01$, FD $h=10^{-4}$, on login-node CPU.

**Table 5.12. Real two-mode bearing cost on the dominant planar face (Part A).** All 15 frames have $K_{\rm phys}=2$ physical minima ($K_{\rm all}=4$ before the $z>0$ filter). `w(other)` is the Laplace weight of the physical mode that is not the hard argmin (the flip in 14/15 frames; the GT-like mode in `000010/72`). Rotation errors are against BOP ground truth in degrees; the Laplace entry is the **projected chordal mean** $\mathrm{Proj}_{SO(3)}(\sum_kw_kR_k)$. The earlier unprojected 9-vector values (median $58.28^\circ$) are not rotation errors and are not used anywhere in this paper.

| frame/img | obj | $n_{\rm plane}$ | $\Delta J$ | other-mode (deg) | $w({\rm other})$ | hard rot err (deg) | Laplace rot err (deg, projected) |
|-|-|-|-|-|-|-|-|
| 000010/15 | 10 | 558 | 9.58e-04 | 69.3 | 0.483 | 1.47 | 33.90 |
| 000010/18 | 10 | 474 | 1.25e-03 | 91.9 | 0.472 | 0.51 | 42.85 |
| 000010/40 | 10 | 482 | 9.71e-04 | 69.1 | 0.482 | 2.20 | 33.07 |
| 000010/60 | 10 | 452 | 1.69e-03 | 104.0 | 0.459 | 0.96 | 46.45 |
| 000010/66 | 10 | 460 | 7.00e-04 | 88.4 | 0.488 | 2.29 | 43.67 |
| 000010/72 | 10 | 501 | 1.25e-04 | 3.8 | 0.504 | 44.46 | 23.02 |
| 000010/80 | 10 | 508 | 5.11e-04 | 54.7 | 0.501 | 1.08 | 27.71 |
| 000010/90 | 10 | 474 | 4.13e-04 | 54.8 | 0.497 | 3.49 | 27.91 |
| 000011/20 | 11 | 413 | 2.07e-03 | 69.1 | 0.608 | 8.14 | 44.24 |
| 000011/24 | 11 | 423 | 1.65e-03 | 131.1 | 0.457 | 1.37 | 54.99 |
| 000011/26 | 11 | 439 | 1.90e-03 | 126.9 | 0.450 | 2.06 | 51.65 |
| 000011/32 | 11 | 474 | 2.10e-03 | 123.9 | 0.451 | 1.78 | 51.60 |
| 000011/33 | 11 | 365 | 1.49e-03 | 149.6 | 0.460 | 1.62 | 59.09 |
| 000011/40 | 11 | 428 | 2.56e-03 | 106.0 | 0.457 | 2.77 | 46.19 |
| 000011/49 | 11 | 347 | 2.10e-03 | 83.7 | 0.528 | 4.95 | 46.75 |

*Evidence level: `independent reproduction (this repo)`; hard and weight columns from `real/linemod/results/two_mode.md` (Part A), Laplace column from `real/linemod/results/pose_error.md` (per-frame table). `pose_error` reproduces the same per-frame $\Delta J$, weights and hard rotation errors exactly (max abs diff $0$; the only $2.1\times10^{-14}$ difference is in the old unprojected value).*

Three points. First, the two-mode structure is real and reproducible: $K_{\rm phys}=2$ in 15/15 frames, with $\Delta J\in[1.25\times10^{-4},2.56\times10^{-3}]$, other-mode angles $3.8^\circ$–$149.6^\circ$, and $w({\rm other})=0.450$–$0.608$. Second, the GT-like mode is the lowest-$J$ physical mode in 14/15 frames, so the hard argmin is accurate there: median rotation error $2.06^\circ$ [$1.47,3.49$] and median ADD-S $3.7$ mm [$3.0,4.3$], with one failure above $10^\circ$ (`000010/72`, $44.5^\circ$). Third — the honest reading — the Laplace posterior mean is **not** an accuracy win on these near-ties: its median projected rotation error is $44.24^\circ$ [$33.90,46.87$], a paired bias of $+36.9^\circ$ [$+27.2,+44.9$] against the hard argmin, and its median ADD-S is $13.9$ mm [$10.3,20.2$] against $3.7$ mm. It puts roughly half the mass on the flip mode. The smallest Hessian determinant in the set is $6.09\times10^{-3}$ (`000011/20`), i.e. the flip mode is itself close to a fold.

**The $58.3^\circ$ correction.** Earlier drafts (and `two_mode.json`/`two_mode.md`) applied the geodesic formula to the *unprojected* 9-vector mean $\sum_kw_kR_k$, which is not a rotation. That quantity is exactly $\arccos(\sum_kw_k\cos e_k)$ (verified to $10^{-14}$) and gives a median of $58.28^\circ$; it is not a geodesic distance and it overstates the error. On the projected chordal mean $\mathrm{Proj}_{SO(3)}(\sum_kw_kR_k)$ the median is $44.24^\circ$, and ADD-S is defined only for the projected pose. The direction of the bias claim is unchanged — the Laplace mean is pulled toward the flip — but its magnitude was overstated in the earlier draft. Some downstream result tables (`theseus_baseline.{md,json}`'s comparison block and one cited-value sentence in `classical_baselines.md`) still carry the old $58.28^\circ$; the paper uses $44.24^\circ$, the valid projected value.

**Aggregate pose error and ADD-S (E15).** Table 5.13 gives the 15-frame aggregates with bootstrap CIs; the per-frame rotation and translation errors and all CIs are in `pose_error.md`/`pose_error.json`. Means are larger than medians for the hard argmin ($5.28^\circ$, $6.0$ mm) because of the single failure frame, while the Laplace means are $42.21^\circ$ and $15.5$ mm; translation medians are $6.2$ mm (hard) and $20.2$ mm (Laplace). MC estimates the same smooth mean as Laplace (mean over 20 seed means versus the Laplace aggregate: $0.23^\circ$, within its seed-level CI) and therefore inherits the bias; it only adds seed noise (between-seed SD $1.011^\circ$), so it is not a fix.

**Table 5.13. Aggregate pose error and ADD-S on the 15 real frames (E15).** Medians with 95% percentile-bootstrap CIs over frames (1000 resamples). The paired bias is the mean per-frame difference against the hard argmin (95% CI over frames); the median paired rotation bias is $+41.8^\circ$. "Within $10^\circ$" counts frames whose estimate is within $10^\circ$ of GT; for the posterior mean this is not a "true mode", since the mean need not be a mode.

| method | rot median [CI] (deg) | ADD-S median [CI] (mm) | ADD median [CI] (mm, ref) | paired rot bias vs hard (deg) | paired ADD-S bias (mm) | within $10^\circ$ / true mode |
|-|-|-|-|-|-|-|
| hard argmin | 2.06 [1.47, 3.49] | 3.7 [3.0, 4.3] | 5.9 [5.2, 11.1] | — | — | 14/15 true mode |
| Laplace (projected) | 44.24 [33.90, 46.87] | 13.9 [10.3, 20.2] | 38.1 [31.0, 41.5] | +36.9 [+27.2, +44.9] | +9.4 [+6.4, +12.6] | 0/15 |
| MC (20 seeds) | 43.80 [33.90, 47.43] | 14.1 [10.3, 20.0] | 38.0 [31.0, 41.5] | +37.2 [+26.7, +45.0] | +9.4 [+6.4, +12.5] | 0/15 |

*Evidence level: `independent reproduction (this repo)`; source `real/linemod/results/pose_error.md` §2. ADD-S matches 1000 uniform-vertex queries (three independent subsamples; mean reported) against the full GT mesh; the three query subsamples differ by at most $1.87$ mm in any frame, and the 1000-query/dense-GT estimator agrees with full-mesh-vs-full-mesh ADD-S to $0.51$ mm. The CIs describe these 15 frames, not a population.*

**Per-frame ADD-S.** Table 5.14 lists the object-space error for all 15 frames. ADD is shown only for the Laplace pose, to make the size of the wrong-metric effect visible.

**Table 5.14. Per-frame ADD-S (mm) and ADD reference (mm) on the 15 real frames.** ADD-S is the primary object-space metric; ADD is same-index distance and is not invariant to the objects' $\sim180^\circ$ symmetry.

| frame/img | obj | ADD-S hard | ADD-S Laplace | ADD-S MC | ADD Laplace (ref) |
|-|-|-|-|-|-|
| 000010/15 | 10 | 3.0 | 10.3 | 10.3 | 31.0 |
| 000010/18 | 10 | 4.3 | 12.2 | 12.3 | 37.6 |
| 000010/40 | 10 | 6.2 | 9.7 | 9.7 | 31.1 |
| 000010/60 | 10 | 3.0 | 13.9 | 14.1 | 40.7 |
| 000010/66 | 10 | 2.7 | 12.1 | 12.0 | 38.0 |
| 000010/72 | 10 | 12.5 | 8.4 | 8.4 | 22.3 |
| 000010/80 | 10 | 3.8 | 8.8 | 8.9 | 24.1 |
| 000010/90 | 10 | 3.1 | 8.5 | 8.4 | 23.3 |
| 000011/20 | 11 | 13.8 | 22.9 | 22.8 | 72.7 |
| 000011/24 | 11 | 2.8 | 20.2 | 20.0 | 40.5 |
| 000011/26 | 11 | 3.7 | 17.6 | 17.5 | 38.1 |
| 000011/32 | 11 | 3.1 | 19.0 | 19.0 | 41.5 |
| 000011/33 | 11 | 3.8 | 23.4 | 23.7 | 41.5 |
| 000011/40 | 11 | 2.2 | 16.8 | 16.7 | 44.7 |
| 000011/49 | 11 | 22.4 | 28.1 | 28.2 | 75.9 |

*Evidence level: `independent reproduction (this repo)`; source `real/linemod/results/pose_error.md` §1.*

Two per-frame readings matter. (i) On `000010/72` the hard argmin is the mode $44.5^\circ$ from GT while the GT-like mode is $3.8^\circ$ away but higher in $J$ by $1.25\times10^{-4}$; Laplace/MC land between the modes ($23.0^\circ$, $8.4$ mm) and are closer than the hard failure. That is a single-frame accident of the loss ranking, not evidence that the posterior mean is more accurate in general. (ii) ADD is $2$–$3\times$ ADD-S for the Laplace pose (median $38.1$ vs $13.9$ mm) because the glue object is long and thin and the $\sim180^\circ$ symmetry relabels it; this is exactly why ADD-S, not ADD, is the reported metric.

**Classical forward solvers on the same correspondences (E16).** Table 5.15 runs cv2's EPnP, ITERATIVE, SQPNP and IPPE, plus a plain local GN/LM on the same bearing cost, on the bitwise-identical 15-frame correspondences (asserted equal to `two_mode.json`; the GT-pose re-projection residual is $1.47$ px median, i.e. the $\sim1$ px nearest-vertex quantisation). These are **forward solvers, not differentiable layers**: the LM is a plain forward optimiser on the same cost, not BPnP, and nothing here differentiates through PnP or supports a gradient claim. IPPE is run on the RANSAC inliers projected onto the fitted plane (it requires exact coplanarity, planarity RMS $0.83$–$1.47$ mm); the other solvers use the raw inliers.

**Table 5.15. Classical forward PnP baselines on the 15 real frames (E16).** Median with 95% bootstrap CI over frames (2000 resamples). "True mode" counts solutions within $10^\circ$ of GT. SQPNP returns one solution in cv2 5.0.0 (documented for non-P3P/IPPE flags and observed on all 15 frames); IPPE returns two planar solutions, and the first (cheapest-reprojection) one is the GT-closer one on all 15 frames, so the oracle row would be identical to "IPPE first".

| method | rot median [CI] (deg) | ADD-S median [CI] (mm) | true mode |
|-|-|-|-|
| EPnP | 2.05 [1.41, 3.71] | 6.5 [5.0, 13.5] | 12/15 |
| ITERATIVE (homography init + LM) | 2.07 [1.60, 3.54] | 6.5 [5.7, 8.7] | 11/15 |
| SQPNP (1 solution) | 2.04 [1.47, 2.77] | 4.6 [4.0, 5.8] | 15/15 |
| IPPE (first, plane-projected) | 2.21 [1.68, 2.76] | 5.5 [4.4, 7.4] | 15/15 |
| local LM, identity init | 106.01 [69.14, 179.82] | 40.7 [28.0, 1708.7] | 3/15 |
| local LM, GT init | 2.06 [1.47, 2.77] | 4.6 [4.0, 5.8] | 15/15 |
| local LM, flip init | 88.39 [69.08, 106.01] | 23.5 [16.9, 28.0] | 0/15 |

*Evidence level: `independent reproduction (this repo)`; source `real/linemod/results/classical_baselines.md`. The LM rows differ only in initialisation; the GT-init row is not a deployable baseline. The ADD-S column here uses that script's 1000-query/1000-GT-vertex estimator, which carries a $+1.1$ mm discretisation floor relative to the dense-GT-mesh ADD-S of E15 (Table 5.13); the two ADD-S columns are therefore not directly comparable, and rotation error is the comparable accuracy column across Tables 5.13 and 5.15.*

Three readings. (i) Forward point accuracy is at the classical level: SQPNP/EPnP/ITERATIVE medians are $2.04$–$2.07^\circ$ and IPPE is $2.21^\circ$, i.e. equal to the hard argmin's $2.06^\circ$ within frame noise; no classical solver comes near the Laplace mean's $44.2^\circ$. (ii) Single-solution solvers are confidently wrong on some frames: EPnP misses GT by more than $10^\circ$ on 3/15 and ITERATIVE on 4/15 (the fourth, `000011/20`, is $10.0^\circ$, at the threshold); on `000010/72` EPnP and ITERATIVE both return the planar flip at $44.3^\circ$/$44.0^\circ$ with a re-projection residual of $0.78$–$0.79$ px, and the flip even has lower bearing cost than the true-mode planar solution on the raw inliers. (iii) Without the $z>0$ physicality filter and multi-start, the local LM baseline fails from identity (3/15 true, 7/15 behind the camera; median $106^\circ$) and stays out of the true mode from the flip init (0/15); only the GT init recovers it. P1's explicit physicality filter and enumeration are doing real work on this cost.

**Theseus LM on the same cost (E17).** Theseus 0.2.3 (`LevenbergMarquardt`, float64 CPU) minimises the same bearing cost; its objective agrees with a numpy reference to $1.1\times10^{-13}$ and its analytic Jacobian matches central differences to $1.3\times10^{-10}$. It is a **local optimiser, not a differentiable-layer comparison**: each call returns one pose, it has no notion of the enumerated mode set, it returns no posterior, and it applies no $z>0$ physicality filter. Table 5.16 gives the initialisation study.

**Table 5.16. Theseus LM on the P1 real bearing cost (E17).** "True mode" is within $10^\circ$ of GT. The random-init row pools 20 inits on each of the 15 frames (300 solves); the multistart row pools 8 inits on each of 5 representative frames (40 solves) and its median is over those 40 trial errors, not a best-of-8 aggregate.

| protocol | rot median (deg) | true mode | outcome breakdown |
|-|-|-|-|
| GT init (15 frames) | 2.06 | 15/15 | recovers the nearest P1 mode (on `000010/72` that is the $3.8^\circ$ mode, not P1's global argmin) |
| identity init (15 frames) | 178.82 | 1/15 | 6 flip, 8 other; 8 behind the camera |
| 20 random inits/frame (300 solves) | 123.91 | 100/300 (33.3%) | 57 flip, 143 other, 134 non-physical |
| 8 inits $\times$ 5 frames (40 solves) | 4.95 | 11/40 (27.5%) | no aggregate posterior; per-trial errors |

*Evidence level: `independent reproduction (this repo)`; source `real/linemod/results/theseus_baseline.md` §3–§6. Non-physical means the estimate puts the object behind the camera (`frac z>0.05 < 0.99`), which the P1 layer rejects by construction.*

The GT-init row is local convergence, not a baseline win. From cold starts Theseus returns whichever basin the descent enters — often the flip mode or a behind-camera stationary point of the unconstrained cost, exactly the solutions P1's enumeration rejects. It is the natural *no-enumeration* reference on this near-tied cost, not an estimator for the Laplace mean; local optimisation does not replace mode enumeration here.

Part B removes the shared-translation failure of the old construction: each of two labelled hypotheses eliminates its **own** translation, and the costs are mixed as $J_\theta=(1-\theta)J_{\rm true}+\theta J_{\rm sym}$. On the real frame `000010/15` the measured tie is at $\theta^\star\approx0.475$ (the construction is designed to tie at $0.5$, but resampling makes the two labelled costs not exactly symmetric); $\Delta J$ changes sign between $\theta=0.46$ and $0.479$, so the crossing is transverse, and the two minima stay within $\sim2^\circ$ of their labelled poses.

**Table 5.17. Controlled transverse crossing on real geometry (Part B, excerpt).** Full 11-row table in `real/linemod/results/two_mode.md`. `grad rel` is the analytic Laplace gradient against central FD of the layer output; MC is mode-level with 400 samples.

| $\theta$ | $\Delta J$ | $w(A)$ | Laplace grad rel | MC grad rel | $\Vert d\phi\Vert$ |
|-|-|-|-|-|-|
| 0.3500 | +1.64e-05 | 0.50031 | 2.59e-10 | 9.51e-03 | 1.129e-02 |
| 0.4600 | +2.20e-06 | 0.50004 | 8.61e-10 | 8.94e-05 | 1.123e-02 |
| 0.4790 | -2.22e-07 | 0.50000 | 7.82e-10 | 1.68e-01 | 1.122e-02 |
| 0.5000 | -2.89e-06 | 0.49995 | 1.76e-09 | 5.09e-02 | 1.121e-02 |
| 0.6500 | -2.17e-05 | 0.49960 | 4.03e-10 | 1.10e-02 | 1.112e-02 |

*Evidence level: `independent reproduction (this repo)`; source `real/linemod/results/two_mode.md` (Part B).*

Across the 11 rows the Laplace gradient agrees with central FD to $1.9\times10^{-10}$–$1.8\times10^{-9}$, while the 400-sample mode-level MC error is $8.9\times10^{-5}$–$1.7\times10^{-1}$; the hard argmin has a jump at the tie and its FD is undefined there, so no hard gradient is reported. This is a correctness result for the closed-form backward pass on a real cost, not an accuracy claim: no front-end is used, and the ADD-S proxy of E15 belongs to Part A, not to this crossing. Part B also exercises the fixed `dlogdetH_dtheta` path (Section 5.3.5).

**The earlier negative attempt (retained and diagnosed).** `run_symmetry.py` paired the same image bearings with model points $P$ and with the symmetry-transformed points $SP$, mixed the two full bearing-cost matrices, and eliminated one shared translation. It produced 0 usable rows in 9 candidate frames (`symmetry.json` is `[]`, `symmetry.md` a header). Two causes were diagnosed. (B1) `sym_of` used the `models_info` symmetry translation in millimetres as metres, offsetting the second point cloud by about $0.69$ m; after correction $J_B$ at its own pose is $4\times10^{-14}$ rather than a spurious value. (B2) Even after (B1), summing two full costs and eliminating a single shared $t$ makes the hypotheses compete for one translation: the minima collapse to a compromise pose ($t\approx-0.1$ versus the true $t_z\approx0.887$), so the cost is unimodal at the crossing. This is a construction failure, not evidence about symmetric objects; the new construction above is the positive result.

*Evidence level: `independent reproduction (this repo)`; sources `real/linemod/results/two_mode.md` (Part A/B), `real/linemod/README.md` (B1/B2 diagnosis), `real/linemod/results/symmetry.json` (old negative).*

### 5.6 Claim-to-evidence map

Table 5.18 maps each claim the paper makes to its strongest evidence and the file or table that carries the number. It is deliberately conservative: a claim is listed as "proved" only where Section 4 gives an argument, and "measured" only where a result file exists.

**Table 5.18. Claim-to-evidence map.**

| claim | evidence level | where |
|-|-|-|
| $x_R=2Uv_2(q)$; 21 $SO(3)$ constraints span the 20-dimensional degree-two Veronese ideal; the $10\times10$ SDP is its Shor relaxation | proved; exact rational rank check | Sections 4.2–4.3; E12 |
| constructed Choi–Lam gap $0.034188$, $\mathrm{rank}(X^\star)=6$, $\lambda_{\min}(Z)=-1.8\times10^{-11}$ | numerical dual certificate on a constructed instance | Table 5.5; E5 |
| $\lambda^\star=4J$, $3\times3$ Riemannian implicit differentiation, antipodal invariance | proved; FD checks at $10^{-7}$ or better | Table 5.8; E4, E6 |
| translation integrated exactly; Laplace rotation error $O(\tau)$ and its constants | derivation; 1D numerical slope 1.07 | Table 5.3; E2, E6 |
| fold behaviour: $\tau^{1/3}$ regulariser and Scorer continuation | derivation; measured 1D and 6D | Tables 5.4, 5.19; E3, E14 |
| 6D crossing: Laplace map smooth, hard argmin a step map | measured | Table 5.2; E1 |
| real two-mode cost: $K_{\rm phys}=2$ in 15/15 frames | measured | Table 5.12; E8 |
| hard argmin $2.06^\circ$ / $3.7$ mm; Laplace $44.24^\circ$ / $13.9$ mm, bias $+36.9^\circ$ | measured pose-error proxy, not a detector benchmark | Tables 5.13–5.14; E15 |
| classical forward solvers reach $2.0$–$2.2^\circ$ median | measured forward reference, not a layer comparison | Table 5.15; E16 |
| Theseus is local: 15/15 from GT, 100/300 from random init | measured local reference | Table 5.16; E17 |
| 40 complex projective roots on 4/4 instances | measured homotopy count of the algebraic system | Section 5.3.5; E18 |
| full-20D $10\times10$/$13\times13$ equivalence; upstream 17D strictly looser | proved; measured physical separation | Tables 5.6–5.7; E12 |
| detector-level ADD(-S)/ADD-S AUC, UTIAS, learned front-end, SDPR/EPro-PnP/BPnP comparisons | not measured | N1–N3 |

*Evidence level: the map itself is `paper claim` for the proved rows and `independent reproduction (this repo)` for the measured rows; "not measured" rows are `unverified`.*

---

## 6. Discussion and Limitations

1. **Scope and status of the SDP equivalence.** The $10\times10$ Veronese result is for the rotation subproblem after exact translation elimination, and its relation to the released $13\times13$ implementation now has a proved three-part structure (Section 5.3.4). (i) With the full 20-dimensional ideal, the $13\times13$ relaxation is *exactly* the $10\times10$ one — equal value and equal rank, by the rank-preserving block Schur bijection — so the $10\times10$ theory does cover the $13\times13$ relaxation once the ideal is complete. (ii) The released 24-constraint implementation spans only 17 dimensions, missing exactly $r_1\cdot r_2=r_1\cdot r_3=r_2\cdot r_3=0$, and is therefore strictly looser in general: $f^\star_{13,\mathrm{up}}\le f^\star_{13,\mathrm{full}}=f^\star_{10,\mathrm{full}}\le f^\star_{\mathrm{QPEP}}$, with $8/8$ strict gaps after symmetry-breaking perturbations of the Choi–Lam instance ($2.5\times10^{-3}$–$1.4\times10^{-2}$, rank $\{4,5\}$ versus 6). (iii) A 6-point physical 3D–3D instance with positive matrix weights and a unique ground-truth rotation separates the two by $1.8207$ with ranks 1 versus 4, even though the full-20D SDP is exactly tight and rank one there. Q2/Q13 is therefore closed, not reopened: full-20D equivalence is proven, the upstream constraint deficit is proven and demonstrated. The supplement's older claim that the released 24-constraint relaxation has the same value as the $10\times10$ one is the statement that was wrong; the $10\times10$ theory is unaffected. Constructed-instance realizability (Q1) is a separate and still open question (item 2).
2. **Realizability of the Choi–Lam instance.** The constructed $M$ is a valid symmetric cost matrix, but we have not shown that it arises from actual PnP or matrix-weighted correspondences. The positive-measure claim for *physical* cost matrices is open (Q1). Until then, Example 4.5 is a statement about the rotation subproblem, not about SDPRLayers' production behavior. This is separate from Q2/Q13: the 6-point instance of Section 5.3.4 *is* physical, and it separates the upstream implementation without relying on the constructed Choi–Lam objective.
3. **Root enumeration.** The forward pass is only as global as the solver. Missed roots, repeated roots, and numerical elimination failures break both the "global" claim and the Laplace sum. We treat completeness as an assumption and verify residuals. The algebraic counting question is now measured: a total-degree homotopy confirms 40 complex projective roots on 4/4 instances (E18), with real projective counts varying by instance ($26/12/12/20$) and $5/2/2/4$ real local minima. What remains is certification of the enumerator actually used in the reported runs, i.e. the missed-root rate of the multi-start Newton against the homotopy on every instance; a per-instance certified real-root count (Sturm/homotopy with multiplicity) is future work (Q3).
4. **Laplace bias, folds, and the two fold theorems.** The $O(\tau)$ error is non-uniform. At a saddle-node (fold) the mode count changes and a real-root-only Laplace sum is not even continuous. We claim smoothness for the exact smoothed posterior (Section 4.6) and, for the Laplace implementation, only away from folds. Section 5 measures both faces: at $\theta=0$ the Laplace map is smooth (its jump decays linearly with FD slope $\approx502$ while the hard map's jump stays $2.86$), but a genuine fold sits at $\theta_f\approx1.31\times10^{-3}$, where that minimum dies, its weight reaches $0.122$, $\Vert d\phi\Vert$ peaks at $3529$, and the pose map steps by $0.31$. Smaller $\tau$ or more starts do not remove the step; the $\det(H+\tau^{1/3}I)$ regularizer reduces the derivative spike by a factor 24 and the near-fold gradient error from $2.9\times10^{-1}$ to $1.2\times10^{-2}$, but only halves the value step ($0.309\to0.107$). Two results explain why.

   *P1-Q5b (real-root proxies must jump).* Let $J_\theta$ undergo a standard codimension-1 fold ($A_2$) at $\theta_f$: on the side where they exist ($\theta\to\theta_f^+$, with $+$ denoting that side), a secondary real minimum $q_+$ and a nearby saddle $q_-$ merge into $q_0$ (one Hessian eigenvalue $\lambda_1(q_+)\to0^+$, transverse Hessian $H_\perp(q_0)\succ0$, and a deeper minimum elsewhere with $J_0=J_{\theta_f}(q_0)>J_{\rm glob}$), and on the other side the real minimum is gone, so $K\to K-1$. Then any proxy that sums only over real local minima and gives the merging mode a positive regularized floor weight $\tilde w_+^{\rm reg}(\theta_f^+)=\frac{(2\pi\tau)^{3/2}}{\sqrt{\det(H_++\eta(\tau)I_3)}}\,e^{-J_0/\tau}$, with the scale-matched floor $\eta(\tau)=c_0\tau^{1/3}$, forces the smoothed loss $\bar\ell_{\rm reg}$ to jump at $\theta_f$ (with $Z_{\rm rest}$ the Laplace partition function of the other modes):
   $$\lim_{\theta\to\theta_f^+}\bar\ell_{\rm reg}-\lim_{\theta\to\theta_f^-}\bar\ell_{\rm reg}=\frac{\tilde w_+^{\rm reg}(\theta_f^+)}{\tilde w_+^{\rm reg}(\theta_f^+)+Z_{\rm rest}(\theta_f)}\big(\ell(q_0)-\bar\ell_{\rm rest}(\theta_f)\big)\neq0,$$
   and for $\eta=c_0\tau^{1/3}$ the jump is $\Theta\big(\tau^{1/3}e^{-(J_0-J_{\rm glob})/\tau}\big)$. It is exponentially small only when $J_0-J_{\rm glob}\gg\tau$, and stays $O(1)$ when $J_0-J_{\rm glob}=O(\tau)$; at any finite $\tau$ it is nonzero. This is a necessity statement, not an implementation defect, and it is why $0.309\to0.107$ does not go to zero.

   *P1-Q5c (complex-conjugate roots and the Scorer continuation).* The $40$-root action-matrix solver already carries the merged pair through the fold as a complex-conjugate pair. In the normal form $J_\theta(u,v)=J_0(\theta)+\tfrac12v^\top H_\perp(\theta)v+\tfrac{c_3(\theta)}{6}u^3-\alpha(\theta)u$ with $c_3>0$, $\alpha(\theta_f)=0$ and $\alpha'(\theta_f)>0$ (Gromoll–Meyer splitting), define the algebraic invariants $J_0=(J(q_+)+J(q_-))/2$ and $\zeta(\theta,\tau)=\alpha(\theta)\big(2/c_3(\theta)\big)^{1/3}\tau^{-2/3}$, and the semi-line Airy (Scorer) integral $\mathcal S(\zeta)=\int_0^{+\infty}e^{-s^3/3+\zeta s}ds$, which is entire with $\mathcal S(0)=\Gamma(1/3)/3^{2/3}\approx1.2879$. Replacing the two real-root weights by the single continuation
   $$\tilde w_{\rm fold}(\theta,\tau)=\frac{2\pi\tau}{\sqrt{\det H_\perp(\theta)}}\,\tau^{1/3}\Big(\frac{2}{c_3(\theta)}\Big)^{1/3}e^{-J_0(\theta)/\tau}\,\mathcal S\big(\zeta(\theta,\tau)\big)$$
   gives a map that is $C^\infty$ across $\theta_f$ with zero value jump, because $(J_0,\alpha,c_3,H_\perp)$ are symmetric functions of the characteristic polynomial and $\mathcal S$ is entire. For $\zeta\gg1$ it reduces to the ordinary 3D Laplace weight of the surviving minimum, $\tilde w_{\rm fold}=\tilde w_{\rm Lap}(q_+)\big(1+O(\tau/\Delta J_{\rm bar})\big)$; for $\zeta\ll-1$ it decays smoothly, $\mathcal S(\zeta)=|\zeta|^{-1}\big(1+O(|\zeta|^{-3})\big)$. **The Scorer continuation is implemented in this repository at the invariant level** (`theory/results/scorer_fold.md`, `independent reproduction (this repo)`): the 6D fold at $\theta_f=1.3185\times10^{-3}$ is tracked on both sides (real minimum/saddle pair and the complex-conjugate pair), the four invariants $(J_0,\alpha,c_3,H_\perp)$ and $\mathcal S(\zeta)$ are continuous across it, and the measured pose jump is $1.21\times10^{-2}$ versus $3.04\times10^{-1}$ (standard) and $1.07\times10^{-1}$ ($\tau^{1/3}$), i.e. at the smooth-map derivative scale ($|d\phi|\sim6\times10^2$, jump/step $6.0\times10^2$ versus $1.5\times10^4$). The caveats are that the identification of the measured invariants with the normal form is leading-order (the quartic's fourth-order coefficient is $\approx8.3$), and that this 6D fold is shallow ($\Delta J\sim10^{-6}\ll\tau$), so the Scorer weight is far from its 3D-Laplace limit *at* the fold. The main table still reports the real-root-only layer; its step is the one P1-Q5b predicts.

**Table 5.19. Scorer continuation vs real-root proxies at the 6D fold** ($\theta_f=1.3185\times10^{-3}$, $\tau=0.01$, grid $\theta_f\pm10^{-4}$ step $2\times10^{-5}$). `jump` is the largest consecutive pose difference; a genuine step keeps `jump/step` growing under refinement, a smooth map has `jump/step` at the `|dpose|` scale ($\sim6\times10^2$).

| variant | max jump | jump / step | K on grid |
|-|-|-|-|
| standard real-root Laplace | $3.04\times10^{-1}$ | $1.52\times10^{4}$ | 3-4 |
| $\det(H+\tau^{1/3}I)$ regularized | $1.07\times10^{-1}$ | $5.34\times10^{3}$ | 3-4 |
| Scorer continuation | $1.21\times10^{-2}$ | $6.03\times10^{2}$ | 4 |

*Evidence level: `independent reproduction (this repo)`; source `theory/results/scorer_fold.md`. The 1D three-way check reproduces the proof's $0.867/0.497/0.3391$ to $\le1.6\times10^{-4}$.* The continuity claim is tested by grid refinement (staggered grids, half-width $10^{-4}$, step $d\in\{4\times10^{-5},\ldots,2.5\times10^{-6}\}$): the Scorer `jump/d` stays $404.3,404.2,404.0,403.7,403.3$ over a $16\times$ range of $d$ (jump $\propto d$, no numerical floor at $d=2.5\times10^{-6}$), whereas the standard map's `jump/d` grows from $7.6\times10^{3}$ to $2.1\times10^{5}$ and the regularized one from $2.7\times10^{3}$ to $4.4\times10^{4}$ (a constant step). This is the decisive smoothness test and it supersedes the earlier 'no refinement test' caveat.

   Consequently the fold neighbourhood is outside the scope of the real-root-only Laplace layer: it needs the Scorer continuation, or a local numerical integral over the merging pair, for a continuity claim. The two propositions are written up in `theory/proofs.md` §7.5 (P1-Q5b §7.5.1, P1-Q5c §7.5.2; §7.3 keeps the earlier gap statement), and the measured record is `experiments/results/fold_6d.md`.
5. **Symmetric objects: a positive construction and its accuracy cost.** The earlier symmetric-object construction failed for diagnosed reasons, but the new real-data construction does produce two genuine modes on 15/15 LINEMOD `eggbox`/`glue` frames (Section 5.5). The measured outcome is not an accuracy win: the hard argmin is accurate at the near-ties (median $2.06^\circ$, ADD-S $3.7$ mm, at the level of a classical forward solver) while the Laplace mean is pulled toward the flip (median $44.24^\circ$, ADD-S $13.9$ mm; paired bias $+36.9^\circ$). The earlier $58.3^\circ$ figure applied the geodesic formula to the unprojected 9-vector mean, which is not a rotation; the direction of the bias claim is unchanged. This is the accuracy/smoothness trade the layer makes, and the behavior on symmetric objects is now measured rather than untested.
6. **Third derivatives.** $\nabla_\theta\log\det H_k$ requires third $q$-derivatives of $J$ and the implicit derivative of $q^{(k)}$. This is tractable for a quartic but non-trivial to implement correctly; it is the main gap between the "3×3 backward" slogan and the full Laplace KL (Q6).
7. **Temperature.** $\tau$ is a cost-scale-dependent hyperparameter. Without scale normalization it is not identifiable; with it, $\tau$ can be learned. The $\tau\to0$ limit recovers the deterministic gradient only for the posterior mean/argmin, not for the raw KL loss.
8. **No inequality constraints, no outlier robustness.** These are delegated to the companion tracks. The present layer assumes $G\succ0$ and a fixed weighted Gram matrix; it does not handle non-penetration, handedness inequalities, or gross outliers.
9. **GPU batching.** The algebraic root enumeration is the engineering bottleneck. A batched $40\times40$ complex eigensolver is plausible but not yet benchmarked (Q10).

---

## 7. Conclusion

We have presented `Laplace-QPEPLayer`, a differentiable pose layer that combines the quaternion algebra of QPEP with a $3\times3$ Riemannian implicit differentiation and a multi-modal Laplace posterior. The theoretical core is an exact orthogonal bridge between the rotation lift and the Veronese embedding, a proof that the 21 standard $SO(3)$ constraints span the degree-two Veronese ideal, a certified Choi–Lam gap instance for the rotation subproblem, and a closed-form backward pass with antipodal invariance. The probabilistic core is an exact translation integral and an analytic Laplace sum over all enumerated rotation minima, with explicit error order and failure modes. The design is deliberately narrow and its claims are scoped: no SDP lift at inference, no Monte Carlo sampling, no inequality constraints, no outlier kernel. Section 5 reports the experiments that are complete: the synthetic 6D branch-crossing benchmark (with an FD-step study, a measured fold, a fold theory and a timing/batching study), the 1D exact-posterior checks, the certificate and Jacobian verification, the ten-vs-thirteen SDP analysis (proved full-20D equivalence, proved and measured upstream looseness, and the 6-point physical separation), and a real-data LINEMOD two-mode study with a pose-error/ADD-S proxy and forward/local (classical PnP, Theseus) references. The real-data result is a trade, not an accuracy win: the hard argmin is accurate at the near-ties (at the level of a classical EPnP/SQPNP, median $2.06^\circ$ and $3.7$ mm ADD-S) while the Laplace mean is biased toward the flip (median $44.24^\circ$ and $13.9$ mm ADD-S), and the earlier $58.3^\circ$ figure was the unprojected 9-vector mean, not a rotation error. What remains is the real-image accuracy step: a learned front-end and the detector-level ADD(-S)/ADD-S AUC benchmark on LINEMOD/LINEMOD-Occlusion, the UTIAS In-The-Dark pipeline, the SDPR and EPro-PnP comparisons, and a batched (GPU) forward; every reported number is tagged with its evidence level.

---

## References

[1] C. Holmes, F. Dümbgen, and T. D. Barfoot, "SDPRLayers: Certifiable Backpropagation Through Polynomial Optimization Problems in Robotics," *IEEE Transactions on Robotics*, 2025. arXiv:2405.19309. DOI: 10.1109/TRO.2025.3578228. Code: `utiasASRL/sdprlayer`.

[2] H. Chen, P. Wang, F. Wang, W. Tian, L. Xiong, and H. Li, "EPro-PnP: Generalized End-to-End Probabilistic Perspective-n-Points for Monocular Object Pose Estimation," in *Proc. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2022 (Best Student Paper). arXiv:2203.13254. Extended version: *IEEE TPAMI*, 2024, arXiv:2303.12787.

[3] J. Wu, Y. Zheng, Z. Gao, Y. Jiang, X. Hu, Y. Zhu, J. Jiao, and M. Liu, "Quadratic Pose Estimation Problems: Globally Optimal Solutions, Solvability/Observability Analysis, and Uncertainty Description," *IEEE Transactions on Robotics*, vol. 38, no. 5, pp. 3314–3335, 2022. DOI: 10.1109/TRO.2022.3155880.

[4] B. Chen, A. Parra, J. Cao, N. Li, and T.-J. Chin, "End-to-End Learnable Geometric Vision by Backpropagating PnP Optimization," in *Proc. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2020, pp. 8100–8109. arXiv:1909.06043.

[5] L. Pineda, T. Fan, M. Monge, S. Venkataraman, P. Sodhi, R. T. Q. Chen, J. Ortiz, D. DeTone, A. Wang, S. Anderson, J. Dong, B. Amos, and M. Mukadam, "Theseus: A Library for Differentiable Nonlinear Optimization," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2022. arXiv:2207.09442.

[6] M.-D. Choi and T.-Y. Lam, "Extremal Positive Semidefinite Forms," *Mathematische Annalen*, vol. 231, pp. 1–18, 1977.

[7] M.-D. Choi, T.-Y. Lam, and B. Reznick, "Real Zeros of Positive Semidefinite Forms. I," *Mathematische Zeitschrift*, vol. 171, pp. 1–26, 1980.

[8] D. Hilbert, "Über die Darstellung definiter Formen als Summe von Formenquadraten," *Mathematische Annalen*, vol. 32, pp. 342–350, 1888.

[9] J. B. Lasserre, "Global Optimization with Polynomials and the Problem of Moments," *SIAM Journal on Optimization*, vol. 11, no. 3, pp. 796–817, 2001.

[10] M. Laurent, "Sum of Squares, Moment Matrices and Optimization over Polynomials," in *Emerging Applications of Algebraic Geometry*, IMA Volumes in Mathematics and Its Applications, vol. 149, Springer, 2009, pp. 157–270.

[11] G. Blekherman, "There Are Significantly More Nonnegative Polynomials Than Sums of Squares," *Israel Journal of Mathematics*, vol. 153, pp. 355–380, 2006.

[12] J. Harris, *Algebraic Geometry: A First Course*, Graduate Texts in Mathematics, vol. 133, Springer, 1992.

[13] N. Bleistein and R. A. Handelsman, *Asymptotic Expansions of Integrals*, Dover, 1986.

[14] R. Wong, *Asymptotic Approximations of Integrals*, Academic Press, 1989.

[15] G. S. Chirikjian, *Stochastic Models, Information Theory, and Lie Groups, Volume 2*, Birkhäuser, 2012. (Haar measure and the $S^3\to SO(3)$ double cover.)

[16] A. L. Dontchev and R. T. Rockafellar, *Implicit Functions and Solution Mappings*, Springer, 2009.

[17] J. Nocedal and S. J. Wright, *Numerical Optimization*, 2nd ed., Springer, 2006.

<!--
Reference notes / TODO before submission:
- sQPEP (Xue et al., 2025) and the adjacent dQP/POLI/Pyatov entries from an earlier internal list are not included
  because their bibliographic details were not independently verified. Add only after checking the DOI.
- The 13x13/24-constraint fact follows the released SDPRLayers code and the repository KB
  (kb/sdpr-forward-net/01.md §3.1); cite the code commit in the camera-ready.
-->
