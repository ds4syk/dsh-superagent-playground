# Laplace-QPEPLayer: Zero-Lift Differentiable Pose Estimation via Quaternion Algebra and Multi-Modal Laplace Smoothing

> **Draft status (2026-11-27, v3).** This draft contains the theory and a results section: a synthetic 6D branch-crossing benchmark (with an FD-step study and a measured fold), 1D exact-posterior checks, an SDP certificate and a $10\times10$ vs $13\times13$ comparison, and a real-data LINEMOD two-mode study. There is still no real-image accuracy claim: no learned front-end, no ADD(-S)/ADD-S/AUC benchmark, and UTIAS In-The-Dark has not been run. The earlier LINEMOD symmetric-object attempt is retained as a diagnosed negative construction result. Numerical claims are tagged `paper claim`, `independent reproduction (this repo)`, or `unverified`; preliminary numbers from an earlier internal formulation are not reported as results. References are real; entries we could not verify are omitted rather than guessed.

---

## Abstract

Differentiable pose estimation layers sit between learned front-ends and geometric back-ends. Deterministic layers built on semidefinite programming (SDP) relaxations provide global optimality when the relaxation is tight, but they lift the problem to a matrix variable, rely on solver infrastructure, and break down when two global branches cross. Probabilistic layers built on Boltzmann distributions avoid hard branch selection, but typically rely on Monte Carlo sampling in a six-dimensional pose space. We show that the quaternion formulation of quadratic pose estimation problems (QPEP) supplies the missing algebraic structure for a third option. First, for the rotation subproblem obtained after exact elimination of translation, we prove that the standard $10$-dimensional rotation lift is *exactly* $2Uv_2(q)$ for an orthogonal $U$, that the 21 $SO(3)$ constraints span precisely the degree-two vanishing ideal of the Veronese variety $\nu_2(\mathbb{RP}^3)$, and that the resulting $10\times10$ SDP is the Veronese (Shor) relaxation of a quartic on $\mathbb{S}^3$. A constructed Choi–Lam instance exhibits a certified relaxation gap of $0.0342$ with $\mathrm{rank}(X^\star)=6$. Second, we derive a closed-form Euler multiplier $\lambda^\star=4J$ and a $3\times3$ Riemannian implicit differentiation on $\mathbb{S}^3$, including exact antipodal invariance. Third, we replace the hard argmin by a multi-modal Laplace posterior: translation is integrated out exactly, and the rotation integral is approximated analytically over *all* QPEP minima with weights $w_k\propto e^{-J_k/\tau}/\sqrt{\det H_k}$. The resulting layer has no Monte Carlo variance, no SDP lift at inference, and a deterministic backward pass that reduces to one $3\times3$ solve per active mode (plus a trace term for the smoothed KL). We analyze the $O(\tau)$ error of the Laplace approximation, its breakdown at branch merging, and the $1/\tau$ scaling of the KL gradient. We report controlled synthetic branch-crossing experiments, 1D exact-posterior checks, the certified gap instance and a numerical $10\times10$ vs $13\times13$ comparison, and a real-data LINEMOD two-mode study: at real near-ties the hard argmin is accurate (median $2.06^\circ$) while the Laplace posterior mean is biased toward the flip (median $58.3^\circ$), so the layer trades accuracy for smoothness there. Real-image accuracy (ADD(-S), ADD-S AUC) on LINEMOD/LINEMOD-Occlusion and UTIAS In-The-Dark is not established.

---

## 1. Introduction

End-to-end learning for geometry requires differentiating through a pose estimator. The estimator is a nonconvex optimization problem with multiple local minima, and the derivative of its *global* solution is a discontinuous object whenever the identity of the global minimum changes. Two families of methods currently dominate.

**Deterministic global layers.** SDPRLayers [1] lifts a homogeneous quadratically constrained quadratic program (QCQP) to a matrix variable $X\succeq0$, solves the Shor relaxation, and, when the relaxation is tight, reads off a rank-one global solution together with a dual certificate. The backward pass differentiates the KKT system of the original QCQP, not the SDP, using an implicit-selection argument that handles redundant constraints. The approach is certifiable and has been demonstrated on matrix-weighted stereo localization. Its price is structural: the forward pass solves an SDP on the CPU, the problem dimension grows with the number of variables, and the guarantees are conditional on tightness and on constraint qualifications. A symmetric object or a near-planar configuration can produce a non-tight relaxation; the layer then has no pose to return.

**Probabilistic layers.** EPro-PnP [2] interprets the PnP residual as a Boltzmann distribution over $SE(3)$ and trains with a KL loss, using adaptive multiple importance sampling. This removes hard branch selection: ambiguous poses receive probability mass, and the gradient averages over hypotheses. The price is variance: Monte Carlo integration in six dimensions can miss distant modes, and the variance of the gradient must be controlled by sample count.

**The gap.** A layer that is (i) globally exhaustive without a convex lift, (ii) differentiable with a constant-size backward pass, and (iii) smooth at branch crossings without sampling, is missing. The QPEP line of work [3] provides exactly the algebraic ingredients: a quaternion parameterization with a single unit-norm constraint, exact elimination of translation, a Gröbner-basis solver that enumerates all KKT roots, and a closed-form observability/covariance analysis. What QPEP does not provide is a derivative with respect to the data and a probabilistic treatment of ambiguity.

This paper builds `Laplace-QPEPLayer` on those ingredients. The optimization variable is $q\in\mathbb{S}^3$; translation is eliminated in closed form; the forward pass enumerates all real KKT roots; the backward pass uses a $3\times3$ Riemannian implicit differentiation; and ambiguity is handled by a Laplace approximation of the Boltzmann posterior over the enumerated minima. The design is deliberately narrow: we do not handle inequality constraints (companion track `Constrained-QPEPLayer`) or robust outlier kernels (companion track `Robust-QPEPLayer`); those are orthogonal extensions.

**Contributions.**

1. **An exact algebraic bridge for the rotation subproblem.** We prove that the homogeneous rotation lift satisfies $x_R(q)=2Uv_2(q)$ with $U\in O(10)$ (Theorem 0.1), that the 21 standard $SO(3)$ constraints span exactly the degree-two vanishing ideal of $\nu_2(\mathbb{RP}^3)$ (Theorem 0.2), and that the associated $10\times10$ SDP is the Veronese (Shor) relaxation of the quartic on $\mathbb{S}^3$ (Theorem 0.3). The scope is explicit: this is the *rotation subproblem after exact translation elimination*, not the $13$-dimensional SDP solved by the released SDPRLayers implementation.
2. **A constant-size Riemannian backward pass.** We derive $\lambda^\star=4J$ from Euler homogeneity, the Riemannian Hessian $H_{\mathbb{S}^3}=E^\top(\nabla^2J-4JI)E$, and the implicit Jacobians of $q^\star$ and $t^\star$ (Theorem 1.1), with exact antipodal invariance. The backward pass solves a $3\times3$ positive-definite system per active mode, instead of a KKT system with redundant constraints.
3. **A multi-modal, sampling-free Laplace layer.** Translation is integrated out exactly; rotation is handled by a Laplace approximation over all enumerated minima, giving analytic weights and gradients (Theorem 1.3). We give the correct measure/normalization constants, prove that the rotation error is $O(\tau)$ away from bifurcations, and identify where the approximation fails.
4. **A differentiability certificate and an honest failure analysis.** We give sufficient conditions for a $C^\infty$ global pose map (Theorem 1.2), show that a natural "exact iff" claim is false with counterexamples, and isolate the branch-merging regime where uniform asymptotics are required. A constructed Choi–Lam instance with a certified gap and a reproducible verification suite accompany the theory.

---

## 2. Related Work

**SDPRLayers [1].** Holmes, Dümbgen, and Barfoot lift a homogeneous QCQP to $X\succeq0$, solve the Shor relaxation, and differentiate the original KKT system with an implicit-selection theorem that tolerates redundant constraints. The released pose estimator uses a $13$-dimensional vector $x=[h;\mathrm{vec}(C);t]$ and 24 quadratic constraints (6 orthogonality, 9 handedness, 9 norm), with the last 18 marked redundant. Our work differs in four ways. (i) *Variable:* we optimize over $q\in\mathbb{S}^3$ (four variables, one constraint) and eliminate $t$ exactly; the dimension of the nonconvex problem is constant and independent of the number of correspondences. (ii) *Global mechanism:* QPEP enumerates all KKT roots and compares objectives; there is no convex relaxation and hence no tightness assumption. (iii) *Backward:* we differentiate the reduced stationarity system with a $3\times3$ Riemannian Hessian; SDPRLayers differentiates a KKT system with redundant rows after row reduction. (iv) *Ambiguity:* we return a smoothed posterior over modes; SDPRLayers returns a single certifiable solution and has no defined behavior when the relaxation is not tight. We do *not* claim that our $10\times10$ analysis applies to the released $13$-dimensional SDP; that is an open question (Section 6, Q2).

**EPro-PnP [2].** Chen et al. define a Boltzmann distribution on $SE(3)$ and train with a KL loss, using adaptive multiple importance sampling. Our posterior is also Boltzmann, but the support is $\mathbb{S}^3/\{\pm1\}$ rather than $SE(3)$, translation is marginalized exactly, and the rotation integral is approximated analytically over an exhaustive list of modes rather than sampled. The trade is bias for variance: EPro-PnP is asymptotically exact but high-variance; we are deterministic but have an $O(\tau)$ Laplace bias away from bifurcations and a non-uniform error near them. A fair comparison must separate these two error sources (Section 5).

**QPEP [3].** Wu et al. unify a large class of pose problems as quadratic problems in a unit quaternion, eliminate translation, solve the resulting polynomial system with a Gröbner basis, and analyze solvability, observability, and covariance. Their solver returns at most 40 candidate complex roots in the released implementation; the global solution is obtained by comparing objectives, not by a convex certificate. QPEP provides no derivative with respect to the data. Our contribution is precisely the missing differentiable layer: the closed-form Euler multiplier, the $3\times3$ implicit differentiation, the antipodal invariance, and the Laplace posterior over the enumerated roots. We rely on QPEP for the forward enumeration and treat completeness of the enumeration as an explicit assumption.

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

**Scope.** This is the rotation subproblem after exact translation elimination. The released SDPRLayers implementation solves a $13\times13$ SDP in $[h;\mathrm{vec}(C);t]$ with 24 constraints; whether the two relaxations have the same value is *not* proved here. We also do not identify our SDP with the textbook second-order Lasserre moment relaxation with a sphere localizing matrix; we call it the degree-two Veronese/Shor relaxation.

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

A QPEP-specific counting issue: a naive count gives $K\le27$ real minima. The released LibQPEP solver returns up to 40 candidate complex roots; 27 has no cited source, and an odd count is incompatible with the free antipodal symmetry $(q,\lambda)\mapsto(-q,\lambda)$ for a finite root set. We therefore write "$K$ enumerated real minima, at most 40 candidates in the released solver", and we treat enumeration completeness as an assumption (A2), verified numerically by residuals rather than assumed.

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

> **Section status (2026-11-27, draft v3).** This section reports the synthetic 6D branch-crossing benchmark (with an FD-step study and a measured fold), the 1D exact-posterior checks, an SDP certificate and a $10\times10$ vs $13\times13$ comparison, and a real-data LINEMOD two-mode study. The LINEMOD study is a controlled ambiguity and correctness measurement, not an accuracy benchmark: there is no learned front-end, no ADD(-S)/ADD-S/AUC number, and UTIAS In-The-Dark has not been run.

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
| E12 | $10\times10$ vs $13\times13$ SDP check | Choi–Lam, random and constructed instances ($13\times13$ constraints replicated from upstream code) | 10 instances + 8 perturbations | run | `sdp/results/sdp_check.md` |
| E13 | Robustness, fold and timing study | synthetic 6D crossing | `n_starts` sweep, 20 timing repetitions, batched forward | run | `results/robustness.md` |
| N1 | LINEMOD / LINEMOD-Occlusion accuracy | full benchmark with SDPR-IS, SDPR-SDP, EPro-PnP, BPnP, Theseus | — | **not run** | TODO |
| N2 | UTIAS In-The-Dark | full SDPRLayers pipeline | — | **not run** | TODO |
| N3 | End-to-end training and GPU batching | — | — | **not run** | TODO |

*Evidence level: E1–E13 are `independent reproduction (this repo)`; N1–N3 are `unverified` (design only).*

The E1 instance is constructed, not fitted (`experiments/code/instance.py`): $M(\theta)=M_{\rm sym}+\theta M_{\rm break}$ with a $P$-invariant double well, $M_{\rm break}$ chosen so that $d(J_B-J_A)/d\theta|_{\theta=0}=2.0$, and a fixed translation block with $G\succ0$; the instance seed is 20261127. Mode enumeration is a batched damped Newton on $\mathbb{S}^3$ with 128 random starts, followed by an $SO(3)$-distance deduplication (threshold $10^{-5}$ rad) and a stationarity filter ($\|\nabla J\|\le10^{-8}$, $\lambda_{\min}(H_{\mathbb S^3})>10^{-8}$). It is not the released LibQPEP solver, so the completeness assumption (A2) is not independently certified in these runs.

Not run, and not estimated: every baseline comparison (SDPR-IS, SDPR-SDP, EPro-PnP, BPnP, Theseus), the LINEMOD/LINEMOD-Occlusion accuracy matrix, the UTIAS In-The-Dark pipeline, end-to-end training, and GPU batched root enumeration. No ADD(-S), ADD-S AUC, detector-based rotation error, translation error, success rate, training curve, end-to-end training throughput, or peak-memory number exists in the repository; these are marked TODO rather than filled with guesses. The LINEMOD entries E8–E9 are controlled ambiguity and correctness measurements on real geometry, not an accuracy benchmark: no learned front-end and no ADD(-S)/ADD-S/AUC.

### 5.2 Metrics

The executed experiments use the following quantities; all are computed from the result files cited in Table 5.1.

- **Gradient relative error.** $\|\hat d\phi-d\phi_{\mathrm{FD}}\|_2/\|d\phi_{\mathrm{FD}}\|_2$ with $\phi=[\mathrm{vec}\,R(q);t^\star(q)]\in\mathbb{R}^{12}$. The reference is a branch-following central FD with $h=10^{-4}$ (E1, E8, E9) or $h=10^{-3}$ (old E7); E13 uses a Richardson-extrapolated reference built from $h=10^{-4}$ and $5\times10^{-5}$. For the Laplace layer the FD differentiates the same smoothed map; for the hard argmin at a global tie the reference branch switches inside $\pm h$, and the entry is reported as N/A rather than filled with a jump. Because the FD itself carries a truncation error, the $h$-scaling is measured separately in E10.
- **MC pose spread.** Euclidean norm of the standard deviation of the 12-vector $\phi$ over 20 independent mode-level MC means, each from $n=200$ mode samples (E1; synthetic units; the translation block is fixed at scale $D_{\text{scale}}=0.30$, so this is a relative spread, not metres). In E9 the MC baseline is a single mode-level estimate with $n=400$ samples per $\theta$; E13 uses $n=200$ with 3 seeds and also reports the 20-seed mean.
- **Real-data pose error (E8).** Rotation error of the hard-argmin pose and of the Laplace mean pose against the BOP ground-truth rotation, in degrees. No translation or ADD metric is reported for E8 because the correspondences are constructed from depth and mesh, not from a detector.
- **Wall-clock.** CPU milliseconds per layer call. E13 reports 20-repetition medians and p95; the synthetic per-point medians of the main-table run are indicative only because they move with machine load. It is not a memory benchmark.
- **1D posterior error.** $|E_{\mathrm{Lap}}-E_{\mathrm{exact}}|$, its ratio to $\tau$, the log-log slope of $|E_{\mathrm{Lap}}-E_{\mathrm{exact}}|$ versus $\tau$, and the relative error of $\hat E=dE/dg$ against the exact quadrature value.
- **Partition-function error (fold).** $|Z_{\mathrm{Lap}}-Z_{\mathrm{exact}}|/Z_{\mathrm{exact}}$, with the same ratio for the $\tau^{1/3}$-regularized volume factor.
- **SDP certificate quantities.** Primal value, dual $\mu$, $\lambda_{\min}(Z)$ of the dual certificate, $\mathrm{rank}(X^\star)$ at tolerance $10^{-6}$, $\lambda_1/\lambda_2$, and the gap $f^\star-\mu$.
- **Derivative checks.** Relative error of the analytic Riemannian Hessian, implicit $q^\star$ and $t^\star$ Jacobians, and $\partial_\theta\log\det H$ against central FD.
- **Not measured.** ADD(-S), ADD-S AUC, rotation error (deg) on a detector, translation error (m), success rate, NLL calibration, training curves, end-to-end training throughput, peak memory, and missed-root rate against an independent certified solver. Not run.

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

**Mode count and fold.** The mode count over the grid is $K\in\{1,2,3,4\}$ ($K=1$ at 30 points, $K=2$ at 38, $K=3$ at 12, $K=4$ at the crossing point); these are the local minima returned by multi-start Newton with the filter of Section 5.1, not a certified root count. The smoothness statement is restricted to $\theta=0$ and $|\theta|\lesssim1.3\times10^{-3}$: a secondary minimum exists only for $|\theta|\lesssim1.4\times10^{-3}$ and disappears at the fold, where the Laplace map is discontinuous (Section 5.5).

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

#### 5.3.4 Rotation-SDP certificate and the $10\times10$ vs $13\times13$ gap

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

**The $13\times13$ SDPRLayers relaxation is looser, and strictly so in general.** The 21 constraints of the $10\times10$ rotation SDP span a 20-dimensional space; the 24 constraints of the $13\times13$ SDPRLayers relaxation span only 17 dimensions, and the missing three are exactly the row-orthogonality constraints $r_1\cdot r_2=r_1\cdot r_3=r_2\cdot r_3=0$ (row norms are already implied by the other constraints). Hence $f^\star_{13}\le f^\star_{10}$ for every instance, with equality only when the $13\times13$ optimum happens to satisfy row orthogonality. The $13\times13$ constraint generators were replicated row-by-row from the released SDPRLayers `pose_est.py` (commit `62d8ac6`, lines 341–417) and compared element-wise against the AST-extracted upstream generator bodies: all 24 constraints match with max $|\Delta|=0$, and $\max_i|x^\top A_i x|=1.4\times10^{-16}$ on random rotations (`sdp_check.json`, `upstream.ran = true`). The $10\times10$ side reuses `verify/verify_theorem0.py` and reproduces the regression value above.

**Table 5.6. The $13\times13$ SDPRLayers relaxation is looser than the $10\times10$ rotation SDP, and strictly so in general.** $f^\star_{10}$ is the $10\times10$ value and $f^\star_{13}$ the $13\times13$ value; `CL_exact` is the constructed Choi–Lam instance and `rowdot` is the constructed counterexample. The difference column is computed from the printed $f$-values; for `CL_exact` and `p6_data` the true differences are $3.7\times10^{-11}$ and $-2.9\times10^{-10}$, below the six-digit columns. Representative rows; the full 10-instance and 8-perturbation tables are in `sdp/results/sdp_check.md`.

| instance | $f^\star_{10}$ | $f^\star_{13}$ | $f^\star_{10}-f^\star_{13}$ | rank $X_{10}^\star$ | rank $X_{13}^\star$ |
|-|-|-|-|-|-|
| `CL_exact` | 1.965812 | 1.965812 | 3.7e-11 | 6 | 6 |
| `CL_pert0` | 1.962860 | 1.960078 | 2.8e-03 | 6 | 5 |
| `CL_pert5` | 1.964571 | 1.950275 | 1.4e-02 | 6 | 4 |
| `rowdot` | 1.000000 | -1.000000 | 2.00 | 10 | 7 |
| `rowdot_gen` | -0.060192 | -2.029230 | 1.97 | 1 | 4 |
| `p6_data` | 0.252326 | 0.252326 | 0.0 | 1 | 1 |

*Evidence level: `independent reproduction (this repo)`; source `sdp/results/sdp_check.md` and `sdp/run_sdp_check.py`.*

The Choi–Lam equality is a coincidence of symmetry: its $13\times13$ optimum lies on the row-orthogonality face, so $f^\star_{13}=f^\star_{10}=1.9658119658$ with rank 6 on both sides. After eight symmetry-breaking perturbations of order $10^{-2}$, $8/8$ instances have $f^\star_{13}<f^\star_{10}$ by $2.5\times10^{-3}$–$1.4\times10^{-2}$, with $\mathrm{rank}(X_{13}^\star)\in\{4,5\}$ against $\mathrm{rank}(X_{10}^\star)=6$. The constructed objective $Q=-r_1\cdot r_2+\tfrac14 I$, which equals 1 identically on $\mathbb{S}^3$, gives $f^\star_{10}=1.000000$ and $f^\star_{13}=-1.000000$ (gap $2.00$) and still $1.97$ after a $10^{-2}$ perturbation. These measurements **confirm the gap in the negative direction**: they do not show that the two relaxations coincide; they show that the $13\times13$ one is strictly looser in general. This does not close Q2 in the sense of a proof revision: the supplement's statement that the $13\times13$ and $10\times10$ values coincide is contradicted by these measured counterexamples, so the theory needs review, and the numerical evidence is what we report here.

#### 5.3.5 Derivative formulas and the Laplace constant

**Table 5.7. Numerical checks of the backward pass (E6) and the closed-form $\partial_\theta\log\det H$ (E4).** All entries are relative errors unless marked absolute.

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

#### 5.3.6 Ablations

**A. Temperature.** Table 5.8 reports the gradient relative error of the exact-logdet Laplace layer for $\tau\in\{0.05,0.02,0.01,0.005\}$ at the crossing, near the crossing, and far from it.

**Table 5.8. $\tau$ sweep (gradient relative error).**

| $\theta$ | $\tau=0.05$ | 0.02 | 0.01 | 0.005 |
|-|-|-|-|-|
| +0.000 | 1.20e-03 | 5.84e-04 | 4.34e-04 | 1.51e-03 |
| +0.010 | 1.29e-05 | 1.07e-04 | 3.46e-05 | 1.64e-05 |
| -0.010 | 1.29e-05 | 1.06e-04 | 3.46e-05 | 1.63e-05 |
| +0.100 | 8.47e-07 | 8.46e-07 | 8.46e-07 | 8.46e-07 |
| -0.100 | 8.43e-07 | 8.42e-07 | 8.42e-07 | 8.42e-07 |

*Evidence level: `independent reproduction (this repo)`; source `results/ablation.md`, section A.*

At the crossing the error is smallest at $\tau=0.01$ ($4.34\times10^{-4}$) and rises again at $\tau=0.005$ ($1.51\times10^{-3}$), which is consistent with the FD reference becoming unreliable when the two modes are nearly degenerate and $\tau$ is small; it is not evidence for an optimal $\tau$. Far from the crossing the error is $\approx8.5\times10^{-7}$ and independent of $\tau$ at the printed precision. No learned-$\tau$ experiment was run.

**B. Volume factor.** Table 5.9 compares the closed-form $\log\det$ derivative, a tangent-space FD, and dropping the volume factor from the weights.

**Table 5.9. Volume-factor variants.** `exact` is the closed form of Section 4.6, `numeric` is a tangent-space FD, `none` drops $\tfrac12\log\det H_k$ from the weights; `pose diff` and `grad diff` are Euclidean norms of the differences in the 12-vector against `exact`.

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

**C. Top-$K$ mode truncation.** Table 5.10 keeps only the $K$ highest-weight modes and compares the result with the sum over all modes.

**Table 5.10. Top-$K$ mode truncation (reference: all modes).**

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

- **CPU only.** All reported runs are CPU-only single-process jobs: the 6D main run is a Slurm `cpu` job, and the ablation, LINEMOD, 1D, `verify/`, SDP-check, robustness, FD-step and regression scripts are short CPU runs (the LINEMOD timings are login-node CPU). No GPU, no Mosek, and no multi-node run was used. The SDP certificate uses cvxpy 1.9.3 with SCS ($\varepsilon=10^{-9}$, `max_iters=200000`); the SDP comparison uses SCS with the same tolerances under cvxpy 1.7.5.
- **Timing (E13).** From 20 repetitions, a full synthetic layer call at $\theta=0$ takes 61.9 ms median (p95 62.1): mode enumeration is 88%, the $K$ implicit $3\times3$ solves 1.6%, and the analytic backward pass (weight gradient and mixing) 0.1%. At $\theta=-0.1$ the whole call is 17.7 ms. The per-point medians in an earlier main-table run were measured under machine load and are not used as the timing claim; they are indicative only. The LINEMOD runs report 24.2–265.9 ms per frame forward (E8) and 33.1–42.6 ms per $\theta$ including FD (E9). Total wall-clock, end-to-end training throughput, and peak memory were not recorded, so we do not quote them.
- **Batching (E13).** A true batched multi-start Newton is only 1.30× faster than the loop at batch size 64 (59.9 versus 46.2 samples/s), 0.78× at 8, and 0.46× at 1: convergence heterogeneity and the per-instance classify loop dominate, so a GPU or custom kernel is needed for a real throughput win. Mode enumeration is the bottleneck, not the $3\times3$ solves.
- **Not run.** GPU batched root enumeration, end-to-end training, and the full LINEMOD/In-The-Dark accuracy matrix. The earlier planning assumption of 1–2 GPUs over 4–6 weeks is a plan, not a measurement (`unverified`).

### 5.5 Reproduction risks and negative results

- **Root enumeration is not certified.** The 6D runs use the batched multi-start Newton of `instance.py`, not LibQPEP; completeness (A2) is assumed and the missed-root rate against an independent certified solver is not measured. Against a 1024-start reference, $n_{\rm starts}\ge64$ reproduces the mode set on the sweep and $32$ matches the median $K$, while $n_{\rm starts}=16$ misses the fragile mode in 10% of independent draws (below); the mode counts in Section 5.3 should therefore be read as lower bounds on the true count, not certified counts.
- **"Certified" gap is numerical.** The SDP certificate is PSD only within SCS tolerance ($\lambda_{\min}(Z)=-1.79\times10^{-11}$); it is not a symbolic certificate. The $13\times13$ comparison of Section 5.3.4 is likewise a numerical result, and it confirms that the $13\times13$ relaxation is strictly looser in general.
- **No baseline has been run.** SDPR-IS, SDPR-SDP, EPro-PnP, BPnP, and Theseus are absent from every table; no accuracy, variance, or timing comparison against them exists. This is the largest gap in the experimental section.
- **The MC baseline is mode-level.** It samples the discrete mode weights ($n=200$, 20 seeds) and does not implement the $SE(3)$ AMIS sampler of EPro-PnP; it measures discrete-mode sampling variance only, and its numbers must not be read as an EPro-PnP comparison.
- **Laplace bias near bifurcations.** The 1D slope ($1.07$), the 1D fold blow-up (relative error $3.24$ at $a=0.01$), the genuine 6D fold at $|\theta|\approx1.4\times10^{-3}$ (Section 5.5), and the real-data near-tie bias (Laplace median $58.3^\circ$ versus hard $2.06^\circ$, Section 5.5) bound the regime where the Laplace sum is reliable. The uniform (Airy) expansion is not implemented.
- **In-The-Dark not attempted.** Its download, run split, and the `pose_sdp_block.py`-at-HEAD issue are untested; no claim is made about the pipeline.

**Fold at $|\theta|\approx1.4\times10^{-3}$.** The 6D synthetic map has a genuine fold just off the crossing. A secondary minimum exists only for $|\theta|\lesssim1.4\times10^{-3}$; as it approaches the fold its $3\times3$ Hessian determinant falls through $3.35\times10^{-1}$ while its Laplace weight rises from $0.09$ to $0.187$ ($0.122$ at that determinant), and then the mode disappears ($K$ falls from 4 to 3). Across the fold the pose map steps by $0.31$ over one $2\times10^{-5}$ step (constant under grid refinement, i.e. a genuine discontinuity), and $\Vert d\phi\Vert$ peaks at $3529$ versus a median of $379$ away from the fold; the effective FD slope reaches $1.5\times10^4$. Neither a smaller $\tau$ nor more starts removes it: $n_{\rm starts}\ge32$ matches the 1024-start median $K$ ($64$ starts reproduce the full mode set on the sweep), while $n_{\rm starts}=16$ misses the fragile mode in 10% of independent draws and a miss moves the pose by up to 28% and the gradient by up to 97%. The smoothness and $O(\tau)$ accuracy statements are therefore restricted to $\theta=0$ and $|\theta|\lesssim1.3\times10^{-3}$; the multi-modal Laplace map is discontinuous where a minimum is born or dies. Evidence: `independent reproduction (this repo)`; sources `results/robustness.md` (section a2) and the robustness round logged in `experiments/rounds_2026-11-27.md`.

**A cheap partial fix for the fold.** Replacing $\det H$ by $\det(H+\epsilon I)$ with $\epsilon=\tau^{1/3}$ (the 1D rule of Section 5.3.3, whose derivative is still closed form, $\mathrm{tr}((H+\epsilon I)^{-1}\dot H)$) largely removes the *derivative* singularity: at $\theta=1.30\times10^{-3}$ the standard $\Vert d\phi\Vert$ is $3529$ while the regularized one is $443$, and the near-fold gradient error falls from $2.90\times10^{-1}$ to $1.19\times10^{-2}$ (a factor 24); away from the fold ($\theta=0.1,0.02$) the two variants agree to the printed precision. It does **not** remove the *value* discontinuity caused by a minimum dying: the pose step across the fold only falls from $0.309$ to $0.107$ and $K$ still changes from 4 to 3. Hessian regularization is therefore a defensible cheap mitigation for the fold neighbourhood, but a continuity claim across the fold still needs the uniform (Airy) expansion or a local numerical integral over the merging pair. Evidence: `independent reproduction (this repo)`; source `results/fold_6d.md` and `code/run_fold_6d.py`.

**Real-data two-mode result (LINEMOD).** A construction that does produce a genuine two-mode cost on real data is implemented in `real/linemod/run_two_mode.py`. Part A uses real depth pixels inside the BOP `mask_visib`, back-projects them with the real intrinsics to bearings, assigns each a nearest real mesh vertex (KD-tree) as the 3-D model point, and uses deterministic RANSAC (2.5 mm inlier band) to select the dominant planar face. A planar target has the classical two-fold pose ambiguity, so the bearing cost has two genuine minima; modes whose transformed points fall behind the camera ($z\le0$) are rejected. The settings are 15 frames (8 `eggbox`, 7 `glue`), $n_{\rm points}=900$ per observation, $n_{\rm starts}=384$, $\tau=0.01$, FD $h=10^{-4}$, on login-node CPU.

**Table 5.11. Real two-mode bearing cost on the dominant planar face (Part A).** All 15 frames have $K_{\rm phys}=2$ physical minima ($K_{\rm all}=4$ before the $z>0$ filter). `w_wrong` is the Laplace weight of the flip mode; rotation errors are against ground truth in degrees.

| frame/img | obj | $n_{\rm plane}$ | $\Delta J$ | flip (deg) | $w_{\rm wrong}$ | hard rot err (deg) | Laplace rot err (deg) |
|-|-|-|-|-|-|-|-|
| 000010/15 | 10 | 558 | 9.58e-04 | 69.3 | 0.483 | 1.47 | 46.56 |
| 000010/18 | 10 | 474 | 1.25e-03 | 91.9 | 0.472 | 0.51 | 59.17 |
| 000010/40 | 10 | 482 | 9.71e-04 | 69.1 | 0.482 | 2.20 | 46.41 |
| 000010/60 | 10 | 452 | 1.69e-03 | 104.0 | 0.459 | 0.96 | 64.56 |
| 000010/66 | 10 | 460 | 7.00e-04 | 88.4 | 0.488 | 2.29 | 58.28 |
| 000010/72 | 10 | 501 | 1.25e-04 | 3.8 | 0.504 | 44.46 | 31.02 |
| 000010/80 | 10 | 508 | 5.11e-04 | 54.7 | 0.501 | 1.08 | 37.98 |
| 000010/90 | 10 | 474 | 4.13e-04 | 54.8 | 0.497 | 3.49 | 37.96 |
| 000011/20 | 11 | 413 | 2.07e-03 | 69.1 | 0.608 | 8.14 | 52.80 |
| 000011/24 | 11 | 423 | 1.65e-03 | 131.1 | 0.457 | 1.37 | 75.98 |
| 000011/26 | 11 | 439 | 1.90e-03 | 126.9 | 0.450 | 2.06 | 73.75 |
| 000011/32 | 11 | 474 | 2.10e-03 | 123.9 | 0.451 | 1.78 | 72.70 |
| 000011/33 | 11 | 365 | 1.49e-03 | 149.6 | 0.460 | 1.62 | 81.79 |
| 000011/40 | 11 | 428 | 2.56e-03 | 106.0 | 0.457 | 2.77 | 65.39 |
| 000011/49 | 11 | 347 | 2.10e-03 | 83.7 | 0.528 | 4.95 | 58.11 |

*Evidence level: `independent reproduction (this repo)`; source `real/linemod/results/two_mode.md` (Part A).*

Three points. First, the two-mode structure is real and reproducible: $K_{\rm phys}=2$ in 15/15 frames, with $\Delta J\in[1.25\times10^{-4},2.56\times10^{-3}]$, flip angles $3.8^\circ$–$149.6^\circ$, and $w_{\rm wrong}=0.450$–$0.608$. Second, the true pose is the global minimum in 14/15 frames and the hard argmin is accurate there: median rotation error $2.06^\circ$, with one failure above $10^\circ$ (`000010/72`, $44.5^\circ$). Third — the honest reading — the Laplace posterior mean is **not** an accuracy win on these near-ties: its median rotation error is $58.3^\circ$ because it puts roughly half the mass on the flip mode. On real near-degeneracies the Laplace layer trades accuracy for smoothness; this experiment does not support an accuracy claim for it. The smallest Hessian determinant in the set is $6.09\times10^{-3}$ (`000011/20`), i.e. the flip mode is itself close to a fold.

Part B removes the shared-translation failure of the old construction: each of two labelled hypotheses eliminates its **own** translation, and the costs are mixed as $J_\theta=(1-\theta)J_{\rm true}+\theta J_{\rm sym}$. On the real frame `000010/15` the measured tie is at $\theta^\star\approx0.475$ (the construction is designed to tie at $0.5$, but resampling makes the two labelled costs not exactly symmetric); $\Delta J$ changes sign between $\theta=0.46$ and $0.479$, so the crossing is transverse, and the two minima stay within $\sim2^\circ$ of their labelled poses.

**Table 5.12. Controlled transverse crossing on real geometry (Part B, excerpt).** Full 11-row table in `real/linemod/results/two_mode.md`. `grad rel` is the analytic Laplace gradient against central FD of the layer output; MC is mode-level with 400 samples.

| $\theta$ | $\Delta J$ | $w(A)$ | Laplace grad rel | MC grad rel | $\Vert d\phi\Vert$ |
|-|-|-|-|-|-|
| 0.3500 | +1.64e-05 | 0.50031 | 2.59e-10 | 9.51e-03 | 1.129e-02 |
| 0.4600 | +2.20e-06 | 0.50004 | 8.61e-10 | 8.94e-05 | 1.123e-02 |
| 0.4790 | -2.22e-07 | 0.50000 | 7.82e-10 | 1.68e-01 | 1.122e-02 |
| 0.5000 | -2.89e-06 | 0.49995 | 1.76e-09 | 5.09e-02 | 1.121e-02 |
| 0.6500 | -2.17e-05 | 0.49960 | 4.03e-10 | 1.10e-02 | 1.112e-02 |

*Evidence level: `independent reproduction (this repo)`; source `real/linemod/results/two_mode.md` (Part B).*

Across the 11 rows the Laplace gradient agrees with central FD to $1.9\times10^{-10}$–$1.8\times10^{-9}$, while the 400-sample mode-level MC error is $8.9\times10^{-5}$–$1.7\times10^{-1}$; the hard argmin has a jump at the tie and its FD is undefined there, so no hard gradient is reported. This is a correctness result for the closed-form backward pass on a real cost, not an accuracy claim: no front-end is used and no ADD(-S)/ADD-S/AUC is computed. Part B also exercises the fixed `dlogdetH_dtheta` path (Section 5.3.5).

**The earlier negative attempt (retained and diagnosed).** `run_symmetry.py` paired the same image bearings with model points $P$ and with the symmetry-transformed points $SP$, mixed the two full bearing-cost matrices, and eliminated one shared translation. It produced 0 usable rows in 9 candidate frames (`symmetry.json` is `[]`, `symmetry.md` a header). Two causes were diagnosed. (B1) `sym_of` used the `models_info` symmetry translation in millimetres as metres, offsetting the second point cloud by about $0.69$ m; after correction $J_B$ at its own pose is $4\times10^{-14}$ rather than a spurious value. (B2) Even after (B1), summing two full costs and eliminating a single shared $t$ makes the hypotheses compete for one translation: the minima collapse to a compromise pose ($t\approx-0.1$ versus the true $t_z\approx0.887$), so the cost is unimodal at the crossing. This is a construction failure, not evidence about symmetric objects; the new construction above is the positive result.

*Evidence level: `independent reproduction (this repo)`; sources `real/linemod/results/two_mode.md` (Part A/B), `real/linemod/README.md` (B1/B2 diagnosis), `real/linemod/results/symmetry.json` (old negative).*

---

## 6. Discussion and Limitations

1. **Scope of the SDP equivalence.** The $10\times10$ Veronese result is for the rotation subproblem after exact translation elimination. The released SDPRLayers solves a $13\times13$ SDP with translation and 24 constraints. The numerical check of Section 5.3.4 confirms the gap in the negative direction: the $13\times13$ constraint set spans 17 dimensions versus 20 and is missing the three row-orthogonality constraints, so it is strictly looser in general ($f^\star_{13}\le f^\star_{10}$, a constructed gap of $2.00$, and $8/8$ strict gaps after symmetry-breaking perturbations). The Choi–Lam equality is a symmetry coincidence. We therefore do not claim the two relaxation values coincide; the supplement's statement that they do is contradicted by these measured counterexamples, and the proof revision is pending. Q2 remains the most important open item.
2. **Realizability of the Choi–Lam instance.** The constructed $M$ is a valid symmetric cost matrix, but we have not shown that it arises from actual PnP or matrix-weighted correspondences. The positive-measure claim for *physical* cost matrices is open (Q1). Until then, Example 4.5 is a statement about the rotation subproblem, not about SDPRLayers' production behavior.
3. **Root enumeration.** The forward pass is only as global as the solver. Missed roots, repeated roots, and numerical elimination failures break both the "global" claim and the Laplace sum. We treat completeness as an assumption and verify residuals, but a certified root count (Sturm/homotopy) is future work (Q3).
4. **Laplace bias and bifurcations.** The $O(\tau)$ error is non-uniform. At a saddle-node, the mode count changes and the Laplace sum is not even continuous; uniform Airy asymptotics or a local numerical integral is required (Q5). We claim smoothness only for the exact smoothed posterior. Section 5 measures both faces: at $\theta=0$ the Laplace map is smooth (its jump decays linearly with FD slope $\approx502$ while the hard map's jump stays $2.86$), but a genuine fold sits at $|\theta|\approx1.4\times10^{-3}$, where a secondary minimum dies, its weight reaches $0.122$, $\Vert d\phi\Vert$ peaks at $3529$, and the pose map steps by $0.31$. Smaller $\tau$ or more starts do not remove it. The smoothness claim therefore holds at $\theta=0$ and inside $|\theta|\lesssim1.3\times10^{-3}$, but not across the fold; the uniform expansion is not implemented. A $\det(H+\tau^{1/3}I)$ regularizer reduces the derivative spike by a factor 24 and the near-fold gradient error from $2.9\times10^{-1}$ to $1.2\times10^{-2}$, but only halves the value step ($0.309\to0.107$), so it is a mitigation, not a fix.
5. **Symmetric objects: a positive construction and its accuracy cost.** The earlier symmetric-object construction failed for diagnosed reasons, but the new real-data construction does produce two genuine modes on 15/15 LINEMOD `eggbox`/`glue` frames (Section 5.5). The measured outcome is not an accuracy win: the hard argmin is accurate at the near-ties (median $2.06^\circ$) while the Laplace mean is pulled toward the flip (median $58.3^\circ$). This is the accuracy/smoothness trade the layer makes, and the behavior on symmetric objects is now measured rather than untested.
6. **Third derivatives.** $\nabla_\theta\log\det H_k$ requires third $q$-derivatives of $J$ and the implicit derivative of $q^{(k)}$. This is tractable for a quartic but non-trivial to implement correctly; it is the main gap between the "3×3 backward" slogan and the full Laplace KL (Q6).
7. **Temperature.** $\tau$ is a cost-scale-dependent hyperparameter. Without scale normalization it is not identifiable; with it, $\tau$ can be learned. The $\tau\to0$ limit recovers the deterministic gradient only for the posterior mean/argmin, not for the raw KL loss.
8. **No inequality constraints, no outlier robustness.** These are delegated to the companion tracks. The present layer assumes $G\succ0$ and a fixed weighted Gram matrix; it does not handle non-penetration, handedness inequalities, or gross outliers.
9. **GPU batching.** The algebraic root enumeration is the engineering bottleneck. A batched $40\times40$ complex eigensolver is plausible but not yet benchmarked (Q10).

---

## 7. Conclusion

We have presented `Laplace-QPEPLayer`, a differentiable pose layer that combines the quaternion algebra of QPEP with a $3\times3$ Riemannian implicit differentiation and a multi-modal Laplace posterior. The theoretical core is an exact orthogonal bridge between the rotation lift and the Veronese embedding, a proof that the 21 standard $SO(3)$ constraints span the degree-two Veronese ideal, a certified Choi–Lam gap instance for the rotation subproblem, and a closed-form backward pass with antipodal invariance. The probabilistic core is an exact translation integral and an analytic Laplace sum over all enumerated rotation minima, with explicit error order and failure modes. The design is deliberately narrow and its claims are scoped: no SDP lift at inference, no Monte Carlo sampling, no inequality constraints, no outlier kernel. Section 5 reports the experiments that are complete: the synthetic 6D branch-crossing benchmark (with an FD-step study, a measured fold, and a timing/batching study), the 1D exact-posterior checks, the certificate and Jacobian verification, the numerical ten-vs-thirteen SDP comparison, and a real-data LINEMOD two-mode study. The real-data result is a trade, not an accuracy win: the hard argmin is accurate at the near-ties while the Laplace mean is biased toward the flip. What remains is the real-image accuracy step: a learned front-end and the ADD(-S)/ADD-S benchmark on LINEMOD/LINEMOD-Occlusion, the UTIAS In-The-Dark pipeline, the SDPR and EPro-PnP comparisons, and a batched (GPU) forward; every reported number is tagged with its evidence level.

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
