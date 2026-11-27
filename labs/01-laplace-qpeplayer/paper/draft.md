# Laplace-QPEPLayer: Zero-Lift Differentiable Pose Estimation via Quaternion Algebra and Multi-Modal Laplace Smoothing

> **Draft status (2026-11-27).** This is a complete technical draft with experiment *design* but no results. Every numerical claim about our method is marked **TODO** or traced to the reproducible checks in `verify/`; preliminary numbers from an earlier internal formulation are not reported as results. References are real; entries we could not verify are omitted rather than guessed.

---

## Abstract

Differentiable pose estimation layers sit between learned front-ends and geometric back-ends. Deterministic layers built on semidefinite programming (SDP) relaxations provide global optimality when the relaxation is tight, but they lift the problem to a matrix variable, rely on solver infrastructure, and break down when two global branches cross. Probabilistic layers built on Boltzmann distributions avoid hard branch selection, but typically rely on Monte Carlo sampling in a six-dimensional pose space. We show that the quaternion formulation of quadratic pose estimation problems (QPEP) supplies the missing algebraic structure for a third option. First, for the rotation subproblem obtained after exact elimination of translation, we prove that the standard $10$-dimensional rotation lift is *exactly* $2Uv_2(q)$ for an orthogonal $U$, that the 21 $SO(3)$ constraints span precisely the degree-two vanishing ideal of the Veronese variety $\nu_2(\mathbb{RP}^3)$, and that the resulting $10\times10$ SDP is the Veronese (Shor) relaxation of a quartic on $\mathbb{S}^3$. A constructed Choi–Lam instance exhibits a certified relaxation gap of $0.0342$ with $\mathrm{rank}(X^\star)=6$. Second, we derive a closed-form Euler multiplier $\lambda^\star=4J$ and a $3\times3$ Riemannian implicit differentiation on $\mathbb{S}^3$, including exact antipodal invariance. Third, we replace the hard argmin by a multi-modal Laplace posterior: translation is integrated out exactly, and the rotation integral is approximated analytically over *all* QPEP minima with weights $w_k\propto e^{-J_k/\tau}/\sqrt{\det H_k}$. The resulting layer has no Monte Carlo variance, no SDP lift at inference, and a deterministic backward pass that reduces to one $3\times3$ solve per active mode (plus a trace term for the smoothed KL). We analyze the $O(\tau)$ error of the Laplace approximation, its breakdown at branch merging, and the $1/\tau$ scaling of the KL gradient. Experiments on LINEMOD/LINEMOD-Occlusion and UTIAS In-The-Dark are specified but not yet run.

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

## 5. Experiments (Design; Results TODO)

> **No numerical results are reported in this draft.** The following protocol is frozen before running experiments to avoid post-hoc metric selection. Every table entry will be marked "paper claim", "independent reproduction", or "unverified" in the accompanying repository, following the project's evidence convention.

### 5.1 Tasks and datasets

1. **Symmetric / ambiguous 6-DoF object pose (LINEMOD and LINEMOD-Occlusion).** Focus on `eggbox` and `glue` (symmetric), plus near-planar frames. This is where the deterministic argmin is most likely to cross branches. Baselines: SDPRLayers (SDPR-IS and SDPR-SDP), EPro-PnP, BPnP.
2. **Matrix-weighted stereo localization (UTIAS In-The-Dark).** Reuse the SDPRLayers pipeline: VGG16 encoder + two UNet decoders, dense matching, matrix weights, 13-dimensional QCQP for the baseline. Our layer replaces the pose block with the QPEP + Laplace block on the same weights. Baselines: SDPR-IS, SDPR-SDP, Theseus (local), SVD.
3. **Controlled branch-crossing benchmark (synthetic).** A two-mode instance with a known flip parameter $\theta\in[-\epsilon,\epsilon]$, known ground-truth pose on each branch, and a known $\Delta J(\theta)$. This isolates the derivative behavior at $\Delta J\to0$ and is the only setting where the exact posterior can be computed by dense quadrature for a bias/variance decomposition.

### 5.2 Metrics

- **Accuracy:** ADD(-S), ADD-S AUC, rotation error (deg), translation error (m).
- **Global correctness:** fraction of samples where the returned pose is within tolerance of the ground truth; fraction of SDPR samples with non-tight relaxation (rank $>1$), reported separately.
- **Differentiability:** gradient error against central finite differences on the synthetic benchmark; gradient variance across seeds (EPro-PnP) vs deterministic spread (ours); cosine similarity between the analytic gradient and the finite-difference gradient near $\Delta J=0$.
- **Efficiency:** forward and backward wall-clock per sample, batch throughput, peak memory; CPU vs GPU; number of modes $K$ and its distribution.
- **Calibration:** negative log-likelihood of the ground-truth pose under the Laplace posterior; reliability of $w_k$ against a dense-grid posterior on the synthetic benchmark.

### 5.3 Ablations

1. **Smoothing:** deterministic argmin vs Laplace with fixed $\tau$ vs learned $\tau$; uniform weights vs $1/\sqrt{\det H_k}$ weights.
2. **Modes:** top-1 only vs all enumerated minima; effect of dropping modes with $w_k<10^{-3}$.
3. **Backward:** analytic $3\times3$ vs autograd through the root solver (to quantify the "trap B" cost and instability).
4. **Hessian:** $H_{\mathbb{S}^3}$ vs Euclidean $\nabla^2J$ (to show the $-4J$ term matters); rotation-vector vs $E$-basis scaling.
5. **Root enumeration:** LibQPEP vs multi-start Riemannian Newton; residual verification; missed-root rate.
6. **SDPR coupling:** SDPR with and without redundant constraints; non-tight samples excluded vs reported.

### 5.4 Compute budget (planning assumption, not a claim)

- 1–2 GPUs of the class available to the group (e.g., RTX 4090/A100), 4–6 weeks wall-clock for the full matrix.
- LibQPEP is CPU/C++; for batched training we will first use a CPU worker pool with a GPU front-end, then evaluate a batched action-matrix eigensolver on GPU.
- The In-The-Dark pipeline is the heaviest item (dense matching over 768×196,608 descriptors); we follow the SDPRLayers configuration and batch size 1 unless memory allows more.

### 5.5 Reproduction risks

- **Mosek license** for SDPR baselines; SCS is a fallback but changes timing and tightness behavior.
- **LibQPEP build** (Eigen/LAPACK/CMake); the official solver returns up to 40 candidates; missed roots must be detected by residual checks.
- **In-The-Dark download** and the exact train/val/test run split; the SDPRLayers repository's `pose_sdp_block.py` is missing at HEAD, so the baseline must be pinned to a known commit or replaced by `SDPPoseEstimator`.
- **EPro-PnP sampling** variance and its AMIS hyperparameters; comparisons must fix the sample budget and report variance across seeds.
- **Our Laplace bias** near bifurcations; the synthetic benchmark must separate bias from variance, otherwise the comparison is unfair to EPro-PnP.

---

## 6. Discussion and Limitations

1. **Scope of the SDP equivalence.** The $10\times10$ Veronese result is for the rotation subproblem after exact translation elimination. The released SDPRLayers solves a $13\times13$ SDP with translation and 24 constraints. We do not claim the two relaxation values coincide. This is the most important open item (Q2).
2. **Realizability of the Choi–Lam instance.** The constructed $M$ is a valid symmetric cost matrix, but we have not shown that it arises from actual PnP or matrix-weighted correspondences. The positive-measure claim for *physical* cost matrices is open (Q1). Until then, Example 4.5 is a statement about the rotation subproblem, not about SDPRLayers' production behavior.
3. **Root enumeration.** The forward pass is only as global as the solver. Missed roots, repeated roots, and numerical elimination failures break both the "global" claim and the Laplace sum. We treat completeness as an assumption and verify residuals, but a certified root count (Sturm/homotopy) is future work (Q3).
4. **Laplace bias and bifurcations.** The $O(\tau)$ error is non-uniform. At a saddle-node, the mode count changes and the Laplace sum is not even continuous; uniform Airy asymptotics or a local numerical integral is required (Q5). We claim smoothness only for the exact smoothed posterior.
5. **Third derivatives.** $\nabla_\theta\log\det H_k$ requires third $q$-derivatives of $J$ and the implicit derivative of $q^{(k)}$. This is tractable for a quartic but non-trivial to implement correctly; it is the main gap between the "3×3 backward" slogan and the full Laplace KL (Q6).
6. **Temperature.** $\tau$ is a cost-scale-dependent hyperparameter. Without scale normalization it is not identifiable; with it, $\tau$ can be learned. The $\tau\to0$ limit recovers the deterministic gradient only for the posterior mean/argmin, not for the raw KL loss.
7. **No inequality constraints, no outlier robustness.** These are delegated to the companion tracks. The present layer assumes $G\succ0$ and a fixed weighted Gram matrix; it does not handle non-penetration, handedness inequalities, or gross outliers.
8. **GPU batching.** The algebraic root enumeration is the engineering bottleneck. A batched $40\times40$ complex eigensolver is plausible but not yet benchmarked (Q10).

---

## 7. Conclusion

We have presented `Laplace-QPEPLayer`, a differentiable pose layer that combines the quaternion algebra of QPEP with a $3\times3$ Riemannian implicit differentiation and a multi-modal Laplace posterior. The theoretical core is an exact orthogonal bridge between the rotation lift and the Veronese embedding, a proof that the 21 standard $SO(3)$ constraints span the degree-two Veronese ideal, a certified Choi–Lam gap instance for the rotation subproblem, and a closed-form backward pass with antipodal invariance. The probabilistic core is an exact translation integral and an analytic Laplace sum over all enumerated rotation minima, with explicit error order and failure modes. The design is deliberately narrow and its claims are scoped: no SDP lift at inference, no Monte Carlo sampling, no inequality constraints, no outlier kernel. The next step is empirical: build the batched forward, run the synthetic branch-crossing benchmark, then the LINEMOD and In-The-Dark pipelines, and report every number with its evidence level.

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
