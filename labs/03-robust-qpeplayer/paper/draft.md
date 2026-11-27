# Robust-QPEPLayer: Curvature-Corrected Implicit Differentiation for RANSAC-Free Robust Pose Estimation

**Draft v0.2 — measured subsets (2026-11-27). Synthetic 3D-3D, 6D gradients, ModelNet40/10, two 3DMatch scenes, KITTI drive 0001, systems, and ablations are independently measured in `experiments/`; SDPRLayers/Theseus/BPnP/Pyatov/POLI and In-The-Dark are not measured and are marked as such.**

## Abstract

Differentiable pose-estimation layers promise end-to-end learning through geometric optimization, but their guarantees are fragile in the presence of outliers. The state of the art either wraps a non-differentiable RANSAC front-end around a certifiable solver, or differentiates a local robust M-estimator with a frozen-weight or unrolled approximation. We present **Robust-QPEPLayer**, a RANSAC-free differentiable pose layer that combines Black–Rangarajan (BR) duality, graduated non-convexity (GNC), and a constant-size quaternion pose solver (QPEP). The forward pass alternates a closed-form outlier-weight update with a globally solved weighted QPEP; the rotation dimension stays four (three on the tangent space) and the number of correspondences enters only through O(N) accumulation. The backward pass differentiates the robust fixed point exactly with a 6×6 system and O(1) memory in the number of GNC iterations. Our main theoretical results are: (i) an exact BR dual for the Geman–McClure family, with the optimal weight equal to the kernel derivative; (ii) a proof that a GNC fixed point is a stationary point of the robust cost, and that the redescending curvature correction is precisely the Schur complement of the BR weight block; (iii) the exact Riemannian Hessian on S³×R³, which contains a constraint-multiplier term −λ_rob I that is absent from the Euclidean formula and whose omission changes the spectrum by O(1); (iv) a rigorous one-dimensional sign-flip proposition showing that frozen-weight IRLS-StopGrad can have the opposite derivative sign to the exact fixed point, together with an existence construction; and (v) a per-observation O(‖r_i‖^{-4}) gradient bound under a uniform invertibility condition, which we explicitly distinguish from forward robustness. We also delineate what the method does **not** guarantee: GNC is not a global algorithm, coherent majority outliers can make the global robust optimum wrong, and a positive Hessian is a local stability certificate, not a robustness certificate. We report measured synthetic and real 3D-3D results: GNC-GM matches RANSAC-1000 success at 0–70% random outliers and is 22–32× faster on synthetic, ModelNet, 3DMatch, and KITTI data; exact IFT matches finite differences at 3–5×10⁻⁸ relative error, while frozen-weight IRLS has a boundary cosine of 0.234 and a 1.07 relative error. The protocol against SDPRLayers+RANSAC, Theseus, BPnP, Pyatov et al., and POLI, and the In-The-Dark end-to-end experiment, remain future work.

---

## 1. Introduction

Optimization layers are now a standard component of differentiable robotics and vision pipelines: a geometric problem is solved in the forward pass, and implicit differentiation of its optimality conditions provides gradients to an upstream network [1, 6, 7]. For pose estimation, the central difficulty is that the underlying problem is non-convex. Local solvers such as Gauss–Newton or Levenberg–Marquardt converge to *a* stationary point, and the gradient of that stationary point is generally not the gradient of the global pose. SDPRLayers [1] addresses this by solving a convex semidefinite relaxation, certifying rank-one optimality, and differentiating the original non-convex KKT system at the certified solution. QPEP [2] takes a different route: it eliminates translation, enumerates all stationary points of a quartic polynomial system with a Gröbner basis, and returns the algebraically global optimum without requiring a tight convex relaxation. Both are global in the forward pass, but neither is robust to outliers: SDPRLayers' real-data experiments wrap the layer in RANSAC and train on clean samples, and a robust SDP with per-correspondence binary variables lifts a 10×10 matrix to (10+9N)×(10+9N) [1].

This paper asks a narrow question: **can we obtain a differentiable pose layer that is robust to gross outliers, has constant dimension in the number of correspondences, and differentiates the robust estimator exactly rather than unrolling it or freezing its weights?** We answer with a layer built from three ingredients:

- **Black–Rangarajan duality** [3], which turns a redescending M-estimator into a joint optimization over the pose and one weight per correspondence;
- **graduated non-convexity** [4, 5], which anneals a convex surrogate towards the redescending kernel;
- **QPEP** [2], which solves each weighted subproblem globally in a fixed number of variables.

The combination is not new in spirit — GNC with a weighted least-squares inner solver is standard in robust registration [4, 5] — but the *differentiable* layer exposes a mathematical structure that, to our knowledge, has not been isolated: the exact derivative of the robust fixed point contains a **redescending curvature correction** that is the Schur complement of the BR weight block, and on S³ it also contains a **constraint-multiplier term** that the natural Euclidean formula misses. Dropping either term produces gradients that can be wrong by a large factor, or even have the wrong sign for boundary outliers.

### 1.1 Contributions

1. **A constant-dimension robust forward layer with an exact dual.** We prove that the BR dual of the Geman–McClure (GM) kernel is exact for every weight and every annealing parameter, that the optimal weight is the kernel derivative, and that the GNC-QPEP iteration is a monotone block-coordinate descent whose limit points are coordinate-wise stationary. The inner weighted QPEP is globally solved; the outer GNC is explicitly *not* a global algorithm. We state the failure modes rather than hiding them.

2. **Exact fixed-point implicit differentiation with a curvature correction.** We prove that a GNC fixed point is a stationary point of the robust cost, derive the Schur-complement identity that identifies the redescending correction with the eliminated weight block, and give the exact Riemannian Hessian on S³×R³:
   $$
   H_{\mathrm{rob}}=B^\top\Big[\sum_i\big(\rho'(e_i)H_i^E+\rho''(e_i)g_i^E(g_i^E)^\top\big)\Big]B-\lambda_{\mathrm{rob}}\operatorname{diag}(I_3,0_3).
   $$
   The term $-\lambda_{\mathrm{rob}}\operatorname{diag}(I_3,0_3)$ is the constraint multiplier of $\mathbf q^\top\mathbf q=1$; it is absent from the Euclidean-Hessian formula and can shift eigenvalues by an O(1) amount. We give a closed form for $\lambda_i$ in the 3D-3D case, $\lambda_i=4\mathbf u_i^\top W_i\mathbf r_i$, which is *not* $4e_i$.

3. **A rigorous sign-flip result and an O(‖r‖^{-4}) gradient bound.** In a one-dimensional location model we prove that, at a strict local minimum with $H_{\mathrm{rob}}>0$,
   $$
   \frac{\mathrm dt^*}{\mathrm dy_j}=\frac{\lambda_\parallel(e_j)}{H_{\mathrm{rob}}},
   \qquad
   \Big(\frac{\mathrm dt^*}{\mathrm dy_j}\Big)_{\mathrm{naive}}=\frac{2\rho'(e_j)}{\sum_i2\rho'(e_i)}>0,
   $$
   and that for $e_j>c^2/3$ the two signs are strictly opposite. We prove that such a configuration exists for two observations with a sufficiently large gap. In 3D-3D we derive the exact observation sensitivity and show that its parallel eigenvalue changes sign at $e=c^2/3$, while the frozen-weight sensitivity never does. We then prove a per-observation bound $\|\partial z^*/\partial\mathbf y_i\|=O(\|\mathbf r_i\|^{-4})$ under a uniform invertibility condition, and show that the forward bias is only $O(\|\mathbf r_i\|^{-3})$; the two should not be conflated.

4. **A unified kernel framework and an implementation/experiment plan.** We give the BR duals of GM, Cauchy, Welsch, and TLS, show that the forward and backward are structurally identical across kernels, and specify a full protocol on UTIAS In-The-Dark, 3DMatch, and KITTI Odometry against SDPRLayers+RANSAC, Theseus, BPnP, Pyatov et al., and POLI. We explicitly list what would falsify the claims.

The layer is complementary to, not dependent on, the other two prototypes in this project: it does not use the $O(10)$ Veronese isomorphism or inequality constraints. It can be combined with either at a later stage (QPEP as the inner solver is the only shared component).

---

## 2. Related Work

**Certifiable and global pose layers.** SDPRLayers [1] is the closest baseline. It lifts a homogeneous QCQP to a Shor SDP, certifies rank-one optimality with a dual matrix, and differentiates the original KKT system with an implicit-selection argument that handles redundant constraints. Its guarantees are strong but conditional on tightness; its real-data pipeline uses RANSAC and clean samples, and its SDP is CPU-bound and grows with the number of variables. QPEP [2] is the other global route: it eliminates translation, solves a quartic polynomial system by Gröbner basis, and returns the algebraically global pose with a covariance. QPEP has no gradient layer. Our work uses QPEP as a *subroutine* and adds a robust outer loop and an exact backward; it does not claim certifiable global optimality of the robust cost.

**Local differentiable optimization.** Theseus [6] provides differentiable nonlinear least squares on Lie groups with implicit differentiation and direct loss minimization. It supports robust kernels as soft costs, but its fixed point is a local stationary point of the penalized objective; it does not expose the redescending curvature correction, and the kernel is typically applied as a scalar weight rather than through a BR dual. BPnP [7] backpropagates through a PnP solver via the implicit function theorem; it has no outlier mechanism. EPro-PnP [8] models pose ambiguity with a Boltzmann distribution and Monte Carlo integration; it is complementary but high-variance and not a redescending M-estimator. Pyatov et al. [9] treat robust two-view geometry as an implicit layer with learnable matching-confidence weights; their scope is the fundamental matrix and two-view geometry, while ours is metric 3D-3D pose on S³×R³ with an explicit second-order theory. POLI [10] learns per-point Gaussian geometry (a statistical manifold) and can supply anisotropic $W_i$; it is a front-end representation, not a robust pose layer, and is complementary: our $W_i$ can be POLI's ellipsoids.

**Robust estimation.** The BR duality [3] unifies outlier processes and robust statistics and gives the dual form we use. GNC was popularized for spatial perception by Yang et al. [4] and TEASER [5]; the latter combines GNC-TLS with a certified rotation subproblem and achieves extreme outlier rejection. Our use of GNC differs in two ways: the inner problem is a full 6-DoF QPEP rather than a rotation-only subproblem, and the contribution is the *differentiable* fixed point, not a standalone robust solver. The GM, Cauchy, Welsch, and TLS kernels are classical [11]; we give their BR duals in one table.

**Implicit differentiation and phantom gradients.** Jacobian-free backpropagation [14] and related fixed-point layers replace the inverse-Jacobian solve by a cheap approximation. Freezing the robust weights is a similar approximation: it is a "phantom gradient" that ignores the derivative of the weight map. Our 1D proposition gives a setting where this phantom gradient has the wrong sign, which is a stronger statement than a numerical bias.

**Novelty relative to the baselines.** The closest differentiable robust layer is Pyatov et al. [9]; the closest robust global solver is TEASER/GNC [4, 5]; the closest pose layer is SDPRLayers [1]. Our differentiators are (i) exact robust fixed-point differentiation with the curvature correction, (ii) the Riemannian constraint term on S³×R³, (iii) the 1D sign-flip theorem, (iv) the O(‖r‖^{-4}) per-observation bound, and (v) the constant-dimension QPEP inner solver. We do not claim to beat certified global solvers on global optimality; we claim a better-composed *robust differentiable layer*.

---

## 3. Problem Formulation and Notation

Let $\mathcal M=\mathbb S^3\times\mathbb R^3$ with $z=(\mathbf q,\mathbf t)$, $\mathbf q^\top\mathbf q=1$. For $i=1,\dots,N$ let $\mathbf p_i\in\mathbb R^3$ be a model point, $\mathbf y_i\in\mathbb R^3$ an observation, and $W_i\in\mathbb S^3_{++}$ a positive-definite weight. The 3D-3D residual and its squared weighted norm are
$$
\mathbf r_i(z,\theta)=R(\mathbf q)\mathbf p_i+\mathbf t-\mathbf y_i,\qquad
e_i(z,\theta)=\|\mathbf r_i\|_{W_i}^2=\mathbf r_i^\top W_i\mathbf r_i\ge0,
$$
where $\theta$ collects the parameters of the layer (correspondences, features, and optionally the weights). We write $\mathbf u_i=R(\mathbf q)\mathbf p_i$ and $J_i=\partial\mathbf u_i/\partial\mathbf q\in\mathbb R^{3\times4}$. Since $\mathbf u_i$ is homogeneous of degree two in $\mathbf q$, Euler's theorem gives $J_i\mathbf q=2\mathbf u_i$.

The robust cost is
$$
F_{\mathrm{rob}}(z;\theta)=\sum_{i=1}^N\rho(e_i(z;\theta)),
$$
with the Geman–McClure kernel
$$
\rho(e)=\frac{c^2e}{c^2+e},\qquad
\rho'(e)=\frac{c^4}{(c^2+e)^2},\qquad
\rho''(e)=-\frac{2c^4}{(c^2+e)^3}.
$$
The GNC surrogate is $\rho_\mu(e)=\mu c^2e/(\mu c^2+e)$ for $\mu\ge1$. The BR dual variable is $w_i\in[0,1]$, with dual potential
$$
\psi_\mu(w)=\mu c^2(\sqrt w-1)^2.
$$
We write $g_i$ and $H_i$ for the Riemannian gradient and Hessian of $e_i$ on $\mathcal M$, and $g_i^E,H_i^E$ for their Euclidean counterparts in $\mathbb R^7$. $P=I_4-\mathbf q\mathbf q^\top$ is the tangent projector at $\mathbf q$; $\Pi=\operatorname{diag}(P,I_3)$; $B=\operatorname{diag}(B_q,I_3)$ is an orthonormal tangent basis, with $B_q\in\mathbb R^{4\times3}$ spanning $\{\mathbf q\}^{\perp}$. The stationarity map is
$$
\Phi(z,\theta)=\sum_i\rho'(e_i(z,\theta))\,g_i(z,\theta)\in T_z\mathcal M.
$$
A GNC fixed point at $\mu=1$ satisfies $w_i=\rho'(e_i)$ and $\Phi=0$; the exact derivative is obtained from $H_{\mathrm{rob}}\, \mathrm dz+\partial_\theta\Phi\,\mathrm d\theta=0$.

**Assumptions.** (A1) $e_i\in C^2$ and all derivatives are continuous; $W_i$ lies in a compact subset of $\mathbb S^3_{++}$. (A2) The weighted QPEP is solved globally by an algebraic solver (QPEP-style enumeration of all KKT roots; numerical failure is reported, not hidden). This is an algebraic-global assumption, not a convex certificate. (A3) At the point of interest the weighted minimizer is unique and $H_{\mathrm{rob}}$ is invertible; for a strict local minimum we strengthen this to $H_{\mathrm{rob}}\succ0$. (A4) Parameters enter smoothly; the main derivations assume $W_i$ fixed, with the $W_i(\theta)$ extension noted in the appendix. (A5) $N\ge3$ and $\sum_iw_iW_i\succ0$; the 3D points are not collinear.

---

## 4. Method and Theory

### 4.1 Black–Rangarajan dual and GNC-QPEP forward

**Lemma 1 (exact BR dual for GM).** For every $\mu\ge1$, $c>0$, and $e\ge0$,
$$
\rho_\mu(e)=\min_{w\in[0,1]}\big\{we+\psi_\mu(w)\big\},
\qquad
w^*(e)=\rho_\mu'(e)=\Big(\frac{\mu c^2}{\mu c^2+e}\Big)^2.
$$
Consequently, for every $z$,
$$
\min_{w\in[0,1]^N}\sum_i\big\{w_ie_i(z)+\psi_\mu(w_i)\big\}=\sum_i\rho_\mu(e_i(z)).
$$

*Proof sketch.* The scalar objective is strictly convex on $(0,1]$; its stationary point is $w^*$, and substituting gives $\rho_\mu(e)$. The multi-observation case follows because the box constraint is a Cartesian product. A complete proof is in the appendix.

**Theorem 1 (GNC-QPEP forward).** Under (A1), (A2), (A5), fix $\mu\ge1$ and alternate
$$
w_i^{(k)}=\rho_\mu'(e_i(z^{(k)})),\qquad
z^{(k+1)}=\arg\min_{z\in\mathcal M}\sum_iw_i^{(k)}e_i(z).
$$
Then:

(i) The $w$-step is exact and the $z$-step is a weighted QPEP. Eliminating $\mathbf t$ gives
$$
\mathbf t(w,\mathbf q)=G(w)^{-1}\sum_iw_iW_i(\mathbf y_i-R(\mathbf q)\mathbf p_i),\qquad G(w)=\sum_iw_iW_i,
$$
and the reduced cost is a quartic polynomial in $\mathbf q$; the KKT system has four (or seven) variables, and $N$ enters only through $O(N)$ accumulation into fixed-size matrices.

(ii) $F_\mu(z^{(k)})$ is nonincreasing; if each weighted minimizer is unique, the iterates converge to a fixed point $(z_\mu^*,w_\mu^*)$ satisfying $w_{\mu,i}^*=\rho_\mu'(e_i(z_\mu^*))$ and $z_\mu^*\in\arg\min_z\sum_iw_{\mu,i}^*e_i(z)$.

(iii) As $\mu\downarrow1$, every accumulation point satisfies $w_i^*=\rho'(e_i(z^*))$ and $\Phi(z^*)=0$; that is, $z^*$ is a stationary point of $F_{\mathrm{rob}}$.

(iv) **No global guarantee.** The global property belongs to the inner weighted QPEP only. The outer GNC is a nonconvex block-coordinate method; it can stop at a saddle, and a coherent majority of outliers can make the global minimizer of $F_{\mathrm{rob}}$ itself wrong (Section 6). The phrase "escapes any outlier contamination" is not a theorem.

The proof of (ii) is the standard monotone block-descent argument; (iii) follows by continuity of $\rho_\mu'$ and the first-order condition of the weighted problem. (iv) is discussed in Section 6 and Proposition 4.

**Algorithm 1** summarizes the forward pass. The annealing schedule is
$$
\mu_0\ge\frac{3\max_ie_i(z^{(0)})}{c^2},\qquad
\mu_{k+1}=\max(1,\mu_k/\gamma),\quad\gamma\in[1.2,2.0],
$$
with a fixed or adaptive number of inner alternations. The choice of $\mu_0$ makes each residual's *radial* second derivative positive; it does not make the full cost convex in $z$.

### 4.2 Fixed points, curvature correction, and the exact Riemannian Hessian

**Lemma 2 (fixed point ⇒ stationarity).** If $z^*$ globally minimizes $\sum_iw_i^*e_i$ with $w_i^*=\rho'(e_i(z^*))$, then $(z^*,w^*)$ is a GNC fixed point and $\Phi(z^*)=0$. Conversely, $\Phi(z^*)=0$ plus the same weighted-global-minimum property gives a fixed point. Stationarity alone is not enough for the converse.

**Lemma 3 (curvature correction = Schur complement).** At a GNC fixed point, let
$$
H_{\mathrm{IRLS}}=\sum_i\rho'(e_i^*)H_i,\qquad
\Delta H_{\mathrm{redesc}}=\sum_i\rho''(e_i^*)\,g_ig_i^\top\preceq0,
$$
and let $D=[g_1,\dots,g_N]$, $\Psi''=\operatorname{diag}(\psi''(w_i^*))$. The Hessian of the joint BR objective is
$$
H_J=\begin{bmatrix}H_{\mathrm{IRLS}}&D\\D^\top&\Psi''\end{bmatrix},
$$
and its Schur complement onto the pose block is
$$
H_{\mathrm{rob}}=H_{\mathrm{IRLS}}-D\Psi''^{-1}D^\top=H_{\mathrm{IRLS}}+\Delta H_{\mathrm{redesc}}.
$$
Since $\psi''(w_i^*)=-1/\rho''(e_i^*)>0$, the redescending correction is exactly the eliminated weight block. In particular, if $H_{\mathrm{IRLS}}\succeq0$ (which holds at a global weighted minimum), then $H_J\succ0\iff H_{\mathrm{rob}}\succ0$. Freezing the weights is not a small approximation: it deletes the Schur complement.

**Lemma 4 (exact Riemannian Hessian on S³×R³).** Let $f\in C^2(\mathcal M)$, with Euclidean gradient $G$ and Hessian $H_E$. Then
$$
\operatorname{Hess}f=\Pi\big(H_E-(\mathbf q^\top G_q)\operatorname{diag}(I_4,0_3)\big)\Pi,
$$
and in an orthonormal tangent basis $B$,
$$
H_f=B^\top H_E B-(\mathbf q^\top G_q)\operatorname{diag}(I_3,0_3).
$$
For $F_{\mathrm{rob}}$,
$$
\boxed{
H_{\mathrm{rob}}=B^\top\Big[\sum_i\big(\rho'(e_i)H_i^E+\rho''(e_i)g_i^E(g_i^E)^\top\big)\Big]B
-\lambda_{\mathrm{rob}}\operatorname{diag}(I_3,0_3),
\quad
\lambda_{\mathrm{rob}}=\mathbf q^\top G_q=\sum_i\rho'(e_i)\lambda_i.
}
$$
If $H_i$ and $g_i$ denote Riemannian quantities, the equivalent form is
$$
H_{\mathrm{rob}}=\sum_i\rho'(e_i)H_i+\sum_i\rho''(e_i)g_ig_i^\top.
$$
The term $-\lambda_{\mathrm{rob}}\operatorname{diag}(I_3,0_3)$ is the multiplier of the unit-quaternion constraint. It cannot be absorbed into a rescaling: for 3D-3D,
$$
\lambda_i=\mathbf q^\top g_{i,q}^E=4\mathbf u_i^\top W_i\mathbf r_i=4\big(e_i-(\mathbf t-\mathbf y_i)^\top W_i\mathbf r_i\big),
$$
which differs from $4e_i$ whenever $(\mathbf t-\mathbf y_i)^\top W_i\mathbf r_i\neq0$.

*Why the $\lambda$ term matters.* On the sphere, $f(\mathbf q)=\frac12\|\mathbf q\|^2$ is constant; its Riemannian Hessian is zero. The Euclidean formula without the multiplier gives $I_3$ in tangent coordinates. In a six-dimensional 3D-3D example with $c=1.0$, the uncorrected formula differs from a geodesic finite-difference Hessian by 25.9% in relative Frobenius norm, while the corrected formula agrees to $9.9\times10^{-8}$; the multiplier is $\lambda_{\mathrm{rob}}=1.033$. This is not a cosmetic correction.

**Lemma 5 (3D-3D weights).** With the notation above,
$$
g_{i,q}^E=2J_i^\top W_i\mathbf r_i,\qquad g_{i,t}^E=2W_i\mathbf r_i,
$$
and $\lambda_i=4\mathbf u_i^\top W_i\mathbf r_i$. The total multiplier is $\lambda_{\mathrm{rob}}=4\sum_i\rho'(e_i)\mathbf u_i^\top W_i\mathbf r_i$, not $4\sum_i\rho'(e_i)e_i$.

### 4.3 Exact implicit differentiation

**Theorem 2 (exact fixed-point IFT).** Under (A1)–(A4), suppose $\Phi(z^*,\theta_0)=0$ and $H_{\mathrm{rob}}(z^*,\theta_0)$ is invertible. Then there is a neighborhood $U$ of $\theta_0$ and a unique $C^1$ branch $\bar z:U\to\mathcal M$ with $\bar z(\theta_0)=z^*$ and $\Phi(\bar z(\theta),\theta)=0$, and
$$
\frac{\mathrm d\bar z}{\mathrm d\theta}=-H_{\mathrm{rob}}^{-1}\frac{\partial\Phi}{\partial\theta}.
$$
If $H_{\mathrm{rob}}\succ0$, the branch is a strict local minimum and remains so for small $\theta$ perturbations; if $H_{\mathrm{rob}}$ has a negative eigenvalue, $z^*$ is a saddle; if $H_{\mathrm{rob}}$ is singular, the implicit-function theorem fails and the stationary branch can bifurcate. The derivative is the derivative of *the selected stationary branch*, not of the global argmin and not necessarily of the GNC output if the algorithm switches branches. For IFT, invertibility is the condition; $\lambda_{\min}>0$ is the stronger condition for a strict local minimum and for persistence of the branch.

**Theorem 3 (right-hand side decomposition).** For a parameter $\theta$,
$$
\frac{\partial\Phi}{\partial\theta}
=\underbrace{\sum_i\rho'(e_i)\frac{\partial g_i}{\partial\theta}}_{\text{frozen-weight term}}
+\underbrace{\sum_i\rho''(e_i)\,g_i\Big(\frac{\partial e_i}{\partial\theta}\Big)^\top}_{\text{implicit weight sensitivity}}.
$$
For $\theta=\mathbf y_i$ in 3D-3D,
$$
\frac{\partial\Phi_t}{\partial\mathbf y_i}=-2\big(\rho'(e_i)W_i+2\rho''(e_i)W_i\mathbf r_i\mathbf r_i^\top W_i\big),
$$
$$
\frac{\partial\Phi_q}{\partial\mathbf y_i}=-2J_i^\top W_i\big(\rho'(e_i)I_3+2\rho''(e_i)\mathbf r_i\mathbf r_i^\top W_i\big).
$$
The frozen-weight approximation keeps only the first term and sets $\partial\Phi_t^{\mathrm{naive}}/\partial\mathbf y_i=-2\rho'(e_i)W_i$.

### 4.4 When the frozen-weight gradient has the wrong sign

With $W_i=I_3$, the parallel eigenvalue of $\partial\Phi_t/\partial\mathbf y_i$ is $-\lambda_\parallel(e_i)$, where
$$
\lambda_\parallel(e)=2(\rho'(e)+2e\rho''(e))=\frac{2c^4(c^2-3e)}{(c^2+e)^3}.
$$
Thus $\lambda_\parallel(0)=2$, $\lambda_\parallel(c^2/3)=0$, $\lambda_\parallel(c^2)=-1/2$, and $\lambda_\parallel(e)\sim-6c^4/e^2$ as $e\to\infty$. The exact and naive parallel eigenvalues differ by $-4e\rho''=8c^4e/(c^2+e)^3>0$; for $e>c^2/3$ they have strictly opposite signs. The most toxic residuals are the *boundary* ones near $e=c^2$, not the most distant gross outliers.

**Proposition 1 (1D sign flip).** Let $F(t)=\sum_i\rho((t-y_i)^2)$ with GM. At a stationary point $t^*$ with $H_{\mathrm{rob}}=\sum_i\lambda_\parallel(e_i)>0$,
$$
\frac{\mathrm dt^*}{\mathrm dy_j}=\frac{\lambda_\parallel(e_j)}{H_{\mathrm{rob}}},
\qquad
\Big(\frac{\mathrm dt^*}{\mathrm dy_j}\Big)_{\mathrm{naive}}=\frac{2\rho'(e_j)}{\sum_i2\rho'(e_i)}>0.
$$
If $e_j>c^2/3$, then $\lambda_\parallel(e_j)<0$, so the exact derivative is strictly negative while the naive derivative is strictly positive. If $H_{\mathrm{rob}}=0$ the implicit-function theorem fails; if $H_{\mathrm{rob}}<0$ the point is not a local minimum.

**Proposition 2 (existence).** For any $c>0$ and $y_2-y_1$ sufficiently large, the two-observation cost has a strict local minimum $t^*$ with $e_1<c^2/3<e_2$ and $H_{\mathrm{rob}}>0$; hence the sign flip of Proposition 1 occurs. The proof uses the signs of $\Phi$ at $0$ and at $Kc^4/d^3$, and the limit $H_{\mathrm{rob}}\to\lambda_\parallel(0)=2>0$.

In 6D, the sign of the pose sensitivity is *not* determined by $\lambda_\parallel$ alone:
$$
\frac{\partial z^*}{\partial\mathbf y_i}=-H_{\mathrm{rob}}^{-1}\frac{\partial\Phi}{\partial\mathbf y_i}.
$$
The inverse Hessian mixes directions. Proposition 1 is the rigorous statement; a 6D numerical cosine of $-0.079$ is an observation, not a theorem. A 6D analogue requires a joint spectral condition on $H_{\mathrm{rob}}$ and $\partial\Phi/\partial\mathbf y_i$ and is left open.

**Proposition 3 (per-observation O(‖r‖^{-4}) bound).** Under (A1), (A5), assume along a path $\|\mathbf y_i\|\to\infty$ that $z^*(\mathbf y_i)$ stays in a compact set and $H_{\mathrm{rob}}$ is uniformly positive definite there with $\lambda_{\min}\ge\mu_0>0$. Then
$$
\Big\|\frac{\partial\Phi}{\partial\mathbf y_i}\Big\|_F\le C\frac{c^4}{(c^2+e_i)^2}\le C\frac{c^4}{\|\mathbf r_i\|^4},
\qquad
\Big\|\frac{\partial z^*}{\partial\mathbf y_i}\Big\|_F\le \frac{C}{\mu_0}\frac{c^4}{\|\mathbf r_i\|^4}.
$$
The forward influence is only $O(\|\mathbf r_i\|^{-3})$:
$$
\|\rho'(e_i)g_i\|\le\frac{2\overline wc^4}{\|\mathbf r_i\|^3}.
$$
The bound is per observation; it does not prevent $N$ outliers from accumulating, and it says nothing about a coherent outlier cluster that becomes the global minimum. "Gradient immunity" is not "forward robustness".

### 4.5 Algorithm, complexity, and the second-order certificate

Algorithm 1 (forward) and Algorithm 2 (backward) are stated below. The backward pass stores $z^*,w^*$ and recomputes $g_i^E,H_i^E$; it solves one 6×6 system per parameter block and never unrolls the GNC iterations. Memory is $O(N)$ for the weights and residuals, and $O(1)$ in the number of GNC iterations. The complexity is $O(N)$ per forward iteration plus a fixed-size algebraic solve, and $O(N)$ for the backward assembly plus $O(6^3)$.

```
Algorithm 1: Forward GNC-QPEP
Input: correspondences (p_i, y_i, W_i), kernel scale c, schedule gamma
Initialize q, t; mu <- 3 max_i e_i / c^2
while mu >= 1:
    repeat K times:
        e_i <- ||R(q) p_i + t - y_i||^2_{W_i}
        w_i <- (mu c^2 / (mu c^2 + e_i))^2
        (q, t) <- QPEP_global(sum_i w_i e_i)
    mu <- max(1, mu / gamma)
return q, t, w, e
```

```
Algorithm 2: Exact backward
Input: upstream gradients, cached z*, w*, e*
for i: compute u_i, r_i, J_i, g_i^E (7), H_i^E (7x7)
G <- sum_i rho'_i g_i^E
H_E <- sum_i (rho'_i H_i^E + rho''_i g_i^E g_i^{E,T})
B_q <- orthonormal basis of q*^perp; B <- blkdiag(B_q, I_3)
H_rob <- B^T H_E B - lambda_rob diag(I_3, 0_3),  lambda_rob = q*^T G_q
RHS <- sum_i rho'_i dg_i/dtheta + sum_i rho''_i g_i (de_i/dtheta)^T
dz <- solve(H_rob, -RHS)
chain dz back to (q, t) and to the upstream loss
```

The **second-order certificate** is the spectrum of $H_{\mathrm{rob}}$. If $H_{\mathrm{rob}}\succ0$, the fixed point is a strict local minimum of the joint BR objective and of $F_{\mathrm{rob}}$; if it has a negative eigenvalue, the fixed point is a saddle; if $\lambda_{\min}$ crosses zero, the branch bifurcates and the gradient norm blows up. This is a local stability diagnostic, not a robustness certificate: a wrong, outlier-consistent pose can be a strict local minimum with $H_{\mathrm{rob}}\succ0$.

### 4.6 What is not claimed

We do not claim (i) global optimality of the robust cost, (ii) certified robustness or a breakdown-point guarantee, (iii) a 6D sign-flip theorem, (iv) that the GNC output is the global argmin, or (v) that the layer replaces RANSAC in adversarial or coherent-outlier regimes. We do claim exact differentiation of the selected robust fixed point, a corrected second-order theory, and a per-observation gradient bound under explicit conditions.

---

## 5. Experiments

> **Status (2026-11-27): measured subsets only.** The synthetic 3D-3D tables, the 6D gradient table, the GT-supervised boundary diagnostic, ModelNet40/ModelNet10, four 3DMatch scenes (oracle; two with an FPFH front-end audit), KITTI raw drive 0001, the inner-solver comparison, the batch benchmark, and four ablations are measured and reproducible from `experiments/`. The comparison against SDPRLayers/Theseus/BPnP/Pyatov/POLI and the UTIAS In-The-Dark end-to-end experiment are **not** measured and are not claimed. No number below is a preliminary author number; all are independent runs from this repository.

### 5.1 Scope and evaluation protocol

The measured task is 3D-3D registration with correspondences $(p_i,y_i)$, residual $r_i=R(q)p_i+t-y_i$, Geman–McClure kernel with $c=0.05$ m unless stated otherwise, and success defined as rotation error $<5^\circ$ and translation error $<0.1$ m. RANSAC uses 3-point samples and threshold $\|r_i\|<c$; the fixed baseline always spends 1000 iterations (the v0 protocol), and the adaptive baseline uses the standard 0.999 confidence bound capped at 1000. Unless stated otherwise, each cell has 50–100 trials and fixed seeds. Runtimes are single-call wall-clock p50/p95 on a shared CPU node; absolute times drift with node load, so all speedups are within-run ratios.

The main forward solver is weighted Kabsch, the closed-form scalar-weight 3D-3D specialization of QPEP; the robust outer loop is GNC-GM or IRLS-GM. This isolates the robust estimator and the backward, not the full QPEP polynomial algebra. A 4×4 Davenport/Horn inner solver is benchmarked separately (§5.6); no 40-root QPEP solver is implemented.

### 5.2 Synthetic forward robustness and cost

Table 5.1 reports random outliers (replaced observations uniform in $[-2,2]^3$), 100 trials/cell, $N=100$. GNC-GM and both RANSAC variants are at 100% through 70%; IRLS drops to 98% at 70%; non-robust Kabsch collapses after 10%.

| $p$ | Kabsch | RANSAC-1000 | RANSAC-adaptive | IRLS-GM | GNC-GM |
|---|---|---|---|---|---|
| 0% | 100 | 100 | 100 | 100 | 100 |
| 10% | 52 | 100 | 100 | 100 | 100 |
| 30% | 0 | 100 | 100 | 100 | 100 |
| 50% | 0 | 100 | 100 | 100 | 100 |
| 70% | 0 | 100 | 100 | 98 | 100 |

Median rotation error is 0.13–0.21$^\circ$ for all robust methods; at $p=30\%$ RANSAC-1000 is 0.13$^\circ$ and GNC 0.14$^\circ$, i.e. tied at two decimals. The meaningful difference is cost: GNC p50 is 1.48–1.63 ms, RANSAC-1000 is 47.7–51.9 ms, and adaptive RANSAC is 0.41–12.36 ms. Against the fixed v0 baseline GNC is **31.6–32.5×** faster for $p\ge10\%$; against adaptive RANSAC it is slower at low $p$ (3× at 10%), comparable at 40%, and 8.3× faster at 70% (25–30× at 80–90%, where adaptive RANSAC saturates at its 1000-iteration cap). RANSAC is not differentiable; GNC's backward is a 6×6 solve.

Random outliers flatter every robust method. A finer breakdown sweep (100–200 trials/cell) shows the RANSAC failure at 80–90%: at 90% random outliers RANSAC-1000 is 51%, GNC 62%, IRLS 32%; at 80% GNC is 89% while RANSAC is 100%. With one coherent competing rigid transform, all methods are at 100% through 45% and then collapse: at 50% the two hypotheses tie (GNC 52%, RANSAC 50%), at 55% GNC is 2% and RANSAC 0%, and at 60% every method returns the **outlier** transform (100% recovered-outlier rate). This is the identifiability failure of Prop. 3.7, not a solver bug. With boundary outliers ($e/c^2\sim U(1/3,3)$), all methods remain at 100% success even at 70%; the robust methods are more accurate (GNC 0.27$^\circ$ vs RANSAC 0.62$^\circ$ and Kabsch 0.33$^\circ$ at 70%), but forward success is not where boundary outliers bite.

### 5.3 Exact gradient vs frozen weights

The 6D gradient table uses $N=30$, ten 40%-boundary instances and ten 20%-random instances, GNC + Newton polish to $\|\Phi\|\sim10^{-13}$, and central finite differences of the polished fixed-point branch ($h=10^{-5}$). The exact IFT Jacobian matches FD to a relative Frobenius error of **4.9e-8** (inliers), **3.1e-8** (boundary), and 1.25e-4 (gross, limited by FD cancellation), with cosine 1.000000. The naive frozen-weight Jacobian is 0.212 (inlier) and **1.07** (boundary) relative error; its boundary cosine is 0.234 (min −0.358). Omitting only the $-\lambda_{\rm rob}$ term gives 9e-4–1e-3 relative error, so the curvature correction is the dominant missing term, not the constraint term.

| bin | method | $n$ | rel. error p50 | cosine p50 | cosine min |
|---|---|---|---|---|---|
| inlier | exact | 414 | 4.9e-8 | 1.000000 | 1.000000 |
| inlier | naive | 414 | 2.12e-1 | 0.985 | 0.791 |
| boundary | exact | 122 | 3.1e-8 | 1.000000 | 1.000000 |
| boundary | naive | 122 | **1.07** | **0.234** | **−0.358** |
| gross | exact | 64 | 1.25e-4* | 1.000000 | 0.9986 |
| gross | naive | 64 | 1.16 | −0.118 | −0.397 |

\*FD-cancellation-limited; the absolute error remains small and the cosine is 1.

For observations with $e>c^2/3$, the mapping-level radial response has the opposite sign for exact vs naive in **100%** of boundary (122/122) and gross (64/64) observations; the pose-translation level also flips in 100% of this well-conditioned 6D sample. The 1D statement is a theorem (Prop. 1); the 6D statement remains a numerical observation, not a theorem.

### 5.4 End-to-end boundary diagnostic

A GT-supervised test-time refinement updates only the injected boundary observations for 20 Adam steps, re-solving the fixed point after each step. At $p=0.5$ and lr $10^{-3}$, the exact gradient reaches a final loss of 1.90e-6 vs 1.66e-5 for naive (ratio 0.07, 14× lower) and a rotation error of 0.029$^\circ$ vs 0.087$^\circ$; exact wins 10/10 paired seeds. At $p=0.7$ and lr $10^{-3}$ the ratio is 0.095 and exact wins 9/10. The all-points control is not discriminative (4–9/10, ratio 0.64–1.04), because inliers can compensate the wrong boundary direction. The diagnostic's gradient check is exact to 1.6e-7 / 3.9e-8 / 9.3e-8 relative error per coordinate; naive is 1.42 / 0.60 / 0.38.

A real float64 PyTorch `autograd.Function` now implements the same forward (GNC + Newton polish) with exact IFT, naive frozen-weight, and stopgrad backwards. `torch.autograd.gradcheck` passes for the exact backward at N=6/8/10 (max absolute error 6–8e-10; max relative 8e-9–1.1e-8, except one near-zero element at 8e-6); the naive backward fails at relative error 1.5–1.8. The torch port agrees with the NumPy implementation to 1e-16 in pose and 3e-16 in dL/dy. A small learned refinement (per-point MLP producing bounded observation corrections, 5 seeds, p=0.4 boundary outliers, 50 Adam steps) reduces median rotation from 0.635$^\circ$ to 0.068$^\circ$ for exact vs 0.106$^\circ$ for naive; exact wins 4/5 paired seeds with a median loss ratio of 0.56. A disjoint held-out protocol is **negative**: a local per-point MLP cannot identify the random global pose correction, both arms overfit, and exact also hits one basin switch. This is a toy validation of the layer implementation, not a trained real-data system; In-The-Dark remains unmeasured (§5.8).

### 5.5 Real geometry

**ModelNet40 / ModelNet10.** With real object point clouds, 50 trials/cell and uniform outliers, GNC-GM is 100% at 0–70% on both datasets; RANSAC-1000 is 100%, IRLS is 98–100%, and Kabsch collapses at 50%. GNC is 1.7–1.9 ms vs 53.5–61.5 ms for fixed RANSAC, i.e. **30–32.5×** faster. ModelNet10 is a second dataset (908 test objects, 10 classes, OFF meshes resampled to 2048 points); robust-method success differs by at most 2 pp and median rotation by at most 0.038$^\circ$ from ModelNet40.

**3DMatch.** Four official fragment scenes (redkitchen, sun3d-home_at, sun3d-hotel_uc, sun3d-home_md) are processed with the official TSDF point clouds and `gt.log`. The file's transform maps fragment `id2` into `id1`; this was verified empirically (forward median NN 0.50 m vs inverse 0.05 m). Base correspondences are GT-mapped nearest neighbours on real overlapping fragments (oracle matches with real scan/occlusion residual), $N=300$, $c=0.05$ m, 20 trials/cell, 8 high-overlap pairs/scene. On random outliers GNC and RANSAC-1000 are 100% through 70%; GNC p50 is 2.2–2.5 ms vs 61–71 ms (27–30×). On the two unseen scenes the pooled random-outlier success is again GNC/RANSAC 100% through 70% (IRLS 98% at 70%), GNC p50 0.83–2.56 ms vs RANSAC 62.5–74.2 ms (~29×), and boundary at 70% is GNC 0.373$^\circ$ vs RANSAC 0.718$^\circ$. Coherent outliers collapse at 60–70% for every method (0%); at 50% the extra-scene GNC success is 38% and IRLS 57%, a tie regime. A separate FPFH front-end (mutual NN + pairwise-distance consistency, no GT in matching) has a natural inlier rate of 80.0% / 67.8%; GNC succeeds 100% / 100%, RANSAC-1000 100% / 97%, Kabsch 100% / 83%. This is a hand-crafted FPFH front-end, not FCGF/Predator.

**KITTI.** Two raw drives (2011_09_26 drives 0001 and 0002, 108 and 77 frames) provide real outdoor LiDAR. Ego motion is derived from OXTS (Mercator + roll/pitch/yaw) and mapped to Velodyne with `calib_imu_to_velo`; the direction is verified (correct NN 0.10–0.12 m vs inverse 0.14–0.18 m). With $N=300$, $c=0.25$ m (LiDAR spacing is ~0.1 m), 10 consecutive-frame pairs and 20 trials/cell, GNC and RANSAC-1000 are 100% through 70% random outliers; GNC p50 is 2.7–2.8 ms vs 61–64 ms (~22×). Coherent 60–70% collapses for all methods; boundary is 100% for GNC and IRLS, with GNC at 0.07–0.09$^\circ$ vs RANSAC 0.14–0.16$^\circ$ at 70%. Drive 0002 reproduces the drive-0001 success table (max 5 pp difference), the coherent transition, and the ~21–23× speedup; two independent runs agree in all non-timing fields. A threshold sweep shows GNC remains 100% at $c\in\{0.15,0.25,0.35\}$, while c=0.25 is the defensible match to the ~0.10 m point spacing.

| dataset | task | random 0–70% | GNC vs fixed RANSAC | failure mode |
|---|---|---|---|---|
| synthetic | 3D-3D, N=100 | 100% | 31.6–32.5× | coherent majority; boundary gradient |
| ModelNet40 | object clouds, N=200 | 100% | 30–31.7× | none up to 70% |
| ModelNet10 | object clouds, N=200 | 100% | 31.7–32.5× | none up to 70% |
| 3DMatch | indoor fragments, 4 scenes, N=300 | 100% | 27–30× | coherent 60–70% |
| KITTI | outdoor LiDAR, N=300 | 100% | ~22× | coherent 60–70% |

### 5.6 Systems and inner solver

A vectorized numpy GNC implementation over $(B,N,3)$ reproduces the loop to rotation $\le1.1\times10^{-14}$ and translation $\le1.2\times10^{-15}$ with identical $\mu$-schedules. At $B=32,N=100$, batch p50 is 8.5 ms vs 56.2 ms for the loop (**6.60×**); at $B=8$ the speedup is 2.95× (N=100) and 3.27× (N=500). $B=1$ is slower in the batch path (0.54×), so the loop remains the right choice for single instances. On an RTX 5090 in float64 eager torch, GPU is launch-bound at ~40 ms and only wins at $B=128$ (5–10×); no CUDA-graph/torch.compile fusion or fp32 is claimed.

A 4×4 Davenport/Horn inner solver is numerically equivalent to weighted Kabsch (max $\|R_d-R_k\|_F=1.07\times10^{-14}$, max cost excess 9.4e-16) but its runtime is only better at large $N$: p50 0.93× (N=20), 1.01× (N=100), 1.27× (N=500). It is a large-N option, not a default-N=100 win. No 40-root polynomial QPEP solver was implemented; for scalar-weight 3D-3D the QPEP stationary-point problem reduces to this 4×4 eigenproblem.

### 5.7 Ablations

- **Kernel.** With the same weighted-Kabsch inner and schedule, GM, Cauchy, and Welsch are all 100% on random and boundary outliers through 70%. GM has the lowest boundary median rotation at 50–70% (0.199/0.229/0.340$^\circ$ vs Cauchy 0.241/0.264/0.370$^\circ$, Welsch 0.220/0.268/0.392$^\circ$). There is no evidence to switch the default.
- **GNC schedule.** Sweeping $\gamma\in\{1.2,1.4,1.7,2.0\}$, $\mu_0$ multiplier $\{0.5,1,4\}$ and polish steps $\{0,5,10\}$ gives 100% success and identical median rotation at $p=0.3/0.5/0.7$; $\gamma=2.0$ is 1.1–1.7× faster than the default, $\gamma=1.2$ is 1.7–2.5× slower. The default is conservative, not uniquely optimal.
- **Anisotropic $W_i$.** With a correct 6-DoF Gauss-Newton inner solve on Cholesky-whitened residuals, $W_i=I$ matches Kabsch to 5.3e-13. Scalar weighted Kabsch is **not** exact for anisotropic $W_i$: its objective ratio at converged GM weights is 1.002–1.007 (diagonal) and 1.035–1.088 (full SPD), with a single-trial maximum of 3.80. Under the isotropic synthetic noise, however, anisotropic $W_i$ is a misspecified model: full-W GNC has a paired median rotation change of +0.055 to +0.124$^\circ$, is better on only 8–32% of trials, and has 92% success at $p=0.7$ (identity: 100%). The exact inner solve is 5–23× slower than scalar Kabsch. We therefore report this as a correctness/negative result: use the nonlinear inner when $W_i$ is genuinely anisotropic, but do not claim an accuracy gain from anisotropy on this data.

### 5.8 What is not measured

The following are **not** claimed and remain required for the full paper: (i) SDPRLayers, SDPRLayers+RANSAC, Theseus-GM, BPnP, Pyatov et al., POLI, and finite-difference baselines on the same real data; (ii) the UTIAS In-The-Dark end-to-end pipeline (the current end-to-end result is a GT-supervised test-time diagnostic, not a trained layer); (iii) learned descriptors (the only descriptor front-end is hand-crafted FPFH); (iv) full 50-scene 3DMatch, low-overlap/long-baseline pairs, KITTI sequences 00–10, or more than two KITTI drives; (v) the full 40-root QPEP polynomial solver, its root-completeness/failure rate, and GPU batching; (vi) anisotropic $W_i$ under matched noise covariance; (vii) a 6D sign-flip theorem (the 6D result is numerical); (viii) global optimality of the robust cost, a breakdown-point guarantee, or adversarial robustness. The coherent-majority failure and the RANSAC-adaptive comparison are reported explicitly because they weaken the strongest possible claims.

## 6. Discussion and Limitations

**No global guarantee.** The inner weighted QPEP is globally solved, but the outer GNC is not. A GNC fixed point is a stationary point of $F_{\mathrm{rob}}$; it need not be a local minimum (if $H_{\mathrm{rob}}$ is indefinite) and need not be the global minimum. The $\mu$-homotopy can bifurcate, and the algorithm can follow a branch that is not globally optimal. We state this explicitly and do not use the phrase "global robust layer".

**Model breakdown.** Even a perfect global solver for $F_{\mathrm{rob}}$ can return the wrong pose when coherent outliers outnumber inliers. A one-dimensional example with one inlier at 0 and two coherent outliers at $M$ has $F(0)\to2c^2$ and $F(M)\to c^2$ as $M\to\infty$, so the global minimum is the outlier pose. This is an identifiability limit, not an algorithmic bug. Practical deployment needs a consensus check, an inlier-count diagnostic, or a prior on the outlier fraction.

**Boundary outliers are the worst.** $\lambda_\parallel$ is most negative at $e=c^2$, not at $e\to\infty$. Gross outliers are gradient-immune at $O(\|\mathbf r\|^{-4})$, but a point just beyond the redescending threshold has the largest gradient influence. This is both a warning for training and an opportunity: boundary outliers are where the curvature correction matters most.

**Gradient immunity vs robustness.** The per-observation bound is conditional on a uniform positive lower bound on $H_{\mathrm{rob}}$. If the Hessian is nearly singular, the constant blows up; if many outliers align, their contributions add; if the outliers are coherent, the global minimum can be wrong. We therefore separate "the gradient through this observation decays" from "the estimator is robust".

**Kernel choice and scale.** The framework covers GM, Cauchy, Welsch, and TLS, but the scale $c$ (or the noise model) is critical. A fixed $c$ may fail under varying noise; a learned $c$ introduces a scale identifiability issue with $W_i$. We recommend fixing $\det W_i=1$ and learning a scalar or log-scale, and reporting sensitivity to $c$.

**PnP and reprojection error.** The derivations are exact for residuals that are affine in $\mathbf t$ and quadratic in the lifted quaternion. Reprojection error is not polynomial; QPEP uses an algebraic surrogate. Extending the $\lambda_i$ formula to reprojection error is future work.

**Fairness against RANSAC.** RANSAC is a strong, well-tuned baseline. Our claim is not "RANSAC is bad"; it is "a differentiable robust layer can remove the non-differentiable wrapper and provide useful gradients, at the cost of a local robust estimator". The experiments must tune RANSAC and report its budget.

**When to use the layer.** Use it when (i) matches are noisy and contain gross outliers, (ii) end-to-end gradients through the pose are needed, (iii) the outlier fraction is below the breakdown regime, and (iv) a 6×6 backward and a fixed-size algebraic forward fit the compute budget. Do not use it when certified global optimality or adversarial robustness is required.

---

## 7. Conclusion

We presented Robust-QPEPLayer, a RANSAC-free differentiable pose layer that combines BR duality, GNC, and a constant-size QPEP inner solver, and differentiates the robust fixed point exactly. The theory isolates two terms that the natural implementation drops: the redescending curvature correction, which is the Schur complement of the BR weight block, and the constraint-multiplier term of the unit-quaternion constraint. The one-dimensional sign-flip proposition gives a rigorous setting in which the frozen-weight gradient has the wrong sign, and the per-observation $O(\|\mathbf r\|^{-4})$ bound quantifies when the gradient through a gross outlier becomes negligible. The method is not a global robust solver and does not replace RANSAC in adversarial regimes; it is a principled, constant-dimension, exactly differentiable robust layer whose failure modes are stated. Measured on synthetic, ModelNet40/10, 3DMatch, and KITTI 3D-3D data, it matches fixed-RANSAC success through 70% random outliers at 22–32× lower runtime, and its exact gradient is 3–5×10⁻⁸ relative to finite differences; the full baseline suite and In-The-Dark training remain future work.

---

## References

[1] C. Holmes, F. Dümbgen, and T. D. Barfoot, "SDPRLayers: Certifiable Backpropagation Through Polynomial Optimization Problems in Robotics," *IEEE Transactions on Robotics*, vol. 41, pp. 4120–4138, 2025. DOI: 10.1109/TRO.2025.3578228. arXiv:2405.19309.

[2] J. Wu, Y. Zheng, Z. Gao, Y. Jiang, X.-C. Hu, Y.-L. Zhu, J.-H. Jiao, and M. Liu, "Quadratic Pose Estimation Problems: Globally Optimal Solutions, Solvability/Observability Analysis, and Uncertainty Description," *IEEE Transactions on Robotics*, vol. 38, no. 5, pp. 3314–3335, 2022. DOI: 10.1109/TRO.2022.3155880.

[3] M. J. Black and A. Rangarajan, "On the Unification of Line Processes, Outlier Rejection, and Robust Statistics with Applications in Early Vision," *International Journal of Computer Vision*, vol. 19, no. 1, pp. 57–91, 1996. DOI: 10.1007/BF00131148.

[4] H. Yang, P. Antonante, V. Tzoumas, and L. Carlone, "Graduated Non-Convexity for Robust Spatial Perception: From Non-Minimal Solvers to Global Outlier Rejection," *IEEE Robotics and Automation Letters*, vol. 5, no. 2, pp. 1127–1134, 2020. DOI: 10.1109/LRA.2020.2969178.

[5] H. Yang, J. Shi, and L. Carlone, "TEASER: Fast and Accurate 3D Registration with Extreme Outlier Rates," in *Proc. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2020.

[6] L. Pineda, T. Fan, M. Monge, S. Venkataraman, P. Sodhi, R. T. Q. Chen, J. Ortiz, D. DeTone, A. Wang, S. Anderson, J. Dong, B. Amos, and M. Mukadam, "Theseus: A Library for Differentiable Nonlinear Optimization," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2022. arXiv:2207.09442.

[7] B. Chen, A. Parra, J. Cao, N. Li, and T.-J. Chin, "End-to-End Learnable Geometric Vision by Backpropagating PnP Optimization," in *Proc. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2020.

[8] H. Chen, P. Wang, F. Wang, W. Tian, L. Xiong, and Q. Li, "EPro-PnP: Generalized End-to-End Probabilistic Perspective-n-Points for Monocular Object Pose Estimation," in *Proc. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2022.

[9] V. Pyatov, I. Koshelev, and S. Lefkimmiatis, "Robust Two-View Geometry Estimation with Implicit Differentiation," in *Proc. IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)*, 2024. arXiv:2410.17983.

[10] J. Lee, J. Kim, W. Shin, G. Kim, and H. Oh, "Learning Point Cloud Geometry as a Statistical Manifold: Theory and Practice," arXiv:2605.10456, 2026.

[11] S. Geman and D. E. McClure, "Bayesian Image Analysis: An Application to Single Photon Emission Tomography," in *Proc. American Statistical Association*, 1985, pp. 12–18.

[12] M. A. Fischler and R. C. Bolles, "Random Sample Consensus: A Paradigm for Model Fitting with Applications to Image Analysis and Automated Cartography," *Communications of the ACM*, vol. 24, no. 6, pp. 381–395, 1981.

[13] P.-A. Absil, R. Mahony, and R. Sepulchre, *Optimization Algorithms on Matrix Manifolds*. Princeton, NJ: Princeton University Press, 2008.

[14] S. Fung, H. Heaton, Q. Li, D. Wu, K. He, and Y. Zhang, "JFB: Jacobian-Free Backpropagation for Implicit Networks," in *Proc. AAAI Conference on Artificial Intelligence*, 2022.

[15] UTIAS ASRL, "UTIAS Long-Term Localization and Mapping Dataset (In-The-Dark)," 2020. [Online]. Available: http://asrl.utias.utoronto.ca/datasets/2020-vtr-dataset/

---

## Appendix A: Proofs

The complete proofs of Lemmas 1–5, Theorems 1–3, Propositions 1–3, the Schur-complement identity, the Riemannian Hessian formula, and the existence construction are in `theory/proofs.md` (in Chinese) and `theory/derivations.md`. The numerical checks are in `verify/`. This appendix will be expanded in the camera-ready version; the current draft states proof sketches in the main text and defers the full algebra to the theory documents to keep the paper self-contained at the level of statements and assumptions.

## Appendix B: Kernel duals

| Kernel | $\rho_\mu(e)$ | $w^*=\rho_\mu'(e)$ | $\psi_\mu(w)$ |
|---|---|---|---|
| Geman–McClure | $ae/(a+e)$ | $(a/(a+e))^2$ | $a(\sqrt w-1)^2$ |
| Cauchy | $a\log(1+e/a)$ | $a/(a+e)$ | $a(w-1-\log w)$ |
| Welsch | $a(1-e^{-e/a})$ | $e^{-e/a}$ | $a(1-w+w\log w)$ |
| TLS | $\min(e,a)$ | $1$ if $e<a$, $0$ if $e>a$ | $a(1-w)$ |

Here $a=\mu c^2$. TLS has a linear $\psi$ and a non-unique weight at $e=a$; it requires the GNC-TLS schedule rather than plain bandwidth annealing.
