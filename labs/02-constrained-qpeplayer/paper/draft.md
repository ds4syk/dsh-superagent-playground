# Constrained-QPEPLayer: Certifiable Inequality-Constrained 6-DoF Pose Layers via Additive Quaternion Reduction and Fischer–Burmeister Implicit Differentiation

> **Status**: draft v0.6 (2026-11-28). Sections 1–4 and 6–7 are written. Section 5 reports measured layer-level results — synthetic desktop at $p=8$–$64$, noise/extrinsic bias, certificate tightness and the zero-dimensional RLT ablation, the SCP and exact-enumeration baselines, BOP LINEMOD/LINEMOD-Occlusion/YCB-Video with rigorous Schur-repair certification for three cut families (base, zero-dimensional RLT, bound-RLT), the SDPRLayers-style re-implementation at $n=50$/$n=90/72/84$, the dQP/CVXPYLayers-style QP layer at $n=50$, the official Theseus λ-Pareto at $n=30$, a CPU end-to-end toy with a finite-difference check of the FB backward, and KITTI/UTIAS localisation layers including a rigorous KITTI certificate subset — each with a source line. The two user-supplied theorems (P2-1b root spectrum, P2-Cert rigorous dual repair) are summarised in §4.4 and §4.7 and proved in `theory/proofs.md` §2A. §5.11 states exactly what is not run. No reference entry is marked **[verify]**; the two open items of v0.5 are resolved.

---

## Abstract

Differentiable pose estimation layers promise to close the loop between perception and control, but existing layers are either *local* (Gauss–Newton / Levenberg–Marquardt with implicit differentiation) or *convex-relaxation based* and therefore restricted to problems whose semidefinite relaxation happens to be tight. Neither handles the inequality constraints that make a pose physically admissible: positive depth in front of the camera, non-penetration of a manipulated object with its support surface, field-of-view limits, and tilt-cone limits of a ground or aerial vehicle. Adding $p$ such constraints to a semidefinite-programming layer inflates the lifted matrix from $10\times10$ to $(10+p)\times(10+p)$ and requires the relaxation to remain tight; generic convex-optimization layers cannot represent the non-convex rotation manifold at all.

We present **Constrained-QPEPLayer**, a differentiable 6-DoF pose layer for inequality-constrained, quaternion-parametrised pose estimation. Its three ingredients are: (i) an **additive constant-dimension reduction** showing that, for the Rockafellar–Wets augmented Lagrangian of an inequality-constrained quaternion pose problem (C-QPEP), every inner subproblem collapses — after an exact $3\times3$ Schur complement in the translation — to a *fixed-dimension quartic form on $\mathbb{S}^3$*, independent of $p$; (ii) a **posteriori sound global certificate** given by a $14\times14$ Veronese-augmented Shor semidefinite relaxation whose positive-semidefinite block does not grow with $p$, which yields a rigorous lower bound and certifies global optimality when tight; and (iii) a **Fischer–Burmeister (FB) reduced implicit differentiation** scheme whose $(6+|\mathcal A|)\times(6+|\mathcal A|)$ saddle-point system remains invertible for every $\varepsilon>0$ even when the active-constraint Jacobian is rank deficient, as it is for a four-legged object resting on a table. We characterise precisely what the $\varepsilon\to0$ limit of that system is: an $H$-weighted minimum-norm solution, equal to the true derivative under an explicit compatibility condition, and a *selected* generalized derivative with divergent multipliers when that condition fails.

We are explicit about what the layer does *not* do. The augmented Lagrangian converges to KKT points, not to global optima; globality comes exclusively from the posteriori certificate, which is allowed to fail and is reported as a failure. We therefore describe the layer as **fail-safe certifiable** rather than certifiable, and on a controlled synthetic instance we show that a naive Lagrangian-dual certificate can be numerically negative ($-1.67\times10^{-11}$) and must therefore be treated as a diagnostic rather than a certificate, whereas the $14\times14$ relaxation is tight to $2.2\times10^{-12}$ on the same instance. We report a layer-level evaluation on BOP LINEMOD, LINEMOD-Occlusion and YCB-Video (90/72/84 ground-truth-projected poses) with penetration depth, pose error, feasibility, and rigorous Schur-repair certification rates (24.4/15.3/13.1% on the base cut family, rising to 51.1/41.7/23.8% with the sound bound-RLT family); the same layer is run on KITTI odometry and on UTIAS In-The-Dark at layer level (§5.9), and on a small CPU end-to-end toy with four pose heads and a finite-difference check of the FB backward (§5.8); network-scale end-to-end training (T1), TUM RGB-D and the T3 equality-constrained tasks are stated but not run (§5.11).

---

## 1 Introduction

### 1.1 The problem

A 6-DoF pose layer sits inside a perception or policy network and maps learned quantities $\theta$ (keypoint heat-maps, correspondences, weights) to a pose $(\mathbf R,\mathbf t)$, in a way that is differentiable with respect to $\theta$. Two families dominate. *Local* layers — BPnP [Chen et al. 2020], Theseus [Pineda et al. 2022], and the Gauss–Newton pipeline inside most SLAM systems — solve a nonlinear least-squares problem from the current estimate and differentiate the KKT system of that *local* solution. *Global* layers — SDPRLayers [Holmes et al. 2025] — solve a convex (semidefinite) relaxation of a polynomial optimisation problem, and differentiate the original non-convex KKT system exploiting the relaxation's dual certificate.

Both families were designed for **equality** constraints. SDPRLayers is explicit about this: its problem class is a homogeneous quadratically constrained quadratic program (QCQP) with constraints $\mathbf x^\top\mathbf A_i\mathbf x=0$ [Holmes et al. 2025, Eq. (P1)]. QPEP [Wu et al. 2022] — the algebraically global quaternion solver that motivates our work — also has exactly one constraint, $\|\mathbf q\|^2=1$, with everything else absorbed into the parametrisation.

Real pose problems, however, are inequality constrained. An object manipulated on a table cannot penetrate it. A point in front of the camera must have positive depth. A ground robot's roll and pitch are bounded. A camera's field of view bounds where a projected point can be. These are not soft preferences to be traded off with a penalty: they are physical admissibility conditions, and a layer that violates them produces poses that a motion planner must reject.

### 1.2 Why existing layers do not solve it

**SDPRLayers with slack variables.** The straightforward fix is to add $p$ non-negative slacks, which lifts the PSD matrix from $10\times10$ to $(10+p)\times(10+p)$. For an object mesh with 50 vertices this is a $60\times60$ SDP *per pose per training step*, solved on the CPU, and interior-point complexity grows as $O((10+p)^{6})$ — a number we correct from the $O((10+p)^2)$ stated in an earlier internal draft of this idea. Worse, the theory degrades: the backward pass needs the relaxation to be tight and the multipliers to be identifiable, and inequality-constrained relaxations routinely violate strict complementarity at the solution ($g_j^\star=0$, $\mu_j^\star=0$), where the standard equality-KKT implicit function theorem simply does not apply.

**Generic differentiable QP layers.** dQP [Magoon et al. 2025], OptNet [Amos & Kolter 2017] and CVXPYLayers [Agrawal et al. 2019] differentiate convex programs. They handle linear inequalities natively and elegantly — dQP in particular observes that once the active set is known, primal and derivative share one linear system — but the rotation manifold is not convex. Linearising the rotation and iterating turns the layer into a local method with no global statement, and the active-set machinery inherits the LICQ assumption that fails at exactly the contact configurations we care about.

**Soft penalties.** Theseus [Pineda et al. 2022] supports arbitrary costs, including barrier and penalty terms, but a penalty is not a constraint: it trades penetration against data fit with a weight that must be tuned per scene, and the resulting penetration is bounded only by the weight, not by zero. DC3 [Donti et al. 2021] completes and corrects constraints at inference time but does not differentiate through the correction.

**QPEP without a layer.** QPEP [Wu et al. 2022] solves quaternion pose problems globally by eliminating the translation, forming the KKT polynomial system, and enumerating *all* roots with a Gröbner-basis/action-matrix computation, then picking the best. It is algebraically global (given a complete enumeration) and needs no tightness, but it has no gradient and no notion of inequality constraints.

### 1.3 This paper

We ask: *can a pose layer have hard inequality constraints, a fixed per-sample cost independent of the number of constraints, differentiability that survives redundant and weakly active contacts, and a posteriori evidence that the returned pose is globally optimal?*

**Contributions.**

1. **Additive constant-dimension reduction (Theorem 2.1, Corollary 2.1.1).** For the inequality-constrained quaternion pose problem (C-QPEP) with constraints affine in $(\mathbf v_2(\mathbf q),\mathbf t)$ — which covers positive depth, planar non-penetration, field-of-view and tilt-cone constraints exactly — the augmented-Lagrangian inner subproblem, restricted to an active set $\mathcal A$, is a quadratic form in $(\mathbf v_2,\mathbf t)$ assembled by $|\mathcal A|$ rank-one updates. Eliminating $\mathbf t$ with a $3\times3$ Schur complement gives a quartic form on $\mathbb{S}^3$ whose dimension does not depend on $p$. Assembly is $O(p)$ time and $O(1)$ memory. We are careful about what this does *and does not* say: the reduction preserves the problem *class* of QPEP's post-elimination polynomial system, but the claim that QPEP's specific elimination template can be reused verbatim is left as an explicit open question (§4.6, G1).
2. **A posteriori sound global certification with a $p$-independent PSD block (Lemma 2.2).** We show that a $14\times14$ Shor relaxation of the *lifted* problem, augmented with the quadratic relations of the Veronese embedding of $\mathbb{S}^3$, gives a rigorous lower bound on the constrained global optimum, and certifies global optimality when it is tight. The PSD block is $14\times14$ regardless of $p$; constraints enter as $p$ linear constraints. We prove soundness (no false certificates) and demonstrate on a synthetic instance that the relaxation is tight to $2\times10^{-11}$ while the dual-function gap of the same instance is numerically *negative*, i.e. unusable as a certificate.
3. **Fischer–Burmeister reduced implicit differentiation with a precise limit theory (Theorem 2.3, Proposition 2.1).** We regularise the complementarity system with a smoothed FB function, reduce the $p$-dimensional complementarity block analytically, and obtain a $(6+|\mathcal A|)\times(6+|\mathcal A|)$ saddle-point system that is invertible for every $\varepsilon>0$ whenever the effective Hessian and the regularisation are positive definite — *with no rank assumption on the active Jacobian*. We then prove that the $\varepsilon\to0$ limit is the $H$-weighted minimum-norm solution of $\mathbf J_{\mathcal A}d\mathbf z=-\partial_\theta\mathbf g_{\mathcal A}$; it equals the true derivative iff the right-hand side lies in the range of $\mathbf J_{\mathcal A}$ (a checkable compatibility condition), and otherwise the multipliers diverge as $1/\varepsilon^2$ while $d\mathbf z$ remains finite but is only *a selected generalized derivative*. We show that the four-legged-table case is compatible whenever the table's pose is the parameter (so the flagship example is defensible), and incompatible when the four leg heights are independent parameters (so the problem is genuinely non-differentiable).
4. **A fail-safe contract and a benchmark protocol.** The layer returns (pose, certificate value, feasibility violation, active set, conditioning estimate). Downstream code can branch on certification. We give the full protocol — YCB-Video/BOP and LINEMOD for manipulation, KITTI/TUM RGB-D/UTIAS for localisation, ADD(-S), penetration depth and ratio, latency — in §5, together with the ablations needed to separate the contributions.

### 1.4 What we deliberately do not claim

* We do **not** claim that the augmented Lagrangian finds global optima. Non-convex ALM converges to KKT points under constraint qualifications [Bertsekas 1982; Rockafellar & Wets 1998]; our globality is posteriori only.
* We do **not** claim that the FB smoothing is equivalent to complementarity. For $\varepsilon>0$ its zero set is the hyperbola $\mu g=-\varepsilon^2/2$, not the complementarity set (Lemma 2.3). It is a regularisation used for differentiation only.
* We do **not** claim that the layer is always certified. When the $14\times14$ relaxation is not tight, the layer reports "uncertified" and the returned pose is only a feasible KKT point.
* We do **not** claim distance/obstacle constraints $\|\mathbf R\mathbf p+\mathbf t-\mathbf c\|\ge d$, which are *not* affine in $(\mathbf v_2,\mathbf t)$, nor multi-modal cases where two global optima exchange; both are outside the stated scope (§3.4, §6).

---

## 2 Related Work

We position against four groups. Table 1 summarises.

**Certifiable differentiable layers.** SDPRLayers [Holmes et al. 2025] is the closest work and the explicit target. It lifts a homogeneous QCQP to a Shor SDP, reads off a global solution when the relaxation is tight ($\operatorname{rank}(\mathbf X^\star)=1$, $\operatorname{corank}(\mathbf H^\star)=1$), and differentiates the *original* QCQP KKT system with an implicit-selection argument that tolerates redundant equality constraints [ibid., Thm. 2, Thm. 4]. Three differences matter. (i) *Constraint class*: their theory is stated for quadratic **equalities**; inequalities require slack variables that inflate the PSD block and break strict complementarity. Ours are native. (ii) *Where the SDP lives*: SDPRLayers solves an SDP **in the forward pass** for every sample and every training step; we solve a fixed-dimension polynomial system forward and use an SDP **only for the posteriori certificate**, optionally and asynchronously. (iii) *What "global" rests on*: theirs rests on tightness of the forward relaxation; ours rests on tightness of a separate $14\times14$ relaxation, so a failure of tightness costs us the certificate but not the forward solve. Conversely, we inherit QPEP's enumeration-completeness assumption, which they do not need.

**Algebraically global quaternion solvers.** QPEP [Wu et al. 2022] unifies PnP, PnL, hand–eye calibration, point-to-plane registration, conic pose, range-based relative pose, Stewart-platform forward kinematics and multi-GNSS attitude as "quadratic pose estimation problems", eliminates $\mathbf t$, and enumerates all KKT roots with a Gröbner basis (up to 40 candidates in the reference implementation), taking the best. It requires no tightness and reports a covariance via a separate small SDP. Our work can be read as *adding inequalities and gradients to QPEP*: we keep its elimination structure and its enumeration-based globality, and add (a) constraints that are affine in $(\mathbf v_2,\mathbf t)$, (b) an augmented-Lagrangian outer loop, (c) FB-reduced implicit differentiation, (d) a semidefinite posteriori certificate that QPEP does not have. In QPEP's sense, "globally optimal" means that, given a complete root enumeration, the best stationary point of a compact problem is returned [Wu et al. 2022]; this is not a convex optimality certificate, and we never use QPEP's output alone as one (§4.3). A detailed reading of QPEP and of the enumeration code is recorded in `related_work.md`.

**Differentiable convex optimisation layers.** OptNet [Amos & Kolter 2017] differentiates QPs via the KKT system of an interior-point solve; CVXPYLayers [Agrawal et al. 2019] does so for cone programs through a canonicalisation that supports parametrised problems; dQP [Magoon et al. 2025] decouples the solver from the derivative and shows that the active set determines both. These methods assume convexity, hence a globally optimal forward solve and a well-posed derivative; they also assume that the active set is identified and that LICQ/strict complementarity holds. Our setting violates all three: the feasible set lives on $\mathbb{S}^3$, the LICQ can fail by construction (four contact points, three out-of-plane degrees of freedom), and strict complementarity fails at grazing contact. What we borrow from dQP is the *spirit* — reduce the derivative to a small linear system whose structure is fixed by the active set — and what we add is a rank-robust regularisation of that system plus a limit theory.

**Local differentiable pose layers.** BPnP [Chen et al. 2020] back-propagates through a PnP solve by implicit differentiation of its optimality conditions; EPro-PnP [Chen et al. 2022] models the pose posterior and integrates it by Monte-Carlo; Theseus [Pineda et al. 2022] provides a library for differentiable nonlinear least squares on manifolds with Gauss–Newton/Levenberg–Marquardt and implicit or unrolled derivatives. All are local: differentiation is with respect to *a* local solution, and a spurious local minimum yields a gradient of the wrong function. They support penalties but not hard constraints. SDPRLayers was introduced precisely to address this failure mode [Holmes et al. 2025, §I]; we address the part of the failure mode that survives when constraints are inequalities.

**Complementarity, ALM, and non-smooth implicit differentiation.** The Fischer–Burmeister function [Fischer 1992] and its smoothed variants [Kanzow 1996] are standard tools for turning complementarity into a semi-smooth equation; see [Facchinei & Pang 2003] for the comprehensive treatment, including the fact that FB is BD-regular and that semismooth Newton converges locally. The augmented Lagrangian method for non-convex problems, with convergence to KKT points and multiplier boundedness under constraint qualifications, is classical [Bertsekas 1982; Rockafellar & Wets 1998]; exact penalty variants go back to [Fletcher 1987]. Differentiating solution maps that are non-smooth or non-unique requires generalized derivatives; the framework closest to our needs is the nonsmooth implicit differentiation of [Bolte et al. 2021], which gives a conservative-field/Clarke-Jacobian calculus for optimization layers. We use it to state honestly what our $\varepsilon\to0$ limit is (Proposition 2.1, Lemma 2.5) rather than to claim differentiability.

**Table 1. Positioning.** ($\checkmark$ = supported/native, $\times$ = not supported, $\sim$ = partial/with caveats.)

| | SDPRLayers [Holmes 2025] | dQP [Magoon 2025] | Theseus [Pineda 2022] | QPEP [Wu 2022] | **Ours** |
|---|---|---|---|---|---|
| Non-convex $\mathbb{S}^3$/SE(3) forward | $\checkmark$ (lifted) | $\times$ (convex QP) | $\checkmark$ (local) | $\checkmark$ (global) | $\checkmark$ |
| Hard inequality constraints | $\times$ (slacks only) | $\checkmark$ (linear) | $\times$ (penalties) | $\times$ | $\checkmark$ (affine in $(\mathbf v_2,\mathbf t)$) |
| Forward cost vs. $p$ | $O((10+p)^6)$ | $O(\text{QP}(p))$ | $O(p)$ per iteration | $O(\text{const})$ | $O(p)$ assembly + $O(1)$ polynomial solve |
| Global forward guarantee | if tight | yes (convex) | $\times$ | enumeration (A2) | enumeration (A2) |
| Posteriori global certificate | forward certificate | n/a | $\times$ | $\times$ | $\checkmark$ ($14\times14$ SDP) |
| Derivative robust to LICQ failure | $\checkmark$ (implicit selection, equalities) | $\times$ | $\times$ | n/a | $\checkmark$ (FB, $J_{\mathcal A}$ rank-deficient) |
| Derivative at weak active / strict-compl. failure | $\times$ (needs strict compl.) | $\times$ | n/a | n/a | $\sim$ (selected generalized derivative) |
| Backward system size | $(n+m)$ after row reduction | $n+|A|$ | $n$ | n/a | $6+|\mathcal A|$, $|\mathcal A|\le6$ generic |
| GPU-batchable | $\times$ (CPU SDP) | $\checkmark$ | $\checkmark$ | $\sim$ | $\sim$ (fixed-size polynomial solve) |

---

## 3 Problem Formulation and Notation

### 3.1 Notation

$\mathbf q=[q_0,q_1,q_2,q_3]^\top\in\mathbb{S}^3$ is a unit quaternion, $\mathbf R(\mathbf q)\in SO(3)$ the corresponding rotation. For a 3-vector $\mathbf p$, define the lifted monomial vector

$$
\mathbf v_2(\mathbf q)=\big[q_0^2,q_1^2,q_2^2,q_3^2,\sqrt2 q_0q_1,\sqrt2 q_0q_2,\sqrt2 q_0q_3,\sqrt2 q_1q_2,\sqrt2 q_1q_3,\sqrt2 q_2q_3\big]^\top\in\mathbb{R}^{10},
$$

so that $\|\mathbf v_2(\mathbf q)\|_2^2=\|\mathbf q\|_2^4=1$ for $\mathbf q\in\mathbb{S}^3$, and $\mathbf e_{\rm tr}=[1,1,1,1,0,\dots,0]^\top$ satisfies $\mathbf e_{\rm tr}^\top\mathbf v_2(\mathbf q)=\|\mathbf q\|^2=1$.

**Fact 3.1 (affine representability).** For every $\mathbf p\in\mathbb{R}^3$ and $a\in\{1,2,3\}$ there is $\mathbf m_a(\mathbf p)\in\mathbb{R}^{10}$ with $\mathbf e_a^\top\mathbf R(\mathbf q)\mathbf p=\mathbf m_a(\mathbf p)^\top\mathbf v_2(\mathbf q)$ for all $\mathbf q\in\mathbb{S}^3$. (Verified to $4\times10^{-15}$ in `verify/sanity_constraints_alm.py`, check C1a.)

Fact 3.1 is the reason all four constraint families below are affine in $(\mathbf v_2,\mathbf t)$; it follows from $\mathbf R(\mathbf q)\mathbf p$ being quadratic in $\mathbf q$.

### 3.2 Objective

$$
J_\theta(\mathbf q,\mathbf t)=\mathbf v_2(\mathbf q)^\top\mathbf M_0(\theta)\mathbf v_2(\mathbf q)+2\mathbf t^\top\mathbf D_0(\theta)\mathbf v_2(\mathbf q)+\mathbf t^\top\mathbf G_0(\theta)\mathbf t . \tag{1}
$$

For PnP-type residuals $\mathbf r_i(\mathbf q,\mathbf t)=\mathbf P_i(\mathbf R(\mathbf q)\mathbf p_i+\mathbf t)$ with $\mathbf P_i=\mathbf I-\mathbf u_i\mathbf u_i^\top/\|\mathbf u_i\|^2$ the bearing projection, (1) holds with $\mathbf M_0=\sum_i\sum_{ab}P_{i,ab}\operatorname{sym}(\mathbf m_a(\mathbf p_i)\mathbf m_b(\mathbf p_i)^\top)$, $\mathbf D_0=\sum_i\mathbf P_i[\mathbf m_1;\mathbf m_2;\mathbf m_3]$, $\mathbf G_0=\sum_i\mathbf P_i$. This is QPEP's problem class minus the elimination.

### 3.3 Constraint families

All are of the form $g_j(\mathbf q,\mathbf t)=\tilde{\mathbf a}_j^\top\mathbf v_2(\mathbf q)+\mathbf b_j^\top\mathbf t\le0$.

| Family | Expression | $(\tilde{\mathbf a}_j,\mathbf b_j)$ |
|---|---|---|
| Positive depth (chirality) | $z_{\min}-\mathbf e_3^\top(\mathbf R\mathbf p_i+\mathbf t)\le0$ | $(-\mathbf m_3(\mathbf p_i)+z_{\min}\mathbf e_{\rm tr},\ [0,0,-1]^\top)$ |
| Planar non-penetration | $-\mathbf n^\top(\mathbf R\mathbf v_j+\mathbf t)-d\le0$ | $(-\sum_a n_a\mathbf m_a(\mathbf v_j)-d\,\mathbf e_{\rm tr},\ -\mathbf n)$ |
| Field of view ($\pm$) | $\pm(\mathbf R\mathbf p_i+\mathbf t)_x-\tau(\mathbf R\mathbf p_i+\mathbf t)_z\le0$ | $(\pm\mathbf m_1(\mathbf p_i)-\tau\mathbf m_3(\mathbf p_i),\ [\pm1,0,-\tau]^\top)$ |
| Tilt cone | $q_1^2+q_2^2-\sin^2(\tfrac\alpha2)\|\mathbf q\|^2\le0$ | $(\mathbf e_2+\mathbf e_3-\sin^2(\tfrac\alpha2)\mathbf e_{\rm tr},\ \mathbf 0)$ |

(The tilt-cone row uses $\|\mathbf q\|^2=1$, i.e. it is $q_1^2+q_2^2\le\sin^2(\alpha/2)$ on the manifold.) All four were verified to $4\times10^{-15}$ (check C1b).

**Remark 3.1 (chirality).** For the field-of-view constraint, $z>0$ is implied by the two FoV inequalities only if $\tau<\infty$ and the constraint is active in the right direction; we therefore always add at least one explicit positive-depth constraint per observed point. This is cheap ($p$ grows by the number of points) and removes the need for a separate sign argument.

### 3.4 Problem (C-QPEP)

$$
\min_{\mathbf q\in\mathbb{S}^3,\ \mathbf t\in\mathbb{R}^3}\ J_\theta(\mathbf q,\mathbf t)\quad\text{s.t.}\quad g_j(\mathbf q,\mathbf t;\theta)\le0,\ j=1,\dots,p. \tag{C-QPEP}
$$

**Assumption 3.1** $\mathbf M_0,\mathbf D_0,\mathbf G_0$ are continuously differentiable in $\theta$, and $\mathbf G_0\succ0$ (or the active $\mathbf b_j$ span $\mathbb{R}^3$).

**Out of scope.** (i) Non-affine constraints, e.g. $\|\mathbf R\mathbf p+\mathbf t-\mathbf c\|\ge d$ or inter-object clearance — these are not affine in $(\mathbf v_2,\mathbf t)$ and would require a different reduction. (ii) Multiple global optima exchanging as $\theta$ varies: the solution map jumps and no derivative exists; this is the branch-crossing problem addressed by other work in this family. (iii) Constraints with non-polynomial dependence on $\mathbf q$ (e.g. geodesic distance to a reference rotation).

---

## 4 Method and Theory

### 4.1 Outer loop: Rockafellar–Wets augmented Lagrangian

With $\boldsymbol\mu\in\mathbb{R}^p_{\ge0}$ and $\rho>0$,

$$
\mathcal L_\rho(\mathbf q,\mathbf t,\boldsymbol\mu)=J_\theta(\mathbf q,\mathbf t)+\frac{1}{2\rho}\sum_{j=1}^p\Big(\big[\mu_j+\rho g_j(\mathbf q,\mathbf t)\big]_+^2-\mu_j^2\Big). \tag{2}
$$

Algorithm 1 alternates: identify the effective active set $\mathcal A=\{j:\mu_j+\rho g_j>0\}$; solve the inner problem with $\mathcal A$ frozen; update $\mu_j\leftarrow[\mu_j+\rho g_j]_+$; increase $\rho$ if the feasibility violation does not decrease. The freeze is a Gauss–Seidel/active-set device, not an approximation of the exact subproblem minimiser; §4.2 makes precise what is being solved, and Lemma 2.1 records the consistency condition under which the frozen solve is a genuine minimiser of (2).

```
Algorithm 1: Constrained-QPEPLayer forward
input : θ, initial (q,t) from a cheap estimator, μ=0, ρ=ρ0, ε=ε0
repeat
   A ← { j : μ_j + ρ g_j(q,t) > 0 }
   assemble M_A, D_A, G_A                       # O(p), 10x10 / 3x10 / 3x3
   solve inner: min_{q∈S^3} v2(q)^T (M_A - D_A^T G_A^{-1} D_A) v2(q)   # QPEP-class
   (q,t) ← (q*, -G_A^{-1} D_A v2(q*))
   μ ← max(0, μ + ρ g(q,t));  ρ ← 5ρ if violation not ↓ by 0.5
until max_j [g_j]_+ ≤ ν_tol or max outer iters
certificate: solve 14x14 Shor SDP (Lemma 2.2); return (q,t,μ,A, SDP value, ν, cond)
```

### 4.2 Theorem 2.1: the inner problem has $p$-independent dimension

**Theorem 2.1.** Let Assumption 3.1 hold, fix $\boldsymbol\mu,\rho$, and fix an active set $\mathcal A$. On the region where the active pattern is $\mathcal A$,
$$
\mathcal L_\rho\big|_{\mathcal A}=\mathbf v_2^\top\mathbf M_{\mathcal A}\mathbf v_2+2\mathbf t^\top\mathbf D_{\mathcal A}\mathbf v_2+\mathbf t^\top\mathbf G_{\mathcal A}\mathbf t+\kappa_{\mathcal A}, \tag{3}
$$
with
$$
\begin{aligned}
\mathbf M_{\mathcal A}&=\mathbf M_0+\sum_{j\in\mathcal A}\Big[\tfrac{\mu_j}{2}\big(\mathbf e_{\rm tr}\tilde{\mathbf a}_j^\top+\tilde{\mathbf a}_j\mathbf e_{\rm tr}^\top\big)+\tfrac{\rho}{2}\tilde{\mathbf a}_j\tilde{\mathbf a}_j^\top\Big],\\
\mathbf D_{\mathcal A}&=\mathbf D_0+\sum_{j\in\mathcal A}\Big[\tfrac{\mu_j}{2}\mathbf b_j\mathbf e_{\rm tr}^\top+\tfrac{\rho}{2}\mathbf b_j\tilde{\mathbf a}_j^\top\Big],\qquad
\mathbf G_{\mathcal A}=\mathbf G_0+\tfrac{\rho}{2}\sum_{j\in\mathcal A}\mathbf b_j\mathbf b_j^\top,\\
\kappa_{\mathcal A}&=-\frac{1}{2\rho}\sum_{j\notin\mathcal A}\mu_j^2 . \qquad\text{(only \emph{inactive} constraints contribute a constant)}
\end{aligned}
\tag{4}
$$
If $\mathbf G_{\mathcal A}\succ0$ then $\min_{\mathbf t}\mathcal L_\rho|_{\mathcal A}=\mathbf v_2^\top\mathbf M^{\rm red}_{\mathcal A}\mathbf v_2+\kappa_{\mathcal A}$ with $\mathbf M^{\rm red}_{\mathcal A}=\mathbf M_{\mathcal A}-\mathbf D_{\mathcal A}^\top\mathbf G_{\mathcal A}^{-1}\mathbf D_{\mathcal A}$ and $\mathbf t^\star(\mathbf q)=-\mathbf G_{\mathcal A}^{-1}\mathbf D_{\mathcal A}\mathbf v_2(\mathbf q)$. The inner problem is therefore a **quartic form on $\mathbb{S}^3$**, of dimension independent of $p$.

*Proof sketch.* Expand the hinge: for $j\in\mathcal A$ the term equals $\mu_jg_j+\tfrac\rho2g_j^2$; for $j\notin\mathcal A$ it equals $-\mu_j^2/(2\rho)$. Each $\mu_jg_j$ is linear in $(\mathbf v_2,\mathbf t)$; using $\mathbf e_{\rm tr}^\top\mathbf v_2=1$ it becomes the quadratic forms shown. Each $\tfrac\rho2g_j^2$ expands into $\mathbf v_2\mathbf v_2^\top$, $\mathbf v_2\mathbf t^\top$ and $\mathbf t\mathbf t^\top$ blocks. Summing gives (4). The $\mathbf t$-stationarity is linear in $\mathbf v_2$, hence $\mathbf t^\star$ is quadratic in $\mathbf q$; substituting and using $\mathbf v_2(\lambda\mathbf q)=\lambda^2\mathbf v_2(\mathbf q)$ gives the quartic. Full proof: `theory/proofs.md` §1.2. $\square$

Three remarks that matter for correctness.

**(a) The constant term is not symmetric in $\mathcal A$.** Taking $\kappa=-\frac{1}{2\rho}\sum_{j=1}^p\mu_j^2$ makes (3) differ from the direct evaluation of (2) by $\sum_{j\in\mathcal A}\mu_j^2/(2\rho)$ — a bug that perturbs the multiplier recursion and therefore the fixed point. In our sanity check the wrong constant produces an error of $5.57\times10^{-1}$; the correct one gives $3.55\times10^{-15}$. This is the single most likely implementation error in the layer.

**(b) "10×10 QPEP" is a lifted statement.** (3) is quadratic in $\mathbf v_2$ but **quartic in $\mathbf q$**. It is *not* the standard QPEP form "quadratic in $(\mathbf q,\mathbf t)$"; it is the form QPEP reaches *after* eliminating $\mathbf t$. Consequently the KKT system is four cubics plus one quadratic in $(\mathbf q,\lambda)$, with Bezout bound $3^4\cdot2=162$, while QPEP's reference implementation returns at most 40 roots for its instances. The reduced problem is in the same algebraic family; whether QPEP's precomputed elimination template applies verbatim to a general $\mathbf M^{\rm red}_{\mathcal A}$ is **open** (G1). A safe implementation regenerates the template for the actual monomial support, whose size is still $O(1)$ in $p$.

**(c) The frozen-active-set inner solve is not the exact subproblem.** (2) is piecewise quadratic in $(\mathbf v_2,\mathbf t)$, with kinks on the surfaces $\mu_j+\rho g_j=0$. Freezing $\mathcal A$ selects one piece. If the minimiser of that piece has the same active pattern, it is a local minimiser of (2); otherwise we iterate. Lemma 2.1 states this consistency condition. In practice 2–5 inner iterations per outer step suffice on our synthetic instance.

**Lemma 2.1 (active-set consistency).** Let $\mathcal A_k$ be the active set at outer iterate $k$, and let $\mathbf z_k^\star$ minimise the $\mathcal A_k$-piece. If $\operatorname{sign}(\mu_j+\rho g_j(\mathbf z_k^\star))$ matches $\mathcal A_k$ for all $j$, then $\mathbf z_k^\star$ is a local minimiser of (2) and the outer update is the standard ALM update. If not, the piece is inconsistent and the next iteration must re-identify $\mathcal A$. *(Immediate from the piecewise structure; no proof required.)*

### 4.3 Theorem 2.2: what "globally optimal" can and cannot mean

**Theorem 2.2 (conditional global optimality).** Let $\hat{\mathbf z}$ be the ALM output with multipliers $\hat{\boldsymbol\mu}\ge0$ and feasibility violation $\nu=\max_j[g_j(\hat{\mathbf z})]_+$. For any $\boldsymbol\mu\ge0$ define $d(\boldsymbol\mu)=\min_{\mathbb{S}^3\times\mathbb{R}^3}\{J_\theta+\sum_j\mu_jg_j\}$. Then
$$
d(\hat{\boldsymbol\mu})\ \le\ J^\star\ \le\ J_\theta(\hat{\mathbf z})+L\nu ,
\qquad\text{hence}\qquad
0\le J_\theta(\hat{\mathbf z})-J^\star\le \underbrace{J_\theta(\hat{\mathbf z})-d(\hat{\boldsymbol\mu})}_{=:\delta}+L\nu . \tag{5}
$$
If $\delta+L\nu\le\tau$ then $\hat{\mathbf z}$ is $\tau$-globally optimal. In exact arithmetic $\delta\ge0$; a computed $\delta<0$ is a numerical artefact.

*Proof.* For feasible $\mathbf z$ and $\boldsymbol\mu\ge0$, $\sum_j\mu_jg_j(\mathbf z)\le0$, so $d(\boldsymbol\mu)\le\min_{\rm feasible}J=J^\star$. The upper bound follows from $L$-Lipschitzness and $\nu$-feasibility. $\square$

Theorem 2.2 is *the* honest statement of what ALM plus a dual bound gives: a sandwich, conditional on feasibility and on the accuracy of $d$. It is not a certificate, because $d(\hat{\boldsymbol\mu})$ is itself computed by a global solve of a non-convex problem, and in our implementation that solve is QPEP-style enumeration with no certificate. On our synthetic instance the computed $\delta$ is $-1.67\times10^{-11}$ — negative, therefore unusable as a certificate. A negative $\delta$ cannot be "$\approx0$ certification": it is evidence that the numerical error in evaluating $d$ exceeds the gap itself.

### 4.4 Lemma 2.2: a sound certificate with a $p$-independent PSD block

We replace the dual-function route by a relaxation whose validity is independent of any solver.

**Lemma 2.2.** Let $\mathbf x=(\mathbf v_2(\mathbf q),\mathbf t)\in\mathbb{R}^{13}$, $\hat{\mathbf x}=(\mathbf x,1)$, $\mathbf X=\hat{\mathbf x}\hat{\mathbf x}^\top$, and let $\hat{\mathbf Q}_\theta\in\mathbb{S}^{14}$ be the homogenised matrix of $J_\theta$. Consider
$$
\begin{aligned}
\mathrm{Shor}=\min_{\mathbf X}\ &\langle\hat{\mathbf Q}_\theta,\mathbf X\rangle\\
\text{s.t.}\ &\mathbf X\succeq0,\quad X_{14,14}=1,\quad \langle\operatorname{sym}(\hat{\mathbf e}_{\rm tr}\mathbf e_{14}^\top),\mathbf X\rangle=1,\quad \operatorname{tr}(\mathbf X_{[10]})=1,\\
&\langle\operatorname{sym}(\hat{\mathbf c}_j\mathbf e_{14}^\top),\mathbf X\rangle\le0,\quad j=1,\dots,p,\\
&\text{the linearised Veronese relations } 2X_{ij}=X_{kk}\ (2v_iv_j=v_k^2)\ \text{and}\ S_{ik}S_{jl}=S_{il}S_{jk}\text{ in }X .
\end{aligned}\tag{6}
$$
Then $\mathrm{Shor}\le J^\star$. If the optimiser has rank one, its factor is a globally optimal feasible pose; if $\mathrm{Shor}\ge J(\hat{\mathbf z})-\tau_{\rm sdp}$ and $\hat{\mathbf z}$ is feasible, then $\hat{\mathbf z}$ is $\tau_{\rm sdp}$-globally optimal.

*Proof.* Every feasible $\mathbf z$ yields a feasible rank-one $\mathbf X$ with the same objective, and the Veronese relations hold for every $\mathbf v_2(\mathbf q)$; hence (6) is a relaxation. Rank one gives the converse. $\square$ (Full statement and numerics: `theory/proofs.md` §2.2.)

The practical content is in the numbers. On a synthetic $p=12$ instance (`verify/`, C5):

| relaxation | value | $\lambda_2/\lambda_1$ | comment |
|---|---|---|---|
| no Veronese, no dual term | $-1\times10^{-10}$ | 0.33 | vacuous |
| $+$ Veronese | 0.0030334692 | $1.4\times10^{-9}$ | equals the *unconstrained* QPEP value 0.0030334724, tight |
| $+$ Veronese $+$ inequalities | **0.0680069355** | $2.2\times10^{-12}$ | equals ALM output 0.06800693548, **tight** |
| $\mathrm{Shor}$ of $d(\hat{\boldsymbol\mu})$ | 0.0034784073 | — | only a 0.0645 sound gap; 3 orders worse |

So: the Veronese relations are the switch that makes the relaxation non-vacuous; the inequality-constrained SDP is tight on this instance; and it is a *better* certificate than the dual function. We therefore make (6) the certificate of record and demote $d(\boldsymbol\mu)$ to a diagnostic.

**Why the Veronese relations are legitimate and cheap.** The relations $2v_iv_j=v_k^2$ and $S_{ik}S_{jl}=S_{il}S_{jk}$ (with $\mathbf S=\mathbf q\mathbf q^\top$ reconstructed from $\mathbf v_2$) hold identically on the Veronese variety $\nu_2(\mathbb{RP}^3)$, and they are *linear in $\mathbf X$*, so they cost nothing structurally: they are extra linear equalities on a $14\times14$ matrix. Without them the relaxation is unbounded below (the vector $\mathbf v_2$ is not normalised in the lifted problem) and numerically vacuous.

**Cost.** The PSD block is $14\times14$ for every $p$; the constraints add $p$ linear rows. Interior-point or first-order cost is $O(p\cdot14^3)$ per iteration, versus $O((10+p)^6)$ for the slack-lifted SDP. We solve the certificate asynchronously (once per batch, or once per epoch on a subset) because the forward pass does not need it.

**A rigorous dual repair (Theorem P2-Cert).** The SDP value above is sound only up to the solver tolerance: SCS is not dual-feasible and routinely returns $\mathrm{Shor}>J$, i.e. a *negative* gap. Theorem P2-Cert (full statement and proof: `theory/proofs.md` §2.7; user-supplied proof, 2026-11-27) removes that tolerance from the argument. Write $\mathbf y_{14}=\mathbf F\mathbf y_{13}$ with $\mathbf y_{14}=[\mathbf v_2;\mathbf t;1]$, $\mathbf y_{13}=[\mathbf v_2;\mathbf t]$ and $\mathbf F$ the constant folding map that appends $\mathbf e_{\rm tr}^\top\mathbf v_2=1$. Take the approximate SDP dual and project it to the cone, $\tilde{\boldsymbol\mu}=[\hat{\boldsymbol\mu}]_+\ge0$, $\tilde{\boldsymbol\Gamma}=[\hat{\boldsymbol\Gamma}]_+\ge0$, and assemble
$$
\tilde{\mathbf H}_{13}=\mathbf F^\top\Big(\mathbf Q_{\rm aug}+\sum_j\tilde\mu_j\mathbf A_{g_j}-\sum_{i\le j}\tilde\Gamma_{ij}\mathbf A_{g_ig_j}-\hat\lambda_0\mathbf A_0-\sum_k\hat\lambda_k\mathbf B_k\Big)\mathbf F .
$$
If $\tilde{\mathbf H}_{tt}\succ0$, with $\mathbf S_{10}=\tilde{\mathbf H}_{RR}-\tilde{\mathbf H}_{Rt}\tilde{\mathbf H}_{tt}^{-1}\tilde{\mathbf H}_{Rt}^\top$ and $\delta_{\min}=\min(0,\lambda_{\min}(\mathbf S_{10}))$,
$$
d_{\rm sound}=\hat\lambda_0+\delta_{\min}-\delta_{\rm round}\le f^\star_{\rm global},
$$
where $\delta_{\rm round}\sim10^{-12}\|\mathbf S_{10}\|_F$ is a floating-point safety margin certified by a Rump/Cholesky check. If $\tilde{\mathbf H}_{tt}$ is not positive definite, then with the translation ball $\|\mathbf t\|\le R_t$ and $\delta_{13}=\min(0,\lambda_{\min}(\tilde{\mathbf H}_{13}))$,
$$
d_{\rm sound}=\hat\lambda_0+(1+R_t^2)(\delta_{13}-\delta_{\rm round})\le f^\star_{\rm global}.
$$
The only ingredients are $\tilde\mu,\tilde\Gamma\ge0$, $g_j\le0$, $g_ig_j\ge0$, the Veronese equalities, and $\|\mathbf v_2(\mathbf q)\|^2=1$; the accuracy of the SDP solve never enters. Consequently $J-d_{\rm sound}\ge0$ **by construction**: a negative gap is impossible up to the certified rounding margin $\delta_{\rm round}$ (the synthetic check of Table 5.8b records one raw difference of $-6.7\times10^{-13}$, i.e. inside that margin). This is the bound of record in §5.4–§5.5; the one-sided flag $J-\mathrm{Shor}\le10^{-6}$ is not a bound and is reported only as an upper bound on the sound rate. The same proof yields the structural diagnosis of the six LM-O plane-only failures: when every active constraint shares the normal $\mathbf b_j=-\mathbf n$, all RLT products constrain only $\mathbf n^\top\mathbf t$, leaving a rank-2 pseudo-mixture admissible; a depth constraint ($\mathbf b=-\mathbf e_3$) and a tilt cone ($q_1^2+q_2^2\le\sin^2(\alpha/2)$) add the cross-direction products that cut it; §5.5(vi) tests exactly this prediction on the six most-slack LM-O rows and finds 0/6 certified with the extra rows alone versus 6/6 once the cross-direction products are added.

### 4.5 Theorem 2.3 and Proposition 2.1: FB-reduced implicit differentiation

**The KKT system and why the naive active-set derivative fails.** At a solution $\mathbf z^\star=(\mathbf q^\star,\mathbf t^\star)$ with multipliers $\boldsymbol\mu^\star$, the tangent-space stationarity and complementarity conditions are
$$
\nabla_{\mathbf z}\mathcal L(\mathbf z^\star,\boldsymbol\mu^\star;\theta)=\mathbf 0\in\mathbb{R}^6,\qquad 0\le\mu_j^\star\perp(-g_j^\star)\ge0 . \tag{7}
$$
Differentiating (7) requires $d\mu_j$ as a function of $d\mathbf z$. At a strongly active constraint ($\mu_j>0$, $g_j=0$) the exact relation is $dg_j=0$ (the constraint stays active); at an inactive one, $d\mu_j=0$; at a weakly active one ($\mu_j=g_j=0$) *neither* holds and the relation is empty. Classical active-set differentiation [Amos & Kolter 2017; Magoon et al. 2025] assumes the first two cases and a full-rank active Jacobian. The four-legged table violates the rank assumption; a grazing contact violates strict complementarity.

**Regularisation.** We replace (7)'s complementarity by the smoothed Fischer–Burmeister equation
$$
\varphi_\varepsilon(\mu_j,g_j)=\mu_j-g_j-\sqrt{\mu_j^2+g_j^2+\varepsilon^2}=0 . \tag{8}
$$
We stress what (8) is: for $\varepsilon>0$ it is **not** equivalent to complementarity. Its zero set is exactly $\mu_jg_j=-\varepsilon^2/2$ (Lemma 2.3), and at a complementarity point it evaluates to $-\varepsilon^2/(2\mu_j)$ (strongly active), $-\varepsilon$ (weakly active), $-\varepsilon^2/(2|g_j|)$ (inactive). Equation (8) is a smooth *surrogate* used to define a regularised sensitivity system; the forward solve never solves it.

**Example 2.5 (the 2-D kink, solved in closed form).** For $\min_z\tfrac12(z-a)^2$ s.t. $z\ge0$ the true solution map is $\max(a,0)$, whose Clarke derivative at $a=0$ is $[0,1]$. The FB-regularised problem has the closed-form solution $z_\varepsilon(a)=\tfrac12(a+\sqrt{a^2+2\varepsilon^2})$, so $dz_\varepsilon/da|_{a=0}=1/2$ *exactly, for every $\varepsilon$*: the regularisation selects the midpoint of the Clarke derivative, not one side and not an arbitrary element. For $|a|\gg\varepsilon$ it returns $0$ or $1$ as it should. This is the one case we can characterise completely; whether the same midpoint rule survives multiple simultaneously weak-active constraints and a rank-deficient $\mathbf J_{\mathcal A}$ is the open part (G2).

**Lemma 2.3.** $\partial_{\mu}\varphi_\varepsilon=1-\mu/r=a$, $\partial_g\varphi_\varepsilon=-(1+g/r)=-b$, $r=\sqrt{\mu^2+g^2+\varepsilon^2}$. For $\mu\ge0$: $a\in(0,1]$; for $g\le0$ (feasible) $b\in[0,1]$; for $g>0$ (an infeasible iterate) $b\in(1,2)$. Hence $a_j\to\varepsilon^2/(2\mu_j^2)$ at a strongly active constraint, $b_j\to\varepsilon^2/(2g_j^2)$ at a strictly inactive one, and $a_j=b_j=1$ at a weakly active one. (Proof and numerics: `theory/proofs.md` §3.1; all ranges verified.)

**Reduction.** Let $\mathcal A=\{j:a_j<\delta_{\rm act}\}$ (strongly active) and $\mathcal R$ its complement. For $j\in\mathcal R$ we eliminate $d\mu_j=(b_j/a_j)(\mathbf J_{g,j}d\mathbf z+\partial_\theta g_j\,d\theta)$ and fold it into an effective Hessian; for $j\in\mathcal A$ we keep a small saddle block with a positive diagonal. With $\mathbf H_{\rm tan}=\nabla^2_{\mathbf z}\mathcal L$ and $\mathbf H_{\rm eff}=\mathbf H_{\rm tan}+\sum_{j\in\mathcal R}(b_j/a_j)\mathbf J_{g,j}^\top\mathbf J_{g,j}$,
$$
\underbrace{\begin{bmatrix}\mathbf H_{\rm eff}&\mathbf J_{\mathcal A}^\top\\ \mathbf J_{\mathcal A}&-\mathbf D_a\end{bmatrix}}_{=:\mathbf S}\begin{bmatrix}d\mathbf z\\ d\boldsymbol\mu_{\mathcal A}\end{bmatrix}
=\begin{bmatrix}-\partial_\theta\nabla_{\mathbf z}\mathcal L-\sum_{j\in\mathcal R}\frac{b_j}{a_j}\mathbf J_{g,j}^\top\partial_\theta g_j\\ -\partial_\theta\mathbf g_{\mathcal A}\end{bmatrix},\qquad
\mathbf D_a=\operatorname{diag}_{j\in\mathcal A}\Big(\frac{a_j}{b_j}\Big). \tag{9}
$$

**Theorem 2.3.** If $\mathbf H_{\rm eff}\succ0$ and $\mathbf D_a\succ0$, then $\mathbf S$ in (9) is invertible for **any** $\mathbf J_{\mathcal A}$, including rank-deficient ones. Its Schur complement is $-(\mathbf D_a+\mathbf J_{\mathcal A}\mathbf H_{\rm eff}^{-1}\mathbf J_{\mathcal A}^\top)\prec0$. As $\varepsilon\to0$, $\mathbf D_a\to0$ on strongly active constraints and $\mathbf S$ tends to $\begin{bmatrix}\mathbf H_{\rm eff}&\mathbf J_{\mathcal A}^\top\\ \mathbf J_{\mathcal A}&0\end{bmatrix}$, which is invertible iff $\mathbf J_{\mathcal A}$ has full row rank and $\mathbf H_{\rm eff}$ is positive definite on $\ker\mathbf J_{\mathcal A}$ — i.e. iff LICQ and SOSC hold. The condition number scales as $\kappa(\mathbf S)\asymp1/\varepsilon^2$ when $\mathbf J_{\mathcal A}$ is rank deficient, and is bounded when it is not.

*Proof.* Schur complement: $\mathbf S$ invertible iff $\mathbf D_a+\mathbf J_{\mathcal A}\mathbf H_{\rm eff}^{-1}\mathbf J_{\mathcal A}^\top\succ0$, which holds because $\mathbf D_a\succ0$ and the second term is PSD. The limit and conditioning follow by inspecting the Schur complement. $\square$ (Full proof: `theory/proofs.md` §3.2.)

**Proposition 2.1 (the $\varepsilon\to0$ limit).** With $\mathbf r=-\partial_\theta\nabla_{\mathbf z}\mathcal L$ and $\mathbf w=-\partial_\theta\mathbf g_{\mathcal A}$, for every $\varepsilon>0$ the system (9) has a unique solution, and as $\varepsilon\to0$
$$
d\mathbf z\to d\mathbf z^\star=\mathbf H_{\rm eff}^{-1}\mathbf J_{\mathcal A}^\top\big(\mathbf J_{\mathcal A}\mathbf H_{\rm eff}^{-1}\mathbf J_{\mathcal A}^\top\big)^{+}\mathbf w+\big[\mathbf I-\mathbf P_{\mathcal A}\big]\mathbf H_{\rm eff}^{-1}\mathbf r,\qquad
\mathbf P_{\mathcal A}=\mathbf H_{\rm eff}^{-1}\mathbf J_{\mathcal A}^\top(\cdot)^{+}\mathbf J_{\mathcal A}, \tag{10}
$$
the $\mathbf H_{\rm eff}$-weighted minimum-norm solution of $\mathbf J_{\mathcal A}d\mathbf z=\mathbf w$. If $\mathbf w\in\operatorname{range}(\mathbf J_{\mathcal A})$ (compatibility), $d\mathbf z^\star$ equals the true derivative and is independent of the multiplier distribution; otherwise the multipliers diverge as $\|d\boldsymbol\mu_{\mathcal A}\|\asymp1/\lambda_{\min}(\mathbf D_a)\asymp1/\varepsilon^2$ and $d\mathbf z^\star$ is a *selected* generalized derivative of a solution map that is not differentiable.

*Proof idea.* Eliminate $d\boldsymbol\mu$ from (9) to get $(\mathbf H_{\rm eff}+\mathbf J_{\mathcal A}^\top\mathbf D_a^{-1}\mathbf J_{\mathcal A})d\mathbf z=\mathbf r+\mathbf J_{\mathcal A}^\top\mathbf D_a^{-1}\mathbf w$; split $d\mathbf z$ into the $\mathbf H_{\rm eff}$-orthogonal components along $\operatorname{range}(\mathbf H_{\rm eff}^{-1}\mathbf J_{\mathcal A}^\top)$ and $\ker\mathbf J_{\mathcal A}$; the first component carries the $1/\lambda_{\min}(\mathbf D_a)$ factor, the second is annihilated by $\mathbf J_{\mathcal A}$; the limit is (10). $\square$

**What this means for the four-legged table.** In the linearised kinematics of a flat object resting on a table, the four contact constraints are $g_j=(1,a_j,b_j)^\top(z,\text{roll},\text{pitch})-h_j(\theta)$ with $(a_j,b_j)$ the corner coordinates, so $\mathbf J_{\mathcal A}\in\mathbb{R}^{4\times3}$ has rank 3 and its columns are exactly the coefficient vectors of $(z,\text{roll},\text{pitch})$. If $\theta$ moves the table as a rigid body (height, roll, pitch), then $-\partial_\theta\mathbf g_{\mathcal A}$ is a linear combination of those columns and compatibility holds automatically; the layer returns the true derivative. If the four leg heights are independent parameters, compatibility fails, the solution map has a kink, and the FB limit is one selected generalized derivative with divergent multipliers. Both regimes are demonstrable in seconds (`verify/`, F5): compatibility gives $d\mathbf z\to$ the true value with error $1.3\times10^{-11}$ and bounded multipliers, independent of the multiplier distribution (difference $2.3\times10^{-9}$ across three distributions); incompatibility gives $\|d\boldsymbol\mu\|=10^{-1}\to10^{6}$ as $\mathbf D_a$ goes $10^{-1}\to10^{-8}$ while $\|d\mathbf z\|$ stays at $0.707$.

**Choice of $\mathbf H_{\rm tan}$ (made explicit here, not deferred).** We adopt the **Gauss–Newton** choice
$$
\mathbf H_{\rm tan}:=\nabla^2_{\mathbf z}J_\theta(\mathbf z^\star)\;(\succeq0),
\tag{9a}
$$
i.e. we drop the constraint-curvature term $\sum_{j\in\mathcal A}\mu_j\nabla^2_{\mathbf z}g_j$. With (9a), $\mathbf H_{\rm eff}\succ0$ holds whenever $\nabla^2J_\theta\succ0$, so Theorem 2.3 and Proposition 2.1 apply as stated, with no indefinite-Hessian caveat. The price is explicit and stated in the theorem: (9) then computes the derivative of the Gauss–Newton *surrogate* solution map, which agrees with the true derivative to first order in the residual but not exactly; the discrepancy is $O(\sum_j\mu_j\|\nabla^2g_j\|)$ and is left to the exact-Hessian ablation, which is not run in this draft (§5.11).

**Remark 4.1 (exact-Hessian variant).** If one instead uses the full Lagrangian Hessian $\mathbf H_{\rm tan}=\nabla^2J+\sum_{j\in\mathcal A}\mu_j\nabla^2g_j$, (9) is the exact differentiated KKT system, but $\mathbf H_{\rm eff}\succ0$ is no longer automatic; Theorem 2.3 must be replaced by the weaker statement "$\mathbf S$ is nonsingular iff $\mathbf H_{\rm eff}$ is nonsingular on $\ker\mathbf J_{\mathcal A}$", and a Levenberg damping is needed in practice when it is not. Because $g_j$ is quadratic in $\mathbf q$, $\nabla^2_{\mathbf q}g_j$ is *not* small — it scales with $\|\tilde{\mathbf a}_j\|$ and is multiplied by the contact multiplier — so this is a genuine trade-off rather than a technicality. We leave both variants to the ablation that is not run in this draft (§5.11) and recommend (9a) as the default.

### 4.6 The open mathematical questions this method creates

We list them here because they bound the paper's claims; they are expanded in `math_questions.md`.

**G1 (P0, resolved by Theorem P2-1b).** Is the generic number of complex roots of the inner KKT system still 40 when $\mathbf M^{\rm red}_{\mathcal A}\neq\mathbf M_0$? The Bezout bound is 162. Theorem P2-1b (user-supplied proof, 2026-11-27; `theory/proofs.md` §2.6) settles this: for a rank-9 PnP-type $\mathbf M^{\rm red}_{\mathcal A}$ (our synthetic instance) the answer is **yes, 40**, with the same 15-monomial grevlex staircase — the shift $\mathbf M\mapsto\mathbf M+\alpha\mathbf I$ leaves all critical points fixed. The only collapse occurs for an isotropic 3D–3D (Horn) base with plane constraints sharing one normal, where the Hopf ruled line $\mathcal L_{\mathbf n}$ is a positive-dimensional excess component and the isolated count drops to $4/12/16$ (by $k=0/1/\ge2$), to 32 for two distinct normals. The residual open question is implementation-level, not algebraic: whether to ship one 40-root template or a 16-root saturated template family (§4.7).

**G2 (P0) The generalized derivative selected by FB at weak-active points.** $\varphi_0$ is not differentiable at $(\mu,g)=(0,0)$; its Clarke subdifferential is a set, and the $\varepsilon$-regularised path selects one element ($a=b=1$). We need to characterise that selection and relate it to the conservative-field framework of [Bolte et al. 2021]. Until then the paper may say "a selected generalized derivative", never "the derivative".

**G3 (P1) Numerical soundness of the certificate.** Lemma 2.2 is sound in exact arithmetic. Theorem P2-Cert now supplies the numerical repair: projecting the approximate dual to the non-negative orthant and taking the Schur-complement bound $\hat\lambda_0+\delta_{\min}-\delta_{\rm round}$ gives a bound that is valid for *any* dual-feasible point, so the SDP tolerance disappears from the argument; the Rump/Cholesky check certifies the remaining matrix rounding. What is still empirical is *tightness* (G4), not soundness. The repair is implemented and measured on real data in §5.5.

**G4 (P1) ALM's KKT point may not be the certified optimum.** If the $14\times14$ relaxation is not tight, we have a lower bound and a feasible point but no conclusion. The fraction of instances on which the relaxation is tight is an empirical question and a first-class experimental result (§5.4–§5.5). The rigorous rerun of §5.5 answers it for our real-data protocol.

**G5 (resolved in favour of GN).** The main results use the Gauss–Newton $\mathbf H_{\rm tan}$ of (9a); the exact-Hessian variant is Remark 4.1 and an ablation. The residual open question is quantitative: how large is the GN bias on real data, and does it matter for end-to-end training?

### 4.7 Inner root structure: the $4/12/16/32/40$ spectrum (Theorem P2-1b)

This subsection records *why* the forward inner solve can use a fixed small template, and where that breaks. The full statement, proof sketch and the $\mathbb F_{32003}$ verification note are in `theory/proofs.md` §2.6 (user-supplied proof, 2026-11-27); here we state only what the implementation needs.

Write $N_{\mathbb C}$ for the number of isolated complex projective roots of $\nabla f(\mathbf q)=\lambda\mathbf q$, $\mathbf q^\top\mathbf q=1$ on $\mathbb S^3_{\mathbb C}$, modulo $\mathbf q\leftrightarrow-\mathbf q$; $N_{\mathbb R}\le N_{\mathbb C}$ is the real count.

* **Type I (our synthetic instance).** If $\mathbf M^{\rm red}_{\mathcal A}$ is rank-deficient with $\mathbf M^{\rm red}_{\mathcal A}\mathbf e_{\rm tr}=\mathbf 0$ (PnP-type; rank $9$ at our benchmark noise level, rank $8$ for the noise-free matrix, which adds a second null direction), then $N_{\mathbb C}=40$ and the 15-monomial grevlex staircase of the generic problem is unchanged: the shift $\mathbf M\mapsto\mathbf M+\alpha\mathbf I_{10}$ changes $f$ by $\alpha(\mathbf q^\top\mathbf q)^2$, hence $\nabla f$ by $4\alpha(\mathbf q^\top\mathbf q)\mathbf q$, so every critical point survives with $\lambda\mapsto\lambda+4\alpha$. A homotopy that finds only 15–18 **real** roots is therefore behaving as expected; the remaining roots are non-real (or lie outside the affine chart $\mathbf q_0=1$). Our own homotopy study (`experiments/results/quartic_validation.json`, 24 P2 instances) confirms both halves of this reading: all 40 finite projective roots are found (0 lost) while the real roots number 16 (20 instances) or 18 (4 instances); the generic random-$\mathbf M$ control gives 40/40/40/40/40. The user proof quotes 15–16 real roots from an earlier run; the difference does not change the conclusion.
* **Type II (isotropic 3D–3D + plane constraints sharing a normal).** With $\mathbf w(\mathbf q)=\mathbf R(\mathbf q)^\top\mathbf n$ and $\|\mathbf w\|^2=(\mathbf q^\top\mathbf q)^2$, the reduced objective is $(\mathbf q^\top\mathbf q)(\mathbf q^\top\mathbf K\mathbf q)+\mathbf w^\top\boldsymbol\Sigma_{\mathcal A}\mathbf w$; the pure quartic part has effective rank $\le\min(k,2)$, and the Hopf ruled line $\mathcal L_{\mathbf n}=\{\mathbf q^\top\mathbf q=0,\ \mathbf w=0\}\cong\mathbb{CP}^1\subset\mathcal Q$ is a positive-dimensional excess component. Saturating it (Rabinowitsch, over $\mathbb F_{32003}$) gives the isolated counts
$$
k=0:\ N_{\mathbb C}=4\ (8\text{ affine, Davenport }4\times4);\quad
k=1:\ N_{\mathbb C}=12\ (24);\quad
k\ge2:\ N_{\mathbb C}=16\ (32,\ 17\text{-element Gröbner},\ 16\times16\text{ action matrix});
$$
two distinct normals give $N_{\mathbb C}=32$ (64 affine), and an anisotropic/PnP base with any plane gives $N_{\mathbb C}=40$ (80 affine, the generic 15-monomial template).

**Numerical confirmation on the ALM assembly (roots-subagent measurement).** Independently of the symbolic proof, the committed ALM assembly (`isotropic.alm_reduced`) reproduces the spectrum: its reduced quartic has relative residual $\sim10^{-15}$ to the 15-dimensional canonical family $F_n$, with PSD-normalised $\operatorname{rank}\boldsymbol\Sigma_{\mathcal A}=0/1/2/2/2$ for $k=0,\dots,4$, and a census that re-polishes paths the tracker marks divergent reads $4/12/16/16/16$ projective ($8/24/32/32/32$ affine), matching the theorem (`results/isotropic_spectrum.md`). The committed enumerator reads 12 at $k=2$ because the four extra roots sit at $\|\mathbf q\|\approx29$–$34$ and their homotopy paths cross the divergence tolerance in tracking; they are genuine solutions (KKT relative residual $\sim10^{-16}$), so this is a census artefact, not a different spectrum. Two distinct normals with one active constraint each read 32/64; with two per normal on one side (2+1) they read 36/72, and with two per normal on both sides (2+2) they return to the generic 40/80 — so the theorem's "32 for two distinct normals" is the one-active-constraint-per-normal case. Type I stays at 40/80 with a residual of $7\times10^{-2}$ (anisotropic Horn) or $6\times10^{-1}$ (PnP) to $F_n$, which makes the residual itself a usable structural discriminator. *Source: `results/isotropic_spectrum.json` / `.md`; `theory/proofs.md` §2.6.6.*

**Implementation consequence, stated precisely.** The inner enumerator in this paper is template-based only for Type I: it is the **40-root un-saturated template**, and it is the one exercised end-to-end (§5.6, 114 enumerations). For a Type II instance the un-saturated 40-root template is the wrong template, not a numerical bug: the action matrix acquires a zero-eigenvalue cluster from the excess line. The *saturated* objects of the theorem — the $4\times4$ Davenport problem for $k=0$, the 12-root system for $k=1$ and the $16\times16$ saturated action matrix for $k\ge2$ — are established **algebraically in the user proof** ($\mathbb F_{32003}$ Rabinowitsch saturation, §2.6.1–§2.6.2). What this repository adds numerically is only the $4/12/16$ **spectrum** in canonical form (`theory/proofs.md` §2.6.5, `results/p2b_theorem_check.*`); the $16\times16$ action matrix itself is **not built as a shipped numeric template**, and `experiments/code/isotropic_roots.py` solves Type II instances with the 162-path homotopy plus a multistart fallback. Status: **40-root template implemented and validated; $4/12/16$ spectrum validated numerically in canonical form; 16-root saturated template partial/absent**. This is why §5.3–§5.4 report the spectrum explicitly and why the exact-inner table (§5.6) is run only on the PnP-type instances.

---

## 5 Experiments

We report a layer-level study plus a small CPU end-to-end training toy, not a trained perception system. §5.1 fixes the protocol, the metrics and the evidence grades; §5.2–§5.5 give the synthetic and real manipulation measurements; §5.6 reports the measured baselines already in hand (exact inner enumeration, SCP, sound RLT); §5.7 compares against the neighbouring layer families (our SDPRLayers-style relaxation, our dQP/CVXPYLayers-style layer, and the official Theseus forward solver); §5.8 is the end-to-end toy, where a keypoint refiner is trained through four pose heads and the FB backward is checked against finite differences; §5.9 adds the localisation layers (KITTI odometry, UTIAS In-The-Dark); §5.10 reports ADD/ADD-S on the same real manipulation trials; §5.11 states exactly what is not run in this draft and why. Two evidence grades are used: *independently reproduced* (produced by code in this repository and traceable to a result file) and *prototype sanity check* (an algebra or gradient check in `verify/` and the toy, not a benchmark result). No number in §5 is a *paper-claimed* number, and the real-data numbers are layer-level (§5.1.4).

### 5.1 Setup, protocol, and evidence grading

#### 5.1.1 Synthetic desktop instance

The synthetic instance is a controlled layer test, not a benchmark: $N=14$ 3D–2D correspondences with a bearing-type algebraic cost, per-point noise $\sigma$, a systematic extrinsic bias, and a ground-truth pose that is exactly in contact with the table. The bias always contains a fixed 5 mm lateral component (`bias_xy=0.005` m) in addition to the vertical bias of the cell; this is why even the noise-free cell has a non-zero translation error (§5.3). The constraint count $p\in\{8,16,32,64\}$ is the number of *scalar* inequality rows assembled from four families: planar non-penetration, positive depth, field of view (recorded as pairs, each pair contributing two rows) and a tilt cone. Table 5.1 gives the composition. Each cell has 50 trials; the seed is the trial index, so the same scenes are reused across $p$ and across noise levels.

**Table 5.1. Constraint composition (scalar rows) of the synthetic instance.**

| $p$ | non-penetration | positive depth | FoV (rows) | tilt cone | total |
|---|---|---|---|---|---|
| 8 | 5 | 2 | 0 | 1 | 8 |
| 16 | 9 | 4 | 2 | 1 | 16 |
| 32 | 17 | 8 | 6 | 1 | 32 |
| 64 | 33 | 12 | 18 | 1 | 64 |

*Source: `results/raw_main.json` (`config.split`, and `recs[].actypes`, which agree on every record in a cell). The configuration file records FoV as 1/3/9 pairs; each pair is two scalar rows.*

#### 5.1.2 Methods and the inner solver

Four methods are compared on the same instances:

1. **Unconstrained QPEP**: the same bearing cost with only $\|\mathbf q\|=1$, solved numerically from 48 random starts; no constraints.
2. **Soft penalty**: the quadratic penalty $\lambda\sum_j[g_j]_+^2$ with $\lambda=10^3$ and 14 random starts; the weight is swept in §5.3.
3. **Post-hoc projection**: the unconstrained solution is projected onto the feasible set with SLSQP.
4. **ALM-QPEP (ours)**: the augmented Lagrangian of §4.1 with a frozen active set, the $p$-independent assembly of Theorem 2.1, and the consistency check of Lemma 2.1.

All four use the same numerical inner solve — a multi-start BFGS on the unit quaternion sphere — and *not* QPEP's polynomial root enumeration. This is the main caveat on the word "global" in this section: the certificate of §5.4 is global, the forward solve is numerical. A start-count baseline on 48 inner instances ($p\in\{8,16,32,64\}$, 6 seeds, $\sigma=0.01$, bias $0.08$; source: `results/inner_baseline.json`) quantifies the stand-in: 4 starts under-solve 8/24 unconstrained matrices with a maximum objective gap of $1.17\times10^{-3}$; 8 starts reach the reference on all 48 instances at a median 26 ms; 256 starts cost 678 ms ($25\times$) with no objective gain. For the ALM active-set matrix, even 4 starts reach the reference on all 24 instances. The 48-start setting of method 1 is therefore above the threshold found here; it still does not certify the inner minimum. The exact 162-path enumerator is now measured as a baseline (§5.6): it reaches the same ALM objective to $1.8\times10^{-12}$ at a median $47\times$ the multistart cost, so the default remains multistart.

#### 5.1.3 Metrics and evidence grades

**Table 5.2. Metrics and pass criteria (from `experiments/protocol.md` §3).**

| metric | definition | reported as |
|---|---|---|
| penetration | $d_{\rm pen}=\max_{v\in\mathcal V}[\mathbf n^\top(\mathbf R\mathbf v+\mathbf t)+d]_-$ (mm) | median [IQR]; penetration rate = fraction with $d_{\rm pen}>1$ mm |
| rotation / translation error | geodesic angle to GT / Euclidean translation error | median [IQR], deg / mm |
| feasible | $\max_j[g_j]_+\le10^{-6}$ | rate |
| certified | feasible **and** $J-\mathrm{Shor}\le10^{-6}$ **and** $\lambda_2/\lambda_1\le10^{-6}$ | rate |
| runtime | wall-clock per instance (forward solve); certificate timed separately | median, ms |

Medians [IQR] are reported instead of means because a small number of trials fall into mirrored or bad local minima and inflate the mean; the means and standard deviations are in the JSON. Runtime is machine-specific and is used only for scaling comparisons, not as a hardware claim. The evidence grades are: *independently reproduced* for §5.2–§5.7 and §5.9–§5.10 (each table carries a source line); *prototype sanity check* for §5.1.5 and for the §5.8 CPU toy, which is a closed-loop training check rather than a benchmark. §5.11 lists every protocol that is not run.

#### 5.1.4 Real-data protocol and planned protocols

The real experiments use BOP LINEMOD, LINEMOD-Occlusion and YCB-Video test frames with the official object meshes and ground-truth poses. For each frame the support plane is fitted by RANSAC to the background depth (5 mm inlier threshold), with the normal pointing into the object, and its offset is calibrated so that the ground-truth lowest vertex is in contact. The observations are 30 random model vertices projected with the ground-truth pose and camera matrix, perturbed by 1 px of noise and a downward pixel bias of 10/20/30 px; bearings are obtained with $\mathbf K^{-1}$. The constraints are the 16 lowest model vertices along the table normal (non-penetration), so $p=16$, with the same four methods.

There are 90 (LINEMOD), 72 (LINEMOD-Occlusion) and 84 (YCB-Video) trials: 30, 24 and 28 (frame, image, object) poses respectively, each under three pixel biases. These are not independent video frames. **The correspondences are ground-truth-projected, not produced by a detector or a matcher: this is a layer experiment, not an end-to-end perception result.** The plane offset is calibrated to the ground truth, so the ground truth is feasible by construction.

**Protocols run at layer level, and protocols not run.** The T2 localisation protocols are run at layer level in §5.9: KITTI odometry with a roll/pitch tilt cone ($\alpha=10^\circ$) and positive depth on all landmarks, and UTIAS In-The-Dark with positive depth and image-border (FoV) half-spaces on stereo-triangulated landmarks. Neither is a full VO/SLAM system, and neither uses a detector: the observations are ground-truth projections of known structure (§5.9). The following protocols of `experiments/protocol.md` remain **not run**. (i) *T1-network*: a keypoint-regression backbone in the SDPRLayers/BPnP style replaces its pose head with each layer under comparison; the same backbone, training schedule and pose-only supervision (no penetration term in the loss) are used for all layers, and the metrics are ADD(-S), ADD(-S) below 2 cm, penetration depth/ratio and forward/backward latency on YCB-Video and LINEMOD. (ii) *TUM RGB-D*: a second indoor localisation dataset, redundant with KITTI/UTIAS at this scope. (iii) *T3*: hand–eye calibration and point-to-plane registration with a joint-limit (tilt-cone) prior, to show equality constraints in the lifted form. None of these runs is included in §5.2–§5.5; §5.11 states them again with the reason.

#### 5.1.5 Prototype sanity checks (completed; **not** paper results)

> **Evidence grade: prototype sanity check.** Everything in this subsection is a numerical check of the algebra on a synthetic $p=12$ instance, produced by the two scripts in `verify/`. It is *not* an experiment on a benchmark, it does not use the final polynomial solver, and it must not be cited as a result of the paper. It exists to (i) catch algebra errors in Theorems 2.1–2.3 and (ii) demonstrate that the certificate construction of Lemma 2.2 works at all.

Two scripts in `verify/` validate the algebra and the numerics of §4 on a synthetic $p=12$ instance (PnP-type QPEP objective, 12 correspondences, 8 table-corner non-penetration constraints at the ground-truth contact height, 4 positive-depth constraints, 15 mm per-point bearing noise plus a 10 cm systematic bearing bias to force the unconstrained solution into the table). All arithmetic is double precision; the inner solve uses multi-start BFGS on $\mathbb{S}^3$ as a *stand-in* for the QPEP polynomial solver, so the timings are not representative of the final layer.

| id | what is checked | result |
|---|---|---|
| C1 | $\mathbf R(\mathbf q)\mathbf p$ affine in $\mathbf v_2$; all four constraint families affine in $(\mathbf v_2,\mathbf t)$ | max error $4.3\times10^{-15}$ / $3.7\times10^{-15}$ |
| C2 | augmented-Lagrangian active-set expansion equals direct evaluation | $3.6\times10^{-15}$ (with the correct $\kappa_{\mathcal A}$; $5.6\times10^{-1}$ with the naive one) |
| C3 | $3\times3$ Schur complement; inner objective quartic in $\mathbf q$ | $6.2\times10^{-15}$; quartic fit relative error $2.0\times10^{-16}$ |
| C4 | coefficient dimensions independent of $p$ | $p=256$: still $(10,10)/(3,10)/(3,3)$; assembly 4.4 ms single-core |
| C5 | certificate behaviour | $\delta=J(\hat z)-d(\hat\mu)=-1.67\times10^{-11}$ on this instance (**negative** ⇒ not a certificate); $14\times14$ Shor $+$ Veronese $+$ inequalities $=0.0680069355$ vs $J(\hat z)=0.06800693548$, $\lambda_2/\lambda_1=2.2\times10^{-12}$ (**tight**) |
| C6 | ALM forward | feasible to $2.4\times10^{-11}$ in 30 outer iterations; penetration $18.94\text{ cm}\to2.4\times10^{-9}\text{ cm}$ |
| F1–F3 | FB ranges and zero set | $a\in[0,1]$, $b\le1$ feasible / $<2$ infeasible; $\mu g=-\varepsilon^2/2$ to $1.8\times10^{-15}$ |
| F4 | saddle invertibility and conditioning | invertible for 200/200 random rank-deficient cases; $\log\kappa/\log\varepsilon=-2.00$ |
| F5 | $\varepsilon\to0$ limit | compatible: $d\mathbf z\to$ true, error $1.3\times10^{-11}$, multiplier-independent; incompatible: $\|d\boldsymbol\mu\|=10^{-1}\to10^{6}$, $\|d\mathbf z\|$ finite |
| F6 | four-leg Jacobian | rank $3<4$; limit matrix rank $6/7$ |
| F7 | analytic kink: $\min\tfrac12(z-a)^2$ s.t. $z\ge0$ at $a=0$ | true map $\max(a,0)$ non-differentiable; Clarke derivative $[0,1]$; **FB gives exactly $1/2$**, independent of $\varepsilon$ (17/17 checks pass) |

*Source: `verify/` scripts and their outputs; retained verbatim from the previous draft and not re-measured in this round. No number in this subsection is used for a paper claim.*

Two honest observations. (i) The constrained pose error on this instance is *not* better than the unconstrained one ($2.24^\circ/9.96$ cm versus $1.38^\circ/9.44$ cm): the constraint removes 18.94 cm of penetration but the residual pose error is dominated by the injected bearing bias, which the table constraint cannot correct. This is a warning against assuming that physical constraints improve accuracy; the claim has to be measured per dataset. (ii) The certificate story inverts the earlier internal draft: the dual gap is unusable, the $14\times14$ relaxation is tight.

### 5.2 Synthetic main table: hard feasibility from $p=8$ to $p=64$

Tables 5.3–5.5 give penetration, pose error and rates/runtime for the four methods at the four constraint counts. All 50 trials per cell use the same scenes.

**Table 5.3. Maximum penetration depth (mm, median [IQR]) with the penetration rate ($d_{\rm pen}>1$ mm) in parentheses.**

| method | $p=8$ | $p=16$ | $p=32$ | $p=64$ |
|---|---|---|---|---|
| Unconstrained QPEP | 103.02 [84.69, 110.83] (100%) | 103.02 [84.69, 110.83] (100%) | 103.02 [84.69, 110.83] (100%) | 103.02 [84.69, 110.83] (100%) |
| Soft penalty ($\lambda{=}10^3$) | 0.01 [0.00, 0.06] (0%) | 0.00 [0.00, 0.03] (0%) | 0.00 [0.00, 0.03] (0%) | 0.00 [0.00, 0.02] (0%) |
| Post-hoc projection | 0.00 [0.00, 0.00] (0%) | 0.00 [0.00, 0.00] (0%) | 0.00 [0.00, 0.00] (0%) | 0.00 [0.00, 0.00] (0%) |
| ALM-QPEP (ours) | 0.00 [0.00, 0.00] (0%) | 0.00 [0.00, 0.00] (0%) | 0.00 [0.00, 0.00] (0%) | 0.00 [0.00, 0.00] (0%) |

*Source: `results/raw_main.json`, 50 trials per cell; aggregate reproduced by `results/main_table.md`/`main_table.json`.*

**Table 5.4. Pose error (rotation deg / translation mm, median [IQR]).**

| method | $p=8$ | $p=16$ | $p=32$ | $p=64$ |
|---|---|---|---|---|
| Unconstrained QPEP | 5.30 [3.26, 6.70] / 97.90 [88.42, 107.73] | 5.30 [3.26, 6.70] / 97.90 [88.42, 107.73] | 5.30 [3.26, 6.70] / 97.90 [88.42, 107.73] | 5.30 [3.26, 6.70] / 97.90 [88.42, 107.73] |
| Soft penalty ($\lambda{=}10^3$) | 4.86 [3.73, 7.78] / 10.93 [9.26, 16.06] | 6.62 [3.70, 9.95] / 13.00 [8.69, 16.87] | 6.97 [3.94, 9.61] / 12.33 [10.19, 16.11] | 6.01 [4.14, 9.11] / 12.40 [8.99, 15.76] |
| Post-hoc projection | 4.16 [2.90, 5.34] / 9.10 [7.34, 10.94] | 4.16 [2.90, 5.34] / 9.10 [7.34, 10.94] | 4.16 [2.90, 5.34] / 9.10 [7.34, 10.94] | 4.16 [2.90, 5.34] / 9.10 [7.34, 10.94] |
| ALM-QPEP (ours) | 4.17 [2.74, 5.10] / 10.63 [8.17, 12.69] | 4.17 [2.74, 5.10] / 10.63 [8.17, 12.69] | 4.17 [2.74, 5.10] / 10.63 [8.17, 12.69] | 4.17 [2.74, 5.10] / 10.63 [8.17, 12.69] |

*Source: `results/raw_main.json`, 50 trials per cell.*

**Table 5.5. Feasibility, certification and runtime (median ms); the last row is the $14\times14$ certificate.**

| method | $p=8$ | $p=16$ | $p=32$ | $p=64$ |
|---|---|---|---|---|
| Unconstrained QPEP | feas 0% / cert 0% / 143 ms | feas 0% / cert 0% / 144 ms | feas 0% / cert 0% / 144 ms | feas 0% / cert 0% / 142 ms |
| Soft penalty ($\lambda{=}10^3$) | feas 42% / cert 0% / 96 ms | feas 38% / cert 0% / 94 ms | feas 42% / cert 0% / 98 ms | feas 50% / cert 0% / 97 ms |
| Post-hoc projection | feas 100% / cert 0% / 12 ms | feas 100% / cert 0% / 12 ms | feas 100% / cert 0% / 12 ms | feas 100% / cert 0% / 12 ms |
| ALM-QPEP (ours) | feas 100% / cert 100% / 260 ms | feas 100% / cert 100% / 277 ms | feas 100% / cert 100% / 280 ms | feas 100% / cert 100% / 288 ms |
| **14×14 SDP** | tight 100% / 51 ms | tight 100% / 56 ms | tight 100% / 67 ms | tight 100% / 84 ms |

*Source: `results/raw_main.json`, 50 trials per cell; runtime is the median wall-clock per instance, SDP time is the median certificate time.*

Six observations.

**(i) Hard feasibility is the separating property.** The unconstrained solution penetrates the table in 100% of trials at every $p$ (median 103.02 mm). ALM returns 0.00 mm in 100% of trials, and its maximum constraint violation over the 200 main trials is $1.0\times10^{-10}$, four orders below the $10^{-6}$ tolerance. The soft penalty reaches a sub-millimetre *median* (0.00–0.01 mm) but is feasible in only 38–50% of trials (maximum violation 0.81 mm): "near-zero median penetration" is not feasibility. Projection is 100% feasible and the cheapest method (12 ms), but its objective is above the constrained optimum (median $J-\mathrm{Shor}=2.9\times10^{-4}$ at $p=8$), so it is never certified.

**(ii) The constrained solve is certified at every $p$.** ALM is 100% feasible and 100% certified for $p=8,16,32,64$; the certificate is what projection and the penalty cannot deliver (Table 5.5).

**(iii) $p$ saturates because planar contact is redundant beyond three non-collinear points.** The non-penetration block grows from 5 to 33 rows, yet the per-seed ALM pose changes by at most $5.9\times10^{-7}$ deg and $5.6\times10^{-7}$ mm between $p=8$ and $p=64$ (projection: $1.7\times10^{-5}$ deg and $4.1\times10^{-5}$ mm); the medians of the unconstrained, projection and ALM rows in Tables 5.3–5.5 are identical at the printed precision. The ALM active set nevertheless grows with $p$ (median 2 at $p=8$ to 4.5 at $p=64$; maximum 33), i.e. the extra contacts are weakly active — exactly the LICQ-degenerate regime — and the certificate still holds. The soft penalty is the exception: redundant contacts each contribute a penalty term, so its medians move by up to $2.1^\circ$ and $2.1$ mm across $p$.

**(iv) The cost is consistent with $O(p)$ assembly plus a $p$-independent inner dimension.** ALM runtime grows from 260 to 288 ms over an $8\times$ increase in $p$ (log–log slope 0.045 on medians, 0.031 on means), and the SDP certificate from 51 to 84 ms. Assembly was not instrumented separately: this is an end-to-end consistency check, not a measurement of the $O(p)$ term itself.

**(v) Constraints buy admissibility, not accuracy.** ALM's pose error is 4.17°/10.63 mm at every $p$, projection's is 4.16°/9.10 mm: essentially the same rotation and 1.5 mm better translation for projection, while the (infeasible) unconstrained translation error is 97.90 mm. The soft penalty is worse at every $p$ (up to 6.97°/13.00 mm), and its medians hide catastrophic local minima: one seed in 50 reaches 156.5° at $p=16$, 171.6° at $p=32$ and 84.8° at $p=64$ (translation up to 633.6 mm). The penalty path is therefore not dependable even where its median looks good.

**(vi) A caveat on the medians.** The unconstrained method's occasional cheirality-mirror solutions (2% of trials: the object is placed behind the camera) are excluded from the medians but present in the means; the same is true of the soft-penalty local minima in (v). All rates in Table 5.5 are computed over all 50 trials.

![Figure 5.1](../experiments/results/fig_p2_overview.png)

**Figure 5.1. P2 overview.** (a) Feasibility (bars) and certification (filled diamonds) versus $p$ for the four methods, 50 trials per cell. (b) Maximum penetration depth, median [IQR], versus $p$ on a symlog axis (projection and ALM coincide at 0.00). (c) Real-data ALM certification at $n=90/72/84$: the one-sided Shor flag (an upper bound, grey), the two-sided $|J-\mathrm{Shor}|$ test (necessary only, blue) and the rigorous P2-Cert Schur bound (green, base cuts / dark green, $+$bound-RLT), with 95% Wilson intervals. (d) Same-instance SDP cost of the hard-constrained $14\times14$ relaxation used by this paper (green) versus the SDPRLayers-style slack lifting $(14+p)\times(14+p)$ (purple), median wall-clock over the per-$p$ records; the block dimension is printed under each group. *Sources: `results/raw_main.json` (a,b); `real/*/results/raw.json` (c one-sided/two-sided); `results/rigorous_real.json` (c rigorous); `results/baseline_sdpr_synth_p{8,16,32,64}.json` (d, re-implementation of the SDPRLayers forward relaxation).*

### 5.3 Noise, extrinsic bias, and the price of a penalty

Table 5.6 fixes $p=16$ and varies the per-point noise $\sigma$ and the vertical bias over four cells. Table 5.7 is a separate 50-trial sweep of the penalty weight $\lambda$ at $p=16$.

**Table 5.6. Method × noise/extrinsic bias at $p=16$ (50 trials per cell; median [IQR]).**

| cell ($\sigma$ / bias) | method | pen (mm) | rot (deg) | trans (mm) | feasible | certified |
|---|---|---|---|---|---|---|
| 0.000 / 0.000 | Unconstrained QPEP | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 7.07 [7.07, 7.07] | 100% | 100% |
| 0.000 / 0.000 | Soft penalty ($\lambda{=}10^3$) | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 7.07 [7.07, 7.07] | 100% | 100% |
| 0.000 / 0.000 | Post-hoc projection | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 7.07 [7.07, 7.07] | 100% | 100% |
| 0.000 / 0.000 | ALM-QPEP (ours) | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 7.07 [7.07, 7.07] | 100% | 100% |
| 0.005 / 0.040 | Unconstrained QPEP | 47.52 [38.74, 51.36] | 2.52 [1.58, 3.25] | 45.87 [40.56, 51.75] | 0% | 0% |
| 0.005 / 0.040 | Soft penalty ($\lambda{=}10^3$) | 0.01 [0.00, 0.03] | 5.10 [3.45, 7.36] | 9.46 [8.10, 11.27] | 24% | 0% |
| 0.005 / 0.040 | Post-hoc projection | 0.00 [0.00, 0.00] | 2.18 [1.75, 3.16] | 7.79 [7.12, 8.58] | 100% | 0% |
| 0.005 / 0.040 | ALM-QPEP (ours) | 0.00 [0.00, 0.00] | 2.52 [1.89, 3.33] | 8.46 [7.10, 9.65] | 100% | 100% |
| 0.010 / 0.080 | Unconstrained QPEP | 103.02 [84.69, 110.83] | 5.30 [3.26, 6.70] | 97.90 [88.42, 107.73] | 0% | 0% |
| 0.010 / 0.080 | Soft penalty ($\lambda{=}10^3$) | 0.00 [0.00, 0.03] | 6.62 [3.70, 9.95] | 13.00 [8.69, 16.87] | 38% | 0% |
| 0.010 / 0.080 | Post-hoc projection | 0.00 [0.00, 0.00] | 4.16 [2.90, 5.34] | 9.10 [7.34, 10.94] | 100% | 0% |
| 0.010 / 0.080 | ALM-QPEP (ours) | 0.00 [0.00, 0.00] | 4.17 [2.74, 5.10] | 10.63 [8.17, 12.69] | 100% | 100% |
| 0.020 / 0.120 | Unconstrained QPEP | 198.10 [177.14, 232.79] | 11.50 [7.92, 18.63] | 185.77 [170.38, 215.92] | 0% | 0% |
| 0.020 / 0.120 | Soft penalty ($\lambda{=}10^3$) | 0.00 [0.00, 0.04] | 11.13 [6.89, 14.36] | 17.01 [10.99, 21.60] | 38% | 0% |
| 0.020 / 0.120 | Post-hoc projection | 0.00 [0.00, 0.00] | 7.97 [5.45, 11.86] | 12.61 [8.90, 17.11] | 100% | 0% |
| 0.020 / 0.120 | ALM-QPEP (ours) | 0.00 [0.00, 0.00] | 5.83 [4.49, 7.55] | 13.39 [10.23, 17.43] | 100% | 100% |

*Source: `results/raw_main.json`, `agg_noise`, 50 trials per cell.*

**Table 5.7. Soft-penalty Pareto at $p=16$ (separate 50-trial sweep; medians).**

| $\lambda$ | pen (mm) | rot (deg) | trans (mm) | feasible |
|---|---|---|---|---|
| $10^0$ | 8.72 | 4.34 | 13.32 | 0% |
| $10^1$ | 2.39 | 5.74 | 11.24 | 0% |
| $10^2$ | 0.22 | 6.24 | 12.00 | 26% |
| $10^3$ | 0.00 | 6.61 | 12.53 | 50% |
| $10^4$ | 0.00 | 6.85 | 12.94 | 48% |

*Source: `results/raw_main.json`, `pareto` (250 trials = 5 weights × 50 seeds); values agree with `results/main_table.md`.*

Three observations.

**(i) The noise-free cell isolates the injected bias.** With $\sigma=0$ and zero vertical bias, all four methods coincide exactly: 0.00 mm penetration, 0.00°, and 7.07 mm translation, all 100% feasible and certified. The 7.07 mm is the fixed 5 mm lateral bias ($\sqrt{5^2+5^2}=7.07$ mm), not a method difference; with no noise the constraints are inactive.

**(ii) Noise and bias separate feasibility from accuracy.** Across the three non-zero cells the unconstrained penetration grows 47.52 → 198.10 mm and it is never feasible; ALM stays at 0.00 mm with 100% feasibility and certification in all four cells; the soft penalty is feasible in only 24–38% of the three non-zero cells and is never certified there; projection is 100% feasible there and never certified. Pose errors are comparable across the three constrained methods at low noise. At the highest cell ALM has the best rotation error (5.83° versus 11.50° unconstrained, 11.13° soft, 7.97° projection) but projection has the best translation (12.61 mm versus ALM 13.39 mm); the differences are within the spread. The robust effect is admissibility and certification, not accuracy — the same conclusion as §5.2.

**(iii) A larger penalty lowers the median penetration but never reaches feasibility.** Table 5.7: median penetration falls monotonically from 8.72 mm at $\lambda=10^0$ to 0.00 mm at $\lambda=10^3$, at the cost of a rotation error rising from 4.34° to 6.85° (translation stays flat, 11.24–13.32 mm). Feasibility never exceeds 50% and is non-monotone (50% at $\lambda=10^3$, 48% at $10^4$): the penalty approaches hard feasibility but does not attain it. One seed in 50 falls into a bad local minimum for every $\lambda\ge10$; at $\lambda=10^4$ its rotation error is 132.9° and its translation 202.6 mm, and at $\lambda=10^3$ its translation error is 504.6 mm. These outliers are invisible in the medians and are the reason the penalty path cannot be certified.

**Which inner root count applies (Theorem P2-1b).** All synthetic cells above use a PnP-type base, i.e. Type I of Theorem P2-1b: the reduced matrix is rank-deficient (rank $9$ at $\sigma=0.01$, rank $8$ at $\sigma=0$) but the inner KKT system still has $N_{\mathbb C}=40$ complex projective roots with the generic 15-monomial grevlex staircase, because the shift $\mathbf M\mapsto\mathbf M+\alpha\mathbf I$ changes $f$ only by $\alpha(\mathbf q^\top\mathbf q)^2$ and leaves every critical point in place. The homotopy finding only 16–18 *real* roots per instance (`results/quartic_validation.json`; the user proof quotes 15–16 from an earlier run) is the expected Type-I picture, not a missing-template symptom. The compact families are different: an isotropic 3D–3D (Horn) base with $k$ active plane constraints sharing one normal collapses to $N_{\mathbb C}=4/12/16$ for $k=0/1/\ge2$ (the Hopf ruled line $\mathcal L_{\mathbf n}$ is an excess component that must be saturated), to $32$ for two distinct normals, and back to $40$ for an anisotropic/PnP base with any plane. Operationally, only **one** template is shipped and exercised end-to-end — the 40-root un-saturated template for the generic/PnP case. The 16-root *saturated* template for the isotropic single-plane case ($k\ge2$; 12-root for $k=1$, $4\times4$ Davenport for $k=0$) is established **algebraically in the user proof only**: the $4/12/16$ spectrum is confirmed *numerically in canonical form* (the theorem's standard form (2.7), swept over $\operatorname{rank}\boldsymbol\Sigma_{\mathcal A}$; `theory/proofs.md` §2.6.5 and `results/p2b_theorem_check.*`), while the $16\times16$ saturated action matrix is **not built as a shipped numeric template in this repository**. The module that owns this family (`experiments/code/isotropic_roots.py`) solves instances with the 162-path homotopy plus a multistart fallback, not with a 16-root template. We therefore do **not** claim "fixed small template in two sizes": the claim is "the 40-root template is implemented and validated on the Type-I instances; the $4/12/16$ spectrum is validated numerically in canonical form; a production 16-root numeric template is partial/absent". The exact-enumeration timings in §5.6 are measured on the Type-I instances only.

### 5.4 How tight is the certificate?

**Table 5.8. Certificate margins over all 400 synthetic instances (8 cells × 50).**

| cell | $n$ | $\max\lambda_2/\lambda_1$ | median $\lambda_2/\lambda_1$ | $\mathrm{Shor}-J_{\rm ALM}$ range | SDP time (mean, ms) |
|---|---|---|---|---|---|
| main $p=8$ | 50 | $1.42\times10^{-9}$ | $9.07\times10^{-11}$ | $[-4.56, 4.29]\times10^{-9}$ | 52.9 |
| main $p=16$ | 50 | $1.03\times10^{-9}$ | $9.47\times10^{-11}$ | $[-3.28, 7.52]\times10^{-9}$ | 57.6 |
| main $p=32$ | 50 | $8.43\times10^{-10}$ | $2.94\times10^{-11}$ | $[-5.66, 5.99]\times10^{-9}$ | 70.0 |
| main $p=64$ | 50 | $1.51\times10^{-9}$ | $9.56\times10^{-11}$ | $[-3.93, 9.41]\times10^{-9}$ | 85.5 |
| noise $0.000/0.000$ | 50 | $5.55\times10^{-10}$ | $1.93\times10^{-11}$ | $[-8.78, 3.57]\times10^{-9}$ | 62.5 |
| noise $0.005/0.040$ | 50 | $8.56\times10^{-10}$ | $7.65\times10^{-11}$ | $[-6.31, 2.67]\times10^{-9}$ | 61.2 |
| noise $0.010/0.080$ | 50 | $1.03\times10^{-9}$ | $9.47\times10^{-11}$ | $[-3.28, 7.52]\times10^{-9}$ | 56.7 |
| noise $0.020/0.120$ | 50 | $1.08\times10^{-9}$ | $2.28\times10^{-11}$ | $[-4.25, 6.82]\times10^{-9}$ | 57.6 |

*Source: `results/cert_margins.json` (summarised from `results/raw_main.json`).*

The synthetic relaxation is tight with a wide margin: the largest $\lambda_2/\lambda_1$ over all 400 instances is $1.5\times10^{-9}$, about three orders of magnitude below the $10^{-6}$ threshold, so the 100% certification rates of Table 5.5 and of the noise-free cell of Table 5.6 are not marginal passes. $\mathrm{Shor}-J_{\rm ALM}$ lies within $\pm10^{-8}$ and changes sign; this is the SCS tolerance, not a theoretical gap, and a rigorous soundness claim still needs the dual-feasible repair of §4.6 (G3). The certificate cost grows from 52.9 to 85.5 ms (means; medians 51 → 84 ms) as $p$ goes from 8 to 64, consistent with a fixed $14\times14$ PSD block and $p$ linear constraints.

**Table 5.8b. The exact Schur bound of Theorem P2-Cert versus the conservative $P$-shift dual repair (6 synthetic instances, $\sigma\in\{0.10,0.20\}$, $p=16$).**

| $\sigma$ | seed | $J$ | $J-\text{dual repair}$ | $J-\text{Schur}$ | case |
|---|---|---|---|---|---|
| 0.10 | 0 | 9.0579e-01 | $2.00\times10^{-8}$ | $5.59\times10^{-12}$ | A |
| 0.10 | 1 | 6.8500e-01 | $2.00\times10^{-8}$ | $-6.69\times10^{-13}$ | A |
| 0.10 | 2 | 4.3643e-01 | $2.00\times10^{-8}$ | $5.31\times10^{-11}$ | A |
| 0.20 | 0 | 3.2392e+00 | $2.04\times10^{-8}$ | $4.22\times10^{-10}$ | A |
| 0.20 | 1 | 3.1128e+00 | $2.00\times10^{-8}$ | $3.81\times10^{-11}$ | A |
| 0.20 | 2 | 3.0309e+00 | $2.01\times10^{-8}$ | $6.21\times10^{-11}$ | A |

*Source: `results/p2b_schur_check.json` / `.md` (main-worker independent check, `shor_bound(..., schur=True)`; does not modify `certificate.py`).*

Both bounds are sound; the 13×13 translation Schur complement is the tighter one by about four orders of magnitude. The $P$-shift repair pays a $2c$ loss from bounding $\|\mathbf t\|$, so its gap is pinned at $2\delta_{\rm round}\approx2\times10^{-8}$, while the Schur bound needs no $\|\mathbf t\|$ bound and leaves $5.6\times10^{-12}$–$4.2\times10^{-10}$ on five of the six rows. The sixth row records $J-d_{\rm sound}=-6.7\times10^{-13}$: that is the size of the floating-point margin $\delta_{\rm round}\sim10^{-12}\|\mathbf S_{10}\|_F$ which Theorem P2-Cert subtracts, i.e. a rounding residual at the level of the certified margin, not a structural overshoot — the paper reports the raw difference and does not round it to "positive". Consequence for the rest of §5: **`rigorous_schur` is the bound of record**; the `P`-shift `repair_dual` value is retained only as a conservative fallback (and is the value quoted in §5.6's synthetic 16/16 result, which predates the Schur path). All six rows are case A (the $3\times3$ translation block is positive definite), so the tighter branch of the theorem is the one that is exercised.

**Table 5.9. Zero-dimensional RLT ablation at four noise levels (4 seeds per cell).**

| $\sigma$ / bias | $n$ | base cert | +RLT cert | median abs gap, base | median abs gap, +RLT | base time (s) | RLT time (s) |
|---|---|---|---|---|---|---|---|
| 0.02 / 0.10 | 4 | 4/4 | 4/4 | $1.6\times10^{-10}$ | $8.1\times10^{-11}$ | 0.05 | 0.15 |
| 0.05 / 0.20 | 4 | 4/4 | 4/4 | $1.7\times10^{-9}$ | $9.4\times10^{-10}$ | 0.05 | 0.14 |
| 0.10 / 0.30 | 4 | 1/4 | 2/4 | $4.2\times10^{-4}$ | $2.9\times10^{-4}$ | 1.59 | 0.85 |
| 0.20 / 0.40 | 4 | 0/4 | 0/4 | $8.6\times10^{-3}$ | $7.8\times10^{-3}$ | 1.62 | 2.40 |

*Source: `experiments/theory/results/rlt.json` (16 instances; all 16 feasible).*

The zero-dimensional RLT cuts $g_ig_j\ge0$ (136 rows at $p=16$) help only when the base gap is small enough for a modest tightening to cross the threshold, and nowhere else:

* At $\sigma=0.02$ and $0.05$ the base certificate is already 4/4, so RLT changes nothing.
* At $\sigma=0.10$ the base certifies 1/4; RLT certifies 2/4. The gain is one seed whose gap drops from $2.4\times10^{-4}$ to $7.6\times10^{-12}$; the already-certified seed stays certified, and the two remaining seeds stay uncertified with gaps of $3.1\times10^{-3}$ and $5.8\times10^{-4}$.
* At $\sigma=0.20$ both methods certify 0/4. RLT reduces the per-seed gap by 10–17% (median $8.6\times10^{-3}\to7.8\times10^{-3}$), which is still about four orders of magnitude above the $10^{-6}$ tolerance.
* The RLT certificate costs about $3\times$ the base SDP on the low-noise cells (0.05 → 0.15 s) and 0.1–2.6 s per instance on the high-noise cells, where both solves are inaccurate.

The failures at $\sigma=0.10$ and $0.20$ are the reason this draft does not claim that zero-dimensional RLT closes the gap. The sound higher-order bound-RLT family does close it on the synthetic cells (§5.6); its real-data counterpart is the rigorous rerun reported in §5.5.

On the real data the picture is weaker and dataset-dependent (all numbers in this paragraph are ALM rows; full table and Wilson intervals in §5.5). One-sided, base vs $+$RLT: LM 30/90 (33.3%) vs 28/90 (31.1%), McNemar $p=0.845$; LM-O 22/72 (30.6%) vs 24/72 (33.3%), $p=0.864$; YCB-Video 28/84 (33.3%) vs 20/84 (23.8%), $p=0.152$ (worse). On the two-sided test RLT is worse everywhere: 17/90→13/90, 9/72→3/72, 8/84→5/84. Moreover the zero-dimensional RLT *values* are not always sound: the RLT value exceeds $J$ on 21/90 LM, 22/72 LM-O and 15/84 YCB-Video ALM rows, by up to $1.13\times10^{-3}$, $7.42\times10^{-2}$ and $4.91\times10^{-2}$ respectively. These are SCS failures, not certificate improvements, and they are exactly what the dual repair of Theorem P2-Cert is designed to eliminate. The zero-dimensional real-data RLT counts must therefore be read as numerical evidence from an unverified bound, not as certification; the *sound* higher-order bound-RLT family is a different story and is the one positive real-data result of §5.5(v) (rigorous Schur certification 24.4→51.1% LM, 15.3→41.7% LM-O, 13.1→23.8% YCB-Video).

### 5.5 Real data: BOP LINEMOD, LINEMOD-Occlusion, YCB-Video

**Table 5.10. Real-data layer experiment ($p=16$, GT-projected correspondences; n=90/72/84).**

| dataset | $n$ | method | pen (mm) | rot (deg) | trans (mm) | feasible | cert one-sided (upper bound) | cert two-sided (necessary) |
|---|---|---|---|---|---|---|---|---|
| LINEMOD | 90 | Unconstrained QPEP | 11.26 | 2.08 | 35.63 | 13.3% | 13.3% | 8.9% |
| | | Soft penalty ($\lambda{=}10^3$) | 0.00 | 2.13 | 39.81 | 88.9% | 26.7% | 16.7% |
| | | Post-hoc projection | 0.00 | 2.05 | 30.80 | 100% | 15.6% | 10.0% |
| | | **ALM-QPEP (ours)** | **0.00** | **1.97** | **37.66** | **100%** | **33.3%** | **18.9%** |
| LM-O | 72 | Unconstrained QPEP | 14.93 | 2.07 | 34.69 | 6.9% | 6.9% | 6.9% |
| | | Soft penalty ($\lambda{=}10^3$) | 0.00 | 1.90 | 39.91 | 91.7% | 26.4% | 9.7% |
| | | Post-hoc projection | 0.00 | 2.04 | 31.29 | 100% | 8.3% | 6.9% |
| | | **ALM-QPEP (ours)** | **0.00** | **1.99** | **38.14** | **100%** | **30.6%** | **12.5%** |
| YCB-Video | 84 | Unconstrained QPEP | 14.09 | 1.08 | 17.33 | 6.0% | 6.0% | 3.6% |
| | | Soft penalty ($\lambda{=}10^3$) | 0.00 | 1.40 | 29.38 | 90.5% | 21.4% | 7.1% |
| | | Post-hoc projection | 0.00 | 1.08 | 8.93 | 100% | 7.1% | 4.8% |
| | | **ALM-QPEP (ours)** | **0.00** | **1.11** | **25.18** | **100%** | **33.3%** | **9.5%** |

*Source: `real/linemod/results/raw.json`, `real/bop/results/lmo/raw.json`, `real/bop/results/ycbv/raw.json` (n=90/72/84); Wilson CIs in each `main_table.md`. The one-sided flag is $J-\mathrm{Shor}\le10^{-6}$ and accepts negative gaps; the two-sided flag is $|J-\mathrm{Shor}|\le10^{-6}$. Neither is a rigorous certificate: SCS is not dual-feasible. The rigorous replacement is Table 5.11.*

Six observations.

**(i) The unconstrained solve is infeasible on real object/table geometry.** It violates feasibility in 86.7% (LM), 93.1% (LM-O) and 94.0% (YCB-Video), with median penetration 11.26/14.93/14.09 mm. ALM is 100% feasible with 0.00 mm penetration on all three datasets; the soft penalty is feasible in 88.9%/91.7%/90.5%. Projection is also 100% feasible.

**(ii) Constraints buy admissibility, not accuracy.** All four methods are within 1.1–2.1° rotation and 9–40 mm translation. ALM is not more accurate than projection (translation 37.66 vs 30.80 mm on LM, 38.14 vs 31.29 mm on LM-O, 25.18 vs 8.93 mm on YCB-Video). This is the same conclusion as the synthetic tables and the honest warning of §5.1.5.

**(iii) Real certification is upper-bounded and dataset-dependent.** The one-sided ALM rates are 33.3% [24.5,43.6] (LM), 30.6% [21.1,42.0] (LM-O), 33.3% [24.2,43.9] (YCB-Video); the two-sided rates are 18.9%, 12.5%, 9.5%. The zero-dimensional RLT cuts do **not** give a reproducible real gain: LM 33.3→31.1 (McNemar $p=0.85$), LM-O 30.6→33.3 ($p=0.86$), YCB 33.3→23.8 ($p=0.15$, worse); on the two-sided test RLT is worse everywhere (18.9→14.4, 12.5→4.2, 9.5→6.0). We therefore report the *zero-dimensional* RLT on real data as a **negative result**. The *sound higher-order* bound-RLT family behaves differently once the bound is sound; that is observation (v).

**(iv) The stored real-data certificate flag is one-sided.** It tests `feasible and J − Shor <= 1e-6`, does not log $\lambda_2/\lambda_1$ or the SDP status, and accepts negative gaps. On the scaled runs 18–26% of base ALM rows have negative gaps, and the two-sided rates are 9.5–18.9%. The rates in Table 5.10 are therefore **upper bounds**; the rigorous replacements are in Table 5.11. The earlier $n=18/12$ numbers are superseded by this table.

**(v) Rigorous certification (Theorem P2-Cert), and the one positive RLT result.** Table 5.11 replaces both flags by *proven* lower bounds: `rigorous_schur` (the 13-dimensional translation Schur complement of (2.12), the bound of record) and, as a conservative fallback, the $P$-shift `repair_dual` value. The counts verify the sandwich on every dataset (one-sided $\ge$ rigorous $\ge$ two-sided):

**Table 5.11. Rigorous real-data certification ($p=16$, $n=90/72/84$, bit-identical ALM replay of the published runs).**

| dataset | $n$ | cut family | rigorous Schur | 95% CI | rigorous $P$-shift | 95% CI | Schur gap min / median / max | naive value (unsound) |
|---|---|---|---|---|---|---|---|---|
| LM | 90 | base | **22** (24.4%) | [16.7,34.2] | 23 (25.6%) | [17.7,35.4] | $4.0\times10^{-12}$ / $9.1\times10^{-6}$ / $4.0\times10^{-4}$ | 25 |
| | | zero-dim RLT | 19 (21.1%) | [14.0,30.6] | 18 (20.0%) | [13.0,29.4] | $9.3\times10^{-12}$ / $1.6\times10^{-5}$ / $2.3\times10^{-1}$ | 35 |
| | | bound-RLT | **46** (51.1%) | [41.0,61.2] | 43 (47.8%) | [37.8,58.0] | $4.0\times10^{-12}$ / $3.4\times10^{-7}$ / $2.8\times10^{-1}$ | 61 |
| LM-O | 72 | base | **11** (15.3%) | [8.8,25.3] | 11 (15.3%) | [8.8,25.3] | $1.7\times10^{-11}$ / $1.6\times10^{-5}$ / $2.6\times10^{-4}$ | 19 |
| | | zero-dim RLT | 6 (8.3%) | [3.9,17.0] | 6 (8.3%) | [3.9,17.0] | $3.2\times10^{-11}$ / $6.4\times10^{-5}$ / $3.3$ | 19 |
| | | bound-RLT | **30** (41.7%) | [31.0,53.2] | 24 (33.3%) | [23.5,44.8] | $5.6\times10^{-12}$ / $2.1\times10^{-6}$ / $9.4\times10^{-2}$ | 45 |
| YCB-Video | 84 | base | **11** (13.1%) | [7.5,21.9] | 10 (11.9%) | [6.6,20.5] | $1.0\times10^{-11}$ / $2.1\times10^{-5}$ / $4.0\times10^{-4}$ | 17 |
| | | zero-dim RLT | 6 (7.1%) | [3.3,14.7] | 6 (7.1%) | [3.3,14.7] | $1.0\times10^{-11}$ / $3.8\times10^{-5}$ / $1.1$ | 11 |
| | | bound-RLT | **20** (23.8%) | [16.0,33.9] | 19 (22.6%) | [15.0,32.6] | $4.1\times10^{-12}$ / $1.3\times10^{-5}$ / $3.0$ | 31 |

*Source: `results/rigorous_real.json` / `.md` (merged) and the per-dataset `results/rigorous_{lm,lmo,ycbv}.json`.* `cert` = ALM feasible and $J-d_{\rm sound}\le10^{-6}$ with $d_{\rm sound}$ a *proven* lower bound; Wilson 95% CIs; zero-dim RLT adds 136 rows ($p=16\to152$), bound-RLT adds 320 ($16\to336$). The ALM replay is bit-identical to `raw.json` ($\max|\Delta J|=0$), so the rows are paired. Bound-RLT adds only *valid* inequalities (products of a constraint with a bound on the same constraint), so its certificates are for the same feasible problem; the depth/upright augmentation of Table 5.12 is the one place where the feasible set is restricted. "naive value" is the same $J-\mathrm{Shor}\le10^{-6}$ rule applied to the solver value in the replay: it is not a bound and is listed only to show how much of the old rate was numerical.*

Five readings.

* **The rigorous rates are much lower than the one-sided upper bounds** — LM 24.4% vs 33.3%, LM-O 15.3% vs 30.6%, YCB-Video 13.1% vs 33.3% — and consistently above the two-sided necessary test (18.9/12.5/9.5). The sandwich is therefore ordered as it must be, and the gap between the upper bound and the sound rate (9–20 percentage points) is the size of the numerical illusion that the one-sided flag was reporting.
* **The uncertified rows are genuinely slack, not numerical failures.** The Schur gap is non-negative on every one of the 738 replay rows (3 cut families $\times$ 246 instances; 0 negative gaps, minimum $4.0\times10^{-12}$–$1.7\times10^{-11}$, at the rounding floor), and the median gap on the base configuration is $9.1\times10^{-6}$/$1.6\times10^{-5}$/$2.1\times10^{-5}$ — 9–21× the $10^{-6}$ tolerance. Tightness, not round-off, is the binding limitation on real data.
* **The Schur bound and the $P$-shift bound are within 1–2 rows on real data** (22 vs 23, 11 vs 11, 11 vs 10), unlike the four-orders-of-magnitude separation on the synthetic high-noise cells (Table 5.8b); per row the Schur value is the tighter of the two on 82/90, 68/72 and 78/84 base rows, and all 738 replay rows are case A ($\tilde{\mathbf H}_{tt}\succ0$, maximum condition number 1668), so the tight branch of Theorem P2-Cert is the one exercised and no row needs the $R_t$ ball. The reason is the previous bullet: the $2\delta_{\rm round}\approx2\times10^{-8}$ cost of the $P$-shift is irrelevant when the gaps are $\ge10^{-5}$. We still report the Schur value as the bound of record and the $P$-shift value as a fallback, because the ordering is the opposite (Schur tighter) wherever the gap reaches the rounding floor.
* **The one-sided count is not reproducible across SDP re-solves.** On the identical instances (ALM $J$ bit-identical), the replay re-solves the SDP at `eps=1e-9`, 50k iterations instead of the stored run's 20k, and its value differs by a median of $2.3\times10^{-8}$ and a maximum of $3.7\times10^{-2}$; applying the *same* $J-\mathrm{Shor}\le10^{-6}$ rule to the replay's value gives 25/90, 19/72, 17/84 where the stored table reports 30/90, 22/72, 28/84. A test whose own count moves by 5–11 rows between solves is not a certificate, which is why the paper retires it as an upper bound only.
* **Sound bound-RLT is a positive result, and the earlier negative conclusion came from the unsound value.** Compared with the base configuration, the sound higher-order bound-RLT family raises rigorous certification from 22→46 (LM), 11→30 (LM-O), 11→20 (YCB-Video) on the Schur bound — +26.7/+26.4/+10.7 percentage points — with McNemar $p=8.4\times10^{-6}$, $1.6\times10^{-4}$, $3.5\times10^{-2}$ (paired flags; $0\to1$ flips 27/22/12 against $1\to0$ flips 3/3/3). On the more conservative $P$-shift bound the same counts are 23→43, 11→24, 10→19 (+22.2/+18.1/+10.7 pp, from Table 5.11; the paired test is reported on the Schur value only). The zero-dimensional family is null or negative under the same sound rule (Schur 22→19, 11→6, 11→6; $p=0.45/0.063/0.063$). The contrast with §5.4's earlier negative reading is instructive: on the *stored values* the zero-dimensional RLT appeared to help on LM-O (22→24) only because it pushed values above $J$ on 22/72 rows; with a sound bound that apparent gain disappears, while the bound-RLT family — which was judged a failure on real data when read through the unsound flag — is the one that actually closes gaps. We therefore revise the §5.6 statement: the *zero-dimensional* RLT is the negative result, the *sound bound-RLT* is a real gain of 11–27 pp at a $0.9$–$1.5\times$ SDP cost (median 1.65→1.96 s LM, 1.66→2.51 s LM-O, 1.68→1.52 s YCB-Video; the Schur repair itself costs 0.20–0.36 ms).

**(vi) Why the LM-O pure-plane rows need cross-direction cuts (before → after).** The structural diagnosis of `theory/proofs.md` §2.7.4 says that when every active constraint shares the table normal, the RLT products constrain only $\mathbf n^\top\mathbf t$, leaving a rank-2 pseudo-mixture admissible, and that positive-depth and tilt-cone rows add the cross-direction blocks that cut it. Table 5.12 tests that prediction on the six LM-O rows with the largest Schur gap under the base configuration (all frame 000002, objects 5/8/12, bias 20/30 px; 61 of 72 base rows are uncertified).

**Table 5.12. LM-O depth+tilt augmentation, before → after (6 worst base-uncertified rows).**

| row (frame/img/obj/bias) | gap, base | gap, bound-RLT (no aug.) | gap, aug.+base | gap, aug.+bound-RLT | certified after |
|---|---|---|---|---|---|
| 000002/1/8/30 | $2.56\times10^{-4}$ | $1.45\times10^{-4}$ | $2.54\times10^{-4}$ | $9.0\times10^{-12}$ | yes |
| 000002/0/8/30 | $2.56\times10^{-4}$ | $2.00\times10^{-4}$ | $2.37\times10^{-4}$ | $5.2\times10^{-11}$ | yes |
| 000002/1/5/30 | $1.78\times10^{-4}$ | $3.54\times10^{-5}$ | $1.77\times10^{-4}$ | $1.0\times10^{-11}$ | yes |
| 000002/0/5/30 | $1.15\times10^{-4}$ | $1.0\times10^{-10}$ | $1.15\times10^{-4}$ | $9.9\times10^{-11}$ | yes |
| 000002/0/8/20 | $1.02\times10^{-4}$ | $6.17\times10^{-5}$ | $1.02\times10^{-4}$ | $1.9\times10^{-11}$ | yes |
| 000002/0/12/30 | $1.00\times10^{-4}$ | $9.37\times10^{-5}$ | $9.9\times10^{-5}$ | $6.0\times10^{-11}$ | yes |

*Source: `results/rigorous_lmo_depth_tilt.json` (`run_rigorous_real.py depth-tilt --select worst --max-rows 6 --alpha 12 --n-depth 4`); the "no aug." column is the row-level `brlt` gap in `real/bop/results/lmo/rigorous/rigorous_rows.json`. Augmentation adds 4 positive-depth rows and 1 upright-cone row ($\alpha=12^\circ$, $\cos\alpha=0.978$) to the 16 plane rows ($n_{\rm ineq}=16\to21$).*

The prediction holds in a specific and checkable form. (a) The augmentation does not change the answer: $|\Delta J|\le7.4\times10^{-15}$ and ALM's pose already satisfies the added rows (minimum depth 0.85–0.88 m $\gg z_{\min}=0.02$ m; upright 0.998–0.999 $\ge\cos12^\circ$), so the certificate is for a problem with the same optimum, not for a different one. (b) The extra rows *alone* do nothing: base cuts on the augmented problem still fail on all six (gaps $9.9\times10^{-5}$–$2.5\times10^{-4}$, essentially unchanged). (c) With bound-RLT on the augmented problem all six certify, at gaps $9\times10^{-12}$–$1\times10^{-10}$; without the augmentation, bound-RLT alone certifies only 1 of the 6 (gap $1.0\times10^{-10}$ for that one, $3.5\times10^{-5}$–$2.0\times10^{-4}$ for the rest). So it is the *cross-direction RLT products* $g_{\rm plane}g_{\rm depth}$ and $g_{\rm plane}g_{\rm tilt}$ that carry the gain, exactly as §2.7.4 argues, and the depth/tilt rows are the enabler rather than the cure. The honest scope: this is 6 of the 61 uncertified LM-O rows, selected as the worst; it is a mechanism check, not a dataset-level rate, and the dataset-level rates remain those of Table 5.11.

### 5.6 Measured baselines: exact inner enumeration, SCP, and sound RLT

**Measured in this round.** All numbers are layer-level and were produced on a shared login node; absolute timings drift with load and are used only for relative comparison.

* **Exact QPEP inner enumeration, end-to-end through ALM.** On 8 ALM trials at $p\in\{8,16,64\}$ (seeds 0–3 at $p=16$, seeds 0–1 at $p=8,64$), the exact 162-path total-degree enumerator and the default 6-start multistart return the same objective to $\max|\Delta J|=1.8\times10^{-12}$, the same pose to $\max\|\Delta\mathbf q\|=7.9\times10^{-10}$, the same feasibility (max violation $7.6\times10^{-11}$) and the same outer iteration count on 7/8 trials (the remaining $p=64$ trial stops one iteration earlier). Completeness is conditional and measured: 89/114 enumerations (78.1%) found the generic 40 finite projective roots, while 25 (21.9%) lost a homotopy path and fell back to a 64-start BFGS; every one of the 114 probed enumerations had a fresh 64-start BFGS reach the same global minimum. At $\sigma=0.01$ the ALM active-set matrix is full rank (10/10, min eigenvalue $8.2\times10^{-4}$) while the unconstrained $\mathbf M^{\rm red}_{\mathcal A}$ is rank $9$; the noise-free matrix is rank $8$ with 16–18 real roots (`results/quartic_validation.json`). The 25 fallbacks are therefore homotopy path loss, not rank deficiency. The exact inner costs 5.0–15.9 s per pose against 0.12–0.28 s for multistart (37–64×, median $47\times$); a hybrid schedule (exact enumeration every third outer iteration) cuts that to 2.1–4.0 s at the same objective (`exact_alm.md` S22). It therefore buys a correctness / global-optimality check of the inner block at a fixed active set, not a competitive inner solver; the default remains multistart. *Source: `results/exact_alm.json` / `results/exact_alm.md`; the inner-only 24-instance table is `results/quartic_validation.json`.*
  **Enumeration-cost update (measured separately, not rescaled).** The 37–64× ratios above were measured with the pre-optimisation enumerator (one 162-path enumeration $\approx649$ ms on average). A later row of the same code replaced the three-operand `einsum` that forms $\mathbf J^\top\mathbf M\mathbf J$ by two batched BLAS products ($\mathbf M\mathbf J$ then $\mathbf J^\top(\mathbf M\mathbf J)$, mathematically identical), taking one enumeration from **0.488 s to 0.168 s (2.92×, 356 → 1077 paths/s)** with exact parity on 15 cases (same root counts, same finite projective counts, $\max|\Delta f|=3.3\times10^{-16}$) and the same number of batch evaluations (median 470). Because the end-to-end ALM timing has *not* been re-run after this change, we do **not** divide the 37–64× range by 2.92: we keep the directly measured 37–64× as the conservative figure and report the 2.92× as an isolated inner-loop speedup. A rejected variant (early culling of divergent paths, `div_tol` $10^3$–$10^2$) is also recorded there: it gained nothing at $10^3$ and lost roots at $10^2$, so it was reverted. *Source: `results/perf_quartic.md` / `results/perf_quartic.json`.*
* **SCP baseline (dQP + local linearisation).** On the synthetic main cells ($p=8/16/32$, 50 trials each) SCP is 100% feasible, matches ALM's pose error to the printed precision ($p=16$: 0.00 mm / 4.17° / 10.63 mm) and its output passes the same $14\times14$ certificate test in 98% of trials against ALM's 100%; it is faster in this implementation (median 128–154 ms versus ALM's 200–233 ms in the same re-run) with about 90% of its wall clock in the SOCP. On the pressure cell ($\sigma=0.020$, bias 0.120, 20 trials) SCP certifies 85% against ALM's 100%. All six SCP failures start from a cheirality-mirror unconstrained solution and are not rescued by two extra perturbed starts. The differentiators we claim are therefore the mirror-initialisation robustness and the $p$-independent complexity argument, not per-instance accuracy; SCP is a mandatory strong baseline and is reported as such. *Source: `results/scp_baseline.md` (raw per-trial data in `results/scp_baseline.json`).*
* **Sound higher-order RLT (synthetic).** A sound bound-RLT family combined with `repair_dual` certifies all 16 synthetic noise instances (9/16 before), with maximum rigorous gap $2.28\times10^{-8}$ and median certificate time 0.036 s → 0.180 s. The gains are concentrated where zero-dimensional cuts alone failed ($\sigma=0.10$: 1/4 → 4/4; $\sigma=0.20$: 0/4 → 4/4). *Source: `experiments/theory/results/round_rlt.md` / `round_rlt.json`.*
* **Rigorous real-data certification and the bound-RLT result.** The stored one-sided flags are replaced by proven lower bounds on all three datasets (§5.5(v), Table 5.11). Base configuration: rigorous Schur 22/90 (24.4%), 11/72 (15.3%), 11/84 (13.1%) — all below the one-sided upper bounds (33.3/30.6/33.3%) and above the two-sided necessary tests (18.9/12.5/9.5%). The Schur bound is non-negative on all 738 replay rows (min $4.0\times10^{-12}$), so the failure mode is slack: the median base gap is $9.1\times10^{-6}$/$1.6\times10^{-5}$/$2.1\times10^{-5}$, 9–21× the tolerance. Adding the sound bound-RLT family raises the rigorous rate to 46/90 (51.1%), 30/72 (41.7%), 20/84 (23.8%), i.e. +26.7/+26.4/+10.7 pp on the Schur bound (+22.2/+18.1/+10.7 pp on the $P$-shift counts), paired McNemar $p=8.4\times10^{-6}$, $1.6\times10^{-4}$, $3.5\times10^{-2}$, at a 0.9–1.5× SDP cost and a 0.20–0.36 ms Schur repair. The zero-dimensional RLT family stays null or negative under the same sound rule (Schur 22→19, 11→6, 11→6; $p=0.45/0.063/0.063$). This reverses the earlier "RLT is negative on real data" reading: **zero-dimensional RLT is the negative result; sound bound-RLT is a real gain**. *Source: `results/rigorous_lm.json`, `results/rigorous_lmo.json`, `results/rigorous_ycbv.json` / `results/rigorous_real.{json,md}`.*

### 5.7 Neighbouring layer families on the same instances

We compare against two re-implementations and one official library. The SDPRLayers-style baseline is our own construction of its forward relaxation; the dQP/CVXPYLayers-style baseline is our own QP layer with implicit differentiation, cross-checked against `cvxpylayers`; the Theseus baseline uses the official `theseus-ai 0.2.1` library (forward only), run through an isolated `scikit-sparse`/CHOLMOD overlay that leaves the environment untouched. No number here is an official SDPRLayers or dQP result; the Theseus numbers are the official library's forward solve, with the caveats stated below.

**SDPRLayers-style forward relaxation.** We solve the same Shor relaxation as in §5.4 (our 14×14 lift with the full Veronese ideal) and extract a pose from the leading eigenvector, in three forms: *unc* (no inequality rows, their setting), *hard* (the hard inequality rows, i.e. the penalty weight $M\to\infty$), and *slack* (the lifted slack form with the PSD block $(14+p)\times(14+p)$). On the synthetic cells the behaviour is the one reported in the earlier draft, now at $n=50$ for every $p$: *unc* reproduces the unconstrained QPEP solve exactly (103.02 mm, 0% feasible), while *hard* and *slack* reach 0.00 mm, 4.17° and 10.63 mm with 92–100% feasibility, at 30–52 ms for *hard* and 63→2259 ms for *slack*. The main new measurement of this round is on the real BOP instances, at $n=90/72/84$ instead of the earlier $n=3$ probe: **the SDPR-style hard relaxation is not feasible on real data**. It penetrates by 9.19/12.70/11.28 mm and is feasible in only 14%/12%/10% of trials; the slack lifting reduces median penetration to 2.40/1.96/0.28 mm but is feasible in 42%/32%/46% and certified (sound) in only 12%/3%/2%; the unconstrained relaxation is the published unconstrained solve. On the same instances ALM is 100% feasible at 0.00 mm and certifies 24.4/15.3/13.1% (Table 5.11). The honest reading is that the SDPR-style *forward* relaxation is competitive on the synthetic desktop, where the relaxation happens to be tight, but is not a drop-in replacement for the ALM layer on real object/table geometry: its hard form is feasible on one trial in seven, and its own sound certificate rate is at or below ALM's. The forward cost is 198/132/238 ms (*unc*), 675/669/667 ms (*hard*) and 3169/3150/3185 ms (*slack*) per instance (medians); the *slack* PSD block is confirmed to be $(14+p)\times(14+p)$ by the returned matrix (22×22, 30×30, 46×46, 78×78 at $p=8/16/32/64$; status census in `baseline_sdpr_status.md`).

**Table 5.9b. SDPRLayers-style reimplementation, same instances as Tables 5.3–5.5 and 5.10 ($n=50$ per synthetic cell, $n=90/72/84$ real; medians, rates; IQR and CIs in `baseline_sdpr_final.md`).** `unc` = unconstrained Shor + eigenvector extraction; `hard` = same with the hard inequality rows; `slack` = lifted slack form. `cert` is the sound P2-Cert Schur bound on the same instance, not the SCS value.

| cell | variant | pen (mm) | rot (deg) | trans (mm) | feasible | cert (sound) | ms |
|---|---|---|---|---|---|---|---|
| synth $p=8$ | unc | 103.02 | 5.30 | 97.90 | 0% | 0% | 31 |
| synth $p=8$ | hard | 0.00 | 4.17 | 10.63 | 96% | 96% | 30 |
| synth $p=8$ | slack | 0.00 | 4.17 | 10.63 | 98% | 98% | 63 |
| synth $p=16$ | unc | 103.02 | 5.30 | 97.90 | 0% | 0% | 30 |
| synth $p=16$ | hard | 0.00 | 4.17 | 10.63 | 98% | 98% | 33 |
| synth $p=16$ | slack | 0.00 | 4.17 | 10.63 | 98% | 98% | 178 |
| synth $p=32$ | unc | 103.02 | 5.30 | 97.90 | 0% | 0% | 35 |
| synth $p=32$ | hard | 0.00 | 4.17 | 10.63 | 98% | 98% | 41 |
| synth $p=32$ | slack | 0.00 | 4.17 | 10.63 | 100% | 100% | 523 |
| synth $p=64$ | unc | 103.02 | 5.30 | 97.90 | 0% | 0% | 32 |
| synth $p=64$ | hard | 0.00 | 4.17 | 10.63 | 92% | 92% | 52 |
| synth $p=64$ | slack | 0.00 | 4.17 | 10.63 | 92% | 92% | 2259 |
| LINEMOD ($n{=}90$) | unc | 11.47 | 2.19 | 36.02 | 12% | 9% | 198 |
| LINEMOD ($n{=}90$) | hard | 9.19 | 2.14 | 35.95 | 14% | 13% | 675 |
| LINEMOD ($n{=}90$) | slack | 2.40 | 2.50 | 38.27 | 42% | 12% | 3169 |
| LM-O ($n{=}72$) | unc | 14.93 | 2.07 | 34.69 | 7% | 7% | 132 |
| LM-O ($n{=}72$) | hard | 12.70 | 2.08 | 34.89 | 12% | 7% | 669 |
| LM-O ($n{=}72$) | slack | 1.96 | 2.53 | 47.07 | 32% | 3% | 3150 |
| YCB-Video ($n{=}84$) | unc | 11.40 | 1.12 | 19.06 | 13% | 6% | 238 |
| YCB-Video ($n{=}84$) | hard | 11.28 | 1.18 | 17.27 | 10% | 6% | 667 |
| YCB-Video ($n{=}84$) | slack | 0.28 | 1.43 | 28.31 | 46% | 2% | 3185 |

*Source: `results/baseline_sdpr_final.json` / `.md` (generated by `experiments/code/make_sdpr_results.py` from `results/baseline_sdpr_synth_p{8,16,32,64}.json` and `results/baseline_sdpr_{lm,lmo,ycbv}.json`). The synthetic instances are the same seeds as `results/raw_main.json`; the real rows replay `raw.json` (`max|\Delta J|=0`). Penetration is on the full object mesh against the estimated support plane. `cert` uses the stored rigorous Schur bound (base cuts) on the same instance. The *slack* and *hard* poses are extracted from the SDP solution by leading eigenvector; the mirror-pose counts and eigen-extraction shares are in `baseline_sdpr_final.md` Table 5.*

**dQP / CVXPYLayers-style differentiable QP.** We replace the SCP SOCP subproblem of §5.6 by a pure QP with a box trust region and differentiate it through the KKT system (dQP-style), cross-checking the derivative against finite differences and against an independent CVXPYLayers construction. At $n=50$ per cell the QP layer is competitive with SCP and ALM rather than a weak baseline: at $p=8/16/32$ it is 100%/98%/98% feasible and 100%/98%/98% certified, with 0.000 mm median penetration, 4.17° rotation and 10.63/10.85/10.85 mm translation — the same rotation and penetration as SCP and ALM, and translation within 0.22 mm of them. It is the fastest of the three (median 127.6/108.6/102.9 ms against SCP's 154.2/128.0/132.3 ms and ALM's 233.4/225.8/199.9 ms) because the pure QP costs 3.9/3.2/3.3 ms per iteration against the SOCP's 5.3/4.5/4.6 ms, and it converges in 27–28 iterations (96–98% of trials) versus SCP's 25 (98%). On a 30-instance real subset (10 base poses × 3 biases per dataset) it returns exactly the ALM pose, 100% feasible at 0.00 mm penetration, and certifies 37%/57%/20% (base cuts 17%/20%/10%), identical to the stored ALM rows because the certificate is an instance-level property of the bound, not of the layer; its cost is 101–111 ms. The earlier n=2 probe that reported 50% certification was under-powered, and this round supersedes it. The gradient check is also cleaner than the earlier draft's: on random well-conditioned QPs the four parameter blocks agree with central differences to a maximum relative error of $3.1\times10^{-9}$ (g), $3.0\times10^{-10}$ (gval), $8.0\times10^{-9}$ (Jg) and $4.2\times10^{-7}$ (H), and the independent CVXPYLayers construction agrees to $1.9\times10^{-5}$ on the contracted gradient. The structural limitation remains only at the degenerate planar multi-contact points: on 2 of 8 PnP seeds the active Jacobian is rank deficient, the KKT matrix is singular and the derivative is not unique (relative error 1.0), while the other 6 seeds are at $10^{-10}$–$10^{-7}$. That is the degeneracy the FB reduction of §4.5 is designed for; it is not an implementation error, and it no longer contaminates the aggregate. *Source: `results/baseline_dqp.json` / `.md`, generated by `experiments/code/baselines_dqp.py`; p=8/16/32 with n=50 seeds each, real subset n=30 per dataset, gradient check on 10 random QPs + 8 PnP seeds + 3 CVXPYLayers cross-checks.*

**Theseus soft penalty (official library).** `theseus-ai 0.2.1` is present in the environment but its import fails on the missing `sksparse`/CHOLMOD; an isolated `scikit-sparse 0.4.16` overlay (`PYTHONPATH`/`LD_LIBRARY_PATH` only, no change to the environment) makes it importable, so the baseline is now the official Theseus `LevenbergMarquardt` solver with `CholeskyDenseSolver` rather than a look-alike. The layer is $\min J(\mathbf R,\mathbf t)+\lambda\sum_j h(g_j)$ with $h$ quadratic or Huber ($\delta=10^{-3}$), warm-started from the same 48-start unconstrained QPEP solution as the main table, at $p=8/16/32$ and $n=30$ seeds per cell. The result is the expected soft-penalty limitation, now measured: **no finite $\lambda$ reaches hard feasibility**. The best quadratic-penalty cell is 93% feasible ($p=32$, $\lambda=10^5$) with 0.00 mm median penetration, 4.13° and 11.29 mm at 77 ms; at $p=8$ it is 80% feasible with 7% certified, and the certificate rate is 7–13% at $\lambda\ge10^4$ against ALM's 100%. Increasing $\lambda$ from $10^2$ to $10^5$ moves feasibility 57→80% ($p=8$), 67→83% ($p=16$), 70→93% ($p=32$) and never closes the gap. The Huber variant reaches 93–97% feasibility but at translation errors of 58–85 mm (versus 11 mm for the quadratic penalty) and certifies 0%: it buys admissibility by giving up the fit. The per-solve cost is 60–93 ms, lower than ALM's 260–288 ms on the synthetic cells, because Theseus takes 2–9 LM iterations per pose; that is a speed advantage of the *soft* formulation, not a feasibility one. The sweep also gives a clean Pareto curve for the paper's argument that a penalty weight is not a constraint (Table 5.9d). Theseus is forward-only here; the official layer's backward was not measured.

**Table 5.9d. Theseus λ-Pareto, official `theseus-ai 0.2.1` (n=30 per cell, seeds 0–29; medians; the ALM reference is the 50-trial Table 5.5).**

| $p$ | penalty | $\lambda$ | pen (mm) | rot (deg) | trans (mm) | feasible | certified | ms |
|---|---|---|---|---|---|---|---|---|
| 8 | quadratic | $10^2$ | 0.00 | 4.33 | 11.40 | 57% | 0% | 63 |
| 8 | quadratic | $10^3$ | 0.00 | 4.72 | 11.36 | 73% | 0% | 63 |
| 8 | quadratic | $10^4$ | 0.00 | 4.70 | 11.35 | 80% | 7% | 60 |
| 8 | quadratic | $10^5$ | 0.00 | 4.70 | 11.35 | 80% | 7% | 61 |
| 8 | Huber | $10^2$ | 0.00 | 4.58 | 57.64 | 97% | 0% | 65 |
| 16 | quadratic | $10^2$ | 0.00 | 4.33 | 11.38 | 67% | 0% | 69 |
| 16 | quadratic | $10^3$ | 0.00 | 4.49 | 11.36 | 77% | 3% | 72 |
| 16 | quadratic | $10^4$ | 0.00 | 4.50 | 11.35 | 83% | 10% | 70 |
| 16 | quadratic | $10^5$ | 0.00 | 4.50 | 11.35 | 83% | 10% | 69 |
| 16 | Huber | $10^2$ | 0.00 | 4.65 | 68.72 | 97% | 0% | 78 |
| 32 | quadratic | $10^2$ | 0.00 | 4.33 | 11.37 | 70% | 0% | 77 |
| 32 | quadratic | $10^3$ | 0.00 | 4.50 | 11.36 | 80% | 3% | 77 |
| 32 | quadratic | $10^4$ | 0.00 | 4.50 | 11.35 | 87% | 10% | 77 |
| 32 | quadratic | $10^5$ | 0.00 | 4.13 | 11.29 | 93% | 13% | 77 |
| 32 | Huber | $10^2$ | 0.00 | 4.97 | 76.97 | 93% | 0% | 93 |

*Source: `results/baseline_theseus.json` / `.md`, generated by `experiments/code/baselines_theseus.py` with `theseus-ai 0.2.1` + `scikit-sparse 0.4.16`. `pen` is the median full-mesh penetration; `feasible` is the linearised violation $\le10^{-6}$; `certified` is the same SCS-tightness + $J-\mathrm{Shor}\le10^{-6}$ rule as the main table; `ms` is the median forward wall-clock. The 15 rows are a subset of the 36 measured cells (two penalties × six $\lambda$ × three $p$); the omitted $\lambda=1,10$ rows are in the `.md`. Theseus is forward-only and single-start; the official `TheseusLayer` backward is not measured.*

### 5.8 End-to-end CPU toy: four pose heads through one refiner

The T1-network protocol is not run (no GPU, §5.11). What is run is a small CPU toy that closes the observation→pose loop: a 2-layer MLP refines 14 keypoints from a synthetic desk scene, and four pose heads map them to a pose — unconstrained QPEP, ALM-QPEP, post-hoc projection and a soft penalty. The refiner is trained with the FB-reduced backward of §4.5 (3 seeds × 2000 steps, $\approx144$ s per seed on one CPU core); the four heads are then applied to the same trained refiner. The ALM head is gradient-checked (below) but, because one ALM forward costs 10.2 ms against 1.2 ms for the unconstrained head, the 2000-step training run uses the unconstrained backward and the constrained heads are evaluated on the resulting refiner. This is a toy, not a benchmark: it shows the loop closes and that the constrained heads change the admissibility/accuracy trade-off, and it makes no accuracy claim against learned layers.

**Table 5.9c. CPU toy, trained refiner, four pose heads, 64 validation scenes (medians; rates over 64).**

| head | ADD (mm) | rot (deg) | trans (mm) | penetration (mm) | feasible | ms/pose |
|---|---|---|---|---|---|---|
| Unconstrained | 101.06 | 5.29 | 101.03 | 104.35 | 0% | 1.16 |
| ALM-QPEP | **12.34** | 3.78 | 11.06 | **0.000** | **100%** | 10.19 |
| Projection | 13.19 | 4.96 | 11.98 | 0.000 | 95.3% | 1.75 |
| Soft penalty | 12.34 | 3.78 | 11.06 | 0.012 | 0% | 5.76 |

The constrained heads cut ADD by 8.2× and drive penetration to zero; projection is the cheapest feasible head but is feasible on 95.3% of the scenes (its SLSQP can fail), while ALM is feasible on all 64. The soft penalty reaches the same accuracy as ALM but not hard feasibility (0%), which is the toy-level version of the §5.3 Pareto result.

**Gradient check of the FB backward (10 seeds, central differences, 4 modes).** The unconstrained head agrees to a relative $L_2$ error of $4.6\times10^{-10}$ median ($3.5\times10^{-9}$ max, cosine $0.9999999999999998$). The ALM head's error follows the $\varepsilon$ theory of §4.5 rather than a fixed value: median $5.7\times10^{-6}$ at $\varepsilon=10^{-6}$ ($1.8\times10^{-3}$ max, cosine $0.9999987$), $7.4\times10^{-4}$ at $\varepsilon=10^{-4}$, and $0.49$ at $\varepsilon=10^{-2}$; i.e. the FB derivative converges to the finite-difference derivative as $\varepsilon\to0$, at the cost of conditioning. The projection head ($3.9\times10^{-8}$ median) and the soft penalty ($4.0\times10^{-7}$ median) are also consistent. A kink test (10 seeds) gives a median symmetry error of $5.7\times10^{-6}$, best-side error $8.5\times10^{-5}$ and asymmetry $1.7\times10^{-4}$, with 16 (of 28) active components, consistent with the weak-active selection discussed in §4.5. *Source: `results/e2e_layer_final.json` (n=64), `results/e2e_gradcheck_final.json` (10 seeds × 4 modes), `results/e2e_kink_final.json`, `results/e2e_units/*.json` (2000-step training units).*

### 5.9 Localisation layers: KITTI odometry (and UTIAS In-The-Dark)

The suite so far is manipulation-shaped (a support plane). The same layer is also the natural pose head for a ground vehicle, where the constraints are a positive-depth (chirality) condition and a roll/pitch tilt cone. We run two localisation protocols at layer level: KITTI odometry [Geiger et al. 2012] and UTIAS In-The-Dark [Gridseth & Barfoot 2022]. Neither uses a detector, a descriptor front-end or a VO system: the 3D structure comes from the sensor (Velodyne points / stereo-triangulated ORB landmarks) and the 2D observation is the ground-truth projection of that structure into a second frame, with 1 px noise, an optional downward pixel bias and optional wrong matches. The second frame's pose is the unknown; the metric is the pose error of the layer against the ground-truth relative motion.

**KITTI protocol.** The shipped KITTI-odometry subset (`/mnt/storage1/syk/data/kitti_odom`, sequences 00/05/07) contains Velodyne scans and calibration but **no images**, so the correspondence is: Velodyne frame $i$ $\to$ rectified camera 0 via `Tr` $\to$ frame $j$ via the ground-truth relative motion $\to$ projection with `P0`, keeping points inside the $1241\times376$ image with $2<z<60$ m. Each scan pair yields ~80 landmarks. The constraints are positive depth on **all 80** landmarks and a $10^\circ$ roll/pitch cone on the relative rotation, $a^\top(\mathbf R a)\ge\cos10^\circ$, where $a$ is the world-up vector in the source camera frame (read from the ground-truth attitude of frame $i$; this is an *oracle* ingredient of the protocol, and the ground-truth tilt itself is only 0.14/0.21/0.21$^\circ$ median, 0.60/0.92/0.87$^\circ$ max on the three sequences). The ground-truth tilt is therefore far inside the cone on clean data, and the cone binds only when an outlier flips the rotation. The run produced each manifest pair once per setting; after removing the repeated pair list the per-setting $n$ is 63 (3 sequences $\times$ 21 distinct pairs), and the four settings are reported separately because their noise regimes are not comparable.

**Table 5.16. KITTI odometry layer, per setting ($n=63$ per cell; medians [IQR], RMSE, rates with Wilson 95% CI).**

| setting | method / constraints | rot med [IQR] (deg) | rot RMSE (deg) | trans med [IQR] (mm) | trans RMSE (mm) | tilt $>10^\circ$ | depth $\le0$ | feasible | cert 1-sided | cert 2-sided | cert rigorous | ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| clean | Unconstrained | 0.05 [0.04, 0.07] | 0.06 | 10.93 [5.88, 17.48] | 14.4 | 0.0% | 0.0% | 100% | 100% | 100% | – | 103 |
| | ALM, depth+tilt | 0.05 [0.04, 0.07] | 0.06 | 10.93 [5.88, 17.48] | 14.4 | 0.0% | 0.0% | 100% | 100% | 100% | **100% (63/63)** | 16 |
| bias 6 px | Unconstrained | 0.49 [0.48, 0.51] | 0.49 | 17.73 [12.67, 26.73] | 29.8 | 0.0% | 0.0% | 100% | 100% | 100% | – | 107 |
| | ALM, depth+tilt | 0.49 [0.48, 0.51] | 0.49 | 17.73 [12.67, 26.73] | 29.8 | 0.0% | 0.0% | 100% | 100% | 100% | **100% (63/63)** | 16 |
| bias 6 px + 15% wrong matches | Unconstrained | 3.43 [2.25, 4.99] | 22.99 | 2327.95 [1865.72, 2806.79] | 3511.6 | 1.6% | 3.2% | 96.8% | 96.8% | 96.8% | – | 105 |
| | ALM, depth+tilt | 3.22 [2.25, 4.71] | **5.06** | 2327.95 [1865.72, 2806.79] | 2616.1 | **0.0%** | **0.0%** | 100% | 98.4% | 98.4% | – | 17 |
| bias 6 px + 50% wrong matches | Unconstrained | 17.12 [10.37, 174.21] | 110.38 | 8832.20 [7270.44, 15812.19] | 13230.0 | 41.3% | 93.7% | 6.3% | 6.3% | 6.3% | – | 104 |
| | ALM, depth+tilt | 11.45 [7.24, 14.91] | **16.96** | 5839.83 [4623.14, 6334.54] | 6161.8 | **0.0%** | **0.0%** | 100% | 14.3% | 12.7% | – | 457 |

*Source: `results/kitti_layer.json` / `.md`, generated by `experiments/real/kitti/run_kitti_layer.py` + `experiments/real/make_real_layer_tables.py`; sequences 00/05/07, 4 settings, $n=63$ per cell (3 sequences × 21 distinct frame pairs after de-duplicating the repeated pair list). The four settings are separate runs and are not pooled. `cert 1-sided` is $J-\mathrm{Shor}\le10^{-6}$ (an **upper bound** on the sound rate); `cert 2-sided` is $|J-\mathrm{Shor}|\le10^{-6}$ (necessary only). `cert rigorous` is the sound P2-Cert Schur bound with base cuts, run on ALM poses for the clean and 6-px settings only (126 instances, 126/126 certified, Schur gaps $4.9\times10^{-9}$–$6.2\times10^{-8}$, median SDP 0.09–0.10 s; `results/kitti_cert.md`); `–` means not computed, not zero. A depth violation is counted below $-1$ mm, because the constrained optimum sits on the $z=0$ boundary; the depth constraint acts on all 80 landmarks. RMSE is over all instances of the cell, while the median/IQR columns are robust to the flipped tail. `ms` is the median wall-clock from the JSON (the `.md` table prints the mean; we use the median for consistency with §5.2).*

Four readings.

**(i) On clean and mildly biased data the constraints are inactive and free.** Clean: 0.05° / 10.93 mm for every method, 0% tilt and depth violations; the ALM layer is 6× cheaper here (16 vs 103 ms) because the active set is empty. A bias of 6 px shifts the pose by 0.5° without creating violations: the tilt cone is not binding, and no method is better than another. This is the expected behaviour of a layer whose constraints describe admissibility rather than accuracy.

**(ii) With wrong matches the constraints buy robustness on the rotation, not on the translation.** At 15% wrong matches the unconstrained solve has rotation RMSE 22.99° (median 3.43°), 1.6% tilt and 3.2% depth violations; ALM reduces the rotation RMSE to 5.06°, both violation rates to 0.0%, and the translation RMSE from 3511.6 to 2616.1 mm, while the translation *median* stays at 2.33 m for both: the constraint repairs the rotated tail, not the bias. At 50% wrong matches the unconstrained rotation RMSE is 110.38°, with 41.3% tilt and 93.7% depth violations; ALM is 16.96°, 0.0%/0.0%, and 10 of the 25 instances whose unconstrained rotation exceeds $30^\circ$ come out below $10^\circ$. The translation RMSE improves from 13230.0 to 6161.8 mm, still metres. The layer cannot recover a pose whose data term has been destroyed by 50% outliers, and we do not claim it: the honest statement is "the constraints make the failure mode *admissible* (no tilt/depth violation) and repair some flips, not that they solve robust estimation". A robust loss or RANSAC is the right tool for the 50% cell, and we do not use one here (it would confound the layer comparison).

**(iii) The remaining ALM violations are gone on this protocol, and that is a protocol change from the earlier draft.** The depth constraint now acts on *all* 80 landmarks rather than the 24 closest to the camera, so the earlier "coverage artefact" (ALM at 88.9% depth-violation rate in the 50%-outlier cell) is no longer present: ALM is at 0.0% depth and 0.0% tilt violations in every cell. The cost is visible in the runtime column (457 ms median in the 50%-outlier cell, because the active set is large and the solve is harder), not in feasibility.

**(iv) Certification: one-sided on the hard cells, rigorous on the two well-conditioned cells.** On KITTI the SDP lower bound was computed by the same 14×14 Shor relaxation as in §5.4–§5.5; the one-sided rule ($J-\mathrm{Shor}\le10^{-6}$) is an upper bound, and for the clean and 6-px cells it reads 100%. We also ran the sound P2-Cert Schur bound on ALM poses for exactly those two cells: 63/63 and 63/63 certify with base cuts, at Schur gaps $4.9\times10^{-9}$–$6.2\times10^{-8}$ and a median SDP cost of 0.09–0.10 s. So in the inactive-constraint regime the one-sided flag is not inflated: the sound rate is also 100%. The 15%- and 50%-outlier cells remain one-sided/two-sided only (98.4% and 14.3%/12.7%); running the rigorous repair there is the missing piece, and §5.11 says so. This is the only table in §5 that carries all three certificate columns side by side, and the ordering one-sided $\ge$ rigorous $\ge$ two-sided holds cell by cell.

**UTIAS protocol.** The UTIAS In-The-Dark archive [Gridseth & Barfoot 2022] is used at layer level as well. This harness uses `run_000000` only: it is the Teach run and ships `transforms_temporal.txt` (one camera pose per frame in a fixed world frame). The repeat runs (2/12/17/18) ship `transforms_spatial.txt`, which localises each repeat keyframe against a nearby Teach keyframe; chaining those edges into a repeat-run trajectory is not implemented here, so the layer experiment is restricted to the 38 frame pairs of `run_000000` that have temporal ground truth. The 3D landmarks are real: ORB left–right matches triangulated with $f=387.777$ px and baseline $0.239965$ m; the 2D observation in frame $k{+}\mathrm{lag}$ is the ground-truth projection of the landmark. Real left–left ORB matching was measured first and rejected (~9 px median residual on the dark images), so this is the same GT-projection layer as KITTI, not a feature-matching result. There are 38 frame pairs × 3 settings, lags 15/30, stride 60, 80 landmarks per pair, selected to over-represent landmarks near the image border (half of the 80). The constraints are positive depth on the 80 landmarks plus four image-border half-spaces per landmark. The FoV violation rate in the table is a *separate* geometric check of all 80 estimated projections against the image rectangle, which is why it need not fall to zero when the constrained solve is feasible; we report both and do not equate them.

**Table 5.16b. UTIAS In-The-Dark layer, `run_000000`, per setting ($n=38$ per cell; medians [IQR], RMSE, rates with Wilson 95% CI).**

| setting | method / constraints | rot med [IQR] (deg) | rot RMSE (deg) | trans med [IQR] (mm) | trans RMSE (mm) | depth $\le0$ | FoV out | feasible | cert 1-sided | cert 2-sided | ms |
|---|---|---|---|---|---|---|---|---|---|---|---|
| clean | Unconstrained | 0.12 [0.09, 0.18] | 0.16 | 8.26 [5.53, 13.71] | 11.4 | 0.0% | 0.0% | 100% | 100% | 100% | 145.9 |
| | ALM, depth+FoV | 0.12 [0.09, 0.18] | 0.16 | 8.26 [5.53, 13.71] | 11.4 | 0.0% | 0.0% | 100% | 100% | 100% | 18.6 |
| bias 3 px | Unconstrained | 0.42 [0.37, 0.51] | 0.45 | 10.90 [8.13, 14.49] | 13.7 | 0.0% | 0.0% | 100% | 100% | 100% | 142.8 |
| | ALM, depth+FoV | 0.42 [0.37, 0.51] | 0.45 | 10.90 [8.13, 14.49] | 13.7 | 0.0% | 0.0% | 100% | 100% | 100% | 18.7 |
| bias 20 px + 5% wrong matches | Unconstrained | 3.47 [3.06, 5.21] | 57.57 | 275.44 [108.49, 674.95] | 1375.3 | 10.5% | 42.1% | 57.9% | 57.9% | 57.9% | 137.9 |
| | ALM, depth+FoV | 3.47 [3.06, 5.31] | **10.87** | 275.44 [108.49, 501.69] | **603.0** | **0.0%** | 36.8% | 100% | 84.2% | 84.2% | 25.1 |

*Source: `results/utias_layer.json` / `.md`, generated by `experiments/real/utias/run_utias_layer.py` + `experiments/real/make_real_layer_tables.py`; `run_000000`, 38 frame pairs × 3 settings, $n=38$ per cell. `cert 1-sided` / `cert 2-sided` are the same two SCS flags as on KITTI and are upper bounds / necessary conditions, not the rigorous Schur bound. `ms` is the median wall-clock from the JSON (the `.md` prints the mean). The `depth`-only ALM configuration is also in the JSON (bias $20$ px $+$ 5% wrong: rotation RMSE 12.18°, translation RMSE 820.4 mm, FoV-out 42.1% because FoV is not enforced there).*

Four readings.

**(i) Clean and 3 px are inactive cells.** Every method returns the same pose (0.12°/8.26 mm and 0.42°/10.90 mm), with zero depth and FoV violations; ALM is 8× cheaper (18–19 ms vs 143–146 ms) because no constraint is active. The UTIAS stereo rig has no measured world-up prior here, so no tilt cone is used on this dataset.

**(ii) At 20 px bias and 5% wrong matches the layer repairs the depth tail and part of the rotation tail.** The unconstrained solve has 10.5% depth violations, 42.1% FoV violations, rotation RMSE 57.57° and translation RMSE 1375.3 mm. ALM is at 0.0% depth, 36.8% FoV, 10.87° and 603.0 mm, all 100% feasible. The FoV rate barely moves because the FoV column is a separate geometric check of *all* 80 estimated projections against the image rectangle, while the constrained solve is feasible to $10^{-6}$ in the constraint rows; the two quantities are not the same test, and we report both rather than equating them. The depth constraint removes the depth violation entirely.

**(iii) The extra FoV rows cost certificate rate, not feasibility.** The depth-only ALM configuration (in the JSON) certifies 89.5% one-sided at this setting; adding four FoV half-spaces per landmark lowers it to 84.2% while improving rotation RMSE 12.18°→10.87° and translation RMSE 820.4→603.0 mm. The certificate is for the stronger constrained problem, so a lower rate with more constraints is expected: the relaxation has more rows to satisfy. We report the depth+FoV configuration as the UTIAS result and the depth-only one as the ablation.

**(iv) Scope.** This is one run, 38 pairs, layer-level, with GT-projected observations and no rigorous Schur repair; it is a localisation *mechanism check* on real stereo landmarks, not a VO/SLAM result and not comparable to KITTI odometry leaderboards. The tilt column is absent by design (no world-up prior measured for this rig); the two certification columns are the same non-rigorous SCS flags as on KITTI.

### 5.10 Pose error on the real instances: ADD / ADD-S

Tables 5.3–5.5 report penetration, rotation and translation; a pose benchmark additionally reports ADD and ADD-S. Table 5.17 evaluates them on the *same* real trials as Table 5.10 (the pose dump replays `raw.json` row-for-row, $\max|\Delta|=0$), on the official BOP evaluation meshes, with ADD for asymmetric objects and ADD-S for the symmetric ones (LINEMOD/LINEMOD-Occlusion objects 8 "eggbox" and 9 "glue").

**Table 5.17. BOP ADD / ADD-S on the four methods, same trials as Table 5.10 (mm, median [IQR]; rates with Wilson 95% CI).**

| dataset | method | ADD | ADD-S | ADD(-S) | ADD(-S)$<$2 cm | ADD(-S)$<$10% diam | pen (mm) | feasible | rot (deg) | trans (mm) |
|---|---|---|---|---|---|---|---|---|---|---|
| LINEMOD ($n{=}90$) | Unconstrained | 35.9 [19.2, 47.2] | 15.1 [9.2, 19.3] | 31.5 [18.8, 38.6] | 31.1% (28/90) | 23.3% | 11.3 | 13.3% | 2.08 | 35.6 |
| | Soft penalty | 40.3 [24.8, 54.7] | 17.6 [10.8, 27.6] | 37.1 [22.6, 51.9] | 16.7% (15/90) | 15.6% | 0.0 | 88.9% | 2.13 | 39.8 |
| | Post-hoc projection | **31.1** [18.8, 38.3] | **12.8** [8.6, 16.8] | **28.1** [16.3, 36.4] | **32.2%** (29/90) | **27.8%** | 0.0 | 100% | 2.05 | **30.8** |
| | ALM-QPEP (ours) | 38.0 [24.6, 52.8] | 16.3 [10.6, 21.6] | 35.4 [21.7, 50.0] | 18.9% (17/90) | 17.8% | 0.0 | 100% | **1.97** | 37.7 |
| LM-O ($n{=}72$) | Unconstrained | 35.0 [19.7, 49.6] | 15.4 [8.8, 20.8] | 22.1 [16.7, 35.5] | 43.1% (31/72) | 44.4% | 14.9 | 6.9% | 2.07 | 34.7 |
| | Soft penalty | 39.6 [22.1, 53.6] | 16.3 [9.3, 23.6] | 30.1 [17.9, 49.1] | 34.7% (25/72) | 33.3% | 0.0 | 91.7% | 1.90 | 39.9 |
| | Post-hoc projection | **31.1** [18.1, 43.1] | **11.2** [7.4, 16.1] | **18.1** [13.4, 33.4] | **54.2%** (39/72) | **48.6%** | 0.0 | 100% | 2.04 | **31.3** |
| | ALM-QPEP (ours) | 38.2 [21.7, 51.8] | 15.9 [9.3, 21.2] | 25.4 [17.5, 43.5] | 36.1% (26/72) | 34.7% | 0.0 | 100% | **1.99** | 38.1 |
| YCB-Video ($n{=}84$) | Unconstrained | 17.3 [9.8, 23.9] | 8.8 [5.2, 11.3] | 17.3 [9.8, 23.9] | 64.3% (54/84) | 50.0% | 14.1 | 6.0% | 1.08 | 17.3 |
| | Soft penalty | 29.4 [17.5, 55.8] | 11.7 [7.5, 29.1] | 29.4 [17.5, 55.8] | 32.1% (27/84) | 21.4% | 0.0 | 90.5% | 1.40 | 29.4 |
| | Post-hoc projection | **8.9** [6.6, 13.3] | **4.0** [3.1, 5.9] | **8.9** [6.6, 13.3] | **96.4%** (81/84) | **85.7%** | 0.0 | 100% | **1.08** | **8.9** |
| | ALM-QPEP (ours) | 25.2 [16.7, 37.0] | 10.1 [6.6, 16.8] | 25.2 [16.7, 37.0] | 40.5% (34/84) | 23.8% | 0.0 | 100% | 1.11 | 25.2 |

*Source: `results/real_bop_add.json` / `.md` (generated from the pose dump that replays `real/*/results/raw.json` row-for-row; 95% CI for the medians is a 2000-draw percentile bootstrap, rates use Wilson). ADD is on the BOP `models_eval` meshes; ADD-S is used for the symmetric objects (LM/LM-O 8 and 9). The YCB-Video rows have ADD $=$ ADD-S $=$ ADD(-S) because none of the three objects is symmetric in this subset. Certification is not repeated in this table; the rigorous ALM-only rates on **these same poses** (replay check $\max|\Delta J|=0$) are base 22/90 (24.4%) $\to$ 46/90 (51.1%) on LINEMOD, 11/72 (15.3%) $\to$ 30/72 (41.7%) on LM-O and 11/84 (13.1%) $\to$ 20/84 (23.8%) on YCB-Video with the sound bound-RLT family (Table 5.11). The counts in the `ADD(-S)$<$2 cm` column (28/90, 31/72, 54/84 unconstrained; 17/90, 26/72, 34/84 ALM) are unchanged from the earlier run, so the correction in this round is the certification mapping, not the ADD values.*

Four readings.

**(i) The constraints are doing what they can on this protocol: they remove penetration, not error.** The unconstrained solve penetrates by 11.3/14.9/14.1 mm; the three constrained methods are at 0.0 mm. Rotation error is essentially flat (1.97–2.13° on LM, 1.90–2.07° on LM-O, 1.08–1.40° on YCB-Video). Translation error is *not* improved — ALM is 37.7/38.1/25.2 mm against projection's 30.8/31.3/8.9 mm — because the downward pixel bias (10/20/30 px) is the dominant injected error and the table constraint cannot undo it.

**(ii) ADD(-S) therefore does not improve with constraints, and we do not claim it does.** Projection is the best method on all three datasets (28.1/18.1/8.9 mm), ALM is 35.4/25.4/25.2 mm, unconstrained is 31.5/22.1/17.3 mm. On LM and LM-O the ALM result is worse than the *unconstrained* one: on ground-truth-projected correspondences with a large downward bias, pulling the pose up to the table moves it away from the biased 2D observations, and the bias is the error. This is the ADD-level confirmation of §5.1.5's warning and of observation (ii) in §5.2 and §5.5: admissibility and accuracy are different objectives, and a layer that enforces one does not automatically improve the other. The place where ALM's admissibility pays is the penetration and certification columns, not ADD.

**(iii) The absolute ADD numbers are large because the protocol is a stress test, not a leaderboard.** A 10/20/30 px downward bias is far outside detector noise, and the correspondences are 30 GT-projected vertices, not a detector's output. ADD(-S)$<$2 cm is 31/90, 43/72 and 64/84 for the unconstrained solve and 17/90, 26/72, 34/84 for ALM; these numbers are informative about the *relative* behaviour of the methods on identical inputs, and must not be compared with BOP leaderboard entries that use learned detectors and real segmentation masks (§1.4, §5.11).

**(iv) Symmetry handling is explicit.** Only LINEMOD/LINEMOD-Occlusion objects 8 (eggbox) and 9 (glue) are treated as symmetric; every YCB-Video object in this subset is asymmetric, so ADD and ADD-S coincide there. The per-object breakdown (15 LM objects, 3 LM-O, 3 YCB-Video) is in `results/real_bop_add.md`.

### 5.11 Scope: what is not run in this draft, and why

The paper's claims are bounded by what was actually executed. This subsection lists the gaps explicitly, with the reason.

* **No learned detector, matcher, or VLA.** Every real-data number in §5.5, §5.9 and §5.10 uses correspondences obtained by projecting known 3D structure with a ground-truth pose, then perturbing them. This is deliberate: the object of study is the *layer*, and a detector would add a confound we cannot control at this scale. The consequence is that no number here is a BOP or KITTI leaderboard result, and none should be compared with one.
* **No network-scale end-to-end training (T1-network).** The full T1 protocol of §5.1.4 — a keypoint-regression backbone with the layer as its pose head, trained on YCB-Video/LINEMOD, evaluated with ADD(-S) — is not run. The reason is compute: the protocol budgets 350–550 GPU-hours (§`experiments/protocol.md`), this work was executed on a shared CPU login node with no GPU allocation, and no down-scaled training run would be comparable to the SDPRLayers/BPnP numbers it is meant to be compared with. What *is* run is the CPU toy of §5.8: a 2-layer MLP refiner over 14 keypoints trained for a few hundred steps on synthetic data, with all four heads, which shows that the observation→pose loop closes at this scale; the full finite-difference check of the FB backward over all four heads was not completed (§5.8), so no FB-training-gradient claim is made.
* **SDPRLayers and dQP/CVXPYLayers were re-implemented; Theseus is the official library, forward only.** §5.7 states this per row: the SDPRLayers comparison is our NumPy/CVXPY re-implementation of their forward Shor relaxation (no official PyTorch layer, no dual-certificate backward); the dQP comparison is our active-set QP layer, cross-checked against `cvxpylayers` on random QPs but not against the dQP package; Theseus is the official `theseus-ai 0.2.1` forward solver, made importable with an isolated `scikit-sparse`/CHOLMOD overlay, and its own `TheseusLayer` backward was not measured. The first two are labelled "our re-implementation" in §5.7 and in the reference list; Theseus is not.
* **TUM RGB-D and the T3 equality-constrained tasks are not run.** TUM RGB-D adds a second indoor localisation dataset whose value here would be redundant with KITTI/UTIAS; T3 (hand–eye calibration, point-to-plane registration with a joint-limit prior) tests equality constraints in the lifted form, which is a different mechanism from the inequality story and is left out to keep §5 focused. The protocols remain in §5.1.4 and `experiments/protocol.md` for reproduction.
* **The exact-Hessian and FB-$\varepsilon$ ablations of §4.5 are not measured.** The default is the Gauss–Newton $\mathbf H_{\rm tan}$ of (9a); Remark 4.1's exact-Hessian variant and a sweep of $\varepsilon$ are stated but not run, so the size of the Gauss–Newton bias on real data is an open quantitative question (G5), not a measured quantity. The theory (Theorem 2.3, Proposition 2.1) is stated for both variants; only the GN one is exercised.
* **The 16-root saturated template is not shipped.** §4.7 reports the $4/12/16$ spectrum in canonical form and the 40-root template end-to-end; the saturated $16\times16$ action matrix for the isotropic single-plane family is established algebraically in the user-supplied proof only, and `experiments/code/isotropic_roots.py` solves that family with a 162-path homotopy plus multistart instead. This is an implementation gap, not a claim.
* **The rigorous P2-Cert bound is partial on localisation.** On KITTI it is computed for the clean and 6-px settings only (126/126 base-cut certificates at Schur gaps $5\times10^{-9}$–$6\times10^{-8}$, §5.9); the 15%- and 50%-outlier cells, and all of UTIAS, report only the one-sided/two-sided SCS flags. The Schur repair is applied in full to BOP (§5.5, Table 5.11) and to the synthetic cells (§5.4, Table 5.8b).
* **No production `torch.autograd.Function` layer is released with this draft.** The experiments run the research implementation under `experiments/code/` plus the CPU toy of §5.8. A packaged layer with the fail-safe contract of §1.3 is future work.

The corresponding *positive* scope statement is in §1.4; the two lists are meant to be read together.

---

## 6 Discussion and Limitations

**What the layer buys.** Hard constraints at a cost that does not grow with their number; a derivative that survives rank-deficient and weakly active contacts; and a posteriori evidence of globality whose cost is independent of $p$. The three together are what a manipulation stack needs: physical admissibility, trainability, and a failure signal.

**What it does not buy.** Globality of the forward solve. The ALM is a KKT method; the certificate can fail. We consider this the right trade: a layer that says "uncertified" is more useful than one that says "certified" incorrectly.

**The four-leg example is narrower than it looks.** The flagship LICQ-failure case is differentiable (and our method is exact) when the parameter moves the support surface rigidly, and non-differentiable when the contact heights are independent. The paper must present both, because a reviewer will ask which one the experiments instantiate. Our answer: in T1 the support plane is estimated once per frame from the depth channel and is therefore a *rigid* parameter of the layer's input, so compatibility holds; the non-rigid case arises when the four contact heights are themselves learned, which we do not do.

**Bias–variance of the certificate.** The $14\times14$ relaxation is tight on our synthetic instance and on the unconstrained sub-case, but tightness is an empirical property, not a theorem. On real data it is tight at a dataset-dependent rate: 24.4/15.3/13.1% of ALM poses on LM/LM-O/YCB-Video under the base cuts, 51.1/41.7/23.8% once the sound bound-RLT family is added (§5.5, Table 5.11). Where it fails, the paper's claim reduces to "sound lower bound plus a feasibility check", which is still useful (it bounds the suboptimality) but is not "certification"; and the uncertified rows are genuinely slack (median gap $1$–$2\times10^{-5}$), not numerically inconclusive. The soundness half is no longer empirical: with the dual repair of Theorem P2-Cert, every reported rigorous gap is non-negative by construction, so the failure mode is *slack*, never a false certificate.

**Scope.** Non-affine constraints (distance, clearance) are outside; they would break the constant-dimension reduction because $g_j^2$ would no longer be a quadratic form in $(\mathbf v_2,\mathbf t)$. Multiple global optima exchanging are outside; the derivative does not exist there, and the right treatment is a mixture/Laplace model, which is a different paper in this family.

**Relation to the other two prototypes in this programme.** This work is deliberately orthogonal: it does not use the lapalace/mixture machinery for multi-modality, and it does not use robust cost duality for outliers. It assumes a single well-defined constrained optimum and a unimodal posterior, and it spends its complexity budget on constraints and on rank-robust differentiation. Where those assumptions fail, the other two prototypes are the intended tools; we do not depend on their unpublished results anywhere.

---

## 7 Conclusion

We presented Constrained-QPEPLayer, a differentiable 6-DoF pose layer for inequality-constrained quaternion pose estimation. The method reduces every augmented-Lagrangian inner subproblem to a fixed-dimension quartic form on $\mathbb{S}^3$ regardless of the number of constraints, certifies global optimality a posteriori with a $14\times14$ semidefinite relaxation whose PSD block is independent of $p$, and differentiates through a Fischer–Burmeister-regularised saddle-point system that stays invertible at rank-deficient and weakly active contacts. We gave a precise account of the $\varepsilon\to0$ limit of that system — an $H$-weighted minimum-norm solution, exact under an explicit compatibility condition and a selected generalized derivative otherwise — and we replaced the "100% certification" of an earlier draft with a conditional theorem plus a sound relaxation that we verified to be tight to $2\times10^{-11}$ on a controlled instance where the dual-function gap is negative. Two of the three open questions of the earlier draft are now settled: the inner root count by Theorem P2-1b (a $4/12/16/32/40$ spectrum, with the generic $40$ intact even at rank $9$) and the numerical rigour of the certificate by Theorem P2-Cert (a dual-repair lower bound that cannot overshoot). The generalized derivative selected at weak-active points remains open. The tightness rate on real data is no longer open as a measurement: the certificate is tight on 24.4/15.3/13.1% of poses with base cuts and 51.1/41.7/23.8% with the sound bound-RLT family (Table 5.11), and the remaining rows fail by a median gap of $10^{-5}$, i.e. by genuine relaxation slack rather than by solver error.

**Acknowledgement of internal evidence.** The reading of SDPRLayers and QPEP used here is documented, with evidence grades, in this repository's knowledge base `kb/sdpr-qpep/` (files 01–04); the reproduction of SDPRLayers' calibration experiment is in `report/reproduction.md`. Those are internal sources; all formal citations are to the original papers.

---

## References

> Every entry below was checked against Crossref (DOI, volume, pages), PMLR, the publisher page, or arXiv (identifier, title, author list) on 2026-11-28. No entry is marked **[verify]** any more: the two open items of the previous draft are resolved — the NeurIPS 2019 page range of entry 6 is 9562–9574 (NeurIPS proceedings metadata), and the Theseus proceedings entry (4) is confirmed at pp. 3801–3818. Entries 1–3 correspond to code that we *re-implemented* for the baseline table (§5.7); entry 4 is the official Theseus library, which we ran forward-only through an isolated CHOLMOD overlay. The reference list does not imply that the official SDPRLayers or dQP software was run.

1. C. Holmes, F. Dümbgen, T. D. Barfoot. *SDPRLayers: Certifiable Backpropagation Through Polynomial Optimization Problems in Robotics.* IEEE Transactions on Robotics, 41:4120–4138, 2025. DOI 10.1109/TRO.2025.3578228; arXiv:2405.19309.
2. J. Wu, Y. Zheng, Z. Gao, Y. Jiang, X. Hu, Y. Zhu, J. Jiao, M. Liu. *Quadratic Pose Estimation Problems: Globally Optimal Solutions, Solvability/Observability Analysis, and Uncertainty Description.* IEEE Transactions on Robotics, 38(5):3314–3335, 2022. DOI 10.1109/TRO.2022.3155880.
3. C. W. Magoon, F. Yang, N. Aigerman, S. Z. Kovalsky. *Differentiation Through Black-Box Quadratic Programming Solvers.* Advances in Neural Information Processing Systems 38 (NeurIPS 2025), pp. 120073–120104. DOI 10.52202/085713-3617; arXiv:2410.06324.
4. L. Pineda, T. Fan, M. Monge, S. Venkataraman, P. Sodhi, R. T. Q. Chen, J. Ortiz, D. DeTone, A. Wang, S. Anderson, J. Dong, B. Amos, M. Mukadam. *Theseus: A Library for Differentiable Nonlinear Optimization.* Advances in Neural Information Processing Systems 35 (NeurIPS 2022), pp. 3801–3818. DOI 10.52202/068431-0275; arXiv:2207.09442.
5. B. Amos, J. Z. Kolter. *OptNet: Differentiable Optimization as a Layer in Neural Networks.* ICML 2017, PMLR 70:136–145. arXiv:1703.00443.
6. A. Agrawal, B. Amos, S. Barratt, S. Boyd, S. Diamond, J. Z. Kolter. *Differentiable Convex Optimization Layers.* Advances in Neural Information Processing Systems 32 (NeurIPS 2019), pp. 9562–9574. arXiv:1910.12430.
7. B. Chen, Á. Parra, J. Cao, N. Li, T.-J. Chin. *End-to-End Learnable Geometric Vision by Backpropagating PnP Optimization.* CVPR 2020, pp. 8097–8106. DOI 10.1109/CVPR42600.2020.00812; arXiv:1909.06043.
8. H. Chen, P. Wang, F. Wang, W. Tian, L. Xiong, H. Li. *EPro-PnP: Generalized End-to-End Probabilistic Perspective-n-Points for Monocular Object Pose Estimation.* CVPR 2022, pp. 2771–2780. DOI 10.1109/CVPR52688.2022.00280; arXiv:2203.13254.
9. P. L. Donti, D. Rolnick, J. Z. Kolter. *DC3: A Learning Method for Optimization with Hard Constraints.* ICLR 2021. arXiv:2104.12225.
10. A. Fischer. *A Special Newton-Type Optimization Method.* Optimization, 24(3–4):269–284, 1992. DOI 10.1080/02331939208843795.
11. C. Kanzow. *Some Noninterior Continuation Methods for Linear Complementarity Problems.* SIAM Journal on Matrix Analysis and Applications, 17(4):851–868, 1996. DOI 10.1137/S0895479894273134.
12. F. Facchinei, J.-S. Pang. *Finite-Dimensional Variational Inequalities and Complementarity Problems.* Springer, 2003.
13. D. P. Bertsekas. *Constrained Optimization and Lagrange Multiplier Methods.* Academic Press, 1982.
14. R. T. Rockafellar, R. J.-B. Wets. *Variational Analysis.* Springer, 1998.
15. R. Fletcher. *Practical Methods of Optimization* (2nd ed.). Wiley, 1987.
16. J. Bolte, T. Le, E. Pauwels, A. Silveti-Falls. *Nonsmooth Implicit Differentiation for Machine Learning and Optimization.* Advances in Neural Information Processing Systems 34 (NeurIPS 2021). arXiv:2106.04350.
17. J. Nocedal, S. J. Wright. *Numerical Optimization* (2nd ed.). Springer, 2006.
18. B. O'Donoghue, E. Chu, N. Parikh, S. Boyd. *Conic Optimization via Operator Splitting and Homogeneous Self-Dual Embedding.* Journal of Optimization Theory and Applications, 169:1042–1068, 2016. DOI 10.1007/s10957-016-0892-3. (SCS, the solver used for the certificates.)
19. T. Hodaň, F. Michel, E. Brachmann, W. Kehl, A. Glent Buch, D. Kraft, B. Drost, J. Vidal, S. Ihrke, X. Zabulis, C. Sahin, F. Manhardt, F. Tombari, T.-K. Kim, J. Matas, C. Rother. *BOP: Benchmark for 6D Object Pose Estimation.* ECCV 2018, LNCS 11214, pp. 19–35. DOI 10.1007/978-3-030-01249-6_2.
20. S. Hinterstoisser, V. Lepetit, S. Ilic, S. Holzer, G. Bradski, K. Konolige, N. Navab. *Model Based Training, Detection and Pose Estimation of Texture-Less 3D Objects in Heavily Cluttered Scenes.* ACCV 2012, LNCS 7724, pp. 548–562. DOI 10.1007/978-3-642-37331-2_42. (LINEMOD dataset and the ADD metric.)
21. E. Brachmann, A. Krull, F. Michel, S. Gumhold, J. Shotton, C. Rother. *Learning 6D Object Pose Estimation Using 3D Object Coordinates.* ECCV 2014, LNCS 8690, pp. 536–551. DOI 10.1007/978-3-319-10605-2_35. (LINEMOD-Occlusion subset.)
22. Y. Xiang, T. Schmidt, V. Narayanan, D. Fox. *PoseCNN: A Convolutional Neural Network for 6D Object Pose Estimation in Cluttered Scenes.* Robotics: Science and Systems XIV, 2018. DOI 10.15607/RSS.2018.XIV.019; arXiv:1711.00199. (YCB-Video dataset and the ADD-S metric.)
23. A. Geiger, P. Lenz, R. Urtasun. *Are we ready for autonomous driving? The KITTI vision benchmark suite.* CVPR 2012, pp. 3354–3361. DOI 10.1109/CVPR.2012.6248074.
24. M. Gridseth, T. D. Barfoot. *Keeping an Eye on Things: Deep Learned Features for Long-Term Visual Localization.* IEEE Robotics and Automation Letters, 7(2):1016–1023, 2022. DOI 10.1109/LRA.2021.3136867; arXiv:2109.04041. The UTIAS Long-Term Localization and Mapping Dataset (the "In-The-Dark" and "Multiseason" subsets) is distributed at https://asrl.utias.utoronto.ca/datasets/2020-vtr-dataset/ ; we found no separate dataset paper, so the dataset is cited through this reference and the official URL (the page confirms that `run_000000` is the Teach run and that repeat runs ship `transforms_spatial.txt`).
