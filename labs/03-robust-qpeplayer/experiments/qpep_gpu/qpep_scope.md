# QPEP scope: scalar-weight 3D-3D, the 4×4 eigen special case, and the 40-root solver

Status: 2026-11-27, P3 engineering note. **No new mathematics is proved here.** This
file fixes the vocabulary used by `paper/draft.md` §5 and by
`experiments/qpep_gpu/`, and it states exactly what the repository does and does
not implement. Evidence is the measured JSON in `experiments/inner/`,
`experiments/systems/`, `experiments/ablations/anisotropic/` and
`experiments/qpep_gpu/`, plus the upstream QPEP description in
`kb/sdpr-qpep/03.md` (Wu et al., T-RO 2022).

---

## 1. Scalar-weight 3D-3D QPEP **is** weighted Kabsch / Horn 4×4

The problem solved by every 3D-3D table in this paper is

$$
\min_{R\in SO(3),\,\mathbf t\in\mathbb R^3}\;
\sum_{i=1}^{N} w_i\,\big\|R\mathbf p_i+\mathbf t-\mathbf y_i\big\|^2 ,
\qquad w_i\ge 0 .
\tag{1}
$$

This is the weighted QPEP with **scalar** weights, i.e. every per-observation
weight matrix is $W_i=w_i I_3$. Eliminating the translation gives
$\mathbf t^\*(R)=\bar{\mathbf y}-R\bar{\mathbf x}$ with normalized weights, and
(1) becomes the orthogonal Procrustes problem

$$
\max_{R\in SO(3)}\operatorname{tr}\!\big(RH\big),
\qquad
H=\sum_i w_i\,(\mathbf p_i-\bar{\mathbf x})(\mathbf y_i-\bar{\mathbf y})^\top .
\tag{2}
$$

Horn's quaternion form writes (2) as $\mathbf q^\top K(H)\mathbf q$ with a
symmetric $4\times4$ matrix $K(H)$; the maximizer is the unit eigenvector of the
largest eigenvalue, and $R(\mathbf q)$ is the standard quaternion-to-rotation
map. The SVD form of weighted Kabsch returns the same proper rotation. For
scalar weights this is an **exact closed form**, not an approximation of the
general QPEP: there is no polynomial system to enumerate, because (2) has a
unique global maximizer generically (the sign of $\mathbf q$ represents the
same rotation).

Measured evidence in this repository:

| check | result | artifact |
|---|---|---|
| Horn/Davenport 4×4 vs weighted Kabsch | max $\lVert R_d-R_k\rVert_F=1.07\times10^{-14}$, max cost excess $9.4\times10^{-16}$, 1000/1000 same solution | `experiments/inner/results/inner_solver.json` |
| batched NumPy 4×4 eigh vs the per-instance loop | rotation $\le1.1\times10^{-14}$, translation $\le1.2\times10^{-15}$, identical $\mu$-schedules | `experiments/systems/results/batch_gnc.json` |
| GPU static 4×4 eigh/ Jacobi vs NumPy reference | see `experiments/qpep_gpu/results/qpep_gpu.{json,md}` (numerical agreement; timing status depends on GPU availability) | `experiments/qpep_gpu/` |

The 4×4 solver is therefore what `paper/draft.md` §5 means by “the main forward
solver is weighted Kabsch, the closed-form scalar-weight 3D-3D specialization of
QPEP”. It is the **inner** solver of GNC-GM; it says nothing about the outer
GNC's global optimality.

### 1.1 What changes when the weights are matrices

For a full SPD $W_i$ the residual is $e_i=\mathbf r_i^\top W_i\mathbf r_i$ and
(1) becomes a nonlinear 6-DoF problem: after eliminating $\mathbf t$ the
rotation-dependent part is no longer $\operatorname{tr}(RH)$. The repository
handles that case with a 6-DoF Gauss–Newton on Cholesky-whitened residuals
(`experiments/ablations/anisotropic/`), and the ablation shows that scalar
Kabsch is **not** exact there (objective ratio 1.035–1.088 for full SPD $W_i$,
single-trial max 3.80). That GN solver is a local nonlinear least-squares
solve, **not** the QPEP algebraic-global solve either.

---

## 2. What the full 40-root polynomial solves

The general QPEP (Wu et al., T-RO 2022) targets the same 3D-3D cost with full
$W_i$, and also PnP, PnL, hand–eye, point-to-plane and Stewart-platform
variants. Its algebraic route is:

1. write the rotation as a unit quaternion $\mathbf q$ and eliminate
   $\mathbf t$ in closed form,
   $\mathbf t(\mathbf q)=G(w)^{-1}\sum_i w_iW_i(\mathbf y_i-R(\mathbf q)\mathbf p_i)$,
   $G(w)=\sum_iw_iW_i$;
2. the remaining cost is a quartic polynomial in $\mathbf q$; the KKT system
   consists of the quartic stationarity equations in $(\mathbf q,\lambda)$ plus
   $\mathbf q^\top\mathbf q=1$;
3. a Gröbner-basis / action-matrix elimination returns **up to 40 candidate
   complex roots** (the official `solver_WQ` path stores a $4\times40$ complex
   matrix, `solution_size=40`, in `LibQPEP/src/QPEP_grobner.cpp`; see
   `kb/sdpr-qpep/03.md` §2.3);
4. each root's real part is normalized to a unit quaternion, $\mathbf t$ is
   recovered by back-substitution, the original cost is evaluated, the roots
   are sorted, and the best one is refined by Levenberg–Marquardt; covariance
   is a separate convex SDP.

So the “40 roots” are the **enumerated stationary points of the general
weighted problem**, not 40 different pose problems and not a 40-dimensional
state. The algebraic-global claim is conditional: it assumes the elimination
did not miss a root and that the numerical basis is well conditioned. It is not
a convex certificate (no dual matrix / rank-one certificate; see
`kb/sdpr-qpep/03.md` §2.4).

For **scalar weights**, the same problem collapses to (2): the quartic system
has the 4×4 eigenproblem as its closed-form solution, the global maximizer is
the top eigenvector, and there is no need to enumerate roots. In that case the
40-root count is irrelevant to the answer — it is a property of the general
solver, not of the scalar-weight special case.

### 2.1 What a faithful implementation would need

- O(N) accumulation of the fixed-size coefficient matrices, then a
  Gröbner/action-matrix solve per problem instance (a C++ library such as
  LibQPEP, or an equivalent polynomial-system implementation);
- complex root extraction, real-part/normalize heuristics, cost evaluation,
  sorting, and an LM polish for each selected root;
- explicit failure handling: missed/duplicate roots, degenerate or multiple
  roots, singular $G(w)$, ill-conditioned action matrix, and a cost gap between
  the best and second-best root;
- a separate covariance SDP (CSDP/SeDuMi) if covariances are wanted;
- for hard real-time use, a batched/GPU story. This is the hard part: the
  elimination has data-dependent control flow and host-side work, so it cannot
  be captured as a static CUDA graph, and a batched action-matrix solve would
  need a custom kernel. The current GPU probe therefore batches the **4×4
  eigen** inner solver, not the 40-root solver.

### 2.2 Why it is not needed or feasible for the current tables

- Every 3D-3D table in the paper uses scalar weights ($W_i=I$ in the main
  protocol). For those tables the 4×4 eigen solve is exact and gives the same
  answer a 40-root solver would: when $W_i=w_iI$, the global optimum is exactly
  the closed-form solution.
- The only full-$W_i$ experiment is the anisotropic ablation, and it already
  uses a correct 6-DoF GN inner solve; adding a 40-root solver there would
  change the inner solver, not the paper's scalar-weight claims, and its
  measured effect would need a new protocol.
- No current number depends on root completeness, failure rate, or a
  certificate. Implementing and validating a 40-root solver (including its
  failure modes) is a separate research/engineering project, not a missing
  measurement for the present claims. The paper already states “no 40-root QPEP
  solver is implemented”; §5.8 lists it as not measured.

---

## 3. Exact non-claims

1. **No 40-root implementation.** This repository does not implement, batch,
   benchmark, or validate the QPEP Gröbner/action-matrix solver, its root
   completeness, or its failure rate.
2. **No general-QPEP evidence from the 4×4 results.** The inner-solver and GPU
   results are evidence for the scalar-weight special case (2) only. They do not
   show that the general QPEP is fast, robust, or correct on full $W_i$, PnP, or
   any other QPEP problem class.
3. **No certificate.** “QPEP is algebraic-global” is not “certifiably globally
   optimal”; the repository has no dual/rank-one certificate for the inner
   problem, and the paper does not claim one.
4. **No global claim for GNC.** The outer GNC is a local block-coordinate
   method. Even a perfect 40-root inner solver would not make the robust cost
   globally optimized, and coherent majority outliers can make the global
   robust optimum wrong (identifiability, not solver quality).
5. **No anisotropic exactness claim for scalar Kabsch.** Scalar weighted Kabsch
   is exact only for scalar weights; for full SPD $W_i$ it is an approximation
   (measured objective ratio 1.035–1.088, max 3.80 in the ablation), and the GN
   inner used there is a local solve, not a global one.
6. **No GPU claim for the full QPEP.** The GPU work in `experiments/qpep_gpu/`
   concerns the 4×4 eigen inner solver and the GNC schedule, not the 40-root
   polynomial algebra; CUDA-graph capture of `torch.linalg.eigh` is a separate
   numerical/engineering result and is reported with its exact error in
   `results/qpep_gpu.md`.
