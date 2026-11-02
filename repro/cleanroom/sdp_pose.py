"""Problem (P6) of Holmes et al. (TRO 2025) as a homogeneous QCQP + Shor SDP
relaxation, solved with CVXPY/SCS.

Problem (P6) -- matrix-weighted stereo localisation
---------------------------------------------------
Given world-frame features ``m_k`` and camera-frame Euclidean measurements
``y_k = m_tilde_k(theta)`` with inverse covariance weights
``W_k = Sigma_k(theta)^{-1}``,

    min_{C in SO(3), t}  sum_k || y_k - C m_k + t ||^2_{W_k}          (P6)

where ``t = C r_p`` is the camera translation expressed in the camera frame
(``r_p`` being the camera position in the world frame; the paper writes the
same cost as ``y_k - C(m_k + t_paper)`` with ``t_paper = -r_p``).  For fixed
``C`` the optimal ``t`` is available in closed form, but we keep it as a
variable, as in the paper.

Homogeneous QCQP (Problem (P1))
-------------------------------
With ``x = [h, vec_col(C), t] in R^13`` (``h`` the homogenisation variable)

    Q(theta) = sum_k Q_k(theta)  in S^13,
    x^T Q x = sum_k || y_k - C m_k + t ||^2_{W_k}   for  h = 1,

and the constraints ``x^T A_i x = 0`` (i = 1..24) encode ``C in SO(3)``:

    6  orthogonality constraints   C_i . C_j - delta_ij h^2 = 0
    9  handedness constraints      (C_j x C_k)_l - h C_li = 0  (cyclic)
    9  row/column norm constraints ||col_j(C)||^2 - ||row_i(C)||^2 = 0

the last 18 being the redundant constraints used by the paper to tighten the
relaxation.  The homogenising constraint is ``x^T A_0 x = 1`` with
``A_0 = e_h e_h^T``.

Shor relaxation (Problem (P3)): drop ``rank(X) = 1``.

Dual certificate
----------------
``H = Q + sum_i lambda_i A_i + lambda_0 A_0 >= 0`` with ``H x = 0`` at the
optimum (Sec. III-B of the paper).  ``H`` is assembled here from the equality
constraint duals returned by the conic solver.
"""

from __future__ import annotations

import time

import cvxpy as cp
import numpy as np

__all__ = [
    "N_VARS",
    "HOMOG_IDX",
    "C_IDX",
    "T_IDX",
    "constraint_matrices",
    "homogenising_matrix",
    "build_objective_matrix",
    "objective_scale",
    "SDPPoseSolver",
    "recover_solution",
]

N_VARS = 13
HOMOG_IDX = 0
C_IDX = slice(1, 10)
T_IDX = slice(10, 13)


# ----------------------------------------------------------------------
# constraint matrices
# ----------------------------------------------------------------------
def _add_block(A, p, q, val):
    """Add ``val * x_p * x_q`` (symmetrically) to the quadratic form."""
    A[p, q] += val
    if p != q:
        A[q, p] += val


def constraint_matrices() -> list[np.ndarray]:
    """Return the 24 symmetric matrices ``A_i`` with ``x^T A_i x = 0``.

    Ordering matches the upstream layer so that the indices of the redundant
    constraints are ``6..23``:

    * ``0..5``: orthogonality ``C_i . C_j - delta_ij h^2 = 0``  (``i <= j``)
    * ``6..14``: handedness ``(C_j x C_k)_l - h C_li = 0``
    * ``15..23``: row/column norm equalities
    """
    A_list = []

    # --- 6 orthogonality constraints: C_i . C_j - delta_ij h^2 = 0 -----
    for i in range(3):
        for j in range(i, 3):
            A = np.zeros((N_VARS, N_VARS))
            for k in range(3):
                p = 1 + 3 * i + k   # c[3*i+k] = C[k, i]
                q = 1 + 3 * j + k
                if i == j:
                    A[p, q] += 1.0
                else:
                    _add_block(A, p, q, 0.5)
            if i == j:
                A[0, 0] -= 1.0      # -delta_ij h^2
            A_list.append(A)

    # --- 9 handedness constraints: (C_j x C_k)_l - h C_li = 0 ----------
    # cyclic triples (i, j, k) of columns, (l, m, n) of rows
    for i, j, k in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        for l, m, n in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
            A = np.zeros((N_VARS, N_VARS))
            # (C_j x C_k)_l = eps_{lmn} (C[m,j] C[n,k] - C[n,j] C[m,k])
            eps = 1.0 if (l, m, n) in ((0, 1, 2), (1, 2, 0), (2, 0, 1)) else -1.0
            p = 1 + 3 * j + m   # c[3*j+m] = C[m, j]
            q = 1 + 3 * k + n   # c[3*k+n] = C[n, k]
            _add_block(A, p, q, 0.5 * eps)
            p2 = 1 + 3 * j + n
            q2 = 1 + 3 * k + m
            _add_block(A, p2, q2, -0.5 * eps)
            # - h C[l, i]
            A[1 + 3 * i + l, 0] += -0.5
            A[0, 1 + 3 * i + l] += -0.5
            A_list.append(A)

    # --- 9 row/column norm equalities ---------------------------------
    for i in range(3):
        for j in range(3):
            A = np.zeros((N_VARS, N_VARS))
            for k in range(3):
                A[1 + 3 * j + k, 1 + 3 * j + k] += 1.0   # ||col_j||^2
                A[1 + i + 3 * k, 1 + i + 3 * k] -= 1.0   # -||row_i||^2
            A_list.append(A)

    return A_list


def homogenising_matrix() -> np.ndarray:
    """``A_0 = e_h e_h^T`` so that ``x^T A_0 x = h^2 = 1``."""
    A0 = np.zeros((N_VARS, N_VARS))
    A0[0, 0] = 1.0
    return A0


# ----------------------------------------------------------------------
# objective matrix
# ----------------------------------------------------------------------
def build_objective_matrix(m: np.ndarray, y: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Assemble ``Q(theta)`` of Problem (P6).

    Parameters
    ----------
    m : (3, N) world-frame landmarks.
    y : (3, N) Euclidean measurements in the camera frame.
    W : (N, 3, 3) inverse covariance weights.

    Returns
    -------
    (13, 13) symmetric matrix ``Q`` such that, for ``h = 1``,

        x^T Q x = sum_k (y_k - C m_k + t)^T W_k (y_k - C m_k + t).

    The blocks are (``h`` = 0, ``C`` = 1:10, ``t`` = 10:13)

        Q_hh = sum_k y_k^T W_k y_k
        Q_CC = sum_k (m_k m_k^T) (x) W_k
        Q_tt = sum_k W_k
        Q_Ch = -sum_k m_k (x) (W_k y_k)
        Q_th =  sum_k W_k y_k
        Q_Ct = -sum_k m_k (x) W_k
    """
    m = np.asarray(m)
    y = np.asarray(y)
    W = np.asarray(W)
    if m.shape[0] != 3:
        m = m.T
    if y.shape[0] != 3:
        y = y.T
    mT = m.T                     # (N,3)
    yT = y.T                     # (N,3)
    N = mT.shape[0]
    dtype = np.result_type(m, y, W, float)
    Q = np.zeros((N_VARS, N_VARS), dtype=dtype)

    Wy = np.einsum("nab,nb->na", W, yT)               # (N,3)

    Q[0, 0] = np.einsum("na,na->", Wy, yT)
    Q[C_IDX, C_IDX] = np.einsum("ni,nj,nab->iajb", mT, mT, W).reshape(9, 9)
    Q[T_IDX, T_IDX] = W.sum(axis=0)
    Qch = -np.einsum("ni,na->ia", mT, Wy).reshape(9)
    Q[C_IDX, 0] = Qch
    Q[0, C_IDX] = Qch
    Qth = Wy.sum(axis=0)
    Q[T_IDX, 0] = Qth
    Q[0, T_IDX] = Qth
    Qct = -np.einsum("ni,nab->iab", mT, W).reshape(9, 3)
    Q[C_IDX, T_IDX] = Qct
    Q[T_IDX, C_IDX] = Qct.T
    # symmetrise against round-off
    Q = 0.5 * (Q + Q.T)
    return Q


def objective_scale(Q: np.ndarray) -> float:
    """Solver-conditioning scale: ``1 / ||Q - Q_hh e_h e_h^T||_F``.

    The constant offset ``Q_hh`` is irrelevant on the feasible set (``h = 1``)
    and the positive scalar does not change the minimiser; freezing the scale
    (rather than recomputing it as a function of theta) keeps the derivative
    of the parameterised cost matrix free of a spurious scale term.  Any
    *constant* positive rescaling leaves the solution map unchanged.
    """
    Qr = np.array(Q, dtype=float)
    Qr[0, 0] = 0.0
    return 1.0 / np.linalg.norm(Qr, ord="fro")


def scale_objective(Q: np.ndarray, scale: float) -> np.ndarray:
    """Remove the constant term and rescale (used for solver conditioning)."""
    Qs = np.array(Q, dtype=float)
    Qs[0, 0] = 0.0
    Qs *= scale
    return 0.5 * (Qs + Qs.T)


# ----------------------------------------------------------------------
# SDP solver
# ----------------------------------------------------------------------
def recover_solution(X: np.ndarray) -> np.ndarray:
    """Recover ``x`` from the rank-1 matrix solution.

    ``X ~= x x^T`` with ``x_0 = 1`` (homogenising constraint ``X_00 = 1``),
    so ``x = X[:, 0]``.  The result is compared against the leading
    eigenvector of ``X`` in :meth:`SDPPoseSolver.solve` for diagnostics.
    """
    x = 0.5 * (X[:, 0] + X[0, :])
    return x


class SDPPoseSolver:
    """Shor SDP relaxation of Problem (P6) for a single pose.

    The CVXPY problem is built once (one ``cp.Parameter`` for the scaled
    objective matrix) and re-solved with new parameter values during the
    bilevel optimisation.

    Parameters
    ----------
    solver : str
        CVXPY solver name (``"SCS"`` by default, as in the paper).
    eps : float
        Absolute and relative solver tolerance (paper: ``1e-9``).
    max_iters : int
        Iteration cap for SCS.
    redun_inds : sequence of int
        Indices (in ``0..23``) of the constraints that are *redundant* with
        respect to the 6 orthogonality constraints; they are excluded from
        the reduced constraint Jacobian ``G_r`` used by Theorem 2.
    """

    def __init__(
        self,
        solver: str = "SCS",
        eps: float = 1e-9,
        max_iters: int = 100000,
        redun_inds=None,
        verbose: bool = False,
    ) -> None:
        self.solver_name = solver
        self.eps = eps
        self.max_iters = max_iters
        self.verbose = verbose
        self.A_list = constraint_matrices()
        self.A0 = homogenising_matrix()
        self.n_constraints = len(self.A_list)  # 24
        self.redun_inds = (
            list(range(6, self.n_constraints)) if redun_inds is None else list(redun_inds)
        )

        n = N_VARS
        self.X = cp.Variable((n, n), symmetric=True)
        self.Qp = cp.Parameter((n, n), symmetric=True)
        cons = []
        self._con_inds = []
        for A in self.A_list:
            cons.append(cp.trace(A @ self.X) == 0.0)
            self._con_inds.append(len(cons) - 1)
        self._homog_ind = len(cons)
        cons.append(cp.trace(self.A0 @ self.X) == 1.0)
        cons.append(self.X >> 0)
        self._psd_ind = len(cons) - 1
        self.problem = cp.Problem(cp.Minimize(cp.trace(self.Qp @ self.X)), cons)
        self.constraints = cons
        assert self.problem.is_dcp(dpp=True)

    # ------------------------------------------------------------------
    def solve(
        self,
        Q_scaled: np.ndarray,
        warm_start: bool = True,
        refine: bool = True,
    ) -> dict:
        """Solve the relaxation for one scaled objective matrix."""
        t0 = time.perf_counter()
        self.Qp.value = 0.5 * (Q_scaled + Q_scaled.T)
        solve_kwargs = dict(warm_start=warm_start, verbose=self.verbose)
        if self.solver_name.upper() == "SCS":
            solve_kwargs.update(
                eps_abs=self.eps, eps_rel=self.eps, max_iters=self.max_iters
            )
        elif self.solver_name.upper() == "CLARABEL":
            solve_kwargs.update(tol_gap_abs=self.eps, tol_gap_rel=self.eps,
                                tol_feas=self.eps)
        self.problem.solve(solver=self.solver_name, **solve_kwargs)
        t1 = time.perf_counter()
        if self.problem.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(f"SDP solve failed: {self.problem.status}")
        X = np.array(self.X.value, dtype=float)
        X = 0.5 * (X + X.T)
        # duals: 24 homogeneous constraints + homogenising constraint
        y = np.array(
            [self.constraints[i].dual_value for i in self._con_inds], dtype=float
        )
        y_hom = float(self.constraints[self._homog_ind].dual_value)
        t2 = time.perf_counter()

        # certificate matrix H = Q + sum_i y_i A_i + y_hom A_0
        H = np.array(Q_scaled, dtype=float)
        for yi, A in zip(y, self.A_list):
            H = H + yi * A
        H = H + y_hom * self.A0
        H = 0.5 * (H + H.T)

        x = recover_solution(X)
        if refine:
            x = self._refine_x(x)

        # diagnostics ---------------------------------------------------
        eigs_X = np.sort(np.linalg.eigvalsh(X))[::-1]
        # lambda_1 / lambda_2, capped at 1e16 (the ratio can otherwise
        # overflow once the second eigenvalue reaches machine precision)
        r_tight = float(eigs_X[0] / max(eigs_X[1], 1e-16 * max(eigs_X[0], 1e-300)))
        r_tight = min(r_tight, 1e16)
        eigs_H = np.linalg.eigvalsh(H)
        # corank at two relative tolerances (the exact certificate has
        # corank one; the tolerance only controls numerical noise)
        scale_H = max(abs(eigs_H[-1]), 1e-300)
        corank_H = int(np.sum(np.abs(eigs_H) < 1e-6 * scale_H))
        corank_H_strict = int(np.sum(np.abs(eigs_H) < 1e-9 * scale_H))
        # ratio of the second *smallest* eigenvalue to the largest one: a
        # proxy for "how close to corank one" the numerical certificate is
        eig2_eig1 = float(np.abs(eigs_H[1]) / scale_H)
        v1 = np.linalg.eigh(X)[1][:, -1]
        v1 = v1 / v1[0] if abs(v1[0]) > 1e-12 else v1
        rec = dict(
            X=X,
            x=x,
            y=y,
            y_hom=y_hom,
            H=H,
            lam=y,
            lam0=y_hom,
            eigs_X=eigs_X,
            r_tight=r_tight,
            eigs_H=eigs_H,
            corank_H=corank_H,
            corank_H_strict=corank_H_strict,
            eig2_eig1=eig2_eig1,
            eigvec_x=v1,
            Hx_res=float(np.linalg.norm(H @ x)),
            H_min_eig=float(eigs_H[0]),
            primal_obj=float(np.trace(Q_scaled @ X)),
            dual_obj=float(-y_hom),
            t_solve=t1 - t0,
            t_cert=t2 - t1,
            status=self.problem.status,
            solver=self.solver_name,
        )
        return rec

    @staticmethod
    def _refine_x(x: np.ndarray) -> np.ndarray:
        """Renormalise the homogenisation entry (``X_00 = 1`` already)."""
        x = np.array(x, dtype=float)
        if abs(x[0]) > 1e-12:
            x = x / x[0]
        return x

    # ------------------------------------------------------------------
    def reduced_gradient_rows(self, x: np.ndarray):
        """Constraint Jacobian rows ``G_r`` of Theorem 2 (7 x 13).

        The 6 orthogonality constraints plus the homogenising constraint; the
        18 redundant constraints are excluded, cf. ``redun_list`` upstream.
        """
        keep = [i for i in range(self.n_constraints) if i not in self.redun_inds]
        rows = [self.A_list[i] @ x for i in keep]
        rows.append(self.A0 @ x)
        return np.vstack(rows), keep

    def all_gradient_rows(self, x: np.ndarray) -> np.ndarray:
        """All 25 constraint Jacobian rows (24 + homogenising)."""
        rows = [A @ x for A in self.A_list]
        rows.append(self.A0 @ x)
        return np.vstack(rows)
