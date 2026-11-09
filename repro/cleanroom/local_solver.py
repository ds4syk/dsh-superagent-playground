"""Local (non-convex) solver baseline for Problem (P6).

Gauss-Newton on SO(3) for the matrix-weighted localisation problem

    min_{C in SO(3), r_p}  sum_k || y_k - C (m_k - r_p) ||^2_{W_k},

with the same iterates/termination philosophy as the Theseus Gauss-Newton
solver used in the paper (Sec. VI-B2: step size 0.2, absolute/relative error
tolerances, at most 500 iterations, warm-started with the previous solution
across outer iterations).

Note that this is exactly Problem (P6) with ``t = C r_p``; the local solver
is initialised either at the ground-truth pose ("local-GT") or at the random
initialisation of the paper's Fig. 3b ("local-RND"), in which case it can
converge to a spurious local minimum.

Implicit differentiation of the local solution
----------------------------------------------
At a local optimum the Riemannian gradient vanishes:

    g_delta = 2 sum_k (W_k e_k) x v_k = 0,        v_k = C (m_k - r_p)
    g_r     = 2 sum_k C^T W_k e_k     = 0,        e_k = y_k(b) - v_k

with ``delta`` a left perturbation of ``C`` (``C <- exp(delta) C``).  Both
``y_k`` and ``W_k`` depend on the parameter ``b``, so the implicit derivative
of the *local* solution map is

    [d delta; d r_p] / db = - (dg/d(delta, r_p))^{-1} dg/db ,

evaluated with central differences in the local coordinates (the Jacobian is
6 x 6, so numerical differentiation is both cheap and accurate).  This is the
same quantity that Theseus returns with ``backward_mode="implicit"``; the
implementation here is independent of Theseus.
"""

from __future__ import annotations

import numpy as np

from camera_model import exp_so3, skew

__all__ = [
    "pose_cost",
    "local_stationarity",
    "solve_pose_local",
    "local_implicit_derivative",
    "outer_loss",
]


# ----------------------------------------------------------------------
# problem container
# ----------------------------------------------------------------------
# The module keeps the landmark positions out of the function signatures used
# by the bilevel loop: they are carried by a module-level context object.
class _LocalProblem:
    """Everything needed to evaluate the local problem for one pose."""

    __slots__ = ("m", "meas", "W", "L")

    def __init__(self, m, meas, W):
        self.m = m                      # (3, N) world-frame landmarks
        self.meas = meas                # (3, N) measurements
        self.W = W                      # (N, 3, 3) weights
        # whitening factors: L_k^T L_k = W_k
        self.L = np.linalg.cholesky(W).transpose(0, 2, 1)   # (N,3,3)

    # -- residuals -----------------------------------------------------
    def residuals(self, C, r_p):
        """Whitened residual matrix (N, 3) and cost."""
        v = C @ (self.m - r_p.reshape(3)[:, None])       # (3,N)
        e = (self.meas - v).T                            # (N,3)
        r = np.einsum("nij,nj->ni", self.L, e)           # (N,3)
        return r, float(np.sum(r * r))

    # -- gradient ------------------------------------------------------
    def gradient(self, C, r_p):
        """Riemannian gradient in local coordinates ``(delta, r_p)``: (6,)."""
        v = C @ (self.m - r_p.reshape(3)[:, None])       # (3,N)
        e = self.meas - v
        We = np.einsum("nij,ni->nj", self.W, e.T)        # (N,3)
        # g_delta = 2 sum_k (W e) x v
        g_d = 2.0 * np.cross(We, v.T).sum(axis=0)
        g_r = 2.0 * (C.T @ We.sum(axis=0))
        return np.concatenate([g_d, g_r])

    # -- Gauss-Newton system -------------------------------------------
    def gn_system(self, C, r_p):
        """Whitened residual ``r`` (3N,) and Jacobian ``A`` (3N, 6).

        ``A`` stacks ``d(W^{1/2} e)/d(delta, dr_p)`` for all points, so that
        the cost is ``r^T r`` and the (Gauss-Newton) Hessian is ``A^T A``.
        """
        v = C @ (self.m - r_p.reshape(3)[:, None])       # (3,N)
        e = (self.meas - v).T                            # (N,3)
        N = self.m.shape[1]
        # d e / d delta = [v]_x   ;   d e / d r_p = C
        Vx = np.zeros((N, 3, 3))
        Vx[:, 0, 1] = -v[2]
        Vx[:, 0, 2] = v[1]
        Vx[:, 1, 0] = v[2]
        Vx[:, 1, 2] = -v[0]
        Vx[:, 2, 0] = -v[1]
        Vx[:, 2, 1] = v[0]
        A_d = np.einsum("nij,njk->nik", self.L, Vx)      # (N,3,3)
        A_r = np.einsum("nij,jk->nik", self.L, C)        # (N,3,3)
        A = np.concatenate([A_d, A_r], axis=2).reshape(3 * N, 6)
        r = np.einsum("nij,nj->ni", self.L, e).reshape(3 * N)
        return A, r

    def gn_step(self, C, r_p):
        """Gauss-Newton step ``(delta, dr)`` for the local coordinates."""
        A, r = self.gn_system(C, r_p)
        step, *_ = np.linalg.lstsq(A, -r, rcond=None)
        return step[:3], step[3:]

    def gn_hessian(self, C, r_p):
        """Gauss-Newton Hessian ``J^T J`` (6 x 6) in the local coordinates."""
        A, _ = self.gn_system(C, r_p)
        return A.T @ A


def pose_cost(C, r_p, meas, W, m=None) -> float:  # noqa: F811  (public helper)
    """Matrix-weighted cost (convenience wrapper; needs ``m``)."""
    if m is None:
        raise ValueError("m must be provided")
    return _LocalProblem(m, meas, W).residuals(C, r_p)[1]


def local_stationarity(C, r_p, meas, W, m) -> np.ndarray:
    """Riemannian gradient ``g`` (6,) at ``(C, r_p)``."""
    return _LocalProblem(m, meas, W).gradient(C, r_p)


# ----------------------------------------------------------------------
# Gauss-Newton solver
# ----------------------------------------------------------------------
def solve_pose_local(
    m,
    meas,
    W,
    C0,
    r_p0,
    max_iter: int = 500,
    step_size: float = 0.2,
    abs_tol: float = 1e-10,
    rel_tol: float = 1e-6,
    lm_lambda0: float = 1e-6,
    stationarity_tol: float = 1e-3,
) -> dict:
    """Levenberg-Marquardt on SO(3) for Problem (P6).

    The paper uses Theseus' Gauss-Newton solver with step size ``0.2``,
    ``max_iterations=500`` and relative error tolerance ``1e-6``.  A plain
    Gauss-Newton iteration with a fixed 0.2 step can run away for the random
    initialisations of Fig. 3b (some far-away stationary points are very
    ill-conditioned), so this implementation uses LM damping (adaptive
    ``lambda``) while keeping the paper's step size and tolerances.  The best
    iterate encountered is returned (cf. theseus ``track_best_solution``).

    Convergence is judged on the relative decrease of the cost *and* on the
    Riemannian gradient norm; a returned solution with a large gradient is
    flagged ``stationary=False``, and the bilevel loop treats that as a
    failed run rather than as convergence to a local minimum.
    """
    prob = _LocalProblem(m, meas, W)
    C = np.array(C0, dtype=float)
    r_p = np.array(r_p0, dtype=float).reshape(3)
    cost = prob.residuals(C, r_p)[1]
    best = (cost, C.copy(), r_p.copy())
    lam = lm_lambda0
    n_iter = 0
    converged = False
    n_rejected = 0
    delta_norm = np.inf
    for it in range(max_iter):
        n_iter = it + 1
        A, r = prob.gn_system(C, r_p)
        g = A.T @ r                       # 0.5 * gradient of the cost
        H = A.T @ A
        diagH = np.maximum(np.diag(H), 1e-12 * max(np.max(np.diag(H)), 1e-30))
        accepted = False
        for _ in range(40):
            try:
                delta = np.linalg.solve(H + lam * np.diag(diagH), -g)
            except np.linalg.LinAlgError:
                lam *= 10.0
                continue
            delta *= step_size
            C_new = exp_so3(delta[:3]) @ C
            r_p_new = r_p + delta[3:]
            cost_new = prob.residuals(C_new, r_p_new)[1]
            if cost_new < cost:
                accepted = True
                break
            lam *= 10.0
            n_rejected += 1
        if not accepted:
            converged = True
            break
        rel_decrease = (cost - cost_new) / max(cost, 1e-300)
        C, r_p, cost = C_new, r_p_new, cost_new
        lam = max(lam / 3.0, 1e-12)
        delta_norm = float(np.linalg.norm(delta))
        if cost < best[0]:
            best = (cost, C.copy(), r_p.copy())
        if cost < abs_tol or rel_decrease < rel_tol:
            converged = True
            break
    best_cost, best_C, best_r = best
    g = prob.gradient(best_C, best_r)
    g_norm = float(np.linalg.norm(g))
    # stationarity measured by the *natural* gradient step ||H^{-1} g||, i.e.
    # the distance to the nearest stationary point in the local coordinates
    A, r = prob.gn_system(best_C, best_r)
    H = A.T @ A
    try:
        nat_step = float(np.linalg.norm(np.linalg.solve(H, A.T @ r)))
    except np.linalg.LinAlgError:
        nat_step = float("inf")
    return dict(
        C=best_C,
        r_p=best_r,
        cost=best_cost,
        n_iter=n_iter,
        converged=converged,
        n_rejected=n_rejected,
        g_norm=g_norm,
        natural_step=nat_step,
        stationary=bool(nat_step < stationarity_tol),
    )


# ----------------------------------------------------------------------
# implicit differentiation of the local solution
# ----------------------------------------------------------------------
def local_implicit_derivative(
    m,
    C,
    r_p,
    meas_fn,
    W_fn,
    b: float,
    fd_step: float = 1e-6,
    b_step: float = 1e-5,
) -> dict:
    """``(dC/db, dr_p/db)`` from the implicit function theorem at a local min.

    ``meas_fn(b)`` / ``W_fn(b)`` re-evaluate the Euclidean measurements and
    weights for a perturbed baseline; everything else is held fixed.
    """
    meas0, W0 = meas_fn(b), W_fn(b)
    prob0 = _LocalProblem(m, meas0, W0)
    g0 = prob0.gradient(C, r_p)

    # dg / d(delta, dr_p) by central differences in the local coordinates
    J = np.zeros((6, 6))
    eps = fd_step
    for i in range(3):
        Cp = exp_so3(eps * np.eye(3)[i]) @ C
        Cm = exp_so3(-eps * np.eye(3)[i]) @ C
        J[:, i] = (_LocalProblem(m, meas0, W0).gradient(Cp, r_p)
                   - _LocalProblem(m, meas0, W0).gradient(Cm, r_p)) / (2 * eps)
    for i in range(3):
        rp = r_p.copy()
        rm = r_p.copy()
        rp[i] += eps
        rm[i] -= eps
        J[:, 3 + i] = (prob0.gradient(C, rp) - prob0.gradient(C, rm)) / (2 * eps)

    # dg / db at fixed (C, r_p)
    gp = _LocalProblem(m, meas_fn(b + b_step), W_fn(b + b_step)).gradient(C, r_p)
    gm = _LocalProblem(m, meas_fn(b - b_step), W_fn(b - b_step)).gradient(C, r_p)
    gb = (gp - gm) / (2 * b_step)

    try:
        sol = np.linalg.solve(J, -gb)
        cond = float(np.linalg.cond(J))
        singular = False
    except np.linalg.LinAlgError:
        sol, *_ = np.linalg.lstsq(J, -gb, rcond=None)
        cond = float(np.linalg.cond(J))
        singular = True
    d_delta = sol[:3]
    dr_p = sol[3:]
    dC = skew(d_delta) @ C
    return dict(
        dC=dC,
        dr_p=dr_p,
        cond=cond,
        singular=singular,
        g_norm=float(np.linalg.norm(g0)),
        J=J,
    )


# ----------------------------------------------------------------------
# outer loss of Problem (P7)
# ----------------------------------------------------------------------
def outer_loss(C, r_p, C_gt, r_gt) -> float:
    """``||r_p - r_gt||^2 + ||C^T C_gt - I||_F^2`` (Eq. P7)."""
    u = r_p.reshape(3) - r_gt.reshape(3)
    return float(u @ u) + float(np.sum((C.T @ C_gt - np.eye(3)) ** 2))
