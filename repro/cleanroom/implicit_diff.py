"""Implicit differentiation of Problem (P6) via Theorem 2 of Holmes et al.

The globally optimal solution of the QCQP satisfies the KKT conditions

    2 H x = 0,        x^T A_i x = 0 (i = 1..m),      x^T A_0 x = 1,

with certificate matrix ``H = Q + sum_i lambda_i A_i + lambda_0 A_0 >= 0``.
Differentiating ``k(z, theta) = 0`` gives ``M dz + N dtheta = 0`` with

    M = 2 [[H, G_r^T], [G_r, 0]],      (Eq. 13/17)
    G_r = [A_1 x, ..., A_r x, A_0 x]^T  (rows of the *linearly independent*
          constraint gradients, Eq. 14/38),
    N = [[2 (I (x) x^T), ...], [0, I (x) vec(x x^T)], [0, 0]]  (Eq. 35).

Because the constraints do not depend on theta, only the cost matrix
contributes, ``(I (x) x^T) vec(dQ) = dQ x``, and the solution Jacobian is

    J = -P M^{-1} [ 2 dQ x ; 0 ],        P = [I_n; 0],          (Eq. 16)

where the factor 2 cancels with the factor 2 in ``M`` if the reduced system

    [[H, G_r^T], [G_r, 0]] [dx; dlambda] = -[dQ x; 0]

is used directly.  The corresponding reverse-mode rule is

    grad_theta = -x^T (dQ/dtheta) y_1,   y = M_red^{-1} [grad_x; 0],   (Eq. 20)

``y_1`` being the first block of ``y`` (the gradient with respect to the KKT
conditions).  This module also implements a finite-difference reference
implementation of the same Jacobian, used to validate the analytic one.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "reduced_kkt_matrix",
    "qcqp_jacobian",
    "qcqp_backward",
    "pose_from_x",
    "pose_jacobian_from_x",
    "loss_pose_and_grad",
    "loss_from_x_and_grad",
    "fd_pose_jacobian",
    "relative_jacobian_difference",
]


# ----------------------------------------------------------------------
# structure of the solution vector
# ----------------------------------------------------------------------
def pose_from_x(x: np.ndarray):
    """``x = [h, vec_col(C), t] -> (C, r_p)`` with ``r_p = C^T t``."""
    c = np.asarray(x)[1:10]
    t = np.asarray(x)[10:13]
    C = c.reshape(3, 3).T
    r_p = C.T @ t
    return C, r_p


def pose_jacobian_from_x(x: np.ndarray, dx: np.ndarray):
    """Convert ``dx/dtheta`` into ``(dC/dtheta, dr_p/dtheta)``.

    ``C`` depends linearly on ``x``; ``r_p = C^T t`` so
    ``dr_p = dC^T t + C^T dt``.
    """
    c = x[1:10]
    t = x[10:13]
    C = c.reshape(3, 3).T
    dc = np.asarray(dx)[1:10]
    dt = np.asarray(dx)[10:13]
    dC = dc.reshape(3, 3).T
    dr_p = dC.T @ t + C.T @ dt
    return dC, dr_p


# ----------------------------------------------------------------------
# Theorem 2: Jacobian and backward pass
# ----------------------------------------------------------------------
def reduced_kkt_matrix(H: np.ndarray, G_r: np.ndarray):
    """``M_red = [[H, G_r^T], [G_r, 0]]`` (symmetric, invertible at a
    solution satisfying the SOSC for the reduced constraint set)."""
    n = H.shape[0]
    r = G_r.shape[0]
    M = np.zeros((n + r, n + r))
    M[:n, :n] = H
    M[:n, n:] = G_r.T
    M[n:, :n] = G_r
    return M


def qcqp_jacobian(H, G_r, x, dQ) -> dict:
    """Solution Jacobian ``dx/dtheta`` of the QCQP (Eq. 16).

    Parameters
    ----------
    H : (13,13) certificate matrix.
    G_r : (r,13) independent constraint Jacobian rows.
    x : (13,) primal solution.
    dQ : (13,13) derivative of the (scaled) cost matrix w.r.t. theta.

    Returns
    -------
    dict with ``dx`` (13,), ``cond`` (condition number of ``M_red``) and the
    residual of the KKT system.
    """
    M = reduced_kkt_matrix(H, G_r)
    n = H.shape[0]
    rhs = np.zeros(n + G_r.shape[0])
    rhs[:n] = -(dQ @ x)
    sol = np.linalg.solve(M, rhs)
    dx = sol[:n]
    resid = np.linalg.norm(M @ sol - rhs)
    return dict(
        dx=dx,
        dlam=sol[n:],
        cond=float(np.linalg.cond(M)),
        residual=float(resid),
        M=M,
    )


def qcqp_backward(H, G_r, x, dQ, grad_x) -> dict:
    """Reverse-mode gradient ``dloss/dtheta`` (Eq. 20).

    ``grad_x`` is ``dloss/dx``.  Returns the scalar gradient as well as the
    intermediate adjoint ``y`` (gradient w.r.t. the KKT conditions).
    """
    M = reduced_kkt_matrix(H, G_r)
    n = H.shape[0]
    rhs = np.zeros(n + G_r.shape[0])
    rhs[:n] = grad_x
    y = np.linalg.solve(M, rhs)
    y1 = y[:n]
    grad_theta = -float(x @ (dQ @ y1))
    # equivalent direct computation via the Jacobian, for cross-checking
    jac = qcqp_jacobian(H, G_r, x, dQ)
    grad_theta_direct = float(grad_x @ jac["dx"])
    return dict(
        grad_theta=grad_theta,
        grad_theta_direct=grad_theta_direct,
        y=y,
        cond=jac["cond"],
        jac=jac,
    )


# ----------------------------------------------------------------------
# outer loss and its gradient w.r.t. x
# ----------------------------------------------------------------------
def loss_pose_and_grad(C, r_p, C_gt, r_gt):
    """Outer loss of Problem (P7) and its gradients w.r.t. ``(C, r_p)``.

        loss = ||r_p - r_gt||^2 + ||C^T C_gt - I||_F^2

    (for orthogonal ``C`` the second term equals ``||C - C_gt||_F^2``).

    The gradients are taken in the *pose parameterisation* ``(C, r_p)`` used
    by the local solver: ``dC`` and ``dr_p`` are independent variations.  The
    QCQP parameterisation ``(C, t = C r_p)`` needs the extra term
    ``2 t u^T`` in ``dL/dC``; see :func:`loss_from_x_and_grad`.
    """
    u = r_p.reshape(3) - r_gt.reshape(3)
    loss = float(u @ u) + float(np.sum((C.T @ C_gt - np.eye(3)) ** 2))
    grad_r = 2.0 * u
    grad_C = 2.0 * (C - C_gt)
    return loss, grad_C, grad_r


def loss_from_x_and_grad(x, C_gt, r_gt):
    """Outer loss as a function of ``x`` and its gradient w.r.t. ``x``.

    In the QCQP parameterisation the independent variables are ``(C, t)``
    with ``r_p = C^T t``, so ``d r_p = dC^T t + C^T dt`` and

        dL/dC = 2 (C - C_gt) + 2 t u^T,      dL/dt = 2 C u,

    with ``u = r_p - r_gt``.  The ``2 t u^T`` term is the difference with the
    pose parameterisation returned by :func:`loss_pose_and_grad`.
    """
    C, r_p = pose_from_x(x)
    loss, grad_C, grad_r = loss_pose_and_grad(C, r_p, C_gt, r_gt)
    u = r_p.reshape(3) - r_gt.reshape(3)
    t = C @ r_p.reshape(3)
    grad_C_qcqp = grad_C + 2.0 * np.outer(t, u)
    grad_x = np.zeros(13)
    # x[1:10] = vec_col(C),   x[10:13] = t = C r_p
    grad_x[1:10] = grad_C_qcqp.T.reshape(9)
    grad_x[10:13] = C @ grad_r
    return loss, grad_x


# ----------------------------------------------------------------------
# finite-difference reference
# ----------------------------------------------------------------------
def fd_pose_jacobian(solve_fn, b: float, h: float) -> dict:
    """Central finite differences of the SDP solution map ``b -> (C, r_p)``.

    ``solve_fn(b)`` must return ``(C, r_p)`` (or a dict containing them).
    Used to validate the implicit-differentiation Jacobian.
    """
    Cp, rp = solve_fn(b + h)
    Cm, rm = solve_fn(b - h)
    dC = (Cp - Cm) / (2.0 * h)
    dr = (rp.reshape(3) - rm.reshape(3)) / (2.0 * h)
    return dict(dC=dC, dr=dr)


def relative_jacobian_difference(dC_a, dr_a, dC_b, dr_b, norm="inf") -> float:
    """Eq. (29): ``||J_est - J_true|| / ||J_true||`` on the pose Jacobian."""
    Ja = np.concatenate([np.asarray(dC_a).reshape(-1), np.asarray(dr_a).reshape(-1)])
    Jb = np.concatenate([np.asarray(dC_b).reshape(-1), np.asarray(dr_b).reshape(-1)])
    diff = Ja - Jb
    if norm == "inf":
        num = np.max(np.abs(diff))
        den = np.max(np.abs(Jb))
    elif norm == "fro":
        num = np.linalg.norm(diff)
        den = np.linalg.norm(Jb)
    else:
        raise ValueError(norm)
    return float(num / max(den, 1e-300))
