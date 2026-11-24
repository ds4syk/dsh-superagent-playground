"""Implicit differentiation of the quaternion KKT system."""
import numpy as np
from .poly import peval, pgrad, phess, pderiv, q2R


def vee(S):
    return np.array([S[2, 1], S[0, 2], S[1, 0]])


def kkt_matrix(J, q, lam):
    g = np.array([peval(gi, q) for gi in pgrad(J)])
    Hp = phess(J)
    H = np.array([[peval(Hp[i][j], q) for j in range(4)] for i in range(4)])
    K = np.zeros((5, 5))
    K[:4, :4] = H - lam * np.eye(4)
    K[:4, 4] = -q
    K[4, :4] = 2.0 * q
    return K, g, H


def kkt_derivative(J, q, dJ, lam=None):
    """Return dq/dtheta and dlambda/dtheta for F=[grad J - lam q; q'q-1]."""
    g = np.array([peval(gi, q) for gi in pgrad(J)])
    if lam is None:
        lam = float(q @ g)
    K, g, H = kkt_matrix(J, q, lam)
    dgrad = np.array([peval(pderiv(dJ, i), q) for i in range(4)])
    rhs = -np.concatenate([dgrad, [0.0]])
    sol = np.linalg.solve(K, rhs)
    return sol[:4], sol[4], K, lam, g


def rotation_derivative(q, dq, h=1e-6):
    """dR/dtheta for the unit-quaternion map, gauge invariant."""
    dR = np.zeros((3, 3))
    for i in range(4):
        e = np.zeros(4)
        e[i] = h
        dR += (q2R(q + e) - q2R(q - e)) / (2 * h) * dq[i]
    return dR


def angular_derivative(q, dq):
    R = q2R(q)
    dR = rotation_derivative(q, dq)
    return vee(dR @ R.T)


def finite_difference_global(solve_global, theta0, h=1e-5):
    """Return dR/dtheta and dq/dtheta from two global solves."""
    qp, Rp, tp = solve_global(theta0 + h)
    qm, Rm, tm = solve_global(theta0 - h)
    if qp @ qm < 0:
        qm, Rm, tm = -qm, Rm, tm
    dR = (Rp - Rm) / (2 * h)
    dq = (qp - qm) / (2 * h)
    omega = vee(dR @ Rp.T)
    return dR, dq, omega, (Rp, tp), (Rm, tm)
