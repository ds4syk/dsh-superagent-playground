"""The rotation-matrix (O(3)) Shor lift of the QPEP algebraic cost.

This is the naive 'apply SDPRLayers' route: 13x13 SDP in x=[vec(R);t;1].
It is tight in value but can return a reflection (det R = -1); the
quaternion (SO(3)) problem is not the same as the O(3) problem.
"""
import numpy as np
import cvxpy as cp

N = 13
E13 = np.zeros((N, N)); E13[12, 12] = 1.0


def build_Q(p, uv):
    n = len(p)
    Q = np.zeros((N, N))
    e = np.eye(3)
    for i in range(n):
        pT = p[i][None, :]
        zrow = np.kron(e[2][None, :], pT)   # e3^T R p, row-major vec(R)
        xrow = np.kron(e[0][None, :], pT)
        yrow = np.kron(e[1][None, :], pT)
        A1 = np.zeros(N); A2 = np.zeros(N)
        A1[:9] = uv[i, 0] * zrow - xrow
        A1[9:12] = uv[i, 0] * e[2] - e[0]
        A2[:9] = uv[i, 1] * zrow - yrow
        A2[9:12] = uv[i, 1] * e[2] - e[1]
        Q += np.outer(A1, A1) + np.outer(A2, A2)
    return Q / n


def orth_constraints():
    out = []
    for a in range(3):
        for b in range(a, 3):
            A = np.zeros((N, N))
            for k in range(3):
                A[3 * k + a, 3 * k + b] += 0.5
                A[3 * k + b, 3 * k + a] += 0.5
            out.append((A, 1.0 if a == b else 0.0))
    return out


def depth_forms(p):
    e = np.eye(3)
    out = []
    for pi in p:
        a = np.zeros(N)
        a[:9] = np.kron(e[2][None, :], pi[None, :])
        a[9:12] = e[2]
        out.append(a)
    return out


def solve(Q, p=None, eps_depth=None, eps=1e-9, max_iters=100000):
    X = cp.Variable((N, N), symmetric=True)
    cons = [X >> 0]
    for A, b in orth_constraints():
        cons.append(cp.trace(A @ X) == b)
    cons.append(cp.trace(E13 @ X) == 1)
    if eps_depth is not None and p is not None:
        for a in depth_forms(p):
            A = 0.5 * (np.outer(a, E13[12]) + np.outer(E13[12], a))
            cons.append(cp.trace(A @ X) >= eps_depth)
    prob = cp.Problem(cp.Minimize(cp.trace(Q @ X)), cons)
    prob.solve(solver=cp.SCS, eps=eps, max_iters=max_iters)
    return X.value, prob.value, prob.status


def recover(X):
    """Return (x, rank_ratio). Uses last row when it is a valid section."""
    w = np.linalg.eigvalsh(X)
    rr = float(w[-1] / max(w[0], 1e-30))
    if abs(X[12, 12]) > 1e-9:
        x = X[12, :] / X[12, 12]
        if x @ x > 1e-8 and abs(x[12] - 1.0) < 1e-6:
            return x, rr
    # top eigenvector, sign so x[12] > 0 if possible
    val, vec = np.linalg.eigh(X)
    x = vec[:, -1] * np.sqrt(max(val[-1], 0.0))
    if x[12] < 0:
        x = -x
    return x, rr


def pose_from_x(x):
    R = x[:9].reshape(3, 3)
    t = x[9:12]
    if abs(x[12]) > 1e-9:
        R = R / x[12]; t = t / x[12]
    return R, t
