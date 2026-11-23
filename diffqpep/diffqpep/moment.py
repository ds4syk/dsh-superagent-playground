"""Lasserre moment relaxations of the quartic QPEP cost on the unit sphere."""
import time
import numpy as np
import cvxpy as cp
from .poly import monomials_deg_le

MON1 = monomials_deg_le(1)
MON2 = monomials_deg_le(2)
IDX1 = {k: i for i, k in enumerate(MON1)}
IDX2 = {k: i for i, k in enumerate(MON2)}


def _monos(k, n=4):
    return monomials_deg_le(k, n)


def solve_moment(W, order=2, eps=1e-9, max_iters=200000, solver=cp.SCS,
                 verbose=False):
    """Solve the order-`order` moment relaxation.

    W is the 15x15 Gram matrix in the degree<=2 monomial basis; the objective
    is <W, M[:15,:15]>.  Returns (M, bound, status, monomials, seconds).
    """
    mon = _monos(order)
    idx = {k: i for i, k in enumerate(mon)}
    m = len(mon)
    M = cp.Variable((m, m), symmetric=True)
    zero = (0, 0, 0, 0)
    cons = [M >> 0, M[idx[zero], idx[zero]] == 1]
    cons.append(sum(M[idx[tuple(1 if j == i else 0 for j in range(4))],
                      idx[tuple(1 if j == i else 0 for j in range(4))]]
                    for i in range(4)) == 1)
    loc = _monos(order - 1)
    for a in range(len(loc)):
        for b in range(a, len(loc)):
            ka, kb = loc[a], loc[b]
            expr = 0
            for i in range(4):
                ei = tuple(1 if j == i else 0 for j in range(4))
                k1 = tuple(ka[j] + ei[j] for j in range(4))
                k2 = tuple(kb[j] + ei[j] for j in range(4))
                expr = expr + M[idx[k1], idx[k2]]
            expr = expr - M[idx[ka], idx[kb]]
            cons.append(expr == 0)
    i2 = [idx[mo] for mo in MON2]
    obj = sum(W[a, b] * M[i2[a], i2[b]]
              for a in range(len(MON2)) for b in range(len(MON2)))
    prob = cp.Problem(cp.Minimize(obj), cons)
    t0 = time.time()
    prob.solve(solver=solver, eps=eps, max_iters=max_iters, verbose=verbose)
    return M.value, prob.value, prob.status, mon, time.time() - t0


def recover_q(M, mon):
    """Recover q from the first column of a rank-one moment matrix."""
    idx = {k: i for i, k in enumerate(mon)}
    q = np.array([M[idx[tuple(1 if j == i else 0 for j in range(4))], idx[(0, 0, 0, 0)]]
                  for i in range(4)])
    n = np.linalg.norm(q)
    if n < 1e-12:
        return None
    return q / n


def rank_ratio(M, tol=1e-7):
    w = np.linalg.eigvalsh(M)
    eff = int(np.sum(w > tol * max(w[-1], 1e-30)))
    return float(w[-1] / max(w[0], 1e-30)), eff, w
