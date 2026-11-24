"""Local solvers for the quaternion-constrained QPEP cost."""
import numpy as np
from scipy.optimize import minimize
from .poly import peval, pgrad


def local_solve(J, q0, maxiter=2000, ftol=1e-14):
    grad = pgrad(J)
    q0 = np.asarray(q0, dtype=float)
    q0 = q0 / np.linalg.norm(q0)

    def f(q):
        return peval(J, q)

    def g(q):
        return np.array([peval(gi, q) for gi in grad])

    cons = {'type': 'eq', 'fun': lambda q: q @ q - 1.0,
            'jac': lambda q: 2.0 * q}
    res = minimize(f, q0, jac=g, constraints=cons, method='SLSQP',
                   options={'ftol': ftol, 'maxiter': maxiter})
    q = res.x / np.linalg.norm(res.x)
    return q, float(peval(J, q)), res


def multistart(J, nstart=20, seed=0, keep=1):
    rng = np.random.default_rng(seed)
    sols = []
    for _ in range(nstart):
        q0 = rng.normal(size=4)
        q, f, _ = local_solve(J, q0)
        sols.append((f, q))
    sols.sort(key=lambda z: z[0])
    # deduplicate sign-equivalent quaternions
    uniq = []
    for f, q in sols:
        if all(min(np.linalg.norm(q - u), np.linalg.norm(q + u)) > 1e-4
               for _, u in uniq):
            uniq.append((f, q))
        if len(uniq) >= keep:
            break
    return uniq


def all_local_minima(J, nstart=30, seed=0, tol=1e-6):
    rng = np.random.default_rng(seed)
    sols = []
    for _ in range(nstart):
        q0 = rng.normal(size=4)
        q, f, _ = local_solve(J, q0)
        sols.append((f, q))
    sols.sort(key=lambda z: z[0])
    uniq = []
    for f, q in sols:
        if all(min(np.linalg.norm(q - u), np.linalg.norm(q + u)) > 1e-3
               for _, u in uniq):
            uniq.append((f, q))
    return uniq
