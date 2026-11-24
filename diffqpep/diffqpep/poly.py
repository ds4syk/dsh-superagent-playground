"""Sparse multivariate polynomials in the quaternion q=(q0,q1,q2,q3).

The QPEP PnP algebraic cost is quartic in q after eliminating the translation.
Everything here is small and dense enough for exact coefficient bookkeeping.
"""
import itertools
import numpy as np


def padd(a, b):
    out = dict(a)
    for k, v in b.items():
        out[k] = out.get(k, 0.0) + v
    return {k: v for k, v in out.items() if abs(v) > 1e-15}


def pscale(a, s):
    return {k: v * s for k, v in a.items() if abs(v * s) > 1e-15}


def pmul(a, b):
    out = {}
    for ka, va in a.items():
        for kb, vb in b.items():
            k = tuple(x + y for x, y in zip(ka, kb))
            out[k] = out.get(k, 0.0) + va * vb
    return {k: v for k, v in out.items() if abs(v) > 1e-15}


def pconst(c):
    return {(0, 0, 0, 0): float(c)}


def pvar(i):
    k = [0, 0, 0, 0]
    k[i] = 1
    return {tuple(k): 1.0}


def peval(p, q):
    s = 0.0
    for k, v in p.items():
        t = v
        for i in range(4):
            t *= q[i] ** k[i]
        s += t
    return s


def pderiv(p, i):
    out = {}
    for k, v in p.items():
        if k[i] == 0:
            continue
        kk = list(k)
        kk[i] -= 1
        out[tuple(kk)] = out.get(tuple(kk), 0.0) + v * k[i]
    return out


def pgrad(p):
    return [pderiv(p, i) for i in range(4)]


def phess(p):
    g = pgrad(p)
    return [[pderiv(g[i], j) for j in range(4)] for i in range(4)]


def q2R(q):
    q0, q1, q2, q3 = q
    return np.array([
        [q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3,
         2 * (q0 * q3 + q1 * q2), 2 * (q1 * q3 - q0 * q2)],
        [2 * (q1 * q2 - q0 * q3),
         q0 * q0 - q1 * q1 + q2 * q2 - q3 * q3, 2 * (q0 * q1 + q2 * q3)],
        [2 * (q0 * q2 + q1 * q3), 2 * (q2 * q3 - q0 * q1),
         q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3]], dtype=float)


def quat_to_R_poly():
    q = [pvar(i) for i in range(4)]
    q0, q1, q2, q3 = q

    def sq(x):
        return pmul(x, x)

    def mul2(x, y):
        return pmul(x, y)

    R = [[None] * 3 for _ in range(3)]
    R[0][0] = padd(padd(sq(q0), sq(q1)), pscale(padd(sq(q2), sq(q3)), -1))
    R[0][1] = pscale(padd(mul2(q0, q3), mul2(q1, q2)), 2)
    R[0][2] = pscale(padd(mul2(q1, q3), pscale(mul2(q0, q2), -1)), 2)
    R[1][0] = pscale(padd(mul2(q1, q2), pscale(mul2(q0, q3), -1)), 2)
    R[1][1] = padd(padd(sq(q0), pscale(sq(q1), -1)),
                   padd(sq(q2), pscale(sq(q3), -1)))
    R[1][2] = pscale(padd(mul2(q0, q1), mul2(q2, q3)), 2)
    R[2][0] = pscale(padd(mul2(q0, q2), mul2(q1, q3)), 2)
    R[2][1] = pscale(padd(mul2(q2, q3), pscale(mul2(q0, q1), -1)), 2)
    R[2][2] = padd(padd(sq(q0), pscale(sq(q1), -1)),
                   padd(pscale(sq(q2), -1), sq(q3)))
    return R


def qpep_residuals(p, uv):
    """Algebraic PnP residuals after eliminating t.

    p: (n,3) world/body points, uv: (n,2) normalized image coordinates.
    Returns the 2n residual polynomials r(q), each quadratic in q.
    """
    n = len(p)
    R = quat_to_R_poly()
    G = np.zeros((3, 3))
    h = [pconst(0) for _ in range(3)]
    b1s, b2s = [], []
    for i in range(n):
        u, v = uv[i]
        p0, p1, p2 = p[i]
        zq = padd(padd(pscale(R[2][0], p0), pscale(R[2][1], p1)),
                  pscale(R[2][2], p2))
        xq = padd(padd(pscale(R[0][0], p0), pscale(R[0][1], p1)),
                  pscale(R[0][2], p2))
        yq = padd(padd(pscale(R[1][0], p0), pscale(R[1][1], p1)),
                  pscale(R[1][2], p2))
        b1 = padd(pscale(zq, u), pscale(xq, -1))
        b2 = padd(pscale(zq, v), pscale(yq, -1))
        A1 = np.array([-1.0, 0.0, u])
        A2 = np.array([0.0, -1.0, v])
        G += np.outer(A1, A1) + np.outer(A2, A2)
        h[0] = padd(h[0], pscale(b1, -1.0))
        h[1] = padd(h[1], pscale(b2, -1.0))
        h[2] = padd(h[2], padd(pscale(b1, u), pscale(b2, v)))
        b1s.append(b1)
        b2s.append(b2)
    Ginv = np.linalg.inv(G)
    hg = []
    for a in range(3):
        acc = pconst(0)
        for b in range(3):
            if abs(Ginv[a, b]) > 1e-15:
                acc = padd(acc, pscale(h[b], Ginv[a, b]))
        hg.append(acc)
    rs = []
    for i in range(n):
        u, v = uv[i]
        r1 = padd(b1s[i], padd(pscale(hg[0], 1.0), pscale(hg[2], -u)))
        r2 = padd(b2s[i], padd(pscale(hg[1], 1.0), pscale(hg[2], -v)))
        rs.extend([r1, r2])
    return rs


def gram_W(rs, monos):
    """PSD Gram matrix W with J(q)=m(q)^T W m(q), m = degree<=2 monomials."""
    idx = {k: i for i, k in enumerate(monos)}
    W = np.zeros((len(monos), len(monos)))
    for r in rs:
        w = np.zeros(len(monos))
        for k, v in r.items():
            w[idx[k]] += v
        W += np.outer(w, w)
    return W


def J_from_res(rs):
    J = pconst(0)
    for r in rs:
        J = padd(J, pmul(r, r))
    return J


def direct_cost(p, uv, q):
    """Direct algebraic cost and optimal translation for a given q."""
    R = q2R(q)
    n = len(p)
    A, b = [], []
    for i in range(n):
        u, v = uv[i]
        X, Y, Z = R[0] @ p[i], R[1] @ p[i], R[2] @ p[i]
        A.append(np.array([[-1.0, 0.0, u], [0.0, -1.0, v]]))
        b.append(np.array([u * Z - X, v * Z - Y]))
    A = np.vstack(A)
    b = np.concatenate(b)
    t, *_ = np.linalg.lstsq(A, -b, rcond=None)
    r = A @ t + b
    return float(r @ r), t


def monomials_deg_le(d, n=4):
    out = []
    for deg in range(d + 1):
        for c in itertools.combinations_with_replacement(range(n), deg):
            k = [0] * n
            for i in c:
                k[i] += 1
            out.append(tuple(k))
    return out
