"""Compare the Pade denominator Q(y), y=x^2, for arctan with the poles of the
n-point / (n+1)-point Gauss-Legendre quadrature of int_0^1 ds/(1+x^2 s^2).
Result (used in report section 4.3): the naive correspondence
   [n/n] Pade  <->  (n+1)-point Gauss-Legendre
is FALSE (the roots -1/s_i^2 do not match).  The quadrature values at x=1 are,
however, closer to pi/4 than the Pade values, so the Pade family is not the
Gauss family.  Hence the report only claims the *mechanism* (Bernstein-ellipse
gain (1+sqrt2)^{2n}), not a closed Legendre formula.
"""
from fractions import Fraction as F
from math import comb, cos, pi
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from pade_arctan import solve_pade_arctan

def Pn(N, x):
    p0, p1 = 1.0, x
    if N == 0:
        return 1.0
    for k in range(2, N + 1):
        p0, p1 = p1, ((2 * k - 1) * x * p1 - (k - 1) * p0) / k
    return p1

def gauss_nodes(N):
    xs, ws = [], []
    for i in range(N):
        x = cos(pi * (i + 0.75) / (N + 0.5))
        for _ in range(200):
            f, f1 = Pn(N, x), Pn(N - 1, x)
            dp = N * (x * f - f1) / (x * x - 1)
            dx = -f / dp
            x += dx
            if abs(dx) < 1e-16:
                break
        f, f1 = Pn(N, x), Pn(N - 1, x)
        dp = N * (x * f - f1) / (x * x - 1)
        ws.append(2.0 / ((1 - x * x) * dp * dp))
        xs.append(x)
    return xs, ws

def roots(coeffs):
    c = [float(v) for v in coeffs]
    while c and abs(c[-1]) < 1e-300:
        c.pop()
    n = len(c) - 1
    c = [v / c[-1] for v in c]
    z = [complex(0.4, 0.9) ** k for k in range(n)]
    for _ in range(3000):
        new = []
        for i, zi in enumerate(z):
            p = sum(c[j] * zi ** j for j in range(n + 1))
            d = 1.0
            for j, zj in enumerate(z):
                if j != i:
                    d *= (zi - zj)
            new.append(zi - p / d if d != 0 else zi)
        if max(abs(new[i] - z[i]) for i in range(n)) < 1e-15:
            z = new
            break
        z = new
    return z

if __name__ == "__main__":
    for n in [2, 3, 4]:
        q, p = solve_pade_arctan(n)
        xs, ws = gauss_nodes(n + 1)
        s = [(1 + t) / 2 for t in xs]
        pred = sorted(-1.0 / (si * si) for si in s)
        got = sorted(roots(q), key=lambda z: z.real)
        print("n=%d" % n)
        print("  Pade Q roots (y=x^2)     :", ["%.6f" % z.real for z in got])
        print("  -1/s_i^2 Gauss nodes     :", ["%.6f" % v for v in pred])
        print("  quadrature sum at x=1    : %.10f" % sum((w / 2) / (1 + si * si) for w, si in zip(ws, s)))
        print("  Pade P(1)/Q(1)           : %.10f" % (float(sum(p)) / float(sum(q))))
        print("  pi/4                     : %.10f" % (pi / 4))
