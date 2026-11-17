"""Show that the Pade remainder at x=1 factorises as
      E_n = -c_n * (3+2*sqrt2)^{-n} * J_n ,    J_n = int_0^1 (t(1-t))^n/(1+t^2) dt ,
   with c_n ~ 0.13 sqrt(n)  (so |E_n| ~ 0.245 * 23.31^{-n}).
The numerical slope of log10|E_n/J_n| converges to -log10(3+2 sqrt2) = -0.765551.
This is the "Legendre gain" (1+sqrt2)^{2n} on top of the mass factor 4^{-n}.
"""
from fractions import Fraction as F
from decimal import Decimal, localcontext
from math import comb, log10, sqrt
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from mp import pi_decimal, set_dps
from pade_arctan import linform

def div_by_one_plus_x2(co):
    w = [int(c) for c in co]; n = len(w) - 1
    q = [0] * (n + 1)
    for j in range(n, 1, -1):
        q[j - 2] = w[j]; w[j - 2] -= w[j]; w[j] = 0
    u = w[1] if n >= 1 else 0
    v = w[0]
    return q[:max(n - 1, 0)], u, v

def J_exact(n):
    """J_n = int_0^1 (t(1-t))^n/(1+t^2) dt = A + (u/2) log 2 + v*pi/4."""
    co = [0] * (2 * n + 1)
    for k in range(n + 1):
        co[n + k] += (-1) ** k * comb(n, k)
    q, u, v = div_by_one_plus_x2(co)
    return sum(F(q[j], j + 1) for j in range(len(q))), u, v

if __name__ == "__main__":
    set_dps(300)
    pi = pi_decimal(300)
    L = log10(3 + 2 * sqrt(2))
    print("log10(3+2sqrt2) = %.6f ; log10(4*(3+2sqrt2)) = %.6f" % (L, log10(4) + L))
    print(" n | log10|E_n| | log10 J_n | log10|E/J| | slope | c_n=|E/J|(3+2sqrt2)^n")
    prev = None
    for n in list(range(1, 31)) + [35, 40, 45, 50]:
        q, p, Q1, P1, E = linform(n, dps=350)
        A, u, v = J_exact(n)
        with localcontext() as ctx:
            ctx.prec = 350
            J = (Decimal(A.numerator) / Decimal(A.denominator)) \
                + Decimal(u) / 2 * Decimal(2).ln() + Decimal(v) * pi / 4
            lr = float(abs(E / J).log10())
        c = 10 ** (lr + n * L)
        if prev is None:
            print("%3d | %10.4f | %9.4f | %10.4f |       | %.4f" % (n, float(abs(E).log10()), float(J.log10()), lr, c))
        else:
            print("%3d | %10.4f | %9.4f | %10.4f | %6.4f | %.4f" % (n, float(abs(E).log10()), float(J.log10()), lr, (lr - prev), c))
        prev = lr
