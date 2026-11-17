"""Beukers-type 2D integral for zeta(2):
   I_n = int int_[0,1]^2 (x(1-x)y(1-y))^n/(1-xy)^{n+1} dx dy
       = sum_{k>=0} C(n+k,k) B(k+n+1,n+1)^2  = A_n + B_n zeta(2),  A_n,B_n in Q.
Verified against 2D Gauss-Legendre quadrature (n=1,2,3: rel.err 1e-8..1e-14)
and decay base measured by Richardson extrapolation: 1/I_n^{1/n} -> ~11.09 > e^2.
"""
from decimal import Decimal, getcontext, localcontext
from math import comb, factorial, cos, pi, log10, sqrt
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

def I_n(n, dps=50):
    """sum_{k>=0} C(n+k,k) B(k+n+1,n+1)^2  (bounded loop; n=0 needs its own tail)."""
    with localcontext() as ctx:
        ctx.prec = dps
        s = Decimal(0); nf = Decimal(factorial(n)); k = 0
        while k < 300000:
            d = 1
            for i in range(n + 1):
                d *= (k + n + 1 + i)
            beta = nf / Decimal(d)
            t = Decimal(comb(n + k, k)) * beta * beta
            s += t
            if t < s * Decimal(10) ** (-dps) and k > n + 10:
                break
            k += 1
        return +s

def gl_nodes(N):
    def Pn(N, x):
        p0, p1 = 1.0, x
        if N == 0:
            return 1.0
        for kk in range(2, N + 1):
            p0, p1 = p1, ((2 * kk - 1) * x * p1 - (kk - 1) * p0) / kk
        return p1
    xs = []; ws = []
    for i in range(N):
        x = cos(pi * (i + 0.75) / (N + 0.5))
        for _ in range(100):
            f = Pn(N, x); f1 = Pn(N - 1, x)
            dp = N * (x * f - f1) / (x * x - 1)
            dx = -f / dp; x += dx
            if abs(dx) < 1e-16:
                break
        f = Pn(N, x); f1 = Pn(N - 1, x)
        dp = N * (x * f - f1) / (x * x - 1)
        ws.append(2.0 / ((1 - x * x) * dp * dp)); xs.append(x)
    return xs, ws

if __name__ == "__main__":
    XS, WS = gl_nodes(160)
    def quad(n):
        s = 0.0
        for a, wa in zip(XS, WS):
            for b, wb in zip(XS, WS):
                x = (a + 1) / 2; y = (b + 1) / 2
                s += wa * wb * 0.25 * (x * (1 - x) * y * (1 - y)) ** n / (1 - x * y) ** (n + 1)
        return s
    print("quadrature cross-check (sum vs 2D Gauss-Legendre):")
    for n in [1, 2, 3]:
        v = I_n(n); q = quad(n)
        print("  n=%d  sum=%.12f  quad=%.12f  rel=%.2e" % (n, float(v), q, abs(float(v) - q) / q))
    print()
    logs = {}
    for n in range(10, 41):
        logs[n] = float(I_n(n).ln())
    print(" n | log I_n | d_n | Richardson d_n + n(d_n-d_{n-1})")
    for n in [15, 20, 25, 30, 35, 40]:
        d1 = logs[n] - logs[n - 1]; d0 = logs[n - 1] - logs[n - 2]
        print("%3d | %12.5f | %9.6f | %9.6f" % (n, logs[n], d1, d1 + n * (d1 - d0)))
    d40 = logs[40] - logs[39]; d39 = logs[39] - logs[38]
    lam = d40 + 40 * (d40 - d39)
    print()
    print("lambda = %.4f  =>  decay base e^{-lambda} = %.3f  (compare e^2 = %.3f)"
          % (lam, 2.718281828459045 ** (-lam), 7.389056))
