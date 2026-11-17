"""Family 2: W_n(x) = (x(1-x))^n , n = 0 mod 4.  Kernel 1/(1+x^2).
   W_n = Q_n (1+x^2) + v_n ,  v_n = (1+i)^n = (-4)^{n/4}  (real integer)
   L_n = int_0^1 W_n/(1+x^2) dx = A_n + v_n * pi/4 ,  A_n = int_0^1 Q_n  (rational)
   4 L_n = 4 A_n + v_n pi ;  D_n = den(4 A_n) ; alpha = 4 D_n A_n in Z, beta = D_n v_n in Z
   RIGOROUS:  L_n >= (1/2) int_0^1 W_n = (1/2) B(n+1,n+1)
              |alpha + beta pi| = 4 D_n L_n >= 2 D_n B(n+1,n+1)
"""
from fractions import Fraction as F
from decimal import Decimal, localcontext
from math import comb, lcm, log10, factorial
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from mp import pi_decimal, set_dps

def W_coeffs(n):
    co = [0] * (2 * n + 1)
    for k in range(n + 1):
        co[n + k] += (-1) ** k * comb(n, k)
    return co

def div1px2(co):
    w = [int(c) for c in co]; n = len(w) - 1
    q = [0] * (n + 1)
    for j in range(n, 1, -1):
        q[j - 2] = w[j]; w[j - 2] -= w[j]; w[j] = 0
    u = w[1] if n >= 1 else 0
    v = w[0]
    return q[:max(n - 1, 0)], u, v

def report(n):
    co = W_coeffs(n)
    Q, u, v = div1px2(co)
    assert u == 0, "log2 term must vanish (n=0 mod 4)"
    A = sum(F(Q[j], j + 1) for j in range(len(Q)))
    assert A.denominator and v != 0
    # 4A denominator
    D = (4 * A).denominator
    alpha = 4 * A * D
    beta = v * D
    assert alpha.denominator == 1 and beta.denominator == 1
    alpha = int(alpha); beta = int(beta)
    return Q, A, v, D, alpha, beta

if __name__ == "__main__":
    set_dps(120)
    pi = pi_decimal(120)
    print("n | deg | log10 D_n | log10|beta| | log10|alpha| | log10(4 D L)_exact | growth/n")
    prev = None
    for n in range(4, 61, 4):
        Q, A, v, D, alpha, beta = report(n)
        with localcontext() as ctx:
            ctx.prec = 120
            L = (Decimal(A.numerator) / Decimal(A.denominator)) + Decimal(v) * pi / 4
            form = abs(Decimal(alpha) + Decimal(beta) * pi)
            lL = float(L.log10()); lf = float(form.log10())
        # rigorous lower bound: 2 D B(n+1,n+1)
        B = F(factorial(n) ** 2, factorial(2 * n + 1))
        lb = 2 * D * B
        row = (n, 2 * n, log10(D), log10(abs(beta)), log10(abs(alpha)), lf,
               (lf - prev) / 4 if prev is not None else float('nan'),
               log10(float(lb)))
        print("%3d | %4d | %9.4f | %10.4f | %11.4f | %18.4f | %8.4f | log10 lb=%.4f" % row)
        prev = lf
    print()
    print("theory: lcm(1..2n) ~ e^{2n} -> log10 = %.4f n ; B(n+1,n+1) ~ 4^{-n} -> log10 = %.4f n"
          % (2 * log10(2.718281828459045), log10(4)))
    print("net growth exponent per unit n: %.4f" % (2 * log10(2.718281828459045) - log10(4)))
