"""Final verified tables for the report."""
from fractions import Fraction as F
from decimal import Decimal, localcontext
from math import lcm, log10, gcd
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from mp import pi_decimal, set_dps
from pade_arctan import linform, solve_pade_arctan
from family_pos import report as fam2

set_dps(200)
pi = pi_decimal(200)

print("=== TABLE 1: arctan Pade family, integer linear forms  alpha + beta*pi = 4 D E_n ===")
print(" n | D=den(P1,Q1) | beta=Q1*D | alpha=-4P1*D | log10|form| | exact check |4D E - (alpha+beta pi)|")
for n in range(1, 9):
    q, p, Q1, P1, E = linform(n, dps=200)
    D = lcm(Q1.denominator, P1.denominator)
    alpha = int(-4 * P1 * D); beta = int(Q1 * D)
    with localcontext() as ctx:
        ctx.prec = 200
        val = Decimal(alpha) + Decimal(beta) * pi
        chk = abs(val - 4 * Decimal(D) * E)
        lf = float(abs(val).log10())
    print("%2d | %12d | %9d | %11d | %10.4f | %.1e" % (n, D, beta, alpha, lf, float(chk)))

print()
print("=== TABLE 2: positive family W=(x(1-x))^n, n=0 mod 4 : alpha + beta*pi = 4 D L_n ===")
print(" n | N=deg | v=(1+i)^n | D | log10|beta| | log10|alpha| | log10|form| | rigorous lb 2 D B(n+1,n+1)")
for n in range(4, 41, 4):
    Q, A, v, D, alpha, beta = fam2(n)
    from math import factorial
    B = F(factorial(n) ** 2, factorial(2 * n + 1))
    lb = 2 * D * B
    with localcontext() as ctx:
        ctx.prec = 120
        val = (Decimal(alpha) + Decimal(beta) * pi)
        lf = float(abs(val).log10())
    print("%2d | %5d | %12d | %10d | %10.4f | %11.4f | %10.4f | %10.4f" % (
        n, 2 * n, v, D, log10(abs(beta)), log10(abs(alpha)), lf, log10(float(lb))))

print()
print("=== TABLE 3: intrinsic quality q_n = |E_n|*|Q1|  and total quality rho_n = D_n^2 q_n ===")
print(" (mu<=2 needs rho_n <= 1; here rho_n grows -> failure)")
print(" n | q_n = |E_n||Q1| | 16^-n      | D_n^2 | rho_n = D_n^2 q_n")
for n in range(4, 9):
    q, p, Q1, P1, E = linform(n, dps=200)
    D = lcm(Q1.denominator, P1.denominator)
    with localcontext() as ctx:
        ctx.prec = 200
        qn = float(abs(E)) * float(abs(Q1))
    print("%2d | %14.4e | %10.4e | %8.3e | %14.4e" % (n, qn, 16.0 ** (-n), float(D) ** 2, qn * float(D) ** 2))
