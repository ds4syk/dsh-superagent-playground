"""Padé approximants to arctan at x=1  ->  integer linear forms in 1 and pi.

arctor x = sum_{k>=0} (-1)^k x^{2k+1}/(2k+1).

Take Q(x)=sum_{j<=n} q_j x^{2j}, P(x)=x*sum_{m<=n} p_m x^{2m} (q_0=1).
Pade condition: Q*arctan - P = O(x^{4n+3}).
  <=> for m=n+1..2n :  sum_{j=0..n} q_j (-1)^{m-j}/(2(m-j)+1) = 0.
Then p_m = sum_j q_j (-1)^{m-j}/(2(m-j)+1) for m<=n.
Error at x=1:  E_n = Q(1)*pi/4 - P(1).
Linear form:   4*E_n = Q(1)*pi - 4*P(1).
"""
from fractions import Fraction as F
from decimal import Decimal, localcontext
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from mp import pi_decimal, set_dps

def solve_pade_arctan(n):
    """Return (q list length n+1, p list length n+1) as Fractions."""
    # unknowns q_1..q_n ; equations for m=n+1..2n
    # sum_{j=1..n} q_j (-1)^{m-j}/(2(m-j)+1) = -(-1)^m/(2m+1)
    N = n
    A = [[F(0)] * (N + 1) for _ in range(N + 1)]  # augmented
    for r, m in enumerate(range(n + 1, 2 * n + 1)):
        for j in range(1, n + 1):
            A[r][j - 1] = F((-1) ** (m - j), 2 * (m - j) + 1)
        A[r][N] = -F((-1) ** m, 2 * m + 1)
    # gaussian elimination over Q
    for c in range(N):
        piv = None
        for r in range(c, N):
            if A[r][c] != 0:
                piv = r; break
        if piv is None:
            raise RuntimeError("singular pade system n=%d" % n)
        A[c], A[piv] = A[piv], A[c]
        pv = A[c][c]
        A[c] = [x / pv for x in A[c]]
        for r in range(N):
            if r != c and A[r][c] != 0:
                f = A[r][c]
                A[r] = [a - f * b for a, b in zip(A[r], A[c])]
    q = [F(1)] + [A[j][N] for j in range(N)]
    # p_m for m=0..n
    p = []
    for m in range(n + 1):
        s = F(0)
        for j in range(0, n + 1):
            k = m - j
            if k >= 0:
                s += q[j] * F((-1) ** k, 2 * k + 1)
        p.append(s)
    return q, p

def linform(n, dps=200):
    """Exact (P1, Q1) and high precision error E_n = Q(1) pi/4 - P(1)."""
    q, p = solve_pade_arctan(n)
    Q1 = sum(q)            # x=1 -> x^{2j}=1
    P1 = sum(p)            # x=1 -> x^{2m+1}=1
    set_dps(dps)
    with localcontext() as ctx:
        ctx.prec = dps
        pi = pi_decimal(dps)
        E = (Decimal(Q1.numerator) / Decimal(Q1.denominator)) * pi / 4 \
            - (Decimal(P1.numerator) / Decimal(P1.denominator))
    return q, p, Q1, P1, +E

def clear_denoms(*frs):
    from math import lcm
    D = 1
    for f in frs:
        D = lcm(D, f.denominator)
    return D

if __name__ == "__main__":
    print(" n |  |E_n|=|Q(1)pi/4-P(1)|   log10|E|   Q1 (log10)  den(Q1,P1)   |B| log10  |L|=|Q1 pi-4P1| log10")
    import math
    prev = None
    for n in range(1, 41):
        q, p, Q1, P1, E = linform(n, dps=250)
        D = clear_denoms(Q1, P1)
        B = Q1 * D          # pi coefficient of D*(Q1 pi - 4 P1)? careful below
        # L := Q1*pi - 4*P1 ;  A = -4*P1*D, Bc = Q1*D  ->  D*L = A + Bc*pi
        A_int = -4 * P1 * D
        B_int = Q1 * D
        assert A_int.denominator == 1 and B_int.denominator == 1
        A_int = int(A_int); B_int = int(B_int)
        # |D*L| = D*4|E|
        logE = float(E.copy_abs().log10()) if E != 0 else float('-inf')
        logL = math.log10(4) + logE + math.log10(D)
        logB = math.log10(abs(B_int)) if B_int != 0 else float('-inf')
        logA = math.log10(abs(A_int)) if A_int != 0 else float('-inf')
        print("%3d | %s  %10.4f  %10.4f  %10d  %10.4f  %10.4f" % (
            n, ("%.3e" % float(E)), logE, math.log10(abs(float(Q1))), D, logB, logL))
