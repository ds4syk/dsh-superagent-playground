"""Identify the Pade denominator/numerator polynomials for arctan with
Legendre / hypergeometric formulas; test a positive-kernel remainder formula."""
from fractions import Fraction as F
from math import comb, lcm, factorial
from decimal import Decimal, localcontext
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from mp import pi_decimal, set_dps
from pade_arctan import solve_pade_arctan

def show(n):
    q, p = solve_pade_arctan(n)
    D = 1
    for f in q + p:
        D = lcm(D, f.denominator)
    qi = [int(f * D) for f in q]
    pi_ = [int(f * D) for f in p]
    print("n=%d  q (den of Pade, x^{2j} coeff, common den %d):" % (n, D))
    print("   q_j =", qi)
    print("   p_m =", pi_)
    # candidate: q_j = (-1)^j C(n,j) C(n+j,j) / (2j+1) ?  etc.
    print("   j:      ", list(range(n + 1)))
    print("   C(n,j)  ", [comb(n, j) for j in range(n + 1)])
    print("   C(n+j,j)", [comb(n + j, j) for j in range(n + 1)])
    print("   (-1)^j*C(n,j)*C(n+j,j) =", [(-1) ** j * comb(n, j) * comb(n + j, j) for j in range(n + 1)])
    print("   ratio q_j / [(-1)^j C(n,j)C(n+j,j)] =",
          [F(qi[j], (-1) ** j * comb(n, j) * comb(n + j, j)) if comb(n, j) * comb(n + j, j) else None
           for j in range(n + 1)])
    print()

if __name__ == "__main__":
    for n in range(1, 6):
        show(n)
