"""Compare (a) reduced denominators of Pade values P(1)/Q(1)
   (b) continued-fraction convergents of arctan(1)  (integers by construction)."""
from fractions import Fraction as F
from decimal import Decimal, localcontext
from math import gcd, log10
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from mp import pi_decimal, set_dps
from pade_arctan import solve_pade_arctan, linform

def cf_convergents(K):
    """Lambert CF: pi/4 = 1/(1 + 1^2/(3 + 2^2/(5 + 3^2/(7+...))))
       a_k = 2k-1 (k>=1), b_k = k^2 ; a_0=0.
       p_k = a_k p_{k-1} + b_k p_{k-2} ; q_k likewise."""
    pm1, p0 = 1, 0
    qm1, q0 = 0, 1
    out = []
    for k in range(1, K + 1):
        ak = 2 * k - 1
        bk = k * k
        p = ak * p0 + bk * pm1
        q = ak * q0 + bk * qm1
        pm1, p0 = p0, p
        qm1, q0 = q0, q
        out.append((k, p, q))
    return out

if __name__ == "__main__":
    set_dps(400)
    pi = pi_decimal(400)
    pi4 = pi / 4
    print("=== CF convergents of arctan(1):  p_k/q_k ===")
    print("  k |  log10 q_k | log10|pi/4-p/q| | log10(q*err) | err rate")
    prev = None
    for k, p, q in cf_convergents(200):
        if k % 10 == 0 or k <= 5:
            with localcontext() as ctx:
                ctx.prec = 400
                err = pi4 - (Decimal(p) / Decimal(q))
                lq = log10(abs(q)) if q else 0
                le = float(err.copy_abs().log10())
            print("%3d | %10.4f | %14.4f | %13.4f | %s" % (
                k, lq, le, lq + le, ("%.4f" % (le / k)) if k else ""))
    print()
    print("=== Padé: reduced denominator of P(1)/Q(1) vs raw clearing ===")
    print(" n | log10 num | log10 den | log10|E| | log10 Q1 | log10|q*pi/4-p|")
    for n in range(1, 41):
        if n % 5 and n > 4:
            continue
        q, p, Q1, P1, E = linform(n, dps=300)
        fr = P1 / Q1
        num, den = fr.numerator, fr.denominator
        # error of rational approx pi/4 ~ P1/Q1 is |E/Q1|
        with localcontext() as ctx:
            ctx.prec = 300
            lE = float(E.copy_abs().log10())
        lQ = log10(abs(float(Q1)))
        # |den * pi/4 - num| = den * |E/Q1|
        print("%3d | %9.4f | %9.4f | %8.4f | %8.4f | %13.4f" % (
            n, log10(abs(num)), log10(den), lE, lQ,
            log10(den) + lE - lQ))
