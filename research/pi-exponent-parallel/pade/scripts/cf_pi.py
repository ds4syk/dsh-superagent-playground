"""Continued fraction of pi:  partial quotients, convergent quality
   ϱ_k = |F_k|*|beta_k| = q_k*|q_k pi - p_k|  (Dirichlet quality of the integer form)
   and the CONDITIONAL exponent of the two-family argument:
       mu(pi) <= 2 + limsup log a_{k+1}/log q_k .
Also: quality of the Lambert CF convergents of pi/4 (form q_k pi - 4 p_k).
"""
from decimal import Decimal, getcontext, localcontext
from fractions import Fraction
from math import log, log10
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from mp import pi_decimal

def cf_from_fraction(fr, count=400):
    """continued fraction partial quotients of a rational (exact)."""
    num, den = fr.numerator, fr.denominator
    a = []
    while den and len(a) < count:
        q, r = divmod(num, den)
        a.append(q)
        num, den = den, r
    return a

def convergents(a, upto=None):
    p_1, p_0 = 1, a[0]
    q_1, q_0 = 0, 1
    out = [(a[0], p_0, q_0)]
    for k in range(1, len(a) if upto is None else min(upto, len(a))):
        p = a[k] * p_0 + p_1
        q = a[k] * q_0 + q_1
        out.append((a[k], p, q))
        p_1, p_0 = p_0, p
        q_1, q_0 = q_0, q
    return out

if __name__ == "__main__":
    DPS = 1500
    setctx = getcontext(); setctx.prec = DPS + 50
    pi = pi_decimal(DPS)
    fra = Fraction(pi)          # exact binary/decimal fraction of the computed pi
    cf = cf_from_fraction(fra, 1200)
    print("first 40 partial quotients of pi:", cf[:40])
    conv = convergents(cf, 1200)
    print()
    print(" k | a_k | log10 q_k | q_k*|q_k pi-p_k| | cond. exp 2+log a_{k+1}/log q_k")
    prev = None
    for k, (ak, p, q) in enumerate(conv[:600]):
        if k in (0,1,2,3,4,5,10,20,50,100,200,300,400,500):
            with localcontext() as ctx:
                ctx.prec = DPS
                err = pi - Decimal(p) / Decimal(q)
                lq = log10(q)
                qual = float((Decimal(q) * abs(err)).copy_sign(1) * Decimal(1))
            nx = conv[k + 1][0] if k + 1 < len(conv) else 0
            ce = (2 + log(nx) / log(q)) if (nx and q > 1) else float('nan')
            print("%4d | %5d | %10.3f | %18.3e | %.6f" % (k, ak, lq, qual * q, ce))
    # statistics
    import statistics
    aq = [c[0] for c in conv[1:800]]
    print()
    print("partial quotients: max=%d  mean=%.3f  geometric mean=%.3f  (Khinchin const=2.685)"
          % (max(aq), statistics.mean(aq), statistics.geometric_mean([x for x in aq if x > 0])))
    print("Levy: log q_k / k -> %.4f (theory pi^2/(12 log 2)=%.4f)"
          % (log(conv[600][2]) / 600, 3.14159265358979 ** 2 / (12 * log(2))))
