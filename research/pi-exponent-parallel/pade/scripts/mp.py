"""Minimal arbitrary-precision toolkit (no external deps).

Decimal-based: pi (Machin), arctan series, high-precision utilities.
Rational helpers via fractions.Fraction.
"""
from decimal import Decimal, getcontext, localcontext
from fractions import Fraction

DPS = 120  # default decimal digits


def set_dps(d):
    global DPS
    DPS = d
    getcontext().prec = d + 20


def pi_decimal(dps=DPS):
    """pi via Machin-like formula  pi = 16*atan(1/5) - 4*atan(1/239)."""
    with localcontext() as ctx:
        ctx.prec = dps + 30
        def atan_inv(n):
            # atan(1/n), n integer
            n = Decimal(n)
            term = 1 / n
            s = term
            k = 1
            n2 = n * n
            sign = -1
            while True:
                term = term / n2
                t = term / (2 * k + 1)
                if t == 0:
                    break
                s += sign * t
                sign = -sign
                k += 1
            return s
        r = 16 * atan_inv(5) - 4 * atan_inv(239)
        return +r


def dec(x):
    if isinstance(x, Decimal):
        return x
    if isinstance(x, Fraction):
        return Decimal(x.numerator) / Decimal(x.denominator)
    return Decimal(x)


def frac_to_dec(fr, dps=None):
    if dps is None:
        dps = DPS
    with localcontext() as ctx:
        ctx.prec = dps + 20
        return +(Decimal(fr.numerator) / Decimal(fr.denominator))


def log10_abs(fr):
    """log10 |fr| for a Fraction, as float (safe for huge values)."""
    if fr == 0:
        return float("-inf")
    n, d = abs(fr.numerator), fr.denominator
    # use bit lengths then refine
    import math
    bn = n.bit_length()
    bd = d.bit_length()
    # scale to ~50 bits then take log
    if bn > 60:
        n_scaled = n >> (bn - 60)
    else:
        n_scaled = n << (60 - bn)
    if bd > 60:
        d_scaled = d >> (bd - 60)
    else:
        d_scaled = d << (60 - bd)
    v = (n_scaled / d_scaled)
    return (math.log10(v) + (bn - bd)) * 0.30102999566398120  # log10(2)


def ln_abs(fr):
    return log10_abs(fr) / 0.4342944819032518


if __name__ == "__main__":
    p = pi_decimal(60)
    print(p)
