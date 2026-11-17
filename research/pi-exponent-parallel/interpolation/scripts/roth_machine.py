#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
roth_machine.py -- exact interpolation / determinant machine for the
"rational interpolation + evaluation at pi" architecture.

All linear algebra over Z (saturated integer kernel via Smith normal form),
all polynomial evaluations of pi done with mpmath at high precision.

Conventions
-----------
P in Z[X_1,...,X_n],  deg_{X_i} P <= d.
Interpolation conditions: for each point x_i = p_i/q_i (i=1..n) and each
multi-index a with |a| < m:
        (d/ dX)^a P (x_i,...,x_i) = 0 .
After clearing the denominator q_i^d the equation has integer coefficients

        (e!/(e-a)!) * p_i^(e-a) * q_i^(d-|e|+|a|)      (e >= a, else 0)

for the coefficient c_e of X^e.  N = (d+1)^n unknowns,
M = n * C(m+n-1, n) equations.

Q(X) := P(X,...,X) in Z[X] is the diagonal restriction; it vanishes to
order m at every x_i, hence is divisible by prod_i (q_i X - p_i)^m.
"""
import math
from fractions import Fraction

import mpmath as mp
import sympy as sp
from sympy.matrices.normalforms import smith_normal_decomp

# ---------------------------------------------------------------- pi / CF ---
def pi_cf(num_terms, dps=140):
    """partial quotients and convergents of pi via mpmath at high precision."""
    mp.mp.dps = dps
    pi = mp.pi
    x = pi
    a = []
    for _ in range(num_terms):
        ai = int(mp.floor(x))
        a.append(ai)
        frac = x - ai
        if frac == 0:
            break
        x = 1 / frac
    # convergents
    p_prev, p_prev2 = 1, 0
    q_prev, q_prev2 = 0, 1
    conv = []
    for ai in a:
        p = ai * p_prev + p_prev2
        q = ai * q_prev + q_prev2
        conv.append((p, q, ai))
        p_prev2, p_prev = p_prev, p
        q_prev2, q_prev = q_prev, q
    return a, conv


def approx_exponent(p, q, dps=160):
    """local exponent kappa with |pi - p/q| = q^{-kappa} (numerically)."""
    mp.mp.dps = dps
    err = abs(mp.pi - mp.mpf(p) / mp.mpf(q))
    if err == 0 or q <= 1:
        return mp.inf
    return float(-mp.log10(err) / mp.log10(q))


# ------------------------------------------------------- interpolation ------
def exp_vectors(n, d):
    """all exponent vectors e in {0..d}^n, in a fixed order."""
    out = [()]
    for _ in range(n):
        out = [e + (k,) for e in out for k in range(d + 1)]
    return out


def multi_indices_below(n, m):
    """all a with |a| < m."""
    out = []
    for total in range(m):
        def rec(prefix, left, slots):
            if slots == 1:
                out.append(prefix + (left,))
                return
            for k in range(left + 1):
                rec(prefix + (k,), left - k, slots - 1)
        rec((), total, n)
    return out


def multinomial_factorials(e, a):
    """prod_i e_i!/(e_i-a_i)!  (0 if a_i > e_i for some i)."""
    out = 1
    for ei, ai in zip(e, a):
        if ai > ei:
            return 0
        for t in range(ei - ai + 1, ei + 1):
            out *= t
    return out


def interp_matrix(n, d, m, points):
    """integer matrix B (M x N); rows = interpolation conditions."""
    exps = exp_vectors(n, d)
    ais = multi_indices_below(n, m)
    rows = []
    for (p, q) in points:
        for a in ais:
            row = []
            for e in exps:
                c = multinomial_factorials(e, a)
                if c == 0:
                    row.append(0)
                    continue
                # (e!/(e-a)!) p^(|e-a|) q^(n*d-|e-a|)
                s_ea = sum(ei - ai for ei, ai in zip(e, a))
                c *= p ** s_ea
                c *= q ** (n * d - s_ea)
                row.append(c)
            rows.append(row)
    return sp.Matrix(rows), exps, ais


def saturated_kernel(B):
    """Z-basis of {x in Z^N : B x = 0}, saturated, as list of integer lists."""
    N = B.cols
    ns = B.nullspace()
    if not ns:
        return []
    K = sp.Matrix.hstack(*ns)              # N x R, rational entries
    # clear denominators columnwise
    cols = []
    for j in range(K.cols):
        den = sp.ilcm(*[sp.denom(K[i, j]) for i in range(K.rows)])
        cols.append([sp.Integer(K[i, j] * den) for i in range(K.rows)])
    K = sp.Matrix.hstack(*[sp.Matrix(c) for c in cols]) if cols else sp.zeros(N, 0)
    if K.cols == 0:
        return []
    S, U, V = smith_normal_decomp(K, domain=sp.ZZ)   # S = U*K*V
    Uinv = U.inv()
    basis = []
    for j in range(K.cols):
        basis.append([int(Uinv[i, j]) for i in range(N)])
    return basis


# ------------------------------------------------------------- polynomials ---
def diag_poly(vec, exps, n):
    """Q(X) = P(X,...,X); returns dict k -> coefficient (k = |e|)."""
    q = {}
    for c, e in zip(vec, exps):
        if c == 0:
            continue
        k = sum(e)
        q[k] = q.get(k, 0) + int(c)
    return {k: v for k, v in q.items() if v != 0}


def poly_height(q):
    return max((abs(v) for v in q.values()), default=0)


def poly_eval(q, x):
    if not q:
        return mp.mpf(0)
    K = max(q)
    s = mp.mpf(0)
    for k in range(K, -1, -1):
        s = s * x + q.get(k, 0)
    return s


def poly_to_str(q):
    if not q:
        return "0"
    K = max(q)
    terms = []
    for k in range(K, -1, -1):
        c = q.get(k, 0)
        if c == 0:
            continue
        if k == 0:
            terms.append(str(c))
        elif k == 1:
            terms.append(f"{c}*X")
        else:
            terms.append(f"{c}*X^{k}")
    return " + ".join(terms)


def vec_height(vec):
    return max(abs(int(c)) for c in vec)


def evaluate_log10_abs(q, dps=200):
    """log10 |Q(pi)| with high precision; returns (log10, value-string)."""
    mp.mp.dps = dps
    val = poly_eval(q, mp.pi)
    av = abs(val)
    if av == 0:
        return None, "0"
    return float(mp.log10(av)), mp.nstr(av, 25)


# ------------------------------------------------------------------- LLL -----
def lll_reduce(vectors, delta=Fraction(3, 4)):
    """LLL on a list of integer row vectors (exact rational arithmetic)."""
    B = [[int(x) for x in v] for v in vectors if any(v)]
    if not B:
        return []
    n = len(B)
    dim = len(B[0])

    def dot(u, v):
        return sum(a * b for a, b in zip(u, v))

    def gram_schmidt(B):
        Bs, mu = [], [[Fraction(0)] * n for _ in range(n)]
        for i in range(n):
            v = [Fraction(x) for x in B[i]]
            for j in range(i):
                mu[i][j] = Fraction(dot(B[i], Bs[j]), dot(Bs[j], Bs[j]))
                v = [v[k] - mu[i][j] * Bs[j][k] for k in range(dim)]
            Bs.append(v)
        return Bs, mu

    Bs, mu = gram_schmidt(B)
    k = 1
    while k < n:
        for j in range(k - 1, -1, -1):
            if abs(mu[k][j]) > Fraction(1, 2):
                r = int(round(mu[k][j]))
                B[k] = [B[k][i] - r * B[j][i] for i in range(dim)]
                Bs, mu = gram_schmidt(B)
        if dot(Bs[k], Bs[k]) >= (delta - mu[k][k - 1] ** 2) * dot(Bs[k - 1], Bs[k - 1]):
            k += 1
        else:
            B[k], B[k - 1] = B[k - 1], B[k]
            Bs, mu = gram_schmidt(B)
            k = max(k - 1, 1)
    return B


# -------------------------------------------------------------- helpers -----
def fmt(x, nd=4):
    if x is None:
        return "None"
    if isinstance(x, float):
        if x != x:
            return "nan"
        if abs(x) >= 1e5 or (x != 0 and abs(x) < 1e-3):
            return f"{x:.{nd}e}"
        return f"{x:.{nd}f}"
    return str(x)


def sieve_primes(n):
    """primes up to n."""
    if n < 2:
        return []
    s = bytearray([1]) * (n + 1)
    s[0:2] = b"\x00\x00"
    for i in range(2, int(n ** 0.5) + 1):
        if s[i]:
            s[i * i::i] = b"\x00" * (((n - i * i) // i) + 1)
    return [i for i in range(n + 1) if s[i]]
