#!/usr/bin/env python3
"""Audit of "candidate constructions": explicit families of rational
approximations to pi, and the exponents a proof would need.

For each family we build exact rational approximations p_N/q_N (fractions
module, exact arithmetic) and measure

    lambda_N := -log10|pi - p_N/q_N| / log10(q_N)      (error exponent)
    Q_N      := q_N^2 |pi - p_N/q_N|                   (Legendre quality)
    growth   := q_{N+1}/q_N                            (denominator growth)

Interpretation
  * Legendre's theorem: any p/q with Q < 1/2 is a convergent of pi.  Hence a
    family with lambda >= 2 (the regime needed by the standard gap principle to
    conclude mu <= 2) must *be* the continued fraction.
  * The table shows what the standard arithmetic constructions actually give:
    lambda between ~0.4 and ~1.2, Q >> 1, i.e. orders of magnitude away from
    the convergents (lambda ~ 2, Q < 1).  See report.md, section 4.

Families
  A  Machin        pi = 16 arctan(1/5) - 4 arctan(1/239), partial sums
  B  arcsin        pi = 6 arcsin(1/2), Taylor partial sums
  C  Wallis        pi/2 = prod 4k^2/(4k^2-1), partial products
  D  integral      Hata-type: I_n = int_0^1 (x(1-x))^n/(1+x^2) dx
                   = A_n + B_n pi + C_n log 2, approximant pi ~ -A_n/B_n
  E  Pade          [L/L] Pade approximant of arctan at x=1, times 4
  F  convergents   p_k/q_k from the certified continued fraction (reference)

Usage: python3 constructions.py --cf ../data/cf_d1e5_a.txt --out ../data/constructions.json
"""
import argparse
import json
from decimal import Decimal, getcontext
from fractions import Fraction
import math
from math import comb

getcontext().prec = 600
PI = Decimal(
    "3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679821480865132823066470938446095505822317253594081284811174502841027019385211055596446229489549303819644288109756659334461284756482337867831652712019091456485669234603486104543266482133936072602491412737245870066063155881748815209209628292540917153643678925903600113305305488204665213841469519415116094330572703657595919530921861173819326117931051185480744623799627495673518857527248912279381830119491298336733624406566430860213949463952247371907021798609437027705392171762931767523846748184676694051320005681271452635608277857713427577896091736371787214684409012249534301465495853710507922796892589235420199561121290219608640344181598136297747713099605187072113499999983729780499510597317328160963185950244594553469083026425223082533446850352619311881710100031378387528865875332083814206171776691473035982534904287554687311595628638823537875937519577818577805321712268066130019278766111959092164201989"
)


def dec_log10(x: Fraction):
    return (Decimal(x.numerator) / Decimal(x.denominator)).log10()


def measure(p: Fraction, name: str, N, extra=None):
    q = p.denominator
    err = abs(PI - Decimal(p.numerator) / Decimal(q))
    lq = Decimal(q).log10()
    lQ = float(lq * 2 + err.log10())          # log10(q^2 * err)
    row = dict(family=name, N=N, log10_q=float(lq),
               log10_err=float(err.log10()),
               lambda_exponent=float(-err.log10() / lq),
               log10_Q_q2err=lQ, is_convergent_legendre=bool(lQ < math.log10(0.5)),
               q_digits=len(str(q)))
    if extra:
        row.update(extra)
    return row


# ------------------------------------------------------------------ A: Machin
def arctan_series(x: Fraction, N: int) -> Fraction:
    s, xp, x2 = Fraction(0), x, x * x
    for k in range(N + 1):
        term = xp / (2 * k + 1)
        s = s + term if k % 2 == 0 else s - term
        xp *= x2
    return s


def machin(N: int) -> Fraction:
    return 16 * arctan_series(Fraction(1, 5), N) - 4 * arctan_series(Fraction(1, 239), N)


# ------------------------------------------------------------------ B: arcsin
def arcsin_half_series(N: int) -> Fraction:
    """6 * sum_{k<=N} C(2k,k) (1/2)^{2k+1}/(4^k (2k+1)) = partial sum of 6 arcsin(1/2)."""
    s = Fraction(0)
    for k in range(N + 1):
        s += Fraction(comb(2 * k, k) * 6, 4 ** k * 2 ** (2 * k + 1) * (2 * k + 1))
    return s


# ----------------------------------------------------------------- C: Wallis
def wallis(N: int) -> Fraction:
    """Partial product 2 * prod_{k=1..N} 4k^2/(4k^2-1)."""
    p = Fraction(2)
    for k in range(1, N + 1):
        p *= Fraction(4 * k * k, 4 * k * k - 1)
    return p


# ---------------------------------------------------------------- D: integral
def integral_family(n: int):
    """A_n, B_n, C_n with int_0^1 (x(1-x))^n/(1+x^2) dx = A_n + B_n pi + C_n log 2."""
    coeffs = [0] * (2 * n + 1)
    for k in range(n + 1):
        coeffs[n + k] += (-1) ** k * comb(n, k)
    Q = [0] * (2 * n + 1)
    rem = coeffs[:]
    for m in range(2 * n, 1, -1):
        qm = rem[m]
        Q[m - 2] = qm
        rem[m] = 0
        rem[m - 2] -= qm          # rem[m]*x^m = rem[m]*x^(m-2)*(x^2+1) - rem[m]*x^(m-2)
    r0, r1 = rem[0], rem[1]
    A = Fraction(0)
    for m, qm in enumerate(Q):
        if qm:
            A += Fraction(qm, m + 1)
    B = Fraction(r0, 4)
    C = Fraction(r1, 2)
    return A, B, C


# ------------------------------------------------------------------ E: Pade
def pade_arctan(L: int):
    """[L/L] Pade approximant of arctan at x=0, evaluated at x=1 (exact).

    f(x) = arctan x = sum c_k x^k,  c_k = (-1)^((k-1)/2)/k for odd k.
    Solve for b_1..b_L from the equations for k = L+1..2L:
        c_k + sum_{j=1..L} b_j c_{k-j} = 0,
    then a_k = c_k + sum_{j<=min(k,L)} b_j c_{k-j}, and f ~ P/Q.
    """
    M = 2 * L + 2
    c = [Fraction(0)] * (M + 2)
    for k in range(len(c)):
        if k % 2 == 1:
            c[k] = Fraction((-1) ** ((k - 1) // 2), k)
    # L x L system for b_1..b_L
    mat = [[Fraction(0)] * L for _ in range(L)]
    rhs = [Fraction(0)] * L
    for row, k in enumerate(range(L + 1, 2 * L + 1)):
        for j in range(1, L + 1):
            if k - j >= 0:
                mat[row][j - 1] = c[k - j]
        rhs[row] = -c[k]
    for col in range(L):
        piv = next((r for r in range(col, L) if mat[r][col] != 0), None)
        if piv is None:
            return None
        mat[col], mat[piv] = mat[piv], mat[col]
        rhs[col], rhs[piv] = rhs[piv], rhs[col]
        pv = mat[col][col]
        mat[col] = [v / pv for v in mat[col]]
        rhs[col] /= pv
        for r in range(L):
            if r != col and mat[r][col] != 0:
                f = mat[r][col]
                mat[r] = [a - f * b for a, b in zip(mat[r], mat[col])]
                rhs[r] -= f * rhs[col]
    b = [Fraction(1)] + rhs
    a = []
    for k in range(L + 1):
        ak = c[k]
        for j in range(1, min(k, L) + 1):
            ak += b[j] * c[k - j]
        a.append(ak)
    num = sum(a)
    den = sum(b)
    return num / den if den != 0 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cf", default="../data/cf_d1e5_a.txt")
    ap.add_argument("--out", default="../data/constructions.json")
    args = ap.parse_args()

    families = {}      # name -> list of (N, p)
    order = []

    def add(fn, name, Ns):
        fam = []
        for N in Ns:
            try:
                p = fn(N)
            except Exception as e:
                print(f"  {name} N={N} failed: {e}")
                continue
            if p is None or p <= 0 or p.denominator == 1:
                continue
            fam.append((N, p))
        if fam:
            families[name] = fam
            order.append(name)

    add(machin, "A_machin", [2, 4, 8, 16, 32, 64, 100])
    add(arcsin_half_series, "B_arcsin", [5, 10, 20, 40, 80, 160])
    add(wallis, "C_wallis", [5, 10, 20, 40, 80, 160])

    def integral_pi(n):
        A, B, C = integral_family(n)
        if C != 0 or B == 0:          # log2 term must vanish for a 2-term form
            return None
        return -A / B

    add(integral_pi, "D_integral", [4, 8, 12, 20, 40, 80, 120])

    def pade_pi(L):
        v = pade_arctan(L)
        return 4 * v if v else None

    add(pade_pi, "E_pade", [4, 6, 10, 16, 24, 32])

    # F: convergents of pi from the certified CF (reference family)
    try:
        A = [int(x) for x in open(args.cf).read().split()]
        fam = []
        pm1, qm1, p, q = 1, 0, A[0], 1
        for k in range(1, 30):
            p, pm1 = A[k] * p + pm1, p
            q, qm1 = A[k] * q + qm1, q
            if k in (2, 4, 6, 8, 10, 12, 16, 20, 25, 28):
                fam.append((k, Fraction(p, q)))
        families["F_convergents"] = fam
        order.append("F_convergents")
    except FileNotFoundError:
        print("no CF file, skipping F")

    rows = []
    for name in order:
        fam = families[name]
        prev = None
        for (N, p) in fam:
            r = measure(p, name, N)
            if prev is not None:
                dq = Decimal(p.denominator) / Decimal(prev.denominator)
                if dq > 1:
                    r["theta_gap_exponent"] = float(
                        dq.log10() / Decimal(prev.denominator).log10())
                    # gap principle: need lambda >= theta + 2 (see report section 6)
                    r["gap_lambda_minus_theta_plus_2"] = (
                        r["lambda_exponent"] - (r["theta_gap_exponent"] + 2.0))
            prev = p
            rows.append(r)

    with open(args.out, "w") as f:
        json.dump(rows, f, indent=2, default=str)

    print(f"{'family':14s} {'N':>4s} {'log10 q':>9s} {'log10(1/e)':>11s} "
          f"{'lambda':>7s} {'log10 Q':>8s} {'theta':>6s} {'lam-(th+2)':>11s} {'conv?':>6s}")
    for r in rows:
        th = r.get("theta_gap_exponent")
        gap = r.get("gap_lambda_minus_theta_plus_2")
        print(f"{r['family']:14s} {str(r['N']):>4s} {r['log10_q']:9.3f} "
              f"{-r['log10_err']:11.3f} {r['lambda_exponent']:7.3f} "
              f"{r['log10_Q_q2err']:8.2f} "
              f"{(f'{th:6.3f}' if th is not None else '   n/a')} "
              f"{(f'{gap:11.3f}' if gap is not None else '        n/a')} "
              f"{str(r['is_convergent_legendre']):>6s}")

    # summary: lambda at the largest N for each family
    print("\nsummary (largest N per family):")
    for name in order:
        rs = [r for r in rows if r["family"] == name]
        if rs:
            r = rs[-1]
            print(f"  {name:14s} log10 q={r['log10_q']:8.2f}  lambda={r['lambda_exponent']:.3f}"
                  f"  log10 Q={r['log10_Q_q2err']:8.2f}")
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
