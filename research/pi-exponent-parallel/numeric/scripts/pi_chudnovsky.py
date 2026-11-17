#!/usr/bin/env python3
"""Independent computation of pi via the Chudnovsky series + binary splitting.

This is a *different algorithm* from scripts/pi_agm.py (rational arithmetic and
integer square root instead of Decimal AGM), used to cross-check the leading
digits.  Pure Python integers (Karatsuba multiplication).

1/pi = 12 sum_{k>=0} (-1)^k (6k)! (13591409 + 545140134 k)
                        / ((3k)! (k!)^3 (640320)^(3k+3/2))

Usage: python3 pi_chudnovsky.py --digits 200000 --out ../data/pi_chud_2e5
"""
import argparse
import json
import math
import sys
import time

sys.set_int_max_str_digits(0)

C = 640320
C3_OVER_24 = C ** 3 // 24


def bs(a, b):
    if b - a == 1:
        if a == 0:
            Pab = Qab = 1
        else:
            Pab = (6 * a - 5) * (2 * a - 1) * (6 * a - 1)
            Qab = a * a * a * C3_OVER_24
        Tab = Pab * (13591409 + 545140134 * a)
        if a & 1:
            Tab = -Tab
        return Pab, Qab, Tab
    m = (a + b) // 2
    Pam, Qam, Tam = bs(a, m)
    Pmb, Qmb, Tmb = bs(m, b)
    return Pam * Pmb, Qam * Qmb, Tam * Qmb + Pam * Tmb


def pi_scaled(d):
    """Return N = floor(pi * 10^d)."""
    # digits per term ~ 14.18
    n = d // 14 + 10
    P, Q, T = bs(0, n)
    # sqrt(10005) to d+40 digits: isqrt(10005 * 10^(2*(d+40)))
    g = 40
    S = math.isqrt(10005 * 10 ** (2 * (d + g)))
    # pi * 10^d = 426880 * Q * 10^d * sqrt(10005) / T
    num = 426880 * Q * S
    den = T * 10 ** g          # since S ~ sqrt(10005) * 10^(d+g) and we want *10^d
    return num // den, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--digits", type=int, default=200_000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    d = args.digits
    out = args.out or f"../data/pi_chud_d{d}"

    t0 = time.time()
    N, n = pi_scaled(d)
    t1 = time.time()
    s = str(N)                      # "314159265..." length d+1
    assert s[0] == "3", s[:10]
    frac = s[1:]
    with open(out + ".txt", "w") as f:
        for i in range(0, len(frac), 200):
            f.write(frac[i:i + 200] + "\n")
    meta = dict(digits=d, terms=n, seconds=round(t1 - t0, 2),
                method="Chudnovsky binary splitting, integer arithmetic",
                python=sys.version.split()[0])
    with open(out + ".json", "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta), flush=True)
    print("head:", s[:60], flush=True)


if __name__ == "__main__":
    main()
