#!/usr/bin/env python3
"""Compute pi to D decimal digits with the Gauss-Legendre AGM iteration.

Uses decimal.Decimal (libmpdec, fast NTT multiplication).  All digits are
obtained analytically; there is *no* floating point involved except the
explicit guard digits (GUARD) which are computed and then discarded.

Outputs
-------
<out>.txt   : fractional digits (D of them, wrapped at WRAP chars/line), no header
<out>.json  : metadata (D, guard, iterations, wall time, checks)

The value written is *truncated* (not rounded): the digit string is the exact
first D fractional digits produced by the AGM value; since the AGM is run with
GUARD >= 60 extra correct digits, truncation error is < 10^-D with a huge
margin (the probability that the (D+1)-th AGM digit is itself wrong is
astronomically small; a cross-check with an independent algorithm is done by
scripts/pi_chudnovsky.py).

Usage:  python3 pi_agm.py [--digits 1000000] [--out ../data/pi_1e6]
"""
import argparse
import json
import sys
import time
from decimal import Decimal, getcontext

GUARD = 80
WRAP = 200

# first 50 decimal digits of pi, used as a sanity check
_HEAD = "14159265358979323846264338327950288419716939937510"


def pi_agm(d: int):
    """Return (pi as Decimal, iterations).  Error ~ 10^-(d+GUARD-10)."""
    getcontext().prec = d + GUARD
    one = Decimal(1)
    two = Decimal(2)
    four = Decimal(4)
    ten = Decimal(10)
    a = one
    b = two.sqrt() / two          # 1/sqrt(2)
    t = one / four                 # 1/4
    p = one                        # 2^k
    eps = ten ** (-(d + GUARD - 20))
    k = 0
    while True:
        an = (a + b) / two
        b = (a * b).sqrt()
        t = t - p * (a - an) * (a - an)
        a = an
        p = p * two
        k += 1
        if k > 3 and (a - b).copy_abs() < eps:
            break
    pi = (a + b) ** 2 / (four * t)
    return pi, k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--digits", type=int, default=1_000_000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    d = args.digits
    out = args.out or f"../data/pi_d{d}"

    t0 = time.time()
    pi, iters = pi_agm(d)
    t1 = time.time()

    s = str(pi)                    # "3.1415..." with d+GUARD-ish digits
    assert s[0] == "3" and s[1] == ".", s[:20]
    frac = s[2:]
    assert len(frac) >= d, (len(frac), d)
    frac = frac[:d]                # truncate

    checks = {}
    checks["head50_ok"] = frac[:50] == _HEAD
    # Feynman point: six consecutive 9s at decimal positions 762..767 (1-based)
    checks["feynman_999999_at_762"] = frac[761:767] == "999999"
    checks["len"] = len(frac)
    if not checks["head50_ok"]:
        print("FATAL: head mismatch", frac[:60], file=sys.stderr)
        sys.exit(2)

    with open(out + ".txt", "w") as f:
        for i in range(0, d, WRAP):
            f.write(frac[i:i + WRAP] + "\n")

    meta = dict(digits=d, guard=GUARD, iterations=iters,
                seconds=round(t1 - t0, 2),
                method="Gauss-Legendre AGM via decimal.Decimal (libmpdec)",
                checks=checks,
                python=sys.version.split()[0])
    with open(out + ".json", "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta), flush=True)
    print("wrote", out + ".txt", flush=True)


if __name__ == "__main__":
    main()
