#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exp_pigeonhole.py -- numerical illustration of the degree obstruction.

For a transcendental alpha (= pi), for each degree D the minimum of |Q(alpha)|
over integer polynomials of degree <= D and height <= H behaves like
H^{-D} up to a constant depending on D.  Consequently a value bound
|Q(alpha)| >= c H^{-A} required to hold for ALL such Q cannot have A < D.

Search is done in double precision (fast), the winner is re-evaluated with
mpmath at 60 digits.  Output: ../data/pigeonhole.csv
"""
import itertools
import math
import os
import time

import mpmath as mp

PI_FLOAT = math.pi
PI_MP = mp.pi
mp.mp.dps = 80


def min_abs_poly(D, H):
    best = None
    best_coeffs = None
    rng = range(-H, H + 1)
    for coeffs in itertools.product(rng, repeat=D + 1):
        if not any(coeffs):
            continue
        val = 0.0
        for k in range(D, -1, -1):
            val = val * PI_FLOAT + coeffs[k]
        av = abs(val)
        if av == 0.0:
            continue
        if best is None or av < best:
            best = av
            best_coeffs = coeffs
    return best, best_coeffs


def exact_abs(coeffs):
    val = mp.mpf(0)
    for k in range(len(coeffs) - 1, -1, -1):
        val = val * PI_MP + coeffs[k]
    return abs(val)


def main():
    rows = []
    print(f"{'D':>2} {'H':>4} {'npolys':>9} {'log10|Q|':>14} {'-log10|Q|/log10 H':>20}  best Q")
    for D, Hs in ((1, (5, 10, 20, 40)), (2, (5, 10, 20, 40)), (3, (3, 5, 8, 12))):
        for H in Hs:
            npoly = (2 * H + 1) ** (D + 1)
            if npoly > 3_000_000:
                continue
            t0 = time.time()
            val, coeffs = min_abs_poly(D, H)
            if coeffs is None:
                continue
            val_exact = exact_abs(coeffs)
            l10 = float(mp.log10(val_exact))
            ratio = -l10 / math.log10(H)
            poly = " + ".join(f"{c}*X^{k}" for k, c in enumerate(coeffs) if c)
            hpoly = max(abs(c) for c in coeffs)
            rows.append((D, H, l10, ratio, poly, hpoly))
            print(f"{D:>2} {H:>4} {npoly:>9} {l10:>14.6f} {ratio:>20.6f}  {poly[:46]}  ({time.time()-t0:.1f}s)")
    out = os.path.join(os.path.dirname(__file__), "..", "data", "pigeonhole.csv")
    with open(os.path.abspath(out), "w") as f:
        f.write("D,H,log10_min_abs,neg_log10_over_log10H,poly,hpolynomial\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]},{r[2]:.10f},{r[3]:.10f},{r[4]},{r[5]}\n")
    print("wrote", os.path.abspath(out))


if __name__ == "__main__":
    main()
