#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exp_products.py -- why products of linear factors do not improve the exponent.

For a convergent p/q of pi, put L = q*pi - p.  Then
    |L| = q^{1-kappa}  with kappa = local approximation exponent,
so the "effective value exponent" of the linear form is A = kappa - 1.

For Q = prod_j (q_j X - p_j) (degree k, height H ~ prod q_j) we have
    |Q(pi)| = prod |L_j| = H^{1-kappa_eff},
so the effective exponent is again ~ kappa_eff - 1, NOT k or anything
growing with the number of factors.  Hence a general value bound
|Q(pi)| >= c H^{-A} can rule out kappa > A+1 only; products never do better
than the linear case A ~ 1.

Outputs ../data/products.csv
"""
import os

import mpmath as mp

import roth_machine as rm

mp.mp.dps = 120


def main():
    a, conv = rm.pi_cf(28)
    print("partial quotients:", a[:20])
    print(f"{'i':>3} {'q':>14} {'kappa_i':>9} {'A_i=-log|q pi-p|/log q':>24} {'q^2|pi-p/q|':>14} {'next a':>7}")
    rows = []
    for i in range(1, min(len(conv) - 1, 22)):
        p, q, ai = conv[i]
        kappa = rm.approx_exponent(p, q)
        mp.mp.dps = 120
        L = abs(q * mp.pi - p)
        A = float(-mp.log10(L) / mp.log10(q))
        qual = float(q * q * abs(mp.pi - mp.mpf(p) / q))
        nxt = a[i + 1] if i + 1 < len(a) else None
        rows.append((i, q, kappa, A, qual, nxt))
        print(f"{i:>3} {q:>14} {kappa:>9.4f} {A:>24.4f} {qual:>14.4e} {str(nxt):>7}")

    # products of k consecutive convergents
    print()
    print("products of consecutive convergents:")
    print(f"{'k':>3} {'indices':>14} {'log10 H':>10} {'log10|Q(pi)|':>14} {'A_eff':>9}")
    rows2 = []
    for k in (1, 2, 3, 4):
        for start in (5, 8, 11, 14, 17):
            pts = conv[start:start + k]
            if len(pts) < k:
                continue
            coeffs = [1]
            for (p, q, _) in pts:
                new = [0] * (len(coeffs) + 1)
                for j, c in enumerate(coeffs):
                    new[j] += c * (-p)
                    new[j + 1] += c * q
                coeffs = new
            H = max(abs(c) for c in coeffs)
            mp.mp.dps = 200
            val = mp.mpf(0)
            for c in reversed(coeffs):
                val = val * mp.pi + c
            l10 = float(mp.log10(abs(val)))
            Aeff = -l10 / mp.log10(H)
            idx = f"{start}..{start+k-1}"
            rows2.append((k, idx, float(mp.log10(H)), l10, float(Aeff)))
            print(f"{k:>3} {idx:>14} {float(mp.log10(H)):>10.4f} {l10:>14.4f} {float(Aeff):>9.4f}")

    out = os.path.join(os.path.dirname(__file__), "..", "data", "products.csv")
    with open(os.path.abspath(out), "w") as f:
        f.write("type,i_or_indices,a,b,c,d,e\n")
        for r in rows:
            f.write(f"single,{r[0]},{r[1]},{r[2]:.6f},{r[3]:.6f},{r[4]:.6e},{r[5]}\n")
        for r in rows2:
            f.write(f"prod,{r[1]},{r[0]},{r[2]:.6f},{r[3]:.6f},{r[4]:.6f},\n")
    print("wrote", os.path.abspath(out))


if __name__ == "__main__":
    main()
