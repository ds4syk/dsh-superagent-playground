#!/usr/bin/env python3
"""End-to-end exact spot checks of the extracted continued fraction.

For selected indices k we compute the convergent p_k/q_k *exactly* (binary
splitting on the 2x2 transfer matrices M_i = [[a_i,1],[1,0]]) and then verify,
with the 10^6-digit expansion of pi,

    (1) sign of  q_k*pi - p_k  equals (-1)^k,
    (2) q_k^2 |pi - p_k/q_k|  agrees with the double-precision prediction
        1/(R_{k+1} + rho_k)  of scripts/cf_analyze.py.

This validates simultaneously: the decimal expansion, the certified quotient
list, and the tail recursions used for the statistics.

Note on precision: computing q_k*pi - p_k involves a cancellation of ~2*log10(q_k)
digits (q_k*pi and p_k agree in their leading ~log10 q_k digits), so the
Decimal context needs 2*log10(q_k) digits while pi must be known to that many
digits as well.

Usage: python3 verify_convergents.py --a ../data/cf_pi_d1000000_a.txt \
          --pi ../data/pi_d1000000.txt --indices 10,100,1000,10000,50000
"""
import argparse
import json
import math
import sys
from decimal import Decimal, getcontext

sys.set_int_max_str_digits(0)


def matmul(X, Y):
    a, b, c, d = X
    e, f, g, h = Y
    return (a * e + b * g, a * f + b * h, c * e + d * g, c * f + d * h)


def product(a, lo, hi):
    """Product M_{lo} ... M_{hi-1} (hi exclusive) = [[p, p', ],[q, q']]."""
    if hi - lo == 1:
        return (a[lo], 1, 1, 0)
    mid = (lo + hi) // 2
    return matmul(product(a, lo, mid), product(a, mid, hi))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--pi", required=True)
    ap.add_argument("--indices", default="10,100,1000,10000,50000")
    ap.add_argument("--out", default="../data/verify_convergents.json")
    args = ap.parse_args()
    with open(args.a) as f:
        A = [int(x) for x in f.read().split()]
    with open(args.pi) as f:
        pi_str = f.read().replace("\n", "")

    idxs = [int(x) for x in args.indices.split(",")]
    out = []
    for k in idxs:
        if k >= len(A) - 1:
            print(f"skip k={k}: only {len(A)} quotients", flush=True)
            continue
        P, Pm1, Q, Qm1 = product(A, 0, k + 1)     # p_k, p_{k-1}, q_k, q_{k-1}
        nq = len(str(Q))
        prec = 2 * nq + 400
        assert len(pi_str) >= prec, (len(pi_str), prec)
        getcontext().prec = prec
        pi = Decimal("3." + pi_str[:prec])
        val = Q * pi - P                           # ~ (-1)^k / q_k
        absval = abs(val)
        q2err = Q * absval                         # = q_k^2 |pi - p_k/q_k|

        # double-precision reference from the same quotients
        logq, rho_prev = 0.0, 0.0
        for j in range(1, k + 1):
            logq += math.log10(A[j] + rho_prev)
            rho_prev = 1.0 / (A[j] + rho_prev)
        Rk = float(A[-1])
        for j in range(len(A) - 2, k, -1):
            Rk = A[j] + 1.0 / Rk
        ref = 1.0 / (Rk + rho_prev)

        rel_dev = float(abs(q2err - Decimal(ref)) / Decimal(ref))
        rec = dict(k=k, digits_q=nq, log10_q=logq,
                   q_squared_error=float(q2err),
                   predicted_from_tails=ref,
                   relative_deviation=rel_dev,
                   sign_ok=(val > 0) == (k % 2 == 0),
                   in_sanity_range=(1e-6 < float(q2err) < 1.0))
        out.append(rec)
        print(f"k={k:6d} digits(q)={nq:6d} q^2|pi-p/q|={float(q2err):.12g} "
              f"predicted={ref:.12g} reldev={rel_dev:.2e} "
              f"sign_ok={rec['sign_ok']}", flush=True)
    with open(args.out, "w") as f:
        json.dump(dict(indices=idxs, checks=out), f, indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
