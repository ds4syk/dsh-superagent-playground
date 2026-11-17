#!/usr/bin/env python3
"""Legendre / best-approximation brute-force check for q <= QMAX.

Facts under test (all exact integer arithmetic with 60 digits of pi):

  (1) Legendre: every p/q with 2 q^2 ||q pi|| < 1 is a convergent of pi.
  (2) Best approximation of the 2nd kind: for q_k <= q < q_{k+1},
          ||q pi|| >= ||q_k pi|| .
  (3) Best approximation of the 1st kind: the running minimum of
          ||q pi|| / q
      is attained at a convergent (record list printed).
  (4) The record lows of ||q pi|| up to QMAX are exactly the convergent
      denominators, with q^2||q pi|| = q/(R_{k+1}+rho_k) values reported.

Usage: python3 best_rationals.py --pi ../data/pi_chud_d1000000.txt --qmax 200000
"""
import argparse
import bisect
import json
import sys

sys.set_int_max_str_digits(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pi", default="../data/pi_chud_d1000000.txt")
    ap.add_argument("--qmax", type=int, default=200_000)
    ap.add_argument("--cf", default="../data/cf_d1e5_a.txt")
    ap.add_argument("--out", default="../data/best_rationals.json")
    args = ap.parse_args()
    with open(args.pi) as f:
        pi_str = f.read().replace("\n", "")
    SCALE = 10 ** 60
    M = 3 * SCALE + int(pi_str[:60])      # floor(pi * 10^60)

    A = [int(x) for x in open(args.cf).read().split()]
    conv = []
    pm1, qm1, p, q = 1, 0, A[0], 1
    conv.append((0, p, q))
    for k in range(1, len(A)):
        p, pm1 = A[k] * p + pm1, p
        q, qm1 = A[k] * q + qm1, q
        if q > args.qmax:
            break
        conv.append((k, p, q))
    conv_pairs = {(x[1], x[2]) for x in conv}
    conv_qs = sorted(x[2] for x in conv)
    k_of_q = {x[2]: x[0] for x in conv}

    legendre_bad = []
    second_kind_bad = []
    rec2 = []          # record lows of ||q pi||
    rec1 = []          # record lows of ||q pi|| / q
    best2 = None
    best1 = None
    for qq in range(1, args.qmax + 1):
        t = (qq * M) % SCALE
        dist = min(t, SCALE - t)
        if best2 is None or dist < best2:
            best2 = dist
            rec2.append((qq, dist))
        if best1 is None or dist * best1[1] < best1[0] * qq:
            best1 = (dist, qq)
            rec1.append((qq, dist))
        # (1) Legendre
        if 2 * dist * qq * qq < SCALE:
            pn = round(qq * M / SCALE)
            if (pn, qq) not in conv_pairs:
                legendre_bad.append((qq, pn, dist / SCALE, dist * qq * qq / SCALE))
        # (2) 2nd kind
        i = bisect.bisect_right(conv_qs, qq) - 1
        if i >= 0:
            kk, pk, qk = conv[i]
            tk = (qk * M) % SCALE
            distk = min(tk, SCALE - tk)
            if dist < distk:
                second_kind_bad.append((qq, kk))

    out = dict(qmax=args.qmax, n_convergents=len(conv),
               convergent_denominators=conv_qs,
               records_2nd_kind=[
                   dict(q=x[0], norm=x[1] / SCALE, q2norm=x[0] * x[0] * x[1] / SCALE,
                        is_convergent=x[0] in k_of_q) for x in rec2],
               records_1st_kind=[
                   dict(q=x[0], norm_over_q=x[1] / SCALE / x[0],
                        is_convergent=x[0] in k_of_q) for x in rec1],
               n_legendre_violations=len(legendre_bad),
               legendre_violations=legendre_bad[:10],
               n_second_kind_violations=len(second_kind_bad),
               second_kind_violations=second_kind_bad[:10])
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    print(f"qmax={args.qmax}, convergents: {conv_qs}")
    print(f"(1) Legendre violations: {len(legendre_bad)}")
    print(f"(2) best-approximation (2nd kind) violations: {len(second_kind_bad)}")
    print("\n(4) record lows of ||q pi||:")
    for r in out["records_2nd_kind"]:
        print(f"   q={r['q']:8d}  ||q pi||={r['norm']:.6e}"
              f"  q^2||q pi||={r['q2norm']:.6f}  convergent={r['is_convergent']}")
    print("\n(3) record lows of ||q pi||/q :")
    for r in out["records_1st_kind"]:
        print(f"   q={r['q']:8d}  ||q pi||/q={r['norm_over_q']:.6e}"
              f"  convergent={r['is_convergent']}")


if __name__ == "__main__":
    main()
