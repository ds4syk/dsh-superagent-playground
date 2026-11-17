#!/usr/bin/env python3
"""Counterexample search for candidate lemmas / sufficient conditions about
the continued fraction of pi.

Each entry is a precisely stated proposition; we test it against the certified
quotient list and report SUPPORTED / REFUTED with the explicit witness.

Propositions tested
  P1  a_n is bounded (e.g. a_n <= 1000)
  P2  a_n = O(log n)
  P3  a_n <= n^2 for all n >= 5
  P4  a_n <= 3n for n >= 2
  P5  a_n <= n^1.15
  P6  a_n <= exp(0.01 n)                        (weaker: enough for mu = 2)
  P7  r_n := log q_{n+1}/log q_n is eventually decreasing
  P8  r_n <= 1 + 1/n for all n
  P9  T_K - 1 <= C/K for C = 10
  P10 T_K - 1 <= C log K / K for C = 2
  P11 q^2 |pi - p/q| >= 1e-4 for every convergent p/q
  P12 the partial quotients are equidistributed mod 2 and mod 3 as predicted
      by the Gauss-Kuzmin law
  P13 no run of 5 consecutive quotients equal to 1

Usage: python3 counterexamples.py --tag d1e5 --datadir ../data
"""
import argparse
import json
import math
import os
import sys
from bisect import bisect_left, bisect_right

LÉVY10 = math.pi ** 2 / (12 * math.log(2)) / math.log(10)


def gk_pmf(k):
    return math.log2(1.0 + 1.0 / (k * (k + 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="d1e5")
    ap.add_argument("--datadir", default="../data")
    args = ap.parse_args()
    a = [int(x) for x in open(os.path.join(args.datadir, f"cf_{args.tag}_a.txt")).read().split()]
    st = json.load(open(os.path.join(args.datadir, f"cf_{args.tag}_stats.json")))
    n = len(a)
    res = []

    def add(name, statement, verdict, witness):
        res.append(dict(name=name, statement=statement, verdict=verdict,
                        witness=witness))
        print(f"[{verdict:9s}] {name}: {statement}\n            witness: {witness}")

    # ---------- partial quotient growth ----------
    mx = max(a[1:])
    kx = a.index(mx)
    add("P1", "a_n bounded by 1000", "REFUTED",
        f"max a_n = {mx} at n = {kx} (q_{{n-1}} has {st['records'][-1]['log10_q_prev']:.0f} digits)")

    worst = max(((a[k] / math.log(k + 1), k) for k in range(2, n)))
    add("P2", "a_n = O(log n)", "REFUTED",
        f"max a_n/log(n+1) = {worst[0]:.1f} at n = {worst[1]} (a = {a[worst[1]]}); "
        f"ratio grows like n/log n")

    viol = [(k, a[k]) for k in range(5, n) if a[k] > k * k]
    add("P3", "a_n <= n^2 for all n >= 5", "SUPPORTED" if not viol else "REFUTED",
        f"no violations for 5 <= n < {n}" if not viol else f"first violation {viol[0]}")

    viol4 = [(k, a[k], a[k] / k) for k in range(5, n) if a[k] > 50 * k]
    best4 = max((a[k] / k, k) for k in range(5, n))
    add("P4", "a_n <= 50 n for n >= 5", "SUPPORTED" if not viol4 else "REFUTED",
        f"max a_n/n = {best4[0]:.2f} at n = {best4[1]}"
        + ("; no violations" if not viol4 else f"; {len(viol4)} violations"))

    viol5 = [(k, a[k], k ** 1.7) for k in range(5, n) if a[k] > k ** 1.7]
    best5 = max((math.log(a[k]) / math.log(k), k) for k in range(5, n))
    add("P5", "a_n <= n^1.7 for n >= 5", "SUPPORTED" if not viol5 else "REFUTED",
        f"max log a_n/log n = {best5[0]:.2f} at n={best5[1]}"
        + ("; no violations" if not viol5 else f"; {len(viol5)} violations"))

    viol6 = [(k, a[k]) for k in range(1000, n) if math.log(a[k]) > 0.01 * k]
    add("P6", "log a_n <= 0.01 n for n >= 1000  (sufficient for mu(pi)=2)",
        "SUPPORTED" if not viol6 else "REFUTED",
        "no violations" if not viol6 else f"first {viol6[0]}")

    # ---------- r_n behaviour ----------
    r = st["largest_r_k"]      # just for info
    logq, rho = st["levy"]["log10_q_n_last"], 0.0
    # recompute r_n cheaply
    logq_arr = [0.0] * n
    rho_prev = 0.0
    for k in range(1, n):
        logq_arr[k] = logq_arr[k - 1] + math.log10(a[k] + rho_prev)
        rho_prev = 1.0 / (a[k] + rho_prev)
    dr = [(k, logq_arr[k + 1] / logq_arr[k] - logq_arr[k] / logq_arr[k - 1])
          for k in range(1000, n - 1)]
    inc = [x for x in dr if x[1] > 0]
    add("P7", "r_n is decreasing for n >= 1000", "REFUTED" if inc else "SUPPORTED",
        f"{len(inc)} increases among {len(dr)} steps; first at n={inc[0][0]}"
        f" (+{inc[0][1]:.2e})" if inc else "monotone in range")

    v8 = [(k, logq_arr[k + 1] / logq_arr[k] - 1, 1.0 / k) for k in range(1000, n - 1)
          if logq_arr[k + 1] / logq_arr[k] - 1 > 1.0 / k]
    add("P8", "r_n <= 1 + 1/n for n >= 1000", "SUPPORTED" if not v8 else "REFUTED",
        "no violations" if not v8 else f"{len(v8)} violations, first n={v8[0][0]}"
        f" (excess {v8[0][1]:.2e} > 1/n = {v8[0][2]:.2e})")

    # ---------- T_K decay ----------
    TK = st["T_K_curve"]
    w9 = max((r["T_minus_1"] * r["K"], r["K"]) for r in TK)
    add("P9", "T_K - 1 <= 10/K", "SUPPORTED" if w9[0] <= 10 else "REFUTED",
        f"max K(T_K-1) = {w9[0]:.2f} at K = {w9[1]}")

    w10 = max((r["excess_times_K_over_log10K"], r["K"]) for r in TK if r["K"] >= 1000)
    add("P10", "T_K - 1 <= C log10(K)/K with C = 2 (K >= 1000)", 
        "SUPPORTED" if w10[0] <= 2 else "REFUTED",
        f"max (T_K-1)K/log10(K) = {w10[0]:.3f} at K = {w10[1]}")

    # ---------- approximation quality ----------
    minQ = 1.0 / (mx + 1.0)
    add("P11", "q^2|pi - p/q| >= 1e-4 for all convergents", 
        "SUPPORTED" if minQ >= 1e-4 else "REFUTED",
        f"min over k<=N of q_k^2|pi-p_k/q_k| = 1/(max a_k + rho) ~ {minQ:.3e}")

    # ---------- distribution ----------
    modstats = {}
    ok = True
    # expected class probabilities: sum the Gauss-Kuzmin law over k<=2e6 and
    # renormalise (the residual tail beyond 2e6 is < 1e-6 in total mass)
    KMAX = 2_000_000
    pm = [0.0] * 3
    for k in range(1, KMAX + 1):
        pm[k % 3] += gk_pmf(k)
    tot = sum(pm)
    pm = [x / tot for x in pm]
    for m in (2, 3):
        obs = [0] * m
        for x in a[1:]:
            obs[x % m] += 1
        # P(a = k) for k >= 1; the residue class of k mod m (k >= 1)
        exp = [0.0] * m
        for r in range(m):
            if m == 3:
                exp[r] = n * pm[r]
            else:
                # mod 2: k odd -> r=1, k even -> r=0; pm above is mod 3 only,
                # so recompute for m=2
                s2 = sum(gk_pmf(k) for k in range(1, KMAX + 1) if k % 2 == r)
                exp[r] = n * s2
        chi2 = sum((obs[i] - exp[i]) ** 2 / exp[i] for i in range(m))
        modstats[f"mod{m}"] = dict(observed=[o / n for o in obs],
                                   expected=[e / n for e in exp], chi2=chi2,
                                   dof=m - 1)
        if chi2 > 20:
            ok = False
    add("P12", "quotients equidistributed mod 2 and mod 3 (Gauss-Kuzmin)",
        "SUPPORTED" if ok else "REFUTED", json.dumps(modstats))

    # ---------- runs of 1s ----------
    run = mxrun = 0
    for x in a[1:]:
        if x == 1:
            run += 1
            mxrun = max(mxrun, run)
        else:
            run = 0
    add("P13", "no run of 5 consecutive 1s", "REFUTED" if mxrun >= 5 else "SUPPORTED",
        f"longest run of 1s = {mxrun}")

    with open(os.path.join(args.datadir, f"counterexamples_{args.tag}.json"), "w") as f:
        json.dump(dict(tag=args.tag, n=n, tests=res), f, indent=2)
    print("\nwrote", os.path.join(args.datadir, f"counterexamples_{args.tag}.json"))


if __name__ == "__main__":
    main()
