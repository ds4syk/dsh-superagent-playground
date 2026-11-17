#!/usr/bin/env python3
"""Gauss-Kuzmin null model for the T_K statistic of a "random" real number.

For a uniformly random x in (0,1), the partial quotients are (to an excellent
approximation for tail statistics, and exactly in the sense that the Gauss
measure is the invariant measure of the continued fraction map) i.i.d. with

    P(a = k) = log2(1 + 1/(k(k+2))),      P(a >= k) = log2(1 + 1/k).

We sample a_k = ceil(1/(2^U - 1)) with U uniform, which reproduces the
Gauss-Kuzmin law exactly, build N quotients, and compute

    r_k = log q_{k+1}/log q_k,     T_K = sup_{k >= K} r_k,

the same statistic as scripts/cf_analyze.py.  This gives the distribution of
T_K for a "random" number, against which pi can be compared.  (Nothing here is
a proof about pi: mu(x) = 2 for almost every x is a theorem, but pi is one
specific point, and the CF of pi is not known to be "random" in any sense.)

Usage: python3 gk_montecarlo.py --n 97210 --trials 200 --out ../data
"""
import argparse
import json
import math
import random

LÉVY10 = math.pi ** 2 / (12 * math.log(2)) / math.log(10)


def sample_quotients(n, rng):
    a = [0] * n
    for i in range(n):
        u = rng.random()
        z = 2.0 ** u
        if z <= 1.0:               # u smaller than double resolution
            a[i] = 10 ** 9
            continue
        v = 1.0 / (z - 1.0)
        # a = floor(1/(2^u - 1)) has exactly the Gauss-Kuzmin law:
        # P(a=k) = log2(1+1/(k(k+2)))  (note: floor, not ceil)
        k = int(v)
        a[i] = max(k, 1)
    return a


def T_curve(a, grid):
    n = len(a)
    logq = [0.0] * n
    rho = [0.0] * n
    rho_prev = 0.0
    for k in range(1, n):
        logq[k] = logq[k - 1] + math.log10(a[k] + rho_prev)
        rho_prev = 1.0 / (a[k] + rho_prev)
        rho[k] = rho_prev
    r = [0.0] * (n - 1)
    for k in range(1, n - 1):
        # q_k = 1 can happen when a_1 = 1 (prob ~0.415 under Gauss-Kuzmin);
        # then log q_k = 0 and r_k is not defined.  Such small-k terms are
        # irrelevant to mu = limsup; mark them as non-contributing.
        r[k] = logq[k + 1] / logq[k] if logq[k] > 0 else -1e18
    T = [0.0] * (n - 1)
    run = -1e9
    for k in range(n - 2, -1, -1):
        if r[k] > run:
            run = r[k]
        T[k] = run
    return [T[K] for K in grid], max(a[1:]) / (n - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=97210)
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20261117)
    ap.add_argument("--out", default="../data")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    n = args.n
    grid = []
    K = 2
    while K < n - 2:
        grid.append(K)
        K = int(K * 1.25) + 1
    grid.append(n - 2)

    curves = []
    ratios = []
    for t in range(args.trials):
        a = sample_quotients(n, rng)
        Tc, mratio = T_curve(a, grid)
        curves.append(Tc)
        ratios.append(mratio)
        if (t + 1) % 25 == 0:
            print(f"  trial {t+1}/{args.trials}", flush=True)

    # quantiles per grid point
    rows = []
    for i, K in enumerate(grid):
        col = sorted(c[i] for c in curves)
        m = len(col)
        rows.append(dict(K=K,
                         q05=col[int(0.05 * m)], q25=col[int(0.25 * m)],
                         median=col[m // 2],
                         q75=col[int(0.75 * m)], q95=col[int(0.95 * m)],
                         mean=sum(col) / m,
                         max=col[-1]))
    ratios.sort()
    summary = dict(n=n, trials=args.trials, seed=args.seed,
                   max_a_over_n=dict(
                       q05=ratios[int(0.05 * len(ratios))],
                       median=ratios[len(ratios) // 2],
                       q95=ratios[int(0.95 * len(ratios))],
                       max=ratios[-1]),
                   curve=rows)
    with open(f"{args.out}/gk_mc_n{n}_trials{args.trials}.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "curve"}, indent=2))
    print("K, median T_K, q95 T_K, q05 T_K")
    for w in rows:
        print(f"{w['K']:8d}  {w['median']:.10f}  {w['q95']:.10f}  {w['q05']:.10f}")


if __name__ == "__main__":
    main()
