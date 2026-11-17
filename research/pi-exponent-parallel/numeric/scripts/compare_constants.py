#!/usr/bin/env python3
"""Comparison of the mu-statistic T_K for several constants.

Control cases where mu = 2 is a THEOREM:
  * sqrt(2): all partial quotients equal 2 (bounded) -> q_n grows like (1+sqrt2)^n,
    log q_{n+1}/log q_n -> 1 trivially.
  * e: partial quotients 2,1,2,1,1,4,1,1,6,... grow LINEARLY (a_n ~ 2n/3), i.e.
    exactly the regime the numerics suggest for pi -- and mu(e) = 2 is a
    classical theorem because the continued fraction is explicit.

This is the key control experiment: a linearly growing partial-quotient sequence
is fully compatible with mu = 2.  What pi lacks is not small quotients but an
explicit/effective description of its continued fraction.

Usage: python3 compare_constants.py --tag d1e5 --n 100000 --out ../data
"""
import argparse
import json
import math
import os


def T_curve(a, grid):
    n = len(a)
    logq = [0.0] * n
    rho_prev = 0.0
    for k in range(1, n):
        logq[k] = logq[k - 1] + math.log10(a[k] + rho_prev)
        rho_prev = 1.0 / (a[k] + rho_prev)
    T = [0.0] * (n - 1)
    run = -1e9
    for k in range(n - 2, 0, -1):
        rk = logq[k + 1] / logq[k] if logq[k] > 0 else 1e9
        if rk > run:
            run = rk
        T[k] = run
    return [T[K] for K in grid], logq[n - 1] / (n - 1)


def cf_e(n):
    a = [2]
    for k in range(1, n):
        a.append(2 * (k + 1) // 3 if k % 3 == 2 else 1)
    return a


def cf_sqrt2(n):
    return [1] + [2] * (n - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="d1e5")
    ap.add_argument("--n", type=int, default=100000)
    ap.add_argument("--datadir", default="../data")
    ap.add_argument("--out", default="../data/compare_constants.json")
    args = ap.parse_args()
    n = args.n
    grid = []
    K = 2
    while K < n - 2:
        grid.append(K)
        K = int(K * 1.25) + 1

    pi = [int(x) for x in
          open(os.path.join(args.datadir, f"cf_{args.tag}_a.txt")).read().split()][:n]
    out = {}
    for name, a in (("pi", pi), ("e", cf_e(n)), ("sqrt2", cf_sqrt2(n))):
        Tc, levy = T_curve(a, grid)
        out[name] = dict(K=grid, T_minus_1=[t - 1.0 for t in Tc],
                         levy_log10_qn_over_n=levy,
                         max_a=max(a[1:]),
                         max_a_over_n=max(a[k] / k for k in range(1, len(a))),
                         argmax_over_n=max(range(1, len(a)), key=lambda k: a[k] / k),
                         max_a_over_n_ge1000=max(a[k] / k for k in range(1000, len(a))))
        print(f"{name:7s} max a_n={max(a[1:]):8d}  max a_n/n={out[name]['max_a_over_n']:8.3f}"
              f" (at n={out[name]['argmax_over_n']})"
              f"  log10 q_n/n={levy:.6f}  T_2-1={Tc[0]-1:.3e}  T_last-1={Tc[-1]-1:.3e}")
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print("wrote", args.out)
    print("\n  K      pi T_K-1      e T_K-1      sqrt2 T_K-1")
    for i in range(0, len(grid), max(1, len(grid) // 14)):
        print(f"{grid[i]:8d}  {out['pi']['T_minus_1'][i]:.3e}  "
              f"{out['e']['T_minus_1'][i]:.3e}  {out['sqrt2']['T_minus_1'][i]:.3e}")


if __name__ == "__main__":
    main()
