#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CF stage-3 arithmetic statistics:
  - run lengths of partial quotients a_n
  - autocorrelation of log a_n
  - q_n mod p for the first primes (streaming recurrence)
  - full factorisation of the first QN_LIMIT denominators q_n (Pollard rho)
  - largest prime factor / omega(q_n) for those
NUMERICAL EXPERIMENT ONLY.
"""

import csv
import json
import math
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "..", "pi-exponent-independent", "cf_d100000.csv")
OUT = HERE

# ---------------------------------------------------------------- primality
def is_prime(n, rounds=20):
    if n < 2:
        return False
    small = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    for p in small:
        if n % p == 0:
            return n == p
    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in small + [random.randrange(2, n - 1) for _ in range(rounds)]:
        if a % n == 0:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def pollard_rho(n):
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    while True:
        c = random.randrange(1, n)
        f = lambda x: (x * x + c) % n
        x, y, d = random.randrange(0, n), 0, 1
        y = x
        while d == 1:
            x = f(x)
            y = f(f(y))
            d = math.gcd(abs(x - y), n)
        if d != n:
            return d


def factor(n):
    if n == 1:
        return []
    if is_prime(n):
        return [n]
    d = pollard_rho(n)
    return factor(d) + factor(n // d)


# ---------------------------------------------------------------- main
def main():
    a = []
    with open(SRC, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            a.append((int(r["n"]), int(r["a_n"])))
    a.sort()
    N = len(a)
    vals = [x[1] for x in a]
    print("N =", N)

    # ---------- run lengths ----------
    def run_stats(seq, target):
        runs = []
        cur = 0
        for v in seq:
            if v == target:
                cur += 1
            else:
                if cur:
                    runs.append(cur)
                cur = 0
        if cur:
            runs.append(cur)
        return runs

    runs_one = run_stats(vals, 1)
    # runs of equal values (any value)
    eq_runs = []
    cur = 1
    for i in range(1, N):
        if vals[i] == vals[i - 1]:
            cur += 1
        else:
            eq_runs.append(cur)
            cur = 1
    eq_runs.append(cur)

    from collections import Counter
    c1 = Counter(runs_one)
    ceq = Counter(eq_runs)
    maxrun_one = max(runs_one) if runs_one else 0
    maxrun_eq = max(eq_runs) if eq_runs else 0

    # ---------- autocorrelation of log a_n ----------
    logs = [math.log(v) for v in vals]
    mu = sum(logs) / N
    var = sum((x - mu) ** 2 for x in logs) / N
    autocorr = {}
    for k in range(1, 11):
        s = 0.0
        for i in range(N - k):
            s += (logs[i] - mu) * (logs[i + k] - mu)
        autocorr[k] = (s / (N - k)) / var if var > 0 else 0.0

    # ---------- q_n mod p ----------
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]
    modstats = {}
    for p in primes:
        qm1, qm2 = 0, 1  # q_{-1}=0, q_0=1
        cnt = [0] * p
        for i in range(N):
            q = (vals[i] * qm2 + qm1) % p
            cnt[q] += 1
            qm1, qm2 = qm2, q
        # chi-square vs uniform
        exp = N / p
        chi2 = sum((c - exp) ** 2 / exp for c in cnt)
        modstats[p] = {"chi2": chi2, "dof": p - 1, "chi2_over_dof": chi2 / (p - 1),
                       "counts": cnt, "zero_hits": cnt[0]}

    # ---------- factor first QN_LIMIT denominators ----------
    QN_LIMIT = 60
    q_prev, q_cur = 0, 1  # q_{-1}, q_0
    fact_rows = []
    for i in range(min(QN_LIMIT, N)):
        q = vals[i] * q_cur + q_prev
        q_prev, q_cur = q_cur, q
        if q <= 1:
            continue
        fs = factor(q)
        fs.sort()
        largest = fs[-1]
        fact_rows.append({
            "n": i + 1,
            "digits": len(str(q)),
            "q": str(q),
            "factors": fs,
            "omega": len(fs),
            "largest_prime_factor": largest,
            "lpf_digits": len(str(largest)),
            "lpf_over_q": largest / q,
        })
        print(f"n={i+1:3d} digits={len(str(q)):3d} omega={len(fs):2d} "
              f"lpf_digits={len(str(largest)):3d} lpf/q={largest/q:.3e}")

    # ---------- summary ----------
    summary = {
        "N": N,
        "runs_of_1": {
            "count": len(runs_one),
            "max": maxrun_one,
            "mean": sum(runs_one) / len(runs_one) if runs_one else 0,
            "hist": dict(sorted(c1.items())),
        },
        "equal_runs": {
            "count": len(eq_runs),
            "max": maxrun_eq,
            "mean": sum(eq_runs) / len(eq_runs) if eq_runs else 0,
            "hist": dict(sorted(ceq.items())[:20]),
        },
        "autocorr_log_a": autocorr,
        "modstats": {str(p): {"chi2": v["chi2"], "dof": v["dof"],
                              "chi2_over_dof": v["chi2_over_dof"],
                              "zero_hits": v["zero_hits"]}
                     for p, v in modstats.items()},
        "factor_first": fact_rows,
    }
    with open(os.path.join(OUT, "cf_arith_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    with open(os.path.join(OUT, "cf_runs.csv"), "w", encoding="utf-8") as f:
        f.write("run_len,count_of_1_runs,count_of_equal_runs\n")
        L = max(max(c1) if c1 else 1, max(ceq) if ceq else 1)
        for r in range(1, L + 1):
            f.write(f"{r},{c1.get(r,0)},{ceq.get(r,0)}\n")

    print("runs of 1:", summary["runs_of_1"]["count"], "max", maxrun_one,
          "mean", round(summary["runs_of_1"]["mean"], 3))
    print("equal runs:", summary["equal_runs"]["count"], "max", maxrun_eq,
          "mean", round(summary["equal_runs"]["mean"], 3))
    print("autocorr:", {k: round(v, 5) for k, v in autocorr.items()})
    print("mod chi2/dof:", {p: round(v["chi2_over_dof"], 3) for p, v in modstats.items()})


if __name__ == "__main__":
    random.seed(20261117)
    main()
