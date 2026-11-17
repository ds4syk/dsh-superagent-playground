#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CF-direction deep dive: numerical statistics of pi's partial quotients.

Reads the (already computed, first-round) continued-fraction data from
../../pi-exponent-independent/cf_d100000.csv (read only), and writes
record values, top partial quotients, distribution fit and envelope
statistics into THIS directory only.

NUMERICAL EXPERIMENT ONLY -- not part of any proof.
"""

import csv
import math
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "..", "pi-exponent-independent", "cf_d100000.csv")
OUT = HERE

LOG10 = math.log(10.0)


def load():
    rows = []
    with open(SRC, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append((int(r["n"]), int(r["a_n"]), float(r["log10_q_n"])))
    return rows


def main():
    rows = load()
    N = len(rows)
    print("rows:", N)

    # ---------- record values ----------
    rec = []
    mx = 0
    for n, a, lq in rows:
        if a > mx:
            mx = a
            rec.append({
                "n": n, "a": a, "log10_q": lq,
                "ratio": math.log(a) / (lq * LOG10) if lq > 0 else None,
            })

    # ---------- top 20 ----------
    top = sorted(rows, key=lambda t: -t[1])[:20]
    top = [{
        "n": n, "a": a, "log10_q": lq,
        "ratio": math.log(a) / (lq * LOG10) if lq > 0 else None,
    } for n, a, lq in top]

    # ---------- envelope: max_{k<=n} log(a_k)/log(q_{k-1}) for n >= n0 ----------
    envelope = []
    for n0 in [10, 100, 1000, 5000, 10000, 50000, 90000]:
        best = None
        for n, a, lq in rows:
            if n < n0 or lq <= 0:
                continue
            r = math.log(a) / (lq * LOG10)
            if best is None or r > best[0]:
                best = (r, n, a)
        envelope.append({"n0": n0, "max_ratio": best[0], "n": best[1], "a": best[2]})

    # ---------- Khinchin distribution fit (a=1..K) ----------
    from collections import Counter
    cnt = Counter(a for _, a, _ in rows if a <= 30)
    K = 30
    total = sum(cnt.values())
    # conditional probabilities given a <= K (correct normalisation)
    pvals = {k: -math.log2(1.0 - 1.0 / (k + 1) ** 2) for k in range(1, K + 1)}
    psum = sum(pvals.values())
    chi2 = 0.0
    table = []
    for k in range(1, K + 1):
        p = pvals[k] / psum
        obs = cnt.get(k, 0)
        exp = p * total
        chi2 += (obs - exp) ** 2 / exp
        table.append({"k": k, "obs": obs, "exp": round(exp, 1), "p": p})

    # ---------- summary ----------
    summary = {
        "N": N,
        "records": rec,
        "top20": top,
        "envelope": envelope,
        "chi2_a_le_30": chi2,
        "dof": K - 1,
        "chi2_over_dof": chi2 / (K - 1),
        "a_gt_1000": sum(1 for _, a, _ in rows if a > 1000),
        "a_gt_100": sum(1 for _, a, _ in rows if a > 100),
        "a_gt_10": sum(1 for _, a, _ in rows if a > 10),
        "max_a": max(a for _, a, _ in rows),
    }
    with open(os.path.join(OUT, "cf_stats_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(os.path.join(OUT, "cf_distribution.csv"), "w", encoding="utf-8") as f:
        f.write("k,obs,exp,p\n")
        for r in table:
            f.write(f"{r['k']},{r['obs']},{r['exp']},{r['p']:.8f}\n")

    print("records:", len(rec))
    for r in rec:
        print(f"  n={r['n']:6d} a={r['a']:8d} log10q={r['log10_q']:10.1f} ratio={r['ratio']:.6f}")
    print("envelope:")
    for e in envelope:
        print(f"  n0={e['n0']:6d} max_ratio={e['max_ratio']:.6f} n={e['n']} a={e['a']}")
    print(f"chi2/dof = {summary['chi2_over_dof']:.4f} (dof={summary['dof']})")


if __name__ == "__main__":
    main()
