#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exp_machine.py -- small explicit runs of the rational-interpolation determinant
machine for pi.

For each parameter set (n,d,m) and a tuple of convergents p_i/q_i we
  * build the integer interpolation matrix B (M x N),
  * compute a saturated Z-basis of ker B and LLL-reduce it,
  * split the reduced basis into vectors with diagonal restriction Q == 0 and
    Q != 0,
  * for the smallest-height Q != 0 compute H_Q, log10|Q(pi)| and the effective
    value exponent A_eff = -log|Q|/log H_Q,
  * compute what the Taylor/arithmetic balance would give:
        kappa > (A_eff+1) * log H_Q / (m log q_max)
    and the naive bound  kappa > n*(A_eff+1).

Output: ../data/machine.csv
"""
import itertools
import os

import mpmath as mp

import roth_machine as rm

mp.mp.dps = 160

CASES = [
    (1, 1, 1, (5,)),
    (2, 1, 1, (2, 3)),
    (2, 2, 1, (2, 3)),
    (2, 2, 2, (2, 3)),
    (3, 1, 1, (2, 3, 4)),
]


def analyze(n, d, m, idxs, conv):
    pts = [(conv[i][0], conv[i][1]) for i in idxs]
    qmax = max(q for _, q in pts)
    B, exps, ais = rm.interp_matrix(n, d, m, pts)
    K = rm.saturated_kernel(B)
    KR = rm.lll_reduce(K)
    rows = []
    for v in KR:
        q = rm.diag_poly(v, exps, n)
        rows.append((v, q))
    zero = [(v, q) for v, q in rows if not q]
    nz = [(v, q) for v, q in rows if q]
    nz.sort(key=lambda t: rm.poly_height(t[1]))
    out = {
        "n": n, "d": d, "m": m, "idxs": idxs,
        "points": pts, "qmax": qmax, "N": B.cols, "M": B.rows,
        "kernel_dim": len(K), "dim_zero": len(zero), "dim_nonzero": len(nz),
    }
    if nz:
        v, q = nz[0]
        H = rm.poly_height(q)
        l10 = rm.evaluate_log10_abs(q)[0]
        Aeff = -l10 / float(mp.log10(H)) if H > 1 else None
        kappa_from_balance = (Aeff + 1) * float(mp.log10(H)) / (m * float(mp.log10(qmax))) if Aeff is not None else None
        out.update({
            "HQ": H, "degQ": max(q), "vecH": rm.vec_height(v),
            "log10Q": l10, "Aeff": Aeff,
            "kappa_balance": kappa_from_balance,
            "kappa_naive_n": n * (Aeff + 1) if Aeff is not None else None,
            "A_needed_for_kappa2": m * 2 * float(mp.log10(qmax)) / float(mp.log10(H)) - 1 if H > 1 else None,
            "Q": rm.poly_to_str(q),
        })
    return out


def main():
    a, conv = rm.pi_cf(24)
    print("partial quotients:", a[:16])
    results = []
    for (n, d, m, idxs) in CASES:
        r = analyze(n, d, m, idxs, conv)
        results.append(r)
        print("=" * 78)
        print(f"n={n} d={d} m={m}  q={[q for _,q in r['points']]}  N={r['N']} M={r['M']} "
              f"dim ker={r['kernel_dim']}  (Q==0: {r['dim_zero']}, Q!=0: {r['dim_nonzero']})")
        if "HQ" in r:
            print(f"  smallest Q!=0: deg={r['degQ']} H_Q={r['HQ']:.6e} |vec|={r['vecH']:.6e}")
            print(f"  log10|Q(pi)|={r['log10Q']:.6f}  A_eff={r['Aeff']:.4f}")
            print(f"  balance kappa > (A_eff+1)log H_Q/(m log q_max) = {r['kappa_balance']:.3f}")
            print(f"  naive n(A_eff+1) = {r['kappa_naive_n']:.3f};  A needed for kappa<=2: {r['A_needed_for_kappa2']:.3f}")
            print(f"  Q = {r['Q'][:110]}")
        else:
            print("  no Q != 0 in the reduced basis")

    out = os.path.join(os.path.dirname(__file__), "..", "data", "machine.csv")
    with open(os.path.abspath(out), "w") as f:
        f.write("n,d,m,idxs,qmax,N,M,kernel_dim,dim_zero,dim_nonzero,HQ,degQ,log10Q,Aeff,kappa_balance,kappa_naive_n,A_needed_for_kappa2\n")
        for r in results:
            f.write(",".join(str(r.get(k, "")) for k in
                             ["n","d","m","idxs","qmax","N","M","kernel_dim","dim_zero","dim_nonzero",
                              "HQ","degQ","log10Q","Aeff","kappa_balance","kappa_naive_n","A_needed_for_kappa2"]) + "\n")
    print("wrote", os.path.abspath(out))


if __name__ == "__main__":
    main()
