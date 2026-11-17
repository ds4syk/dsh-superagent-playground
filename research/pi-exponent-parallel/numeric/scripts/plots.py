#!/usr/bin/env python3
"""Figures for the pi continued-fraction / irrationality-measure experiments.

Usage: python3 plots.py --tag d1e5 --gk ../data/gk_mc_n97210_trials120.json
Produces PNGs in ../data/plots/.
"""
import argparse
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LÉVY10 = math.pi ** 2 / (12 * math.log(2)) / math.log(10)   # 0.51532041


def load_json(p):
    with open(p) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="d1e5")
    ap.add_argument("--datadir", default="../data")
    ap.add_argument("--gk", default="../data/gk_mc_n97210_trials120.json")
    args = ap.parse_args()
    dd = args.datadir
    outdir = os.path.join(dd, "plots")
    os.makedirs(outdir, exist_ok=True)
    st = load_json(os.path.join(dd, f"cf_{args.tag}_stats.json"))
    gk = load_json(args.gk) if os.path.exists(args.gk) else None

    # ---------------------------------------------------------------- fig 1
    rows = st["T_K_curve"]
    kmax_plot = int(0.92 * st["n_quotients"])
    rows = [r for r in rows if r["K"] <= kmax_plot]
    K = [r["K"] for r in rows]
    T = [r["T_minus_1"] for r in rows]
    plt.figure(figsize=(7.2, 4.8))
    plt.loglog(K, T, "o-", ms=3, lw=1.0, label=r"$\pi$: $T_K-1$ (certified)")
    if gk:
        gk = {"curve": [c for c in gk["curve"] if c["K"] <= kmax_plot]}
        gkK = [c["K"] for c in gk["curve"]]
        for key, style, lab in (("median", "--", "Gauss-Kuzmin median"),
                                ("q95", ":", "Gauss-Kuzmin 95%"),
                                ("q05", ":", "Gauss-Kuzmin 5%")):
            plt.loglog(gkK, [c[key] - 1.0 for c in gk["curve"]], style,
                       color="gray", lw=1.0, label=lab)
    guide = [1.94 * math.log10(k) / k for k in K]
    plt.loglog(K, guide, "-.", color="tab:red", lw=1.0,
               label=r"$1.94\,\log_{10}K/K$ guide")
    plt.xlabel(r"$K$  (terms ignored before $K$)")
    plt.ylabel(r"$T_K-1=\sup_{k\geq K}\log q_{k+1}/\log q_k-1$")
    plt.title(r"$\mu(\pi)\leq 2 \Leftrightarrow T_K\to 1$;  data: $T_K-1=O(\log K/K)$")
    plt.legend(fontsize=8)
    plt.grid(True, which="both", alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"fig1_TK_{args.tag}.png"), dpi=150)
    plt.close()

    # ---------------------------------------------------------------- fig 2
    a = [int(x) for x in open(os.path.join(dd, f"cf_{args.tag}_a.txt")).read().split()]
    n = len(a)
    idx = list(range(1, n))
    vals = a[1:]
    step = max(1, n // 60000)
    plt.figure(figsize=(7.2, 4.8))
    plt.loglog(idx[::step], vals[::step], ",", color="tab:blue",
               label=r"$a_n$  (every %d-th term)" % step)
    rec = [(r["index"], r["a"]) for r in st["records"] if r["index"] > 0]
    plt.loglog([x[0] for x in rec], [x[1] for x in rec], "o-", ms=3, lw=1,
               color="tab:orange", label="records")
    nn = [k for k in idx]
    plt.loglog(nn, [1.44 * k for k in nn], "--", color="gray", lw=1,
               label=r"$1.44\,n$ (GK mean max)")
    plt.loglog(nn, [2.08 * k for k in nn], ":", color="gray", lw=1,
               label=r"$2.08\,n$ (GK median max)")
    plt.loglog(nn, [k ** 1.15 for k in nn], "-.", color="tab:red", lw=1,
               label=r"$n^{1.15}$ (sufficient growth)")
    plt.xlabel(r"$n$")
    plt.ylabel(r"$a_n$")
    plt.title(r"partial quotients of $\pi$:  $a_n\leq 3n$ observed,  $a_n=n^{O(1)}$ suffices")
    plt.legend(fontsize=8)
    plt.grid(True, which="both", alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"fig2_quotients_{args.tag}.png"), dpi=150)
    plt.close()

    # ---------------------------------------------------------------- fig 3
    tot = n
    tail = []
    from bisect import bisect_left
    a_sorted = sorted(a)
    ks_plot = [1, 2, 3, 5, 10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000]
    for k in ks_plot:
        obs = (tot - bisect_left(a_sorted, k)) / tot
        pred = math.log2(1 + 1.0 / k)
        tail.append((k, obs, pred))
    plt.figure(figsize=(7.2, 4.8))
    plt.loglog([x[0] for x in tail], [x[1] for x in tail], "o-", label=r"$\pi$: $\#\{a_n\geq k\}/N$")
    plt.loglog([x[0] for x in tail], [x[2] for x in tail], "s--", label=r"Gauss-Kuzmin $\log_2(1+1/k)$")
    plt.xlabel(r"$k$")
    plt.ylabel("tail probability")
    plt.title("partial-quotient distribution vs Gauss-Kuzmin")
    plt.legend(fontsize=9)
    plt.grid(True, which="both", alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"fig3_distribution_{args.tag}.png"), dpi=150)
    plt.close()

    # ---------------------------------------------------------------- fig 4
    cons = load_json(os.path.join(dd, "constructions.json"))
    fams = {}
    for r in cons:
        fams.setdefault(r["family"], []).append(r)
    plt.figure(figsize=(7.2, 4.8))
    markers = "os^DvP*"
    for i, (f, rs) in enumerate(sorted(fams.items())):
        rs = [r for r in rs if r["log10_q"] > 0.5]
        plt.semilogx([r["log10_q"] for r in rs], [r["lambda_exponent"] for r in rs],
                     marker=markers[i % len(markers)], ms=4, lw=1, label=f)
    plt.axhline(2.0, color="k", ls="--", lw=1, label=r"$\lambda=2$ (Legendre: only convergents)")
    plt.axhline(1.0, color="gray", ls=":", lw=1)
    plt.xlabel(r"$\log_{10} q_N$")
    plt.ylabel(r"$\lambda_N=-\log_{10}|\pi-p_N/q_N|\,/\,\log_{10}q_N$")
    plt.title("candidate constructions: error exponent vs denominator size")
    plt.legend(fontsize=8)
    plt.grid(True, which="both", alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "fig4_constructions.png"), dpi=150)
    plt.close()

    # ---------------------------------------------------------------- fig 5
    cc = os.path.join(dd, "compare_constants.json")
    if os.path.exists(cc):
        c = load_json(cc)
        plt.figure(figsize=(7.2, 4.8))
        for nm, style in (("pi", "o-"), ("e", "s--"), ("sqrt2", "^-.")):
            if nm not in c:
                continue
            KK = c[nm]["K"]
            TT = c[nm]["T_minus_1"]
            pairs = [(k, t) for k, t in zip(KK, TT) if t > 0]
            plt.loglog([x[0] for x in pairs], [x[1] for x in pairs], style,
                       ms=3, lw=1,
                       label=(r"$\pi$: $\max_{n\geq1000} a_n/n$=%.1f" % c[nm]["max_a_over_n_ge1000"]
                              if nm == "pi" else
                              (r"$e$: $\max_{n\geq1000} a_n/n$=%.2f" % c[nm]["max_a_over_n_ge1000"]
                               if nm == "e" else
                               r"$\sqrt{2}$: bounded quotients")))
        plt.xlabel(r"$K$")
        plt.ylabel(r"$T_K-1$")
        plt.title(r"$\mu=2$ holds for $e$ and $\sqrt{2}$ (theorems), despite the linear quotients of $e$")
        plt.legend(fontsize=9)
        plt.grid(True, which="both", alpha=0.25)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "fig5_constants.png"), dpi=150)
        plt.close()

    print("figures written to", outdir)


if __name__ == "__main__":
    main()
