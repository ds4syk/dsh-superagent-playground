#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_fig.py -- two-panel figure for the interpolation report.

(a) products of k consecutive convergents of pi: log10|Q(pi)| vs log10 H(Q).
    All slopes are ~ -1, independent of k: products do not improve A.
(b) pigeonhole minima: min |Q(pi)| over deg <= D, height <= H, vs H.
    Slopes ~ -D: the sharp exponent degrades with the degree.

Output: ../data/fig_no_go.png
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import math

import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")

# ---- read products.csv
singles = []
prods = {}
with open(os.path.join(DATA, "products.csv")) as f:
    for row in csv.DictReader(f):
        if row["type"] == "single":
            q = float(row["a"])
            kappa = float(row["b"])
            Aeff = float(row["c"])
            singles.append((q, kappa, Aeff))
        else:
            # prod,indices,k,log10H,log10Q,Aeff
            k = int(row["a"])
            prods.setdefault(k, []).append((float(row["b"]), float(row["c"])))

# ---- read pigeonhole.csv
pig = {}
with open(os.path.join(DATA, "pigeonhole.csv")) as f:
    for row in csv.DictReader(f):
        D = int(row["D"])
        h = float(row["hpolynomial"])
        pig.setdefault(D, []).append((math.log10(h), float(row["log10_min_abs"]), float(row["H"])))

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
ax = axes[0]
colors = {1: "#1f77b4", 2: "#ff7f0e", 3: "#2ca02c", 4: "#d62728"}
for k in sorted(prods):
    xs = [p[0] for p in prods[k]]
    ys = [p[1] for p in prods[k]]
    ax.scatter(xs, ys, s=22, color=colors[k], label=f"product of {k} convergents")
# reference line slope -1 through the cloud
xr = [min(x for k in prods for x, _ in prods[k]), max(x for k in prods for x, _ in prods[k])]
ax.plot(xr, [-xr[0] - 0.4 + (-(x - xr[0])) for x in xr], "k--", lw=1, label="slope -1")
ax.set_xlabel(r"$\log_{10} H(Q)$")
ax.set_ylabel(r"$\log_{10}|Q(\pi)|$")
ax.set_title("(a) products of convergents: effective exponent $A_{\\rm eff}\\approx 1$")
ax.legend(fontsize=8, loc="upper right")
ax.grid(alpha=0.3)

ax = axes[1]
markers = {1: "o", 2: "s", 3: "^"}
for D in sorted(pig):
    xs = [p[0] for p in pig[D]]
    ys = [p[1] for p in pig[D]]
    ax.plot(xs, ys, markers[D] + "-", ms=4, label=f"deg $\\leq {D}$ (at height of best Q)")
    x0 = xs[0]
    y0 = ys[0]
    xr = [x0, max(p[0] for p in pig[D])]
    ax.plot(xr, [y0 - D * (x - x0) for x in xr], ":", lw=1,
            color=ax.lines[-1].get_color(), alpha=0.7)
ax.set_xlabel(r"$\log_{10} H(Q_{\rm best})$")
ax.set_ylabel(r"$\log_{10}|Q_{\rm best}(\pi)|$")
ax.set_title("(b) pigeonhole minima: slope steepens with $D$")
ax.legend(fontsize=8, loc="lower left")
ax.grid(alpha=0.3)

fig.tight_layout()
out = os.path.join(DATA, "fig_no_go.png")
fig.savefig(out, dpi=170)
print("wrote", os.path.abspath(out))
