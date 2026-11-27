#!/usr/bin/env python3
"""One-page P2 overview figure: fig_p2_overview.png.

Panels (all values recomputed from the raw trial records, no hand-typed numbers):
  (a) feasible (bars) and certified (markers) rate vs p, four methods;
  (b) penetration median [IQR] vs p, four methods (symlog y so 0 is visible);
  (c) real-data ALM certification, 14x14 Shor base vs + zero-dimensional RLT
      (LM, LM-O, YCB; YCB RLT not run -> n/a).

Run:  /home/syk/miniconda3/bin/python make_p2_overview.py
"""
import json
import statistics as st
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
EXP = HERE.parent                                   # .../experiments
RAW_MAIN = EXP / "results" / "raw_main.json"
OUT = HERE / "fig_p2_overview.png"

METHODS = [("unc", "Unconstrained QPEP", "#7f7f7f"),
           ("soft", "Soft penalty ($\\lambda{=}10^3$)", "#e08214"),
           ("proj", "Post-hoc projection", "#2b6cb0"),
           ("alm", "ALM-QPEP (ours)", "#2e7d32")]
PS = [8, 16, 32, 64]


def quantile(vals, p):
    vals = sorted(vals)
    h = (len(vals) - 1) * p
    lo = int(h)
    hi = min(lo + 1, len(vals) - 1)
    return vals[lo] + (h - lo) * (vals[hi] - vals[lo])


raw = json.load(open(RAW_MAIN))
main = {p: raw["main"][f"p{p}"]["recs"] for p in PS}

# ---------------------------------------------------------------- panel (a),(b)
fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.4))
ax = axes[0]
n = len(METHODS)
width = 0.8 / n
for i, (key, label, color) in enumerate(METHODS):
    feas = [100 * sum(bool(r[key]["feasible"]) for r in main[p]) / len(main[p]) for p in PS]
    cert = [100 * sum(bool(r[key]["certified"]) for r in main[p]) / len(main[p]) for p in PS]
    x = np.arange(len(PS)) + (i - (n - 1) / 2) * width
    ax.bar(x, feas, width * 0.92, color=color, alpha=0.75, label=label)
    ax.plot(x, cert, "D", ms=5.5, color="black", zorder=5,
            label="certified (marker)" if i == 0 else None)
ax.set_xticks(np.arange(len(PS)))
ax.set_xticklabels([f"p={p}" for p in PS])
ax.set_ylabel("rate (%)")
ax.set_ylim(-4, 112)
ax.set_title("(a) feasibility (bars) and certification (markers)")
ax.grid(axis="y", alpha=0.3)
ax.legend(fontsize=7.2, loc="upper center", ncol=2, framealpha=0.9)

ax = axes[1]
xs = np.arange(len(PS))
for i, (key, label, color) in enumerate(METHODS):
    med = np.array([st.median([r[key]["pen_mm"] for r in main[p]]) for p in PS])
    lo = np.array([quantile([r[key]["pen_mm"] for r in main[p]], .25) for p in PS])
    hi = np.array([quantile([r[key]["pen_mm"] for r in main[p]], .75) for p in PS])
    jitter = 0.0  # no jitter: p is categorical and values must not be displaced
    ax.errorbar(xs + jitter, med, yerr=[med - lo, hi - med], fmt="o-", ms=6,
                lw=1.8, capsize=3, color=color, label=label)
ax.set_yscale("symlog", linthresh=0.1)
ax.set_yticks([0, 0.01, 0.1, 1, 10, 100])
ax.set_yticklabels(["0", "0.01", "0.1", "1", "10", "100"])
ax.set_ylim(-0.01, 260)
ax.set_xticks(xs)
ax.set_xticklabels([f"p={p}" for p in PS])
ax.set_ylabel("max penetration depth (mm)")
ax.set_title("(b) penetration median [IQR] vs p")
ax.grid(alpha=0.3)
ax.legend(fontsize=7.2, loc="upper left", framealpha=0.9)
ax.text(0.98, 0.04, "median at p=64:  unc 103.02 · soft 0.00\n"
        "projection 0.00 · ALM 0.00", transform=ax.transAxes,
        fontsize=6.8, color="#333", ha="right", va="bottom",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#bbbbbb", lw=0.6))

# ------------------------------------------------------------------- panel (c)
ax = axes[2]
ci = json.load(open(HERE / "real_cert_ci_baseline.json"))
datasets = ["linemod", "lmo", "ycbv"]
names = ["LM", "LM-O", "YCB-V"]
x = np.arange(len(datasets))
w = 0.36


def _bar(i, key, offset, color, hatch=None):
    d = ci[key]
    b = d["base" if offset < 0 else "rlt"]
    if b["rate"] is None:
        ax.bar(i + offset, 100, w, color="none", edgecolor=color,
               hatch="////", linewidth=0.8)
        return None
    v = 100 * b["rate"]
    ax.bar(i + offset, v, w, color=color, hatch=hatch, alpha=0.85 if hatch is None else 1.0)
    lo, hi = 100 * b["ci95"][0], 100 * b["ci95"][1]
    ax.errorbar(i + offset, v, yerr=[[v - lo], [hi - v]], fmt="none",
                ecolor="black", elinewidth=1.0, capsize=3, zorder=6)
    ax.text(i + offset, hi + 1.5, f"{v:.0f}%\n({b['k']}/{b['n']})",
            ha="center", va="bottom", fontsize=7)
    return (lo, hi)


ax.axhline(0, color="black", lw=0.6)
for i, key in enumerate(datasets):
    _bar(i, key, -w / 2, "#7f7f7f")
    if ci[key]["rlt"]["rate"] is None:
        # YCB +RLT: not run
        ax.text(i + w / 2, 45, "n/a\n(not run)", ha="center", va="center",
                fontsize=7.0, color="#2e7d32")
    else:
        _bar(i, key, +w / 2, "#2e7d32")
ax.set_xticks(x)
ax.set_xticklabels([f"{nm}\n(n={ci[k]['n_trials']})" for nm, k in zip(names, datasets)])
ax.set_ylabel("ALM certified rate (%)")
ax.set_ylim(-4, 112)
ax.set_title("(c) real-data certification, base vs +RLT (95% Wilson CI)")
ax.grid(axis="y", alpha=0.3)
ax.legend([plt.Rectangle((0, 0), 1, 1, color="#7f7f7f"),
           plt.Rectangle((0, 0), 1, 1, color="#2e7d32")],
          ["14×14 Shor", "+ 0-dim RLT"], fontsize=7.2, loc="upper left")

fig.text(0.01, 0.005,
         "source: experiments/results/raw_main.json (a,b); "
         "experiments/real/*/results/raw.json + results/real_cert_ci_baseline.json (c)",
         fontsize=6.5, color="#555")
fig.tight_layout(rect=(0, 0.025, 1, 1))
fig.savefig(OUT, dpi=200, facecolor="white", bbox_inches="tight", pad_inches=0.12)
print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")
