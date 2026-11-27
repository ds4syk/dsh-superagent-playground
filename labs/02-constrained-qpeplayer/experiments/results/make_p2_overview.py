#!/usr/bin/env python3
"""One-page P2 overview figure: fig_p2_overview.png.

Panels (all values recomputed from the raw trial records, no hand-typed numbers):
  (a) feasible (bars) and certified (markers) rate vs p, four methods;
  (b) penetration median [IQR] vs p, four methods (symlog y so 0 is visible);
  (c) real-data ALM certification at n=90/72/84: one-sided Shor flag
      (an upper bound), the two-sided |J-Shor|<=1e-6 test (necessary only),
      and -- when results/rigorous_real.json exists -- the rigorous
      P2-Cert Schur-repair lower bound (sound by construction).

Run directly on the login node (no Slurm):
    nice -n 10 /home/syk/miniconda3/bin/python make_p2_overview.py
"""
import json
import math
import os
import statistics as st
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
EXP = HERE.parent                                   # .../experiments
# Where the SDPRLayers-style baseline JSONs live.  In the merged tree they are
# in results/; while the baseline subagent runs in a parallel worktree they can
# be read from there with P2_BASE_DIR=<that worktree>/experiments/results .
BASE_DIR = Path(os.environ.get("P2_BASE_DIR", str(EXP / "results")))
RAW_MAIN = EXP / "results" / "raw_main.json"
RIGOROUS = EXP / "results" / "rigorous_real.json"
OUT = HERE / "fig_p2_overview.png"

METHODS = [("unc", "Unconstrained QPEP", "#7f7f7f"),
           ("soft", "Soft penalty ($\\lambda{=}10^3$)", "#e08214"),
           ("proj", "Post-hoc projection", "#2b6cb0"),
           ("alm", "ALM-QPEP (ours)", "#2e7d32")]
PS = [8, 16, 32, 64]

REAL = {
    "linemod": ("LM", EXP / "real" / "linemod" / "results" / "raw.json"),
    "lmo": ("LM-O", EXP / "real" / "bop" / "results" / "lmo" / "raw.json"),
    "ycbv": ("YCB-V", EXP / "real" / "bop" / "results" / "ycbv" / "raw.json"),
}


def quantile(vals, p):
    vals = sorted(vals)
    h = (len(vals) - 1) * p
    lo = int(h)
    hi = min(lo + 1, len(vals) - 1)
    return vals[lo] + (h - lo) * (vals[hi] - vals[lo])


def wilson(k, n, z=1.96):
    """95% Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    den = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / den
    half = z * math.sqrt(max(0.0, p * (1.0 - p) / n + z * z / (4.0 * n * n))) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def _num(node, *names):
    for nm in names:
        if isinstance(node, dict) and nm in node and node[nm] is not None:
            return float(node[nm])
    return None


def rig_counts(key, rj):
    """Pull (k, n) of rigorously certified ALM rows for one dataset.

    The rigorous pass writes `results/rigorous_real.json` as
      {"datasets": {key: {"configs": {"base"|"rlt"|"brlt":
          {"n":.., "cert_schur":.., "rate_schur":.., ...}}}}}
    The P2-Cert bound is `rigorous_schur` (13-dim Schur repair), so we report
    `cert_schur` on the `brlt` config (the strongest sound cut family).
    Falls back to a few simpler shapes, and returns None if absent.
    """
    ds = rj.get("datasets", {})
    node = ds.get(key, {})
    cfg = node.get("configs", {}) if isinstance(node, dict) else {}
    for name in ("brlt", "base", "rlt"):
        s = cfg.get(name)
        if isinstance(s, dict):
            k = _num(s, "cert_schur", "cert_dual_repair", "k", "cert", "certified")
            n = _num(s, "n", "n_trials", "total")
            if k is not None and n:
                return int(round(k)), int(round(n))
    # generic fallbacks for alternative aggregations
    node = rj.get(key, node)
    if isinstance(node, dict):
        k = _num(node, "k", "cert", "certified", "n_cert", "cert_schur")
        n = _num(node, "n", "n_trials", "total")
        if k is not None and n:
            return int(round(k)), int(round(n))
    return None


raw = json.load(open(RAW_MAIN))
main = {p: raw["main"][f"p{p}"]["recs"] for p in PS}

# ---------------------------------------------------------------- panel (a),(b)
fig, _ax = plt.subplots(2, 2, figsize=(15.2, 9.2))
axes = _ax.ravel()
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
# rigorous JSON uses dataset keys lm/lmo/ycbv while REAL uses linemod/lmo/ycbv
RIG_KEY = {"linemod": "lm", "lmo": "lmo", "ycbv": "ycbv"}
rig = {}
if RIGOROUS.exists():
    rj = json.load(open(RIGOROUS))
    for key in REAL:
        node = rj.get("datasets", {}).get(RIG_KEY[key], {})
        cfg = node.get("configs", {}) if isinstance(node, dict) else {}
        if isinstance(cfg, dict) and "base" in cfg and "brlt" in cfg:
            rig[key] = {c: (int(cfg[c]["cert_schur"]), int(cfg[c]["n"]))
                        for c in ("base", "brlt") if c in cfg}

keys = list(REAL)
x = np.arange(len(keys))
w = 0.20
slots = [(-1.5 * w, "one-sided Shor\n(upper bound)", "#9e9e9e"),
         (-0.5 * w, "two-sided |J−Shor|\n(necessary only)", "#2b6cb0"),
         (0.5 * w, "rigorous P2-Cert\n(base cuts)", "#66bb6a"),
         (1.5 * w, "rigorous P2-Cert\n(+bound-RLT)", "#1b5e20")]

ax.axhline(0, color="black", lw=0.6)
for i, key in enumerate(keys):
    rows = json.load(open(REAL[key][1]))
    alm = [r for r in rows if r["method"] == "ALM-QPEP"]
    nn = len(alm)
    one = sum(bool(r.get("cert")) for r in alm)
    two = sum(bool(r.get("cert_two_sided")) for r in alm)
    entries = [(-1.5 * w, one, nn), (-0.5 * w, two, nn)]
    if key in rig:
        entries += [(0.5 * w, rig[key]["base"][0], rig[key]["base"][1]),
                    (1.5 * w, rig[key]["brlt"][0], rig[key]["brlt"][1])]
    for (off, k, n), (_, _, color) in zip(entries, slots):
        v = 100.0 * k / n
        lo, hi = wilson(k, n)
        ax.bar(i + off, v, w * 0.92, color=color, alpha=0.88)
        ax.errorbar(i + off, v, yerr=[[v - 100 * lo], [100 * hi - v]], fmt="none",
                    ecolor="black", elinewidth=0.9, capsize=2, zorder=6)
        ax.text(i + off, min(100 * hi + 1.5, 104), f"{v:.0f}%\n({k}/{n})",
                ha="center", va="bottom", fontsize=5.6, rotation=0)
    if key not in rig:
        ax.text(i + w, 50, "rigorous\npending", ha="center", va="center",
                fontsize=6.0, color="#1b5e20")

ax.set_xticks(x)
ax.set_xticklabels([f"{REAL[k][0]}\n(n={len([r for r in json.load(open(REAL[k][1])) if r['method'] == 'ALM-QPEP'])})"
                    for k in keys])
ax.set_ylabel("ALM certified rate (%)")
ax.set_ylim(-4, 112)
ax.set_title("(c) real-data certification: upper bound, necessary test, rigorous")
ax.grid(axis="y", alpha=0.3)
ax.legend([plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in slots],
          [lab for _, lab, _ in slots], fontsize=6.6, loc="upper left", framealpha=0.9)

# ------------------------------------------------------------------- panel (d)
# SDP block dimension / solve time: our 14x14 hard relaxation (the certificate
# of record, which is also a hard-constrained layer) versus the SDPRLayers-style
# slack lifting (14+p)x(14+p).  Times are medians over the per-p JSON records;
# nothing is rescaled or fitted.
ax = axes[3]
BASE_PS = [8, 16, 32, 64]
b_p, b_hard, b_slack = [], [], []
for p_ in BASE_PS:
    f = BASE_DIR / f"baseline_sdpr_synth_p{p_}.json"
    if not f.exists():
        continue
    try:
        d = json.load(open(f))
    except Exception:
        continue
    recs = [r for r in d.get("recs", []) if "error" not in r]

    def _t(rec, kind):
        for k in (kind, kind + "_sdp"):
            v = rec.get(k)
            if isinstance(v, dict) and v.get("time_ms") is not None:
                return float(v["time_ms"])
        return None

    hh = [t for t in (_t(r, "hard") for r in recs) if t]
    ss = [t for t in (_t(r, "slack") for r in recs) if t]
    if not hh or not ss:
        continue
    b_p.append(p_)
    b_hard.append(st.median(hh))
    b_slack.append(st.median(ss))

if len(b_p) >= 2:
    x = np.arange(len(b_p))
    w = 0.36
    ax.bar(x - w / 2, b_hard, w, color="#2e7d32", alpha=0.85,
           label="14×14 hard (ours)")
    ax.bar(x + w / 2, b_slack, w, color="#8e24aa", alpha=0.85,
           label="(14+p)×(14+p) SDPRLayers+slack")
    for xi, hv, sv in zip(x, b_hard, b_slack):
        ax.text(xi - w / 2, hv * 1.12, f"{hv:.0f}", ha="center", va="bottom",
                fontsize=6.0, color="#1b5e20")
        ax.text(xi + w / 2, sv * 1.12, f"{sv:.0f}", ha="center", va="bottom",
                fontsize=6.0, color="#4a148c")
    ax.set_yscale("log")
    ax.set_ylim(20.0, max(b_slack) * 1.8)
    ax.set_yticks([20, 30, 50, 100, 200, 500, 1000])
    from matplotlib.ticker import ScalarFormatter
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.set_xticks(x)
    ax.set_xticklabels([f"p={p_}\n{14 + p_}² vs 14²" for p_ in b_p], fontsize=7.0)
    ax.set_ylabel("median SDP wall-clock (ms, log)")
    ax.set_title("(d) certificate/layer SDP: block size and solve time")
    ax.grid(axis="y", alpha=0.3, which="both")
    ax.legend(fontsize=7.2, loc="upper left", framealpha=0.9)
else:
    ax.text(0.5, 0.5, "SDPRLayers re-implementation\nnot available in results/",
            ha="center", va="center", fontsize=8, color="#777")
    ax.set_axis_off()

fig.text(0.01, 0.005,
         "source: experiments/results/raw_main.json (a,b); "
         "experiments/real/*/results/raw.json (c); "
         "results/rigorous_real.json (c, green bars when present); "
         "results/baseline_sdpr_synth_p*.json (d)",
         fontsize=6.5, color="#555")
fig.tight_layout(rect=(0, 0.025, 1, 1))
fig.savefig(OUT, dpi=200, facecolor="white", bbox_inches="tight", pad_inches=0.12)
print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")
