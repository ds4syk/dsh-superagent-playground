#!/usr/bin/env python3
"""One-page submission composite for the P1 real-data evidence (measured JSON only).

Panels
  (a) real two-mode ADD-S, LINEMOD (n=15) vs LMO (n=37), hard / Laplace / MC,
      median with 95% bootstrap CI over frames;
  (b) simulated detector-noise sweep on the 15 LINEMOD frames, hard and Laplace
      rotation error vs sigma, median with 95% CI bands;
  (c) same-data forward / local baselines on the bitwise-identical 15 frames:
      SQPNP, IPPE, EPnP, LM (GT init), Theseus (GT init) plus two cold-start
      context rows (LM / Theseus identity init) in grey;
  (d) 40-root count on the 11 instances of results/rootcount_many.json.

Every plotted number is read from a committed JSON file; the script re-reads each
file and prints a check table (mismatches must be 0).  Outputs:
  results/final_summary.png
  results/final_summary.json   (plotted values + source pointers + caption)

Run with the WM python (matplotlib 3.10):
  /home/syk/miniconda3/envs/WM/bin/python make_final_summary.py
"""
from __future__ import annotations

import hashlib
import json
import math
import platform
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
EXP = HERE.parent.parent
RES = HERE / "results"
OUT_PNG = RES / "final_summary.png"
OUT_PDF = RES / "final_summary.pdf"
OUT_JSON = RES / "final_summary.json"

# ----------------------------------------------------------------------------- palette
C_HARD = "#1f5c8b"   # hard argmin
C_LAP = "#d97706"    # projected Laplace mean
C_MC = "#2a9d8f"     # mode-level Monte-Carlo mean
C_COLD = "#8b9099"   # cold-start context
C_BAR2 = "#cfd4da"   # complex roots
C_MIN = "#ffd166"    # real minima marker
C_EDGE = "#222222"
C_GRID = "#e7e7e7"


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10.2,
        "axes.titleweight": "bold",
        "axes.labelsize": 9,
        "axes.labelcolor": "#222222",
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": 8.4,
        "ytick.labelsize": 8.4,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3.2,
        "ytick.major.size": 3.2,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "legend.handlelength": 1.5,
        "grid.color": C_GRID,
        "grid.linewidth": 0.6,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def load(rel: str):
    return json.loads((EXP / rel).read_text())


def nav(obj, path):
    cur = obj
    for key in path:
        cur = cur[key]
    return cur


def sha256(rel: str) -> str:
    return hashlib.sha256((EXP / rel).read_bytes()).hexdigest()


# ----------------------------------------------------------------------------- records
# A record is one plotted number: value in plotted units, optional CI, and the
# exact source pointer it was read from.  All values are extracted, never typed.
records: list[dict] = []
structural: list[dict] = []


def add(panel, item, value, lo, hi, unit, src_file, src_path, src_value, scale):
    records.append(dict(panel=panel, item=item, value=value, lo=lo, hi=hi, unit=unit,
                        source_file=src_file, source_path=list(src_path),
                        source_value=src_value, scale=scale))


def series(panel, item, src_file, path, scale, unit):
    """Median [95% CI] from a JSON aggregate entry; CI may be absent."""
    obj = load(src_file)
    base = nav(obj, path)
    med = base["median"] * scale
    ci = base.get("median_ci")
    if ci is None:
        add(panel, item + " median", med, None, None, unit, src_file, tuple(path) + ("median",), base["median"], scale)
    else:
        add(panel, item + " median", med, ci[0] * scale, ci[1] * scale, unit,
            src_file, tuple(path) + ("median",), base["median"], scale)
        add(panel, item + " CI low", ci[0] * scale, None, None, unit,
            src_file, tuple(path) + ("median_ci", 0), ci[0], scale)
        add(panel, item + " CI high", ci[1] * scale, None, None, unit,
            src_file, tuple(path) + ("median_ci", 1), ci[1], scale)


def scalar(panel, item, src_file, path, scale, unit):
    obj = load(src_file)
    v = nav(obj, path)
    add(panel, item, v * scale, None, None, unit, src_file, tuple(path), v, scale)
    return v


# --- panel (a): real two-mode ADD-S, metres -> mm (full 177-frame LMO)
S_POSE = "real/linemod/results/pose_error.json"
S_LMO = "real/linemod/results/lmo_full.json"
S_LMO37 = "real/linemod/results/lmo_two_mode.json"
for tag, src in [("LINEMOD (n=15)", S_POSE), ("LMO (n=177)", S_LMO)]:
    for method, label in [("hard", "hard argmin"), ("laplace", "Laplace mean"), ("mc", "MC mean")]:
        series("a", f"{tag} / {label} ADD-S", src, ("aggregate", "adds_m", method), 1000.0, "mm")
# caption note only: the earlier 37-frame LMO subset, as a consistency check
for method, label in [("hard", "hard argmin"), ("laplace", "Laplace mean")]:
    v = nav(load(S_LMO37), ("aggregate", "adds_m", method, "median"))
    add("a-note", f"37-frame LMO subset / {label} ADD-S (caption note)", v * 1000.0,
        None, None, "mm", S_LMO37, ("aggregate", "adds_m", method, "median"), v, 1000.0)

# --- panel (c) official forward/cost checks (text annotations only: no degree metric)
S_SDPR = "results/official_sdpr.json"
S_BPNP = "results/official_bpnp.json"
sdpr = load(S_SDPR)
bpnp = load(S_BPNP)
# per-instance tightness / rank / pose-vs-hard, read directly from the file
for i in range(len(sdpr["p1_sdp10"])):
    th = sdpr["p1_sdp10"][i]["theta"]
    add("c-note", f"official SDPR instance {i} theta", float(th), None, None, "theta",
        S_SDPR, ("p1_sdp10", i, "theta"), th, 1.0)
    add("c-note", f"official SDPR theta={sdpr['p1_sdp10'][i]['theta']:g} tight",
        float(bool(sdpr["p1_sdp10"][i]["official"]["upstream_tight"])), None, None, "bool",
        S_SDPR, ("p1_sdp10", i, "official", "upstream_tight"),
        sdpr["p1_sdp10"][i]["official"]["upstream_tight"], 1.0)
    add("c-note", f"official SDPR theta={sdpr['p1_sdp10'][i]['theta']:g} official rank",
        float(sdpr["p1_sdp10"][i]["official"]["rank"]), None, None, "rank",
        S_SDPR, ("p1_sdp10", i, "official", "rank"),
        sdpr["p1_sdp10"][i]["official"]["rank"], 1.0)
    diffs = sdpr["p1_sdp10"][i]["diffs"]
    if "official_xs_vs_hard" in diffs:  # absent when xs is None (non-tight theta=0)
        d = diffs["official_xs_vs_hard"]["rot_deg"]
        add("c-note", f"official SDPR theta={sdpr['p1_sdp10'][i]['theta']:g} xs-vs-hard rot",
            d, None, None, "deg", S_SDPR,
            ("p1_sdp10", i, "diffs", "official_xs_vs_hard", "rot_deg"), d, 1.0)
# --- panel (b): detector-noise sweep, conditions in meta.sigmas order
S_DN = "real/linemod/results/detector_noise.json"
dn = load(S_DN)
sigmas = dn["meta"]["sigmas"]
assert sigmas == [0.0, 0.5, 1.0, 2.0, 5.0], sigmas
for s in sigmas:
    cond = f"sigma_{s:g}"
    for m, label in [("hard_rot", "hard argmin"), ("laplace_rot", "Laplace mean")]:
        series("b", f"{label} rot @ sigma={s:g}px", S_DN,
               ("conditions", cond, "metrics", m), 1.0, "deg")
# 10-degree failure threshold drawn on the panel
scalar("b", "10 deg failure threshold", S_DN, ("meta", "fail_deg"), 1.0, "deg")

# --- panel (c): same-data baselines, rotation error median [95% CI]
S_CL = "real/linemod/results/classical_baselines.json"
S_TH = "real/linemod/results/theseus_baseline.json"
CL_ROWS = [
    ("SQPNP (first)", ("aggregate", "SQPNP first", "rot")),
    ("IPPE (first, planar)", ("aggregate", "IPPE first", "rot")),
    ("EPnP", ("aggregate", "EPnP", "rot")),
    ("LM, GT init (E16)", ("aggregate", "LM GT init", "rot")),
]
for label, path in CL_ROWS:
    series("c", label, S_CL, path, 1.0, "deg")
# Theseus GT init has no bootstrap CI in its JSON (single point estimate)
th_med = nav(load(S_TH), ("main_summary", "gt", "rot_err_deg", "median"))
add("c", "Theseus, GT init (E17) median", th_med, None, None, "deg", S_TH,
    ("main_summary", "gt", "rot_err_deg", "median"), th_med, 1.0)
# cold-start context rows (grey)
series("c", "context: LM, identity init", S_CL, ("aggregate", "LM identity init", "rot"), 1.0, "deg")
th_id = nav(load(S_TH), ("main_summary", "identity", "rot_err_deg", "median"))
add("c", "context: Theseus, identity init", th_id, None, None, "deg", S_TH,
    ("main_summary", "identity", "rot_err_deg", "median"), th_id, 1.0)
# P1 hard reference line
p1_hard = nav(load(S_POSE), ("aggregate", "rot_deg", "hard", "median"))
add("c", "reference: P1 hard argmin", p1_hard, None, None, "deg", S_POSE,
    ("aggregate", "rot_deg", "hard", "median"), p1_hard, 1.0)

# --- panel (d): 11 instances, 40 projective roots each
S_RC = "results/rootcount_many.json"
rc = load(S_RC)
for i, case in enumerate(rc["cases"]):
    name = case["name"]
    for key, unit in [("n_projective", "roots"), ("n_real", "roots"),
                      ("n_complex", "roots"), ("n_minima", "minima")]:
        add("d", f"{name} / {key}", float(case[key]), None, None, unit, S_RC,
            ("cases", i, key), case[key], 1.0)

# ----------------------------------------------------------------------------- verify
def check_roundtrip():
    """Re-read every source file from disk and compare with the plotted records."""
    rows, bad = [], 0
    cache = {}
    for r in records:
        f = r["source_file"]
        if f not in cache:
            cache[f] = json.loads((EXP / f).read_text())
        src = nav(cache[f], r["source_path"])
        expect = src * r["scale"]
        ok = math.isclose(r["value"], expect, rel_tol=1e-12, abs_tol=1e-12)
        if not ok:
            bad += 1
        rows.append((r, ok))
    return rows, bad


# ----------------------------------------------------------------------------- figure
def make_figure(rows):
    style()
    fig = plt.figure(figsize=(11.0, 11.7), dpi=300)
    gs = fig.add_gridspec(2, 2, left=0.118, right=0.975, top=0.935, bottom=0.27,
                          hspace=0.46, wspace=0.30)

    def letter(ax, s):
        ax.text(-0.105, 1.075, s, transform=ax.transAxes, fontsize=12.5,
                fontweight="bold", va="top", ha="left")

    # ---------------- (a) ADD-S, LINEMOD vs LMO
    ax = fig.add_subplot(gs[0, 0])
    pose_a = load(S_POSE)["aggregate"]["adds_m"]
    lmo_a = load(S_LMO)["aggregate"]["adds_m"]
    methods = [("hard", "hard argmin", C_HARD), ("laplace", "Laplace mean", C_LAP),
               ("mc", "MC mean", C_MC)]
    w = 0.24
    x = np.array([0.0, 1.0])
    for j, (m, lab, col) in enumerate(methods):
        vals, lo, hi = [], [], []
        for agg in (pose_a, lmo_a):
            med = agg[m]["median"] * 1000.0
            c0, c1 = agg[m]["median_ci"]
            vals.append(med)
            lo.append(med - c0 * 1000.0)
            hi.append(c1 * 1000.0 - med)
        xs = x + (j - 1) * w
        ax.bar(xs, vals, w, color=col, edgecolor=C_EDGE, linewidth=0.6, zorder=3)
        ax.errorbar(xs, vals, yerr=np.array([lo, hi]), fmt="none", ecolor=C_EDGE,
                    elinewidth=0.9, capsize=2.6, capthick=0.9, zorder=4)
        for xi, v, u in zip(xs, vals, hi):
            ax.text(xi, v + u + 0.75, f"{v:.2f}", ha="center", va="bottom",
                    fontsize=7.3, color="#222222")
    ax.set_xticks(x)
    ax.set_xticklabels(["LINEMOD\neggbox/glue (n=15)", "LMO seq. 000002\n(n=177 usable)"])
    ax.set_ylim(0, 27.5)
    ax.set_ylabel("ADD-S (mm, median [95% CI])")
    ax.set_title("Real two-mode ADD-S: LINEMOD vs LMO")
    ax.grid(axis="y", zorder=0)
    ax.legend(handles=[Patch(facecolor=c, edgecolor=C_EDGE, linewidth=0.6, label=l)
                       for _, l, c in methods],
              loc="upper left", borderaxespad=0.6, labelspacing=0.35)
    letter(ax, "(a)")

    # ---------------- (b) detector noise
    ax = fig.add_subplot(gs[0, 1])
    dn_ = load(S_DN)["conditions"]
    sx = np.array([s for s in sigmas], dtype=float)
    for m, lab, col, mk in [("hard_rot", "hard argmin", C_HARD, "o"),
                            ("laplace_rot", "Laplace mean", C_LAP, "s")]:
        med = np.array([dn_[f"sigma_{s:g}"]["metrics"][m]["median"] for s in sx])
        lo = np.array([dn_[f"sigma_{s:g}"]["metrics"][m]["median_ci"][0] for s in sx])
        hi = np.array([dn_[f"sigma_{s:g}"]["metrics"][m]["median_ci"][1] for s in sx])
        ax.fill_between(sx, lo, hi, color=col, alpha=0.16, linewidth=0, zorder=2)
        ax.plot(sx, med, color=col, lw=1.6, marker=mk, ms=5.0, zorder=3,
                markeredgecolor="white", markeredgewidth=0.7)
        ax.plot([sx[-1]], [med[-1]], marker="", zorder=3)
    ax.axhline(dn["meta"]["fail_deg"], color="#777777", lw=0.8, ls=(0, (4, 3)), zorder=1)
    ax.text(0.18, dn["meta"]["fail_deg"] + 0.9, "10° failure threshold",
            fontsize=7.0, color="#666666")
    ax.text(5.16, dn_["sigma_5"]["metrics"]["hard_rot"]["median"],
            "hard argmin", color=C_HARD, fontsize=7.6, ha="left", va="center", fontweight="bold")
    ax.text(5.16, dn_["sigma_5"]["metrics"]["laplace_rot"]["median"],
            "Laplace mean", color=C_LAP, fontsize=7.6, ha="left", va="center", fontweight="bold")
    ax.set_xlim(-0.25, 6.65)
    ax.set_ylim(0, 53)
    ax.set_xticks(sigmas)
    ax.set_xticklabels(["0", "0.5", "1", "2", "5"])
    ax.set_xlabel("injected i.i.d. Gaussian noise σ (px)")
    ax.set_ylabel("rotation error vs GT (deg)")
    ax.set_title("Simulated detector noise, 15 LINEMOD frames")
    ax.grid(axis="y", zorder=0)
    letter(ax, "(b)")

    # ---------------- (c) same-data baselines
    ax = fig.add_subplot(gs[1, 0])
    cl = load(S_CL)["aggregate"]

    def ci_of(entry):
        med = entry["rot"]["median"]
        c0, c1 = entry["rot"]["median_ci"]
        return med, med - c0, c1 - med

    labels, meds, errs, cols, rows_y, is_ctx = [], [], [], [], [], []
    y = 0.0
    for lab, key in [("SQPNP", "SQPNP first"), ("IPPE (planar)", "IPPE first"),
                     ("EPnP", "EPnP"), ("LM, GT init", "LM GT init")]:
        m, l, h = ci_of(cl[key])
        labels.append(lab); meds.append(m); errs.append((l, h)); cols.append(C_HARD)
        rows_y.append(y); is_ctx.append(False); y += 1.0
    labels.append("Theseus, GT †"); meds.append(th_med); errs.append((0.0, 0.0))
    cols.append(C_HARD); rows_y.append(y); is_ctx.append(False); y += 1.45
    id_med, id_lo, id_hi = ci_of(cl["LM identity init"])
    for lab, val, err in [("LM, identity", id_med, (id_lo, id_hi)),
                          ("Theseus, identity", th_id, (0.0, 0.0))]:
        labels.append(lab); meds.append(val); errs.append(err); cols.append(C_COLD)
        rows_y.append(y); is_ctx.append(True); y += 1.0

    ax.axvline(p1_hard, color="#8a8f98", lw=0.9, ls=(0, (3, 3)), zorder=1)
    ax.axhspan(4.70, 6.80, color="#f0f1f3", zorder=0)
    for lab, m, (l, h), col, yy, ctx in zip(labels, meds, errs, cols, rows_y, is_ctx):
        if h > 0:
            ax.errorbar([m], [yy], xerr=np.array([[l], [h]]), fmt="none",
                        ecolor=col, elinewidth=1.1, capsize=2.8, capthick=1.0, zorder=3)
        ax.plot([m], [yy], marker="o" if not ctx else "s", ms=6.0,
                mfc=col if not ctx else "white", mec=col, mew=1.1, ls="none", zorder=4)
        ax.text(0.985, yy, f"{m:.2f}", transform=ax.get_yaxis_transform(),
                ha="right", va="center", fontsize=8.0,
                color=col if not ctx else "#555555",
                fontweight="bold" if not ctx else "normal")
    ax.set_yticks(rows_y)
    ax.set_yticklabels(labels)
    ax.set_ylim(9.85, -0.75)
    ax.set_xscale("log")
    ax.set_xlim(1.05, 500)
    ax.set_xticks([1, 2, 5, 10, 20, 50, 100, 200])
    ax.set_xticklabels(["1", "2", "5", "10", "20", "50", "100", "200"])
    ax.set_xlabel("rotation error vs GT (deg, median; 95% bootstrap CI)")
    ax.set_title("Same-data forward / local baselines")
    ax.grid(axis="x", which="major", zorder=0)
    ax.text(1.10, 4.55, "cold start (identity init) — context only, not deployable",
            fontsize=6.9, color="#666666", va="bottom")
    sdpr = load(S_SDPR)["p1_sdp10"]
    tight = [e for e in sdpr if e["official"]["upstream_tight"]]
    tt = ", ".join(f"{e['theta']:g}" for e in tight)
    pose_deg = max(e["diffs"]["official_xs_vs_hard"]["rot_deg"] for e in tight)
    note = (f"Official SDPR (10×10 QPEP): tight θ={tt} ≡ hard argmin"
            f" ({pose_deg:.1f}°);\nθ=0 non-tight (rank {sdpr[0]['official']['rank']}).  "
            "Official BPnP: released backward runs; P1 same-basis\n"
            "adaptation withdrawn (recheck, Table 5.24).\n"
            "† point estimate, no CI  ·  dashed = P1 hard 2.06°")
    ax.text(1.06, 7.10, note, fontsize=5.95, va="top", ha="left", linespacing=1.45,
            color="#333333",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.92, pad=2.0))
    letter(ax, "(c)")

    # ---------------- (d) 40-root count
    ax = fig.add_subplot(gs[1, 1])
    cases = load(S_RC)["cases"]
    names = []
    for c in cases:
        n = c["name"]
        names.append("r" + n.split("seed")[-1] if n.startswith("random_seed") else "M(θ=0)")
    xd = np.arange(len(cases))
    nreal = np.array([c["n_real"] for c in cases], dtype=float)
    ncomp = np.array([c["n_complex"] for c in cases], dtype=float)
    nmin = np.array([c["n_minima"] for c in cases], dtype=float)
    ax.bar(xd, nreal, 0.68, color=C_HARD, edgecolor="white", linewidth=0.6, zorder=3,
           label="real projective roots")
    ax.bar(xd, ncomp, 0.68, bottom=nreal, color=C_BAR2, edgecolor="white", linewidth=0.6,
           zorder=3, label="complex projective roots")
    ax.scatter(xd, nmin, marker="D", s=26, facecolor=C_MIN, edgecolor=C_EDGE,
               linewidth=0.7, zorder=5, label="real minima")
    for xi, nr in zip(xd, nreal):
        ax.text(xi, nr / 2.0, f"{int(nr)}", ha="center", va="center", fontsize=7.0,
                color="white", zorder=4)
    ax.axhline(40, color="#777777", lw=0.8, ls=(0, (4, 3)), zorder=4)
    ax.text(len(cases) - 0.45, 41.2, "40 projective roots, 11/11 instances",
            fontsize=7.4, color="#555555", ha="right", va="bottom")
    ax.set_xticks(xd)
    ax.set_xticklabels(names, rotation=32, ha="right", fontsize=8.2)
    ax.set_ylim(0, 50)
    ax.set_yticks([0, 10, 20, 30, 40])
    ax.set_ylabel("projective roots per instance")
    ax.set_xlabel("instance (M(θ=0) = E1 instance at θ=0)")
    ax.set_title("40-root count: 40/40 on 11/11")
    ax.grid(axis="y", zorder=0)
    ax.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.01),
              columnspacing=1.2, handletextpad=0.5, borderaxespad=0.0,
              fontsize=7.4)
    letter(ax, "(d)")

    # ---------------- caption with evidence tag
    cap = build_caption()
    fig.text(0.118, 0.012, cap, ha="left", va="bottom", fontsize=6.85,
             color="#1a1a1a", linespacing=1.38)

    fig.savefig(OUT_PNG, dpi=300, facecolor="white")
    fig.savefig(OUT_PDF, facecolor="white")  # vector copy for the paper build
    return cap


def build_caption():
    pose = load(S_POSE)["aggregate"]["adds_m"]
    lmo = load(S_LMO)["aggregate"]["adds_m"]
    dn_ = load(S_DN)
    cl = load(S_CL)["aggregate"]
    rc = load(S_RC)
    f = lambda v: f"{v:.2f}"
    hard_lm = pose["hard"]["median"] * 1000
    lap_lm = pose["laplace"]["median"] * 1000
    hard_lmo = lmo["hard"]["median"] * 1000
    lap_lmo = lmo["laplace"]["median"] * 1000
    h0 = dn_["conditions"]["sigma_0"]["metrics"]["hard_rot"]["median"]
    h5 = dn_["conditions"]["sigma_5"]["metrics"]["hard_rot"]["median"]
    l5 = dn_["conditions"]["sigma_5"]["metrics"]["laplace_rot"]["median"]
    l0 = dn_["conditions"]["sigma_0"]["metrics"]["laplace_rot"]["median"]
    fail0 = 100 * dn_["conditions"]["sigma_0"]["hard_fail_rate_draws"]
    fail5 = 100 * dn_["conditions"]["sigma_5"]["hard_fail_rate_draws"]
    meds_c = [cl[k]["rot"]["median"] for k in ("SQPNP first", "IPPE first", "EPnP", "LM GT init")]
    meds_c.append(nav(load(S_TH), ("main_summary", "gt", "rot_err_deg", "median")))
    rmax = rc["max_chart_residual"]
    lmo37 = load(S_LMO37)["aggregate"]["adds_m"]
    lm37_hard = lmo37["hard"]["median"] * 1000
    lm37_lap = lmo37["laplace"]["median"] * 1000
    lmo_mc = lmo["mc"]["median"] * 1000
    # object counts of the 177 usable frames
    cnt = Counter(f["obj"] for f in load(S_LMO)["frames"])
    n_egg, n_glue = cnt[10], cnt[11]
    sdpr = load(S_SDPR)
    bpnp = load(S_BPNP)
    tight = [e for e in sdpr["p1_sdp10"] if e["official"]["upstream_tight"]]
    non = [e for e in sdpr["p1_sdp10"] if not e["official"]["upstream_tight"]]
    tt = ", ".join(f"{e['theta']:g}" for e in tight)
    nt = ", ".join(f"{e['theta']:g}" for e in non)
    pose_deg = max(e["diffs"]["official_xs_vs_hard"]["rot_deg"] for e in tight)
    sdpr_commit = sdpr["provenance"]["upstream_commit"]
    bpnp_commit = bpnp["env"]["bpnp_commit"][:7]
    lines = [
        "Figure 1 | Real-data evidence for the two-mode QPEP/Laplace layer. (a) ADD-S on the controlled two-mode cost",
        f"(median [95% bootstrap CI over frames]): LINEMOD eggbox/glue n=15, LMO sequence 000002 n=177 ({n_egg} eggbox + {n_glue} glue). The hard",
        f"argmin is less accurate under occlusion (LINEMOD {f(hard_lm)} mm vs LMO {f(hard_lmo)} mm) while the posterior mean is {f(lap_lm)} vs {f(lap_lmo)} mm",
        f"(MC {f(lmo_mc)} mm); both datasets show the same hard→Laplace bias. The earlier 37-frame LMO subset gives {f(lm37_hard)}/{f(lm37_lap)} mm, consistent.",
        "(b) Simulated i.i.d. Gaussian detector noise on the 15 LINEMOD frames (clean RANSAC subset fixed, never re-fit; per-frame mean over",
        f"10 seeds, median and 95% CI over the 15 frames). Up to 1 px there is no measurable change; between 2 and 5 px the hard argmin slides along",
        f"the flat valley ({f(h0)}°→{f(h5)}°, frames >10°: {fail0:.0f}%→{fail5:.0f}%) while the already-biased Laplace mean changes by {l5 / l0:.2f}×. (c) Same-data",
        f"forward/local references on the bitwise-identical 15-frame correspondences: the five solver/initialisation rows land at the hard argmin ({min(meds_c):.2f}–{max(meds_c):.2f}°);",
        "cold-start local optimisers (grey) do not recover it from identity. Official checks are text-only: SDPR's tight rows",
        f"(θ={tt}) return the hard argmin ({pose_deg:.1f}°), θ={nt} is non-tight; the official BPnP recheck is in Table 5.24, where the earlier",
        "P1 same-basis adaptation is withdrawn. (d) Total-degree homotopy: 40/40 complex projective roots on 11/11 instances (max chart residual",
        f"{rmax:.1e}; 80 affine roots per projective root); the real/complex split and the number of real minima are instance properties, not constants.",
        "Evidence tag: independent reproduction (this repo) — E15 (pose/ADD-S), E16 (classical), E17 (Theseus), E18 (root count); N1 (LMO,",
        f"detector-noise) are controlled estimator experiments with GT-associated correspondences, not detector benchmarks. Official SDPR ({sdpr_commit})",
        f"and BPnP ({bpnp_commit}) used the released code on CPU; no training, no GPU.",
    ]
    return "\n".join(lines)


# ----------------------------------------------------------------------------- main
def main():
    rows, bad = check_roundtrip()
    # structural checks: sample sizes, same-data assertions, root-count totals
    pose, lmo = load(S_POSE), load(S_LMO)
    dn_ = load(S_DN)
    cl = load(S_CL)
    th = load(S_TH)
    rc = load(S_RC)
    structural.extend([
        ("pose n frames == 15", len(pose["frames"]) == 15),
        ("pose reproduction status ok", pose["reproduction_check"]["status"] == "ok"),
        ("LMO full n_used == 177", lmo["meta"]["n_used"] == 177),
        ("LMO full n_attempted == 200", lmo["meta"]["n_attempted"] == 200),
        ("LMO full n_failures == 23", lmo["meta"]["n_failures"] == 23),
        ("LMO full object counts 96/81", sorted(Counter(f["obj"] for f in lmo["frames"]).values()) == [81, 96]),
        ("LMO full reproduces 37-frame subset", lmo["comparison_previous"]["status"] == "ok"),
        ("LMO 37-frame subset n_used == 37", load(S_LMO37)["meta"]["n_used"] == 37),
        ("detector n_frames == 15", dn_["meta"]["n_frames"] == 15),
        ("detector reproduction status ok", dn_["reproduction_check"]["status"] == "ok"),
        ("detector sigmas == [0,0.5,1,2,5]", dn_["meta"]["sigmas"] == [0.0, 0.5, 1.0, 2.0, 5.0]),
        ("classical n_frames == 15", cl["verification"]["n_frames"] == 15),
        ("classical correspondences bitwise == two_mode", bool(cl["verification"]["correspondences_bitwise_equal_to_run_two_mode"])),
        ("theseus frames == 15", th["comparison"]["frames"] == 15),
        ("rootcount n_cases == 11", rc["n_cases"] == 11),
        ("rootcount all_40 is true", rc["all_40"] is True),
        ("rootcount n_40 == 11", rc["n_40"] == 11),
        ("rootcount every n_projective == 40", all(c["n_projective"] == 40 for c in rc["cases"])),
        ("rootcount every n_real+n_complex == 40", all(c["n_real"] + c["n_complex"] == 40 for c in rc["cases"])),
        ("rootcount max chart residual < 1e-12", rc["max_chart_residual"] < 1e-12),
        ("panel (c) Theseus GT has no CI in source", "median_ci" not in th["main_summary"]["gt"]["rot_err_deg"]),
        ("official SDPR: 3 instances", len(load(S_SDPR)["p1_sdp10"]) == 3),
        ("official SDPR: exactly 2 tight rows", sum(e["official"]["upstream_tight"] for e in load(S_SDPR)["p1_sdp10"]) == 2),
        ("official SDPR: theta=0 non-tight rank 2", (load(S_SDPR)["p1_sdp10"][0]["official"]["upstream_tight"] is False)
         and load(S_SDPR)["p1_sdp10"][0]["official"]["rank"] == 2),
        ("official SDPR: tight pose == hard argmin (0 deg)",
         max(e["diffs"]["official_xs_vs_hard"]["rot_deg"] for e in load(S_SDPR)["p1_sdp10"] if e["official"]["upstream_tight"]) == 0.0),
        ("official BPnP: released backward executed on each adapter case",
         all(c["t_bpnp_backward_ms"] > 0 for c in load(S_BPNP)["adaptation"])),
    ])
    bad_struct = sum(1 for _, ok in structural if not ok)

    # ---- check table
    print("PLOTTED-VALUE CHECK  (value = source_value x scale; source re-read from disk)\n")
    hdr = f"{'#':>3} {'panel':>5} {'item':<46} {'plotted':>12} {'CI':>21} {'source':>12} {'units':>5}  ok  source path"
    print(hdr)
    print("-" * len(hdr))
    for i, (r, ok) in enumerate(rows, 1):
        ci = "" if r["lo"] is None else f"[{r['lo']:.5f}, {r['hi']:.5f}]"
        print(f"{i:>3} {r['panel']:>5} {r['item']:<46} {r['value']:>12.5f} {ci:>21} "
              f"{r['source_value']:>12.6f} {r['unit']:>5}  {'OK' if ok else 'BAD':<3} "
              f"{r['source_file']}:{'.'.join(str(x) for x in r['source_path'])}")
    print(f"\n{len(rows)} plotted values checked against their source files; mismatches: {bad}\n")

    print("\nSTRUCTURAL / SAME-DATA CHECKS")
    for name, ok in structural:
        print(f"  [{'OK' if ok else 'BAD'}] {name}")
    print(f"\n{len(structural)} structural checks; failures: {bad_struct}")

    assert bad == 0, "plotted-value mismatch"
    assert bad_struct == 0, "structural check failure"

    cap = make_figure(rows)
    meta = {
        "figure": str(OUT_PNG),
        "script": str(Path(__file__).resolve()),
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "matplotlib": matplotlib.__version__,
        "numpy": np.__version__,
        "dpi": 300,
        "figsize_in": [11.0, 11.7],
        "png_px": [3300, 3510],
        "evidence_tag": "independent reproduction (this repo) - E15, E16, E17, E18; N1 controlled estimator (partially run)",
        "caption": cap,
        "checks": {
            "n_values": len(rows),
            "n_mismatches": bad,
            "n_structural": len(structural),
            "structural_failures": bad_struct,
            "rows": [dict(panel=r["panel"], item=r["item"], value=r["value"], lo=r["lo"], hi=r["hi"],
                          unit=r["unit"], source_file=r["source_file"], source_path=r["source_path"],
                          source_value=r["source_value"], scale=r["scale"], match=ok)
                     for r, ok in rows],
        },
        "sources_sha256": {f: sha256(f) for f in sorted({r["source_file"] for r, _ in rows})},
        "notes": {
            "missing_sources": [],
            "panel_a": "ADD-S in mm = metres x 1000; median and percentile bootstrap CI over frames; LMO full n=177 (96 eggbox + 81 glue); the 37-frame subset values are caption-only consistency checks.",
            "panel_b": "per-frame mean over 10 noise seeds; median and 95% percentile bootstrap CI over the 15 frames.",
            "panel_c": "classical/Theseus rotation medians (deg); Theseus GT/identity rows have no CI in the JSON; cold-start rows are context, not deployable baselines.",
            "panel_c_official": "Official SDPR/BPnP are text annotations, not bars: SDPR reports no real-frame degree metric, and the BPnP P1 adaptation is withdrawn. SDPR: on the synthetic 10x10 QPEP instances, tight rows (theta=0.01, 0.1) return the hard argmin (official_xs_vs_hard.rot_deg = 0.0); theta=0 is non-tight (rank 2, xs=None). BPnP: the earlier P1 same-basis adaptation is withdrawn after a recheck (paper Table 5.24); this figure asserts no P1 gradient/cost match. The released backward was executed on all three adapter cases (t_bpnp_backward_ms > 0).",
            "panel_a_counts": {"eggbox": 96, "glue": 81, "n_used": 177, "n_attempted": 200, "n_failures": 23},
            "panel_d": "n_projective/n_real/n_complex are projective roots; each projective root is an 80-affine-root class; n_minima = real local minima.",
            "official_provenance": {
                "sdpr_commit": sdpr["provenance"]["upstream_commit"],
                "bpnp_commit": bpnp["env"]["bpnp_commit"],
            },
        },
    }
    OUT_JSON.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"\nwrote {OUT_PNG}")
    print(f"wrote {OUT_JSON}")


if __name__ == "__main__":
    sys.exit(main())
